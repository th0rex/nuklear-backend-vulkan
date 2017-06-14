// Copyright 2017 th0rex
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern crate nuklear_rust;
#[macro_use]
extern crate quick_error;
#[macro_use]
extern crate vulkano;
#[macro_use]
extern crate vulkano_shader_derive;

use std::mem::size_of;
use std::sync::Arc;

use nuklear_rust::{NkBuffer, NkContext, NkConvertConfig, NkDrawVertexLayoutAttribute,
                   NkDrawVertexLayoutElements, NkDrawVertexLayoutFormat, NkHandle, NkRect};
use quick_error::ResultExt;
use vulkano::OomError;
use vulkano::buffer::{BufferSlice, BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandAddError, CommandBufferBuilder,
                              CommandBufferBuilderError, DynamicState};
use vulkano::command_buffer::cb::{AbstractStorageLayer, SubmitSyncLayer, UnsafeCommandBuffer};
use vulkano::command_buffer::commands_raw::CmdCopyBufferToImageError;
use vulkano::command_buffer::pool::StandardCommandPool;
use vulkano::descriptor::descriptor_set::{DescriptorSet, SimpleDescriptorSetBuilder,
                                          SimpleDescriptorSetBufferExt,
                                          SimpleDescriptorSetImageExt};
use vulkano::descriptor::pipeline_layout::{PipelineLayout, PipelineLayoutDescUnion};
use vulkano::device::{Device, Queue};
use vulkano::format::R8G8B8A8Unorm;
use vulkano::framebuffer::{Framebuffer, RenderPass, RenderPassDesc, Subpass};
use vulkano::image::{Dimensions, ImmutableImage, SwapchainImage};
use vulkano::instance::QueueFamily;
use vulkano::pipeline::{GraphicsPipeline, GraphicsPipelineParams};
use vulkano::pipeline::blend::Blend;
use vulkano::pipeline::depth_stencil::DepthStencil;
use vulkano::pipeline::input_assembly::InputAssembly;
use vulkano::pipeline::multisample::Multisample;
use vulkano::pipeline::vertex::{SingleBufferDefinition, Vertex};
use vulkano::pipeline::viewport::{Scissor, Viewport, ViewportsState};
use vulkano::sampler::Sampler;
use vulkano::swapchain::Swapchain;

mod render_pass;

use render_pass::CustomRenderPassDesc;

quick_error! {
    #[derive(Clone, Debug)]
    pub enum Error {
        CopyImageError(err: CommandBufferBuilderError<CmdCopyBufferToImageError>) {
            display("error trying to copy image: {:?}", err)
            from()
        }
        DrawIndexed(err: CommandAddError) {
            display("could not add command to `draw_indexed` call: {:?}", err)
            from()
        }
        TextureNotFound {
            display("nuklear sent a texture id to draw that was never created")
        }
        VulkanOom(err: OomError) {
            display("vulkan is out of memory: {:?}", err)
            from()
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;

struct Buffers {
    index_buffer: Arc<CpuAccessibleBuffer<[u16]>>,
    index_buffer_slice: BufferSlice<[u16], Arc<CpuAccessibleBuffer<[u16]>>>,
    index_count: usize,
    uniform_buffer: Arc<CpuAccessibleBuffer<vs::ty::Data>>,
    vertex_buffer: Arc<CpuAccessibleBuffer<[vs::Vertex]>>,
    vertex_count: usize,
}

impl Buffers {
    fn new(dimensions: [u32; 2],
           vertex_count: usize,
           index_count: usize,
           device: &Arc<Device>,
           queue_family: Option<QueueFamily>)
           -> Buffers {
        let index_buffer = unsafe {
            CpuAccessibleBuffer::uninitialized_array(device.clone(),
                                                     index_count,
                                                     BufferUsage::all(),
                                                     queue_family)
                    .expect("failed to create index buffer")
        };

        Buffers {
            index_buffer: index_buffer.clone(),
            index_buffer_slice: BufferSlice::from_typed_buffer_access(index_buffer.clone()),
            index_count,
            uniform_buffer: CpuAccessibleBuffer::from_data(device.clone(),
                                                           BufferUsage::all(),
                                                           queue_family,
                                                           vs::ty::Data {
                                                               scale: [2f32 / dimensions[0] as f32,
                                                                       2f32 / dimensions[1] as f32],
                                                               transform: [-1f32, -1f32],
                                                           })
                    .expect("failed to create uniform buffer"),
            vertex_buffer: unsafe {
                CpuAccessibleBuffer::uninitialized_array(device.clone(),
                                                         vertex_count,
                                                         BufferUsage::all(),
                                                         queue_family)
                        .expect("failed to create vertex buffer")
            },
            vertex_count,
        }
    }
}

struct Texture {
    buffer: Arc<CpuAccessibleBuffer<[u8]>>,
    set: Arc<DescriptorSet + Send + Sync>,
    texture: Arc<ImmutableImage<R8G8B8A8Unorm>>,
}

impl Texture {
    fn new(data: &[u8],
           width: u32,
           height: u32,
           device: &Arc<Device>,
           family: Option<QueueFamily>,
           sampler: &Arc<Sampler>,
           uniforms: &Arc<CpuAccessibleBuffer<vs::ty::Data>>,
           pipeline: &Pipeline)
           -> Texture {
        let texture = ImmutableImage::new(device.clone(),
                                          Dimensions::Dim2d { width, height },
                                          R8G8B8A8Unorm,
                                          family)
                .unwrap();

        let buffer = CpuAccessibleBuffer::from_iter(device.clone(),
                                                    BufferUsage::all(),
                                                    family,
                                                    data.iter().cloned())
                .expect("failed to create texture buffer");

        let set = {
            let builder = SimpleDescriptorSetBuilder::new(pipeline.clone(), 0);
            let builder = uniforms.clone().add_me(builder, "constants");
            let builder = (texture.clone(), sampler.clone()).add_me(builder, "sTexture");
            builder.build()
        };

        Texture {
            buffer,
            set: Arc::new(set),
            texture,
        }
    }
}

type CustomFrameBuffer = Framebuffer<Arc<RenderPass<CustomRenderPassDesc>>,
                                     ((), Arc<SwapchainImage>)>;

type ImageCommandBuffer =
    SubmitSyncLayer<AbstractStorageLayer<UnsafeCommandBuffer<Arc<StandardCommandPool>>>>;

type Pipeline = Arc<GraphicsPipeline<SingleBufferDefinition<vs::Vertex>,
                                     PipelineLayout<PipelineLayoutDescUnion<vs::Layout,
                                                                            fs::Layout>>,
                                     Arc<RenderPass<CustomRenderPassDesc>>>>;

pub struct Renderer {
    buffers: Buffers,
    device: Arc<Device>,
    dimensions: [u32; 2],
    frame_buffers: Vec<Arc<CustomFrameBuffer>>,
    fs: Arc<fs::Shader>,
    pipeline: Pipeline,
    queue: Arc<Queue>,
    render_pass: Arc<RenderPass<CustomRenderPassDesc>>,
    sampler: Arc<Sampler>,
    swapchain: Arc<Swapchain>,
    textures: Vec<Texture>,
    vertex_layout_elements: NkDrawVertexLayoutElements,
    vs: Arc<vs::Shader>,
}

impl Renderer {
    pub fn new(device: Arc<Device>,
               queue: Arc<Queue>,
               swapchain: Arc<Swapchain>,
               images: &[Arc<SwapchainImage>])
               -> Renderer {
        assert!(images.len() >= 1);
        let dimensions = images[0].dimensions();
        let buffers = Buffers::new(dimensions,
                                   512 * 1024,
                                   128 * 1024,
                                   &device,
                                   Some(queue.family()));

        let vs = Arc::new(vs::Shader::load(&device).expect("failed to load vertex shader"));
        let fs = Arc::new(fs::Shader::load(&device).expect("failed to load fragment shader"));

        let render_pass = Arc::new(CustomRenderPassDesc { color: (swapchain.format(), 1) }
                                       .build_render_pass(device.clone())
                                       .unwrap());

        let pipeline = Renderer::create_pipeline(&device, dimensions, &fs, &render_pass, &vs);

        let sampler = Sampler::simple_repeat_linear_no_mipmap(device.clone());

        let frame_buffers = Renderer::create_frame_buffers(images, &render_pass);

        Renderer {
            buffers,
            device,
            dimensions,
            frame_buffers,
            fs: fs.clone(),
            pipeline,
            queue,
            render_pass,
            sampler,
            swapchain,
            textures: vec![],
            vertex_layout_elements: Renderer::get_vle(),
            vs: vs.clone(),
        }
    }

    pub fn add_texture(&mut self, data: &[u8], width: u32, height: u32) -> NkHandle {
        self.textures
            .push(Texture::new(data,
                               width,
                               height,
                               &self.device,
                               Some(self.queue.family()),
                               &self.sampler,
                               &self.buffers.uniform_buffer,
                               &self.pipeline));

        NkHandle::from_id(self.textures.len() as i32 - 1)
    }

    #[inline]
    pub fn get_frame_buffer(&self, image_num: usize) -> Option<Arc<CustomFrameBuffer>> {
        self.frame_buffers.get(image_num).map(|x| x.clone())
    }

    pub fn initial_commands(&self) -> Result<ImageCommandBuffer> {
        let mut command_buffer = AutoCommandBufferBuilder::new(self.device.clone(),
                                                               self.queue.family())?;

        for texture in &self.textures {
            command_buffer =
                command_buffer
                    .copy_buffer_to_image(texture.buffer.clone(), texture.texture.clone())?;
        }

        command_buffer.build().map_err(From::from)
    }

    pub fn initialize_convert_config(&self, config: &mut NkConvertConfig) {
        config.set_vertex_layout(&self.vertex_layout_elements);
        config.set_vertex_size(size_of::<vs::Vertex>());
    }

    // RENDER PASS MUST HAVE ALREADY BEGON (begin_render_pass needs to be called)
    pub fn render(&mut self,
                  ctx: &mut NkContext,
                  nk_cmd_buffer: &mut NkBuffer,
                  config: &NkConvertConfig,
                  mut command_buffer: AutoCommandBufferBuilder)
                  -> Result<AutoCommandBufferBuilder> {
        self.convert(ctx, nk_cmd_buffer, config);

        let mut start = 0;
        let mut end = 0;

        for cmd in ctx.draw_command_iterator(&nk_cmd_buffer) {
            if cmd.elem_count() < 1 {
                continue;
            }

            end = start + cmd.elem_count();

            let texture = self.find_texture(cmd.texture().id().unwrap())
                .ok_or(Error::TextureNotFound)?;

            let &NkRect { x, y, w, h } = cmd.clip_rect();

            let scissor = Scissor {
                origin: [if x < 0f32 { 0 } else { x as _ },
                         if y < 0f32 { 0 } else { y as _ }],
                dimensions: [if x < 0f32 { (x + w) as _ } else { w as _ },
                             if y < 0f32 { (h + y) as _ } else { h as _ }],
            };

            let slice = self.buffers.index_buffer_slice.clone();

            command_buffer = command_buffer
                .draw_indexed(self.pipeline.clone(),
                              DynamicState {
                                  scissors: Some(vec![scissor]),
                                  ..Default::default()
                              },
                              self.buffers.vertex_buffer.clone(),
                              slice.slice(start as usize..end as usize).unwrap(),
                              texture.set.clone(),
                              ())?;

            start = end;
        }

        Ok(command_buffer)
    }

    pub fn resize(&mut self, dimensions: [u32; 2]) -> Result<Arc<Swapchain>> {
        self.dimensions = dimensions;
        let (swapchain, images) = self.swapchain.recreate_with_dimension(dimensions)?;

        self.pipeline = Renderer::create_pipeline(&self.device,
                                                  dimensions,
                                                  &self.fs,
                                                  &self.render_pass,
                                                  &self.vs);
        self.frame_buffers = Renderer::create_frame_buffers(&images, &self.render_pass);
        self.swapchain = swapchain.clone();

        let mut data = self.buffers.uniform_buffer.write().unwrap();
        data.scale = [2f32 / dimensions[0] as f32, 2f32 / dimensions[1] as f32];

        Ok(swapchain)
    }

    fn convert(&mut self,
               ctx: &mut NkContext,
               nk_cmd_buffer: &mut NkBuffer,
               config: &NkConvertConfig) {
        let mut vertex_buffer = self.buffers.vertex_buffer.write().unwrap();
        let mut vertex_buffer = unsafe {
            std::slice::from_raw_parts_mut(&mut *vertex_buffer as *mut [_] as *mut u8,
                                           size_of::<vs::Vertex>() * self.buffers.vertex_count)
        };
        let mut vertex_buffer = NkBuffer::with_fixed(&mut vertex_buffer);

        let mut index_buffer = self.buffers.index_buffer.write().unwrap();
        let mut index_buffer = unsafe {
            std::slice::from_raw_parts_mut(&mut *index_buffer as *mut [_] as *mut u8,
                                           size_of::<u16>() * self.buffers.index_count)
        };
        let mut index_buffer = NkBuffer::with_fixed(&mut index_buffer);

        ctx.convert(nk_cmd_buffer, &mut vertex_buffer, &mut index_buffer, config);
    }

    fn create_frame_buffers(images: &[Arc<SwapchainImage>],
                            pass: &Arc<RenderPass<CustomRenderPassDesc>>)
                            -> Vec<Arc<CustomFrameBuffer>> {
        images
            .iter()
            .map(|image| {
                     Arc::new(Framebuffer::start(pass.clone())
                                  .add(image.clone())
                                  .unwrap()
                                  .build()
                                  .unwrap())
                 })
            .collect()
    }

    fn create_pipeline(device: &Arc<Device>,
                       dimensions: [u32; 2],
                       fs: &Arc<fs::Shader>,
                       render_pass: &Arc<RenderPass<CustomRenderPassDesc>>,
                       vs: &Arc<vs::Shader>)
                       -> Pipeline {
        let pipeline_params = GraphicsPipelineParams {
            vertex_input: SingleBufferDefinition::new(),
            vertex_shader: vs.main_entry_point(),
            input_assembly: InputAssembly::triangle_list(),
            tessellation: None,
            geometry_shader: None,
            viewport: ViewportsState::DynamicScissors {
                viewports: vec![Viewport {
                                    origin: [0f32, 0f32],
                                    depth_range: 0f32..1f32,
                                    dimensions: [dimensions[0] as f32, dimensions[1] as f32],
                                }],
            },
            raster: Default::default(),
            multisample: Multisample::disabled(),
            fragment_shader: fs.main_entry_point(),
            depth_stencil: DepthStencil::disabled(),
            blend: Blend::alpha_blending(),
            render_pass: Subpass::from(render_pass.clone(), 0).unwrap(),
        };

        Arc::new(GraphicsPipeline::new(device.clone(), pipeline_params).unwrap())
    }

    fn find_texture(&self, id: i32) -> Option<&Texture> {
        self.textures.get(id as usize)
    }

    fn get_shader_offset(name: &str) -> u32 {
        vs::Vertex::member(name).unwrap().offset as _
    }

    fn get_vle() -> NkDrawVertexLayoutElements {
        use NkDrawVertexLayoutAttribute::{NK_VERTEX_ATTRIBUTE_COUNT, NK_VERTEX_COLOR,
                                          NK_VERTEX_POSITION, NK_VERTEX_TEXCOORD};
        use NkDrawVertexLayoutFormat::{NK_FORMAT_COUNT, NK_FORMAT_FLOAT,
                                       NK_FORMAT_R32G32B32A32_FLOAT};

        NkDrawVertexLayoutElements::new(&[(NK_VERTEX_POSITION,
                                           NK_FORMAT_FLOAT,
                                           Renderer::get_shader_offset("pos")),
                                          (NK_VERTEX_TEXCOORD,
                                           NK_FORMAT_FLOAT,
                                           Renderer::get_shader_offset("uv")),
                                          (NK_VERTEX_COLOR,
                                           NK_FORMAT_R32G32B32A32_FLOAT,
                                           Renderer::get_shader_offset("color")),
                                          (NK_VERTEX_ATTRIBUTE_COUNT,
                                           NK_FORMAT_COUNT,
                                           Renderer::get_shader_offset("_count"))])
    }
}

mod vs {
    #[derive(Clone, Debug)]
    pub struct Vertex {
        pub pos: [f32; 2],
        pub uv: [f32; 2],
        pub color: [f32; 4],
        pub _count: i32,
    }
    impl_vertex!(Vertex, pos, uv, color, _count);

    #[derive(VulkanoShader)]
    #[ty = "vertex"]
    #[src = "
#version 450
layout(location = 0) in vec2 pos;
layout(location = 1) in vec2 uv;
layout(location = 2) in vec4 color;
layout(location = 3) in int _count;

layout(set=0, binding=0) uniform Data {
    vec2 scale;
    vec2 transform;
} constants;

out gl_PerVertex {
    vec4 gl_Position;
};

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec2 outUV;

void main() {
    outColor = color;
    outUV = uv;
    gl_Position = vec4(pos * constants.scale + constants.transform, 0, 1);
}
"]
    struct Dummy;
}

mod fs {
    #[derive(VulkanoShader)]
    #[ty = "fragment"]
    #[src = "
#version 450

layout(location = 0) in vec4 color;
layout(location = 1) in vec2 uv;

layout(location = 0) out vec4 fragmentColor;

layout(set = 0, binding = 1) uniform sampler2D sTexture;

void main() {
    fragmentColor = color * texture(sTexture, uv.st);
}
"]
    struct Dummy;
}
