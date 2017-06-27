// Copyright 2017 th0rex
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cmp::max;
use std::sync::Arc;
use std::time::Duration;

use nuklear_rust::{NkAllocator, NkAntiAliasing, NkBuffer, NkButton, NkContext, NkConvertConfig,
                   NkDrawNullTexture, NkFont, NkFontAtlas, NkFontAtlasFormat, NkFontConfig, NkKey};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferBuilder};
use vulkano::device::Device;
use vulkano::instance::{DeviceExtensions, Instance, PhysicalDevice};
use vulkano::swapchain::{acquire_next_image, SurfaceTransform, Swapchain};
use vulkano::sync::{GpuFuture, now};
use vulkano_win::{required_extensions, VkSurfaceBuild, Window};
use winit::{ElementState, Event, EventsLoop, MouseButton, MouseScrollDelta, VirtualKeyCode,
            WindowBuilder, WindowEvent};

use super::{Error, Renderer, Result};

/// Determines how to proceed after each frame.
pub enum Action {
    /// Close the application.
    Close,
    /// Continue running.
    Continue,
}

/// A trait that should be implemented for anything that is a state for nuklear to use.
/// It must provide a way to load required media, a function to render the UI
/// and ways to initialize various nuklear structs.
pub trait State {
    /// The Media that this state is able to load and that it will use.
    type Media;

    /// Create the command buffer for nuklear.
    fn create_command_buffer(alloc: &mut NkAllocator) -> NkBuffer {
        NkBuffer::with_size(alloc, 64 * 1024)
    }

    /// Create the convert config for nuklear.
    fn create_convert_config(null: NkDrawNullTexture) -> NkConvertConfig {
        let mut convert_config = NkConvertConfig::default();
        convert_config.set_null(null.clone());
        convert_config.set_circle_segment_count(22);
        convert_config.set_curve_segment_count(22);
        convert_config.set_arc_segment_count(22);
        convert_config.set_global_alpha(1f32);
        convert_config.set_shape_aa(NkAntiAliasing::NK_ANTI_ALIASING_ON);
        convert_config.set_line_aa(NkAntiAliasing::NK_ANTI_ALIASING_ON);
        convert_config
    }

    /// Create the font config to be used.
    fn create_font_config() -> NkFontConfig;

    /// Creates a new instance of `Media` and initializes it.
    /// Also must return a main font to use for the application.
    fn load_media(
        cfg: &mut NkFontConfig,
        atlas: &mut NkFontAtlas,
        renderer: &mut Renderer,
    ) -> (Self::Media, Box<NkFont>);

    /// Renders the UI and returns an `Action` that determines whether the window should be
    /// closed or not.
    ///
    /// **IMPORTANT**: This shouldn't do any updating (like polling some futures or what ever),
    /// stuff like that should take place in the `update` function for ideal performance when using
    /// multiple threads.
    fn render_ui(&mut self, ctx: &mut NkContext, media: &mut Self::Media) -> Action;

    /// Update things that should run while vulkan is rendering things.
    /// This is the place where most of your polling of futures or what ever should happen.
    fn update(&mut self) {}
}

/// A way to build the `UI` structure of this module.
#[derive(Default)]
pub struct Builder<'a> {
    background_color: Option<[u8; 4]>,
    dimensions: [u32; 2],
    images: Option<u32>,
    index_count: Option<usize>,
    min_dimensions: Option<[u32; 2]>,
    max_dimensions: Option<[u32; 2]>,
    window_title: Option<&'a str>,
    vertex_count: Option<usize>,
}

impl<'a> Builder<'a> {
    /// Creates a new `Builder` with the given `dimensions`.
    /// `dimensions[0]` is the width and `dimensions[1]` is the height.
    pub fn new(dimensions: [u32; 2]) -> Self {
        Self {
            dimensions,
            ..Default::default()
        }
    }

    /// Sets the background color of the window.
    /// Format is `[r, g, b, a]`.
    #[inline]
    pub fn background_color(mut self, color: [u8; 4]) -> Self {
        self.background_color = Some(color);
        self
    }

    /// Sets the number of framebuffers to use.
    #[inline]
    pub fn images(mut self, images: u32) -> Self {
        self.images = Some(images);
        self
    }

    /// Sets the index count for the renderer.
    #[inline]
    pub fn index_count(mut self, count: usize) -> Self {
        self.index_count = Some(count);
        self
    }

    /// Sets the minimum dimensions for the window.
    #[inline]
    pub fn min_dimensions(mut self, dimensions: [u32; 2]) -> Self {
        self.min_dimensions = Some(dimensions);
        self
    }

    /// Sets the maximum dimensions for the window.
    #[inline]
    pub fn max_dimensions(mut self, dimensions: [u32; 2]) -> Self {
        self.max_dimensions = Some(dimensions);
        self
    }

    /// Sets the title of the window.
    #[inline]
    pub fn title(mut self, title: &'a str) -> Self {
        self.window_title = Some(title);
        self
    }

    /// Sets the vertex count for the renderer.
    #[inline]
    pub fn vertex_count(mut self, count: usize) -> Self {
        self.vertex_count = Some(count);
        self
    }

    /// Builds the `UI`.
    pub fn build<S: State>(self, state: S) -> Result<InitUI<S>> {
        let instance = {
            let extensions = required_extensions();

            Instance::new(None, &extensions, None)?
        };

        let physical = PhysicalDevice::enumerate(&instance).next().ok_or(
            Error::NoDeviceFound,
        )?;

        let events_loop = Arc::new(EventsLoop::new());
        let mut window = WindowBuilder::new();

        window = window.with_dimensions(self.dimensions[0], self.dimensions[1]);

        if let Some(t) = self.window_title {
            window = window.with_title(t);
        }

        if let Some(min) = self.min_dimensions {
            window = window.with_min_dimensions(min[0], min[1]);
        }

        if let Some(max) = self.max_dimensions {
            window = window.with_max_dimensions(max[0], max[1]);
        }

        let window = window
            .build_vk_surface(&*events_loop, instance.clone())
            .expect("could not create window");

        let queue = physical
            .queue_families()
            .find(|&q| {
                q.supports_graphics() && window.surface().is_supported(q).unwrap_or(false)
            })
            .ok_or(Error::NoQueueFound)?;

        let (device, mut queue) = {
            let device_ext = DeviceExtensions {
                khr_swapchain: true,
                ..DeviceExtensions::none()
            };

            Device::new(
                &physical,
                physical.supported_features(),
                &device_ext,
                [(queue, 0.5)].iter().cloned(),
            ).expect("failed to create device")
        };

        let queue = queue.next().ok_or(Error::NoQueueFound)?;

        let (swapchain, images) = {
            // Theres nothing we can do, swapchain::surface::CapabilitiesError is not exported.
            let caps = window.surface().capabilities(physical).expect(
                "failed to get capabilities of window surface",
            );

            let dimensions = caps.current_extent.unwrap_or(self.dimensions);

            let present = caps.present_modes.iter().next().unwrap();
            let alpha = caps.supported_composite_alpha.iter().next().unwrap();
            let format = caps.supported_formats[0].0;

            Swapchain::new(
                device.clone(),
                window.surface().clone(),
                max(caps.min_image_count, self.images.unwrap_or(0)),
                format,
                dimensions,
                1,
                caps.supported_usage_flags,
                &queue,
                SurfaceTransform::Identity,
                alpha,
                present,
                true,
                None,
            )?
        };

        Ok(InitUI {
            events_loop,
            renderer: Renderer::with_count(
                device.clone(),
                queue.clone(),
                swapchain.clone(),
                &images,
                self.vertex_count,
                self.index_count,
            ),
            state,
            swapchain,
            window,
        })
    }
}

pub struct InitUI<S: State> {
    events_loop: Arc<EventsLoop>,
    renderer: Renderer,
    state: S,
    swapchain: Arc<Swapchain>,
    window: Window,
}

impl<S: State> InitUI<S> {
    pub fn initialize(mut self) -> UI<S> {
        let mut font_config = S::create_font_config();
        let mut alloc = NkAllocator::new_vec();
        let mut atlas = NkFontAtlas::new(&mut alloc);

        let (media, font) = S::load_media(&mut font_config, &mut atlas, &mut self.renderer);

        let font_texture = {
            let (b, w, h) = atlas.bake(NkFontAtlasFormat::NK_FONT_ATLAS_RGBA32);
            self.renderer.add_texture(b, w, h)
        };

        let mut null = NkDrawNullTexture::default();
        atlas.end(font_texture, Some(&mut null));

        let ctx = NkContext::new(&mut alloc, &font.handle());

        let mut convert_config = S::create_convert_config(null.clone());
        self.renderer.initialize_convert_config(&mut convert_config);

        let command_buffer = S::create_command_buffer(&mut alloc);

        UI {
            command_buffer,
            context: ctx,
            convert_config,
            events_loop: self.events_loop,
            font,
            media,
            mx: 0,
            my: 0,
            renderer: self.renderer,
            state: self.state,
            swapchain: self.swapchain,
            window: self.window,
        }
    }
}

pub struct UI<S: State> {
    command_buffer: NkBuffer,
    context: NkContext,
    convert_config: NkConvertConfig,
    events_loop: Arc<EventsLoop>,
    font: Box<NkFont>,
    media: S::Media,
    mx: i32,
    my: i32,
    renderer: Renderer,
    state: S,
    swapchain: Arc<Swapchain>,
    // This field is actually never used, but we need to keep the window alive.
    // If we would let it drop, we would not be able to handle ANY events.
    #[allow(unused)]
    window: Window,
}

impl<S: State> UI<S> {
    pub fn run(mut self) {
        let device = self.renderer.get_device();
        let mut previous_frame = Box::new(now(device.clone())) as Box<GpuFuture>;

        previous_frame = self.load_textures(previous_frame);

        loop {
            if let Action::Close = self.state.render_ui(&mut self.context, &mut self.media) {
                break;
            }

            previous_frame.cleanup_finished();

            previous_frame = self.render_frame(previous_frame);

            self.state.update();

            if let Action::Close = self.update_ui() {
                break;
            }
        }
    }

    fn load_textures(&self, previous_frame: Box<GpuFuture>) -> Box<GpuFuture> {
        let &Self {
            ref renderer,
            ref swapchain,
            ..
        } = self;
        let queue = renderer.get_queue();

        let (image_num, acquire_future) =
            acquire_next_image(swapchain.clone(), Duration::new(1, 0)).unwrap();

        let command_buffer = renderer.initial_commands().unwrap().build().unwrap();
        Box::new(
            previous_frame
                .join(acquire_future)
                .then_execute(queue.clone(), command_buffer)
                .unwrap()
                .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
                .then_signal_fence_and_flush()
                .unwrap(),
        )
    }

    fn render_frame(&mut self, mut previous_frame: Box<GpuFuture>) -> Box<GpuFuture> {
        let &mut Self {
            ref mut context,
            ref convert_config,
            command_buffer: ref mut nk_command_buffer,
            ref mut renderer,
            ref swapchain,
            ..
        } = self;
        let queue = renderer.get_queue();

        let device = renderer.get_device();

        let (image_num, acquire_future) =
            acquire_next_image(swapchain.clone(), Duration::new(1, 0)).unwrap();

        let frame_buffer = renderer.get_frame_buffer(image_num).unwrap();

        let command_buffer = AutoCommandBufferBuilder::new(device.clone(), queue.family())
            .unwrap()
            .begin_render_pass(
                frame_buffer,
                false,
                vec![[0.2f32, 0.2f32, 0.5f32, 1f32].into()],
            )
            .unwrap();

        let command_buffer = renderer
            .render(context, nk_command_buffer, convert_config, command_buffer)
            .unwrap()
            .end_render_pass()
            .unwrap()
            .build()
            .unwrap();

        previous_frame = Box::new(
            previous_frame
                .join(acquire_future)
                .then_execute(queue.clone(), command_buffer)
                .unwrap()
                .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
                .then_signal_fence_and_flush()
                .unwrap(),
        );

        context.clear();

        previous_frame
    }

    fn update_ui(&mut self) -> Action {
        let mut done = false;

        let &mut Self {
            context: ref mut ctx,
            ref mut mx,
            ref mut my,
            ref mut renderer,
            ref mut swapchain,
            ..
        } = self;

        ctx.input_begin();
        self.events_loop.poll_events(|ev| {
            let Event::WindowEvent { event, .. } = ev;

            match event {
                WindowEvent::Closed => done = true,
                WindowEvent::ReceivedCharacter(c) => {
                    ctx.input_unicode(c);
                }
                WindowEvent::KeyboardInput(s, _, k, _) => {
                    if let Some(k) = k {
                        let key = match k {
                            VirtualKeyCode::Back => NkKey::NK_KEY_BACKSPACE,
                            VirtualKeyCode::Delete => NkKey::NK_KEY_DEL,
                            VirtualKeyCode::Up => NkKey::NK_KEY_UP,
                            VirtualKeyCode::Down => NkKey::NK_KEY_DOWN,
                            VirtualKeyCode::Left => NkKey::NK_KEY_LEFT,
                            VirtualKeyCode::Right => NkKey::NK_KEY_RIGHT,
                            _ => NkKey::NK_KEY_NONE,
                        };

                        ctx.input_key(key, s == ElementState::Pressed);
                    }
                }
                WindowEvent::MouseMoved(x, y) => {
                    *mx = x;
                    *my = y;
                    ctx.input_motion(x, y);
                }
                WindowEvent::MouseInput(s, b) => {
                    let button = match b {
                        MouseButton::Left => NkButton::NK_BUTTON_LEFT,
                        MouseButton::Middle => NkButton::NK_BUTTON_MIDDLE,
                        MouseButton::Right => NkButton::NK_BUTTON_RIGHT,
                        _ => NkButton::NK_BUTTON_MAX,
                    };

                    ctx.input_button(button, *mx, *my, s == ElementState::Pressed)
                }
                WindowEvent::MouseWheel(d, _) => {
                    if let MouseScrollDelta::LineDelta(_, y) = d {
                        ctx.input_scroll(y * 22f32);
                    } else if let MouseScrollDelta::PixelDelta(_, y) = d {
                        ctx.input_scroll(y);
                    }
                }
                WindowEvent::Resized(w, h) => {
                    *swapchain = renderer.resize([w, h]).expect("Resize failed");
                }
                _ => (),
            }
        });
        ctx.input_end();

        if done {
            Action::Close
        } else {
            Action::Continue
        }
    }
}
