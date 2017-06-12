// Copyright 2017 th0rex
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use vulkano::format::{ClearValue, Format};
use vulkano::framebuffer::{LayoutAttachmentDescription, LayoutPassDependencyDescription, LayoutPassDescription, LoadOp, RenderPassDesc, RenderPassDescClearValues, StoreOp};
use vulkano::image::ImageLayout;

pub struct CustomRenderPassDesc {
    pub color: (Format, u32),
}

unsafe impl RenderPassDesc for CustomRenderPassDesc {
    fn num_attachments(&self) -> usize {
        1
    }

    fn attachment_desc(&self, id: usize) -> Option<LayoutAttachmentDescription> {
        if id == 0 {
            Some(LayoutAttachmentDescription {
                     format: self.color.0,
                     samples: self.color.1,
                     load: LoadOp::Clear,
                     store: StoreOp::Store,
                     stencil_load: LoadOp::Clear,
                     stencil_store: StoreOp::Store,
                     initial_layout: ImageLayout::Undefined,
                     final_layout: ImageLayout::ColorAttachmentOptimal,
                 })
        } else {
            unreachable!();
            None
        }
    }

    fn num_subpasses(&self) -> usize {
        1
    }

    fn subpass_desc(&self, id: usize) -> Option<LayoutPassDescription> {
        if id == 0 {
            Some(LayoutPassDescription {
                     color_attachments: vec![(0, ImageLayout::ColorAttachmentOptimal)],
                     depth_stencil: None,
                     input_attachments: vec![],
                     resolve_attachments: vec![],
                     preserve_attachments: vec![],
                 })
        } else {
            unreachable!();
            None
        }
    }

    fn num_dependencies(&self) -> usize {
        0
    }

    fn dependency_desc(&self, id: usize) -> Option<LayoutPassDependencyDescription> {
        unreachable!();
        None
    }
}

unsafe impl RenderPassDescClearValues<Vec<ClearValue>> for CustomRenderPassDesc {
    fn convert_clear_values(&self, values: Vec<ClearValue>) -> Box<Iterator<Item = ClearValue>> {
        Box::new(values.into_iter())
    }
}
