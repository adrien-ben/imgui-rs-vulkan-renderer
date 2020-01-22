mod vulkan;

use ash::{
    version::{DeviceV1_0, InstanceV1_0},
    vk, Device, Instance,
};
use imgui::{Context, DrawCmd, DrawCmdParams, DrawData};
use mesh::*;
use std::error::Error;
use ultraviolet::projection::orthographic_vk;
use vulkan::*;

pub trait RendererVkContext {
    fn instance(&self) -> &Instance;
    fn physical_device(&self) -> vk::PhysicalDevice;
    fn device(&self) -> &Device;
    fn graphics_queue(&self) -> vk::Queue;
    fn command_pool(&self) -> vk::CommandPool;
}

pub struct Renderer {
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    fonts_texture: Texture,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    mesh: Option<Mesh>,
}

impl Renderer {
    pub fn new<C: RendererVkContext>(
        vk_context: &C,
        render_pass: vk::RenderPass,
        imgui: &mut Context,
    ) -> Result<Self, Box<dyn Error>> {
        // Descriptor set layout
        let descriptor_set_layout = create_vulkan_descriptor_set_layout(vk_context.device())?;

        // Pipeline and layout
        let pipeline_layout =
            create_vulkan_pipeline_layout(vk_context.device(), descriptor_set_layout)?;
        let pipeline = create_vulkan_pipeline(vk_context.device(), pipeline_layout, render_pass)?;

        // Font texture
        let fonts_texture = {
            let mut fonts = imgui.fonts();
            let atlas_texture = fonts.build_rgba32_texture();
            let memory_properties = unsafe {
                vk_context
                    .instance()
                    .get_physical_device_memory_properties(vk_context.physical_device())
            };

            execute_one_time_commands(
                vk_context.device(),
                vk_context.graphics_queue(),
                vk_context.command_pool(),
                |buffer| {
                    Texture::cmd_from_rgba(
                        vk_context.device(),
                        buffer,
                        memory_properties,
                        atlas_texture.width,
                        atlas_texture.height,
                        vk::Format::R8G8B8A8_UNORM,
                        &atlas_texture.data,
                    )
                },
            )??
        };

        // Descriptor set
        let (descriptor_pool, descriptor_set) = create_vulkan_descriptor_set(
            vk_context.device(),
            descriptor_set_layout,
            &fonts_texture,
        )?;

        Ok(Self {
            pipeline,
            pipeline_layout,
            descriptor_set_layout,
            fonts_texture,
            descriptor_pool,
            descriptor_set,
            mesh: None,
        })
    }

    pub fn cmd_draw<C: RendererVkContext>(
        &mut self,
        vk_context: &C,
        command_buffer: vk::CommandBuffer,
        draw_data: &DrawData,
    ) -> Result<(), Box<dyn Error>> {
        if self.mesh.is_none() {
            self.mesh.replace(Mesh::new(vk_context, draw_data)?);
        }
        let mesh = self.mesh.as_mut().unwrap();
        mesh.refresh(vk_context, draw_data)?;

        unsafe {
            vk_context.device().cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            )
        };

        unsafe {
            vk_context.device().cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.descriptor_set],
                &[],
            )
        };

        let framebuffer_width = draw_data.framebuffer_scale[0] * draw_data.display_size[0];
        let framebuffer_height = draw_data.framebuffer_scale[1] * draw_data.display_size[1];
        let viewports = [vk::Viewport {
            width: framebuffer_width,
            height: framebuffer_height,
            max_depth: 1.0,
            ..Default::default()
        }];

        unsafe {
            vk_context
                .device()
                .cmd_set_viewport(command_buffer, 0, &viewports)
        };

        // Ortho projection
        let projection =
            orthographic_vk(0.0, framebuffer_width, 0.0, -framebuffer_height, -1.0, 1.0);
        unsafe {
            let push = any_as_u8_slice(&projection);
            vk_context.device().cmd_push_constants(
                command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                push,
            )
        };

        unsafe {
            vk_context.device().cmd_bind_index_buffer(
                command_buffer,
                mesh.indices,
                0,
                vk::IndexType::UINT16,
            )
        };

        unsafe {
            vk_context
                .device()
                .cmd_bind_vertex_buffers(command_buffer, 0, &[mesh.vertices], &[0])
        };

        let mut index_offset = 0;
        let mut vertex_offset = 0;
        let clip_offset = draw_data.display_pos;
        let clip_scale = draw_data.framebuffer_scale;
        for draw_list in draw_data.draw_lists() {
            for command in draw_list.commands() {
                match command {
                    DrawCmd::Elements {
                        count,
                        cmd_params:
                            DrawCmdParams {
                                clip_rect,
                                vtx_offset,
                                idx_offset,
                                ..
                            },
                    } => {
                        unsafe {
                            let clip_x = (clip_rect[0] - clip_offset[0]) * clip_scale[0];
                            let clip_y = (clip_rect[1] - clip_offset[1]) * clip_scale[1];
                            let clip_w = (clip_rect[2] - clip_offset[0]) * clip_scale[0] - clip_x;
                            let clip_h = (clip_rect[3] - clip_offset[1]) * clip_scale[1] - clip_y;

                            let scissors = [vk::Rect2D {
                                offset: vk::Offset2D {
                                    x: clip_x as _,
                                    y: clip_y as _,
                                },
                                extent: vk::Extent2D {
                                    width: clip_w as _,
                                    height: clip_h as _,
                                },
                            }];
                            vk_context
                                .device()
                                .cmd_set_scissor(command_buffer, 0, &scissors);
                        }

                        unsafe {
                            vk_context.device().cmd_draw_indexed(
                                command_buffer,
                                count as _,
                                1,
                                index_offset + idx_offset as u32,
                                vertex_offset + vtx_offset as i32,
                                0,
                            )
                        };
                    }
                    _ => (), // Ignored for now
                }
            }

            index_offset += draw_list.idx_buffer().len() as u32;
            vertex_offset += draw_list.vtx_buffer().len() as i32;
        }

        Ok(())
    }

    pub fn drop(&mut self, device: &Device) {
        unsafe {
            if let Some(mesh) = self.mesh.take() {
                mesh.drop(device);
            }
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.fonts_texture.drop(device);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}

mod mesh {

    use super::{vulkan::*, RendererVkContext};
    use ash::{
        version::{DeviceV1_0, InstanceV1_0},
        vk, Device,
    };
    use imgui::{DrawData, DrawVert};
    use std::error::Error;

    pub struct Mesh {
        pub vertices: vk::Buffer,
        vertices_mem: vk::DeviceMemory,
        vertex_count: usize,
        pub indices: vk::Buffer,
        indices_mem: vk::DeviceMemory,
        index_count: usize,
    }

    impl Mesh {
        pub fn new<C: RendererVkContext>(
            vk_context: &C,
            draw_data: &DrawData,
        ) -> Result<Self, Box<dyn Error>> {
            let vertices = create_vertices(draw_data);
            let vertex_count = vertices.len();
            let indices = create_indices(draw_data);
            let index_count = indices.len();

            // Create a vertex buffer
            let memory_properties = unsafe {
                vk_context
                    .instance()
                    .get_physical_device_memory_properties(vk_context.physical_device())
            };
            let (vertices, vertices_mem) =
                create_vertex_buffer(&vertices, vk_context.device(), memory_properties)?;

            // Create an index buffer
            let (indices, indices_mem) =
                create_index_buffer(&indices, vk_context.device(), memory_properties)?;

            Ok(Mesh {
                vertices,
                vertices_mem,
                vertex_count,
                indices,
                indices_mem,
                index_count,
            })
        }

        pub fn refresh<C: RendererVkContext>(
            &mut self,
            vk_context: &C,
            draw_data: &DrawData,
        ) -> Result<(), Box<dyn Error>> {
            let memory_properties = unsafe {
                vk_context
                    .instance()
                    .get_physical_device_memory_properties(vk_context.physical_device())
            };

            let vertices = create_vertices(draw_data);
            if draw_data.total_vtx_count as usize > self.vertex_count {
                log::trace!("Resizing vertex buffers");
                self.destroy_vertices(vk_context.device());
                let vertex_count = vertices.len();
                let (vertices, vertices_mem) =
                    create_vertex_buffer(&vertices, vk_context.device(), memory_properties)?;

                self.vertices = vertices;
                self.vertices_mem = vertices_mem;
                self.vertex_count = vertex_count;
            } else {
                update_buffer_content(vk_context.device(), self.vertices_mem, &vertices)?;
            }

            let indices = create_indices(draw_data);
            if draw_data.total_idx_count as usize > self.index_count {
                log::trace!("Resizing index buffers");
                self.destroy_indices(vk_context.device());
                let index_count = indices.len();
                let (indices, indices_mem) =
                    create_index_buffer(&indices, vk_context.device(), memory_properties)?;
                self.indices = indices;
                self.indices_mem = indices_mem;
                self.index_count = index_count;
            } else {
                update_buffer_content(vk_context.device(), self.indices_mem, &indices)?;
            }

            Ok(())
        }

        pub fn drop(mut self, device: &Device) {
            self.destroy_indices(device);
            self.destroy_vertices(device);
        }

        fn destroy_vertices(&mut self, device: &Device) {
            unsafe {
                device.destroy_buffer(self.vertices, None);
                device.free_memory(self.vertices_mem, None);
            }
        }
        fn destroy_indices(&mut self, device: &Device) {
            unsafe {
                device.destroy_buffer(self.indices, None);
                device.free_memory(self.indices_mem, None);
            }
        }
    }

    fn create_vertices(draw_data: &DrawData) -> Vec<DrawVert> {
        let vertex_count = draw_data.total_vtx_count as usize;
        let mut vertices = Vec::with_capacity(vertex_count);
        for draw_list in draw_data.draw_lists() {
            vertices.extend_from_slice(draw_list.vtx_buffer());
        }
        vertices
    }

    fn create_indices(draw_data: &DrawData) -> Vec<u16> {
        let index_count = draw_data.total_idx_count as usize;
        let mut indices = Vec::with_capacity(index_count);
        for draw_list in draw_data.draw_lists() {
            indices.extend_from_slice(draw_list.idx_buffer());
        }
        indices
    }
}
