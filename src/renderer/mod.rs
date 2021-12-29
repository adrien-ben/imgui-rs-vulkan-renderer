mod allocator;
pub mod vulkan;

use crate::RendererError;
use ash::{vk, Device};
use imgui::{Context, DrawCmd, DrawCmdParams, DrawData, TextureId, Textures};
use mesh::*;
use ultraviolet::projection::orthographic_vk;
use vulkan::*;

use self::allocator::Allocator;

#[cfg(not(feature = "gpu-allocator"))]
use ash::Instance;

#[cfg(feature = "gpu-allocator")]
use {
    gpu_allocator::vulkan::Allocator as GpuAllocator,
    std::sync::{Arc, Mutex},
};

/// Convenient return type for function that can return a [`RendererError`].
///
/// [`RendererError`]: enum.RendererError.html
pub type RendererResult<T> = Result<T, RendererError>;

/// Vulkan renderer for imgui.
///
/// It records rendering command to the provided command buffer at each call to [`cmd_draw`].
///
/// The renderer holds a set of vertex/index buffers per in flight frames. Vertex and index buffers
/// are resized at each call to [`cmd_draw`] if draw data does not fit.
///
/// [`cmd_draw`]: #method.cmd_draw
pub struct Renderer {
    device: Device,
    allocator: Allocator,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    fonts_texture: Option<Texture>,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    textures: Textures<vk::DescriptorSet>,
    in_flight_frames: usize,
    frames: Option<Frames>,
}

impl Renderer {
    /// Initialize and return a new instance of the renderer.
    ///
    /// At initialization all Vulkan resources are initialized and font texture is created and
    /// uploaded to the gpu. Vertex and index buffers are not created yet.
    ///
    /// # Arguments
    ///
    /// * `instance` - A reference to a Vulkan instance.
    /// * `physical_device` - A Vulkan physical device.
    /// * `device` - A Vulkan device.
    /// * `queue` - A Vulkan queue.
    ///             It will be used to submit commands during initialization to upload
    ///             data to the gpu. The type of queue must be supported by the following
    ///             commands: [vkCmdCopyBufferToImage](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdCopyBufferToImage.html),
    ///             [vkCmdPipelineBarrier](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdPipelineBarrier.html)
    /// * `command_pool` - A Vulkan command pool used to allocate command buffers to upload textures to the gpu.
    /// * `in_flight_frames` - The number of in flight frames of the application.
    /// * `render_pass` - The render pass used to render the gui.
    /// * `imgui` - The imgui context.
    ///
    /// # Errors
    ///
    /// * [`RendererError`] - If the number of in flight frame in incorrect.
    /// * [`RendererError`] - If any Vulkan or io error is encountered during initialization.
    #[cfg(not(feature = "gpu-allocator"))]
    pub fn new(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        device: Device,
        queue: vk::Queue,
        command_pool: vk::CommandPool,
        in_flight_frames: usize,
        render_pass: vk::RenderPass,
        imgui: &mut Context,
    ) -> RendererResult<Self> {
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        Self::from_allocator(
            device,
            queue,
            command_pool,
            Allocator::new(memory_properties),
            in_flight_frames,
            render_pass,
            imgui,
        )
    }

    /// Initialize and return a new instance of the renderer.
    ///
    /// At initialization all Vulkan resources are initialized and font texture is created and
    /// uploaded to the gpu. Vertex and index buffers are not created yet.
    ///
    /// # Arguments
    ///
    /// * `gpu_allocator` - The allocator that will be used to allocator buffer and image memory.
    /// * `device` - A Vulkan device.
    /// * `queue` - A Vulkan queue.
    ///             It will be used to submit commands during initialization to upload
    ///             data to the gpu. The type of queue must be supported by the following
    ///             commands: [vkCmdCopyBufferToImage](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdCopyBufferToImage.html),
    ///             [vkCmdPipelineBarrier](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdPipelineBarrier.html)
    /// * `command_pool` - A Vulkan command pool used to allocate command buffers to upload textures to the gpu.
    /// * `in_flight_frames` - The number of in flight frames of the application.
    /// * `render_pass` - The render pass used to render the gui.
    /// * `imgui` - The imgui context.
    ///
    /// # Errors
    ///
    /// * [`RendererError`] - If the number of in flight frame in incorrect.
    /// * [`RendererError`] - If any Vulkan or io error is encountered during initialization.
    #[cfg(feature = "gpu-allocator")]
    pub fn new(
        gpu_allocator: Arc<Mutex<GpuAllocator>>, // TODO: Another way ?
        device: Device,
        queue: vk::Queue,
        command_pool: vk::CommandPool,
        in_flight_frames: usize,
        render_pass: vk::RenderPass,
        imgui: &mut Context,
    ) -> RendererResult<Self> {
        Self::from_allocator(
            device,
            queue,
            command_pool,
            Allocator::new(gpu_allocator),
            in_flight_frames,
            render_pass,
            imgui,
        )
    }

    fn from_allocator(
        device: Device,
        queue: vk::Queue,
        command_pool: vk::CommandPool,
        mut allocator: Allocator,
        in_flight_frames: usize,
        render_pass: vk::RenderPass,
        imgui: &mut Context,
    ) -> RendererResult<Self> {
        if in_flight_frames == 0 {
            return Err(RendererError::Init(String::from(
                "'in_flight_frames' parameter should be at least one",
            )));
        }

        // Descriptor set layout
        let descriptor_set_layout = create_vulkan_descriptor_set_layout(&device)?;

        // Pipeline and layout
        let pipeline_layout = create_vulkan_pipeline_layout(&device, descriptor_set_layout)?;
        let pipeline = create_vulkan_pipeline(&device, pipeline_layout, render_pass)?;

        // Fonts texture
        let fonts_texture = {
            let mut fonts = imgui.fonts();
            let atlas_texture = fonts.build_rgba32_texture();

            Texture::from_rgba8(
                &device,
                queue,
                command_pool,
                &mut allocator,
                atlas_texture.width,
                atlas_texture.height,
                atlas_texture.data,
            )?
        };

        let mut fonts = imgui.fonts();
        fonts.tex_id = TextureId::from(usize::MAX);

        // Descriptor pool
        let descriptor_pool = create_vulkan_descriptor_pool(&device, 1)?;

        // Descriptor set
        let descriptor_set = create_vulkan_descriptor_set(
            &device,
            descriptor_set_layout,
            descriptor_pool,
            fonts_texture.image_view,
            fonts_texture.sampler,
        )?;

        // Textures
        let textures = Textures::new();

        Ok(Self {
            device,
            allocator,
            pipeline,
            pipeline_layout,
            descriptor_set_layout,
            fonts_texture: Some(fonts_texture),
            descriptor_pool,
            descriptor_set,
            textures,
            in_flight_frames,
            frames: None,
        })
    }

    /// Change the render pass to render to.
    ///
    /// Useful if you need to render to a new render pass but don't want to rebuild
    /// the entire renderer. It will rebuild the graphics pipeline from scratch so it
    /// is an expensive operation.
    ///
    /// # Arguments
    ///
    /// * `render_pass` - The render pass used to render the gui.
    ///
    /// # Errors
    ///
    /// * [`RendererError`] - If any Vulkan error is encountered during pipeline creation.
    pub fn set_render_pass(&mut self, render_pass: vk::RenderPass) -> RendererResult<()> {
        unsafe { self.device.destroy_pipeline(self.pipeline, None) };
        self.pipeline = create_vulkan_pipeline(&self.device, self.pipeline_layout, render_pass)?;
        Ok(())
    }

    /// Returns the texture mapping used by the renderer to lookup textures.
    ///
    /// Textures are provided by the application as `vk::DescriptorSet`s.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let descriptor_set = ...;
    /// // Insert a vk::DescriptorSet in the renderer textures map.
    /// // The renderer returns a generated texture id.
    /// let texture_id = renderer.textures().insert(descriptor_set);
    /// ...
    /// // Create an `Image` that references the texture by its id.
    /// Image::new(texture_id, [100, 100]).build(&ui);
    /// ```
    ///
    /// # Caveat
    ///
    /// Provided `vk::DescriptorSet`s must be created with a descriptor set layout that is compatible with the one used by the renderer.
    /// See [Pipeline Layout Compatibility](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/html/vkspec.html#descriptorsets-compatibility).
    pub fn textures(&mut self) -> &mut Textures<vk::DescriptorSet> {
        &mut self.textures
    }

    fn lookup_descriptor_set(&self, texture_id: TextureId) -> RendererResult<vk::DescriptorSet> {
        if texture_id.id() == usize::MAX {
            Ok(self.descriptor_set)
        } else if let Some(descriptor_set) = self.textures.get(texture_id) {
            Ok(*descriptor_set)
        } else {
            Err(RendererError::BadTexture(texture_id))
        }
    }

    /// Update the fonts texture after having added new fonts to imgui.
    ///
    /// # Arguments
    ///
    /// * `queue` - A Vulkan queue.
    ///             It will be used to submit commands during initialization to upload
    ///             data to the gpu. The type of queue must be supported by the following
    ///             commands: [vkCmdCopyBufferToImage](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdCopyBufferToImage.html),
    ///             [vkCmdPipelineBarrier](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdPipelineBarrier.html)
    /// * `command_pool` - A Vulkan command pool used to allocate command buffers to upload textures to the gpu.
    /// * `imgui` - The imgui context.
    ///
    /// # Errors
    ///
    /// * [`RendererError`] - If any error is encountered during texture update.
    pub fn update_fonts_texture(
        &mut self,
        queue: vk::Queue,
        command_pool: vk::CommandPool,
        imgui: &mut Context,
    ) -> RendererResult<()> {
        // Generate the new fonts texture
        let fonts_texture = {
            let mut fonts = imgui.fonts();
            let atlas_texture = fonts.build_rgba32_texture();

            Texture::from_rgba8(
                &self.device,
                queue,
                command_pool,
                &mut self.allocator,
                atlas_texture.width,
                atlas_texture.height,
                atlas_texture.data,
            )?
        };

        let mut fonts = imgui.fonts();
        fonts.tex_id = TextureId::from(usize::MAX);

        // Free Descriptor set the create a new one
        let old_descriptor_set = self.descriptor_set;
        unsafe {
            self.device
                .free_descriptor_sets(self.descriptor_pool, &[old_descriptor_set])?
        };
        self.descriptor_set = create_vulkan_descriptor_set(
            &self.device,
            self.descriptor_set_layout,
            self.descriptor_pool,
            fonts_texture.image_view,
            fonts_texture.sampler,
        )?;

        // Free old fonts texture
        let mut old_texture = self.fonts_texture.replace(fonts_texture);
        if let Some(texture) = old_texture.take() {
            texture.destroy(&self.device, &mut self.allocator)?;
        }

        Ok(())
    }

    /// Record commands required to render the gui.RendererError.
    ///
    /// # Arguments
    ///
    /// * `command_buffer` - The Vulkan command buffer that command will be recorded to.
    /// * `draw_data` - A reference to the imgui `DrawData` containing rendering data.
    ///
    /// # Errors
    ///
    /// * [`RendererError`] - If any Vulkan error is encountered during command recording.
    pub fn cmd_draw(
        &mut self,
        command_buffer: vk::CommandBuffer,
        draw_data: &DrawData,
    ) -> RendererResult<()> {
        if draw_data.total_vtx_count == 0 {
            return Ok(());
        }

        if self.frames.is_none() {
            self.frames.replace(Frames::new(
                &self.device,
                &mut self.allocator,
                draw_data,
                self.in_flight_frames,
            )?);
        }

        let mesh = self.frames.as_mut().unwrap().next();
        mesh.update(&self.device, &mut self.allocator, draw_data)?;

        unsafe {
            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
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

        unsafe { self.device.cmd_set_viewport(command_buffer, 0, &viewports) };

        // Ortho projection
        let projection = orthographic_vk(
            0.0,
            draw_data.display_size[0],
            0.0,
            -draw_data.display_size[1],
            -1.0,
            1.0,
        );
        unsafe {
            let push = any_as_u8_slice(&projection);
            self.device.cmd_push_constants(
                command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                push,
            )
        };

        unsafe {
            self.device.cmd_bind_index_buffer(
                command_buffer,
                mesh.indices,
                0,
                vk::IndexType::UINT16,
            )
        };

        unsafe {
            self.device
                .cmd_bind_vertex_buffers(command_buffer, 0, &[mesh.vertices], &[0])
        };

        let mut index_offset = 0;
        let mut vertex_offset = 0;
        let mut current_texture_id: Option<TextureId> = None;
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
                                texture_id,
                                vtx_offset,
                                idx_offset,
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
                            self.device.cmd_set_scissor(command_buffer, 0, &scissors);
                        }

                        if Some(texture_id) != current_texture_id {
                            let descriptor_set = self.lookup_descriptor_set(texture_id)?;
                            unsafe {
                                self.device.cmd_bind_descriptor_sets(
                                    command_buffer,
                                    vk::PipelineBindPoint::GRAPHICS,
                                    self.pipeline_layout,
                                    0,
                                    &[descriptor_set],
                                    &[],
                                )
                            };
                            current_texture_id = Some(texture_id);
                        }

                        unsafe {
                            self.device.cmd_draw_indexed(
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
}

impl Drop for Renderer {
    fn drop(&mut self) {
        log::debug!("Destroying ImGui Renderer");
        let device = &self.device;

        unsafe {
            if let Some(frames) = self.frames.take() {
                frames
                    .destroy(device, &mut self.allocator)
                    .expect("Failed to destroy frame data");
            }
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.fonts_texture
                .take()
                .unwrap()
                .destroy(device, &mut self.allocator)
                .expect("Failed to fronts data");
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}

// Structure holding data for all frames in flight.
struct Frames {
    index: usize,
    count: usize,
    meshes: Vec<Mesh>,
}

impl Frames {
    fn new(
        device: &Device,
        allocator: &mut Allocator,
        draw_data: &DrawData,
        count: usize,
    ) -> RendererResult<Self> {
        let meshes = (0..count)
            .map(|_| Mesh::new(device, allocator, draw_data))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            index: 0,
            count,
            meshes,
        })
    }

    fn next(&mut self) -> &mut Mesh {
        let result = &mut self.meshes[self.index];
        self.index = (self.index + 1) % self.count;
        result
    }

    fn destroy(self, device: &Device, allocator: &mut Allocator) -> RendererResult<()> {
        for mesh in self.meshes.into_iter() {
            mesh.destroy(device, allocator)?;
        }
        Ok(())
    }
}

mod mesh {

    use super::allocator::{Allocate, Allocator, Memory};
    use super::vulkan::*;
    use crate::RendererResult;
    use ash::{vk, Device};
    use imgui::{DrawData, DrawVert};
    use std::mem::size_of;

    /// Vertex and index buffer resources for one frame in flight.
    pub struct Mesh {
        pub vertices: vk::Buffer,
        vertices_mem: Memory,
        vertex_count: usize,
        pub indices: vk::Buffer,
        indices_mem: Memory,
        index_count: usize,
    }

    impl Mesh {
        pub fn new(
            device: &Device,
            allocator: &mut Allocator,
            draw_data: &DrawData,
        ) -> RendererResult<Self> {
            let vertices = create_vertices(draw_data);
            let vertex_count = vertices.len();
            let indices = create_indices(draw_data);
            let index_count = indices.len();

            // Create a vertex buffer
            let (vertices, vertices_mem) = create_and_fill_buffer(
                device,
                allocator,
                &vertices,
                vk::BufferUsageFlags::VERTEX_BUFFER,
            )?;

            // Create an index buffer
            let (indices, indices_mem) = create_and_fill_buffer(
                device,
                allocator,
                &indices,
                vk::BufferUsageFlags::INDEX_BUFFER,
            )?;

            Ok(Mesh {
                vertices,
                vertices_mem,
                vertex_count,
                indices,
                indices_mem,
                index_count,
            })
        }

        pub fn update(
            &mut self,
            device: &Device,
            allocator: &mut Allocator,
            draw_data: &DrawData,
        ) -> RendererResult<()> {
            let vertices = create_vertices(draw_data);
            if draw_data.total_vtx_count as usize > self.vertex_count {
                log::trace!("Resizing vertex buffers");

                let vertex_count = vertices.len();
                let size = vertex_count * size_of::<DrawVert>();
                let (vertices, vertices_mem) =
                    allocator.create_buffer(device, size, vk::BufferUsageFlags::VERTEX_BUFFER)?;

                self.vertex_count = vertex_count;

                let old_vertices = self.vertices;
                self.vertices = vertices;

                let old_vertices_mem = std::mem::replace(&mut self.vertices_mem, vertices_mem);

                allocator.destroy_buffer(device, old_vertices, old_vertices_mem)?;
            }
            allocator.update_buffer(device, &self.vertices_mem, &vertices)?;

            let indices = create_indices(draw_data);
            if draw_data.total_idx_count as usize > self.index_count {
                log::trace!("Resizing index buffers");

                let index_count = indices.len();
                let size = index_count * size_of::<u16>();
                let (indices, indices_mem) =
                    allocator.create_buffer(device, size, vk::BufferUsageFlags::INDEX_BUFFER)?;

                self.index_count = index_count;

                let old_indices = self.indices;
                self.indices = indices;

                let old_indices_mem = std::mem::replace(&mut self.indices_mem, indices_mem);

                allocator.destroy_buffer(device, old_indices, old_indices_mem)?;
            }
            allocator.update_buffer(device, &self.indices_mem, &indices)?;

            Ok(())
        }

        pub fn destroy(self, device: &Device, allocator: &mut Allocator) -> RendererResult<()> {
            allocator.destroy_buffer(device, self.vertices, self.vertices_mem)?;
            allocator.destroy_buffer(device, self.indices, self.indices_mem)?;
            Ok(())
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
