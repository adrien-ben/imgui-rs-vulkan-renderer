//! Vulkan helpers.
//!
//! A set of functions used to ease Vulkan resources creations. These are supposed to be internal but
//! are exposed since they might help users create descriptors sets when using the custom textures.

use crate::{Options, RendererResult};
use ash::{vk, Device};
pub(crate) use buffer::*;
use std::{ffi::CString, mem};
pub(crate) use texture::*;

/// Return a `&[u8]` for any sized object passed in.
pub(crate) unsafe fn any_as_u8_slice<T: Sized>(any: &T) -> &[u8] {
    let ptr = (any as *const T) as *const u8;
    std::slice::from_raw_parts(ptr, std::mem::size_of::<T>())
}

/// Create a descriptor set layout compatible with the graphics pipeline.
pub fn create_vulkan_descriptor_set_layout(
    device: &Device,
) -> RendererResult<vk::DescriptorSetLayout> {
    log::debug!("Creating vulkan descriptor set layout");
    let bindings = [vk::DescriptorSetLayoutBinding::builder()
        .binding(0)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
        .build()];

    let descriptor_set_create_info =
        vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);

    unsafe { Ok(device.create_descriptor_set_layout(&descriptor_set_create_info, None)?) }
}

pub(crate) fn create_vulkan_pipeline_layout(
    device: &Device,
    descriptor_set_layout: vk::DescriptorSetLayout,
) -> RendererResult<vk::PipelineLayout> {
    use ultraviolet::mat::Mat4;

    log::debug!("Creating vulkan pipeline layout");
    let push_const_range = [vk::PushConstantRange {
        stage_flags: vk::ShaderStageFlags::VERTEX,
        offset: 0,
        size: mem::size_of::<Mat4>() as u32,
    }];

    let descriptor_set_layouts = [descriptor_set_layout];
    let layout_info = vk::PipelineLayoutCreateInfo::builder()
        .set_layouts(&descriptor_set_layouts)
        .push_constant_ranges(&push_const_range);
    let pipeline_layout = unsafe { device.create_pipeline_layout(&layout_info, None)? };
    Ok(pipeline_layout)
}

pub(crate) fn create_vulkan_pipeline(
    device: &Device,
    pipeline_layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
    options: Options,
) -> RendererResult<vk::Pipeline> {
    let entry_point_name = CString::new("main").unwrap();

    let vertex_shader_source = std::include_bytes!("../shaders/shader.vert.spv");
    let fragment_shader_source = std::include_bytes!("../shaders/shader.frag.spv");

    let vertex_source = read_shader_from_source(vertex_shader_source)?;
    let vertex_create_info = vk::ShaderModuleCreateInfo::builder().code(&vertex_source);
    let vertex_module = unsafe { device.create_shader_module(&vertex_create_info, None)? };

    let fragment_source = read_shader_from_source(fragment_shader_source)?;
    let fragment_create_info = vk::ShaderModuleCreateInfo::builder().code(&fragment_source);
    let fragment_module = unsafe { device.create_shader_module(&fragment_create_info, None)? };

    let shader_states_infos = [
        vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex_module)
            .name(&entry_point_name)
            .build(),
        vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment_module)
            .name(&entry_point_name)
            .build(),
    ];

    let binding_desc = [vk::VertexInputBindingDescription::builder()
        .binding(0)
        .stride(20)
        .input_rate(vk::VertexInputRate::VERTEX)
        .build()];
    let attribute_desc = [
        vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(0)
            .build(),
        vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(8)
            .build(),
        vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(2)
            .format(vk::Format::R8G8B8A8_UNORM)
            .offset(16)
            .build(),
    ];

    let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::builder()
        .vertex_binding_descriptions(&binding_desc)
        .vertex_attribute_descriptions(&attribute_desc);

    let input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);

    let rasterizer_info = vk::PipelineRasterizationStateCreateInfo::builder()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::CLOCKWISE)
        .depth_bias_enable(false)
        .depth_bias_constant_factor(0.0)
        .depth_bias_clamp(0.0)
        .depth_bias_slope_factor(0.0);

    let viewports = [Default::default()];
    let scissors = [Default::default()];
    let viewport_info = vk::PipelineViewportStateCreateInfo::builder()
        .viewports(&viewports)
        .scissors(&scissors);

    let multisampling_info = vk::PipelineMultisampleStateCreateInfo::builder()
        .sample_shading_enable(false)
        .rasterization_samples(vk::SampleCountFlags::TYPE_1)
        .min_sample_shading(1.0)
        .alpha_to_coverage_enable(false)
        .alpha_to_one_enable(false);

    let color_blend_attachments = [vk::PipelineColorBlendAttachmentState::builder()
        .color_write_mask(
            vk::ColorComponentFlags::R
                | vk::ColorComponentFlags::G
                | vk::ColorComponentFlags::B
                | vk::ColorComponentFlags::A,
        )
        .blend_enable(true)
        .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .alpha_blend_op(vk::BlendOp::ADD)
        .build()];
    let color_blending_info = vk::PipelineColorBlendStateCreateInfo::builder()
        .logic_op_enable(false)
        .logic_op(vk::LogicOp::COPY)
        .attachments(&color_blend_attachments)
        .blend_constants([0.0, 0.0, 0.0, 0.0]);

    let depth_stencil_state_create_info = vk::PipelineDepthStencilStateCreateInfo::builder()
        .depth_test_enable(options.enable_depth_test)
        .depth_write_enable(options.enable_depth_write)
        .depth_compare_op(vk::CompareOp::ALWAYS)
        .depth_bounds_test_enable(false)
        .stencil_test_enable(false)
        .build();

    let dynamic_states = [vk::DynamicState::SCISSOR, vk::DynamicState::VIEWPORT];
    let dynamic_states_info =
        vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_states);

    let pipeline_info = [vk::GraphicsPipelineCreateInfo::builder()
        .stages(&shader_states_infos)
        .vertex_input_state(&vertex_input_info)
        .input_assembly_state(&input_assembly_info)
        .rasterization_state(&rasterizer_info)
        .viewport_state(&viewport_info)
        .multisample_state(&multisampling_info)
        .color_blend_state(&color_blending_info)
        .depth_stencil_state(&depth_stencil_state_create_info)
        .dynamic_state(&dynamic_states_info)
        .layout(pipeline_layout)
        .render_pass(render_pass)
        .subpass(0)
        .build()];

    let pipeline = unsafe {
        device
            .create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_info, None)
            .map_err(|e| e.1)?[0]
    };

    unsafe {
        device.destroy_shader_module(vertex_module, None);
        device.destroy_shader_module(fragment_module, None);
    }

    Ok(pipeline)
}

fn read_shader_from_source(source: &[u8]) -> RendererResult<Vec<u32>> {
    use std::io::Cursor;
    let mut cursor = Cursor::new(source);
    Ok(ash::util::read_spv(&mut cursor)?)
}

/// Create a descriptor pool of sets compatible with the graphics pipeline.
pub fn create_vulkan_descriptor_pool(
    device: &Device,
    max_sets: u32,
) -> RendererResult<vk::DescriptorPool> {
    log::debug!("Creating vulkan descriptor pool");

    let sizes = [vk::DescriptorPoolSize {
        ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        descriptor_count: 1,
    }];
    let create_info = vk::DescriptorPoolCreateInfo::builder()
        .pool_sizes(&sizes)
        .max_sets(max_sets)
        .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);
    unsafe { Ok(device.create_descriptor_pool(&create_info, None)?) }
}

/// Create a descriptor set compatible with the graphics pipeline from a texture.
pub fn create_vulkan_descriptor_set(
    device: &Device,
    set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    image_view: vk::ImageView,
    sampler: vk::Sampler,
) -> RendererResult<vk::DescriptorSet> {
    log::debug!("Creating vulkan descriptor set");

    let set = {
        let set_layouts = [set_layout];
        let allocate_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&set_layouts);

        unsafe { device.allocate_descriptor_sets(&allocate_info)?[0] }
    };

    unsafe {
        let image_info = [vk::DescriptorImageInfo {
            sampler,
            image_view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        }];

        let writes = [vk::WriteDescriptorSet::builder()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&image_info)
            .build()];
        device.update_descriptor_sets(&writes, &[])
    }

    Ok(set)
}

mod buffer {

    use crate::{
        renderer::allocator::{Allocate, Allocator, Memory},
        RendererResult,
    };
    use ash::vk;
    use ash::Device;
    use std::mem;

    pub fn create_and_fill_buffer<T>(
        device: &Device,
        allocator: &mut Allocator,
        data: &[T],
        usage: vk::BufferUsageFlags,
    ) -> RendererResult<(vk::Buffer, Memory)>
    where
        T: Copy,
    {
        let size = data.len() * mem::size_of::<T>();
        let (buffer, memory) = allocator.create_buffer(device, size, usage)?;
        allocator.update_buffer(device, &memory, data)?;
        Ok((buffer, memory))
    }
}

mod texture {

    use super::buffer::*;
    use crate::renderer::allocator::{Allocate, Allocator, Memory};
    use crate::RendererResult;
    use ash::vk;
    use ash::Device;

    /// Helper struct representing a sampled texture.
    pub struct Texture {
        pub image: vk::Image,
        image_mem: Memory,
        pub image_view: vk::ImageView,
        pub sampler: vk::Sampler,
    }

    impl Texture {
        /// Create a texture from an `u8` array containing an rgba image.
        ///
        /// The image data is device local and it's format is R8G8B8A8_UNORM.
        ///     
        /// # Arguments
        ///
        /// * `device` - The Vulkan logical device.
        /// * `queue` - The queue with transfer capabilities to execute commands.
        /// * `command_pool` - The command pool used to create a command buffer used to record commands.
        /// * `allocator` - Allocator used to allocate memory for the image.
        /// * `width` - The width of the image.
        /// * `height` - The height of the image.
        /// * `data` - The image data.
        pub fn from_rgba8(
            device: &Device,
            queue: vk::Queue,
            command_pool: vk::CommandPool,
            allocator: &mut Allocator,
            width: u32,
            height: u32,
            data: &[u8],
        ) -> RendererResult<Self> {
            let (texture, staging_buff, staging_mem) =
                execute_one_time_commands(device, queue, command_pool, |buffer| {
                    Self::cmd_from_rgba(device, allocator, buffer, width, height, data)
                })??;

            allocator.destroy_buffer(device, staging_buff, staging_mem)?;

            Ok(texture)
        }

        fn cmd_from_rgba(
            device: &Device,
            allocator: &mut Allocator,
            command_buffer: vk::CommandBuffer,
            width: u32,
            height: u32,
            data: &[u8],
        ) -> RendererResult<(Self, vk::Buffer, Memory)> {
            let (buffer, buffer_mem) = create_and_fill_buffer(
                device,
                allocator,
                data,
                vk::BufferUsageFlags::TRANSFER_SRC,
            )?;

            let (image, image_mem) = allocator.create_image(device, width, height)?;

            // Transition the image layout and copy the buffer into the image
            // and transition the layout again to be readable from fragment shader.
            {
                let mut barrier = vk::ImageMemoryBarrier::builder()
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(image)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .build();

                unsafe {
                    device.cmd_pipeline_barrier(
                        command_buffer,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[barrier],
                    )
                };

                let region = vk::BufferImageCopy::builder()
                    .buffer_offset(0)
                    .buffer_row_length(0)
                    .buffer_image_height(0)
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                    .image_extent(vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    })
                    .build();
                unsafe {
                    device.cmd_copy_buffer_to_image(
                        command_buffer,
                        buffer,
                        image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &[region],
                    )
                }

                barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
                barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
                barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
                barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

                unsafe {
                    device.cmd_pipeline_barrier(
                        command_buffer,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::FRAGMENT_SHADER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[barrier],
                    )
                };
            }

            let image_view = {
                let create_info = vk::ImageViewCreateInfo::builder()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });

                unsafe { device.create_image_view(&create_info, None)? }
            };

            let sampler = {
                let sampler_info = vk::SamplerCreateInfo::builder()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::REPEAT)
                    .address_mode_v(vk::SamplerAddressMode::REPEAT)
                    .address_mode_w(vk::SamplerAddressMode::REPEAT)
                    .anisotropy_enable(false)
                    .max_anisotropy(1.0)
                    .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
                    .unnormalized_coordinates(false)
                    .compare_enable(false)
                    .compare_op(vk::CompareOp::ALWAYS)
                    .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                    .mip_lod_bias(0.0)
                    .min_lod(0.0)
                    .max_lod(1.0);
                unsafe { device.create_sampler(&sampler_info, None)? }
            };

            let texture = Self {
                image,
                image_mem,
                image_view,
                sampler,
            };

            Ok((texture, buffer, buffer_mem))
        }

        /// Free texture's resources.
        pub fn destroy(self, device: &Device, allocator: &mut Allocator) -> RendererResult<()> {
            unsafe {
                device.destroy_sampler(self.sampler, None);
                device.destroy_image_view(self.image_view, None);
                allocator.destroy_image(device, self.image, self.image_mem)?;
            }
            Ok(())
        }
    }

    fn execute_one_time_commands<R, F: FnOnce(vk::CommandBuffer) -> R>(
        device: &Device,
        queue: vk::Queue,
        pool: vk::CommandPool,
        executor: F,
    ) -> RendererResult<R> {
        let command_buffer = {
            let alloc_info = vk::CommandBufferAllocateInfo::builder()
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_pool(pool)
                .command_buffer_count(1);

            unsafe { device.allocate_command_buffers(&alloc_info)?[0] }
        };
        let command_buffers = [command_buffer];

        // Begin recording
        {
            let begin_info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            unsafe { device.begin_command_buffer(command_buffer, &begin_info)? };
        }

        // Execute user function
        let executor_result = executor(command_buffer);

        // End recording
        unsafe { device.end_command_buffer(command_buffer)? };

        // Submit and wait
        {
            let submit_info = vk::SubmitInfo::builder()
                .command_buffers(&command_buffers)
                .build();
            let submit_infos = [submit_info];
            unsafe {
                device.queue_submit(queue, &submit_infos, vk::Fence::null())?;
                device.queue_wait_idle(queue)?;
            };
        }

        // Free
        unsafe { device.free_command_buffers(pool, &command_buffers) };

        Ok(executor_result)
    }
}
