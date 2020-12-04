//! Vulkan helpers.
//!
//! A set of functions used to ease Vulkan resources creations. These are supposed to be internal but
//! are exposed since they might help users create descriptors sets when using the custom textures.

use crate::RendererResult;
use ash::{version::DeviceV1_0, vk, Device};
pub(crate) use buffer::*;
use std::{ffi::CString, mem};
pub use texture::*;

#[cfg(feature = "vma")]
use vk_mem;

#[cfg(not(feature = "vma"))]
pub type Memory = vk::DeviceMemory;

#[cfg(not(feature = "vma"))]
pub type Allocator = Option<vk::AllocationCallbacks>;

#[cfg(feature = "vma")]
pub type Memory = vk_mem::Allocation;

#[cfg(feature = "vma")]
pub type Allocator = vk_mem::Allocator;

pub struct XX {}

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
        .color_write_mask(vk::ColorComponentFlags::all())
        .blend_enable(true)
        .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
        .alpha_blend_op(vk::BlendOp::ADD)
        .build()];
    let color_blending_info = vk::PipelineColorBlendStateCreateInfo::builder()
        .logic_op_enable(false)
        .logic_op(vk::LogicOp::COPY)
        .attachments(&color_blend_attachments)
        .blend_constants([0.0, 0.0, 0.0, 0.0]);

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
        .max_sets(max_sets);
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

    use super::{Allocator, Memory};
    use crate::RendererResult;
    use ash::prelude::VkResult;
    use ash::{version::DeviceV1_0, vk, Device};
    use core::ffi::c_void;
    use std::mem;

    #[cfg(feature = "vma")]
    use crate::error::RendererError;

    /*pub fn create_and_fill_buffer<T: Copy>(
        device: &Device,
        allocator: &Allocator,
        mem_properties: vk::PhysicalDeviceMemoryProperties,
        usage: vk::BufferUsageFlags,
        data: &[T],
    ) -> RendererResult<(vk::Buffer, Memory)> {
        let size = data.len() * mem::size_of::<T>();
        let (buffer, memory) = create_host_coherent_buffer(device, allocator, mem_properties, usage, size)?;
        update_buffer_content(device, allocator, &memory, data)?;
        Ok((buffer, memory))
    }*/

    pub fn create_host_coherent_buffer(
        device: &Device,
        allocator: &Allocator,
        mem_properties: vk::PhysicalDeviceMemoryProperties,
        usage: vk::BufferUsageFlags,
        size: usize,
    ) -> RendererResult<(vk::Buffer, Memory)> {
        let buffer_create_info = vk::BufferCreateInfo::builder()
            .size(size as _)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .build();
        create_buffer(device, allocator, mem_properties, &buffer_create_info)
    }

    pub fn create_staging_buffer(
        device: &Device,
        allocator: &Allocator,
        mem_properties: vk::PhysicalDeviceMemoryProperties,
        usage: vk::BufferUsageFlags,
        size: usize,
    ) -> RendererResult<(vk::Buffer, Memory)> {
        let buffer_create_info = vk::BufferCreateInfo::builder()
            .size(size as _)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .build();
        create_buffer(device, allocator, mem_properties, &buffer_create_info)
    }

    pub fn update_buffer_content<T: Copy>(
        device: &Device,
        allocator: &Allocator,
        buffer_memory: &Memory,
        data: &[T],
    ) -> RendererResult<()> {
        unsafe {
            let size = (data.len() * mem::size_of::<T>()) as _;

            let data_ptr = map_memory(
                device,
                allocator,
                buffer_memory,
                0,
                size,
                vk::MemoryMapFlags::empty(),
            )?;
            let mut align = ash::util::Align::new(data_ptr, mem::align_of::<T>() as _, size);
            align.copy_from_slice(&data);
            unmap_memory(device, allocator, buffer_memory);
        };
        Ok(())
    }

    #[cfg(not(feature = "vma"))]
    fn create_buffer(
        device: &Device,
        allocator: &Allocator,
        mem_properties: vk::PhysicalDeviceMemoryProperties,
        buffer_create_info: &vk::BufferCreateInfo,
    ) -> RendererResult<(vk::Buffer, Memory)> {
        let buffer = unsafe { device.create_buffer(&buffer_create_info, allocator.as_ref())? };

        let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let mem_type = find_memory_type(
            mem_requirements,
            mem_properties,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        let memory_allocate_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(mem_requirements.size)
            .memory_type_index(mem_type);
        let memory = unsafe { device.allocate_memory(&memory_allocate_info, allocator.as_ref())? };
        unsafe { device.bind_buffer_memory(buffer, memory, 0)? };

        Ok((buffer, memory))
    }

    #[cfg(not(feature = "vma"))]
    pub(crate) fn destroy_buffer(
        device: &Device,
        allocator: &Allocator,
        buffer: vk::Buffer,
        memory: &Memory,
    ) {
        unsafe {
            device.destroy_buffer(buffer, allocator.as_ref());
            device.free_memory(*memory, None);
        }
    }

    #[cfg(feature = "vma")]
    fn create_buffer(
        _device: &Device,
        allocator: &Allocator,
        _mem_properties: vk::PhysicalDeviceMemoryProperties,
        buffer_create_info: &vk::BufferCreateInfo,
    ) -> RendererResult<(vk::Buffer, Memory)> {
        let allocation_create_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::CpuToGpu,
            ..Default::default()
        };

        let (buffer, allocation, _allocation_info) = allocator
            .create_buffer(&buffer_create_info, &allocation_create_info)
            .unwrap();

        Ok((buffer, allocation))
    }

    #[cfg(feature = "vma")]
    pub(crate) fn destroy_buffer(
        _device: &Device,
        allocator: &Allocator,
        buffer: vk::Buffer,
        memory: &Memory,
    ) {
        allocator.destroy_buffer(buffer, memory);
    }

    #[cfg(not(feature = "vma"))]
    fn map_memory(
        device: &Device,
        _allocator: &Allocator,
        memory: &vk::DeviceMemory,
        offset: vk::DeviceSize,
        size: vk::DeviceSize,
        flags: vk::MemoryMapFlags,
    ) -> VkResult<*mut c_void> {
        unsafe { device.map_memory(*memory, offset, size, flags) }
    }

    #[cfg(not(feature = "vma"))]
    fn unmap_memory(device: &Device, _allocator: &Allocator, memory: &vk::DeviceMemory) {
        unsafe { device.unmap_memory(*memory) }
    }

    #[cfg(feature = "vma")]
    fn map_memory(
        _device: &Device,
        allocator: &Allocator,
        memory: &vk_mem::Allocation,
        _offset: vk::DeviceSize,
        _size: vk::DeviceSize,
        _flags: vk::MemoryMapFlags,
    ) -> VkResult<*mut c_void> {
        Ok(allocator.map_memory(memory).unwrap() as _)
    }

    #[cfg(feature = "vma")]
    fn unmap_memory(_device: &Device, allocator: &Allocator, memory: &vk_mem::Allocation) {
        allocator.unmap_memory(memory).unwrap()
    }

    pub fn find_memory_type(
        requirements: vk::MemoryRequirements,
        mem_properties: vk::PhysicalDeviceMemoryProperties,
        required_properties: vk::MemoryPropertyFlags,
    ) -> u32 {
        for i in 0..mem_properties.memory_type_count {
            if requirements.memory_type_bits & (1 << i) != 0
                && mem_properties.memory_types[i as usize]
                    .property_flags
                    .contains(required_properties)
            {
                return i;
            }
        }
        panic!("Failed to find suitable memory type.")
    }
}

mod texture {

    use super::{buffer::*, Allocator, Memory};
    use crate::RendererResult;
    use ash::vk;
    use ash::{version::DeviceV1_0, Device};

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
        /// * `transfer_queue` - The queue with transfer capabilities to execute commands.
        /// * `command_pool` - The command pool used to create a command buffer used to record commands.
        /// * `mem_properties` - The memory properties of the Vulkan physical device.
        /// * `width` - The width of the image.
        /// * `height` - The height of the image.
        /// * `data` - The image data.
        pub fn from_rgba8(
            device: &Device,
            transfer_queue: vk::Queue,
            command_pool: vk::CommandPool,
            mem_properties: vk::PhysicalDeviceMemoryProperties,
            allocator: &Allocator,
            width: u32,
            height: u32,
            data: &[u8],
        ) -> RendererResult<Self> {
            let (texture, staging_buff, staging_mem) =
                execute_one_time_commands(device, transfer_queue, command_pool, |buffer| {
                    Self::cmd_from_rgba(
                        device,
                        allocator,
                        mem_properties,
                        buffer,
                        width,
                        height,
                        data,
                    )
                })??;

            destroy_buffer(device, allocator, staging_buff, &staging_mem);

            Ok(texture)
        }

        fn cmd_from_rgba(
            device: &Device,
            allocator: &Allocator,
            mem_properties: vk::PhysicalDeviceMemoryProperties,
            command_buffer: vk::CommandBuffer,
            width: u32,
            height: u32,
            data: &[u8],
        ) -> RendererResult<(Self, vk::Buffer, Memory)> {
            // TODO this buffer should be CPU only ?
            let (buffer, buffer_mem) = create_staging_buffer(
                device,
                allocator,
                mem_properties,
                vk::BufferUsageFlags::TRANSFER_SRC,
                std::mem::size_of_val(data),
            )?;
            update_buffer_content(device, allocator, &buffer_mem, data)?;

            let (image, image_mem) = {
                let extent = vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                };

                let image_create_info = vk::ImageCreateInfo::builder()
                    .image_type(vk::ImageType::TYPE_2D)
                    .extent(extent)
                    .mip_levels(1)
                    .array_layers(1)
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .flags(vk::ImageCreateFlags::empty());

                create_image(device, allocator, mem_properties, &image_create_info)?
            };

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
        pub fn destroy(&mut self, device: &Device, allocator: &Allocator) {
            unsafe {
                device.destroy_sampler(self.sampler, None);
                device.destroy_image_view(self.image_view, None);
                destroy_image(device, allocator, self.image, &self.image_mem);
            }
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

    #[cfg(not(feature = "vma"))]
    fn create_image(
        device: &Device,
        allocator: &Allocator,
        mem_properties: vk::PhysicalDeviceMemoryProperties,
        image_create_info: &vk::ImageCreateInfo,
    ) -> RendererResult<(vk::Image, Memory)> {
        let image = unsafe { device.create_image(&image_create_info, None)? };
        let mem_requirements = unsafe { device.get_image_memory_requirements(image) };
        let mem_type_index = find_memory_type(
            mem_requirements,
            mem_properties,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );
        let memory_allocate_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(mem_requirements.size)
            .memory_type_index(mem_type_index);
        let memory = unsafe {
            let mem = device.allocate_memory(&memory_allocate_info, allocator.as_ref())?;
            device.bind_image_memory(image, mem, 0)?;
            mem
        };
        Ok((image, memory))
    }

    #[cfg(not(feature = "vma"))]
    fn destroy_image(device: &Device, allocator: &Allocator, image: vk::Image, memory: &Memory) {
        unsafe {
            device.destroy_image(image, allocator.as_ref());
            device.free_memory(*memory, None);
        }
    }

    #[cfg(feature = "vma")]
    fn create_image(
        _device: &Device,
        allocator: &Allocator,
        _mem_properties: vk::PhysicalDeviceMemoryProperties,
        image_create_info: &vk::ImageCreateInfo,
    ) -> RendererResult<(vk::Image, Memory)> {
        let allocation_create_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::GpuOnly,
            ..Default::default()
        };

        let (image, allocation, _allocation_info) = allocator
            .create_image(&image_create_info, &allocation_create_info)
            .unwrap();

        Ok((image, allocation))
    }

    #[cfg(feature = "vma")]
    fn destroy_image(_device: &Device, allocator: &Allocator, image: vk::Image, memory: &Memory) {
        allocator.destroy_image(image, memory);
    }
}
