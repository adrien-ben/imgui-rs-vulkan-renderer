use ash::prelude::VkResult;
use ash::{version::DeviceV1_0, vk, Device};
pub use buffer::*;
pub use texture::*;

pub fn execute_one_time_commands<R, F: FnOnce(vk::CommandBuffer) -> R>(
    device: &Device,
    queue: vk::Queue,
    pool: vk::CommandPool,
    executor: F,
) -> VkResult<R> {
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

pub fn create_vulkan_descriptor_set_layout(device: &Device) -> VkResult<vk::DescriptorSetLayout> {
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

pub fn create_vulkan_descriptor_set(
    device: &Device,
    set_layout: vk::DescriptorSetLayout,
    texture: &texture::Texture,
) -> VkResult<(vk::DescriptorPool, vk::DescriptorSet)> {
    log::debug!("Creating vulkan descriptor set");
    let descriptor_pool = {
        let sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1,
        }];
        let create_info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(&sizes)
            .max_sets(1);
        unsafe { device.create_descriptor_pool(&create_info, None)? }
    };

    let set = {
        let set_layouts = [set_layout];
        let allocate_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&set_layouts);

        unsafe { device.allocate_descriptor_sets(&allocate_info)?[0] }
    };

    unsafe {
        let image_info = [vk::DescriptorImageInfo {
            sampler: texture.sampler,
            image_view: texture.image_view,
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

    Ok((descriptor_pool, set))
}

mod buffer {

    //use crate::VkResult;
    use ash::prelude::VkResult;
    use ash::{version::DeviceV1_0, vk, Device};
    use std::mem;

    pub fn create_and_fill_buffer<T: Copy>(
        data: &[T],
        device: &Device,
        usage: vk::BufferUsageFlags,
        mem_properties: vk::PhysicalDeviceMemoryProperties,
    ) -> VkResult<(vk::Buffer, vk::DeviceMemory)> {
        let size = data.len() * mem::size_of::<T>();
        let (buffer, memory) = create_buffer(size, device, usage, mem_properties)?;
        update_buffer_content(device, memory, data)?;
        Ok((buffer, memory))
    }

    pub fn create_buffer(
        size: usize,
        device: &Device,
        usage: vk::BufferUsageFlags,
        mem_properties: vk::PhysicalDeviceMemoryProperties,
    ) -> VkResult<(vk::Buffer, vk::DeviceMemory)> {
        let buffer_info = vk::BufferCreateInfo::builder()
            .size(size as _)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .build();
        let buffer = unsafe { device.create_buffer(&buffer_info, None)? };

        let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let mem_type = find_memory_type(
            mem_requirements,
            mem_properties,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        let alloc_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(mem_requirements.size)
            .memory_type_index(mem_type);
        let memory = unsafe { device.allocate_memory(&alloc_info, None)? };
        unsafe { device.bind_buffer_memory(buffer, memory, 0)? };

        Ok((buffer, memory))
    }

    pub fn update_buffer_content<T: Copy>(
        device: &Device,
        buffer_memory: vk::DeviceMemory,
        data: &[T],
    ) -> VkResult<()> {
        unsafe {
            let size = (data.len() * mem::size_of::<T>()) as _;

            let data_ptr =
                device.map_memory(buffer_memory, 0, size, vk::MemoryMapFlags::empty())?;
            let mut align = ash::util::Align::new(data_ptr, mem::align_of::<T>() as _, size);
            align.copy_from_slice(&data);
            device.unmap_memory(buffer_memory);
        };
        Ok(())
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

    use super::buffer::*;
    use ash::prelude::VkResult;
    use ash::vk;
    use ash::{version::DeviceV1_0, Device};

    pub struct Texture {
        buffer: vk::Buffer,
        buffer_mem: vk::DeviceMemory,
        pub image: vk::Image,
        image_mem: vk::DeviceMemory,
        pub image_view: vk::ImageView,
        pub sampler: vk::Sampler,
    }

    impl Texture {
        pub fn cmd_from_rgba(
            device: &Device,
            command_buffer: vk::CommandBuffer,
            mem_properties: vk::PhysicalDeviceMemoryProperties,
            width: u32,
            height: u32,
            format: vk::Format,
            data: &[u8],
        ) -> VkResult<Self> {
            let (buffer, buffer_mem) = create_and_fill_buffer(
                data,
                device,
                vk::BufferUsageFlags::TRANSFER_SRC,
                mem_properties,
            )?;

            let (image, image_mem) = {
                let extent = vk::Extent3D {
                    width: width,
                    height: height,
                    depth: 1,
                };

                let image_info = vk::ImageCreateInfo::builder()
                    .image_type(vk::ImageType::TYPE_2D)
                    .extent(extent)
                    .mip_levels(1)
                    .array_layers(1)
                    .format(format)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .flags(vk::ImageCreateFlags::empty());

                let image = unsafe { device.create_image(&image_info, None)? };
                let mem_requirements = unsafe { device.get_image_memory_requirements(image) };
                let mem_type_index = find_memory_type(
                    mem_requirements,
                    mem_properties,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                );

                let alloc_info = vk::MemoryAllocateInfo::builder()
                    .allocation_size(mem_requirements.size)
                    .memory_type_index(mem_type_index);
                let memory = unsafe {
                    let mem = device.allocate_memory(&alloc_info, None)?;
                    device.bind_image_memory(image, mem, 0)?;
                    mem
                };

                (image, memory)
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
                        width: width,
                        height: height,
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
                    .format(format)
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

            Ok(Self {
                buffer,
                buffer_mem,
                image,
                image_mem,
                image_view,
                sampler,
            })
        }
        pub fn destroy(&mut self, device: &Device) {
            unsafe {
                device.destroy_sampler(self.sampler, None);
                device.destroy_image_view(self.image_view, None);
                device.destroy_image(self.image, None);
                device.free_memory(self.image_mem, None);
                device.destroy_buffer(self.buffer, None);
                device.free_memory(self.buffer_mem, None);
            }
        }
    }
}
