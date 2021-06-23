use crate::{RendererResult, RendererVkContext};
use ash::version::{DeviceV1_0, InstanceV1_0};
use ash::vk;

use super::{AllocatorTrait, Memory};

pub struct DefaultAllocator;

impl DefaultAllocator {
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

impl AllocatorTrait for DefaultAllocator {
    fn create_buffer<C: RendererVkContext>(
        &self,
        vk_context: &C,
        size: usize,
        usage: vk::BufferUsageFlags,
    ) -> RendererResult<(vk::Buffer, Memory)> {
        let memory_properties = unsafe {
            vk_context
                .instance()
                .get_physical_device_memory_properties(vk_context.physical_device())
        };

        let buffer_info = vk::BufferCreateInfo::builder()
            .size(size as _)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .build();

        let device = vk_context.device();

        let buffer = unsafe { device.create_buffer(&buffer_info, None)? };

        let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let mem_type = Self::find_memory_type(
            mem_requirements,
            memory_properties,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        let alloc_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(mem_requirements.size)
            .memory_type_index(mem_type);
        let memory = unsafe { device.allocate_memory(&alloc_info, None)? };
        unsafe { device.bind_buffer_memory(buffer, memory, 0)? };

        Ok((buffer, Memory::DeviceMemory(memory)))
    }

    fn create_image<C: RendererVkContext>(
        &self,
        vk_context: &C,
        width: u32,
        height: u32,
    ) -> RendererResult<(vk::Image, Memory)> {
        let extent = vk::Extent3D {
            width,
            height,
            depth: 1,
        };

        let image_info = vk::ImageCreateInfo::builder()
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

        let device = vk_context.device();
        let image = unsafe { device.create_image(&image_info, None)? };
        let mem_requirements = unsafe { device.get_image_memory_requirements(image) };
        let memory_properties = unsafe {
            vk_context
                .instance()
                .get_physical_device_memory_properties(vk_context.physical_device())
        };
        let mem_type_index = Self::find_memory_type(
            mem_requirements,
            memory_properties,
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

        Ok((image, Memory::DeviceMemory(memory)))
    }
}
