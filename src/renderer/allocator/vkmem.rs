use crate::{RendererResult, RendererVkContext};
use ash::vk;
use vk_mem::{AllocationCreateInfo, MemoryUsage};

use super::{AllocatorTrait, Memory};

pub struct VkMemAllocator;

impl AllocatorTrait for VkMemAllocator {
    fn create_buffer<C: RendererVkContext>(
        &self,
        vk_context: &C,
        size: usize,
        usage: vk::BufferUsageFlags,
    ) -> RendererResult<(vk::Buffer, Memory)> {
        let buffer_info = vk::BufferCreateInfo::builder()
            .size(size as _)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .build();

        let buffer_alloc_info = AllocationCreateInfo {
            usage: MemoryUsage::CpuOnly,
            ..Default::default()
        };

        let (buffer, buffer_alloc, buffer_alloc_info) = vk_context
            .vk_mem_allocator()
            .create_buffer(&buffer_info, &buffer_alloc_info)?;
        log::debug!("Allocated buffer. Allocation info: {:?}", buffer_alloc_info);

        Ok((buffer, Memory::VkMemAllocation(buffer_alloc)))
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

        let image_alloc_info = AllocationCreateInfo {
            usage: MemoryUsage::GpuOnly,
            ..Default::default()
        };

        let (image, image_alloc, image_alloc_info) = vk_context
            .vk_mem_allocator()
            .create_image(&image_info, &image_alloc_info)?;
        log::debug!("Allocated image. Allocation info: {:?}", image_alloc_info);

        Ok((image, Memory::VkMemAllocation(image_alloc)))
    }
}
