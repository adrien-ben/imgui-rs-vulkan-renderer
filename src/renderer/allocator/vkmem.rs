use super::Allocate;
use crate::{RendererError, RendererResult};
use ash::{vk, Device};
use std::sync::{Arc, Mutex, MutexGuard};
use vk_mem::{Allocation, AllocationCreateInfo, Allocator as GpuAllocator, MemoryUsage};

/// Abstraction over memory used by Vulkan resources.
pub type Memory = Allocation;

pub struct Allocator {
    pub allocator: Arc<Mutex<GpuAllocator>>,
}

impl Allocator {
    pub fn new(allocator: Arc<Mutex<vk_mem::Allocator>>) -> Self {
        Self { allocator }
    }

    fn get_allocator(&self) -> RendererResult<MutexGuard<GpuAllocator>> {
        self.allocator.lock().map_err(|e| {
            RendererError::Allocator(format!(
                "Failed to acquire lock on allocator: {}",
                e.to_string()
            ))
        })
    }
}

impl Allocate for Allocator {
    type Memory = Memory;

    fn create_buffer(
        &mut self,
        _device: &Device,
        size: usize,
        usage: vk::BufferUsageFlags,
    ) -> RendererResult<(vk::Buffer, Self::Memory)> {
        let buffer_info = vk::BufferCreateInfo::builder()
            .size(size as _)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .build();

        let buffer_alloc_info = AllocationCreateInfo {
            usage: MemoryUsage::CpuOnly,
            ..Default::default()
        };

        let allocator = self.get_allocator()?;

        let (buffer, allocation, buffer_alloc_info) =
            allocator.create_buffer(&buffer_info, &buffer_alloc_info)?;
        log::debug!("Allocated buffer. Allocation info: {:?}", buffer_alloc_info);

        Ok((buffer, allocation))
    }

    fn create_image(
        &mut self,
        _device: &Device,
        width: u32,
        height: u32,
    ) -> RendererResult<(vk::Image, Self::Memory)> {
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

        let allocator = self.get_allocator()?;

        let (image, allocation, image_alloc_info) =
            allocator.create_image(&image_info, &image_alloc_info)?;
        log::debug!("Allocated image. Allocation info: {:?}", image_alloc_info);

        Ok((image, allocation))
    }

    fn destroy_buffer(
        &mut self,
        _device: &Device,
        buffer: vk::Buffer,
        memory: Self::Memory,
    ) -> RendererResult<()> {
        let allocator = self.get_allocator()?;

        allocator.destroy_buffer(buffer, &memory);

        Ok(())
    }

    fn destroy_image(
        &mut self,
        _device: &Device,
        image: vk::Image,
        memory: Self::Memory,
    ) -> RendererResult<()> {
        let allocator = self.get_allocator()?;

        allocator.destroy_image(image, &memory);

        Ok(())
    }

    fn update_buffer<T: Copy>(
        &mut self,
        _device: &Device,
        memory: &Self::Memory,
        data: &[T],
    ) -> RendererResult<()> {
        let size = (data.len() * std::mem::size_of::<T>()) as _;

        let allocator = self.get_allocator()?;
        let data_ptr = allocator.map_memory(memory)? as *mut std::ffi::c_void;
        let mut align =
            unsafe { ash::util::Align::new(data_ptr, std::mem::align_of::<T>() as _, size) };
        align.copy_from_slice(data);
        allocator.unmap_memory(memory);

        Ok(())
    }
}
