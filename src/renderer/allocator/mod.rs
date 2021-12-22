mod default;

use crate::RendererResult;
use ash::{vk, Device};

use self::default::DefaultAllocator;

/// Abstraction over memory used by Vulkan resources.
pub enum Memory {
    DeviceMemory(vk::DeviceMemory),
}

/// Base allocator trait for all implementations.
pub trait AllocatorTrait {
    /// Create a Vulkan buffer.
    ///
    /// # Arguments
    ///
    /// * `vk_context` - A reference to a type implementing the [`RendererVkContext`] trait.
    /// * `size` - The size in bytes of the buffer.
    /// * `usage` - The buffer usage flags.
    fn create_buffer(
        &self,
        device: &Device,
        size: usize,
        usage: vk::BufferUsageFlags,
    ) -> RendererResult<(vk::Buffer, Memory)>;

    /// Create a Vulkan image.
    ///
    /// This creates a 2D RGBA8_UNORM image with TRANSFER_DST and SAMPLED flags.
    ///
    /// # Arguments
    ///
    /// * `vk_context` - A reference to a type implementing the [`RendererVkContext`] trait.
    /// * `width` - The width of the image to create.
    /// * `height` - The height of the image to create.
    fn create_image(
        &self,
        device: &Device,
        width: u32,
        height: u32,
    ) -> RendererResult<(vk::Image, Memory)>;

    /// Destroys a buffer.
    ///
    /// # Arguments
    ///
    /// * `vk_context` - A reference to a type implementing the [`RendererVkContext`] trait.
    /// * `buffer` - The buffer to destroy.
    /// * `memory` - The buffer memory to destroy.
    fn destroy_buffer(
        &self,
        device: &Device,
        buffer: vk::Buffer,
        memory: &Memory,
    ) -> RendererResult<()> {
        match memory {
            Memory::DeviceMemory(memory) => unsafe {
                device.destroy_buffer(buffer, None);
                device.free_memory(*memory, None);
            },
        }

        Ok(())
    }

    /// Destroys an image.
    ///
    /// # Arguments
    ///
    /// * `vk_context` - A reference to a type implementing the [`RendererVkContext`] trait.
    /// * `image` - The image to destroy.
    /// * `memory` - The image memory to destroy.
    fn destroy_image(
        &self,
        device: &Device,
        image: vk::Image,
        memory: &Memory,
    ) -> RendererResult<()> {
        match memory {
            Memory::DeviceMemory(memory) => unsafe {
                device.destroy_image(image, None);
                device.free_memory(*memory, None);
            },
        }

        Ok(())
    }

    /// Update buffer data
    ///
    /// # Arguments
    ///
    /// * `vk_context` - A reference to a type implementing the [`RendererVkContext`] trait.
    /// * `buffer_memory` - The memory of the buffer to update.
    /// * `data` - The data to update the buffer with.
    fn update_buffer<T: Copy>(
        &self,
        device: &Device,
        buffer_memory: &Memory,
        data: &[T],
    ) -> RendererResult<()> {
        let size = (data.len() * std::mem::size_of::<T>()) as _;
        unsafe {
            match buffer_memory {
                Memory::DeviceMemory(memory) => {
                    let data_ptr =
                        device.map_memory(*memory, 0, size, vk::MemoryMapFlags::empty())?;
                    let mut align =
                        ash::util::Align::new(data_ptr, std::mem::align_of::<T>() as _, size);
                    align.copy_from_slice(data);
                    device.unmap_memory(*memory);
                }
            }
        };
        Ok(())
    }
}

/// Vulkan resource allocator.
///
/// Create Vulkan resources using memory like buffers and images.
///
/// # Variants
///
/// * `Default` - Default allocator.
pub enum Allocator {
    Default(DefaultAllocator),
}

impl Allocator {
    /// Get a default allocator.
    pub fn defaut(memory_properties: vk::PhysicalDeviceMemoryProperties) -> Self {
        Self::Default(DefaultAllocator { memory_properties })
    }
}

impl AllocatorTrait for Allocator {
    fn create_buffer(
        &self,
        device: &Device,
        size: usize,
        usage: vk::BufferUsageFlags,
    ) -> RendererResult<(vk::Buffer, Memory)> {
        match self {
            Self::Default(allocator) => allocator.create_buffer(device, size, usage),
        }
    }

    fn create_image(
        &self,
        device: &Device,
        width: u32,
        height: u32,
    ) -> RendererResult<(vk::Image, Memory)> {
        match self {
            Self::Default(allocator) => allocator.create_image(device, width, height),
        }
    }
}
