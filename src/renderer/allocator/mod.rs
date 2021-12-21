mod default;

use crate::{RendererResult, RendererVkContext};
use ash::vk;

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
    fn create_buffer<C: RendererVkContext>(
        &self,
        vk_context: &C,
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
    fn create_image<C: RendererVkContext>(
        &self,
        vk_context: &C,
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
    fn destroy_buffer<C: RendererVkContext>(
        &self,
        vk_context: &C,
        buffer: vk::Buffer,
        memory: &Memory,
    ) -> RendererResult<()> {
        match memory {
            Memory::DeviceMemory(memory) => unsafe {
                let device = vk_context.device();
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
    fn destroy_image<C: RendererVkContext>(
        &self,
        vk_context: &C,
        image: vk::Image,
        memory: &Memory,
    ) -> RendererResult<()> {
        match memory {
            Memory::DeviceMemory(memory) => unsafe {
                let device = vk_context.device();
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
    fn update_buffer<C: RendererVkContext, T: Copy>(
        &self,
        vk_context: &C,
        buffer_memory: &Memory,
        data: &[T],
    ) -> RendererResult<()> {
        let size = (data.len() * std::mem::size_of::<T>()) as _;
        unsafe {
            match buffer_memory {
                Memory::DeviceMemory(memory) => {
                    let device = vk_context.device();
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
    pub fn defaut() -> Self {
        Self::Default(DefaultAllocator)
    }
}

impl AllocatorTrait for Allocator {
    fn create_buffer<C: RendererVkContext>(
        &self,
        vk_context: &C,
        size: usize,
        usage: vk::BufferUsageFlags,
    ) -> RendererResult<(vk::Buffer, Memory)> {
        match self {
            Self::Default(allocator) => allocator.create_buffer(vk_context, size, usage),
        }
    }

    fn create_image<C: RendererVkContext>(
        &self,
        vk_context: &C,
        width: u32,
        height: u32,
    ) -> RendererResult<(vk::Image, Memory)> {
        match self {
            Self::Default(allocator) => allocator.create_image(vk_context, width, height),
        }
    }
}
