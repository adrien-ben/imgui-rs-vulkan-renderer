#[cfg(not(any(feature = "gpu-allocator", feature = "vk-mem")))]
mod default;

#[cfg(not(any(feature = "gpu-allocator", feature = "vk-mem")))]
pub use self::default::{Allocator, Memory};

#[cfg(feature = "gpu-allocator")]
mod gpu;

#[cfg(feature = "gpu-allocator")]
pub use self::gpu::{Allocator, Memory};

#[cfg(feature = "vk-mem")]
mod vkmem;

#[cfg(feature = "vk-mem")]
pub use self::vkmem::{Allocator, Memory};

use crate::RendererResult;
use ash::{vk, Device};

/// Base allocator trait for all implementations.
pub trait Allocate {
    type Memory;

    /// Create a Vulkan buffer.
    ///
    /// # Arguments
    ///
    /// * `device` - A reference to Vulkan device.
    /// * `size` - The size in bytes of the buffer.
    /// * `usage` - The buffer usage flags.
    fn create_buffer(
        &mut self,
        device: &Device,
        size: usize,
        usage: vk::BufferUsageFlags,
    ) -> RendererResult<(vk::Buffer, Self::Memory)>;

    /// Create a Vulkan image.
    ///
    /// This creates a 2D RGBA8_UNORM image with TRANSFER_DST and SAMPLED flags.
    ///
    /// # Arguments
    ///
    /// * `device` - A reference to Vulkan device.
    /// * `width` - The width of the image to create.
    /// * `height` - The height of the image to create.
    fn create_image(
        &mut self,
        device: &Device,
        width: u32,
        height: u32,
    ) -> RendererResult<(vk::Image, Self::Memory)>;

    /// Destroys a buffer.
    ///
    /// # Arguments
    ///
    /// * `device` - A reference to Vulkan device.
    /// * `buffer` - The buffer to destroy.
    /// * `memory` - The buffer memory to destroy.
    fn destroy_buffer(
        &mut self,
        device: &Device,
        buffer: vk::Buffer,
        memory: Self::Memory,
    ) -> RendererResult<()>;

    /// Destroys an image.
    ///
    /// # Arguments
    ///
    /// * `device` - A reference to Vulkan device.
    /// * `image` - The image to destroy.
    /// * `memory` - The image memory to destroy.
    fn destroy_image(
        &mut self,
        device: &Device,
        image: vk::Image,
        memory: Self::Memory,
    ) -> RendererResult<()>;

    /// Update buffer data
    ///
    /// # Arguments
    ///
    /// * `device` - A reference to Vulkan device.
    /// * `memory` - The memory of the buffer to update.
    /// * `data` - The data to update the buffer with.
    fn update_buffer<T: Copy>(
        &mut self,
        device: &Device,
        memory: &mut Self::Memory,
        data: &[T],
    ) -> RendererResult<()>;
}
