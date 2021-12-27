mod default;

#[cfg(feature = "gpu-allocator")]
mod gpu;

use std::convert::{TryFrom, TryInto};

use crate::RendererResult;
use ash::{vk, Device};

use self::default::DefaultAllocator;

#[cfg(feature = "gpu-allocator")]
use {
    self::gpu::GpuAllocator,
    gpu_allocator::vulkan::Allocation,
    std::sync::{Arc, Mutex},
};

/// Abstraction over memory used by Vulkan resources.
pub enum Memory {
    DeviceMemory(vk::DeviceMemory),
    #[cfg(feature = "gpu-allocator")]
    GpuAllocation(Allocation),
}

impl TryFrom<Memory> for vk::DeviceMemory {
    type Error = String;

    fn try_from(memory: Memory) -> Result<Self, Self::Error> {
        match memory {
            Memory::DeviceMemory(memory) => Ok(memory),
            _ => Err("Incompatible memory type".into()),
        }
    }
}

impl<'a> TryFrom<&'a Memory> for &'a vk::DeviceMemory {
    type Error = String;

    fn try_from(memory: &'a Memory) -> Result<Self, Self::Error> {
        match memory {
            Memory::DeviceMemory(memory) => Ok(memory),
            _ => Err("Incompatible memory type".into()),
        }
    }
}

#[cfg(feature = "gpu-allocator")]
impl TryFrom<Memory> for Allocation {
    type Error = String;

    fn try_from(memory: Memory) -> Result<Self, Self::Error> {
        match memory {
            Memory::GpuAllocation(allocation) => Ok(allocation),
            _ => Err("Incompatible memory type".into()),
        }
    }
}

#[cfg(feature = "gpu-allocator")]
impl<'a> TryFrom<&'a Memory> for &'a Allocation {
    type Error = String;

    fn try_from(memory: &'a Memory) -> Result<Self, Self::Error> {
        match memory {
            Memory::GpuAllocation(allocation) => Ok(allocation),
            _ => Err("Incompatible memory type".into()),
        }
    }
}

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
        memory: &Self::Memory,
        data: &[T],
    ) -> RendererResult<()>;
}

/// Vulkan resource allocator.
///
/// Create Vulkan resources using memory like buffers and images.
///
/// # Variants
///
/// * `Default` - Default allocator.
/// * `Gpu` - Allocator using [`gpu-allocator`] internally
///
/// [`gpu-allocator`]: https://github.com/Traverse-Research/gpu-allocator

pub enum Allocator {
    Default(DefaultAllocator),
    #[cfg(feature = "gpu-allocator")]
    Gpu(GpuAllocator),
}

impl Allocator {
    /// Get a default allocator.
    pub fn defaut(memory_properties: vk::PhysicalDeviceMemoryProperties) -> Self {
        Self::Default(DefaultAllocator { memory_properties })
    }

    #[cfg(feature = "gpu-allocator")]
    pub fn gpu(allocator: Arc<Mutex<gpu_allocator::vulkan::Allocator>>) -> Self {
        Self::Gpu(GpuAllocator { allocator })
    }
}

impl Allocate for Allocator {
    type Memory = Memory;

    fn create_buffer(
        &mut self,
        device: &Device,
        size: usize,
        usage: vk::BufferUsageFlags,
    ) -> RendererResult<(vk::Buffer, Self::Memory)> {
        match self {
            Self::Default(allocator) => allocator
                .create_buffer(device, size, usage)
                .map(|(buffer, mem)| (buffer, Memory::DeviceMemory(mem))),
            #[cfg(feature = "gpu-allocator")]
            Self::Gpu(allocator) => allocator
                .create_buffer(device, size, usage)
                .map(|(buffer, mem)| (buffer, Memory::GpuAllocation(mem))),
        }
    }

    fn create_image(
        &mut self,
        device: &Device,
        width: u32,
        height: u32,
    ) -> RendererResult<(vk::Image, Self::Memory)> {
        match self {
            Self::Default(allocator) => allocator
                .create_image(device, width, height)
                .map(|(image, mem)| (image, Memory::DeviceMemory(mem))),
            #[cfg(feature = "gpu-allocator")]
            Self::Gpu(allocator) => allocator
                .create_image(device, width, height)
                .map(|(image, mem)| (image, Memory::GpuAllocation(mem))),
        }
    }

    fn destroy_buffer(
        &mut self,
        device: &Device,
        buffer: vk::Buffer,
        memory: Self::Memory,
    ) -> RendererResult<()> {
        match self {
            Self::Default(allocator) => {
                allocator.destroy_buffer(device, buffer, memory.try_into().unwrap())
            }
            #[cfg(feature = "gpu-allocator")]
            Self::Gpu(allocator) => {
                allocator.destroy_buffer(device, buffer, memory.try_into().unwrap())
            }
        }
    }

    fn destroy_image(
        &mut self,
        device: &Device,
        image: vk::Image,
        memory: Self::Memory,
    ) -> RendererResult<()> {
        match self {
            Self::Default(allocator) => {
                allocator.destroy_image(device, image, memory.try_into().unwrap())
            }
            #[cfg(feature = "gpu-allocator")]
            Self::Gpu(allocator) => {
                allocator.destroy_image(device, image, memory.try_into().unwrap())
            }
        }
    }

    fn update_buffer<T: Copy>(
        &mut self,
        device: &Device,
        memory: &Self::Memory,
        data: &[T],
    ) -> RendererResult<()> {
        match self {
            Self::Default(allocator) => {
                allocator.update_buffer(device, memory.try_into().unwrap(), data)
            }
            #[cfg(feature = "gpu-allocator")]
            Self::Gpu(allocator) => {
                allocator.update_buffer(device, memory.try_into().unwrap(), data)
            }
        }
    }
}
