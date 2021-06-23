mod default;
#[cfg(feature = "vkmem")]
mod vkmem;

use crate::{RendererResult, RendererVkContext};
use ash::version::DeviceV1_0;
use ash::vk;

pub use self::default::DefaultAllocator;
#[cfg(feature = "vkmem")]
pub use self::vkmem::VkMemAllocator;

pub enum Memory {
    DeviceMemory(vk::DeviceMemory),
    #[cfg(feature = "vkmem")]
    VkMemAllocation(vk_mem::Allocation),
}

pub trait AllocatorTrait {
    fn create_buffer<C: RendererVkContext>(
        &self,
        vk_context: &C,
        size: usize,
        usage: vk::BufferUsageFlags,
    ) -> RendererResult<(vk::Buffer, Memory)>;

    fn create_image<C: RendererVkContext>(
        &self,
        vk_context: &C,
        width: u32,
        height: u32,
    ) -> RendererResult<(vk::Image, Memory)>;

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
            #[cfg(feature = "vkmem")]
            Memory::VkMemAllocation(allocation) => {
                vk_context
                    .vk_mem_allocator()
                    .destroy_buffer(buffer, &allocation)?;
            }
        }

        Ok(())
    }

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
            #[cfg(feature = "vkmem")]
            Memory::VkMemAllocation(allocation) => {
                vk_context
                    .vk_mem_allocator()
                    .destroy_image(image, &allocation)?;
            }
        }

        Ok(())
    }

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
                    align.copy_from_slice(&data);
                    device.unmap_memory(*memory);
                }
                #[cfg(feature = "vkmem")]
                Memory::VkMemAllocation(allocation) => {
                    let allocator = vk_context.vk_mem_allocator();
                    let data_ptr = allocator.map_memory(allocation)? as *mut std::ffi::c_void;
                    let mut align =
                        ash::util::Align::new(data_ptr, std::mem::align_of::<T>() as _, size);
                    align.copy_from_slice(&data);
                    allocator.unmap_memory(allocation)?;
                }
            }
        };
        Ok(())
    }
}

pub enum Allocator {
    Default(DefaultAllocator),
    #[cfg(feature = "vkmem")]
    VkMem(VkMemAllocator),
}

impl Allocator {
    pub fn defaut() -> Self {
        Self::Default(DefaultAllocator)
    }

    #[cfg(feature = "vkmem")]
    pub fn vk_mem() -> Self {
        Self::VkMem(VkMemAllocator)
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
            #[cfg(feature = "vkmem")]
            Self::VkMem(allocator) => allocator.create_buffer(vk_context, size, usage),
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
            #[cfg(feature = "vkmem")]
            Self::VkMem(allocator) => allocator.create_image(vk_context, width, height),
        }
    }
}
