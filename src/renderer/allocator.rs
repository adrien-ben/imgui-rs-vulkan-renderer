use crate::{RendererResult, RendererVkContext};
use ash::{prelude::VkResult, vk};
#[cfg(not(feature = "vma"))]
use ash::{
    version::{DeviceV1_0, InstanceV1_0},
    Device,
};
use core::ffi::c_void;

#[cfg(not(feature = "vma"))]
pub(crate) type Memory = vk::DeviceMemory;

#[cfg(not(feature = "vma"))]
pub(crate) struct Allocator {
    device: Device,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
}

#[cfg(not(feature = "vma"))]
impl Allocator {
    pub fn new(vk_context: &dyn RendererVkContext, _frame_in_use_count: u32) -> Self {
        let memory_properties = unsafe {
            vk_context
                .instance()
                .get_physical_device_memory_properties(vk_context.physical_device())
        };
        Allocator {
            device: vk_context.device().clone(), // !!!
            memory_properties,
        }
    }

    pub fn destroy(&mut self) {}

    pub fn create_buffer(
        &self,
        buffer_create_info: &vk::BufferCreateInfo,
    ) -> RendererResult<(vk::Buffer, Memory)> {
        let buffer = unsafe { self.device.create_buffer(&buffer_create_info, None)? };
        let mem_requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let mem_type = Self::find_memory_type(
            mem_requirements,
            self.memory_properties,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );
        let memory_allocate_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(mem_requirements.size)
            .memory_type_index(mem_type);
        let memory = unsafe {
            let memory = self.device.allocate_memory(&memory_allocate_info, None)?;
            self.device.bind_buffer_memory(buffer, memory, 0)?;
            memory
        };
        Ok((buffer, memory))
    }

    pub fn destroy_buffer(&self, buffer: vk::Buffer, memory: &Memory) {
        unsafe {
            self.device.destroy_buffer(buffer, None);
            self.device.free_memory(*memory, None);
        }
    }

    pub fn map_memory(&self, memory: &vk::DeviceMemory) -> VkResult<*mut c_void> {
        unsafe {
            self.device
                .map_memory(*memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::default())
        }
    }

    pub fn unmap_memory(&self, memory: &vk::DeviceMemory) {
        unsafe { self.device.unmap_memory(*memory) }
    }

    pub fn create_image(
        &self,
        image_create_info: &vk::ImageCreateInfo,
    ) -> RendererResult<(vk::Image, Memory)> {
        let image = unsafe { self.device.create_image(&image_create_info, None)? };
        let mem_requirements = unsafe { self.device.get_image_memory_requirements(image) };
        let mem_type_index = Self::find_memory_type(
            mem_requirements,
            self.memory_properties,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );
        let memory_allocate_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(mem_requirements.size)
            .memory_type_index(mem_type_index);
        let memory = unsafe {
            let memory = self.device.allocate_memory(&memory_allocate_info, None)?;
            self.device.bind_image_memory(image, memory, 0)?;
            memory
        };
        Ok((image, memory))
    }

    pub fn destroy_image(&self, image: vk::Image, memory: &Memory) {
        unsafe {
            self.device.destroy_image(image, None);
            self.device.free_memory(*memory, None);
        }
    }

    fn find_memory_type(
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

#[cfg(feature = "vma")]
use vk_mem;

#[cfg(feature = "vma")]
pub(crate) type Memory = vk_mem::Allocation;

#[cfg(feature = "vma")]
pub(crate) struct Allocator {
    allocator: vk_mem::Allocator,
}

#[cfg(feature = "vma")]
impl Allocator {
    pub fn new(vk_context: &dyn RendererVkContext, frame_in_use_count: u32) -> Self {
        let allocator = {
            let create_info = vk_mem::AllocatorCreateInfo {
                physical_device: vk_context.physical_device(),
                device: vk_context.device().clone(),
                instance: vk_context.instance().clone(),
                flags: vk_mem::AllocatorCreateFlags::NONE,
                preferred_large_heap_block_size: 0,
                frame_in_use_count,
                heap_size_limits: None,
            };
            let allocator = match vk_mem::Allocator::new(&create_info) {
                Ok(v) => v,
                Err(e) => panic!(e.to_string()),
            };
            allocator
        };
        Allocator { allocator }
    }

    pub fn destroy(&mut self) {
        self.allocator.destroy();
    }

    pub fn create_buffer(
        &self,
        buffer_create_info: &vk::BufferCreateInfo,
    ) -> RendererResult<(vk::Buffer, Memory)> {
        let allocation_create_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::CpuToGpu,
            ..Default::default()
        };

        let (buffer, allocation, _allocation_info) = self
            .allocator
            .create_buffer(&buffer_create_info, &allocation_create_info)
            .unwrap();

        Ok((buffer, allocation))
    }

    pub fn destroy_buffer(&self, buffer: vk::Buffer, memory: &Memory) {
        self.allocator
            .destroy_buffer(buffer, memory)
            .expect("Failed to destroy buffer!");
    }

    pub fn map_memory(&self, memory: &vk_mem::Allocation) -> VkResult<*mut c_void> {
        Ok(self.allocator.map_memory(memory).unwrap() as _)
    }

    pub fn unmap_memory(&self, memory: &vk_mem::Allocation) {
        self.allocator.unmap_memory(memory).unwrap()
    }

    pub fn create_image(
        &self,
        image_create_info: &vk::ImageCreateInfo,
    ) -> RendererResult<(vk::Image, Memory)> {
        let allocation_create_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::GpuOnly,
            ..Default::default()
        };

        let (image, allocation, _allocation_info) = self
            .allocator
            .create_image(&image_create_info, &allocation_create_info)
            .unwrap();

        Ok((image, allocation))
    }

    pub fn destroy_image(&self, image: vk::Image, memory: &Memory) {
        self.allocator
            .destroy_image(image, memory)
            .expect("Failed to destroy image!");
    }
}
