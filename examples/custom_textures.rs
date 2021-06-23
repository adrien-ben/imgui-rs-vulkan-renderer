mod common;

use ash::{version::DeviceV1_0, vk};
use common::*;
use imgui::*;
use imgui_rs_vulkan_renderer::vulkan::*;
use imgui_rs_vulkan_renderer::{Allocator, RendererVkContext};

use std::error::Error;

use image::{self, GenericImageView};
use simple_logger::SimpleLogger;

const APP_NAME: &str = "custom textures";

struct CustomTexturesApp {
    allocator: Allocator,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    _descriptor_set: vk::DescriptorSet,
    my_texture: Texture,
    my_texture_id: Option<TextureId>,
    lenna: Option<Lenna>,
}

struct Lenna {
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    _descriptor_set: vk::DescriptorSet,
    texture: Texture,
    texture_id: TextureId,
    size: [f32; 2],
}

impl CustomTexturesApp {
    fn new<C: RendererVkContext>(
        vk_context: &C,
        textures: &mut Textures<vk::DescriptorSet>,
    ) -> Self {
        const WIDTH: usize = 100;
        const HEIGHT: usize = 100;

        let allocator = Allocator::defaut();

        // Generate dummy texture
        let my_texture = {
            let mut data = Vec::with_capacity(WIDTH * HEIGHT);
            for i in 0..WIDTH {
                for j in 0..HEIGHT {
                    // Insert RGB values
                    data.push(i as u8);
                    data.push(j as u8);
                    data.push((i + j) as u8);
                    data.push(255 as u8);
                }
            }

            Texture::from_rgba8(vk_context, &allocator, WIDTH as u32, HEIGHT as u32, &data).unwrap()
        };

        let descriptor_set_layout =
            create_vulkan_descriptor_set_layout(vk_context.device()).unwrap();

        let descriptor_pool = create_vulkan_descriptor_pool(vk_context.device(), 1).unwrap();

        let descriptor_set = create_vulkan_descriptor_set(
            &vk_context.device(),
            descriptor_set_layout,
            descriptor_pool,
            my_texture.image_view,
            my_texture.sampler,
        )
        .unwrap();

        let texture_id = textures.insert(descriptor_set);

        let my_texture_id = Some(texture_id);

        // Lenna
        let lenna = Some(Lenna::new(vk_context, &allocator, textures).unwrap());

        CustomTexturesApp {
            allocator,
            descriptor_set_layout,
            descriptor_pool,
            _descriptor_set: descriptor_set,
            my_texture,
            my_texture_id,
            lenna,
        }
    }

    fn show_textures(&self, ui: &Ui) {
        Window::new(im_str!("Hello textures"))
            .size([400.0, 600.0], Condition::FirstUseEver)
            .build(ui, || {
                ui.text(im_str!("Hello textures!"));
                if let Some(my_texture_id) = self.my_texture_id {
                    ui.text("Some generated texture");
                    Image::new(my_texture_id, [100.0, 100.0]).build(ui);
                }

                if let Some(lenna) = &self.lenna {
                    ui.text("Say hello to Lenna.jpg");
                    lenna.show(ui);
                }
            });
    }
}

impl Lenna {
    fn new<C: RendererVkContext>(
        vk_context: &C,
        allocator: &Allocator,
        textures: &mut Textures<vk::DescriptorSet>,
    ) -> Result<Self, Box<dyn Error>> {
        let lenna_bytes = include_bytes!("../assets/images/Lenna.jpg");
        let image =
            image::load_from_memory_with_format(lenna_bytes, image::ImageFormat::Jpeg).unwrap();
        let (width, height) = image.dimensions();
        let data = image.into_rgba8();

        let texture = Texture::from_rgba8(vk_context, allocator, width, height, &data).unwrap();

        let descriptor_set_layout =
            create_vulkan_descriptor_set_layout(vk_context.device()).unwrap();

        let descriptor_pool = create_vulkan_descriptor_pool(vk_context.device(), 1).unwrap();

        let descriptor_set = create_vulkan_descriptor_set(
            vk_context.device(),
            descriptor_set_layout,
            descriptor_pool,
            texture.image_view,
            texture.sampler,
        )
        .unwrap();

        let texture_id = textures.insert(descriptor_set);
        Ok(Lenna {
            descriptor_set_layout,
            descriptor_pool,
            _descriptor_set: descriptor_set,
            texture,
            texture_id,
            size: [width as f32, height as f32],
        })
    }

    fn show(&self, ui: &Ui) {
        Image::new(self.texture_id, self.size).build(ui);
    }

    fn destroy<C: RendererVkContext>(&mut self, context: &C, allocator: &Allocator) {
        unsafe {
            let device = context.device();
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.texture.destroy(context, allocator).unwrap();
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}

impl App for CustomTexturesApp {
    fn destroy(&mut self, context: &VulkanContext) {
        unsafe {
            let device = context.device();
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.my_texture.destroy(context, &self.allocator).unwrap();
            if let Some(lenna) = &mut self.lenna {
                lenna.destroy(context, &self.allocator);
            }
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    SimpleLogger::new().init()?;
    let mut system = System::new(APP_NAME)?;
    let my_app = CustomTexturesApp::new(&system.vulkan_context, system.renderer.textures());
    system.run(my_app, move |_, ui, app| {
        app.show_textures(ui);
    })?;

    Ok(())
}
