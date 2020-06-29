mod common;

use ash::{
    version::{DeviceV1_0, InstanceV1_0},
    vk,
};
use common::*;
use imgui::*;
use imgui_rs_vulkan_renderer::vulkan::*;
use imgui_rs_vulkan_renderer::RendererVkContext;

use std::error::Error;
use std::io::Cursor;

use image::{jpeg::JpegDecoder, ImageDecoder};

const APP_NAME: &str = "custom textures";

struct CustomTexturesApp {
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

            let memory_properties = unsafe {
                vk_context
                    .instance()
                    .get_physical_device_memory_properties(vk_context.physical_device())
            };

            Texture::from_rgba8(
                vk_context.device(),
                vk_context.queue(),
                vk_context.command_pool(),
                memory_properties,
                WIDTH as u32,
                HEIGHT as u32,
                &data,
            )
            .unwrap()
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
        let lenna = Some(Lenna::new(vk_context, textures).unwrap());

        CustomTexturesApp {
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

    pub fn destroy<C: RendererVkContext>(&mut self, context: &C) {
        unsafe {
            let device = context.device();
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.my_texture.destroy(device);
            if let Some(lenna) = &mut self.lenna {
                lenna.destroy(context);
            }
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}

impl Lenna {
    fn new<C: RendererVkContext>(
        vk_context: &C,
        textures: &mut Textures<vk::DescriptorSet>,
    ) -> Result<Self, Box<dyn Error>> {
        let lenna_bytes = include_bytes!("../resources/Lenna.jpg");
        let byte_stream = Cursor::new(lenna_bytes.as_ref());
        let decoder = JpegDecoder::new(byte_stream)?;

        let (width, height) = decoder.dimensions();
        let mut image = vec![0; decoder.total_bytes() as usize];
        decoder.read_image(&mut image)?;
        let image = Self::rgb_to_rgba(&image);

        let memory_properties = unsafe {
            vk_context
                .instance()
                .get_physical_device_memory_properties(vk_context.physical_device())
        };

        let texture = Texture::from_rgba8(
            vk_context.device(),
            vk_context.queue(),
            vk_context.command_pool(),
            memory_properties,
            width,
            height,
            &image,
        )
        .unwrap();

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

    fn rgb_to_rgba(input: &[u8]) -> Vec<u8> {
        let count = input.len() / 3;
        let mut output = vec![0; 4 * count];

        let in_pixels = input[..3 * count].chunks_exact(3);
        let out_pixels = output[..4 * count].chunks_exact_mut(4);

        for (pixel, outp) in in_pixels.zip(out_pixels) {
            let r = pixel[0];
            let g = pixel[1];
            let b = pixel[2];

            outp[0] = r;
            outp[1] = g;
            outp[2] = b;
            outp[3] = 255;
        }

        output
    }

    fn show(&self, ui: &Ui) {
        Image::new(self.texture_id, self.size).build(ui);
    }

    fn destroy<C: RendererVkContext>(&mut self, context: &C) {
        unsafe {
            let device = context.device();
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.texture.destroy(device);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    simple_logger::init()?;
    let mut system = System::new(APP_NAME)?;
    let mut my_app = CustomTexturesApp::new(&system.vulkan_context, system.renderer.textures());
    system.run(|_, ui| {
        my_app.show_textures(ui);
    })?;
    my_app.destroy(&system.vulkan_context);

    Ok(())
}
