mod common;

use ash::{vk, Device, Instance};
use common::*;
use imgui::*;
use imgui_rs_vulkan_renderer::vulkan::*;

use std::error::Error;

use image::{self, GenericImageView};
use simple_logger::SimpleLogger;

const APP_NAME: &str = "custom textures";

fn main() -> Result<(), Box<dyn Error>> {
    SimpleLogger::new().init()?;
    let mut system = System::new(APP_NAME)?;
    let my_app = CustomTexturesApp::new(
        &system.vulkan_context.instance,
        &system.vulkan_context.device,
        system.vulkan_context.physical_device,
        system.vulkan_context.graphics_queue,
        system.vulkan_context.command_pool,
        system.renderer.textures(),
    );
    system.run(my_app, move |_, ui, app| {
        app.show_textures(ui);
    })?;

    Ok(())
}

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
    fn new(
        instance: &Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        queue: vk::Queue,
        command_pool: vk::CommandPool,
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

            let memory_properties =
                unsafe { instance.get_physical_device_memory_properties(physical_device) };

            Texture::from_rgba8(
                device,
                queue,
                command_pool,
                memory_properties,
                WIDTH as u32,
                HEIGHT as u32,
                &data,
            )
            .unwrap()
        };

        let descriptor_set_layout = create_vulkan_descriptor_set_layout(device).unwrap();

        let descriptor_pool = create_vulkan_descriptor_pool(device, 1).unwrap();

        let descriptor_set = create_vulkan_descriptor_set(
            device,
            descriptor_set_layout,
            descriptor_pool,
            my_texture.image_view,
            my_texture.sampler,
        )
        .unwrap();

        let texture_id = textures.insert(descriptor_set);

        let my_texture_id = Some(texture_id);

        // Lenna
        let lenna = Some(
            Lenna::new(
                instance,
                device,
                physical_device,
                queue,
                command_pool,
                textures,
            )
            .unwrap(),
        );

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
        Window::new("Hello textures")
            .size([400.0, 400.0], Condition::FirstUseEver)
            .build(ui, || {
                ui.text("Hello textures!");
                if let Some(my_texture_id) = self.my_texture_id {
                    ui.text("Some generated texture");
                    Image::new(my_texture_id, [100.0, 100.0]).build(ui);
                }

                if let Some(lenna) = &self.lenna {
                    ui.text("Say hello to Lenna.jpg");
                    lenna.show(ui);
                }

                // Example of using custom textures on a button
                if let Some(lenna) = &self.lenna {
                    ui.text("The Lenna buttons");

                    {
                        ui.invisible_button("Boring Button", [100.0, 100.0]);
                        // See also `imgui::Ui::style_color`
                        let tint_none = [1.0, 1.0, 1.0, 1.0];
                        let tint_green = [0.5, 1.0, 0.5, 1.0];
                        let tint_red = [1.0, 0.5, 0.5, 1.0];

                        let tint = match (
                            ui.is_item_hovered(),
                            ui.is_mouse_down(imgui::MouseButton::Left),
                        ) {
                            (false, false) => tint_none,
                            (false, true) => tint_none,
                            (true, false) => tint_green,
                            (true, true) => tint_red,
                        };

                        let draw_list = ui.get_window_draw_list();
                        draw_list
                            .add_image(lenna.texture_id, ui.item_rect_min(), ui.item_rect_max())
                            .col(tint)
                            .build();
                    }

                    {
                        ui.same_line();

                        // Button using quad positioned image
                        ui.invisible_button("Exciting Button", [100.0, 100.0]);

                        // Button bounds
                        let min = ui.item_rect_min();
                        let max = ui.item_rect_max();

                        // get corner coordinates
                        let tl = [
                            min[0],
                            min[1] + (ui.frame_count() as f32 / 10.0).cos() * 10.0,
                        ];
                        let tr = [
                            max[0],
                            min[1] + (ui.frame_count() as f32 / 10.0).sin() * 10.0,
                        ];
                        let bl = [min[0], max[1]];
                        let br = max;

                        let draw_list = ui.get_window_draw_list();
                        draw_list
                            .add_image_quad(lenna.texture_id, tl, tr, br, bl)
                            .build();
                    }

                    // Rounded image
                    {
                        ui.same_line();
                        ui.invisible_button("Smooth Button", [100.0, 100.0]);

                        let draw_list = ui.get_window_draw_list();
                        draw_list
                            .add_image_rounded(
                                lenna.texture_id,
                                ui.item_rect_min(),
                                ui.item_rect_max(),
                                16.0,
                            )
                            // Tint brighter for visiblity of corners
                            .col([2.0, 0.5, 0.5, 1.0])
                            // Rounding on each corner can be changed separately
                            .round_top_left(ui.frame_count() / 60 % 4 == 0)
                            .round_top_right((ui.frame_count() + 1) / 60 % 4 == 1)
                            .round_bot_right((ui.frame_count() + 3) / 60 % 4 == 2)
                            .round_bot_left((ui.frame_count() + 2) / 60 % 4 == 3)
                            .build();
                    }
                }
            });
    }
}

impl Lenna {
    fn new(
        instance: &Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        queue: vk::Queue,
        command_pool: vk::CommandPool,
        textures: &mut Textures<vk::DescriptorSet>,
    ) -> Result<Self, Box<dyn Error>> {
        let lenna_bytes = include_bytes!("../assets/images/Lenna.jpg");
        let image =
            image::load_from_memory_with_format(lenna_bytes, image::ImageFormat::Jpeg).unwrap();
        let (width, height) = image.dimensions();
        let data = image.into_rgba8();

        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        let texture = Texture::from_rgba8(
            device,
            queue,
            command_pool,
            memory_properties,
            width,
            height,
            &data,
        )
        .unwrap();

        let descriptor_set_layout = create_vulkan_descriptor_set_layout(device).unwrap();

        let descriptor_pool = create_vulkan_descriptor_pool(device, 1).unwrap();

        let descriptor_set = create_vulkan_descriptor_set(
            device,
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

    fn destroy(&mut self, device: &Device) {
        unsafe {
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.texture.destroy(device);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}

impl App for CustomTexturesApp {
    fn destroy(&mut self, context: &VulkanContext) {
        unsafe {
            let device = &context.device;
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.my_texture.destroy(device);
            if let Some(lenna) = &mut self.lenna {
                lenna.destroy(device);
            }
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}
