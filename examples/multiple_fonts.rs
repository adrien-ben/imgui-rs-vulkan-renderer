mod common;

use common::*;
use imgui::*;
use simple_logger::SimpleLogger;
use std::error::Error;

const APP_NAME: &str = "multiple fonts";

fn main() -> Result<(), Box<dyn Error>> {
    SimpleLogger::new().init()?;

    let mut system = System::<()>::new(APP_NAME)?;

    let dokdo = system.imgui.fonts().add_font(&[FontSource::TtfData {
        data: include_bytes!("../assets/fonts/Dokdo-Regular.ttf"),
        size_pixels: system.font_size,
        config: None,
    }]);
    let roboto = system.imgui.fonts().add_font(&[FontSource::TtfData {
        data: include_bytes!("../assets/fonts/Roboto-Regular.ttf"),
        size_pixels: system.font_size,
        config: None,
    }]);

    system.update_fonts_texture()?;

    system.run((), move |run, ui, _| {
        Window::new("Hello world").opened(run).build(ui, || {
            ui.text("Hello, I'm the default font!");
            let _roboto = ui.push_font(roboto);
            ui.text("Hello, I'm Roboto Regular!");
            let _dokdo = ui.push_font(dokdo);
            ui.text("Hello, I'm Dokdo Regular!");
            _dokdo.pop();
            ui.text("Hello, I'm Roboto Regular again!");
            _roboto.pop();
            ui.text("Hello, I'm the default font again!");
        });
    })?;

    Ok(())
}
