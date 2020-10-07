mod common;

use common::*;
use imgui::*;
use simple_logger::SimpleLogger;
use std::error::Error;

const APP_NAME: &str = "hello world";

fn main() -> Result<(), Box<dyn Error>> {
    SimpleLogger::new().init()?;
    System::new(APP_NAME)?.run((), |_, ui, _| {
        Window::new(im_str!("Hello world"))
            .size([300.0, 100.0], Condition::FirstUseEver)
            .build(ui, || {
                ui.text(im_str!("Hello world!"));
                ui.text(im_str!("こんにちは世界！"));
                ui.text(im_str!("This...is...imgui-rs!"));
                ui.separator();
                let mouse_pos = ui.io().mouse_pos;
                ui.text(format!(
                    "Mouse Position: ({:.1},{:.1})",
                    mouse_pos[0], mouse_pos[1]
                ));
            });
    })?;

    Ok(())
}
