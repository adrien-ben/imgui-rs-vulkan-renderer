mod common;

use common::*;
use imgui::*;
use simple_logger::SimpleLogger;
use std::error::Error;

const APP_NAME: &str = "long list";

fn main() -> Result<(), Box<dyn Error>> {
    SimpleLogger::new().init()?;

    let mut stable_str = String::new();

    System::new(APP_NAME)?.run((), move |_, ui, _| {
        if let Some(_window) = imgui::Window::new("Input text callbacks")
            .size([500.0, 300.0], Condition::FirstUseEver)
            .begin(ui)
        {
            if ui.input_text("input stable", &mut stable_str).build() {
                dbg!(&stable_str);
            }

            let mut per_frame_buf = String::new();
            ui.input_text("input per frame", &mut per_frame_buf).build();

            if ui.is_item_deactivated_after_edit() {
                dbg!(&per_frame_buf);
            }
        }
    })?;

    Ok(())
}
