mod common;

use common::*;
use imgui::*;
use simple_logger::SimpleLogger;
use std::error::Error;

const APP_NAME: &str = "long list";

fn main() -> Result<(), Box<dyn Error>> {
    SimpleLogger::new().init()?;
    let lots_of_words: Vec<String> = (0..10000).map(|x| format!("Line {}", x)).collect();
    System::new(APP_NAME)?.run((), move |_, ui, _| {
        Window::new("Hello long world")
            .size([300.0, 110.0], Condition::FirstUseEver)
            .build(ui, || {
                let mut clipper = imgui::ListClipper::new(lots_of_words.len() as i32)
                    .items_height(ui.current_font_size())
                    .begin(ui);
                while clipper.step() {
                    for row_num in clipper.display_start()..clipper.display_end() {
                        ui.text(&lots_of_words[row_num as usize]);
                    }
                }
            });
    })?;

    Ok(())
}
