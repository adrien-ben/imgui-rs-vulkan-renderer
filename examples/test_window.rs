mod common;

use common::*;
use std::error::Error;

const APP_NAME: &str = "test window";

fn main() -> Result<(), Box<dyn Error>> {
    simple_logger::init()?;
    System::new(APP_NAME)?.run((), |run, ui, _| ui.show_demo_window(run))?;

    Ok(())
}
