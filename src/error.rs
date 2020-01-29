use ash::vk;
use std::{error::Error, fmt, io};

#[derive(Debug)]
pub enum RendererError {
    Vulkan(vk::Result),
    Io(io::Error),
}

impl fmt::Display for RendererError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RendererError::Vulkan(error) => write!(f, "A Vulkan error occured: {}", error),
            RendererError::Io(error) => write!(f, "An io error occured: {}", error),
        }
    }
}

impl Error for RendererError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            RendererError::Vulkan(error) => Some(error),
            RendererError::Io(error) => Some(error),
        }
    }
}

impl From<vk::Result> for RendererError {
    fn from(error: vk::Result) -> RendererError {
        RendererError::Vulkan(error)
    }
}

impl From<io::Error> for RendererError {
    fn from(error: io::Error) -> RendererError {
        RendererError::Io(error)
    }
}
