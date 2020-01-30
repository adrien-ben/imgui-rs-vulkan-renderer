use ash::vk;
use std::{error::Error, fmt, io};

/// Crates error type.
#[derive(Debug)]
pub enum RendererError {
    /// Errors coming from calls to Vulkan functions.
    Vulkan(vk::Result),
    /// Io errors.
    Io(io::Error),
    /// Initialization errors.
    Init(String),
    /// Post destruction renderer usage error.
    Destroyed,
}

impl fmt::Display for RendererError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use RendererError::*;
        match self {
            Vulkan(error) => write!(f, "A Vulkan error occured: {}", error),
            Io(error) => write!(f, "An io error occured: {}", error),
            Init(message) => write!(
                f,
                "An error occured when initializing the renderer: {}",
                message
            ),
            Destroyed => write!(f, "You are trying to use the renderer after destoying it"),
        }
    }
}

impl Error for RendererError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        use RendererError::*;
        match self {
            Vulkan(error) => Some(error),
            Io(error) => Some(error),
            Init(..) | Destroyed => None,
        }
    }
}

#[doc(hidden)]
impl From<vk::Result> for RendererError {
    fn from(error: vk::Result) -> RendererError {
        RendererError::Vulkan(error)
    }
}

#[doc(hidden)]
impl From<io::Error> for RendererError {
    fn from(error: io::Error) -> RendererError {
        RendererError::Io(error)
    }
}
