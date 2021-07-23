use ash::vk;
use imgui::TextureId;
use std::{error::Error, fmt, io};

/// Crates error type.
#[derive(Debug)]
pub enum RendererError {
    /// Errors coming from calls to Vulkan functions.
    Vulkan(vk::Result),
    /// Errors coming from calls to vk-mem functions.
    #[cfg(feature = "vkmem")]
    VkMem(vk_mem::Error),
    /// Io errors.
    Io(io::Error),
    /// Initialization errors.
    Init(String),
    /// Texture lookup error.
    BadTexture(TextureId),
    /// Post destruction renderer usage error.
    Destroyed,
}

impl fmt::Display for RendererError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use RendererError::*;
        match self {
            Vulkan(error) => write!(f, "A Vulkan error occured: {}", error),
            #[cfg(feature = "vkmem")]
            VkMem(error) => write!(f, "A vk-mem error occured: {}", error),
            Io(error) => write!(f, "An io error occured: {}", error),
            Init(message) => write!(
                f,
                "An error occured when initializing the renderer: {}",
                message
            ),
            BadTexture(texture_id) => write!(f, "Bad texture ID: {}", texture_id.id()),
            Destroyed => write!(f, "You are trying to use the renderer after destoying it"),
        }
    }
}

impl Error for RendererError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        use RendererError::*;
        match self {
            Vulkan(error) => Some(error),
            #[cfg(feature = "vkmem")]
            VkMem(error) => Some(error),
            Io(error) => Some(error),
            Init(..) | BadTexture(..) | Destroyed => None,
        }
    }
}

#[doc(hidden)]
impl From<vk::Result> for RendererError {
    fn from(error: vk::Result) -> RendererError {
        RendererError::Vulkan(error)
    }
}

#[cfg(feature = "vkmem")]
#[doc(hidden)]
impl From<vk_mem::Error> for RendererError {
    fn from(error: vk_mem::Error) -> RendererError {
        RendererError::VkMem(error)
    }
}

#[doc(hidden)]
impl From<io::Error> for RendererError {
    fn from(error: io::Error) -> RendererError {
        RendererError::Io(error)
    }
}
