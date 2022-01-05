//! # imgui-rs-vulkan-renderer
//!
//! A Vulkan renderer for [imgui-rs][imgui-rs] using [Ash][ash].
//!
//! ## Compatibility
//!
//! | crate | imgui | ash  | gpu-allocator (feature) | vk-mem (feature)                |
//! |-------|-------|------|-------------------------|---------------------------------|
//! | 1.0.0 | 0.8   | 0.35 | 0.14                    | 0.2.3 ([forked][forked-mem-rs]) |
//!
//! ## How it works
//!
//! The renderer records drawing command to a command buffer supplied by the application. Here is a little breakdown of the features of this crate and how they work.
//!
//! - Vertex/Index buffers
//!
//! The renderer creates a vertex buffer and a index buffer that will be updated every time
//! `Renderer::cmd_draw` is called. If the vertex/index count is more than what the buffers can
//! actually hold then the buffers are resized (actually destroyed then re-created).
//!
//! - Frames in flight
//!
//! The renderer support having multiple frames in flight. You need to specify the number of frames
//! during initialization of the renderer. The renderer manages one vertex and index buffer per frame.
//!
//! - No draw call execution
//!
//! The `Renderer::cmd_draw` only record commands to a command buffer supplied by the application. It does not submit anything to the gpu.
//!
//! - Custom textures
//!
//! The renderer supports custom textures. See `Renderer::textures` for details.
//!
//! - Custom Vulkan allocators
//!
//! Custom Vulkan allocators are not supported for the moment.
//!
//! ## Features
//!
//! ### gpu-allocator
//!
//! This feature adds support for gpu-allocator. It changes `Renderer::new` which now takes
//! a `Arc<Mutex<gpu_allocator::vulkan::Allocator>>`. All internal allocator are then done using the allocator.
//!
//! ### vk-mem
//!
//! This feature adds support for [vk-mem-rs][vk-mem-rs]. It adds `Renderer::with_vk_mem_allocator` which takes
//! a `Arc<Mutex<vk_mem::Allocator>>`. All internal allocator are then done using the allocator.
//!
//! ## Integration
//!
//! You can find an example of integration in the [common module](examples/common/mod.rs) of the examples.
//!
//! ## Examples
//!
//! You can run a set of examples by running the following command:
//!
//! ```sh
//! ## If you want to enable validation layers
//! export VK_LAYER_PATH=$VULKAN_SDK/Bin
//! export VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation
//!
//! ## Or with Powershell
//! $env:VK_LAYER_PATH = "$env:VULKAN_SDK\Bin"
//! $env:VK_INSTANCE_LAYERS = "VK_LAYER_KHRONOS_validation"
//!
//! ## If you changed the shader code (you'll need glslangValidator on you PATH)
//! ## There is also a PowerShell version (compile_shaders.ps1)
//! ./compile_shaders.sh
//!
//! ## Run an example
//! cargo run --example <example>
//!
//! ## Example can be one of the following value:
//! ## - collapsing_header
//! ## - color_button
//! ## - custom_textures
//! ## - disablement
//! ## - draw_list
//! ## - hello_world
//! ## - keyboard
//! ## - long_list
//! ## - multiple fonts
//! ## - progress_bar
//! ## - radio_button
//! ## - slider : TODO: crashes
//! ## - tables_api
//! ## - test_drawing_channels_split
//! ## - test_window_impl
//! ## - test_window
//! ## - text_callbacks
//! ## - text_input
//! ```
//!
//! [imgui-rs]: https://github.com/Gekkio/imgui-rs
//! [ash]: https://github.com/MaikKlein/ash
//! [gpu-allocator]: https://github.com/Traverse-Research/gpu-allocator
//! [example]: https://github.com/adrien-ben/imgui-rs-vulkan-renderer/blob/master/examples/common/mod.rs

mod error;
mod renderer;

pub use error::*;
pub use renderer::*;
