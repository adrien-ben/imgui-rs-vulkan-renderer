//! # imgui-rs-vulkan-renderer
//!
//! A Vulkan renderer for [imgui-rs][imgui-rs] using [Ash][ash].
//!
//! ## Compatibility
//!
//! | crate  | imgui | ash          | gpu-allocator (feature) | vk-mem (feature)                        |
//! |--------|-------|--------------|-------------------------|-----------------------------------------|
//! | 1.13.0 | 0.12  | [0.34, 0.37] | 0.26                    | 0.3.0                                   |
//! | 1.12.0 | 0.11  | [0.34, 0.37] | 0.25                    | 0.3.0                                   |
//! | 1.11.0 | 0.11  | [0.34, 0.37] | 0.25                    | 0.2.3 ([forked][forked-mem-rs-034-037]) |
//! | 1.10.0 | 0.11  | [0.34, 0.37] | 0.23                    | 0.2.3 ([forked][forked-mem-rs-034-037]) |
//! | 1.9.0  | 0.11  | [0.34, 0.37] | 0.22                    | 0.2.3 ([forked][forked-mem-rs-034-037]) |
//! | 1.8.0  | 0.10  | [0.34, 0.37] | 0.22                    | 0.2.3 ([forked][forked-mem-rs-034-037]) |
//! | 1.7.0  | 0.10  | [0.34, 0.37] | [0.19, 0.21]            | 0.2.3 ([forked][forked-mem-rs-034-037]) |
//! | 1.6.x  | 0.9   | [0.34, 0.37] | [0.19, 0.21]            | 0.2.3 ([forked][forked-mem-rs-034-037]) |
//! | 1.5.0  | 0.8   | [0.34, 0.37] | 0.19                    | 0.2.3 ([forked][forked-mem-rs-034-037]) |
//! | 1.4.0  | 0.8   | [0.34, 0.37] | 0.18                    | 0.2.3 ([forked][forked-mem-rs-034-037]) |
//! | 1.3.0  | 0.8   | [0.34, 0.37] | 0.18                    | 0.2.3 ([forked][forked-mem-rs-034-037]) |
//! | 1.2.0  | 0.8   | 0.36         | 0.17                    | 0.2.3 ([forked][forked-mem-rs-036])     |
//! | 1.1.x  | 0.8   | 0.35         | 0.15                    | 0.2.3 ([forked][forked-mem-rs-035])     |
//! | 1.0.0  | 0.8   | 0.35         | 0.14                    | 0.2.3 ([forked][forked-mem-rs-035])     |
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
//! ### dynamic-rendering
//!
//! This feature is useful if you want to integrate the library in an app making use of Vulkan's dynamic rendering.
//! When enabled, functions that usually takes a `vk::RenderPass` as argument will now take a `vk::Format` which is
//! the format of the color attachment the UI will be drawn to.
//!
//! ## Integration
//!
//! You can find an example of integration in the [common module](examples/common/mod.rs) of the examples.
//!
//! ```rust
//! // Example with default allocator
//! let renderer = Renderer::with_default_allocator(
//!     &instance,
//!     physical_device,
//!     device.clone(),
//!     graphics_queue,
//!     command_pool,
//!     render_pass,
//!     &mut imgui,
//!     Some(Options {
//!         in_flight_frames: 1,
//!         ..Default::default()
//!     }),
//! ).unwrap();
//! ```
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
//! ## - creating_windows
//! ## - custom_textures
//! ## - disablement
//! ## - draw_list
//! ## - hello_world
//! ## - id_wrangling
//! ## - keyboard
//! ## - long_list
//! ## - long_table
//! ## - multiple_fonts
//! ## - progress_bar
//! ## - radio_button
//! ## - slider
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
