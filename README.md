# imgui-rs-vulkan-renderer

A Vulkan renderer for imgui-rs using Ash.

## Examples

You can run a set of examples by running the following command:

```sh
# If you want to enable validation layers
export VK_LAYER_PATH=$VULKAN_SDK/Bin
export VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation

# If you changed the shader code (you'll need glslangValidator on you PATH)
# There is also a PowerShell version (compile_shaders.ps1)
./compile_shaders.sh

# Run an example
cargo run --example <example>

# Example can be one of the following value:
# - color_button
# - hello_world
# - progress_bar
# - radio_button
# - test_drawing_channels_split
# - test_window_impl
# - test_window
```
