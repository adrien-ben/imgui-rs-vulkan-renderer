# Changelog

## 1.1.0

- Add optional depth test/write support.
- **BREAKING** Added `Options` parameter to all renderer instanciation methods.
- **BREAKING** Remove in_flight_frames parameter from all renderer instanciation methods. Moved to `Options`.
- Bump gpu-allocator to 0.15

## 1.0.0

- Rework API:
    - Remove the trait `RendererVkContext` so no need to pass a ref to each method.
    - Remove `Renderer::destroy` and implement `Drop`.
- Add [`gpu-allocator`](https://github.com/Traverse-Research/gpu-allocator) support.
- Add multiple font support with `Renderer::update_fonts_texture`.
- Bump dependencies:
    - imgui 0.8
    - ash 0.35
- Add and update examples.

## 0.8.1

- Fix vk-mem-rs compatibility with ash 0.33 (by setting an upper bound)

## 0.8.0

- Add [vk-mem-rs](https://github.com/gwihlidal/vk-mem-rs) support.
- Hide internals previously exposed for examples.

## 0.7.0

- Bump imgui-rs to 0.7

## 0.6.1

- Do not apply framebuffer scaling on projection matrix

## 0.6.0

- Bump imgui-rs to 0.6.0

## 0.5.0

- Bump imgui-rs to 0.5.0

## 0.4.2

- Add `Renderer::set_render_pass` to change the target render pass

## 0.4.1

- Add custom texture support

## 0.4.0

- Bump imgui-rs to 0.4.0
- Add `collapsing_header` example

## 0.3.2

- Protect against empty DrawData

## 0.3.1

- Make imgui and ash version requirements more flexible

## 0.3.0

- Bump imgui-rs to 0.3.0

## 0.1.1

- First release after experimenting and yanking the 0.1.0 ! ;)
