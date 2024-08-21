# Changelog

## 1.14.0

- Bump ash to 0.38 by @filnet in #41

## 1.13.0

- Bump imgui to 0.12
- Bump gpu-allocator to 0.26
- Bump dev dependencies

## 1.12.0

- Bump vk-mem to 0.3.0

## 1.11.0

- Bump gpu-allocator to 0.25 by @nice-sprite in #38

## 1.10.0

- Bump gpu-allocator to 0.23

## 1.9.0

- Bump imgui to 0.11

## 1.8.0

- Bump gpu-allocator to 0.22

## 1.7.0

- Bump imgui to 0.10

## 1.6.1

- Avoid passing negative values to `cmd_set_scissor`

## 1.6.0

- Bump imgui to 0.9
- Allow gpu-allocator >= 0.19 & <= 0.21
- Update examples and bump dev dependencies

## 1.5.0

- Bump gpu-allocator to 0.19
- **BREAKING** (only with dynamic-rendering feature) Allow passing depth attachment format through `DynamicRendering` struct

## 1.4.0

- Add support for Vulkan dynamic rendering

## 1.3.0

- Bump dependencies
    - ash >= 0.34 <= 0.37
    - gpu-allocator 0.18
    - vk-mem (forked version compatible with ash >= 0.34 <= 0.37)

## 1.2.0

- Bump dependencies
    - ash 0.36
    - gpu-allocator 0.17
    - vk-mem (forked version compatible with ash 0.36)

## 1.1.1

- Fix blend function

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
