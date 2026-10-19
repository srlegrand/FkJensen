# FkJensen

A 3D viewport that renders on the CPU and nothing else. No GPU API is
linked (no Vulkan, OpenGL, Metal, wgpu): the frame is rasterised on the
cores with rayon and copied to the window through shared memory
(softbuffer). It opens USD files (usda, usdc, usdz) through the pure-Rust
`openusd` crate, so there is no C++ dependency either.

This is the first stone of a CPU-and-RAM-only renderer. What it does today
is small and measured; the design is laid out so the next steps (vertex
splatting for dense meshes, cluster level of detail, a lighting cache, SIMD
inner loops) fit without a rewrite.

## Build

Rust 1.85 or later (edition 2024). No system libraries are needed to build.

    cargo build --release

Builds on Linux x86-64 and aarch64 (Raspberry Pi), macOS and Windows. The
window backend is winit; on Linux it uses X11 or Wayland, whichever is
running.

## Run

    cargo run --release -- path/to/scene.usd

Headless, for tests and screenshots (no window, no display needed):

    cargo run --release -- scene.usd --png out.png --size 1920x1080

The window title shows the frame rate, triangle counts, light count, and
the time of each pass in milliseconds: setup, bin, raster, shade.

Test scenes: the USD Working Group assets
(https://github.com/usd-wg/assets, `full_assets/`) open directly:
`OpenChessSet/chess_set.usda`, `CarbonFrameBike/CarbonFrameBike.usdz`,
`Teapot/Teapot.usd`, `McUsd/McUsd.usda` (which has a distant light in the file; file lights are read with USD's intensity conventions mapped to the renderer's, a distant light at the USD fallback of 50000 giving irradiance 2).

## Controls

    left drag          orbit            middle drag / shift+left   pan
    right drag         look (fly)       wheel                      dolly
    W A S D Q E        fly (shift: faster)
    F                  frame the scene  B    toggle back-face culling
    1 (hold) + click   place a light on the surface under the cursor
    2 (hold) + click   move the selected light there
    Tab                select next light       Delete     remove selected light
    + / -              light intensity         [ / ]      exposure
    C                  cycle light colour      L          toggle light gizmos
    G                  toggle wireframe        R          reset lights to the file's
    P                  save frame.png          Esc        quit
    drop a .usd/.usda/.usdc/.usdz file on the window to open it

A placed light sits off the surface along its normal, a quarter of the
scene's radius away, so it lights the spot that was clicked. Lights are
drawn as discs; the selected one has a yellow ring. A disc goes dim when it
is behind geometry.

## How a frame is made

`src/raster.rs`, four passes, all parallel:

1. Setup: every triangle is projected, clipped against the near plane and
   written as a screen-space record. Trivially off-screen triangles are
   dropped here.
2. Bin: each record is listed in every 64 x 64 tile its box touches.
3. Visibility: each tile is rasterised by one thread into its own depth and
   triangle-id buffer. Depth is 1/w, so it is perspective-correct and
   needs no far plane. No atomics, no locks.
4. Shade: each pixel that has a triangle reconstructs its barycentric
   weights, interpolates position, normal and colour from the source
   triangle, and is lit once: a sky and ground hemisphere, then each light
   with Lambert diffuse and a small Blinn-Phong highlight. Point lights fall
   off with 1/d², scaled so intensity 1 at one scene radius gives
   irradiance 1.

Shading after visibility is what keeps the cost proportional to pixels:
a hidden triangle is never shaded. The same structure takes vertex
splatting in pass 3 later, for meshes with more vertices than pixels.

`src/usd_load.rs` turns the stage into one world-space triangle soup
(`src/scene.rs`): positions, normals, colours, indices, in flat arrays.

## Measured

On the machine this was written on (2 cores, no SIMD, 19/10/2026), one
1280 x 720 frame:

| scene | triangles | setup | bin | raster | shade | total |
|---|---|---|---|---|---|---|
| Teapot.usd | 2,472 | 0.8 ms | 0.1 ms | 6.9 ms | 9.9 ms | 18 ms |
| CarbonFrameBike.usdz | 135,648 | 17.7 ms | 1.3 ms | 13.6 ms | 5.3 ms | 38 ms |
| chess_set.usda | 799,384 | 99.0 ms | 12.2 ms | 32.4 ms | 12.7 ms | 157 ms |

Setup and shade scale with cores; nothing has been vectorised yet.

## Not done yet

- Shadows, reflections, global illumination: every light is unshadowed.
- SIMD: the inner loops are scalar.
- Vertex splatting and cluster level of detail: a 50 M triangle mesh is
  set up triangle by triangle every frame.
- Materials and textures: `displayColor` only, 0.7 grey when absent.
- Subdivision surfaces are drawn as their control cages.
- Face-varying normals and colours are averaged onto vertices.
- Point instancers, curves and points prims are not read.
- Time samples: the default time only.
- Light rigs are not saved.
- Text overlay: the numbers are in the window title.
- Front faces: USD's right-handed winding with the camera's handedness
  decides; `B` culls back faces when a file's winding is consistent.
