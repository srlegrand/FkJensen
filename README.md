# FkJensen

A 3D viewport that renders on the CPU and nothing else. No GPU API is
linked (no Vulkan, OpenGL, Metal, wgpu): the frame is rasterised on the
cores with rayon and copied to the window through shared memory
(softbuffer).

It does not read USD. Scenes are in FkJensen's own `.fkj` format
(`docs/format.md`), with geometry already cut into the clusters the renderer
draws.

The built-in set has three distances, all real geometry, in metres:

- foreground: the Open Chess Set on a table, under two small lamps;
- mid ground: a café shopfront 40 m away across a paved square, with a
  string of festoon bulbs and a street lamp;
- far background: mountains 4 to 12 km away, up to about 2 km high.

![The set from the wide view](docs/set_wide.png)

![The chess table from the close view](docs/set_close.png)

## Light

Two kinds of light, in physical units.

**Point lights.** A point light is a position, a colour and a luminous flux
in lumens. There is no falloff radius: the light gives lumens / 4π candela,
and a surface facing it at d metres receives that over d² lux. Lumens are
kept between 1 and 2,000,000, about the output of an 18 kW HMI, the largest
common film light, so nothing brighter than a real lamp can be made. The
set opens with a 400 lm lamp and a 120 lm fill at the table, fourteen
200 lm festoon bulbs at the café and a 12,000 lm LED street lamp.

**The sky.** Light from the whole sky, with nothing between it and any
surface, so a mountain 10 km away and a pawn are lit by it the same way. It
is set by its illuminance on an upward-facing surface, in lux; the set
opens at 30 lux, dusk. The sky behind the scene is drawn from the same
values.

### Light reach

A light has no radius, but the distance beyond which it can no longer be
seen is fixed by physics and the exposure, and the renderer computes it
every frame (`raster::light_reach`):

    reach = sqrt(candela / cut)

`cut` is the illuminance below which a light's contribution is dropped
(`raster::cut_lux`):

- with the sky on, 1/64 of what the sky gives the darkest-facing surface
  (one lit only by light bounced off the ground), a change of about 1.6%;
- always, at least half an 8-bit step at black for the current exposure,
  so with the sky off a light reaches as far as it can still be seen in the
  dark.

In the setup pass each visible cluster gets a bit mask of the lights whose
reach touches its bounding sphere; the shade pass evaluates only those.
At the opening sky and exposure (cut 0.094 lux):

| light | lumens | reach |
|---|---|---|
| table fill | 120 | 10 m |
| table lamp | 400 | 18 m |
| festoon bulb | 200 | 13 m |
| street lamp | 12,000 | 101 m |
| brightest allowed | 2,000,000 | 1.3 km |

So the table lamps stop short of the café 40 m away, the café's bulbs stop
short of the table, the street lamp lights both, and no light can reach the
mountains, however strong it is made. The tests check each of these
statements, and that rendering the whole set with and without light reach
differs by at most one 8-bit step on any pixel.

![Lights evaluated per pixel](docs/light_count.png)

Light count view (`H`), left: the lights as the set opens; dark grey is
none, then blue 1, green 2, yellow 3, red 4, white 5 or more. Right: one
light at the table set to 2,000,000 lm. It reaches the square, the café and
the nearby valley, and still not the mountains.

The same bound is meant to gate shadow maps and bounce light when they
come: a light's shadow map only covers its reach, and light bounced from
the chess table never reaches the mountains.

## Build

Rust 1.85 or later (edition 2024). No system libraries are needed to build.

    cargo build --release

Builds on Linux x86-64 and aarch64 (Raspberry Pi), macOS and Windows. The
window backend is winit; on Linux it uses X11 or Wayland, whichever is
running.

## Run

    cargo run --release                     the built-in set
    cargo run --release -- scene.fkj        another .fkj scene

Headless, for tests and screenshots (no window, no display needed):

    cargo run --release -- --png out.png --size 1920x1080
    cargo run --release -- --png out.png --close
    cargo run --release -- --png out.png --view lights --light -0.2,1.1,0.22,2000000
    cargo run --release -- --png out.png --sky 0 --exposure 0.02

- `--light x,y,z[,lumens]` replaces the scene's lights (800 lm when no
  lumens are given); repeatable. Positions in metres.
- `--sky LUX` sets the sky, 0 for none, and the exposure to match.
- `--exposure E` sets the display value per cd/m².
- `--view lights` writes the light count view.
- `--no-reach` evaluates every light at every pixel, to compare.
- `--close` starts on the close framing.

The window title shows the frame rate, the selected light's lumens,
position and reach, the sky, how many light evaluations the frame did and
what share light reach skipped, cluster counts, and the time of each pass
in milliseconds.

## Controls

    Lights: point lights in lumens, 1 lm to 2,000,000 lm, plus the sky
    left drag on a light   move it, facing the camera (stays at its depth)
      + Ctrl               move it level (keeps its height)
      + wheel              push it away / pull it closer
    arrows                 nudge the selected light, relative to the view
    Page Up / Page Down    raise / lower the selected light (Shift: 5x)
    1 (hold) + click       add an 800 lm light above the surface under the cursor
    2 (hold) + click       put the selected light above that surface
    Tab                    select next light     Delete   remove selected light
    + / -                  lumens x1.25          C        cycle colour
    L                      show / hide lights    R        reset to the scene's lights
    K                      sky on / off          , / .    sky dimmer / brighter (x2)
    H                      light count view: how many lights each pixel evaluates
    X                      light reach off / on (to compare cost and picture)

    Camera
    left drag              orbit                 middle drag / shift+left   pan
    right drag             look (fly)            wheel                      dolly
    W A S D Q E            fly (shift: faster)   F    wide view / close on the subject

    View
    [ / ]                  exposure              0    exposure from the sky
    B                      back-face culling
    G                      wireframe             P    save frame.png
    Esc                    quit
    drop an .fkj file on the window to open it

Lights are drawn as discs in their colour; the selected one is larger, has
a yellow ring, and a dashed line drops from it to the ground with a dot
where it lands. A disc goes dim when it is behind geometry. Clicking a disc
selects the light and starts dragging it; clicking anywhere else orbits. A
rig holds 64 lights at most.

![Dragging the table lamp, raising it to about 320,000 lm, then the light count view](docs/light_drag.png)

There are no shadows yet: a light lights every surface that faces it, and
the sky lights everything.

## How a frame is made

`src/raster.rs`, four passes, all parallel:

1. Setup: each cluster is tested against the view frustum as a whole, and,
   with back-face culling on, against its normal cone. A surviving
   cluster gets its light mask, its vertices (64 at most) are projected
   into a local array, and its triangles are clipped against the near plane
   and written as screen-space records carrying the mask.
2. Bin: each record is listed in every 64 x 64 tile its box touches.
3. Visibility: each tile is rasterised by one thread into its own depth and
   triangle-id buffer. Depth is 1/w, so it is perspective-correct and
   needs no far plane. No atomics, no locks.
4. Shade: each pixel that has a triangle reconstructs its barycentric
   weights, interpolates position, normal and albedo, and computes
   radiance in cd/m²: the sky's irradiance on that normal, then each light
   in the mask, Lambert diffuse and a small Blinn-Phong highlight. Pixels
   with no triangle show the sky. Exposure and a tone curve take radiance to
   the display.

`src/scene.rs` holds the scene and builds clusters. `src/fkj.rs` reads
and writes `.fkj`. `src/world.rs` makes the table, square, café, valley and
mountains. `src/lights.rs` is the light rig and the picking and dragging
maths.

## Measured

On the machine this was written on (2 cores of an Intel Xeon at 2.1 GHz,
no SIMD, 19/10/2026), the built-in set (1,521,804 triangles, 24,379
clusters, 17 lights), one 1280 x 720 frame, three runs each:

| view | light reach | light evaluations | shade pass | whole frame |
|---|---|---|---|---|
| wide | on | 3,165,795 | 85 to 105 ms | 295 to 356 ms |
| wide | off | 13,407,645 | 229 to 279 ms | 443 to 517 ms |
| close | on | 2,767,684 | 97 to 108 ms | 328 to 341 ms |
| close | off | 15,667,200 | 252 to 382 ms | 476 to 610 ms |

The machine is shared, so the spread between runs is wide.

Light reach skips 76% of the light evaluations in the wide view and 82% in
the close view. The café's pixels still evaluate its fifteen lights, which
is correct: they light it.

Not measured yet: a Raspberry Pi.

## Not done yet

- Shadows, reflections, global illumination: lights are unshadowed and the
  sky is unoccluded.
- SIMD: the inner loops are scalar.
- Vertex splatting and cluster level of detail.
- Points, curves and volumes in `.fkj` (tags reserved).
- Materials and textures: albedo per vertex only.
- Light reach is per cluster, not per pixel or per tile: a large cluster
  near a light evaluates it on all its pixels.
- The chess set's clusters were baked before clusters were bounded in
  space; the procedural set's are bounded.
- No converter in the repository: the chess set was baked once.
- Light rigs and the sky setting are not saved.
- Text overlay: the numbers are in the window title.

Credit: FkJensen was created by Simon Legrand. The chess set is CC BY 4.0;
see `assets/README.md`.
