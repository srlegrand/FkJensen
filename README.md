# FkJensen

A 3D viewport that renders on the CPU and nothing else, following the path
3D graphics would have taken if GPUs had never existed (see
`DESIGN_PHILOSOPHY.md`): REYES-style shading kept between frames, shadow
maps, point-based bounce light, no ray tracing. No GPU API is linked; the frame is made on the cores
with rayon and copied to the window through shared memory (softbuffer).

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

**Shadows.** Each point light has a cube shadow map, 512 x 512 per face
(Williams 1978), read with a 3 x 3 percentage-closer filter for soft edges
(Reeves, Salesin, Cook 1987) and a normal offset against self-shadowing.
A map only holds the geometry inside its light's reach: whatever can shadow
a point the light reaches is nearer the light than that point. Maps are
kept until their light moves or changes strength. `M` turns them off. The
sky is not occluded yet.

**Bounce light.** Point-based colour bleeding (Christensen 2008), no rays
(`src/bounce.rs`). The surfaces become a cloud of discs (surfels): one per
large triangle, up to six per small cluster binned by facing, 153,753 for
the built-in set. Each is lit directly, by the sky and the point lights
through their shadow maps: that lit cloud is the set of micro lights. The
surfels sit in a tree whose nodes carry summed area, normal and mean
radiance.

Bounce light is gathered at receivers: one at each vertex of a large
cluster (a floor, a wall, a table top), blended across its triangles, and
one for each small cluster. A receiver walks the tree nearest first, taking
near surfels one by one and far groups as one disc, and lays each onto a
6 x 8 x 8 cube raster around itself: a small or far disc by its solid
angle, a large near one by meeting each texel's direction with the disc.
The raster fills nearest first, so a nearer surface hides what is behind
it; a surfel seen from behind sends nothing but still hides what is
beyond, so a receiver under the chess board does not see through it. The
raster is folded into order-1 spherical harmonics, from which each pixel
takes the irradiance for its own normal. One bounce. With bounce on, the
sky's ground term is left out: the real ground is in the cloud.

Light reach applies here too. A changed light relights only the surfels in
its reach, and throws away only the receivers within its reach plus the
reach of the brightest change in bounce light. Turning a table lamp up to
2,000,000 lm does not gather the mountains' bounce light again. In the
window, gathering gets 30 ms a frame, nearest the camera first, and catches
up over the following frames; a still render waits for all of it. `I` turns
bounce off.

![The café at night, lit by one 30,000 lm lamp above the awning, without and with bounce light: the shadowed shopfront under the awning picks up light from the lit pavement](docs/bounce.png)

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

The same bound limits each shadow map to the geometry in its light's reach,
and is meant to limit bounce light when it comes.

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
    cargo run --release -- --png out.png --sky 0 --exposure 0.01 --look -6,1.6,-25,0,2.2,-40 --light 1,6,-35,30000

- `--light x,y,z[,lumens]` replaces the scene's lights (800 lm when no
  lumens are given); repeatable. Positions in metres.
- `--sky LUX` sets the sky, 0 for none, and the exposure to match.
- `--exposure E` sets the display value per cd/m².
- `--view lights` writes the light count view.
- `--no-reach` evaluates every light at every point, to compare.
- `--no-shadows` turns shadow maps off.
- `--no-bounce` turns bounce light off.
- `--close` starts on the close framing.
- `--look ex,ey,ez,tx,ty,tz` puts the camera at e looking at t, in metres.
- `--no-chess` leaves the chess set out of the built-in set (about 90 K
  triangles instead of 1.5 M), for machines with little memory.
- `--shadow-res N` sets the shadow map texels per cube face edge (512).
- After `--png`, the peak memory of the run is printed (Linux).

Peak memory of a 320 x 180 `--png` render on x86-64, 20/10/2026:

| | peak memory |
|---|---|
| `--no-chess --no-shadows --no-bounce` | 25 MB |
| `--no-chess --no-bounce --shadow-res 128` | 32 MB |
| `--no-chess --shadow-res 128` | 46 MB |
| `--no-shadows --no-bounce` (with the chess set) | 341 MB |
| everything on | 473 MB |
- `--bench-drag` (with `--png`) then moves the first light 10 cm five times,
  as a drag would, and times each frame.

The window title shows the frame rate, the selected light's lumens,
position and reach, the sky, whether visibility was kept from the last
frame, how many shading points were shaded out of those in view, shadow map
time, how many receivers are still waiting for bounce light, and the time
of each pass in milliseconds.

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
    M                      shadow maps on / off
    I                      bounce light on / off

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

![Dragging the table lamp from left to right: the shadows follow](docs/light_drag.png)

## How a frame is made

1. **What changed.** Lights are compared with the last frame's. A changed
   light throws away the shading of the clusters inside its old and new
   reach, and its shadow map if it moved or changed strength. A change of
   sky, exposure, light reach, shadows or number of lights throws away all
   shading.
2. **Shadow maps** for lights that have none (`src/shadow.rs`), in parallel.
   Then **surfels** inside a changed light's reach are lit again, and the
   bounce receivers that change could reach are thrown away.
3. **Visibility** (`src/raster.rs`), skipped while the camera has not moved.
   Clusters are culled against the view frustum as a whole (and their
   normal cone, with back-face culling on), then their triangles are set
   up, binned into 64 x 64 buckets and hidden with a depth buffer, one
   bucket per thread, no atomics. Each visible cluster gets a dicing level
   from the size of its triangles on screen.
4. **Shading** (`src/reyes.rs`). Triangles are diced into micropolygons in
   world space, 2^level parts per edge so that a micropolygon is about two
   pixels across, up to 32 parts; level 0 shades the mesh's own vertices.
   Only triangles under visible pixels are diced and shaded, and only those
   whose shading was not kept: radiance in cd/m² from the sky, then each
   light in reach through its shadow map, Lambert diffuse and a small
   Blinn-Phong highlight. Diffuse light is kept until a light, the sky or
   the exposure throws it away; the highlight depends on the eye and is
   redone when the camera moves. Shading is two-sided: normals face the
   eye, and a triangle seen from its other side is shaded again.
5. **Bounce gather** at visible receivers that have none, nearest the
   camera first, within the time budget.
6. **Resolve.** Each pixel interpolates the shaded micropolygon under it
   and adds bounce light from its receivers for its normal. While
   visibility is kept, only pixels of clusters shaded or gathered this
   frame are resolved again; the rest of the picture is kept.

`src/scene.rs` holds the scene and builds clusters. `src/fkj.rs` reads
and writes `.fkj`. `src/world.rs` makes the table, square, café, valley and
mountains. `src/lights.rs` is the light rig and the picking and dragging
maths. `src/shadow.rs` and `src/bounce.rs` are the shadow maps and the
bounce light.

## Measured

On the machine this was written on (2 cores of an Intel Xeon at 2.1 GHz,
shared, no SIMD, 20/10/2026), the built-in set (1,521,804 triangles, 26,456
clusters, 17 lights with shadow maps, 153,753 surfels), 1280 x 720, two runs
each.

First frame, nothing kept, everything gathered:

| | wide view | close view |
|---|---|---|
| shadow maps (17) | 527 to 613 ms | 517 to 599 ms |
| lighting the surfels | 127 to 128 ms | 112 to 114 ms |
| bounce gather | 1.63 to 1.74 s (1,467 clusters) | 2.42 to 2.45 s (7,195 clusters) |
| visibility | 222 to 240 ms | 197 to 211 ms |
| shading | 240 to 309 ms (2.2 M points) | 113 to 120 ms (1.1 M points) |
| resolve | 65 to 69 ms | 75 to 76 ms |
| total | 2.89 to 3.04 s | 3.45 to 3.56 s |

Dragging the table lamp, `--bench-drag`, steps 1 and 5 of each run, with
the window's 30 ms bounce budget:

| | wide view | close view |
|---|---|---|
| visibility | kept | kept |
| shadow map (1, 1.44 M triangles) | 98 to 103 ms | 96 to 129 ms |
| lighting the surfels | 13 to 21 ms (65 K surfels) | 15 to 16 ms (65 K surfels) |
| bounce gather | 32 to 36 ms (128 clusters; about 1,050 waiting) | 31 to 39 ms (128 clusters; about 7,060 waiting) |
| shading | 89 to 140 ms (554 K points) | 101 to 115 ms (937 K points) |
| resolve | 33 to 41 ms | 64 to 73 ms |
| total | 272 to 338 ms | 321 to 350 ms |

While a light is being dragged, the bounce light near it lags; after the
drag stops it catches up, about 2 s for the chess set in the close view.
In the window, the set's bounce light was complete about 4 s after start.
A frame where nothing changed takes 1 to 3 ms.

Light reach, first frame of the wide view without bounce, on and off:

| | reach on | reach off |
|---|---|---|
| triangles drawn into shadow maps | 4,364,338 | 25,913,128 |
| shadow maps | 609 ms | 1,371 ms |
| light evaluations | 3,991,073 | 37,537,156 |
| shading | 338 ms | 2,047 ms |

Where a drag frame goes: re-drawing the lamp's shadow map, which holds the
whole chess set, and re-shading every point in the lamp's reach with all
the lights that reach it, not just the one that moved.

Not measured yet: a Raspberry Pi.

## Not done yet

- Reflections (environment and per-object cube maps): not started.
- Bounce light: one bounce only. A large surfel near a receiver covers its
  raster through a disc enlarged by a third, so that tiled discs leave no
  gaps; the outline of a nearby surface counts a little more light than it
  should. Small clusters have one receiver, so a cluster that wraps a tight
  curve gets one set of spherical harmonics for all of it. Gathering is
  the slowest pass of a first frame.
- The sky is not occluded.
- A moved light re-shades its points with every light in reach; keeping
  each light's share apart would let it redo only its own.
- Shadow maps are drawn at full detail: a lamp over the chess set draws
  all 1.4 M of its triangles every time it moves.
- Adaptive refinement (sparse samples where the picture is smooth) and
  front-to-back early stopping: not started.
- SIMD: the inner loops are scalar.
- Vertex splatting and cluster level of detail.
- Points, curves and volumes in `.fkj` (tags reserved).
- Materials and textures: albedo per vertex only.
- Light reach is per cluster and per point, not per pixel.
- The shading cache holds 142 to 178 MB for the views above and is not
  capped or evicted yet. With the chess set the program needs at least
  341 MB: geometry is decoded to f32 at load and every visible triangle
  gets a setup record each frame, so it does not fit a 256 MB Raspberry
  Pi 1. Quantised geometry in memory, cluster level of detail and
  per-bucket setup are what it needs.
- Shading points are not shared between neighbouring triangles above
  level 0, so edges are shaded twice.
- No converter in the repository: the chess set was baked from USD outside
  it (`docs/format.md`).
- Light rigs and the sky setting are not saved.
- Text overlay: the numbers are in the window title.

Credit: FkJensen was created by Simon Legrand. The chess set is CC BY 4.0;
see `assets/README.md`.
