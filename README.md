# FkJensen

A 3D viewport that renders on the CPU and nothing else, following the path
3D graphics would have taken if GPUs had never existed (see
`DESIGN_PHILOSOPHY.md`): REYES-style shading kept between frames, shadow
maps, point-based bounce light, no ray tracing. No GPU API is linked; the frame is made on the cores
with rayon and copied to the window through shared memory (softbuffer).

It does not read USD. Scenes are in FkJensen's own `.fkj` format
(`docs/format.md`), with geometry already cut into the clusters the renderer
draws.

The built-in set, all real geometry, in metres:

- a T-rex (10.8 M triangles, a ZBrush sculpt, with its texture baked to
  vertex colours and tiles) and a car (17 K triangles), on a 100 m paved
  ground plane;
- far background: mountains 4 to 12 km away, up to about 2 km high.

On machines whose plan leaves it out (see Fitting the machine), the T-rex is
its 28 K-triangle version.

![The set from the wide view](docs/set_wide.png)

![The set from the close view](docs/set_close.png)

## Light

Three kinds of light: point lights and the sky in physical units, and the
sun.

**Point lights.** A point light is a position, a colour and a luminous flux
in lumens. There is no falloff radius: the light gives lumens / 4π candela,
and a surface facing it at d metres receives that over d² lux. Lumens are
kept between 1 and 2,000,000, about the output of an 18 kW HMI, the largest
common film light, so nothing brighter than a real lamp can be made. The
set opens with three lights sized like film lights on a night exterior: a
1,000,000 lm warm key high to the front left, a 40,000 lm cool fill and a
200,000 lm rim behind.

**The sun.** Infinitely far: parallel light and parallel shadows, the same
on everything in the scene, with no reach. Its brightness runs from 0
(none) to 1000, which lights a white surface facing it to full white at the
current exposure; it is not in lumens. Its shadows come from an
orthographic shadow map in three cascades round the subject (about 1.6
times its radius, 600 m, 25 km), each 4 times the point lights' texels a
side, up to 2048, read with the same 3 x 3 filter (`src/sun.rs`).

**Time of day.** One value, in hours, sets the sun's direction (east at 6,
south at noon at 60 degrees, west at 18) and colour, the sky's light and
colour, the backdrop's colours and the exposure: about 13,000 lux of sky at
noon, 400 at sunset, 3 at the end of civil twilight, 0.05 at night. The
exposure follows the sky three quarters of the way, so noon reads brighter
than dusk and night darker, and stops following below 3 lux so that lamps
carry the night. The set opens at 18:12, dusk, where the sky is 30 lux.

**Lighting contrast**, 0 to 1, shapes the sky's light only: 0 is flat
(every facing gets the same, twice as much: overcast), 0.5 the default
(sideways gets half), 1 overhead only (half as much in all). Together with
the sun's brightness it sets key against fill. **Visible sky contrast**, 0
to 1, is the backdrop only: the zenith to horizon gradient and the sun's
glow; it re-shades nothing.

**The sky.** Light from the whole sky, with nothing between it and any
surface, so a mountain 10 km away and the T-rex are lit by it the same way. It
is set by its illuminance on an upward-facing surface, in lux; the set
opens at 30 lux, dusk. The sky behind the scene is drawn from the same
values.

**Shadows.** The sun's map is above. Each point light has a cube shadow map, 512 x 512 per face
(Williams 1978), read with a 3 x 3 percentage-closer filter for soft edges
(Reeves, Salesin, Cook 1987) and a normal offset against self-shadowing.
A map only holds the geometry inside its light's reach: whatever can shadow
a point the light reaches is nearer the light than that point. Maps are
kept until their light moves or changes strength. `M` turns them off. The
sky is not occluded yet.

**Bounce light.** Point-based colour bleeding (Christensen 2008), no rays
(`src/bounce.rs`). The surfaces become a cloud of discs (surfels): one per
large triangle, up to six per small cluster binned by facing, 561,573 for
the built-in set. Each is lit directly, by the sky and the point lights
through their shadow maps: that lit cloud is the set of micro lights. The
surfels sit in a tree whose nodes carry summed area, normal and mean
radiance.

Bounce light is gathered at receivers: one at each vertex of a large
cluster (a floor, a wall), blended across its triangles, and
one for each small cluster. A receiver walks the tree nearest first, taking
near surfels one by one and far groups as one disc, and lays each onto a
6 x 8 x 8 cube raster around itself: a small or far disc by its solid
angle, a large near one by meeting each texel's direction with the disc.
The raster fills nearest first, so a nearer surface hides what is behind
it; a surfel seen from behind sends nothing but still hides what is
beyond, so a receiver under the car does not see through it. The
raster is folded into order-1 spherical harmonics, from which each pixel
takes the irradiance for its own normal. One bounce. With bounce on, the
sky's ground term is left out: the real ground is in the cloud.

Light reach applies here too. A changed light relights only the surfels in
its reach, and throws away only the receivers within its reach plus the
reach of the brightest change in bounce light. Turning a light up to
2,000,000 lm does not gather the mountains' bounce light again. In the
window, gathering gets 30 ms a frame, nearest the camera first, and catches
up over the following frames; a still render waits for all of it. `I` turns
bounce off.

![Bounce light on the earlier set (a café, 20/10/2026), lit by one 30,000 lm lamp above the awning, without and with bounce: the shadowed shopfront picks up light from the lit pavement](docs/bounce.png)

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
| fill | 40,000 | 184 m |
| rim | 200,000 | 411 m |
| key | 1,000,000 | 920 m |
| brightest allowed | 2,000,000 | 1.3 km |
| a 400 lm table lamp, for scale | 400 | 18 m |

So the three lights cover the ground plane and the near valley, and no
light can reach the mountains, however strong it is made. The tests check
that, and that rendering the whole set with and without light reach
differs by at most one 8-bit step on any pixel.

![Lights evaluated per cluster](docs/light_count.png)

Light count view (`--debug-view light-count`): yellow, all three lights,
on the plane and the models; none on the mountains.

The same bound limits each shadow map to the geometry in its light's reach,
and is meant to limit bounce light when it comes.

## Fitting the machine

At start FkJensen works out what it is running on (`src/hardware.rs`) and
chooses its own settings. It reads:

- CPU model, architecture, cores and top clock (`/proc/cpuinfo`, sysfs);
- SIMD and floating-point features (`is_x86_feature_detected!` on x86-64,
  the `Features` line on ARM);
- cache sizes per level (`/sys/devices/system/cpu/cpu0/cache`);
- total and available memory (`/proc/meminfo`).

And it measures, in 0.1 to 0.7 s:

- memory read bandwidth, on one core and on all, over a buffer twice the
  last cache (within an eighth of free memory and 512 MB);
- shading points per second on all cores, running the renderer's own
  `shade_point` on synthetic points with the sky and two lights.

From those it chooses:

| choice | rule |
|---|---|
| render resolution | a pixel budget: a whole frame shaded in 150 ms at 3 points per pixel; the window or screen renders at the scale that fits it, from 6.25% to 100%, scaled up with nearest pixels |
| `--png` size without `--size` | about 2 s of shading, up to 1280 x 720 |
| high-resolution T-rex | at least 3,200 MB available (peak 2,523 MB measured), and its 10.8 M triangles' memory traffic per frame (100 bytes each) under 250 ms; otherwise the 28 K-triangle T-rex |
| shadow maps | 512, 256 or 128 texels per face by speed, halved until 17 lights fit in a tenth of free memory; off below 64 |
| bounce light | on at an eighth of the reference machine's speed or more, with a gather budget of 10 to 30 ms per window frame |
| bucket size | depth and ids of a bucket in half the L2 cache, 16 to 64 pixels |

It prints what it found and each choice with its reason. `--hardware`
prints them and exits. Any option on the command line overrides the plan
(`--size`, `--scale`, `--shadow-res`, `--[no-]highres`, `--[no-]shadows`,
`--[no-]bounce`); `--no-auto` skips detection and uses fixed defaults.
In the window `Z` and `Shift+Z` change the render scale.

On the machine this was written on (21/10/2026):

    CPU: Intel(R) Xeon(R) Processor @ 2.10GHz (x86_64), 2 cores, up to 2100 MHz
    features: sse2 sse4.1 avx avx2 fma avx512f
    caches (per core or shared): L1 data 48 KB, L1 instruction 32 KB, L2 unified 2048 KB, L3 unified 266240 KB
    memory: 7.84 GB total, 7.28 GB available
    measured: memory read 10.35 GB/s on one core, 18.85 GB/s on all; shading benchmark 16.33 M points/s on all cores (749 ms to measure)
      frames of about 408 K pixels (62% of a 1280 x 720 window): about 8.16 M shading points/s in a frame, 150 ms per frame at 3 points per pixel
      --png without --size renders 1280x720 (about 2 s of shading)
      high-resolution T-rex: 7452 MB available, its triangles take 57 ms of memory traffic per frame
      shadow maps 512 x 512 per face (107.0 MB for 17 lights)
      bounce light on, gathered 29 ms per window frame
      buckets 64 x 64 (32 KB of depth and ids, L2 2048 KB)
      texture tile cache up to 931 MB
      shading cache up to 1862 MB (added 22/10/2026)

The shading cache's budget is kept every frame: past it, clusters off
screen are dropped, least recently seen first; when what is on screen alone
does not fit, micropolygons grow (4, 8, up to 32 pixels apart) until it
does, and shrink back once there is room. A small machine renders coarser
or slower; it does not run out of memory. Without a plan the budget is
1 GB, or 16 MB on 32-bit machines.

On a Raspberry Pi 1 Model B Rev 1 (59 MB available, about 80 K shading
points per second measured in a frame on 20/10/2026) the rules give, as the
tests check: the low-resolution T-rex, no bounce light, 128 x 128 shadow maps, frames
of about 4 K pixels (its 1080 x 1920 screen renders at the 6.25% floor,
68 x 120) and stills of about 304 x 171. Not yet run on the Pi.

## Models and texture tiles

Models come in through `tools/fkj-import`, which reads glTF (.glb, .gltf),
binary FBX and OBJ, and writes `.fkj`:

    cd tools/fkj-import && cargo build --release && cd ../..
    tools/fkj-import/target/release/fkj-import trex.glb -o assets/models/trex.fkj --length 12.3
    tools/fkj-import/target/release/fkj-import car.fbx -o assets/models/car.fkj --length 4.79 --textures path/to/textures
    tools/fkj-import/target/release/fkj-import trex_highres.obj -o assets/models/trex_highres.fkj --length 12.3 --texture trex_basecolor.png

An OBJ's materials are not read: `--texture` gives its base colour
texture. The high-resolution T-rex is a ZBrush export with the same UVs as
the glTF one, so it takes the glTF's base colour texture.

It sets the model on the ground, centred, Y up, in metres (`--length`:
longest horizontal side; `--height`; `--rotate-x`, `--rotate-y` in
degrees), cuts it into clusters, and bakes its base colour textures to
per-triangle tiles, in the manner of Ptex: a triangle's tile at level L is
its colour at the points of the triangle cut into 2^L parts per edge, and a
shading point reads its colour by interpolating between them, with no UVs
and no filtering at render time (`src/texture.rs`, `docs/format.md`). Levels 1 to 5 are stored, each
filtered from the source's mip chain at its own spacing; a triangle's
finest level stops where its points would be finer than the source's
texels or closer than `--texel` (2 mm). A triangle covering two texels or
fewer gets no tiles: its three vertex colours hold them.

At load the tiles stay in the file. When a cluster is shaded at level L,
its tiles at that level are read with one positioned read and kept in a
cache capped by the machine's plan (an eighth of available memory). Only
visible triangles' textures, at the size they are seen at, reach memory.

The built-in set places the T-rex (`trex_highres.fkj`, or `trex.fkj` when
the plan says so or the high-resolution file is missing) and `car.fkj` on
the ground plane. They are in the repository, in `assets/models/`, stored
with Git LFS (credits in `assets/README.md`); clone with Git LFS
installed.
Models are looked for in `$FKJENSEN_ASSETS`, then `assets/models/` and
`assets/local/` in the current folder, in the source tree the program was
built from, and beside the program. `assets/local/` is ignored by git, for
models that are not to be published.

Measured on 21/10/2026 (2 cores of a Xeon at 2.1 GHz):

| model | source | triangles | clusters | tiles | import |
|---|---|---|---|---|---|
| T-rex | glTF, one 4096 x 4096 texture | 28,152 | 770 | 53.4 MB | 2.1 s |
| car | FBX, 17 textures up to 1024 x 1024 | 17,243 | 575 | 37.6 MB | 1.0 s |
| T-rex, high resolution | OBJ (639 MB, 5.4 M quads), the glTF's texture | 10,810,368 | 268,130 | 116.4 MB | 16.3 s |

The high-resolution file is 304 MB: 188 MB of geometry and 116 MB of tiles.
About half its clusters need no tiles, since most of its triangles cover
under two texels.

Decoded at full size, without mips, those textures alone are 71 MB of RGB.
A 1280 x 720 frame of the wide view read 4.1 MB of tiles; one filling the
frame with the T-rex read 3.4 MB.

## Visual debug views

`--visual-debug` adds views that draw each stage of the renderer over the
frame, with a caption saying what the colours mean, for screenshots that
explain how it works. `V` and `Shift+V` step through them in the window;
`P` saves the frame. They are built on 64-bit machines only: they are for
illustration on a capable machine, may read texture tiles and allocate
freely, and nothing in the renderer depends on them, so the Raspberry Pi
build leaves them out.

    cargo run --release -- --visual-debug
    cargo run --release -- --png out.png --debug-view all          one file per view
    cargo run --release -- --png out.png --debug-view reshaded --bench-drag
    cargo run --release -- --png out.png --close --debug-view shadow-map --select 2

`--select N` picks the light the light views show (1 is the first).

![Six of the views: clusters, micropolygons, light reach, a shadow map unfolded, the surfel cloud, dicing levels](docs/debug_views.png)

| view | what it shows |
|---|---|
| `clusters` | one colour per cluster |
| `triangles` | one colour per triangle id: the visibility buffer |
| `depth` | distance from the eye, log scale over the range in view |
| `normals` | world-space normals |
| `buckets` | the bucket grid, with the number of triangles binned into each |
| `dice` | each triangle's dicing level |
| `micropolygons` | the grid each triangle was shaded on, even on screen, edges drawn |
| `reshaded` | clusters shaded (red) and bounce gathered (green) in the last frames |
| `albedo` | base colour: vertex colours or texture tiles |
| `tiles` | the tile level read for each cluster, dim where finer levels stay on disk |
| `sky` | the sky's light alone |
| `direct` | the kept shading, without bounce |
| `highlight` | the highlight alone, x4 |
| `bounce` | bounce light alone, x4 |
| `light-count` | how many lights reach each cluster |
| `reach` | the selected light's illuminance in bands that halve, out to its reach |
| `shadow` | the selected light's shadow test, with its cube map as an inset |
| `sun` | the sun's shadow test, coloured by the cascade each point reads |
| `shadow-map` | the selected light's cube map, unfolded |
| `surfels` | the surfel cloud as lit discs |
| `receivers` | bounce receivers: gathered or waiting |
| `micro-raster` | the 6 x 8 x 8 cube raster of the receiver under the cursor, x4, and what its spherical harmonics give back |

## Without a display server

When neither X nor Wayland is running, as on a Raspberry Pi at its text
console, FkJensen draws straight to the Linux framebuffer (`src/fbdev.rs`)
instead of opening a window: each frame is scaled up to the screen and
converted to the framebuffer's pixel format (16, 24 or 32 bits, read with
the `FBIOGET_VSCREENINFO` and `FBIOGET_FSCREENINFO` ioctls), and the
keyboard and mouse are read from `/dev/input/event*`. The controls are the
same; a small cross shows the mouse. Esc quits and gives the console back.
`--fb` uses the framebuffer even when a display server is running.

The user needs to be in the `video` group (framebuffer) and the `input`
group (keyboard, mouse). Over ssh, the status line that the window title
would show is printed to the ssh terminal every 2 s; on the console itself
it is left out, since it would be drawn over the picture.

The render resolution follows the screen: a frame holds the machine's pixel
budget (`--hardware` prints it), scaled up to fill the screen.

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
    cargo run --release -- --png out.png --view lights --light -10,8,10,2000000
    cargo run --release -- --png out.png --sky 0 --exposure 0.02
    cargo run --release -- --png out.png --look 3,4.5,9,0,4,0

- `--light x,y,z[,lumens]` replaces the scene's lights (800 lm when no
  lumens are given); repeatable. Positions in metres.
- `--time HH:MM` sets the time of day (`--time 12:00`, `--time 18.5`).
- `--sun B` sets the sun's brightness, 0 to 1000 (600 at start).
- `--light-contrast C` and `--sky-contrast C`, 0 to 1 (0.5 at start).
- `--sky LUX` sets the sky, 0 for none, and the exposure to match, after the time of day.
- `--exposure E` sets the display value per cd/m².
- `--view lights` writes the light count view.
- `--no-reach` evaluates every light at every point, to compare.
- `--no-shadows` turns shadow maps off.
- `--no-bounce` turns bounce light off.
- `--close` starts on the close framing.
- `--look ex,ey,ez,tx,ty,tz` puts the camera at e looking at t, in metres.
- `--no-highres` uses the 28 K-triangle T-rex instead of the 10.8 M one,
  for machines with little memory.
- `--shadow-res N` sets the shadow map texels per cube face edge (512).
- After `--png`, the peak memory of the run is printed (Linux).

Peak memory of a 320 x 180 `--png` render on x86-64, 20/10/2026, with the
earlier set (chess table and café; `--no-chess` left the chess set out).
With the T-rex and the car, 1280 x 720, everything on: 2,523 MB
(21/10/2026).

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
    T / Shift+T            time of day later / earlier (15 min): sun, sky, exposure
    Y / Shift+Y            sun brighter / dimmer (0 to 1000, steps of 50)
    J / Shift+J            lighting contrast up / down: flat overcast to overhead
    U / Shift+U            visible sky contrast up / down (the backdrop only)
    K                      sky on / off          , / .    sky dimmer / brighter (x2)
    H                      light count view: how many lights each pixel evaluates
    X                      light reach off / on (to compare cost and picture)
    M                      shadow maps on / off
    I                      bounce light on / off
    Z / Shift+Z            render resolution finer / coarser (x1.25)
    V / Shift+V            next / previous visual debug view (--visual-debug)

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

![Dragging a lamp on the earlier set (20/10/2026): the shadows follow](docs/light_drag.png)

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
   bucket per thread, no atomics.
4. **Shading** (`src/reyes.rs`). Each triangle is diced from its own size
   on screen, whatever the mesh's topology: a grid laid on it as seen,
   rows along its longest edge and rows up to the opposite corner, 2^level
   + 1 points each way so that points are at most two pixels apart on
   screen, up to 256 parts. The grid is even on screen however thin or
   receding the triangle is, and covers the rectangle on that edge, twice
   the triangle; its points are placed on the triangle's plane in world
   space (perspective-correct). A triangle under two pixels shades the
   mesh's own vertices. A kept grid stays while its points are between a
   quarter of the rate and the rate apart on the new view. Only triangles
   under visible pixels are diced and shaded, and only those
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
and writes `.fkj`. `src/world.rs` makes the ground plane, valley and mountains,
and the set's lights. `src/lights.rs` is the light rig and the picking and dragging
maths. `src/shadow.rs` and `src/bounce.rs` are the shadow maps and the
bounce light. `src/texture.rs` is the texture tiles and their cache;
`tools/fkj-import` is the model converter, a separate crate.

## Measured

With the T-rex and the car (10,925,731 triangles, 271,903 clusters, 3
lights, 561,573 surfels), 1280 x 720, 2 cores of an Intel Xeon at 2.1 GHz,
21/10/2026, one run:

| | first frame, wide view | dragging the key light, steps 1 to 5 |
|---|---|---|
| shadow maps | 2,135 ms (3 maps, 32.6 M triangles) | 679 to 831 ms (1 map, 10.9 M triangles) |
| lighting the surfels | 665 ms | 87 to 107 ms (498 K surfels) |
| bounce gather | 7.50 s (20,433 receivers) | 47 to 51 ms (32 receivers; 20,306 waiting) |
| visibility | 2.01 s | kept |
| shading | 1,379 ms (3.7 M points) | 892 to 960 ms (2.7 M points in 20,338 clusters) |
| resolve | 106 ms | 58 to 67 ms |
| total | 13.8 s | 1.77 to 1.95 s |

A drag frame is now dominated by the key light's shadow map holding all
10.9 M triangles of the T-rex, and by re-shading what its 920 m reach
covers, which is everything but the mountains.

The tables below are the earlier set (chess table and café), kept for
comparison. On the machine this was written on (2 cores of an Intel Xeon at 2.1 GHz,
shared, no SIMD, 20/10/2026), the built-in set (1,521,804 triangles, 26,456
clusters, 17 lights with shadow maps, 153,753 surfels), 1280 x 720, two runs
each. These tables are without the T-rex and the car; with the
T-rex and the car (45,395 more triangles, 21/10/2026) the first frame of
the wide view took about 4.0 s, 2.7 s of it bounce gather.

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

Where a drag frame went: re-drawing the lamp's shadow map, which held the
whole chess set, and re-shading every point in the lamp's reach with all
the lights that reach it, not just the one that moved.

Not measured yet: a Raspberry Pi.

## Not done yet

- Reflections (environment and per-object cube maps): not started.
- Visual debug views: no view yet for culling (frustum and normal cone),
  for the surfel tree, or for the texture cache over time; text is a 5 x 7
  pixel font in capitals.
- Bounce light: one bounce only. A large surfel near a receiver covers its
  raster through a disc enlarged by a third, so that tiled discs leave no
  gaps; the outline of a nearby surface counts a little more light than it
  should. Small clusters have one receiver, so a cluster that wraps a tight
  curve gets one set of spherical harmonics for all of it. Gathering is
  the slowest pass of a first frame.
- The sky is not occluded.
- The sun: its shadow cascades are fixed round the subject, not the camera,
  so a close look far from the subject gets coarse sun shadows; its disc is
  drawn larger than the real one so it reads; no moon, no atmosphere or
  aerial perspective; the light count view leaves it out.
- A moved light re-shades its points with every light in reach; keeping
  each light's share apart would let it redo only its own.
- Shadow maps are drawn at full detail: the key light draws all 10.9 M
  triangles of the T-rex every time it moves.
- Adaptive refinement (sparse samples where the picture is smooth) and
  front-to-back early stopping: not started.
- SIMD: the inner loops are scalar.
- Vertex splatting and cluster level of detail.
- Texture tiles: base colour only (no normal or roughness maps); bounce
  light on a textured surface is coloured by its vertex colours; the importer does not read USD or ASCII FBX, and
  reads FBX transforms without pivots or offsets.
- Points, curves and volumes in `.fkj` (tags reserved).
- Materials: albedo only, per vertex or from texture tiles.
- Light reach is per cluster and per point, not per pixel.
- The shading cache holds 118 MB for that view (186 MB before even
  dicing). Only the shading cache has a budget: geometry, shadow maps
  and surfels are as large as the scene makes them. With the
  high-resolution T-rex the program needs
  about 2.5 GB, and with the low-resolution one it still does not fit: geometry is decoded to f32 at load and every visible triangle
  gets a setup record each frame, so it does not fit a 256 MB Raspberry
  Pi 1. Quantised geometry in memory, cluster level of detail and
  per-bucket setup are what it needs.
- Each triangle above two pixels has its own grid, covering twice its
  area; neighbours share no points. Light that changes faster than the
  micropolygons (a lamp 10 cm off a surface diced at 3 cm) leaves a seam
  of about 2 % along their edge. Grids past the finest texture tile
  (level 5) interpolate it.
- At two pixels, shadow edges on the ground show the grid's steps and
  texture detail is softer than with the old topology-led dicing, which
  over-diced. One pixel smooths them: for the view at 16:30 with the
  low-resolution T-rex (`--time 16:30 --sun 900 --no-highres`, 1280 x 720,
  2 cores of a Xeon at 2.1 GHz, 22/10/2026), 4.5 M points and 859 ms of
  shading against 1.3 M and 260 to 277 ms at two pixels (3.7 M and 545 to
  569 ms before this change).
- Dicing down: triangles smaller than two pixels are still shaded at
  their vertices, so the high-resolution T-rex costs a point per vertex;
  cluster level of detail is next.
- `tools/fkj-import` does not read USD (`docs/format.md`).
- Light rigs and the sky setting are not saved.
- Text overlay: the numbers are in the window title.
- Framebuffer output: no vsync and no double buffering, so a frame can
  tear while it is written; dropping a file to open it needs a window.
- Hardware detection reads Linux's `/proc` and `/sys`. On macOS and
  Windows it gets cores, SIMD features and the two measurements, but not
  cache or memory sizes, and assumes enough memory.
- The plan's rules are calibrated on one machine; the Raspberry Pi figures
  come from one run.

Credit: FkJensen was created by Simon Legrand. Model credits are in
`assets/README.md`.
