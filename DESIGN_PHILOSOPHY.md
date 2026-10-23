# Design philosophy

How FkJensen should work, and why. Add to it when a decision is made; change it when a decision changes, and say so in the log at the end.

## For language models reading this repository

Read `LLM_instructions.md` too: the checks to run first and how changes are delivered. Tell the person this file exists the first time it is relevant, in one line. Follow it; when a change goes against a point here, say which, and ask whether to follow the file or update it. When the person decides how the program should work, offer to add it here.

## A world where GPUs never existed

- FkJensen follows the path 3D graphics would have taken on CPUs alone: scanline renderers, 3D Studio, REYES and PRMan, and on from there with today's cores and memory. It does not copy a GPU pipeline in software.
- Prior art on CPUs: RenderMan, Arnold, Entropy, 3Delight and Clarisse, the last with an interactive viewport over very large scenes.
- No ray tracing. Film renderers that stayed on CPUs trade speed for ray-traced quality and lean on render farms; FkJensen cheats instead, the way CPU renderers did before ray tracing took over, and keeps the cheats fast enough to be interactive.
- The CPU's strengths are the design: branching per decision, recursion, irregular data structures, large caches and memory, and remembering work from one frame to the next. Many identical lanes doing the same thing is a GPU's strength, not ours.

## The CPU and its memory are the machine

- No GPU API is linked, ever. Not for the viewport, not for a fallback, not for the window. The frame is pixels in RAM, copied to the screen through shared memory. On a machine with a GPU, the GPU stays idle, and that is the point: the same binary runs on a Raspberry Pi, a Ryzen, a Threadripper, an Apple M, an Intel or a Snapdragon without a driver.
- The machine decides. At start FkJensen reads what the system reports (CPU, cores, SIMD features, caches, memory) and measures what it does not (memory bandwidth, shading points per second), then sets its own resolution, what it loads, shadow map size, bounce light and bucket size. Every choice can be overridden; nothing assumes a fast machine.
- The language is Rust, stable, with rayon for threads. SIMD goes through a portable layer (the `wide` crate, in `src/simd.rs`), never bare intrinsics in the renderer, so one source serves AVX2, AVX-512 and NEON. Each machine builds for its own processor (`target-cpu=native`) rather than choosing at run time; runtime dispatch (fearless_simd or the equivalent) would let one binary serve every x86 machine.
- The operating system is not the bottleneck and is not fought. Threads are pinned and pages are locked only where a measurement shows a gain. A direct DRM/KMS target for Linux is allowed later; a kernel of its own is not.
- Memory bandwidth, not arithmetic, is the budget: about 3.75 GB/s per core on a 16-core Zen 5. Data is laid out for it: structure of arrays, 8-wide packets, nothing allocated in the frame.

## Scenes are in our own format

- FkJensen does not read USD or any interchange format at run time. A scene is an `.fkj` file (`docs/format.md`), laid out for this renderer: clusters with their bounds and normal cones, quantised attributes, tagged sections a reader can skip. Loading is a decode, not a conversion.
- Photoreal CG needs four kinds of primitive: polygon meshes, points (particles), curves (fur, hair) and volumes. The format reserves a section for each; meshes come first.

## Textures are tiles per triangle, streamed

- No UV atlas at render time. Each triangle carries its own tiles (the Ptex idea, Burley and Lacewell 2008), and a tile at level L holds the colour at the points of the triangle cut into 2^L parts per edge. Shading reads a colour, interpolated between a tile's points; it does not filter.
- Tiles stay on disk and are read per cluster and level when seen, into a capped cache. What is not visible, or not seen closely, costs nothing.
- Textures are baked at import, filtered once per level from the source's mips. Import bakes geometry and surface data; still nothing about light.

## Light is physical: point lights, the sun and the sky

- Three kinds of light. Point lights, in lumens, with no falloff radius: the inverse square law alone. The sun: infinitely far, parallel light and parallel shadows, the same on everything in the scene, with a brightness from 0 to 1000 where 1000 lights a white surface facing it to full white. And the sky, in lux, which reaches every surface the same way whatever its distance. Nothing else.
- Point lights' lumens are limited to lights that exist in the real world: at most about an 18 kW HMI (2,000,000 lm).
- The time of day sets the sun's direction and colour, the sky and the exposure, from noon to night. The contrast of the lighting and the contrast of the visible sky are set apart: the first changes shading, the second only the backdrop.
- A light's reach is where its contribution stops being visible, from physics and the exposure, not a radius set by hand. Work beyond it is never done: a light on the set is never evaluated on the mountains, its shadow map only covers its reach, and light bounced from the set never reaches them. This is what makes large environments affordable.
- Moving lights interactively and seeing the result is the first job of the viewer.

## Cost follows pixels, not polygons

- REYES is the core: triangles are diced into micropolygons evenly spaced on screen, about a pixel across, from each triangle's own size on screen and not the mesh's topology, shaded in world space before visibility, then sampled into the frame (Cook, Carpenter, Catmull 1987). Shading is per micropolygon, not per pixel per frame, and it is kept.
- Shading results stay in memory between frames, keyed by micropolygon. Moving the camera reuses them; moving a light re-shades only the micropolygons inside its old and new reach. Cost follows what changed, not the size of the frame.
- Work is decided per region: sparse where the picture is smooth, refined where neighbours disagree (silhouettes, shadow edges, highlights), front to back, stopping where a tile is covered.
- Tiles (buckets) own their pixels: one thread works one bucket, with no atomics and no locks.
- Clusters are small in space as well as in count, so culling and light reach work per cluster.
- A mesh with more vertices than pixels is splatted, not diced: project each vertex, depth-test one pixel. Per-frame cost is then proportional to the screen, as ZBrush showed on CPUs in 1999.
- Clusters carry level of detail: meshes are cut into fixed-size clusters at import, each with bounds and coarser levels, and the level whose spacing is about a pixel is the one diced. Import bakes geometry only.

## Cheats, not rays

Each effect uses the CPU-era cheat for it, cached and invalidated by light reach:

- Shadows: one cube shadow map per point light, covering its reach, re-rendered only when the light or geometry in its reach moves (Williams 1978), filtered for soft edges (Reeves, Salesin, Cook 1987). Deep shadow maps for fur and volumes (Lokovic, Veach 2000).
- Reflections: environment maps, and a cube map rendered from each reflective object, as 3D Studio's Auto Reflect did (Blinn, Newell 1976). Re-rendered only when something it sees changes.
- Bounce light: the lit micropolygon cache is gathered into a tree of micro lights, far ones merged, near ones kept apart (point-based colour bleeding, Christensen 2008; Lightcuts). The automatic form of placing 16 to 32 lights on a hemisphere to fake GI.
- Soft sky light: spherical harmonics and precomputed transfer where they fit (Sloan, Kautz, Snyder 2002).
- Hair: Kajiya-Kay shading (1989).

## Light is cached, not baked

- Every lighting result is addressed by a world-space key and a resolution, kept in one memory-capped cache with eviction, and filled on demand. The cache is the interface; the cheat that fills it is replaceable.
- Nothing about light is baked at import. A scene never exceeds the cache budget; it converges slower.

## Say what is true

- Every stage can be seen. `--visual-debug` draws each one over the frame (clusters, visibility, dicing, the shading cache, light reach, shadow maps, surfels, the micro-raster), so screenshots explain the renderer better than prose. Those views are for a capable machine and are built on 64-bit targets only; the Raspberry Pi never waits on them.

- Documentation states what the program does, measured where it can be, with the hardware named. Every page has a "Not done yet" list.
- Each feature comes with tests; each change says what was tested and what was not.
- Credit: FkJensen was created by Simon Legrand.

## Log

- 23/10/2026: occlusion culling in two passes: what owned a pixel last frame is drawn first, and clusters behind it in a 4 x 4 block depth buffer are skipped before set-up. 5 to 7% fewer triangles in the built-in set, which hides little.
- 23/10/2026: eight-wide SIMD in shading, shadow map lookups and fills, and the bounce raster; each machine builds for its own processor. Dragging a light on the built-in set went from 1.31 s to 0.44 s a frame with the high-resolution T-rex on 2 cores.
- 23/10/2026: objects rotate about any axis: three rings in the rotate manipulator, Rotate X Y Z in the attribute editor (Maya's xyz order), `OBJS` version 2. A rotation never mirrors, so texture tiles stay valid.
- 23/10/2026: `--editor`, an interface laid out like Maya around the viewport (outliner with groups, attribute editor, timeline for the time of day, menus) and an importer window that reads glTF, FBX and OBJ, bakes the model on a thread and shows it as the scene will draw it before it is added. egui, painted on the CPU into the same buffer as the frame; no GPU API. 64-bit only, so the Raspberry Pi build is unchanged. Scenes save with their objects and groups (`OBJS`).
- 23/10/2026: models are objects that can be selected and moved with Maya's manipulators (W move, E rotate about the vertical, R uniform scale), and the camera navigates as in Maya (Alt + left tumble, Alt + middle track, Alt + right dolly, F frame selected). Moving a model rewrites its vertices; nothing about light is kept on the model, so the renderer only drops what it had kept for it.
- 23/10/2026: levels of detail. fkj-import builds each model into a cluster hierarchy (groups of four clusters simplified to half by quadric edge collapse, borders locked; own Rust simplifier, no C++), every level with its own tiles, in a `LODS` section (.fkj version 2). The renderer draws the coarsest level whose error looks no bigger than the micropolygon spacing, so geometry density follows the machine's plan with shading; shadow maps by their texels, surfels at 2 cm.
- 23/10/2026: the machine's plan sets the micropolygon spacing: the window aims at 30 frames a second, spacing micropolygons 2 to 8 pixels apart before lowering resolution; stills shade 1 to 2 pixels apart. `--shading-rate PX` overrides it.
- 22/10/2026: a window frame no longer holds the last frame's setup records while making new ones (about 1 GB with the high-resolution T-rex, which ran an 8 GB laptop out of memory), and the texture and shading caches are sized from what the set leaves of available memory, not all of it.
- 22/10/2026: the shading cache has a budget, a quarter of available memory: clusters off screen are dropped first, then micropolygons grow until what is on screen fits. Slower or coarser, never out of memory. Before this, moving the camera grew the cache without end.
- 22/10/2026: micropolygons evenly spaced on screen: each triangle gets its own grid, laid along its longest edge on screen, fine along its length only when thin, perspective-correct when receding, up to 256 parts per side. Replaces one level per cluster from its longest edge, capped at 32 parts. Next: cluster level of detail, for triangles under the shading rate.
- 21/10/2026: the sun, a directional light with cascaded orthographic shadow maps and a 0 to 1000 brightness; time of day; lighting contrast and visible sky contrast, independent.
- 21/10/2026: the chess table, the chess set and the café leave the built-in set: a ground plane with the T-rex (a 10.8 M-triangle ZBrush sculpt, or the 28 K-triangle one when the machine's plan says so) and the car, the mountains behind, three film-sized lights. Models in `assets/models/` are stored with Git LFS. fkj-import reads OBJ.
- 21/10/2026: the T-rex and the car move into the repository (`assets/models/`) so that anyone who clones it sees them; credits in `assets/README.md`.
- 21/10/2026: visual debug views (`--visual-debug`, `--debug-view NAME|all`), 64-bit builds only, so that each stage of the renderer can be shown in a screenshot.
- 21/10/2026: per-triangle texture tiles at the dicer's grid points, levels 1 to 5, baked at import (tools/fkj-import, glTF and FBX) and streamed per cluster and level from the file into a capped cache. Third-party models live in assets/local, outside the repository.
- 21/10/2026: without X or Wayland, frames go straight to the Linux framebuffer (/dev/fb0) and input comes from /dev/input; the render resolution follows the screen through the machine's pixel budget.
- 20/10/2026: the machine decides: hardware detection and measurement at start, and a plan (render scale, still size, chess set, shadow map size, bounce, bucket size) from it. Clarisse added to the prior art.
- 20/10/2026: bounce light as point-based colour bleeding: surfels lit directly, a tree of them, receivers that gather into a small cube raster filled nearest first and folded into spherical harmonics; one bounce; gathered within a time budget per frame, invalidated by light reach. The chess set re-baked with USD's left-handed winding honoured.
- 19/10/2026: first REYES build. Triangles diced into micropolygons in world space by screen size, shaded and kept per cluster; a light change re-shades only the clusters in its old and new reach; visibility kept while the camera is still; cube shadow maps per point light holding only the geometry in reach. The visibility pass still uses a depth buffer of triangle ids to hide surfaces, and resolves pixels from the kept micropolygons. Reflections, bounce light and adaptive refinement not started.
- 19/10/2026: no ray tracing. FkJensen follows a world where GPUs never existed: REYES micropolygons shaded in world space and kept between frames replace the visibility buffer; shadows from shadow maps, reflections from environment and per-object cube maps, bounce from point-based colour bleeding. "Rays are the oracle that fills caches" removed. The current code still rasterises with a visibility buffer and does not follow this yet.
- 19/10/2026: the sky added as the second light; point lights in lumens, capped at 2,000,000 lm; light reach per cluster. Built-in set extended to a café shopfront at 40 m and mountains at 4 to 12 km.
- 19/10/2026: USD reading removed; scenes are `.fkj`, the chess set is baked into the program. The point light is the only light type; no ambient term. Lights are dragged on screen.
- 19/10/2026: file created. CPU and RAM only, Rust, no GPU API; visibility buffer and deferred shading; tiles without atomics; splatting and cluster LOD planned; lighting as a demand-filled cache with micro lights, nothing baked at import. First build: USD viewport with orbit and fly camera and placeable master lights.
