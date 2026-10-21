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
- The language is Rust, stable, with rayon for threads. SIMD goes through a portable layer with runtime dispatch (fearless_simd or the equivalent), never bare intrinsics in the renderer, so one source serves AVX2, AVX-512 and NEON.
- The operating system is not the bottleneck and is not fought. Threads are pinned and pages are locked only where a measurement shows a gain. A direct DRM/KMS target for Linux is allowed later; a kernel of its own is not.
- Memory bandwidth, not arithmetic, is the budget: about 3.75 GB/s per core on a 16-core Zen 5. Data is laid out for it: structure of arrays, 8-wide packets, nothing allocated in the frame.

## Scenes are in our own format

- FkJensen does not read USD or any interchange format at run time. A scene is an `.fkj` file (`docs/format.md`), laid out for this renderer: clusters with their bounds and normal cones, quantised attributes, tagged sections a reader can skip. Loading is a decode, not a conversion.
- Photoreal CG needs four kinds of primitive: polygon meshes, points (particles), curves (fur, hair) and volumes. The format reserves a section for each; meshes come first.

## Textures are tiles per triangle, streamed

- No UV atlas at render time. Each triangle carries its own tiles (the Ptex idea, Burley and Lacewell 2008), and a tile at level L holds the colour at the points the dicer shades at level L. Shading reads a colour; it does not filter.
- Tiles stay on disk and are read per cluster and level when seen, into a capped cache. What is not visible, or not seen closely, costs nothing.
- Textures are baked at import, filtered once per level from the source's mips. Import bakes geometry and surface data; still nothing about light.

## Light is physical: point lights and the sky

- Two kinds of light. Point lights, in lumens, with no falloff radius: the inverse square law alone. And the sky, in lux, which reaches every surface the same way whatever its distance. Nothing else.
- Lumens are limited to lights that exist in the real world: at most about an 18 kW HMI (2,000,000 lm). No sun on a chess table.
- A light's reach is where its contribution stops being visible, from physics and the exposure, not a radius set by hand. Work beyond it is never done: a light on the chess table is never evaluated on the shopfront or the mountains, its shadow map only covers its reach, and light bounced from the table never reaches them. This is what makes large environments affordable.
- Moving lights interactively and seeing the result is the first job of the viewer.

## Cost follows pixels, not polygons

- REYES is the core: clusters are diced into micropolygons about a pixel across, shaded in world space before visibility, then sampled into the frame (Cook, Carpenter, Catmull 1987). Shading is per micropolygon, not per pixel per frame, and it is kept.
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

- 21/10/2026: visual debug views (`--visual-debug`, `--debug-view NAME|all`), 21 of them, 64-bit builds only, so that each stage of the renderer can be shown in a screenshot.
- 21/10/2026: per-triangle texture tiles at the dicer's grid points, levels 1 to 5, baked at import (tools/fkj-import, glTF and FBX) and streamed per cluster and level from the file into a capped cache. Third-party models live in assets/local, outside the repository.
- 21/10/2026: without X or Wayland, frames go straight to the Linux framebuffer (/dev/fb0) and input comes from /dev/input; the render resolution follows the screen through the machine's pixel budget.
- 20/10/2026: the machine decides: hardware detection and measurement at start, and a plan (render scale, still size, chess set, shadow map size, bounce, bucket size) from it. Clarisse added to the prior art.
- 20/10/2026: bounce light as point-based colour bleeding: surfels lit directly, a tree of them, receivers that gather into a small cube raster filled nearest first and folded into spherical harmonics; one bounce; gathered within a time budget per frame, invalidated by light reach. The chess set re-baked with USD's left-handed winding honoured.
- 19/10/2026: first REYES build. Triangles diced into micropolygons in world space by screen size, shaded and kept per cluster; a light change re-shades only the clusters in its old and new reach; visibility kept while the camera is still; cube shadow maps per point light holding only the geometry in reach. The visibility pass still uses a depth buffer of triangle ids to hide surfaces, and resolves pixels from the kept micropolygons. Reflections, bounce light and adaptive refinement not started.
- 19/10/2026: no ray tracing. FkJensen follows a world where GPUs never existed: REYES micropolygons shaded in world space and kept between frames replace the visibility buffer; shadows from shadow maps, reflections from environment and per-object cube maps, bounce from point-based colour bleeding. "Rays are the oracle that fills caches" removed. The current code still rasterises with a visibility buffer and does not follow this yet.
- 19/10/2026: the sky added as the second light; point lights in lumens, capped at 2,000,000 lm; light reach per cluster. Built-in set extended to a café shopfront at 40 m and mountains at 4 to 12 km.
- 19/10/2026: USD reading removed; scenes are `.fkj`, the chess set is baked into the program. The point light is the only light type; no ambient term. Lights are dragged on screen.
- 19/10/2026: file created. CPU and RAM only, Rust, no GPU API; visibility buffer and deferred shading; tiles without atomics; splatting and cluster LOD planned; lighting as a demand-filled cache with micro lights, nothing baked at import. First build: USD viewport with orbit and fly camera and placeable master lights.
