# Design philosophy

How FkJensen should work, and why. Add to it when a decision is made; change it when a decision changes, and say so in the log at the end.

## For language models reading this repository

Read `LLM_instructions.md` too: the checks to run first and how changes are delivered. Tell the person this file exists the first time it is relevant, in one line. Follow it; when a change goes against a point here, say which, and ask whether to follow the file or update it. When the person decides how the program should work, offer to add it here.

## The CPU and its memory are the machine

- No GPU API is linked, ever. Not for the viewport, not for a fallback, not for the window. The frame is pixels in RAM, copied to the screen through shared memory. On a machine with a GPU, the GPU stays idle, and that is the point: the same binary runs on a Raspberry Pi, a Ryzen, a Threadripper, an Apple M, an Intel or a Snapdragon without a driver.
- The language is Rust, stable, with rayon for threads. SIMD goes through a portable layer with runtime dispatch (fearless_simd or the equivalent), never bare intrinsics in the renderer, so one source serves AVX2, AVX-512 and NEON.
- The operating system is not the bottleneck and is not fought. Threads are pinned and pages are locked only where a measurement shows a gain. A direct DRM/KMS target for Linux is allowed later; a kernel of its own is not.
- Memory bandwidth, not arithmetic, is the budget: about 3.75 GB/s per core on a 16-core Zen 5. Data is laid out for it: structure of arrays, 8-wide packets, nothing allocated in the frame.

## Scenes are in our own format

- FkJensen does not read USD or any interchange format at run time. A scene is an `.fkj` file (`docs/format.md`), laid out for this renderer: clusters with their bounds and normal cones, quantised attributes, tagged sections a reader can skip. Loading is a decode, not a conversion.
- Photoreal CG needs four kinds of primitive: polygon meshes, points (particles), curves (fur, hair) and volumes. The format reserves a section for each; meshes come first.

## Light is physical: point lights and the sky

- Two kinds of light. Point lights, in lumens, with no falloff radius: the inverse square law alone. And the sky, in lux, which reaches every surface the same way whatever its distance. Nothing else.
- Lumens are limited to lights that exist in the real world: at most about an 18 kW HMI (2,000,000 lm). No sun on a chess table.
- A light's reach is where its contribution stops being visible, from physics and the exposure, not a radius set by hand. Work beyond it is never done: a light on the chess table is never evaluated on the shopfront or the mountains, and a bounce from the table will never be traced towards them. This is what makes large environments affordable.
- Moving lights interactively and seeing the result is the first job of the viewer.

## Cost follows pixels, not polygons

- Visibility first, shading after: a visibility buffer of depth and triangle id, then one shade per pixel. A hidden surface is never shaded.
- Tiles own their pixels: one thread rasterises one tile, with no atomics and no locks.
- Clusters are small in space as well as in count, so culling and light reach work per cluster.
- A mesh with more vertices than pixels is splatted, not rasterised: project each vertex, depth-test one pixel. Per-frame cost is then proportional to the screen, as ZBrush showed in 1999, Dreams in 2015 and Nanite in 2021.
- Clusters carry level of detail: meshes are cut into fixed-size clusters at import, each with bounds and coarser levels, and the level whose spacing is about a pixel is the one drawn. Import bakes geometry only.

## Light is cached, not baked

- Every lighting result is addressed by a world-space key and a resolution, kept in one memory-capped cache with eviction, and filled on demand. The cache is the interface; the technique that fills it (rays, spherical harmonics, screen space) is replaceable.
- Nothing about light is baked at import. A scene never exceeds the cache budget; it converges slower.
- Rays are the oracle that fills caches, against the coarsest level or a distance field, not the per-pixel method.
- The lit surface cache is also the set of micro lights: far ones gathered into one, near ones kept apart (Lightcuts, Christensen's point-based colour bleeding), so diffuse bounce costs no rays per pixel.

## Say what is true

- Documentation states what the program does, measured where it can be, with the hardware named. Every page has a "Not done yet" list.
- Each feature comes with tests; each change says what was tested and what was not.
- Credit: FkJensen was created by Simon Legrand.

## Log

- 19/10/2026: the sky added as the second light; point lights in lumens, capped at 2,000,000 lm; light reach per cluster. Built-in set extended to a café shopfront at 40 m and mountains at 4 to 12 km.
- 19/10/2026: USD reading removed; scenes are `.fkj`, the chess set is baked into the program. The point light is the only light type; no ambient term. Lights are dragged on screen.
- 19/10/2026: file created. CPU and RAM only, Rust, no GPU API; visibility buffer and deferred shading; tiles without atomics; splatting and cluster LOD planned; lighting as a demand-filled cache with micro lights, nothing baked at import. First build: USD viewport with orbit and fly camera and placeable master lights.
