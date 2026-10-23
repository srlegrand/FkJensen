# The .fkj scene format

FkJensen does not read USD. Scenes are stored in its own format, laid out
for the way this renderer draws: geometry already cut into small clusters,
each with the bounds the renderer culls with, so loading is a decode and the
first frame needs no preparation. `src/fkj.rs` reads and writes it.

The four kinds of primitive the format is meant to hold are triangle meshes,
points (particles), curves (fur, hair) and volumes. Version 1 implements
meshes; the other three have reserved tags and are skipped by this reader.
Version 2 is version 1 with a `LODS` section: a mesh built into levels of
detail. A file with levels is written as version 2 so that a reader that
does not know them refuses it, rather than drawing every level at once.

## Layout

Little-endian throughout. A file is a header and a list of sections.

    header    16 bytes
      [u8; 4]   "FKJ\0"
      u32       version, 1 (2 with LODS)
      u32       number of sections
      u32       0

    section
      [u8; 4]   tag
      u32       section version, 1
      u64       payload length in bytes
      payload
      zero padding to a multiple of 8 bytes

A reader skips any section whose tag it does not know, using the length.

| tag | contents | read by this build |
|---|---|---|
| `MESH` | clustered triangle mesh | yes |
| `INST` | extra copies of mesh clusters at other transforms | yes |
| `LITE` | point lights, lumens (section version 2) | yes; version 1 skipped |
| `TEXR` | per-triangle texture tiles | yes, streamed from the file |
| `LODS` | each cluster's level of detail | yes |
| `OBJS` | objects and groups, for the editor | yes |
| `PNTS` | points, particles | reserved |
| `CURV` | curves, fur and hair | reserved |
| `VOLM` | volumes | reserved |

## MESH

All geometry is in world space, metres, Y up.

    u32       cluster count C
    u32       vertex count V
    u32       triangle count T
    u32       0
    f32 x 3   bounds min
    f32 x 3   bounds max
    C x 64 bytes   clusters
    V x 4 bytes    normals: two u16, octahedral encoding
    V x 4 bytes    albedo: sRGB u8 red, green, blue, then 255
    V x 6 bytes    positions: three u16 per vertex, cluster by cluster
    T x 3 bytes    triangles: three u8 indices local to their cluster

A cluster holds at most 64 vertices and 124 triangles, so a triangle's
corners fit in one byte each. It is also kept small in space: the builder
closes a cluster before its box grows past 16 typical triangle sizes, since
a cluster that spans a whole mesh defeats frustum culling and light reach. Clusters own contiguous runs of vertices and
triangles and are stored in that order.

    cluster, 64 bytes
      u32       first vertex
      u16       vertex count
      u16       triangle count
      u32       first triangle
      u32       0
      f32 x 3   bounding sphere centre
      f32       bounding sphere radius (includes quantisation error)
      f32 x 3   position origin
      f32 x 3   position step
      i8 x 3    normal cone axis, snorm (value / 127)
      u8        0
      f32       normal cone cosine: every face normal n has
                n . axis >= this; 0 or less means do not cone-cull

A position is `origin + q * step` with `q` the three u16 values, so the
precision is the cluster's size divided by 65535: about 0.2 micrometres for
a 1.5 cm cluster of the chess set.

The normal cone lets the renderer drop a cluster whose every face points
away from the camera, with back-face culling on.

## INST

    u32       group count
    u32       0
    per group
      u32       first cluster
      u32       cluster count
      u32       transform count N
      u32       0
      N x 12 f32  affine transforms: rows 0, 1, 2 of a 3 x 4 matrix

Each transform makes one more copy of the group's clusters, applied to the
stored world-space geometry. The stored copy is drawn too. The reader
expands instances into plain clusters at load: they cost memory like any
other geometry, and only save file size. The chess set uses it for the
pawns: one pawn per side is stored, seven copies each come from `INST`.

## TEXR

Texture tiles, one set per triangle, in the manner of Ptex (Burley and
Lacewell 2008): no UV atlas at render time. A triangle's tile at level L
holds its colour at the (2^L + 1)(2^L + 2) / 2 points of the triangle cut
into 2^L parts per edge; a shading point, wherever the dicer puts it, reads
its colour interpolated between them. Level 0 is the three corners: the
vertex colours of `MESH`.

    u32       cluster count (the MESH section's)
    u32       0
    u64       bytes from the start of this payload to the pool
    per cluster, 16 bytes
      u8        finest level stored, 0 for none (vertex colours only)
      u8 x 7    0
      u64       offset of its tiles in the pool
    pool

A cluster's tiles are its levels 1 to its finest, one after the other;
each level holds its triangles' tiles in the cluster's triangle order, each
tile its points in order (i towards the triangle's second corner, j
towards its third), 3 bytes of sRGB each. The finest level is at most 5
(561 points a triangle).

The reader keeps the table and leaves the pool in the file. When a cluster
is shaded with grids up to level L, the block of its tiles at min(L, 5,
finest) is read with
one positioned read and kept in a memory-capped cache, least recently used
dropped first. Only the tiles of visible triangles, at the level they are
seen at, are read.

An `INST` copy of a cluster uses the same tiles; a mirrored copy keeps
vertex colours only, since mirroring turns its tiles over.

## LODS

Levels of detail, built by `tools/fkj-import` (`src/lod.rs`): the
clusters of `MESH` hold every level, the model as made (level 0) and each
coarser level, about half the triangles of the one below. A cluster records
its level, its error (how far, in metres, it may stray from the model as
made) with the sphere that error is seen from, and the same two for the
clusters that replace it. The renderer draws a cluster when its error,
seen from the eye, is under the threshold and its replacement's is not:
one level for each stretch of surface, and no cracks, because the edges
where levels meet were never moved (Karis, Stubbs, Wihlidal, "Nanite",
SIGGRAPH 2021).

    u32       cluster count (the MESH section's)
    u32       0
    per cluster, 44 bytes
      u8        level, 0 for the model as made
      u8 x 3    0
      f32       error, metres (0 at level 0)
      f32 x 3   centre of the sphere the error is seen from
      f32       its radius
      f32       error of what replaces it, metres; infinite when nothing does
      f32 x 3   centre of that sphere
      f32       its radius

A cluster's error is never above its replacement's, and its sphere is
inside its replacement's, so what is drawn changes steadily with distance.
Each level has its own texture tiles in `TEXR`. Clusters without `LODS`
(the ground, the mountains, older files) are always drawn.

## OBJS

What the editor selects and moves (`Scene::objects`): runs of clusters
with a pivot, and groups. Optional; a reader without it draws the same
geometry. Section version 2.

    u32       object count
    u32       0
    per object
      u32       first cluster
      u32       cluster count, 0 for a group
      u32       parent + 1, 0 for none (an index into this list)
      u32       name length, bytes
      f32 x 3   pivot, metres: the point it turns and scales about
      f32 x 3   rotation since first placed, radians about X, Y and Z,
                applied in that order
      f32       uniform scale since first placed
      u8 x n    the name, UTF-8, padded with zeros to 4 bytes

Geometry is stored where the objects stand; the rotation and scale are
kept so the editor shows them. Version 1 (23/10/2026) held one f32, the
turn about Y, in place of the rotation; it is still read. A group with nothing under it is its pivot. A
parent chain that loops is cut at the object that closes it.

## SAMP

Surface samples (`src/samples.rs`) and their sky transfer
(`src/transfer.rs`). Optional; a reader without it draws the same
geometry with the sky unoccluded. Section version 1.

    u32       sample count
    u32       0
    per sample, in order of cluster, 60 bytes
      f32 x 3   position, metres
      u16 x 2   unit normal, octahedral (as MESH)
      f32       area it stands for, m²
      u32       cluster it lies on (finest level of detail)
      f32 x 9   sky transfer: what it sees of the sky past the model
                itself, cosine weighted, as order-2 real spherical
                harmonics (Y00, Y1-1, Y10, Y11, Y2-2, Y2-1, Y20, Y21, Y22),
                in world space; NaN when not computed

Instances (`INST`) copy the samples of the clusters they copy, the
transfer turned with them when the transform is a rotation and a uniform
scale, else NaN.

## LITE

Section version 2.

    u32       light count
    u32       0
    per light, 32 bytes
      f32 x 3   position, metres
      f32 x 3   colour, linear RGB, luminance 1
      f32       luminous flux, lumens
      f32       0

Every light is a point light with no falloff radius: intensity is
lumens / 4π candela, and the illuminance at distance d is that over d² lux.
Lumens are kept between 1 and 2,000,000 (about an 18 kW HMI). Version 1
sections, which held a relative intensity, are skipped and reported.

The sky is not stored in the file; it is a viewer setting.

## The built-in scene

The built-in set (21/10/2026) is the procedural ground plane, valley and
mountains of `src/world.rs`, made directly into clusters, with
`assets/models/trex_highres.fkj` (or `trex.fkj`) and `car.fkj` placed on
it. The models are made with `tools/fkj-import` (below).

## The chess set

`assets/chess_set.fkj` was the subject of the built-in set until
21/10/2026; it stays in the repository and opens with
`fkjensen assets/chess_set.fkj`. It was baked on 19/10/2026, and again on 20/10/2026, from the USD Working Group's
Open Chess Set (`full_assets/OpenChessSet/chess_set.usda` at
https://github.com/usd-wg/assets), by a converter that is not part of this
repository:

- every mesh, in world space, with the pawn point instancers expanded;
- winding as USD authors it: 17 of the 21 meshes are `leftHanded`, and
  their triangles are turned counter-clockwise (the first bake missed
  this, so their normals pointed inwards);
- clusters bounded in space as well as in count;
- vertex normals smoothed from the faces where the source has none,
  face-varying ones averaged onto vertices;
- each vertex's albedo is the average of the base colour texture over a
  5 x 5 texel footprint at each of its UV corners, converted to linear;
- pawn tops are glass in the source, with no base colour texture: their
  transmission colour is used as albedo;
- no lights: the file holds geometry only. The lights of the built-in set
  are in `src/world.rs`.

Stored: 799,384 triangles, 815,974 vertices, 14,252 clusters, 14.7 MB.
Loaded, with the pawn instances: 1,430,784 triangles, 1,355,968 vertices,
23,520 clusters.

## Converting models

`tools/fkj-import` converts glTF (.glb, .gltf), binary FBX and OBJ models to
`.fkj`, with their base colour textures baked to tiles. A triangle's finest
level is the coarser of as many points as the source texture has texels
over it, and points no closer than `--texel` metres (2 mm by default); a
cluster stores the finest level any of its triangles needs. Each level is
filtered from the source's mip chain at its own point spacing. A triangle
over two texels or fewer needs no tiles: its vertex colours hold them.
Normal, roughness and other maps are not read. An OBJ's materials are not
read; `--texture` gives its base colour. Every model gets its surface
samples and their sky transfer (`SAMP`); `fkj-import --sky in.fkj -o
out.fkj` adds them to a file made before them.

## Not done yet

- Points, curves and volumes: tags reserved, nothing written or read.
- Instances are expanded at load rather than drawn from one copy.
- Materials: albedo only, per vertex or from texture tiles. No roughness,
  metalness, normal maps, transmission.
- The chess set has no tiles yet: it was baked from USD with colours per
  vertex, so detail finer than the vertex spacing is lost (the board's
  squares have soft edges). `tools/fkj-import` does not read USD.
- Levels of detail are all loaded: a small machine cannot yet leave the
  finest ones in the file.
- Seams (texture coordinates, normals, separate parts) never move when a
  level is simplified, so models made of many parts, like the car,
  simplify little.
- No compression beyond quantisation; tiles are raw sRGB8.
- Tiles are one level per cluster: a cluster's largest triangle sets it for
  all of them.
- The converter that baked the chess set from USD is not in the
  repository, so that file cannot be regenerated from here.
