# The .fkj scene format

FkJensen does not read USD. Scenes are stored in its own format, laid out
for the way this renderer draws: geometry already cut into small clusters,
each with the bounds the renderer culls with, so loading is a decode and the
first frame needs no preparation. `src/fkj.rs` reads and writes it.

The four kinds of primitive the format is meant to hold are triangle meshes,
points (particles), curves (fur, hair) and volumes. Version 1 implements
meshes; the other three have reserved tags and are skipped by this reader.

## Layout

Little-endian throughout. A file is a header and a list of sections.

    header    16 bytes
      [u8; 4]   "FKJ\0"
      u32       version, 1
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
| `LITE` | point lights | yes |
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
corners fit in one byte each. Clusters own contiguous runs of vertices and
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

## LITE

    u32       light count
    u32       0
    per light, 32 bytes
      f32 x 3   position
      f32 x 3   colour, linear RGB
      f32       intensity
      f32       0

Every light is a point light. Irradiance from a light falls off with 1/d²
and is scaled so that intensity 1 at one scene radius gives irradiance 1.

## The built-in scene

`assets/chess_set.fkj` is compiled into the program and opens when no file
is given. It was baked once, on 19/10/2026, from the USD Working Group's
Open Chess Set (`full_assets/OpenChessSet/chess_set.usda` at
https://github.com/usd-wg/assets), by a converter that is not part of this
repository:

- every mesh, in world space, with the pawn point instancers expanded;
- vertex normals smoothed from the faces where the source has none,
  face-varying ones averaged onto vertices;
- each vertex's albedo is the average of the base colour texture over a
  5 x 5 texel footprint at each of its UV corners, converted to linear;
- pawn tops are glass in the source, with no base colour texture: their
  transmission colour is used as albedo;
- two point lights, a warm key and a cool fill, placed for this viewer.

Stored: 799,384 triangles, 821,485 vertices, 12,903 clusters, 14.7 MB.
Loaded, with the pawn instances: 1,430,784 triangles, 1,366,001 vertices,
21,443 clusters.

## Not done yet

- Points, curves and volumes: tags reserved, nothing written or read.
- Instances are expanded at load rather than drawn from one copy.
- Materials: albedo per vertex only. No roughness, metalness, normal maps,
  transmission. Detail finer than the vertex spacing is lost, which shows on
  the board's squares (soft edges).
- Positions, normals and colours are decoded to f32 at load; the renderer
  does not yet read the quantised data directly, which would cut memory
  traffic per frame.
- Level of detail: clusters have no coarser levels yet.
- No compression beyond quantisation.
- The converter that baked the chess set is not in the repository, so the
  file cannot be regenerated from here.
