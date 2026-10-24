//! The scene the rasteriser reads: clustered triangle meshes in flat SoA
//! arrays, and point lights.
//!
//! Geometry is cut into clusters of at most `CLUSTER_VERTS` vertices and
//! `CLUSTER_TRIS` triangles. Each cluster owns a contiguous run of vertices
//! and of triangles, and carries a bounding sphere and a normal cone, so the
//! renderer can drop a whole cluster that is off screen or faces away before
//! touching its vertices. Geometry is kept as the `.fkj` file stores it
//! (`src/fkj.rs`): positions 16 bits per axis inside their cluster's box,
//! normals as 16-bit octahedral pairs, colours as 8-bit sRGB, corners as a
//! byte into their cluster's vertices; 13 bytes a vertex and 3 a triangle
//! where f32 took 36 and 12. Loading copies them; reading a vertex decodes
//! it (`Scene::pos`, `normal`, `color`, `tri`).


// Building clusters (`push_mesh`) is used by the tests and the baking tool;
// the viewer gets its clusters from the file.
#![allow(dead_code)]

use glam::Vec3;

/// Most vertices in one cluster. Local indices fit a byte.
pub const CLUSTER_VERTS: usize = 64;

/// A unit vector as two 16-bit octahedral coordinates (Cigolle et al.,
/// "A Survey of Efficient Representations for Independent Unit Vectors",
/// 2014): within about 0.003 degrees.
pub fn oct_encode(n: Vec3) -> [u16; 2] {
    let n = n.normalize_or(Vec3::Y);
    let l1 = n.x.abs() + n.y.abs() + n.z.abs();
    let (mut x, mut y) = (n.x / l1, n.y / l1);
    if n.z < 0.0 {
        let (ox, oy) = (x, y);
        x = (1.0 - oy.abs()) * ox.signum();
        y = (1.0 - ox.abs()) * oy.signum();
    }
    let q = |v: f32| ((v.clamp(-1.0, 1.0) * 0.5 + 0.5) * 65535.0).round() as u16;
    [q(x), q(y)]
}

#[inline]
pub fn oct_decode(e: [u16; 2]) -> Vec3 {
    let x = e[0] as f32 / 65535.0 * 2.0 - 1.0;
    let y = e[1] as f32 / 65535.0 * 2.0 - 1.0;
    let z = 1.0 - x.abs() - y.abs();
    let (x, y) = if z < 0.0 { ((1.0 - y.abs()) * x.signum(), (1.0 - x.abs()) * y.signum()) } else { (x, y) };
    Vec3::new(x, y, z).normalize_or(Vec3::Y)
}

pub fn linear_to_srgb8(v: f32) -> u8 {
    let v = v.clamp(0.0, 1.0);
    let s = if v <= 0.0031308 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 };
    (s * 255.0).round() as u8
}

pub fn srgb8_to_linear(v: u8) -> f32 {
    let s = v as f32 / 255.0;
    if s <= 0.04045 { s / 12.92 } else { ((s + 0.055) / 1.055).powf(2.4) }
}

/// `srgb8_to_linear` of every byte.
static SRGB: std::sync::LazyLock<[f32; 256]> = std::sync::LazyLock::new(|| std::array::from_fn(|i| srgb8_to_linear(i as u8)));

/// A cluster's quantisation box for `ps`: its corner and the step of one
/// unit of a 16-bit coordinate.
pub fn quant_box(ps: &[Vec3]) -> (Vec3, Vec3) {
    let (mut lo, mut hi) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
    for p in ps {
        lo = lo.min(*p);
        hi = hi.max(*p);
    }
    if ps.is_empty() {
        return (Vec3::ZERO, Vec3::ZERO);
    }
    (lo, (hi - lo) / 65535.0)
}

#[inline]
pub fn quantise(p: Vec3, lo: Vec3, step: Vec3) -> [u16; 3] {
    std::array::from_fn(|k| if step[k] > 0.0 { ((p[k] - lo[k]) / step[k]).round().clamp(0.0, 65535.0) as u16 } else { 0 })
}

#[inline]
pub fn dequantise(q: [u16; 3], lo: Vec3, step: Vec3) -> Vec3 {
    lo + Vec3::new(q[0] as f32, q[1] as f32, q[2] as f32) * step
}
/// Most triangles in one cluster.
pub const CLUSTER_TRIS: usize = 124;

/// The brightest point light allowed, in lumens: about what an 18 kW HMI,
/// the largest common film light, puts out. Lights are kept to what exists
/// in the real world.
pub const MAX_LUMENS: f32 = 2_000_000.0;
/// The dimmest, about a candle's worth (a candle is about 12 lm).
pub const MIN_LUMENS: f32 = 1.0;

/// A point light. Its strength is its luminous flux in lumens; there is no
/// falloff radius. Light leaves it equally in every direction, so its
/// intensity is lumens / 4π candela and the illuminance it puts on a surface
/// facing it at distance d metres is that over d² lux.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointLight {
    pub position: Vec3,
    /// Linear RGB, scaled so its luminance is 1 (`lights::normalise_colour`):
    /// the colour tints the light, the lumens say how strong it is.
    pub color: Vec3,
    pub lumens: f32,
}

impl PointLight {
    /// Luminous intensity in candela.
    pub fn candela(&self) -> f32 {
        self.lumens / (4.0 * std::f32::consts::PI)
    }

    /// Illuminance in lux on a surface facing the light at `d` metres.
    pub fn lux_at(&self, d: f32) -> f32 {
        self.candela() / (d * d).max(1e-12)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cluster {
    pub first_vertex: u32,
    pub vertex_count: u32,
    /// Index of the first triangle (triangle i uses indices 3i..3i+3).
    pub first_tri: u32,
    pub tri_count: u32,
    /// Bounding sphere.
    pub centre: Vec3,
    pub radius: f32,
    /// Normal cone: every face normal n has `n.dot(cone_axis) >= cone_cos`.
    /// `cone_cos <= 0` means the cone is too wide to cull with.
    pub cone_axis: Vec3,
    pub cone_cos: f32,
    /// Level of detail (`lod.rs`): 0 for the model as made, each level up
    /// about half the triangles of the one below.
    pub lod_level: u8,
    /// How far, in metres, this cluster may stray from the model as made,
    /// and the sphere that error is seen from (its group's).
    pub lod_error: f32,
    pub lod_centre: Vec3,
    pub lod_radius: f32,
    /// The same for the coarser clusters that replace it and its group;
    /// `f32::INFINITY` when nothing does.
    pub parent_error: f32,
    pub parent_centre: Vec3,
    pub parent_radius: f32,
    /// Its vertices' quantisation box, as stored: a stored position is
    /// `qlo + q * qstep`.
    pub qlo: Vec3,
    pub qstep: Vec3,
    /// From a quantised position to the world: the box, then its object's
    /// transform (`Object::xf`). Moving an object changes this, not its
    /// vertices.
    pub xf: glam::Affine3A,
    /// Its object's rotation, for normals, and whether it has one.
    pub turn: glam::Quat,
    pub turned: bool,
}

impl Default for Cluster {
    fn default() -> Self {
        Self {
            first_vertex: 0,
            vertex_count: 0,
            first_tri: 0,
            tri_count: 0,
            centre: Vec3::ZERO,
            radius: 0.0,
            cone_axis: Vec3::Y,
            cone_cos: -1.0,
            lod_level: 0,
            lod_error: 0.0,
            lod_centre: Vec3::ZERO,
            lod_radius: 0.0,
            parent_error: f32::INFINITY,
            parent_centre: Vec3::ZERO,
            parent_radius: 0.0,
            qlo: Vec3::ZERO,
            qstep: Vec3::ZERO,
            xf: glam::Affine3A::IDENTITY,
            turn: glam::Quat::IDENTITY,
            turned: false,
        }
    }
}

impl Cluster {
    /// Set its quantisation box, with no object transform.
    pub fn set_quant(&mut self, lo: Vec3, step: Vec3) {
        self.qlo = lo;
        self.qstep = step;
        self.xf = glam::Affine3A::from_scale_rotation_translation(step, glam::Quat::IDENTITY, lo);
        self.turn = glam::Quat::IDENTITY;
        self.turned = false;
    }

    /// Whether its vertices are where they are stored (no object transform).
    pub fn in_place(&self) -> bool {
        !self.turned && self.xf == glam::Affine3A::from_scale_rotation_translation(self.qstep, glam::Quat::IDENTITY, self.qlo)
    }

    /// Whether this cluster is the one to draw for a view in which an error
    /// of one metre at a sphere looks `scale(centre, radius)` units big
    /// (pixels, texels), when errors up to `threshold` units may show: its
    /// own error is small enough and its replacement's is not (Karis,
    /// Stubbs, Wihlidal, "Nanite", 2021). Exactly one cluster of each
    /// stretch of surface passes. A cluster with no levels always does.
    #[inline]
    pub fn lod_drawn(&self, threshold: f32, scale: impl Fn(Vec3, f32) -> f32) -> bool {
        (self.lod_error == 0.0 || self.lod_error * scale(self.lod_centre, self.lod_radius) <= threshold)
            && (self.parent_error.is_infinite() || self.parent_error * scale(self.parent_centre, self.parent_radius) > threshold)
    }

    /// Whether it belongs to a level-of-detail hierarchy at all.
    pub fn has_lod(&self) -> bool {
        self.lod_error > 0.0 || self.parent_error.is_finite()
    }

    /// Move its spheres as `f` moves points, with lengths times `scale`.
    pub fn move_lod(&mut self, f: impl Fn(Vec3) -> Vec3, scale: f32) {
        self.lod_centre = f(self.lod_centre);
        self.lod_radius *= scale;
        self.lod_error *= scale;
        self.parent_centre = f(self.parent_centre);
        self.parent_radius *= scale;
        self.parent_error *= scale;
    }
}

/// For a perspective view from `eye`: pixels per metre at the nearest point
/// of a sphere, for `lod_drawn`. `px_per_unit` is pixels per unit of
/// tangent (half the frame height over tan(fov / 2)).
#[inline]
pub fn perspective_scale(eye: Vec3, px_per_unit: f32, near: f32) -> impl Fn(Vec3, f32) -> f32 {
    move |c, r| px_per_unit / (c.distance(eye) - r).max(near)
}

#[derive(Default)]
pub struct Scene {
    /// World-space positions, one per vertex, 16 bits per axis inside its
    /// cluster's box (`Cluster::qlo`, `qstep`); read with `pos`.
    pub qpos: Vec<[u16; 3]>,
    /// World-space unit normals, octahedral; read with `normal`.
    pub qnrm: Vec<[u16; 2]>,
    /// Albedo, 8-bit sRGB; read (linear) with `color`.
    pub qcol: Vec<[u8; 3]>,
    /// Each triangle's corners, as indices into its cluster's vertices.
    /// A triangle only references vertices of its own cluster.
    pub tris: Vec<[u8; 3]>,
    pub clusters: Vec<Cluster>,
    /// Bounds of all positions.
    pub bounds_min: Vec3,
    pub bounds_max: Vec3,
    /// The subject the camera frames with `F` and opens on (the chess table
    /// in the built-in scene). The whole scene when not set.
    pub focus: Option<(Vec3, Vec3)>,
    /// The lights stored with the scene; `R` in the viewer resets to these.
    pub lights: Vec<PointLight>,
    /// Per-triangle texture tiles, streamed (`texture.rs`); empty when the
    /// scene has only vertex colours.
    pub textures: crate::texture::Textures,
    /// What can be selected and moved: runs of clusters with a pivot.
    /// Geometry outside every object (the ground, the mountains) stays put.
    pub objects: Vec<Object>,
    /// Points spread evenly over each object's surface (`samples.rs`), in
    /// order of their cluster, moved with it.
    pub samples: Vec<crate::samples::Sample>,
    /// Where the geometry outside every object came from, for saving.
    pub world: World,
}

/// Where a scene's geometry outside every object (the ground, the
/// mountains) comes from.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum World {
    /// None, or read from a file it is not kept in (an old .fkj scene).
    #[default]
    Unsaved,
    /// The program's own ground and mountains (`world.rs`).
    BuiltIn,
    /// An .fja asset.
    Asset(std::path::PathBuf),
}

/// A model in the scene: its clusters (and so its vertices, which they
/// hold in one run), and how it has been placed. Geometry is kept in world
/// space; moving an object rewrites its vertices. A group is an object with
/// no clusters: moving it moves everything under it (`parent`).
#[derive(Clone, Debug, PartialEq)]
pub struct Object {
    pub name: String,
    pub clusters: std::ops::Range<usize>,
    /// The point it turns and scales about: the middle of its base when
    /// placed.
    pub pivot: Vec3,
    /// Rotation since placed, radians about X, Y and Z, applied in that
    /// order (Maya's xyz), and uniform scale since placed.
    pub rotation: Vec3,
    pub scale: f32,
    /// Bounds of its finest level's vertices, in the world (round its
    /// oriented box).
    pub lo: Vec3,
    pub hi: Vec3,
    /// From its geometry as stored to the world. Its vertices are never
    /// rewritten when it moves: this changes.
    pub xf: glam::Affine3A,
    /// Bounds of its finest level as stored: with `xf`, its oriented box.
    pub box_lo: Vec3,
    pub box_hi: Vec3,
    /// The group it is in, if any (an index into `Scene::objects`).
    pub parent: Option<usize>,
    /// The .fja asset file it was read from, if any (`fjs.rs`).
    pub asset: Option<std::path::PathBuf>,
}

impl Object {
    /// A new object of clusters `clusters` whose geometry is stored where it
    /// stands.
    pub fn new(name: String, clusters: std::ops::Range<usize>, pivot: Vec3, rotation: Vec3, scale: f32, parent: Option<usize>) -> Object {
        Object { name, clusters, pivot, rotation, scale, lo: pivot, hi: pivot, xf: glam::Affine3A::IDENTITY, box_lo: pivot, box_hi: pivot, parent, asset: None }
    }

    /// The eight corners of its oriented box, in the world.
    pub fn corners(&self) -> [Vec3; 8] {
        let (a, b) = (self.box_lo, self.box_hi);
        std::array::from_fn(|i| self.xf.transform_point3(Vec3::new(if i & 1 == 0 { a.x } else { b.x }, if i & 2 == 0 { a.y } else { b.y }, if i & 4 == 0 { a.z } else { b.z })))
    }
}

/// The rotation of Euler angles `r` (radians), X first, then Y, then Z.
pub fn euler_quat(r: Vec3) -> glam::Quat {
    glam::Quat::from_euler(glam::EulerRot::ZYX, r.z, r.y, r.x)
}

/// A sample's sky transfer turned by `q`. A uniform scale changes nothing:
/// a model's own occlusion does not depend on its size.
fn turned_sky(t: &[f32; 9], q: glam::Quat, _scale: f32) -> [f32; 9] {
    if t[0].is_nan() { *t } else { crate::transfer::rotate(t, q) }
}

/// Euler angles (radians, X then Y then Z) of a rotation.
pub fn quat_euler(q: glam::Quat) -> Vec3 {
    let (z, y, x) = q.to_euler(glam::EulerRot::ZYX);
    Vec3::new(x, y, z)
}

impl Scene {
    pub fn new() -> Self {
        Self {
            bounds_min: Vec3::splat(f32::INFINITY),
            bounds_max: Vec3::splat(f32::NEG_INFINITY),
            ..Default::default()
        }
    }

    pub fn triangle_count(&self) -> usize {
        self.tris.len()
    }

    pub fn vertex_count(&self) -> usize {
        self.qpos.len()
    }

    /// Bytes the geometry takes.
    pub fn geometry_bytes(&self) -> usize {
        self.qpos.len() * 6 + self.qnrm.len() * 4 + self.qcol.len() * 3 + self.tris.len() * 3 + self.clusters.len() * std::mem::size_of::<Cluster>()
    }

    /// Position of vertex `v` of cluster `c`.
    #[inline]
    pub fn pos(&self, c: &Cluster, v: usize) -> Vec3 {
        let q = self.qpos[v];
        c.xf.transform_point3(Vec3::new(q[0] as f32, q[1] as f32, q[2] as f32))
    }

    /// Normal of vertex `v` of cluster `c`.
    #[inline]
    pub fn normal(&self, c: &Cluster, v: usize) -> Vec3 {
        let n = oct_decode(self.qnrm[v]);
        if c.turned { c.turn * n } else { n }
    }

    /// Linear RGB albedo.
    #[inline]
    pub fn color(&self, v: usize) -> Vec3 {
        let c = self.qcol[v];
        let l = &*SRGB;
        Vec3::new(l[c[0] as usize], l[c[1] as usize], l[c[2] as usize])
    }

    /// The scene vertices of triangle `t` of cluster `c`.
    #[inline]
    pub fn tri(&self, c: &Cluster, t: u32) -> [usize; 3] {
        let f = c.first_vertex as usize;
        self.tris[t as usize].map(|l| f + l as usize)
    }

    /// The corner positions of triangle `t` of cluster `c`.
    #[inline]
    pub fn tri_pos(&self, c: &Cluster, t: u32) -> [Vec3; 3] {
        self.tri(c, t).map(|v| self.pos(c, v))
    }

    /// Cluster `c`'s vertex positions, in order.
    #[inline]
    pub fn cluster_pos<'a>(&'a self, c: &'a Cluster) -> impl Iterator<Item = Vec3> + 'a {
        self.qpos[c.first_vertex as usize..(c.first_vertex + c.vertex_count) as usize].iter().map(|q| c.xf.transform_point3(Vec3::new(q[0] as f32, q[1] as f32, q[2] as f32)))
    }

    /// Every vertex position, decoded, in order (for tests and tools).
    pub fn positions(&self) -> Vec<Vec3> {
        let mut out = vec![Vec3::ZERO; self.qpos.len()];
        for c in &self.clusters {
            for (k, p) in self.cluster_pos(c).enumerate() {
                out[c.first_vertex as usize + k] = p;
            }
        }
        out
    }

    /// Every normal, decoded, in order.
    pub fn normals(&self) -> Vec<Vec3> {
        let mut out = vec![Vec3::Y; self.qnrm.len()];
        for c in &self.clusters {
            for v in c.first_vertex as usize..(c.first_vertex + c.vertex_count) as usize {
                out[v] = self.normal(c, v);
            }
        }
        out
    }

    /// Every colour, decoded, in order.
    pub fn colors(&self) -> Vec<Vec3> {
        (0..self.qcol.len()).map(|v| self.color(v)).collect()
    }

    /// Three scene vertex indices per triangle (for tests and tools).
    pub fn indices(&self) -> Vec<u32> {
        let mut out = vec![0; self.tris.len() * 3];
        for c in &self.clusters {
            for t in c.first_tri..c.first_tri + c.tri_count {
                for (k, v) in self.tri(c, t).into_iter().enumerate() {
                    out[t as usize * 3 + k] = v as u32;
                }
            }
        }
        out
    }

    /// Grow each sphere of clusters `range` to hold its decoded vertices.
    fn fit_spheres(&mut self, range: std::ops::Range<usize>) {
        use rayon::prelude::*;
        let qpos = &self.qpos;
        self.clusters[range].par_iter_mut().for_each(|c| {
            let fv = c.first_vertex as usize;
            let r = qpos[fv..fv + c.vertex_count as usize].iter().map(|q| c.xf.transform_point3(Vec3::new(q[0] as f32, q[1] as f32, q[2] as f32)).distance(c.centre)).fold(0.0, f32::max);
            c.radius = c.radius.max(r);
        });
    }

    /// Append one cluster's vertices, quantised; returns its box.
    fn push_vertices(&mut self, ps: &[Vec3], ns: &[Vec3], cs: &[Vec3]) -> (Vec3, Vec3) {
        let (lo, step) = quant_box(ps);
        self.qpos.extend(ps.iter().map(|p| quantise(*p, lo, step)));
        self.qnrm.extend(ns.iter().map(|n| oct_encode(*n)));
        self.qcol.extend(cs.iter().map(|c| [c.x, c.y, c.z].map(linear_to_srgb8)));
        (lo, step)
    }

    /// Move every vertex of clusters `range` by `f` and turn every normal
    /// by `q`, quantising each cluster again in its new box: the result is
    /// stored where it stands (any object transform baked in). For whole
    /// scenes at load; objects move by their transform instead.
    pub fn map_vertices(&mut self, range: std::ops::Range<usize>, f: impl Fn(Vec3) -> Vec3 + Sync, q: Option<glam::Quat>) {
        use rayon::prelude::*;
        // Normals of turned clusters take their turn too.
        let turns: Vec<(std::ops::Range<usize>, glam::Quat)> = self.clusters[range.clone()]
            .iter()
            .filter(|c| c.turned || q.is_some())
            .map(|c| (c.first_vertex as usize..(c.first_vertex + c.vertex_count) as usize, q.unwrap_or(glam::Quat::IDENTITY) * c.turn))
            .collect();
        let (Some(first), Some(last)) = (self.clusters.get(range.start).copied(), range.end.checked_sub(1).and_then(|e| self.clusters.get(e)).copied()) else { return };
        let fv = first.first_vertex as usize;
        let end = (last.first_vertex + last.vertex_count) as usize;
        // Split the vertex arrays into each cluster's run, in parallel.
        let mut pos_runs: Vec<&mut [[u16; 3]]> = Vec::with_capacity(range.len());
        let mut rest = &mut self.qpos[fv..end];
        let mut at = fv;
        for c in &self.clusters[range.clone()] {
            let skip = c.first_vertex as usize - at;
            let (_, r) = rest.split_at_mut(skip);
            let (run, r) = r.split_at_mut(c.vertex_count as usize);
            pos_runs.push(run);
            rest = r;
            at = (c.first_vertex + c.vertex_count) as usize;
        }
        self.clusters[range].par_iter_mut().zip(pos_runs).for_each(|(c, run)| {
            let ps: Vec<Vec3> = run.iter().map(|q| f(c.xf.transform_point3(Vec3::new(q[0] as f32, q[1] as f32, q[2] as f32)))).collect();
            let (lo, step) = quant_box(&ps);
            for (q, p) in run.iter_mut().zip(&ps) {
                *q = quantise(*p, lo, step);
            }
            c.set_quant(lo, step);
        });
        let _ = (fv, end);
        for (r, t) in turns {
            for n in &mut self.qnrm[r] {
                *n = oct_encode(t * oct_decode(*n));
            }
        }
    }

    pub fn centre(&self) -> Vec3 {
        (self.bounds_min + self.bounds_max) * 0.5
    }

    pub fn radius(&self) -> f32 {
        ((self.bounds_max - self.bounds_min).length() * 0.5).max(1e-3)
    }

    pub fn focus_bounds(&self) -> (Vec3, Vec3) {
        self.focus.unwrap_or((self.bounds_min, self.bounds_max))
    }

    pub fn focus_centre(&self) -> Vec3 {
        let (a, b) = self.focus_bounds();
        (a + b) * 0.5
    }

    pub fn focus_radius(&self) -> f32 {
        let (a, b) = self.focus_bounds();
        ((b - a).length() * 0.5).max(1e-3)
    }

    /// Move every vertex, cluster and light by `d`.
    pub fn translate(&mut self, d: Vec3) {
        self.map_vertices(0..self.clusters.len(), |p| p + d, None);
        for c in &mut self.clusters {
            c.centre += d;
            c.move_lod(|p| p + d, 1.0);
        }
        self.fit_spheres(0..self.clusters.len());
        for s in &mut self.samples {
            s.position += d;
        }
        for l in &mut self.lights {
            l.position += d;
        }
        self.bounds_min += d;
        self.bounds_max += d;
        if let Some((a, b)) = &mut self.focus {
            *a += d;
            *b += d;
        }
    }

    /// Rotate about the vertical axis by `yaw` radians, scale by `scale`,
    /// then move by `d`: how a model is placed in a set. Rotation and
    /// uniform scale keep each triangle's corner order, so its texture tile
    /// stays valid.
    pub fn place(&mut self, yaw: f32, scale: f32, d: Vec3) {
        let q = glam::Quat::from_rotation_y(yaw);
        self.map_vertices(0..self.clusters.len(), |p| q * (p * scale) + d, Some(q));
        for c in &mut self.clusters {
            c.centre = q * (c.centre * scale) + d;
            c.radius *= scale;
            c.cone_axis = q * c.cone_axis;
            c.move_lod(|p| q * (p * scale) + d, scale);
        }
        self.fit_spheres(0..self.clusters.len());
        for s in &mut self.samples {
            s.position = q * (s.position * scale) + d;
            s.normal = q * s.normal;
            s.area *= scale * scale;
            s.sky = turned_sky(&s.sky, q, scale);
        }
        for l in &mut self.lights {
            l.position = q * (l.position * scale) + d;
        }
        let (mut lo, mut hi) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
        for c in &self.clusters {
            for p in self.cluster_pos(c) {
                lo = lo.min(p);
                hi = hi.max(p);
            }
        }
        self.bounds_min = lo;
        self.bounds_max = hi;
    }

    /// Append another scene's geometry, lights, texture tiles and objects.
    pub fn append(&mut self, other: Scene) {
        let cbase = self.clusters.len();
        let obase = self.objects.len();
        self.objects.extend(other.objects.iter().map(|o| Object { clusters: o.clusters.start + cbase..o.clusters.end + cbase, parent: o.parent.map(|p| p + obase), ..o.clone() }));
        let other_clusters = other.clusters.len();
        if !other.textures.is_empty() || !self.textures.is_empty() {
            self.textures.append(&other.textures, cbase, other_clusters);
        }
        let vbase = self.qpos.len() as u32;
        let tbase = self.triangle_count() as u32;
        self.qpos.extend(other.qpos);
        self.qnrm.extend(other.qnrm);
        self.qcol.extend(other.qcol);
        self.tris.extend(other.tris);
        self.clusters.extend(other.clusters.iter().map(|c| Cluster { first_vertex: c.first_vertex + vbase, first_tri: c.first_tri + tbase, ..*c }));
        self.bounds_min = self.bounds_min.min(other.bounds_min);
        self.bounds_max = self.bounds_max.max(other.bounds_max);
        self.lights.extend(other.lights);
        self.samples.extend(other.samples.iter().map(|s| crate::samples::Sample { cluster: s.cluster + cbase as u32, ..*s }));
    }

    /// Clusters `ranges` (in order) as a scene of their own, moved by
    /// `back`: their vertices, triangles, levels of detail, texture tiles
    /// (still read from where they are) and samples. No objects or lights.
    pub fn extract(&self, ranges: &[std::ops::Range<usize>], back: glam::Affine3A) -> Scene {
        let (k3, turn, _) = back.to_scale_rotation_translation();
        let k = k3.x;
        let turn = turn.normalize();
        let still = turn.abs_diff_eq(glam::Quat::IDENTITY, 1e-7);
        let mut out = Scene::default();
        out.textures.sources = self.textures.sources.clone();
        let mut index = std::collections::HashMap::new();
        for r in ranges {
            for ci in r.clone() {
                let c = &self.clusters[ci];
                let (fv, nv) = (c.first_vertex as usize, c.vertex_count as usize);
                let (ft, nt) = (c.first_tri as usize, c.tri_count as usize);
                let mut h = *c;
                h.first_vertex = out.qpos.len() as u32;
                h.first_tri = out.tris.len() as u32;
                out.qpos.extend_from_slice(&self.qpos[fv..fv + nv]);
                out.qnrm.extend_from_slice(&self.qnrm[fv..fv + nv]);
                out.qcol.extend_from_slice(&self.qcol[fv..fv + nv]);
                out.tris.extend_from_slice(&self.tris[ft..ft + nt]);
                h.xf = back * c.xf;
                if !still || c.turned {
                    h.turn = (turn * c.turn).normalize();
                    h.turned = true;
                }
                h.centre = back.transform_point3(c.centre);
                h.radius = c.radius * k * (1.0 + 1e-5) + 1e-6;
                h.cone_axis = turn * c.cone_axis;
                h.move_lod(|p| back.transform_point3(p), k);
                index.insert(ci as u32, out.clusters.len() as u32);
                out.clusters.push(h);
                out.textures.entries.push(self.textures.entry(ci));
            }
            for s in &self.samples[self.samples_in(r.clone())] {
                out.samples.push(crate::samples::Sample { position: back.transform_point3(s.position), normal: turn * s.normal, area: s.area * k * k, cluster: index[&s.cluster], sky: turned_sky(&s.sky, turn, k) });
            }
        }
        if out.textures.entries.iter().all(|e| e.level == 0) {
            out.textures = Default::default();
        }
        out.samples.sort_by_key(|s| s.cluster);
        let (mut lo, mut hi) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
        for c in &out.clusters {
            for p in out.cluster_pos(c) {
                lo = lo.min(p);
                hi = hi.max(p);
            }
        }
        if out.clusters.is_empty() {
            (lo, hi) = (Vec3::ZERO, Vec3::ZERO);
        }
        out.bounds_min = lo;
        out.bounds_max = hi;
        out
    }

    /// Object `i` alone, where it stood before it was first moved (its
    /// asset's own space), as one object at the pivot, rotation and scale
    /// it had there: what an .fja holds.
    pub fn extract_object(&self, i: usize) -> Scene {
        let o = &self.objects[i];
        let back = o.xf.inverse();
        let (k3, turn, _) = o.xf.to_scale_rotation_translation();
        let mut out = self.extract(&[o.clusters.clone()], back);
        let n = out.clusters.len();
        let rest_rotation = quat_euler(turn.normalize().inverse() * euler_quat(o.rotation));
        let mut ob = Object::new(o.name.clone(), 0..n, back.transform_point3(o.pivot), rest_rotation, o.scale / k3.x.max(1e-12), None);
        ob.asset = None;
        out.objects.push(ob);
        out.refresh_bounds(0);
        out
    }

    /// The runs of clusters outside every object, in order.
    pub fn loose_clusters(&self) -> Vec<std::ops::Range<usize>> {
        let mut owned: Vec<std::ops::Range<usize>> = self.objects.iter().map(|o| o.clusters.clone()).filter(|r| !r.is_empty()).collect();
        owned.sort_by_key(|r| r.start);
        let mut out = Vec::new();
        let mut at = 0;
        for r in owned {
            if r.start > at {
                out.push(at..r.start);
            }
            at = at.max(r.end);
        }
        if at < self.clusters.len() {
            out.push(at..self.clusters.len());
        }
        out
    }

    /// The samples lying on clusters `clusters`.
    pub fn samples_in(&self, clusters: std::ops::Range<usize>) -> std::ops::Range<usize> {
        let a = self.samples.partition_point(|s| (s.cluster as usize) < clusters.start);
        let b = self.samples.partition_point(|s| (s.cluster as usize) < clusters.end);
        a..b
    }

    /// Spread samples over object `i`'s finest level (`samples.rs`),
    /// replacing any it had. Returns how many and their spacing.
    pub fn scatter_samples(&mut self, i: usize) -> (usize, f32) {
        self.scatter_over(self.objects[i].clusters.clone())
    }

    /// Spread samples over the finest level of clusters `range`, replacing
    /// any they had.
    pub fn scatter_over(&mut self, range: std::ops::Range<usize>) -> (usize, f32) {
        let old = self.samples_in(range.clone());
        self.samples.drain(old.clone());
        if range.is_empty() {
            return (0, 0.0);
        }
        let finest = self.clusters[range.clone()].iter().map(|c| c.lod_level).min().unwrap_or(0);
        // Every finest-level triangle, as (cluster, triangle).
        let tris: Vec<(u32, u32)> = range.clone().filter(|&c| self.clusters[c].lod_level == finest).flat_map(|c| (self.clusters[c].first_tri..self.clusters[c].first_tri + self.clusters[c].tri_count).map(move |t| (c as u32, t))).collect();
        let get = |k: usize| {
            let (ci, t) = tris[k];
            let c = &self.clusters[ci as usize];
            let v = self.tri(c, t);
            (ci, v.map(|v| self.pos(c, v)), v.map(|v| self.normal(c, v)))
        };
        let area: f32 = {
            use rayon::prelude::*;
            (0..tris.len()).into_par_iter().map(|k| crate::samples::tri_area(get(k).1) as f64).sum::<f64>() as f32
        };
        let h = crate::samples::spacing_for(area);
        let new = crate::samples::scatter(tris.len(), get, area, h);
        let n = new.len();
        self.samples.splice(old.start..old.start, new);
        (n, h)
    }

    /// Samples for every object that has none.
    pub fn scatter_missing_samples(&mut self) {
        for i in 0..self.objects.len() {
            let r = self.objects[i].clusters.clone();
            if !r.is_empty() && self.samples_in(r).is_empty() {
                let t = std::time::Instant::now();
                let (n, h) = self.scatter_samples(i);
                eprintln!("{}: {n} surface samples {:.1} cm apart in {:.2} s", self.objects[i].name, h * 100.0, t.elapsed().as_secs_f64());
            }
        }
    }

    /// Make clusters `clusters` an object turning about `pivot`, already
    /// turned by `rotation` (Euler angles, radians).
    pub fn add_object(&mut self, name: String, clusters: std::ops::Range<usize>, pivot: Vec3, rotation: Vec3) -> usize {
        self.objects.push(Object::new(name, clusters, pivot, rotation, 1.0, None));
        let i = self.objects.len() - 1;
        self.refresh_bounds(i);
        i
    }

    /// Make an empty group at `pivot`.
    pub fn add_group(&mut self, name: String, pivot: Vec3) -> usize {
        let n = self.clusters.len();
        self.objects.push(Object::new(name, n..n, pivot, Vec3::ZERO, 1.0, None));
        self.objects.len() - 1
    }

    pub fn is_group(&self, i: usize) -> bool {
        self.objects[i].clusters.is_empty()
    }

    /// The objects directly under `parent` (None: at the top), in order.
    pub fn children(&self, parent: Option<usize>) -> Vec<usize> {
        (0..self.objects.len()).filter(|&i| self.objects[i].parent == parent).collect()
    }

    /// Everything under `i`, at any depth.
    pub fn descendants(&self, i: usize) -> Vec<usize> {
        let mut out = Vec::new();
        let mut todo = vec![i];
        while let Some(j) = todo.pop() {
            for k in self.children(Some(j)) {
                out.push(k);
                todo.push(k);
            }
        }
        out
    }

    /// Put `child` under `parent` (None: at the top). Refused, returning
    /// false, when that would put an object under itself.
    pub fn set_parent(&mut self, child: usize, parent: Option<usize>) -> bool {
        if let Some(p) = parent {
            if p == child || self.descendants(child).contains(&p) || p >= self.objects.len() {
                return false;
            }
        }
        self.objects[child].parent = parent;
        true
    }

    /// The runs of clusters that move with object `i`: its own and those
    /// of everything under it.
    pub fn moving_clusters(&self, i: usize) -> Vec<std::ops::Range<usize>> {
        std::iter::once(i).chain(self.descendants(i)).map(|j| self.objects[j].clusters.clone()).filter(|r| !r.is_empty()).collect()
    }

    /// Measure an object's box again from its finest level's vertices,
    /// as stored, and its bounds in the world from it.
    pub fn refresh_bounds(&mut self, i: usize) {
        use rayon::prelude::*;
        let range = self.objects[i].clusters.clone();
        if range.is_empty() {
            let p = self.objects[i].pivot;
            self.objects[i].lo = p;
            self.objects[i].hi = p;
            self.objects[i].box_lo = p;
            self.objects[i].box_hi = p;
            return;
        }
        let back = self.objects[i].xf.inverse();
        let finest = self.clusters[range.clone()].iter().map(|c| c.lod_level).min().unwrap_or(0);
        let (lo, hi) = self.clusters[range]
            .par_iter()
            .filter(|c| c.lod_level == finest)
            .map(|c| {
                let (mut lo, mut hi) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
                for p in self.cluster_pos(c) {
                    let p = back.transform_point3(p);
                    lo = lo.min(p);
                    hi = hi.max(p);
                }
                (lo, hi)
            })
            .reduce(|| (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)), |a, b| (a.0.min(b.0), a.1.max(b.1)));
        self.objects[i].box_lo = lo;
        self.objects[i].box_hi = hi;
        self.world_bounds(i);
    }

    /// Bounds in the world of object `i`'s finest level's vertices: tighter
    /// than `object_bounds` when it is turned, for framing it.
    pub fn vertex_bounds(&self, i: usize) -> (Vec3, Vec3) {
        use rayon::prelude::*;
        let range = self.objects[i].clusters.clone();
        let finest = self.clusters[range.clone()].iter().map(|c| c.lod_level).min().unwrap_or(0);
        self.clusters[range]
            .par_iter()
            .filter(|c| c.lod_level == finest)
            .map(|c| self.cluster_pos(c).fold((Vec3::INFINITY, Vec3::NEG_INFINITY), |(lo, hi), p| (lo.min(p), hi.max(p))))
            .reduce(|| (Vec3::INFINITY, Vec3::NEG_INFINITY), |a, b| (a.0.min(b.0), a.1.max(b.1)))
    }

    /// An object's bounds in the world, round its oriented box.
    fn world_bounds(&mut self, i: usize) {
        let o = &self.objects[i];
        if o.clusters.is_empty() {
            let p = o.pivot;
            self.objects[i].lo = p;
            self.objects[i].hi = p;
            return;
        }
        let cs = o.corners();
        let lo = cs.iter().fold(Vec3::INFINITY, |a, p| a.min(*p));
        let hi = cs.iter().fold(Vec3::NEG_INFINITY, |a, p| a.max(*p));
        self.objects[i].lo = lo;
        self.objects[i].hi = hi;
    }

    /// The object holding cluster `c`, if any.
    pub fn object_of(&self, c: usize) -> Option<usize> {
        self.objects.iter().position(|o| o.clusters.contains(&c))
    }

    /// Bounds of an object and everything under it. A group with nothing
    /// under it is its pivot.
    pub fn object_bounds(&self, i: usize) -> (Vec3, Vec3) {
        std::iter::once(i)
            .chain(self.descendants(i))
            .filter(|&j| !self.objects[j].clusters.is_empty())
            .map(|j| (self.objects[j].lo, self.objects[j].hi))
            .reduce(|(a, b), (c, d)| (a.min(c), b.max(d)))
            .unwrap_or((self.objects[i].lo, self.objects[i].hi))
    }

    /// Place object `i` at `pivot`, turned to `rotation` (Euler angles,
    /// radians) and scaled to `scale`, all absolute, from how it was first
    /// placed: its vertices, normals, clusters and levels of detail move
    /// with it. A rotation and a uniform scale keep each triangle's corners
    /// in order (no mirroring), so its texture tile stays valid. Everything
    /// under it moves with it, about its pivot.
    pub fn set_object(&mut self, i: usize, pivot: Vec3, rotation: Vec3, scale: f32) {
        let o = self.objects[i].clone();
        let scale = scale.max(1e-4);
        let k = scale / o.scale;
        if pivot == o.pivot && rotation == o.rotation && k == 1.0 {
            return;
        }
        let q = euler_quat(rotation) * euler_quat(o.rotation).inverse();
        let map = |p: Vec3| q * ((p - o.pivot) * k) + pivot;
        for j in self.descendants(i) {
            let d = self.objects[j].clone();
            self.place_one(j, map(d.pivot), quat_euler(q * euler_quat(d.rotation)), d.scale * k);
        }
        self.place_one(i, pivot, rotation, scale);
    }

    /// `set_object` for object `i` alone.
    fn place_one(&mut self, i: usize, pivot: Vec3, rotation: Vec3, scale: f32) {
        let o = self.objects[i].clone();
        let k = scale / o.scale;
        let q = euler_quat(rotation) * euler_quat(o.rotation).inverse();
        let map = |p: Vec3| q * ((p - o.pivot) * k) + pivot;
        if o.clusters.is_empty() {
            let ob = &mut self.objects[i];
            ob.pivot = pivot;
            ob.rotation = rotation;
            ob.scale = scale;
            self.world_bounds(i);
            return;
        }
        // The vertices stay as stored; the object's transform and its
        // clusters' bounds move.
        let d = glam::Affine3A::from_translation(pivot) * glam::Affine3A::from_scale_rotation_translation(Vec3::splat(k), q, Vec3::ZERO) * glam::Affine3A::from_translation(-o.pivot);
        {
            use rayon::prelude::*;
            self.clusters[o.clusters.clone()].par_iter_mut().for_each(|c| {
                c.xf = d * c.xf;
                c.turn = (q * c.turn).normalize();
                c.turned = true;
                c.centre = map(c.centre);
                // A hair wider: the bounds move by arithmetic, not by
                // measuring the vertices again.
                c.radius = c.radius * k * (1.0 + 1e-5) + 1e-6;
                c.cone_axis = q * c.cone_axis;
                c.move_lod(map, k);
            });
        }
        self.objects[i].xf = d * o.xf;
        let r = self.samples_in(o.clusters.clone());
        for s in &mut self.samples[r] {
            s.position = map(s.position);
            s.normal = q * s.normal;
            s.area *= k * k;
            s.sky = turned_sky(&s.sky, q, k);
        }
        let ob = &mut self.objects[i];
        ob.pivot = pivot;
        ob.rotation = rotation;
        ob.scale = scale;
        self.world_bounds(i);
        let (lo, hi) = (self.objects[i].lo, self.objects[i].hi);
        self.bounds_min = self.bounds_min.min(lo);
        self.bounds_max = self.bounds_max.max(hi);
    }

    /// Append a mesh already in world space, cutting it into clusters.
    /// `normals` may be empty (smooth normals are computed from the faces);
    /// `colors` may hold one value for the whole mesh or be empty (0.7 grey).
    pub fn push_mesh(&mut self, positions: &[Vec3], normals: &[Vec3], colors: &[Vec3], tri_indices: &[u32]) {
        self.push_mesh_mapped(positions, normals, colors, tri_indices);
    }

    /// `push_mesh`, returning for each triangle added (in the scene's
    /// order) the index of the source triangle it came from. Corner order is
    /// kept, so per-triangle data (texture tiles) can follow.
    pub fn push_mesh_mapped(&mut self, positions: &[Vec3], normals: &[Vec3], colors: &[Vec3], tri_indices: &[u32]) -> Vec<u32> {
        let mut source = Vec::with_capacity(tri_indices.len() / 3);
        let n = positions.len();
        let normals: Vec<Vec3> = if normals.len() == n {
            normals.to_vec()
        } else {
            let mut acc = vec![Vec3::ZERO; n];
            for t in tri_indices.chunks_exact(3) {
                let (a, b, c) = (t[0] as usize, t[1] as usize, t[2] as usize);
                let f = (positions[b] - positions[a]).cross(positions[c] - positions[a]);
                acc[a] += f;
                acc[b] += f;
                acc[c] += f;
            }
            acc.into_iter().map(|v| v.normalize_or(Vec3::Y)).collect()
        };
        let colors: Vec<Vec3> = if colors.len() == n {
            colors.to_vec()
        } else {
            vec![colors.first().copied().unwrap_or(Vec3::splat(0.7)); n]
        };

        for c in build_clusters(positions, tri_indices) {
            let first_vertex = self.qpos.len() as u32;
            let first_tri = self.tris.len() as u32;
            let pick = |a: &[Vec3]| c.vertices.iter().map(|&v| a[v as usize]).collect::<Vec<_>>();
            let (qlo, qstep) = self.push_vertices(&pick(positions), &pick(&normals), &pick(&colors));
            self.tris.extend_from_slice(&c.local_tris);
            source.extend_from_slice(&c.source_tris);
            let mut cl = Cluster {
                first_vertex,
                vertex_count: c.vertices.len() as u32,
                first_tri,
                tri_count: c.local_tris.len() as u32,
                ..Default::default()
            };
            cl.set_quant(qlo, qstep);
            self.finish_cluster(&mut cl);
            self.clusters.push(cl);
        }
        source
    }

    /// Append a mesh's level-of-detail hierarchy (`lod::build`), one scene
    /// cluster per hierarchy cluster, with its errors and spheres. Returns,
    /// per triangle added, its corners as vertices of the mesh given.
    pub fn push_lod_mesh(&mut self, positions: &[Vec3], normals: &[Vec3], colors: &[Vec3], clusters: &[crate::lod::LodCluster]) -> Vec<[u32; 3]> {
        let mut corners = Vec::new();
        for lc in clusters {
            let first_vertex = self.qpos.len() as u32;
            let first_tri = self.tris.len() as u32;
            let mut local: std::collections::HashMap<u32, u8> = std::collections::HashMap::new();
            let mut verts: Vec<u32> = Vec::new();
            for t in &lc.tris {
                let l = t.map(|v| {
                    *local.entry(v).or_insert_with(|| {
                        verts.push(v);
                        (verts.len() - 1) as u8
                    })
                });
                self.tris.push(l);
                corners.push(*t);
            }
            assert!(verts.len() <= CLUSTER_VERTS, "a level-of-detail cluster has more than {CLUSTER_VERTS} vertices");
            let pick = |a: &[Vec3]| verts.iter().map(|&v| a[v as usize]).collect::<Vec<_>>();
            let (qlo, qstep) = self.push_vertices(&pick(positions), &pick(normals), &pick(colors));
            let mut cl = Cluster {
                first_vertex,
                vertex_count: verts.len() as u32,
                first_tri,
                tri_count: lc.tris.len() as u32,
                lod_level: lc.level,
                lod_error: lc.error,
                lod_centre: lc.sphere.0,
                lod_radius: lc.sphere.1,
                parent_error: lc.parent_error,
                parent_centre: lc.parent_sphere.0,
                parent_radius: lc.parent_sphere.1,
                ..Default::default()
            };
            cl.set_quant(qlo, qstep);
            self.finish_cluster(&mut cl);
            self.clusters.push(cl);
        }
        corners
    }

    /// Fill a cluster's sphere and cone from its vertices and triangles, and
    /// grow the scene bounds.
    pub fn finish_cluster(&mut self, cl: &mut Cluster) {
        let vs: Vec<Vec3> = self.cluster_pos(cl).collect();
        let (mut lo, mut hi) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
        for p in &vs {
            lo = lo.min(*p);
            hi = hi.max(*p);
        }
        let centre = (lo + hi) * 0.5;
        let radius = vs.iter().map(|p| p.distance(centre)).fold(0.0, f32::max);
        self.bounds_min = self.bounds_min.min(lo);
        self.bounds_max = self.bounds_max.max(hi);

        let mut face_normals = Vec::with_capacity(cl.tri_count as usize);
        let mut axis = Vec3::ZERO;
        for t in cl.first_tri..cl.first_tri + cl.tri_count {
            let [a, b, c] = self.tri_pos(cl, t);
            let f = (b - a).cross(c - a);
            if f.length_squared() > 0.0 {
                let f = f.normalize();
                face_normals.push(f);
                axis += f;
            }
        }
        let (axis, cos) = if axis.length_squared() < 1e-12 || face_normals.is_empty() {
            (Vec3::Y, -1.0)
        } else {
            let axis = axis.normalize();
            (axis, face_normals.iter().map(|f| f.dot(axis)).fold(1.0, f32::min))
        };
        cl.centre = centre;
        cl.radius = radius;
        cl.cone_axis = axis;
        cl.cone_cos = cos;
    }
}

/// One cluster as the builder makes it: source vertex ids, and triangles in
/// local indices.
pub struct BuiltCluster {
    /// Source triangle of each local triangle.
    pub source_tris: Vec<u32>,
    pub vertices: Vec<u32>,
    pub local_tris: Vec<[u8; 3]>,
}

/// Cut one mesh into clusters. Triangles are sorted along a Morton curve of
/// their centroids so neighbours land together, then packed greedily until a
/// cluster would exceed `CLUSTER_VERTS` or `CLUSTER_TRIS`.
pub fn build_clusters(positions: &[Vec3], tri_indices: &[u32]) -> Vec<BuiltCluster> {
    let n_tris = tri_indices.len() / 3;
    if n_tris == 0 {
        return Vec::new();
    }
    // Bounds of the vertices these triangles use (not of all positions: the
    // level-of-detail builder calls this once per group of a large mesh).
    let (mut lo, mut hi) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
    for &i in tri_indices {
        lo = lo.min(positions[i as usize]);
        hi = hi.max(positions[i as usize]);
    }
    let ext = (hi - lo).max(Vec3::splat(1e-20));
    let mut order: Vec<(u64, u32)> = (0..n_tris)
        .map(|t| {
            let i = t * 3;
            let c = (positions[tri_indices[i] as usize] + positions[tri_indices[i + 1] as usize] + positions[tri_indices[i + 2] as usize]) / 3.0;
            let q = ((c - lo) / ext * 2_097_151.0).clamp(Vec3::ZERO, Vec3::splat(2_097_151.0));
            (morton3(q.x as u64, q.y as u64, q.z as u64), t as u32)
        })
        .collect();
    order.sort_unstable_by_key(|k| k.0);

    // A cluster must also stay small in space, not just in count: the
    // Morton curve jumps now and then, and a cluster straddling a jump would
    // span the whole mesh, which defeats frustum culling and light reach.
    // Its box may not grow past 16 typical triangle sizes (a full cluster of
    // 124 triangles in a regular patch is about 11 across).
    let mut sizes: Vec<f32> = (0..n_tris)
        .map(|t| {
            let i = t * 3;
            let (a, b, c) = (positions[tri_indices[i] as usize], positions[tri_indices[i + 1] as usize], positions[tri_indices[i + 2] as usize]);
            (a - b).length().max((b - c).length()).max((c - a).length())
        })
        .collect();
    let mid = sizes.len() / 2;
    let typical = *sizes.select_nth_unstable_by(mid, |a, b| a.total_cmp(b)).1;
    let max_diag = typical * 16.0;

    let mut out = Vec::new();
    let mut cur = BuiltCluster { vertices: Vec::new(), local_tris: Vec::new(), source_tris: Vec::new() };
    let (mut blo, mut bhi) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
    // Source vertex -> local index in the current cluster.
    let mut local: std::collections::HashMap<u32, u8> = std::collections::HashMap::new();
    for &(_, t) in &order {
        let i = t as usize * 3;
        let tri = [tri_indices[i], tri_indices[i + 1], tri_indices[i + 2]];
        let new_verts = tri.iter().enumerate().filter(|(k, v)| !local.contains_key(v) && !tri[..*k].contains(v)).count();
        let (mut tlo, mut thi) = (blo, bhi);
        for v in tri {
            tlo = tlo.min(positions[v as usize]);
            thi = thi.max(positions[v as usize]);
        }
        let too_big = !cur.local_tris.is_empty() && (thi - tlo).length() > max_diag;
        if too_big || cur.local_tris.len() + 1 > CLUSTER_TRIS || cur.vertices.len() + new_verts > CLUSTER_VERTS {
            out.push(std::mem::replace(&mut cur, BuiltCluster { vertices: Vec::new(), local_tris: Vec::new(), source_tris: Vec::new() }));
            local.clear();
            (tlo, thi) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
            for v in tri {
                tlo = tlo.min(positions[v as usize]);
                thi = thi.max(positions[v as usize]);
            }
        }
        (blo, bhi) = (tlo, thi);
        let mut lt = [0u8; 3];
        for (k, v) in tri.iter().enumerate() {
            lt[k] = *local.entry(*v).or_insert_with(|| {
                cur.vertices.push(*v);
                (cur.vertices.len() - 1) as u8
            });
        }
        cur.local_tris.push(lt);
        cur.source_tris.push(t);
    }
    if !cur.local_tris.is_empty() {
        out.push(cur);
    }
    out
}

fn spread3(mut x: u64) -> u64 {
    x &= 0x1f_ffff;
    x = (x | x << 32) & 0x1f00000000ffff;
    x = (x | x << 16) & 0x1f0000ff0000ff;
    x = (x | x << 8) & 0x100f00f00f00f00f;
    x = (x | x << 4) & 0x10c30c30c30c30c3;
    x = (x | x << 2) & 0x1249249249249249;
    x
}

fn morton3(x: u64, y: u64, z: u64) -> u64 {
    spread3(x) | spread3(y) << 1 | spread3(z) << 2
}

#[cfg(test)]
pub mod tests {
    use super::*;

    /// A sphere-ish test mesh: a UV sphere with `seg` segments.
    pub fn uv_sphere(seg: usize) -> (Vec<Vec3>, Vec<u32>) {
        let mut p = Vec::new();
        let mut idx = Vec::new();
        let rings = seg / 2;
        for r in 0..=rings {
            let th = std::f32::consts::PI * r as f32 / rings as f32;
            for s in 0..seg {
                let ph = std::f32::consts::TAU * s as f32 / seg as f32;
                p.push(Vec3::new(th.sin() * ph.cos(), th.cos(), th.sin() * ph.sin()));
            }
        }
        for r in 0..rings {
            for s in 0..seg {
                let a = (r * seg + s) as u32;
                let b = (r * seg + (s + 1) % seg) as u32;
                let c = a + seg as u32;
                let d = b + seg as u32;
                idx.extend_from_slice(&[a, b, c, b, d, c]);
            }
        }
        (p, idx)
    }

    /// Positions decode to within half a step of their cluster's box,
    /// normals within 0.003 degrees, colours to the nearest 8-bit sRGB
    /// value; 13 bytes a vertex and 3 a triangle.
    #[test]
    fn geometry_is_held_quantised() {
        let (p, idx) = uv_sphere(32);
        let colors: Vec<Vec3> = p.iter().map(|v| (*v * 0.5 + Vec3::splat(0.5)).powf(2.2)).collect();
        let normals: Vec<Vec3> = p.iter().map(|v| v.normalize()).collect();
        let mut s = Scene::new();
        let source = s.push_mesh_mapped(&p, &normals, &colors, &idx);
        // Each scene vertex came from the source vertex at the same corner.
        let mut src_of = std::collections::HashMap::new();
        let ind = s.indices();
        for (t, &st) in source.iter().enumerate() {
            for k in 0..3 {
                src_of.insert(ind[t * 3 + k], idx[st as usize * 3 + k]);
            }
        }
        for c in &s.clusters {
            for v in c.first_vertex..c.first_vertex + c.vertex_count {
                let q = src_of[&v] as usize;
                assert!((s.pos(c, v as usize) - p[q]).abs().cmple(c.qstep * 0.5 + Vec3::splat(1e-6)).all());
                assert!(s.normal(c, v as usize).dot(normals[q]) > 0.999_999);
                assert!((s.color(v as usize) - colors[q]).abs().max_element() < 0.01);
                assert!(s.pos(c, v as usize).distance(c.centre) <= c.radius);
            }
        }
        let n = s.vertex_count() * 13 + s.triangle_count() * 3;
        assert_eq!(s.geometry_bytes(), n + s.clusters.len() * std::mem::size_of::<Cluster>());
    }

    /// An object's samples lie on its surface and move with it; a model
    /// added after it keeps its own.
    #[test]
    fn samples_follow_their_object() {
        let (p, idx) = uv_sphere(48);
        let mut s = Scene::new();
        s.push_mesh(&p, &[], &[], &idx);
        let o = s.add_object("ball".into(), 0..s.clusters.len(), Vec3::ZERO, Vec3::ZERO);
        let (n, h) = s.scatter_samples(o);
        assert!(n > 1000, "{n}");
        assert_eq!(h, crate::samples::MIN_SPACING);
        let on_sphere = |s: &Scene, c: Vec3, r: f32| s.samples.iter().all(|x| (x.position.distance(c) - r).abs() < 0.01 && x.normal.dot((x.position - c).normalize()) > 0.95);
        assert!(on_sphere(&s, Vec3::ZERO, 1.0));
        s.set_object(o, Vec3::new(3.0, 1.0, 0.0), Vec3::new(0.4, 0.0, 0.2), 2.0);
        assert!(on_sphere(&s, Vec3::new(3.0, 1.0, 0.0), 2.0));
        assert!((s.samples.iter().map(|x| x.area).sum::<f32>() - 16.0 * std::f32::consts::PI).abs() < 0.5);
        let mut other = Scene::new();
        other.push_mesh(&p, &[], &[], &idx);
        let k = other.add_object("other".into(), 0..other.clusters.len(), Vec3::ZERO, Vec3::ZERO);
        other.scatter_samples(k);
        let before = s.samples.len();
        let first = s.clusters.len();
        s.append(other);
        assert_eq!(s.samples_in(0..first).len(), before);
        assert!(s.samples_in(first..s.clusters.len()).len() > 1000);
    }

    /// Moving an object changes its transform, not its stored vertices;
    /// its points, normals and oriented box follow, and a file saved after
    /// the move holds it where it stands.
    #[test]
    fn objects_move_by_their_transform() {
        let (p, idx) = uv_sphere(24);
        let stretched: Vec<Vec3> = p.iter().map(|v| *v * Vec3::new(2.0, 0.5, 1.0)).collect();
        let mut s = Scene::new();
        s.push_mesh(&stretched, &[], &[], &idx);
        let o = s.add_object("egg".into(), 0..s.clusters.len(), Vec3::ZERO, Vec3::ZERO);
        let (stored, before, normals) = (s.qpos.clone(), s.positions(), s.normals());
        let q = euler_quat(Vec3::new(0.3, 0.9, -0.4));
        let to = Vec3::new(1.0, 2.0, 3.0);
        s.set_object(o, to, Vec3::new(0.3, 0.9, -0.4), 1.5);
        assert_eq!(s.qpos, stored, "vertices are not rewritten");
        for ((a, b), (na, nb)) in before.iter().zip(s.positions()).zip(normals.iter().zip(s.normals())) {
            assert!((q * (*a * 1.5) + to).distance(b) < 1e-4);
            assert!((q * *na).dot(nb) > 0.9999);
        }
        // The box turns with it: its corners are the stored box's, moved.
        let ob = &s.objects[o];
        assert!((ob.box_hi - Vec3::new(2.0, 0.5, 1.0)).abs().max_element() < 0.01, "{}", ob.box_hi);
        let far = ob.corners()[7];
        assert!(far.distance(q * (ob.box_hi * 1.5) + to) < 1e-4);
        for c in &s.clusters {
            for p in s.cluster_pos(c) {
                assert!(p.distance(c.centre) <= c.radius + 1e-4);
            }
        }
        // Saved and read again: where it stands, as stored geometry.
        let (t, _) = crate::fkj::read(&crate::fkj::write(&s, &[])).unwrap();
        assert!(t.clusters.iter().all(|c| c.in_place()));
        for (a, b) in s.positions().iter().zip(t.positions()) {
            assert!(a.distance(b) < 1e-4, "{a} {b}");
        }
        for (a, b) in s.normals().iter().zip(t.normals()) {
            assert!(a.dot(b) > 0.9999);
        }
    }

    /// Moving an object quantises it again in its new boxes: a move and its
    /// way back leave it within a step of where it was.
    #[test]
    fn moving_and_back_stays_within_a_step() {
        let (p, idx) = uv_sphere(24);
        let mut s = Scene::new();
        s.push_mesh(&p, &[], &[], &idx);
        let o = s.add_object("ball".into(), 0..s.clusters.len(), Vec3::ZERO, Vec3::ZERO);
        let before = s.positions();
        for _ in 0..20 {
            s.set_object(o, Vec3::new(3.0, 1.0, -2.0), Vec3::new(0.3, 1.1, -0.7), 1.7);
            s.set_object(o, Vec3::ZERO, Vec3::ZERO, 1.0);
        }
        let step = s.clusters.iter().map(|c| c.qstep.length()).fold(0.0, f32::max);
        for (a, b) in before.iter().zip(s.positions()) {
            assert!(a.distance(b) < 20.0 * step + 1e-5, "{a} {b}");
        }
    }

    #[test]
    fn clusters_respect_limits_and_keep_every_triangle() {
        let (p, idx) = uv_sphere(96);
        let mut s = Scene::new();
        s.push_mesh(&p, &[], &[], &idx);
        assert_eq!(s.triangle_count(), idx.len() / 3);
        let mut seen = std::collections::HashSet::new();
        for c in &s.clusters {
            assert!(c.vertex_count as usize <= CLUSTER_VERTS);
            assert!(c.tri_count as usize <= CLUSTER_TRIS);
            for t in c.first_tri..c.first_tri + c.tri_count {
                for k in 0..3 {
                    let v = s.indices()[t as usize * 3 + k];
                    assert!(v >= c.first_vertex && v < c.first_vertex + c.vertex_count, "triangle leaves its cluster");
                }
                // Same triangle as a set of source positions, to check none is lost or doubled.
                let key: Vec<[i32; 3]> = (0..3)
                    .map(|k| {
                        let q = s.positions()[s.indices()[t as usize * 3 + k] as usize] * 1e4;
                        [q.x as i32, q.y as i32, q.z as i32]
                    })
                    .collect();
                let mut key = key;
                key.sort();
                assert!(seen.insert(key));
            }
            for v in c.first_vertex..c.first_vertex + c.vertex_count {
                assert!(s.positions()[v as usize].distance(c.centre) <= c.radius + 1e-5);
            }
        }
        assert_eq!(seen.len(), idx.len() / 3);
        // Clusters should be well filled, not one triangle each.
        assert!(s.clusters.len() * 40 < idx.len() / 3 * 2, "{} clusters for {} triangles", s.clusters.len(), idx.len() / 3);
    }

    /// Moving, turning and scaling an object moves its vertices, normals
    /// and clusters together, keeps its triangles' shapes, and leaves the
    /// rest of the scene where it was.
    #[test]
    fn objects_move_turn_and_scale() {
        let (p, idx) = uv_sphere(32);
        let mut s = Scene::new();
        s.push_mesh(&[Vec3::new(-50.0, 0.0, -50.0), Vec3::new(50.0, 0.0, -50.0), Vec3::new(0.0, 0.0, 50.0)], &[], &[], &[0, 1, 2]);
        let ground = s.positions().clone();
        let first = s.clusters.len();
        let lifted: Vec<Vec3> = p.iter().map(|v| *v + Vec3::Y).collect();
        s.push_mesh(&lifted, &[], &[], &idx);
        s.add_object("ball".into(), first..s.clusters.len(), Vec3::ZERO, Vec3::ZERO);
        let before = s.positions().clone();
        let (yaw, k, to) = (0.7f32, 2.0f32, Vec3::new(3.0, 0.0, -2.0));
        s.set_object(0, to, Vec3::new(0.0, yaw, 0.0), k);
        let q = glam::Quat::from_rotation_y(yaw);
        let vbase = s.clusters[first].first_vertex as usize;
        for v in vbase..s.positions().len() {
            let want = q * (before[v] * k) + to;
            assert!(s.positions()[v].distance(want) < 1e-4);
        }
        assert_eq!(&s.positions()[..ground.len()], &ground[..], "the ground stays put");
        for c in &s.clusters[first..] {
            for v in c.first_vertex..c.first_vertex + c.vertex_count {
                assert!(s.positions()[v as usize].distance(c.centre) <= c.radius + 1e-4);
            }
        }
        // And back, from the absolute placement.
        s.set_object(0, Vec3::ZERO, Vec3::ZERO, 1.0);
        for v in vbase..s.positions().len() {
            assert!(s.positions()[v].distance(before[v]) < 1e-4);
        }
        assert_eq!(s.object_of(first), Some(0));
        assert_eq!(s.object_of(0), None);
    }

    /// A group moves, turns and scales what is under it about its own
    /// pivot, at any depth; nothing can be put under itself.
    #[test]
    fn groups_carry_their_children() {
        let (p, idx) = uv_sphere(16);
        let mut s = Scene::new();
        let a0 = s.clusters.len();
        s.push_mesh(&p.iter().map(|v| *v * 0.5 + Vec3::new(2.0, 0.5, 0.0)).collect::<Vec<_>>(), &[], &[], &idx);
        let a = s.add_object("a".into(), a0..s.clusters.len(), Vec3::new(2.0, 0.0, 0.0), Vec3::ZERO);
        let b0 = s.clusters.len();
        s.push_mesh(&p.iter().map(|v| *v * 0.5 + Vec3::new(-2.0, 0.5, 0.0)).collect::<Vec<_>>(), &[], &[], &idx);
        let b = s.add_object("b".into(), b0..s.clusters.len(), Vec3::new(-2.0, 0.0, 0.0), Vec3::ZERO);
        let g = s.add_group("g".into(), Vec3::ZERO);
        let top = s.add_group("top".into(), Vec3::ZERO);
        assert!(s.set_parent(a, Some(g)) && s.set_parent(b, Some(g)) && s.set_parent(g, Some(top)));
        assert!(!s.set_parent(top, Some(a)), "a cycle");
        assert!(!s.set_parent(g, Some(g)));
        assert_eq!(s.descendants(top).len(), 3);
        let (lo, hi) = s.object_bounds(top);
        assert!((lo.x + 2.5).abs() < 0.01 && (hi.x - 2.5).abs() < 0.01, "{lo} {hi}");
        // A half turn and twice the size about the origin: a goes to x = -4.
        s.set_object(top, Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, std::f32::consts::PI, 0.0), 2.0);
        let oa = &s.objects[a];
        assert!(oa.pivot.distance(Vec3::new(-4.0, 0.0, 1.0)) < 1e-4, "{}", oa.pivot);
        assert!((oa.scale - 2.0).abs() < 1e-5 && euler_quat(oa.rotation).angle_between(glam::Quat::from_rotation_y(std::f32::consts::PI)) < 1e-4);
        let (lo, hi) = s.object_bounds(a);
        assert!(((lo + hi) * 0.5).distance(Vec3::new(-4.0, 1.0, 1.0)) < 0.02, "{lo} {hi}");
        assert_eq!(s.moving_clusters(top).len(), 2);
        // Out of the group: moving the group leaves it.
        assert!(s.set_parent(a, None));
        let before = s.objects[a].pivot;
        s.set_object(top, Vec3::ZERO, Vec3::ZERO, 1.0);
        assert_eq!(s.objects[a].pivot, before);
    }

    /// Euler angles turn X first, then Y, then Z, as Maya's xyz order, and
    /// come back from the rotation they make.
    #[test]
    fn euler_angles_follow_maya_order() {
        let h = std::f32::consts::FRAC_PI_2;
        // About Y a quarter turn: X goes to -Z.
        assert!((euler_quat(Vec3::new(0.0, h, 0.0)) * Vec3::X).distance(Vec3::NEG_Z) < 1e-5);
        // X then Y: Y goes to Z, then to X.
        assert!((euler_quat(Vec3::new(h, h, 0.0)) * Vec3::Y).distance(Vec3::X) < 1e-5);
        let r = Vec3::new(0.3, -0.7, 1.1);
        assert!(quat_euler(euler_quat(r)).distance(r) < 1e-5);
    }

    /// Turned about any axis, an object's vertices, normals and bounds
    /// follow, and turning it back restores it.
    #[test]
    fn objects_turn_about_any_axis() {
        let (p, idx) = uv_sphere(16);
        let mut s = Scene::new();
        let lifted: Vec<Vec3> = p.iter().map(|v| *v * Vec3::new(2.0, 0.5, 0.5) + Vec3::Y).collect();
        s.push_mesh(&lifted, &[], &[], &idx);
        let o = s.add_object("long".into(), 0..s.clusters.len(), Vec3::ZERO, Vec3::ZERO);
        let before = s.positions().clone();
        let r = Vec3::new(0.0, 0.0, std::f32::consts::FRAC_PI_2);
        s.set_object(o, Vec3::ZERO, r, 1.0);
        let q = euler_quat(r);
        for (a, b) in before.iter().zip(&s.positions()) {
            assert!((q * *a).distance(*b) < 1e-4);
        }
        // Stood on end: tall now, not long.
        let (lo, hi) = s.object_bounds(o);
        assert!(hi.y - lo.y > 3.9 && hi.x - lo.x < 1.1, "{lo} {hi}");
        for c in &s.clusters {
            for v in c.first_vertex..c.first_vertex + c.vertex_count {
                assert!(s.positions()[v as usize].distance(c.centre) <= c.radius + 1e-4);
            }
        }
        s.set_object(o, Vec3::ZERO, Vec3::ZERO, 1.0);
        for (a, b) in before.iter().zip(&s.positions()) {
            assert!(a.distance(*b) < 1e-4);
        }
    }

    #[test]
    fn clusters_stay_small_in_space() {
        // A long strip of quads: Morton order jumps along it, and no cluster
        // may span the jump.
        let mut p = Vec::new();
        let mut idx = Vec::new();
        for k in 0..4000u32 {
            let x = k as f32;
            p.extend_from_slice(&[Vec3::new(x, 0.0, 0.0), Vec3::new(x + 1.0, 0.0, 0.0), Vec3::new(x + 1.0, 0.0, -1.0), Vec3::new(x, 0.0, -1.0)]);
            let b = k * 4;
            idx.extend_from_slice(&[b, b + 1, b + 2, b, b + 2, b + 3]);
        }
        let mut s = Scene::new();
        s.push_mesh(&p, &[], &[], &idx);
        for c in &s.clusters {
            assert!(c.radius < 16.0, "cluster radius {}", c.radius);
        }
    }

    #[test]
    fn normal_cone_contains_every_face() {
        let (p, idx) = uv_sphere(64);
        let mut s = Scene::new();
        s.push_mesh(&p, &[], &[], &idx);
        for c in &s.clusters {
            for t in c.first_tri..c.first_tri + c.tri_count {
                let i = t as usize * 3;
                let (a, b, cc) = (s.positions()[s.indices()[i] as usize], s.positions()[s.indices()[i + 1] as usize], s.positions()[s.indices()[i + 2] as usize]);
                let f = (b - a).cross(cc - a);
                if f.length_squared() > 0.0 {
                    assert!(f.normalize().dot(c.cone_axis) >= c.cone_cos - 1e-5);
                }
            }
        }
    }
}
