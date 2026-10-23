//! The scene the rasteriser reads: clustered triangle meshes in flat SoA
//! arrays, and point lights.
//!
//! Geometry is cut into clusters of at most `CLUSTER_VERTS` vertices and
//! `CLUSTER_TRIS` triangles. Each cluster owns a contiguous run of vertices
//! and of triangles, and carries a bounding sphere and a normal cone, so the
//! renderer can drop a whole cluster that is off screen or faces away before
//! touching its vertices. This is the layout the `.fkj` file stores
//! (`src/fkj.rs`); loading decodes it straight into these arrays.


// Building clusters (`push_mesh`) is used by the tests and the baking tool;
// the viewer gets its clusters from the file.
#![allow(dead_code)]

use glam::Vec3;

/// Most vertices in one cluster. Local indices fit a byte.
pub const CLUSTER_VERTS: usize = 64;
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
        }
    }
}

impl Cluster {
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
    /// World-space positions, one per vertex.
    pub positions: Vec<Vec3>,
    /// World-space unit normals, one per vertex.
    pub normals: Vec<Vec3>,
    /// Linear RGB albedo, one per vertex.
    pub colors: Vec<Vec3>,
    /// Three global vertex indices per triangle. A triangle only references
    /// vertices of its own cluster.
    pub indices: Vec<u32>,
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
    /// Turn about the vertical, radians, and uniform scale, since placed.
    pub yaw: f32,
    pub scale: f32,
    /// Bounds of its finest level's vertices.
    pub lo: Vec3,
    pub hi: Vec3,
    /// The group it is in, if any (an index into `Scene::objects`).
    pub parent: Option<usize>,
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
        self.indices.len() / 3
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
        for p in &mut self.positions {
            *p += d;
        }
        for c in &mut self.clusters {
            c.centre += d;
            c.move_lod(|p| p + d, 1.0);
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
        for p in &mut self.positions {
            *p = q * (*p * scale) + d;
        }
        for n in &mut self.normals {
            *n = q * *n;
        }
        for c in &mut self.clusters {
            c.centre = q * (c.centre * scale) + d;
            c.radius *= scale;
            c.cone_axis = q * c.cone_axis;
            c.move_lod(|p| q * (p * scale) + d, scale);
        }
        for l in &mut self.lights {
            l.position = q * (l.position * scale) + d;
        }
        let (mut lo, mut hi) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
        for p in &self.positions {
            lo = lo.min(*p);
            hi = hi.max(*p);
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
        let vbase = self.positions.len() as u32;
        let tbase = self.triangle_count() as u32;
        self.positions.extend(other.positions);
        self.normals.extend(other.normals);
        self.colors.extend(other.colors);
        self.indices.extend(other.indices.iter().map(|i| i + vbase));
        self.clusters.extend(other.clusters.iter().map(|c| Cluster { first_vertex: c.first_vertex + vbase, first_tri: c.first_tri + tbase, ..*c }));
        self.bounds_min = self.bounds_min.min(other.bounds_min);
        self.bounds_max = self.bounds_max.max(other.bounds_max);
        self.lights.extend(other.lights);
    }

    /// Make clusters `clusters` an object turning about `pivot`, already
    /// turned by `yaw`.
    pub fn add_object(&mut self, name: String, clusters: std::ops::Range<usize>, pivot: Vec3, yaw: f32) -> usize {
        self.objects.push(Object { name, clusters, pivot, yaw, scale: 1.0, lo: Vec3::ZERO, hi: Vec3::ZERO, parent: None });
        let i = self.objects.len() - 1;
        self.refresh_bounds(i);
        i
    }

    /// Make an empty group at `pivot`.
    pub fn add_group(&mut self, name: String, pivot: Vec3) -> usize {
        let n = self.clusters.len();
        self.objects.push(Object { name, clusters: n..n, pivot, yaw: 0.0, scale: 1.0, lo: pivot, hi: pivot, parent: None });
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

    /// Measure an object's bounds again from its finest level's vertices.
    pub fn refresh_bounds(&mut self, i: usize) {
        use rayon::prelude::*;
        let range = self.objects[i].clusters.clone();
        if range.is_empty() {
            let p = self.objects[i].pivot;
            self.objects[i].lo = p;
            self.objects[i].hi = p;
            return;
        }
        let finest = self.clusters[range.clone()].iter().map(|c| c.lod_level).min().unwrap_or(0);
        let (lo, hi) = self.clusters[range]
            .par_iter()
            .filter(|c| c.lod_level == finest)
            .map(|c| {
                let (mut lo, mut hi) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
                for p in &self.positions[c.first_vertex as usize..(c.first_vertex + c.vertex_count) as usize] {
                    lo = lo.min(*p);
                    hi = hi.max(*p);
                }
                (lo, hi)
            })
            .reduce(|| (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)), |a, b| (a.0.min(b.0), a.1.max(b.1)));
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

    /// Place object `i` at `pivot`, turned to `yaw` and scaled to `scale`
    /// (absolute, from how it was first placed): its vertices, normals,
    /// clusters and levels of detail move with it. Turning about the
    /// vertical and scaling the same on every axis keep each triangle's
    /// corners in order, so its texture tile stays valid.
    /// Everything under it moves with it, about its pivot.
    pub fn set_object(&mut self, i: usize, pivot: Vec3, yaw: f32, scale: f32) {
        let o = self.objects[i].clone();
        let scale = scale.max(1e-4);
        let (dyaw, k) = (yaw - o.yaw, scale / o.scale);
        if pivot == o.pivot && dyaw == 0.0 && k == 1.0 {
            return;
        }
        let q = glam::Quat::from_rotation_y(dyaw);
        let map = |p: Vec3| q * ((p - o.pivot) * k) + pivot;
        for j in self.descendants(i) {
            let d = self.objects[j].clone();
            self.place_one(j, map(d.pivot), d.yaw + dyaw, d.scale * k);
        }
        self.place_one(i, pivot, yaw, scale);
    }

    /// `set_object` for object `i` alone.
    fn place_one(&mut self, i: usize, pivot: Vec3, yaw: f32, scale: f32) {
        use rayon::prelude::*;
        let o = self.objects[i].clone();
        let (dyaw, k) = (yaw - o.yaw, scale / o.scale);
        let q = glam::Quat::from_rotation_y(dyaw);
        let map = |p: Vec3| q * ((p - o.pivot) * k) + pivot;
        if o.clusters.is_empty() {
            let ob = &mut self.objects[i];
            ob.pivot = pivot;
            ob.yaw = yaw;
            ob.scale = scale;
            self.refresh_bounds(i);
            return;
        }
        let cs = &self.clusters[o.clusters.clone()];
        let (Some(first), Some(last)) = (cs.first(), cs.last()) else { return };
        let verts = first.first_vertex as usize..(last.first_vertex + last.vertex_count) as usize;
        self.positions[verts.clone()].par_iter_mut().for_each(|p| *p = map(*p));
        self.normals[verts].par_iter_mut().for_each(|n| *n = q * *n);
        for c in &mut self.clusters[o.clusters.clone()] {
            c.centre = map(c.centre);
            c.radius *= k;
            c.cone_axis = q * c.cone_axis;
            c.move_lod(map, k);
        }
        let ob = &mut self.objects[i];
        ob.pivot = pivot;
        ob.yaw = yaw;
        ob.scale = scale;
        self.refresh_bounds(i);
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
            let first_vertex = self.positions.len() as u32;
            let first_tri = (self.indices.len() / 3) as u32;
            for &v in &c.vertices {
                self.positions.push(positions[v as usize]);
                self.normals.push(normals[v as usize]);
                self.colors.push(colors[v as usize]);
            }
            for t in &c.local_tris {
                self.indices.extend(t.iter().map(|&l| first_vertex + l as u32));
            }
            source.extend_from_slice(&c.source_tris);
            let mut cl = Cluster {
                first_vertex,
                vertex_count: c.vertices.len() as u32,
                first_tri,
                tri_count: c.local_tris.len() as u32,
                ..Default::default()
            };
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
            let first_vertex = self.positions.len() as u32;
            let first_tri = (self.indices.len() / 3) as u32;
            let mut local: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
            for t in &lc.tris {
                for &v in t {
                    let l = *local.entry(v).or_insert_with(|| {
                        self.positions.push(positions[v as usize]);
                        self.normals.push(normals[v as usize]);
                        self.colors.push(colors[v as usize]);
                        self.positions.len() as u32 - 1
                    });
                    self.indices.push(l);
                }
                corners.push(*t);
            }
            let mut cl = Cluster {
                first_vertex,
                vertex_count: self.positions.len() as u32 - first_vertex,
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
            self.finish_cluster(&mut cl);
            self.clusters.push(cl);
        }
        corners
    }

    /// Fill a cluster's sphere and cone from its vertices and triangles, and
    /// grow the scene bounds.
    pub fn finish_cluster(&mut self, cl: &mut Cluster) {
        let vs = &self.positions[cl.first_vertex as usize..(cl.first_vertex + cl.vertex_count) as usize];
        let (mut lo, mut hi) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
        for p in vs {
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
            let i = t as usize * 3;
            let (a, b, c) = (
                self.positions[self.indices[i] as usize],
                self.positions[self.indices[i + 1] as usize],
                self.positions[self.indices[i + 2] as usize],
            );
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
                    let v = s.indices[t as usize * 3 + k];
                    assert!(v >= c.first_vertex && v < c.first_vertex + c.vertex_count, "triangle leaves its cluster");
                }
                // Same triangle as a set of source positions, to check none is lost or doubled.
                let key: Vec<[i32; 3]> = (0..3)
                    .map(|k| {
                        let q = s.positions[s.indices[t as usize * 3 + k] as usize] * 1e4;
                        [q.x as i32, q.y as i32, q.z as i32]
                    })
                    .collect();
                let mut key = key;
                key.sort();
                assert!(seen.insert(key));
            }
            for v in c.first_vertex..c.first_vertex + c.vertex_count {
                assert!(s.positions[v as usize].distance(c.centre) <= c.radius + 1e-5);
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
        let ground = s.positions.clone();
        let first = s.clusters.len();
        let lifted: Vec<Vec3> = p.iter().map(|v| *v + Vec3::Y).collect();
        s.push_mesh(&lifted, &[], &[], &idx);
        s.add_object("ball".into(), first..s.clusters.len(), Vec3::ZERO, 0.0);
        let before = s.positions.clone();
        let (yaw, k, to) = (0.7f32, 2.0f32, Vec3::new(3.0, 0.0, -2.0));
        s.set_object(0, to, yaw, k);
        let q = glam::Quat::from_rotation_y(yaw);
        let vbase = s.clusters[first].first_vertex as usize;
        for v in vbase..s.positions.len() {
            let want = q * (before[v] * k) + to;
            assert!(s.positions[v].distance(want) < 1e-4);
        }
        assert_eq!(&s.positions[..ground.len()], &ground[..], "the ground stays put");
        for c in &s.clusters[first..] {
            for v in c.first_vertex..c.first_vertex + c.vertex_count {
                assert!(s.positions[v as usize].distance(c.centre) <= c.radius + 1e-4);
            }
        }
        // And back, from the absolute placement.
        s.set_object(0, Vec3::ZERO, 0.0, 1.0);
        for v in vbase..s.positions.len() {
            assert!(s.positions[v].distance(before[v]) < 1e-4);
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
        let a = s.add_object("a".into(), a0..s.clusters.len(), Vec3::new(2.0, 0.0, 0.0), 0.0);
        let b0 = s.clusters.len();
        s.push_mesh(&p.iter().map(|v| *v * 0.5 + Vec3::new(-2.0, 0.5, 0.0)).collect::<Vec<_>>(), &[], &[], &idx);
        let b = s.add_object("b".into(), b0..s.clusters.len(), Vec3::new(-2.0, 0.0, 0.0), 0.0);
        let g = s.add_group("g".into(), Vec3::ZERO);
        let top = s.add_group("top".into(), Vec3::ZERO);
        assert!(s.set_parent(a, Some(g)) && s.set_parent(b, Some(g)) && s.set_parent(g, Some(top)));
        assert!(!s.set_parent(top, Some(a)), "a cycle");
        assert!(!s.set_parent(g, Some(g)));
        assert_eq!(s.descendants(top).len(), 3);
        let (lo, hi) = s.object_bounds(top);
        assert!((lo.x + 2.5).abs() < 0.01 && (hi.x - 2.5).abs() < 0.01, "{lo} {hi}");
        // A half turn and twice the size about the origin: a goes to x = -4.
        s.set_object(top, Vec3::new(0.0, 0.0, 1.0), std::f32::consts::PI, 2.0);
        let oa = &s.objects[a];
        assert!(oa.pivot.distance(Vec3::new(-4.0, 0.0, 1.0)) < 1e-4, "{}", oa.pivot);
        assert!((oa.scale - 2.0).abs() < 1e-5 && (oa.yaw - std::f32::consts::PI).abs() < 1e-5);
        let (lo, hi) = s.object_bounds(a);
        assert!(((lo + hi) * 0.5).distance(Vec3::new(-4.0, 1.0, 1.0)) < 0.02, "{lo} {hi}");
        assert_eq!(s.moving_clusters(top).len(), 2);
        // Out of the group: moving the group leaves it.
        assert!(s.set_parent(a, None));
        let before = s.objects[a].pivot;
        s.set_object(top, Vec3::ZERO, 0.0, 1.0);
        assert_eq!(s.objects[a].pivot, before);
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
                let (a, b, cc) = (s.positions[s.indices[i] as usize], s.positions[s.indices[i + 1] as usize], s.positions[s.indices[i + 2] as usize]);
                let f = (b - a).cross(cc - a);
                if f.length_squared() > 0.0 {
                    assert!(f.normalize().dot(c.cone_axis) >= c.cone_cos - 1e-5);
                }
            }
        }
    }
}
