//! Shading the REYES way, kept between frames.
//!
//! Triangles are diced into micropolygons in world space, a level per
//! cluster chosen from its size on screen (`raster::dice_level`): level 0
//! shades the mesh's own vertices, level L cuts each triangle into 2^L parts
//! per edge. Shading happens on those points, in world space, before the
//! picture is made, and the results are kept (Cook, Carpenter, Catmull
//! 1987). The picture is then resolved by interpolating the shaded points
//! under each pixel.
//!
//! What is kept, and what throws it away:
//!
//! - Diffuse light at each point: kept until a light whose old or new reach
//!   touches the point's cluster changes, or the sky or exposure changes.
//!   Moving a lamp on the chess table re-shades the table and nothing else.
//! - The highlight at each point: depends on the eye, so also redone when
//!   the camera moves.
//! - Visibility: kept while the camera does not move. Dragging a light
//!   skips it entirely.
//! - Shadow maps: kept until their light moves or changes strength.
//!
//! Only points under visible pixels are shaded: a cluster records which of
//! its triangles (or, at level 0, vertices) are needed and which are done.

use std::sync::atomic::{AtomicU64, Ordering};

use glam::{Mat4, Vec3};
use rayon::prelude::*;

use crate::raster::{self, Camera, Frame, NO_TRI, NOT_VISIBLE, RenderSettings, SPECULAR, Sky, View};
use crate::scene::{PointLight, Scene};
use crate::shadow::{CubeShadow, SHADOW_RES};

/// Shading kept for one cluster.
#[derive(Default)]
struct ClusterShade {
    /// Dicing level of what is stored; `NOT_VISIBLE` when nothing is.
    level: u8,
    /// Per unit (vertex at level 0, triangle above), where its points start
    /// in `diffuse` and `spec`; `u32::MAX` before first use.
    offset: Vec<u32>,
    diffuse: Vec<Vec3>,
    spec: Vec<Vec3>,
    /// Units whose diffuse light is current.
    done: u128,
    /// Units whose highlight is current for `spec_eye`.
    spec_done: u128,
    spec_eye: Vec3,
    /// The frame in which anything here was last shaded.
    stamp: u64,
    /// Units seen from their back when last shaded. Shading is two-sided:
    /// normals are turned towards the eye, so a unit seen from the other
    /// side has its diffuse light redone.
    back: u128,
}

impl ClusterShade {
    fn reset(&mut self, level: u8, units: usize) {
        self.level = level;
        self.offset.clear();
        self.offset.resize(units, u32::MAX);
        self.diffuse.clear();
        self.spec.clear();
        self.done = 0;
        self.spec_done = 0;
        self.back = 0;
    }
}

/// Points per triangle at a dicing level.
pub fn grid_points(level: u8) -> usize {
    let n = 1usize << level;
    (n + 1) * (n + 2) / 2
}

/// Index of grid point (i, j), i steps towards the triangle's second vertex
/// and j towards its third, i + j <= n.
#[inline]
fn grid_index(n: usize, i: usize, j: usize) -> usize {
    i * (n + 1) - i * i.saturating_sub(1) / 2 + j
}

#[derive(PartialEq, Clone, Copy)]
struct VisKey {
    view: Mat4,
    proj: Mat4,
    size: (usize, usize),
    cull: bool,
    wireframe: bool,
}

static NEXT_STAMP: AtomicU64 = AtomicU64::new(1);

/// Everything kept from one frame to the next.
pub struct Renderer {
    scene_key: (usize, usize, usize),
    cache: Vec<ClusterShade>,
    shadows: Vec<Option<CubeShadow>>,
    prev_lights: Vec<PointLight>,
    prev_reach: Vec<f32>,
    prev_global: Option<(Sky, f32, bool, bool)>,
    vis_key: Option<VisKey>,
    vis_stamp: u64,
    levels: Vec<u8>,
    /// Units needed by the current visibility, per cluster (triangles).
    need: Vec<u128>,
    visible: Vec<u32>,
    /// The last resolved picture, before light gizmos are drawn on it.
    base: Vec<u32>,
    /// Which view the base was resolved with.
    base_view: Option<View>,
    frame_no: u64,
}

impl Default for Renderer {
    fn default() -> Self {
        Self::new()
    }
}

impl Renderer {
    pub fn new() -> Self {
        Self {
            scene_key: (0, 0, 0),
            cache: Vec::new(),
            shadows: Vec::new(),
            prev_lights: Vec::new(),
            prev_reach: Vec::new(),
            prev_global: None,
            vis_key: None,
            vis_stamp: 0,
            levels: Vec::new(),
            need: Vec::new(),
            visible: Vec::new(),
            base: Vec::new(),
            base_view: None,
            frame_no: 0,
        }
    }

    /// Memory held by the shading cache and the shadow maps, in bytes.
    pub fn memory(&self) -> usize {
        let shade: usize = self.cache.iter().map(|c| (c.diffuse.capacity() + c.spec.capacity()) * 12 + c.offset.capacity() * 4).sum();
        let shadows: usize = self.shadows.iter().flatten().map(|s| s.bytes()).sum();
        shade + shadows
    }

    pub fn render(&mut self, scene: &Scene, cam: &Camera, lights: &[PointLight], settings: &RenderSettings, frame: &mut Frame) {
        let n_lights = lights.len().min(64);
        let lights = &lights[..n_lights];
        let key = (scene.positions.as_ptr() as usize, scene.positions.len(), scene.clusters.len());
        if key != self.scene_key {
            *self = Self::new();
            self.scene_key = key;
            self.cache = (0..scene.clusters.len()).map(|_| ClusterShade { level: NOT_VISIBLE, ..Default::default() }).collect();
        }
        frame.stats = Default::default();
        self.frame_no += 1;
        let frame_no = self.frame_no;
        // Whether every pixel has to be resolved again, or only those of
        // clusters shaded this frame.
        let mut full_resolve = false;

        // Light reach for this frame.
        let cut = raster::cut_lux(&settings.sky, settings.exposure);
        let reach: Vec<f32> = lights.iter().map(|l| if settings.light_reach { raster::light_reach(l, cut) } else { f32::INFINITY }).collect();

        // What changed since the last frame decides what is thrown away.
        let global = (settings.sky, settings.exposure, settings.light_reach, settings.shadows);
        if self.prev_global != Some(global) || self.prev_lights.len() != lights.len() {
            full_resolve = true;
            for c in &mut self.cache {
                c.done = 0;
                c.spec_done = 0;
            }
            self.shadows = (0..lights.len()).map(|_| None).collect();
        } else {
            let mut spheres: Vec<(Vec3, f32)> = Vec::new();
            for (i, l) in lights.iter().enumerate() {
                let old = &self.prev_lights[i];
                if old != l {
                    spheres.push((old.position, self.prev_reach[i]));
                    spheres.push((l.position, reach[i]));
                    if old.position != l.position || old.lumens != l.lumens {
                        self.shadows[i] = None;
                    }
                }
            }
            if !spheres.is_empty() {
                self.cache.par_iter_mut().zip(scene.clusters.par_iter()).for_each(|(e, c)| {
                    if spheres.iter().any(|(p, r)| p.distance(c.centre) <= c.radius + r) {
                        e.done = 0;
                        e.spec_done = 0;
                    }
                });
            }
        }
        self.prev_global = Some(global);
        self.prev_lights = lights.to_vec();
        self.prev_reach = reach.clone();

        // Shadow maps for lights that have none.
        let ts = std::time::Instant::now();
        if settings.shadows {
            let missing: Vec<usize> = (0..lights.len()).filter(|&i| self.shadows[i].is_none()).collect();
            let built: Vec<(usize, CubeShadow)> = missing
                .par_iter()
                .map(|&i| {
                    // With light reach off, a map still has to stop somewhere:
                    // the whole scene.
                    let r = if reach[i].is_finite() { reach[i] } else { scene.radius() * 4.0 + lights[i].position.distance(scene.centre()) };
                    (i, CubeShadow::build(scene, lights[i].position, r, SHADOW_RES))
                })
                .collect();
            frame.stats.shadow_maps_built = built.len();
            for (i, m) in built {
                frame.stats.shadow_tris += m.tris;
                self.shadows[i] = Some(m);
            }
        }
        frame.stats.shadow_ms = ts.elapsed().as_secs_f32() * 1000.0;

        // Visibility, unless the camera and the frame are as they were.
        let vk = VisKey { view: cam.view, proj: cam.proj, size: (frame.width, frame.height), cull: settings.cull_backfaces, wireframe: settings.wireframe };
        if self.vis_key == Some(vk) && frame.vis_stamp == self.vis_stamp && self.vis_stamp != 0 {
            frame.stats.visibility_kept = true;
            frame.stats.clusters_in = scene.clusters.len();
            frame.stats.clusters_drawn = self.levels.iter().filter(|l| **l != NOT_VISIBLE).count();
            frame.stats.tris_in = scene.triangle_count();
            frame.stats.tris_drawn = frame.setups.len();
        } else {
            full_resolve = true;
            self.levels = raster::visibility(scene, cam, settings.cull_backfaces, settings.wireframe, frame);
            self.vis_key = Some(vk);
            self.vis_stamp = NEXT_STAMP.fetch_add(1, Ordering::Relaxed);
            frame.vis_stamp = self.vis_stamp;
            // Which triangles of which clusters are seen.
            self.need.clear();
            self.need.resize(scene.clusters.len(), 0);
            for &id in &frame.tri_id {
                if id != NO_TRI {
                    let s = &frame.setups[id as usize];
                    let c = &scene.clusters[s.cluster as usize];
                    self.need[s.cluster as usize] |= 1u128 << (s.tri - c.first_tri);
                }
            }
            self.visible = (0..scene.clusters.len() as u32).filter(|&c| self.need[c as usize] != 0).collect();
        }

        // Shade what is needed and not already kept.
        let t_shade = std::time::Instant::now();
        let evals = AtomicU64::new(0);
        let shaded = AtomicU64::new(0);
        let used = AtomicU64::new(0);
        let clusters_shaded = AtomicU64::new(0);
        let shadows: Vec<Option<&CubeShadow>> = (0..lights.len()).map(|i| if settings.shadows { self.shadows[i].as_ref() } else { None }).collect();
        let eye = cam.eye;
        let levels = &self.levels;
        let need = &self.need;
        let sky = settings.sky;
        if settings.view == View::Shaded {
            self.cache.par_iter_mut().enumerate().for_each(|(ci, e)| {
                let need_tris = need[ci];
                if need_tris == 0 {
                    return;
                }
                let c = &scene.clusters[ci];
                let want = levels[ci];
                // Keep what is stored if it is as fine as needed and not more
                // than one level finer.
                if e.level == NOT_VISIBLE || e.level < want || e.level > want.saturating_add(1) {
                    let units = if want == 0 { c.vertex_count } else { c.tri_count } as usize;
                    e.reset(want, units);
                }
                if e.spec_eye != eye {
                    e.spec_done = 0;
                    e.spec_eye = eye;
                }
                let level = e.level;
                let need_units = if level == 0 { vertex_mask(scene, c, need_tris) } else { need_tris };
                used.fetch_add(
                    if level == 0 { need_units.count_ones() as u64 } else { need_units.count_ones() as u64 * grid_points(level) as u64 },
                    Ordering::Relaxed,
                );
                let todo = need_units & !(e.done & e.spec_done);
                if todo == 0 {
                    return;
                }
                clusters_shaded.fetch_add(1, Ordering::Relaxed);
                e.stamp = frame_no;
                // Lights that reach this cluster.
                let in_reach: Vec<usize> = (0..lights.len()).filter(|&i| lights[i].position.distance(c.centre) <= c.radius + reach[i]).collect();
                let per = if level == 0 { 1 } else { grid_points(level) };
                let mut bits = todo;
                let (mut n_shaded, mut n_evals) = (0u64, 0u64);
                while bits != 0 {
                    let u = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    if e.offset[u] == u32::MAX {
                        e.offset[u] = e.diffuse.len() as u32;
                        e.diffuse.resize(e.diffuse.len() + per, Vec3::ZERO);
                        e.spec.resize(e.spec.len() + per, Vec3::ZERO);
                    }
                    let off = e.offset[u] as usize;
                    // Which side the eye is on: from the vertex normal at
                    // level 0, from the triangle's face otherwise.
                    let back = if level == 0 {
                        let v = c.first_vertex as usize + u;
                        scene.normals[v].dot(eye - scene.positions[v]) < 0.0
                    } else {
                        let t = (c.first_tri as usize + u) * 3;
                        let (a, b, cc) = (scene.positions[scene.indices[t] as usize], scene.positions[scene.indices[t + 1] as usize], scene.positions[scene.indices[t + 2] as usize]);
                        let fnrm = scene.normals[scene.indices[t] as usize] + scene.normals[scene.indices[t + 1] as usize] + scene.normals[scene.indices[t + 2] as usize];
                        fnrm.dot(eye - (a + b + cc) / 3.0) < 0.0
                    };
                    let bit = 1u128 << u;
                    let diffuse_too = e.done & bit == 0 || ((e.back & bit != 0) != back);
                    if back {
                        e.back |= bit;
                    } else {
                        e.back &= !bit;
                    }
                    let side = if back { -1.0 } else { 1.0 };
                    if level == 0 {
                        let v = c.first_vertex as usize + u;
                        let (d, s, k) = shade_point(scene.positions[v], scene.normals[v] * side, scene.colors[v], eye, sky, lights, &in_reach, &reach, &shadows);
                        if diffuse_too {
                            e.diffuse[off] = d;
                        }
                        e.spec[off] = s;
                        n_evals += k;
                        n_shaded += 1;
                    } else {
                        let t = (c.first_tri as usize + u) * 3;
                        let (i0, i1, i2) = (scene.indices[t] as usize, scene.indices[t + 1] as usize, scene.indices[t + 2] as usize);
                        let n = 1usize << level;
                        let inv = 1.0 / n as f32;
                        let mut k = 0;
                        for i in 0..=n {
                            for j in 0..=(n - i) {
                                let (b1, b2) = (i as f32 * inv, j as f32 * inv);
                                let b0 = 1.0 - b1 - b2;
                                let p = scene.positions[i0] * b0 + scene.positions[i1] * b1 + scene.positions[i2] * b2;
                                let nn = (scene.normals[i0] * b0 + scene.normals[i1] * b1 + scene.normals[i2] * b2).normalize_or(Vec3::Y) * side;
                                let a = scene.colors[i0] * b0 + scene.colors[i1] * b1 + scene.colors[i2] * b2;
                                let (d, s, ev) = shade_point(p, nn, a, eye, sky, lights, &in_reach, &reach, &shadows);
                                if diffuse_too {
                                    e.diffuse[off + k] = d;
                                }
                                e.spec[off + k] = s;
                                n_evals += ev;
                                k += 1;
                            }
                        }
                        n_shaded += per as u64;
                    }
                }
                e.done |= todo;
                e.spec_done |= todo;
                evals.fetch_add(n_evals, Ordering::Relaxed);
                shaded.fetch_add(n_shaded, Ordering::Relaxed);
            });
        }
        frame.stats.shade_ms = t_shade.elapsed().as_secs_f32() * 1000.0;
        frame.stats.points_shaded = shaded.into_inner();
        frame.stats.points_used = used.into_inner();
        frame.stats.clusters_shaded = clusters_shaded.into_inner() as usize;
        frame.stats.light_evals = evals.into_inner();
        frame.stats.light_evals_all = frame.stats.points_shaded * lights.len() as u64;

        // Resolve: each pixel interpolates the shaded points under it.
        let t_res = std::time::Instant::now();
        let (width, height) = (frame.width, frame.height);
        let inv_view = cam.view.inverse();
        let exposure = settings.exposure;
        let view = settings.view;
        if self.base.len() != width * height || self.base_view != Some(view) {
            self.base = vec![0; width * height];
            full_resolve = true;
        }
        self.base_view = Some(view);
        frame.stats.full_resolve = full_resolve;
        let cache = &self.cache;
        let tri_id = &frame.tri_id;
        let setups = &frame.setups;
        self.base.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
            for x in 0..width {
                let id = tri_id[y * width + x];
                if !full_resolve && (id == NO_TRI || cache[setups[id as usize].cluster as usize].stamp != frame_no) {
                    // Nothing under this pixel changed: keep it.
                    continue;
                }
                if id == NO_TRI {
                    row[x] = match view {
                        View::Shaded => raster::pack(raster::tonemap(sky.radiance(raster::sky_dir(cam, width, height, x, y, &inv_view)) * exposure)),
                        View::LightCount => raster::pack(Vec3::splat(0.02)),
                    };
                    continue;
                }
                let s = &setups[id as usize];
                let c = &scene.clusters[s.cluster as usize];
                if view == View::LightCount {
                    let k = (0..lights.len()).filter(|&i| lights[i].position.distance(c.centre) <= c.radius + reach[i]).count();
                    row[x] = raster::pack(raster::heat(k as u32));
                    continue;
                }
                let b = raster::barycentrics(s, x as f32 + 0.5, y as f32 + 0.5);
                let w = [
                    b[0] * s.bary[0][0] + b[1] * s.bary[1][0] + b[2] * s.bary[2][0],
                    b[0] * s.bary[0][1] + b[1] * s.bary[1][1] + b[2] * s.bary[2][1],
                    b[0] * s.bary[0][2] + b[1] * s.bary[1][2] + b[2] * s.bary[2][2],
                ];
                let e = &cache[s.cluster as usize];
                let radiance = lookup(scene, c, e, s.tri, w);
                row[x] = raster::pack(raster::tonemap(radiance * exposure));
            }
        });
        frame.color.copy_from_slice(&self.base);
        frame.stats.resolve_ms = t_res.elapsed().as_secs_f32() * 1000.0;
    }
}

/// Vertices used by the triangles in `tris`, as a mask over the cluster's
/// vertices.
fn vertex_mask(scene: &Scene, c: &crate::scene::Cluster, tris: u128) -> u128 {
    let mut m = 0u128;
    let mut bits = tris;
    while bits != 0 {
        let t = bits.trailing_zeros();
        bits &= bits - 1;
        let i = (c.first_tri + t) as usize * 3;
        for k in 0..3 {
            m |= 1u128 << (scene.indices[i + k] - c.first_vertex);
        }
    }
    m
}

/// Radiance at a point inside triangle `tri` with weights `w` over its
/// vertices, from the kept shading.
fn lookup(scene: &Scene, c: &crate::scene::Cluster, e: &ClusterShade, tri: u32, w: [f32; 3]) -> Vec3 {
    let at = |i: usize| e.diffuse[i] + e.spec[i];
    let t = tri as usize * 3;
    if e.level == 0 {
        let mut r = Vec3::ZERO;
        for k in 0..3 {
            let u = (scene.indices[t + k] - c.first_vertex) as usize;
            let off = e.offset.get(u).copied().unwrap_or(u32::MAX);
            if off != u32::MAX {
                r += at(off as usize) * w[k];
            }
        }
        return r;
    }
    let u = (tri - c.first_tri) as usize;
    let off = e.offset.get(u).copied().unwrap_or(u32::MAX);
    if off == u32::MAX || e.level == NOT_VISIBLE {
        return Vec3::ZERO;
    }
    let off = off as usize;
    let n = 1usize << e.level;
    let nf = n as f32;
    let (u1, u2) = (w[1].clamp(0.0, 1.0) * nf, w[2].clamp(0.0, 1.0) * nf);
    let i = (u1.floor() as usize).min(n - 1);
    let j = (u2.floor() as usize).min(n - 1 - i);
    let (fi, fj) = (u1 - i as f32, u2 - j as f32);
    if fi + fj <= 1.0 || i + j + 1 >= n {
        // Lower micro-triangle (i, j), (i+1, j), (i, j+1).
        let (fi, fj) = if fi + fj > 1.0 {
            let s = fi + fj;
            (fi / s, fj / s)
        } else {
            (fi, fj)
        };
        at(off + grid_index(n, i, j)) * (1.0 - fi - fj) + at(off + grid_index(n, i + 1, j)) * fi + at(off + grid_index(n, i, j + 1)) * fj
    } else {
        // Upper micro-triangle (i+1, j+1), (i, j+1), (i+1, j).
        at(off + grid_index(n, i + 1, j + 1)) * (fi + fj - 1.0) + at(off + grid_index(n, i, j + 1)) * (1.0 - fi) + at(off + grid_index(n, i + 1, j)) * (1.0 - fj)
    }
}

/// Diffuse and highlight radiance (cd/m²) leaving a point towards `eye`,
/// and how many lights were evaluated. `n` is already turned to the eye's
/// side.
#[allow(clippy::too_many_arguments)]
fn shade_point(
    p: Vec3,
    n: Vec3,
    albedo: Vec3,
    eye: Vec3,
    sky: Sky,
    lights: &[PointLight],
    in_reach: &[usize],
    reach: &[f32],
    shadows: &[Option<&CubeShadow>],
) -> (Vec3, Vec3, u64) {
    let inv_pi = std::f32::consts::FRAC_1_PI;
    let mut diffuse = albedo * sky.irradiance(n) * inv_pi;
    let mut spec = Vec3::ZERO;
    let v = (eye - p).normalize_or(n);
    let mut evals = 0;
    for &i in in_reach {
        let l = &lights[i];
        let d = l.position - p;
        let d2 = d.length_squared().max(1e-12);
        if d2 > reach[i] * reach[i] {
            continue;
        }
        evals += 1;
        let ldir = d / d2.sqrt();
        let ndl = n.dot(ldir);
        if ndl <= 0.0 {
            continue;
        }
        let vis = match shadows[i] {
            Some(m) => m.lit(p, n),
            None => 1.0,
        };
        if vis <= 0.0 {
            continue;
        }
        let e = l.color * (l.candela() / d2 * ndl * vis);
        diffuse += e * albedo * inv_pi;
        let h = (ldir + v).normalize_or(ldir);
        spec += e * (n.dot(h).max(0.0).powf(48.0) * SPECULAR * inv_pi);
    }
    (diffuse, spec, evals)
}

/// One frame from nothing: a fresh renderer.
#[cfg(test)]
pub fn render(scene: &Scene, cam: &Camera, lights: &[PointLight], settings: &RenderSettings, frame: &mut Frame) {
    Renderer::new().render(scene, cam, lights, settings, frame);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_index_walks_every_point_once() {
        for level in 0..=5u8 {
            let n = 1usize << level;
            let mut seen = vec![false; grid_points(level)];
            let mut k = 0;
            for i in 0..=n {
                for j in 0..=(n - i) {
                    let g = grid_index(n, i, j);
                    assert_eq!(g, k, "level {level} ({i}, {j})");
                    assert!(!seen[g]);
                    seen[g] = true;
                    k += 1;
                }
            }
            assert!(seen.iter().all(|s| *s));
        }
    }

    use crate::raster::Sky;

    fn cam_at(eye_yaw: f32, w: usize, h: usize, target: Vec3, dist: f32, pitch: f32) -> Camera {
        let c = crate::camera::OrbitCamera { target, distance: dist, yaw: eye_yaw, pitch, fov_y: 45f32.to_radians(), scene_radius: 1.0 };
        Camera { view: c.view(), proj: c.proj(w as f32 / h as f32), eye: c.eye(), near: c.near() }
    }

    fn world() -> (Scene, Vec<PointLight>) {
        let mut s = crate::world::build();
        let mut lights = crate::world::table_lights();
        lights.extend(std::mem::take(&mut s.lights));
        (s, lights)
    }

    fn worst(a: &Frame, b: &Frame) -> i32 {
        let mut worst = 0;
        for (x, y) in a.color.iter().zip(&b.color) {
            for sh in [0, 8, 16] {
                worst = worst.max((((x >> sh) & 0xff) as i32 - ((y >> sh) & 0xff) as i32).abs());
            }
        }
        worst
    }

    /// Dragging a light: visibility is kept, only clusters in its reach are
    /// shaded again, and the picture is the one a fresh render gives.
    #[test]
    fn moving_a_light_reshades_only_its_reach() {
        let (scene, mut lights) = world();
        let (w, h) = (240, 135);
        let cam = cam_at(0.0, w, h, Vec3::new(0.0, 0.85, 0.0), 6.0, 0.07);
        let settings = RenderSettings::new(Sky { lux: 30.0, on: true });
        let mut r = Renderer::new();
        let mut frame = Frame::new(w, h);
        r.render(&scene, &cam, &lights, &settings, &mut frame);
        let first = frame.stats;
        assert!(!first.visibility_kept);
        // Move the table lamp 20 cm.
        lights[0].position += Vec3::new(0.2, 0.0, 0.0);
        r.render(&scene, &cam, &lights, &settings, &mut frame);
        let s = frame.stats;
        assert!(s.visibility_kept);
        assert!(!s.full_resolve);
        assert_eq!(s.shadow_maps_built, 1);
        assert!(s.clusters_shaded > 0 && s.clusters_shaded * 3 < first.clusters_shaded, "{} of {}", s.clusters_shaded, first.clusters_shaded);
        let mut fresh = Frame::new(w, h);
        render(&scene, &cam, &lights, &settings, &mut fresh);
        assert_eq!(worst(&frame, &fresh), 0);
        // Nothing changed: nothing is shaded.
        r.render(&scene, &cam, &lights, &settings, &mut frame);
        assert_eq!(frame.stats.points_shaded, 0);
        assert_eq!(frame.stats.shadow_maps_built, 0);
    }

    /// Orbiting the camera keeps diffuse shading where the dicing level
    /// holds; the picture stays close to a fresh render.
    #[test]
    fn moving_the_camera_keeps_what_it_can() {
        let (scene, lights) = world();
        let (w, h) = (240, 135);
        let settings = RenderSettings::new(Sky { lux: 30.0, on: true });
        let mut r = Renderer::new();
        let mut frame = Frame::new(w, h);
        r.render(&scene, &cam_at(0.0, w, h, Vec3::new(0.0, 0.85, 0.0), 6.0, 0.07), &lights, &settings, &mut frame);
        let cam2 = cam_at(0.02, w, h, Vec3::new(0.0, 0.85, 0.0), 6.0, 0.07);
        r.render(&scene, &cam2, &lights, &settings, &mut frame);
        assert!(!frame.stats.visibility_kept);
        assert_eq!(frame.stats.shadow_maps_built, 0, "shadow maps do not depend on the camera");
        let mut fresh = Frame::new(w, h);
        render(&scene, &cam2, &lights, &settings, &mut fresh);
        assert!(worst(&frame, &fresh) <= 6, "{}", worst(&frame, &fresh));
    }

    /// A card between a light and the floor darkens the floor under it.
    #[test]
    fn the_renderer_draws_shadows() {
        let mut scene = Scene::new();
        let quad = |c: Vec3, h: f32| [c + Vec3::new(-h, 0.0, h), c + Vec3::new(h, 0.0, h), c + Vec3::new(h, 0.0, -h), c + Vec3::new(-h, 0.0, -h)];
        scene.push_mesh(&quad(Vec3::ZERO, 2.0), &[], &[Vec3::splat(0.5)], &[0, 1, 2, 0, 2, 3]);
        scene.push_mesh(&quad(Vec3::new(0.0, 0.6, 0.0), 0.25), &[], &[Vec3::splat(0.5)], &[0, 1, 2, 0, 2, 3]);
        let lights = [PointLight { position: Vec3::new(0.0, 1.6, 0.0), color: Vec3::ONE, lumens: 2000.0 }];
        let (w, h) = (160, 120);
        // Looking down from the side, so the floor under the card shows.
        let cam = cam_at(0.0, w, h, Vec3::ZERO, 3.0, 1.0);
        let on = RenderSettings::new(Sky { lux: 1.0, on: false });
        let mut a = Frame::new(w, h);
        render(&scene, &cam, &lights, &RenderSettings { exposure: 0.01, ..on }, &mut a);
        let mut b = Frame::new(w, h);
        render(&scene, &cam, &lights, &RenderSettings { exposure: 0.01, shadows: false, ..RenderSettings::new(Sky { lux: 1.0, on: false }) }, &mut b);
        // Pixels where the floor shows in both and differs: under the card's shadow.
        let darker = a.color.iter().zip(&b.color).filter(|(x, y)| (**x & 0xff) + 20 < (**y & 0xff)).count();
        assert!(darker > 50, "{darker} pixels darker with shadows");
        let brighter = a.color.iter().zip(&b.color).filter(|(x, y)| (**x & 0xff) > (**y & 0xff)).count();
        assert_eq!(brighter, 0);
    }
}
