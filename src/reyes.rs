//! Shading the REYES way, kept between frames.
//!
//! Each visible triangle is diced into micropolygons from its own size on
//! screen, not its cluster's or its mesh's: each edge gets a level from its
//! length in pixels (`raster::dice_level`), and the triangle's grid takes
//! the finest of its three. Level L cuts each edge into 2^L parts. The grid
//! is even in the triangle's screen weights, so its points are evenly
//! spaced on screen however the triangle recedes. Points on an edge between
//! that edge's own samples are interpolated along it, so two triangles
//! sharing an edge agree along it. A triangle whose edges are all under
//! the shading rate (`RenderSettings::shading_rate_px`) uses the cluster's
//! shaded vertices.
//!
//! Shading happens on those points, in world space, before the picture is
//! made, and the results are kept (Cook, Carpenter, Catmull 1987). The
//! picture is then resolved by interpolating the shaded points under each
//! pixel.
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

use crate::bounce::{self, Bounce};
use crate::raster::{self, Camera, Frame, NO_TRI, NOT_VISIBLE, RenderSettings, SPECULAR, Sky, View};
use crate::scene::{PointLight, Scene};
use crate::shadow::CubeShadow;
use crate::sun::SunShadow;

/// How one triangle was diced, and where its points are kept.
///
/// Its grid lies on the triangle as seen when it was diced: rows along its
/// longest edge on screen, 2^lu + 1 points each, and 2^lv + 1 rows up to the
/// opposite corner, so points are about the shading rate apart
/// both ways on screen however thin or receding the triangle is. The grid
/// covers the rectangle on that edge, twice the triangle; points outside it
/// are shaded on the triangle's plane, for interpolating up to its edges.
#[derive(Clone, Copy)]
struct TriDice {
    /// The finer of lu and lv; 0: the triangle reads the cluster's
    /// vertices; `NOT_VISIBLE`: not diced.
    level: u8,
    lu: u8,
    lv: u8,
    /// The longest edge on screen when diced (from corner `base` to the
    /// next).
    base: u8,
    /// Depth (clip w) and position on screen (pixels from the centre) of
    /// each corner when diced.
    w: [f32; 3],
    px: [glam::Vec2; 3],
    /// Where its points start in `diffuse` and `spec`, and how many fit.
    offset: u32,
    cap: u32,
}

impl Default for TriDice {
    fn default() -> Self {
        Self { level: NOT_VISIBLE, lu: 0, lv: 0, base: 0, w: [1.0; 3], px: [glam::Vec2::ZERO; 3], offset: 0, cap: 0 }
    }
}

impl TriDice {
    fn points(&self) -> usize {
        if self.level == 0 || self.level == NOT_VISIBLE { 0 } else { ((1usize << self.lu) + 1) * ((1usize << self.lv) + 1) }
    }

    /// The grid's origin and its two sides on screen.
    fn frame(&self) -> (glam::Vec2, glam::Vec2, glam::Vec2) {
        let k = self.base as usize;
        let (o, ex) = (self.px[k], self.px[(k + 1) % 3] - self.px[k]);
        let len = ex.length().max(1e-6);
        let up = ex.perp() / len;
        let h = (self.px[(k + 2) % 3] - o).dot(up);
        let ey = if h.abs() < 1e-6 { up * 1e-6 } else { up * h };
        (o, ex, ey)
    }

    /// Screen weights of grid point (u, v), u and v from 0 to 1.
    fn grid_to_screen(&self, u: f32, v: f32) -> [f32; 3] {
        let (o, ex, ey) = self.frame();
        let q = o + ex * u + ey * v - self.px[0];
        let (e1, e2) = (self.px[1] - self.px[0], self.px[2] - self.px[0]);
        let det = e1.perp_dot(e2);
        let det = if det.abs() < 1e-12 { 1e-12 } else { det };
        let (s1, s2) = (q.perp_dot(e2) / det, e1.perp_dot(q) / det);
        [1.0 - s1 - s2, s1, s2]
    }

    /// Grid position (u, v), each 0 to 1, of screen weights `s`.
    fn screen_to_grid(&self, s: [f32; 3]) -> (f32, f32) {
        let (o, ex, ey) = self.frame();
        let q = self.px[0] * s[0] + self.px[1] * s[1] + self.px[2] * s[2] - o;
        ((q.dot(ex) / ex.length_squared()).clamp(0.0, 1.0), (q.dot(ey) / ey.length_squared()).clamp(0.0, 1.0))
    }
}

/// A triangle diced for the view `vp` on a `size` pixel frame. Corners
/// behind the near plane are taken at it.
fn dice_triangle(vp: Mat4, p: [Vec3; 3], near: f32, size: (usize, usize), rate: f32) -> TriDice {
    let (hw, hh) = (size.0 as f32 * 0.5, size.1 as f32 * 0.5);
    let c = p.map(|q| vp * q.extend(1.0));
    let w = c.map(|q| q.w.max(near));
    let px = [0, 1, 2].map(|k| glam::Vec2::new(c[k].x / w[k] * hw, c[k].y / w[k] * hh));
    let base = (0..3).max_by(|&a, &b| px[a].distance_squared(px[(a + 1) % 3]).total_cmp(&px[b].distance_squared(px[(b + 1) % 3]))).unwrap_or(0);
    let mut d = TriDice { base: base as u8, w, px, ..Default::default() };
    let (_, ex, ey) = d.frame();
    d.lu = raster::dice_level_at(ex.length(), rate);
    d.lv = raster::dice_level_at(ey.length(), rate);
    d.level = d.lu.max(d.lv);
    d
}

/// Widest spacing on screen, under view `vp`, of the grid `d` keeps, from
/// its steps at the triangle's corners; `None` when a corner is behind the
/// near plane.
fn kept_spacing(d: &TriDice, p: [Vec3; 3], vp: Mat4, near: f32, size: (usize, usize)) -> Option<f32> {
    let (hw, hh) = (size.0 as f32 * 0.5, size.1 as f32 * 0.5);
    let (du, dv) = (1.0 / (1u32 << d.lu) as f32, 1.0 / (1u32 << d.lv) as f32);
    let to_px = |u: f32, v: f32| -> Option<glam::Vec2> {
        let b = screen_to_tri(d.grid_to_screen(u, v), d.w);
        let q = vp * (p[0] * b[0] + p[1] * b[1] + p[2] * b[2]).extend(1.0);
        (q.w >= near).then(|| glam::Vec2::new(q.x / q.w * hw, q.y / q.w * hh))
    };
    let mut widest = 0f32;
    for k in 0..3 {
        let mut s = [0.0; 3];
        s[k] = 1.0;
        let (u, v) = d.screen_to_grid(s);
        let (u2, v2) = (if u > 0.5 { u - du } else { u + du }, if v > 0.5 { v - dv } else { v + dv });
        let a = to_px(u, v)?;
        widest = widest.max(a.distance(to_px(u2, v)?)).max(a.distance(to_px(u, v2)?));
    }
    Some(widest)
}

/// A triangle's own weights from its screen weights `s`, for corner
/// depths `w` (perspective-correct interpolation).
#[inline]
fn screen_to_tri(s: [f32; 3], w: [f32; 3]) -> [f32; 3] {
    let b = [s[0] / w[0], s[1] / w[1], s[2] / w[2]];
    let t = b[0] + b[1] + b[2];
    let t = if t.abs() < 1e-30 { 1e-30 } else { t };
    [b[0] / t, b[1] / t, b[2] / t]
}

/// Screen weights from a triangle's own weights `b`.
#[inline]
fn tri_to_screen(b: [f32; 3], w: [f32; 3]) -> [f32; 3] {
    let s = [b[0] * w[0], b[1] * w[1], b[2] * w[2]];
    let t = 1.0 / (s[0] + s[1] + s[2]).max(1e-30);
    [s[0] * t, s[1] * t, s[2] * t]
}

/// Shading kept for one cluster.
#[derive(Default)]
struct ClusterShade {
    /// Finest level kept for any triangle; `NOT_VISIBLE` when none is.
    level: u8,
    /// Per triangle: `UNDICED`, `ON_VERTICES`, or which of `grids` it has.
    state: Vec<u8>,
    /// Grids of triangles diced above level 0; an entry at `NOT_VISIBLE` is
    /// free, its points kept for the next triangle that fits.
    grids: Vec<TriDice>,
    /// Grid points of triangles diced above level 0.
    diffuse: Vec<Vec3>,
    spec: Vec<Vec3>,
    /// Grid points in use, against `diffuse.len()`: the rest is garbage
    /// left by triangles diced again at a larger size.
    live: usize,
    /// Grid triangles whose diffuse light is current.
    done: u128,
    /// Grid triangles whose highlight is current for `spec_eye`.
    spec_done: u128,
    /// Grid triangles seen from their back when last shaded. Shading is
    /// two-sided: normals are turned towards the eye, so a triangle seen
    /// from the other side has its diffuse light redone.
    back: u128,
    /// The cluster's vertices, shaded for triangles at level 0.
    vdiffuse: Vec<Vec3>,
    vspec: Vec<Vec3>,
    vdone: u128,
    vspec_done: u128,
    vback: u128,
    spec_eye: Vec3,
    /// The frame in which anything here was last shaded.
    stamp: u64,
    /// The frame in which it was last on screen.
    seen: u64,
}

/// A triangle not diced yet.
const UNDICED: u8 = u8::MAX;
/// A triangle under two pixels, read from the cluster's vertices.
const ON_VERTICES: u8 = u8::MAX - 1;

impl ClusterShade {
    fn empty() -> Self {
        Self { level: NOT_VISIBLE, ..Default::default() }
    }

    /// Bytes held.
    fn bytes(&self) -> usize {
        (self.diffuse.capacity() + self.spec.capacity() + self.vdiffuse.capacity() + self.vspec.capacity()) * 12 + self.state.capacity() + self.grids.capacity() * std::mem::size_of::<TriDice>()
    }

    /// Dicing level of triangle `u` (`NOT_VISIBLE`: not diced).
    fn tri_level(&self, u: usize) -> u8 {
        match self.state.get(u).copied().unwrap_or(UNDICED) {
            UNDICED => NOT_VISIBLE,
            ON_VERTICES => 0,
            i => self.grids[i as usize].level,
        }
    }

    /// Grid of triangle `u`, when it has one.
    fn grid(&self, u: usize) -> Option<&TriDice> {
        match self.state.get(u).copied().unwrap_or(UNDICED) {
            UNDICED | ON_VERTICES => None,
            i => Some(&self.grids[i as usize]),
        }
    }

    fn init(&mut self, tris: usize, verts: usize) {
        self.state = vec![UNDICED; tris];
        self.vdiffuse = vec![Vec3::ZERO; verts];
        self.vspec = vec![Vec3::ZERO; verts];
        self.forget_grids();
        self.vdone = 0;
        self.vspec_done = 0;
        self.vback = 0;
    }

    /// Light changed: everything kept is shaded again, grids kept.
    fn forget_light(&mut self) {
        self.done = 0;
        self.spec_done = 0;
        self.vdone = 0;
        self.vspec_done = 0;
    }

    /// Drops every grid and its memory, keeping the vertices.
    fn forget_grids(&mut self) {
        for st in &mut self.state {
            if *st != ON_VERTICES {
                *st = UNDICED;
            }
        }
        self.grids = Vec::new();
        self.diffuse = Vec::new();
        self.spec = Vec::new();
        self.live = 0;
        self.done = 0;
        self.spec_done = 0;
        self.back = 0;
    }
}

#[derive(PartialEq, Clone, Copy)]
struct VisKey {
    tile: usize,
    view: Mat4,
    proj: Mat4,
    size: (usize, usize),
    cull: bool,
    wireframe: bool,
    lod_px: u32,
}

static NEXT_STAMP: AtomicU64 = AtomicU64::new(1);

/// Everything kept from one frame to the next.
pub struct Renderer {
    scene_key: (usize, usize, usize),
    cache: Vec<ClusterShade>,
    shadows: Vec<Option<CubeShadow>>,
    prev_lights: Vec<PointLight>,
    prev_reach: Vec<f32>,
    prev_global: Option<(Sky, raster::Sun, f32, bool, bool, bool, usize)>,
    /// The sun's shadow map, and what it was built for (direction, texels,
    /// scene).
    sun_shadow: Option<SunShadow>,
    sun_key: Option<(Vec3, usize, (usize, usize, usize))>,
    prev_backdrop: Option<raster::Backdrop>,
    /// The surfel cloud and the bounce light at each receiver.
    bounce: Option<Bounce>,
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
    /// The shading rate is multiplied by this to fit the cache budget.
    rate_scale: f32,
}

#[cfg_attr(not(target_pointer_width = "64"), allow(dead_code))]
/// A read-only look at what the renderer keeps, for the visual debug views.
pub struct Peek<'a> {
    pub frame_no: u64,
    cache: &'a [ClusterShade],
    pub shadows: &'a [Option<CubeShadow>],
    pub bounce: Option<&'a Bounce>,
    pub sun_shadow: Option<&'a SunShadow>,
}

#[cfg_attr(not(target_pointer_width = "64"), allow(dead_code))]
impl Peek<'_> {
    /// Dicing level of the shading kept for cluster `c` (`NOT_VISIBLE`: none).
    pub fn kept_level(&self, c: usize) -> u8 {
        self.cache.get(c).map(|e| e.level).unwrap_or(NOT_VISIBLE)
    }

    /// Dicing level kept for triangle `tri` of cluster `c`.
    pub fn tri_level(&self, c: usize, tri: u32, scene: &Scene) -> u8 {
        self.cache.get(c).map(|e| e.tri_level((tri - scene.clusters[c].first_tri) as usize)).unwrap_or(NOT_VISIBLE)
    }

    /// Where weights `w` of triangle `tri` of cluster `c` fall in its kept
    /// grid, in grid steps, and the grid's level. `None` at level 0 or when
    /// not diced.
    pub fn grid_at(&self, c: usize, tri: u32, scene: &Scene, w: [f32; 3]) -> Option<(f32, f32, u8)> {
        let d = self.tri(c, tri, scene)?;
        if d.level == 0 || d.level == NOT_VISIBLE {
            return None;
        }
        let (u, v) = d.screen_to_grid(tri_to_screen(w, d.w));
        Some((u * (1u32 << d.lu) as f32, v * (1u32 << d.lv) as f32, d.level))
    }

    fn tri(&self, c: usize, tri: u32, scene: &Scene) -> Option<&TriDice> {
        self.cache.get(c)?.grid((tri - scene.clusters[c].first_tri) as usize)
    }

    /// The frame in which cluster `c` was last shaded.
    pub fn shaded_in(&self, c: usize) -> u64 {
        self.cache.get(c).map(|e| e.stamp).unwrap_or(0)
    }

    /// Kept radiance of triangle `tri` of cluster `c` at weights `w`.
    pub fn radiance(&self, scene: &Scene, c: usize, tri: u32, w: [f32; 3], part: Part) -> Vec3 {
        match self.cache.get(c) {
            Some(e) => lookup_part(scene, &scene.clusters[c], e, tri, w, part),
            None => Vec3::ZERO,
        }
    }
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
            sun_shadow: None,
            sun_key: None,
            prev_backdrop: None,
            bounce: None,
            vis_key: None,
            vis_stamp: 0,
            levels: Vec::new(),
            need: Vec::new(),
            visible: Vec::new(),
            base: Vec::new(),
            base_view: None,
            frame_no: 0,
            rate_scale: 1.0,
        }
    }

    /// For tests: in which frame each cluster's bounce light was last
    /// gathered.
    #[cfg(test)]
    pub fn bounce_stamps(&self) -> Option<&[u64]> {
        self.bounce.as_ref().map(|b| &b.stamp[..])
    }

    #[cfg_attr(not(target_pointer_width = "64"), allow(dead_code))]
    pub fn peek(&self) -> Peek<'_> {
        Peek { frame_no: self.frame_no, cache: &self.cache, shadows: &self.shadows, bounce: self.bounce.as_ref(), sun_shadow: self.sun_shadow.as_ref() }
    }

    /// Memory held by the shading cache and the shadow maps, in bytes.
    /// Clusters `moved` were moved (an object's geometry rewritten): their
    /// shading is dropped, everything else is lit again (its shadows may
    /// have changed), and the shadow maps and visibility are made again.
    /// Bounce light is made again too when `bounce`; while an object is
    /// being dragged it is left, and redone when the drag ends.
    pub fn moved(&mut self, moved: std::ops::Range<usize>, bounce: bool) {
        for c in moved {
            if let Some(e) = self.cache.get_mut(c) {
                *e = ClusterShade::empty();
            }
        }
        self.prev_global = None;
        self.sun_key = None;
        self.vis_key = None;
        if bounce {
            self.bounce = None;
        }
    }

    pub fn memory(&self) -> usize {
        let shade: usize = self.cache.iter().map(|c| c.bytes()).sum();
        let shadows: usize = self.shadows.iter().flatten().map(|s| s.bytes()).sum();
        shade + shadows + self.sun_shadow.as_ref().map(|s| s.bytes()).unwrap_or(0)
    }

    pub fn render(&mut self, scene: &Scene, cam: &Camera, lights: &[PointLight], settings: &RenderSettings, frame: &mut Frame) {
        let n_lights = lights.len().min(64);
        let lights = &lights[..n_lights];
        let key = (scene.positions.as_ptr() as usize, scene.positions.len(), scene.clusters.len());
        if key != self.scene_key {
            *self = Self::new();
            self.scene_key = key;
            self.cache = (0..scene.clusters.len()).map(|_| ClusterShade::empty()).collect();
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
        let global = (settings.sky, settings.sun, settings.exposure, settings.light_reach, settings.shadows && settings.shadow_res > 0, settings.bounce, settings.shadow_res);
        let global_changed = self.prev_global != Some(global) || self.prev_lights.len() != lights.len();
        // Spheres (old and new position, with reach) of lights that changed.
        let mut spheres: Vec<(Vec3, f32)> = Vec::new();
        if global_changed {
            full_resolve = true;
            for c in &mut self.cache {
                c.forget_light();
            }
            self.shadows = (0..lights.len()).map(|_| None).collect();
        } else {
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
                        e.forget_light();
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
                    (i, CubeShadow::build(scene, lights[i].position, r, settings.shadow_res))
                })
                .collect();
            frame.stats.shadow_maps_built = built.len();
            for (i, m) in built {
                frame.stats.shadow_tris += m.tris;
                self.shadows[i] = Some(m);
            }
        }
        // The sun's map, when it is up and its direction or the scene changed.
        if settings.shadows && settings.sun.up() {
            let res = (settings.shadow_res * 4).clamp(256, 2048);
            let key = (settings.sun.dir, res, key);
            if self.sun_key != Some(key) {
                let r = scene.focus_radius();
                let m = SunShadow::build(scene, settings.sun.dir, scene.focus_centre(), &[r * 1.6, 600.0, 25_000.0], res);
                frame.stats.shadow_maps_built += 1;
                frame.stats.shadow_tris += m.tris;
                self.sun_shadow = Some(m);
                self.sun_key = Some(key);
            }
        }
        frame.stats.shadow_ms = ts.elapsed().as_secs_f32() * 1000.0;
        let sun = SunLit {
            lux: settings.sun.lux(settings.exposure),
            dir: settings.sun.dir,
            shadow: if settings.shadows && settings.sun.up() { self.sun_shadow.as_ref() } else { None },
        };
        let sun = &sun;

        // Bounce: light the surfels again where light changed, then throw
        // away the receivers that bounce could reach from there.
        let tb = std::time::Instant::now();
        let shadows: Vec<Option<&CubeShadow>> = (0..lights.len()).map(|i| if settings.shadows { self.shadows[i].as_ref() } else { None }).collect();
        let sky = settings.sky;
        if settings.bounce {
            let b = self.bounce.get_or_insert_with(|| Bounce::new(scene));
            let all = global_changed;
            if all || !spheres.is_empty() {
                let lit = AtomicU64::new(0);
                // The brightest change, as an intensity (cd): how far the
                // change in bounce light can be seen.
                let changed = std::sync::Mutex::new(0f32);
                b.radiance.par_iter_mut().zip(b.surfels.par_iter()).for_each(|(r, sf)| {
                    if !all && !spheres.iter().any(|(p, rr)| p.distance(sf.position) <= rr + (sf.area / std::f32::consts::PI).sqrt()) {
                        return;
                    }
                    let in_reach: Vec<usize> = (0..lights.len()).filter(|&i| lights[i].position.distance(sf.position) <= reach[i]).collect();
                    // Direct light only, without the sky's ground term: the
                    // ground is in the cloud.
                    let new = shade_point(sf.position, sf.normal, sf.albedo, sf.position + sf.normal, sky, false, lights, &in_reach, &reach, &shadows, sun).0;
                    let delta = (new - *r).abs().max_element() * sf.area;
                    *r = new;
                    lit.fetch_add(1, Ordering::Relaxed);
                    if delta > 0.0 {
                        let mut m = changed.lock().unwrap();
                        *m = m.max(delta);
                    }
                });
                frame.stats.surfels_lit = lit.into_inner();
                b.refit();
                let bounce_reach = (changed.into_inner().unwrap() / cut.max(1e-12)).sqrt();
                if all {
                    b.valid.iter_mut().for_each(|v| *v = false);
                } else {
                    for (ci, c) in scene.clusters.iter().enumerate() {
                        if spheres.iter().any(|(p, r)| p.distance(c.centre) <= c.radius + r + bounce_reach) {
                            b.valid[ci] = false;
                        }
                    }
                }
            }
        }
        frame.stats.bounce_ms = tb.elapsed().as_secs_f32() * 1000.0;

        // Visibility, unless the camera and the frame are as they were.
        let vk = VisKey { tile: settings.tile, view: cam.view, proj: cam.proj, size: (frame.width, frame.height), cull: settings.cull_backfaces, wireframe: settings.wireframe, lod_px: settings.lod_px().to_bits() };
        if self.vis_key == Some(vk) && frame.vis_stamp == self.vis_stamp && self.vis_stamp != 0 {
            frame.stats.visibility_kept = true;
            frame.stats.clusters_in = scene.clusters.len();
            frame.stats.clusters_drawn = self.levels.iter().filter(|l| **l != NOT_VISIBLE).count();
            frame.stats.tris_in = scene.triangle_count();
            frame.stats.tris_drawn = frame.setups.len();
        } else {
            full_resolve = true;
            self.levels = raster::visibility(scene, cam, settings.cull_backfaces, settings.wireframe, settings.tile, settings.lod_px(), frame);
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
        let eye = cam.eye;
        let need = &self.need;
        let ground = !settings.bounce;
        let vp = cam.proj * cam.view;
        let (near, size) = (cam.near, (frame.width, frame.height));
        let rate = settings.shading_rate_px.max(0.25) * self.rate_scale;
        if settings.view == View::Shaded {
            self.cache.par_iter_mut().enumerate().for_each(|(ci, e)| {
                let need_tris = need[ci];
                if need_tris == 0 {
                    return;
                }
                let c = &scene.clusters[ci];
                let fv = c.first_vertex as usize;
                e.seen = frame_no;
                if e.state.len() != c.tri_count as usize {
                    e.init(c.tri_count as usize, c.vertex_count as usize);
                }
                // Too much left by grids that moved to larger places:
                // start the grids over.
                if e.diffuse.len() > 2 * e.live + 4096 {
                    e.forget_grids();
                }
                if e.spec_eye != eye {
                    e.spec_done = 0;
                    e.vspec_done = 0;
                    e.spec_eye = eye;
                }
                // Dice each needed triangle from its own size on screen.
                // A kept grid stays while it is at most one level finer,
                // edge by edge, and its spacing on screen within a factor
                // of two.
                let (mut vneed, mut tneed, mut n_used) = (0u128, 0u128, 0u64);
                let mut bits = need_tris;
                while bits != 0 {
                    let u = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    let t = (c.first_tri as usize + u) * 3;
                    let idx = [scene.indices[t] as usize, scene.indices[t + 1] as usize, scene.indices[t + 2] as usize];
                    let corners = idx.map(|i| scene.positions[i]);
                    let cur = match e.state[u] {
                        UNDICED | ON_VERTICES => None,
                        i => Some(i as usize),
                    };
                    // A kept grid stays while its points are no further
                    // apart on screen than the rate and no more than a
                    // level finer than a fresh one.
                    let keep = cur.is_some_and(|i| kept_spacing(&e.grids[i], corners, vp, near, size).is_some_and(|s| s <= rate * 1.05 && s > rate * 0.25));
                    if !keep {
                        let fresh = dice_triangle(vp, corners, near, size, rate);
                        if let Some(i) = cur {
                            e.live -= e.grids[i].points();
                            e.grids[i].level = NOT_VISIBLE;
                        }
                        if fresh.level == 0 {
                            e.state[u] = ON_VERTICES;
                            for i in idx {
                                vneed |= 1u128 << (i - fv);
                            }
                            continue;
                        }
                        // Its own entry, or a free one, keeping their points.
                        let i = cur.or_else(|| e.grids.iter().position(|g| g.level == NOT_VISIBLE)).unwrap_or_else(|| {
                            e.grids.push(TriDice::default());
                            e.grids.len() - 1
                        });
                        let pts = fresh.points();
                        let (mut offset, mut cap) = (e.grids[i].offset, e.grids[i].cap);
                        if (cap as usize) < pts {
                            offset = e.diffuse.len() as u32;
                            cap = pts as u32;
                            e.diffuse.resize(e.diffuse.len() + pts, Vec3::ZERO);
                            e.spec.resize(e.spec.len() + pts, Vec3::ZERO);
                        }
                        e.grids[i] = TriDice { offset, cap, ..fresh };
                        e.state[u] = i as u8;
                        e.live += pts;
                        e.done &= !(1u128 << u);
                        e.spec_done &= !(1u128 << u);
                    }
                    if e.state[u] == ON_VERTICES {
                        for i in idx {
                            vneed |= 1u128 << (i - fv);
                        }
                        continue;
                    }
                    tneed |= 1u128 << u;
                    n_used += e.grids[e.state[u] as usize].points() as u64;
                }
                n_used += vneed.count_ones() as u64;
                used.fetch_add(n_used, Ordering::Relaxed);
                e.level = (0..e.state.len()).map(|u| e.tri_level(u)).filter(|l| *l != NOT_VISIBLE).max().unwrap_or(NOT_VISIBLE);
                let vtodo = vneed & !(e.vdone & e.vspec_done);
                let ttodo = tneed & !(e.done & e.spec_done);
                if vtodo == 0 && ttodo == 0 {
                    return;
                }
                clusters_shaded.fetch_add(1, Ordering::Relaxed);
                e.stamp = frame_no;
                // Lights that reach this cluster.
                let in_reach: Vec<usize> = (0..lights.len()).filter(|&i| lights[i].position.distance(c.centre) <= c.radius + reach[i]).collect();
                let (mut n_shaded, mut n_evals) = (0u64, 0u64);

                // Vertices, for triangles at level 0.
                let mut bits = vtodo;
                while bits != 0 {
                    let u = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    let v = fv + u;
                    let back = scene.normals[v].dot(eye - scene.positions[v]) < 0.0;
                    let bit = 1u128 << u;
                    let diffuse_too = e.vdone & bit == 0 || ((e.vback & bit != 0) != back);
                    if back {
                        e.vback |= bit;
                    } else {
                        e.vback &= !bit;
                    }
                    let side = if back { -1.0 } else { 1.0 };
                    let (d, s, k) = shade_point(scene.positions[v], scene.normals[v] * side, scene.colors[v], eye, sky, ground, lights, &in_reach, &reach, &shadows, sun);
                    if diffuse_too {
                        e.vdiffuse[u] = d;
                    }
                    e.vspec[u] = s;
                    n_evals += k;
                    n_shaded += 1;
                }
                e.vdone |= vtodo;
                e.vspec_done |= vtodo;

                // Grids, even on screen. Texture tiles at the finest level
                // needed, read from the file if not in the cache; grids
                // finer than the finest tile sample it.
                let finest = {
                    let mut l = 0;
                    let mut bits = ttodo;
                    while bits != 0 {
                        l = l.max(e.tri_level(bits.trailing_zeros() as usize));
                        bits &= bits - 1;
                    }
                    l
                };
                let tiles = if ttodo != 0 { scene.textures.block(ci, c.tri_count as usize, finest.min(crate::texture::MAX_LEVEL)) } else { None };
                let mut bits = ttodo;
                while bits != 0 {
                    let u = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    let d = e.grids[e.state[u] as usize];
                    let t = (c.first_tri as usize + u) * 3;
                    let (i0, i1, i2) = (scene.indices[t] as usize, scene.indices[t + 1] as usize, scene.indices[t + 2] as usize);
                    let (p0, p1, p2) = (scene.positions[i0], scene.positions[i1], scene.positions[i2]);
                    let fnrm = scene.normals[i0] + scene.normals[i1] + scene.normals[i2];
                    let back = fnrm.dot(eye - (p0 + p1 + p2) / 3.0) < 0.0;
                    let bit = 1u128 << u;
                    let diffuse_too = e.done & bit == 0 || ((e.back & bit != 0) != back);
                    if back {
                        e.back |= bit;
                    } else {
                        e.back &= !bit;
                    }
                    let side = if back { -1.0 } else { 1.0 };
                    let (nu, nv) = (1usize << d.lu, 1usize << d.lv);
                    let mut g = d.offset as usize;
                    for a in 0..=nu {
                        for b in 0..=nv {
                            let [b0, b1, b2] = screen_to_tri(d.grid_to_screen(a as f32 / nu as f32, b as f32 / nv as f32), d.w);
                            let p = p0 * b0 + p1 * b1 + p2 * b2;
                            let nn = (scene.normals[i0] * b0 + scene.normals[i1] * b1 + scene.normals[i2] * b2).normalize_or(Vec3::Y) * side;
                            let alb = match &tiles {
                                Some(t) => t.sample(u, b1, b2),
                                None => (scene.colors[i0] * b0 + scene.colors[i1] * b1 + scene.colors[i2] * b2).max(Vec3::ZERO),
                            };
                            let (dd, s, ev) = shade_point(p, nn, alb, eye, sky, ground, lights, &in_reach, &reach, &shadows, sun);
                            if diffuse_too {
                                e.diffuse[g] = dd;
                            }
                            e.spec[g] = s;
                            g += 1;
                            n_evals += ev;
                            n_shaded += 1;
                        }
                    }
                }
                e.done |= ttodo;
                e.spec_done |= ttodo;
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

        // Gather bounce light at visible receivers that need it, nearest the
        // camera first, within the time budget.
        let tg = std::time::Instant::now();
        if settings.bounce && settings.view == View::Shaded {
            let b = self.bounce.as_mut().expect("made above");
            let bounce_reach = b.reach(cut);
            let mut todo: Vec<u32> = self.visible.iter().copied().filter(|&c| !b.valid[c as usize]).collect();
            todo.sort_by(|a, c| scene.clusters[*a as usize].centre.distance_squared(eye).total_cmp(&scene.clusters[*c as usize].centre.distance_squared(eye)));
            let mut done = 0;
            for batch in todo.chunks(32) {
                if let Some(ms) = settings.bounce_budget_ms {
                    if tg.elapsed().as_secs_f32() * 1000.0 > ms {
                        break;
                    }
                }
                let bref = &*b;
                let results: Vec<(u32, Vec<bounce::Sh>)> = batch.par_iter().map(|&c| (c, bref.gather_cluster(c as usize, bounce_reach))).collect();
                for (c, shs) in results {
                    let first = b.first_receiver[c as usize] as usize;
                    for (k, sh) in shs.into_iter().enumerate() {
                        b.sh[first + k] = sh;
                    }
                    b.valid[c as usize] = true;
                    b.stamp[c as usize] = frame_no;
                }
                done += batch.len();
            }
            frame.stats.receivers_updated = done;
            frame.stats.receivers_pending = todo.len() - done;
        }
        frame.stats.gather_ms = tg.elapsed().as_secs_f32() * 1000.0;

        // Resolve: each pixel interpolates the shaded points under it.
        let t_res = std::time::Instant::now();
        let (width, height) = (frame.width, frame.height);
        let inv_view = cam.view.inverse();
        let exposure = settings.exposure;
        let view = settings.view;
        if self.prev_backdrop != Some(settings.backdrop) {
            full_resolve = true;
            self.prev_backdrop = Some(settings.backdrop);
        }
        if self.base.len() != width * height || self.base_view != Some(view) {
            self.base = vec![0; width * height];
            full_resolve = true;
        }
        self.base_view = Some(view);
        frame.stats.full_resolve = full_resolve;
        let cache = &self.cache;
        let tri_id = &frame.tri_id;
        let setups = &frame.setups;
        let bounce = if settings.bounce { self.bounce.as_ref() } else { None };
        self.base.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
            for x in 0..width {
                let id = tri_id[y * width + x];
                let changed = |c: usize| cache[c].stamp == frame_no || bounce.is_some_and(|b| b.stamp[c] == frame_no);
                if !full_resolve && (id == NO_TRI || !changed(setups[id as usize].cluster as usize)) {
                    // Nothing under this pixel changed: keep it.
                    continue;
                }
                if id == NO_TRI {
                    row[x] = match view {
                        View::Shaded => raster::pack(raster::tonemap(raster::backdrop_radiance(&sky, &settings.backdrop, &settings.sun, exposure, raster::sky_dir(cam, width, height, x, y, &inv_view)) * exposure)),
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
                let mut radiance = lookup(scene, c, e, s.tri, w);
                if let Some(b) = bounce {
                    // Bounce light, from the cluster's spherical harmonics,
                    // with the normal turned to the eye.
                    let t = s.tri as usize * 3;
                    let (i0, i1, i2) = (scene.indices[t] as usize, scene.indices[t + 1] as usize, scene.indices[t + 2] as usize);
                    let p = scene.positions[i0] * w[0] + scene.positions[i1] * w[1] + scene.positions[i2] * w[2];
                    let mut n = (scene.normals[i0] * w[0] + scene.normals[i1] * w[1] + scene.normals[i2] * w[2]).normalize_or(Vec3::Y);
                    if n.dot(eye - p) < 0.0 {
                        n = -n;
                    }
                    let albedo = scene.colors[i0] * w[0] + scene.colors[i1] * w[1] + scene.colors[i2] * w[2];
                    radiance += albedo * b.irradiance_at(scene, s.cluster as usize, s.tri, w, n) * std::f32::consts::FRAC_1_PI;
                }
                row[x] = raster::pack(raster::tonemap(radiance * exposure));
            }
        });
        frame.color.copy_from_slice(&self.base);
        frame.stats.resolve_ms = t_res.elapsed().as_secs_f32() * 1000.0;
        self.fit_budget(settings.cache_budget, frame);
    }

    /// Keeps the shading cache within `budget` bytes. Clusters off screen
    /// go first, least recently seen first, down to three quarters of the
    /// budget. When what is on screen alone does not fit, the shading rate
    /// doubles (up to 16 times) and the grids on screen are dropped, to be
    /// diced coarser next frame; it halves again once the cache is under an
    /// eighth of the budget. Slower or coarser, never out of memory.
    fn fit_budget(&mut self, budget: usize, frame: &mut Frame) {
        let frame_no = self.frame_no;
        let mut total: usize = self.cache.par_iter().map(|e| e.bytes()).sum();
        if total > budget {
            let mut old: Vec<(u64, u32)> = self.cache.iter().enumerate().filter(|(_, e)| e.seen != frame_no && e.bytes() > 0).map(|(i, e)| (e.seen, i as u32)).collect();
            old.sort_unstable();
            for (_, i) in old {
                if total <= budget / 4 * 3 {
                    break;
                }
                let e = &mut self.cache[i as usize];
                total -= e.bytes();
                *e = ClusterShade::empty();
                frame.stats.clusters_evicted += 1;
            }
            if total > budget && self.rate_scale < 16.0 {
                self.rate_scale *= 2.0;
                for e in &mut self.cache {
                    if e.grids.capacity() > 0 {
                        total -= e.bytes();
                        e.forget_grids();
                        total += e.bytes();
                    }
                }
            }
        } else if total < budget / 8 && self.rate_scale > 1.0 {
            self.rate_scale /= 2.0;
        }
        frame.stats.cache_bytes = total;
        frame.stats.rate_scale = self.rate_scale;
    }
}

/// Radiance at a point inside triangle `tri` with weights `w` over its
/// vertices, from the kept shading.
fn lookup(scene: &Scene, c: &crate::scene::Cluster, e: &ClusterShade, tri: u32, w: [f32; 3]) -> Vec3 {
    lookup_part(scene, c, e, tri, w, Part::All)
}

#[cfg_attr(not(target_pointer_width = "64"), allow(dead_code))]
/// Which part of the kept shading to read.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Part {
    All,
    Highlight,
}

fn lookup_part(scene: &Scene, c: &crate::scene::Cluster, e: &ClusterShade, tri: u32, w: [f32; 3], part: Part) -> Vec3 {
    let u = (tri - c.first_tri) as usize;
    let state = e.state.get(u).copied().unwrap_or(UNDICED);
    if state == UNDICED {
        return Vec3::ZERO;
    }
    if state == ON_VERTICES {
        let t = tri as usize * 3;
        let mut r = Vec3::ZERO;
        for k in 0..3 {
            let v = (scene.indices[t + k] - c.first_vertex) as usize;
            r += match part {
                Part::All => e.vdiffuse[v] + e.vspec[v],
                Part::Highlight => e.vspec[v],
            } * w[k];
        }
        return r;
    }
    let d = &e.grids[state as usize];
    let at = |i: usize| match part {
        Part::All => e.diffuse[i] + e.spec[i],
        Part::Highlight => e.spec[i],
    };
    let (u, v) = d.screen_to_grid(tri_to_screen(w, d.w));
    let (nu, nv) = (1usize << d.lu, 1usize << d.lv);
    let (fu, fv) = (u * nu as f32, v * nv as f32);
    let (a, b) = ((fu as usize).min(nu - 1), (fv as usize).min(nv - 1));
    let (fu, fv) = (fu - a as f32, fv - b as f32);
    let p = |a: usize, b: usize| at(d.offset as usize + a * (nv + 1) + b);
    (p(a, b) * (1.0 - fu) + p(a + 1, b) * fu) * (1.0 - fv) + (p(a, b + 1) * (1.0 - fu) + p(a + 1, b + 1) * fu) * fv
}

/// The sun as `shade_point` needs it: illuminance facing it, direction,
/// shadow map.
pub struct SunLit<'a> {
    pub lux: Vec3,
    pub dir: Vec3,
    pub shadow: Option<&'a SunShadow>,
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
    ground: bool,
    lights: &[PointLight],
    in_reach: &[usize],
    reach: &[f32],
    shadows: &[Option<&CubeShadow>],
    sun: &SunLit,
) -> (Vec3, Vec3, u64) {
    let inv_pi = std::f32::consts::FRAC_1_PI;
    let mut diffuse = albedo * sky.irradiance_from(n, ground) * inv_pi;
    let mut spec = Vec3::ZERO;
    let v = (eye - p).normalize_or(n);
    let mut evals = 0;
    // The sun: everywhere, parallel, through its own map.
    if sun.lux != Vec3::ZERO {
        let ndl = n.dot(sun.dir);
        if ndl > 0.0 {
            let vis = sun.shadow.map(|m| m.lit(p, n)).unwrap_or(1.0);
            if vis > 0.0 {
                let e = sun.lux * (ndl * vis);
                diffuse += e * albedo * inv_pi;
                let h = (sun.dir + v).normalize_or(sun.dir);
                spec += e * (n.dot(h).max(0.0).powf(48.0) * SPECULAR * inv_pi);
            }
        }
    }
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

/// Shading points per second on this machine, all cores, measured for
/// about `seconds`: the real `shade_point` on synthetic points, the sky and
/// two lights in reach (about what a point of the built-in set evaluates),
/// no shadow maps.
pub fn shading_rate(seconds: f64) -> f64 {
    let lights = [
        PointLight { position: Vec3::new(1.0, 2.0, 1.0), color: Vec3::ONE, lumens: 800.0 },
        PointLight { position: Vec3::new(-2.0, 1.0, 0.5), color: Vec3::new(1.0, 0.8, 0.6), lumens: 400.0 },
    ];
    let in_reach = [0usize, 1];
    let reach = [f32::INFINITY; 2];
    let shadows: [Option<&CubeShadow>; 2] = [None, None];
    let sky = Sky::new(30.0, true);
    let eye = Vec3::new(0.0, 1.0, 5.0);
    let run = |n: usize| -> f64 {
        let t = std::time::Instant::now();
        let s: f32 = (0..n)
            .into_par_iter()
            .map(|i| {
                let a = i as f32 * 0.618;
                let n = Vec3::new(a.sin(), (a * 1.7).cos(), (a * 0.3).sin()).normalize_or(Vec3::Y);
                let p = n * 0.5 + Vec3::new(0.0, 0.5, 0.0);
                let (d, sp, _) = shade_point(p, n, Vec3::splat(0.5), eye, sky, true, &lights, &in_reach, &reach, &shadows, &SunLit { lux: Vec3::ZERO, dir: Vec3::Y, shadow: None });
                d.x + sp.y
            })
            .sum();
        std::hint::black_box(s);
        t.elapsed().as_secs_f64()
    };
    let mut n = 2000;
    loop {
        let dt = run(n);
        if dt >= seconds || n >= 1 << 26 {
            return n as f64 / dt.max(1e-9);
        }
        n *= 2;
    }
}

/// One frame from nothing: a fresh renderer.
#[cfg(test)]
pub fn render(scene: &Scene, cam: &Camera, lights: &[PointLight], settings: &RenderSettings, frame: &mut Frame) {
    Renderer::new().render(scene, cam, lights, settings, frame);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A ground triangle seen at a grazing angle: wide and thin on screen,
    /// receding. Its grid is fine along its length only, its points no
    /// further apart than the shading rate, and there are about as many as
    /// its area on screen needs.
    #[test]
    fn dicing_is_even_on_screen() {
        let (w, h) = (640, 360);
        let cam = cam_at(0.0, w, h, Vec3::new(0.0, 0.0, -10.0), 30.0, 0.06);
        let vp = cam.proj * cam.view;
        let p = [Vec3::new(-6.0, 0.0, 0.0), Vec3::new(6.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -30.0)];
        let d = dice_triangle(vp, p, cam.near, (w, h), raster::SHADING_RATE_PX);
        let to_px = |b: [f32; 3]| {
            let q = vp * (p[0] * b[0] + p[1] * b[1] + p[2] * b[2]).extend(1.0);
            glam::Vec2::new(q.x / q.w * w as f32 * 0.5, q.y / q.w * h as f32 * 0.5)
        };
        let corner = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]].map(to_px);
        let area = ((corner[1] - corner[0]).perp_dot(corner[2] - corner[0]) * 0.5).abs();
        let (nu, nv) = (1usize << d.lu, 1usize << d.lv);
        let at = |a: usize, b: usize| to_px(screen_to_tri(d.grid_to_screen(a as f32 / nu as f32, b as f32 / nv as f32), d.w));
        let (mut widest, mut closest) = (0f32, f32::MAX);
        for a in 0..=nu {
            for b in 0..=nv {
                for (a2, b2) in [(a + 1, b), (a, b + 1)] {
                    if a2 <= nu && b2 <= nv {
                        let s = at(a, b).distance(at(a2, b2));
                        widest = widest.max(s);
                        closest = closest.min(s);
                    }
                }
            }
        }
        let rate = raster::SHADING_RATE_PX;
        assert!(widest <= rate * 1.001 && closest > rate * 0.5 * 0.999, "{closest} to {widest} px apart");
        // Even micropolygons of the rate would need area / rate^2 points;
        // the grid covers twice the triangle, and rounding to powers of two
        // can double each way.
        let ideal = area / (rate * rate);
        assert!(area > 2000.0 && (d.points() as f32) < 8.0 * ideal, "{} points for {ideal}", d.points());
        // One grid even on the whole triangle, cut as finely as its longest
        // edge needs, would hold far more.
        let longest = (0..3).map(|k| corner[k].distance(corner[(k + 1) % 3])).fold(0f32, f32::max);
        let n = (longest / rate).ceil();
        assert!(n * n / 2.0 > 1.5 * d.points() as f32, "{} against {}", n * n / 2.0, d.points());
    }

    /// Weights to grid and back give the same point.
    #[test]
    fn grid_points_are_found_where_they_were_made() {
        let d = TriDice { lu: 4, lv: 2, level: 4, base: 1, w: [1.0, 7.0, 2.5], px: [glam::Vec2::new(-40.0, 3.0), glam::Vec2::new(10.0, -20.0), glam::Vec2::new(90.0, 11.0)], ..Default::default() };
        for (u, v) in [(0.1, 0.2), (0.5, 0.5), (0.9, 0.3), (0.25, 0.95), (0.02, 0.6)] {
            let (uu, vv) = d.screen_to_grid(d.grid_to_screen(u, v));
            assert!((uu - u).abs() < 1e-4 && (vv - v).abs() < 1e-4, "({u}, {v}) came back ({uu}, {vv})");
        }
        // The triangle lies in the grid: its corners are on the grid's edges.
        for k in 0..3 {
            let mut s = [0.0; 3];
            s[k] = 1.0;
            let (u, v) = d.screen_to_grid(s);
            assert!((0.0..=1.0).contains(&u) && (0.0..=1.0).contains(&v));
        }
        let b = [0.2, 0.3, 0.5];
        let r = screen_to_tri(tri_to_screen(b, d.w), d.w);
        assert!((0..3).all(|k| (r[k] - b[k]).abs() < 1e-5));
    }

    /// Two triangles of very different size on screen share an edge, each
    /// with its own grid. Along the edge they read the same light to well
    /// under an 8-bit step, with a lamp 30 cm off the surface and
    /// micropolygons of about 3 cm. Grids do not share points, so light
    /// changing faster than the micropolygons (a lamp 10 cm off: 2 %)
    /// shows as a seam.
    #[test]
    fn neighbours_agree_along_a_shared_edge() {
        let mut scene = Scene::new();
        // Shared edge 0-2; triangle (0, 1, 2) reaches far down, (0, 2, 3)
        // barely up.
        let pos = [Vec3::new(-0.25, 0.0, 0.0), Vec3::new(0.0, -3.0, 0.0), Vec3::new(0.25, 0.0, 0.0), Vec3::new(0.0, 0.05, 0.0)];
        scene.push_mesh(&pos, &[], &[Vec3::splat(0.6)], &[0, 1, 2, 0, 2, 3]);
        let lights = [PointLight { position: Vec3::new(0.07, 0.0, 0.3), color: Vec3::ONE, lumens: 300.0 }];
        let (w, h) = (320, 320);
        let cam = cam_at(0.0, w, h, Vec3::new(0.0, -1.2, 0.0), 6.0, 0.0);
        let settings = RenderSettings { shadows: false, bounce: false, ..RenderSettings::new(Sky::new(30.0, true)) };
        let mut r = Renderer::new();
        let mut f = Frame::new(w, h);
        r.render(&scene, &cam, &lights, &settings, &mut f);
        let peek = r.peek();
        let (big, small) = (peek.tri_level(0, 0, &scene), peek.tri_level(0, 1, &scene));
        assert!(big >= small + 2, "levels {big} and {small}");
        let mut worst = 0f32;
        for k in 0..=200 {
            let t = k as f32 / 200.0;
            // The same point: 1 - t of corner 0, t of corner 2.
            let a = peek.radiance(&scene, 0, 0, [1.0 - t, 0.0, t], Part::All);
            let b = peek.radiance(&scene, 0, 1, [1.0 - t, t, 0.0], Part::All);
            worst = worst.max((a - b).abs().max_element() / a.max_element().max(1e-6));
        }
        assert!(worst < 0.004, "{worst}");
    }

    use crate::raster::Sky;

    fn cam_at(eye_yaw: f32, w: usize, h: usize, target: Vec3, dist: f32, pitch: f32) -> Camera {
        let c = crate::camera::OrbitCamera { target, distance: dist, yaw: eye_yaw, pitch, fov_y: 45f32.to_radians(), scene_radius: 1.0 };
        Camera { view: c.view(), proj: c.proj(w as f32 / h as f32), eye: c.eye(), near: c.near() }
    }

    fn world() -> (Scene, Vec<PointLight>) {
        let s = crate::world::build();
        (s, crate::world::small_lights())
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
        let settings = RenderSettings::new(Sky::new(30.0, true));
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
        // Bounce light kept beyond the reach of the change differs from a
        // fresh gather by less than the cut: at most one 8-bit step.
        assert!(worst(&frame, &fresh) <= 1, "{}", worst(&frame, &fresh));
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
        let settings = RenderSettings::new(Sky::new(30.0, true));
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

    /// Orbiting and zooming for a while keeps the shading cache under its
    /// budget: what is off screen is dropped.
    #[test]
    fn the_cache_stays_within_its_budget() {
        let (scene, lights) = world();
        let (w, h) = (320, 180);
        let budget = 24 << 20;
        let settings = RenderSettings { cache_budget: budget, ..RenderSettings::new(Sky::new(30.0, true)) };
        let mut r = Renderer::new();
        let mut f = Frame::new(w, h);
        let mut evicted = 0;
        for i in 0..80 {
            let cam = cam_at(i as f32 * 0.08, w, h, Vec3::new(0.0, 0.5, 0.0), 3.0 + (i as f32 * 0.3).sin() * 2.0, 0.1 + (i as f32 * 0.2).cos() * 0.08);
            r.render(&scene, &cam, &lights, &settings, &mut f);
            assert!(f.stats.cache_bytes <= budget || f.stats.rate_scale > 1.0, "frame {i}: {} bytes", f.stats.cache_bytes);
            evicted += f.stats.clusters_evicted;
        }
        assert!(evicted > 0);
        assert_eq!(f.stats.rate_scale, 1.0, "what is on screen fits");
    }

    /// With a budget too small for what is on screen, micropolygons grow
    /// until it fits, and the picture is still drawn.
    #[test]
    fn a_tiny_budget_makes_micropolygons_coarser_not_a_crash() {
        let (scene, lights) = world();
        let (w, h) = (320, 180);
        let budget = 1 << 20;
        let settings = RenderSettings { cache_budget: budget, ..RenderSettings::new(Sky::new(30.0, true)) };
        let cam = cam_at(0.0, w, h, Vec3::new(0.0, 0.5, 0.0), 2.0, 0.1);
        let mut r = Renderer::new();
        let mut f = Frame::new(w, h);
        for _ in 0..6 {
            r.render(&scene, &cam, &lights, &settings, &mut f);
        }
        assert!(f.stats.rate_scale > 1.0);
        assert!(f.stats.cache_bytes <= budget, "{} bytes", f.stats.cache_bytes);
        let lit = f.color.iter().filter(|p| (**p & 0xff) > 10).count();
        assert!(lit > w * h / 2, "{lit} pixels drawn");
    }

    /// Moving an object, then changing the sky: each time the kept picture
    /// is the one a fresh render gives (the moved clusters are diced again,
    /// everything is lit again, shadows and bounce are made again).
    #[test]
    fn moving_an_object_and_relighting_match_a_fresh_render() {
        let (mut scene, lights) = world();
        let (p, idx) = crate::scene::tests::uv_sphere(24);
        let first = scene.clusters.len();
        let ball: Vec<Vec3> = p.iter().map(|v| *v * 0.3 + Vec3::new(0.0, 0.6, 0.0)).collect();
        scene.push_mesh(&ball, &[], &[Vec3::new(0.8, 0.3, 0.2)], &idx);
        let n = scene.clusters.len();
        scene.add_object("ball".into(), first..n, Vec3::new(0.0, 0.3, 0.0), Vec3::ZERO);
        let (w, h) = (200, 112);
        let cam = cam_at(0.0, w, h, Vec3::new(0.0, 0.6, 0.0), 4.0, 0.15);
        let mut settings = RenderSettings::new(Sky::new(30.0, true));
        let mut r = Renderer::new();
        let mut f = Frame::new(w, h);
        r.render(&scene, &cam, &lights, &settings, &mut f);
        let before = f.color.clone();
        scene.set_object(0, Vec3::new(0.6, 0.3, 0.2), Vec3::new(0.4, 0.5, -0.3), 1.4);
        r.moved(scene.objects[0].clusters.clone(), true);
        r.render(&scene, &cam, &lights, &settings, &mut f);
        assert_ne!(f.color, before, "the picture changed");
        let mut fresh = Frame::new(w, h);
        render(&scene, &cam, &lights, &settings, &mut fresh);
        assert!(worst(&f, &fresh) <= 1, "after the move: {}", worst(&f, &fresh));
        settings.sky = Sky::new(300.0, true);
        r.render(&scene, &cam, &lights, &settings, &mut f);
        render(&scene, &cam, &lights, &settings, &mut fresh);
        assert!(worst(&f, &fresh) <= 1, "after the sky changed: {}", worst(&f, &fresh));
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
        let on = RenderSettings::new(Sky::new(1.0, false));
        let mut a = Frame::new(w, h);
        render(&scene, &cam, &lights, &RenderSettings { exposure: 0.01, ..on }, &mut a);
        let mut b = Frame::new(w, h);
        render(&scene, &cam, &lights, &RenderSettings { exposure: 0.01, shadows: false, ..RenderSettings::new(Sky::new(1.0, false)) }, &mut b);
        // Pixels where the floor shows in both and differs: under the card's shadow.
        let darker = a.color.iter().zip(&b.color).filter(|(x, y)| (**x & 0xff) + 20 < (**y & 0xff)).count();
        assert!(darker > 50, "{darker} pixels darker with shadows");
        let brighter = a.color.iter().zip(&b.color).filter(|(x, y)| (**x & 0xff) > (**y & 0xff)).count();
        assert_eq!(brighter, 0);
    }

    /// A textured quad is shaded from its tiles, not its vertex colours.
    #[test]
    fn shading_reads_texture_tiles() {
        use crate::texture::{level_offset, Source, TexEntry};
        use std::sync::Arc;
        let mut scene = Scene::new();
        let quad = [Vec3::new(-1.0, -1.0, 0.0), Vec3::new(1.0, -1.0, 0.0), Vec3::new(1.0, 1.0, 0.0), Vec3::new(-1.0, 1.0, 0.0)];
        scene.push_mesh(&quad, &[], &[Vec3::splat(0.5)], &[0, 1, 2, 0, 2, 3]);
        let c = scene.clusters[0];
        // Pure red tiles at levels 1 to 5.
        let len = level_offset(c.tri_count as usize, 6) as usize;
        let pool: Vec<u8> = (0..len / 3).flat_map(|_| [255u8, 0, 0]).collect();
        scene.textures.entries = vec![TexEntry { level: 5, source: 0, offset: 0 }];
        scene.textures.sources = vec![Arc::new(Source::Memory(Arc::new(pool)))];
        let (w, h) = (160, 120);
        let cam = cam_at(0.0, w, h, Vec3::ZERO, 3.0, 0.0);
        let settings = RenderSettings { shadows: false, bounce: false, ..RenderSettings::new(Sky::new(100.0, true)) };
        let mut f = Frame::new(w, h);
        render(&scene, &cam, &[], &settings, &mut f);
        let px = f.color[h / 2 * w + w / 2];
        let (r, g, b) = ((px >> 16) & 255, (px >> 8) & 255, px & 255);
        assert!(r > 40 && g < 5 && b < 5, "red from the tiles, got {r} {g} {b}");
        assert!(scene.textures.stats().reads >= 1);
    }
}
