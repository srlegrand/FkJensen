//! Visibility: which surface each pixel sees.
//!
//! Three passes, each parallel across cores with rayon:
//!
//! 1. **Setup**: clusters are tested against the view frustum (and, with
//!    back-face culling on, against their normal cone) as a whole; the
//!    vertices of each surviving cluster are projected into a small local
//!    array and its triangles are clipped against the near plane, optionally
//!    back-face culled, and given a screen-space record. Each visible
//!    cluster also gets a dicing level from the screen size of its
//!    triangles, for the shading in `reyes.rs`.
//! 2. **Bin**: each record is listed in every 64 x 64 bucket its box touches.
//! 3. **Hide**: each bucket is rasterised by one thread into its own part
//!    of the depth and triangle-id buffers. No atomics, no locks.
//!
//! Nothing is shaded here. Shading happens on diced micropolygons in world
//! space and is kept between frames (`reyes.rs`); when the camera has not
//! moved, this whole module is skipped.
//!
//! Also here: the sky, light reach, and small drawing helpers.

use glam::{Mat4, Vec2, Vec3, Vec4};
use rayon::prelude::*;

use crate::scene::{Cluster, PointLight, Scene};

/// Default bucket edge in pixels; `RenderSettings::tile` sets it per machine.
pub const TILE: usize = 64;
pub const NO_TRI: u32 = u32::MAX;

pub struct Camera {
    pub view: Mat4,
    pub proj: Mat4,
    pub eye: Vec3,
    pub near: f32,
}

pub struct Frame {
    pub width: usize,
    pub height: usize,
    /// 0x00RRGGBB, sRGB, what softbuffer wants.
    pub color: Vec<u32>,
    /// 1/w per pixel, 0 where empty. Larger is closer.
    pub depth: Vec<f32>,
    /// Index into the frame's setup records, or NO_TRI.
    pub tri_id: Vec<u32>,
    /// Per-frame setup records, kept so the app can pick (unproject a pixel).
    pub setups: Vec<TriSetup>,
    pub stats: Stats,
    /// Which renderer's visibility the buffers hold (0: none), so a
    /// renderer only reuses visibility it made itself in this frame.
    pub vis_stamp: u64,
}

#[derive(Default, Clone, Copy, Debug)]
pub struct Stats {
    pub clusters_in: usize,
    pub clusters_drawn: usize,
    /// Point light evaluations done while shading this frame.
    pub light_evals: u64,
    /// What they would have been without light reach: shading points
    /// shaded this frame x lights.
    pub light_evals_all: u64,
    pub tris_in: usize,
    pub tris_drawn: usize,
    /// Whether visibility was kept from the last frame (camera unchanged).
    pub visibility_kept: bool,
    /// Shading points shaded this frame (the rest came from the cache).
    pub points_shaded: u64,
    /// Shading points the frame used.
    pub points_used: u64,
    /// Clusters that had any shading done this frame.
    pub clusters_shaded: usize,
    /// Whether every pixel was resolved, or only those of clusters shaded
    /// this frame.
    pub full_resolve: bool,
    /// Surfels lit again, receivers whose bounce was gathered, receivers in
    /// view still waiting, and the time spent.
    pub surfels_lit: u64,
    pub receivers_updated: usize,
    pub receivers_pending: usize,
    pub bounce_ms: f32,
    pub gather_ms: f32,
    /// Shadow maps rebuilt this frame, and triangles drawn into them.
    pub shadow_maps_built: usize,
    pub shadow_tris: u64,
    pub setup_ms: f32,
    pub bin_ms: f32,
    pub raster_ms: f32,
    pub shadow_ms: f32,
    pub shade_ms: f32,
    pub resolve_ms: f32,
}

/// A triangle after projection: screen positions, 1/w, and each corner's
/// weights over the source triangle's three vertices (identity unless the
/// triangle was clipped).
#[derive(Clone, Copy)]
pub struct TriSetup {
    pub x: [f32; 3],
    pub y: [f32; 3],
    pub inv_w: [f32; 3],
    pub bary: [[f32; 3]; 3],
    pub tri: u32,
    /// The cluster the triangle belongs to.
    pub cluster: u32,
    pub min_x: u16,
    pub min_y: u16,
    pub max_x: u16,
    pub max_y: u16,
}

/// The sky: light from every direction above, with nothing between it and
/// any surface, so a mountain 10 km away and a pawn are lit by it alike.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sky {
    /// Illuminance on an upward-facing surface, in lux.
    pub lux: f32,
    pub on: bool,
}

/// Sky colour seen straight up and at the horizon, luminance about 1.
const SKY_ZENITH: Vec3 = Vec3::new(0.55, 0.78, 1.45);
const SKY_HORIZON: Vec3 = Vec3::new(1.15, 1.0, 0.85);
/// Light coming back up from the ground, as a share of the sky's.
const GROUND_BOUNCE: f32 = 0.2;
const GROUND_TINT: Vec3 = Vec3::new(1.0, 1.0, 0.9);
/// Colour of the sky's light on an upward-facing surface (luminance 1).
const SKY_IRRADIANCE_TINT: Vec3 = Vec3::new(0.82, 0.95, 1.25);

impl Sky {
    /// Illuminance in lux, per colour channel, on a surface facing `n`.
    #[allow(dead_code)]
    pub fn irradiance(&self, n: Vec3) -> Vec3 {
        self.irradiance_from(n, true)
    }

    /// The same, with or without the light the ground sends back up. With
    /// bounce light on, the ground term is left out: the real ground is in
    /// the surfel cloud and bounces its own light.
    pub fn irradiance_from(&self, n: Vec3, ground: bool) -> Vec3 {
        if !self.on {
            return Vec3::ZERO;
        }
        let up = n.y * 0.5 + 0.5;
        let g = if ground { GROUND_TINT * GROUND_BOUNCE * (1.0 - up) } else { Vec3::ZERO };
        self.lux * (SKY_IRRADIANCE_TINT * up + g)
    }

    /// Radiance in cd/m², per colour channel, seen looking along `dir`.
    pub fn radiance(&self, dir: Vec3) -> Vec3 {
        if !self.on {
            return Vec3::ZERO;
        }
        let l = self.lux / std::f32::consts::PI;
        if dir.y >= 0.0 {
            let t = dir.y.sqrt();
            l * SKY_HORIZON.lerp(SKY_ZENITH, t)
        } else {
            l * GROUND_BOUNCE * GROUND_TINT
        }
    }
}

/// What the shade pass writes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum View {
    Shaded,
    /// How many point lights each pixel evaluates: shows light reach.
    LightCount,
}

pub struct RenderSettings {
    pub cull_backfaces: bool,
    pub wireframe: bool,
    /// Display value per cd/m², before the tone curve.
    pub exposure: f32,
    pub sky: Sky,
    /// Skip lights at pixels beyond their reach. Off only to compare.
    pub light_reach: bool,
    /// Shadow maps for the point lights.
    pub shadows: bool,
    /// Texels per cube face edge of each shadow map.
    pub shadow_res: usize,
    /// Bucket edge in pixels.
    pub tile: usize,
    /// Bounce light (point-based colour bleeding).
    pub bounce: bool,
    /// Time per frame for gathering bounce light, in ms; None: all of it.
    pub bounce_budget_ms: Option<f32>,
    pub view: View,
}

impl RenderSettings {
    pub fn new(sky: Sky) -> Self {
        Self { cull_backfaces: false, wireframe: false, exposure: default_exposure(sky.lux), sky, light_reach: true, shadows: true, shadow_res: crate::shadow::SHADOW_RES, tile: TILE, bounce: true, bounce_budget_ms: None, view: View::Shaded }
    }
}

/// An exposure that keeps sky-lit surfaces in the lower half of the
/// display, so lamps read against them.
pub fn default_exposure(sky_lux: f32) -> f32 {
    std::f32::consts::PI / (8.0 * sky_lux.max(1.0))
}

/// Largest radiance a surface sends per lux it receives: white diffuse
/// (1/π) plus the highlight's peak.
const BRDF_PEAK: f32 = (1.0 + SPECULAR) / std::f32::consts::PI;
pub const SPECULAR: f32 = 0.12;
/// Steepest slope of the tone curve.
const TONE_SLOPE: f32 = 1.25;

/// Illuminance below which a light's contribution is dropped, in lux.
///
/// With the sky on: 1/64 of what the sky gives the darkest-facing surface
/// (one lit only by ground bounce), a change of about 1.6%, at or under one
/// 8-bit step in the mid tones. Always also: half an 8-bit step at black for
/// the current exposure, so with the sky off a light reaches as far as it
/// can still be seen in the dark.
pub fn cut_lux(sky: &Sky, exposure: f32) -> f32 {
    let black_step = 0.5 / (255.0 * 12.92); // linear value of half an sRGB step at 0
    let floor = black_step / (BRDF_PEAK * TONE_SLOPE * exposure.max(1e-12));
    let sky_part = if sky.on { sky.lux * GROUND_BOUNCE / 64.0 } else { 0.0 };
    floor.max(sky_part)
}

/// How far, in metres, a light's illuminance stays at or above `cut_lux`.
pub fn light_reach(l: &PointLight, cut: f32) -> f32 {
    (l.candela() / cut.max(1e-12)).sqrt()
}

impl Frame {
    pub fn new(width: usize, height: usize) -> Self {
        let n = width * height;
        Self {
            width,
            height,
            color: vec![0; n],
            depth: vec![0.0; n],
            tri_id: vec![NO_TRI; n],
            setups: Vec::new(),
            stats: Stats::default(),
            vis_stamp: 0,
        }
    }

    pub fn resize(&mut self, width: usize, height: usize) {
        if width != self.width || height != self.height {
            *self = Self::new(width, height);
        }
    }
}

struct ClipVert {
    p: Vec4,
    bary: [f32; 3],
}

fn lerp_vert(a: &ClipVert, b: &ClipVert, t: f32) -> ClipVert {
    ClipVert {
        p: a.p.lerp(b.p, t),
        bary: [
            a.bary[0] + (b.bary[0] - a.bary[0]) * t,
            a.bary[1] + (b.bary[1] - a.bary[1]) * t,
            a.bary[2] + (b.bary[2] - a.bary[2]) * t,
        ],
    }
}

/// Clip a triangle against w >= near. Returns 0, 1 or 2 triangles.
fn clip_near(v: [ClipVert; 3], near: f32, out: &mut Vec<[ClipVert; 3]>) {
    let inside: Vec<bool> = v.iter().map(|c| c.p.w >= near).collect();
    let n_in = inside.iter().filter(|b| **b).count();
    match n_in {
        3 => out.push(v),
        0 => {}
        _ => {
            // Sutherland-Hodgman against one plane.
            let mut poly: Vec<ClipVert> = Vec::with_capacity(4);
            for i in 0..3 {
                let j = (i + 1) % 3;
                let (a, b) = (&v[i], &v[j]);
                if inside[i] {
                    poly.push(ClipVert { p: a.p, bary: a.bary });
                }
                if inside[i] != inside[j] {
                    let t = (near - a.p.w) / (b.p.w - a.p.w);
                    poly.push(lerp_vert(a, b, t));
                }
            }
            for k in 1..poly.len().saturating_sub(1) {
                out.push([
                    ClipVert { p: poly[0].p, bary: poly[0].bary },
                    ClipVert { p: poly[k].p, bary: poly[k].bary },
                    ClipVert { p: poly[k + 1].p, bary: poly[k + 1].bary },
                ]);
            }
        }
    }
}

fn setup_tri(tri: u32, v: &[ClipVert; 3], width: usize, height: usize, cull: bool) -> Option<TriSetup> {
    let mut x = [0f32; 3];
    let mut y = [0f32; 3];
    let mut inv_w = [0f32; 3];
    let (w, h) = (width as f32, height as f32);
    for i in 0..3 {
        let iw = 1.0 / v[i].p.w;
        inv_w[i] = iw;
        // NDC to pixels, y down.
        x[i] = (v[i].p.x * iw * 0.5 + 0.5) * w;
        y[i] = (0.5 - v[i].p.y * iw * 0.5) * h;
    }
    // Signed area in screen space (y down). A triangle wound counter-
    // clockwise as seen from its front (right-handed, the USD and OpenGL
    // convention) has negative area here, so positive area is a back face.
    let area = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]);
    if area == 0.0 || !area.is_finite() {
        return None;
    }
    if cull && area > 0.0 {
        return None;
    }
    let min_x = x.iter().cloned().fold(f32::INFINITY, f32::min).floor().max(0.0);
    let min_y = y.iter().cloned().fold(f32::INFINITY, f32::min).floor().max(0.0);
    let max_x = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max).ceil().min(w - 1.0);
    let max_y = y.iter().cloned().fold(f32::NEG_INFINITY, f32::max).ceil().min(h - 1.0);
    if max_x < min_x || max_y < min_y {
        return None;
    }
    let mut s = TriSetup {
        x,
        y,
        inv_w,
        bary: [v[0].bary, v[1].bary, v[2].bary],
        tri,
        cluster: 0,
        min_x: min_x as u16,
        min_y: min_y as u16,
        max_x: max_x as u16,
        max_y: max_y as u16,
    };
    // Make every triangle clockwise on screen so one edge test serves all.
    if area < 0.0 {
        s.x.swap(1, 2);
        s.y.swap(1, 2);
        s.inv_w.swap(1, 2);
        s.bary.swap(1, 2);
    }
    Some(s)
}

/// The frustum planes (left, right, bottom, top, near) of a view-projection
/// matrix, normalised, as (normal, d) with inside meaning `n.dot(p) + d >= 0`.
/// The projection is infinite, so there is no far plane.
pub fn frustum_planes(vp: Mat4, near: f32) -> [Vec4; 5] {
    let (r0, r1, r3) = (vp.row(0), vp.row(1), vp.row(3));
    let norm = |p: Vec4| p / p.truncate().length().max(1e-20);
    [norm(r3 + r0), norm(r3 - r0), norm(r3 + r1), norm(r3 - r1), norm(r3 - Vec4::new(0.0, 0.0, 0.0, near))]
}

/// Whether a cluster can be skipped: outside a frustum plane, or (with
/// back-face culling) every face turned away from the eye.
fn cluster_hidden(c: &Cluster, planes: &[Vec4; 5], eye: Vec3, cull_backfaces: bool) -> bool {
    for p in planes {
        if p.truncate().dot(c.centre) + p.w < -c.radius {
            return true;
        }
    }
    if cull_backfaces && c.cone_cos > 0.0 {
        let d = c.centre - eye;
        let dist = d.length();
        if dist > c.radius {
            // The cone of view directions to the sphere is asin(r/d) wide.
            // Every face is back-facing when each view direction is within
            // 90 degrees of every face normal.
            let spread = c.cone_cos.clamp(-1.0, 1.0).acos();
            let limit = std::f32::consts::FRAC_PI_2 - spread - (c.radius / dist).asin();
            if limit > 0.0 && (d / dist).dot(c.cone_axis).clamp(-1.0, 1.0).acos() < limit {
                return true;
            }
        }
    }
    false
}

/// Shading points per micropolygon edge target, in pixels.
pub const SHADING_RATE_PX: f32 = 2.0;
/// Finest dicing level: a triangle is cut into 2^level parts per edge.
pub const MAX_DICE_LEVEL: u8 = 5;
/// Dicing level of a cluster that is not on screen.
pub const NOT_VISIBLE: u8 = u8::MAX;

/// Dicing level for a cluster whose longest on-screen triangle edge is
/// `edge_px` pixels: the smallest level whose micropolygon edges are at most
/// `SHADING_RATE_PX`, up to `MAX_DICE_LEVEL`.
pub fn dice_level(edge_px: f32) -> u8 {
    if !(edge_px > SHADING_RATE_PX) {
        return 0;
    }
    ((edge_px / SHADING_RATE_PX).log2().ceil() as i32).clamp(0, MAX_DICE_LEVEL as i32) as u8
}

/// Work out what each pixel sees. Fills `frame.depth`, `frame.tri_id` and
/// `frame.setups`, and returns each cluster's dicing level (`NOT_VISIBLE`
/// for clusters culled whole).
pub fn visibility(scene: &Scene, cam: &Camera, cull: bool, wireframe: bool, tile: usize, frame: &mut Frame) -> Vec<u8> {
    let tile = tile.max(8);
    let width = frame.width;
    let height = frame.height;
    let vp = cam.proj * cam.view;
    let t0 = std::time::Instant::now();

    // Pass 1: cull clusters, then project each survivor's vertices into a
    // local array and set up its triangles.
    let near = cam.near;
    let planes = frustum_planes(vp, near);
    let eye = cam.eye;
    let parts: Vec<(Vec<TriSetup>, Vec<(u32, u8)>)> = scene
        .clusters
        .par_chunks(64)
        .enumerate()
        .map(|(ci, chunk)| {
            let mut out: Vec<TriSetup> = Vec::new();
            let mut levels: Vec<(u32, u8)> = Vec::new();
            let mut clipped: Vec<[ClipVert; 3]> = Vec::new();
            let mut clip = [Vec4::ZERO; crate::scene::CLUSTER_VERTS];
            for (k, c) in chunk.iter().enumerate() {
                let cluster = (ci * 64 + k) as u32;
                if cluster_hidden(c, &planes, eye, cull) {
                    continue;
                }
                let fv = c.first_vertex as usize;
                for (k, p) in scene.positions[fv..fv + c.vertex_count as usize].iter().enumerate() {
                    clip[k] = vp * p.extend(1.0);
                }
                // Longest triangle edge on screen, for the dicing level.
                let mut edge_px = 0f32;
                let to_px = |p: Vec4| Vec2::new((p.x / p.w * 0.5 + 0.5) * width as f32, (0.5 - p.y / p.w * 0.5) * height as f32);
                for tri in c.first_tri..c.first_tri + c.tri_count {
                    let t = tri as usize * 3;
                    let p0 = clip[scene.indices[t] as usize - fv];
                    let p1 = clip[scene.indices[t + 1] as usize - fv];
                    let p2 = clip[scene.indices[t + 2] as usize - fv];
                    // Trivial rejects against the frustum in clip space.
                    if (p0.w < near && p1.w < near && p2.w < near)
                        || (p0.x > p0.w && p1.x > p1.w && p2.x > p2.w)
                        || (p0.x < -p0.w && p1.x < -p1.w && p2.x < -p2.w)
                        || (p0.y > p0.w && p1.y > p1.w && p2.y > p2.w)
                        || (p0.y < -p0.w && p1.y < -p1.w && p2.y < -p2.w)
                    {
                        continue;
                    }
                    let v = [
                        ClipVert { p: p0, bary: [1.0, 0.0, 0.0] },
                        ClipVert { p: p1, bary: [0.0, 1.0, 0.0] },
                        ClipVert { p: p2, bary: [0.0, 0.0, 1.0] },
                    ];
                    if p0.w >= near && p1.w >= near && p2.w >= near {
                        let (a, b, cc) = (to_px(p0), to_px(p1), to_px(p2));
                        edge_px = edge_px.max(a.distance(b)).max(b.distance(cc)).max(cc.distance(a));
                        if let Some(mut s) = setup_tri(tri, &v, width, height, cull) {
                            s.cluster = cluster;
                            out.push(s);
                        }
                    } else {
                        // Crosses the near plane: as close as it gets, so
                        // dice it as finely as allowed.
                        edge_px = f32::INFINITY;
                        clipped.clear();
                        clip_near(v, near, &mut clipped);
                        for cv in &clipped {
                            if let Some(mut s) = setup_tri(tri, cv, width, height, cull) {
                                s.cluster = cluster;
                                out.push(s);
                            }
                        }
                    }
                }
                levels.push((cluster, dice_level(edge_px)));
            }
            (out, levels)
        })
        .collect();
    let mut level = vec![NOT_VISIBLE; scene.clusters.len()];
    let mut setups: Vec<TriSetup> = Vec::with_capacity(parts.iter().map(|p| p.0.len()).sum());
    for (s, l) in parts {
        setups.extend(s);
        for (c, lv) in l {
            level[c as usize] = lv;
        }
    }
    let clusters_drawn = level.iter().filter(|l| **l != NOT_VISIBLE).count();
    let t1 = std::time::Instant::now();

    // Pass 2: bin into buckets. Each chunk of records writes its own per-
    // bucket lists; the lists are then joined per bucket.
    let tiles_x = width.div_ceil(tile);
    let tiles_y = height.div_ceil(tile);
    let n_tiles = tiles_x * tiles_y;
    let chunk_lists: Vec<Vec<Vec<u32>>> = setups
        .par_chunks(8192)
        .enumerate()
        .map(|(ci, chunk)| {
            let mut lists: Vec<Vec<u32>> = vec![Vec::new(); n_tiles];
            let base = (ci * 8192) as u32;
            for (k, s) in chunk.iter().enumerate() {
                let id = base + k as u32;
                let tx0 = s.min_x as usize / tile;
                let tx1 = s.max_x as usize / tile;
                let ty0 = s.min_y as usize / tile;
                let ty1 = s.max_y as usize / tile;
                for ty in ty0..=ty1 {
                    for tx in tx0..=tx1 {
                        lists[ty * tiles_x + tx].push(id);
                    }
                }
            }
            lists
        })
        .collect();
    let tile_lists: Vec<Vec<u32>> = (0..n_tiles)
        .into_par_iter()
        .map(|t| {
            let mut v = Vec::new();
            for cl in &chunk_lists {
                v.extend_from_slice(&cl[t]);
            }
            v
        })
        .collect();
    let t2 = std::time::Instant::now();

    // Pass 3: hide. Each bucket is rasterised into a scratch of its own,
    // then copied out, so the buffers are written without sharing.
    let tile_results: Vec<(Vec<f32>, Vec<u32>)> = (0..n_tiles)
        .into_par_iter()
        .map(|t| {
            let tx = t % tiles_x;
            let ty = t / tiles_x;
            let x0 = tx * tile;
            let y0 = ty * tile;
            let x1 = (x0 + tile).min(width);
            let y1 = (y0 + tile).min(height);
            let tw = x1 - x0;
            let th = y1 - y0;
            let mut depth = vec![0f32; tw * th];
            let mut id = vec![NO_TRI; tw * th];
            for &sid in &tile_lists[t] {
                let s = &setups[sid as usize];
                raster_tri(s, sid, x0, y0, x1, y1, tw, &mut depth, &mut id, wireframe);
            }
            (depth, id)
        })
        .collect();
    for (t, (depth, id)) in tile_results.iter().enumerate() {
        let tx = t % tiles_x;
        let ty = t / tiles_x;
        let x0 = tx * tile;
        let y0 = ty * tile;
        let x1 = (x0 + tile).min(width);
        let y1 = (y0 + tile).min(height);
        let tw = x1 - x0;
        for (row, y) in (y0..y1).enumerate() {
            let dst = y * width + x0;
            frame.depth[dst..dst + tw].copy_from_slice(&depth[row * tw..row * tw + tw]);
            frame.tri_id[dst..dst + tw].copy_from_slice(&id[row * tw..row * tw + tw]);
        }
    }
    let t3 = std::time::Instant::now();

    frame.stats.clusters_in = scene.clusters.len();
    frame.stats.clusters_drawn = clusters_drawn;
    frame.stats.tris_in = scene.triangle_count();
    frame.stats.tris_drawn = setups.len();
    frame.stats.setup_ms = (t1 - t0).as_secs_f32() * 1000.0;
    frame.stats.bin_ms = (t2 - t1).as_secs_f32() * 1000.0;
    frame.stats.raster_ms = (t3 - t2).as_secs_f32() * 1000.0;
    frame.setups = setups;
    level
}

/// The direction of the view ray through pixel (x, y), for the sky behind
/// everything.
pub fn sky_dir(cam: &Camera, width: usize, height: usize, x: usize, y: usize, inv_view: &Mat4) -> Vec3 {
    let ndc = Vec2::new((x as f32 + 0.5) / width as f32 * 2.0 - 1.0, 1.0 - (y as f32 + 0.5) / height as f32 * 2.0);
    inv_view.transform_vector3(Vec3::new(ndc.x / cam.proj.x_axis.x, ndc.y / cam.proj.y_axis.y, -1.0)).normalize_or(Vec3::Y)
}

#[inline]
fn raster_tri(
    s: &TriSetup,
    sid: u32,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    tw: usize,
    depth: &mut [f32],
    id: &mut [u32],
    wireframe: bool,
) {
    let bx0 = (s.min_x as usize).max(x0);
    let bx1 = (s.max_x as usize + 1).min(x1);
    let by0 = (s.min_y as usize).max(y0);
    let by1 = (s.max_y as usize + 1).min(y1);
    if bx0 >= bx1 || by0 >= by1 {
        return;
    }
    // Edge functions, clockwise on screen. e_i(p) = (x_j - x_i)(p.y - y_i) - (y_j - y_i)(p.x - x_i)
    let (ax, ay, bx, by, cx, cy) = (s.x[0], s.y[0], s.x[1], s.y[1], s.x[2], s.y[2]);
    let area = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay);
    if area <= 0.0 {
        return;
    }
    let inv_area = 1.0 / area;
    // Steps per pixel in x and y for each edge.
    let (a0, b0) = (-(cy - by), cx - bx); // edge bc, weight for vertex a
    let (a1, b1) = (-(ay - cy), ax - cx); // edge ca, weight for vertex b
    let (a2, b2) = (-(by - ay), bx - ax); // edge ab, weight for vertex c
    let px = bx0 as f32 + 0.5;
    let py = by0 as f32 + 0.5;
    let mut e0_row = (cx - bx) * (py - by) - (cy - by) * (px - bx);
    let mut e1_row = (ax - cx) * (py - cy) - (ay - cy) * (px - cx);
    let mut e2_row = (bx - ax) * (py - ay) - (by - ay) * (px - ax);
    // 1/w is affine in screen space.
    let dz_dx = (s.inv_w[0] * a0 + s.inv_w[1] * a1 + s.inv_w[2] * a2) * inv_area;
    let dz_dy = (s.inv_w[0] * b0 + s.inv_w[1] * b1 + s.inv_w[2] * b2) * inv_area;
    let mut z_row = (s.inv_w[0] * e0_row + s.inv_w[1] * e1_row + s.inv_w[2] * e2_row) * inv_area;
    // Wireframe: a pixel is on an edge when one edge function is within a
    // pixel's step of zero.
    let edge_w = [
        (a0 * a0 + b0 * b0).sqrt() * 0.75,
        (a1 * a1 + b1 * b1).sqrt() * 0.75,
        (a2 * a2 + b2 * b2).sqrt() * 0.75,
    ];
    for y in by0..by1 {
        let mut e0 = e0_row;
        let mut e1 = e1_row;
        let mut e2 = e2_row;
        let mut z = z_row;
        let row = (y - y0) * tw;
        for x in bx0..bx1 {
            if e0 >= 0.0 && e1 >= 0.0 && e2 >= 0.0 {
                let inside = !wireframe || e0 < edge_w[0] || e1 < edge_w[1] || e2 < edge_w[2];
                let i = row + (x - x0);
                if inside && z > depth[i] {
                    depth[i] = z;
                    id[i] = sid;
                }
            }
            e0 += a0;
            e1 += a1;
            e2 += a2;
            z += dz_dx;
        }
        e0_row += b0;
        e1_row += b1;
        e2_row += b2;
        z_row += dz_dy;
    }
}

/// Perspective-correct barycentric weights of a pixel centre.
#[inline]
pub fn barycentrics(s: &TriSetup, px: f32, py: f32) -> [f32; 3] {
    let (ax, ay, bx, by, cx, cy) = (s.x[0], s.y[0], s.x[1], s.y[1], s.x[2], s.y[2]);
    let area = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay);
    let e0 = (cx - bx) * (py - by) - (cy - by) * (px - bx);
    let e1 = (ax - cx) * (py - cy) - (ay - cy) * (px - cx);
    let e2 = (bx - ax) * (py - ay) - (by - ay) * (px - ax);
    let inv = 1.0 / area;
    let l0 = e0 * inv * s.inv_w[0];
    let l1 = e1 * inv * s.inv_w[1];
    let l2 = e2 * inv * s.inv_w[2];
    let sum = l0 + l1 + l2;
    if sum.abs() < 1e-20 {
        return [1.0, 0.0, 0.0];
    }
    [l0 / sum, l1 / sum, l2 / sum]
}

/// World position under a pixel, from the visibility buffer.
pub fn unproject(frame: &Frame, scene: &Scene, x: usize, y: usize) -> Option<(Vec3, Vec3)> {
    let id = *frame.tri_id.get(y * frame.width + x)?;
    if id == NO_TRI {
        return None;
    }
    let s = &frame.setups[id as usize];
    let b = barycentrics(s, x as f32 + 0.5, y as f32 + 0.5);
    let w0 = b[0] * s.bary[0][0] + b[1] * s.bary[1][0] + b[2] * s.bary[2][0];
    let w1 = b[0] * s.bary[0][1] + b[1] * s.bary[1][1] + b[2] * s.bary[2][1];
    let w2 = b[0] * s.bary[0][2] + b[1] * s.bary[1][2] + b[2] * s.bary[2][2];
    let t = s.tri as usize * 3;
    let (i0, i1, i2) = (scene.indices[t] as usize, scene.indices[t + 1] as usize, scene.indices[t + 2] as usize);
    let p = scene.positions[i0] * w0 + scene.positions[i1] * w1 + scene.positions[i2] * w2;
    let n = (scene.normals[i0] * w0 + scene.normals[i1] * w1 + scene.normals[i2] * w2).normalize_or(Vec3::Y);
    Some((p, n))
}

/// A world point on screen: pixel x, y and 1/w. None when behind the near
/// plane.
pub fn project(cam: &Camera, width: usize, height: usize, p: Vec3) -> Option<(f32, f32, f32)> {
    let c = cam.proj * cam.view * p.extend(1.0);
    if c.w <= cam.near {
        return None;
    }
    let iw = 1.0 / c.w;
    Some(((c.x * iw * 0.5 + 0.5) * width as f32, (0.5 - c.y * iw * 0.5) * height as f32, iw))
}

/// The world-space ray through a pixel centre: (origin at the eye, unit
/// direction).
pub fn pixel_ray(cam: &Camera, width: usize, height: usize, x: f32, y: f32) -> (Vec3, Vec3) {
    let ndc = Vec2::new(x / width as f32 * 2.0 - 1.0, 1.0 - y / height as f32 * 2.0);
    // The direction in view space comes straight from the projection's
    // focal terms; going through the inverse of the whole matrix loses
    // precision with a small near plane.
    let d_view = Vec3::new(ndc.x / cam.proj.x_axis.x, ndc.y / cam.proj.y_axis.y, -1.0);
    let d = cam.view.inverse().transform_vector3(d_view);
    (cam.eye, d.normalize_or(Vec3::NEG_Z))
}

/// Draw a one-pixel line between two world points, without depth test,
/// dashed when `dashed`.
pub fn draw_line(frame: &mut Frame, cam: &Camera, a: Vec3, b: Vec3, color: Vec3, dashed: bool) {
    let (Some(pa), Some(pb)) = (project(cam, frame.width, frame.height, a), project(cam, frame.width, frame.height, b)) else { return };
    let packed = pack(color);
    let steps = (pb.0 - pa.0).abs().max((pb.1 - pa.1).abs()).ceil().min(8192.0) as usize;
    for i in 0..=steps {
        if dashed && (i / 4) % 2 == 1 {
            continue;
        }
        let t = if steps == 0 { 0.0 } else { i as f32 / steps as f32 };
        let x = (pa.0 + (pb.0 - pa.0) * t) as i32;
        let y = (pa.1 + (pb.1 - pa.1) * t) as i32;
        if x >= 0 && y >= 0 && (x as usize) < frame.width && (y as usize) < frame.height {
            frame.color[y as usize * frame.width + x as usize] = packed;
        }
    }
}

/// Draw a filled disc into the colour buffer, depth-tested at its centre.
pub fn draw_disc(frame: &mut Frame, cam: &Camera, centre: Vec3, radius_px: f32, color: Vec3, ring: bool) {
    let Some((sx, sy, iw)) = project(cam, frame.width, frame.height, centre) else { return };
    let (cx, cy) = (sx as i32, sy as i32);
    let occluded = if cx >= 0 && cy >= 0 && (cx as usize) < frame.width && (cy as usize) < frame.height {
        frame.depth[cy as usize * frame.width + cx as usize] > iw
    } else {
        false
    };
    let col = if occluded { color * 0.35 } else { color };
    let packed = pack(col);
    let r = radius_px.ceil() as i32;
    for dy in -r..=r {
        for dx in -r..=r {
            let d2 = (dx * dx + dy * dy) as f32;
            if d2 > radius_px * radius_px {
                continue;
            }
            if ring && d2 < (radius_px - 1.5) * (radius_px - 1.5) {
                continue;
            }
            let (x, y) = (cx + dx, cy + dy);
            if x >= 0 && y >= 0 && (x as usize) < frame.width && (y as usize) < frame.height {
                frame.color[y as usize * frame.width + x as usize] = packed;
            }
        }
    }
}

#[inline]
pub fn tonemap(c: Vec3) -> Vec3 {
    // Simple filmic-ish curve, keeps highlights from clipping to white.
    let c = c.max(Vec3::ZERO);
    c / (Vec3::ONE + c) * TONE_SLOPE
}

/// Colour for a count of lights: dark grey for none, then blue, green,
/// yellow, red, white.
pub fn heat(k: u32) -> Vec3 {
    const RAMP: [Vec3; 6] = [
        Vec3::new(0.012, 0.012, 0.014),
        Vec3::new(0.02, 0.08, 0.5),
        Vec3::new(0.03, 0.45, 0.08),
        Vec3::new(0.7, 0.6, 0.02),
        Vec3::new(0.8, 0.08, 0.02),
        Vec3::new(0.9, 0.9, 0.9),
    ];
    RAMP[(k as usize).min(RAMP.len() - 1)]
}

#[inline]
pub fn pack(c: Vec3) -> u32 {
    let to8 = |v: f32| -> u32 {
        let v = v.clamp(0.0, 1.0);
        // sRGB transfer
        let s = if v <= 0.0031308 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 };
        (s * 255.0 + 0.5) as u32
    };
    (to8(c.x) << 16) | (to8(c.y) << 8) | to8(c.z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reyes::render;

    fn quad_scene() -> Scene {
        let mut s = Scene::new();
        let p = [Vec3::new(-1.0, -1.0, 0.0), Vec3::new(1.0, -1.0, 0.0), Vec3::new(1.0, 1.0, 0.0), Vec3::new(-1.0, 1.0, 0.0)];
        s.push_mesh(&p, &[], &[Vec3::splat(0.8)], &[0, 1, 2, 0, 2, 3]);
        s
    }

    fn settings() -> RenderSettings {
        RenderSettings { exposure: 0.05, ..RenderSettings::new(Sky { lux: 100.0, on: false }) }
    }

    fn camera(w: usize, h: usize) -> Camera {
        let cam = crate::camera::OrbitCamera { target: Vec3::ZERO, distance: 3.0, yaw: 0.0, pitch: 0.0, fov_y: 45f32.to_radians(), scene_radius: 1.5 };
        Camera { view: cam.view(), proj: cam.proj(w as f32 / h as f32), eye: cam.eye(), near: cam.near() }
    }

    #[test]
    fn quad_covers_centre_and_not_corner() {
        let scene = quad_scene();
        let (w, h) = (320, 200);
        let mut frame = Frame::new(w, h);
        let settings = settings();
        let lights = [PointLight { position: Vec3::new(0.0, 0.0, 3.0), color: Vec3::ONE, lumens: 100.0 }];
        render(&scene, &camera(w, h), &lights, &settings, &mut frame);
        assert_eq!(frame.stats.tris_drawn, 2);
        assert_ne!(frame.tri_id[h / 2 * w + w / 2], NO_TRI);
        assert_eq!(frame.tri_id[0], NO_TRI);
        let (p, n) = unproject(&frame, &scene, w / 2, h / 2).unwrap();
        assert!(p.length() < 0.02, "centre pixel unprojects to the origin, got {p}");
        assert!((n.z.abs() - 1.0).abs() < 1e-3);
    }

    #[test]
    fn triangle_behind_camera_is_clipped_not_drawn_wrong() {
        let mut scene = Scene::new();
        // One triangle crossing the camera plane: two vertices in front, one behind.
        let p = [Vec3::new(-1.0, -1.0, 0.0), Vec3::new(1.0, -1.0, 0.0), Vec3::new(0.0, 1.0, 10.0)];
        scene.push_mesh(&p, &[], &[], &[0, 1, 2]);
        let (w, h) = (160, 100);
        let mut frame = Frame::new(w, h);
        let settings = settings();
        render(&scene, &camera(w, h), &[], &settings, &mut frame);
        // Clipping yields one or two pieces, and every drawn pixel has finite depth.
        assert!(frame.stats.tris_drawn >= 1 && frame.stats.tris_drawn <= 2);
        assert!(frame.depth.iter().all(|d| d.is_finite()));
    }

    #[test]
    fn lights_fall_off_with_distance_squared() {
        let scene = quad_scene();
        let (w, h) = (160, 100);
        let settings = settings();
        let centre = h / 2 * w + w / 2;
        let mut dark = Frame::new(w, h);
        render(&scene, &camera(w, h), &[], &settings, &mut dark);
        assert_ne!(dark.tri_id[centre], NO_TRI);
        assert_eq!(dark.color[centre], 0, "no lights, no sky: the quad is black");
        let mut lit = Frame::new(w, h);
        let near_light = [PointLight { position: Vec3::new(0.0, 0.0, 0.5), color: Vec3::ONE, lumens: 50.0 }];
        render(&scene, &camera(w, h), &near_light, &settings, &mut lit);
        let far_light = [PointLight { position: Vec3::new(0.0, 0.0, 2.0), color: Vec3::ONE, lumens: 50.0 }];
        let mut dim = Frame::new(w, h);
        render(&scene, &camera(w, h), &far_light, &settings, &mut dim);
        let g = |c: u32| (c >> 8) & 0xff;
        assert!(g(lit.color[centre]) > g(dim.color[centre]), "a closer light is brighter (1/d²)");
        assert!(g(dim.color[centre]) > 0);
    }

    #[test]
    fn clusters_outside_the_view_are_skipped() {
        let mut scene = Scene::new();
        let (p, idx) = crate::scene::tests::uv_sphere(64);
        // One sphere in front of the camera, one far behind it.
        scene.push_mesh(&p, &[], &[], &idx);
        let behind: Vec<Vec3> = p.iter().map(|v| *v * 0.5 + Vec3::new(0.0, 0.0, 20.0)).collect();
        scene.push_mesh(&behind, &[], &[], &idx);
        let (w, h) = (200, 150);
        let mut frame = Frame::new(w, h);
        let settings = settings();
        render(&scene, &camera(w, h), &[], &settings, &mut frame);
        assert!(frame.stats.clusters_drawn * 2 <= frame.stats.clusters_in + 1, "{:?}", frame.stats);
        assert!(frame.stats.clusters_drawn > 0);
        // Back-face culling drops roughly the far half of the front sphere too.
        let before = frame.stats.clusters_drawn;
        let settings = RenderSettings { cull_backfaces: true, ..settings };
        let mut culled = Frame::new(w, h);
        render(&scene, &camera(w, h), &[], &settings, &mut culled);
        assert!(culled.stats.clusters_drawn < before, "{} vs {}", culled.stats.clusters_drawn, before);
        // And the picture is the same: no visible pixel is lost.
        let covered = |f: &Frame| f.tri_id.iter().filter(|t| **t != NO_TRI).count();
        assert_eq!(covered(&frame), covered(&culled));
    }

    #[test]
    fn pixel_ray_goes_through_projected_point() {
        let (w, h) = (320, 200);
        let cam = camera(w, h);
        let p = Vec3::new(0.3, -0.2, 0.4);
        let (x, y, _) = project(&cam, w, h, p).unwrap();
        let (o, d) = pixel_ray(&cam, w, h, x, y);
        let closest = o + d * (p - o).dot(d);
        assert!(closest.distance(p) < 1e-3, "{closest} vs {p}");
    }

    #[test]
    fn the_sky_lights_near_and_far_surfaces_alike() {
        // A small card 3 m away and a huge one 5 km away, both facing the
        // camera, same albedo: lit by the sky only they read the same.
        let mut scene = Scene::new();
        let card = |c: Vec3, half: f32| [c + Vec3::new(-half, -half, 0.0), c + Vec3::new(half, -half, 0.0), c + Vec3::new(half, half, 0.0), c + Vec3::new(-half, half, 0.0)];
        scene.push_mesh(&card(Vec3::new(-0.3, 0.0, 0.0), 0.2), &[], &[Vec3::splat(0.5)], &[0, 1, 2, 0, 2, 3]);
        scene.push_mesh(&card(Vec3::new(1500.0, 0.0, -5000.0), 900.0), &[], &[Vec3::splat(0.5)], &[0, 1, 2, 0, 2, 3]);
        let (w, h) = (200, 100);
        let mut frame = Frame::new(w, h);
        let settings = RenderSettings::new(Sky { lux: 500.0, on: true });
        render(&scene, &camera(w, h), &[], &settings, &mut frame);
        let near = frame.color[h / 2 * w + w / 2 - 10];
        let far = frame.color[h / 2 * w + w / 2 + 30];
        assert_ne!(frame.tri_id[h / 2 * w + w / 2 - 10], frame.tri_id[h / 2 * w + w / 2 + 30]);
        assert_eq!(near, far);
        assert_ne!(near, 0);
    }

    #[test]
    fn reach_is_where_illuminance_meets_the_cut() {
        let sky = Sky { lux: 30.0, on: true };
        let cut = cut_lux(&sky, default_exposure(sky.lux));
        for lm in [12.0, 400.0, 12_000.0, crate::scene::MAX_LUMENS] {
            let l = PointLight { position: Vec3::ZERO, color: Vec3::ONE, lumens: lm };
            let r = light_reach(&l, cut);
            assert!((l.lux_at(r) / cut - 1.0).abs() < 1e-3);
        }
        // Sky off: only the black-level floor, so lights reach further.
        let dark = Sky { on: false, ..sky };
        assert!(cut_lux(&dark, default_exposure(sky.lux)) < cut);
    }

    #[test]
    fn lights_are_skipped_on_clusters_out_of_reach() {
        let mut scene = Scene::new();
        let (p, idx) = crate::scene::tests::uv_sphere(32);
        scene.push_mesh(&p, &[], &[], &idx);
        // Off to the side so the near sphere does not hide it.
        let far: Vec<Vec3> = p.iter().map(|v| *v * 50.0 + Vec3::new(180.0, 0.0, -400.0)).collect();
        scene.push_mesh(&far, &[], &[], &idx);
        let lights = [PointLight { position: Vec3::new(0.0, 1.5, 1.5), color: Vec3::ONE, lumens: 400.0 }];
        let (w, h) = (200, 150);
        let settings = RenderSettings::new(Sky { lux: 30.0, on: true });
        let mut culled = Frame::new(w, h);
        render(&scene, &camera(w, h), &lights, &settings, &mut culled);
        let mut full = Frame::new(w, h);
        render(&scene, &camera(w, h), &lights, &RenderSettings { light_reach: false, ..RenderSettings::new(Sky { lux: 30.0, on: true }) }, &mut full);
        assert!(culled.stats.light_evals < full.stats.light_evals, "{:?} vs {:?}", culled.stats, full.stats);
        assert_eq!(full.stats.light_evals, full.stats.light_evals_all);
        // The picture does not change by more than one 8-bit step anywhere.
        let mut worst = 0;
        for (a, b) in culled.color.iter().zip(&full.color) {
            for sh in [0, 8, 16] {
                worst = worst.max((((a >> sh) & 0xff) as i32 - ((b >> sh) & 0xff) as i32).abs());
            }
        }
        assert!(worst <= 1, "worst channel difference {worst}");
    }
}
