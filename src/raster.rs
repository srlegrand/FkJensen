//! A tile-binned CPU rasteriser with a visibility buffer and deferred shading.
//!
//! One frame is four passes, each parallel across cores with rayon:
//!
//! 1. **Setup**: clusters are tested against the view frustum (and, with
//!    back-face culling on, against their normal cone) as a whole; the
//!    vertices of each surviving cluster are projected into a small local
//!    array and its triangles are clipped against the near plane, optionally
//!    back-face culled, and given a screen-space record.
//! 2. **Bin**: each record is listed in every 64 x 64 tile its box touches.
//! 3. **Visibility**: each tile is rasterised by one thread into its own part
//!    of the depth and triangle-id buffers. No atomics, no locks: a tile
//!    belongs to one thread for the pass.
//! 4. **Shade**: each pixel that has a triangle reconstructs its barycentric
//!    weights, interpolates position, normal and colour, and is lit once by
//!    the point lights. There is no ambient term: a surface no light reaches
//!    is black.
//!
//! Shading after visibility is what keeps the cost proportional to pixels
//! rather than to triangles: a triangle covered by another is never shaded.
//!
//! No SIMD yet: the inner loops are plain scalar Rust. That is the next step
//! and the layout (SoA records, per-tile ownership) is chosen for it.

use glam::{Mat4, Vec2, Vec3, Vec4};
use rayon::prelude::*;

use crate::scene::{Cluster, PointLight, Scene};

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
}

#[derive(Default, Clone, Copy, Debug)]
pub struct Stats {
    pub clusters_in: usize,
    pub clusters_drawn: usize,
    pub tris_in: usize,
    pub tris_drawn: usize,
    pub setup_ms: f32,
    pub bin_ms: f32,
    pub raster_ms: f32,
    pub shade_ms: f32,
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
    pub min_x: u16,
    pub min_y: u16,
    pub max_x: u16,
    pub max_y: u16,
}

pub struct RenderSettings {
    pub cull_backfaces: bool,
    pub wireframe: bool,
    pub exposure: f32,
    /// Scene radius, used to scale point light falloff so intensities stay
    /// sensible across scene sizes.
    pub scene_radius: f32,
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
fn frustum_planes(vp: Mat4, near: f32) -> [Vec4; 5] {
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

pub fn render(scene: &Scene, cam: &Camera, lights: &[PointLight], settings: &RenderSettings, frame: &mut Frame) {
    let width = frame.width;
    let height = frame.height;
    let vp = cam.proj * cam.view;
    let t0 = std::time::Instant::now();

    // Pass 1: cull clusters, then project each survivor's vertices into a
    // local array and set up its triangles.
    let cull = settings.cull_backfaces;
    let near = cam.near;
    let planes = frustum_planes(vp, near);
    let eye = cam.eye;
    let clusters_drawn = std::sync::atomic::AtomicUsize::new(0);
    let setups: Vec<TriSetup> = scene
        .clusters
        .par_chunks(64)
        .flat_map_iter(|chunk| {
            let mut out: Vec<TriSetup> = Vec::new();
            let mut clipped: Vec<[ClipVert; 3]> = Vec::new();
            let mut clip = [Vec4::ZERO; crate::scene::CLUSTER_VERTS];
            let mut drawn = 0;
            for c in chunk {
                if cluster_hidden(c, &planes, eye, cull) {
                    continue;
                }
                drawn += 1;
                let fv = c.first_vertex as usize;
                for (k, p) in scene.positions[fv..fv + c.vertex_count as usize].iter().enumerate() {
                    clip[k] = vp * p.extend(1.0);
                }
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
                        if let Some(s) = setup_tri(tri, &v, width, height, cull) {
                            out.push(s);
                        }
                    } else {
                        clipped.clear();
                        clip_near(v, near, &mut clipped);
                        for cv in &clipped {
                            if let Some(s) = setup_tri(tri, cv, width, height, cull) {
                                out.push(s);
                            }
                        }
                    }
                }
            }
            clusters_drawn.fetch_add(drawn, std::sync::atomic::Ordering::Relaxed);
            out.into_iter()
        })
        .collect();
    let t1 = std::time::Instant::now();

    // Pass 2: bin into tiles. Each chunk of records writes its own per-tile
    // lists; the lists are then joined per tile.
    let tiles_x = width.div_ceil(TILE);
    let tiles_y = height.div_ceil(TILE);
    let n_tiles = tiles_x * tiles_y;
    let chunk_lists: Vec<Vec<Vec<u32>>> = setups
        .par_chunks(8192)
        .enumerate()
        .map(|(ci, chunk)| {
            let mut lists: Vec<Vec<u32>> = vec![Vec::new(); n_tiles];
            let base = (ci * 8192) as u32;
            for (k, s) in chunk.iter().enumerate() {
                let id = base + k as u32;
                let tx0 = s.min_x as usize / TILE;
                let tx1 = s.max_x as usize / TILE;
                let ty0 = s.min_y as usize / TILE;
                let ty1 = s.max_y as usize / TILE;
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

    // Pass 3: visibility. Each tile is rasterised into its own rows of the
    // frame buffers through a per-tile scratch, then copied out, so the
    // buffers are written without sharing.
    let tile_results: Vec<(Vec<f32>, Vec<u32>)> = (0..n_tiles)
        .into_par_iter()
        .map(|t| {
            let tx = t % tiles_x;
            let ty = t / tiles_x;
            let x0 = tx * TILE;
            let y0 = ty * TILE;
            let x1 = (x0 + TILE).min(width);
            let y1 = (y0 + TILE).min(height);
            let tw = x1 - x0;
            let th = y1 - y0;
            let mut depth = vec![0f32; tw * th];
            let mut id = vec![NO_TRI; tw * th];
            for &sid in &tile_lists[t] {
                let s = &setups[sid as usize];
                raster_tri(s, sid, x0, y0, x1, y1, tw, &mut depth, &mut id, settings.wireframe);
            }
            (depth, id)
        })
        .collect();
    for (t, (depth, id)) in tile_results.iter().enumerate() {
        let tx = t % tiles_x;
        let ty = t / tiles_x;
        let x0 = tx * TILE;
        let y0 = ty * TILE;
        let x1 = (x0 + TILE).min(width);
        let y1 = (y0 + TILE).min(height);
        let tw = x1 - x0;
        for (row, y) in (y0..y1).enumerate() {
            let dst = y * width + x0;
            frame.depth[dst..dst + tw].copy_from_slice(&depth[row * tw..row * tw + tw]);
            frame.tri_id[dst..dst + tw].copy_from_slice(&id[row * tw..row * tw + tw]);
        }
    }
    let t3 = std::time::Instant::now();

    // Pass 4: shade, one row per task.
    let r2 = settings.scene_radius * settings.scene_radius;
    let exposure = settings.exposure;
    let bg = pack(Vec3::new(0.10, 0.105, 0.115));
    let tri_id = &frame.tri_id;
    let setups_ref = &setups;
    frame
        .color
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row)| {
            for x in 0..width {
                let id = tri_id[y * width + x];
                if id == NO_TRI {
                    row[x] = bg;
                    continue;
                }
                let s = &setups_ref[id as usize];
                let b = barycentrics(s, x as f32 + 0.5, y as f32 + 0.5);
                // Weights over the source triangle's vertices.
                let w0 = b[0] * s.bary[0][0] + b[1] * s.bary[1][0] + b[2] * s.bary[2][0];
                let w1 = b[0] * s.bary[0][1] + b[1] * s.bary[1][1] + b[2] * s.bary[2][1];
                let w2 = b[0] * s.bary[0][2] + b[1] * s.bary[1][2] + b[2] * s.bary[2][2];
                let t = s.tri as usize * 3;
                let (i0, i1, i2) = (scene.indices[t] as usize, scene.indices[t + 1] as usize, scene.indices[t + 2] as usize);
                let p = scene.positions[i0] * w0 + scene.positions[i1] * w1 + scene.positions[i2] * w2;
                let mut n = (scene.normals[i0] * w0 + scene.normals[i1] * w1 + scene.normals[i2] * w2).normalize_or(Vec3::Y);
                let albedo = scene.colors[i0] * w0 + scene.colors[i1] * w1 + scene.colors[i2] * w2;
                let v = (eye - p).normalize_or(Vec3::Y);
                // Two-sided: face the normal towards the viewer.
                if n.dot(v) < 0.0 {
                    n = -n;
                }
                let mut radiance = Vec3::ZERO;
                for l in lights {
                    let d = l.position - p;
                    let d2 = d.length_squared().max(1e-12);
                    let ldir = d / d2.sqrt();
                    let irr = l.color * (l.intensity * r2 / d2);
                    let ndl = n.dot(ldir);
                    if ndl <= 0.0 {
                        continue;
                    }
                    let h = (ldir + v).normalize_or(ldir);
                    let spec = n.dot(h).max(0.0).powf(48.0) * 0.12;
                    radiance += irr * (albedo * ndl + Vec3::splat(spec * ndl));
                }
                row[x] = pack(tonemap(radiance * exposure));
            }
        });
    let t4 = std::time::Instant::now();

    frame.stats = Stats {
        clusters_in: scene.clusters.len(),
        clusters_drawn: clusters_drawn.into_inner(),
        tris_in: scene.triangle_count(),
        tris_drawn: setups.len(),
        setup_ms: (t1 - t0).as_secs_f32() * 1000.0,
        bin_ms: (t2 - t1).as_secs_f32() * 1000.0,
        raster_ms: (t3 - t2).as_secs_f32() * 1000.0,
        shade_ms: (t4 - t3).as_secs_f32() * 1000.0,
    };
    frame.setups = setups;
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
fn tonemap(c: Vec3) -> Vec3 {
    // Simple filmic-ish curve, keeps highlights from clipping to white.
    let c = c.max(Vec3::ZERO);
    c / (Vec3::ONE + c) * 1.25
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

    fn quad_scene() -> Scene {
        let mut s = Scene::new();
        let p = [Vec3::new(-1.0, -1.0, 0.0), Vec3::new(1.0, -1.0, 0.0), Vec3::new(1.0, 1.0, 0.0), Vec3::new(-1.0, 1.0, 0.0)];
        s.push_mesh(&p, &[], &[Vec3::splat(0.8)], &[0, 1, 2, 0, 2, 3]);
        s
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
        let settings = RenderSettings { cull_backfaces: false, wireframe: false, exposure: 1.0, scene_radius: 1.5 };
        let lights = [PointLight { position: Vec3::new(0.0, 0.0, 3.0), color: Vec3::ONE, intensity: 1.0 }];
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
        let settings = RenderSettings { cull_backfaces: false, wireframe: false, exposure: 1.0, scene_radius: 1.5 };
        render(&scene, &camera(w, h), &[], &settings, &mut frame);
        // Clipping yields one or two pieces, and every drawn pixel has finite depth.
        assert!(frame.stats.tris_drawn >= 1 && frame.stats.tris_drawn <= 2);
        assert!(frame.depth.iter().all(|d| d.is_finite()));
    }

    #[test]
    fn only_point_lights_light_the_scene() {
        let scene = quad_scene();
        let (w, h) = (160, 100);
        let settings = RenderSettings { cull_backfaces: false, wireframe: false, exposure: 1.0, scene_radius: 1.5 };
        let centre = h / 2 * w + w / 2;
        let mut dark = Frame::new(w, h);
        render(&scene, &camera(w, h), &[], &settings, &mut dark);
        assert_ne!(dark.tri_id[centre], NO_TRI);
        assert_eq!(dark.color[centre], 0, "no lights, no ambient: the quad is black");
        let mut lit = Frame::new(w, h);
        let near_light = [PointLight { position: Vec3::new(0.0, 0.0, 0.5), color: Vec3::ONE, intensity: 0.05 }];
        render(&scene, &camera(w, h), &near_light, &settings, &mut lit);
        let far_light = [PointLight { position: Vec3::new(0.0, 0.0, 2.0), color: Vec3::ONE, intensity: 0.05 }];
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
        let settings = RenderSettings { cull_backfaces: false, wireframe: false, exposure: 1.0, scene_radius: 1.5 };
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
}
