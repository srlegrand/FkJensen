//! A tile-binned CPU rasteriser with a visibility buffer and deferred shading.
//!
//! One frame is four passes, each parallel across cores with rayon:
//!
//! 1. **Setup**: every triangle is projected, clipped against the near plane,
//!    optionally back-face culled, and gets a screen-space record.
//! 2. **Bin**: each record is listed in every 64 x 64 tile its box touches.
//! 3. **Visibility**: each tile is rasterised by one thread into its own part
//!    of the depth and triangle-id buffers. No atomics, no locks: a tile
//!    belongs to one thread for the pass.
//! 4. **Shade**: each pixel that has a triangle reconstructs its barycentric
//!    weights, interpolates position, normal and colour, and is lit once.
//!
//! Shading after visibility is what keeps the cost proportional to pixels
//! rather than to triangles: a triangle covered by another is never shaded.
//!
//! No SIMD yet: the inner loops are plain scalar Rust. That is the next step
//! and the layout (SoA records, per-tile ownership) is chosen for it.

use glam::{Mat4, Vec2, Vec3, Vec4};
use rayon::prelude::*;

use crate::scene::{Light, Scene};

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
    pub ambient: f32,
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
    // Signed area in screen space (y down): positive = clockwise on screen =
    // counter-clockwise in a y-up world, the front face.
    let area = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]);
    if area == 0.0 || !area.is_finite() {
        return None;
    }
    if cull && area < 0.0 {
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

pub fn render(scene: &Scene, cam: &Camera, lights: &[Light], settings: &RenderSettings, frame: &mut Frame) {
    let width = frame.width;
    let height = frame.height;
    let vp = cam.proj * cam.view;
    let t0 = std::time::Instant::now();

    // Pass 1: project vertices, then set up triangles in parallel chunks.
    let clip: Vec<Vec4> = scene
        .positions
        .par_chunks(4096)
        .flat_map_iter(|chunk| chunk.iter().map(move |p| vp * p.extend(1.0)))
        .collect();

    let cull = settings.cull_backfaces;
    let near = cam.near;
    let setups: Vec<TriSetup> = scene
        .indices
        .par_chunks(3 * 2048)
        .enumerate()
        .flat_map_iter(move |(ci, chunk)| {
            let base_tri = (ci * 2048) as u32;
            let mut out: Vec<TriSetup> = Vec::with_capacity(chunk.len() / 3);
            let mut clipped: Vec<[ClipVert; 3]> = Vec::new();
            for (k, t) in chunk.chunks_exact(3).enumerate() {
                let tri = base_tri + k as u32;
                let p0 = clip[t[0] as usize];
                let p1 = clip[t[1] as usize];
                let p2 = clip[t[2] as usize];
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
                    for c in &clipped {
                        if let Some(s) = setup_tri(tri, c, width, height, cull) {
                            out.push(s);
                        }
                    }
                }
            }
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
    let eye = cam.eye;
    let sky = Vec3::new(0.55, 0.65, 0.80) * settings.ambient;
    let ground = Vec3::new(0.25, 0.22, 0.20) * settings.ambient;
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
                let hemi = n.y * 0.5 + 0.5;
                let mut radiance = albedo * (sky * hemi + ground * (1.0 - hemi));
                for l in lights {
                    let (ldir, irr) = match *l {
                        Light::Point { position, color, intensity } => {
                            let d = position - p;
                            let d2 = d.length_squared().max(1e-6);
                            (d / d2.sqrt(), color * (intensity * r2 / d2))
                        }
                        Light::Distant { direction, color, intensity } => (-direction, color * intensity),
                    };
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

/// Draw a filled disc into the colour buffer, depth-tested at its centre.
pub fn draw_disc(frame: &mut Frame, cam: &Camera, centre: Vec3, radius_px: f32, color: Vec3, ring: bool) {
    let vp = cam.proj * cam.view;
    let c = vp * centre.extend(1.0);
    if c.w <= cam.near {
        return;
    }
    let iw = 1.0 / c.w;
    let sx = (c.x * iw * 0.5 + 0.5) * frame.width as f32;
    let sy = (0.5 - c.y * iw * 0.5) * frame.height as f32;
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

#[allow(dead_code)]
pub fn screen_to_ndc(x: f32, y: f32, width: usize, height: usize) -> Vec2 {
    Vec2::new(x / width as f32 * 2.0 - 1.0, 1.0 - y / height as f32 * 2.0)
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
        let settings = RenderSettings { cull_backfaces: false, wireframe: false, ambient: 0.3, exposure: 1.0, scene_radius: 1.5 };
        let lights = [Light::Point { position: Vec3::new(0.0, 0.0, 3.0), color: Vec3::ONE, intensity: 1.0 }];
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
        let settings = RenderSettings { cull_backfaces: false, wireframe: false, ambient: 0.3, exposure: 1.0, scene_radius: 1.5 };
        render(&scene, &camera(w, h), &[], &settings, &mut frame);
        // Clipping yields one or two pieces, and every drawn pixel has finite depth.
        assert!(frame.stats.tris_drawn >= 1 && frame.stats.tris_drawn <= 2);
        assert!(frame.depth.iter().all(|d| d.is_finite()));
    }
}
