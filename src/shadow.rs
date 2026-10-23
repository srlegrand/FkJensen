//! Shadows from cube shadow maps, one per point light (Williams 1978),
//! filtered for soft edges (Reeves, Salesin, Cook 1987). No rays.
//!
//! A light's map only holds the geometry inside its reach: anything that
//! can shadow a point the light reaches is nearer the light than that point,
//! so it is inside the reach too. A table lamp's map holds the table and the
//! chess set, never the café or the mountains. Maps are kept between frames
//! and rebuilt only when their light moves or changes strength.

use glam::{Vec3, Vec4};
use rayon::prelude::*;

use crate::scene::Scene;
use crate::simd::{bits, pcf3, F, I, V3};

/// Texels per cube face edge, by default (`--shadow-res` changes it).
pub const SHADOW_RES: usize = 512;
/// Nothing nearer the light than this is drawn into its map, in metres.
const NEAR: f32 = 0.002;

/// The six faces: (axis, right, up).
const FACES: [(Vec3, Vec3, Vec3); 6] = [
    (Vec3::X, Vec3::NEG_Z, Vec3::Y),
    (Vec3::NEG_X, Vec3::Z, Vec3::Y),
    (Vec3::Y, Vec3::X, Vec3::NEG_Z),
    (Vec3::NEG_Y, Vec3::X, Vec3::Z),
    (Vec3::Z, Vec3::X, Vec3::Y),
    (Vec3::NEG_Z, Vec3::NEG_X, Vec3::Y),
];

pub struct CubeShadow {
    pub position: Vec3,
    res: usize,
    /// Per face, per texel: 1 / distance along the face axis to the nearest
    /// surface, 0 where there is none.
    inv_depth: Vec<f32>,
    /// Triangles drawn when it was built.
    pub tris: u64,
}

/// Which face a direction falls on.
fn face_of(q: Vec3) -> usize {
    let a = q.abs();
    if a.x >= a.y && a.x >= a.z {
        if q.x >= 0.0 { 0 } else { 1 }
    } else if a.y >= a.z {
        if q.y >= 0.0 { 2 } else { 3 }
    } else if q.z >= 0.0 {
        4
    } else {
        5
    }
}

impl CubeShadow {
    /// Draw every cluster inside `reach` of `position` into the six faces,
    /// at the level of detail whose error is under a texel as seen from
    /// the light.
    pub fn build(scene: &Scene, position: Vec3, reach: f32, res: usize) -> CubeShadow {
        // A face spans 90 degrees: res / 2 texels per unit of tangent.
        let lod_scale = crate::scene::perspective_scale(position, res as f32 * 0.5, 1e-3);
        let near_clusters: Vec<usize> = (0..scene.clusters.len())
            .filter(|&i| {
                let c = &scene.clusters[i];
                c.centre.distance(position) <= c.radius + reach && c.lod_drawn(1.0, &lod_scale)
            })
            .collect();
        let faces: Vec<(Vec<f32>, u64)> = (0..6)
            .into_par_iter()
            .map(|f| draw_face(scene, &near_clusters, position, f, res))
            .collect();
        let mut inv_depth = Vec::with_capacity(6 * res * res);
        let mut tris = 0;
        for (d, t) in faces {
            inv_depth.extend(d);
            tris += t;
        }
        CubeShadow { position, res, inv_depth, tris }
    }

    #[cfg_attr(not(target_pointer_width = "64"), allow(dead_code))]
    /// Texels per face edge and the faces' 1 / depth, face by face in the
    /// order (+X, -X, +Y, -Y, +Z, -Z), rows from the face's lower edge.
    pub fn faces(&self) -> (usize, &[f32]) {
        (self.res, &self.inv_depth)
    }

    /// How much of the light reaches point `p` with normal `n`: 1 lit, 0 in
    /// shadow, in between at a soft edge. 3 x 3 percentage-closer filter.
    pub fn lit(&self, p: Vec3, n: Vec3) -> f32 {
        let res = self.res as f32;
        let to_light = self.position - p;
        let dist = to_light.length();
        if dist < 1e-6 {
            return 1.0;
        }
        // Normal offset: push the lookup off the surface by about a texel
        // and a half at this distance, towards the light's side, so a
        // surface does not shadow itself.
        let n = if n.dot(to_light) < 0.0 { -n } else { n };
        let texel = 2.0 * dist / res;
        let q = p + n * (texel * 1.5) - self.position;
        let f = face_of(q);
        let (a, r, u) = FACES[f];
        let w = q.dot(a);
        if w <= NEAR {
            return 1.0;
        }
        let sx = (q.dot(r) / w * 0.5 + 0.5) * res - 0.5;
        let sy = (q.dot(u) / w * 0.5 + 0.5) * res - 0.5;
        let (cx, cy) = (sx.round() as i32, sy.round() as i32);
        let face = &self.inv_depth[f * self.res * self.res..(f + 1) * self.res * self.res];
        // An occluder must be nearer than the point by a small share of the
        // distance.
        let limit = 1.0 / (w * (1.0 - 0.004));
        let mut lit = 0.0;
        for dy in -1..=1 {
            for dx in -1..=1 {
                let x = (cx + dx).clamp(0, self.res as i32 - 1) as usize;
                let y = (cy + dy).clamp(0, self.res as i32 - 1) as usize;
                if face[y * self.res + x] <= limit {
                    lit += 1.0;
                }
            }
        }
        lit / 9.0
    }

    /// `lit` for eight points at once, for the lanes in `mask` (others
    /// come back 0).
    pub fn lit8(&self, p: V3, n: V3, mask: u32) -> [f32; 8] {
        let res = self.res as f32;
        let to_light = V3::splat(self.position).sub(p);
        let dist = to_light.length();
        let n = n.facing(to_light);
        let texel = dist * 2.0 / res;
        let q = p.add(n.scale(texel * 1.5)).sub(V3::splat(self.position));
        // The face each lane falls on, as `face_of`, without branches:
        // w along the face's axis, (s, t) along its right and up.
        let (ax, ay, az) = (q.x.abs(), q.y.abs(), q.z.abs());
        let on_x = ax.simd_ge(ay) & ax.simd_ge(az);
        let on_y = !on_x & ay.simd_ge(az);
        let on_z = !(on_x | on_y);
        let (px, py, pz) = (q.x.simd_ge(F::ZERO), q.y.simd_ge(F::ZERO), q.z.simd_ge(F::ZERO));
        // +X (-z, y)  -X (z, y)  +Y (x, -z)  -Y (x, z)  +Z (x, y)  -Z (-x, y)
        let w = on_x.select(ax, on_y.select(ay, az));
        let sx = on_x.select(px.select(-q.z, q.z), on_y.select(q.x, pz.select(q.x, -q.x)));
        let sy = on_y.select(py.select(-q.z, q.z), q.y);
        let face = on_x.select(px.select(F::splat(0.0), F::splat(1.0)), on_y.select(py.select(F::splat(2.0), F::splat(3.0)), pz.select(F::splat(4.0), F::splat(5.0))));
        let _ = on_z;
        let w_safe = w.max(F::splat(1e-30));
        let cx = ((sx / w_safe * 0.5 + 0.5) * res - 0.5).round_int();
        let cy = ((sy / w_safe * 0.5 + 0.5) * res - 0.5).round_int();
        let base = face.round_int() * I::splat((self.res * self.res) as i32);
        // Lanes at the light, or nearer it than the maps start, are lit.
        let near = bits(dist.simd_lt(F::splat(1e-6)) | w.simd_le(F::splat(NEAR)));
        let look = mask & !near;
        let limit = (F::ONE / (w * (1.0 - 0.004)).max(F::splat(1e-30))).to_array();
        let mut out = pcf3(&self.inv_depth, self.res, base, cx, cy, look, |d, j| d <= limit[j]);
        for (j, o) in out.iter_mut().enumerate() {
            if mask & near & (1 << j) != 0 {
                *o = 1.0;
            }
        }
        out
    }

    /// Draw the map again where the boxes `moved` fall on it (where an
    /// object was and is), with everything inside `reach` that covers
    /// those texels. Returns the triangles drawn.
    pub fn redraw(&mut self, scene: &Scene, moved: &[(Vec3, Vec3)], reach: f32) -> u64 {
        let (pos, res) = (self.position, self.res);
        let lod_scale = crate::scene::perspective_scale(pos, res as f32 * 0.5, 1e-3);
        let near_clusters: Vec<usize> = (0..scene.clusters.len())
            .into_par_iter()
            .filter(|&i| {
                let c = &scene.clusters[i];
                c.centre.distance(pos) <= c.radius + reach && c.lod_drawn(1.0, &lod_scale)
            })
            .collect();
        let tris: u64 = self
            .inv_depth
            .par_chunks_mut(res * res)
            .enumerate()
            .map(|(f, face)| {
                // One rect round every box on this face, drawn once.
                let rect = moved.iter().filter_map(|&(lo, hi)| face_rect(pos, f, res, lo, hi)).reduce(|a, b| [a[0].min(b[0]), a[1].min(b[1]), a[2].max(b[2]), a[3].max(b[3])]);
                let Some(rect) = rect else { return 0 };
                for y in rect[1]..=rect[3] {
                    face[y * res + rect[0]..=y * res + rect[2]].fill(0.0);
                }
                let over = |o: &[usize; 4]| o[0] <= rect[2] && o[2] >= rect[0] && o[1] <= rect[3] && o[3] >= rect[1];
                let hit: Vec<usize> = near_clusters
                    .iter()
                    .copied()
                    .filter(|&i| {
                        let c = &scene.clusters[i];
                        face_rect(pos, f, res, c.centre - Vec3::splat(c.radius), c.centre + Vec3::splat(c.radius)).is_some_and(|o| over(&o))
                    })
                    .collect();
                draw_face_into(face, scene, &hit, pos, f, res, rect)
            })
            .sum();
        tris
    }

    pub fn bytes(&self) -> usize {
        self.inv_depth.len() * 4
    }
}

fn draw_face(scene: &Scene, clusters: &[usize], pos: Vec3, f: usize, res: usize) -> (Vec<f32>, u64) {
    let mut buf = vec![0f32; res * res];
    let tris = draw_face_into(&mut buf, scene, clusters, pos, f, res, [0, 0, res - 1, res - 1]);
    (buf, tris)
}

/// The texels of face `f` a box covers, as seen from `pos`: x0, y0, x1,
/// y1 inclusive; the whole face when the box reaches the light; None when
/// it is not on this face.
fn face_rect(pos: Vec3, f: usize, res: usize, lo: Vec3, hi: Vec3) -> Option<[usize; 4]> {
    let (a, r, u) = FACES[f];
    let rs = res as f32;
    let (mut x0, mut y0, mut x1, mut y1) = (f32::INFINITY, f32::INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
    let mut behind = 0;
    for k in 0..8 {
        let p = Vec3::new(if k & 1 == 0 { lo.x } else { hi.x }, if k & 2 == 0 { lo.y } else { hi.y }, if k & 4 == 0 { lo.z } else { hi.z });
        let q = p - pos;
        let w = q.dot(a);
        if w <= NEAR {
            behind += 1;
            continue;
        }
        let (sx, sy) = ((q.dot(r) / w * 0.5 + 0.5) * rs, (q.dot(u) / w * 0.5 + 0.5) * rs);
        x0 = x0.min(sx);
        y0 = y0.min(sy);
        x1 = x1.max(sx);
        y1 = y1.max(sy);
    }
    if behind == 8 {
        return None;
    }
    if behind > 0 {
        return Some([0, 0, res - 1, res - 1]);
    }
    if x1 < 0.0 || y1 < 0.0 || x0 >= rs || y0 >= rs {
        return None;
    }
    let c = |v: f32| (v.max(0.0) as usize).min(res - 1);
    Some([c(x0.floor()), c(y0.floor()), c(x1.ceil()), c(y1.ceil())])
}

/// Draw `clusters` into face `f` of `buf`, only the texels in `rect`
/// (x0, y0, x1, y1 inclusive). Returns the triangles drawn.
fn draw_face_into(buf: &mut [f32], scene: &Scene, clusters: &[usize], pos: Vec3, f: usize, res: usize, rect: [usize; 4]) -> u64 {
    let (a, r, u) = FACES[f];
    let mut tris = 0u64;
    let s2 = std::f32::consts::FRAC_1_SQRT_2;
    let planes = [(a - r) * s2, (a + r) * s2, (a - u) * s2, (a + u) * s2];
    let mut clip = [Vec4::ZERO; crate::scene::CLUSTER_VERTS];
    let mut poly: Vec<Vec4> = Vec::with_capacity(4);
    for &ci in clusters {
        let c = &scene.clusters[ci];
        let qc = c.centre - pos;
        if planes.iter().any(|n| n.dot(qc) < -c.radius) || qc.dot(a) < -c.radius {
            continue;
        }
        for (k, p) in scene.cluster_pos(c).enumerate() {
            let q = p - pos;
            clip[k] = Vec4::new(q.dot(r), q.dot(u), 0.0, q.dot(a));
        }
        for tri in c.first_tri..c.first_tri + c.tri_count {
            let v = scene.tris[tri as usize].map(|l| clip[l as usize]);
            if (v[0].w < NEAR && v[1].w < NEAR && v[2].w < NEAR)
                || (v[0].x > v[0].w && v[1].x > v[1].w && v[2].x > v[2].w)
                || (v[0].x < -v[0].w && v[1].x < -v[1].w && v[2].x < -v[2].w)
                || (v[0].y > v[0].w && v[1].y > v[1].w && v[2].y > v[2].w)
                || (v[0].y < -v[0].w && v[1].y < -v[1].w && v[2].y < -v[2].w)
            {
                continue;
            }
            tris += 1;
            // Clip against the near plane.
            poly.clear();
            for i in 0..3 {
                let (p, q) = (v[i], v[(i + 1) % 3]);
                if p.w >= NEAR {
                    poly.push(p);
                }
                if (p.w >= NEAR) != (q.w >= NEAR) {
                    let t = (NEAR - p.w) / (q.w - p.w);
                    poly.push(p.lerp(q, t));
                }
            }
            for k in 1..poly.len().saturating_sub(1) {
                fill(buf, res, poly[0], poly[k], poly[k + 1], rect);
            }
        }
    }
    tris
}

/// Rasterise one triangle into a face, keeping the largest 1/w.
fn fill(buf: &mut [f32], res: usize, a: Vec4, b: Vec4, c: Vec4, clip: [usize; 4]) {
    let rs = res as f32;
    let to = |p: Vec4| (((p.x / p.w) * 0.5 + 0.5) * rs, ((p.y / p.w) * 0.5 + 0.5) * rs, 1.0 / p.w);
    let (ax, ay, az) = to(a);
    let (bx, by, bz) = to(b);
    let (cx, cy, cz) = to(c);
    let area = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay);
    if area == 0.0 || !area.is_finite() {
        return;
    }
    let x0 = (ax.min(bx).min(cx).floor().max(0.0) as usize).max(clip[0]);
    let y0 = (ay.min(by).min(cy).floor().max(0.0) as usize).max(clip[1]);
    let x1 = ((ax.max(bx).max(cx).ceil().min(rs - 1.0)).max(0.0) as usize).min(clip[2]);
    let y1 = ((ay.max(by).max(cy).ceil().min(rs - 1.0)).max(0.0) as usize).min(clip[3]);
    if x0 > x1 || y0 > y1 {
        return;
    }
    crate::simd::fill_max(buf, res, (ax, ay, az), (bx, by, bz), (cx, cy, cz), x0, x1, y0, y1, 1.0 / area);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A floor and a small card above it, a light above the card.
    fn blocker_scene() -> Scene {
        let mut s = Scene::new();
        let quad = |c: Vec3, h: f32| [c + Vec3::new(-h, 0.0, h), c + Vec3::new(h, 0.0, h), c + Vec3::new(h, 0.0, -h), c + Vec3::new(-h, 0.0, -h)];
        s.push_mesh(&quad(Vec3::ZERO, 2.0), &[], &[], &[0, 1, 2, 0, 2, 3]);
        s.push_mesh(&quad(Vec3::new(0.0, 0.5, 0.0), 0.2), &[], &[], &[0, 1, 2, 0, 2, 3]);
        s
    }

    #[test]
    fn a_card_shadows_the_floor_under_it_only() {
        let s = blocker_scene();
        let light = Vec3::new(0.0, 1.5, 0.0);
        let m = CubeShadow::build(&s, light, 10.0, 256);
        assert_eq!(m.lit(Vec3::new(0.0, 0.0, 0.0), Vec3::Y), 0.0, "under the card");
        assert_eq!(m.lit(Vec3::new(1.2, 0.0, 0.3), Vec3::Y), 1.0, "beside it");
        assert_eq!(m.lit(Vec3::new(0.05, 0.5, 0.05), Vec3::Y), 1.0, "the card does not shadow itself");
        // Directly on every face of the cube.
        for d in [Vec3::X, Vec3::NEG_X, Vec3::Z, Vec3::NEG_Z] {
            let p = Vec3::new(d.x * 1.5, 0.0, d.z * 1.5);
            assert_eq!(m.lit(p, Vec3::Y), 1.0, "{p}");
        }
    }

    /// Eight at a time gives what one at a time gives.
    #[test]
    fn eight_lookups_match_one() {
        let s = blocker_scene();
        let m = CubeShadow::build(&s, Vec3::new(0.0, 1.5, 0.0), 10.0, 128);
        let mut pts = Vec::new();
        for i in 0..64 {
            let t = i as f32 * 0.37;
            pts.push((Vec3::new(t.sin() * 1.8, (i % 3) as f32 * 0.25, (t * 1.3).cos() * 1.8), Vec3::new(0.1 * t.cos(), 1.0, 0.0).normalize()));
        }
        // And the light's own position.
        pts[5].0 = Vec3::new(0.0, 1.5, 0.0);
        for chunk in pts.chunks(8) {
            let lane = |f: fn(&(Vec3, Vec3)) -> f32| F::from(std::array::from_fn::<f32, 8, _>(|j| f(&chunk[j])));
            let p = V3 { x: lane(|c| c.0.x), y: lane(|c| c.0.y), z: lane(|c| c.0.z) };
            let n = V3 { x: lane(|c| c.1.x), y: lane(|c| c.1.y), z: lane(|c| c.1.z) };
            let got = m.lit8(p, n, 0xff);
            for j in 0..8 {
                assert_eq!(got[j], m.lit(chunk[j].0, chunk[j].1), "{:?}", chunk[j]);
            }
        }
    }

    /// Moving the card and drawing the map again where it was and is
    /// gives the map of a fresh build.
    #[test]
    fn redrawing_where_something_moved_matches_a_fresh_map() {
        let mut s = blocker_scene();
        let light = Vec3::new(0.0, 1.5, 0.0);
        let mut m = CubeShadow::build(&s, light, 10.0, 128);
        // The card is the second cluster: slide it 0.6 m along x.
        let c = 1;
        let bounds = |s: &Scene| {
            let ps: Vec<Vec3> = s.cluster_pos(&s.clusters[c]).collect();
            (ps.iter().fold(Vec3::INFINITY, |a, p| a.min(*p)), ps.iter().fold(Vec3::NEG_INFINITY, |a, p| a.max(*p)))
        };
        let from = bounds(&s);
        s.map_vertices(c..c + 1, |p| p + Vec3::X * 0.6, None);
        s.clusters[c].centre.x += 0.6;
        let to = bounds(&s);
        m.redraw(&s, &[from, to], 10.0);
        let fresh = CubeShadow::build(&s, light, 10.0, 128);
        assert_eq!(m.inv_depth, fresh.inv_depth);
        assert_eq!(m.lit(Vec3::new(0.0, 0.0, 0.0), Vec3::Y), 1.0, "where it was is lit");
    }

    #[test]
    fn geometry_out_of_reach_is_not_drawn() {
        let s = blocker_scene();
        let far = CubeShadow::build(&s, Vec3::new(0.0, 5.0, 0.0), 0.5, 64);
        assert_eq!(far.tris, 0, "neither the card nor the floor is within 0.5 m");
        let near = CubeShadow::build(&s, Vec3::new(0.0, 1.5, 0.0), 10.0, 64);
        assert!(near.tris >= 4);
    }

    #[test]
    fn faces_cover_every_direction() {
        for i in 0..500 {
            let t = i as f32 * 0.731;
            let q = Vec3::new(t.sin(), (t * 1.3).cos(), (t * 0.7).sin() * (t * 2.1).cos());
            let (a, r, u) = FACES[face_of(q)];
            let w = q.dot(a);
            assert!(w > 0.0);
            assert!((q.dot(r) / w).abs() <= 1.0 + 1e-5 && (q.dot(u) / w).abs() <= 1.0 + 1e-5);
        }
    }
}
