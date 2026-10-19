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

/// Texels per cube face edge.
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
    /// Draw every cluster inside `reach` of `position` into the six faces.
    pub fn build(scene: &Scene, position: Vec3, reach: f32, res: usize) -> CubeShadow {
        let near_clusters: Vec<usize> = (0..scene.clusters.len())
            .filter(|&i| {
                let c = &scene.clusters[i];
                c.centre.distance(position) <= c.radius + reach
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

    pub fn bytes(&self) -> usize {
        self.inv_depth.len() * 4
    }
}

fn draw_face(scene: &Scene, clusters: &[usize], pos: Vec3, f: usize, res: usize) -> (Vec<f32>, u64) {
    let (a, r, u) = FACES[f];
    let mut buf = vec![0f32; res * res];
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
        let fv = c.first_vertex as usize;
        for (k, p) in scene.positions[fv..fv + c.vertex_count as usize].iter().enumerate() {
            let q = *p - pos;
            clip[k] = Vec4::new(q.dot(r), q.dot(u), 0.0, q.dot(a));
        }
        for tri in c.first_tri..c.first_tri + c.tri_count {
            let t = tri as usize * 3;
            let v = [clip[scene.indices[t] as usize - fv], clip[scene.indices[t + 1] as usize - fv], clip[scene.indices[t + 2] as usize - fv]];
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
                fill(&mut buf, res, poly[0], poly[k], poly[k + 1]);
            }
        }
    }
    (buf, tris)
}

/// Rasterise one triangle into a face, keeping the largest 1/w.
fn fill(buf: &mut [f32], res: usize, a: Vec4, b: Vec4, c: Vec4) {
    let rs = res as f32;
    let to = |p: Vec4| (((p.x / p.w) * 0.5 + 0.5) * rs, ((p.y / p.w) * 0.5 + 0.5) * rs, 1.0 / p.w);
    let (ax, ay, az) = to(a);
    let (bx, by, bz) = to(b);
    let (cx, cy, cz) = to(c);
    let area = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay);
    if area == 0.0 || !area.is_finite() {
        return;
    }
    let x0 = ax.min(bx).min(cx).floor().max(0.0) as usize;
    let y0 = ay.min(by).min(cy).floor().max(0.0) as usize;
    let x1 = (ax.max(bx).max(cx).ceil().min(rs - 1.0)).max(0.0) as usize;
    let y1 = (ay.max(by).max(cy).ceil().min(rs - 1.0)).max(0.0) as usize;
    if x0 > x1 || y0 > y1 {
        return;
    }
    let inv = 1.0 / area;
    for y in y0..=y1 {
        let py = y as f32 + 0.5;
        for x in x0..=x1 {
            let px = x as f32 + 0.5;
            let w0 = ((cx - bx) * (py - by) - (cy - by) * (px - bx)) * inv;
            let w1 = ((ax - cx) * (py - cy) - (ay - cy) * (px - cx)) * inv;
            let w2 = ((bx - ax) * (py - ay) - (by - ay) * (px - ax)) * inv;
            if w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0 {
                let z = w0 * az + w1 * bz + w2 * cz;
                let i = y * res + x;
                if z > buf[i] {
                    buf[i] = z;
                }
            }
        }
    }
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
