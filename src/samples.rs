//! Surface samples: points spread evenly over a model's surface, the same
//! distance apart whatever its triangles. A floor of two triangles gets as
//! many as a sculpted one of a million over the same area, so whatever is
//! kept at them (sky occlusion, bounce light) is as fine on a low-polygon
//! surface as on a dense one.
//!
//! They are placed by dart throwing (Poisson-disk sampling, Cook 1986):
//! candidates scattered at random over the triangles in proportion to
//! their area, then taken in random order, each kept unless a kept sample
//! on the same side of the surface is closer than the spacing. The two
//! sides of a thin part (a claw, a car's wing) do not crowd each other
//! out. A model's samples come from its finest level of detail, so the
//! cloud does not change with what is drawn.

use std::collections::HashMap;

use glam::Vec3;

/// Closest samples are ever placed, metres.
pub const MIN_SPACING: f32 = 0.01;
/// Most samples on one model: its spacing grows to keep under this.
pub const MAX_SAMPLES: usize = 150_000;
/// Candidates scattered per sample's worth of area. Four is enough for the
/// dart throwing to come close to filling the surface.
const CANDIDATES: f32 = 4.0;
/// Two samples are on the same side of a surface when their normals are
/// closer than this (cosine).
const SAME_SIDE: f32 = 0.3;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sample {
    pub position: Vec3,
    /// Unit normal, from the model's smooth normals.
    pub normal: Vec3,
    /// The surface it stands for, m²: its model's area over its number of
    /// samples.
    pub area: f32,
    /// The scene cluster it lies on (finest level of detail).
    pub cluster: u32,
    /// Its sky transfer (`transfer.rs`): what it sees of the sky as
    /// order-2 spherical harmonics, in world space. NaN when not computed.
    pub sky: [f32; 9],
}

/// Spacing for a model of `area` m².
pub fn spacing_for(area: f32) -> f32 {
    (area / MAX_SAMPLES as f32).sqrt().max(MIN_SPACING)
}

/// A number in [0, 1) from two integers, the same on every run.
fn rand(a: u64, b: u64) -> f32 {
    let mut z = a.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ b.wrapping_add(0x632b_e59b_d9b4_e019).rotate_left(29);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    (z >> 40) as f32 / (1u64 << 24) as f32
}

/// Area of a triangle.
pub fn tri_area(p: [Vec3; 3]) -> f32 {
    (p[1] - p[0]).cross(p[2] - p[0]).length() * 0.5
}

/// Spread samples about `h` apart over `count` triangles, `tri(i)` giving
/// triangle i's cluster, corners and corner normals. `area` is their
/// total area.
pub fn scatter(count: usize, tri: impl Fn(usize) -> (u32, [Vec3; 3], [Vec3; 3]) + Sync, area: f32, h: f32) -> Vec<Sample> {
    use rayon::prelude::*;
    let density = CANDIDATES / (h * h);
    // Candidates, each with a random key that sets the order they are
    // taken in.
    let mut cand: Vec<(f32, Vec3, Vec3, u32)> = (0..count)
        .into_par_iter()
        .with_min_len(4096)
        .flat_map_iter(|i| {
            let (cluster, p, n) = tri(i);
            let e = tri_area(p) * density;
            let k = e.floor() as u64 + (rand(i as u64, 0) < e.fract()) as u64;
            let face = (p[1] - p[0]).cross(p[2] - p[0]).normalize_or(Vec3::Y);
            (0..k).map(move |j| {
                // Uniform over the triangle.
                let (r1, r2) = (rand(i as u64, 2 * j + 1).sqrt(), rand(i as u64, 2 * j + 2));
                let (a, b, c) = (1.0 - r1, r1 * (1.0 - r2), r1 * r2);
                let pos = p[0] * a + p[1] * b + p[2] * c;
                let nrm = (n[0] * a + n[1] * b + n[2] * c).normalize_or(face);
                (rand(i as u64 ^ 0x5555, j), pos, nrm, cluster)
            })
        })
        .collect();
    cand.par_sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    // Dart throwing on a grid of cells `h` wide.
    let cell = |p: Vec3| ((p.x / h).floor() as i32, (p.y / h).floor() as i32, (p.z / h).floor() as i32);
    let mut grid: HashMap<(i32, i32, i32), Vec<u32>> = HashMap::with_capacity(cand.len() / 3);
    let mut kept: Vec<(Vec3, Vec3, u32)> = Vec::new();
    let h2 = h * h;
    for &(_, p, n, c) in &cand {
        let (x, y, z) = cell(p);
        let mut free = true;
        'look: for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if let Some(v) = grid.get(&(x + dx, y + dy, z + dz)) {
                        for &k in v {
                            let (q, m, _) = kept[k as usize];
                            if q.distance_squared(p) < h2 && m.dot(n) > SAME_SIDE {
                                free = false;
                                break 'look;
                            }
                        }
                    }
                }
            }
        }
        if free {
            grid.entry((x, y, z)).or_default().push(kept.len() as u32);
            kept.push((p, n, c));
        }
    }
    let each = if kept.is_empty() { 0.0 } else { area / kept.len() as f32 };
    let mut out: Vec<Sample> = kept.into_iter().map(|(position, normal, cluster)| Sample { position, normal, area: each, cluster, sky: [f32::NAN; 9] }).collect();
    out.sort_by_key(|s| s.cluster);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quad_tris(c: Vec3, u: Vec3, v: Vec3, n: usize) -> Vec<[Vec3; 3]> {
        let mut out = Vec::new();
        let at = |i: usize, j: usize| c + u * (i as f32 / n as f32 * 2.0 - 1.0) + v * (j as f32 / n as f32 * 2.0 - 1.0);
        for j in 0..n {
            for i in 0..n {
                out.push([at(i, j), at(i + 1, j), at(i + 1, j + 1)]);
                out.push([at(i, j), at(i + 1, j + 1), at(i, j + 1)]);
            }
        }
        out
    }

    fn run(tris: &[[Vec3; 3]], h: f32) -> Vec<Sample> {
        let area: f32 = tris.iter().map(|t| tri_area(*t)).sum();
        scatter(tris.len(), |i| (i as u32, tris[i], [Vec3::ZERO; 3]), area, h)
    }

    /// The same area gets the same number of samples whether it is two
    /// triangles or twenty thousand, and none are closer than the spacing
    /// on the same side.
    #[test]
    fn samples_follow_area_not_triangles() {
        let h = 0.05;
        let coarse = run(&quad_tris(Vec3::ZERO, Vec3::X, Vec3::NEG_Z, 1), h);
        let fine = run(&quad_tris(Vec3::ZERO, Vec3::X, Vec3::NEG_Z, 100), h);
        let (a, b) = (coarse.len() as f32, fine.len() as f32);
        assert!(a > 800.0, "{a}");
        assert!((a / b - 1.0).abs() < 0.05, "{a} on 2 triangles, {b} on 20,000");
        for s in [&coarse, &fine] {
            for (i, p) in s.iter().enumerate() {
                for q in &s[i + 1..] {
                    assert!(p.position.distance(q.position) >= h * 0.999);
                }
            }
        }
        // Spread over the whole square: every 20 cm cell has some.
        for gx in 0..10 {
            for gz in 0..10 {
                let (x0, z0) = (-1.0 + gx as f32 * 0.2, -1.0 + gz as f32 * 0.2);
                assert!(coarse.iter().any(|s| s.position.x >= x0 && s.position.x < x0 + 0.2 && s.position.z >= z0 && s.position.z < z0 + 0.2));
            }
        }
        let total: f32 = coarse.iter().map(|s| s.area).sum();
        assert!((total - 4.0).abs() < 1e-3);
    }

    /// Two sides of a thin sheet 1 mm apart each keep their own samples.
    #[test]
    fn both_sides_of_a_thin_part_are_sampled() {
        let mut tris = quad_tris(Vec3::ZERO, Vec3::X * 0.5, Vec3::NEG_Z * 0.5, 2);
        let under: Vec<[Vec3; 3]> = tris.iter().map(|t| [t[0] - Vec3::Y * 0.001, t[2] - Vec3::Y * 0.001, t[1] - Vec3::Y * 0.001]).collect();
        let n_top = tris.len();
        tris.extend(under);
        let s = run(&tris, 0.05);
        let up = s.iter().filter(|s| s.normal.y > 0.9).count();
        let down = s.iter().filter(|s| s.normal.y < -0.9).count();
        assert!(up > 100 && (up as f32 / down as f32 - 1.0).abs() < 0.15, "{up} up, {down} down");
        assert!(s.iter().all(|s| (s.cluster as usize >= n_top) == (s.normal.y < 0.0)));
    }
}
