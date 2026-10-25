//! Shadow fields: each object carries, baked once with its asset, a grid
//! round its geometry from which the shadows it casts are traced, for any
//! light, wherever the object stands. The idea is Simon Legrand's
//! (precomputed per-object shadow data, sharp where the occluder is close,
//! blurred with distance); the grid follows Zhou et al. (precomputed shadow
//! fields, 2005), Wright (mesh distance field shadows, Unreal Engine,
//! 2015) and Quilez's soft-shadow estimate.
//!
//! Two parts, either or both:
//!
//! - **Distance**: in each cell, how far the nearest opaque triangle is
//!   (unsigned, in eighths of a cell, up to 31 cells). A shadow ray steps
//!   by that distance; how close it passes to the surface against how far
//!   it has gone (`d / (k t)`, `k` the light's angular size) is the
//!   penumbra, so a shadow is sharp where the occluder is close to what it
//!   falls on and soft where it is far.
//! - **Density**: for cut-out surfaces (leaves on cards), how much of the
//!   cell's volume their solid parts block: optical depth per cell, from
//!   their solid area as thin plates turned every way (each blocks half its
//!   area on average). A ray through it keeps `exp(-sum)` of the light, so
//!   a canopy is dappled and thins at its edges.
//!
//! Fields are in the asset's own space (as its geometry is stored) and are
//! shared by instances. Rays are traced through them in `fieldtrace.rs`.

use glam::Vec3;

use crate::scene::Scene;

/// A distance byte is this many per cell.
pub const DIST_STEPS: f32 = 8.0;
/// A density byte is this many per unit of optical depth across a cell.
pub const TAU_STEPS: f32 = 64.0;
/// Cells along the longest side of an object's field.
pub const RES: usize = 128;
/// A triangle whose mask is solid over at least this share of it is
/// treated as opaque (distance), else as foliage (density).
pub const OPAQUE_SHARE: f32 = 0.9;

#[derive(Clone, Debug, PartialEq)]
pub struct ShadowField {
    /// The grid's first cell centre, in the asset's space, its cell size
    /// and its cells along x, y and z.
    pub lo: Vec3,
    pub h: f32,
    pub n: [usize; 3],
    /// Distance to the nearest opaque surface per cell (`DIST_STEPS` a
    /// cell), x fastest; empty when the object has no opaque surface.
    pub dist: Vec<u8>,
    /// Optical depth per cell (`TAU_STEPS` per unit); empty when the
    /// object has no cut-out surface.
    pub density: Vec<u8>,
    /// Cells to the nearest cell with anything in it, density or a
    /// surface within a cell (chamfer distance, made when read, not
    /// stored): a ray steps over empty space in one go.
    pub empty: Vec<u8>,
    /// What a ray reads, in bricks of 4 by 4 by 4 cells (one cache line
    /// each), `nb` of them along each axis: `dist`, `density`, `empty`.
    pub(crate) nb: [usize; 3],
    /// The last corner a look-up starts from, and one on along y and z
    /// across a brick's side.
    pub(crate) top: Vec3,
    pub(crate) wrap: [usize; 2],
    pub(crate) bdist: Vec<u8>,
    pub(crate) bdensity: Vec<u8>,
    pub(crate) bempty: Vec<u8>,
}


/// `grid` (x fastest, `n` cells) in bricks of 4 cells a side, `nb` of
/// them a side.
fn bricked(grid: &[u8], n: [usize; 3], nb: [usize; 3]) -> Vec<u8> {
    if grid.is_empty() {
        return Vec::new();
    }
    let mut out = vec![0u8; nb[0] * nb[1] * nb[2] * 64];
    for z in 0..n[2] {
        for y in 0..n[1] {
            for x in 0..n[0] {
                out[brick_idx(nb, x, y, z)] = grid[(z * n[1] + y) * n[0] + x];
            }
        }
    }
    out
}

#[inline(always)]
pub(crate) fn brick_idx(nb: [usize; 3], x: usize, y: usize, z: usize) -> usize {
    ((((z >> 2) * nb[1] + (y >> 2)) * nb[0] + (x >> 2)) << 6) | ((z & 3) << 4) | ((y & 3) << 2) | (x & 3)
}

/// Cells to the nearest cell with density, or with a surface within a
/// cell, two chamfer passes (3-4-5 weights over 3), at most 255.
fn empty_space(dist: &[u8], density: &[u8], n: [usize; 3]) -> Vec<u8> {
    if dist.is_empty() && density.is_empty() {
        return Vec::new();
    }
    let idx = |x: usize, y: usize, z: usize| (z * n[1] + y) * n[0] + x;
    let full = |i: usize| density.get(i).is_some_and(|&v| v > 0) || dist.get(i).is_some_and(|&v| (v as f32) < DIST_STEPS * 1.5);
    let mut d: Vec<u16> = (0..n[0] * n[1] * n[2]).map(|i| if full(i) { 0 } else { 765 }).collect();
    let pass = |d: &mut Vec<u16>, fwd: bool| {
        let range = |m: usize| -> Vec<usize> { if fwd { (0..m).collect() } else { (0..m).rev().collect() } };
        let s: i32 = if fwd { -1 } else { 1 };
        for z in range(n[2]) {
            for y in range(n[1]) {
                for x in range(n[0]) {
                    let mut best = d[idx(x, y, z)];
                    for dz in [0, s] {
                        for dy in [-1, 0, 1] {
                            for dx in [-1, 0, 1] {
                                // Only neighbours already passed.
                                if dz == 0 && (dy == -s || (dy == 0 && dx != s)) {
                                    continue;
                                }
                                let (qx, qy, qz) = (x as i32 + dx, y as i32 + dy, z as i32 + dz);
                                if qx < 0 || qy < 0 || qz < 0 || qx >= n[0] as i32 || qy >= n[1] as i32 || qz >= n[2] as i32 {
                                    continue;
                                }
                                let w = [0u16, 3, 4, 5][(dx.abs() + dy.abs() + dz.abs()) as usize];
                                best = best.min(d[idx(qx as usize, qy as usize, qz as usize)] + w);
                            }
                        }
                    }
                    d[idx(x, y, z)] = best;
                }
            }
        }
    };
    pass(&mut d, true);
    pass(&mut d, false);
    d.into_iter().map(|v| (v / 3).min(255) as u8).collect()
}

/// How long a bake took, for the log.
#[derive(Clone, Copy, Debug, Default)]
pub struct BakeReport {
    pub opaque_tris: usize,
    pub cutout_tris: usize,
    pub seconds: f64,
}

/// Closest distance from `p` to triangle a b c (Ericson, Real-Time
/// Collision Detection, 5.1.5).
pub fn point_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> f32 {
    let (ab, ac, ap) = (b - a, c - a, p - a);
    let (d1, d2) = (ab.dot(ap), ac.dot(ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return ap.length();
    }
    let bp = p - b;
    let (d3, d4) = (ab.dot(bp), ac.dot(bp));
    if d3 >= 0.0 && d4 <= d3 {
        return bp.length();
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (p - (a + ab * v)).length();
    }
    let cp = p - c;
    let (d5, d6) = (ab.dot(cp), ac.dot(cp));
    if d6 >= 0.0 && d5 <= d6 {
        return cp.length();
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (p - (a + ac * w)).length();
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (p - (b + (c - b) * w)).length();
    }
    let denom = 1.0 / (va + vb + vc);
    let (v, w) = (vb * denom, vc * denom);
    (p - (a + ab * v + ac * w)).length()
}

impl ShadowField {
    /// From its grids, with what a ray reads made from them.
    pub fn assemble(lo: Vec3, h: f32, n: [usize; 3], dist: Vec<u8>, density: Vec<u8>) -> ShadowField {
        let empty = empty_space(&dist, &density, n);
        let nb = [n[0].div_ceil(4), n[1].div_ceil(4), n[2].div_ceil(4)];
        let (bdist, bdensity, bempty) = (bricked(&dist, n, nb), bricked(&density, n, nb), bricked(&empty, n, nb));
        let top = Vec3::new(n[0] as f32 - 1.0001, n[1] as f32 - 1.0001, n[2] as f32 - 1.0001);
        let wrap = [nb[0] * 64 - 12, nb[0] * nb[1] * 64 - 48];
        ShadowField { lo, h, n, dist, density, empty, nb, top, wrap, bdist, bdensity, bempty }
    }

    /// Bake the field of clusters `range` of `scene` as they are stored
    /// (an asset alone, its transform the identity), `res` cells along its
    /// longest side. The level of detail whose error is under a quarter of
    /// a cell stands for the surface. Cut-out triangles (those whose alpha
    /// tiles are solid over less than `OPAQUE_SHARE`) go into the density,
    /// the rest into the distance.
    pub fn bake(scene: &Scene, range: std::ops::Range<usize>, res: usize) -> (ShadowField, BakeReport) {
        use rayon::prelude::*;
        let t0 = std::time::Instant::now();
        // Bounds of the finest level.
        let finest = scene.clusters[range.clone()].iter().map(|c| c.lod_level).min().unwrap_or(0);
        let (mut lo, mut hi) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
        for c in scene.clusters[range.clone()].iter().filter(|c| c.lod_level == finest) {
            for p in scene.cluster_pos(c) {
                lo = lo.min(p);
                hi = hi.max(p);
            }
        }
        if !lo.is_finite() {
            return (ShadowField::assemble(Vec3::ZERO, 1.0, [2, 2, 2], Vec::new(), Vec::new()), BakeReport::default());
        }
        let res = res.max(8);
        let h = ((hi - lo).max_element() / (res as f32 - 4.0)).max(1e-5);
        let lo = lo - Vec3::splat(2.0 * h);
        let n = (((hi - lo) / h).ceil().as_uvec3() + glam::UVec3::splat(3)).to_array().map(|x| x as usize);
        // The surface at a level of detail a quarter of a cell from the
        // model as made.
        let at = |c: &crate::scene::Cluster| c.lod_drawn(1.0, |_, _| 4.0 / h);
        let mut opaque: Vec<[Vec3; 3]> = Vec::new();
        // Cut-out triangles: their corners, and their alpha block and
        // triangle within it.
        let mut cutout: Vec<([Vec3; 3], std::sync::Arc<crate::texture::Block>, usize)> = Vec::new();
        for ci in range.clone() {
            let c = &scene.clusters[ci];
            if !at(c) {
                continue;
            }
            let mask = scene.textures.alpha_block(ci, c.tri_count as usize, crate::texture::MAX_LEVEL);
            for t in 0..c.tri_count {
                let tri = scene.tri_pos(c, c.first_tri + t);
                match &mask {
                    Some(b) if b.coverage(t as usize) < OPAQUE_SHARE => cutout.push((tri, b.clone(), t as usize)),
                    _ => opaque.push(tri),
                }
            }
        }
        let cells = n[0] * n[1] * n[2];
        let dist = if opaque.is_empty() { Vec::new() } else { bake_distance(&opaque, lo, h, n) };
        let density = if cutout.is_empty() {
            Vec::new()
        } else {
            // Each solid alpha point stands for its share of its
            // triangle's area, laid into the cell it falls in.
            let mut area = vec![0f32; cells];
            let parts: Vec<Vec<(usize, f32)>> = cutout
                .par_iter()
                .map(|(tri, b, t)| {
                    let l = b.level;
                    let m = 1usize << l;
                    let pts = crate::texture::tile_points(l);
                    let a = (tri[1] - tri[0]).cross(tri[2] - tri[0]).length() * 0.5;
                    let each = a / pts as f32;
                    let mut out = Vec::new();
                    for i in 0..=m {
                        for j in 0..=(m - i) {
                            let (b1, b2) = (i as f32 / m as f32, j as f32 / m as f32);
                            if !b.solid(*t, b1, b2) {
                                continue;
                            }
                            let p = tri[0] * (1.0 - b1 - b2) + tri[1] * b1 + tri[2] * b2;
                            let g = ((p - lo) / h).round().as_ivec3();
                            if g.min_element() < 0 || g.x as usize >= n[0] || g.y as usize >= n[1] || g.z as usize >= n[2] {
                                continue;
                            }
                            out.push(((g.z as usize * n[1] + g.y as usize) * n[0] + g.x as usize, each));
                        }
                    }
                    out
                })
                .collect();
            for part in parts {
                for (k, a) in part {
                    area[k] += a;
                }
            }
            // Thin plates turned every way block half their area: optical
            // depth across a cell is area / (2 h^2).
            area.iter().map(|&a| (a / (2.0 * h * h) * TAU_STEPS).round().min(255.0) as u8).collect()
        };
        let report = BakeReport { opaque_tris: opaque.len(), cutout_tris: cutout.len(), seconds: t0.elapsed().as_secs_f64() };
        (ShadowField::assemble(lo, h, n, dist, density), report)
    }

    /// Bytes it holds (as stored).
    pub fn bytes(&self) -> usize {
        self.dist.len() + self.density.len()
    }

    /// Its box in its own space.
    pub fn bounds(&self) -> (Vec3, Vec3) {
        (self.lo, self.lo + Vec3::new(self.n[0] as f32 - 1.0, self.n[1] as f32 - 1.0, self.n[2] as f32 - 1.0) * self.h)
    }

    pub fn write(&self, out: &mut Vec<u8>) {
        for v in self.lo.to_array() {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&self.h.to_le_bytes());
        for k in self.n {
            out.extend_from_slice(&(k as u32).to_le_bytes());
        }
        out.extend_from_slice(&(self.dist.len() as u32).to_le_bytes());
        out.extend_from_slice(&(self.density.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.dist);
        out.extend_from_slice(&self.density);
        while out.len() % 4 != 0 {
            out.push(0);
        }
    }

    /// Read one written by `write` from `b` at `*at`.
    pub fn read(b: &[u8], at: &mut usize) -> Result<ShadowField, String> {
        let mut take = |k: usize| -> Result<&[u8], String> {
            let s = b.get(*at..*at + k).ok_or("SHDF is truncated")?;
            *at += k;
            Ok(s)
        };
        let f = |s: &[u8]| f32::from_le_bytes(s.try_into().unwrap());
        let u = |s: &[u8]| u32::from_le_bytes(s.try_into().unwrap()) as usize;
        let lo = Vec3::new(f(take(4)?), f(take(4)?), f(take(4)?));
        let h = f(take(4)?);
        let n = [u(take(4)?), u(take(4)?), u(take(4)?)];
        let (nd, nt) = (u(take(4)?), u(take(4)?));
        let cells = n[0].checked_mul(n[1]).and_then(|v| v.checked_mul(n[2])).ok_or("SHDF grid too large")?;
        if (nd != 0 && nd != cells) || (nt != 0 && nt != cells) || !(h > 0.0) || n.iter().any(|&k| k < 2) {
            return Err("SHDF grid does not match its size".into());
        }
        let dist = take(nd)?.to_vec();
        let density = take(nt)?.to_vec();
        *at = at.div_ceil(4) * 4;
        Ok(ShadowField::assemble(lo, h, n, dist, density))
    }
}

/// Unsigned distance to `tris` at every cell centre, in bytes. Triangles
/// are binned in blocks of 4 cells; each cell looks outward block ring by
/// block ring until no nearer triangle can be left.
fn bake_distance(tris: &[[Vec3; 3]], lo: Vec3, h: f32, n: [usize; 3]) -> Vec<u8> {
    use rayon::prelude::*;
    let cs = 4.0 * h;
    let cn = n.map(|x| x / 4 + 1);
    let mut bins: Vec<Vec<u32>> = vec![Vec::new(); cn[0] * cn[1] * cn[2]];
    let top = glam::IVec3::new(cn[0] as i32 - 1, cn[1] as i32 - 1, cn[2] as i32 - 1);
    for (k, t) in tris.iter().enumerate() {
        let a = ((t[0].min(t[1]).min(t[2]) - lo) / cs).floor().as_ivec3().clamp(glam::IVec3::ZERO, top);
        let b = ((t[0].max(t[1]).max(t[2]) - lo) / cs).floor().as_ivec3().clamp(glam::IVec3::ZERO, top);
        let big = a != b;
        for z in a.z..=b.z {
            for y in a.y..=b.y {
                for x in a.x..=b.x {
                    // Only the bins it passes through, not all its box's (a
                    // big slanted triangle's box is most of the grid).
                    if big {
                        let centre = lo + (Vec3::new(x as f32, y as f32, z as f32) + 0.5) * cs;
                        if point_triangle(centre, t[0], t[1], t[2]) > cs * 0.87 {
                            continue;
                        }
                    }
                    bins[(z as usize * cn[1] + y as usize) * cn[0] + x as usize].push(k as u32);
                }
            }
        }
    }
    let cap = 255.0 / DIST_STEPS * h;
    let near = chessboard(&bins, cn);
    let bidx = |q: glam::IVec3| (q.z as usize * cn[1] + q.y as usize) * cn[0] + q.x as usize;
    (0..n[0] * n[1] * n[2])
        .into_par_iter()
        .map(|i| {
            let (x, y, z) = (i % n[0], (i / n[0]) % n[1], i / (n[0] * n[1]));
            let p = lo + Vec3::new(x as f32, y as f32, z as f32) * h;
            let c = glam::IVec3::new((x / 4) as i32, (y / 4) as i32, (z / 4) as i32);
            let mut best = f32::INFINITY;
            // No triangle in a nearer ring than the nearest bin with one.
            let first = near[bidx(c)] as i32;
            for r in first..(cn.iter().max().copied().unwrap_or(1) as i32 + 1) {
                // Nothing in ring r can be nearer than (r - 1) blocks.
                if best <= (r as f32 - 1.0).max(0.0) * cs || (r as f32 - 1.0) * cs > cap {
                    break;
                }
                // The shell of ring r only.
                for dz in -r..=r {
                    for dy in -r..=r {
                        let side = dz.abs() == r || dy.abs() == r;
                        let mut dx = -r;
                        while dx <= r {
                            let q = c + glam::IVec3::new(dx, dy, dz);
                            dx += if side || r == 0 { 1 } else { 2 * r };
                            if q.min_element() < 0 || q.x > top.x || q.y > top.y || q.z > top.z {
                                continue;
                            }
                            for &k in &bins[bidx(q)] {
                                let t = tris[k as usize];
                                best = best.min(point_triangle(p, t[0], t[1], t[2]));
                            }
                        }
                    }
                }
            }
            (best / h * DIST_STEPS).round().min(255.0) as u8
        })
        .collect()
}

/// Per bin, how many bins away (in the largest of the three axes) the
/// nearest bin holding a triangle is, at most 255: two passes over the 26
/// neighbours, exact for that measure.
fn chessboard(bins: &[Vec<u32>], n: [usize; 3]) -> Vec<u8> {
    let idx = |x: usize, y: usize, z: usize| (z * n[1] + y) * n[0] + x;
    let mut d: Vec<u8> = bins.iter().map(|b| if b.is_empty() { 255 } else { 0 }).collect();
    for fwd in [true, false] {
        let s: i64 = if fwd { -1 } else { 1 };
        for zi in 0..n[2] {
            let z = if fwd { zi } else { n[2] - 1 - zi };
            for yi in 0..n[1] {
                let y = if fwd { yi } else { n[1] - 1 - yi };
                for xi in 0..n[0] {
                    let x = if fwd { xi } else { n[0] - 1 - xi };
                    let mut best = d[idx(x, y, z)];
                    for dz in [0, s] {
                        for dy in -1i64..=1 {
                            for dx in -1i64..=1 {
                                // Only neighbours already passed.
                                if dz == 0 && (dy == -s || (dy == 0 && dx != s)) {
                                    continue;
                                }
                                let (qx, qy, qz) = (x as i64 + dx, y as i64 + dy, z as i64 + dz);
                                if qx < 0 || qy < 0 || qz < 0 || qx >= n[0] as i64 || qy >= n[1] as i64 || qz >= n[2] as i64 {
                                    continue;
                                }
                                best = best.min(d[idx(qx as usize, qy as usize, qz as usize)].saturating_add(1));
                            }
                        }
                    }
                    d[idx(x, y, z)] = best;
                }
            }
        }
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Written and read back, a field is the same.
    #[test]
    fn round_trip() {
        let mut s = Scene::new();
        let p = [Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z];
        s.push_mesh(&p, &[], &[], &[0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3]);
        let (f, _) = ShadowField::bake(&s, 0..s.clusters.len(), 32);
        let f = &f;
        let mut b = Vec::new();
        f.write(&mut b);
        assert_eq!(b.len() % 4, 0);
        let mut at = 0;
        let g = ShadowField::read(&b, &mut at).unwrap();
        assert_eq!(at, b.len());
        assert_eq!(*f, g);
        assert!(ShadowField::read(&b[..b.len() / 2], &mut 0).is_err());
    }
}
