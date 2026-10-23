//! Eight lanes at a time: points as a structure of arrays, for the hot
//! loops (shading, shadow map lookups, rasterising).
//!
//! `wide` gives f32x8 as one AVX register when the build targets a
//! processor that has AVX (`.cargo/config.toml` builds for the machine it
//! runs on), two SSE or NEON registers otherwise, and plain arrays on the
//! Raspberry Pi 1, so the same code runs everywhere.

use glam::Vec3;
pub use wide::{f32x8 as F, i32x8 as I};

/// Eight points or vectors.
#[derive(Clone, Copy, Debug)]
pub struct V3 {
    pub x: F,
    pub y: F,
    pub z: F,
}

impl V3 {
    #[inline]
    pub fn splat(v: Vec3) -> V3 {
        V3 { x: F::splat(v.x), y: F::splat(v.y), z: F::splat(v.z) }
    }

    #[inline]
    pub fn dot(self, o: V3) -> F {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    #[inline]
    pub fn length(self) -> F {
        self.dot(self).sqrt()
    }

    #[inline]
    pub fn scale(self, k: F) -> V3 {
        V3 { x: self.x * k, y: self.y * k, z: self.z * k }
    }

    #[inline]
    pub fn add(self, o: V3) -> V3 {
        V3 { x: self.x + o.x, y: self.y + o.y, z: self.z + o.z }
    }

    #[inline]
    pub fn sub(self, o: V3) -> V3 {
        V3 { x: self.x - o.x, y: self.y - o.y, z: self.z - o.z }
    }

    /// Each lane turned to face `towards` (flipped where it points away).
    #[inline]
    pub fn facing(self, towards: V3) -> V3 {
        let away = self.dot(towards).simd_lt(F::ZERO);
        V3 { x: away.select(-self.x, self.x), y: away.select(-self.y, self.y), z: away.select(-self.z, self.z) }
    }

    #[cfg(test)]
    pub fn lane(self, j: usize) -> Vec3 {
        Vec3::new(self.x.to_array()[j], self.y.to_array()[j], self.z.to_array()[j])
    }
}

/// Lane mask bits of a comparison.
#[inline]
pub fn bits(m: F) -> u32 {
    m.to_bitmask()
}

/// The 3 x 3 percentage-closer filter round texels (cx, cy) of a `res`
/// square map starting at `base` in `texels`, for the lanes in `mask`:
/// the share of the nine texels for which `pass(texel, lane)` holds.
/// Index arithmetic is eight wide; the loads are one lane at a time.
#[inline]
pub fn pcf3(texels: &[f32], res: usize, base: I, cx: I, cy: I, mask: u32, pass: impl Fn(f32, usize) -> bool) -> [f32; 8] {
    let hi = I::splat(res as i32 - 1);
    let zero = I::splat(0);
    let r = I::splat(res as i32);
    let mut count = [0u32; 8];
    for dy in -1..=1 {
        let y = (cy + I::splat(dy)).max(zero).min(hi);
        let row = base + y * r;
        for dx in -1..=1 {
            let x = (cx + I::splat(dx)).max(zero).min(hi);
            let idx = (row + x).to_array();
            for j in 0..8 {
                if mask & (1 << j) != 0 && pass(texels[idx[j] as usize], j) {
                    count[j] += 1;
                }
            }
        }
    }
    count.map(|c| c as f32 / 9.0)
}

/// Rasterise a triangle into a `res` square map, keeping the largest z,
/// eight texels of a row at a time: corners (x, y in texels, z), texels
/// `x0..=x1`, `y0..=y1` (already inside the map), `inv` = 1 / its signed
/// area. Each texel's weights are worked out as the one-at-a-time loop
/// did, so the map is the same. Returns whether any texel was covered.
#[allow(clippy::too_many_arguments)]
pub fn fill_max(buf: &mut [f32], res: usize, a: (f32, f32, f32), b: (f32, f32, f32), c: (f32, f32, f32), x0: usize, x1: usize, y0: usize, y1: usize, inv: f32) -> bool {
    let (ax, ay, az) = a;
    let (bx, by, bz) = b;
    let (cx, cy, cz) = c;
    let lane = F::from([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]);
    let invf = F::splat(inv);
    let mut any = false;
    for y in y0..=y1 {
        let py = y as f32 + 0.5;
        let row = &mut buf[y * res..(y + 1) * res];
        let mut x = x0;
        while x <= x1 {
            let n = (x1 + 1 - x).min(8);
            let px = F::splat(x as f32) + lane;
            let w0 = (F::splat((cx - bx) * (py - by)) - F::splat(cy - by) * (px - F::splat(bx))) * invf;
            let w1 = (F::splat((ax - cx) * (py - cy)) - F::splat(ay - cy) * (px - F::splat(cx))) * invf;
            let w2 = (F::splat((bx - ax) * (py - ay)) - F::splat(by - ay) * (px - F::splat(ax))) * invf;
            let inside = w0.simd_ge(F::ZERO) & w1.simd_ge(F::ZERO) & w2.simd_ge(F::ZERO);
            let m = inside.to_bitmask() & ((1u32 << n) - 1);
            if m != 0 {
                any = true;
                let z = (w0 * az + w1 * bz + w2 * cz).to_array();
                for j in 0..n {
                    if m & (1 << j) != 0 && z[j] > row[x + j] {
                        row[x + j] = z[j];
                    }
                }
            }
            x += 8;
        }
    }
    any
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The eight-wide fill writes what the texel-by-texel loop writes.
    #[test]
    fn fill_matches_texel_by_texel() {
        let res = 37;
        let tris: [((f32, f32, f32), (f32, f32, f32), (f32, f32, f32)); 3] = [((1.2, 3.3, 0.5), (30.7, 5.1, 2.0), (12.4, 33.9, -1.0)), ((0.0, 0.0, 1.0), (36.9, 0.0, 1.0), (0.0, 36.9, 1.0)), ((5.5, 5.5, 3.0), (6.5, 5.5, 3.0), (5.5, 6.5, 3.0))];
        for (a, b, c) in tris {
            let area = (b.0 - a.0) * (c.1 - a.1) - (c.0 - a.0) * (b.1 - a.1);
            let inv = 1.0 / area;
            let (x0, x1) = (a.0.min(b.0).min(c.0).floor().max(0.0) as usize, a.0.max(b.0).max(c.0).ceil().min(res as f32 - 1.0) as usize);
            let (y0, y1) = (a.1.min(b.1).min(c.1).floor().max(0.0) as usize, a.1.max(b.1).max(c.1).ceil().min(res as f32 - 1.0) as usize);
            let mut want = vec![f32::NEG_INFINITY; res * res];
            for y in y0..=y1 {
                let py = y as f32 + 0.5;
                for x in x0..=x1 {
                    let px = x as f32 + 0.5;
                    let w0 = ((c.0 - b.0) * (py - b.1) - (c.1 - b.1) * (px - b.0)) * inv;
                    let w1 = ((a.0 - c.0) * (py - c.1) - (a.1 - c.1) * (px - c.0)) * inv;
                    let w2 = ((b.0 - a.0) * (py - a.1) - (b.1 - a.1) * (px - a.0)) * inv;
                    if w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0 {
                        let z = w0 * a.2 + w1 * b.2 + w2 * c.2;
                        want[y * res + x] = want[y * res + x].max(z);
                    }
                }
            }
            let mut got = vec![f32::NEG_INFINITY; res * res];
            fill_max(&mut got, res, a, b, c, x0, x1, y0, y1, inv);
            assert_eq!(got, want);
        }
    }

    #[test]
    fn lanes_flip_to_face() {
        let n = V3 { x: F::from([1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), y: F::ZERO, z: F::ZERO };
        let f = n.facing(V3::splat(Vec3::X));
        assert_eq!(f.lane(0), Vec3::X);
        assert_eq!(f.lane(1), Vec3::X);
        assert!((V3::splat(Vec3::new(3.0, 4.0, 0.0)).length().to_array()[5] - 5.0).abs() < 1e-6);
    }
}
