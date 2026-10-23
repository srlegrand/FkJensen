//! What the importers produce: triangles with a position, normal and UV per
//! corner, a material per triangle, and the textures.

use std::path::Path;

use glam::{Vec2, Vec3};

#[derive(Clone, Debug)]
pub struct Material {
    pub name: String,
    /// Linear RGB, multiplied with the texture.
    pub color: Vec3,
    pub texture: Option<usize>,
}

/// A base colour texture, linear RGB, with its mip chain.
pub struct Texture {
    pub name: String,
    /// Level 0 first; each level half the size of the last.
    pub levels: Vec<(usize, usize, Vec<Vec3>)>,
}

fn srgb_to_linear(v: u8) -> f32 {
    let s = v as f32 / 255.0;
    if s <= 0.04045 { s / 12.92 } else { ((s + 0.055) / 1.055).powf(2.4) }
}

impl Texture {
    pub fn load(path: &Path) -> Result<Texture, String> {
        let img = image::open(path).map_err(|e| format!("{}: {e}", path.display()))?.to_rgb8();
        Ok(Texture::from_rgb8(path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(), img.width() as usize, img.height() as usize, img.as_raw()))
    }

    pub fn from_rgb8(name: String, w: usize, h: usize, rgb: &[u8]) -> Texture {
        let lut: Vec<f32> = (0..=255u8).map(srgb_to_linear).collect();
        let base: Vec<Vec3> = rgb.chunks_exact(3).map(|c| Vec3::new(lut[c[0] as usize], lut[c[1] as usize], lut[c[2] as usize])).collect();
        let mut levels = vec![(w, h, base)];
        while {
            let (w, h, _) = levels.last().unwrap();
            *w > 1 || *h > 1
        } {
            let (pw, ph, prev) = levels.last().unwrap();
            let (nw, nh) = ((pw / 2).max(1), (ph / 2).max(1));
            let mut next = vec![Vec3::ZERO; nw * nh];
            for y in 0..nh {
                for x in 0..nw {
                    let mut acc = Vec3::ZERO;
                    let mut n = 0.0;
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let (sx, sy) = ((x * 2 + dx).min(pw - 1), (y * 2 + dy).min(ph - 1));
                            acc += prev[sy * pw + sx];
                            n += 1.0;
                        }
                    }
                    next[y * nw + x] = acc / n;
                }
            }
            levels.push((nw, nh, next));
        }
        Texture { name, levels }
    }

    pub fn size(&self) -> (usize, usize) {
        (self.levels[0].0, self.levels[0].1)
    }

    /// Bilinear sample at `uv` (wrapping), at a fractional mip level `lod`.
    pub fn sample(&self, uv: Vec2, lod: f32) -> Vec3 {
        let lod = lod.clamp(0.0, (self.levels.len() - 1) as f32);
        let l0 = lod.floor() as usize;
        let l1 = (l0 + 1).min(self.levels.len() - 1);
        let f = lod - l0 as f32;
        let a = self.bilinear(l0, uv);
        if f <= 0.0 || l1 == l0 { a } else { a.lerp(self.bilinear(l1, uv), f) }
    }

    fn bilinear(&self, level: usize, uv: Vec2) -> Vec3 {
        let (w, h, px) = &self.levels[level];
        let (w, h) = (*w, *h);
        let x = uv.x.rem_euclid(1.0) * w as f32 - 0.5;
        let y = uv.y.rem_euclid(1.0) * h as f32 - 0.5;
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        // UVs outside 0..1 repeat the texture, but filtering does not reach
        // across its edges: these textures are atlases, where the other
        // edge is another part of the model.
        let clamp = |v: i64, n: usize| v.clamp(0, n as i64 - 1) as usize;
        let p = |xi: i64, yi: i64| px[clamp(yi, h) * w + clamp(xi, w)];
        let (x0, y0) = (x0 as i64, y0 as i64);
        let top = p(x0, y0).lerp(p(x0 + 1, y0), fx);
        let bottom = p(x0, y0 + 1).lerp(p(x0 + 1, y0 + 1), fx);
        top.lerp(bottom, fy)
    }
}

#[derive(Default)]
pub struct Model {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub uvs: Vec<Vec2>,
    pub tris: Vec<[u32; 3]>,
    pub tri_material: Vec<u32>,
    pub materials: Vec<Material>,
    pub textures: Vec<Texture>,
    /// What the reader could not find or read (missing textures), for
    /// whoever is importing to see.
    pub warnings: Vec<String>,
}

impl Model {
    pub fn bounds(&self) -> (Vec3, Vec3) {
        let mut lo = Vec3::splat(f32::INFINITY);
        let mut hi = Vec3::splat(f32::NEG_INFINITY);
        for p in &self.positions {
            lo = lo.min(*p);
            hi = hi.max(*p);
        }
        (lo, hi)
    }

    /// Apply a transform to positions and normals.
    pub fn transform(&mut self, m: glam::Mat4) {
        let nm = m.inverse().transpose();
        for p in &mut self.positions {
            *p = m.transform_point3(*p);
        }
        for n in &mut self.normals {
            *n = nm.transform_vector3(*n).normalize_or(Vec3::ZERO);
        }
        if m.determinant() < 0.0 {
            for t in &mut self.tris {
                t.swap(1, 2);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mips_average_and_sampling_wraps() {
        // 2 x 2: black, white / white, black.
        let t = Texture::from_rgb8("t".into(), 2, 2, &[0, 0, 0, 255, 255, 255, 255, 255, 255, 0, 0, 0]);
        assert_eq!(t.levels.len(), 2);
        assert!((t.levels[1].2[0] - Vec3::splat(0.5)).length() < 1e-6);
        // The centre of texel (0, 0) is black; UVs repeat; the corner does
        // not blend with the opposite edge; the top mip is grey everywhere.
        assert!(t.sample(Vec2::new(0.25, 0.25), 0.0).length() < 1e-6);
        assert!((t.sample(Vec2::new(1.25, -0.75), 0.0) - t.sample(Vec2::new(0.25, 0.25), 0.0)).length() < 1e-6);
        assert!(t.sample(Vec2::new(0.0, 0.0), 0.0).length() < 1e-6);
        assert!((t.sample(Vec2::new(0.1, 0.9), 1.0) - Vec3::splat(0.5)).length() < 1e-6);
    }
}
