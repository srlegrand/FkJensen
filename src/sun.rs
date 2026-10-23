//! The sun and the time of day.
//!
//! The sun is infinitely far: parallel light and parallel shadows, the same
//! on everything in the scene, with no reach. Its shadows come from an
//! orthographic shadow map in three cascades round the subject (a few tens
//! of metres, then hundreds, then the whole set out to the mountains), each
//! read with a 3 x 3 percentage-closer filter like the point lights' cube
//! maps.
//!
//! `daylight` turns an hour of the day into the sun's direction and colour
//! and the sky's light and look, from noon to night.

use glam::{Vec3, Vec4};
use rayon::prelude::*;

use crate::scene::Scene;
use crate::simd::{bits, pcf3, F, I, V3};

/// Everything the hour of the day sets.
#[derive(Clone, Copy, Debug)]
pub struct Daylight {
    /// Unit direction towards the sun; below the horizon at night.
    pub sun_dir: Vec3,
    pub sun_color: Vec3,
    /// Sky illuminance on an upward-facing surface, lux.
    pub sky_lux: f32,
    pub sky_tint: Vec3,
    pub zenith: Vec3,
    pub horizon: Vec3,
}

/// Highest the sun climbs, at noon.
const NOON_ELEVATION: f32 = 60.0;

/// Sun elevation in degrees at `hours` (0 to 24): rises at 6, sets at 18.
pub fn elevation(hours: f32) -> f32 {
    NOON_ELEVATION * (std::f32::consts::PI * (hours - 6.0) / 12.0).sin()
}

/// Sky illuminance (no sun) for a sun at `elev` degrees: about 13,000 lux at
/// noon, 400 at sunset, 3 at the end of civil twilight (-6), 0.05 from
/// astronomical night (-12), about a full moon's.
pub fn sky_lux(elev: f32) -> f32 {
    let lg = if elev >= 0.0 {
        return 400.0 + 14_600.0 * elev.to_radians().sin();
    } else if elev >= -6.0 {
        2.6 + (0.48 - 2.6) * (-elev / 6.0)
    } else if elev >= -12.0 {
        0.48 + (-1.3 - 0.48) * ((-elev - 6.0) / 6.0)
    } else {
        -1.3
    };
    10f32.powf(lg)
}

fn lerp3(a: Vec3, b: Vec3, t: f32) -> Vec3 {
    a.lerp(b, t.clamp(0.0, 1.0))
}

/// Day, golden hour, dusk and night, blended by the sun's elevation.
pub fn daylight(hours: f32) -> Daylight {
    let h = hours.rem_euclid(24.0);
    let e = elevation(h);
    // East (+x) at sunrise, south (+z, behind the opening camera) at noon,
    // west at sunset.
    let az = std::f32::consts::PI * (h - 6.0) / 12.0;
    let el = e.to_radians();
    let sun_dir = Vec3::new(el.cos() * az.cos(), el.sin(), el.cos() * az.sin()).normalize();

    let day = (Vec3::new(0.82, 0.95, 1.25), Vec3::new(0.35, 0.6, 1.5), Vec3::new(1.0, 1.05, 1.1));
    let gold = (Vec3::new(1.05, 0.95, 0.9), Vec3::new(0.45, 0.6, 1.2), Vec3::new(1.5, 1.05, 0.7));
    let dusk = (crate::raster::SKY_IRRADIANCE_TINT, crate::raster::SKY_ZENITH, crate::raster::SKY_HORIZON);
    let night = (Vec3::new(0.6, 0.75, 1.3), Vec3::new(0.25, 0.35, 0.9), Vec3::new(0.45, 0.5, 0.8));
    let mix = |a: (Vec3, Vec3, Vec3), b: (Vec3, Vec3, Vec3), t: f32| (lerp3(a.0, b.0, t), lerp3(a.1, b.1, t), lerp3(a.2, b.2, t));
    let (sky_tint, zenith, horizon) = if e >= 12.0 {
        day
    } else if e >= 0.0 {
        mix(gold, day, e / 12.0)
    } else if e >= -6.0 {
        mix(dusk, gold, (e + 6.0) / 6.0)
    } else {
        mix(night, dusk, (e + 12.0) / 6.0)
    };
    // White high up, orange towards the horizon.
    let sun_color = lerp3(Vec3::new(1.0, 0.55, 0.25), Vec3::new(1.0, 0.97, 0.92), e / 20.0);
    Daylight { sun_dir, sun_color: crate::lights::normalise_colour(sun_color), sky_lux: sky_lux(e), sky_tint, zenith, horizon }
}

/// The hour of the evening whose sky is `lux`: the set opens at dusk.
pub fn hour_for_dusk_lux(lux: f32) -> f32 {
    // Search the evening for the closest sky.
    let mut best = (f32::INFINITY, 18.0);
    for k in 0..=600 {
        let h = 17.0 + k as f32 * 0.005;
        let d = (sky_lux(elevation(h)).ln() - lux.ln()).abs();
        if d < best.0 {
            best = (d, h);
        }
    }
    best.1
}

struct Cascade {
    centre: Vec3,
    half: f32,
    res: usize,
    /// Per texel: how far towards the sun the nearest surface is (p . dir);
    /// -inf where there is none.
    height: Vec<f32>,
}

pub struct SunShadow {
    pub dir: Vec3,
    right: Vec3,
    up: Vec3,
    cascades: Vec<Cascade>,
    /// Triangles drawn.
    pub tris: u64,
}

impl SunShadow {
    /// Draw the scene into cascades centred on `centre`, of half-widths
    /// `halves` (smallest first), `res` texels a side.
    pub fn build(scene: &Scene, dir: Vec3, centre: Vec3, halves: &[f32], res: usize) -> SunShadow {
        let d = dir.normalize();
        let right = if d.y.abs() > 0.99 { Vec3::X } else { Vec3::Y.cross(d).normalize() };
        let up = d.cross(right);
        let mut tris = 0;
        let cascades = halves
            .iter()
            .map(|&half| {
                let texel = 2.0 * half / res as f32;
                let (height, n) = scene
                    .clusters
                    .par_chunks(256)
                    .fold(
                        || (vec![f32::NEG_INFINITY; res * res], 0u64),
                        |(mut buf, mut n), chunk| {
                            for c in chunk {
                                // The level of detail whose error is under
                                // a texel of this cascade.
                                if !c.lod_drawn(texel, |_, _| 1.0) {
                                    continue;
                                }
                                let q = c.centre - centre;
                                if q.dot(right).abs() > half + c.radius || q.dot(up).abs() > half + c.radius {
                                    continue;
                                }
                                // Too small for this cascade's texels.
                                if c.radius * 2.0 < texel * 0.5 {
                                    continue;
                                }
                                for t in c.first_tri..c.first_tri + c.tri_count {
                                    let ps = scene.tri_pos(c, t);
                                    let v = [0, 1, 2].map(|k| {
                                        let p = ps[k] - centre;
                                        Vec4::new((p.dot(right) / half * 0.5 + 0.5) * res as f32, (p.dot(up) / half * 0.5 + 0.5) * res as f32, p.dot(d), 0.0)
                                    });
                                    n += fill_max(&mut buf, res, v[0], v[1], v[2], [0.0, 0.0, f32::INFINITY, f32::INFINITY]) as u64;
                                }
                            }
                            (buf, n)
                        },
                    )
                    .reduce(
                        || (vec![f32::NEG_INFINITY; res * res], 0),
                        |(mut a, na), (b, nb)| {
                            for (x, y) in a.iter_mut().zip(b) {
                                *x = x.max(y);
                            }
                            (a, na + nb)
                        },
                    );
                tris += n;
                Cascade { centre, half, res, height }
            })
            .collect();
        SunShadow { dir: d, right, up, cascades, tris }
    }

    /// Which cascade holds `p` (0 is the finest), if any.
    pub fn cascade_of(&self, p: Vec3) -> Option<usize> {
        self.cascades.iter().position(|c| {
            let q = p - c.centre;
            q.dot(self.right).abs() < c.half * 0.97 && q.dot(self.up).abs() < c.half * 0.97
        })
    }

    /// How much sunlight reaches `p` with normal `n`: 1 lit, 0 in shadow.
    pub fn lit(&self, p: Vec3, n: Vec3) -> f32 {
        let Some(k) = self.cascade_of(p) else { return 1.0 };
        let c = &self.cascades[k];
        let texel = 2.0 * c.half / c.res as f32;
        let n = if n.dot(self.dir) < 0.0 { -n } else { n };
        let q = p + n * (texel * 1.5) - c.centre;
        let sx = (q.dot(self.right) / c.half * 0.5 + 0.5) * c.res as f32 - 0.5;
        let sy = (q.dot(self.up) / c.half * 0.5 + 0.5) * c.res as f32 - 0.5;
        let h = q.dot(self.dir) + texel;
        let (cx, cy) = (sx.round() as i32, sy.round() as i32);
        let mut lit = 0.0;
        for dy in -1..=1 {
            for dx in -1..=1 {
                let x = (cx + dx).clamp(0, c.res as i32 - 1) as usize;
                let y = (cy + dy).clamp(0, c.res as i32 - 1) as usize;
                if c.height[y * c.res + x] <= h {
                    lit += 1.0;
                }
            }
        }
        lit / 9.0
    }

    /// `lit` for eight points at once, for the lanes in `mask` (others
    /// come back 0). A lane outside every cascade is lit.
    pub fn lit8(&self, p: V3, n: V3, mask: u32) -> [f32; 8] {
        let mut out = [0f32; 8];
        let mut todo = mask;
        let (right, up, dir) = (V3::splat(self.right), V3::splat(self.up), V3::splat(self.dir));
        let n = n.facing(dir);
        for c in &self.cascades {
            if todo == 0 {
                break;
            }
            let q = p.sub(V3::splat(c.centre));
            let edge = F::splat(c.half * 0.97);
            let inside = bits(q.dot(right).abs().simd_lt(edge) & q.dot(up).abs().simd_lt(edge)) & todo;
            if inside == 0 {
                continue;
            }
            todo &= !inside;
            let texel = 2.0 * c.half / c.res as f32;
            let q = p.add(n.scale(F::splat(texel * 1.5))).sub(V3::splat(c.centre));
            let res = c.res as f32;
            let half = F::splat(c.half);
            let sx = ((q.dot(right) / half * 0.5 + 0.5) * res - 0.5).round_int();
            let sy = ((q.dot(up) / half * 0.5 + 0.5) * res - 0.5).round_int();
            let h = (q.dot(dir) + texel).to_array();
            let lit = pcf3(&c.height, c.res, I::splat(0), sx, sy, inside, |t, j| t <= h[j]);
            for j in 0..8 {
                if inside & (1 << j) != 0 {
                    out[j] = lit[j];
                }
            }
        }
        for (j, o) in out.iter_mut().enumerate() {
            if todo & (1 << j) != 0 {
                *o = 1.0;
            }
        }
        out
    }

    /// Draw the cascades again where the boxes `moved` fall on them (where
    /// an object was and is), with everything that covers those texels.
    /// Returns the triangles drawn.
    pub fn redraw(&mut self, scene: &Scene, moved: &[(Vec3, Vec3)]) -> u64 {
        let (right, up, d) = (self.right, self.up, self.dir);
        let mut tris = 0;
        for cas in &mut self.cascades {
            let (centre, half, res) = (cas.centre, cas.half, cas.res);
            let texel = 2.0 * half / res as f32;
            let to_tex = |p: Vec3| {
                let q = p - centre;
                ((q.dot(right) / half * 0.5 + 0.5) * res as f32, (q.dot(up) / half * 0.5 + 0.5) * res as f32)
            };
            // A sphere's texels, as a rect.
            let rect_of = |c: Vec3, r: f32| {
                let (x, y) = to_tex(c);
                let k = r / texel;
                [x - k, y - k, x + k, y + k]
            };
            // One rect round every box, drawn once.
            let Some(r) = moved.iter().map(|&(lo, hi)| rect_of((lo + hi) * 0.5, (hi - lo).length() * 0.5)).reduce(|a, b| [a[0].min(b[0]), a[1].min(b[1]), a[2].max(b[2]), a[3].max(b[3])]) else { continue };
            let rect = [r[0].floor().max(0.0), r[1].floor().max(0.0), r[2].ceil().min(res as f32 - 1.0), r[3].ceil().min(res as f32 - 1.0)];
            if rect[0] > rect[2] || rect[1] > rect[3] {
                continue;
            }
            for y in rect[1] as usize..=rect[3] as usize {
                cas.height[y * res + rect[0] as usize..=y * res + rect[2] as usize].fill(f32::NEG_INFINITY);
            }
            // What covers the rect, found in parallel, drawn in order.
            let hit: Vec<usize> = (0..scene.clusters.len())
                .into_par_iter()
                .filter(|&i| {
                    let c = &scene.clusters[i];
                    if !c.lod_drawn(texel, |_, _| 1.0) || c.radius * 2.0 < texel * 0.5 {
                        return false;
                    }
                    let o = rect_of(c.centre, c.radius);
                    !(o[0] > rect[2] + 1.0 || o[2] < rect[0] - 1.0 || o[1] > rect[3] + 1.0 || o[3] < rect[1] - 1.0)
                })
                .collect();
            let height = &mut cas.height;
            for i in hit {
                let c = &scene.clusters[i];
                for t in c.first_tri..c.first_tri + c.tri_count {
                    let ps = scene.tri_pos(c, t);
                    let v = [0, 1, 2].map(|k| {
                        let p = ps[k];
                        let (x, y) = to_tex(p);
                        Vec4::new(x, y, (p - centre).dot(d), 0.0)
                    });
                    tris += fill_max(height, res, v[0], v[1], v[2], rect) as u64;
                }
            }
        }
        tris
    }

    pub fn bytes(&self) -> usize {
        self.cascades.iter().map(|c| c.height.len() * 4).sum()
    }
}

/// Rasterise a triangle (x, y in texels, z the height) keeping the largest
/// height, inside `clip` (x0, y0, x1, y1 inclusive); whether it touched
/// any texel.
fn fill_max(buf: &mut [f32], res: usize, a: Vec4, b: Vec4, c: Vec4, clip: [f32; 4]) -> bool {
    let rs = res as f32;
    let area = (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y);
    if area == 0.0 || !area.is_finite() {
        return false;
    }
    let x0 = a.x.min(b.x).min(c.x).floor().max(0.0).max(clip[0]);
    let y0 = a.y.min(b.y).min(c.y).floor().max(0.0).max(clip[1]);
    let x1 = a.x.max(b.x).max(c.x).ceil().min(rs - 1.0).min(clip[2]);
    let y1 = a.y.max(b.y).max(c.y).ceil().min(rs - 1.0).min(clip[3]);
    if x0 > x1 || y0 > y1 {
        return false;
    }
    crate::simd::fill_max(buf, res, (a.x, a.y, a.z), (b.x, b.y, b.z), (c.x, c.y, c.z), x0 as usize, x1 as usize, y0 as usize, y1 as usize, 1.0 / area)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noon_is_high_and_south_night_is_dark() {
        let noon = daylight(12.0);
        assert!(noon.sun_dir.y > 0.8 && noon.sun_dir.z > 0.0, "{:?}", noon.sun_dir);
        assert!(noon.sky_lux > 10_000.0);
        let night = daylight(23.0);
        assert!(night.sun_dir.y < 0.0);
        assert!(night.sky_lux < 0.1);
        let dusk = hour_for_dusk_lux(30.0);
        assert!(dusk > 18.0 && dusk < 19.0, "{dusk}");
        assert!((sky_lux(elevation(dusk)) / 30.0 - 1.0).abs() < 0.1);
        // The sky gets darker all through the evening.
        let mut last = f32::INFINITY;
        for k in 0..40 {
            let l = daylight(12.0 + k as f32 * 0.25).sky_lux;
            assert!(l <= last + 1e-3, "{k}");
            last = l;
        }
    }

    #[test]
    fn a_card_shades_the_floor_under_the_sun() {
        let mut s = Scene::new();
        let quad = |c: Vec3, h: f32| [c + Vec3::new(-h, 0.0, h), c + Vec3::new(h, 0.0, h), c + Vec3::new(h, 0.0, -h), c + Vec3::new(-h, 0.0, -h)];
        s.push_mesh(&quad(Vec3::ZERO, 4.0), &[], &[], &[0, 1, 2, 0, 2, 3]);
        s.push_mesh(&quad(Vec3::new(0.0, 1.0, 0.0), 0.5), &[], &[], &[0, 1, 2, 0, 2, 3]);
        let m = SunShadow::build(&s, Vec3::Y, Vec3::ZERO, &[5.0, 50.0], 256);
        assert_eq!(m.lit(Vec3::ZERO, Vec3::Y), 0.0, "under the card");
        assert_eq!(m.lit(Vec3::new(2.0, 0.0, 2.0), Vec3::Y), 1.0, "in the open");
        assert_eq!(m.lit(Vec3::new(0.1, 1.0, 0.1), Vec3::Y), 1.0, "the card does not shade itself");
        // A low sun from +x throws the shadow towards -x.
        let low = SunShadow::build(&s, Vec3::new(1.0, 1.0, 0.0).normalize(), Vec3::ZERO, &[5.0], 256);
        // Eight at a time gives what one at a time gives, across cascades
        // and outside them.
        let two = SunShadow::build(&s, Vec3::new(0.3, 1.0, 0.2).normalize(), Vec3::ZERO, &[2.0, 50.0], 128);
        for k in 0..8 {
            let pts: [Vec3; 8] = std::array::from_fn(|j| {
                let t = (k * 8 + j) as f32 * 0.53;
                Vec3::new(t.sin() * (1.0 + k as f32 * 10.0), (j % 2) as f32, t.cos() * 3.0)
            });
            let p = V3 { x: F::from(pts.map(|p| p.x)), y: F::from(pts.map(|p| p.y)), z: F::from(pts.map(|p| p.z)) };
            let got = two.lit8(p, V3::splat(Vec3::Y), 0xff);
            for j in 0..8 {
                assert_eq!(got[j], two.lit(pts[j], Vec3::Y), "{}", pts[j]);
            }
        }
        assert_eq!(low.lit(Vec3::new(-1.0, 0.0, 0.0), Vec3::Y), 0.0);
        assert_eq!(low.lit(Vec3::new(1.0, 0.0, 0.0), Vec3::Y), 1.0);
    }
}
