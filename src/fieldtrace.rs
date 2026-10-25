//! Shadow rays through the objects' shadow fields (`field.rs`), for a
//! frame: each object's field placed where it stands, and eight rays at a
//! time traced through it towards a light, eight lanes wide. A ray is
//! traced in the object's space through the inverse of its transform; a
//! scale that differs by axis is handled by taking the shortest a local
//! length can become in the world.

use glam::{Affine3A, Vec3};

use crate::field::{brick_idx, ShadowField, DIST_STEPS, TAU_STEPS};
use crate::scene::Scene;

/// What a ray goes towards.
#[derive(Clone, Copy, Debug)]
pub enum Towards {
    /// A light infinitely far, the way to it (unit) and its angular size
    /// (tan of its half width).
    Sun { dir: Vec3, k: f32 },
    /// A light at `at`, `size` its radius.
    Point { at: Vec3, size: f32 },
}

/// The most a point is moved off its surface before a ray leaves it,
/// metres (fields with large cells: the ground's).
pub const NUDGE_MAX: f32 = 0.25;

/// A ray on its way through a field.
struct Ray {
    gp: Vec3,
    gd: Vec3,
    gd_len: f32,
    t: f32,
    t_end: f32,
    k: f32,
}

/// An object's field placed in the world for a frame: from the world to
/// its own space, how short a length of its own can get in the world, and
/// a sphere round it.
#[derive(Clone)]
pub struct Placed {
    pub field: std::sync::Arc<ShadowField>,
    pub to_local: Affine3A,
    /// The shortest a unit length of its own space becomes in the world
    /// (a distance in its field, times this, is a safe step in the world).
    pub shrink: f32,
    /// A cell's size in the world (its longest).
    pub cell: f32,
    pub centre: Vec3,
    pub radius: f32,
}

impl Placed {
    pub fn new(field: std::sync::Arc<ShadowField>, xf: &Affine3A) -> Placed {
        let m = glam::Mat3::from(xf.matrix3);
        let (x, y, z) = (m.x_axis, m.y_axis, m.z_axis);
        let l = x.length().max(y.length()).max(z.length());
        let square = x.dot(y).abs() + y.dot(z).abs() + x.dot(z).abs() <= 1e-5 * l * l;
        // The smallest singular value: the shortest axis when the axes
        // are square to each other (a rotation and a scale), else a bound
        // below it (|A v| >= |v| / |A^-1|, the Frobenius norm over it).
        let shrink = if square {
            x.length().min(y.length()).min(z.length())
        } else {
            let inv = m.inverse();
            1.0 / inv.x_axis.length().max(inv.y_axis.length()).max(inv.z_axis.length()).max(1e-20) / 3f32.sqrt()
        };
        let grow = m.x_axis.length().max(m.y_axis.length()).max(m.z_axis.length());
        let (a, b) = field.bounds();
        let centre = xf.transform_point3((a + b) * 0.5);
        let radius = (b - a).length() * 0.5 * grow * 3f32.sqrt();
        Placed { cell: field.h * grow, to_local: xf.inverse(), shrink, field, centre, radius }
    }

    /// How much light from `light` reaches each point `p` (normal `n`,
    /// world) of `mask` past this object, 0 to 1 (1 for the others).
    /// `falloff` is the distance over which a shadow fades with its
    /// occluder's distance (0: it does not). Eight rays a step at a time
    /// in eight lanes; only the look-ups go one by one.
    pub fn light8(&self, p: &[Vec3; 8], n: &[Vec3; 8], light: Towards, falloff: f32, mask: u32) -> [f32; 8] {
        use crate::simd::{F, V3};
        let rays: [Option<Ray>; 8] = std::array::from_fn(|j| if mask & (1 << j) != 0 { self.start(p[j], n[j], light) } else { None });
        let mut active = (0..8).filter(|&j| rays[j].is_some()).fold(0u32, |m, j| m | 1 << j);
        if active == 0 {
            return [1.0; 8];
        }
        let lane = |g: fn(&Ray) -> f32| F::from(std::array::from_fn::<f32, 8, _>(|j| rays[j].as_ref().map_or(0.0, g)));
        let gp = V3 { x: lane(|r| r.gp.x), y: lane(|r| r.gp.y), z: lane(|r| r.gp.z) };
        let gd = V3 { x: lane(|r| r.gd.x), y: lane(|r| r.gd.y), z: lane(|r| r.gd.z) };
        let (gd_len, t_end, kk) = (lane(|r| r.gd_len), lane(|r| r.t_end), lane(|r| r.k));
        let mut t = lane(|r| r.t);
        let (mut lit, mut depth) = (F::splat(1.0), F::splat(0.0));
        let f = &*self.field;
        let (has_d, has_t) = (!f.bdist.is_empty(), !f.bdensity.is_empty());
        let cell = F::splat(self.cell);
        let half = F::splat(self.cell * 0.5);
        let thick = half;
        let empty_unit = F::splat(f.h * self.shrink);
        let (zero, one, two) = (F::splat(0.0), F::splat(1.0), F::splat(2.0));
        let top = V3::splat(f.top);
        // Cells to anything (nearest-cell look-up, `out` cells out of the
        // grid) far enough that there is nothing to look up: no surface,
        // no leaves, out of the penumbra (`kt`).
        let (ew, tw) = (f.h * self.shrink, self.cell * 0.5);
        let is_empty = |e: f32, out: f32, kt: f32| {
            let e = (e - out).max(out - 1.0) - 2.0;
            e >= 2.0 && e * ew - tw >= kt
        };
        for _ in 0..256 {
            let q0 = gp.add(gd.scale(t));
            let q = V3 { x: q0.x.max(zero).min(top.x), y: q0.y.max(zero).min(top.y), z: q0.z.max(zero).min(top.z) };
            // Cells out of the grid (in its margin): everything is at least
            // that far, and the nearest within it at most that nearer.
            let out = q0.sub(q).length();
            let i = V3 { x: q.x.floor(), y: q.y.floor(), z: q.z.floor() };
            let fr = q.sub(i);
            let (ix, iy, iz) = (i.x.fast_trunc_int().to_array(), i.y.fast_trunc_int().to_array(), i.z.fast_trunc_int().to_array());
            let (fx, fy, fz) = (fr.x.to_array(), fr.y.to_array(), fr.z.to_array());
            let kt = kk * t;
            let (out_l, kt_l) = (out.to_array(), kt.to_array());
            // The look-ups, lane by lane: what is nearest, the corners.
            let mut e = [0f32; 8];
            let mut dc = [[0f32; 8]; 8];
            let mut tc = [[0f32; 8]; 8];
            for j in 0..8 {
                if active & (1 << j) == 0 {
                    continue;
                }
                let (x, y, z) = (ix[j] as usize, iy[j] as usize, iz[j] as usize);
                let base = brick_idx(f.nb, x, y, z);
                let ox = if x & 3 != 3 { 1 } else { 64 - 3 };
                let oy = if y & 3 != 3 { 4 } else { f.wrap[0] };
                let oz = if z & 3 != 3 { 16 } else { f.wrap[1] };
                let near = base + if fx[j] >= 0.5 { ox } else { 0 } + if fy[j] >= 0.5 { oy } else { 0 } + if fz[j] >= 0.5 { oz } else { 0 };
                e[j] = f.bempty[near] as f32;
                if is_empty(e[j], out_l[j], kt_l[j]) {
                    // Empty space out of the penumbra: the corners are not
                    // needed.
                    continue;
                }
                let at = [0, ox, oy, oy + ox, oz, oz + ox, oz + oy, oz + oy + ox];
                if has_d {
                    for c in 0..8 {
                        dc[c][j] = f.bdist[base + at[c]] as f32;
                    }
                }
                if has_t {
                    for c in 0..8 {
                        tc[c][j] = f.bdensity[base + at[c]] as f32;
                    }
                }
            }
            let tri = |c: &[[f32; 8]; 8]| {
                let v = |n: usize| F::from(c[n]);
                let l = |a: F, b: F, w: F| a + (b - a) * w;
                let (c00, c10) = (l(v(0), v(1), fr.x), l(v(2), v(3), fr.x));
                let (c01, c11) = (l(v(4), v(5), fr.x), l(v(6), v(7), fr.x));
                l(l(c00, c10, fr.y), l(c01, c11, fr.y), fr.z)
            };
            // Lanes still going: the others keep what they have.
            let go = F::from(std::array::from_fn::<f32, 8, _>(|j| if active & (1 << j) != 0 { 1.0 } else { 0.0 })).simd_gt(zero);
            let fade = if falloff > 0.0 { (-t / F::splat(falloff)).exp() } else { one };
            // Cells to anything, less the nearest-cell look-up's error and
            // the chamfer's: over empty space in one step.
            let e = (F::from(e) - out).max(out - one) - two;
            let empty = e.simd_ge(two) & (e * empty_unit - thick).simd_ge(kt);
            let mut step = cell;
            if has_d {
                let d = empty.select(kt, (tri(&dc) / F::splat(DIST_STEPS) - out).max(out) * empty_unit - thick);
                let hit = d.simd_le(F::splat(1e-5)) & !empty;
                let pen = d.simd_lt(kt) & !hit;
                let soft = one - (one - d / kt.max(F::splat(1e-20))) * fade;
                lit = go.select(hit.select(lit.min(one - fade), pen.select(lit.min(soft), lit)), lit);
                let near = if has_t { cell.min(d.max(half)) } else { d.max(half) };
                step = hit.select(half, near);
            }
            if has_t {
                let tau = out.simd_lt(F::splat(0.5)).select(tri(&tc) / F::splat(TAU_STEPS), zero);
                depth += (go & !empty).select(tau * (step / cell) * fade, zero);
            }
            step = empty.select((e / gd_len).max(cell), step);
            t += go.select(step, zero);
            let done = t.simd_ge(t_end) | lit.simd_le(zero) | depth.simd_gt(F::splat(5.5));
            active &= !done.to_bitmask();
            if active == 0 {
                break;
            }
        }
        let (lit, depth) = (lit.to_array(), depth.to_array());
        std::array::from_fn(|j| {
            if rays[j].is_none() {
                1.0
            } else if lit[j] <= 0.0 || depth[j] > 5.5 {
                0.0
            } else {
                lit[j] * (-depth[j]).exp()
            }
        })
    }

    /// A ray from `p` towards the light set off into the field, or None
    /// when it does not cross it.
    #[inline]
    fn start(&self, p: Vec3, n: Vec3, light: Towards) -> Option<Ray> {
        let f = &*self.field;
        if f.bdist.is_empty() && f.bdensity.is_empty() {
            return None;
        }
        // On or by it, off its surface first: a cell and a half (as a
        // shadow map's normal offset), so that a ray along it does not
        // graze it.
        let lp = self.to_local.transform_point3(p);
        let (a, b) = f.bounds();
        let by = (lp - a).min_element() > -f.h && (b - lp).min_element() > -f.h;
        let p = if by { p + n * (self.cell * 1.5).min(NUDGE_MAX) } else { p };
        let (dir, tmax, k) = match light {
            Towards::Sun { dir, k } => (dir, 1e5, k),
            Towards::Point { at, size } => {
                let d = at - p;
                let dist = d.length().max(1e-6);
                (d / dist, (dist - size).max(0.0), (size / dist).min(1.0))
            }
        };
        let oc = self.centre - p;
        let along = oc.dot(dir);
        // How far out a ray can pass and still be in its penumbra: its
        // width where the ray leaves the object (a point light's size at
        // most), at most as far as the field tells distances.
        let margin = (k * (along + self.radius).clamp(0.0, tmax)).min(self.cell * 31.0);
        // Quick out: the ray misses the sphere round it.
        let r = self.radius + margin;
        if along < -r || along - r > tmax || (oc - dir * along).length_squared() > r * r {
            return None;
        }
        let lp = self.to_local.transform_point3(p);
        let ld = self.to_local.transform_vector3(dir);
        // Into its box and the margin round it: the slab test in its own
        // space (t stays the world distance, the direction there is not
        // unit).
        let (a, b) = (a - Vec3::splat(margin / self.shrink), b + Vec3::splat(margin / self.shrink));
        let inv = Vec3::ONE / ld;
        let (t1, t2) = ((a - lp) * inv, (b - lp) * inv);
        let (tn, tf) = (t1.min(t2).max_element(), t1.max(t2).min_element());
        // Not to the light itself either: it may sit close to the surface
        // (a headlamp on its car).
        let t_end = tf.min(tmax - self.cell * 1.5);
        // Not from the surface itself: a cell on.
        let t = tn.max(self.cell);
        if t >= t_end {
            return None;
        }
        // In cells from the first: where it starts and how far a metre
        // takes it.
        let gd = ld / f.h;
        Some(Ray { gp: (lp - f.lo) / f.h, gd, gd_len: gd.length().max(1e-12), t, t_end, k })
    }
}

/// Fields for a frame: the objects that have one, placed.
#[derive(Clone, Default)]
pub struct Fields {
    pub placed: Vec<Placed>,
    /// 0, 1, 2...: every one, for a light infinitely far.
    pub all: Vec<usize>,
}

impl Fields {
    pub fn of(scene: &Scene) -> Fields {
        let placed = scene
            .objects
            .iter()
            .filter(|o| !o.clusters.is_empty())
            .filter_map(|o| o.field.and_then(|k| scene.fields.get(k)).map(|f| Placed::new(f.clone(), &o.xf)))
            .collect::<Vec<_>>();
        let all = (0..placed.len()).collect();
        Fields { placed, all }
    }

    /// Those that could shade within `reach` of `at` (all for a light
    /// infinitely far, `reach` infinite).
    pub fn near(&self, at: Vec3, reach: f32) -> Vec<usize> {
        (0..self.placed.len()).filter(|&i| reach.is_infinite() || self.placed[i].centre.distance(at) < reach + self.placed[i].radius).collect()
    }

    /// Of `which`, those whose sphere comes within `r` of the segment from
    /// `a` along `dir` for `len` (a group of points' sphere and its way to
    /// a light), or within its penumbra there (`k` times how far, the
    /// light's angular size; at most `size`, a point light's; at most as
    /// far as a field tells distances), into `out`; false when more than
    /// `out` holds.
    #[allow(clippy::too_many_arguments)]
    pub fn along(&self, which: &[usize], a: Vec3, r: f32, dir: Vec3, len: f32, k: f32, size: f32, out: &mut [usize], count: &mut usize) -> bool {
        *count = 0;
        for &i in which {
            let o = &self.placed[i];
            let q = o.centre - a;
            let t = q.dot(dir).clamp(0.0, len);
            let reach = o.radius + r + (k * (t + o.radius)).min(size).min(o.cell * 31.0);
            if (q - dir * t).length_squared() <= reach * reach {
                if *count == out.len() {
                    return false;
                }
                out[*count] = i;
                *count += 1;
            }
        }
        true
    }

    /// How much light from `light` reaches each point `p` (normal `n`) of
    /// `mask` past the objects `which`, each one's shadow multiplied in; 0
    /// for the other lanes.
    pub fn light8(&self, which: &[usize], p: &[Vec3; 8], n: &[Vec3; 8], light: Towards, falloff: f32, mask: u32) -> [f32; 8] {
        let mut v: [f32; 8] = std::array::from_fn(|j| if mask & (1 << j) != 0 { 1.0 } else { 0.0 });
        for &i in which {
            let left = (0..8).filter(|&j| v[j] > 0.0).fold(0u32, |m, j| m | 1 << j);
            if left == 0 {
                break;
            }
            let l = self.placed[i].light8(p, n, light, falloff, left);
            for j in 0..8 {
                v[j] *= l[j];
            }
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 1 m cube at the origin (resting on the ground), its field baked.
    fn cube() -> Scene {
        crate::primitives::part("cube").unwrap()
    }

    fn eight(v: Vec3) -> [Vec3; 8] {
        [v; 8]
    }

    /// Under the cube is dark, beside it lit, from a light above; and a
    /// moved cube shadows where it went.
    #[test]
    fn shadows_where_it_stands() {
        let mut s = cube();
        let fields = Fields::of(&s);
        let up = Vec3::Y;
        let light = Towards::Point { at: Vec3::new(0.0, 5.0, 0.0), size: 0.0 };
        let p = [Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0), Vec3::new(0.3, 0.0, 0.2), Vec3::new(-3.0, 0.0, 1.0), Vec3::ZERO, Vec3::ZERO, Vec3::ZERO, Vec3::ZERO];
        let v = fields.light8(&fields.all, &p, &eight(up), light, 0.0, 0b1111);
        assert_eq!(v[0], 0.0, "under it");
        assert_eq!(v[1], 1.0, "beside it");
        assert_eq!(v[2], 0.0, "under it, off centre");
        assert_eq!(v[3], 1.0, "away");
        // The sun low from +X: the cube's shadow falls towards -X.
        let sun = Towards::Sun { dir: Vec3::new(1.0, 0.3, 0.0).normalize(), k: 0.0 };
        let p = [Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.5, 0.0, 0.0), Vec3::new(-1.0, 0.0, 2.0), Vec3::ZERO, Vec3::ZERO, Vec3::ZERO, Vec3::ZERO, Vec3::ZERO];
        let v = fields.light8(&fields.all, &p, &eight(up), sun, 0.0, 0b111);
        assert_eq!(v[0], 0.0, "in its shadow");
        assert_eq!(v[1], 1.0, "on its lit side");
        assert_eq!(v[2], 1.0, "beside its shadow");
        // Moved 3 m along Z: its shadow goes with it.
        s.set_object(0, Vec3::new(0.0, 0.0, 3.0), Vec3::ZERO, Vec3::ONE);
        let fields = Fields::of(&s);
        let p = [Vec3::new(0.0, 0.0, 3.0), Vec3::ZERO, Vec3::ZERO, Vec3::ZERO, Vec3::ZERO, Vec3::ZERO, Vec3::ZERO, Vec3::ZERO];
        let v = fields.light8(&fields.all, &p, &eight(up), light, 0.0, 0b11);
        assert_eq!(v[1], 1.0, "where it was");
        let light = Towards::Point { at: Vec3::new(0.0, 5.0, 3.0), size: 0.0 };
        let v = fields.light8(&fields.all, &p, &eight(up), light, 0.0, 0b1);
        assert_eq!(v[0], 0.0, "where it is");
    }

    /// A big light's shadow is soft at its edge, a point's hard; and the
    /// softness fades with falloff.
    #[test]
    fn soft_with_size() {
        let s = cube();
        let fields = Fields::of(&s);
        let up = Vec3::Y;
        // Just outside the shadow's edge (0.625 m out), the cube's top 1 m
        // above.
        let p = eight(Vec3::new(0.7, 0.0, 0.0));
        let hard = fields.light8(&fields.all, &p, &eight(up), Towards::Point { at: Vec3::new(0.0, 5.0, 0.0), size: 0.0 }, 0.0, 1)[0];
        let soft = fields.light8(&fields.all, &p, &eight(up), Towards::Point { at: Vec3::new(0.0, 5.0, 0.0), size: 1.0 }, 0.0, 1)[0];
        assert_eq!(hard, 1.0);
        assert!(soft > 0.0 && soft < 0.9, "{soft}");
        let faded = fields.light8(&fields.all, &p, &eight(up), Towards::Point { at: Vec3::new(0.0, 5.0, 0.0), size: 1.0 }, 0.3, 1)[0];
        assert!(faded > soft, "{faded} {soft}");
    }

    /// A light inside an object's box, by its surface (a car's lamp), is
    /// not shadowed by the surface it sits on.
    #[test]
    fn a_lamp_on_its_surface_lights() {
        let s = cube();
        let fields = Fields::of(&s);
        // On the cube's top, a lamp 2 cm over it 40 cm away.
        let p = eight(Vec3::new(-0.2, 1.0, 0.0));
        let v = fields.light8(&fields.all, &p, &eight(Vec3::Y), Towards::Point { at: Vec3::new(0.2, 1.02, 0.0), size: 0.01 }, 0.0, 1)[0];
        assert!(v > 0.9, "{v}");
    }
}
