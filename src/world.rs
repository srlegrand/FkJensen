//! The built-in set around the chess board, made procedurally at start-up,
//! in metres, Y up, the camera looking towards -Z:
//!
//! - foreground: a table, 0.78 m high, with the chess set on it, at the origin;
//! - mid ground: a café shopfront 40 m away across a paved square, with an
//!   awning, windows, a door, a sign, tables outside, neighbouring
//!   buildings, a street lamp and a string of festoon bulbs;
//! - far background: mountains from 4 km to 12 km away, up to about 2 km
//!   high, on a valley floor that runs to the horizon.
//!
//! Everything is real geometry, cut into the same clusters as the chess set.
//! The point is distance: the lights at the table cannot reach the shop and
//! nothing but the sky reaches the mountains, and the renderer skips that
//! work (see `raster.rs`, light reach).

use glam::Vec3;

use crate::lights::normalise_colour;
use crate::scene::{PointLight, Scene};

/// Height of the table top.
pub const TABLE_TOP: f32 = 0.78;
/// Distance from the table to the shop's front wall.
pub const SHOP_FRONT_Z: f32 = -40.0;
/// Nearest edge of the mountain range.
pub const MOUNTAINS_NEAR_Z: f32 = -4000.0;
pub const MOUNTAINS_FAR_Z: f32 = -12000.0;

/// Flat-shaded geometry collected before it is clustered.
#[derive(Default)]
struct Mesh {
    p: Vec<Vec3>,
    n: Vec<Vec3>,
    c: Vec<Vec3>,
    i: Vec<u32>,
}

impl Mesh {
    fn quad(&mut self, a: Vec3, b: Vec3, c: Vec3, d: Vec3, col: Vec3) {
        // a b c d counter-clockwise seen from the front.
        let n = (b - a).cross(d - a).normalize_or(Vec3::Y);
        let base = self.p.len() as u32;
        for v in [a, b, c, d] {
            self.p.push(v);
            self.n.push(n);
            self.c.push(col);
        }
        self.i.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    /// An axis-aligned box, every face outward.
    fn cube(&mut self, lo: Vec3, hi: Vec3, col: Vec3) {
        let v = |x: bool, y: bool, z: bool| Vec3::new(if x { hi.x } else { lo.x }, if y { hi.y } else { lo.y }, if z { hi.z } else { lo.z });
        let (f, t) = (false, true);
        self.quad(v(f, f, t), v(t, f, t), v(t, t, t), v(f, t, t), col); // +z
        self.quad(v(t, f, f), v(f, f, f), v(f, t, f), v(t, t, f), col); // -z
        self.quad(v(t, f, t), v(t, f, f), v(t, t, f), v(t, t, t), col); // +x
        self.quad(v(f, f, f), v(f, f, t), v(f, t, t), v(f, t, f), col); // -x
        self.quad(v(f, t, t), v(t, t, t), v(t, t, f), v(f, t, f), col); // +y
        self.quad(v(f, f, f), v(t, f, f), v(t, f, t), v(f, f, t), col); // -y
    }

    /// A horizontal grid at height `y`, `nx` x `nz` cells, facing up.
    fn ground(&mut self, x0: f32, x1: f32, z0: f32, z1: f32, y: f32, nx: usize, nz: usize, col: impl Fn(f32, f32) -> Vec3) {
        for j in 0..nz {
            for k in 0..nx {
                let xa = x0 + (x1 - x0) * k as f32 / nx as f32;
                let xb = x0 + (x1 - x0) * (k + 1) as f32 / nx as f32;
                let za = z0 + (z1 - z0) * j as f32 / nz as f32;
                let zb = z0 + (z1 - z0) * (j + 1) as f32 / nz as f32;
                let c = col((xa + xb) * 0.5, (za + zb) * 0.5);
                // Counter-clockwise seen from above (+y): x right, z towards the viewer.
                let (lo_z, hi_z) = (za.min(zb), za.max(zb));
                self.quad(Vec3::new(xa, y, hi_z), Vec3::new(xb, y, hi_z), Vec3::new(xb, y, lo_z), Vec3::new(xa, y, lo_z), c);
            }
        }
    }

    /// Ground covering rectangle `outer` minus rectangle `inner` (both
    /// [x0, x1, z0, z1], inner inside outer), in cells of about `cell` metres.
    fn ring(&mut self, outer: [f32; 4], inner: [f32; 4], y: f32, cell: f32, col: impl Fn(f32, f32) -> Vec3 + Copy) {
        let n = |a: f32, b: f32| (((b - a) / cell).ceil() as usize).max(1);
        let [ox0, ox1, oz0, oz1] = outer;
        let [ix0, ix1, iz0, iz1] = inner;
        // Strips below and above the inner rectangle in z, full width, then
        // left and right of it.
        for (x0, x1, z0, z1) in [(ox0, ox1, oz0, iz0), (ox0, ox1, iz1, oz1), (ox0, ix0, iz0, iz1), (ix1, ox1, iz0, iz1)] {
            if x1 > x0 && z1 > z0 {
                self.ground(x0, x1, z0, z1, y, n(x0, x1), n(z0, z1), col);
            }
        }
    }

    fn into_scene(self, scene: &mut Scene) {
        scene.push_mesh(&self.p, &self.n, &self.c, &self.i);
    }
}

fn hash(x: i32, z: i32) -> f32 {
    let mut h = (x as u32).wrapping_mul(0x8da6b343) ^ (z as u32).wrapping_mul(0xd8163841) ^ 0x9e3779b9;
    h ^= h >> 13;
    h = h.wrapping_mul(0x5bd1e995);
    h ^= h >> 15;
    (h & 0xffffff) as f32 / 16_777_215.0
}

fn value_noise(x: f32, z: f32) -> f32 {
    let (xi, zi) = (x.floor() as i32, z.floor() as i32);
    let (fx, fz) = (x - x.floor(), z - z.floor());
    let s = |t: f32| t * t * (3.0 - 2.0 * t);
    let (sx, sz) = (s(fx), s(fz));
    let a = hash(xi, zi) + (hash(xi + 1, zi) - hash(xi, zi)) * sx;
    let b = hash(xi, zi + 1) + (hash(xi + 1, zi + 1) - hash(xi, zi + 1)) * sx;
    a + (b - a) * sz
}

/// Ridged multi-octave noise, 0 to about 1.
fn ridges(x: f32, z: f32) -> f32 {
    let mut sum = 0.0;
    let mut amp = 0.55;
    let mut f = 1.0;
    for _ in 0..6 {
        let v = 1.0 - (value_noise(x * f, z * f) * 2.0 - 1.0).abs();
        sum += v * v * amp;
        amp *= 0.5;
        f *= 2.03;
    }
    sum
}

/// Mountain height in metres at (x, z).
pub fn mountain_height(x: f32, z: f32) -> f32 {
    // Rises from nothing at the near edge, highest in the middle of the
    // range, eases off again at the far edge.
    let t = ((z - MOUNTAINS_NEAR_Z) / (MOUNTAINS_FAR_Z - MOUNTAINS_NEAR_Z)).clamp(0.0, 1.0);
    let envelope = (t * std::f32::consts::PI).sin().powf(0.6);
    2300.0 * envelope * ridges(x / 3200.0 + 7.3, z / 3200.0 - 2.1)
}

fn mountains(scene: &mut Scene) {
    let (nx, nz) = (240usize, 90usize);
    let (x0, x1) = (-16000.0f32, 16000.0f32);
    let (z0, z1) = (MOUNTAINS_NEAR_Z, MOUNTAINS_FAR_Z);
    let at = |k: usize, j: usize| {
        let x = x0 + (x1 - x0) * k as f32 / nx as f32;
        let z = z0 + (z1 - z0) * j as f32 / nz as f32;
        // 2 m down so the foot of the range sinks under the valley floor.
        Vec3::new(x, mountain_height(x, z) - 2.0, z)
    };
    let mut p = Vec::with_capacity((nx + 1) * (nz + 1));
    let mut c = Vec::with_capacity(p.capacity());
    for j in 0..=nz {
        for k in 0..=nx {
            let v = at(k, j);
            // Grass low, rock in the middle, snow on the tops, with a little
            // variation so the slopes read.
            let jitter = hash(k as i32 * 7, j as i32 * 13) * 0.06;
            let col = if v.y > 1500.0 + jitter * 1500.0 {
                Vec3::new(0.80, 0.82, 0.85)
            } else if v.y > 400.0 {
                Vec3::new(0.20 + jitter, 0.18 + jitter, 0.16 + jitter)
            } else {
                Vec3::new(0.10, 0.14 + jitter, 0.07)
            };
            p.push(v);
            c.push(col);
        }
    }
    let mut idx = Vec::with_capacity(nx * nz * 6);
    let w = (nx + 1) as u32;
    for j in 0..nz as u32 {
        for k in 0..nx as u32 {
            // z decreases with j, so (k, j+1) is further away: counter-
            // clockwise from above is (k,j) (k+1,j) (k+1,j+1) (k,j+1).
            let a = j * w + k;
            let b = a + 1;
            let cc = a + w + 1;
            let d = a + w;
            idx.extend_from_slice(&[a, b, cc, a, cc, d]);
        }
    }
    scene.push_mesh(&p, &[], &c, &idx);
}

/// The table, square, shop, valley and mountains, and their lights. Returns
/// the scene and where the chess set's board centre goes.
pub fn build() -> Scene {
    let mut scene = Scene::new();

    // Table: 1.1 m square top, four legs.
    let wood = Vec3::new(0.28, 0.15, 0.08);
    let mut m = Mesh::default();
    m.cube(Vec3::new(-0.55, TABLE_TOP - 0.04, -0.55), Vec3::new(0.55, TABLE_TOP, 0.55), wood);
    for (x, z) in [(-0.48, -0.48), (0.43, -0.48), (-0.48, 0.43), (0.43, 0.43)] {
        m.cube(Vec3::new(x, 0.0, z), Vec3::new(x + 0.05, TABLE_TOP - 0.04, z + 0.05), wood * 0.8);
    }
    m.into_scene(&mut scene);

    // The square: 1 m paving slabs, so the clusters stay small and a light
    // only touches the ones near it.
    let mut m = Mesh::default();
    m.ground(-40.0, 40.0, -62.0, 12.0, 0.0, 80, 74, |x, z| {
        let v = 0.24 + hash(x.floor() as i32, z.floor() as i32) * 0.05;
        Vec3::new(v, v * 0.97, v * 0.93)
    });
    m.into_scene(&mut scene);

    // Valley floor to the horizon, just below the square, in rings of
    // growing cell size: 10 m cells out to a few hundred metres, then 100 m,
    // then 500 m. Small cells near the square keep its clusters small, so a
    // light there only touches the ground near it.
    let mut m = Mesh::default();
    let grass = |x: f32, z: f32| {
        let v = hash((x / 37.0) as i32, (z / 37.0) as i32) * 0.03;
        Vec3::new(0.09 + v, 0.12 + v, 0.06)
    };
    let plaza = [-40.0, 40.0, -62.0, 12.0];
    let r1 = [-400.0, 400.0, -450.0, 400.0];
    let r2 = [-4000.0, 4000.0, -4000.0, 4000.0];
    let r3 = [-24000.0, 24000.0, -16000.0, 8000.0];
    m.ring(r1, plaza, -0.05, 10.0, grass);
    m.ring(r2, r1, -0.05, 100.0, grass);
    m.ring(r3, r2, -0.05, 500.0, grass);
    m.into_scene(&mut scene);

    shopfront(&mut scene);
    mountains(&mut scene);

    scene.focus = Some((Vec3::new(-0.55, 0.0, -0.55), Vec3::new(0.55, TABLE_TOP + 0.15, 0.55)));
    scene
}

fn shopfront(scene: &mut Scene) {
    let z = SHOP_FRONT_Z;
    let mut m = Mesh::default();
    let render = Vec3::new(0.62, 0.55, 0.45);
    let dark_green = Vec3::new(0.04, 0.12, 0.08);
    let glass = Vec3::new(0.02, 0.025, 0.03);
    let frame = Vec3::new(0.05, 0.05, 0.05);

    // The café: 18 m wide, two storeys, 8 m deep.
    m.cube(Vec3::new(-9.0, 0.0, z - 8.0), Vec3::new(9.0, 7.2, z), render);
    // Shop window, two panes, and its frame, standing just proud of the wall.
    m.cube(Vec3::new(-7.5, 0.5, z), Vec3::new(-0.6, 3.0, z + 0.05), frame);
    m.cube(Vec3::new(-7.35, 0.65, z + 0.05), Vec3::new(-4.15, 2.85, z + 0.06), glass);
    m.cube(Vec3::new(-3.95, 0.65, z + 0.05), Vec3::new(-0.75, 2.85, z + 0.06), glass);
    // Door.
    m.cube(Vec3::new(0.6, 0.0, z), Vec3::new(2.2, 2.5, z + 0.06), dark_green);
    m.cube(Vec3::new(0.85, 1.1, z + 0.06), Vec3::new(1.95, 2.25, z + 0.07), glass);
    // Second window right of the door.
    m.cube(Vec3::new(3.0, 0.5, z), Vec3::new(7.5, 3.0, z + 0.05), frame);
    m.cube(Vec3::new(3.15, 0.65, z + 0.05), Vec3::new(7.35, 2.85, z + 0.06), glass);
    // Sign band.
    m.cube(Vec3::new(-8.5, 3.55, z), Vec3::new(8.5, 4.35, z + 0.12), dark_green);
    m.cube(Vec3::new(-3.0, 3.75, z + 0.12), Vec3::new(3.0, 4.15, z + 0.13), Vec3::new(0.75, 0.65, 0.35));
    // Awning, sloping out and down over the windows.
    let stripe = |k: usize| if k % 2 == 0 { Vec3::new(0.55, 0.06, 0.05) } else { Vec3::new(0.85, 0.82, 0.75) };
    let n_stripes = 24;
    for k in 0..n_stripes {
        let x0 = -8.0 + 16.0 * k as f32 / n_stripes as f32;
        let x1 = -8.0 + 16.0 * (k + 1) as f32 / n_stripes as f32;
        let (back, front) = (Vec3::new(0.0, 3.45, z + 0.1), Vec3::new(0.0, 2.75, z + 1.9));
        m.quad(Vec3::new(x0, front.y, front.z), Vec3::new(x1, front.y, front.z), Vec3::new(x1, back.y, back.z), Vec3::new(x0, back.y, back.z), stripe(k));
        // Underside, so it is not see-through from below.
        m.quad(Vec3::new(x0, back.y - 0.02, back.z), Vec3::new(x1, back.y - 0.02, back.z), Vec3::new(x1, front.y - 0.02, front.z), Vec3::new(x0, front.y - 0.02, front.z), stripe(k) * 0.6);
    }
    // Upstairs windows.
    for k in 0..5 {
        let x = -7.5 + k as f32 * 3.3;
        m.cube(Vec3::new(x, 4.9, z), Vec3::new(x + 1.6, 6.6, z + 0.05), frame);
        m.cube(Vec3::new(x + 0.1, 5.0, z + 0.05), Vec3::new(x + 1.5, 6.5, z + 0.06), glass);
    }
    // Roof edge.
    m.cube(Vec3::new(-9.2, 7.2, z - 8.2), Vec3::new(9.2, 7.45, z + 0.2), Vec3::new(0.18, 0.17, 0.16));
    // Pavement and kerb.
    m.cube(Vec3::new(-20.0, 0.0, z), Vec3::new(20.0, 0.12, z + 4.0), Vec3::new(0.35, 0.34, 0.32));
    // Neighbours.
    m.cube(Vec3::new(-20.0, 0.0, z - 9.0), Vec3::new(-9.0, 8.0, z - 0.3), Vec3::new(0.35, 0.22, 0.18));
    m.cube(Vec3::new(9.0, 0.0, z - 8.5), Vec3::new(19.0, 6.6, z - 0.4), Vec3::new(0.55, 0.57, 0.6));
    for k in 0..3 {
        let x = -18.5 + k as f32 * 3.4;
        for row in 0..2 {
            let y = 1.2 + row as f32 * 3.2;
            m.cube(Vec3::new(x, y, z - 0.3), Vec3::new(x + 1.4, y + 1.8, z - 0.25), glass);
        }
    }
    for k in 0..3 {
        let x = 10.0 + k as f32 * 3.0;
        for row in 0..2 {
            let y = 1.2 + row as f32 * 2.9;
            m.cube(Vec3::new(x, y, z - 0.4), Vec3::new(x + 1.5, y + 1.9, z - 0.35), glass);
        }
    }
    // Tables and chairs outside.
    for (k, x) in [-6.0f32, -2.5, 4.5].iter().enumerate() {
        let tz = z + 2.6 + (k % 2) as f32 * 0.3;
        let metal = Vec3::new(0.06, 0.06, 0.06);
        m.cube(Vec3::new(x - 0.35, 0.72, tz - 0.35), Vec3::new(x + 0.35, 0.75, tz + 0.35), metal);
        m.cube(Vec3::new(x - 0.03, 0.12, tz - 0.03), Vec3::new(x + 0.03, 0.72, tz + 0.03), metal);
        for side in [-1.0f32, 1.0] {
            let cx = x + side * 0.7;
            m.cube(Vec3::new(cx - 0.2, 0.45, tz - 0.2), Vec3::new(cx + 0.2, 0.48, tz + 0.2), Vec3::new(0.3, 0.17, 0.08));
            m.cube(Vec3::new(cx + side * 0.18 - 0.02, 0.48, tz - 0.2), Vec3::new(cx + side * 0.18 + 0.02, 0.95, tz + 0.2), Vec3::new(0.3, 0.17, 0.08));
        }
    }
    // Street lamp, half way between the table and the shop.
    let lamp = Vec3::new(6.5, 5.6, -20.0);
    m.cube(Vec3::new(lamp.x - 0.07, 0.0, lamp.z - 0.07), Vec3::new(lamp.x + 0.07, lamp.y + 0.15, lamp.z + 0.07), Vec3::new(0.05, 0.06, 0.06));
    m.cube(Vec3::new(lamp.x - 0.3, lamp.y + 0.1, lamp.z - 0.2), Vec3::new(lamp.x + 0.3, lamp.y + 0.25, lamp.z + 0.2), Vec3::new(0.05, 0.06, 0.06));
    m.into_scene(scene);

    // The café's festoon string, sagging between the awning ends.
    let warm = normalise_colour(Vec3::new(1.0, 0.72, 0.42));
    let n = 14;
    for k in 0..n {
        let t = k as f32 / (n - 1) as f32;
        let x = -8.5 + 17.0 * t;
        let sag = 0.45 * (1.0 - (2.0 * t - 1.0).powi(2));
        scene.lights.push(PointLight { position: Vec3::new(x, 4.6 - sag, z + 2.3), color: warm, lumens: 200.0 });
    }
    // An LED street light: about 12,000 lm.
    scene.lights.push(PointLight { position: lamp, color: normalise_colour(Vec3::new(0.9, 0.95, 1.0)), lumens: 12_000.0 });
}

/// The lights at the chess table: a small warm lamp as key, a dimmer cool
/// fill. Positions are relative to the table top's centre.
pub fn table_lights() -> Vec<PointLight> {
    let top = Vec3::new(0.0, TABLE_TOP, 0.0);
    vec![
        PointLight { position: top + Vec3::new(-0.2, 0.32, 0.22), color: normalise_colour(Vec3::new(1.0, 0.8, 0.6)), lumens: 400.0 },
        PointLight { position: top + Vec3::new(0.3, 0.2, -0.25), color: normalise_colour(Vec3::new(0.65, 0.8, 1.0)), lumens: 120.0 },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn three_distances_are_where_they_say() {
        let s = build();
        let (fmin, fmax) = s.focus_bounds();
        assert!(fmax.y < 1.0 && fmin.x > -1.0 && fmax.x < 1.0, "the table is about a metre across");
        assert!(fmin.y >= 0.0);
        // Shop geometry and mountain geometry exist at their distances.
        let mut shop = 0;
        let mut far = 0;
        for c in &s.clusters {
            if c.centre.z < SHOP_FRONT_Z + 1.0 && c.centre.z > SHOP_FRONT_Z - 12.0 && c.centre.y > 0.5 {
                shop += 1;
            }
            if c.centre.z < MOUNTAINS_NEAR_Z + 500.0 && c.centre.y > 100.0 {
                far += 1;
            }
        }
        assert!(shop > 10, "{shop} shop clusters");
        assert!(far > 100, "{far} mountain clusters");
        let peak = s.positions.iter().map(|p| p.y).fold(0.0, f32::max);
        assert!(peak > 1200.0 && peak < 2500.0, "peak {peak}");
    }

    #[test]
    fn mountains_start_flat_at_the_near_edge() {
        for x in [-5000.0, 0.0, 3000.0] {
            assert!(mountain_height(x, MOUNTAINS_NEAR_Z).abs() < 1.0);
        }
    }
}
