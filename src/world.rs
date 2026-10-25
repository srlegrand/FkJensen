//! The built-in set, made procedurally at start-up, in metres, Y up, the
//! camera looking towards -Z:
//!
//! - a paved ground plane around the origin, where the T-rex and the car
//!   stand (`assets/models/`, placed by `main.rs`);
//! - far background: mountains from 4 km to 12 km away, up to about 2 km
//!   high, on a valley floor that runs to the horizon.
//!
//! Everything is real geometry, cut into clusters. The point is distance:
//! nothing but the sky reaches the mountains, and the renderer skips that
//! work (see `raster.rs`, light reach).

use glam::Vec3;

use crate::lights::normalise_colour;
use crate::scene::{PointLight, Scene};

/// Where the T-rex and the car stand, and which way they face (degrees
/// about Y).
pub const TREX_AT: Vec3 = Vec3::new(0.0, 0.0, 0.0);
pub const TREX_YAW: f32 = 30.0;
pub const CAR_AT: Vec3 = Vec3::new(7.5, 0.0, 5.0);
pub const CAR_YAW: f32 = -35.0;
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

/// The parts of the built-in world, each an object of its own that can be
/// moved, changed or deleted like any other: the names `part` makes.
pub const PARTS: [&str; 2] = ["ground", "mountains"];

/// What a scene file calls a built-in part as its asset.
pub fn asset_name(part: &str) -> String {
    format!("built-in:{part}")
}

/// The built-in part an asset name stands for, if any: a part of the
/// world or a primitive (`primitives.rs`).
pub fn part_of(asset: &std::path::Path) -> Option<&str> {
    asset.to_str()?.strip_prefix("built-in:").filter(|p| PARTS.contains(p) || crate::primitives::NAMES.contains(p))
}

/// One part of the built-in world as a scene of one object, its asset
/// `built-in:<name>`, an environment (open ground for bounce light, no
/// surface samples): the ground (the paved square and the valley floor to
/// the horizon) or the mountains.
pub fn part(name: &str) -> Option<Scene> {
    if crate::primitives::NAMES.contains(&name) {
        return crate::primitives::part(name);
    }
    let mut scene = Scene::new();
    match name {
        "ground" => {
            // The square: 1 m paving slabs, so the clusters stay small and
            // a light only touches the ones near it.
            let mut m = Mesh::default();
            m.ground(-50.0, 50.0, -50.0, 50.0, 0.0, 100, 100, |x, z| {
                let v = 0.24 + hash(x.floor() as i32, z.floor() as i32) * 0.05;
                Vec3::new(v, v * 0.97, v * 0.93)
            });
            m.into_scene(&mut scene);
            // Valley floor to the horizon, just below the square, in rings
            // of growing cell size: 10 m cells out to a few hundred metres,
            // then 100 m, then 500 m. Small cells near the square keep its
            // clusters small, so a light there only touches the ground near
            // it.
            let mut m = Mesh::default();
            let grass = |x: f32, z: f32| {
                let v = hash((x / 37.0) as i32, (z / 37.0) as i32) * 0.03;
                Vec3::new(0.09 + v, 0.12 + v, 0.06)
            };
            let plaza = [-50.0, 50.0, -50.0, 50.0];
            let r1 = [-400.0, 400.0, -450.0, 400.0];
            let r2 = [-4000.0, 4000.0, -4000.0, 4000.0];
            let r3 = [-24000.0, 24000.0, -16000.0, 8000.0];
            m.ring(r1, plaza, -0.05, 10.0, grass);
            m.ring(r2, r1, -0.05, 100.0, grass);
            m.ring(r3, r2, -0.05, 500.0, grass);
            m.into_scene(&mut scene);
            // Paving and grass: rough, a dull sheen at grazing angles.
            scene.materials = vec![crate::scene::Material { name: "ground".into(), roughness: 0.85, ..Default::default() }];
        }
        "mountains" => {
            mountains(&mut scene);
            scene.materials = vec![crate::scene::Material { name: "rock".into(), roughness: 0.85, ..Default::default() }];
        }
        _ => return None,
    }
    let n = scene.clusters.len();
    let i = scene.add_object(name.to_string(), 0..n, Vec3::ZERO, Vec3::ZERO);
    scene.objects[i].asset = Some(asset_name(name).into());
    scene.objects[i].environment = true;
    Some(scene)
}

/// The ground and the mountains, each an object (`part`). No lights: see
/// `set_lights`.
pub fn build() -> Scene {
    let mut scene = Scene::new();
    for name in PARTS {
        scene.append(part(name).expect("a built-in part"));
    }
    // Until the models are added, the subject is the middle of the plane.
    scene.focus = Some(crate::scene::set_middle());
    scene
}

/// The lights of the set: a warm key, a cool fill and a rim, sized like
/// film lights on a night exterior.
pub fn set_lights() -> Vec<PointLight> {
    vec![
        // An 18 kW HMI's worth, high and to the front left.
        PointLight { position: Vec3::new(-14.0, 16.0, 18.0), color: normalise_colour(Vec3::new(1.0, 0.85, 0.7)), lumens: 1_000_000.0, ..PointLight::DEFAULT },
        PointLight { position: Vec3::new(16.0, 5.0, 6.0), color: normalise_colour(Vec3::new(0.7, 0.8, 1.0)), lumens: 40_000.0, ..PointLight::DEFAULT },
        PointLight { position: Vec3::new(3.0, 10.0, -16.0), color: normalise_colour(Vec3::new(1.0, 1.0, 1.0)), lumens: 200_000.0, ..PointLight::DEFAULT },
    ]
}

/// Two small lamps near the origin (400 lm and 120 lm), whose reach is a
/// few metres: for tests that need a light to touch only part of the set.

#[cfg(test)]
pub fn small_lights() -> Vec<PointLight> {
    vec![
        PointLight { position: Vec3::new(-0.2, 1.1, 0.22), color: normalise_colour(Vec3::new(1.0, 0.8, 0.6)), lumens: 400.0, ..PointLight::DEFAULT },
        PointLight { position: Vec3::new(0.3, 1.0, -0.25), color: normalise_colour(Vec3::new(0.65, 0.8, 1.0)), lumens: 120.0, ..PointLight::DEFAULT },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ground_and_mountains_are_where_they_say() {
        let s = build();
        let mut ground = 0;
        let mut far = 0;
        for c in &s.clusters {
            if c.centre.length() < 30.0 && c.centre.y.abs() < 0.5 {
                ground += 1;
            }
            if c.centre.z < MOUNTAINS_NEAR_Z + 500.0 && c.centre.y > 100.0 {
                far += 1;
            }
        }
        assert!(ground > 50, "{ground} ground clusters near the origin");
        assert!(far > 100, "{far} mountain clusters");
        // Nothing stands on the plane but what main.rs places there.
        assert!(s.clusters.iter().filter(|c| c.centre.length() < 100.0).all(|c| c.centre.y < 0.5));
        let peak = s.positions().iter().map(|p| p.y).fold(0.0, f32::max);
        assert!(peak > 1200.0 && peak < 2500.0, "peak {peak}");
    }

    #[test]
    fn mountains_start_flat_at_the_near_edge() {
        for x in [-5000.0, 0.0, 3000.0] {
            assert!(mountain_height(x, MOUNTAINS_NEAR_Z).abs() < 1.0);
        }
    }
}
