//! Maya's polygon primitives, built in: a cube, a sphere, a cylinder and a
//! plane, a metre across, standing on the ground at the origin (their pivot
//! the middle of their base), mid grey, one material. A scene file names
//! them as the assets `built-in:cube` and so on.

use glam::Vec3;

use crate::scene::Scene;

pub const NAMES: [&str; 4] = ["cube", "sphere", "cylinder", "plane"];

const GREY: Vec3 = Vec3::splat(0.5);

/// Positions, normals and triangles (counter-clockwise seen from outside).
struct M {
    p: Vec<Vec3>,
    n: Vec<Vec3>,
    i: Vec<u32>,
}

impl M {
    fn new() -> M {
        M { p: Vec::new(), n: Vec::new(), i: Vec::new() }
    }

    /// A grid of `u` x `v` quads over a face: corner `o`, sides `a` and
    /// `b` (outward is `a` x `b`).
    fn face(&mut self, o: Vec3, a: Vec3, b: Vec3, u: usize, v: usize) {
        let n = a.cross(b).normalize();
        let base = self.p.len() as u32;
        for j in 0..=v {
            for k in 0..=u {
                self.p.push(o + a * (k as f32 / u as f32) + b * (j as f32 / v as f32));
                self.n.push(n);
            }
        }
        let w = (u + 1) as u32;
        for j in 0..v as u32 {
            for k in 0..u as u32 {
                let q = base + j * w + k;
                self.i.extend_from_slice(&[q, q + 1, q + w + 1, q, q + w + 1, q + w]);
            }
        }
    }
}

fn mesh(name: &str) -> Option<M> {
    let mut m = M::new();
    match name {
        "cube" => {
            // Each face a 4 x 4 grid, so the cluster bounds stay tight.
            let h = 0.5;
            let s = 4;
            m.face(Vec3::new(-h, 0.0, h), Vec3::X, Vec3::Y, s, s); // +Z
            m.face(Vec3::new(h, 0.0, -h), -Vec3::X, Vec3::Y, s, s); // -Z
            m.face(Vec3::new(h, 0.0, h), -Vec3::Z, Vec3::Y, s, s); // +X
            m.face(Vec3::new(-h, 0.0, -h), Vec3::Z, Vec3::Y, s, s); // -X
            m.face(Vec3::new(-h, 1.0, h), Vec3::X, -Vec3::Z, s, s); // +Y
            m.face(Vec3::new(-h, 0.0, -h), Vec3::X, Vec3::Z, s, s); // -Y
        }
        "plane" => m.face(Vec3::new(-0.5, 0.0, 0.5), Vec3::X, -Vec3::Z, 10, 10),
        "sphere" => {
            let (seg, rings) = (40usize, 20usize);
            for r in 0..=rings {
                let th = std::f32::consts::PI * r as f32 / rings as f32;
                for s in 0..=seg {
                    let ph = std::f32::consts::TAU * s as f32 / seg as f32;
                    let d = Vec3::new(th.sin() * ph.cos(), th.cos(), -th.sin() * ph.sin());
                    m.p.push(d * 0.5 + Vec3::new(0.0, 0.5, 0.0));
                    m.n.push(d);
                }
            }
            let w = (seg + 1) as u32;
            for r in 0..rings as u32 {
                for s in 0..seg as u32 {
                    let a = r * w + s;
                    m.i.extend_from_slice(&[a, a + w, a + w + 1, a, a + w + 1, a + 1]);
                }
            }
        }
        "cylinder" => {
            let seg = 40usize;
            let rows = 4usize;
            let ring = |s: usize| {
                let ph = std::f32::consts::TAU * s as f32 / seg as f32;
                Vec3::new(ph.cos(), 0.0, -ph.sin())
            };
            // The side.
            for j in 0..=rows {
                for s in 0..=seg {
                    let d = ring(s);
                    m.p.push(d * 0.5 + Vec3::Y * (j as f32 / rows as f32));
                    m.n.push(d);
                }
            }
            let w = (seg + 1) as u32;
            for j in 0..rows as u32 {
                for s in 0..seg as u32 {
                    let a = j * w + s;
                    m.i.extend_from_slice(&[a, a + 1, a + w + 1, a, a + w + 1, a + w]);
                }
            }
            // The caps.
            for (y, up) in [(0.0f32, -1.0f32), (1.0, 1.0)] {
                let c = m.p.len() as u32;
                m.p.push(Vec3::new(0.0, y, 0.0));
                m.n.push(Vec3::Y * up);
                for s in 0..=seg {
                    m.p.push(ring(s) * 0.5 + Vec3::Y * y);
                    m.n.push(Vec3::Y * up);
                }
                for s in 0..seg as u32 {
                    let (a, b) = (c + 1 + s, c + 2 + s);
                    if up > 0.0 {
                        m.i.extend_from_slice(&[c, a, b]);
                    } else {
                        m.i.extend_from_slice(&[c, b, a]);
                    }
                }
            }
        }
        _ => return None,
    }
    Some(m)
}

/// Primitive `name` as a scene of one object, its asset `built-in:<name>`.
pub fn part(name: &str) -> Option<Scene> {
    let m = mesh(name)?;
    let mut scene = Scene::new();
    let colors = vec![GREY; m.p.len()];
    scene.push_mesh(&m.p, &m.n, &colors, &m.i);
    scene.materials = vec![crate::scene::Material { name: format!("{name}1"), ..Default::default() }];
    let n = scene.clusters.len();
    let i = scene.add_object(name.to_string(), 0..n, Vec3::ZERO, Vec3::ZERO);
    scene.objects[i].asset = Some(crate::world::asset_name(name).into());
    // Its shadow field, made with it (a few thousand cells: quick).
    let (f, _) = crate::field::ShadowField::bake(&scene, 0..n, 48);
    scene.fields = vec![std::sync::Arc::new(f)];
    scene.objects[i].field = Some(0);
    scene.scatter_missing_samples();
    Some(scene)
}
