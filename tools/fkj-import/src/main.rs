//! fkj-import: turn a glTF (.glb, .gltf), binary FBX or OBJ model into an .fkj
//! scene for FkJensen, with its textures baked into per-triangle tiles.
//!
//! Usage:
//!   fkj-import model.glb -o model.fkj --length 12.3
//!   fkj-import car.fbx -o car.fkj --length 4.8 --textures path/to/textures
//!   fkj-import sculpt.obj -o sculpt.fkj --length 12.3 --texture colour.png
//!
//! The model is set on the ground (lowest point at y = 0), centred on x and
//! z, Y up, in metres: `--length` scales its longest horizontal side to that
//! many metres, `--height` its height. `--rotate-x` and `--rotate-y` turn it
//! first, in degrees.
//!
//! Textures: each triangle gets a tile per level, 1 to its finest, holding
//! the colour at the dicer's grid points for that level (src/texture.rs in
//! FkJensen). The finest level of a triangle is the coarser of: as many
//! points as the source texture has texels over it, and points no closer
//! than `--texel` metres (default 0.002). A cluster stores the finest level
//! any of its triangles needs. Each level is filtered from the source's mip
//! chain at its own spacing. Vertex colours hold the coarsest view.

#[path = "../../../src/fkj.rs"]
mod fkj;
#[path = "../../../src/scene.rs"]
mod scene;
#[path = "../../../src/texture.rs"]
mod texture;

mod fbx;
mod gltf_in;
mod obj;
mod model;

use std::path::PathBuf;
use std::sync::Arc;

use glam::{Mat4, Vec2, Vec3};
use rayon::prelude::*;

use model::Model;
use scene::Scene;
use texture::{Source, TexEntry, MAX_LEVEL};

/// Bake settings.
pub struct Options {
    /// Smallest spacing between texture points, metres.
    pub texel: f32,
    pub max_level: u8,
}

fn linear_to_srgb8(v: f32) -> u8 {
    let v = v.clamp(0.0, 1.0);
    let s = if v <= 0.0031308 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 };
    (s * 255.0).round() as u8
}

/// Area of a triangle's UVs in source texels, and its texture.
fn texel_area(m: &Model, t: usize) -> Option<(f32, &model::Texture)> {
    let mat = &m.materials[m.tri_material[t] as usize];
    let tex = &m.textures[mat.texture?];
    let (w, h) = tex.size();
    let [a, b, c] = m.tris[t].map(|i| m.uvs[i as usize] * Vec2::new(w as f32, h as f32));
    Some((((b - a).perp_dot(c - a)).abs() * 0.5, tex))
}

/// The finest tile level a source triangle needs (0: vertex colours are
/// enough, or untextured).
fn wanted_level(m: &Model, t: usize, o: &Options) -> u8 {
    let Some((area, _)) = texel_area(m, t) else { return 0 };
    // Two texels or fewer: the three vertex colours hold them.
    if area <= 2.0 {
        return 0;
    }
    // Points at level L are about (2^L)^2 / 2: as many as the texels.
    let l_src = (2.0 * area).sqrt().log2().ceil();
    let [a, b, c] = m.tris[t].map(|i| m.positions[i as usize]);
    let edge = (a - b).length().max((b - c).length()).max((c - a).length());
    let l_world = (edge / o.texel).max(1.0).log2().floor();
    (l_src.min(l_world).max(0.0) as u8).min(o.max_level)
}

/// Linear colour of source triangle `t` at barycentric (b1, b2), filtered
/// for points `spacing` texels apart.
fn colour_at(m: &Model, t: usize, b1: f32, b2: f32, levels_per_edge: f32) -> Vec3 {
    let mat = &m.materials[m.tri_material[t] as usize];
    let Some(ti) = mat.texture else { return mat.color };
    let tex = &m.textures[ti];
    let [ua, ub, uc] = m.tris[t].map(|i| m.uvs[i as usize]);
    let uv = ua * (1.0 - b1 - b2) + ub * b1 + uc * b2;
    let area = texel_area(m, t).map(|(a, _)| a).unwrap_or(1.0);
    // Distance between neighbouring points, in texels.
    let spacing = (2.0 * area).sqrt() / levels_per_edge;
    mat.color * tex.sample(uv, spacing.max(1.0).log2())
}

pub struct Baked {
    pub scene: Scene,
    pub levels: Vec<usize>,
    pub pool_bytes: usize,
}

/// Cluster the model and bake its tiles.
pub fn bake(m: &Model, o: &Options) -> Baked {
    // Vertex colours: the texture at each vertex, filtered over about one
    // triangle, from the first triangle that uses the vertex.
    let mut first_tri = vec![usize::MAX; m.positions.len()];
    for (t, tri) in m.tris.iter().enumerate() {
        for &v in tri {
            if first_tri[v as usize] == usize::MAX {
                first_tri[v as usize] = t;
            }
        }
    }
    let colors: Vec<Vec3> = (0..m.positions.len())
        .map(|v| {
            let t = first_tri[v];
            if t == usize::MAX {
                return Vec3::splat(0.7);
            }
            let k = m.tris[t].iter().position(|&x| x as usize == v).unwrap();
            let (b1, b2) = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)][k];
            colour_at(m, t, b1, b2, 1.0)
        })
        .collect();
    let flat: Vec<u32> = m.tris.iter().flatten().copied().collect();
    let normals = if m.normals.iter().any(|n| n.length_squared() < 0.5) { Vec::new() } else { m.normals.clone() };
    let mut scene = Scene::new();
    let source = scene.push_mesh_mapped(&m.positions, &normals, &colors, &flat);

    let wanted: Vec<u8> = (0..m.tris.len()).map(|t| wanted_level(m, t, o)).collect();
    let mut entries = Vec::with_capacity(scene.clusters.len());
    let mut levels = vec![0usize; MAX_LEVEL as usize + 1];
    // Each cluster's tiles, baked in parallel, then laid end to end.
    let blocks: Vec<(u8, Vec<u8>)> = scene
        .clusters
        .par_iter()
        .map(|c| {
            let tris: Vec<usize> = (c.first_tri..c.first_tri + c.tri_count).map(|t| source[t as usize] as usize).collect();
            let level = tris.iter().map(|&s| wanted[s]).max().unwrap_or(0);
            let mut data = Vec::new();
            for l in 1..=level {
                let n = 1usize << l;
                let inv = 1.0 / n as f32;
                for &s in &tris {
                    for i in 0..=n {
                        for j in 0..=(n - i) {
                            let col = colour_at(m, s, i as f32 * inv, j as f32 * inv, n as f32);
                            data.extend_from_slice(&[linear_to_srgb8(col.x), linear_to_srgb8(col.y), linear_to_srgb8(col.z)]);
                        }
                    }
                }
            }
            (level, data)
        })
        .collect();
    let mut pool = Vec::new();
    for (level, data) in blocks {
        levels[level as usize] += 1;
        if level == 0 {
            entries.push(TexEntry::default());
        } else {
            entries.push(TexEntry { level, source: 0, offset: pool.len() as u64 });
            pool.extend_from_slice(&data);
        }
    }
    let pool_bytes = pool.len();
    scene.textures.entries = entries;
    scene.textures.sources = vec![Arc::new(Source::Memory(Arc::new(pool)))];
    Baked { scene, levels, pool_bytes }
}

/// Turn, then scale and move the model onto the ground at the origin.
pub fn normalise(m: &mut Model, rotate_x: f32, rotate_y: f32, length: Option<f32>, height: Option<f32>) -> f32 {
    m.transform(Mat4::from_rotation_y(rotate_y.to_radians()) * Mat4::from_rotation_x(rotate_x.to_radians()));
    let (lo, hi) = m.bounds();
    let size = hi - lo;
    let scale = match (length, height) {
        (Some(l), _) => l / size.x.max(size.z).max(1e-9),
        (None, Some(h)) => h / size.y.max(1e-9),
        (None, None) => 1.0,
    };
    let centre = Vec3::new((lo.x + hi.x) * 0.5, lo.y, (lo.z + hi.z) * 0.5);
    m.transform(Mat4::from_scale(Vec3::splat(scale)) * Mat4::from_translation(-centre));
    scale
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut input: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut length = None;
    let mut height = None;
    let mut rotate_x = 0.0;
    let mut rotate_y = 0.0;
    let mut tex_dirs = Vec::new();
    let mut texture: Option<PathBuf> = None;
    let mut opts = Options { texel: 0.002, max_level: MAX_LEVEL };
    let mut i = 0;
    let num = |a: Option<&String>, what: &str| -> f32 {
        a.and_then(|s| s.parse().ok()).unwrap_or_else(|| {
            eprintln!("{what} takes a number");
            std::process::exit(2)
        })
    };
    while i < args.len() {
        match args[i].as_str() {
            "-o" | "--out" => {
                out = args.get(i + 1).map(PathBuf::from);
                i += 1;
            }
            "--length" => {
                length = Some(num(args.get(i + 1), "--length"));
                i += 1;
            }
            "--height" => {
                height = Some(num(args.get(i + 1), "--height"));
                i += 1;
            }
            "--rotate-x" => {
                rotate_x = num(args.get(i + 1), "--rotate-x");
                i += 1;
            }
            "--rotate-y" => {
                rotate_y = num(args.get(i + 1), "--rotate-y");
                i += 1;
            }
            "--texel" => {
                opts.texel = num(args.get(i + 1), "--texel");
                i += 1;
            }
            "--max-level" => {
                opts.max_level = (num(args.get(i + 1), "--max-level") as u8).clamp(1, MAX_LEVEL);
                i += 1;
            }
            "--texture" => {
                texture = args.get(i + 1).map(PathBuf::from);
                i += 1;
            }
            "--textures" => {
                if let Some(d) = args.get(i + 1) {
                    tex_dirs.push(PathBuf::from(d));
                }
                i += 1;
            }
            "-h" | "--help" => {
                eprintln!("usage: fkj-import model.(glb|gltf|fbx|obj) -o out.fkj [--length M | --height M] [--rotate-x DEG] [--rotate-y DEG] [--texel M] [--max-level N] [--textures DIR] [--texture FILE (OBJ)]");
                return;
            }
            a => input = Some(PathBuf::from(a)),
        }
        i += 1;
    }
    let Some(input) = input else {
        eprintln!("usage: fkj-import model.(glb|gltf|fbx|obj) -o out.fkj [--length M | --height M] ...");
        std::process::exit(2);
    };
    let out = out.unwrap_or_else(|| input.with_extension("fkj"));
    let t0 = std::time::Instant::now();
    let ext = input.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
    let model = match ext.as_str() {
        "glb" | "gltf" => gltf_in::load(&input),
        "fbx" => fbx::load(&input, &tex_dirs),
        "obj" => obj::load(&input, texture.as_deref()),
        _ => Err(format!("{}: only .glb, .gltf, .fbx and .obj are read", input.display())),
    };
    let mut model = model.unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });
    let scale = normalise(&mut model, rotate_x, rotate_y, length, height);
    let (lo, hi) = model.bounds();
    eprintln!(
        "{}: {} triangles, {} vertices, {} materials, {} textures ({}); scaled by {scale:.4} to {:.2} x {:.2} x {:.2} m",
        input.display(),
        model.tris.len(),
        model.positions.len(),
        model.materials.len(),
        model.textures.len(),
        model.textures.iter().map(|t| format!("{} {}x{}", t.name, t.size().0, t.size().1)).collect::<Vec<_>>().join(", "),
        hi.x - lo.x,
        hi.y - lo.y,
        hi.z - lo.z
    );
    let baked = bake(&model, &opts);
    let bytes = fkj::write(&baked.scene, &[]);
    if let Some(dir) = out.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    std::fs::write(&out, &bytes).unwrap_or_else(|e| {
        eprintln!("cannot write {}: {e}", out.display());
        std::process::exit(1);
    });
    let hist: Vec<String> = baked.levels.iter().enumerate().filter(|(_, n)| **n > 0).map(|(l, n)| format!("level {l}: {n}")).collect();
    eprintln!(
        "wrote {}: {} clusters ({}), tiles {:.1} MB of {:.1} MB, in {:.1} s",
        out.display(),
        baked.scene.clusters.len(),
        hist.join(", "),
        baked.pool_bytes as f64 / 1e6,
        bytes.len() as f64 / 1e6,
        t0.elapsed().as_secs_f64()
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec2;

    /// A 1 x 1 m quad textured with a 2 x 2 checker: the baked tiles hold
    /// the checker's colours at the right points, and reading the file back
    /// through FkJensen's streaming reader gives the same tiles.
    #[test]
    fn a_checker_quad_bakes_and_round_trips() {
        let mut m = Model::default();
        m.positions = vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, -1.0), Vec3::new(0.0, 0.0, -1.0)];
        m.normals = vec![Vec3::Y; 4];
        m.uvs = vec![Vec2::new(0.0, 1.0), Vec2::new(1.0, 1.0), Vec2::new(1.0, 0.0), Vec2::new(0.0, 0.0)];
        m.tris = vec![[0, 1, 2], [0, 2, 3]];
        m.tri_material = vec![0, 0];
        // 64 x 64, red left half, blue right half.
        let mut rgb = Vec::new();
        for _y in 0..64 {
            for x in 0..64 {
                rgb.extend_from_slice(if x < 32 { &[255, 0, 0] } else { &[0, 0, 255] });
            }
        }
        m.textures.push(model::Texture::from_rgb8("t".into(), 64, 64, &rgb));
        m.materials.push(model::Material { name: "m".into(), color: Vec3::ONE, texture: Some(0) });
        let baked = bake(&m, &Options { texel: 0.001, max_level: 5 });
        assert_eq!(baked.scene.clusters.len(), 1);
        let e = baked.scene.textures.entry(0);
        // 64 x 64 texels over two triangles: 2048 each, level 6 wanted,
        // capped at 5.
        assert_eq!(e.level, 5);
        // Write, read back from a file, and compare a tile point.
        let dir = std::env::temp_dir().join(format!("fkj-import-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("q.fkj");
        std::fs::write(&path, fkj::write(&baked.scene, &[])).unwrap();
        let (s2, _) = fkj::read_file(&path).unwrap();
        assert_eq!(s2.textures.entry(0).level, 5);
        let c = &s2.clusters[0];
        let b = s2.textures.block(0, c.tri_count as usize, 5).unwrap();
        for t in 0..c.tri_count as usize {
            let i = (c.first_tri as usize + t) * 3;
            let corners = [s2.positions[s2.indices[i] as usize], s2.positions[s2.indices[i + 1] as usize], s2.positions[s2.indices[i + 2] as usize]];
            let n = 32;
            for (gi, gj) in [(0usize, 0usize), (8, 4), (20, 3), (2, 25)] {
                if gi + gj > n {
                    continue;
                }
                let (b1, b2) = (gi as f32 / n as f32, gj as f32 / n as f32);
                let p = corners[0] * (1.0 - b1 - b2) + corners[1] * b1 + corners[2] * b2;
                let col = b.point(t, gi, gj);
                // Away from the seam at x = 0.5 the colour is pure.
                if (p.x - 0.5).abs() > 0.05 {
                    if p.x < 0.5 {
                        assert!(col.x > 0.9 && col.z < 0.1, "red expected at {p}: {col}");
                    } else {
                        assert!(col.z > 0.9 && col.x < 0.1, "blue expected at {p}: {col}");
                    }
                }
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn levels_follow_texels_and_the_world_cap() {
        let mut m = Model::default();
        m.positions = vec![Vec3::ZERO, Vec3::X * 0.01, Vec3::Z * -0.01];
        m.uvs = vec![Vec2::ZERO, Vec2::X, Vec2::Y];
        m.normals = vec![Vec3::Y; 3];
        m.tris = vec![[0, 1, 2]];
        m.tri_material = vec![0];
        m.textures.push(model::Texture::from_rgb8("t".into(), 16, 16, &vec![128; 16 * 16 * 3]));
        m.materials.push(model::Material { name: "m".into(), color: Vec3::ONE, texture: Some(0) });
        // 128 texels: level 4 by texels; a 1 cm triangle at 2 mm: level 2.
        assert_eq!(wanted_level(&m, 0, &Options { texel: 0.002, max_level: 5 }), 2);
        assert_eq!(wanted_level(&m, 0, &Options { texel: 0.0001, max_level: 5 }), 4);
        m.materials[0].texture = None;
        assert_eq!(wanted_level(&m, 0, &Options { texel: 0.002, max_level: 5 }), 0);
    }
}
