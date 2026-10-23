//! Baking a model for FkJensen: levels of detail, texture tiles, and the
//! scene that is written to .fkj. Shared by fkj-import and the editor's
//! importer window (src/editor in FkJensen).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use glam::{Mat4, Vec2, Vec3};
use rayon::prelude::*;

use crate::lod;
use crate::model::{self, Model};
use crate::scene::Scene;
use crate::texture::{self, Source, TexEntry, MAX_LEVEL};

/// The model file extensions `load` reads.
pub const FORMATS: [&str; 4] = ["glb", "gltf", "fbx", "obj"];

/// Read a model of any format `FORMATS` lists. `texture` gives an OBJ its
/// base colour; `tex_dirs` are searched for an FBX's textures.
pub fn load(input: &Path, texture: Option<&Path>, tex_dirs: &[PathBuf]) -> Result<Model, String> {
    let ext = input.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
    match ext.as_str() {
        "glb" | "gltf" => crate::gltf_in::load(input),
        "fbx" => crate::fbx::load(input, tex_dirs),
        "obj" => crate::obj::load(input, texture),
        _ => Err(format!("{}: only .glb, .gltf, .fbx and .obj are read", input.display())),
    }
}

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

/// A triangle to bake: its corners as model vertices, and the model
/// triangle whose material it has (itself at level 0; the one it was made
/// from on coarser levels).
#[derive(Clone, Copy)]
pub struct BakeTri {
    pub corners: [u32; 3],
    pub material_of: u32,
}

impl BakeTri {
    fn of(m: &Model, t: usize) -> Self {
        BakeTri { corners: m.tris[t], material_of: t as u32 }
    }
}

/// Area of a triangle's UVs in source texels, and its texture.
fn texel_area(m: &Model, t: BakeTri) -> Option<(f32, &model::Texture)> {
    let mat = &m.materials[m.tri_material[t.material_of as usize] as usize];
    let tex = &m.textures[mat.texture?];
    let (w, h) = tex.size();
    let [a, b, c] = t.corners.map(|i| m.uvs[i as usize] * Vec2::new(w as f32, h as f32));
    Some((((b - a).perp_dot(c - a)).abs() * 0.5, tex))
}

/// The finest tile level a triangle needs (0: vertex colours are enough,
/// or untextured). Points no closer than `--texel`, nor than `spacing`
/// metres: a coarser level of detail is only drawn where its error is
/// under a micropolygon, so its tiles need be no finer than its error.
fn wanted_level(m: &Model, t: BakeTri, o: &Options, spacing: f32) -> u8 {
    let Some((area, _)) = texel_area(m, t) else { return 0 };
    // Two texels or fewer: the three vertex colours hold them.
    if area <= 2.0 {
        return 0;
    }
    // Points at level L are about (2^L)^2 / 2: as many as the texels.
    let l_src = (2.0 * area).sqrt().log2().ceil();
    let [a, b, c] = t.corners.map(|i| m.positions[i as usize]);
    let edge = (a - b).length().max((b - c).length()).max((c - a).length());
    let l_world = (edge / o.texel.max(spacing * 2.0)).max(1.0).log2().floor();
    (l_src.min(l_world).max(0.0) as u8).min(o.max_level)
}

/// Linear colour of triangle `t` at barycentric (b1, b2), filtered for
/// points `spacing` texels apart.
fn colour_at(m: &Model, t: BakeTri, b1: f32, b2: f32, levels_per_edge: f32) -> Vec3 {
    let mat = &m.materials[m.tri_material[t.material_of as usize] as usize];
    let Some(ti) = mat.texture else { return mat.color };
    let tex = &m.textures[ti];
    let [ua, ub, uc] = t.corners.map(|i| m.uvs[i as usize]);
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
    /// Triangles and clusters per level of detail.
    pub lod: lod::LodStats,
}

/// Cluster the model, build its levels of detail (unless `levels` is
/// false), and bake the tiles of every level.
pub fn bake(m: &Model, o: &Options, levels: bool) -> Baked {
    bake_logged(m, o, levels, &|s| eprintln!("  {s}"))
}

/// `bake`, telling `log` how the levels of detail are coming along.
pub fn bake_logged(m: &Model, o: &Options, levels: bool, log: &dyn Fn(String)) -> Baked {
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
            colour_at(m, BakeTri::of(m, t), b1, b2, 1.0)
        })
        .collect();
    let normals = if m.normals.len() == m.positions.len() && m.normals.iter().all(|n| n.length_squared() >= 0.5) {
        m.normals.clone()
    } else {
        let mut acc = vec![Vec3::ZERO; m.positions.len()];
        for t in &m.tris {
            let [a, b, c] = t.map(|i| i as usize);
            let f = (m.positions[b] - m.positions[a]).cross(m.positions[c] - m.positions[a]);
            acc[a] += f;
            acc[b] += f;
            acc[c] += f;
        }
        acc.into_iter().map(|v| v.normalize_or(Vec3::Y)).collect()
    };
    let mut scene = Scene::new();
    // The triangle each scene triangle bakes from.
    let (tris, lod_stats): (Vec<BakeTri>, lod::LodStats) = if levels {
        let (clusters, stats) = lod::build_logged(&m.positions, &m.tris, log);
        let corners = scene.push_lod_mesh(&m.positions, &normals, &colors, &clusters);
        let sources = clusters.iter().flat_map(|c| c.source.iter().copied());
        (corners.into_iter().zip(sources).map(|(corners, s)| BakeTri { corners, material_of: s }).collect(), stats)
    } else {
        let flat: Vec<u32> = m.tris.iter().flatten().copied().collect();
        let source = scene.push_mesh_mapped(&m.positions, &normals, &colors, &flat);
        let stats = lod::LodStats { tris: vec![m.tris.len()], clusters: vec![scene.clusters.len()] };
        (source.iter().map(|&s| BakeTri::of(m, s as usize)).collect(), stats)
    };
    // Surface samples and what each sees of the sky past the model itself
    // (src/samples.rs, src/transfer.rs in FkJensen).
    let ts = std::time::Instant::now();
    let (n, h) = scene.scatter_over(0..scene.clusters.len());
    crate::transfer::compute(&mut scene.samples);
    log(format!("sky transfer: {n} surface samples, at least {:.1} cm apart, in {:.1} s", h * 100.0, ts.elapsed().as_secs_f64()));
    let mut error = vec![0f32; tris.len()];
    for c in &scene.clusters {
        error[c.first_tri as usize..(c.first_tri + c.tri_count) as usize].fill(c.lod_error);
    }
    let wanted: Vec<u8> = tris.par_iter().zip(&error).map(|(&t, &e)| wanted_level(m, t, o, e)).collect();
    // Each cluster's finest level and its place in the pool, then the
    // tiles baked in parallel straight into it.
    let cluster_levels: Vec<u8> = scene.clusters.iter().map(|c| (c.first_tri as usize..(c.first_tri + c.tri_count) as usize).map(|t| wanted[t]).max().unwrap_or(0)).collect();
    let mut levels = vec![0usize; MAX_LEVEL as usize + 1];
    let mut entries = Vec::with_capacity(scene.clusters.len());
    let mut total = 0usize;
    for (c, &level) in scene.clusters.iter().zip(&cluster_levels) {
        levels[level as usize] += 1;
        if level == 0 {
            entries.push(TexEntry::default());
        } else {
            entries.push(TexEntry { level, source: 0, offset: total as u64 });
            total += texture::level_offset(c.tri_count as usize, level + 1) as usize;
        }
    }
    let mut pool = vec![0u8; total];
    let mut slices: Vec<&mut [u8]> = Vec::with_capacity(scene.clusters.len());
    let mut rest: &mut [u8] = &mut pool;
    for (c, &level) in scene.clusters.iter().zip(&cluster_levels) {
        let len = if level == 0 { 0 } else { texture::level_offset(c.tri_count as usize, level + 1) as usize };
        let (head, tail) = rest.split_at_mut(len);
        slices.push(head);
        rest = tail;
    }
    slices.into_par_iter().zip(scene.clusters.par_iter()).zip(cluster_levels.par_iter()).for_each(|((out, c), &level)| {
        let range = c.first_tri as usize..(c.first_tri + c.tri_count) as usize;
        let mut k = 0;
        for l in 1..=level {
            let n = 1usize << l;
            let inv = 1.0 / n as f32;
            for t in range.clone() {
                for i in 0..=n {
                    for j in 0..=(n - i) {
                        let col = colour_at(m, tris[t], i as f32 * inv, j as f32 * inv, n as f32);
                        out[k..k + 3].copy_from_slice(&[linear_to_srgb8(col.x), linear_to_srgb8(col.y), linear_to_srgb8(col.z)]);
                        k += 3;
                    }
                }
            }
        }
        debug_assert_eq!(k, out.len());
    });
    let pool_bytes = pool.len();
    scene.textures.entries = entries;
    scene.textures.sources = vec![Arc::new(Source::Memory(Arc::new(pool)))];
    Baked { scene, levels, pool_bytes, lod: lod_stats }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fkj;
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
        let baked = bake(&m, &Options { texel: 0.001, max_level: 5 }, false);
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
            let corners = [s2.positions()[s2.indices()[i] as usize], s2.positions()[s2.indices()[i + 1] as usize], s2.positions()[s2.indices()[i + 2] as usize]];
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
        let t = BakeTri::of(&m, 0);
        assert_eq!(wanted_level(&m, t, &Options { texel: 0.002, max_level: 5 }, 0.0), 2);
        assert_eq!(wanted_level(&m, t, &Options { texel: 0.0001, max_level: 5 }, 0.0), 4);
        // A coarse level with 2.5 mm of error: points 5 mm apart at most.
        assert_eq!(wanted_level(&m, t, &Options { texel: 0.0001, max_level: 5 }, 0.0025), 1);
        m.materials[0].texture = None;
        assert_eq!(wanted_level(&m, t, &Options { texel: 0.002, max_level: 5 }, 0.0), 0);
    }
}
