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
//!
//! Levels of detail (src/lod.rs in FkJensen): the mesh is built into a
//! cluster hierarchy, each level about half the triangles of the one below,
//! and every level gets its own tiles from the same textures. `--no-lod`
//! keeps the mesh as made only.

// The modules shared with FkJensen (both ways) each have parts only the
// other side uses.
#![allow(dead_code)]

#[path = "../../../src/fkj.rs"]
mod fkj;
#[path = "../../../src/scene.rs"]
mod scene;
#[path = "../../../src/texture.rs"]
mod texture;
#[path = "../../../src/samples.rs"]
mod samples;
#[path = "../../../src/lod.rs"]
mod lod;

mod fbx;
mod gltf_in;
mod obj;
mod model;

mod bake;

use std::path::PathBuf;

use bake::{bake, normalise, Options};
use texture::MAX_LEVEL;

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
    let mut levels = true;
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
            "--no-lod" => levels = false,
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
                eprintln!("usage: fkj-import model.(glb|gltf|fbx|obj) -o out.fkj [--length M | --height M] [--rotate-x DEG] [--rotate-y DEG] [--texel M] [--max-level N] [--textures DIR] [--texture FILE (OBJ)] [--no-lod]");
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
    let model = bake::load(&input, texture.as_deref(), &tex_dirs);
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
    let baked = bake(&model, &opts, levels);
    eprintln!(
        "levels of detail: {}",
        baked.lod.tris.iter().zip(&baked.lod.clusters).enumerate().map(|(l, (t, c))| format!("pass {l}: {t} triangles in {c} clusters")).collect::<Vec<_>>().join(", ")
    );
    if let Some(dir) = out.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let written = std::fs::File::create(&out).and_then(|f| {
        let mut w = std::io::BufWriter::with_capacity(1 << 20, f);
        fkj::write_to(&mut w, &baked.scene, &[])?;
        std::io::Write::flush(&mut w)
    });
    if let Err(e) = written {
        eprintln!("cannot write {}: {e}", out.display());
        std::process::exit(1);
    }
    let bytes = std::fs::metadata(&out).map(|m| m.len() as usize).unwrap_or(0);
    let hist: Vec<String> = baked.levels.iter().enumerate().filter(|(_, n)| **n > 0).map(|(l, n)| format!("level {l}: {n}")).collect();
    eprintln!(
        "wrote {}: {} clusters ({}), tiles {:.1} MB of {:.1} MB, in {:.1} s",
        out.display(),
        baked.scene.clusters.len(),
        hist.join(", "),
        baked.pool_bytes as f64 / 1e6,
        bytes as f64 / 1e6,
        t0.elapsed().as_secs_f64()
    );
}
