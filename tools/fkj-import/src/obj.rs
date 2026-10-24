//! Wavefront OBJ into a `Model`: positions, UVs and normals (computed
//! smooth when the file has none), polygons fanned into triangles. Materials
//! are not read: one material, with the texture given by `--texture`, if any.
//! Written for large sculpt exports (millions of polygons): one pass over
//! the file, no per-line allocation.

use std::collections::HashMap;
use std::path::Path;

use glam::{Vec2, Vec3};

use crate::model::{Material, Model, Texture};

fn floats<const N: usize>(s: &str) -> Option<[f32; N]> {
    let mut out = [0f32; N];
    let mut it = s.split_ascii_whitespace();
    for o in out.iter_mut() {
        *o = it.next()?.parse().ok()?;
    }
    Some(out)
}

/// An OBJ index (1-based, or negative from the end) to 0-based.
fn index(s: &str, count: usize) -> Option<usize> {
    let i: i64 = s.parse().ok()?;
    if i > 0 {
        Some(i as usize - 1)
    } else if i < 0 && (-i) as usize <= count {
        Some(count - (-i) as usize)
    } else {
        None
    }
}

pub fn load(path: &Path, texture: Option<&Path>) -> Result<Model, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut pos: Vec<Vec3> = Vec::new();
    let mut uv: Vec<Vec2> = Vec::new();
    let mut nrm: Vec<Vec3> = Vec::new();
    // Welded corners: (position, uv, normal) index triple to model vertex.
    let mut weld: HashMap<(u32, u32, u32), u32> = HashMap::new();
    let mut m = Model::default();
    let mut vert_pos: Vec<u32> = Vec::new();
    let mut corners: Vec<u32> = Vec::with_capacity(8);
    for (ln, line) in text.lines().enumerate() {
        let line = line.trim_start();
        let (tag, rest) = match line.find(|c: char| c.is_ascii_whitespace()) {
            Some(k) => (&line[..k], &line[k..]),
            None => continue,
        };
        match tag {
            "v" => pos.push(Vec3::from(floats::<3>(rest).ok_or_else(|| format!("line {}: bad v", ln + 1))?)),
            "vt" => {
                let mut it = rest.split_ascii_whitespace();
                let u: f32 = it.next().and_then(|s| s.parse().ok()).ok_or_else(|| format!("line {}: bad vt", ln + 1))?;
                let v: f32 = it.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                // OBJ's v runs up the image; the textures here run down.
                uv.push(Vec2::new(u, 1.0 - v));
            }
            "vn" => nrm.push(Vec3::from(floats::<3>(rest).ok_or_else(|| format!("line {}: bad vn", ln + 1))?)),
            "f" => {
                corners.clear();
                for c in rest.split_ascii_whitespace() {
                    let mut p = c.split('/');
                    let vi = p.next().and_then(|s| index(s, pos.len())).ok_or_else(|| format!("line {}: bad face", ln + 1))?;
                    let ti = p.next().filter(|s| !s.is_empty()).and_then(|s| index(s, uv.len())).map(|i| i as u32).unwrap_or(u32::MAX);
                    let ni = p.next().filter(|s| !s.is_empty()).and_then(|s| index(s, nrm.len())).map(|i| i as u32).unwrap_or(u32::MAX);
                    let key = (vi as u32, ti, ni);
                    let id = *weld.entry(key).or_insert_with(|| {
                        m.positions.push(pos[vi]);
                        m.uvs.push(if ti == u32::MAX { Vec2::ZERO } else { uv[ti as usize] });
                        m.normals.push(if ni == u32::MAX { Vec3::ZERO } else { nrm[ni as usize] });
                        vert_pos.push(vi as u32);
                        (m.positions.len() - 1) as u32
                    });
                    corners.push(id);
                }
                for k in 1..corners.len().saturating_sub(1) {
                    m.tris.push([corners[0], corners[k], corners[k + 1]]);
                    m.tri_material.push(0);
                }
            }
            _ => {}
        }
    }
    drop(text);
    drop(weld);
    if m.tris.is_empty() {
        return Err(format!("{}: no faces", path.display()));
    }
    // Smooth normals per position (so UV seams do not crease), area-weighted.
    if m.normals.iter().any(|n| n.length_squared() < 0.5) {
        let mut acc = vec![Vec3::ZERO; pos.len()];
        for t in &m.tris {
            let [a, b, c] = t.map(|i| m.positions[i as usize]);
            let f = (b - a).cross(c - a);
            for &i in t {
                acc[vert_pos[i as usize] as usize] += f;
            }
        }
        for (i, n) in m.normals.iter_mut().enumerate() {
            if n.length_squared() < 0.5 {
                *n = acc[vert_pos[i] as usize].normalize_or(Vec3::Y);
            }
        }
    }
    let tex = match texture {
        Some(p) => {
            m.textures.push(Texture::load(p)?);
            Some(0)
        }
        None => None,
    };
    m.materials.push(Material { name: "obj".into(), color: if tex.is_some() { Vec3::ONE } else { Vec3::splat(0.6) }, texture: tex, roughness: 0.5, metallic: 0.0 });
    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quads_fan_and_normals_are_made() {
        let dir = std::env::temp_dir().join(format!("fkj-obj-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("q.obj");
        std::fs::write(&p, "# quad\nv 0 0 0\nv 1 0 0\nv 1 0 -1\nv 0 0 -1\nvt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nf 1/1 2/2 3/3 4/4\n").unwrap();
        let m = load(&p, None).unwrap();
        assert_eq!(m.tris.len(), 2);
        assert_eq!(m.positions.len(), 4);
        assert!(m.normals.iter().all(|n| (*n - Vec3::Y).length() < 1e-5), "{:?}", m.normals);
        // v flipped: OBJ (0, 0) is the image's bottom left.
        assert_eq!(m.uvs[0], Vec2::new(0.0, 1.0));
    }
}
