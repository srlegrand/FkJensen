//! glTF 2.0 (.glb, .gltf) meshes into a `Model`: every mesh the default
//! scene's nodes reach, with their world transforms, the base colour factor
//! and texture of each primitive's material.

use std::collections::HashMap;
use std::path::Path;

use glam::{Mat4, Vec2, Vec3};

use crate::model::{Material, Model, Texture};

pub fn load(path: &Path) -> Result<Model, String> {
    let (doc, buffers, images) = gltf::import(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut model = Model::default();
    let mut tex_of_image: HashMap<usize, usize> = HashMap::new();
    let mut mat_of: HashMap<Option<usize>, u32> = HashMap::new();

    let scene = doc.default_scene().or_else(|| doc.scenes().next()).ok_or("glTF has no scene")?;
    let mut stack: Vec<(gltf::Node, Mat4)> = scene.nodes().map(|n| (n, Mat4::IDENTITY)).collect();
    while let Some((node, parent)) = stack.pop() {
        let m = parent * Mat4::from_cols_array_2d(&node.transform().matrix());
        for c in node.children() {
            stack.push((c, m));
        }
        let Some(mesh) = node.mesh() else { continue };
        let nm = m.inverse().transpose();
        let flip = m.determinant() < 0.0;
        for prim in mesh.primitives() {
            if prim.mode() != gltf::mesh::Mode::Triangles {
                continue;
            }
            let reader = prim.reader(|b| Some(&buffers[b.index()]));
            let Some(pos) = reader.read_positions() else { continue };
            let pos: Vec<Vec3> = pos.map(Vec3::from).collect();
            let nrm: Vec<Vec3> = reader.read_normals().map(|n| n.map(Vec3::from).collect()).unwrap_or_default();
            let mat = prim.material();
            let pbr = mat.pbr_metallic_roughness();
            let tex_info = pbr.base_color_texture();
            let set = tex_info.as_ref().map(|t| t.tex_coord()).unwrap_or(0);
            let uv: Vec<Vec2> = reader.read_tex_coords(set).map(|t| t.into_f32().map(Vec2::from).collect()).unwrap_or_default();
            let idx: Vec<u32> = match reader.read_indices() {
                Some(i) => i.into_u32().collect(),
                None => (0..pos.len() as u32).collect(),
            };

            // Material: base colour factor (linear) and texture.
            let key = mat.index();
            let mid = *mat_of.entry(key).or_insert_with(|| {
                let f = pbr.base_color_factor();
                let texture = tex_info.as_ref().map(|t| {
                    let img = t.texture().source().index();
                    *tex_of_image.entry(img).or_insert_with(|| {
                        let data = &images[img];
                        let rgb = to_rgb8(data);
                        model.textures.push(Texture::from_rgb8(format!("image {img}"), data.width as usize, data.height as usize, &rgb));
                        model.textures.len() - 1
                    })
                });
                model.materials.push(Material { name: mat.name().unwrap_or("material").to_string(), color: Vec3::new(f[0], f[1], f[2]), texture, roughness: pbr.roughness_factor(), metallic: pbr.metallic_factor(), alpha: None });
                (model.materials.len() - 1) as u32
            });

            let base = model.positions.len() as u32;
            for (i, p) in pos.iter().enumerate() {
                model.positions.push(m.transform_point3(*p));
                model.normals.push(nrm.get(i).map(|n| nm.transform_vector3(*n).normalize_or(Vec3::ZERO)).unwrap_or(Vec3::ZERO));
                model.uvs.push(uv.get(i).copied().unwrap_or(Vec2::ZERO));
            }
            for t in idx.chunks_exact(3) {
                let (a, b, c) = (base + t[0], base + t[1], base + t[2]);
                model.tris.push(if flip { [a, c, b] } else { [a, b, c] });
                model.tri_material.push(mid);
            }
        }
    }
    if model.tris.is_empty() {
        return Err("no triangle meshes in the glTF scene".into());
    }
    Ok(model)
}

fn to_rgb8(img: &gltf::image::Data) -> Vec<u8> {
    use gltf::image::Format::*;
    let p = &img.pixels;
    match img.format {
        R8G8B8 => p.clone(),
        R8G8B8A8 => p.chunks_exact(4).flat_map(|c| [c[0], c[1], c[2]]).collect(),
        R8 => p.iter().flat_map(|&v| [v, v, v]).collect(),
        R8G8 => p.chunks_exact(2).flat_map(|c| [c[0], c[0], c[0]]).collect(),
        R16G16B16 => p.chunks_exact(6).flat_map(|c| [c[1], c[3], c[5]]).collect(),
        R16G16B16A16 => p.chunks_exact(8).flat_map(|c| [c[1], c[3], c[5]]).collect(),
        _ => vec![200; img.width as usize * img.height as usize * 3],
    }
}
