//! A reader for binary FBX (versions 7.x), enough for static meshes: the
//! node tree, then models, geometry, normals, UVs, materials and their
//! diffuse textures, and the model hierarchy's transforms.

use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

use glam::{EulerRot, Mat4, Quat, Vec2, Vec3};

use crate::model::{Material, Model};

#[derive(Debug, Clone)]
#[allow(dead_code)] // every FBX property type is read; not all are used
pub enum Prop {
    I16(i16),
    Bool(bool),
    I32(i32),
    F32(f32),
    F64(f64),
    I64(i64),
    F32s(Vec<f32>),
    F64s(Vec<f64>),
    I64s(Vec<i64>),
    I32s(Vec<i32>),
    Bools(Vec<bool>),
    Str(String),
    Raw(Vec<u8>),
}

impl Prop {
    fn str(&self) -> &str {
        match self {
            Prop::Str(s) => s,
            _ => "",
        }
    }
    fn i64(&self) -> i64 {
        match *self {
            Prop::I64(v) => v,
            Prop::I32(v) => v as i64,
            Prop::I16(v) => v as i64,
            _ => 0,
        }
    }
    fn f64(&self) -> f64 {
        match *self {
            Prop::F64(v) => v,
            Prop::F32(v) => v as f64,
            Prop::I32(v) => v as f64,
            Prop::I64(v) => v as f64,
            _ => 0.0,
        }
    }
    fn f64s(&self) -> Vec<f64> {
        match self {
            Prop::F64s(v) => v.clone(),
            Prop::F32s(v) => v.iter().map(|x| *x as f64).collect(),
            _ => Vec::new(),
        }
    }
    fn i32s(&self) -> Vec<i32> {
        match self {
            Prop::I32s(v) => v.clone(),
            Prop::I64s(v) => v.iter().map(|x| *x as i32).collect(),
            _ => Vec::new(),
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct Node {
    pub name: String,
    pub props: Vec<Prop>,
    pub children: Vec<Node>,
}

impl Node {
    pub fn child(&self, name: &str) -> Option<&Node> {
        self.children.iter().find(|c| c.name == name)
    }
    fn children_named<'a>(&'a self, name: &'a str) -> impl Iterator<Item = &'a Node> + 'a {
        self.children.iter().filter(move |c| c.name == name)
    }
    /// A `Properties70` entry's values (after its four header strings).
    fn p70(&self, key: &str) -> Option<&[Prop]> {
        let p = self.child("Properties70")?;
        p.children_named("P").find(|n| n.props.first().map(|x| x.str()) == Some(key)).map(|n| &n.props[4.min(n.props.len())..])
    }
    fn p70_vec3(&self, key: &str) -> Option<Vec3> {
        let v = self.p70(key)?;
        (v.len() >= 3).then(|| Vec3::new(v[0].f64() as f32, v[1].f64() as f32, v[2].f64() as f32))
    }
}

struct Reader<'a> {
    b: &'a [u8],
    at: usize,
    wide: bool,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], String> {
        let end = self.at.checked_add(n).filter(|e| *e <= self.b.len()).ok_or("FBX file is truncated")?;
        let s = &self.b[self.at..end];
        self.at = end;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn off(&mut self) -> Result<u64, String> {
        Ok(if self.wide { u64::from_le_bytes(self.take(8)?.try_into().unwrap()) } else { self.u32()? as u64 })
    }

    fn array<T>(&mut self, size: usize, conv: impl Fn(&[u8]) -> T) -> Result<Vec<T>, String> {
        let len = self.u32()? as usize;
        let enc = self.u32()?;
        let clen = self.u32()? as usize;
        let raw = self.take(clen)?;
        let data: Vec<u8> = if enc == 1 {
            let mut out = Vec::with_capacity(len * size);
            flate2::read::ZlibDecoder::new(raw).read_to_end(&mut out).map_err(|e| format!("FBX array: {e}"))?;
            out
        } else {
            raw.to_vec()
        };
        if data.len() < len * size {
            return Err("FBX array shorter than its length".into());
        }
        Ok(data[..len * size].chunks_exact(size).map(conv).collect())
    }

    fn prop(&mut self) -> Result<Prop, String> {
        let t = self.u8()?;
        Ok(match t {
            b'Y' => Prop::I16(i16::from_le_bytes(self.take(2)?.try_into().unwrap())),
            b'C' => Prop::Bool(self.u8()? != 0),
            b'I' => Prop::I32(i32::from_le_bytes(self.take(4)?.try_into().unwrap())),
            b'F' => Prop::F32(f32::from_le_bytes(self.take(4)?.try_into().unwrap())),
            b'D' => Prop::F64(f64::from_le_bytes(self.take(8)?.try_into().unwrap())),
            b'L' => Prop::I64(i64::from_le_bytes(self.take(8)?.try_into().unwrap())),
            b'f' => Prop::F32s(self.array(4, |c| f32::from_le_bytes(c.try_into().unwrap()))?),
            b'd' => Prop::F64s(self.array(8, |c| f64::from_le_bytes(c.try_into().unwrap()))?),
            b'l' => Prop::I64s(self.array(8, |c| i64::from_le_bytes(c.try_into().unwrap()))?),
            b'i' => Prop::I32s(self.array(4, |c| i32::from_le_bytes(c.try_into().unwrap()))?),
            b'b' => Prop::Bools(self.array(1, |c| c[0] != 0)?),
            b'S' => {
                let n = self.u32()? as usize;
                Prop::Str(String::from_utf8_lossy(self.take(n)?).into_owned())
            }
            b'R' => {
                let n = self.u32()? as usize;
                Prop::Raw(self.take(n)?.to_vec())
            }
            other => return Err(format!("FBX property type {:?} not known", other as char)),
        })
    }

    /// One node, or None at a null record.
    fn node(&mut self) -> Result<Option<Node>, String> {
        let end = self.off()? as usize;
        let n_props = self.off()?;
        let _list_len = self.off()?;
        let name_len = self.u8()? as usize;
        if end == 0 {
            return Ok(None);
        }
        let name = String::from_utf8_lossy(self.take(name_len)?).into_owned();
        let mut props = Vec::with_capacity(n_props as usize);
        for _ in 0..n_props {
            props.push(self.prop()?);
        }
        let mut children = Vec::new();
        while self.at < end {
            match self.node()? {
                Some(n) => children.push(n),
                None => break,
            }
        }
        self.at = end;
        Ok(Some(Node { name, props, children }))
    }
}

/// Parse a binary FBX file into its top-level nodes.
pub fn parse(bytes: &[u8]) -> Result<Vec<Node>, String> {
    if bytes.len() < 27 || &bytes[..20] != b"Kaydara FBX Binary  " {
        return Err("not a binary FBX file (ASCII FBX is not read)".into());
    }
    let version = u32::from_le_bytes(bytes[23..27].try_into().unwrap());
    let mut r = Reader { b: bytes, at: 27, wide: version >= 7500 };
    let mut out = Vec::new();
    while r.at + 13 < r.b.len() {
        match r.node()? {
            Some(n) => out.push(n),
            None => break,
        }
    }
    Ok(out)
}

/// Values per polygon-vertex from a layer element, resolving its mapping
/// and reference modes. `stride` floats per value.
fn layer_values(el: &Node, data_name: &str, index_name: &str, stride: usize, poly_vertex_to_vertex: &[usize], poly_of_corner: &[usize]) -> Option<Vec<Vec<f64>>> {
    let data = el.child(data_name)?.props.first()?.f64s();
    let mapping = el.child("MappingInformationType").and_then(|n| n.props.first()).map(|p| p.str().to_string()).unwrap_or_default();
    let reference = el.child("ReferenceInformationType").and_then(|n| n.props.first()).map(|p| p.str().to_string()).unwrap_or_default();
    let index = el.child(index_name).and_then(|n| n.props.first()).map(|p| p.i32s());
    let value = |i: usize| -> Vec<f64> { data.get(i * stride..i * stride + stride).map(|s| s.to_vec()).unwrap_or_else(|| vec![0.0; stride]) };
    let pick = |k: usize| -> Vec<f64> {
        if reference == "IndexToDirect" {
            let i = index.as_ref().and_then(|ix| ix.get(k)).copied().unwrap_or(0).max(0) as usize;
            value(i)
        } else {
            value(k)
        }
    };
    let n = poly_vertex_to_vertex.len();
    Some(match mapping.as_str() {
        "ByPolygonVertex" => (0..n).map(pick).collect(),
        "ByVertice" | "ByVertex" | "ByControlPoint" => (0..n).map(|c| pick(poly_vertex_to_vertex[c])).collect(),
        "ByPolygon" => (0..n).map(|c| pick(poly_of_corner[c])).collect(),
        "AllSame" => (0..n).map(|_| pick(0)).collect(),
        _ => return None,
    })
}

fn local_matrix(model: &Node) -> Mat4 {
    let t = model.p70_vec3("Lcl Translation").unwrap_or(Vec3::ZERO);
    let r = model.p70_vec3("Lcl Rotation").unwrap_or(Vec3::ZERO);
    let s = model.p70_vec3("Lcl Scaling").unwrap_or(Vec3::ONE);
    let pre = model.p70_vec3("PreRotation").unwrap_or(Vec3::ZERO);
    let post = model.p70_vec3("PostRotation").unwrap_or(Vec3::ZERO);
    // FBX's default rotation order eEulerXYZ: x first, then y, then z.
    let euler = |v: Vec3| Quat::from_euler(EulerRot::ZYX, v.z.to_radians(), v.y.to_radians(), v.x.to_radians());
    Mat4::from_translation(t) * Mat4::from_quat(euler(pre) * euler(r) * euler(post).inverse()) * Mat4::from_scale(s)
}

/// Load an FBX file's meshes into a `Model`, in the file's units and axes
/// (the hierarchy's transforms applied). Textures are found by file name
/// in the FBX's folder, `textures/` beside it or under it, or `tex_dirs`.
pub fn load(path: &Path, tex_dirs: &[std::path::PathBuf]) -> Result<Model, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let nodes = parse(&bytes)?;
    let objects = nodes.iter().find(|n| n.name == "Objects").ok_or("FBX has no Objects")?;
    let connections = nodes.iter().find(|n| n.name == "Connections");

    let by_id: HashMap<i64, &Node> = objects.children.iter().filter_map(|o| Some((o.props.first()?.i64(), o))).collect();
    // child -> parent links, in file order, and property links.
    let mut parent: HashMap<i64, i64> = HashMap::new();
    let mut children_of: HashMap<i64, Vec<i64>> = HashMap::new();
    let mut prop_links: Vec<(i64, i64, String)> = Vec::new();
    if let Some(c) = connections {
        for l in c.children_named("C") {
            let kind = l.props.first().map(|p| p.str()).unwrap_or("");
            let (a, b) = (l.props.get(1).map(|p| p.i64()).unwrap_or(0), l.props.get(2).map(|p| p.i64()).unwrap_or(0));
            match kind {
                "OO" => {
                    children_of.entry(b).or_default().push(a);
                    if by_id.get(&a).map(|n| n.name.as_str()) == Some("Model") {
                        parent.insert(a, b);
                    }
                }
                "OP" => prop_links.push((a, b, l.props.get(3).map(|p| p.str().to_string()).unwrap_or_default())),
                _ => {}
            }
        }
    }
    let world = |mut id: i64| -> Mat4 {
        let mut m = Mat4::IDENTITY;
        let mut guard = 0;
        while let Some(n) = by_id.get(&id).filter(|n| n.name == "Model") {
            m = local_matrix(n) * m;
            id = *parent.get(&id).unwrap_or(&0);
            guard += 1;
            if guard > 64 {
                break;
            }
        }
        m
    };

    let base = path.parent().unwrap_or(Path::new("."));
    let mut dirs: Vec<std::path::PathBuf> = vec![base.to_path_buf(), base.join("textures"), base.join("../textures")];
    dirs.extend(tex_dirs.iter().cloned());

    let mut model = Model::default();
    let mut material_index: HashMap<i64, u32> = HashMap::new();
    let mut texture_cache: HashMap<String, Option<usize>> = HashMap::new();

    for (&geom_id, geom) in by_id.iter().filter(|(_, n)| n.name == "Geometry" && n.props.get(2).map(|p| p.str()) == Some("Mesh")) {
        let Some(verts) = geom.child("Vertices").and_then(|n| n.props.first()).map(|p| p.f64s()) else { continue };
        let Some(pvi) = geom.child("PolygonVertexIndex").and_then(|n| n.props.first()).map(|p| p.i32s()) else { continue };
        // The models that use this geometry.
        let models: Vec<i64> = children_of.iter().filter(|(_, ch)| ch.contains(&geom_id)).map(|(p, _)| *p).filter(|p| by_id.get(p).map(|n| n.name.as_str()) == Some("Model")).collect();
        let corner_vertex: Vec<usize> = pvi.iter().map(|&i| if i < 0 { (-i - 1) as usize } else { i as usize }).collect();
        let mut poly_of_corner = Vec::with_capacity(pvi.len());
        let mut polys: Vec<(usize, usize)> = Vec::new();
        let mut start = 0;
        for (k, &i) in pvi.iter().enumerate() {
            poly_of_corner.push(polys.len());
            if i < 0 {
                polys.push((start, k + 1));
                start = k + 1;
            }
        }
        let normals = geom.child("LayerElementNormal").and_then(|el| layer_values(el, "Normals", "NormalsIndex", 3, &corner_vertex, &poly_of_corner));
        let uvs = geom.child("LayerElementUV").and_then(|el| layer_values(el, "UV", "UVIndex", 2, &corner_vertex, &poly_of_corner));
        let mats: Option<Vec<i32>> = geom.child("LayerElementMaterial").map(|el| {
            let m = el.child("Materials").and_then(|n| n.props.first()).map(|p| p.i32s()).unwrap_or_default();
            let mapping = el.child("MappingInformationType").and_then(|n| n.props.first()).map(|p| p.str().to_string()).unwrap_or_default();
            if mapping == "AllSame" { vec![m.first().copied().unwrap_or(0); polys.len()] } else { m }
        });

        for &model_id in &models {
            let m = world(model_id);
            let nm = m.inverse().transpose();
            // The model's materials, in connection order: the index the
            // geometry's material layer uses.
            let model_mats: Vec<i64> = children_of.get(&model_id).map(|v| v.iter().copied().filter(|c| by_id.get(c).map(|n| n.name.as_str()) == Some("Material")).collect()).unwrap_or_default();
            let mut local_to_global: Vec<u32> = Vec::new();
            for &mid in &model_mats {
                let g = *material_index.entry(mid).or_insert_with(|| {
                    let mn = by_id[&mid];
                    let color = mn.p70_vec3("DiffuseColor").or_else(|| mn.p70_vec3("Diffuse")).unwrap_or(Vec3::splat(0.8));
                    // The texture linked to the diffuse colour, and the one
                    // linked to transparency or opacity (a cut-out mask).
                    let tex_node = prop_links.iter().find(|(_, b, p)| *b == mid && (p.contains("Diffuse") || p.contains("BaseColor") || p.contains("Base Color"))).and_then(|(a, _, _)| by_id.get(a));
                    let alpha_node = prop_links.iter().find(|(_, b, p)| *b == mid && (p.contains("Transparen") || p.contains("Opacity"))).and_then(|(a, _, _)| by_id.get(a));
                    let mut load_tex = |t: &&Node| -> Option<usize> {
                        let file = t.child("RelativeFilename").or_else(|| t.child("FileName")).and_then(|n| n.props.first()).map(|p| p.str().to_string())?;
                        let base_name = file.rsplit(['\\', '/']).next().unwrap_or(&file).to_string();
                        texture_cache
                            .entry(base_name.clone())
                            .or_insert_with(|| {
                                let found = dirs.iter().map(|d| d.join(&base_name)).find(|p| p.exists());
                                match found {
                                    Some(p) => match crate::model::Texture::load(&p) {
                                        Ok(t) => {
                                            model.textures.push(t);
                                            Some(model.textures.len() - 1)
                                        }
                                        Err(e) => {
                                            eprintln!("  {e}");
                                            model.warnings.push(e);
                                            None
                                        }
                                    },
                                    None => {
                                        eprintln!("  texture {base_name} not found");
                                        model.warnings.push(format!("texture {base_name} not found"));
                                        None
                                    }
                                }
                            })
                            .clone()
                    };
                    let tex = tex_node.and_then(&mut load_tex);
                    let alpha = alpha_node.and_then(&mut load_tex);
                    let name = mn.props.get(1).map(|p| p.str().split('\0').next().unwrap_or("").to_string()).unwrap_or_default();
                    // With a texture, the texture is the colour: Blender
                    // writes a placeholder diffuse colour alongside it.
                    model.materials.push(Material { name, color: if tex.is_some() { Vec3::ONE } else { color }, texture: tex, roughness: 0.5, metallic: 0.0, alpha });
                    (model.materials.len() - 1) as u32
                });
                local_to_global.push(g);
            }
            if local_to_global.is_empty() {
                model.materials.push(Material { name: "default".into(), color: Vec3::splat(0.7), texture: None, roughness: 0.5, metallic: 0.0, alpha: None });
                local_to_global.push((model.materials.len() - 1) as u32);
            }

            // Vertices per polygon corner, welded where position, normal and
            // UV agree.
            let mut weld: HashMap<(usize, [u32; 3], [u32; 2]), u32> = HashMap::new();
            for (pi, &(a, b)) in polys.iter().enumerate() {
                let mat = mats.as_ref().and_then(|v| v.get(pi)).copied().unwrap_or(0).max(0) as usize;
                let mat = local_to_global[mat.min(local_to_global.len() - 1)];
                let mut ids = Vec::with_capacity(b - a);
                for c in a..b {
                    let v = corner_vertex[c];
                    let p = Vec3::new(verts[v * 3] as f32, verts[v * 3 + 1] as f32, verts[v * 3 + 2] as f32);
                    let n = normals.as_ref().map(|n| Vec3::new(n[c][0] as f32, n[c][1] as f32, n[c][2] as f32)).unwrap_or(Vec3::ZERO);
                    let uv = uvs.as_ref().map(|u| Vec2::new(u[c][0] as f32, u[c][1] as f32)).unwrap_or(Vec2::ZERO);
                    let key = (v, [n.x.to_bits(), n.y.to_bits(), n.z.to_bits()], [uv.x.to_bits(), uv.y.to_bits()]);
                    let id = *weld.entry(key).or_insert_with(|| {
                        model.positions.push(m.transform_point3(p));
                        model.normals.push(nm.transform_vector3(n).normalize_or(Vec3::ZERO));
                        // FBX UVs have v up; textures are stored top row first.
                        model.uvs.push(Vec2::new(uv.x, 1.0 - uv.y));
                        (model.positions.len() - 1) as u32
                    });
                    ids.push(id);
                }
                let flip = m.determinant() < 0.0;
                for k in 1..ids.len().saturating_sub(1) {
                    let t = if flip { [ids[0], ids[k + 1], ids[k]] } else { [ids[0], ids[k], ids[k + 1]] };
                    model.tris.push(t);
                    model.tri_material.push(mat);
                }
            }
        }
    }
    if model.tris.is_empty() {
        return Err("no meshes found in the FBX".into());
    }
    // A material with no mask that shares its colour texture with one
    // that has a mask (a second leaf material over the same leaf atlas)
    // takes that mask: the atlas's cut-out is the leaves' shape.
    for k in 0..model.materials.len() {
        if model.materials[k].alpha.is_some() {
            continue;
        }
        let Some(t) = model.materials[k].texture else { continue };
        if let Some(src) = model.materials.iter().find(|m| m.texture == Some(t) && m.alpha.is_some()) {
            let (a, from) = (src.alpha, src.name.clone());
            let msg = format!("{}: no mask of its own; takes {from}'s, which shares its colour texture", model.materials[k].name);
            eprintln!("  {msg}");
            model.warnings.push(msg);
            model.materials[k].alpha = a;
        }
    }
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a tiny binary FBX by hand: one node with an int and a
    /// zlib-compressed double array, and a child.
    #[test]
    fn nodes_and_compressed_arrays_parse() {
        use std::io::Write;
        let mut zdata = Vec::new();
        {
            let mut z = flate2::write::ZlibEncoder::new(&mut zdata, flate2::Compression::default());
            for v in [1.5f64, -2.0, 3.25] {
                z.write_all(&v.to_le_bytes()).unwrap();
            }
        }
        let mut props = Vec::new();
        props.push(b'I');
        props.extend_from_slice(&7i32.to_le_bytes());
        props.push(b'd');
        props.extend_from_slice(&3u32.to_le_bytes());
        props.extend_from_slice(&1u32.to_le_bytes());
        props.extend_from_slice(&(zdata.len() as u32).to_le_bytes());
        props.extend_from_slice(&zdata);
        let child_name = b"Kid";
        let header_len = 27usize;
        let node_start = header_len;
        let child_start = node_start + 13 + 4 + props.len();
        let child_end = child_start + 13 + child_name.len();
        let node_end = child_end + 13;
        let mut f = Vec::new();
        f.extend_from_slice(b"Kaydara FBX Binary  \0\x1a\0");
        f.extend_from_slice(&7400u32.to_le_bytes());
        f.extend_from_slice(&(node_end as u32).to_le_bytes());
        f.extend_from_slice(&2u32.to_le_bytes());
        f.extend_from_slice(&(props.len() as u32).to_le_bytes());
        f.push(4);
        f.extend_from_slice(b"Test");
        f.extend_from_slice(&props);
        f.extend_from_slice(&(child_end as u32).to_le_bytes());
        f.extend_from_slice(&0u32.to_le_bytes());
        f.extend_from_slice(&0u32.to_le_bytes());
        f.push(child_name.len() as u8);
        f.extend_from_slice(child_name);
        f.extend_from_slice(&[0u8; 13]);
        f.extend_from_slice(&[0u8; 13]);
        let nodes = parse(&f).unwrap();
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].name, "Test");
        assert_eq!(nodes[0].props[0].i64(), 7);
        assert_eq!(nodes[0].props[1].f64s(), vec![1.5, -2.0, 3.25]);
        assert_eq!(nodes[0].children.len(), 1);
        assert_eq!(nodes[0].children[0].name, "Kid");
    }

    #[test]
    fn fbx_euler_order_is_x_then_y_then_z() {
        let mut n = Node { name: "Model".into(), ..Default::default() };
        let p = |k: &str, v: [f64; 3]| Node {
            name: "P".into(),
            props: vec![Prop::Str(k.into()), Prop::Str("".into()), Prop::Str("".into()), Prop::Str("A".into()), Prop::F64(v[0]), Prop::F64(v[1]), Prop::F64(v[2])],
            children: vec![],
        };
        n.children.push(Node { name: "Properties70".into(), props: vec![], children: vec![p("Lcl Rotation", [90.0, 90.0, 0.0])] });
        let m = local_matrix(&n);
        // X by 90 takes +Y to +Z; then Y by 90 takes +Z to +X.
        let v = m.transform_vector3(Vec3::Y);
        assert!((v - Vec3::X).length() < 1e-5, "{v}");
    }
}
