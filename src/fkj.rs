//! The `.fkj` scene file: FkJensen's own format, laid out for this renderer.
//!
//! Little-endian. A 16-byte header, then tagged sections. A reader skips any
//! section whose tag it does not know, so new primitive types can be added
//! without breaking old files or old readers. `docs/format.md` describes
//! every byte.
//!
//! ```text
//! header   "FKJ\0"  u32 version (1, or 2 with LODS)  u32 section count  u32 0
//! section  [u8;4] tag  u32 section version  u64 payload bytes  payload  (padded to 8)
//! ```
//!
//! Tags: `MESH` clustered triangles, `INST` copies of a run of clusters at
//! other transforms, `LITE` point lights in lumens, `TEXR` per-triangle
//! texture tiles (`texture.rs`), streamed from the file and never read
//! whole, `LODS` each cluster's level of detail (`lod.rs`). A file with
//! `LODS` is version 2, so that a reader without it refuses the file rather
//! than drawing every level at once. `OBJS` names the objects and groups
//! the editor moves (`Scene::objects`). Reserved for the primitives still
//! to come: `PNTS` (points, particles), `CURV` (curves, fur and hair),
//! `VOLM` (volumes).

// The writer is used by the tests and by the tool that baked
// assets/chess_set.fkj; the viewer itself only reads.
#![allow(dead_code)]

use glam::{Mat4, Vec3, Vec4};

use std::sync::Arc;

use crate::scene::{oct_encode, Cluster, PointLight, Scene};
use crate::texture::{level_offset, Source, TexEntry};

pub const MAGIC: [u8; 4] = *b"FKJ\0";
/// Version written when no cluster has levels of detail; `VERSION_LOD`
/// when one does. Both are read.
pub const VERSION: u32 = 1;
pub const VERSION_LOD: u32 = 2;
pub const TAG_MESH: [u8; 4] = *b"MESH";
pub const TAG_LITE: [u8; 4] = *b"LITE";
pub const TAG_INST: [u8; 4] = *b"INST";
pub const TAG_TEXR: [u8; 4] = *b"TEXR";
pub const TAG_LODS: [u8; 4] = *b"LODS";
pub const TAG_OBJS: [u8; 4] = *b"OBJS";
pub const TAG_SAMP: [u8; 4] = *b"SAMP";
pub const TAG_MATS: [u8; 4] = *b"MATS";
pub const TAG_ALPH: [u8; 4] = *b"ALPH";
pub const TAG_SHDF: [u8; 4] = *b"SHDF";

/// Extra copies of clusters `first_cluster..first_cluster + cluster_count`
/// of the `MESH` section, each at `transform` applied to the stored
/// (world-space) geometry. The stored copy is drawn as well.
#[derive(Clone, Debug, PartialEq)]
pub struct InstanceGroup {
    pub first_cluster: u32,
    pub cluster_count: u32,
    pub transforms: Vec<Mat4>,
}

const CLUSTER_BYTES: usize = 64;

// ---------------------------------------------------------------- encoding

struct W(Vec<u8>);

impl W {
    fn u8(&mut self, v: u8) {
        self.0.push(v);
    }
    fn u16(&mut self, v: u16) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn f32(&mut self, v: f32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn v3(&mut self, v: Vec3) {
        self.f32(v.x);
        self.f32(v.y);
        self.f32(v.z);
    }
    fn pad(&mut self, align: usize) {
        while self.0.len() % align != 0 {
            self.0.push(0);
        }
    }
}

/// Encode a scene. Positions are quantised to 16 bits per axis inside each
/// cluster's box, normals to 16-bit octahedral pairs, colours to 8-bit sRGB.
/// `instances` adds copies of stored clusters; pass `&[]` for none.
pub fn write(scene: &Scene, instances: &[InstanceGroup]) -> Vec<u8> {
    let mut v = Vec::new();
    write_to(&mut v, scene, instances).expect("writing to memory");
    v
}

/// `write`, streamed to `out`: texture tiles go from their source to `out`
/// a cluster at a time, never all in memory.
pub fn write_to(out: &mut impl std::io::Write, scene: &Scene, instances: &[InstanceGroup]) -> std::io::Result<()> {
    let mut head = W(Vec::new());
    head.0.extend_from_slice(&MAGIC);
    let lods = scene.clusters.iter().any(|c| c.has_lod());
    head.u32(if lods { VERSION_LOD } else { VERSION });
    let textured = !scene.textures.is_empty();
    let objs = !scene.objects.is_empty();
    let samp = !scene.samples.is_empty();
    let mats = !scene.materials.is_empty();
    let alph = textured && (0..scene.clusters.len()).any(|c| scene.textures.has_alpha(c));
    let shdf: Vec<(usize, &crate::field::ShadowField)> = scene.objects.iter().enumerate().filter_map(|(i, o)| o.field.and_then(|k| scene.fields.get(k)).map(|f| (i, &**f))).collect();
    head.u32(2 + !instances.is_empty() as u32 + textured as u32 + lods as u32 + objs as u32 + samp as u32 + mats as u32 + alph as u32 + !shdf.is_empty() as u32);
    head.u32(0);
    out.write_all(&head.0)?;

    // MESH
    let mut m = W(Vec::new());
    m.u32(scene.clusters.len() as u32);
    m.u32(scene.vertex_count() as u32);
    m.u32(scene.triangle_count() as u32);
    m.u32(0);
    m.v3(scene.bounds_min);
    m.v3(scene.bounds_max);
    // Geometry is held as the file stores it (`scene.rs`) and written as
    // is, except that of objects moved by their transform: it is written
    // where it now stands, quantised in its new box.
    let world = |c: &Cluster| -> Vec<Vec3> { scene.cluster_pos(c).collect() };
    let boxes: Vec<(Vec3, Vec3)> = scene.clusters.iter().map(|c| if c.in_place() { (c.qlo, c.qstep) } else { crate::scene::quant_box(&world(c)) }).collect();
    for (c, &(lo, step)) in scene.clusters.iter().zip(&boxes) {
        m.u32(c.first_vertex);
        m.u16(c.vertex_count as u16);
        m.u16(c.tri_count as u16);
        m.u32(c.first_tri);
        m.u32(0);
        m.v3(c.centre);
        // Its sphere already holds the quantised positions.
        m.f32(c.radius);
        m.v3(lo);
        m.v3(step);
        let q = |v: f32| (v.clamp(-1.0, 1.0) * 127.0).round() as i8 as u8;
        m.u8(q(c.cone_axis.x));
        m.u8(q(c.cone_axis.y));
        m.u8(q(c.cone_axis.z));
        m.u8(0);
        // The 8-bit axis can be off by about 0.6 degrees: widen the cone.
        m.f32(if c.cone_cos > 0.0 { (c.cone_cos.acos() + 0.012).cos() } else { -1.0 });
    }
    // Normals, turned with their object.
    let mut nrm: Vec<[u16; 2]> = scene.qnrm.clone();
    for c in scene.clusters.iter().filter(|c| c.turned) {
        for v in c.first_vertex as usize..(c.first_vertex + c.vertex_count) as usize {
            nrm[v] = oct_encode(scene.normal(c, v));
        }
    }
    for e in &nrm {
        m.u16(e[0]);
        m.u16(e[1]);
    }
    for c in &scene.qcol {
        m.0.extend_from_slice(c);
        m.u8(255);
    }
    for (c, &(lo, step)) in scene.clusters.iter().zip(&boxes) {
        if c.in_place() {
            for q in &scene.qpos[c.first_vertex as usize..(c.first_vertex + c.vertex_count) as usize] {
                for v in q {
                    m.u16(*v);
                }
            }
        } else {
            for p in world(c) {
                for v in crate::scene::quantise(p, lo, step) {
                    m.u16(v);
                }
            }
        }
    }
    for c in &scene.clusters {
        m.0.extend_from_slice(scene.tris[c.first_tri as usize..(c.first_tri + c.tri_count) as usize].as_flattened());
    }
    section(out, TAG_MESH, 1, &m.0)?;

    // INST: u32 group count, u32 0, then per group u32 first cluster,
    // u32 cluster count, u32 transform count, u32 0, and that many affine
    // transforms as 12 f32 (the three rows of a 3 x 4 matrix).
    if !instances.is_empty() {
        let mut g = W(Vec::new());
        g.u32(instances.len() as u32);
        g.u32(0);
        for ig in instances {
            g.u32(ig.first_cluster);
            g.u32(ig.cluster_count);
            g.u32(ig.transforms.len() as u32);
            g.u32(0);
            for t in &ig.transforms {
                for r in 0..3 {
                    let row = t.row(r);
                    g.f32(row.x);
                    g.f32(row.y);
                    g.f32(row.z);
                    g.f32(row.w);
                }
            }
        }
        section(out, TAG_INST, 1, &g.0)?;
    }

    // LITE, version 2: lumens.
    let mut l = W(Vec::new());
    l.u32(scene.lights.len() as u32);
    l.u32(0);
    for li in &scene.lights {
        l.v3(li.position);
        l.v3(li.color);
        l.f32(li.lumens);
        l.f32(0.0);
    }
    section(out, TAG_LITE, 2, &l.0)?;

    // TEXR: u32 cluster count, u32 0, u64 bytes from the payload's start to
    // the pool; per cluster u8 finest level (0: none), 7 bytes 0, u64
    // offset of its tiles in the pool; then the pool. A cluster's tiles are
    // level 1 to its finest, each level triangle by triangle, sRGB8 RGB.
    if textured {
        let n = scene.clusters.len();
        let table_end = (16 + n * 16) as u64;
        let mut t = W(Vec::new());
        t.u32(n as u32);
        t.u32(0);
        t.u64(table_end);
        // Each cluster's tiles: where they come from and how many bytes.
        let mut parts: Vec<Option<(usize, u64, usize)>> = Vec::with_capacity(n);
        let mut pool_len = 0u64;
        for (ci, c) in scene.clusters.iter().enumerate() {
            let e = scene.textures.entry(ci);
            let part = (e.level > 0 && scene.textures.sources.get(e.source as usize).is_some()).then(|| (e.source as usize, e.offset, level_offset(c.tri_count as usize, e.level + 1) as usize));
            match part {
                Some((_, _, len)) => {
                    t.u8(e.level);
                    for _ in 0..7 {
                        t.u8(0);
                    }
                    t.u64(pool_len);
                    pool_len += len as u64;
                }
                None => {
                    for _ in 0..16 {
                        t.u8(0);
                    }
                }
            }
            parts.push(part);
        }
        let len = t.0.len() as u64 + pool_len;
        out.write_all(&TAG_TEXR)?;
        out.write_all(&1u32.to_le_bytes())?;
        out.write_all(&len.to_le_bytes())?;
        out.write_all(&t.0)?;
        for (src, off, l) in parts.into_iter().flatten() {
            out.write_all(&scene.textures.sources[src].read(off, l)?)?;
        }
        out.write_all(&[0u8; 8][..((8 - len % 8) % 8) as usize])?;
    }

    // ALPH: u32 cluster count, u32 0, u64 bytes from the payload's start to
    // the pool; per cluster u64 offset of its alpha tiles in the pool
    // (u64::MAX: opaque); then the pool. A cluster's alpha tiles are laid
    // out as its TEXR tiles are, one byte a point (opaque from 128).
    if alph {
        let n = scene.clusters.len();
        let table_end = (16 + n * 8) as u64;
        let mut t = W(Vec::new());
        t.u32(n as u32);
        t.u32(0);
        t.u64(table_end);
        let mut parts: Vec<Option<(usize, u64, usize)>> = Vec::with_capacity(n);
        let mut pool_len = 0u64;
        for (ci, c) in scene.clusters.iter().enumerate() {
            let e = scene.textures.entry(ci);
            let part = e.alpha.filter(|_| e.level > 0 && scene.textures.alpha_sources.get(e.source as usize).is_some_and(|s| s.is_some())).map(|a| (e.source as usize, a, (level_offset(c.tri_count as usize, e.level + 1) / 3) as usize));
            match part {
                Some((_, _, len)) => {
                    t.u64(pool_len);
                    pool_len += len as u64;
                }
                None => t.u64(u64::MAX),
            }
            parts.push(part);
        }
        let len = t.0.len() as u64 + pool_len;
        out.write_all(&TAG_ALPH)?;
        out.write_all(&1u32.to_le_bytes())?;
        out.write_all(&len.to_le_bytes())?;
        out.write_all(&t.0)?;
        for (src, off, l) in parts.into_iter().flatten() {
            let s = scene.textures.alpha_sources[src].as_ref().expect("checked above");
            out.write_all(&s.read(off, l)?)?;
        }
        out.write_all(&[0u8; 8][..((8 - len % 8) % 8) as usize])?;
    }

    // LODS: u32 cluster count, u32 0; per cluster, 44 bytes: u8 level, 3
    // bytes 0, f32 error, centre and radius of the sphere it is seen from,
    // then the same three for what replaces it (error infinite: nothing).
    if lods {
        let mut l = W(Vec::new());
        l.u32(scene.clusters.len() as u32);
        l.u32(0);
        for c in &scene.clusters {
            l.u8(c.lod_level);
            l.u8(0);
            l.u8(0);
            l.u8(0);
            l.f32(c.lod_error);
            l.v3(c.lod_centre);
            l.f32(c.lod_radius);
            l.f32(c.parent_error);
            l.v3(c.parent_centre);
            l.f32(c.parent_radius);
        }
        section(out, TAG_LODS, 1, &l.0)?;
    }

    // OBJS, version 2: u32 object count, u32 0; per object u32 first
    // cluster, u32 cluster count (0: a group), u32 parent + 1 (0: none),
    // u32 name bytes, pivot, rotation (3 f32, radians about X, Y, Z, in
    // that order), f32 scale, then the name in UTF-8 padded to 4. Version
    // 1 had one f32, the turn about Y, in place of the rotation.
    if objs {
        let mut o = W(Vec::new());
        o.u32(scene.objects.len() as u32);
        o.u32(0);
        for ob in &scene.objects {
            o.u32(ob.clusters.start as u32);
            o.u32(ob.clusters.len() as u32);
            o.u32(ob.parent.map(|p| p as u32 + 1).unwrap_or(0));
            o.u32(ob.name.len() as u32);
            o.v3(ob.pivot);
            o.v3(ob.rotation);
            o.v3(ob.scale);
            o.0.extend_from_slice(ob.name.as_bytes());
            o.pad(4);
        }
        section(out, TAG_OBJS, 3, &o.0)?;
    }

    // SAMP: u32 sample count, u32 0; per sample, 60 bytes: position (3
    // f32), normal (2 u16, octahedral), f32 area, u32 cluster, then its sky
    // transfer, 9 f32 (NaN when not computed). In order of cluster.
    if samp {
        let mut sm = W(Vec::with_capacity(8 + scene.samples.len() * 60));
        sm.u32(scene.samples.len() as u32);
        sm.u32(0);
        for x in &scene.samples {
            sm.v3(x.position);
            let e = oct_encode(x.normal);
            sm.u16(e[0]);
            sm.u16(e[1]);
            sm.f32(x.area);
            sm.u32(x.cluster);
            for v in x.sky {
                sm.f32(v);
            }
        }
        section(out, TAG_SAMP, 1, &sm.0)?;
    }

    // MATS: u32 material count M, u32 cluster count C, u32 triangle count
    // T, u32 0; per material: tint (3 f32, linear RGB), f32 roughness, f32
    // metallic, f32 specular, u32 name bytes, the name in UTF-8 padded to
    // 4; per cluster u32 its first material; per triangle u8 its material
    // from its cluster's first; padded to 4.
    // SHDF: u32 field count, u32 0; per field u32 the object it belongs
    // to (an index into OBJS), then the field (`field.rs`): f32 x 3 its
    // first cell, f32 cell size, u32 x 3 cells along x, y, z, u32 distance
    // bytes and u32 density bytes (each 0 or the cell count), the bytes,
    // padded to 4.
    if !shdf.is_empty() {
        let mut sh = W(Vec::new());
        sh.u32(shdf.len() as u32);
        sh.u32(0);
        for (i, f) in &shdf {
            sh.u32(*i as u32);
            f.write(&mut sh.0);
        }
        section(out, TAG_SHDF, 1, &sh.0)?;
    }

    if mats {
        let t = scene.triangle_count();
        let mut ms = W(Vec::with_capacity(16 + scene.materials.len() * 40 + scene.clusters.len() * 4 + t + 4));
        ms.u32(scene.materials.len() as u32);
        ms.u32(scene.clusters.len() as u32);
        ms.u32(t as u32);
        ms.u32(0);
        for m in &scene.materials {
            ms.v3(m.tint);
            ms.f32(m.roughness);
            ms.f32(m.metallic);
            ms.f32(m.specular);
            ms.u32(m.name.len() as u32);
            ms.0.extend_from_slice(m.name.as_bytes());
            ms.pad(4);
        }
        for c in &scene.clusters {
            ms.u32(c.material_base);
        }
        ms.0.extend((0..t).map(|k| scene.tri_mat.get(k).copied().unwrap_or(0)));
        ms.pad(4);
        section(out, TAG_MATS, 1, &ms.0)?;
    }
    Ok(())
}

fn section(out: &mut impl std::io::Write, tag: [u8; 4], version: u32, payload: &[u8]) -> std::io::Result<()> {
    out.write_all(&tag)?;
    out.write_all(&version.to_le_bytes())?;
    out.write_all(&(payload.len() as u64).to_le_bytes())?;
    out.write_all(payload)?;
    out.write_all(&[0u8; 8][..(8 - payload.len() % 8) % 8])
}

// ---------------------------------------------------------------- decoding

struct R<'a> {
    b: &'a [u8],
    at: usize,
}

impl<'a> R<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.at.checked_add(n).filter(|e| *e <= self.b.len()).ok_or("file is truncated")?;
        let s = &self.b[self.at..end];
        self.at = end;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16, String> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> Result<f32, String> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn v3(&mut self) -> Result<Vec3, String> {
        Ok(Vec3::new(self.f32()?, self.f32()?, self.f32()?))
    }
}

/// What reading a file found besides the scene.
#[derive(Default, Debug)]
pub struct ReadReport {
    /// Tags of sections this reader does not handle yet, skipped.
    pub skipped: Vec<String>,
}

/// Read a whole `.fkj` held in memory. Texture tiles, if any, are kept in
/// memory too.
pub fn read(bytes: &[u8]) -> Result<(Scene, ReadReport), String> {
    let mut r = R { b: bytes, at: 0 };
    let sections = header(r.take(16)?)?;
    let mut list: Vec<([u8; 4], u32, Vec<u8>)> = Vec::new();
    let mut tex: Option<(Vec<u8>, Source)> = None;
    let mut alpha: Option<(Vec<u8>, Source)> = None;
    for _ in 0..sections {
        let tag: [u8; 4] = r.take(4)?.try_into().unwrap();
        let section_version = r.u32()?;
        let len = r.u64()? as usize;
        let payload = r.take(len)?;
        r.at = r.at.div_ceil(8) * 8;
        if tag == TAG_TEXR {
            let table_end = texr_table_end(payload)?;
            tex = Some((payload[..table_end].to_vec(), Source::Memory(Arc::new(payload[table_end..].to_vec()))));
        } else if tag == TAG_ALPH && section_version == 1 {
            let table_end = alph_table_end(payload)?;
            alpha = Some((payload[..table_end].to_vec(), Source::Memory(Arc::new(payload[table_end..].to_vec()))));
        } else {
            list.push((tag, section_version, payload.to_vec()));
        }
    }
    assemble(list, tex, alpha)
}

/// Read a `.fkj` file. Everything but the texture tiles is read; the tiles
/// stay in the file and are read when a cluster needs them.
pub fn read_file(path: &std::path::Path) -> Result<(Scene, ReadReport), String> {
    use std::io::{Read, Seek, SeekFrom};
    let name = path.display();
    let mut f = std::fs::File::open(path).map_err(|e| format!("cannot read {name}: {e}"))?;
    let mut buf = [0u8; 16];
    f.read_exact(&mut buf).map_err(|_| "file is truncated".to_string())?;
    let sections = header(&buf)?;
    let mut list = Vec::new();
    let mut tex_table: Option<(Vec<u8>, u64)> = None;
    let mut alpha_table: Option<(Vec<u8>, u64)> = None;
    let mut at = 16u64;
    for _ in 0..sections {
        f.seek(SeekFrom::Start(at)).map_err(|e| e.to_string())?;
        f.read_exact(&mut buf).map_err(|_| "file is truncated".to_string())?;
        let tag: [u8; 4] = buf[0..4].try_into().unwrap();
        let version = u32::from_le_bytes(buf[4..8].try_into().unwrap());
        let len = u64::from_le_bytes(buf[8..16].try_into().unwrap());
        let start = at + 16;
        if tag == TAG_TEXR {
            let mut head = [0u8; 16];
            f.read_exact(&mut head).map_err(|_| "file is truncated".to_string())?;
            let table_end = texr_table_end(&head)?;
            let mut table = vec![0u8; table_end];
            table[..16].copy_from_slice(&head);
            f.read_exact(&mut table[16..]).map_err(|_| "file is truncated".to_string())?;
            tex_table = Some((table, start + table_end as u64));
        } else if tag == TAG_ALPH && version == 1 {
            let mut head = [0u8; 16];
            f.read_exact(&mut head).map_err(|_| "file is truncated".to_string())?;
            let table_end = alph_table_end(&head)?;
            let mut table = vec![0u8; table_end];
            table[..16].copy_from_slice(&head);
            f.read_exact(&mut table[16..]).map_err(|_| "file is truncated".to_string())?;
            alpha_table = Some((table, start + table_end as u64));
        } else {
            let mut payload = vec![0u8; len as usize];
            f.read_exact(&mut payload).map_err(|_| "file is truncated".to_string())?;
            list.push((tag, version, payload));
        }
        at = (start + len).div_ceil(8) * 8;
    }
    let tex = match tex_table {
        Some((table, pool)) => {
            let file = std::fs::File::open(path).map_err(|e| format!("cannot read {name}: {e}"))?;
            Some((table, Source::File { file, pool }))
        }
        None => None,
    };
    let alpha = match alpha_table {
        Some((table, pool)) => {
            let file = std::fs::File::open(path).map_err(|e| format!("cannot read {name}: {e}"))?;
            Some((table, Source::File { file, pool }))
        }
        None => None,
    };
    assemble(list, tex, alpha)
}

/// Whether the file at `path` has a section `tag` (headers only read).
pub fn has_section(path: &std::path::Path, tag: [u8; 4]) -> Result<bool, String> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut buf = [0u8; 16];
    f.read_exact(&mut buf).map_err(|_| "file is truncated".to_string())?;
    let n = header(&buf)?;
    let mut at = 16u64;
    for _ in 0..n {
        f.seek(SeekFrom::Start(at)).map_err(|e| e.to_string())?;
        f.read_exact(&mut buf).map_err(|_| "file is truncated".to_string())?;
        if buf[0..4] == tag {
            return Ok(true);
        }
        let len = u64::from_le_bytes(buf[8..16].try_into().unwrap());
        at = (at + 16 + len).div_ceil(8) * 8;
    }
    Ok(false)
}

fn alph_table_end(payload: &[u8]) -> Result<usize, String> {
    if payload.len() < 16 {
        return Err("ALPH section is truncated".into());
    }
    let n = u32::from_le_bytes(payload[0..4].try_into().unwrap()) as usize;
    let end = u64::from_le_bytes(payload[8..16].try_into().unwrap()) as usize;
    if end != 16 + n * 8 {
        return Err("ALPH table size does not match its cluster count".into());
    }
    Ok(end)
}

fn header(b: &[u8]) -> Result<u32, String> {
    if b.len() < 16 || b[0..4] != MAGIC {
        return Err("not an .fkj file".into());
    }
    let version = u32::from_le_bytes(b[4..8].try_into().unwrap());
    if version != VERSION && version != VERSION_LOD {
        return Err(format!(".fkj version {version}, this build reads versions {VERSION} and {VERSION_LOD}"));
    }
    Ok(u32::from_le_bytes(b[8..12].try_into().unwrap()))
}

fn texr_table_end(payload: &[u8]) -> Result<usize, String> {
    if payload.len() < 16 {
        return Err("TEXR section is truncated".into());
    }
    let n = u32::from_le_bytes(payload[0..4].try_into().unwrap()) as usize;
    let end = u64::from_le_bytes(payload[8..16].try_into().unwrap()) as usize;
    if end != 16 + n * 16 {
        return Err("TEXR table size does not match its cluster count".into());
    }
    Ok(end)
}

fn assemble(list: Vec<([u8; 4], u32, Vec<u8>)>, tex: Option<(Vec<u8>, Source)>, alpha: Option<(Vec<u8>, Source)>) -> Result<(Scene, ReadReport), String> {
    let mut scene = Scene::new();
    let mut report = ReadReport::default();
    let mut inst_payloads: Vec<&[u8]> = Vec::new();
    let mut lods: Option<&[u8]> = None;
    let mut objs: Option<(&[u8], u32)> = None;
    let mut samp: Option<&[u8]> = None;
    let mut mats: Option<&[u8]> = None;
    let mut shdf: Option<&[u8]> = None;
    for (tag, section_version, payload) in &list {
        match *tag {
            TAG_OBJS => objs = Some((payload, *section_version)),
            TAG_MESH => read_mesh(payload, &mut scene)?,
            TAG_LODS => lods = Some(payload),
            TAG_LITE if *section_version == 2 => read_lights(payload, &mut scene)?,
            TAG_LITE => report.skipped.push(format!("LITE version {section_version} (relative units, not lumens)")),
            TAG_INST => inst_payloads.push(payload),
            TAG_SAMP if *section_version == 1 => samp = Some(payload),
            TAG_MATS if *section_version == 1 => mats = Some(payload),
            TAG_SHDF if *section_version == 1 => shdf = Some(payload),
            other => report.skipped.push(String::from_utf8_lossy(&other).into_owned()),
        }
    }
    let stored_clusters = scene.clusters.len();
    if let Some(b) = lods {
        read_lods(b, &mut scene)?;
    }
    if let Some((table, source)) = tex {
        let mut r = R { b: &table, at: 16 };
        let n = u32::from_le_bytes(table[0..4].try_into().unwrap()) as usize;
        if n != stored_clusters {
            return Err(format!("TEXR has {n} clusters, MESH has {stored_clusters}"));
        }
        let mut entries = Vec::with_capacity(n);
        for _ in 0..n {
            let level = r.u8()?;
            r.take(7)?;
            let offset = r.u64()?;
            if level > crate::texture::MAX_LEVEL {
                return Err("TEXR level beyond the finest dicing level".into());
            }
            entries.push(TexEntry { level, source: 0, offset, alpha: None });
        }
        scene.textures.entries = entries;
        scene.textures.sources = vec![Arc::new(source)];
        if let Some((table, source)) = alpha {
            let n = u32::from_le_bytes(table[0..4].try_into().unwrap()) as usize;
            if n != stored_clusters {
                return Err(format!("ALPH has {n} clusters, MESH has {stored_clusters}"));
            }
            let mut r = R { b: &table, at: 16 };
            for e in scene.textures.entries.iter_mut() {
                let a = r.u64()?;
                e.alpha = (a != u64::MAX).then_some(a);
            }
            scene.textures.alpha_sources = vec![Some(Arc::new(source))];
        }
    }
    if let Some(b) = samp {
        read_samples(b, &mut scene, stored_clusters)?;
    }
    if let Some(b) = mats {
        read_materials(b, &mut scene, stored_clusters)?;
    }
    // Instances refer to the clusters as stored, so they are expanded after
    // every MESH section has been read.
    for p in inst_payloads {
        expand_instances(p, &mut scene, stored_clusters)?;
    }
    if let Some((b, v)) = objs {
        read_objects(b, v, &mut scene, stored_clusters)?;
    }
    if let Some(b) = shdf {
        let mut at = 8usize;
        let n = b.get(0..4).map(|s| u32::from_le_bytes(s.try_into().unwrap()) as usize).ok_or("SHDF is truncated")?;
        for _ in 0..n {
            let i = b.get(at..at + 4).map(|s| u32::from_le_bytes(s.try_into().unwrap()) as usize).ok_or("SHDF is truncated")?;
            at += 4;
            let f = crate::field::ShadowField::read(b, &mut at)?;
            if let Some(o) = scene.objects.get_mut(i) {
                o.field = Some(scene.fields.len());
                scene.fields.push(std::sync::Arc::new(f));
            }
        }
    }
    if scene.clusters.is_empty() {
        scene.bounds_min = Vec3::splat(-1.0);
        scene.bounds_max = Vec3::splat(1.0);
    }
    Ok((scene, report))
}

/// Write `materials` into the file at `path` in place of the material
/// table it has: every other section is copied as it is (texture tiles
/// included), and which material each triangle has stays. Through a file
/// beside it, which then replaces it.
pub fn rewrite_materials(path: &std::path::Path, materials: &[crate::scene::Material]) -> Result<(), String> {
    use std::io::{Read, Seek, SeekFrom, Write};
    let name = path.display();
    let err = |e: std::io::Error| format!("{name}: {e}");
    let mut f = std::fs::File::open(path).map_err(err)?;
    let mut head = [0u8; 16];
    f.read_exact(&mut head).map_err(|_| format!("{name}: file is truncated"))?;
    let count = header(&head)?;
    // Each section: tag, where it starts (its own header) and its length
    // with padding.
    let mut list: Vec<([u8; 4], u64, u64)> = Vec::new();
    let mut at = 16u64;
    let mut buf = [0u8; 16];
    let (mut clusters, mut tris) = (0usize, 0usize);
    let mut old: Option<Vec<u8>> = None;
    for _ in 0..count {
        f.seek(SeekFrom::Start(at)).map_err(err)?;
        f.read_exact(&mut buf).map_err(|_| format!("{name}: file is truncated"))?;
        let tag: [u8; 4] = buf[0..4].try_into().unwrap();
        let version = u32::from_le_bytes(buf[4..8].try_into().unwrap());
        let len = u64::from_le_bytes(buf[8..16].try_into().unwrap());
        let next = (at + 16 + len).div_ceil(8) * 8;
        if tag == TAG_MESH {
            let mut m = [0u8; 12];
            f.read_exact(&mut m).map_err(err)?;
            clusters = u32::from_le_bytes(m[0..4].try_into().unwrap()) as usize;
            tris = u32::from_le_bytes(m[8..12].try_into().unwrap()) as usize;
        }
        if tag == TAG_MATS && version == 1 {
            let mut p = vec![0u8; len as usize];
            f.read_exact(&mut p).map_err(err)?;
            old = Some(p);
        } else {
            list.push((tag, at, next - at));
        }
        at = next;
    }
    // Which material each cluster starts at and each triangle has: as
    // they were, else the first.
    let (bases, ids) = match old {
        Some(p) => {
            let mut r = R { b: &p, at: 0 };
            let m = r.u32()? as usize;
            r.at = 16;
            for _ in 0..m {
                r.take(24)?;
                let len = r.u32()? as usize;
                r.take(len)?;
                r.at = r.at.div_ceil(4) * 4;
            }
            let bases: Vec<u32> = (0..clusters).map(|_| r.u32()).collect::<Result<_, _>>()?;
            (bases, r.take(tris)?.to_vec())
        }
        None => (vec![0; clusters], vec![0; tris]),
    };
    let top = materials.len().saturating_sub(1) as u32;
    let mut ms = W(Vec::new());
    ms.u32(materials.len() as u32);
    ms.u32(clusters as u32);
    ms.u32(tris as u32);
    ms.u32(0);
    for m in materials {
        ms.v3(m.tint);
        ms.f32(m.roughness);
        ms.f32(m.metallic);
        ms.f32(m.specular);
        ms.u32(m.name.len() as u32);
        ms.0.extend_from_slice(m.name.as_bytes());
        ms.pad(4);
    }
    for b in bases {
        ms.u32(b.min(top));
    }
    ms.0.extend_from_slice(&ids);
    ms.pad(4);
    let tmp = path.with_extension("fja.part");
    let written = (|| -> std::io::Result<()> {
        let mut out = std::io::BufWriter::with_capacity(1 << 20, std::fs::File::create(&tmp)?);
        out.write_all(&head[0..8])?;
        out.write_all(&(list.len() as u32 + 1).to_le_bytes())?;
        out.write_all(&head[12..16])?;
        for &(_, start, len) in &list {
            f.seek(SeekFrom::Start(start))?;
            std::io::copy(&mut (&mut f).take(len), &mut out)?;
        }
        section(&mut out, TAG_MATS, 1, &ms.0)?;
        out.flush()
    })();
    match written.and_then(|_| std::fs::rename(&tmp, path)) {
        Ok(()) => Ok(()),
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            Err(err(e))
        }
    }
}

fn read_materials(b: &[u8], scene: &mut Scene, stored_clusters: usize) -> Result<(), String> {
    let mut r = R { b, at: 0 };
    let m = r.u32()? as usize;
    let c = r.u32()? as usize;
    let t = r.u32()? as usize;
    r.u32()?;
    if c != stored_clusters || t != scene.triangle_count() || m > 1 << 20 {
        return Err(format!("MATS has {c} clusters and {t} triangles, MESH {stored_clusters} and {}", scene.triangle_count()));
    }
    let mut materials = Vec::with_capacity(m);
    for _ in 0..m {
        let tint = r.v3()?;
        let roughness = r.f32()?;
        let metallic = r.f32()?;
        let specular = r.f32()?;
        let len = r.u32()? as usize;
        let name = String::from_utf8_lossy(r.take(len)?).into_owned();
        r.at = r.at.div_ceil(4) * 4;
        materials.push(crate::scene::Material { name, tint, roughness, metallic, specular }.clamped());
    }
    for k in 0..c {
        let base = r.u32()?;
        scene.clusters[k].material_base = base.min(m.saturating_sub(1) as u32);
    }
    scene.tri_mat = r.take(t)?.to_vec();
    scene.materials = materials;
    Ok(())
}

fn read_mesh(b: &[u8], scene: &mut Scene) -> Result<(), String> {
    let mut r = R { b, at: 0 };
    let n_clusters = r.u32()? as usize;
    let n_verts = r.u32()? as usize;
    let n_tris = r.u32()? as usize;
    r.u32()?;
    let bmin = r.v3()?;
    let bmax = r.v3()?;
    // Check sizes before allocating anything.
    let need = n_clusters * CLUSTER_BYTES + n_verts * (4 + 4 + 6) + n_tris * 3;
    if b.len() - r.at < need {
        return Err("MESH section is shorter than its counts say".into());
    }
    let vbase = scene.vertex_count() as u32;
    let tbase = scene.triangle_count() as u32;

    scene.clusters.reserve(n_clusters);
    for _ in 0..n_clusters {
        let first_vertex = r.u32()?;
        let vertex_count = r.u16()? as u32;
        let tri_count = r.u16()? as u32;
        let first_tri = r.u32()?;
        r.u32()?;
        let centre = r.v3()?;
        let radius = r.f32()?;
        let lo = r.v3()?;
        let step = r.v3()?;
        let ax = Vec3::new(r.u8()? as i8 as f32, r.u8()? as i8 as f32, r.u8()? as i8 as f32) / 127.0;
        r.u8()?;
        let cone_cos = r.f32()?;
        if first_vertex as usize + vertex_count as usize > n_verts || first_tri as usize + tri_count as usize > n_tris {
            return Err("cluster points outside the mesh".into());
        }
        scene.clusters.push(Cluster {
            first_vertex: vbase + first_vertex,
            vertex_count,
            first_tri: tbase + first_tri,
            tri_count,
            centre,
            radius,
            cone_axis: ax.normalize_or(Vec3::Y),
            cone_cos,
            ..Default::default()
        });
        scene.clusters.last_mut().unwrap().set_quant(lo, step);
    }
    // Held as stored: copied, not decoded.
    scene.qnrm.reserve(n_verts);
    for _ in 0..n_verts {
        scene.qnrm.push([r.u16()?, r.u16()?]);
    }
    scene.qcol.reserve(n_verts);
    for _ in 0..n_verts {
        let c = r.take(4)?;
        scene.qcol.push([c[0], c[1], c[2]]);
    }
    // Positions are stored cluster by cluster; clusters cover the vertex
    // range in order.
    scene.qpos.resize(vbase as usize + n_verts, [0; 3]);
    let clusters = &scene.clusters[scene.clusters.len() - n_clusters..];
    for c in clusters {
        for v in c.first_vertex..c.first_vertex + c.vertex_count {
            scene.qpos[v as usize] = [r.u16()?, r.u16()?, r.u16()?];
        }
    }
    scene.tris.resize(tbase as usize + n_tris, [0; 3]);
    for c in clusters {
        for t in c.first_tri..c.first_tri + c.tri_count {
            let l = [r.u8()?, r.u8()?, r.u8()?];
            if l.iter().any(|&l| l as u32 >= c.vertex_count) {
                return Err("triangle index outside its cluster".into());
            }
            scene.tris[t as usize] = l;
        }
    }
    scene.bounds_min = scene.bounds_min.min(bmin);
    scene.bounds_max = scene.bounds_max.max(bmax);
    Ok(())
}

fn read_samples(b: &[u8], scene: &mut Scene, stored_clusters: usize) -> Result<(), String> {
    let mut r = R { b, at: 0 };
    let n = r.u32()? as usize;
    r.u32()?;
    if b.len() - r.at < n * 60 {
        return Err("SAMP section is shorter than its count says".into());
    }
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let position = r.v3()?;
        let normal = crate::scene::oct_decode([r.u16()?, r.u16()?]);
        let area = r.f32()?;
        let cluster = r.u32()?;
        let mut sky = [0f32; 9];
        for v in &mut sky {
            *v = r.f32()?;
        }
        if cluster as usize >= stored_clusters {
            return Err("a sample lies on a cluster past the mesh".into());
        }
        out.push(crate::samples::Sample { position, normal, area, cluster, sky });
    }
    if out.windows(2).any(|w| w[0].cluster > w[1].cluster) {
        return Err("SAMP is not in order of cluster".into());
    }
    scene.samples.extend(out);
    Ok(())
}

fn expand_instances(b: &[u8], scene: &mut Scene, stored_clusters: usize) -> Result<(), String> {
    let mut r = R { b, at: 0 };
    let groups = r.u32()?;
    r.u32()?;
    for _ in 0..groups {
        let first = r.u32()? as usize;
        let count = r.u32()? as usize;
        let n = r.u32()? as usize;
        r.u32()?;
        if first + count > stored_clusters {
            return Err("INST group points outside the mesh".into());
        }
        if b.len() - r.at < n * 48 {
            return Err("INST section is shorter than its counts say".into());
        }
        for _ in 0..n {
            let mut rows = [Vec4::ZERO; 3];
            for row in &mut rows {
                *row = Vec4::new(r.f32()?, r.f32()?, r.f32()?, r.f32()?);
            }
            let m = Mat4::from_cols(rows[0], rows[1], rows[2], Vec4::W).transpose();
            let nm = m.inverse().transpose();
            let scale = m.x_axis.truncate().length().max(m.y_axis.truncate().length()).max(m.z_axis.truncate().length());
            let base = scene.clusters.len();
            // The copy's samples: turned with it when it is a rotation and
            // a uniform scale, else without sky transfer.
            let rigid = m.determinant() > 0.0
                && [m.x_axis, m.y_axis, m.z_axis].iter().all(|a| (a.truncate().length() - scale).abs() < 1e-4 * scale);
            let turn = glam::Quat::from_mat3(&glam::Mat3::from_mat4(m).mul_scalar(1.0 / scale));
            let copies: Vec<crate::samples::Sample> = scene.samples[scene.samples_in(first..first + count)]
                .iter()
                .map(|x| crate::samples::Sample {
                    position: m.transform_point3(x.position),
                    normal: nm.transform_vector3(x.normal).normalize_or(Vec3::Y),
                    area: x.area * scale * scale,
                    cluster: (base + x.cluster as usize - first) as u32,
                    sky: if rigid && !x.sky[0].is_nan() { crate::transfer::rotate(&x.sky, turn) } else { [f32::NAN; 9] },
                })
                .collect();
            for ci in first..first + count {
                let c = scene.clusters[ci];
                let vbase = scene.vertex_count() as u32;
                let tbase = scene.triangle_count() as u32;
                let vr = c.first_vertex as usize..(c.first_vertex + c.vertex_count) as usize;
                let ps: Vec<Vec3> = scene.cluster_pos(&c).map(|p| m.transform_point3(p)).collect();
                let (qlo, qstep) = crate::scene::quant_box(&ps);
                for (v, p) in vr.clone().zip(&ps) {
                    let nn = nm.transform_vector3(scene.normal(&c, v)).normalize_or(Vec3::Y);
                    let col = scene.qcol[v];
                    scene.qpos.push(crate::scene::quantise(*p, qlo, qstep));
                    scene.qnrm.push(oct_encode(nn));
                    scene.qcol.push(col);
                    scene.bounds_min = scene.bounds_min.min(*p);
                    scene.bounds_max = scene.bounds_max.max(*p);
                }
                // A mirroring transform flips the winding; keep front faces front.
                let flip = m.determinant() < 0.0;
                for t in c.first_tri..c.first_tri + c.tri_count {
                    let l = scene.tris[t as usize];
                    scene.tris.push(if flip { [l[0], l[2], l[1]] } else { l });
                }
                if !scene.textures.entries.is_empty() {
                    let e = scene.textures.entry(ci);
                    let at = scene.clusters.len();
                    scene.textures.entries.resize(at, TexEntry::default());
                    // A mirrored copy swaps two corners, which would turn its
                    // tiles over: it keeps vertex colours only.
                    scene.textures.entries.push(if m.determinant() < 0.0 { TexEntry::default() } else { e });
                }
                let axis = nm.transform_vector3(c.cone_axis).normalize_or(Vec3::Y);
                scene.clusters.push(Cluster {
                    first_vertex: vbase,
                    first_tri: tbase,
                    centre: m.transform_point3(c.centre),
                    radius: c.radius * scale,
                    cone_axis: axis,
                    // A non-uniform scale bends the cone; drop it rather than cull wrongly.
                    cone_cos: if (m.x_axis.truncate().length() - scale).abs() < 1e-4 * scale
                        && (m.y_axis.truncate().length() - scale).abs() < 1e-4 * scale
                        && (m.z_axis.truncate().length() - scale).abs() < 1e-4 * scale
                    {
                        c.cone_cos
                    } else {
                        -1.0
                    },
                    ..c
                });
                let last = scene.clusters.last_mut().unwrap();
                last.set_quant(qlo, qstep);
                last.move_lod(|p| m.transform_point3(p), scale);
                // Hold the copy's quantised positions.
                let r = ps.iter().map(|p| crate::scene::dequantise(crate::scene::quantise(*p, qlo, qstep), qlo, qstep).distance(last.centre)).fold(0.0, f32::max);
                last.radius = last.radius.max(r);
            }
            scene.samples.extend(copies);
        }
    }
    Ok(())
}

fn read_objects(b: &[u8], version: u32, scene: &mut Scene, stored_clusters: usize) -> Result<(), String> {
    let mut r = R { b, at: 0 };
    let n = r.u32()? as usize;
    r.u32()?;
    let mut objects = Vec::with_capacity(n.min(1 << 16));
    for _ in 0..n {
        let first = r.u32()? as usize;
        let count = r.u32()? as usize;
        let parent = r.u32()? as usize;
        let len = r.u32()? as usize;
        let pivot = r.v3()?;
        let rotation = if version >= 2 { r.v3()? } else { Vec3::new(0.0, r.f32()?, 0.0) };
        let scale = if version >= 3 { r.v3()? } else { Vec3::splat(r.f32()?) };
        let scale = Vec3::select(scale.cmpgt(Vec3::ZERO), scale, Vec3::ONE);
        let name = String::from_utf8_lossy(r.take(len)?).into_owned();
        r.at = r.at.div_ceil(4) * 4;
        if first + count > stored_clusters || parent > n {
            return Err("OBJS refers past the stored clusters or objects".into());
        }
        let clusters = if count == 0 { scene.clusters.len()..scene.clusters.len() } else { first..first + count };
        objects.push(crate::scene::Object::new(name, clusters, pivot, rotation, scale, parent.checked_sub(1)));
    }
    scene.objects = objects;
    // A parent chain that loops (a damaged file) is cut at the top.
    for i in 0..n {
        let mut seen = 0;
        let mut p = scene.objects[i].parent;
        while let Some(q) = p {
            seen += 1;
            if seen > n {
                scene.objects[i].parent = None;
                break;
            }
            p = scene.objects[q].parent;
        }
    }
    for i in 0..n {
        scene.own_materials(i);
        scene.refresh_bounds(i);
    }
    Ok(())
}

fn read_lods(b: &[u8], scene: &mut Scene) -> Result<(), String> {
    let mut r = R { b, at: 0 };
    let n = r.u32()? as usize;
    r.u32()?;
    if n != scene.clusters.len() {
        return Err(format!("LODS has {n} clusters, MESH has {}", scene.clusters.len()));
    }
    if b.len() - r.at < n * 44 {
        return Err("LODS section is shorter than its count says".into());
    }
    for c in &mut scene.clusters {
        c.lod_level = r.u8()?;
        r.take(3)?;
        c.lod_error = r.f32()?;
        c.lod_centre = r.v3()?;
        c.lod_radius = r.f32()?;
        c.parent_error = r.f32()?;
        c.parent_centre = r.v3()?;
        c.parent_radius = r.f32()?;
        if !(c.lod_error >= 0.0) || !(c.parent_error >= c.lod_error) {
            return Err("LODS: a cluster's error is negative or above what replaces it".into());
        }
    }
    Ok(())
}

fn read_lights(b: &[u8], scene: &mut Scene) -> Result<(), String> {
    let mut r = R { b, at: 0 };
    let n = r.u32()? as usize;
    r.u32()?;
    for _ in 0..n {
        let position = r.v3()?;
        let color = r.v3()?;
        let lumens = r.f32()?;
        r.f32()?;
        scene.lights.push(PointLight { position, color, lumens, ..PointLight::DEFAULT });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::oct_decode;
    use crate::scene::tests::uv_sphere;

    fn sample() -> Scene {
        let (p, idx) = uv_sphere(48);
        let colors: Vec<Vec3> = p.iter().map(|v| (*v * 0.5 + Vec3::splat(0.5)).powf(2.2)).collect();
        let mut s = Scene::new();
        s.push_mesh(&p, &[], &colors, &idx);
        s.scatter_over(0..s.clusters.len());
        crate::transfer::compute(&mut s.samples);
        s.lights.push(PointLight { position: Vec3::new(1.0, 2.0, 3.0), color: Vec3::new(1.0, 0.5, 0.25), lumens: 800.0, ..PointLight::DEFAULT });
        s.materials = vec![crate::scene::Material { name: "skin".into(), roughness: 0.3, ..Default::default() }, crate::scene::Material { name: "chrome".into(), metallic: 1.0, roughness: 0.1, tint: Vec3::new(0.9, 0.8, 0.7), specular: 0.5 }];
        s.tri_mat = (0..s.triangle_count()).map(|t| (t % 3 == 0) as u8).collect();
        s
    }

    /// Geometry is held as the file stores it, so a round trip is exact.
    #[test]
    fn round_trip_keeps_geometry_exactly() {
        let a = sample();
        let bytes = write(&a, &[]);
        let (b, rep) = read(&bytes).unwrap();
        assert!(rep.skipped.is_empty());
        assert_eq!(a.indices(), b.indices());
        assert_eq!(a.clusters.len(), b.clusters.len());
        assert_eq!(a.lights, b.lights);
        assert_eq!(a.positions(), b.positions());
        assert_eq!(a.normals(), b.normals());
        assert_eq!(a.colors(), b.colors());
        assert_eq!(a.samples.len(), b.samples.len());
        assert_eq!(a.materials, b.materials);
        assert_eq!(a.tri_mat, b.tri_mat);
        assert_eq!(a.clusters.iter().map(|c| c.material_base).collect::<Vec<_>>(), b.clusters.iter().map(|c| c.material_base).collect::<Vec<_>>());
        for c in &b.clusters {
            for v in c.first_vertex..c.first_vertex + c.vertex_count {
                assert!(b.positions()[v as usize].distance(c.centre) <= c.radius);
            }
        }
    }

    #[test]
    fn unknown_sections_are_skipped() {
        let a = sample();
        let mut bytes = write(&a, &[]);
        // Append a CURV section and bump the section count.
        let n = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        bytes[8..12].copy_from_slice(&(n + 1).to_le_bytes());
        bytes.extend_from_slice(b"CURV");
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&5u64.to_le_bytes());
        bytes.extend_from_slice(&[1, 2, 3, 4, 5, 0, 0, 0]);
        let (b, rep) = read(&bytes).unwrap();
        assert_eq!(rep.skipped, vec!["CURV".to_string()]);
        assert_eq!(b.triangle_count(), a.triangle_count());
    }

    #[test]
    fn truncated_and_foreign_files_are_errors() {
        let bytes = write(&sample(), &[]);
        assert!(read(&bytes[..bytes.len() / 2]).is_err());
        assert!(read(b"#usda 1.0\n").is_err());
    }

    #[test]
    fn octahedral_normals_cover_the_sphere() {
        for i in 0..200 {
            let t = i as f32 * 0.37;
            let n = Vec3::new(t.sin() * (t * 1.7).cos(), (t * 0.9).cos(), t.cos() * (t * 2.3).sin()).normalize();
            assert!(oct_decode(oct_encode(n)).dot(n) > 0.99999, "{n}");
        }
        for n in [Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y, Vec3::Z, -Vec3::Z] {
            assert!(oct_decode(oct_encode(n)).dot(n) > 0.99999, "{n}");
        }
    }

    #[test]
    fn instances_expand_into_moved_copies() {
        let a = sample();
        let n = a.clusters.len() as u32;
        let moves = vec![Mat4::from_translation(Vec3::new(5.0, 0.0, 0.0)), Mat4::from_translation(Vec3::new(0.0, 0.0, -3.0))];
        let bytes = write(&a, &[InstanceGroup { first_cluster: 0, cluster_count: n, transforms: moves.clone() }]);
        let (b, _) = read(&bytes).unwrap();
        assert_eq!(b.triangle_count(), a.triangle_count() * 3);
        assert_eq!(b.clusters.len(), a.clusters.len() * 3);
        let k = a.positions().len();
        for (i, m) in moves.iter().enumerate() {
            for v in 0..k {
                let want = m.transform_point3(b.positions()[v]);
                assert!(b.positions()[k * (i + 1) + v].distance(want) < 1e-5);
            }
        }
        assert!(b.bounds_max.x > 5.5 && b.bounds_min.z < -3.5);
        for c in &b.clusters {
            for v in c.first_vertex..c.first_vertex + c.vertex_count {
                assert!(b.positions()[v as usize].distance(c.centre) <= c.radius + 1e-5);
            }
            for t in c.first_tri..c.first_tri + c.tri_count {
                for j in 0..3 {
                    let v = b.indices()[t as usize * 3 + j];
                    assert!(v >= c.first_vertex && v < c.first_vertex + c.vertex_count);
                }
            }
        }
    }

    #[test]
    fn old_relative_lights_are_skipped_not_misread() {
        let mut bytes = write(&sample(), &[]);
        // Find the LITE section and mark it version 1.
        let at = bytes.windows(4).position(|w| w == b"LITE").unwrap();
        bytes[at + 4..at + 8].copy_from_slice(&1u32.to_le_bytes());
        let (b, rep) = read(&bytes).unwrap();
        assert!(b.lights.is_empty());
        assert_eq!(rep.skipped.len(), 1);
        assert!(rep.skipped[0].starts_with("LITE version 1"));
    }

    /// A mesh built into levels of detail is written as version 2 and reads
    /// back with every cluster's level, errors and spheres.
    #[test]
    fn objects_and_groups_round_trip() {
        let mut s = sample();
        let n = s.clusters.len();
        let a = s.add_object("bishop".into(), 0..n, Vec3::new(0.5, 0.0, 0.0), Vec3::new(0.1, 0.3, -0.2));
        s.objects[a].scale = Vec3::new(1.5, 2.0, 0.5);
        let g = s.add_group("pieces".into(), Vec3::ONE);
        s.set_parent(a, Some(g));
        let (back, report) = read(&write(&s, &[])).unwrap();
        assert!(report.skipped.is_empty());
        assert_eq!(back.objects.len(), 2);
        let (oa, og) = (&back.objects[a], &back.objects[g]);
        assert_eq!((oa.name.as_str(), oa.clusters.clone(), oa.parent, oa.rotation, oa.scale), ("bishop", 0..n, Some(g), Vec3::new(0.1, 0.3, -0.2), Vec3::new(1.5, 2.0, 0.5)));
        assert_eq!((og.name.as_str(), og.clusters.is_empty(), og.parent, og.pivot), ("pieces", true, None, Vec3::ONE));
        assert!(oa.hi.x > oa.lo.x);
    }

    /// Version 1 objects, turned about Y only, still read.
    #[test]
    fn version_one_objects_read_as_a_turn_about_y() {
        let mut s = sample();
        let n = s.clusters.len();
        let mut w = W(Vec::new());
        w.u32(1);
        w.u32(0);
        w.u32(0);
        w.u32(n as u32);
        w.u32(0);
        w.u32(4);
        w.v3(Vec3::ONE);
        w.f32(0.5);
        w.f32(2.0);
        w.0.extend_from_slice(b"rook");
        read_objects(&w.0, 1, &mut s, n).unwrap();
        let o = &s.objects[0];
        assert_eq!((o.name.as_str(), o.rotation, o.scale, o.pivot), ("rook", Vec3::new(0.0, 0.5, 0.0), Vec3::splat(2.0), Vec3::ONE));
    }

    #[test]
    fn levels_of_detail_round_trip() {
        let (p, idx) = uv_sphere(64);
        let tris: Vec<[u32; 3]> = idx.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
        let (lods, _) = crate::lod::build(&p, &tris);
        let mut a = Scene::new();
        a.push_lod_mesh(&p, &p, &vec![Vec3::splat(0.5); p.len()], &lods);
        let bytes = write(&a, &[]);
        assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), VERSION_LOD);
        let (b, rep) = read(&bytes).unwrap();
        assert!(rep.skipped.is_empty());
        assert!(a.clusters.iter().any(|c| c.lod_level > 1));
        for (x, y) in a.clusters.iter().zip(&b.clusters) {
            assert_eq!((x.lod_level, x.lod_error, x.lod_centre, x.lod_radius), (y.lod_level, y.lod_error, y.lod_centre, y.lod_radius));
            assert_eq!((x.parent_error, x.parent_centre, x.parent_radius), (y.parent_error, y.parent_centre, y.parent_radius));
        }
        // Without levels, still version 1.
        assert_eq!(u32::from_le_bytes(write(&sample(), &[])[4..8].try_into().unwrap()), VERSION);
    }

    #[test]
    fn texture_tiles_round_trip_and_follow_instances() {
        use crate::texture::{level_offset, tile_points};
        let mut a = sample();
        // Level-2 tiles for every cluster: the byte value is the cluster.
        let mut pool = Vec::new();
        let mut entries = Vec::new();
        for (ci, c) in a.clusters.iter().enumerate() {
            entries.push(TexEntry { level: 2, source: 0, offset: pool.len() as u64, alpha: None });
            let len = level_offset(c.tri_count as usize, 3) as usize;
            pool.extend(std::iter::repeat_n(ci as u8, len));
        }
        a.textures.entries = entries;
        a.textures.sources = vec![Arc::new(Source::Memory(Arc::new(pool)))];
        let n = a.clusters.len() as u32;
        let bytes = write(&a, &[InstanceGroup { first_cluster: 0, cluster_count: n, transforms: vec![Mat4::from_translation(Vec3::X * 4.0)] }]);
        let (b, _) = read(&bytes).unwrap();
        assert_eq!(b.clusters.len(), a.clusters.len() * 2);
        for ci in 0..b.clusters.len() {
            let src = ci % a.clusters.len();
            let blk = b.textures.block(ci, b.clusters[ci].tri_count as usize, 2).unwrap();
            assert_eq!(blk.data.len(), b.clusters[ci].tri_count as usize * tile_points(2) * 3);
            assert!(blk.data.iter().all(|v| *v == src as u8), "cluster {ci} reads its own tiles");
        }
    }

    /// Alpha tiles go out in ALPH and come back, from memory and from a
    /// file, for the clusters that have them only.
    #[test]
    fn alpha_tiles_round_trip() {
        use crate::texture::{level_offset, tile_points};
        let mut a = sample();
        let (mut pool, mut alpha, mut entries) = (Vec::new(), Vec::new(), Vec::new());
        for (ci, c) in a.clusters.iter().enumerate() {
            let len = level_offset(c.tri_count as usize, 3) as usize;
            let cut = ci % 2 == 0;
            entries.push(TexEntry { level: 2, source: 0, offset: pool.len() as u64, alpha: cut.then_some(alpha.len() as u64) });
            pool.extend(std::iter::repeat_n(7u8, len));
            if cut {
                alpha.extend(std::iter::repeat_n(ci as u8 * 10, len / 3));
            }
        }
        a.textures.entries = entries;
        a.textures.sources = vec![Arc::new(Source::Memory(Arc::new(pool)))];
        a.textures.alpha_sources = vec![Some(Arc::new(Source::Memory(Arc::new(alpha))))];
        let bytes = write(&a, &[]);
        let path = std::env::temp_dir().join(format!("fkj-alpha-{}.fkj", std::process::id()));
        std::fs::write(&path, &bytes).unwrap();
        for (b, _) in [read(&bytes).unwrap(), read_file(&path).unwrap()] {
            for ci in 0..b.clusters.len() {
                assert_eq!(b.textures.has_alpha(ci), ci % 2 == 0);
                if let Some(blk) = b.textures.alpha_block(ci, b.clusters[ci].tri_count as usize, 2) {
                    assert_eq!(blk.data.len(), b.clusters[ci].tri_count as usize * tile_points(2));
                    assert!(blk.data.iter().all(|v| *v == ci as u8 * 10), "cluster {ci} reads its own alpha");
                }
            }
        }
        let _ = std::fs::remove_file(&path);
    }
}
