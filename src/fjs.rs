//! Scenes as two kinds of file:
//!
//! - `.fja`, an asset: one object with everything baked for it at import
//!   (clustered mesh, levels of detail, texture tiles, surface samples and
//!   their sky transfer), in the binary format of `fkj.rs`
//!   (`docs/format.md`), in its own space. Heavy for a dense mesh; written
//!   once, read by every scene that uses it.
//! - `.fjs`, a scene: a small JSON file that points at assets and holds
//!   what is about the scene, not the assets: where each object stands
//!   (pivot, rotation, scale) and what group it is in, the lights, the
//!   directional lights, the sun, the sky and exposure, the time of day,
//!   the camera and the render settings. Written by hand or by the editor.
//!
//! An asset path in a scene is relative to the scene's folder, or
//! absolute; when it is not found there, the file's name is looked for in
//! the asset folders (`assets/models`, `assets/local`, `$FKJENSEN_ASSETS`).
//! `"world": "built-in"` is the program's own ground and mountains.

use std::path::{Path, PathBuf};

use glam::Vec3;
use serde::{Deserialize, Serialize};

use crate::scene::{Object, PointLight, Scene, World};

pub const VERSION: u32 = 1;

/// An .fjs file.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct SceneFile {
    /// Format version, `VERSION`.
    pub fjs: u32,
    /// "built-in", an .fja holding the geometry outside every object, or
    /// nothing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub world: Option<String>,
    /// Objects and groups, a group before what is in it.
    #[serde(default)]
    pub objects: Vec<ObjectEntry>,
    /// Point lights.
    #[serde(default)]
    pub lights: Vec<LightEntry>,
    #[serde(default)]
    pub directional: Vec<DirEntry>,
    /// Time of day, hours. Sets the sun, the sky and the exposure before
    /// the values below, which override it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub time: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sun: Option<SunEntry>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sky: Option<SkyEntry>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backdrop: Option<BackdropEntry>,
    /// Display value per cd/m², before the tone curve.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exposure: Option<f32>,
    /// The perspective camera, Maya's persp.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub camera: Option<CameraEntry>,
    /// The scene's cameras.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cameras: Vec<NamedCamera>,
    /// The camera the view looks through, by name; persp when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub view: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub render: Option<RenderEntry>,
}

/// An object (with an asset) or a group (without).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ObjectEntry {
    pub name: String,
    /// The .fja it is; none for a group.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub asset: Option<String>,
    /// Where it stands: the point it turns and scales about, metres.
    pub pivot: [f32; 3],
    /// Degrees about X, Y and Z, applied in that order (Maya's xyz).
    #[serde(default)]
    pub rotation: [f32; 3],
    #[serde(default = "one")]
    pub scale: f32,
    /// The group it is in: an index into `objects`, of an earlier entry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<usize>,
    /// How its materials differ from its asset's, in this scene.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub materials: Vec<MaterialEntry>,
    /// An instance of an earlier entry (an index into `objects`): its
    /// geometry, materials and texture tiles, stored once. `asset` and
    /// `materials` are then that entry's.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instance_of: Option<usize>,
}

/// A change to one of an object's materials: only what differs from the
/// asset.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct MaterialEntry {
    /// Which of the asset's materials, from 0, and its name (for reading;
    /// the index decides).
    pub index: usize,
    #[serde(default)]
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tint: Option<[f32; 3]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub roughness: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metallic: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub specular: Option<f32>,
}

/// Object `i`'s materials that differ from its asset's.
pub fn material_changes(scene: &Scene, i: usize) -> Vec<MaterialEntry> {
    let o = &scene.objects[i];
    let close = |a: f32, b: f32| (a - b).abs() <= 1e-6;
    o.materials
        .clone()
        .enumerate()
        .filter_map(|(k, mi)| {
            let m = &scene.materials[mi];
            let a = o.asset_materials.get(k).cloned().unwrap_or_default();
            let e = MaterialEntry {
                index: k,
                name: m.name.clone(),
                tint: (!m.tint.abs_diff_eq(a.tint, 1e-6)).then(|| a3(m.tint)),
                roughness: (!close(m.roughness, a.roughness)).then(|| round6(m.roughness)),
                metallic: (!close(m.metallic, a.metallic)).then(|| round6(m.metallic)),
                specular: (!close(m.specular, a.specular)).then(|| round6(m.specular)),
            };
            (e.tint.is_some() || e.roughness.is_some() || e.metallic.is_some() || e.specular.is_some()).then_some(e)
        })
        .collect()
}

/// Apply a scene's changes to object `i`'s materials.
pub fn apply_material_changes(scene: &mut Scene, i: usize, changes: &[MaterialEntry]) {
    let r = scene.objects[i].materials.clone();
    for e in changes {
        let Some(m) = r.clone().nth(e.index).and_then(|k| scene.materials.get_mut(k)) else { continue };
        if let Some(t) = e.tint {
            m.tint = v3(t);
        }
        if let Some(x) = e.roughness {
            m.roughness = x;
        }
        if let Some(x) = e.metallic {
            m.metallic = x;
        }
        if let Some(x) = e.specular {
            m.specular = x;
        }
        *m = m.clone().clamped();
    }
}

/// Write object `i`'s materials as they are now into its asset's file,
/// which then has them; every scene using it sees them.
pub fn push_materials(scene: &mut Scene, i: usize) -> Result<PathBuf, String> {
    let o = &scene.objects[i];
    if o.asset.as_deref().and_then(crate::world::part_of).is_some() {
        return Err(format!("{} is built into the program: its materials stay with the scene", o.name));
    }
    let path = o.asset.clone().filter(|p| p.exists()).ok_or_else(|| format!("{} has no asset file yet: save the scene first", o.name))?;
    let mats = scene.materials[o.materials.clone()].to_vec();
    crate::fkj::rewrite_materials(&path, &mats)?;
    // Every object reading that file now rests on these.
    for ob in scene.objects.iter_mut().filter(|ob| ob.asset.as_ref() == Some(&path)) {
        ob.asset_materials = mats.clone();
    }
    Ok(path)
}

fn one() -> f32 {
    1.0
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
pub struct LightEntry {
    /// In the world, metres, whether parented or not.
    pub position: [f32; 3],
    /// Linear RGB.
    pub color: [f32; 3],
    pub lumens: f32,
    /// The object it is parented to: an index into `objects`. It moves
    /// with it, keeping where it is relative to it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<usize>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
pub struct DirEntry {
    pub on: bool,
    /// Degrees: azimuth clockwise from -Z, elevation above the horizon.
    pub azimuth: f32,
    pub elevation: f32,
    pub color: [f32; 3],
    pub lux: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
pub struct SunEntry {
    /// Towards the sun.
    pub direction: [f32; 3],
    pub color: [f32; 3],
    /// 0 to 1000.
    pub brightness: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
pub struct SkyEntry {
    pub on: bool,
    pub lux: f32,
    pub tint: [f32; 3],
    /// Lighting contrast, 0 to 1.
    pub contrast: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
pub struct BackdropEntry {
    pub zenith: [f32; 3],
    pub horizon: [f32; 3],
    /// Visible sky contrast, 0 to 1.
    pub contrast: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
pub struct CameraEntry {
    pub target: [f32; 3],
    pub distance: f32,
    /// Degrees about the vertical, and above the horizon.
    pub yaw: f32,
    pub pitch: f32,
    /// Vertical field of view, degrees.
    pub fov: f32,
}

/// A scene camera: its name and where it is, as `CameraEntry`.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct NamedCamera {
    pub name: String,
    #[serde(flatten)]
    pub camera: CameraEntry,
}

/// Render settings that belong to the scene. The micropolygon spacing and
/// the frame size follow the machine and are not saved.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Default)]
pub struct RenderEntry {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shadows: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shadow_res: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bounce: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub light_reach: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub occlusion_culling: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adaptive: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sky_occlusion: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub incremental_moves: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backface_culling: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wireframe: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub drag_preview: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bake_over_frames: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub moving_rate: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub moving_detail: Option<f32>,
}

pub fn v3(a: [f32; 3]) -> Vec3 {
    Vec3::from_array(a)
}

/// A vector for the file, rounded to 6 significant digits so the file
/// reads and diffs cleanly.
pub fn a3(v: Vec3) -> [f32; 3] {
    v.to_array().map(round6)
}

pub fn round6(x: f32) -> f32 {
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    let p = 10f64.powi(5 - (x.abs() as f64).log10().floor() as i32);
    ((x as f64 * p).round() / p) as f32
}

/// Read and check an .fjs file.
pub fn read(path: &Path) -> Result<SceneFile, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    parse(&text).map_err(|e| format!("{}: {e}", path.display()))
}

pub fn parse(text: &str) -> Result<SceneFile, String> {
    let f: SceneFile = serde_json::from_str(text).map_err(|e| e.to_string())?;
    if f.fjs == 0 || f.fjs > VERSION {
        return Err(format!("scene format version {} (this build reads 1 to {VERSION})", f.fjs));
    }
    for (i, o) in f.objects.iter().enumerate() {
        if o.parent.is_some_and(|p| p >= i) {
            return Err(format!("object {i} ({}) is in a group that does not come before it", o.name));
        }
    }
    Ok(f)
}

pub fn write(path: &Path, f: &SceneFile) -> Result<(), String> {
    let text = to_text(f)?;
    std::fs::write(path, text).map_err(|e| format!("{}: {e}", path.display()))
}

/// The file's text: JSON indented by two, with each list of numbers on
/// one line.
pub fn to_text(f: &SceneFile) -> Result<String, String> {
    let pretty = serde_json::to_string_pretty(f).map_err(|e| e.to_string())?;
    let mut out = String::with_capacity(pretty.len());
    let mut rest = pretty.as_str();
    while let Some(k) = rest.find('[') {
        out.push_str(&rest[..k]);
        rest = &rest[k..];
        let inner = rest[1..].find(']').map(|e| &rest[1..1 + e]);
        match inner {
            Some(inner) if !inner.trim().is_empty() && inner.chars().all(|c| c.is_ascii_digit() || matches!(c, '-' | '+' | '.' | 'e' | 'E' | ',' | ' ' | '\n')) => {
                let items: Vec<&str> = inner.split(',').map(str::trim).collect();
                out.push('[');
                out.push_str(&items.join(", "));
                out.push(']');
                rest = &rest[inner.len() + 2..];
            }
            _ => {
                out.push('[');
                rest = &rest[1..];
            }
        }
    }
    out.push_str(rest);
    out.push('\n');
    Ok(out)
}

/// Where asset `name` of a scene in folder `dir` is: as written (relative
/// to `dir`), else its file name in `search`.
pub fn resolve(name: &str, dir: &Path, search: &[PathBuf]) -> Option<PathBuf> {
    let p = Path::new(name);
    let direct = if p.is_absolute() { p.to_path_buf() } else { dir.join(p) };
    if direct.exists() {
        return Some(direct);
    }
    let file = p.file_name()?;
    search.iter().map(|d| d.join(file)).find(|q| q.exists())
}

/// `path` as written in a scene in folder `dir`: relative when it is
/// inside it, else absolute; `/` between folders on every system.
pub fn relative(path: &Path, dir: &Path) -> String {
    let abs = |p: &Path| std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
    let (p, d) = (abs(path), abs(dir));
    match p.strip_prefix(&d) {
        Ok(r) => r.components().map(|c| c.as_os_str().to_string_lossy().into_owned()).collect::<Vec<_>>().join("/"),
        Err(_) => p.display().to_string(),
    }
}

/// Read an .fja asset as a scene of one object, in its own space, at the
/// pivot, rotation and scale it was stored with (the origin, unturned, at
/// scale 1 when it has none).
pub fn load_asset(path: &Path) -> Result<Scene, String> {
    let t = std::time::Instant::now();
    let (mut s, _) = crate::fkj::read_file(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let n = s.clusters.len();
    let name = path.file_stem().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let rest = s.objects.iter().find(|o| o.clusters == (0..n)).map(|o| (o.name.clone(), o.pivot, o.rotation, o.scale));
    let (oname, pivot, rotation, scale) = rest.unwrap_or((name.clone(), Vec3::ZERO, Vec3::ZERO, 1.0));
    s.objects = vec![Object::new(if oname.is_empty() { name } else { oname }, 0..n, pivot, rotation, scale, None)];
    s.objects[0].asset = Some(path.to_path_buf());
    s.own_materials(0);
    s.refresh_bounds(0);
    s.lights.clear();
    s.scatter_missing_samples();
    eprintln!("{}: read in {:.2} s: {} triangles, {} clusters", path.display(), t.elapsed().as_secs_f64(), s.triangle_count(), n);
    Ok(s)
}

/// Write object `i` as an .fja at `path`, in its own space.
pub fn write_asset(scene: &Scene, i: usize, path: &Path) -> Result<(), String> {
    write_scene_file(&scene.extract_object(i), path)
}

/// Write `scene` (an asset, or the world) to `path`, through a file beside
/// it: its texture tiles may be read from the file being replaced.
pub fn write_scene_file(scene: &Scene, path: &Path) -> Result<(), String> {
    let tmp = path.with_extension("fja.part");
    let written = std::fs::File::create(&tmp).and_then(|f| {
        let mut w = std::io::BufWriter::with_capacity(1 << 20, f);
        crate::fkj::write_to(&mut w, scene, &[])?;
        std::io::Write::flush(&mut w)
    });
    match written.and_then(|_| std::fs::rename(&tmp, path)) {
        Ok(()) => Ok(()),
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            Err(format!("{}: {e}", path.display()))
        }
    }
}

/// The scene a file describes: its world, then its objects, each read from
/// its asset and placed, and its lights. `dir` is the scene's folder.
pub fn build(f: &SceneFile, dir: &Path, search: &[PathBuf]) -> Result<Scene, String> {
    let mut scene = match f.world.as_deref() {
        // Files from before the ground and mountains were objects.
        Some("built-in") => crate::world::build(),
        Some(w) => {
            let p = resolve(w, dir, search).ok_or_else(|| format!("world asset {w} not found"))?;
            let mut s = load_asset(&p)?;
            s.objects.clear();
            s.world = World::Asset(p);
            s
        }
        None => Scene::default(),
    };
    let mut index = Vec::with_capacity(f.objects.len());
    for o in &f.objects {
        let parent = o.parent.map(|p| index[p]);
        if let Some(k) = o.instance_of {
            let j = *index.get(k).ok_or_else(|| format!("{}: instance_of {k} is not an earlier object", o.name))?;
            let i = scene.instance_object(j, o.name.clone());
            scene.objects[i].parent = parent;
            index.push(i);
            continue;
        }
        match &o.asset {
            Some(a) => {
                let mut m = match crate::world::part_of(Path::new(a)) {
                    Some(part) => crate::world::part(part).expect("a built-in part"),
                    None => load_asset(&resolve(a, dir, search).ok_or_else(|| format!("{}: asset {a} not found", o.name))?)?,
                };
                m.objects[0].name = o.name.clone();
                m.objects[0].parent = None;
                let i = scene.objects.len();
                scene.append(m);
                scene.objects[i].parent = parent;
                apply_material_changes(&mut scene, i, &o.materials);
                index.push(i);
            }
            None => {
                let i = scene.add_group(o.name.clone(), v3(o.pivot));
                scene.objects[i].parent = parent;
                index.push(i);
            }
        }
    }
    // Groups come before what is in them, so placing in order moves each
    // object last as itself: where the file says.
    for (o, &i) in f.objects.iter().zip(&index) {
        scene.set_object(i, v3(o.pivot), v3(o.rotation.map(f32::to_radians)), o.scale);
    }
    scene.lights = f.lights.iter().map(|l| PointLight { position: v3(l.position), color: v3(l.color), lumens: l.lumens.clamp(crate::scene::MIN_LUMENS, crate::scene::MAX_LUMENS) }).collect();
    scene.light_parents = f.lights.iter().map(|l| l.parent.and_then(|p| index.get(p).copied())).collect();
    scene.focus = scene.subject();
    Ok(scene)
}

/// A name for a new asset file: `name` with what a file name cannot hold
/// replaced, not taken by another file in `dir` (`taken` holds names
/// already chosen).
pub fn asset_name(name: &str, dir: &Path, taken: &[PathBuf]) -> PathBuf {
    let clean: String = name.chars().map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect();
    let clean = if clean.is_empty() { "object".to_string() } else { clean };
    for k in 1.. {
        let file = if k == 1 { format!("{clean}.fja") } else { format!("{clean}_{k}.fja") };
        let p = dir.join(file);
        if !p.exists() && !taken.contains(&p) {
            return p;
        }
    }
    unreachable!()
}

/// Write an .fja for every object that has none on disk, and for the
/// geometry outside every object when it came from no file, beside the
/// scene at `scene_path`. Each object then refers to its file. Returns
/// the files written.
pub fn save_assets(scene: &mut Scene, scene_path: &Path) -> Result<Vec<PathBuf>, String> {
    let dir = scene_path.parent().map(Path::to_path_buf).unwrap_or_default();
    let stem = scene_path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "scene".into());
    let mut written = Vec::new();
    let mut done: std::collections::HashMap<(u32, u32), usize> = std::collections::HashMap::new();
    for i in 0..scene.objects.len() {
        // An instance of an object already written is in its asset.
        if let Some(g) = scene.geometry_of(i) {
            if let Some(&j) = done.get(&g) {
                scene.objects[i].asset = scene.objects[j].asset.clone();
                continue;
            }
            done.insert(g, i);
        }
        if scene.objects[i].clusters.is_empty() || scene.objects[i].asset.as_ref().is_some_and(|p| p.exists() || crate::world::part_of(p).is_some()) {
            continue;
        }
        let p = asset_name(&scene.objects[i].name, &dir, &written);
        write_asset(scene, i, &p)?;
        scene.objects[i].asset = Some(p.clone());
        // The asset holds its materials as they are.
        scene.objects[i].asset_materials = scene.materials[scene.objects[i].materials.clone()].to_vec();
        written.push(p);
    }
    if scene.world == World::Unsaved || matches!(&scene.world, World::Asset(p) if !p.exists()) {
        let loose = scene.loose_clusters();
        if !loose.is_empty() {
            let p = asset_name(&format!("{stem}_world"), &dir, &written);
            write_scene_file(&scene.extract(&loose, glam::Affine3A::IDENTITY), &p)?;
            scene.world = World::Asset(p.clone());
            written.push(p);
        }
    }
    Ok(written)
}

/// The scene's objects, world and lights as file entries (the rest is the
/// app's). Assets must have been saved (`save_assets`).
pub fn entries(scene: &Scene, scene_path: &Path, lights: &[PointLight], light_parents: &[Option<usize>]) -> SceneFile {
    let dir = scene_path.parent().map(Path::to_path_buf).unwrap_or_default();
    // Groups before what is in them: parents first, keeping the order
    // otherwise.
    let n = scene.objects.len();
    let mut order: Vec<usize> = Vec::with_capacity(n);
    let mut placed = vec![false; n];
    while order.len() < n {
        let before = order.len();
        for i in 0..n {
            if !placed[i] && scene.objects[i].parent.is_none_or(|p| placed[p]) {
                placed[i] = true;
                order.push(i);
            }
        }
        if order.len() == before {
            // A loop (never made by the editor): the rest at the top.
            for (i, p) in placed.iter_mut().enumerate() {
                if !*p {
                    *p = true;
                    order.push(i);
                }
            }
        }
    }
    let mut at = vec![0; n];
    for (k, &i) in order.iter().enumerate() {
        at[i] = k;
    }
    // An instance names the first entry with its geometry.
    let mut first_with: std::collections::HashMap<(u32, u32), usize> = std::collections::HashMap::new();
    let source: Vec<Option<usize>> = order
        .iter()
        .map(|&i| {
            let g = scene.geometry_of(i)?;
            let first = *first_with.entry(g).or_insert(i);
            (first != i).then_some(first)
        })
        .collect();
    let objects = order
        .iter()
        .zip(&source)
        .map(|(&i, &src)| {
            let o = &scene.objects[i];
            if let Some(j) = src {
                return ObjectEntry {
                    name: o.name.clone(),
                    asset: None,
                    pivot: a3(o.pivot),
                    rotation: a3(o.rotation).map(|r| round6(r.to_degrees())),
                    scale: round6(o.scale),
                    parent: o.parent.filter(|&p| p < n && at[p] < at[i]).map(|p| at[p]),
                    materials: Vec::new(),
                    instance_of: Some(at[j]),
                };
            }
            ObjectEntry {
                name: o.name.clone(),
                asset: if o.clusters.is_empty() {
                    None
                } else {
                    o.asset.as_ref().map(|p| if crate::world::part_of(p).is_some() { p.display().to_string() } else { relative(p, &dir) })
                },
                pivot: a3(o.pivot),
                rotation: a3(o.rotation).map(|r| round6(r.to_degrees())),
                scale: round6(o.scale),
                parent: o.parent.filter(|&p| p < n && at[p] < at[i]).map(|p| at[p]),
                materials: if o.clusters.is_empty() { Vec::new() } else { material_changes(scene, i) },
                instance_of: None,
            }
        })
        .collect();
    let world = match &scene.world {
        World::Asset(p) => Some(relative(p, &dir)),
        World::Unsaved => None,
    };
    SceneFile {
        fjs: VERSION,
        world,
        objects,
        lights: lights
            .iter()
            .enumerate()
            .map(|(k, l)| LightEntry { position: a3(l.position), color: a3(l.color), lumens: round6(l.lumens), parent: light_parents.get(k).copied().flatten().filter(|&p| p < n).map(|p| at[p]) })
            .collect(),
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("fjs-test-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// A ball object moved, turned and scaled, and a group over it.
    fn ball_scene() -> Scene {
        let (p, idx) = crate::scene::tests::uv_sphere(12);
        let mut s = Scene::default();
        let ball: Vec<Vec3> = p.iter().map(|v| *v * 0.5 + Vec3::new(0.0, 0.5, 0.0)).collect();
        s.push_mesh(&ball, &[], &[Vec3::new(0.8, 0.3, 0.2)], &idx);
        let n = s.clusters.len();
        s.add_object("ball one".into(), 0..n, Vec3::ZERO, Vec3::ZERO);
        s.scatter_missing_samples();
        let g = s.add_group("group".into(), Vec3::new(1.0, 0.0, 0.0));
        s.objects[0].parent = Some(g);
        s.set_object(0, Vec3::new(2.0, 0.3, -1.0), Vec3::new(0.2, 0.9, -0.1), 1.5);
        s.set_object(g, Vec3::new(1.5, 0.0, 0.5), Vec3::new(0.0, 0.4, 0.0), 1.0);
        s
    }

    /// Saving writes the object as an asset in its own space and the scene
    /// as JSON pointing at it; opening puts every vertex back where it was.
    #[test]
    fn a_scene_saves_and_opens_with_its_objects_where_they_were() {
        let d = tmp("round");
        let mut s = ball_scene();
        s.objects[1].name = "groupé [1]".into();
        let path = d.join("my scene.fjs");
        let written = save_assets(&mut s, &path).unwrap();
        assert_eq!(written, vec![d.join("ball_one.fja")]);
        let mut f = entries(&s, &path, &[PointLight { position: Vec3::ONE, color: Vec3::ONE, lumens: 900.0 }], &[]);
        f.time = Some(17.5);
        write(&path, &f).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("\"asset\": \"ball_one.fja\""), "{text}");
        let g = read(&path).unwrap();
        assert_eq!(g, f);
        // The group comes first, the ball in it.
        assert_eq!(g.objects[0].name, "groupé [1]");
        assert_eq!(g.objects[1].parent, Some(0));
        let t = build(&g, &d, &[]).unwrap();
        assert_eq!(t.objects.len(), 2);
        let (a, b) = (&s.objects[0], t.objects.iter().find(|o| o.name == "ball one").unwrap());
        assert!(a.pivot.distance(b.pivot) < 1e-4 && (a.scale - b.scale).abs() < 1e-5);
        for (ca, cb) in s.clusters[a.clusters.clone()].iter().zip(&t.clusters[b.clusters.clone()]) {
            for (va, vb) in (ca.first_vertex..ca.first_vertex + ca.vertex_count).zip(cb.first_vertex..cb.first_vertex + cb.vertex_count) {
                assert!(s.pos(ca, va as usize).distance(t.pos(cb, vb as usize)) < 1e-3);
                assert!(s.normal(ca, va as usize).distance(t.normal(cb, vb as usize)) < 0.02);
            }
        }
        assert_eq!(t.samples.len(), s.samples.len());
        assert_eq!(t.lights.len(), 1);
        // The asset is in its own space: where the ball was made.
        let asset = load_asset(&d.join("ball_one.fja")).unwrap();
        assert!(asset.objects[0].pivot.length() < 1e-4, "{}", asset.objects[0].pivot);
        assert!((asset.bounds_min.y - 0.0).abs() < 0.01 && (asset.bounds_max.y - 1.0).abs() < 0.01);
        // Saving again writes no asset: the object has its file.
        let mut t2 = t;
        assert!(save_assets(&mut t2, &path).unwrap().is_empty());
        let _ = std::fs::remove_dir_all(&d);
    }

    /// A material changed in a scene is saved as a change in the .fjs and
    /// the asset keeps its own; pushed, the asset has it and the scene
    /// has no change left to save.
    #[test]
    fn material_changes_stay_in_the_scene_until_pushed() {
        let d = tmp("mats");
        let mut s = ball_scene();
        // Two materials on the ball: the first half of its triangles each.
        let r = s.objects[0].materials.clone();
        assert_eq!(r.len(), 1);
        s.materials.push(crate::scene::Material { name: "second".into(), ..Default::default() });
        s.objects[0].materials = r.start..r.start + 2;
        let n = s.triangle_count();
        s.tri_mat = (0..n).map(|t| (t * 2 >= n) as u8).collect();
        s.objects[0].asset_materials = s.materials[s.objects[0].materials.clone()].to_vec();
        let path = d.join("m.fjs");
        save_assets(&mut s, &path).unwrap();
        // Change the second material in the scene.
        let k = s.objects[0].materials.start + 1;
        s.materials[k].roughness = 0.9;
        s.materials[k].metallic = 1.0;
        let f = entries(&s, &path, &[], &[]);
        let ball = f.objects.iter().find(|o| o.name == "ball one").unwrap();
        assert_eq!(ball.materials.len(), 1);
        assert_eq!((ball.materials[0].index, ball.materials[0].roughness, ball.materials[0].metallic, ball.materials[0].tint), (1, Some(0.9), Some(1.0), None));
        write(&path, &f).unwrap();
        let t = build(&read(&path).unwrap(), &d, &[]).unwrap();
        let tb = t.objects.iter().position(|o| o.name == "ball one").unwrap();
        let tm = &t.materials[t.objects[tb].materials.clone()];
        assert_eq!((tm[1].roughness, tm[1].metallic), (0.9, 1.0));
        assert_eq!(t.objects[tb].asset_materials[1].roughness, 0.5, "the asset keeps its own");
        // Each triangle kept its material through the asset.
        let c0 = &t.clusters[t.objects[tb].clusters.start];
        assert_eq!(t.material(c0, c0.first_tri).name, s.material(&s.clusters[s.objects[0].clusters.start], s.clusters[s.objects[0].clusters.start].first_tri).name);
        // Pushed: the file has it, and nothing is left to save as a change.
        let mut t = t;
        push_materials(&mut t, tb).unwrap();
        let a = load_asset(&d.join("ball_one.fja")).unwrap();
        assert_eq!((a.materials[1].roughness, a.materials[1].metallic), (0.9, 1.0));
        assert_eq!(a.tri_mat, crate::fjs::load_asset(&d.join("ball_one.fja")).unwrap().tri_mat);
        assert!(material_changes(&t, tb).is_empty());
        let _ = std::fs::remove_dir_all(&d);
    }

    /// The ground and the mountains are objects with built-in assets: a
    /// deleted one stays deleted when the scene is saved and opened, and
    /// no file is written for them. Old files' `"world": "built-in"`
    /// still gives both.
    #[test]
    fn the_built_in_world_is_objects() {
        let d = tmp("builtin");
        let mut s = crate::world::build();
        assert_eq!(s.objects.iter().map(|o| o.name.as_str()).collect::<Vec<_>>(), vec!["ground", "mountains"]);
        assert!(s.loose_clusters().is_empty());
        let m = s.objects.iter().position(|o| o.name == "mountains").unwrap();
        s.remove_objects(&[m]);
        let path = d.join("w.fjs");
        assert!(save_assets(&mut s, &path).unwrap().is_empty());
        let f = entries(&s, &path, &[], &[]);
        assert_eq!(f.world, None);
        assert_eq!(f.objects.len(), 1);
        assert_eq!(f.objects[0].asset.as_deref(), Some("built-in:ground"));
        let t = build(&f, &d, &[]).unwrap();
        assert_eq!(t.objects.len(), 1);
        assert!(t.objects[0].environment);
        assert_eq!(t.clusters.len(), s.clusters.len());
        let old = parse(r#"{ "fjs": 1, "world": "built-in" }"#).unwrap();
        assert_eq!(build(&old, &d, &[]).unwrap().objects.len(), 2);
        // The camera's subject is the objects, not kilometres of ground: a
        // subject that size put the near plane tens of metres out and hid
        // everything closer.
        assert_eq!(t.focus, Some(crate::scene::set_middle()));
        let with = parse(r#"{ "fjs": 1, "world": "built-in", "objects": [ { "name": "c", "asset": "built-in:cube", "pivot": [7.5, 0.0, 5.0] } ] }"#).unwrap();
        let (lo, hi) = build(&with, &d, &[]).unwrap().focus.unwrap();
        assert!(lo.distance(glam::Vec3::new(7.0, 0.0, 4.5)) < 1e-3 && hi.distance(glam::Vec3::new(8.0, 1.0, 5.5)) < 1e-3, "{lo} {hi}");
        let _ = std::fs::remove_dir_all(&d);
    }

    /// An asset path not found beside the scene is looked for by name in
    /// the asset folders.
    #[test]
    fn assets_are_found_beside_the_scene_or_by_name() {
        let d = tmp("find");
        let other = d.join("library");
        std::fs::create_dir_all(&other).unwrap();
        std::fs::write(other.join("rock.fja"), b"x").unwrap();
        assert_eq!(resolve("models/rock.fja", &d, std::slice::from_ref(&other)), Some(other.join("rock.fja")));
        assert_eq!(resolve("library/rock.fja", &d, &[]), Some(d.join("library/rock.fja")));
        assert_eq!(resolve("nothing.fja", &d, std::slice::from_ref(&other)), None);
        assert_eq!(relative(&other.join("rock.fja"), &d), "library/rock.fja");
        let _ = std::fs::remove_dir_all(&d);
    }

    /// A hand-written scene: only what is given is set.
    #[test]
    fn a_minimal_scene_parses() {
        let f = parse(r#"{ "fjs": 1, "world": "built-in", "objects": [ { "name": "g", "pivot": [0, 0, 0] } ] }"#).unwrap();
        assert_eq!(f.objects[0].scale, 1.0);
        assert!(f.camera.is_none() && f.render.is_none());
        assert!(parse(r#"{ "fjs": 9 }"#).is_err());
        assert!(parse(r#"{ "fjs": 1, "objects": [ { "name": "a", "pivot": [0,0,0], "parent": 0 } ] }"#).is_err());
    }

    /// Loose geometry from a file not kept (an old .fkj scene) is saved as
    /// the scene's world asset.
    #[test]
    fn loose_geometry_becomes_the_world_asset() {
        let d = tmp("world");
        let mut s = ball_scene();
        let quad = [Vec3::new(-5.0, 0.0, 5.0), Vec3::new(5.0, 0.0, 5.0), Vec3::new(5.0, 0.0, -5.0), Vec3::new(-5.0, 0.0, -5.0)];
        s.push_mesh(&quad, &[], &[Vec3::splat(0.5)], &[0, 1, 2, 0, 2, 3]);
        assert_eq!(s.loose_clusters().len(), 1);
        let path = d.join("old.fjs");
        let written = save_assets(&mut s, &path).unwrap();
        assert!(written.contains(&d.join("old_world.fja")), "{written:?}");
        let f = entries(&s, &path, &[], &[]);
        assert_eq!(f.world.as_deref(), Some("old_world.fja"));
        let t = build(&f, &d, &[]).unwrap();
        assert_eq!(t.clusters.len(), s.clusters.len());
        assert_eq!(t.loose_clusters().len(), 1);
        let _ = std::fs::remove_dir_all(&d);
    }
}
