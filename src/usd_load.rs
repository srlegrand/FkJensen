//! Reads a USD stage (usda, usdc, usdz) into a flat `Scene`.
//!
//! What is read: every `Mesh` the default traversal reaches, instance proxies
//! included, with its world transform, points, face topology (triangulated as
//! fans), vertex normals when authored per vertex, and `displayColor`
//! (constant or per vertex). Invisible prims and `guide`/`proxy` purposes
//! are skipped. `upAxis = Z` is turned into Y up; `metersPerUnit` scales to
//! metres. `SphereLight` and `DistantLight` prims become lights.
//!
//! Not read yet: face-varying normals and colours (averaged to vertices),
//! subdivision (meshes are drawn as polygons), textures and materials,
//! point instancers, curves and points prims, time samples (the default time
//! is read).

use std::path::Path;
use std::time::Instant;

use glam::{Mat4, Vec3};
use openusd::{gf, usd};
use openusd_schemas::geom::{self, GprimSchema, ImageableSchema, Interpolation, MeshSchema, PointBasedSchema, Primvar, StageMetadata};
use openusd_schemas::lux;

use crate::scene::{Light, Scene};

pub struct LoadReport {
    pub scene: Scene,
    pub open_secs: f64,
    pub read_secs: f64,
    pub warnings: Vec<String>,
}

fn to_mat4(m: &gf::Matrix4d) -> Mat4 {
    // USD matrices are row-major with row vectors (p' = p M). glam is
    // column-major with column vectors (p' = M p). Reading USD's 16 values as
    // a column-major array is exactly the transpose, which is the matrix glam
    // wants.
    let a: [f32; 16] = std::array::from_fn(|i| m.0[i] as f32);
    Mat4::from_cols_array(&a)
}

fn v3(v: &gf::Vec3f) -> Vec3 {
    Vec3::new(v.x, v.y, v.z)
}

pub fn load(path: &Path) -> Result<LoadReport, String> {
    let t0 = Instant::now();
    let stage = usd::Stage::builder()
        .schema_registry(openusd_schemas::schema_registry())
        .open(path.to_str().ok_or("path is not UTF-8")?)
        .map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let open_secs = t0.elapsed().as_secs_f64();

    let mut warnings: Vec<String> = stage.composition_errors().iter().map(|e| e.to_string()).collect();

    let t1 = Instant::now();
    let mut scene = Scene::new();

    // Stage-level axis and unit: applied as one correction on every world
    // matrix, never baked into points by the file.
    let up_z = stage.up_axis().ok().flatten().map(|t| t.as_str() == "Z").unwrap_or(false);
    let mpu = stage.meters_per_unit().ok().flatten().unwrap_or(0.01) as f32;
    let mut correction = Mat4::from_scale(Vec3::splat(mpu));
    if up_z {
        // Z up to Y up: rotate -90 degrees about X.
        correction *= Mat4::from_rotation_x(-std::f32::consts::FRAC_PI_2);
    }

    let mut paths = Vec::new();
    stage
        .traverse(usd::PrimPredicate::DEFAULT_PROXIES, |p| paths.push(p.clone()))
        .map_err(|e| e.to_string())?;

    let mut xf = geom::XformCache::default();

    for p in &paths {
        let prim = match stage.prim(p.clone()) {
            Ok(p) => p,
            Err(_) => continue,
        };
        let ty = prim.type_name().ok().flatten().map(|t| t.as_str().to_string()).unwrap_or_default();

        match ty.as_str() {
            "Mesh" => {
                match read_mesh(&prim, &mut xf, &correction, &mut scene) {
                    Ok(true) => scene.prims_read += 1,
                    Ok(false) => scene.prims_skipped += 1,
                    Err(e) => {
                        scene.prims_skipped += 1;
                        warnings.push(format!("{p}: {e}"));
                    }
                }
            }
            "SphereLight" | "DistantLight" | "RectLight" | "DiskLight" => {
                if let Some(l) = read_light(&prim, &ty, &mut xf, &correction) {
                    scene.lights.push(l);
                }
            }
            _ => {}
        }
    }

    if !scene.bounds_min.is_finite() {
        scene.bounds_min = Vec3::ZERO;
        scene.bounds_max = Vec3::ZERO;
    }
    let read_secs = t1.elapsed().as_secs_f64();

    Ok(LoadReport { scene, open_secs, read_secs, warnings })
}

fn read_mesh(prim: &usd::Prim, xf: &mut geom::XformCache, correction: &Mat4, scene: &mut Scene) -> Result<bool, String> {
    let mesh = geom::Mesh::from_prim(prim.clone())
        .map_err(|e| e.to_string())?
        .ok_or("not a Mesh under the registry")?;

    // Visibility and purpose. Inherited visibility is resolved by walking up.
    if !is_visible(prim) {
        return Ok(false);
    }
    if let Ok(Some(purpose)) = mesh.purpose() {
        use geom::Purpose::*;
        if matches!(purpose, Guide | Proxy) {
            return Ok(false);
        }
    }

    let points: Vec<gf::Vec3f> = mesh.points().map_err(|e| e.to_string())?.unwrap_or_default();
    if points.is_empty() {
        return Ok(false);
    }
    let counts: Vec<i32> = mesh.face_vertex_counts().map_err(|e| e.to_string())?.unwrap_or_default();
    let fvi: Vec<i32> = mesh.face_vertex_indices().map_err(|e| e.to_string())?.unwrap_or_default();

    let world = *correction * to_mat4(&xf.local_to_world_transform(prim).map_err(|e| e.to_string())?);
    let normal_m = world.inverse().transpose();

    // Left-handed meshes (orientation = leftHanded, or a mirroring transform)
    // need their winding flipped so the front faces stay the front faces.
    let left_handed = mesh
        .orientation_attr()
        .get::<openusd::tf::Token>()
        .ok()
        .flatten()
        .map(|t| t.as_str() == "leftHanded")
        .unwrap_or(false);
    let mirrored = world.determinant() < 0.0;
    let flip = left_handed != mirrored;

    let n_pts = points.len();
    let positions: Vec<Vec3> = points.iter().map(|p| world.transform_point3(v3(p))).collect();

    // Triangulate as fans. Corner index k of face f -> vertex fvi[off+k].
    let mut tris: Vec<u32> = Vec::with_capacity(fvi.len());
    // For face-varying data we need the corner index per triangle corner.
    let mut tri_corners: Vec<u32> = Vec::new();
    let mut off = 0usize;
    for &c in &counts {
        let c = c.max(0) as usize;
        if off + c > fvi.len() {
            break;
        }
        for k in 1..c.saturating_sub(1) {
            let (a, b, cc) = (fvi[off] as u32, fvi[off + k] as u32, fvi[off + k + 1] as u32);
            if (a as usize) < n_pts && (b as usize) < n_pts && (cc as usize) < n_pts {
                if flip {
                    tris.extend_from_slice(&[a, cc, b]);
                    tri_corners.extend_from_slice(&[off as u32, (off + k + 1) as u32, (off + k) as u32]);
                } else {
                    tris.extend_from_slice(&[a, b, cc]);
                    tri_corners.extend_from_slice(&[off as u32, (off + k) as u32, (off + k + 1) as u32]);
                }
            }
        }
        off += c;
    }
    if tris.is_empty() {
        return Ok(false);
    }

    // Normals: per vertex when authored that way; face-varying ones are
    // averaged onto vertices (a seam loses its hard edge, noted in the docs).
    let mut normals: Vec<Vec3> = Vec::new();
    if let Some(pv) = Primvar::from_attribute(mesh.normals_attr()) {
        let interp = pv.interpolation().unwrap_or(Interpolation::Vertex);
        if let Ok(Some(ns)) = pv.compute_flattened::<Vec<gf::Vec3f>>(None) {
            match interp {
                Interpolation::Vertex | Interpolation::Varying if ns.len() == n_pts => {
                    normals = ns.iter().map(|n| normal_m.transform_vector3(v3(n)).normalize_or(Vec3::Y)).collect();
                }
                Interpolation::FaceVarying if ns.len() == fvi.len() => {
                    let mut acc = vec![Vec3::ZERO; n_pts];
                    for (corner, n) in ns.iter().enumerate() {
                        let v = fvi[corner] as usize;
                        if v < n_pts {
                            acc[v] += v3(n);
                        }
                    }
                    normals = acc.iter().map(|n| normal_m.transform_vector3(*n).normalize_or(Vec3::Y)).collect();
                }
                _ => {}
            }
        }
    }
    if flip {
        for n in &mut normals {
            *n = -*n;
        }
    }
    let _ = &tri_corners; // kept for the face-varying path to come

    // displayColor: constant, vertex, uniform (per face, averaged to vertex)
    // or face-varying (averaged to vertex).
    let mut colors: Vec<Vec3> = Vec::new();
    if let Some(pv) = Primvar::from_attribute(mesh.display_color_attr()) {
        if let Ok(Some(cs)) = pv.compute_flattened::<Vec<gf::Vec3f>>(None) {
            let interp = pv.interpolation().unwrap_or(Interpolation::Constant);
            match interp {
                Interpolation::Vertex | Interpolation::Varying if cs.len() == n_pts => {
                    colors = cs.iter().map(v3).collect();
                }
                Interpolation::Uniform if cs.len() == counts.len() => {
                    let mut acc = vec![Vec3::ZERO; n_pts];
                    let mut cnt = vec![0u32; n_pts];
                    let mut off = 0usize;
                    for (f, &c) in counts.iter().enumerate() {
                        for k in 0..c.max(0) as usize {
                            if let Some(&v) = fvi.get(off + k) {
                                let v = v as usize;
                                if v < n_pts {
                                    acc[v] += v3(&cs[f]);
                                    cnt[v] += 1;
                                }
                            }
                        }
                        off += c.max(0) as usize;
                    }
                    colors = acc.iter().zip(&cnt).map(|(a, &n)| if n > 0 { *a / n as f32 } else { Vec3::splat(0.7) }).collect();
                }
                Interpolation::FaceVarying if cs.len() == fvi.len() => {
                    let mut acc = vec![Vec3::ZERO; n_pts];
                    let mut cnt = vec![0u32; n_pts];
                    for (corner, c) in cs.iter().enumerate() {
                        let v = fvi[corner] as usize;
                        if v < n_pts {
                            acc[v] += v3(c);
                            cnt[v] += 1;
                        }
                    }
                    colors = acc.iter().zip(&cnt).map(|(a, &n)| if n > 0 { *a / n as f32 } else { Vec3::splat(0.7) }).collect();
                }
                _ => {
                    if let Some(c) = cs.first() {
                        colors = vec![v3(c)];
                    }
                }
            }
        }
    }

    scene.push_mesh(&positions, &normals, &colors, &tris);
    Ok(true)
}

fn is_visible(prim: &usd::Prim) -> bool {
    let mut cur = Some(prim.clone());
    while let Some(p) = cur {
        if let Ok(Some(img)) = geom::Imageable::from_prim(p.clone()) {
            if let Ok(Some(geom::Visibility::Invisible)) = img.visibility() {
                return false;
            }
        }
        cur = p.parent();
    }
    true
}

fn read_light(prim: &usd::Prim, ty: &str, xf: &mut geom::XformCache, correction: &Mat4) -> Option<Light> {
    let api = lux::LightAPI::from_prim_unchecked(prim.clone());
    let intensity = api.intensity().ok().flatten().unwrap_or(1.0);
    let color = api.color().ok().flatten().map(|c| v3(&c)).unwrap_or(Vec3::ONE);
    let world = *correction * to_mat4(&xf.local_to_world_transform(prim).ok()?);
    match ty {
        "DistantLight" => {
            // A USD light shines down its local -Z.
            let dir = world.transform_vector3(-Vec3::Z).normalize_or(-Vec3::Y);
            Some(Light::Distant { direction: dir, color, intensity })
        }
        _ => Some(Light::Point { position: world.transform_point3(Vec3::ZERO), color, intensity }),
    }
}
