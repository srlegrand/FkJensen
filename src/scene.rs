//! The flat scene the rasteriser reads: world-space triangles in SoA arrays,
//! and the lights.
//!
//! Nothing here knows about USD. The loader fills it; the renderer reads it.

use glam::Vec3;

/// One triangle soup for the whole scene. Every mesh is transformed to world
/// space at load, so the per-frame work is one matrix per vertex, not one
/// per prim.
#[derive(Default)]
pub struct Scene {
    /// World-space positions, one per vertex.
    pub positions: Vec<Vec3>,
    /// World-space unit normals, one per vertex.
    pub normals: Vec<Vec3>,
    /// Linear RGB, one per vertex.
    pub colors: Vec<Vec3>,
    /// Three indices per triangle.
    pub indices: Vec<u32>,
    /// Bounds of all positions.
    pub bounds_min: Vec3,
    pub bounds_max: Vec3,
    /// Lights read from the file, if any (the user's "master lights" are
    /// kept by the app, not here).
    pub lights: Vec<Light>,
    /// How many prims were read and how many skipped, for the title bar.
    pub prims_read: usize,
    pub prims_skipped: usize,
}

/// A light the renderer understands. Point lights are what the user places;
/// distant lights come from files.
#[derive(Clone, Copy, Debug)]
pub enum Light {
    Point {
        position: Vec3,
        color: Vec3,
        /// Radiant intensity scale. Falloff is 1/d².
        intensity: f32,
    },
    Distant {
        /// Direction the light travels (from the light towards the scene).
        direction: Vec3,
        color: Vec3,
        intensity: f32,
    },
}

impl Scene {
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    pub fn centre(&self) -> Vec3 {
        (self.bounds_min + self.bounds_max) * 0.5
    }

    pub fn radius(&self) -> f32 {
        ((self.bounds_max - self.bounds_min).length() * 0.5).max(1e-3)
    }

    /// Append a mesh already in world space. `normals` may be empty, in which
    /// case smooth normals are computed from the faces.
    pub fn push_mesh(&mut self, positions: &[Vec3], normals: &[Vec3], colors: &[Vec3], tri_indices: &[u32]) {
        let base = self.positions.len() as u32;
        let n = positions.len();
        self.positions.extend_from_slice(positions);

        if normals.len() == n {
            self.normals.extend_from_slice(normals);
        } else {
            let mut acc = vec![Vec3::ZERO; n];
            for t in tri_indices.chunks_exact(3) {
                let (a, b, c) = (t[0] as usize, t[1] as usize, t[2] as usize);
                let fn_ = (positions[b] - positions[a]).cross(positions[c] - positions[a]);
                acc[a] += fn_;
                acc[b] += fn_;
                acc[c] += fn_;
            }
            self.normals.extend(acc.into_iter().map(|v| v.normalize_or(Vec3::Y)));
        }

        if colors.len() == n {
            self.colors.extend_from_slice(colors);
        } else {
            let c = colors.first().copied().unwrap_or(Vec3::splat(0.7));
            self.colors.extend(std::iter::repeat_n(c, n));
        }

        self.indices.extend(tri_indices.iter().map(|i| i + base));

        for p in positions {
            self.bounds_min = self.bounds_min.min(*p);
            self.bounds_max = self.bounds_max.max(*p);
        }
    }

    pub fn new() -> Self {
        Self {
            bounds_min: Vec3::splat(f32::INFINITY),
            bounds_max: Vec3::splat(f32::NEG_INFINITY),
            ..Default::default()
        }
    }
}
