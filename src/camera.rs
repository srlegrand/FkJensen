//! An orbit camera that can also fly.
//!
//! Orbit: drag with the left button, pan with the middle button (or Shift +
//! left), dolly with the wheel or the right button. Fly: W A S D Q E move,
//! the right button looks. F frames the scene.

use glam::{Mat4, Vec3};

pub struct OrbitCamera {
    pub target: Vec3,
    pub distance: f32,
    /// Radians, around Y.
    pub yaw: f32,
    /// Radians, above the horizon.
    pub pitch: f32,
    pub fov_y: f32,
    pub scene_radius: f32,
}

impl OrbitCamera {
    pub fn new() -> Self {
        Self {
            target: Vec3::ZERO,
            distance: 5.0,
            yaw: 0.6,
            pitch: 0.35,
            fov_y: 45f32.to_radians(),
            scene_radius: 1.0,
        }
    }

    pub fn frame(&mut self, centre: Vec3, radius: f32) {
        self.target = centre;
        self.scene_radius = radius;
        self.distance = radius / (self.fov_y * 0.5).tan() * 0.9;
    }

    pub fn forward(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        // From eye towards target.
        -Vec3::new(cp * sy, sp, cp * cy)
    }

    pub fn right(&self) -> Vec3 {
        self.forward().cross(Vec3::Y).normalize_or(Vec3::X)
    }

    pub fn up(&self) -> Vec3 {
        self.right().cross(self.forward()).normalize_or(Vec3::Y)
    }

    pub fn eye(&self) -> Vec3 {
        self.target - self.forward() * self.distance
    }

    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx * 0.008;
        self.pitch = (self.pitch + dy * 0.008).clamp(-1.55, 1.55);
    }

    pub fn pan(&mut self, dx: f32, dy: f32, height_px: f32) {
        // One pixel at the target's distance.
        let per_px = 2.0 * self.distance * (self.fov_y * 0.5).tan() / height_px;
        self.target -= self.right() * dx * per_px;
        self.target += self.up() * dy * per_px;
    }

    pub fn dolly(&mut self, steps: f32) {
        self.distance *= (1.0 - steps * 0.1).max(0.1);
        self.distance = self.distance.max(self.scene_radius * 0.001);
    }

    pub fn look(&mut self, dx: f32, dy: f32) {
        // Turn the eye in place: keep the eye, move the target.
        let eye = self.eye();
        self.yaw -= dx * 0.004;
        self.pitch = (self.pitch - dy * 0.004).clamp(-1.55, 1.55);
        self.target = eye + self.forward() * self.distance;
    }

    pub fn fly(&mut self, local: Vec3, dt: f32, fast: bool) {
        // Scales with how far out the camera sits, so the same keys cross a
        // table or a valley.
        let speed = self.scene_radius.max(self.distance) * if fast { 2.0 } else { 0.6 };
        let d = (self.right() * local.x + Vec3::Y * local.y + self.forward() * local.z) * speed * dt;
        self.target += d;
    }

    pub fn near(&self) -> f32 {
        (self.scene_radius * 0.002).max(1e-4)
    }

    pub fn view(&self) -> Mat4 {
        glam::camera::rh::view::look_at_mat4(self.eye(), self.target, Vec3::Y)
    }

    pub fn proj(&self, aspect: f32) -> Mat4 {
        glam::camera::rh::proj::directx::perspective_infinite(self.fov_y, aspect, self.near())
    }
}
