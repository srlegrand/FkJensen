//! The light rig the viewer edits: point lights, which one is selected, and
//! the maths for picking and dragging them on screen.

use glam::Vec3;

use crate::raster::{self, Camera};
use crate::scene::{MAX_LUMENS, MIN_LUMENS, PointLight};

/// Most lights in a rig: the renderer keeps one bit per light per cluster.
pub const MAX_LIGHTS: usize = 64;

/// Tints the `C` key cycles through, before normalising.
pub const LIGHT_COLOURS: [Vec3; 6] = [
    Vec3::new(1.0, 1.0, 1.0),
    Vec3::new(1.0, 0.85, 0.65),
    Vec3::new(0.65, 0.8, 1.0),
    Vec3::new(1.0, 0.55, 0.4),
    Vec3::new(0.5, 1.0, 0.6),
    Vec3::new(0.9, 0.6, 1.0),
];

/// Scale a colour so its luminance (Rec. 709 weights) is 1, so the colour
/// only tints and the lumens alone set the strength.
pub fn normalise_colour(c: Vec3) -> Vec3 {
    let y = c.dot(Vec3::new(0.2126, 0.7152, 0.0722));
    if y > 0.0 { c / y } else { Vec3::ONE }
}

/// How far from a light's disc centre, in pixels, a click still grabs it.
pub const PICK_RADIUS_PX: f32 = 14.0;

pub struct LightRig {
    pub lights: Vec<PointLight>,
    pub selected: usize,
    pub show: bool,
}

/// The surface a dragged light slides on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DragPlane {
    /// Facing the camera, through the light: the light stays under the
    /// cursor at the same depth.
    Screen,
    /// Horizontal, at the light's height: the light moves over the set
    /// without going up or down.
    Ground,
}

impl LightRig {
    pub fn new(lights: Vec<PointLight>) -> Self {
        Self { lights, selected: 0, show: true }
    }

    pub fn selected_mut(&mut self) -> Option<&mut PointLight> {
        self.lights.get_mut(self.selected)
    }

    pub fn selected(&self) -> Option<&PointLight> {
        self.lights.get(self.selected)
    }

    /// The light whose disc is under pixel (x, y), nearest the camera when
    /// discs overlap. `px` is the size of a screen pixel in frame pixels
    /// (below 1 when the frame is rendered smaller than the window).
    pub fn pick(&self, cam: &Camera, width: usize, height: usize, x: f32, y: f32, px: f32) -> Option<usize> {
        let radius = PICK_RADIUS_PX * px;
        let mut best: Option<(usize, f32)> = None;
        for (i, l) in self.lights.iter().enumerate() {
            let Some((sx, sy, iw)) = raster::project(cam, width, height, l.position) else { continue };
            let d2 = (sx - x) * (sx - x) + (sy - y) * (sy - y);
            if d2 <= radius * radius && best.is_none_or(|(_, biw)| iw > biw) {
                best = Some((i, iw));
            }
        }
        best.map(|(i, _)| i)
    }

    /// Add a light; false when the rig is full.
    pub fn add(&mut self, position: Vec3, lumens: f32) -> bool {
        if self.lights.len() >= MAX_LIGHTS {
            return false;
        }
        let color = normalise_colour(LIGHT_COLOURS[self.lights.len() % LIGHT_COLOURS.len()]);
        self.lights.push(PointLight { position, color, lumens: lumens.clamp(MIN_LUMENS, MAX_LUMENS) });
        self.selected = self.lights.len() - 1;
        true
    }

    /// Multiply the selected light's lumens, kept between `MIN_LUMENS` and
    /// `MAX_LUMENS`.
    pub fn scale_selected(&mut self, k: f32) {
        if let Some(l) = self.selected_mut() {
            l.lumens = (l.lumens * k).clamp(MIN_LUMENS, MAX_LUMENS);
        }
    }

    pub fn remove_selected(&mut self) {
        if self.selected < self.lights.len() {
            self.lights.remove(self.selected);
            self.selected = self.selected.min(self.lights.len().saturating_sub(1));
        }
    }

    pub fn select_next(&mut self) {
        if !self.lights.is_empty() {
            self.selected = (self.selected + 1) % self.lights.len();
        }
    }

    pub fn cycle_colour(&mut self) {
        if let Some(l) = self.selected_mut() {
            let i = LIGHT_COLOURS.iter().position(|k| (normalise_colour(*k) - l.color).length() < 1e-3).map(|i| (i + 1) % LIGHT_COLOURS.len()).unwrap_or(0);
            l.color = normalise_colour(LIGHT_COLOURS[i]);
        }
    }
}

/// Where a light being dragged goes when the cursor is at pixel (x, y).
/// `anchor` is the light's position when the drag started; the light slides
/// on a plane through it. None when the cursor ray misses the plane (ground
/// plane seen edge-on, or behind the camera).
pub fn drag_target(cam: &Camera, width: usize, height: usize, x: f32, y: f32, anchor: Vec3, plane: DragPlane) -> Option<Vec3> {
    let (o, d) = raster::pixel_ray(cam, width, height, x, y);
    let n = match plane {
        DragPlane::Screen => (anchor - cam.eye).normalize_or(Vec3::NEG_Z),
        DragPlane::Ground => Vec3::Y,
    };
    let denom = d.dot(n);
    if denom.abs() < 1e-4 {
        return None;
    }
    let t = (anchor - o).dot(n) / denom;
    if t <= 0.0 {
        return None;
    }
    Some(o + d * t)
}

/// Move a point towards or away from the eye along the line through both,
/// by `steps` (positive = away), each step 10% of its distance.
pub fn push_pull(eye: Vec3, p: Vec3, steps: f32) -> Vec3 {
    let d = p - eye;
    eye + d * (1.0 + steps * 0.1).max(0.05)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cam(w: usize, h: usize) -> Camera {
        let c = crate::camera::OrbitCamera { target: Vec3::ZERO, distance: 4.0, yaw: 0.5, pitch: 0.4, fov_y: 45f32.to_radians(), scene_radius: 1.0 };
        Camera { view: c.view(), proj: c.proj(w as f32 / h as f32), eye: c.eye(), near: c.near() }
    }

    #[test]
    fn click_on_a_disc_picks_that_light_and_nearest_wins() {
        let (w, h) = (640, 360);
        let cam = cam(w, h);
        let mut rig = LightRig::new(Vec::new());
        rig.add(Vec3::new(0.5, 0.5, 0.0), 1.0);
        rig.add(Vec3::new(-0.5, 0.2, 0.3), 1.0);
        let (x, y, _) = raster::project(&cam, w, h, rig.lights[1].position).unwrap();
        assert_eq!(rig.pick(&cam, w, h, x + 5.0, y - 5.0, 1.0), Some(1));
        assert_eq!(rig.pick(&cam, w, h, x + 40.0, y, 1.0), None);
        // A second light on the same pixel ray, further away: the nearer one wins.
        let far = cam.eye + (rig.lights[1].position - cam.eye) * 2.0;
        rig.add(far, 1.0);
        assert_eq!(rig.pick(&cam, w, h, x, y, 1.0), Some(1));
    }

    #[test]
    fn screen_drag_keeps_the_light_under_the_cursor_at_its_depth() {
        let (w, h) = (640, 360);
        let cam = cam(w, h);
        let anchor = Vec3::new(0.3, 0.6, -0.2);
        let (_, _, iw0) = raster::project(&cam, w, h, anchor).unwrap();
        for (x, y) in [(100.0, 50.0), (600.0, 300.0), (320.0, 180.0)] {
            let p = drag_target(&cam, w, h, x, y, anchor, DragPlane::Screen).unwrap();
            let (sx, sy, _) = raster::project(&cam, w, h, p).unwrap();
            assert!((sx - x).abs() < 0.01 && (sy - y).abs() < 0.01, "({sx},{sy}) vs ({x},{y})");
            // Same distance along the view axis as the anchor, about.
            let n = (anchor - cam.eye).normalize();
            assert!(((p - cam.eye).dot(n) - (anchor - cam.eye).dot(n)).abs() < 1e-3);
        }
        let _ = iw0;
    }

    #[test]
    fn ground_drag_keeps_height() {
        let (w, h) = (640, 360);
        let cam = cam(w, h);
        let anchor = Vec3::new(0.2, 0.4, 0.1);
        let p = drag_target(&cam, w, h, 200.0, 250.0, anchor, DragPlane::Ground).unwrap();
        assert!((p.y - anchor.y).abs() < 1e-5);
        let (sx, sy, _) = raster::project(&cam, w, h, p).unwrap();
        assert!((sx - 200.0).abs() < 0.01 && (sy - 250.0).abs() < 0.01);
    }

    #[test]
    fn push_pull_moves_along_the_view_line() {
        let eye = Vec3::new(0.0, 1.0, 5.0);
        let p = Vec3::new(1.0, 0.0, 0.0);
        let q = push_pull(eye, p, 2.0);
        assert!(((q - eye).normalize() - (p - eye).normalize()).length() < 1e-5);
        assert!((q - eye).length() > (p - eye).length());
        assert!((push_pull(eye, p, -2.0) - eye).length() < (p - eye).length());
    }

    #[test]
    fn lumens_stay_within_real_lights() {
        let mut rig = LightRig::new(Vec::new());
        rig.add(Vec3::ZERO, 800.0);
        for _ in 0..200 {
            rig.scale_selected(1.25);
        }
        assert_eq!(rig.lights[0].lumens, MAX_LUMENS);
        for _ in 0..400 {
            rig.scale_selected(0.8);
        }
        assert_eq!(rig.lights[0].lumens, MIN_LUMENS);
        rig.add(Vec3::ZERO, 1e12);
        assert_eq!(rig.lights[1].lumens, MAX_LUMENS);
    }

    #[test]
    fn rig_is_capped() {
        let mut rig = LightRig::new(Vec::new());
        for _ in 0..MAX_LIGHTS {
            assert!(rig.add(Vec3::ZERO, 100.0));
        }
        assert!(!rig.add(Vec3::ZERO, 100.0));
    }

    #[test]
    fn colours_are_normalised_to_unit_luminance() {
        for c in LIGHT_COLOURS {
            let n = normalise_colour(c);
            assert!((n.dot(Vec3::new(0.2126, 0.7152, 0.0722)) - 1.0).abs() < 1e-5);
        }
    }
}
