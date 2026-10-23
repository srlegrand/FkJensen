//! Manipulators, as in Maya: Q select, W move, E rotate, R scale.
//!
//! Move has an arrow per axis (X red, Y green, Z blue) and a square at the
//! centre that moves in the plane facing the eye. Rotate is a ring about
//! the vertical (models stand on the ground). Scale is a cube at the centre
//! and an arrow per axis, all scaling the same on every axis (so texture
//! tiles stay valid). Handles are a fixed size on screen.

use glam::Vec3;

use crate::raster::{self, Camera, Frame};

/// Length of an axis handle on screen, pixels.
pub const HANDLE_PX: f32 = 90.0;
/// How close the pointer must be to a handle to take it, pixels.
pub const PICK_PX: f32 = 8.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tool {
    Select,
    Move,
    Rotate,
    Scale,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Handle {
    Axis(usize),
    /// Move: the plane facing the eye. Scale: uniform.
    Centre,
    /// Rotate about the vertical.
    Ring,
}

pub const AXES: [Vec3; 3] = [Vec3::X, Vec3::Y, Vec3::Z];

pub fn axis_colour(k: usize) -> Vec3 {
    [Vec3::new(0.95, 0.25, 0.2), Vec3::new(0.35, 0.9, 0.3), Vec3::new(0.25, 0.45, 1.0)][k]
}

const HOT: Vec3 = Vec3::new(1.0, 0.95, 0.2);

/// World length that looks `HANDLE_PX` long at `p`.
pub fn handle_length(cam: &Camera, height: usize, p: Vec3) -> f32 {
    let dist = (p - cam.eye).dot(-cam.view.inverse().z_axis.truncate()).max(cam.near);
    HANDLE_PX * dist * 2.0 / (cam.proj.y_axis.y * height as f32)
}

fn screen(cam: &Camera, frame: &Frame, p: Vec3) -> Option<glam::Vec2> {
    raster::project(cam, frame.width, frame.height, p).map(|(x, y, _)| glam::Vec2::new(x, y))
}

/// Distance from `m` to the segment a-b on screen.
fn seg_dist(m: glam::Vec2, a: glam::Vec2, b: glam::Vec2) -> f32 {
    let ab = b - a;
    let t = ((m - a).dot(ab) / ab.length_squared().max(1e-9)).clamp(0.0, 1.0);
    m.distance(a + ab * t)
}

fn ring_points(pivot: Vec3, r: f32) -> Vec<Vec3> {
    (0..=64).map(|k| {
        let a = k as f32 / 64.0 * std::f32::consts::TAU;
        pivot + Vec3::new(a.cos(), 0.0, a.sin()) * r
    }).collect()
}

/// The handle under pixel `m`, if any.
pub fn pick(cam: &Camera, frame: &Frame, tool: Tool, pivot: Vec3, m: glam::Vec2, px: f32) -> Option<Handle> {
    let len = handle_length(cam, frame.height, pivot);
    let c = screen(cam, frame, pivot)?;
    let near = PICK_PX * px;
    match tool {
        Tool::Select => None,
        Tool::Move | Tool::Scale => {
            if (m - c).abs().max_element() <= 7.0 * px {
                return Some(Handle::Centre);
            }
            let mut best: Option<(f32, usize)> = None;
            for (k, a) in AXES.iter().enumerate() {
                let Some(e) = screen(cam, frame, pivot + *a * len) else { continue };
                let d = seg_dist(m, c, e);
                if d <= near && best.is_none_or(|(bd, _)| d < bd) {
                    best = Some((d, k));
                }
            }
            best.map(|(_, k)| Handle::Axis(k))
        }
        Tool::Rotate => {
            let pts: Vec<glam::Vec2> = ring_points(pivot, len).into_iter().filter_map(|p| screen(cam, frame, p)).collect();
            let d = pts.windows(2).map(|w| seg_dist(m, w[0], w[1])).fold(f32::INFINITY, f32::min);
            (d <= near).then_some(Handle::Ring)
        }
    }
}

/// Draw the tool's handles at `pivot`, `hot` highlighted.
pub fn draw(frame: &mut Frame, cam: &Camera, tool: Tool, pivot: Vec3, hot: Option<Handle>, px: f32) {
    let len = handle_length(cam, frame.height, pivot);
    match tool {
        Tool::Select => {}
        Tool::Move | Tool::Scale => {
            for (k, a) in AXES.iter().enumerate() {
                let col = if hot == Some(Handle::Axis(k)) { HOT } else { axis_colour(k) };
                let tip = pivot + *a * len;
                thick_line(frame, cam, pivot, tip, col);
                if tool == Tool::Move {
                    raster::draw_disc(frame, cam, tip, 4.5 * px, col, false);
                } else {
                    square(frame, cam, tip, 4.0 * px, col, false);
                }
            }
            let col = if hot == Some(Handle::Centre) { HOT } else { Vec3::splat(0.9) };
            square(frame, cam, pivot, 6.0 * px, col, tool == Tool::Move);
        }
        Tool::Rotate => {
            let col = if hot == Some(Handle::Ring) { HOT } else { axis_colour(1) };
            let pts = ring_points(pivot, len);
            for w in pts.windows(2) {
                thick_line(frame, cam, w[0], w[1], col);
            }
        }
    }
}

fn thick_line(frame: &mut Frame, cam: &Camera, a: Vec3, b: Vec3, col: Vec3) {
    raster::draw_line(frame, cam, a, b, col, false);
    // A second line one pixel over reads better on a busy picture.
    let (Some(pa), Some(pb)) = (raster::project(cam, frame.width, frame.height, a), raster::project(cam, frame.width, frame.height, b)) else { return };
    let d = glam::Vec2::new(pb.0 - pa.0, pb.1 - pa.1);
    let n = if d.x.abs() > d.y.abs() { (0, 1) } else { (1, 0) };
    let packed = raster::pack(col);
    let steps = d.abs().max_element().ceil().min(8192.0) as usize;
    for i in 0..=steps {
        let t = if steps == 0 { 0.0 } else { i as f32 / steps as f32 };
        let x = (pa.0 + d.x * t) as i32 + n.0;
        let y = (pa.1 + d.y * t) as i32 + n.1;
        if x >= 0 && y >= 0 && (x as usize) < frame.width && (y as usize) < frame.height {
            frame.color[y as usize * frame.width + x as usize] = packed;
        }
    }
}

fn square(frame: &mut Frame, cam: &Camera, p: Vec3, half: f32, col: Vec3, outline: bool) {
    let Some((x, y, _)) = raster::project(cam, frame.width, frame.height, p) else { return };
    let packed = raster::pack(col);
    let h = half.ceil() as i32;
    for dy in -h..=h {
        for dx in -h..=h {
            if outline && dx.abs() < h - 1 && dy.abs() < h - 1 {
                continue;
            }
            let (px, py) = (x as i32 + dx, y as i32 + dy);
            if px >= 0 && py >= 0 && (px as usize) < frame.width && (py as usize) < frame.height {
                frame.color[py as usize * frame.width + px as usize] = packed;
            }
        }
    }
}

/// Draw the twelve edges of a box.
pub fn draw_box(frame: &mut Frame, cam: &Camera, lo: Vec3, hi: Vec3, col: Vec3) {
    let c = |i: usize| Vec3::new(if i & 1 == 0 { lo.x } else { hi.x }, if i & 2 == 0 { lo.y } else { hi.y }, if i & 4 == 0 { lo.z } else { hi.z });
    for (a, b) in [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)] {
        raster::draw_line(frame, cam, c(a), c(b), col, false);
    }
}

/// Where the pointer ray at pixel (x, y) comes closest to the line through
/// `origin` along `axis`, as a distance along it.
pub fn along_axis(cam: &Camera, frame: &Frame, x: f32, y: f32, origin: Vec3, axis: Vec3) -> Option<f32> {
    let (o, d) = raster::pixel_ray(cam, frame.width, frame.height, x, y);
    // Closest points of two lines.
    let w = origin - o;
    let (a, b, c) = (axis.dot(axis), axis.dot(d), d.dot(d));
    let (e, f) = (axis.dot(w), d.dot(w));
    let den = a * c - b * b;
    if den.abs() < 1e-6 {
        return None;
    }
    Some((b * f - c * e) / den)
}

/// Where the pointer ray meets the plane through `p` with normal `n`.
pub fn on_plane(cam: &Camera, frame: &Frame, x: f32, y: f32, p: Vec3, n: Vec3) -> Option<Vec3> {
    let (o, d) = raster::pixel_ray(cam, frame.width, frame.height, x, y);
    let den = d.dot(n);
    if den.abs() < 1e-6 {
        return None;
    }
    let t = (p - o).dot(n) / den;
    (t > 0.0).then(|| o + d * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cam(w: usize, h: usize) -> Camera {
        let c = crate::camera::OrbitCamera { target: Vec3::ZERO, distance: 10.0, yaw: 0.6, pitch: 0.3, fov_y: 45f32.to_radians(), scene_radius: 1.0 };
        Camera { view: c.view(), proj: c.proj(w as f32 / h as f32), eye: c.eye(), near: c.near() }
    }

    /// Each handle is picked where it is drawn, and nothing is picked away
    /// from them.
    #[test]
    fn handles_are_picked_where_drawn() {
        let (w, h) = (640, 360);
        let c = cam(w, h);
        let f = Frame::new(w, h);
        let p = Vec3::new(0.5, 0.0, -0.3);
        let len = handle_length(&c, h, p);
        let at = |q: Vec3| {
            let (x, y, _) = raster::project(&c, w, h, q).unwrap();
            glam::Vec2::new(x, y)
        };
        for k in 0..3 {
            assert_eq!(pick(&c, &f, Tool::Move, p, at(p + AXES[k] * len * 0.7), 1.0), Some(Handle::Axis(k)));
        }
        assert_eq!(pick(&c, &f, Tool::Move, p, at(p), 1.0), Some(Handle::Centre));
        assert_eq!(pick(&c, &f, Tool::Rotate, p, at(p + Vec3::new(0.0, 0.0, len)), 1.0), Some(Handle::Ring));
        assert_eq!(pick(&c, &f, Tool::Move, p, at(p) + glam::Vec2::new(200.0, 150.0), 1.0), None);
        assert_eq!(pick(&c, &f, Tool::Select, p, at(p), 1.0), None);
        // About 90 pixels long on screen.
        let l = at(p).distance(at(p + Vec3::Y * len));
        assert!((70.0..=100.0).contains(&l), "{l}");
    }

    /// Dragging along an axis follows the pointer's closest point on it.
    #[test]
    fn axis_drag_finds_the_point_under_the_pointer() {
        let (w, h) = (640, 360);
        let c = cam(w, h);
        let f = Frame::new(w, h);
        let target = Vec3::new(1.7, 0.0, 0.0);
        let (x, y, _) = raster::project(&c, w, h, target).unwrap();
        let t = along_axis(&c, &f, x, y, Vec3::ZERO, Vec3::X).unwrap();
        assert!((t - 1.7).abs() < 0.02, "{t}");
        let q = on_plane(&c, &f, x, y, Vec3::ZERO, Vec3::Y).unwrap();
        assert!(q.distance(target) < 0.02, "{q}");
    }
}
