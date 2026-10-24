//! fkjensen: a CPU-only 3D viewport. No GPU API is linked; the frame is
//! rasterised on the cores and copied to the window through shared memory.
//!
//! Usage:
//!   fkjensen                           open the built-in set: the T-rex and
//!                                      the car on a ground plane, mountains
//!   fkjensen scene.fkj                 open an .fkj scene
//!   fkjensen --png out.png             render one frame to a file, no window
//!   fkjensen --size 1920x1080 --png out.png
//!   fkjensen --light x,y,z[,lumens]    replace the scene's lights (repeatable)
//!   fkjensen --time 12:00              time of day: sun, sky and exposure (opens at dusk)
//!   fkjensen --sun B                   sun brightness, 0 to 1000 (1000: white at full)
//!   fkjensen --dir AZ,EL,LUX           add a directional light: azimuth and elevation in degrees, illuminance in lux (up to 4)
//!   fkjensen --light-contrast C        0 flat overcast, 0.5 default, 1 overhead only
//!   fkjensen --sky-contrast C          the visible sky's gradient and sun glow, 0 to 1
//!   fkjensen --sky LUX                 sky illuminance, 0 for no sky
//!   fkjensen --exposure E              display value per cd/m²
//!   fkjensen --view lights             show how many lights each pixel evaluates
//!   fkjensen --no-reach                evaluate every light everywhere (to compare)
//!   fkjensen --no-occlusion            draw what is hidden behind what was seen (to compare)
//!   fkjensen --no-adaptive             shade every grid point (to compare)
//!   fkjensen --no-sky-occlusion        the sky not taken by objects' own shapes (to compare)
//!   fkjensen --relight-on-move         light everything again when an object moves (to compare)
//!   fkjensen --png out.png --bench-move  then move the first model five times, timing each frame
//!   fkjensen --png out.png --bench-dir   then turn the first directional light five times, timing each frame
//!   fkjensen --close                   start on the close framing of the subject
//!   fkjensen --look ex,ey,ez,tx,ty,tz  camera at e looking at t, in metres
//!   fkjensen --no-shadows              point lights without shadow maps
//!   fkjensen --no-bounce               no bounce light
//!   fkjensen --hardware                print what this machine is and what FkJensen chooses for it
//!   fkjensen --fb                      draw to /dev/fb0 even when a display server is running (Linux)
//!   fkjensen --no-auto                 do not measure the machine; use the fixed defaults
//!   fkjensen --scale F                 render the window at F of its size (0.0625 to 1)
//!   fkjensen --highres / --shadows / --bounce   force on what the machine's plan left off
//!   fkjensen --no-highres              the low-resolution T-rex (28 K triangles, not 10.8 M)
//!   fkjensen --shadow-res N            shadow map texels per cube face edge (512)
//!   fkjensen --png out.png --bench-drag  then move the first light five times, timing each frame
//!   fkjensen --png out.png --bench-orbit then turn the camera five times, timing each frame
//!   fkjensen --editor [scene.fkj]      the editor: outliner, attribute editor, timeline, importer (64-bit)
//!   fkjensen --editor --png out.png [--import model.glb]  the editor drawn to a file
//!
//! Controls are listed in README.md and printed at start.

mod bounce;
mod camera;
#[cfg(target_pointer_width = "64")]
mod debug;
#[cfg(target_os = "linux")]
mod fbdev;
mod fkj;
mod hardware;
mod lights;
mod lod;
mod manip;
mod raster;
mod reyes;
mod samples;
mod transfer;
mod scene;
mod shadow;
mod simd;
mod sun;
mod texture;
mod world;
#[cfg(target_pointer_width = "64")]
mod editor;
// The model readers of tools/fkj-import, for the editor's importer window
// (parts only the command-line tool uses go unused here).
#[cfg(target_pointer_width = "64")]
#[allow(dead_code)]
#[path = "../tools/fkj-import/src/bake.rs"]
mod bake;
#[cfg(target_pointer_width = "64")]
#[allow(dead_code)]
#[path = "../tools/fkj-import/src/fbx.rs"]
mod fbx;
#[cfg(target_pointer_width = "64")]
#[allow(dead_code)]
#[path = "../tools/fkj-import/src/gltf_in.rs"]
mod gltf_in;
#[cfg(target_pointer_width = "64")]
#[allow(dead_code)]
#[path = "../tools/fkj-import/src/model.rs"]
mod model;
#[cfg(target_pointer_width = "64")]
#[allow(dead_code)]
#[path = "../tools/fkj-import/src/obj.rs"]
mod obj;

use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::Instant;

use glam::Vec3;
use rayon::prelude::*;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use camera::OrbitCamera;
use lights::{DragPlane, LightRig};
use raster::{Camera, Frame, RenderSettings, Sky, View};
use scene::{PointLight, Scene};

/// The sky the built-in set opens with, in lux: dusk, dark enough for lamps
/// to show.
const DEFAULT_SKY_LUX: f32 = 30.0;


const CONTROLS: &str = "\
Controls
  Selecting and moving (as in Maya)
  click                  select a model or a light; click empty space to clear
  Q / W / E / R          select / move / rotate / scale manipulator
  drag an arrow          move along that axis (X red, Y green, Z blue)
  drag the centre        move facing the camera (move), scale (scale)
  drag a ring            turn about that axis (rotate; X red, Y green, Z blue)
  F                      frame the selection (the subject when none)
  A                      frame the subject from afar

  Lights: point lights in lumens, 1 lm to 2,000,000 lm, the sun and the sky
  left drag on a light   move it, facing the camera (stays at its depth)
    + Ctrl               move it level, over the set (keeps its height)
    + wheel              push it away / pull it closer
  arrows                 nudge the selected light, relative to the view
  Page Up / Page Down    raise / lower the selected light (Shift: 5x)
  1 (hold) + click       add a light above the surface under the cursor
  2 (hold) + click       put the selected light above that surface
  Tab                    select next light     Delete   remove selected light
  + / -                  lumens x1.25          C        cycle colour
  L                      show / hide lights    Ctrl+R   reset to the scene's lights
  T / Shift+T            time of day later / earlier (15 min): sun, sky, exposure
  Y / Shift+Y            sun brighter / dimmer (0 to 1000, steps of 50)
  J / Shift+J            lighting contrast up / down: flat overcast to overhead
  U / Shift+U            visible sky contrast up / down (the backdrop only)
  K                      sky on / off          , / .    sky dimmer / brighter (x2)
  H                      light count view: how many lights each pixel evaluates
  X                      light reach off / on (to compare cost and picture)
  M                      shadow maps on / off
  I                      bounce light on / off
  Z / Shift+Z            render resolution finer / coarser (x1.25)
  V / Shift+V            next / previous visual debug view (--visual-debug)

  Camera (as in Maya)
  Alt + left drag        tumble                Alt + middle drag / middle drag   track
  Alt + right drag       dolly                 wheel                             dolly
  right drag             look; with W A S D Q E held, fly (shift: faster)

  View
  [ / ]                  exposure              0    exposure from the sky
  B                      back-face culling
  G                      wireframe             P    save frame.png
  Esc                    quit
  drop an .fkj file on the window to open it";

/// The visual debug view shown, when `--visual-debug` is on. Built on
/// 64-bit targets only.
#[cfg(target_pointer_width = "64")]
type VisualDebug = Option<debug::DebugView>;
#[cfg(not(target_pointer_width = "64"))]
type VisualDebug = Option<()>;

/// What is selected.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Selection {
    Nothing,
    Object(usize),
    Light(usize),
}

/// A manipulator handle being dragged, and how things were when it was
/// taken.
struct ManipDrag {
    target: Selection,
    tool: manip::Tool,
    handle: manip::Handle,
    pivot: Vec3,
    /// Euler angles, radians.
    rotation: Vec3,
    scale: f32,
    /// Where the pointer took the handle: along the axis, on the plane, the
    /// angle on the ground plane, or the pointer's x.
    grab: Vec3,
    /// The handle's length in the world when taken.
    len: f32,
}

/// A light being dragged with the mouse.
struct Drag {
    light: usize,
    /// Where the light was when the plane was last set.
    anchor: Vec3,
    plane: DragPlane,
}

struct App {
    window: Option<Rc<Window>>,
    context: Option<softbuffer::Context<Rc<Window>>>,
    surface: Option<softbuffer::Surface<Rc<Window>, Rc<Window>>>,
    surface_sized: bool,
    frame: Frame,
    scene: Scene,
    scene_name: String,
    cam: OrbitCamera,
    rig: LightRig,
    settings: RenderSettings,
    /// The frame is rendered at this fraction of the window's size and
    /// scaled up when presented.
    render_scale: f32,
    drag: Option<Drag>,
    /// The manipulator shown (Q W E R) and what it acts on.
    tool: manip::Tool,
    selection: Selection,
    manip: Option<ManipDrag>,
    /// The handle under the pointer.
    hot: Option<manip::Handle>,
    alt: bool,
    /// Set by Esc.
    quit: bool,
    /// Size of what the frame is shown on (window or framebuffer).
    output: (usize, usize),
    /// When set, the render scale follows the output size so that a frame
    /// holds about this many pixels (from the machine's plan).
    /// Shading points a window frame may take, from the machine's plan:
    /// the micropolygon spacing and render scale follow it as the window
    /// is resized. None once either is set by hand.
    point_budget: Option<f64>,
    /// Shading, visibility and shadow maps kept between frames.
    renderer: reyes::Renderer,
    /// Which framing `F` last chose.
    wide: bool,
    // input state
    mouse: (f32, f32),
    /// The cursor in window pixels, for drag deltas.
    raw_mouse: (f32, f32),
    last_mouse: (f32, f32),
    buttons: [bool; 3],
    shift: bool,
    ctrl: bool,
    keys: std::collections::HashSet<KeyCode>,
    last_frame: Instant,
    fps_accum: (f32, u32),
    title_fps: f32,
    /// Time of day in hours: sets the sun's direction and colour, the sky
    /// and the exposure (`daylight`).
    hours: f32,
    /// `--visual-debug`: the view shown (Some(Final) for the picture).
    #[cfg_attr(not(target_pointer_width = "64"), allow(dead_code))]
    visual: VisualDebug,
}

impl App {
    fn new(scene: Scene, scene_name: String) -> Self {
        let mut cam = OrbitCamera::new();
        frame_camera(&mut cam, &scene, true);
        let rig = LightRig::new(default_rig(&scene));
        Self {
            window: None,
            context: None,
            surface: None,
            surface_sized: false,
            frame: Frame::new(1, 1),
            // In the window, bounce light is gathered for 30 ms a frame and
            // catches up over the following frames.
            settings: RenderSettings { bounce_budget_ms: Some(30.0), ..RenderSettings::new(Sky::new(DEFAULT_SKY_LUX, true)) },
            scene,
            scene_name,
            cam,
            rig,
            drag: None,
            tool: manip::Tool::Move,
            selection: Selection::Nothing,
            manip: None,
            hot: None,
            alt: false,
            quit: false,
            output: (0, 0),
            point_budget: None,
            renderer: reyes::Renderer::new(),
            wide: true,
            mouse: (0.0, 0.0),
            raw_mouse: (0.0, 0.0),
            last_mouse: (0.0, 0.0),
            render_scale: 1.0,
            buttons: [false; 3],
            shift: false,
            ctrl: false,
            keys: Default::default(),
            last_frame: Instant::now(),
            fps_accum: (0.0, 0),
            title_fps: 0.0,
            visual: None,
            hours: sun::hour_for_dusk_lux(DEFAULT_SKY_LUX),
        }
    }

    fn open(&mut self, path: PathBuf) -> Result<(), String> {
        match load_file(&path) {
            Ok(mut scene) => {
                prepare_opened(&mut scene, &path);
                self.scene = scene;
                self.scene_name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                self.wide = true;
                frame_camera(&mut self.cam, &self.scene, self.wide);
                self.rig = LightRig::new(default_rig(&self.scene));
                self.drag = None;
                self.manip = None;
                self.selection = Selection::Nothing;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    /// Set the sun, the sky and the exposure from `hours`, keeping the
    /// sun's brightness and both contrasts.
    fn apply_daylight(&mut self) {
        let d = sun::daylight(self.hours);
        let s = &mut self.settings;
        s.sun.dir = d.sun_dir;
        s.sun.color = d.sun_color;
        s.sky.lux = d.sky_lux;
        s.sky.tint = d.sky_tint;
        s.backdrop.zenith = d.zenith;
        s.backdrop.horizon = d.horizon;
        s.exposure = raster::default_exposure(d.sky_lux);
    }

    fn render_camera(&self) -> Camera {
        let aspect = self.frame.width as f32 / self.frame.height.max(1) as f32;
        Camera { view: self.cam.view(), proj: self.cam.proj(aspect), eye: self.cam.eye(), near: self.cam.near() }
    }

    fn render(&mut self) {
        let cam = self.render_camera();
        self.renderer.render(&self.scene, &cam, &self.rig.lights, &self.settings, &mut self.frame);
        #[cfg(target_pointer_width = "64")]
        if let Some(v) = self.visual {
            let ctx = debug::Ctx {
                scene: &self.scene,
                cam: &cam,
                lights: &self.rig.lights,
                selected: Some(self.rig.selected),
                settings: &self.settings,
                peek: self.renderer.peek(),
                // No pointer yet (a still): the middle of the frame.
                cursor: if self.mouse == (0.0, 0.0) { None } else { Some(self.mouse) },
            };
            debug::draw(v, &ctx, &mut self.frame);
        }
        if self.rig.show {
            let k = self.render_scale.max(0.25);
            let floor = self.floor();
            for (i, l) in self.rig.lights.iter().enumerate() {
                let sel = i == self.rig.selected;
                if sel {
                    // A dashed line down to the floor and a small mark there,
                    // so the light's height reads at a glance.
                    let foot = Vec3::new(l.position.x, floor, l.position.z);
                    let gold = Vec3::new(1.0, 0.9, 0.2) * 0.7;
                    raster::draw_line(&mut self.frame, &cam, l.position, foot, gold, true);
                    raster::draw_disc(&mut self.frame, &cam, foot, 2.5 * k, gold, false);
                }
                let disc = (l.color / l.color.max_element()).min(Vec3::ONE);
                raster::draw_disc(&mut self.frame, &cam, l.position, if sel { 7.0 } else { 4.0 } * k, disc, false);
                if sel {
                    raster::draw_disc(&mut self.frame, &cam, l.position, 11.0 * k, Vec3::new(1.0, 0.9, 0.2), true);
                }
            }
        }
        // The selection and its manipulator.
        let k = self.render_scale.max(0.25);
        if let Selection::Object(i) = self.selection {
            // Each box turns with its object; a group shows those under it.
            for j in std::iter::once(i).chain(self.scene.descendants(i)) {
                if !self.scene.objects[j].clusters.is_empty() {
                    manip::draw_corners(&mut self.frame, &cam, &self.scene.objects[j].corners(), Vec3::new(0.35, 0.95, 0.55));
                }
            }
        }
        if let Some((pivot, tool)) = self.gizmo() {
            let hot = self.manip.as_ref().map(|m| m.handle).or(self.hot);
            manip::draw(&mut self.frame, &cam, tool, pivot, hot, k);
        }
    }

    /// Where the manipulator is and which one: lights only move.
    fn gizmo(&self) -> Option<(Vec3, manip::Tool)> {
        match self.selection {
            Selection::Object(i) => self.scene.objects.get(i).map(|o| (o.pivot, self.tool)),
            Selection::Light(i) if self.tool == manip::Tool::Move && self.rig.show => self.rig.lights.get(i).map(|l| (l.position, self.tool)),
            _ => None,
        }
        .filter(|(_, t)| *t != manip::Tool::Select)
    }

    /// The handle under the pointer, if any.
    fn pick_handle(&self) -> Option<manip::Handle> {
        let (pivot, tool) = self.gizmo()?;
        let cam = self.render_camera();
        manip::pick(&cam, &self.frame, tool, pivot, glam::Vec2::new(self.mouse.0, self.mouse.1), self.render_scale.max(0.25))
    }

    fn start_manip(&mut self, handle: manip::Handle) {
        let Some((pivot, tool)) = self.gizmo() else { return };
        let cam = self.render_camera();
        let len = manip::handle_length(&cam, self.frame.height, pivot);
        let (x, y) = self.mouse;
        let (rotation, scale) = match self.selection {
            Selection::Object(i) => (self.scene.objects[i].rotation, self.scene.objects[i].scale),
            _ => (Vec3::ZERO, 1.0),
        };
        let grab = match (tool, handle) {
            (_, manip::Handle::Axis(k)) => Vec3::splat(manip::along_axis(&cam, &self.frame, x, y, pivot, manip::AXES[k]).unwrap_or(0.0)),
            (manip::Tool::Move, manip::Handle::Centre) => manip::on_plane(&cam, &self.frame, x, y, pivot, (pivot - cam.eye).normalize_or(Vec3::Z)).unwrap_or(pivot),
            (_, manip::Handle::Ring(k)) => {
                // x: the angle taken; y: 1 when followed in the ring's plane.
                let flat = manip::ring_faces_eye(&cam, pivot, k);
                Vec3::new(manip::ring_angle(&cam, &self.frame, pivot, k, glam::Vec2::new(x, y), flat).unwrap_or(0.0), flat as u8 as f32, 0.0)
            }
            _ => Vec3::splat(x),
        };
        self.manip = Some(ManipDrag { target: self.selection, tool, handle, pivot, rotation, scale, grab, len });
    }

    fn update_manip(&mut self) {
        let Some(m) = &self.manip else { return };
        let cam = self.render_camera();
        let (x, y) = self.mouse;
        let (mut pivot, mut rotation, mut scale) = (m.pivot, m.rotation, m.scale);
        match (m.tool, m.handle) {
            (manip::Tool::Move, manip::Handle::Axis(k)) => {
                if let Some(t) = manip::along_axis(&cam, &self.frame, x, y, m.pivot, manip::AXES[k]) {
                    pivot = m.pivot + manip::AXES[k] * (t - m.grab.x);
                }
            }
            (manip::Tool::Move, manip::Handle::Centre) => {
                if let Some(q) = manip::on_plane(&cam, &self.frame, x, y, m.pivot, (m.pivot - cam.eye).normalize_or(Vec3::Z)) {
                    pivot = m.pivot + (q - m.grab);
                }
            }
            (manip::Tool::Rotate, manip::Handle::Ring(k)) => {
                if let Some(a) = manip::ring_angle(&cam, &self.frame, m.pivot, k, glam::Vec2::new(x, y), m.grab.y > 0.5) {
                    let turn = a - m.grab.x;
                    let q = glam::Quat::from_axis_angle(manip::AXES[k], turn) * scene::euler_quat(m.rotation);
                    rotation = scene::quat_euler(q);
                }
            }
            (manip::Tool::Scale, manip::Handle::Axis(k)) => {
                if let Some(t) = manip::along_axis(&cam, &self.frame, x, y, m.pivot, manip::AXES[k]) {
                    scale = m.scale * (1.0 + (t - m.grab.x) / m.len).max(0.01);
                }
            }
            (manip::Tool::Scale, _) => scale = m.scale * ((x - m.grab.x) / (120.0 * self.render_scale.max(0.25))).exp(),
            _ => {}
        }
        match m.target {
            Selection::Object(i) => self.move_object(i, pivot, rotation, scale),
            Selection::Light(i) => {
                if let Some(l) = self.rig.lights.get_mut(i) {
                    l.position = pivot;
                }
            }
            Selection::Nothing => {}
        }
    }

    /// Place object `i` (and what is under it), telling the renderer where
    /// it was and where it is, so only what that changes is redone.
    fn move_object(&mut self, i: usize, pivot: Vec3, rotation: Vec3, scale: f32) {
        let from = self.scene.object_bounds(i);
        self.scene.set_object(i, pivot, rotation, scale);
        let to = self.scene.object_bounds(i);
        for r in self.scene.moving_clusters(i) {
            self.renderer.moved(r, from, to, false);
        }
    }

    /// Object `i` stopped moving: bounce light is made again.
    fn object_settled(&mut self, i: usize) {
        let b = self.scene.object_bounds(i);
        for r in self.scene.moving_clusters(i) {
            self.renderer.moved(r, b, b, true);
        }
    }

    /// The pointer let go of a manipulator.
    fn end_manip(&mut self) {
        if let Some(m) = self.manip.take() {
            if let Selection::Object(i) = m.target {
                self.object_settled(i);
            }
        }
    }

    /// Select what is under the pointer: a light, else a model, else
    /// nothing. Returns the light, for dragging.
    fn select_at_pointer(&mut self) -> Option<usize> {
        if self.rig.show {
            let cam = self.render_camera();
            if let Some(i) = self.rig.pick(&cam, self.frame.width, self.frame.height, self.mouse.0, self.mouse.1, self.render_scale) {
                self.rig.selected = i;
                self.selection = Selection::Light(i);
                return Some(i);
            }
        }
        let (x, y) = (self.mouse.0 as usize, self.mouse.1 as usize);
        let id = if x < self.frame.width && y < self.frame.height { self.frame.tri_id[y * self.frame.width + x] } else { raster::NO_TRI };
        self.selection = if id == raster::NO_TRI {
            Selection::Nothing
        } else {
            let c = self.frame.setups[id as usize].cluster as usize;
            self.scene.object_of(c).map(Selection::Object).unwrap_or(Selection::Nothing)
        };
        None
    }

    /// Frame the selection, or the subject when nothing is selected.
    fn frame_selection(&mut self) {
        match self.selection {
            Selection::Object(i) => {
                let (lo, hi) = self.scene.object_bounds(i);
                self.cam.frame((lo + hi) * 0.5, ((hi - lo).length() * 0.5).max(1e-3));
            }
            Selection::Light(i) => {
                if let Some(l) = self.rig.lights.get(i) {
                    let r = self.scene.focus_radius() * 0.15;
                    self.cam.frame(l.position, r);
                }
            }
            // The whole scene runs to the mountains: the subject instead.
            Selection::Nothing => self.cam.frame(self.scene.focus_centre(), self.scene.focus_radius()),
        }
    }

    fn present(&mut self) {
        let Some(w) = &self.window else { return };
        let size = w.inner_size();
        let (ww, wh) = (size.width.max(1) as usize, size.height.max(1) as usize);
        let Some(surface) = self.surface.as_mut() else { return };
        let Ok(mut buf) = surface.buffer_mut() else { return };
        let (fw, fh) = (self.frame.width, self.frame.height);
        if fw == ww && fh == wh {
            let n = buf.len().min(self.frame.color.len());
            buf[..n].copy_from_slice(&self.frame.color[..n]);
        } else if buf.len() >= ww * wh {
            // Scale the frame up to the window, nearest pixel.
            let xs: Vec<usize> = (0..ww).map(|x| (x * fw / ww).min(fw - 1)).collect();
            let src = &self.frame.color;
            buf[..ww * wh].par_chunks_mut(ww).enumerate().for_each(|(y, row)| {
                let sy = (y * fh / wh).min(fh - 1) * fw;
                for (x, d) in row.iter_mut().enumerate() {
                    *d = src[sy + xs[x]];
                }
            });
        }
        let _ = buf.present();
    }

    /// Keep the frame and the surface the size of the window. Called before
    /// every draw, since the first redraw can come before any resize event.
    fn sync_size(&mut self) {
        let Some(w) = &self.window else { return };
        let size = w.inner_size();
        let (pw, ph) = (size.width.max(1), size.height.max(1));
        if !self.surface_sized || self.output != (pw as usize, ph as usize) {
            if let Some(s) = self.surface.as_mut() {
                let _ = s.resize(NonZeroU32::new(pw).unwrap(), NonZeroU32::new(ph).unwrap());
                self.surface_sized = true;
            }
            self.output = (pw as usize, ph as usize);
            self.fit_frame(pw as usize, ph as usize);
        }
    }

    fn tick(&mut self) {
        self.sync_size();
        let dt = self.advance();
        self.render();
        self.present();
        if self.count_frame(dt) {
            self.update_title();
        }
    }

    /// Time since the last frame: fly the camera and nudge the selected
    /// light by the keys held. Returns the seconds passed.
    fn advance(&mut self) -> f32 {
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32().min(0.1);
        self.last_frame = now;
        let k = &self.keys;
        // Fly keys.
        let mut local = Vec3::ZERO;
        if k.contains(&KeyCode::KeyW) { local.z += 1.0 }
        if k.contains(&KeyCode::KeyS) { local.z -= 1.0 }
        if k.contains(&KeyCode::KeyD) { local.x += 1.0 }
        if k.contains(&KeyCode::KeyA) { local.x -= 1.0 }
        if k.contains(&KeyCode::KeyE) { local.y += 1.0 }
        if k.contains(&KeyCode::KeyQ) { local.y -= 1.0 }
        // Flying is for while the right button looks round; the keys are
        // the manipulators otherwise.
        if local != Vec3::ZERO && self.buttons[2] {
            self.cam.fly(local, dt, self.shift);
        }
        // Light nudge keys, held: relative to the view, flattened to the
        // ground for left/right/forward/back.
        let mut nudge = Vec3::ZERO;
        let fwd = Vec3::new(self.cam.forward().x, 0.0, self.cam.forward().z).normalize_or(Vec3::NEG_Z);
        let right = Vec3::new(self.cam.right().x, 0.0, self.cam.right().z).normalize_or(Vec3::X);
        if k.contains(&KeyCode::ArrowUp) { nudge += fwd }
        if k.contains(&KeyCode::ArrowDown) { nudge -= fwd }
        if k.contains(&KeyCode::ArrowRight) { nudge += right }
        if k.contains(&KeyCode::ArrowLeft) { nudge -= right }
        if k.contains(&KeyCode::PageUp) { nudge += Vec3::Y }
        if k.contains(&KeyCode::PageDown) { nudge -= Vec3::Y }
        if nudge != Vec3::ZERO {
            // A fifth of the subject's radius per second, or a tenth of the
            // light's distance from the camera if that is more; five times
            // that with Shift.
            let eye = self.cam.eye();
            let base = self.scene.focus_radius();
            if let Some(l) = self.rig.selected_mut() {
                let speed = (base * 0.2).max(l.position.distance(eye) * 0.1) * if self.shift { 5.0 } else { 1.0 };
                l.position += nudge * speed * dt;
            }
        }
        dt
    }

    /// Count a frame of `dt` seconds; true when the frame rate was measured
    /// again (twice a second).
    fn count_frame(&mut self, dt: f32) -> bool {
        self.fps_accum.0 += dt;
        self.fps_accum.1 += 1;
        if self.fps_accum.0 >= 0.5 {
            self.title_fps = self.fps_accum.1 as f32 / self.fps_accum.0;
            self.fps_accum = (0.0, 0);
            return true;
        }
        false
    }

    fn update_title(&self) {
        if let Some(w) = &self.window {
            w.set_title(&self.status_line());
        }
    }

    /// What the window title shows.
    fn status_line(&self) -> String {
        let s = self.frame.stats;
        let cut = raster::cut_lux(&self.settings.sky, self.settings.exposure);
        let light = match self.rig.selected() {
            Some(l) => format!(
                "light {}/{} {:.0} lm at ({:.2}, {:.2}, {:.2}) reach {}",
                self.rig.selected + 1,
                self.rig.lights.len(),
                l.lumens,
                l.position.x,
                l.position.y,
                l.position.z,
                if self.settings.light_reach { format_metres(raster::light_reach(l, cut)) } else { "off".into() }
            ),
            None => "no lights".into(),
        };
        let sky = format!(
            "{:02}:{:02}  sun {:.0}{}  sky {}  contrast light {:.2} sky {:.2}",
            self.hours as u32,
            ((self.hours.fract()) * 60.0).round() as u32 % 60,
            self.settings.sun.brightness,
            if self.settings.sun.dir.y <= 0.0 { " (set)" } else { "" },
            if self.settings.sky.on { format!("{:.1} lux", self.settings.sky.lux) } else { "off".into() },
            self.settings.sky.contrast,
            self.settings.backdrop.contrast
        );
        format!(
            "FkJensen  {}  {:.0} fps  {}x{} ({:.0}% of the window)  {light}  {sky}  {}  shaded {} of {} points{}  cache {} of {} MB  shadows {} ({:.0} ms)  bounce {}  {}  setup {:.0} raster {:.0} shade {:.0} resolve {:.0} ms  CPU only",
            self.scene_name,
            self.title_fps,
            self.frame.width,
            self.frame.height,
            self.render_scale * 100.0,
            if s.visibility_kept { "visibility kept" } else { "visibility new" },
            s.points_shaded,
            s.points_used,
            if s.rate_scale > 1.0 { format!(", {:.1} px apart to fit memory", self.settings.shading_rate_px * s.rate_scale) } else { format!(", {:.1} px apart", self.settings.shading_rate_px) },
            s.cache_bytes >> 20,
            self.settings.cache_budget >> 20,
            if self.settings.shadows { "on" } else { "off" },
            s.shadow_ms,
            if !self.settings.bounce { "off".to_string() } else if s.receivers_pending > 0 { format!("{} receivers to go", s.receivers_pending) } else { "done".to_string() },
            {
                let t = self.scene.textures.stats();
                if self.scene.textures.is_empty() { "no textures".to_string() } else { format!("textures {:.1} MB held, {:.1} MB read", t.resident as f64 / 1e6, t.bytes_read as f64 / 1e6) }
            },
            s.setup_ms + s.bin_ms,
            s.raster_ms,
            s.shade_ms,
            s.resolve_ms
        )
    }

    /// Height of the ground under the scene: 0 for the built-in set.
    fn floor(&self) -> f32 {
        if self.scene.bounds_min.y < 0.0 && self.scene.bounds_max.y > 0.0 { 0.0 } else { self.scene.bounds_min.y }
    }

    /// Put a light above the surface under the cursor: a new one, or the
    /// selected one when `replace`.
    fn place_light(&mut self, replace: bool) {
        let (x, y) = (self.mouse.0 as usize, self.mouse.1 as usize);
        let Some((p, n)) = raster::unproject(&self.frame, &self.scene, x, y) else { return };
        // Hold the light off the surface, along its normal, so it lights the
        // spot that was clicked: a quarter of the subject's radius, or a
        // twentieth of the distance from the camera when that is more.
        let off = (self.scene.focus_radius() * 0.25).max(p.distance(self.cam.eye()) * 0.05);
        let pos = p + n * off;
        if replace {
            if let Some(l) = self.rig.selected_mut() {
                l.position = pos;
                return;
            }
        }
        if !self.rig.add(pos, 800.0) {
            eprintln!("the rig holds {} lights at most", lights::MAX_LIGHTS);
        }
    }

    fn drag_plane(&self) -> DragPlane {
        if self.ctrl { DragPlane::Ground } else { DragPlane::Screen }
    }

    /// Restart the drag plane from where the light is now, so switching
    /// Ctrl mid-drag does not make the light jump.
    fn reanchor_drag(&mut self) {
        let plane = self.drag_plane();
        if let Some(d) = self.drag.as_mut() {
            if let Some(l) = self.rig.lights.get(d.light) {
                d.anchor = l.position;
                d.plane = plane;
            }
        }
    }

    fn update_drag(&mut self) {
        let Some(d) = &self.drag else { return };
        let cam = self.render_camera();
        if let Some(p) = lights::drag_target(&cam, self.frame.width, self.frame.height, self.mouse.0, self.mouse.1, d.anchor, d.plane) {
            if let Some(l) = self.rig.lights.get_mut(d.light) {
                l.position = p;
            }
        }
    }

    /// The cursor moved to (x, y) in output pixels.
    fn on_cursor(&mut self, x: f32, y: f32) {
        self.raw_mouse = (x, y);
        // Picking and dragging work in frame pixels.
        self.mouse = (x * self.render_scale, y * self.render_scale);
        let dx = self.raw_mouse.0 - self.last_mouse.0;
        let dy = self.raw_mouse.1 - self.last_mouse.1;
        self.last_mouse = self.raw_mouse;
        let h = self.frame.height as f32 / self.render_scale;
        if self.manip.is_some() {
            self.update_manip();
        } else if self.drag.is_some() {
            self.update_drag();
        } else if self.alt && self.buttons[0] {
            self.cam.orbit(dx, dy);
        } else if self.alt && self.buttons[2] {
            self.cam.dolly((dx + dy) * 0.02);
        } else if self.buttons[1] {
            self.cam.pan(dx, dy, h);
        } else if self.buttons[2] {
            self.cam.look(dx, dy);
        } else if !self.buttons[0] {
            self.hot = self.pick_handle();
        }
    }

    /// Mouse button 0 (left), 1 (middle) or 2 (right) went down or up.
    fn on_button(&mut self, idx: usize, pressed: bool) {
        self.buttons[idx] = pressed;
        if idx == 0 && !pressed {
            self.drag = None;
            self.end_manip();
        }
        if pressed && idx == 0 {
            if self.keys.contains(&KeyCode::Digit1) {
                self.place_light(false);
                self.buttons[0] = false;
            } else if self.keys.contains(&KeyCode::Digit2) {
                self.place_light(true);
                self.buttons[0] = false;
            } else if self.alt {
                // Tumbling.
            } else if let Some(h) = self.pick_handle().filter(|_| !self.ctrl) {
                self.start_manip(h);
            } else if let Some(i) = self.select_at_pointer() {
                self.drag = Some(Drag { light: i, anchor: self.rig.lights[i].position, plane: self.drag_plane() });
            }
        }
    }

    fn on_wheel(&mut self, steps: f32) {
        if self.manip.is_some() {
            return;
        }
        if let Some(d) = &self.drag {
            let eye = self.cam.eye();
            if let Some(l) = self.rig.lights.get_mut(d.light) {
                l.position = lights::push_pull(eye, l.position, -steps);
            }
            self.reanchor_drag();
        } else {
            self.cam.dolly(steps);
        }
    }

    /// Size the frame for an output of `ow` x `oh` pixels: micropolygon
    /// spacing, then render scale, follow the point budget unless set by
    /// hand.
    fn fit_frame(&mut self, ow: usize, oh: usize) {
        if let Some(b) = self.point_budget {
            let (r, scale) = hardware::window_rate(b, ow * oh);
            self.settings.shading_rate_px = r;
            self.render_scale = scale;
        }
        let fw = ((ow as f32 * self.render_scale).round() as usize).max(1);
        let fh = ((oh as f32 * self.render_scale).round() as usize).max(1);
        self.frame.resize(fw, fh);
    }

    /// A key went down or up. Esc sets `quit`.
    fn key(&mut self, code: KeyCode, pressed: bool) {
        if pressed {
            self.keys.insert(code);
        } else {
            self.keys.remove(&code);
        }
        match code {
            KeyCode::ShiftLeft | KeyCode::ShiftRight => self.shift = pressed,
            KeyCode::ControlLeft | KeyCode::ControlRight => {
                self.ctrl = pressed;
                self.reanchor_drag();
            }
            KeyCode::AltLeft | KeyCode::AltRight => self.alt = pressed,
            _ => {}
        }
        if !pressed {
            return;
        }
        match code {
            KeyCode::Escape => self.quit = true,
            // While the right button is held these keys fly.
            KeyCode::KeyQ | KeyCode::KeyW | KeyCode::KeyE | KeyCode::KeyA if self.buttons[2] => {}
            KeyCode::KeyQ => self.tool = manip::Tool::Select,
            KeyCode::KeyW => self.tool = manip::Tool::Move,
            KeyCode::KeyE => self.tool = manip::Tool::Rotate,
            KeyCode::KeyR if !self.ctrl => self.tool = manip::Tool::Scale,
            KeyCode::KeyF => self.frame_selection(),
            KeyCode::KeyA => {
                self.wide = true;
                frame_camera(&mut self.cam, &self.scene, true);
            }
            KeyCode::KeyB => self.settings.cull_backfaces = !self.settings.cull_backfaces,
            KeyCode::KeyG => self.settings.wireframe = !self.settings.wireframe,
            KeyCode::KeyL => self.rig.show = !self.rig.show,
            KeyCode::KeyR => {
                self.rig = LightRig::new(default_rig(&self.scene));
                self.selection = Selection::Nothing;
            }
            KeyCode::Tab => {
                self.rig.select_next();
                self.selection = Selection::Light(self.rig.selected);
            }
            KeyCode::Delete | KeyCode::Backspace => {
                if let Selection::Light(_) = self.selection {
                    self.drag = None;
                    self.manip = None;
                    self.rig.remove_selected();
                    self.selection = Selection::Nothing;
                }
            }
            KeyCode::Equal | KeyCode::NumpadAdd => self.rig.scale_selected(1.25),
            KeyCode::Minus | KeyCode::NumpadSubtract => self.rig.scale_selected(0.8),
            KeyCode::BracketRight => self.settings.exposure *= 1.25,
            KeyCode::BracketLeft => self.settings.exposure /= 1.25,
            KeyCode::Digit0 => self.settings.exposure = raster::default_exposure(self.settings.sky.lux),
            KeyCode::KeyK => self.settings.sky.on = !self.settings.sky.on,
            KeyCode::Comma => self.settings.sky.lux = (self.settings.sky.lux / 2.0).max(0.01),
            KeyCode::Period => self.settings.sky.lux = (self.settings.sky.lux * 2.0).min(120_000.0),
            KeyCode::KeyH => {
                self.settings.view = if self.settings.view == View::Shaded { View::LightCount } else { View::Shaded };
            }
            KeyCode::KeyX => self.settings.light_reach = !self.settings.light_reach,
            KeyCode::KeyM => self.settings.shadows = !self.settings.shadows,
            KeyCode::KeyI => self.settings.bounce = !self.settings.bounce,
            KeyCode::KeyZ => {
                // Z: render at a finer resolution, Shift+Z: coarser.
                let s = if self.shift { self.render_scale / 1.25 } else { self.render_scale * 1.25 };
                self.render_scale = s.clamp(0.0625, 1.0);
                self.point_budget = None;
                self.surface_sized = false;
                let (ow, oh) = self.output;
                if ow > 0 {
                    self.fit_frame(ow, oh);
                }
            }
            KeyCode::KeyC => self.rig.cycle_colour(),
            KeyCode::KeyT => {
                self.hours = (self.hours + if self.shift { -0.25 } else { 0.25 }).rem_euclid(24.0);
                self.apply_daylight();
            }
            KeyCode::KeyY => {
                let b = self.settings.sun.brightness + if self.shift { -50.0 } else { 50.0 };
                self.settings.sun.brightness = b.clamp(0.0, raster::SUN_MAX);
            }
            KeyCode::KeyJ => {
                let c = self.settings.sky.contrast + if self.shift { -0.05 } else { 0.05 };
                self.settings.sky.contrast = (c.clamp(0.0, 1.0) * 20.0).round() / 20.0;
            }
            KeyCode::KeyU => {
                let c = self.settings.backdrop.contrast + if self.shift { -0.05 } else { 0.05 };
                self.settings.backdrop.contrast = (c.clamp(0.0, 1.0) * 20.0).round() / 20.0;
            }
            #[cfg(target_pointer_width = "64")]
            KeyCode::KeyV => {
                if let Some(v) = self.visual {
                    let v = v.step(if self.shift { -1 } else { 1 });
                    eprintln!("view: {}", v.name());
                    self.visual = Some(v);
                }
            }
            KeyCode::KeyP => match save_png(&self.frame, "frame.png") {
                Ok(()) => eprintln!("saved frame.png"),
                Err(e) => eprintln!("cannot save frame.png: {e}"),
            },
            _ => {}
        }
    }

}

/// A scene read from a file, made ready to edit: a file without objects
/// becomes one object, and the camera's subject is the objects (a saved
/// set's ground runs for kilometres).
fn prepare_opened(scene: &mut Scene, path: &Path) {
    if scene.objects.is_empty() && !scene.clusters.is_empty() {
        let (lo, hi) = (scene.bounds_min, scene.bounds_max);
        let name = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        let n = scene.clusters.len();
        scene.add_object(name, 0..n, Vec3::new((lo.x + hi.x) * 0.5, lo.y, (lo.z + hi.z) * 0.5), Vec3::ZERO);
    }
    scene.scatter_missing_samples();
    if scene.focus.is_none() {
        scene.focus = (0..scene.objects.len()).filter(|&i| !scene.objects[i].clusters.is_empty()).map(|i| scene.object_bounds(i)).reduce(|(a, b), (c, d)| (a.min(c), b.max(d)));
    }
}

/// The scene's own lights, or one 800 lm light above and to the side of
/// the subject when it has none.
fn default_rig(scene: &Scene) -> Vec<PointLight> {
    if !scene.lights.is_empty() {
        return scene.lights.iter().take(lights::MAX_LIGHTS).copied().collect();
    }
    let c = scene.focus_centre();
    let r = scene.focus_radius();
    vec![PointLight { position: c + Vec3::new(1.2, 1.6, 1.0) * r, color: lights::normalise_colour(lights::LIGHT_COLOURS[1]), lumens: 800.0 }]
}

/// Frame the subject from the front. Close: from above, filling the view.
/// Wide: from about three times its radius away, at eye level, so what is
/// behind the subject is in the picture (for the built-in set, the
/// mountains behind the T-rex and the car).
fn frame_camera(cam: &mut OrbitCamera, scene: &Scene, wide: bool) {
    cam.frame(scene.focus_centre(), scene.focus_radius());
    cam.yaw = 0.0;
    if wide {
        cam.distance = scene.focus_radius() * 2.2;
        cam.pitch = 0.07;
    } else {
        cam.pitch = 0.45;
    }
}

/// "18:30", "18.5" or "6" as hours.
fn parse_hours(s: &str) -> Option<f32> {
    match s.split_once(':') {
        Some((h, m)) => Some(h.trim().parse::<f32>().ok()? + m.trim().parse::<f32>().ok()? / 60.0),
        None => s.trim().parse().ok(),
    }
    .map(|h: f32| h.rem_euclid(24.0))
}

fn format_metres(m: f32) -> String {
    if m >= 1000.0 { format!("{:.1} km", m / 1000.0) } else if m >= 10.0 { format!("{m:.0} m") } else { format!("{m:.2} m") }
}

/// The built-in set as the tests use it: the ground, the mountains and the
/// set's lights, with the models only if present.
#[cfg(test)]
fn built_in() -> Result<Scene, String> {
    built_in_with(false)
}

/// The built-in set: the ground plane and the mountains, the T-rex
/// (high-resolution when `highres`) and the car, and their lights.
fn built_in_with(highres: bool) -> Result<Scene, String> {
    let mut scene = world::build();
    if let Some((lo, hi)) = add_models(&mut scene, highres) {
        scene.focus = Some((lo, hi));
    }
    scene.lights = world::set_lights();
    scene.scatter_missing_samples();
    eprintln!(
        "built-in set: {} triangles, {} clusters, {} lights, {:.1} km across",
        scene.triangle_count(),
        scene.clusters.len(),
        scene.lights.len(),
        (scene.bounds_max.x - scene.bounds_min.x) / 1000.0
    );
    Ok(scene)
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes().with_title("FkJensen").with_inner_size(PhysicalSize::new(1280, 720));
        let window = Rc::new(event_loop.create_window(attrs).expect("window"));
        let context = softbuffer::Context::new(window.clone()).expect("softbuffer context");
        let surface = softbuffer::Surface::new(&context, window.clone()).expect("softbuffer surface");
        self.window = Some(window);
        self.context = Some(context);
        self.surface = Some(surface);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(_) => self.surface_sized = false,
            WindowEvent::RedrawRequested => {
                self.tick();
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }
            WindowEvent::CursorMoved { position, .. } => self.on_cursor(position.x as f32, position.y as f32),
            WindowEvent::MouseInput { state, button, .. } => {
                let idx = match button {
                    MouseButton::Left => 0,
                    MouseButton::Middle => 1,
                    MouseButton::Right => 2,
                    _ => return,
                };
                self.on_button(idx, state == ElementState::Pressed);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let steps = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 40.0,
                };
                self.on_wheel(steps);
            }
            WindowEvent::KeyboardInput { event: KeyEvent { physical_key: PhysicalKey::Code(code), state, repeat, .. }, .. } => {
                if !repeat {
                    self.key(code, state == ElementState::Pressed);
                    if self.quit {
                        event_loop.exit();
                    }
                }
            }
            WindowEvent::DroppedFile(path) => {
                if let Err(e) = self.open(path) {
                    eprintln!("{e}");
                }
            }
            _ => {}
        }
    }
}

/// Read an .fkj file. Its texture tiles stay in the file and are streamed.
fn load_file(path: &Path) -> Result<Scene, String> {
    let t = Instant::now();
    let name = path.display().to_string();
    let (scene, report) = fkj::read_file(path).map_err(|e| format!("{name}: {e}"))?;
    let tiled = scene.textures.entries.iter().filter(|e| e.level > 0).count();
    eprintln!(
        "{name}: read in {:.2} s: {} triangles, {} vertices, {} clusters ({} with texture tiles, left in the file), {} lights",
        t.elapsed().as_secs_f64(),
        scene.triangle_count(),
        scene.vertex_count(),
        scene.clusters.len(),
        tiled,
        scene.lights.len()
    );
    for tag in &report.skipped {
        eprintln!("  skipped a {tag} section: not read by this build yet");
    }
    Ok(scene)
}

/// Models placed in the built-in set: files (the high-resolution one first,
/// used when the machine's plan allows), position on the ground (metres),
/// turn about the vertical (degrees). Made with tools/fkj-import, in
/// assets/models.
const SET_MODELS: [(&str, Option<&str>, Vec3, f32); 2] = [
    ("trex.fkj", Some("trex_highres.fkj"), world::TREX_AT, world::TREX_YAW),
    ("car.fkj", None, world::CAR_AT, world::CAR_YAW),
];

/// Where models are looked for: $FKJENSEN_ASSETS; then assets/models (in
/// the repository) and assets/local (kept out of it), each in the current
/// folder, in the source tree the program was built from, and beside the
/// program.
fn assets_dirs() -> Vec<PathBuf> {
    let mut d = Vec::new();
    if let Some(p) = std::env::var_os("FKJENSEN_ASSETS") {
        d.push(PathBuf::from(p));
    }
    let exe = std::env::current_exe().ok().and_then(|e| e.parent().map(|p| p.to_path_buf()));
    for sub in ["assets/models", "assets/local"] {
        d.push(PathBuf::from(sub));
        d.push(Path::new(env!("CARGO_MANIFEST_DIR")).join(sub));
        if let Some(e) = &exe {
            d.push(e.join(sub));
        }
    }
    d
}

/// Add the set's models that are present, and return their bounds.
fn add_models(scene: &mut Scene, highres: bool) -> Option<(Vec3, Vec3)> {
    let dirs = assets_dirs();
    let find = |f: &str| dirs.iter().map(|d| d.join(f)).find(|p| p.exists());
    let mut bounds: Option<(Vec3, Vec3)> = None;
    for (file, high, at, yaw) in SET_MODELS {
        let path = high.filter(|_| highres).and_then(find).or_else(|| find(file));
        let Some(path) = path else {
            eprintln!("{file} not found in {}", dirs.iter().map(|d| d.display().to_string()).collect::<Vec<_>>().join(", "));
            continue;
        };
        match load_file(&path) {
            Ok(mut m) => {
                m.place(yaw.to_radians(), 1.0, at);
                let name = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
                let n = m.clusters.len();
                m.add_object(name, 0..n, at, Vec3::new(0.0, yaw.to_radians(), 0.0));
                let (lo, hi) = (m.bounds_min, m.bounds_max);
                bounds = Some(match bounds {
                    Some((a, b)) => (a.min(lo), b.max(hi)),
                    None => (lo, hi),
                });
                eprintln!("{}: {} triangles", path.display(), m.triangle_count());
                scene.append(m);
            }
            Err(e) => eprintln!("{e}"),
        }
    }
    bounds
}

fn save_png(frame: &Frame, path: &str) -> Result<(), String> {
    let file = std::fs::File::create(path).map_err(|e| e.to_string())?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), frame.width as u32, frame.height as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    let mut w = enc.write_header().map_err(|e| e.to_string())?;
    let mut data = Vec::with_capacity(frame.width * frame.height * 3);
    for &c in &frame.color {
        data.extend_from_slice(&[(c >> 16) as u8, (c >> 8) as u8, c as u8]);
    }
    w.write_image_data(&data).map_err(|e| e.to_string())
}

/// The most memory the process has held, from Linux's /proc.
fn peak_memory_mb() -> Option<f64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|l| l.starts_with("VmHWM:"))?;
    let kb: f64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb / 1024.0)
}

fn parse_light(s: &str) -> Option<PointLight> {
    let v: Vec<f32> = s.split(',').map(|x| x.trim().parse().ok()).collect::<Option<_>>()?;
    let lumens = |l: f32| l.clamp(scene::MIN_LUMENS, scene::MAX_LUMENS);
    match v.as_slice() {
        [x, y, z] => Some(PointLight { position: Vec3::new(*x, *y, *z), color: Vec3::ONE, lumens: 800.0 }),
        [x, y, z, l] => Some(PointLight { position: Vec3::new(*x, *y, *z), color: Vec3::ONE, lumens: lumens(*l) }),
        _ => None,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut path: Option<PathBuf> = None;
    let mut png_out: Option<String> = None;
    let mut size: Option<(usize, usize)> = None;
    let mut cli_lights: Vec<PointLight> = Vec::new();
    let mut sky_lux: Option<f32> = None;
    let mut exposure: Option<f32> = None;
    let mut view = View::Shaded;
    let mut reach = true;
    let mut occlusion = true;
    let mut adaptive = true;
    let mut sky_occlusion = true;
    let mut incremental_moves = true;
    let mut bench_move = false;
    let mut bench_dir = false;
    let mut close = false;
    let mut shadows: Option<bool> = None;
    let mut bounce: Option<bool> = None;
    let mut auto = true;
    let mut hardware_only = false;
    let mut scale: Option<f32> = None;
    let mut shading_rate: Option<f32> = None;
    let mut bench_drag = false;
    let mut bench_orbit = false;
    let mut use_fb = false;
    let mut look: Option<(Vec3, Vec3)> = None;
    let mut highres: Option<bool> = None;
    let mut shadow_res: Option<usize> = None;
    let mut visual_debug = false;
    let mut hours: Option<f32> = None;
    let mut sun_brightness: Option<f32> = None;
    let mut dir_lights: Vec<raster::DirLight> = Vec::new();
    let mut light_contrast: Option<f32> = None;
    let mut sky_contrast: Option<f32> = None;
    let mut debug_view: Option<String> = None;
    let mut select: Option<usize> = None;
    let mut editor = false;
    let mut import: Option<PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--png" => {
                png_out = args.get(i + 1).cloned();
                i += 1;
            }
            "--size" => {
                if let Some((w, h)) = args.get(i + 1).and_then(|s| s.split_once('x')) {
                    size = Some((w.parse().unwrap_or(1280), h.parse().unwrap_or(720)));
                }
                i += 1;
            }
            "--light" => {
                match args.get(i + 1).and_then(|s| parse_light(s)) {
                    Some(l) => cli_lights.push(l),
                    None => {
                        eprintln!("--light takes x,y,z or x,y,z,lumens");
                        std::process::exit(2);
                    }
                }
                i += 1;
            }
            "--time" => {
                i += 1;
                hours = args.get(i).and_then(|s| parse_hours(s));
            }
            "--sun" => {
                i += 1;
                sun_brightness = args.get(i).and_then(|s| s.parse::<f32>().ok());
            }
            "--dir" => {
                i += 1;
                let v: Vec<f32> = args.get(i).map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect()).unwrap_or_default();
                if v.len() != 3 {
                    eprintln!("--dir takes azimuth,elevation,lux");
                    std::process::exit(2);
                }
                dir_lights.push(raster::DirLight { on: true, dir: raster::DirLight::towards(v[0], v[1]), color: Vec3::ONE, lux: v[2].clamp(0.0, raster::DIR_MAX_LUX) });
            }
            "--light-contrast" => {
                i += 1;
                light_contrast = args.get(i).and_then(|s| s.parse::<f32>().ok());
            }
            "--sky-contrast" => {
                i += 1;
                sky_contrast = args.get(i).and_then(|s| s.parse::<f32>().ok());
            }
            "--sky" => {
                sky_lux = args.get(i + 1).and_then(|s| s.parse().ok());
                i += 1;
            }
            "--exposure" => {
                exposure = args.get(i + 1).and_then(|s| s.parse().ok());
                i += 1;
            }
            "--view" => {
                if args.get(i + 1).map(|s| s.as_str()) == Some("lights") {
                    view = View::LightCount;
                }
                i += 1;
            }
            "--no-reach" => reach = false,
            "--no-occlusion" => occlusion = false,
            "--no-adaptive" => adaptive = false,
            "--no-sky-occlusion" => sky_occlusion = false,
            "--relight-on-move" => incremental_moves = false,
            "--bench-move" => bench_move = true,
            "--bench-dir" => bench_dir = true,
            "--close" => close = true,
            "--no-shadows" => shadows = Some(false),
            "--shadows" => shadows = Some(true),
            "--no-bounce" => bounce = Some(false),
            "--bounce" => bounce = Some(true),
            "--no-auto" => auto = false,
            "--hardware" => hardware_only = true,
            "--shading-rate" => {
                shading_rate = args.get(i + 1).and_then(|s| s.parse::<f32>().ok()).filter(|r| *r >= 0.25 && *r <= 64.0);
                if shading_rate.is_none() {
                    eprintln!("--shading-rate takes the pixels between micropolygons, 0.25 to 64, e.g. 2");
                    std::process::exit(2);
                }
                i += 1;
            }
            "--scale" => {
                scale = args.get(i + 1).and_then(|s| s.parse().ok());
                if scale.is_none() {
                    eprintln!("--scale takes a fraction of the window size, e.g. 0.5");
                    std::process::exit(2);
                }
                i += 1;
            }
            "--bench-drag" => bench_drag = true,
            "--bench-orbit" => bench_orbit = true,
            "--editor" => editor = true,
            "--import" => {
                i += 1;
                import = args.get(i).map(PathBuf::from);
                editor = true;
            }
            "--fb" => use_fb = true,
            "--visual-debug" => visual_debug = true,
            "--debug-view" => {
                i += 1;
                debug_view = args.get(i).cloned();
                visual_debug = true;
            }
            "--select" => {
                i += 1;
                select = args.get(i).and_then(|v| v.parse::<usize>().ok()).map(|n| n.saturating_sub(1));
            }
            "--no-highres" => highres = Some(false),
            "--highres" => highres = Some(true),
            "--shadow-res" => {
                shadow_res = args.get(i + 1).and_then(|s| s.parse().ok());
                if shadow_res.is_none() {
                    eprintln!("--shadow-res takes a number of texels, e.g. 128");
                    std::process::exit(2);
                }
                i += 1;
            }
            "--look" => {
                // --look ex,ey,ez,tx,ty,tz: camera at e looking at t.
                let v: Option<Vec<f32>> = args.get(i + 1).map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect());
                match v {
                    Some(v) if v.len() == 6 => look = Some((Vec3::new(v[0], v[1], v[2]), Vec3::new(v[3], v[4], v[5]))),
                    _ => {
                        eprintln!("--look takes ex,ey,ez,tx,ty,tz");
                        std::process::exit(2);
                    }
                }
                i += 1;
            }
            "-h" | "--help" => {
                eprintln!("usage: fkjensen [scene.fkj] [--png out.png] [--size WxH] [--light x,y,z[,lumens]] [--time HH:MM] [--sun 0-1000] [--dir AZ,EL,LUX] [--light-contrast 0-1] [--sky-contrast 0-1] [--sky LUX] [--exposure E] [--view lights] [--no-reach] [--no-occlusion] [--no-adaptive] [--no-sky-occlusion] [--relight-on-move] [--[no-]shadows] [--[no-]bounce] [--[no-]highres] [--shadow-res N] [--scale F] [--shading-rate PX] [--no-auto] [--hardware] [--close] [--look ex,ey,ez,tx,ty,tz] [--bench-drag] [--bench-orbit] [--bench-move] [--bench-dir] [--fb] [--visual-debug] [--debug-view NAME|all] [--select N] [--editor [--import MODEL]]\n{CONTROLS}");
                return;
            }
            a => path = Some(PathBuf::from(a)),
        }
        i += 1;
    }

    // What this machine is, and what to do on it. Command-line options
    // override the plan.
    let plan = if auto {
        let hw = hardware::Hardware::detect();
        let plan = hw.plan();
        eprintln!("{}", hw.report());
        for r in &plan.reasons {
            eprintln!("  {r}");
        }
        if hardware_only {
            return;
        }
        Some(plan)
    } else {
        None
    };
    let highres = highres.or(plan.as_ref().map(|p| p.highres)).unwrap_or(true);

    let (mut scene, name) = match &path {
        Some(p) => match load_file(p) {
            Ok(mut s) => {
                prepare_opened(&mut s, p);
                (s, p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default())
            }
            Err(e) => {
                eprintln!("{e}");
                std::process::exit(1);
            }
        },
        None => match built_in_with(highres) {
            Ok(s) => (s, "T-rex, car, mountains".into()),
            Err(e) => {
                eprintln!("{e}");
                std::process::exit(1);
            }
        },
    };
    if !cli_lights.is_empty() {
        scene.lights = cli_lights;
    }

    let mut app = App::new(scene, name);
    app.apply_daylight();
    app.settings.sun.brightness = 600.0;
    if let Some(h) = hours {
        app.hours = h;
        app.apply_daylight();
    }
    for (k, d) in dir_lights.into_iter().take(raster::MAX_DIR_LIGHTS).enumerate() {
        app.settings.dir_lights[k] = d;
    }
    if let Some(b) = sun_brightness {
        app.settings.sun.brightness = b.clamp(0.0, raster::SUN_MAX);
    }
    if let Some(c) = light_contrast {
        app.settings.sky.contrast = c.clamp(0.0, 1.0);
    }
    if let Some(c) = sky_contrast {
        app.settings.backdrop.contrast = c.clamp(0.0, 1.0);
    }
    if let Some(l) = sky_lux {
        app.settings.sky.lux = l.max(0.0);
        app.settings.sky.on = l > 0.0;
        app.settings.exposure = raster::default_exposure(l);
    }
    if let Some(e) = exposure {
        app.settings.exposure = e;
    }
    app.settings.view = view;
    if close {
        app.wide = false;
        frame_camera(&mut app.cam, &app.scene, false);
    }
    if let Some((eye, target)) = look {
        app.cam.look_from(eye, target);
    }
    app.settings.light_reach = reach;
    app.settings.occlusion = occlusion;
    app.settings.adaptive = adaptive;
    app.settings.sky_occlusion = sky_occlusion;
    app.settings.incremental_moves = incremental_moves;
    if let Some(p) = &plan {
        app.settings.shadows = p.shadows;
        app.settings.shadow_res = p.shadow_res;
        app.settings.bounce = p.bounce;
        app.settings.bounce_budget_ms = Some(p.bounce_budget_ms);
        app.settings.tile = p.tile;
        app.render_scale = p.render_scale;
        app.settings.shading_rate_px = p.window_rate_px;
        app.point_budget = Some(p.point_budget);
        app.scene.textures.set_budget(p.texture_cache);
        app.settings.cache_budget = p.shade_cache;
    }
    if let Some(v) = shadows {
        app.settings.shadows = v;
    }
    if let Some(r) = shadow_res {
        app.settings.shadow_res = r.clamp(16, 4096);
    }
    if let Some(v) = bounce {
        app.settings.bounce = v;
    }
    if let Some(v) = scale {
        app.render_scale = v.clamp(0.0625, 1.0);
        app.point_budget = None;
    }
    if let Some(r) = shading_rate {
        app.settings.shading_rate_px = r;
        app.point_budget = None;
    }
    let size = size.or(plan.as_ref().map(|p| p.png_size)).unwrap_or((1280, 720));
    if let Some(n) = select {
        if n < app.rig.lights.len() {
            app.rig.selected = n;
        }
    }
    // Visual debug views: illustration on a capable machine, 64-bit only.
    #[allow(unused_mut)]
    let mut debug_all = false;
    if visual_debug {
        #[cfg(target_pointer_width = "64")]
        {
            let v = match debug_view.as_deref() {
                None | Some("all") => {
                    debug_all = debug_view.is_some();
                    debug::DebugView::Final
                }
                Some(name) => match debug::DebugView::from_name(name) {
                    Some(v) => v,
                    None => {
                        let names: Vec<&str> = debug::ALL.iter().map(|v| v.name()).collect();
                        eprintln!("--debug-view takes all or one of: {}", names.join(", "));
                        std::process::exit(1);
                    }
                },
            };
            app.visual = Some(v);
            eprintln!("visual debug on: V / Shift+V step through the views");
        }
        #[cfg(not(target_pointer_width = "64"))]
        {
            let _ = &debug_view;
            eprintln!("--visual-debug is not in this build (64-bit machines only)");
        }
    }

    if editor {
        #[cfg(target_pointer_width = "64")]
        {
            match png_out {
                // The editor drawn to a file, no window.
                Some(out) => {
                    if let Err(e) = editor::shot(app, size.0, size.1, &out, import) {
                        eprintln!("cannot write {out}: {e}");
                        std::process::exit(1);
                    }
                    eprintln!("wrote {out}");
                }
                None => {
                    if import.is_some() {
                        eprintln!("--import is for --png (drag a model onto the editor, or File > Import)");
                    }
                    eprintln!("{CONTROLS}");
                    editor::run(app, path);
                }
            }
            return;
        }
        #[cfg(not(target_pointer_width = "64"))]
        {
            let _ = &import;
            eprintln!("--editor is not in this build (64-bit machines only)");
            std::process::exit(2);
        }
    }

    if let Some(out) = png_out {
        // A still frame waits for all the bounce light, and has the
        // plan's still shading rate.
        app.settings.bounce_budget_ms = None;
        if let (None, Some(p)) = (shading_rate, &plan) {
            app.settings.shading_rate_px = p.still_rate_px;
        }
        app.frame.resize(size.0, size.1);
        let t = Instant::now();
        app.render();
        let s = app.frame.stats;
        eprintln!(
            "rendered {}x{} in {:.1} ms on {} threads: texture tiles {} reads, {:.1} MB read, {:.1} MB held; shadows {:.1} ms ({} maps, {} triangles), bounce: {} surfels lit in {:.1} ms, {} receivers gathered in {:.1} ms; setup {:.1}, bin {:.1}, raster {:.1}, shade {:.1}, resolve {:.1} ms; {} of {} clusters, {} of {} triangles; {} shading points, {} light evaluations of {} without reach; cache {:.0} MB of {:.0}{}",
            size.0,
            size.1,
            t.elapsed().as_secs_f32() * 1000.0,
            rayon::current_num_threads(),
            app.scene.textures.stats().reads,
            app.scene.textures.stats().bytes_read as f64 / 1e6,
            app.scene.textures.stats().resident as f64 / 1e6,
            s.shadow_ms,
            s.shadow_maps_built,
            s.shadow_tris,
            s.surfels_lit,
            s.bounce_ms,
            s.receivers_updated,
            s.gather_ms,
            s.setup_ms,
            s.bin_ms,
            s.raster_ms,
            s.shade_ms,
            s.resolve_ms,
            s.clusters_drawn,
            s.clusters_in,
            s.tris_drawn,
            s.tris_in,
            s.points_shaded,
            s.light_evals,
            s.light_evals_all,
            app.renderer.memory() as f64 / 1e6,
            app.settings.cache_budget as f64 / 1e6,
            if s.rate_scale > 1.0 { format!(", micropolygons {:.1} px apart to fit it", app.settings.shading_rate_px * s.rate_scale) } else { format!("; micropolygons {:.1} px apart", app.settings.shading_rate_px) }
        );
        if let Err(e) = save_png(&app.frame, &out) {
            eprintln!("cannot write {out}: {e}");
            std::process::exit(1);
        }
        eprintln!("wrote {out}");
        // Every debug view of the same frame, one file each.
        #[cfg(target_pointer_width = "64")]
        if debug_all {
            for v in debug::ALL {
                app.visual = Some(v);
                app.render();
                let p = out.replace(".png", &format!("_{}.png", v.name()));
                match save_png(&app.frame, &p) {
                    Ok(()) => eprintln!("wrote {p}"),
                    Err(e) => eprintln!("cannot write {p}: {e}"),
                }
            }
            app.visual = Some(debug::DebugView::Final);
        }
        let _ = debug_all;
        if let Some(mb) = peak_memory_mb() {
            eprintln!("peak memory {mb:.0} MB");
        }
        if bench_drag {
            // Move the first light 10 cm at a time, as a drag would, and
            // time each frame with everything kept from the one before, with
            // the window's bounce budget.
            app.settings.bounce_budget_ms = Some(30.0);
            for step in 1..=5 {
                if let Some(l) = app.rig.lights.get_mut(0) {
                    l.position.x += 0.1;
                }
                let t = Instant::now();
                app.render();
                let s = app.frame.stats;
                eprintln!(
                    "drag step {step}: {:.1} ms (cache {} MB): visibility {}, shadows {:.1} ms ({} maps, {} triangles), bounce {:.1} ms ({} surfels lit) + gather {:.1} ms ({} receivers, {} waiting), shade {:.1} ms ({} points in {} clusters), resolve {:.1} ms",
                    t.elapsed().as_secs_f32() * 1000.0,
                    s.cache_bytes >> 20,
                    if s.visibility_kept { "kept" } else { "redone" },
                    s.shadow_ms,
                    s.shadow_maps_built,
                    s.shadow_tris,
                    s.bounce_ms,
                    s.surfels_lit,
                    s.gather_ms,
                    s.receivers_updated,
                    s.receivers_pending,
                    s.shade_ms,
                    s.points_shaded,
                    s.clusters_shaded,
                    s.resolve_ms
                );
            }
            let drag_out = out.replace(".png", "_dragged.png");
            if save_png(&app.frame, &drag_out).is_ok() {
                eprintln!("wrote {drag_out}");
            }
        }
        if bench_dir {
            // Turn the first directional light 5 degrees at a time, as
            // dragging its azimuth would.
            app.settings.bounce_budget_ms = Some(30.0);
            if !app.settings.dir_lights[0].on {
                app.settings.dir_lights[0] = raster::DirLight { on: true, ..Default::default() };
                app.render();
            }
            for step in 1..=5 {
                let d = app.settings.dir_lights[0];
                let (az, el) = raster::DirLight::angles(d.dir);
                app.settings.dir_lights[0].dir = raster::DirLight::towards(az + 5.0, el);
                let t = Instant::now();
                app.render();
                let s = app.frame.stats;
                eprintln!(
                    "turn step {step}: {:.1} ms: shadows {:.1} ms ({} maps, {} triangles), surfels {:.1} ms, shade {:.1} ms ({} points), gather {:.1} ms, resolve {:.1} ms",
                    t.elapsed().as_secs_f32() * 1000.0,
                    s.shadow_ms,
                    s.shadow_maps_built,
                    s.shadow_tris,
                    s.bounce_ms,
                    s.shade_ms,
                    s.points_shaded,
                    s.gather_ms,
                    s.resolve_ms
                );
            }
        }
        if bench_move && !app.scene.objects.is_empty() {
            // Move the first model 10 cm at a time, as a drag would.
            app.settings.bounce_budget_ms = Some(30.0);
            for step in 1..=5 {
                let o = app.scene.objects[0].clone();
                app.move_object(0, o.pivot + Vec3::new(0.1, 0.0, 0.0), o.rotation, o.scale);
                let t = Instant::now();
                app.render();
                let s = app.frame.stats;
                eprintln!(
                    "move step {step}: {:.1} ms: shadows {:.1} ms ({} maps drawn again in part or whole, {} triangles), {} clusters lit again, visibility {:.1} ms, shade {:.1} ms ({} points), gather {:.1} ms, resolve {:.1} ms",
                    t.elapsed().as_secs_f32() * 1000.0,
                    s.shadow_ms,
                    s.shadow_maps_built,
                    s.shadow_tris,
                    s.clusters_relit,
                    s.setup_ms + s.bin_ms + s.raster_ms,
                    s.shade_ms,
                    s.points_shaded,
                    s.gather_ms,
                    s.resolve_ms
                );
            }
            app.object_settled(0);
            let t = Instant::now();
            app.render();
            eprintln!("let go: {:.1} ms (bounce light made again)", t.elapsed().as_secs_f32() * 1000.0);
            let move_out = out.replace(".png", "_moved.png");
            if save_png(&app.frame, &move_out).is_ok() {
                eprintln!("wrote {move_out}");
            }
        }
        if bench_orbit {
            // Turn the camera round its target a degree at a time, as a
            // tumble would: visibility is made again each frame, with the
            // last frame's to start from.
            app.settings.bounce_budget_ms = Some(30.0);
            for step in 1..=5 {
                app.cam.orbit(2.2, 0.0);
                let t = Instant::now();
                app.render();
                let s = app.frame.stats;
                eprintln!(
                    "orbit step {step}: {:.1} ms: setup {:.1} bin {:.1} raster {:.1} ms, {} triangles in {} clusters ({} drawn first, {} hidden behind them), shade {:.1} ms ({} points), gather {:.1} ms, resolve {:.1} ms",
                    t.elapsed().as_secs_f32() * 1000.0,
                    s.setup_ms,
                    s.bin_ms,
                    s.raster_ms,
                    s.tris_drawn,
                    s.clusters_drawn,
                    s.clusters_first,
                    s.clusters_occluded,
                    s.shade_ms,
                    s.points_shaded,
                    s.gather_ms,
                    s.resolve_ms
                );
            }
            let orbit_out = out.replace(".png", "_orbited.png");
            if save_png(&app.frame, &orbit_out).is_ok() {
                eprintln!("wrote {orbit_out}");
            }
        }
        return;
    }

    eprintln!("{CONTROLS}");

    // No X or Wayland (a Raspberry Pi on its text console): draw straight
    // to the framebuffer instead of opening a window.
    #[cfg(target_os = "linux")]
    if use_fb || fbdev::no_display_server() {
        if let Err(e) = run_framebuffer(&mut app) {
            eprintln!("{e}");
            std::process::exit(1);
        }
        return;
    }
    #[cfg(not(target_os = "linux"))]
    let _ = use_fb;

    let event_loop = match EventLoop::new() {
        Ok(e) => e,
        Err(e) => {
            eprintln!("cannot open a window: {e}");
            std::process::exit(1);
        }
    };
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut app).expect("run");
}

/// Render to /dev/fb0, reading keys and mouse from /dev/input, until Esc.
#[cfg(target_os = "linux")]
fn run_framebuffer(app: &mut App) -> Result<(), String> {
    let mut fb = fbdev::Framebuffer::open("/dev/fb0")?;
    let mut input = fbdev::Input::open();
    let g = fb.geometry;
    app.output = (g.width, g.height);
    app.fit_frame(g.width, g.height);
    eprintln!(
        "no display server: drawing to /dev/fb0, {}x{} at {} bits per pixel, rendering {}x{}; {} input devices{}",
        g.width,
        g.height,
        g.format.bits,
        app.frame.width,
        app.frame.height,
        input.device_count(),
        if input.device_count() == 0 { " (add the user to the input group for keyboard and mouse)" } else { "" }
    );
    // Over ssh the status goes to the ssh terminal; on the console itself
    // it would be drawn over the picture, so it is left out there.
    let over_ssh = std::env::var_os("SSH_CONNECTION").is_some();
    let console = fbdev::Console::take();
    let r = fb_loop(app, &mut fb, || input.poll(), over_ssh, None);
    drop(console);
    r
}

/// The framebuffer main loop: input, a frame, show it. Stops on Esc or
/// after `max_frames`.
#[cfg(target_os = "linux")]
fn fb_loop(app: &mut App, fb: &mut fbdev::Framebuffer, mut poll: impl FnMut() -> fbdev::Events, status: bool, max_frames: Option<usize>) -> Result<(), String> {
    let g = fb.geometry;
    let mut cursor = (g.width as f32 / 2.0, g.height as f32 / 2.0);
    let mut last_status = Instant::now();
    let mut frames = 0;
    app.last_frame = Instant::now();
    while !app.quit && max_frames.is_none_or(|m| frames < m) {
        frames += 1;
        let ev = poll();
        for (k, down) in ev.keys {
            app.key(k, down);
            let (ow, oh) = app.output;
            if (app.frame.width, app.frame.height) != (((ow as f32 * app.render_scale).round() as usize).max(1), ((oh as f32 * app.render_scale).round() as usize).max(1)) {
                app.fit_frame(ow, oh);
            }
        }
        if ev.motion != (0, 0) {
            cursor.0 = (cursor.0 + ev.motion.0 as f32).clamp(0.0, g.width as f32 - 1.0);
            cursor.1 = (cursor.1 + ev.motion.1 as f32).clamp(0.0, g.height as f32 - 1.0);
            app.on_cursor(cursor.0, cursor.1);
        }
        for (b, down) in ev.buttons {
            app.on_button(b, down);
        }
        if ev.wheel != 0 {
            app.on_wheel(ev.wheel as f32);
        }
        app.tick();
        fb.show(&app.frame.color, app.frame.width, app.frame.height, Some((cursor.0 as usize, cursor.1 as usize))).map_err(|e| format!("/dev/fb0: {e}"))?;
        if status && last_status.elapsed().as_secs_f32() >= 2.0 {
            eprintln!("{}", app.status_line());
            last_status = Instant::now();
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A ball on a floor, the ball an object, in an app with no window,
    /// rendered once at 320 x 180.
    pub(crate) fn ball_app() -> App {
        let mut scene = Scene::new();
        scene.push_mesh(&[Vec3::new(-5.0, 0.0, -5.0), Vec3::new(5.0, 0.0, -5.0), Vec3::new(5.0, 0.0, 5.0), Vec3::new(-5.0, 0.0, 5.0)], &[], &[], &[0, 2, 1, 0, 3, 2]);
        let first = scene.clusters.len();
        let (p, idx) = scene::tests::uv_sphere(24);
        let ball: Vec<Vec3> = p.iter().map(|v| *v * 0.5 + Vec3::new(0.0, 0.5, 0.0)).collect();
        scene.push_mesh(&ball, &[], &[], &idx);
        let n = scene.clusters.len();
        scene.add_object("ball".into(), first..n, Vec3::ZERO, Vec3::ZERO);
        scene.focus = Some((Vec3::new(-1.0, 0.0, -1.0), Vec3::new(1.0, 1.0, 1.0)));
        let mut app = App::new(scene, "ball".into());
        app.cam.look_from(Vec3::new(2.0, 2.5, 4.0), Vec3::new(0.0, 0.4, 0.0));
        app.settings.bounce = false;
        app.output = (320, 180);
        app.fit_frame(320, 180);
        app.render();
        app
    }

    fn screen_of(app: &App, p: Vec3) -> (f32, f32) {
        let (x, y, _) = raster::project(&app.render_camera(), app.frame.width, app.frame.height, p).unwrap();
        (x, y)
    }

    /// Clicking the ball selects it; dragging the move tool's X arrow moves
    /// it along X only; clicking the floor clears the selection; F frames it.
    #[test]
    fn select_and_move_with_the_manipulator() {
        let mut app = ball_app();
        let (x, y) = screen_of(&app, Vec3::new(0.0, 0.5, 0.0));
        app.on_cursor(x, y);
        app.on_button(0, true);
        app.on_button(0, false);
        assert_eq!(app.selection, Selection::Object(0));
        app.key(KeyCode::KeyW, true);
        app.key(KeyCode::KeyW, false);
        app.render();
        // Take the X arrow two thirds along, drag it further along X.
        let len = manip::handle_length(&app.render_camera(), app.frame.height, Vec3::ZERO);
        let (ax, ay) = screen_of(&app, Vec3::X * len * 0.66);
        app.on_cursor(ax, ay);
        assert_eq!(app.hot, Some(manip::Handle::Axis(0)));
        app.on_button(0, true);
        let (bx, by) = screen_of(&app, Vec3::X * (len * 0.66 + 1.0));
        app.on_cursor(bx, by);
        app.on_button(0, false);
        let p = app.scene.objects[0].pivot;
        assert!((p.x - 1.0).abs() < 0.05 && p.y.abs() < 1e-4 && p.z.abs() < 1e-4, "{p}");
        app.render();
        // The floor: nothing selected.
        let (fx, fy) = screen_of(&app, Vec3::new(-3.0, 0.0, -1.0));
        app.on_cursor(fx, fy);
        app.on_button(0, true);
        app.on_button(0, false);
        assert_eq!(app.selection, Selection::Nothing);
        // Select again and frame it.
        app.selection = Selection::Object(0);
        app.key(KeyCode::KeyF, true);
        assert!(app.cam.target.distance(Vec3::new(1.0, 0.5, 0.0)) < 0.2, "{}", app.cam.target);
    }

    /// Rotate and scale turn and grow the object about its pivot; Alt and
    /// the left button tumble without selecting.
    #[test]
    fn rotate_scale_and_tumble() {
        let mut app = ball_app();
        app.selection = Selection::Object(0);
        app.key(KeyCode::KeyE, true);
        app.render();
        let len = manip::handle_length(&app.render_camera(), app.frame.height, Vec3::ZERO);
        // Take the Y ring at 45 degrees (where no other ring is) and carry
        // it a quarter turn round.
        let ring = |a: f32| Vec3::new(a.cos(), 0.0, a.sin()) * len;
        let q45 = std::f32::consts::FRAC_PI_4;
        let (rx, ry) = screen_of(&app, ring(q45));
        app.on_cursor(rx, ry);
        assert_eq!(app.hot, Some(manip::Handle::Ring(1)));
        app.on_button(0, true);
        let (sx, sy) = screen_of(&app, ring(q45 - std::f32::consts::FRAC_PI_2));
        app.on_cursor(sx, sy);
        app.on_button(0, false);
        let r = app.scene.objects[0].rotation;
        let turned = scene::euler_quat(r) * Vec3::X;
        assert!(turned.distance(Vec3::NEG_Z) < 0.1, "{r} turns X to {turned}");
        // The Z ring tips it over: X goes up.
        app.scene.set_object(0, Vec3::ZERO, Vec3::ZERO, 1.0);
        app.render();
        let ringz = |a: f32| Vec3::new(a.cos(), a.sin(), 0.0) * len;
        let (rx, ry) = screen_of(&app, ringz(-q45));
        app.on_cursor(rx, ry);
        assert_eq!(app.hot, Some(manip::Handle::Ring(2)));
        app.on_button(0, true);
        let (sx, sy) = screen_of(&app, ringz(q45));
        app.on_cursor(sx, sy);
        app.on_button(0, false);
        let turned = scene::euler_quat(app.scene.objects[0].rotation) * Vec3::X;
        assert!(turned.distance(Vec3::Y) < 0.15, "X turned to {turned}");
        app.key(KeyCode::KeyR, true);
        app.render();
        let (cx, cy) = screen_of(&app, Vec3::ZERO);
        app.on_cursor(cx, cy);
        assert_eq!(app.hot, Some(manip::Handle::Centre));
        app.on_button(0, true);
        app.on_cursor(cx + 120.0, cy);
        app.on_button(0, false);
        let k = app.scene.objects[0].scale;
        assert!((k - std::f32::consts::E).abs() < 0.1, "{k}");
        // Alt + left drag tumbles and keeps the selection.
        let yaw_cam = app.cam.yaw;
        app.key(KeyCode::AltLeft, true);
        app.on_cursor(10.0, 10.0);
        app.on_button(0, true);
        app.on_cursor(60.0, 10.0);
        app.on_button(0, false);
        app.key(KeyCode::AltLeft, false);
        assert_ne!(app.cam.yaw, yaw_cam);
        assert_eq!(app.selection, Selection::Object(0));
    }

    #[test]
    fn light_argument_parses() {
        assert_eq!(parse_light("1,2,3").unwrap().position, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(parse_light("1, 2, 3, 500").unwrap().lumens, 500.0);
        assert_eq!(parse_light("1,2,3,1e9").unwrap().lumens, scene::MAX_LUMENS);
        assert!(parse_light("1,2").is_none());
        assert!(parse_light("a,b,c").is_none());
    }

    #[test]
    fn the_set_has_its_lights_and_ground() {
        let scene = built_in().unwrap();
        assert_eq!(scene.lights.len(), 3);
        assert!(scene.triangle_count() > 10_000);
    }

    /// The point of light reach: whatever the set's lights do, up to the
    /// brightest light allowed, none of them is ever
    /// evaluated on the mountains, at the default sky and exposure.
    #[test]
    fn no_light_reaches_the_mountains() {
        let mut scene = built_in().unwrap();
        for l in &mut scene.lights {
            l.lumens = scene::MAX_LUMENS;
        }
        let sky = Sky::new(DEFAULT_SKY_LUX, true);
        let cut = raster::cut_lux(&sky, raster::default_exposure(sky.lux));
        let mut mountain_clusters = 0;
        for c in &scene.clusters {
            // Mountain clusters: in the range and off the valley floor.
            if c.centre.z < world::MOUNTAINS_NEAR_Z && c.centre.y > 20.0 {
                mountain_clusters += 1;
                for l in &scene.lights {
                    assert!(l.position.distance(c.centre) > c.radius + raster::light_reach(l, cut));
                }
            }
        }
        assert!(mountain_clusters > 200, "{mountain_clusters}");
    }

    /// Rendering the whole set with and without light reach: the picture
    /// changes by at most one 8-bit step, the work drops.
    #[test]
    fn light_reach_does_not_change_the_picture() {
        let scene = built_in().unwrap();
        let (w, h) = (320, 180);
        let mut cam = OrbitCamera::new();
        frame_camera(&mut cam, &scene, true);
        let c = Camera { view: cam.view(), proj: cam.proj(w as f32 / h as f32), eye: cam.eye(), near: cam.near() };
        let sky = Sky::new(DEFAULT_SKY_LUX, true);
        let mut a = Frame::new(w, h);
        reyes::render(&scene, &c, &scene.lights, &RenderSettings::new(sky), &mut a);
        let mut b = Frame::new(w, h);
        reyes::render(&scene, &c, &scene.lights, &RenderSettings { light_reach: false, ..RenderSettings::new(sky) }, &mut b);
        let mut worst = 0;
        for (x, y) in a.color.iter().zip(&b.color) {
            for sh in [0, 8, 16] {
                worst = worst.max((((x >> sh) & 0xff) as i32 - ((y >> sh) & 0xff) as i32).abs());
            }
        }
        assert!(worst <= 1, "worst channel difference {worst}");
        // The valley and the mountains evaluate none of the lights.
        assert!(a.stats.light_evals < b.stats.light_evals, "{} vs {}", a.stats.light_evals, b.stats.light_evals);
    }

    /// A light turned up to the brightest allowed changes the bounce light
    /// on the ground near it, and not on the mountains:
    /// their bounce light is not gathered again.
    #[test]
    fn bounce_changes_stay_local() {
        let scene = built_in().unwrap();
        let mut lights = scene.lights.clone();
        let (w, h) = (192, 108);
        let mut cam = OrbitCamera::new();
        frame_camera(&mut cam, &scene, true);
        let c = Camera { view: cam.view(), proj: cam.proj(w as f32 / h as f32), eye: cam.eye(), near: cam.near() };
        let settings = RenderSettings::new(Sky::new(DEFAULT_SKY_LUX, true));
        let mut r = reyes::Renderer::new();
        let mut f = Frame::new(w, h);
        r.render(&scene, &c, &lights, &settings, &mut f);
        lights[0].lumens = scene::MAX_LUMENS;
        r.render(&scene, &c, &lights, &settings, &mut f);
        let regathered = f.stats.receivers_updated;
        assert!(regathered > 0);
        let stamps = r.bounce_stamps().unwrap();
        let second = stamps.iter().copied().max().unwrap();
        let mut mountains = 0;
        for (ci, cl) in scene.clusters.iter().enumerate() {
            if cl.centre.z < world::MOUNTAINS_NEAR_Z && cl.centre.y > 20.0 && stamps[ci] != 0 {
                mountains += 1;
                assert!(stamps[ci] < second, "mountain cluster {ci} gathered again");
            }
        }
        assert!(mountains > 20, "{mountains} mountain clusters in view");
    }

    /// The framebuffer loop on a file standing in for /dev/fb0: frames
    /// arrive, and a mouse drag from a light's disc moves that light.
    #[cfg(target_os = "linux")]
    #[test]
    fn framebuffer_loop_draws_and_drags() {
        let mut scene = world::build();
        scene.lights = world::small_lights();
        let mut app = App::new(scene, "test".into());
        app.settings = RenderSettings { shadows: false, bounce: false, ..RenderSettings::new(Sky::new(DEFAULT_SKY_LUX, true)) };
        let (w, h) = (192usize, 108usize);
        let dir = std::env::temp_dir().join(format!("fkjensen-fbloop-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("fb");
        let g = fbdev::Geometry { width: w, height: h, stride: w * 4, offset: 0, format: fbdev::Format { bits: 32, red: (16, 8), green: (8, 8), blue: (0, 8) } };
        std::fs::write(&path, vec![0u8; g.stride * h]).unwrap();
        let f = std::fs::OpenOptions::new().read(true).write(true).open(&path).unwrap();
        let mut fb = fbdev::Framebuffer::with_geometry(f, g);
        app.output = (w, h);
        app.fit_frame(w, h);
        // One frame, to find where the first light is on screen.
        fb_loop(&mut app, &mut fb, fbdev::Events::default, false, Some(1)).unwrap();
        let cam = app.render_camera();
        let (lx, ly, _) = raster::project(&cam, app.frame.width, app.frame.height, app.rig.lights[0].position).unwrap();
        let before = app.rig.lights[0].position;
        // Scripted input: move the cursor from the centre onto the light,
        // press, move 20 pixels right, release.
        let mut script = vec![
            fbdev::Events { motion: ((lx - w as f32 / 2.0) as i32, (ly - h as f32 / 2.0) as i32), ..Default::default() },
            fbdev::Events { buttons: vec![(0, true)], ..Default::default() },
            fbdev::Events { motion: (20, 0), ..Default::default() },
            fbdev::Events { buttons: vec![(0, false)], ..Default::default() },
        ]
        .into_iter();
        fb_loop(&mut app, &mut fb, || script.next().unwrap_or_default(), false, Some(4)).unwrap();
        let after = app.rig.lights[0].position;
        assert!(after.distance(before) > 0.01, "the light moved: {before} -> {after}");
        let out = std::fs::read(&path).unwrap();
        assert!(out.iter().any(|b| *b != 0), "a frame reached the framebuffer");
        // Esc stops the loop.
        let mut esc = vec![fbdev::Events { keys: vec![(KeyCode::Escape, true)], ..Default::default() }].into_iter();
        fb_loop(&mut app, &mut fb, || esc.next().unwrap_or_default(), false, Some(100)).unwrap();
        assert!(app.quit);
        std::fs::remove_dir_all(&dir).ok();
    }
}
