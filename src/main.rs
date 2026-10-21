//! fkjensen: a CPU-only 3D viewport. No GPU API is linked; the frame is
//! rasterised on the cores and copied to the window through shared memory.
//!
//! Usage:
//!   fkjensen                           open the built-in set: chess table,
//!                                      shopfront, mountains
//!   fkjensen scene.fkj                 open an .fkj scene
//!   fkjensen --png out.png             render one frame to a file, no window
//!   fkjensen --size 1920x1080 --png out.png
//!   fkjensen --light x,y,z[,lumens]    replace the scene's lights (repeatable)
//!   fkjensen --sky LUX                 sky illuminance, 0 for no sky
//!   fkjensen --exposure E              display value per cd/m²
//!   fkjensen --view lights             show how many lights each pixel evaluates
//!   fkjensen --no-reach                evaluate every light everywhere (to compare)
//!   fkjensen --close                   start on the close framing of the subject
//!   fkjensen --look ex,ey,ez,tx,ty,tz  camera at e looking at t, in metres
//!   fkjensen --no-shadows              point lights without shadow maps
//!   fkjensen --no-bounce               no bounce light
//!   fkjensen --hardware                print what this machine is and what FkJensen chooses for it
//!   fkjensen --fb                      draw to /dev/fb0 even when a display server is running (Linux)
//!   fkjensen --no-auto                 do not measure the machine; use the fixed defaults
//!   fkjensen --scale F                 render the window at F of its size (0.0625 to 1)
//!   fkjensen --chess / --shadows / --bounce   force on what the machine's plan left off
//!   fkjensen --no-chess                the built-in set without the chess set
//!   fkjensen --shadow-res N            shadow map texels per cube face edge (512)
//!   fkjensen --png out.png --bench-drag  then move the first light five times, timing each frame
//!
//! Controls are listed in README.md and printed at start.

mod bounce;
mod camera;
#[cfg(target_os = "linux")]
mod fbdev;
mod fkj;
mod hardware;
mod lights;
mod raster;
mod reyes;
mod scene;
mod shadow;
mod world;

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

/// The chess set, baked into the binary. See docs/format.md and
/// assets/README.md.
const CHESS_SET: &[u8] = include_bytes!("../assets/chess_set.fkj");

const CONTROLS: &str = "\
Controls
  Lights: point lights in lumens, 1 lm to 2,000,000 lm, plus the sky
  left drag on a light   move it, facing the camera (stays at its depth)
    + Ctrl               move it level, over the set (keeps its height)
    + wheel              push it away / pull it closer
  arrows                 nudge the selected light, relative to the view
  Page Up / Page Down    raise / lower the selected light (Shift: 5x)
  1 (hold) + click       add a light above the surface under the cursor
  2 (hold) + click       put the selected light above that surface
  Tab                    select next light     Delete   remove selected light
  + / -                  lumens x1.25          C        cycle colour
  L                      show / hide lights    R        reset to the scene's lights
  K                      sky on / off          , / .    sky dimmer / brighter (x2)
  H                      light count view: how many lights each pixel evaluates
  X                      light reach off / on (to compare cost and picture)
  M                      shadow maps on / off
  I                      bounce light on / off
  Z / Shift+Z            render resolution finer / coarser (x1.25)

  Camera
  left drag              orbit                 middle drag / shift+left   pan
  right drag             look (fly)            wheel                      dolly
  W A S D Q E            fly (shift: faster)   F    wide view / close on the subject

  View
  [ / ]                  exposure              0    exposure from the sky
  B                      back-face culling
  G                      wireframe             P    save frame.png
  Esc                    quit
  drop an .fkj file on the window to open it";

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
    /// Set by Esc.
    quit: bool,
    /// Size of what the frame is shown on (window or framebuffer).
    output: (usize, usize),
    /// When set, the render scale follows the output size so that a frame
    /// holds about this many pixels (from the machine's plan).
    pixel_budget: Option<f64>,
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
            settings: RenderSettings { bounce_budget_ms: Some(30.0), ..RenderSettings::new(Sky { lux: DEFAULT_SKY_LUX, on: true }) },
            scene,
            scene_name,
            cam,
            rig,
            drag: None,
            quit: false,
            output: (0, 0),
            pixel_budget: None,
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
        }
    }

    fn open(&mut self, path: PathBuf) {
        match load_file(&path) {
            Ok(scene) => {
                self.scene = scene;
                self.scene_name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                self.wide = true;
                frame_camera(&mut self.cam, &self.scene, self.wide);
                self.rig = LightRig::new(default_rig(&self.scene));
                self.drag = None;
            }
            Err(e) => eprintln!("{e}"),
        }
    }

    fn render_camera(&self) -> Camera {
        let aspect = self.frame.width as f32 / self.frame.height.max(1) as f32;
        Camera { view: self.cam.view(), proj: self.cam.proj(aspect), eye: self.cam.eye(), near: self.cam.near() }
    }

    fn render(&mut self) {
        let cam = self.render_camera();
        self.renderer.render(&self.scene, &cam, &self.rig.lights, &self.settings, &mut self.frame);
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
        if local != Vec3::ZERO {
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
        self.render();
        self.present();
        self.fps_accum.0 += dt;
        self.fps_accum.1 += 1;
        if self.fps_accum.0 >= 0.5 {
            self.title_fps = self.fps_accum.1 as f32 / self.fps_accum.0;
            self.fps_accum = (0.0, 0);
            self.update_title();
        }
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
        let sky = if self.settings.sky.on { format!("sky {:.0} lux", self.settings.sky.lux) } else { "sky off".into() };
        format!(
            "FkJensen  {}  {:.0} fps  {}x{} ({:.0}% of the window)  {light}  {sky}  {}  shaded {} of {} points  shadows {} ({:.0} ms)  bounce {}  setup {:.0} raster {:.0} shade {:.0} resolve {:.0} ms  CPU only",
            self.scene_name,
            self.title_fps,
            self.frame.width,
            self.frame.height,
            self.render_scale * 100.0,
            if s.visibility_kept { "visibility kept" } else { "visibility new" },
            s.points_shaded,
            s.points_used,
            if self.settings.shadows { "on" } else { "off" },
            s.shadow_ms,
            if !self.settings.bounce { "off".to_string() } else if s.receivers_pending > 0 { format!("{} receivers to go", s.receivers_pending) } else { "done".to_string() },
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
        if self.drag.is_some() {
            self.update_drag();
        } else if self.buttons[0] && !self.shift {
            self.cam.orbit(dx, dy);
        } else if self.buttons[1] || (self.buttons[0] && self.shift) {
            self.cam.pan(dx, dy, h);
        } else if self.buttons[2] {
            self.cam.look(dx, dy);
        }
    }

    /// Mouse button 0 (left), 1 (middle) or 2 (right) went down or up.
    fn on_button(&mut self, idx: usize, pressed: bool) {
        self.buttons[idx] = pressed;
        if idx == 0 && !pressed {
            self.drag = None;
        }
        if pressed && idx == 0 {
            if self.keys.contains(&KeyCode::Digit1) {
                self.place_light(false);
                self.buttons[0] = false;
            } else if self.keys.contains(&KeyCode::Digit2) {
                self.place_light(true);
                self.buttons[0] = false;
            } else if self.rig.show {
                let cam = self.render_camera();
                if let Some(i) = self.rig.pick(&cam, self.frame.width, self.frame.height, self.mouse.0, self.mouse.1, self.render_scale) {
                    self.rig.selected = i;
                    self.drag = Some(Drag { light: i, anchor: self.rig.lights[i].position, plane: self.drag_plane() });
                }
            }
        }
    }

    fn on_wheel(&mut self, steps: f32) {
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

    /// Size the frame for an output of `ow` x `oh` pixels: the render scale
    /// follows the pixel budget unless it was set by hand.
    fn fit_frame(&mut self, ow: usize, oh: usize) {
        if let Some(b) = self.pixel_budget {
            self.render_scale = ((b / (ow * oh).max(1) as f64).sqrt() as f32).clamp(0.0625, 1.0);
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
            _ => {}
        }
        if !pressed {
            return;
        }
        match code {
            KeyCode::Escape => self.quit = true,
            KeyCode::KeyF => {
                self.wide = !self.wide;
                frame_camera(&mut self.cam, &self.scene, self.wide);
            }
            KeyCode::KeyB => self.settings.cull_backfaces = !self.settings.cull_backfaces,
            KeyCode::KeyG => self.settings.wireframe = !self.settings.wireframe,
            KeyCode::KeyL => self.rig.show = !self.rig.show,
            KeyCode::KeyR => self.rig = LightRig::new(default_rig(&self.scene)),
            KeyCode::Tab => self.rig.select_next(),
            KeyCode::Delete | KeyCode::Backspace => {
                self.drag = None;
                self.rig.remove_selected();
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
                self.pixel_budget = None;
                self.surface_sized = false;
                let (ow, oh) = self.output;
                if ow > 0 {
                    self.fit_frame(ow, oh);
                }
            }
            KeyCode::KeyC => self.rig.cycle_colour(),
            KeyCode::KeyP => match save_png(&self.frame, "frame.png") {
                Ok(()) => eprintln!("saved frame.png"),
                Err(e) => eprintln!("cannot save frame.png: {e}"),
            },
            _ => {}
        }
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
/// Wide: from nine times its radius away, at eye level, so what is behind
/// the subject is in the picture (for the built-in set, the shop and the
/// mountains behind the chess table).
fn frame_camera(cam: &mut OrbitCamera, scene: &Scene, wide: bool) {
    cam.frame(scene.focus_centre(), scene.focus_radius());
    cam.yaw = 0.0;
    if wide {
        cam.distance = scene.focus_radius() * 9.0;
        cam.pitch = 0.07;
    } else {
        cam.pitch = 0.45;
    }
}

fn format_metres(m: f32) -> String {
    if m >= 1000.0 { format!("{:.1} km", m / 1000.0) } else if m >= 10.0 { format!("{m:.0} m") } else { format!("{m:.2} m") }
}

/// The built-in set: the chess set from `assets/chess_set.fkj` on the
/// table, in the procedural square with the shop and the mountains.
#[cfg(test)]
fn built_in() -> Result<Scene, String> {
    built_in_with(true)
}

/// The built-in set, with or without the chess set (1.4 M triangles; the
/// set without it is about 90 K, for machines with little memory).
fn built_in_with(with_chess: bool) -> Result<Scene, String> {
    if !with_chess {
        let mut scene = world::build();
        let shop_lights = std::mem::take(&mut scene.lights);
        scene.lights = world::table_lights();
        scene.lights.extend(shop_lights);
        eprintln!("built-in set without the chess set: {} triangles, {} clusters, {} lights", scene.triangle_count(), scene.clusters.len(), scene.lights.len());
        return Ok(scene);
    }
    let mut chess = load_bytes(CHESS_SET, "built-in chess set")?;
    // Board centred on the table, resting on its top.
    let c = chess.centre();
    chess.translate(Vec3::new(-c.x, world::TABLE_TOP - chess.bounds_min.y, -c.z));
    let (chess_lo, chess_hi) = (chess.bounds_min, chess.bounds_max);
    let mut scene = world::build();
    let shop_lights = std::mem::take(&mut scene.lights);
    scene.append(chess);
    scene.lights = world::table_lights();
    scene.lights.extend(shop_lights);
    // The subject is the board and its pieces, with a margin.
    scene.focus = Some((Vec3::new(chess_lo.x - 0.05, world::TABLE_TOP - 0.05, chess_lo.z - 0.05), Vec3::new(chess_hi.x + 0.05, chess_hi.y, chess_hi.z + 0.05)));
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
            WindowEvent::DroppedFile(path) => self.open(path),
            _ => {}
        }
    }
}

fn load_bytes(bytes: &[u8], name: &str) -> Result<Scene, String> {
    let t = Instant::now();
    let (scene, report) = fkj::read(bytes).map_err(|e| format!("{name}: {e}"))?;
    eprintln!(
        "{name}: read in {:.2} s: {} triangles, {} vertices, {} clusters, {} lights",
        t.elapsed().as_secs_f64(),
        scene.triangle_count(),
        scene.positions.len(),
        scene.clusters.len(),
        scene.lights.len()
    );
    for tag in &report.skipped {
        eprintln!("  skipped a {tag} section: not read by this build yet");
    }
    Ok(scene)
}

fn load_file(path: &Path) -> Result<Scene, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    load_bytes(&bytes, &path.display().to_string())
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
    let mut close = false;
    let mut shadows: Option<bool> = None;
    let mut bounce: Option<bool> = None;
    let mut auto = true;
    let mut hardware_only = false;
    let mut scale: Option<f32> = None;
    let mut bench_drag = false;
    let mut use_fb = false;
    let mut look: Option<(Vec3, Vec3)> = None;
    let mut chess: Option<bool> = None;
    let mut shadow_res: Option<usize> = None;
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
            "--close" => close = true,
            "--no-shadows" => shadows = Some(false),
            "--shadows" => shadows = Some(true),
            "--no-bounce" => bounce = Some(false),
            "--bounce" => bounce = Some(true),
            "--no-auto" => auto = false,
            "--hardware" => hardware_only = true,
            "--scale" => {
                scale = args.get(i + 1).and_then(|s| s.parse().ok());
                if scale.is_none() {
                    eprintln!("--scale takes a fraction of the window size, e.g. 0.5");
                    std::process::exit(2);
                }
                i += 1;
            }
            "--bench-drag" => bench_drag = true,
            "--fb" => use_fb = true,
            "--no-chess" => chess = Some(false),
            "--chess" => chess = Some(true),
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
                eprintln!("usage: fkjensen [scene.fkj] [--png out.png] [--size WxH] [--light x,y,z[,lumens]] [--sky LUX] [--exposure E] [--view lights] [--no-reach] [--[no-]shadows] [--[no-]bounce] [--[no-]chess] [--shadow-res N] [--scale F] [--no-auto] [--hardware] [--close] [--look ex,ey,ez,tx,ty,tz] [--bench-drag] [--fb]\n{CONTROLS}");
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
    let with_chess = chess.or(plan.as_ref().map(|p| p.with_chess)).unwrap_or(true);

    let (mut scene, name) = match &path {
        Some(p) => match load_file(p) {
            Ok(s) => (s, p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()),
            Err(e) => {
                eprintln!("{e}");
                std::process::exit(1);
            }
        },
        None => match built_in_with(with_chess) {
            Ok(s) => (s, "chess table, shop, mountains".into()),
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
    if let Some(l) = sky_lux {
        app.settings.sky = Sky { lux: l.max(0.0), on: l > 0.0 };
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
    if let Some(p) = &plan {
        app.settings.shadows = p.shadows;
        app.settings.shadow_res = p.shadow_res;
        app.settings.bounce = p.bounce;
        app.settings.bounce_budget_ms = Some(p.bounce_budget_ms);
        app.settings.tile = p.tile;
        app.render_scale = p.render_scale;
        app.pixel_budget = Some(p.pixel_budget);
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
        app.pixel_budget = None;
    }
    let size = size.or(plan.as_ref().map(|p| p.png_size)).unwrap_or((1280, 720));

    if let Some(out) = png_out {
        // A still frame waits for all the bounce light.
        app.settings.bounce_budget_ms = None;
        app.frame.resize(size.0, size.1);
        let t = Instant::now();
        app.render();
        let s = app.frame.stats;
        eprintln!(
            "rendered {}x{} in {:.1} ms on {} threads: shadows {:.1} ms ({} maps, {} triangles), bounce: {} surfels lit in {:.1} ms, {} receivers gathered in {:.1} ms; setup {:.1}, bin {:.1}, raster {:.1}, shade {:.1}, resolve {:.1} ms; {} of {} clusters, {} of {} triangles; {} shading points, {} light evaluations of {} without reach; cache {:.0} MB",
            size.0,
            size.1,
            t.elapsed().as_secs_f32() * 1000.0,
            rayon::current_num_threads(),
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
            app.renderer.memory() as f64 / 1e6
        );
        if let Err(e) = save_png(&app.frame, &out) {
            eprintln!("cannot write {out}: {e}");
            std::process::exit(1);
        }
        eprintln!("wrote {out}");
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
                    "drag step {step}: {:.1} ms: visibility {}, shadows {:.1} ms ({} maps, {} triangles), bounce {:.1} ms ({} surfels lit) + gather {:.1} ms ({} receivers, {} waiting), shade {:.1} ms ({} points in {} clusters), resolve {:.1} ms",
                    t.elapsed().as_secs_f32() * 1000.0,
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

    #[test]
    fn built_in_chess_set_decodes() {
        let (scene, report) = fkj::read(CHESS_SET).unwrap();
        assert!(report.skipped.is_empty());
        assert!(scene.triangle_count() > 500_000, "{}", scene.triangle_count());
        assert!(scene.lights.is_empty(), "the chess file holds geometry only");
        assert!(scene.radius() > 0.1 && scene.radius() < 1.0, "the set is about half a metre across");
        for c in &scene.clusters {
            assert!(c.vertex_count as usize <= scene::CLUSTER_VERTS && c.tri_count as usize <= scene::CLUSTER_TRIS);
        }
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
    fn chess_set_stands_on_the_table() {
        let scene = built_in().unwrap();
        let (lo, hi) = scene.focus_bounds();
        assert!(hi.y > world::TABLE_TOP + 0.05, "pieces above the table top");
        assert!(lo.y < world::TABLE_TOP && lo.y > world::TABLE_TOP - 0.1);
        assert_eq!(scene.lights.len(), 2 + 15);
    }

    /// The point of light reach: whatever the lights at the table and the
    /// shop do, up to the brightest light allowed, none of them is ever
    /// evaluated on the mountains, at the default sky and exposure.
    #[test]
    fn no_light_reaches_the_mountains() {
        let mut scene = built_in().unwrap();
        for l in &mut scene.lights {
            l.lumens = scene::MAX_LUMENS;
        }
        let sky = Sky { lux: DEFAULT_SKY_LUX, on: true };
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

    /// The table lamps at their default strength do not reach the shop,
    /// and the shop's lights do not reach the table, while the street lamp
    /// (12,000 lm) reaches both.
    #[test]
    fn table_and_shop_lights_stay_where_they_are() {
        let scene = built_in().unwrap();
        let sky = Sky { lux: DEFAULT_SKY_LUX, on: true };
        let cut = raster::cut_lux(&sky, raster::default_exposure(sky.lux));
        let table = Vec3::new(0.0, world::TABLE_TOP, 0.0);
        let shop = Vec3::new(0.0, 2.0, world::SHOP_FRONT_Z);
        for (i, l) in scene.lights.iter().enumerate() {
            let r = raster::light_reach(l, cut);
            if i < 2 {
                assert!(l.position.distance(shop) > r, "table light {i} reaches the shop: {r} m");
            } else if l.lumens < 1000.0 {
                assert!(l.position.distance(table) > r, "shop light {i} reaches the table: {r} m");
            } else {
                assert!(l.position.distance(table) < r && l.position.distance(shop) < r);
            }
        }
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
        let sky = Sky { lux: DEFAULT_SKY_LUX, on: true };
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
        // The shop is lit by its own fifteen lights, so its pixels still
        // evaluate most of them; the table, square, valley and mountains do not.
        assert!(a.stats.light_evals * 2 < b.stats.light_evals, "{} vs {}", a.stats.light_evals, b.stats.light_evals);
    }

    /// A table lamp turned up to the brightest light allowed changes the
    /// bounce light on the table and the square, and not on the mountains:
    /// their bounce light is not gathered again.
    #[test]
    fn bounce_changes_stay_local() {
        let scene = built_in().unwrap();
        let mut lights = scene.lights.clone();
        let (w, h) = (192, 108);
        let mut cam = OrbitCamera::new();
        frame_camera(&mut cam, &scene, true);
        let c = Camera { view: cam.view(), proj: cam.proj(w as f32 / h as f32), eye: cam.eye(), near: cam.near() };
        let settings = RenderSettings::new(Sky { lux: DEFAULT_SKY_LUX, on: true });
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
        scene.lights = world::table_lights();
        let mut app = App::new(scene, "test".into());
        app.settings = RenderSettings { shadows: false, bounce: false, ..RenderSettings::new(Sky { lux: DEFAULT_SKY_LUX, on: true }) };
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
