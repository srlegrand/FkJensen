//! fkjensen: a CPU-only 3D viewport. No GPU API is linked; the frame is
//! rasterised on the cores and copied to the window through shared memory.
//!
//! Usage:
//!   fkjensen                           open the built-in chess set
//!   fkjensen scene.fkj                 open an .fkj scene
//!   fkjensen --png out.png             render one frame to a file, no window
//!   fkjensen --size 1920x1080 --png out.png
//!   fkjensen --light x,y,z[,intensity] replace the stored lights (repeatable)
//!
//! Controls are listed in README.md and printed at start.

mod camera;
mod fkj;
mod lights;
mod raster;
mod scene;

use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::Instant;

use glam::Vec3;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use camera::OrbitCamera;
use lights::{DragPlane, LightRig};
use raster::{Camera, Frame, RenderSettings};
use scene::{PointLight, Scene};

/// The chess set, baked into the binary. See docs/format.md and
/// assets/README.md.
const CHESS_SET: &[u8] = include_bytes!("../assets/chess_set.fkj");

const CONTROLS: &str = "\
Controls
  Lights (the only light type is a point light)
  left drag on a light   move it, facing the camera (stays at its depth)
    + Ctrl               move it level, over the set (keeps its height)
    + wheel              push it away / pull it closer
  arrows                 nudge the selected light, relative to the view
  Page Up / Page Down    raise / lower the selected light (Shift: 5x)
  1 (hold) + click       add a light above the surface under the cursor
  2 (hold) + click       put the selected light above that surface
  Tab                    select next light     Delete   remove selected light
  + / -                  intensity             C        cycle colour
  L                      show / hide lights    R        reset to the scene's lights

  Camera
  left drag              orbit                 middle drag / shift+left   pan
  right drag             look (fly)            wheel                      dolly
  W A S D Q E            fly (shift: faster)   F                          frame the scene

  View
  [ / ]                  exposure              B    back-face culling
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
    drag: Option<Drag>,
    // input state
    mouse: (f32, f32),
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
        cam.frame(scene.centre(), scene.radius());
        let rig = LightRig::new(default_rig(&scene));
        Self {
            window: None,
            context: None,
            surface: None,
            surface_sized: false,
            frame: Frame::new(1, 1),
            settings: RenderSettings { cull_backfaces: false, wireframe: false, exposure: 1.0, scene_radius: scene.radius() },
            scene,
            scene_name,
            cam,
            rig,
            drag: None,
            mouse: (0.0, 0.0),
            last_mouse: (0.0, 0.0),
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
                self.cam.frame(self.scene.centre(), self.scene.radius());
                self.settings.scene_radius = self.scene.radius();
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
        raster::render(&self.scene, &cam, &self.rig.lights, &self.settings, &mut self.frame);
        if self.rig.show {
            let floor = self.scene.bounds_min.y;
            for (i, l) in self.rig.lights.iter().enumerate() {
                let sel = i == self.rig.selected;
                // A dashed line down to the floor of the scene and a small
                // mark there, so the light's height reads at a glance.
                let foot = Vec3::new(l.position.x, floor, l.position.z);
                let grey = if sel { Vec3::new(1.0, 0.9, 0.2) * 0.7 } else { Vec3::splat(0.35) };
                raster::draw_line(&mut self.frame, &cam, l.position, foot, grey, true);
                raster::draw_disc(&mut self.frame, &cam, foot, 2.5, grey, false);
                raster::draw_disc(&mut self.frame, &cam, l.position, if sel { 7.0 } else { 5.0 }, l.color, false);
                if sel {
                    raster::draw_disc(&mut self.frame, &cam, l.position, 11.0, Vec3::new(1.0, 0.9, 0.2), true);
                }
            }
        }
    }

    fn present(&mut self) {
        let Some(surface) = self.surface.as_mut() else { return };
        let Ok(mut buf) = surface.buffer_mut() else { return };
        let n = buf.len().min(self.frame.color.len());
        buf[..n].copy_from_slice(&self.frame.color[..n]);
        let _ = buf.present();
    }

    /// Keep the frame and the surface the size of the window. Called before
    /// every draw, since the first redraw can come before any resize event.
    fn sync_size(&mut self) {
        let Some(w) = &self.window else { return };
        let size = w.inner_size();
        let (pw, ph) = (size.width.max(1), size.height.max(1));
        if pw as usize != self.frame.width || ph as usize != self.frame.height || !self.surface_sized {
            if let Some(s) = self.surface.as_mut() {
                let _ = s.resize(NonZeroU32::new(pw).unwrap(), NonZeroU32::new(ph).unwrap());
                self.surface_sized = true;
            }
            self.frame.resize(pw as usize, ph as usize);
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
            // A fifth of the scene's radius per second, five times that with Shift.
            let speed = self.scene.radius() * if self.shift { 1.0 } else { 0.2 };
            if let Some(l) = self.rig.selected_mut() {
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
        let Some(w) = &self.window else { return };
        let s = self.frame.stats;
        let light = match self.rig.selected() {
            Some(l) => format!(
                "light {}/{} at ({:.3}, {:.3}, {:.3}) x{:.2}",
                self.rig.selected + 1,
                self.rig.lights.len(),
                l.position.x,
                l.position.y,
                l.position.z,
                l.intensity
            ),
            None => "no lights".into(),
        };
        w.set_title(&format!(
            "FkJensen  {}  {}x{}  {:.0} fps  {light}  {} tris ({} drawn, {}/{} clusters)  setup {:.1} bin {:.1} raster {:.1} shade {:.1} ms  CPU only",
            self.scene_name,
            self.frame.width,
            self.frame.height,
            self.title_fps,
            s.tris_in,
            s.tris_drawn,
            s.clusters_drawn,
            s.clusters_in,
            s.setup_ms,
            s.bin_ms,
            s.raster_ms,
            s.shade_ms
        ));
    }

    /// Put a light above the surface under the cursor: a new one, or the
    /// selected one when `replace`.
    fn place_light(&mut self, replace: bool) {
        let (x, y) = (self.mouse.0 as usize, self.mouse.1 as usize);
        let Some((p, n)) = raster::unproject(&self.frame, &self.scene, x, y) else { return };
        // Hold the light off the surface, along its normal, by a share of
        // the scene's size, so it lights the spot that was clicked.
        let pos = p + n * self.scene.radius() * 0.25;
        if replace {
            if let Some(l) = self.rig.selected_mut() {
                l.position = pos;
                return;
            }
        }
        self.rig.add(pos, 1.0);
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

    fn key(&mut self, code: KeyCode, pressed: bool, event_loop: &ActiveEventLoop) {
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
            KeyCode::Escape => event_loop.exit(),
            KeyCode::KeyF => self.cam.frame(self.scene.centre(), self.scene.radius()),
            KeyCode::KeyB => self.settings.cull_backfaces = !self.settings.cull_backfaces,
            KeyCode::KeyG => self.settings.wireframe = !self.settings.wireframe,
            KeyCode::KeyL => self.rig.show = !self.rig.show,
            KeyCode::KeyR => self.rig = LightRig::new(default_rig(&self.scene)),
            KeyCode::Tab => self.rig.select_next(),
            KeyCode::Delete | KeyCode::Backspace => {
                self.drag = None;
                self.rig.remove_selected();
            }
            KeyCode::Equal | KeyCode::NumpadAdd => self.scale_light(1.25),
            KeyCode::Minus | KeyCode::NumpadSubtract => self.scale_light(0.8),
            KeyCode::BracketRight => self.settings.exposure *= 1.25,
            KeyCode::BracketLeft => self.settings.exposure /= 1.25,
            KeyCode::KeyC => self.rig.cycle_colour(),
            KeyCode::KeyP => match save_png(&self.frame, "frame.png") {
                Ok(()) => eprintln!("saved frame.png"),
                Err(e) => eprintln!("cannot save frame.png: {e}"),
            },
            _ => {}
        }
    }

    fn scale_light(&mut self, k: f32) {
        if let Some(l) = self.rig.selected_mut() {
            l.intensity *= k;
        }
    }
}

/// The scene's own lights, or one key light above and to the side when it
/// has none, so nothing opens black.
fn default_rig(scene: &Scene) -> Vec<PointLight> {
    if !scene.lights.is_empty() {
        return scene.lights.clone();
    }
    let c = scene.centre();
    let r = scene.radius();
    vec![PointLight { position: c + Vec3::new(1.2, 1.6, 1.0) * r, color: lights::LIGHT_COLOURS[1], intensity: 3.0 }]
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
            WindowEvent::CursorMoved { position, .. } => {
                self.mouse = (position.x as f32, position.y as f32);
                let dx = self.mouse.0 - self.last_mouse.0;
                let dy = self.mouse.1 - self.last_mouse.1;
                self.last_mouse = self.mouse;
                let h = self.frame.height as f32;
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
            WindowEvent::MouseInput { state, button, .. } => {
                let pressed = state == ElementState::Pressed;
                let idx = match button {
                    MouseButton::Left => 0,
                    MouseButton::Middle => 1,
                    MouseButton::Right => 2,
                    _ => return,
                };
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
                        if let Some(i) = self.rig.pick(&cam, self.frame.width, self.frame.height, self.mouse.0, self.mouse.1) {
                            self.rig.selected = i;
                            self.drag = Some(Drag { light: i, anchor: self.rig.lights[i].position, plane: self.drag_plane() });
                        }
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let steps = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 40.0,
                };
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
            WindowEvent::KeyboardInput { event: KeyEvent { physical_key: PhysicalKey::Code(code), state, repeat, .. }, .. } => {
                if !repeat {
                    self.key(code, state == ElementState::Pressed, event_loop);
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

fn parse_light(s: &str) -> Option<PointLight> {
    let v: Vec<f32> = s.split(',').map(|x| x.trim().parse().ok()).collect::<Option<_>>()?;
    match v.as_slice() {
        [x, y, z] => Some(PointLight { position: Vec3::new(*x, *y, *z), color: Vec3::ONE, intensity: 1.0 }),
        [x, y, z, i] => Some(PointLight { position: Vec3::new(*x, *y, *z), color: Vec3::ONE, intensity: *i }),
        _ => None,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut path: Option<PathBuf> = None;
    let mut png_out: Option<String> = None;
    let mut size = (1280usize, 720usize);
    let mut cli_lights: Vec<PointLight> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--png" => {
                png_out = args.get(i + 1).cloned();
                i += 1;
            }
            "--size" => {
                if let Some((w, h)) = args.get(i + 1).and_then(|s| s.split_once('x')) {
                    size = (w.parse().unwrap_or(1280), h.parse().unwrap_or(720));
                }
                i += 1;
            }
            "--light" => {
                match args.get(i + 1).and_then(|s| parse_light(s)) {
                    Some(l) => cli_lights.push(l),
                    None => {
                        eprintln!("--light takes x,y,z or x,y,z,intensity");
                        std::process::exit(2);
                    }
                }
                i += 1;
            }
            "-h" | "--help" => {
                eprintln!("usage: fkjensen [scene.fkj] [--png out.png] [--size WxH] [--light x,y,z[,intensity]]\n{CONTROLS}");
                return;
            }
            a => path = Some(PathBuf::from(a)),
        }
        i += 1;
    }

    let (mut scene, name) = match &path {
        Some(p) => match load_file(p) {
            Ok(s) => (s, p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()),
            Err(e) => {
                eprintln!("{e}");
                std::process::exit(1);
            }
        },
        None => match load_bytes(CHESS_SET, "built-in chess set") {
            Ok(s) => (s, "chess set".into()),
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

    if let Some(out) = png_out {
        app.frame.resize(size.0, size.1);
        let t = Instant::now();
        app.render();
        let s = app.frame.stats;
        eprintln!(
            "rendered {}x{} in {:.1} ms on {} threads: setup {:.1}, bin {:.1}, raster {:.1}, shade {:.1} ms, {} of {} clusters, {} of {} triangles drawn",
            size.0,
            size.1,
            t.elapsed().as_secs_f32() * 1000.0,
            rayon::current_num_threads(),
            s.setup_ms,
            s.bin_ms,
            s.raster_ms,
            s.shade_ms,
            s.clusters_drawn,
            s.clusters_in,
            s.tris_drawn,
            s.tris_in
        );
        if let Err(e) = save_png(&app.frame, &out) {
            eprintln!("cannot write {out}: {e}");
            std::process::exit(1);
        }
        eprintln!("wrote {out}");
        return;
    }

    eprintln!("{CONTROLS}");
    let event_loop = EventLoop::new().expect("event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut app).expect("run");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn built_in_chess_set_decodes() {
        let (scene, report) = fkj::read(CHESS_SET).unwrap();
        assert!(report.skipped.is_empty());
        assert!(scene.triangle_count() > 500_000, "{}", scene.triangle_count());
        assert!(!scene.lights.is_empty());
        assert!(scene.radius() > 0.1 && scene.radius() < 1.0, "the set is about half a metre across");
        for c in &scene.clusters {
            assert!(c.vertex_count as usize <= scene::CLUSTER_VERTS && c.tri_count as usize <= scene::CLUSTER_TRIS);
        }
    }

    #[test]
    fn light_argument_parses() {
        assert_eq!(parse_light("1,2,3").unwrap().position, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(parse_light("1, 2, 3, 0.5").unwrap().intensity, 0.5);
        assert!(parse_light("1,2").is_none());
        assert!(parse_light("a,b,c").is_none());
    }
}
