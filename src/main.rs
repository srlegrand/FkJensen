//! cpuview: a CPU-only USD viewport. No GPU API is linked; the frame is
//! rasterised on the cores and copied to the window through shared memory.
//!
//! Usage:
//!   cpuview scene.usd                 open a window
//!   cpuview scene.usd --png out.png   render one frame to a file, no window
//!   cpuview scene.usd --size 1920x1080 --png out.png
//!
//! Controls are listed in README.md and printed at start.

mod camera;
mod raster;
mod scene;
mod usd_load;

use std::num::NonZeroU32;
use std::path::PathBuf;
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
use raster::{Camera, Frame, RenderSettings};
use scene::{Light, Scene};

const CONTROLS: &str = "\
Controls
  left drag          orbit            middle drag / shift+left   pan
  right drag         look (fly)       wheel                      dolly
  W A S D Q E        fly (shift: faster)
  F                  frame the scene  B    toggle back-face culling
  1 (hold) + click   place a light on the surface under the cursor
  2 (hold) + click   move the selected light there
  Tab                select next light       Delete     remove selected light
  + / -              light intensity         [ / ]      exposure
  C                  cycle light colour      L          toggle light gizmos
  G                  toggle wireframe        R          reset lights to the file's
  P                  save frame.png          Esc        quit
  drop a .usd/.usda/.usdc/.usdz file on the window to open it";

/// What the user placed: the master lights.
struct LightRig {
    lights: Vec<Light>,
    selected: usize,
    show: bool,
}

const LIGHT_COLOURS: [Vec3; 6] = [
    Vec3::new(1.0, 1.0, 1.0),
    Vec3::new(1.0, 0.85, 0.65),
    Vec3::new(0.65, 0.8, 1.0),
    Vec3::new(1.0, 0.55, 0.4),
    Vec3::new(0.5, 1.0, 0.6),
    Vec3::new(0.9, 0.6, 1.0),
];

struct App {
    window: Option<Rc<Window>>,
    context: Option<softbuffer::Context<Rc<Window>>>,
    surface: Option<softbuffer::Surface<Rc<Window>, Rc<Window>>>,
    surface_sized: bool,
    frame: Frame,
    scene: Scene,
    scene_path: Option<PathBuf>,
    cam: OrbitCamera,
    rig: LightRig,
    settings: RenderSettings,
    // input state
    mouse: (f32, f32),
    last_mouse: (f32, f32),
    buttons: [bool; 3],
    shift: bool,
    keys: std::collections::HashSet<KeyCode>,
    last_frame: Instant,
    fps_accum: (f32, u32),
    title_fps: f32,
}

impl App {
    fn new(scene: Scene, scene_path: Option<PathBuf>) -> Self {
        let mut cam = OrbitCamera::new();
        cam.frame(scene.centre(), scene.radius());
        let rig = LightRig { lights: default_rig(&scene), selected: 0, show: true };
        Self {
            window: None,
            context: None,
            surface: None,
            surface_sized: false,
            frame: Frame::new(1, 1),
            settings: RenderSettings {
                cull_backfaces: false,
                wireframe: false,
                ambient: 0.35,
                exposure: 1.0,
                scene_radius: scene.radius(),
            },
            scene,
            scene_path,
            cam,
            rig,
            mouse: (0.0, 0.0),
            last_mouse: (0.0, 0.0),
            buttons: [false; 3],
            shift: false,
            keys: Default::default(),
            last_frame: Instant::now(),
            fps_accum: (0.0, 0),
            title_fps: 0.0,
        }
    }

    fn open(&mut self, path: PathBuf) {
        match usd_load::load(&path) {
            Ok(r) => {
                report(&r, &path);
                self.scene = r.scene;
                self.scene_path = Some(path);
                self.cam.frame(self.scene.centre(), self.scene.radius());
                self.settings.scene_radius = self.scene.radius();
                self.rig.lights = default_rig(&self.scene);
                self.rig.selected = 0;
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
            for (i, l) in self.rig.lights.iter().enumerate() {
                if let Light::Point { position, color, .. } = l {
                    let sel = i == self.rig.selected;
                    raster::draw_disc(&mut self.frame, &cam, *position, if sel { 7.0 } else { 5.0 }, *color, false);
                    if sel {
                        raster::draw_disc(&mut self.frame, &cam, *position, 11.0, Vec3::new(1.0, 0.9, 0.2), true);
                    }
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
        // Fly keys.
        let mut local = Vec3::ZERO;
        let k = &self.keys;
        if k.contains(&KeyCode::KeyW) { local.z += 1.0 }
        if k.contains(&KeyCode::KeyS) { local.z -= 1.0 }
        if k.contains(&KeyCode::KeyD) { local.x += 1.0 }
        if k.contains(&KeyCode::KeyA) { local.x -= 1.0 }
        if k.contains(&KeyCode::KeyE) { local.y += 1.0 }
        if k.contains(&KeyCode::KeyQ) { local.y -= 1.0 }
        if local != Vec3::ZERO {
            self.cam.fly(local, dt, self.shift);
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
        let name = self.scene_path.as_ref().and_then(|p| p.file_name()).map(|n| n.to_string_lossy().to_string()).unwrap_or("no file".into());
        w.set_title(&format!(
            "cpuview  {name}  {}x{}  {:.0} fps  {} tris ({} drawn)  {} lights  setup {:.1} bin {:.1} raster {:.1} shade {:.1} ms  CPU only",
            self.frame.width, self.frame.height, self.title_fps, s.tris_in, s.tris_drawn, self.rig.lights.len(),
            s.setup_ms, s.bin_ms, s.raster_ms, s.shade_ms
        ));
    }

    fn place_light(&mut self, replace: bool) {
        let (x, y) = (self.mouse.0 as usize, self.mouse.1 as usize);
        let Some((p, n)) = raster::unproject(&self.frame, &self.scene, x, y) else { return };
        // Hold the light off the surface, along its normal, by a share of
        // the scene's size, so it lights the spot that was clicked.
        let pos = p + n * self.scene.radius() * 0.25;
        let colour = LIGHT_COLOURS[self.rig.lights.len() % LIGHT_COLOURS.len()];
        if replace {
            if let Some(Light::Point { position, .. }) = self.rig.lights.get_mut(self.rig.selected) {
                *position = pos;
                return;
            }
        }
        self.rig.lights.push(Light::Point { position: pos, color: colour, intensity: 1.0 });
        self.rig.selected = self.rig.lights.len() - 1;
    }

    fn key(&mut self, code: KeyCode, pressed: bool, event_loop: &ActiveEventLoop) {
        if pressed {
            self.keys.insert(code);
        } else {
            self.keys.remove(&code);
        }
        match code {
            KeyCode::ShiftLeft | KeyCode::ShiftRight => self.shift = pressed,
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
            KeyCode::KeyR => {
                self.rig.lights = default_rig(&self.scene);
                self.rig.selected = 0;
            }
            KeyCode::Tab => {
                if !self.rig.lights.is_empty() {
                    self.rig.selected = (self.rig.selected + 1) % self.rig.lights.len();
                }
            }
            KeyCode::Delete | KeyCode::Backspace => {
                if self.rig.selected < self.rig.lights.len() {
                    self.rig.lights.remove(self.rig.selected);
                    self.rig.selected = self.rig.selected.min(self.rig.lights.len().saturating_sub(1));
                }
            }
            KeyCode::Equal | KeyCode::NumpadAdd => self.scale_light(1.25),
            KeyCode::Minus | KeyCode::NumpadSubtract => self.scale_light(0.8),
            KeyCode::BracketRight => self.settings.exposure *= 1.25,
            KeyCode::BracketLeft => self.settings.exposure /= 1.25,
            KeyCode::KeyC => {
                if let Some(l) = self.rig.lights.get_mut(self.rig.selected) {
                    let (c, next) = match l {
                        Light::Point { color, .. } | Light::Distant { color, .. } => {
                            let i = LIGHT_COLOURS.iter().position(|k| (*k - *color).length() < 1e-3).map(|i| (i + 1) % LIGHT_COLOURS.len()).unwrap_or(0);
                            (color, LIGHT_COLOURS[i])
                        }
                    };
                    *c = next;
                }
            }
            KeyCode::KeyP => {
                if let Err(e) = save_png(&self.frame, "frame.png") {
                    eprintln!("cannot save frame.png: {e}");
                } else {
                    eprintln!("saved frame.png");
                }
            }
            _ => {}
        }
    }

    fn scale_light(&mut self, k: f32) {
        if let Some(l) = self.rig.lights.get_mut(self.rig.selected) {
            match l {
                Light::Point { intensity, .. } | Light::Distant { intensity, .. } => *intensity *= k,
            }
        }
    }
}

fn default_rig(scene: &Scene) -> Vec<Light> {
    if !scene.lights.is_empty() {
        return scene.lights.clone();
    }
    // One key light above and to the side, so a file with no lights is
    // not black.
    let c = scene.centre();
    let r = scene.radius();
    vec![Light::Point { position: c + Vec3::new(1.2, 1.6, 1.0) * r, color: LIGHT_COLOURS[1], intensity: 3.0 }]
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes().with_title("cpuview").with_inner_size(PhysicalSize::new(1280, 720));
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
                if self.buttons[0] && !self.shift {
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
                if pressed && idx == 0 {
                    if self.keys.contains(&KeyCode::Digit1) {
                        self.place_light(false);
                        self.buttons[0] = false;
                    } else if self.keys.contains(&KeyCode::Digit2) {
                        self.place_light(true);
                        self.buttons[0] = false;
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let steps = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 40.0,
                };
                self.cam.dolly(steps);
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

fn report(r: &usd_load::LoadReport, path: &std::path::Path) {
    eprintln!(
        "{}: opened in {:.2} s, read in {:.2} s: {} prims, {} skipped, {} triangles, {} file lights",
        path.display(),
        r.open_secs,
        r.read_secs,
        r.scene.prims_read,
        r.scene.prims_skipped,
        r.scene.triangle_count(),
        r.scene.lights.len()
    );
    for w in r.warnings.iter().take(20) {
        eprintln!("  warning: {w}");
    }
    if r.warnings.len() > 20 {
        eprintln!("  ... {} more warnings", r.warnings.len() - 20);
    }
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

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut path: Option<PathBuf> = None;
    let mut png_out: Option<String> = None;
    let mut size = (1280usize, 720usize);
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--png" => {
                png_out = args.get(i + 1).cloned();
                i += 1;
            }
            "--size" => {
                if let Some(s) = args.get(i + 1) {
                    if let Some((w, h)) = s.split_once('x') {
                        size = (w.parse().unwrap_or(1280), h.parse().unwrap_or(720));
                    }
                }
                i += 1;
            }
            "-h" | "--help" => {
                eprintln!("usage: cpuview [scene.usd] [--png out.png] [--size WxH]\n{CONTROLS}");
                return;
            }
            a => path = Some(PathBuf::from(a)),
        }
        i += 1;
    }

    let mut scene = Scene::new();
    if let Some(p) = &path {
        match usd_load::load(p) {
            Ok(r) => {
                report(&r, p);
                scene = r.scene;
            }
            Err(e) => {
                eprintln!("{e}");
                std::process::exit(1);
            }
        }
    }
    if !scene.bounds_min.is_finite() || scene.positions.is_empty() {
        scene.bounds_min = Vec3::splat(-1.0);
        scene.bounds_max = Vec3::splat(1.0);
    }

    let mut app = App::new(scene, path);

    if let Some(out) = png_out {
        app.frame.resize(size.0, size.1);
        let t = Instant::now();
        app.render();
        let s = app.frame.stats;
        eprintln!(
            "rendered {}x{} in {:.1} ms on {} threads: setup {:.1}, bin {:.1}, raster {:.1}, shade {:.1} ms, {} of {} triangles drawn",
            size.0, size.1, t.elapsed().as_secs_f32() * 1000.0, rayon::current_num_threads(),
            s.setup_ms, s.bin_ms, s.raster_ms, s.shade_ms, s.tris_drawn, s.tris_in
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
