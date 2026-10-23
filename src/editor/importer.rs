//! The importer window: pick a model (glTF, FBX, OBJ) and its textures,
//! set its size and turn, and see it baked as FkJensen will draw it, levels
//! of detail and texture tiles included, before adding it to the scene.
//!
//! Reading and baking run on a thread of their own (tools/fkj-import's
//! code), so the editor stays live; a model that fails to read reports why
//! and never takes the editor down.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver};
use std::time::Instant;

use glam::Vec3;

use super::files::{self, Browser, Mode, Outcome};
use crate::bake::{self, Baked, Options};
use crate::camera::OrbitCamera;
use crate::raster::{self, Camera, Frame, RenderSettings, Sky};
use crate::reyes::Renderer;
use crate::scene::Scene;

/// The user texture the preview is shown with.
pub const PREVIEW_TEXTURE: u64 = 2;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Size {
    /// As the file has it (glTF is in metres; FBX often in centimetres).
    AsIs,
    /// Longest horizontal side, metres.
    Length,
    Height,
}

/// What the model holds, before baking.
#[derive(Clone, Debug, Default)]
pub struct ModelInfo {
    pub triangles: usize,
    pub vertices: usize,
    pub materials: usize,
    pub textures: Vec<(String, usize, usize)>,
    /// Size after scaling, metres.
    pub size: Vec3,
    pub scale: f32,
}

enum Msg {
    Log(String),
    Warn(String),
    Info(ModelInfo),
    Done(Result<Box<Baked>, String>),
}

struct Job {
    rx: Receiver<Msg>,
    started: Instant,
}

/// A baked model, drawn in the window.
struct Preview {
    scene: Scene,
    name: String,
    renderer: Renderer,
    cam: OrbitCamera,
    settings: RenderSettings,
    frame: Frame,
    /// Tiles per finest level, pool bytes, triangles and clusters per LOD
    /// pass.
    levels: Vec<usize>,
    pool_bytes: usize,
    lod: (Vec<usize>, Vec<usize>),
    /// Frames still to render: a few after anything changes, so streamed
    /// detail settles, then none while nothing moves.
    dirty: u32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Pick {
    Model,
    Texture,
    Folder,
}

pub struct Importer {
    pub path: String,
    /// An OBJ's base colour texture.
    pub texture: String,
    /// Folders an FBX's textures are looked for in (beside the model too).
    pub tex_dirs: Vec<String>,
    pub size: Size,
    pub metres: f32,
    pub rotate_x: f32,
    pub rotate_y: f32,
    pub lod: bool,
    /// Smallest texture point spacing, millimetres.
    pub texel_mm: f32,
    job: Option<Job>,
    log: Vec<(String, bool)>,
    info: Option<ModelInfo>,
    preview: Option<Preview>,
    browser: Option<(Browser, Pick)>,
    last_dir: Option<PathBuf>,
    /// The preview's size in pixels, from the last layout.
    want: (usize, usize),
    pub wireframe: bool,
}

/// What the window asks of the editor.
pub enum Request {
    Nothing,
    Close,
    /// Add this scene as one object named so.
    Add(Scene, String),
}

impl Importer {
    pub fn new(start_dir: Option<PathBuf>) -> Importer {
        Importer {
            path: String::new(),
            texture: String::new(),
            tex_dirs: Vec::new(),
            size: Size::AsIs,
            metres: 1.0,
            rotate_x: 0.0,
            rotate_y: 0.0,
            lod: true,
            texel_mm: 2.0,
            job: None,
            log: Vec::new(),
            info: None,
            preview: None,
            browser: None,
            last_dir: start_dir,
            want: (480, 360),
            wireframe: false,
        }
    }

    pub fn busy(&self) -> bool {
        self.job.is_some()
    }

    pub fn ready(&self) -> bool {
        self.preview.is_some() && self.job.is_none()
    }

    /// Read and bake `self.path` with the options set, on a thread.
    pub fn start(&mut self) {
        let path = PathBuf::from(self.path.trim());
        if !path.is_file() {
            self.log = vec![(format!("{} not found", path.display()), true)];
            return;
        }
        if let Some(d) = path.parent() {
            self.last_dir = Some(d.to_path_buf());
        }
        let (tx, rx) = channel();
        let texture = Some(PathBuf::from(self.texture.trim())).filter(|p| !p.as_os_str().is_empty());
        let mut dirs: Vec<PathBuf> = self.tex_dirs.iter().map(PathBuf::from).collect();
        if let Some(d) = path.parent() {
            // Beside the model, and the usual "textures" folder there.
            dirs.push(d.to_path_buf());
            dirs.push(d.join("textures"));
            dirs.push(d.join("Textures"));
        }
        let (length, height) = match self.size {
            Size::AsIs => (None, None),
            Size::Length => (Some(self.metres), None),
            Size::Height => (None, Some(self.metres)),
        };
        let (rx_deg, ry_deg, lod) = (self.rotate_x, self.rotate_y, self.lod);
        let opts = Options { texel: (self.texel_mm / 1000.0).max(1e-5), max_level: crate::texture::MAX_LEVEL };
        self.log = vec![(format!("reading {}", path.display()), false)];
        self.info = None;
        std::thread::spawn(move || {
            let send = |m: Msg| {
                let _ = tx.send(m);
            };
            let work = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<Box<Baked>, String> {
                let t = Instant::now();
                let mut m = bake::load(&path, texture.as_deref(), &dirs)?;
                if m.tris.is_empty() {
                    return Err(format!("{}: no triangles", path.display()));
                }
                for w in &m.warnings {
                    send(Msg::Warn(w.clone()));
                }
                send(Msg::Log(format!("read in {:.1} s", t.elapsed().as_secs_f32())));
                let scale = bake::normalise(&mut m, rx_deg, ry_deg, length, height);
                let (lo, hi) = m.bounds();
                send(Msg::Info(ModelInfo {
                    triangles: m.tris.len(),
                    vertices: m.positions.len(),
                    materials: m.materials.len(),
                    textures: m.textures.iter().map(|t| (t.name.clone(), t.size().0, t.size().1)).collect(),
                    size: hi - lo,
                    scale,
                }));
                send(Msg::Log(if lod { "building levels of detail and texture tiles".into() } else { "baking texture tiles".into() }));
                let t = Instant::now();
                let baked = bake::bake_logged(&m, &opts, lod, &|s| send(Msg::Log(s)));
                send(Msg::Log(format!("baked in {:.1} s", t.elapsed().as_secs_f32())));
                Ok(Box::new(baked))
            }));
            let r = work.unwrap_or_else(|e| {
                let why = e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default();
                Err(format!("the reader stopped on this file: {why}"))
            });
            send(Msg::Done(r));
        });
        self.job = Some(Job { rx, started: Instant::now() });
    }

    /// Take what the thread has sent.
    fn poll(&mut self) {
        let Some(job) = &self.job else { return };
        let mut done = None;
        while let Ok(m) = job.rx.try_recv() {
            match m {
                Msg::Log(s) => self.log.push((s, false)),
                Msg::Warn(s) => self.log.push((s, true)),
                Msg::Info(i) => self.info = Some(i),
                Msg::Done(r) => done = Some(r),
            }
        }
        let Some(r) = done else { return };
        let secs = job.started.elapsed().as_secs_f32();
        self.job = None;
        match r {
            Ok(b) => {
                self.log.push((format!("ready in {secs:.1} s"), false));
                let name = Path::new(self.path.trim()).file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "model".into());
                self.preview = Some(Preview::new(*b, name));
            }
            Err(e) => self.log.push((e, true)),
        }
    }

    /// Render the preview if it needs it; the picture to show, 0x00RRGGBB.
    pub fn render_preview(&mut self) -> Option<(usize, usize, &[u32])> {
        let p = self.preview.as_mut()?;
        let (w, h) = (self.want.0.max(16), self.want.1.max(16));
        if (p.frame.width, p.frame.height) != (w, h) {
            p.frame.resize(w, h);
            p.dirty = p.dirty.max(3);
        }
        if p.settings.wireframe != self.wireframe {
            p.settings.wireframe = self.wireframe;
            p.dirty = p.dirty.max(3);
        }
        if p.dirty == 0 {
            return None;
        }
        p.dirty -= 1;
        let cam = Camera { view: p.cam.view(), proj: p.cam.proj(w as f32 / h as f32), eye: p.cam.eye(), near: p.cam.near() };
        p.renderer.render(&p.scene, &cam, &[], &p.settings, &mut p.frame);
        Some((w, h, &p.frame.color))
    }

    pub fn show(&mut self, ctx: &egui::Context) -> Request {
        self.poll();
        if self.job.is_some() {
            ctx.request_repaint();
        }
        let mut request = Request::Nothing;
        let mut open = true;
        let screen = ctx.content_rect();
        egui::Window::new("Import model")
            .open(&mut open)
            .default_size(egui::vec2((screen.width() * 0.75).min(1100.0), (screen.height() * 0.75).min(700.0)))
            .max_height(screen.height() - 40.0)
            .default_pos(screen.center())
            .pivot(egui::Align2::CENTER_CENTER)
            .resizable(true)
            .collapsible(false)
            .show(ctx, |ui| {
                egui::Panel::bottom("import buttons").show(ui, |ui| {
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        let add = ui.add_enabled(self.ready(), egui::Button::new("Add to scene"));
                        if add.clicked() {
                            if let Some(p) = self.preview.take() {
                                request = Request::Add(p.scene, p.name);
                            }
                        }
                        if ui.button("Close").clicked() {
                            request = Request::Close;
                        }
                        if self.busy() {
                            ui.spinner();
                            ui.label(format!("working, {:.0} s", self.job.as_ref().map(|j| j.started.elapsed().as_secs_f32()).unwrap_or(0.0)));
                        }
                    });
                });
                egui::Panel::left("import options").resizable(false).exact_size(310.0).show(ui, |ui| {
                    egui::ScrollArea::vertical().id_salt("import options").auto_shrink([false, false]).show(ui, |ui| self.options(ui));
                });
                egui::CentralPanel::default().show(ui, |ui| {
                    let h = ui.available_height();
                    self.view(ui, h);
                });
            });
        if !open {
            request = Request::Close;
        }
        if let Some((b, pick)) = &mut self.browser {
            let pick = *pick;
            match b.show(ctx) {
                Outcome::Open => {}
                Outcome::Cancelled => self.browser = None,
                Outcome::Chosen(p) => {
                    self.last_dir = Some(if p.is_dir() { p.clone() } else { p.parent().map(Path::to_path_buf).unwrap_or_default() });
                    self.browser = None;
                    let s = p.display().to_string();
                    match pick {
                        Pick::Model => {
                            self.path = s;
                            self.start();
                        }
                        Pick::Texture => self.texture = s,
                        Pick::Folder => {
                            if !self.tex_dirs.contains(&s) {
                                self.tex_dirs.push(s);
                            }
                        }
                    }
                }
            }
        }
        request
    }

    fn options(&mut self, ui: &mut egui::Ui) {
        ui.strong("Model");
        ui.horizontal(|ui| {
            ui.add(egui::TextEdit::singleline(&mut self.path).desired_width(200.0).hint_text("model file"));
            if ui.button("Browse...").clicked() {
                self.browser = Some((Browser::new("Import a model", Mode::Open, &bake::FORMATS, self.last_dir.as_deref(), ""), Pick::Model));
            }
        });
        ui.weak(format!("reads .{}", bake::FORMATS.join(" .")));
        ui.add_space(6.0);
        ui.strong("Textures");
        ui.horizontal(|ui| {
            ui.add(egui::TextEdit::singleline(&mut self.texture).desired_width(200.0).hint_text("base colour (OBJ)"));
            if ui.button("Browse...").clicked() {
                self.browser = Some((Browser::new("Base colour texture", Mode::Open, &["png", "jpg", "jpeg"], self.last_dir.as_deref(), ""), Pick::Texture));
            }
        });
        let mut drop = None;
        for (k, d) in self.tex_dirs.iter().enumerate() {
            ui.horizontal(|ui| {
                if ui.small_button("x").clicked() {
                    drop = Some(k);
                }
                ui.label(d);
            });
        }
        if let Some(k) = drop {
            self.tex_dirs.remove(k);
        }
        if ui.button("Add a texture folder...").clicked() {
            self.browser = Some((Browser::new("Folder with textures", Mode::Folder, &[], self.last_dir.as_deref(), ""), Pick::Folder));
        }
        ui.weak("glTF holds its own; FBX textures are looked for beside the model and in these folders");
        ui.add_space(6.0);
        ui.strong("Size and turn");
        egui::Grid::new("import size").num_columns(2).show(ui, |ui| {
            ui.label("Size");
            egui::ComboBox::from_id_salt("size mode")
                .selected_text(match self.size {
                    Size::AsIs => "as in the file",
                    Size::Length => "length",
                    Size::Height => "height",
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.size, Size::AsIs, "as in the file");
                    ui.selectable_value(&mut self.size, Size::Length, "length");
                    ui.selectable_value(&mut self.size, Size::Height, "height");
                });
            ui.end_row();
            if self.size != Size::AsIs {
                ui.label(if self.size == Size::Length { "Length" } else { "Height" });
                ui.add(egui::DragValue::new(&mut self.metres).speed(0.01).range(0.001..=10_000.0).suffix(" m"));
                ui.end_row();
            }
            ui.label("Turn X");
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut self.rotate_x).speed(1.0).suffix("°"));
                if ui.small_button("-90").clicked() {
                    self.rotate_x -= 90.0;
                }
                if ui.small_button("+90").clicked() {
                    self.rotate_x += 90.0;
                }
            });
            ui.end_row();
            ui.label("Turn Y");
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut self.rotate_y).speed(1.0).suffix("°"));
                if ui.small_button("-90").clicked() {
                    self.rotate_y -= 90.0;
                }
                if ui.small_button("+90").clicked() {
                    self.rotate_y += 90.0;
                }
            });
            ui.end_row();
            ui.label("Texture detail");
            ui.add(egui::DragValue::new(&mut self.texel_mm).speed(0.05).range(0.05..=100.0).suffix(" mm"));
            ui.end_row();
        });
        ui.checkbox(&mut self.lod, "Levels of detail");
        ui.add_space(4.0);
        let label = if self.preview.is_some() || self.info.is_some() { "Import again with these settings" } else { "Import" };
        if ui.add_enabled(!self.busy() && !self.path.trim().is_empty(), egui::Button::new(label)).clicked() {
            self.start();
        }
        ui.add_space(6.0);
        if let Some(i) = &self.info {
            ui.strong("What it holds");
            egui::Grid::new("import info").num_columns(2).show(ui, |ui| {
                ui.label("Triangles");
                ui.label(thousands(i.triangles));
                ui.end_row();
                ui.label("Vertices");
                ui.label(thousands(i.vertices));
                ui.end_row();
                ui.label("Materials");
                ui.label(i.materials.to_string());
                ui.end_row();
                ui.label("Size");
                ui.label(format!("{} x {} x {}", metres(i.size.x), metres(i.size.y), metres(i.size.z)));
                ui.end_row();
                if i.scale != 1.0 {
                    ui.label("Scaled by");
                    ui.label(format!("{:.4}", i.scale));
                    ui.end_row();
                }
                ui.label("Textures");
                ui.label(if i.textures.is_empty() { "none".to_string() } else { i.textures.len().to_string() });
                ui.end_row();
            });
            for (name, w, h) in &i.textures {
                ui.weak(format!("{name}  {w} x {h}"));
            }
        }
        if let Some(p) = &self.preview {
            ui.add_space(4.0);
            ui.strong("Baked");
            let tiled: usize = p.levels.iter().skip(1).sum();
            ui.label(format!("{} clusters, {} with texture tiles ({})", p.scene.clusters.len(), tiled, files::size_text(p.pool_bytes as u64)));
            if p.lod.0.len() > 1 {
                ui.label(format!("{} levels of detail: {} to {} triangles", p.lod.0.len(), thousands(*p.lod.0.first().unwrap()), thousands(*p.lod.0.last().unwrap())));
            }
        }
        ui.add_space(6.0);
        egui::ScrollArea::vertical().id_salt("import log").max_height(160.0).stick_to_bottom(true).show(ui, |ui| {
            for (line, warn) in &self.log {
                if *warn {
                    ui.colored_label(ui.visuals().warn_fg_color, line);
                } else {
                    ui.weak(line);
                }
            }
        });
    }

    /// The preview, `height` points tall in all.
    fn view(&mut self, ui: &mut egui::Ui, height: f32) {
        let top = ui.cursor().top();
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.wireframe, "Wireframe");
            if let Some(p) = &mut self.preview {
                if ui.button("Frame").clicked() {
                    frame(&mut p.cam, &p.scene);
                    p.dirty = 3;
                }
                let s = p.frame.stats;
                ui.weak(format!("drawing {} of {} triangles, {} clusters", thousands(s.tris_drawn), thousands(s.tris_in), s.clusters_drawn));
            }
        });
        let avail = egui::vec2(ui.available_width(), height - (ui.cursor().top() - top) - 4.0);
        let size = egui::vec2(avail.x.max(160.0), avail.y.max(120.0));
        let ppp = ui.ctx().pixels_per_point();
        self.want = ((size.x * ppp) as usize, (size.y * ppp) as usize);
        let Some(p) = &mut self.preview else {
            let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
            ui.painter().rect_filled(rect, 4.0, ui.visuals().extreme_bg_color);
            let text = if self.job.is_some() { "baking..." } else { "choose a model to see it here" };
            ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, text, egui::FontId::proportional(16.0), ui.visuals().weak_text_color());
            return;
        };
        let img = egui::Image::new(egui::load::SizedTexture::new(egui::TextureId::User(PREVIEW_TEXTURE), size)).sense(egui::Sense::click_and_drag());
        let r = ui.add(img);
        let d = r.drag_delta() * ppp;
        if r.dragged_by(egui::PointerButton::Primary) {
            p.cam.orbit(d.x, d.y);
            p.dirty = 3;
        } else if r.dragged_by(egui::PointerButton::Secondary) {
            p.cam.dolly((d.x + d.y) * 0.02);
            p.dirty = 3;
        } else if r.dragged_by(egui::PointerButton::Middle) {
            p.cam.pan(d.x, d.y, size.y * ppp);
            p.dirty = 3;
        }
        if r.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                p.cam.dolly(scroll / 40.0);
                p.dirty = 3;
            }
        }
        r.on_hover_text("drag: tumble   right drag or wheel: dolly   middle drag: track");
    }
}

fn frame(cam: &mut OrbitCamera, scene: &Scene) {
    cam.frame(scene.centre(), scene.radius());
    cam.yaw = 0.6;
    cam.pitch = 0.3;
}

impl Preview {
    fn new(b: Baked, name: String) -> Preview {
        let mut cam = OrbitCamera::new();
        frame(&mut cam, &b.scene);
        // Afternoon light: the sun from the side, the sky round it.
        let d = crate::sun::daylight(15.0);
        let mut settings = RenderSettings::new(Sky::new(d.sky_lux, true));
        settings.sun.dir = d.sun_dir;
        settings.sun.color = d.sun_color;
        settings.sun.brightness = 600.0;
        settings.sky.tint = d.sky_tint;
        settings.backdrop.zenith = d.zenith;
        settings.backdrop.horizon = d.horizon;
        settings.exposure = raster::default_exposure(d.sky_lux);
        settings.bounce = false;
        settings.shadows = true;
        settings.shading_rate_px = 1.5;
        settings.cache_budget = 256 << 20;
        settings.bounce_budget_ms = Some(20.0);
        Preview {
            lod: (b.lod.tris.clone(), b.lod.clusters.clone()),
            levels: b.levels,
            pool_bytes: b.pool_bytes,
            scene: b.scene,
            name,
            renderer: Renderer::new(),
            cam,
            settings,
            frame: Frame::new(16, 16),
            dirty: 3,
        }
    }
}

pub fn thousands(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (k, c) in s.chars().enumerate() {
        if k > 0 && (s.len() - k) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

pub fn metres(m: f32) -> String {
    if m >= 100.0 {
        format!("{m:.0} m")
    } else if m >= 1.0 {
        format!("{m:.2} m")
    } else {
        format!("{:.1} cm", m * 100.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An OBJ goes through the importer's thread, comes back baked, and
    /// the preview renders it; a file that is not a model is an error in
    /// the log, not a crash.
    #[test]
    fn imports_on_a_thread_and_previews() {
        let d = std::env::temp_dir().join(format!("fkj-importer-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let obj = d.join("tri.obj");
        std::fs::write(&obj, "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 2 3\nf 1 3 4\nf 1 4 2\nf 2 4 3\n").unwrap();
        let mut im = Importer::new(None);
        im.path = obj.display().to_string();
        im.size = Size::Height;
        im.metres = 2.0;
        im.start();
        let t = Instant::now();
        while im.busy() && t.elapsed().as_secs() < 60 {
            im.poll();
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(im.ready(), "{:?}", im.log);
        let info = im.info.clone().unwrap();
        assert_eq!(info.triangles, 4);
        assert!((info.size.y - 2.0).abs() < 1e-4);
        im.want = (64, 48);
        let (w, h, px) = im.render_preview().unwrap();
        assert_eq!((w, h), (64, 48));
        let bg = px[0];
        assert!(px.iter().any(|&c| c != bg), "the model shows");

        let bad = d.join("bad.glb");
        std::fs::write(&bad, b"not a model at all").unwrap();
        im.path = bad.display().to_string();
        im.start();
        while im.busy() {
            im.poll();
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(im.log.iter().any(|(_, warn)| *warn), "{:?}", im.log);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn numbers_read_easily() {
        assert_eq!(thousands(1234567), "1,234,567");
        assert_eq!(thousands(12), "12");
        assert_eq!(metres(0.25), "25.0 cm");
    }
}
