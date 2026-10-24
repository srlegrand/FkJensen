//! The editor (`--editor`): the renderer's viewport inside an egui window
//! laid out like Maya. Menus and tools along the top, the outliner on the
//! left (objects and their groups, lights, sun, sky, render settings), the
//! attribute editor on the right, the timeline along the bottom (time of
//! day), and the viewport in the middle, which takes the same mouse and
//! keys as the plain window.
//!
//! Everything is drawn on the CPU: egui is tessellated and filled by
//! `paint.rs` into the same buffer the frame is copied into, and shown
//! through softbuffer like the plain window. 64-bit machines only.

pub mod files;
pub mod importer;
pub mod paint;
mod edit;
mod materials;
mod navigate;
mod panels;
pub mod style;

use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::Instant;

use rayon::prelude::*;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use crate::{manip, App, Selection};
use files::{Browser, Mode, Outcome};

/// What the attribute editor shows.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Inspect {
    /// The selected object or light.
    Selection,
    Sun,
    /// A directional light (an index into `RenderSettings::dir_lights`).
    Directional(usize),
    Sky,
    Render,
}

/// How big the viewport's frame is.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FrameSize {
    /// 1920 x 1080, whatever the window's size.
    Hd,
    /// Width and height set by hand.
    Custom(usize, usize),
    /// The viewport's own pixels, times the resolution percentage.
    Viewport,
}

impl FrameSize {
    pub fn pixels(self) -> Option<(usize, usize)> {
        match self {
            FrameSize::Hd => Some((1920, 1080)),
            FrameSize::Custom(w, h) => Some((w.max(16), h.max(16))),
            FrameSize::Viewport => None,
        }
    }
}

/// Where an editor frame's time went, ms.
#[derive(Clone, Copy, Default, Debug)]
pub struct FrameTimes {
    /// egui laying out the interface.
    pub layout: f32,
    /// The renderer making the viewport's frame.
    pub render: f32,
    /// The frame copied and scaled into the window's picture.
    pub compose: f32,
    /// The interface tessellated and filled over it.
    pub paint: f32,
    /// The picture handed to the window (the window's own frame loop
    /// sets it).
    pub present: f32,
    /// From the start of one frame to the start of the next.
    pub total: f32,
}

impl FrameTimes {
    pub fn line(&self) -> String {
        format!(
            "frame {:.0} ms: interface {:.0}, render {:.0}, copy {:.0}, paint {:.0}, present {:.0}",
            self.total, self.layout, self.render, self.compose, self.paint, self.present
        )
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Purpose {
    Open,
    Save,
}

pub struct Editor {
    pub app: App,
    ctx: egui::Context,
    painter: paint::Painter,
    /// The viewport in window pixels: x, y, width, height.
    view: (usize, usize, usize, usize),
    pub inspect: Inspect,
    last_selection: Selection,
    /// The timeline plays the day at `speed` hours a second.
    pub playing: bool,
    pub speed: f32,
    /// The scene file, once opened or saved.
    pub path: Option<PathBuf>,
    browser: Option<(Browser, Purpose)>,
    pub importer: Option<importer::Importer>,
    /// The last thing done or gone wrong, shown in the status bar.
    message: Option<(String, bool, Instant)>,
    /// An object moved from the attribute editor: its bounce light is made
    /// again once the value stops changing.
    pending_bounce: Option<usize>,
    /// Mouse buttons that went down in the viewport and are still down.
    captured: [bool; 3],
    /// The window's picture, 0x00RRGGBB.
    pub buf: Vec<u32>,
    pub size: (usize, usize),
    pub quit: bool,
    /// The point budget the machine's plan gave, to go back to.
    plan_budget: Option<f64>,
    pub frame_size: FrameSize,
    /// The last frame's times, and when this frame began.
    pub times: FrameTimes,
    began: Option<Instant>,
    /// Buttons over the viewport to move the camera without a mouse.
    pub navigate: bool,
    /// Where each navigation control was last drawn (points), by its label;
    /// the second "Left" is the track row's.
    nav_rects: Vec<(&'static str, egui::Rect)>,
    /// Where the viewport's Show menu button was last drawn (points).
    show_rect: egui::Rect,
    /// Where its "Light count" entry was last drawn, when open.
    show_item: egui::Rect,
    /// What Edit > Copy took.
    clip: Option<edit::Clip>,
}

impl Editor {
    pub fn new(app: App) -> Editor {
        let ctx = egui::Context::default();
        style::apply(&ctx);
        let plan_budget = app.point_budget;
        let last_selection = app.selection;
        let mut app = app;
        app.frame_size = FrameSize::Hd.pixels();
        Editor {
            app,
            ctx,
            painter: paint::Painter::new(),
            view: (0, 0, 0, 0),
            inspect: Inspect::Selection,
            last_selection,
            playing: false,
            speed: 1.0,
            path: None,
            browser: None,
            importer: None,
            message: None,
            pending_bounce: None,
            captured: [false; 3],
            buf: Vec::new(),
            size: (0, 0),
            quit: false,
            plan_budget,
            frame_size: FrameSize::Hd,
            times: FrameTimes::default(),
            began: None,
            navigate: true,
            nav_rects: Vec::new(),
            show_rect: egui::Rect::NOTHING,
            show_item: egui::Rect::NOTHING,
            clip: None,
        }
    }

    /// Use frame size `f`, from the next frame.
    pub fn set_frame_size(&mut self, f: FrameSize) {
        self.frame_size = f;
        self.app.frame_size = f.pixels();
        let (ow, oh) = self.app.output;
        if ow > 0 && oh > 0 {
            self.app.fit_frame(ow, oh);
        }
    }

    pub fn say(&mut self, text: impl Into<String>, error: bool) {
        let text = text.into();
        eprintln!("{text}");
        self.message = Some((text, error, Instant::now()));
    }

    /// One frame of `w` x `h` pixels: lay out the interface, render the
    /// viewport, draw both into `buf`.
    pub fn frame(&mut self, raw: egui::RawInput, w: usize, h: usize) -> egui::PlatformOutput {
        let t0 = Instant::now();
        if let Some(b) = self.began {
            self.times.total = (t0 - b).as_secs_f32() * 1000.0;
        }
        self.began = Some(t0);
        let ctx = self.ctx.clone();
        let out = ctx.run_ui(raw, |ui| self.ui(ui));
        self.times.layout = t0.elapsed().as_secs_f32() * 1000.0;
        self.painter.set_textures(&out.textures_delta);
        let ppp = out.pixels_per_point;

        // The viewport: sized from this layout, rendered now.
        let (_, _, vw, vh) = self.view;
        let dt = self.app.advance();
        if self.playing {
            self.app.hours = (self.app.hours + self.speed * dt).rem_euclid(24.0);
            self.app.apply_daylight();
        }
        let tr = Instant::now();
        if vw > 0 && vh > 0 {
            if self.app.output != (vw, vh) {
                self.app.output = (vw, vh);
                self.app.fit_frame(vw, vh);
            }
            self.app.render();
        }
        self.times.render = tr.elapsed().as_secs_f32() * 1000.0;
        self.app.count_frame(dt);
        if self.pending_bounce.is_some() && !ctx.egui_is_using_pointer() {
            if let Some(i) = self.pending_bounce.take() {
                self.app.object_settled(i);
            }
        }
        if let Some(im) = &mut self.importer {
            if let Some((pw, ph, px)) = im.render_preview() {
                self.painter.set_user(importer::PREVIEW_TEXTURE, pw, ph, px);
            }
        }

        let tc = Instant::now();
        self.compose(w, h);
        let tp = Instant::now();
        self.times.compose = (tp - tc).as_secs_f32() * 1000.0;
        let prims = ctx.tessellate(out.shapes, ppp);
        self.painter.paint(&prims, ppp, &mut self.buf, w, h);
        self.painter.free_textures(&out.textures_delta);
        self.times.paint = tp.elapsed().as_secs_f32() * 1000.0;
        out.platform_output
    }

    /// The frame into the viewport's place in `buf`, scaled to it.
    fn compose(&mut self, w: usize, h: usize) {
        self.size = (w, h);
        self.buf.resize(w * h, 0);
        let (vx, vy, vw, vh) = self.view;
        let f = &self.app.frame;
        let (fw, fh) = (f.width, f.height);
        if vw == 0 || vh == 0 || fw == 0 || fh == 0 || vx + vw > w || vy + vh > h {
            return;
        }
        // Bars round a frame of its own size.
        let (sx, sy, sw, sh) = self.app.shown();
        if (sx, sy, sw, sh) != (0, 0, vw, vh) {
            const BAR: u32 = 0x0018_1818;
            self.buf[vy * w..(vy + vh) * w].par_chunks_mut(w).for_each(|row| row[vx..vx + vw].fill(BAR));
        }
        let (vx, vy, vw, vh) = (vx + sx, vy + sy, sw.min(vw - sx), sh.min(vh - sy));
        let xs: Vec<usize> = (0..vw).map(|x| (x * fw / vw).min(fw - 1)).collect();
        let src = &f.color;
        self.buf[vy * w..(vy + vh) * w].par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            let sy = (y * fh / vh).min(fh - 1) * fw;
            for (x, d) in row[vx..vx + vw].iter_mut().enumerate() {
                *d = src[sy + xs[x]];
            }
        });
    }

    // ------------------------------------------------------------ viewport

    /// Whether window pixel (x, y) is on the viewport with nothing of the
    /// interface (a menu, a window) over it.
    fn on_view(&self, x: f32, y: f32) -> bool {
        let (vx, vy, vw, vh) = self.view;
        if x < vx as f32 || y < vy as f32 || x >= (vx + vw) as f32 || y >= (vy + vh) as f32 {
            return false;
        }
        if self.browser.is_some() || self.ctx.egui_is_using_pointer() {
            return false;
        }
        let ppp = self.ctx.pixels_per_point();
        self.ctx.layer_id_at(egui::pos2(x / ppp, y / ppp)).is_none_or(|l| l.order == egui::Order::Background)
    }

    pub fn view_cursor(&mut self, x: f32, y: f32) {
        if self.captured.iter().any(|b| *b) || self.on_view(x, y) {
            let (vx, vy) = self.frame_origin();
            self.app.on_cursor(x - vx, y - vy);
        }
    }

    pub fn view_button(&mut self, idx: usize, pressed: bool, x: f32, y: f32) {
        if pressed {
            if self.on_view(x, y) {
                self.captured[idx] = true;
                let (vx, vy) = self.frame_origin();
                self.app.on_cursor(x - vx, y - vy);
                self.app.on_button(idx, true);
                self.sync_selection();
            }
        } else if self.captured[idx] {
            self.captured[idx] = false;
            self.app.on_button(idx, false);
        }
    }

    /// Where the frame's top left corner is in the window.
    fn frame_origin(&self) -> (f32, f32) {
        let (vx, vy, _, _) = self.view;
        let (sx, sy, _, _) = self.app.shown();
        ((vx + sx) as f32, (vy + sy) as f32)
    }

    pub fn view_wheel(&mut self, steps: f32, x: f32, y: f32) {
        if self.on_view(x, y) || self.captured.iter().any(|b| *b) {
            self.app.on_wheel(steps);
        }
    }

    /// A key, for the viewport unless a text box is being typed in. The
    /// editor's own shortcuts (Ctrl+O, S, I, G) and Esc stay with it.
    pub fn view_key(&mut self, code: KeyCode, pressed: bool) {
        let modifier = matches!(code, KeyCode::ShiftLeft | KeyCode::ShiftRight | KeyCode::ControlLeft | KeyCode::ControlRight | KeyCode::AltLeft | KeyCode::AltRight);
        if pressed && !modifier {
            if self.ctx.text_edit_focused() || code == KeyCode::Escape || self.browser.is_some() {
                return;
            }
            if self.app.ctrl && matches!(code, KeyCode::KeyO | KeyCode::KeyS | KeyCode::KeyI | KeyCode::KeyG | KeyCode::KeyN) {
                return;
            }
            // Edit's keys, as Maya's.
            match code {
                KeyCode::Delete | KeyCode::Backspace => {
                    self.delete_selected();
                    return;
                }
                KeyCode::KeyC if self.app.ctrl => {
                    self.copy_selected();
                    return;
                }
                KeyCode::KeyV if self.app.ctrl => {
                    self.paste();
                    return;
                }
                KeyCode::KeyD if self.app.ctrl && !self.app.shift => {
                    self.duplicate();
                    return;
                }
                _ => {}
            }
        }
        self.app.key(code, pressed);
        self.sync_selection();
    }

    /// A file dropped on the window: a scene opens, a model goes to the
    /// importer.
    pub fn dropped(&mut self, path: PathBuf) {
        let ext = path.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
        if ext == "fjs" || ext == "fkj" {
            self.open(path);
        } else if ext == "fja" {
            self.add_asset(path);
        } else if crate::bake::FORMATS.contains(&ext.as_str()) {
            self.import(Some(path));
        } else {
            self.say(format!("{}: not a scene (.fjs, .fkj), an asset (.fja) or a model (.{})", path.display(), crate::bake::FORMATS.join(" .")), true);
        }
    }

    /// When the viewport changed the selection, show it.
    fn sync_selection(&mut self) {
        if self.app.selection != self.last_selection {
            self.last_selection = self.app.selection;
            self.inspect = Inspect::Selection;
        }
    }

    pub fn select(&mut self, s: Selection) {
        if let Selection::Light(i) = s {
            self.app.rig.selected = i;
        }
        self.app.selection = s;
        self.last_selection = s;
        self.inspect = Inspect::Selection;
    }

    // --------------------------------------------------------------- files

    pub fn open(&mut self, path: PathBuf) {
        match self.app.open(path.clone()) {
            Ok(()) => {
                self.say(format!("opened {}", path.display()), false);
                self.path = Some(path);
                self.last_selection = Selection::Nothing;
                self.pending_bounce = None;
            }
            Err(e) => self.say(e, true),
        }
    }

    /// Write the scene as an .fjs, and an .fja beside it for each object
    /// not already in one.
    pub fn save(&mut self, path: PathBuf) {
        let path = if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("fjs")) { path } else { path.with_extension("fjs") };
        match self.app.save(&path) {
            Ok(files) => {
                let assets = files.len().saturating_sub(1);
                let bytes: u64 = files.iter().map(|f| std::fs::metadata(f).map(|m| m.len()).unwrap_or(0)).sum();
                let extra = if assets > 0 { format!(" and {assets} asset{}", if assets == 1 { "" } else { "s" }) } else { String::new() };
                self.say(format!("saved {}{extra} ({})", path.display(), files::size_text(bytes)), false);
                self.path = Some(path);
            }
            Err(e) => self.say(format!("cannot save {}: {e}", path.display()), true),
        }
    }

    /// Add an .fja asset to the scene as an object, at the middle of what
    /// the camera looks at when it was made elsewhere.
    pub fn add_asset(&mut self, path: PathBuf) {
        match crate::fjs::load_asset(&path) {
            Ok(m) => {
                let first = self.app.scene.objects.len();
                let name = m.objects[0].name.clone();
                let tris = m.triangle_count();
                self.app.scene.append(m);
                self.app.manip = None;
                self.app.drag = None;
                self.select(Selection::Object(first));
                self.say(format!("added {name} from {} ({} triangles in all its levels)", path.display(), importer::thousands(tris)), false);
            }
            Err(e) => self.say(e, true),
        }
    }

    fn start_dir(&self) -> Option<PathBuf> {
        self.path.as_ref().and_then(|p| p.parent().map(Path::to_path_buf))
    }

    pub fn ask_open(&mut self) {
        self.browser = Some((Browser::new("Open a scene", Mode::Open, &["fjs", "fkj"], self.start_dir().as_deref(), ""), Purpose::Open));
    }

    pub fn ask_save(&mut self) {
        let name = self.path.as_ref().and_then(|p| p.file_name()).map(|n| n.to_string_lossy().into_owned()).map(|n| Path::new(&n).with_extension("fjs").display().to_string()).unwrap_or_else(|| "scene.fjs".into());
        self.browser = Some((Browser::new("Save the scene", Mode::Save, &["fjs"], self.start_dir().as_deref(), &name), Purpose::Save));
    }

    /// Open the importer, on `model` if given.
    pub fn import(&mut self, model: Option<PathBuf>) {
        if let Some(m) = model.as_ref().filter(|m| m.extension().is_some_and(|e| e.eq_ignore_ascii_case("fja"))) {
            self.add_asset(m.clone());
            return;
        }
        let im = self.importer.get_or_insert_with(|| importer::Importer::new(None));
        if let Some(m) = model {
            im.path = m.display().to_string();
            im.start();
        }
    }

    /// Put an imported model in the scene, at the origin, as one object.
    fn add_model(&mut self, mut model: crate::scene::Scene, name: String) {
        let n = model.clusters.len();
        model.objects.clear();
        model.add_object(name.clone(), 0..n, glam::Vec3::ZERO, glam::Vec3::ZERO);
        model.scatter_missing_samples();
        let tris = model.triangle_count();
        let first = self.app.scene.objects.len();
        self.app.scene.append(model);
        self.app.manip = None;
        self.app.drag = None;
        self.select(Selection::Object(first));
        self.say(format!("added {name} ({} triangles in all its levels)", importer::thousands(tris)), false);
    }

    /// Put the selected object in a new group at its pivot.
    pub fn group_selected(&mut self) {
        let Selection::Object(i) = self.app.selection else {
            self.say("select an object to group", true);
            return;
        };
        let pivot = self.app.scene.objects[i].pivot;
        let parent = self.app.scene.objects[i].parent;
        let name = format!("{}_group", self.app.scene.objects[i].name);
        let g = self.app.scene.add_group(name, pivot);
        self.app.scene.set_parent(g, parent);
        self.app.scene.set_parent(i, Some(g));
        self.select(Selection::Object(g));
    }

    pub fn new_group(&mut self) {
        let n = self.app.scene.objects.iter().filter(|o| o.clusters.is_empty()).count();
        let pivot = self.app.cam.target;
        let g = self.app.scene.add_group(format!("group{}", n + 1), glam::Vec3::new(pivot.x, self.app.floor(), pivot.z));
        self.select(Selection::Object(g));
    }

    // ------------------------------------------------------------------ ui

    fn shortcuts(&mut self, ctx: &egui::Context) {
        use egui::{Key, Modifiers};
        if self.browser.is_some() {
            return;
        }
        let hit = |m, k| ctx.input_mut(|i| i.consume_key(m, k));
        if hit(Modifiers::COMMAND, Key::O) {
            self.ask_open();
        }
        if hit(Modifiers::COMMAND | Modifiers::SHIFT, Key::S) {
            self.ask_save();
        } else if hit(Modifiers::COMMAND, Key::S) {
            match self.path.clone() {
                Some(p) => self.save(p),
                None => self.ask_save(),
            }
        }
        if hit(Modifiers::COMMAND, Key::I) {
            self.import(None);
        }
        if hit(Modifiers::COMMAND, Key::G) {
            self.group_selected();
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        self.shortcuts(&ctx);
        egui::Panel::top("menu").show(ui, |ui| self.menu(ui));
        egui::Panel::top("tools").show(ui, |ui| self.toolbar(ui));
        egui::Panel::bottom("status").show(ui, |ui| self.status(ui));
        egui::Panel::bottom("timeline").resizable(false).show(ui, |ui| self.timeline(ui));
        egui::Panel::left("outliner").resizable(true).default_size(200.0).min_size(140.0).show(ui, |ui| self.outliner(ui));
        egui::Panel::right("attributes").resizable(true).default_size(280.0).min_size(200.0).show(ui, |ui| {
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| self.attributes(ui));
        });
        egui::CentralPanel::default().frame(egui::Frame::NONE).show(ui, |ui| self.viewport(ui));

        if let Some((b, purpose)) = &mut self.browser {
            let purpose = *purpose;
            match b.show(&ctx) {
                Outcome::Open => {}
                Outcome::Cancelled => self.browser = None,
                Outcome::Chosen(p) => {
                    self.browser = None;
                    match purpose {
                        Purpose::Open => self.open(p),
                        Purpose::Save => self.save(p),
                    }
                }
            }
        }
        if let Some(im) = &mut self.importer {
            match im.show(&ctx) {
                importer::Request::Nothing => {}
                importer::Request::Close => self.importer = None,
                importer::Request::Add(scene, name) => {
                    self.importer = None;
                    self.add_model(scene, name);
                }
            }
        }
    }

    fn menu(&mut self, ui: &mut egui::Ui) {
        egui::MenuBar::new().ui(ui, |ui| {
            ui.menu_button("File", |ui| {
                if ui.add(egui::Button::new("Open...").shortcut_text("Ctrl+O")).clicked() {
                    self.ask_open();
                }
                let save = ui.add(egui::Button::new("Save").shortcut_text("Ctrl+S"));
                if save.clicked() {
                    match self.path.clone() {
                        Some(p) => self.save(p),
                        None => self.ask_save(),
                    }
                }
                if ui.add(egui::Button::new("Save As...").shortcut_text("Ctrl+Shift+S")).clicked() {
                    self.ask_save();
                }
                ui.separator();
                if ui.add(egui::Button::new("Import...").shortcut_text("Ctrl+I")).clicked() {
                    self.import(None);
                }
                ui.separator();
                if ui.button("Quit").clicked() {
                    self.quit = true;
                }
            });
            ui.menu_button("Edit", |ui| {
                let some = self.can_edit();
                if ui.add_enabled(some, egui::Button::new("Copy").shortcut_text("Ctrl+C")).clicked() {
                    self.copy_selected();
                    ui.close();
                }
                if ui.add_enabled(self.clip.is_some(), egui::Button::new("Paste").shortcut_text("Ctrl+V")).clicked() {
                    self.paste();
                    ui.close();
                }
                ui.separator();
                if ui.add_enabled(some, egui::Button::new("Duplicate").shortcut_text("Ctrl+D")).clicked() {
                    self.duplicate();
                    ui.close();
                }
                ui.add_enabled(false, egui::Button::new("Duplicate Instanced").shortcut_text("Ctrl+Shift+D")).on_disabled_hover_text("Instances share one copy of the geometry: next to be built");
                ui.separator();
                if ui.add_enabled(some, egui::Button::new("Delete").shortcut_text("Del")).clicked() {
                    self.delete_selected();
                    ui.close();
                }
                ui.separator();
                if ui.add(egui::Button::new("Group").shortcut_text("Ctrl+G")).clicked() {
                    self.group_selected();
                }
                if ui.button("New empty group").clicked() {
                    self.new_group();
                }
                ui.separator();
                if ui.button("Add a light").clicked() {
                    self.add_light();
                }
                if ui.add(egui::Button::new("Reset the lights").shortcut_text("Ctrl+R")).clicked() {
                    self.app.rig = crate::lights::LightRig::new(crate::default_rig(&self.app.scene));
                    self.select(Selection::Nothing);
                }
            });
            ui.menu_button("View", |ui| {
                if ui.add(egui::Button::new("Frame the selection").shortcut_text("F")).clicked() {
                    self.app.frame_selection();
                }
                if ui.add(egui::Button::new("Frame the subject from afar").shortcut_text("A")).clicked() {
                    self.app.wide = true;
                    crate::frame_camera(&mut self.app.cam, &self.app.scene, true);
                }
                ui.separator();
                ui.checkbox(&mut self.navigate, "Navigation buttons");
                ui.checkbox(&mut self.app.rig.show, "Lights (L)");
                ui.checkbox(&mut self.app.settings.wireframe, "Wireframe (G)");
                ui.checkbox(&mut self.app.settings.cull_backfaces, "Back-face culling (B)");
            });
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.weak(format!("{}   {:.0} fps", self.app.scene_name, self.app.title_fps));
            });
        });
    }

    fn toolbar(&mut self, ui: &mut egui::Ui) {
        use style::icon;
        ui.horizontal(|ui| {
            for (tool, glyph, name) in [(manip::Tool::Select, icon::CURSOR, "Select (Q)"), (manip::Tool::Move, icon::MOVE, "Move (W)"), (manip::Tool::Rotate, icon::ROTATE, "Rotate (E)"), (manip::Tool::Scale, icon::SCALE, "Scale (R)")] {
                if ui.selectable_label(self.app.tool == tool, style::ic(glyph, 14.0)).on_hover_text(name).clicked() {
                    self.app.tool = tool;
                }
            }
            ui.separator();
            if ui.button(style::ic(icon::IMPORT, 14.0)).on_hover_text("Import a model (Ctrl+I)").clicked() {
                self.import(None);
            }
            if ui.button(style::ic(icon::FRAME, 14.0)).on_hover_text("Frame the selection (F)").clicked() {
                self.app.frame_selection();
            }
        });
    }

    fn status(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if let Some((text, error, at)) = &self.message {
                if at.elapsed().as_secs() < 12 {
                    if *error {
                        ui.colored_label(ui.visuals().error_fg_color, text);
                    } else {
                        ui.label(text);
                    }
                }
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let s = self.app.frame.stats;
                ui.weak(format!(
                    "{}  |  {}x{}  {} triangles drawn  {:.1} px micropolygons  cache {} MB  CPU only",
                    self.times.line(),
                    self.app.frame.width,
                    self.app.frame.height,
                    importer::thousands(s.tris_drawn),
                    self.app.settings.shading_rate_px * s.rate_scale.max(1.0),
                    s.cache_bytes >> 20
                ));
            });
        });
    }

    fn viewport(&mut self, ui: &mut egui::Ui) {
        let rect = ui.max_rect();
        let ppp = ui.ctx().pixels_per_point();
        let x = (rect.min.x * ppp).round().max(0.0) as usize;
        let y = (rect.min.y * ppp).round().max(0.0) as usize;
        let w = (rect.width() * ppp).round().max(0.0) as usize;
        let h = (rect.height() * ppp).round().max(0.0) as usize;
        self.view = (x, y, w, h);
        // The tool and what is selected, over the picture's corner.
        let what = match self.app.selection {
            Selection::Object(i) => self.app.scene.objects.get(i).map(|o| o.name.clone()).unwrap_or_default(),
            Selection::Light(i) => format!("light {}", i + 1),
            Selection::Nothing => "nothing selected".into(),
        };
        let tool = match self.app.tool {
            manip::Tool::Select => "select",
            manip::Tool::Move => "move",
            manip::Tool::Rotate => "rotate",
            manip::Tool::Scale => "scale",
        };
        let text = format!("{tool}   {what}   {:02}:{:02}", self.app.hours as u32, (self.app.hours.fract() * 60.0) as u32);
        let ctx = ui.ctx().clone();
        // Over the picture's corner: what it shows, then the tool and what
        // is selected.
        egui::Area::new(egui::Id::unique("viewport menu")).order(egui::Order::Foreground).fixed_pos(rect.min + egui::vec2(6.0, 4.0)).show(&ctx, |ui| {
            egui::Frame::NONE.fill(egui::Color32::from_black_alpha(150)).corner_radius(3.0).inner_margin(egui::Margin::symmetric(4, 2)).show(ui, |ui| {
                ui.horizontal(|ui| {
                    self.show_menu(ui);
                    ui.label(egui::RichText::new(text).color(egui::Color32::from_gray(230)));
                });
            });
        });
        self.navigation(&ctx, rect);
    }

    /// The viewport's Show menu: the picture, the light count, or a visual
    /// debug view, the views in menus by subject so the list never runs
    /// off the window.
    fn show_menu(&mut self, ui: &mut egui::Ui) {
        let name = match self.app.visual {
            Some(v) => v.title(),
            None if self.app.settings.view == crate::raster::View::LightCount => "light count".to_string(),
            None => "picture".to_string(),
        };
        let menu = ui.menu_button(format!("Show: {name}"), |ui| {
            if ui.selectable_label(self.app.visual.is_none() && self.app.settings.view == crate::raster::View::Shaded, "Picture").clicked() {
                self.app.visual = None;
                self.app.settings.view = crate::raster::View::Shaded;
                ui.close();
            }
            let lc = ui.selectable_label(self.app.settings.view == crate::raster::View::LightCount, "Light count");
            self.show_item = lc.rect;
            if lc.clicked() {
                self.app.visual = None;
                self.app.settings.view = crate::raster::View::LightCount;
                ui.close();
            }
            ui.separator();
            for group in crate::debug::GROUPS {
                ui.menu_button(group, |ui| {
                    for v in crate::debug::ALL.into_iter().filter(|v| v.group() == group && *v != crate::debug::DebugView::Final) {
                        if ui.selectable_label(self.app.visual == Some(v), v.title()).on_hover_text(format!("--debug-view {}", v.name())).clicked() {
                            self.app.visual = Some(v);
                            self.app.settings.view = crate::raster::View::Shaded;
                            ui.close();
                        }
                    }
                });
            }
        });
        self.show_rect = menu.response.rect;
    }
}

// ------------------------------------------------------------------ window

struct Runner {
    ed: Editor,
    window: Option<Rc<Window>>,
    surface: Option<softbuffer::Surface<Rc<Window>, Rc<Window>>>,
    context: Option<softbuffer::Context<Rc<Window>>>,
    state: Option<egui_winit::State>,
    cursor: (f32, f32),
    surface_size: (u32, u32),
}

impl ApplicationHandler for Runner {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes().with_title("FkJensen").with_inner_size(PhysicalSize::new(1600, 900));
        let window = match event_loop.create_window(attrs) {
            Ok(w) => Rc::new(w),
            Err(e) => {
                eprintln!("cannot open a window: {e}");
                event_loop.exit();
                return;
            }
        };
        let context = softbuffer::Context::new(window.clone()).expect("softbuffer context");
        let surface = softbuffer::Surface::new(&context, window.clone()).expect("softbuffer surface");
        let state = egui_winit::State::new(self.ed.ctx.clone(), egui::ViewportId::ROOT, &window, Some(window.scale_factor() as f32), None, None);
        self.state = Some(state);
        self.window = Some(window);
        self.context = Some(context);
        self.surface = Some(surface);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let (Some(window), Some(state)) = (self.window.clone(), self.state.as_mut()) else { return };
        let _ = state.on_window_event(&window, &event);
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                let raw = state.take_egui_input(&window);
                let size = window.inner_size();
                let (w, h) = (size.width.max(1), size.height.max(1));
                let out = self.ed.frame(raw, w as usize, h as usize);
                if let Some(state) = self.state.as_mut() {
                    state.handle_platform_output(&window, out);
                }
                self.ed.app.title_fps = self.ed.app.title_fps.max(0.0);
                if let Some(surface) = self.surface.as_mut() {
                    if self.surface_size != (w, h) {
                        let _ = surface.resize(NonZeroU32::new(w).unwrap(), NonZeroU32::new(h).unwrap());
                        self.surface_size = (w, h);
                    }
                    let t = Instant::now();
                    if let Ok(mut buf) = surface.buffer_mut() {
                        let n = buf.len().min(self.ed.buf.len());
                        buf[..n].copy_from_slice(&self.ed.buf[..n]);
                        let _ = buf.present();
                    }
                    self.ed.times.present = t.elapsed().as_secs_f32() * 1000.0;
                }
                if self.ed.quit {
                    event_loop.exit();
                }
                window.request_redraw();
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor = (position.x as f32, position.y as f32);
                self.ed.view_cursor(self.cursor.0, self.cursor.1);
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let idx = match button {
                    MouseButton::Left => 0,
                    MouseButton::Middle => 1,
                    MouseButton::Right => 2,
                    _ => return,
                };
                self.ed.view_button(idx, state == ElementState::Pressed, self.cursor.0, self.cursor.1);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let steps = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 40.0,
                };
                self.ed.view_wheel(steps, self.cursor.0, self.cursor.1);
            }
            WindowEvent::KeyboardInput { event: KeyEvent { physical_key: PhysicalKey::Code(code), state, repeat, .. }, .. } => {
                if !repeat {
                    self.ed.view_key(code, state == ElementState::Pressed);
                }
            }
            WindowEvent::Focused(false) => {
                // Keys let go while the window was away would stay down.
                for code in self.ed.app.keys.clone() {
                    self.ed.app.key(code, false);
                }
            }
            WindowEvent::DroppedFile(path) => self.ed.dropped(path),
            _ => {}
        }
    }
}

/// Run the editor in a window until it is closed.
pub fn run(app: App, scene_path: Option<PathBuf>) {
    let mut ed = Editor::new(app);
    ed.path = scene_path;
    let event_loop = match EventLoop::new() {
        Ok(e) => e,
        Err(e) => {
            eprintln!("cannot open a window: {e}");
            std::process::exit(1);
        }
    };
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut r = Runner { ed, window: None, surface: None, context: None, state: None, cursor: (0.0, 0.0), surface_size: (0, 0) };
    if let Err(e) = event_loop.run_app(&mut r) {
        eprintln!("{e}");
    }
}

/// Draw the editor without a window, `w` x `h`, and write it to `out`
/// (for documentation and checks). With `import`, the importer is open on
/// that model, shown once it has baked.
/// `bench_move`: then move the first object 10 cm along X five times, as a
/// drag would, and let go, printing each frame's times.
pub fn shot(app: App, w: usize, h: usize, out: &str, import: Option<PathBuf>, bench_move: bool) -> Result<(), String> {
    let mut ed = Editor::new(app);
    let raw = || egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(w as f32, h as f32))), ..Default::default() };
    if let Some(m) = import {
        ed.import(Some(m));
        let t = Instant::now();
        while ed.importer.as_ref().is_some_and(|i| i.busy()) && t.elapsed().as_secs() < 1800 {
            ed.frame(raw(), w, h);
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    }
    // A few frames: layout settles, fonts arrive, the importer's preview
    // renders.
    for _ in 0..4 {
        ed.frame(raw(), w, h);
    }
    if let Some(m) = ed.app.scene.first_model().filter(|_| bench_move) {
        let line = |ed: &Editor, what: &str| {
            let s = ed.app.frame.stats;
            eprintln!("{what}: {} (layout and paint at {w} x {h}); renderer: visibility {:.0}, shade {:.0}, resolve {:.0}, shadows {:.0}, layer {:.0} ms{}", ed.times.line(), s.setup_ms + s.bin_ms + s.raster_ms, s.shade_ms, s.resolve_ms, s.shadow_ms, s.layer_ms, if s.layer_captured { " (captured)" } else { "" });
        };
        ed.select(Selection::Object(m));
        ed.frame(raw(), w, h);
        line(&ed, "still");
        for step in 1..=5 {
            let o = ed.app.scene.objects[m].clone();
            ed.app.move_object(m, o.pivot + glam::Vec3::new(0.1, 0.0, 0.0), o.rotation, o.scale);
            ed.frame(raw(), w, h);
            line(&ed, &format!("move step {step}"));
        }
        ed.app.object_settled(m);
        for k in 1..=60 {
            ed.frame(raw(), w, h);
            let baking = ed.app.frame.stats.baking;
            line(&ed, &format!("let go, frame {k}{}", if baking { " (baking)" } else { "" }));
            if !baking && k > 2 {
                break;
            }
        }
    }
    let frame = crate::raster::Frame { width: w, height: h, color: ed.buf.clone(), ..crate::raster::Frame::new(1, 1) };
    crate::save_png(&frame, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn editor() -> Editor {
        let mut ed = Editor::new(crate::tests::ball_app());
        ed.app.point_budget = None;
        ed.app.render_scale = 1.0;
        ed.set_frame_size(FrameSize::Viewport);
        ed
    }

    fn raw(w: usize, h: usize) -> egui::RawInput {
        egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(w as f32, h as f32))), ..Default::default() }
    }

    /// The layout puts the viewport between the panels, the frame is sized
    /// to it, and clicking the ball there selects it and shows it in the
    /// attribute editor.
    #[test]
    fn viewport_sits_between_the_panels_and_takes_the_mouse() {
        let mut ed = editor();
        let (w, h) = (1100, 700);
        for _ in 0..2 {
            ed.frame(raw(w, h), w, h);
        }
        let (vx, vy, vw, vh) = ed.view;
        assert!(vx >= 150 && vy >= 30 && vx + vw <= w - 200 && vy + vh <= h - 50, "{:?}", ed.view);
        assert_eq!((ed.app.frame.width, ed.app.frame.height), (vw, vh));
        // The ball is in the middle of the picture.
        let (cx, cy) = (ed.app.frame.width / 2, ed.app.frame.height / 2);
        let c = ed.app.frame.color[cy * ed.app.frame.width + cx];
        assert_eq!(ed.buf[(vy + cy) * w + vx + cx], c, "the frame is in the window at the viewport");
        ed.app.selection = Selection::Nothing;
        ed.inspect = Inspect::Sky;
        let (mx, my) = ((vx + vw / 2) as f32, (vy + vh / 2) as f32);
        ed.view_cursor(mx, my);
        ed.view_button(0, true, mx, my);
        ed.view_button(0, false, mx, my);
        assert_eq!(ed.app.selection, Selection::Object(0));
        assert_eq!(ed.inspect, Inspect::Selection);
        // Over a panel the viewport takes nothing.
        ed.app.selection = Selection::Nothing;
        ed.view_button(0, true, 20.0, 300.0);
        ed.view_button(0, false, 20.0, 300.0);
        assert_eq!(ed.app.selection, Selection::Nothing);
    }

    /// The navigation buttons move the camera: dragging the tumble pad
    /// turns it, holding In brings it closer, All frames the subject.
    #[test]
    fn navigation_buttons_move_the_camera() {
        let mut ed = editor();
        let (w, h) = (1100, 700);
        for _ in 0..2 {
            ed.frame(raw(w, h), w, h);
        }
        let find = |ed: &Editor, name: &str| ed.nav_rects.iter().find(|(n, _)| *n == name).map(|(_, r)| r.center()).unwrap_or_else(|| panic!("no {name}"));
        let ev = |events: Vec<egui::Event>| egui::RawInput { events, ..raw(w, h) };
        let press = |p: egui::Pos2, down: bool| egui::Event::PointerButton { pos: p, button: egui::PointerButton::Primary, pressed: down, modifiers: Default::default() };
        // Drag the tumble pad 40 points to the right.
        let yaw = ed.app.cam.yaw;
        let p = find(&ed, "Tumble");
        ed.frame(ev(vec![egui::Event::PointerMoved(p), press(p, true)]), w, h);
        for k in 1..=4 {
            ed.frame(ev(vec![egui::Event::PointerMoved(p + egui::vec2(10.0 * k as f32, 0.0))]), w, h);
        }
        ed.frame(ev(vec![press(p + egui::vec2(40.0, 0.0), false)]), w, h);
        assert!(ed.app.cam.yaw < yaw - 0.1, "{} then {}", yaw, ed.app.cam.yaw);
        // Hold In.
        let d = ed.app.cam.distance;
        let p = find(&ed, "In");
        ed.frame(ev(vec![egui::Event::PointerMoved(p), press(p, true)]), w, h);
        for _ in 0..3 {
            std::thread::sleep(std::time::Duration::from_millis(20));
            ed.frame(ev(vec![]), w, h);
        }
        ed.frame(ev(vec![press(p, false)]), w, h);
        assert!(ed.app.cam.distance < d, "{d} then {}", ed.app.cam.distance);
        // All: framed from afar again.
        let p = find(&ed, "All");
        ed.frame(ev(vec![egui::Event::PointerMoved(p), press(p, true)]), w, h);
        ed.frame(ev(vec![press(p, false)]), w, h);
        let mut wide = crate::camera::OrbitCamera::new();
        crate::frame_camera(&mut wide, &ed.app.scene, true);
        assert!((ed.app.cam.distance - wide.distance).abs() < 1e-3 && ed.app.cam.yaw.abs() < 1e-6);
        // The viewport under the buttons is not clicked through.
        assert_eq!(ed.app.selection, Selection::Nothing);
    }

    /// The Show menu sits over the viewport: it opens there and sets what
    /// the viewport shows.
    #[test]
    fn the_show_menu_is_over_the_viewport() {
        let mut ed = editor();
        let (w, h) = (1100, 700);
        for _ in 0..2 {
            ed.frame(raw(w, h), w, h);
        }
        let (vx, vy, vw, _) = ed.view;
        let ppp = 1.0;
        let r = ed.show_rect;
        assert!(r.min.x * ppp >= vx as f32 && r.min.y * ppp >= vy as f32 && r.max.x * ppp < (vx + vw) as f32, "{r:?} in {:?}", ed.view);
        let ev = |events: Vec<egui::Event>| egui::RawInput { events, ..raw(w, h) };
        let press = |p: egui::Pos2, down: bool| egui::Event::PointerButton { pos: p, button: egui::PointerButton::Primary, pressed: down, modifiers: Default::default() };
        let c = r.center();
        ed.frame(ev(vec![egui::Event::PointerMoved(c), press(c, true)]), w, h);
        ed.frame(ev(vec![press(c, false)]), w, h);
        ed.frame(ev(vec![]), w, h);
        let p = ed.show_item.center();
        assert!(p.y > r.max.y, "the menu opened under the button: {:?}", ed.show_item);
        ed.frame(ev(vec![egui::Event::PointerMoved(p), press(p, true)]), w, h);
        ed.frame(ev(vec![press(p, false)]), w, h);
        let picked = ed.app.settings.view == crate::raster::View::LightCount;
        assert!(picked, "Light count not picked from the viewport's Show menu");
        assert_eq!(ed.app.selection, Selection::Nothing, "the click did not go through to the viewport");
    }

    /// Edit: duplicate an object (and its group), copy and paste a light,
    /// delete an object with what is under it; the scene stays whole.
    #[test]
    fn edit_duplicates_copies_and_deletes() {
        let mut ed = editor();
        ed.select(Selection::Object(0));
        ed.group_selected();
        let Selection::Object(g) = ed.app.selection else { panic!() };
        let tris = ed.app.scene.triangle_count();
        // Duplicate the group: a new group with a new ball under it.
        ed.duplicate();
        let Selection::Object(g2) = ed.app.selection else { panic!() };
        assert_ne!(g2, g);
        assert_eq!(ed.app.scene.objects.len(), 4);
        let kids: Vec<usize> = (0..4).filter(|&k| ed.app.scene.objects[k].parent == Some(g2)).collect();
        assert_eq!(kids.len(), 1);
        assert_eq!(ed.app.scene.triangle_count(), tris + ed.app.scene.objects[0].clusters.clone().map(|c| ed.app.scene.clusters[c].tri_count as usize).sum::<usize>());
        assert!(ed.app.scene.objects[kids[0]].pivot.distance(ed.app.scene.objects[0].pivot) < 1e-5);
        // A light: copy, paste.
        let n = ed.app.rig.lights.len();
        ed.select(Selection::Light(0));
        ed.copy_selected();
        ed.paste();
        assert_eq!(ed.app.rig.lights.len(), n + 1);
        assert_eq!(ed.app.rig.lights[n].position, ed.app.rig.lights[0].position);
        // Delete the first group: it and its ball go; the copy stays and
        // still draws.
        ed.select(Selection::Object(g));
        ed.delete_selected();
        assert_eq!(ed.app.scene.objects.len(), 2);
        assert_eq!(ed.app.scene.triangle_count(), tris);
        let (w, h) = (640, 400);
        for _ in 0..2 {
            ed.frame(raw(w, h), w, h);
        }
        assert!(ed.app.frame.stats.tris_drawn > 0);
        // Delete the light pasted.
        ed.select(Selection::Light(n));
        ed.delete_selected();
        assert_eq!(ed.app.rig.lights.len(), n);
    }

    /// At 1920 x 1080 the frame keeps its size as the window changes, is
    /// shown as large as fits with bars round it, and the pointer still
    /// lands on what is under it.
    #[test]
    fn a_1080p_frame_keeps_its_size_and_takes_the_mouse() {
        let mut ed = editor();
        ed.set_frame_size(FrameSize::Hd);
        for (w, h) in [(1100, 700), (1400, 600)] {
            for _ in 0..2 {
                ed.frame(raw(w, h), w, h);
            }
            assert_eq!((ed.app.frame.width, ed.app.frame.height), (1920, 1080));
            let (vx, vy, vw, vh) = ed.view;
            let (sx, sy, sw, sh) = ed.app.shown();
            assert!(sx + sw <= vw && sy + sh <= vh && (sw == vw || sh == vh), "{:?} in {:?}", (sx, sy, sw, sh), ed.view);
            assert!(((sw as f32 / sh as f32) - 16.0 / 9.0).abs() < 0.02);
            ed.app.selection = Selection::Nothing;
            let (mx, my) = ((vx + sx + sw / 2) as f32, (vy + sy + sh / 2) as f32);
            ed.view_cursor(mx, my);
            ed.view_button(0, true, mx, my);
            ed.view_button(0, false, mx, my);
            assert_eq!(ed.app.selection, Selection::Object(0), "window {w} x {h}");
        }
        ed.set_frame_size(FrameSize::Custom(640, 480));
        ed.frame(raw(1100, 700), 1100, 700);
        assert_eq!((ed.app.frame.width, ed.app.frame.height), (640, 480));
    }

    /// Moving an object from the attribute editor moves its group's
    /// children; saving writes an .fjs and an .fja for the ball, and
    /// opening it keeps the objects, the groups, the lights, the time of
    /// day, the camera and the render settings.
    #[test]
    fn groups_save_and_open() {
        let mut ed = editor();
        ed.select(Selection::Object(0));
        ed.group_selected();
        let Selection::Object(g) = ed.app.selection else { panic!() };
        assert_eq!(ed.app.scene.objects[0].parent, Some(g));
        let before = ed.app.scene.objects[0].pivot;
        ed.app.scene.set_object(g, glam::Vec3::new(1.0, 0.0, 0.0), glam::Vec3::ZERO, 1.0);
        assert!((ed.app.scene.objects[0].pivot - before - glam::Vec3::X).length() < 1e-5);
        ed.app.hours = 15.25;
        ed.app.apply_daylight();
        ed.app.settings.adaptive = false;
        ed.app.settings.dir_lights[1] = crate::raster::DirLight { on: true, dir: crate::raster::DirLight::towards(100.0, 30.0), color: glam::Vec3::new(1.0, 0.5, 0.25), lux: 5000.0 };
        ed.app.cam.yaw = 0.7;
        let dir = std::env::temp_dir().join(format!("fkj-editor-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("groups.fjs");
        ed.save(path.clone());
        assert!(path.is_file(), "{:?}", ed.message);
        assert!(dir.join("ball.fja").is_file(), "{:?}", std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().file_name()).collect::<Vec<_>>());
        let mut other = editor();
        other.open(path.clone());
        let s = &other.app.scene;
        assert_eq!(s.objects.len(), 2, "{:?}", other.message);
        let ball = s.objects.iter().position(|o| o.name == "ball").unwrap();
        let group = s.objects.iter().position(|o| o.name == "ball_group").unwrap();
        assert_eq!(s.objects[ball].parent, Some(group));
        assert!(s.objects[ball].pivot.distance(ed.app.scene.objects[0].pivot) < 1e-4);
        assert_eq!(other.app.rig.lights.len(), ed.app.rig.lights.len());
        assert!((other.app.hours - 15.25).abs() < 1e-4);
        assert!(!other.app.settings.adaptive);
        assert!(other.app.settings.dir_lights[0].on && (other.app.settings.dir_lights[0].lux - 5000.0).abs() < 1e-2);
        assert!((other.app.cam.yaw - 0.7).abs() < 1e-4);
        assert!((other.app.settings.exposure - ed.app.settings.exposure).abs() <= ed.app.settings.exposure * 1e-5);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The timeline plays the day: the sun and the sky follow.
    #[test]
    fn the_timeline_plays_the_day() {
        let mut ed = editor();
        ed.app.hours = 12.0;
        ed.app.apply_daylight();
        let noon = ed.app.settings.sun.dir;
        ed.playing = true;
        ed.speed = 3.0;
        ed.app.last_frame = Instant::now() - std::time::Duration::from_millis(100);
        ed.frame(raw(640, 400), 640, 400);
        assert!(ed.app.hours > 12.2, "{}", ed.app.hours);
        assert_ne!(ed.app.settings.sun.dir, noon);
    }

    /// A model from the importer joins the scene as one object, selected.
    #[test]
    fn imported_models_join_the_scene() {
        let mut ed = editor();
        let mut m = crate::scene::Scene::new();
        let (p, idx) = crate::scene::tests::uv_sphere(12);
        m.push_mesh(&p, &[], &[], &idx);
        let before = ed.app.scene.clusters.len();
        ed.add_model(m, "sphere".into());
        assert_eq!(ed.app.scene.objects.len(), 2);
        assert_eq!(ed.app.scene.objects[1].name, "sphere");
        assert_eq!(ed.app.scene.objects[1].clusters.start, before);
        assert_eq!(ed.app.selection, Selection::Object(1));
        ed.frame(raw(800, 500), 800, 500);
    }

    /// The editor draws a whole frame: panels, text and the viewport.
    #[test]
    fn draws_a_frame() {
        let mut ed = editor();
        let (w, h) = std::env::var("FKJ_EDITOR_SIZE").ok().and_then(|s| s.split_once('x').map(|(a, b)| (a.parse().unwrap(), b.parse().unwrap()))).unwrap_or((900, 600));
        for _ in 0..3 {
            ed.frame(raw(w, h), w, h);
        }
        let distinct: std::collections::HashSet<u32> = ed.buf.iter().copied().collect();
        assert!(distinct.len() > 50, "{} colours", distinct.len());
        if let Ok(p) = std::env::var("FKJ_EDITOR_SHOT") {
            let save = |ed: &Editor, name: &str| {
                let frame = crate::raster::Frame { width: w, height: h, color: ed.buf.clone(), ..crate::raster::Frame::new(1, 1) };
                crate::save_png(&frame, &p.replace(".png", name)).unwrap();
            };
            save(&ed, ".png");
            ed.select(Selection::Object(0));
            ed.group_selected();
            ed.select(Selection::Object(0));
            ed.frame(raw(w, h), w, h);
            ed.frame(raw(w, h), w, h);
            save(&ed, "_object.png");
            ed.inspect = Inspect::Render;
            ed.frame(raw(w, h), w, h);
            save(&ed, "_render.png");
            ed.ask_open();
            ed.frame(raw(w, h), w, h);
            ed.frame(raw(w, h), w, h);
            save(&ed, "_browser.png");
        }
    }
}
