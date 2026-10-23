//! The outliner, the attribute editor and the timeline.

use glam::Vec3;

use super::importer::{metres, thousands};
use super::{Editor, Inspect};
use crate::lights::{self, LIGHT_COLOURS};
use crate::{raster, scene, Selection};

/// What a row of the outliner asked for, done once the tree is drawn.
enum Action {
    Select(Selection),
    Inspect(Inspect),
    /// Put object .0 under .1 (None: at the top).
    Parent(usize, Option<usize>),
    Frame(usize),
}

fn hhmm(h: f32) -> String {
    let m = (h.rem_euclid(24.0) * 60.0).round() as u32 % (24 * 60);
    format!("{:02}:{:02}", m / 60, m % 60)
}

impl Editor {
    pub(super) fn add_light(&mut self) {
        let c = self.app.cam.target;
        let r = self.app.scene.focus_radius();
        let at = c + Vec3::new(0.0, r * 0.8, 0.0);
        if self.app.rig.add(at, 800.0) {
            let i = self.app.rig.lights.len() - 1;
            self.app.rig.show = true;
            self.select(Selection::Light(i));
        } else {
            self.say(format!("the rig holds {} lights at most", lights::MAX_LIGHTS), true);
        }
    }

    pub(super) fn delete_light(&mut self) {
        if let Selection::Light(i) = self.app.selection {
            self.app.rig.selected = i;
            self.app.drag = None;
            self.app.manip = None;
            self.app.rig.remove_selected();
            self.select(Selection::Nothing);
        }
    }

    // ------------------------------------------------------------ outliner

    pub(super) fn outliner(&mut self, ui: &mut egui::Ui) {
        ui.add_space(2.0);
        ui.strong("Outliner");
        ui.separator();
        let mut actions = Vec::new();
        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            let objects = egui::CollapsingHeader::new("Objects").default_open(true).show(ui, |ui| {
                let top = self.app.scene.children(None);
                if top.is_empty() {
                    ui.weak("none: File > Import adds one");
                }
                for i in top {
                    self.object_row(ui, i, &mut actions);
                }
            });
            // Dropped on the heading: out of any group.
            if let Some(p) = objects.header_response.dnd_release_payload::<usize>() {
                actions.push(Action::Parent(*p, None));
            }
            egui::CollapsingHeader::new("Lights").default_open(true).show(ui, |ui| {
                for (i, l) in self.app.rig.lights.iter().enumerate() {
                    let sel = self.app.selection == Selection::Light(i) && self.inspect == Inspect::Selection;
                    let r = ui.selectable_label(sel, format!("light {}   {} lm", i + 1, thousands(l.lumens.round() as usize)));
                    if r.clicked() {
                        actions.push(Action::Select(Selection::Light(i)));
                    }
                }
                if ui.small_button("+ light").clicked() {
                    actions.push(Action::Inspect(Inspect::Selection));
                    self.add_light();
                }
            });
            ui.separator();
            for (what, name) in [(Inspect::Sun, "Sun"), (Inspect::Sky, "Sky and exposure"), (Inspect::Render, "Render settings")] {
                if ui.selectable_label(self.inspect == what, name).clicked() {
                    actions.push(Action::Inspect(what));
                }
            }
        });
        for a in actions {
            match a {
                Action::Select(s) => self.select(s),
                Action::Inspect(i) => self.inspect = i,
                Action::Parent(child, parent) => {
                    // Keep where it is in the world: only the hierarchy
                    // changes.
                    if !self.app.scene.set_parent(child, parent) {
                        self.say("an object cannot go under itself", true);
                    }
                }
                Action::Frame(i) => {
                    self.select(Selection::Object(i));
                    self.app.frame_selection();
                }
            }
        }
    }

    fn object_row(&self, ui: &mut egui::Ui, i: usize, actions: &mut Vec<Action>) {
        let scene = &self.app.scene;
        let o = &scene.objects[i];
        let kids = scene.children(Some(i));
        let sel = self.app.selection == Selection::Object(i) && self.inspect == Inspect::Selection;
        let text = if o.clusters.is_empty() { egui::RichText::new(&o.name).italics() } else { egui::RichText::new(&o.name) };
        let mut row = |ui: &mut egui::Ui| {
            let r = ui.selectable_label(sel, text.clone()).interact(egui::Sense::drag());
            if r.drag_started() {
                r.dnd_set_drag_payload(i);
            }
            if r.dnd_hover_payload::<usize>().is_some_and(|p| *p != i) {
                ui.painter().rect_stroke(r.rect, 2.0, egui::Stroke::new(1.5, ui.visuals().selection.stroke.color), egui::StrokeKind::Inside);
            }
            if let Some(p) = r.dnd_release_payload::<usize>() {
                if *p != i {
                    actions.push(Action::Parent(*p, Some(i)));
                }
            }
            if r.clicked() {
                actions.push(Action::Select(Selection::Object(i)));
            }
            if r.double_clicked() {
                actions.push(Action::Frame(i));
            }
            r.context_menu(|ui| {
                if ui.button("Frame").clicked() {
                    actions.push(Action::Frame(i));
                    ui.close();
                }
                if o.parent.is_some() && ui.button("Out of its group").clicked() {
                    actions.push(Action::Parent(i, None));
                    ui.close();
                }
            });
        };
        if kids.is_empty() {
            row(ui);
        } else {
            let id = ui.make_persistent_id(("outliner", i));
            egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, true).show_header(ui, |ui| row(ui)).body(|ui| {
                for k in kids {
                    self.object_row(ui, k, actions);
                }
            });
        }
    }

    // ---------------------------------------------------------- attributes

    pub(super) fn attributes(&mut self, ui: &mut egui::Ui) {
        ui.add_space(2.0);
        ui.strong("Attribute Editor");
        ui.separator();
        match (self.inspect, self.app.selection) {
            (Inspect::Selection, Selection::Object(i)) if i < self.app.scene.objects.len() => self.object_attributes(ui, i),
            (Inspect::Selection, Selection::Light(i)) if i < self.app.rig.lights.len() => self.light_attributes(ui, i),
            (Inspect::Sun, _) => self.sun_attributes(ui),
            (Inspect::Sky, _) => self.sky_attributes(ui),
            (Inspect::Render, _) => self.render_attributes(ui),
            _ => {
                ui.weak("Select an object or a light, in the viewport or the outliner.");
            }
        }
    }

    fn object_attributes(&mut self, ui: &mut egui::Ui, i: usize) {
        let group = self.app.scene.is_group(i);
        ui.horizontal(|ui| {
            ui.label(if group { "group" } else { "object" });
            ui.text_edit_singleline(&mut self.app.scene.objects[i].name);
        });
        ui.add_space(4.0);
        let o = self.app.scene.objects[i].clone();
        let (mut p, mut r, mut k) = (o.pivot, o.rotation * (180.0 / std::f32::consts::PI), o.scale);
        let speed = (self.app.scene.focus_radius() as f64 * 0.004).max(0.0005);
        let mut changed = false;
        egui::CollapsingHeader::new("Transform").default_open(true).show(ui, |ui| {
            egui::Grid::new("transform").num_columns(4).show(ui, |ui| {
                ui.label("Translate");
                for v in [&mut p.x, &mut p.y, &mut p.z] {
                    changed |= ui.add(egui::DragValue::new(v).speed(speed).max_decimals(3)).changed();
                }
                ui.end_row();
                ui.label("Rotate");
                for v in [&mut r.x, &mut r.y, &mut r.z] {
                    changed |= ui.add(egui::DragValue::new(v).speed(0.5).max_decimals(2).suffix("°")).changed();
                }
                ui.end_row();
                ui.label("Scale");
                changed |= ui.add(egui::DragValue::new(&mut k).speed(0.005).range(0.001..=1000.0).max_decimals(3)).changed();
                ui.end_row();
            });
            ui.weak("Rotates X, then Y, then Z. Scales the same on every axis, so the texture tiles stay valid.");
        });
        if changed {
            self.app.move_object(i, p, r * (std::f32::consts::PI / 180.0), k);
            self.pending_bounce = Some(i);
        }
        let (lo, hi) = self.app.scene.object_bounds(i);
        let s = &self.app.scene;
        egui::CollapsingHeader::new("Information").default_open(true).show(ui, |ui| {
            egui::Grid::new("object info").num_columns(2).show(ui, |ui| {
                if let Some(pa) = o.parent {
                    ui.label("Group");
                    ui.label(&s.objects[pa].name);
                    ui.end_row();
                }
                if group {
                    ui.label("Holds");
                    ui.label(format!("{} objects", s.descendants(i).len()));
                    ui.end_row();
                } else {
                    let cs = &s.clusters[o.clusters.clone()];
                    let finest = cs.iter().map(|c| c.lod_level).min().unwrap_or(0);
                    let tris: usize = cs.iter().filter(|c| c.lod_level == finest).map(|c| c.tri_count as usize).sum();
                    let levels = cs.iter().map(|c| c.lod_level).max().unwrap_or(0) - finest + 1;
                    let tiled = o.clusters.clone().filter(|&c| s.textures.entry(c).level > 0).count();
                    ui.label("Triangles");
                    ui.label(thousands(tris));
                    ui.end_row();
                    ui.label("Levels of detail");
                    ui.label(levels.to_string());
                    ui.end_row();
                    ui.label("Clusters");
                    ui.label(format!("{} ({} with texture tiles)", thousands(cs.len()), thousands(tiled)));
                    ui.end_row();
                }
                let d = hi - lo;
                ui.label("Size");
                ui.label(format!("{} x {} x {}", metres(d.x), metres(d.y), metres(d.z)));
                ui.end_row();
            });
        });
    }

    fn light_attributes(&mut self, ui: &mut egui::Ui, i: usize) {
        ui.label(format!("light {}", i + 1));
        ui.add_space(4.0);
        let speed = (self.app.scene.focus_radius() as f64 * 0.004).max(0.0005);
        let cut = raster::cut_lux(&self.app.settings.sky, self.app.settings.exposure);
        let reach_on = self.app.settings.light_reach;
        let l = &mut self.app.rig.lights[i];
        egui::Grid::new("light").num_columns(4).show(ui, |ui| {
            ui.label("Position");
            for v in [&mut l.position.x, &mut l.position.y, &mut l.position.z] {
                ui.add(egui::DragValue::new(v).speed(speed).max_decimals(3));
            }
            ui.end_row();
            ui.label("Lumens");
            ui.add(egui::Slider::new(&mut l.lumens, scene::MIN_LUMENS..=scene::MAX_LUMENS).logarithmic(true).max_decimals(0));
            ui.end_row();
            ui.label("Colour");
            let mut c = l.color.to_array();
            if egui::color_picker::color_edit_button_rgb(ui, &mut c).changed() {
                l.color = lights::normalise_colour(Vec3::from_array(c));
            }
            ui.end_row();
        });
        ui.horizontal_wrapped(|ui| {
            for c in LIGHT_COLOURS {
                let n = lights::normalise_colour(c);
                let show = (n / n.max_element()).min(Vec3::ONE);
                let col = egui::Color32::from_rgb((show.x * 255.0) as u8, (show.y * 255.0) as u8, (show.z * 255.0) as u8);
                if ui.add(egui::Button::new("").fill(col).min_size(egui::vec2(22.0, 16.0))).clicked() {
                    l.color = n;
                }
            }
        });
        if reach_on {
            ui.weak(format!("reaches {} (where it falls under what the exposure shows)", crate::format_metres(raster::light_reach(l, cut))));
        }
        ui.add_space(6.0);
        if ui.button("Delete the light").clicked() {
            self.delete_light();
        }
    }

    fn sun_attributes(&mut self, ui: &mut egui::Ui) {
        ui.label("Sun");
        ui.add_space(4.0);
        let mut h = self.app.hours;
        egui::Grid::new("sun").num_columns(2).show(ui, |ui| {
            ui.label("Time of day");
            if ui.add(egui::Slider::new(&mut h, 0.0..=24.0).custom_formatter(|v, _| hhmm(v as f32)).custom_parser(|s| crate::parse_hours(s).map(|h| h as f64))).changed() {
                self.app.hours = h.rem_euclid(24.0);
                self.app.apply_daylight();
            }
            ui.end_row();
            ui.label("Brightness");
            ui.add(egui::Slider::new(&mut self.app.settings.sun.brightness, 0.0..=raster::SUN_MAX).max_decimals(0));
            ui.end_row();
            let e = self.app.settings.sun.dir.y.clamp(-1.0, 1.0).asin().to_degrees();
            ui.label("Elevation");
            ui.label(if e > 0.0 { format!("{e:.1}°") } else { format!("{e:.1}°, set") });
            ui.end_row();
        });
        ui.weak("The timeline at the bottom plays the day. Time sets the sun, the sky and the exposure.");
    }

    fn sky_attributes(&mut self, ui: &mut egui::Ui) {
        ui.label("Sky and exposure");
        ui.add_space(4.0);
        let s = &mut self.app.settings;
        egui::Grid::new("sky").num_columns(2).show(ui, |ui| {
            ui.label("Sky");
            ui.checkbox(&mut s.sky.on, "on");
            ui.end_row();
            ui.label("Illuminance");
            ui.add(egui::Slider::new(&mut s.sky.lux, 0.01..=120_000.0).logarithmic(true).suffix(" lux").max_decimals(2));
            ui.end_row();
            ui.label("Lighting contrast");
            ui.add(egui::Slider::new(&mut s.sky.contrast, 0.0..=1.0)).on_hover_text("0 flat overcast, 1 from overhead only");
            ui.end_row();
            ui.label("Sky contrast");
            ui.add(egui::Slider::new(&mut s.backdrop.contrast, 0.0..=1.0)).on_hover_text("the visible sky's gradient and sun glow");
            ui.end_row();
            ui.label("Exposure");
            ui.add(egui::Slider::new(&mut s.exposure, 1e-6..=1.0).logarithmic(true).max_decimals(6));
            ui.end_row();
        });
        if ui.button("Exposure from the sky").clicked() {
            s.exposure = raster::default_exposure(s.sky.lux);
        }
    }

    fn render_attributes(&mut self, ui: &mut egui::Ui) {
        ui.label("Render settings");
        ui.add_space(4.0);
        let mut refit = false;
        egui::Grid::new("render").num_columns(2).show(ui, |ui| {
            ui.label("Micropolygons");
            if ui.add(egui::Slider::new(&mut self.app.settings.shading_rate_px, 0.5..=16.0).logarithmic(true).suffix(" px apart").max_decimals(2)).changed() {
                self.app.point_budget = None;
            }
            ui.end_row();
            ui.label("Resolution");
            let mut pct = self.app.render_scale * 100.0;
            if ui.add(egui::Slider::new(&mut pct, 6.25..=100.0).suffix(" %").max_decimals(0)).changed() {
                self.app.render_scale = pct / 100.0;
                self.app.point_budget = None;
                refit = true;
            }
            ui.end_row();
            ui.label("");
            if ui.add_enabled(self.plan_budget.is_some() && self.app.point_budget.is_none(), egui::Button::new("Fit to this machine")).clicked() {
                self.app.point_budget = self.plan_budget;
                refit = true;
            }
            ui.end_row();
            ui.label("Shadows");
            ui.checkbox(&mut self.app.settings.shadows, "");
            ui.end_row();
            ui.label("Shadow map");
            egui::ComboBox::from_id_salt("shadow res").selected_text(format!("{} texels", self.app.settings.shadow_res)).show_ui(ui, |ui| {
                for r in [128, 256, 512, 1024, 2048] {
                    ui.selectable_value(&mut self.app.settings.shadow_res, r, format!("{r} texels"));
                }
            });
            ui.end_row();
            ui.label("Bounce light");
            ui.checkbox(&mut self.app.settings.bounce, "");
            ui.end_row();
            ui.label("Light reach");
            ui.checkbox(&mut self.app.settings.light_reach, "");
            ui.end_row();
            ui.label("Occlusion culling");
            ui.checkbox(&mut self.app.settings.occlusion, "");
            ui.end_row();
            ui.label("Adaptive shading");
            ui.checkbox(&mut self.app.settings.adaptive, "");
            ui.end_row();
            ui.label("Sky occlusion");
            ui.checkbox(&mut self.app.settings.sky_occlusion, "");
            ui.end_row();
            ui.label("Moves relight their shadows only");
            ui.checkbox(&mut self.app.settings.incremental_moves, "");
            ui.end_row();
            ui.label("Back-face culling");
            ui.checkbox(&mut self.app.settings.cull_backfaces, "");
            ui.end_row();
            ui.label("Wireframe");
            ui.checkbox(&mut self.app.settings.wireframe, "");
            ui.end_row();
            ui.label("Show");
            let name = match self.app.visual {
                Some(v) => v.name(),
                None if self.app.settings.view == raster::View::LightCount => "light count",
                None => "picture",
            };
            egui::ComboBox::from_id_salt("debug view").selected_text(name).show_ui(ui, |ui| {
                if ui.selectable_label(self.app.visual.is_none() && self.app.settings.view == raster::View::Shaded, "picture").clicked() {
                    self.app.visual = None;
                    self.app.settings.view = raster::View::Shaded;
                }
                if ui.selectable_label(self.app.settings.view == raster::View::LightCount, "light count").clicked() {
                    self.app.visual = None;
                    self.app.settings.view = raster::View::LightCount;
                }
                for v in crate::debug::ALL {
                    if ui.selectable_label(self.app.visual == Some(v), v.name()).clicked() {
                        self.app.visual = Some(v);
                        self.app.settings.view = raster::View::Shaded;
                    }
                }
            });
            ui.end_row();
        });
        if refit {
            let (ow, oh) = self.app.output;
            if ow > 0 {
                self.app.fit_frame(ow, oh);
            }
        }
        ui.add_space(6.0);
        let s = self.app.frame.stats;
        egui::CollapsingHeader::new("Last frame").default_open(true).show(ui, |ui| {
            egui::Grid::new("stats").num_columns(2).show(ui, |ui| {
                let mut row = |k: &str, v: String| {
                    ui.label(k);
                    ui.label(v);
                    ui.end_row();
                };
                row("Size", format!("{} x {}", self.app.frame.width, self.app.frame.height));
                row("Frame rate", format!("{:.1} fps", self.app.title_fps));
                row("Triangles", format!("{} of {}", thousands(s.tris_drawn), thousands(s.tris_in)));
                row("Clusters", format!("{} of {}", thousands(s.clusters_drawn), thousands(s.clusters_in)));
                if s.clusters_occluded > 0 {
                    row("Hidden, skipped", thousands(s.clusters_occluded));
                }
                row("Points shaded", format!("{} of {}", thousands(s.points_shaded as usize), thousands(s.points_used as usize)));
                if s.points_interpolated > 0 {
                    row("Interpolated", thousands(s.points_interpolated as usize));
                }
                row("Shading cache", format!("{} of {} MB", s.cache_bytes >> 20, self.app.settings.cache_budget >> 20));
                row("Shadows", format!("{:.0} ms", s.shadow_ms));
                row("Bounce", if !self.app.settings.bounce { "off".into() } else if s.receivers_pending > 0 { format!("{} receivers to go", thousands(s.receivers_pending)) } else { "done".into() });
                row("Setup, raster", format!("{:.0}, {:.0} ms", s.setup_ms + s.bin_ms, s.raster_ms));
                row("Shade, resolve", format!("{:.0}, {:.0} ms", s.shade_ms, s.resolve_ms));
            });
        });
    }

    // ------------------------------------------------------------ timeline

    pub(super) fn timeline(&mut self, ui: &mut egui::Ui) {
        ui.add_space(3.0);
        ui.horizontal(|ui| {
            if ui.button(if self.playing { "Pause" } else { "Play" }).clicked() {
                self.playing = !self.playing;
            }
            egui::ComboBox::from_id_salt("speed")
                .width(90.0)
                .selected_text(match self.speed {
                    s if s < 0.1 => "1 min/s",
                    s if s < 0.5 => "10 min/s",
                    s if s < 1.5 => "1 h/s",
                    _ => "3 h/s",
                })
                .show_ui(ui, |ui| {
                    for (s, n) in [(1.0 / 60.0, "1 min/s"), (10.0 / 60.0, "10 min/s"), (1.0, "1 h/s"), (3.0, "3 h/s")] {
                        ui.selectable_value(&mut self.speed, s, n);
                    }
                });
            let mut h = self.app.hours;
            let r = ui.add(egui::DragValue::new(&mut h).speed(0.02).custom_formatter(|v, _| hhmm(v as f32)).custom_parser(|s| crate::parse_hours(s).map(|h| h as f64)));
            if r.changed() {
                self.app.hours = h.rem_euclid(24.0);
                self.app.apply_daylight();
            }
            self.time_ruler(ui);
        });
        ui.add_space(3.0);
    }

    /// The day as Maya's time slider: an hour per tick, night shaded, the
    /// time a red mark; click or drag to set it.
    fn time_ruler(&mut self, ui: &mut egui::Ui) {
        let size = egui::vec2(ui.available_width() - 4.0, 34.0);
        let (rect, r) = ui.allocate_exact_size(size, egui::Sense::click_and_drag());
        let p = ui.painter_at(rect);
        let v = ui.visuals();
        p.rect_filled(rect, 2.0, v.extreme_bg_color);
        let x_of = |h: f32| rect.left() + h / 24.0 * rect.width();
        // Day and night: brighter where the sun is up.
        let steps = 96;
        for k in 0..steps {
            let h = (k as f32 + 0.5) * 24.0 / steps as f32;
            let up = crate::sun::daylight(h).sun_dir.y;
            let t = (up * 3.0).clamp(0.0, 1.0);
            let c = egui::Color32::from_rgb((28.0 + 40.0 * t) as u8, (30.0 + 44.0 * t) as u8, (40.0 + 46.0 * t) as u8);
            let a = egui::pos2(x_of(k as f32 * 24.0 / steps as f32), rect.top());
            let b = egui::pos2(x_of((k + 1) as f32 * 24.0 / steps as f32) + 0.5, rect.bottom());
            p.rect_filled(egui::Rect::from_min_max(a, b), 0.0, c);
        }
        let every = if rect.width() / 24.0 > 34.0 { 1 } else if rect.width() / 24.0 > 17.0 { 2 } else { 3 };
        for q in 0..=96 {
            let h = q as f32 / 4.0;
            let x = x_of(h);
            let tall = if q % 4 == 0 { 10.0 } else { 4.0 };
            p.line_segment([egui::pos2(x, rect.bottom() - tall), egui::pos2(x, rect.bottom())], egui::Stroke::new(1.0, v.weak_text_color()));
            if q % 4 == 0 && (q / 4) % every == 0 && q < 96 {
                p.text(egui::pos2(x + 3.0, rect.top() + 2.0), egui::Align2::LEFT_TOP, format!("{}", q / 4), egui::FontId::proportional(11.0), v.text_color());
            }
        }
        let x = x_of(self.app.hours);
        let red = egui::Color32::from_rgb(230, 70, 60);
        p.line_segment([egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())], egui::Stroke::new(2.0, red));
        let label = hhmm(self.app.hours);
        let tx = if x + 46.0 > rect.right() { x - 44.0 } else { x + 3.0 };
        p.rect_filled(egui::Rect::from_min_size(egui::pos2(tx - 1.0, rect.bottom() - 16.0), egui::vec2(38.0, 14.0)), 2.0, red);
        p.text(egui::pos2(tx + 1.0, rect.bottom() - 15.0), egui::Align2::LEFT_TOP, label, egui::FontId::proportional(11.0), egui::Color32::WHITE);
        if let Some(pos) = r.interact_pointer_pos() {
            if r.dragged() || r.clicked() || r.drag_started() {
                let h = ((pos.x - rect.left()) / rect.width() * 24.0).clamp(0.0, 24.0 - 1.0 / 60.0);
                self.app.hours = (h * 60.0).round() / 60.0;
                self.app.apply_daylight();
            }
        }
    }
}
