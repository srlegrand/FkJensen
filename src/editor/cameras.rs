//! Cameras in the editor, as in Maya: the viewport's View and Panels
//! menus (Create Camera from View, Look Through Selected, Perspective),
//! the outliner's Cameras, and a camera's attributes. Looking through a
//! camera, navigating moves it.

use super::Editor;
use crate::Selection;

impl Editor {
    /// The viewport's View menu, as Maya's panel menu.
    pub(super) fn view_menu(&mut self, ui: &mut egui::Ui) {
        ui.menu_button("View", |ui| {
            if ui.button("Create Camera from View").clicked() {
                let k = self.app.add_camera(true);
                self.select(Selection::Camera(k));
                ui.close();
            }
            let sel = match self.app.selection {
                Selection::Camera(k) => Some(k),
                _ => None,
            };
            if ui.add_enabled(sel.is_some(), egui::Button::new("Look Through Selected Camera")).clicked() {
                self.app.look_through(sel);
                ui.close();
            }
            ui.separator();
            if ui.add(egui::Button::new("Frame Selection").shortcut_text("F")).clicked() {
                self.app.frame_selection();
                ui.close();
            }
            if ui.add(egui::Button::new("Frame All").shortcut_text("A")).clicked() {
                self.app.wide = true;
                crate::frame_camera(&mut self.app.cam, &self.app.scene, true);
                ui.close();
            }
        });
    }

    /// The viewport's Panels menu: which camera the view looks through.
    pub(super) fn panels_menu(&mut self, ui: &mut egui::Ui) {
        ui.menu_button("Panels", |ui| {
            ui.menu_button("Perspective", |ui| {
                if ui.selectable_label(self.app.through.is_none(), "persp").clicked() {
                    self.app.look_through(None);
                    ui.close();
                }
                for k in 0..self.app.cameras.len() {
                    if ui.selectable_label(self.app.through == Some(k), self.app.cameras[k].name.clone()).clicked() {
                        self.app.look_through(Some(k));
                        ui.close();
                    }
                }
            });
        });
    }

    /// The outliner's cameras: persp and the scene's; click to select,
    /// double click to look through.
    pub(super) fn camera_rows(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("Cameras").default_open(true).show(ui, |ui| {
            let mut look: Option<Option<usize>> = None;
            let mut select = None;
            let r = ui.selectable_label(false, if self.app.through.is_none() { "persp   (viewing)" } else { "persp" });
            if r.double_clicked() {
                look = Some(None);
            }
            r.context_menu(|ui| {
                if ui.button("Look Through").clicked() {
                    look = Some(None);
                    ui.close();
                }
            });
            for k in 0..self.app.cameras.len() {
                let name = &self.app.cameras[k].name;
                let text = if self.app.through == Some(k) { format!("{name}   (viewing)") } else { name.clone() };
                let r = ui.selectable_label(self.app.selection == Selection::Camera(k) && self.inspect == super::Inspect::Selection, text);
                if r.clicked() {
                    select = Some(k);
                }
                if r.double_clicked() {
                    look = Some(Some(k));
                }
                r.context_menu(|ui| {
                    if ui.button("Look Through").clicked() {
                        look = Some(Some(k));
                        ui.close();
                    }
                });
            }
            if let Some(k) = select {
                self.select(Selection::Camera(k));
            }
            if let Some(k) = look {
                self.app.look_through(k);
            }
        });
    }

    /// A camera's attributes: name, where it is and looks, its angle of
    /// view.
    pub(super) fn camera_attributes(&mut self, ui: &mut egui::Ui, k: usize) {
        ui.horizontal(|ui| {
            ui.label("camera");
            ui.text_edit_singleline(&mut self.app.cameras[k].name);
        });
        ui.add_space(4.0);
        let viewing = self.app.through == Some(k);
        // Looking through it, the view's camera is the one that moves.
        let mut c = if viewing { self.app.cam } else { self.app.cameras[k].cam };
        let mut eye = c.eye();
        let (mut yaw, mut pitch) = (c.yaw.to_degrees(), -c.pitch.to_degrees());
        let mut fov = c.fov_y.to_degrees();
        let mut dist = c.distance;
        let speed = (self.app.scene.focus_radius() as f64 * 0.004).max(0.0005);
        let mut changed = false;
        egui::Grid::new("camera").num_columns(4).show(ui, |ui| {
            ui.label("Position");
            for v in [&mut eye.x, &mut eye.y, &mut eye.z] {
                changed |= ui.add(egui::DragValue::new(v).speed(speed).max_decimals(3)).changed();
            }
            ui.end_row();
            ui.label("Aim").on_hover_text("Degrees: about the vertical, and up from the horizon");
            changed |= ui.add(egui::DragValue::new(&mut yaw).speed(0.5).max_decimals(2).suffix("°")).changed();
            changed |= ui.add(egui::DragValue::new(&mut pitch).speed(0.5).range(-89.0..=89.0).max_decimals(2).suffix("°")).changed();
            ui.end_row();
            ui.label("Angle of view").on_hover_text("Vertical, degrees");
            changed |= ui.add(egui::DragValue::new(&mut fov).speed(0.2).range(1.0..=170.0).max_decimals(1).suffix("°")).changed();
            ui.end_row();
            ui.label("Centre of interest").on_hover_text("How far ahead it turns about when tumbled");
            changed |= ui.add(egui::DragValue::new(&mut dist).speed(speed).range(0.01..=100000.0).max_decimals(3)).changed();
            ui.end_row();
        });
        if changed {
            c.yaw = yaw.to_radians();
            c.pitch = (-pitch).to_radians();
            c.fov_y = fov.to_radians();
            c.distance = dist.max(0.01);
            c.target = eye + c.forward() * c.distance;
            if viewing {
                self.app.cam = c;
            }
            self.app.cameras[k].cam = c;
        }
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            if viewing {
                if ui.button("Back to persp").clicked() {
                    self.app.look_through(None);
                }
            } else if ui.button("Look through").clicked() {
                self.app.look_through(Some(k));
            }
        });
        ui.weak("Looking through it, tumbling, tracking and dollying move it.");
    }
}
