//! Maya's Create menu and the shelf: polygon primitives, lights, an empty
//! group, each made where Maya makes it (the origin; a light above what
//! the camera looks at) and selected.

use super::style::{ic, icon};
use super::{Editor, Inspect};
use crate::Selection;

/// What the shelf and the Create menu make.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Make {
    Primitive(&'static str),
    PointLight,
    Directional,
    Group,
}

/// The shelf's buttons: what each makes, its icon and its name.
pub const SHELF: [(Make, &str, &str); 7] = [
    (Make::Primitive("sphere"), icon::SPHERE, "Polygon sphere"),
    (Make::Primitive("cube"), icon::CUBE, "Polygon cube"),
    (Make::Primitive("cylinder"), icon::CYLINDER, "Polygon cylinder"),
    (Make::Primitive("plane"), icon::PLANE, "Polygon plane"),
    (Make::PointLight, icon::LIGHT, "Point light"),
    (Make::Directional, icon::SUN, "Directional light"),
    (Make::Group, icon::GROUP, "Empty group"),
];

impl Editor {
    pub fn make(&mut self, what: Make) {
        match what {
            Make::Primitive(name) => {
                let Some(m) = crate::world::part(name) else { return };
                let i = self.app.scene.objects.len();
                self.app.scene.append(m);
                // A name not taken: sphere1, sphere2...
                let mut k = 1;
                while self.app.scene.objects.iter().any(|o| o.name == format!("{name}{k}")) {
                    k += 1;
                }
                self.app.scene.objects[i].name = format!("{name}{k}");
                self.app.manip = None;
                self.app.drag = None;
                self.select(Selection::Object(i));
            }
            Make::PointLight => {
                self.add_light();
                self.inspect = Inspect::Selection;
            }
            Make::Directional => match self.app.settings.dir_lights.iter().position(|d| !d.on) {
                Some(k) => {
                    self.app.settings.dir_lights[k] = crate::raster::DirLight { on: true, ..Default::default() };
                    self.inspect = Inspect::Directional(k);
                }
                None => self.say(format!("{} directional lights at most", crate::raster::MAX_DIR_LIGHTS), true),
            },
            Make::Group => self.new_group(),
        }
    }

    /// Create, as Maya's: Polygon Primitives, Lights, Empty Group.
    pub(super) fn create_menu(&mut self, ui: &mut egui::Ui) {
        ui.menu_button("Create", |ui| {
            ui.menu_button("Polygon Primitives", |ui| {
                for (what, _, name) in SHELF.iter().filter(|s| matches!(s.0, Make::Primitive(_))) {
                    if ui.button(name.trim_start_matches("Polygon ")).clicked() {
                        self.make(*what);
                        ui.close();
                    }
                }
            });
            ui.menu_button("Lights", |ui| {
                if ui.button("Point Light").clicked() {
                    self.make(Make::PointLight);
                    ui.close();
                }
                if ui.button("Directional Light").clicked() {
                    self.make(Make::Directional);
                    ui.close();
                }
                ui.separator();
                if ui.add(egui::Button::new("The set's lights again").shortcut_text("Ctrl+R")).clicked() {
                    self.app.rig = crate::lights::LightRig::new(crate::default_rig(&self.app.scene));
                    self.select(Selection::Nothing);
                    ui.close();
                }
            });
            ui.separator();
            if ui.button("Empty Group").clicked() {
                self.make(Make::Group);
                ui.close();
            }
        });
    }

    /// The shelf's create buttons: an icon each, its name on hover.
    pub(super) fn shelf_create(&mut self, ui: &mut egui::Ui) {
        for (k, (what, glyph, name)) in SHELF.iter().enumerate() {
            // A gap between primitives, lights and the rest.
            if k == 4 || k == 6 {
                ui.separator();
            }
            if ui.button(ic(glyph, 14.0)).on_hover_text(*name).clicked() {
                self.make(*what);
            }
        }
    }
}
