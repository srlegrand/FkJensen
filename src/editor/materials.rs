//! The attribute editor's materials for the selected object: each of its
//! asset's materials, edited for this scene (saved in the .fjs as changes
//! from the asset), with a way to reset one and to push them all into the
//! asset itself.

use super::Editor;
use crate::scene::Material;

impl Editor {
    pub(super) fn material_attributes(&mut self, ui: &mut egui::Ui, i: usize) {
        let range = self.app.scene.objects[i].materials.clone();
        if range.is_empty() {
            return;
        }
        let mut changed = false;
        let mut push = false;
        egui::CollapsingHeader::new(format!("Materials ({})", range.len())).default_open(true).show(ui, |ui| {
            for (k, mi) in range.clone().enumerate() {
                let asset = self.app.scene.objects[i].asset_materials.get(k).cloned().unwrap_or_default();
                let m = &mut self.app.scene.materials[mi];
                let edited = *m != asset;
                let title = if edited { format!("{} (changed)", m.name) } else { m.name.clone() };
                egui::CollapsingHeader::new(title).id_salt(("material", i, k)).default_open(range.len() <= 3).show(ui, |ui| {
                    egui::Grid::new(("material grid", i, k)).num_columns(2).show(ui, |ui| {
                        ui.label("Tint").on_hover_text("Multiplies the asset's colour (its texture tiles or vertex colours)");
                        let mut c = m.tint.to_array();
                        if ui.color_edit_button_rgb(&mut c).changed() {
                            m.tint = glam::Vec3::from_array(c);
                            changed = true;
                        }
                        ui.end_row();
                        ui.label("Roughness").on_hover_text("0: a mirror; 1: no highlight to speak of");
                        changed |= ui.add(egui::Slider::new(&mut m.roughness, 0.02..=1.0).max_decimals(3)).changed();
                        ui.end_row();
                        ui.label("Metallic").on_hover_text("1: no diffuse, the colour is what it reflects");
                        changed |= ui.add(egui::Slider::new(&mut m.metallic, 0.0..=1.0).max_decimals(3)).changed();
                        ui.end_row();
                        ui.label("Specular").on_hover_text("A dielectric's reflectance facing the eye: 0.5 is 4%, most materials; 1 is 8%");
                        changed |= ui.add(egui::Slider::new(&mut m.specular, 0.0..=1.0).max_decimals(3)).changed();
                        ui.end_row();
                    });
                    if ui.add_enabled(edited, egui::Button::new("Reset to the asset's")).clicked() {
                        *m = Material { name: m.name.clone(), ..asset.clone() };
                        changed = true;
                    }
                });
            }
            ui.add_space(4.0);
            let o = &self.app.scene.objects[i];
            let edited = self.app.scene.materials[o.materials.clone()] != o.asset_materials[..];
            let has_file = o.asset.as_ref().is_some_and(|p| p.exists());
            ui.horizontal(|ui| {
                let hint = if has_file { "Write these materials into the asset's file: every scene using it gets them" } else { "The object has no asset file yet: saving the scene writes one, with these materials" };
                push = ui.add_enabled(edited && has_file, egui::Button::new("Push to asset")).on_hover_text(hint).on_disabled_hover_text(hint).clicked();
                if ui.add_enabled(edited, egui::Button::new("Reset all")).clicked() {
                    let back = self.app.scene.objects[i].asset_materials.clone();
                    for (k, mi) in range.clone().enumerate() {
                        if let Some(a) = back.get(k) {
                            self.app.scene.materials[mi] = a.clone();
                        }
                    }
                    changed = true;
                }
            });
            if edited {
                ui.weak("Changes are this scene's, saved with it, until pushed to the asset.");
            }
        });
        if changed {
            let clusters = self.app.scene.objects[i].clusters.clone();
            self.app.renderer.materials_changed(clusters);
        }
        if push {
            match crate::fjs::push_materials(&mut self.app.scene, i) {
                Ok(p) => self.say(format!("materials written into {}", p.display()), false),
                Err(e) => self.say(e, true),
            }
        }
    }
}
