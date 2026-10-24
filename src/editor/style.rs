//! The editor's look: Inter at small sizes, tight spacing, and Phosphor's
//! icons (`icon`) for the shelf and the tools. Both fonts are in
//! `assets/fonts/` with their licences (SIL Open Font License for Inter,
//! MIT for Phosphor), built into the program.

use std::sync::Arc;

use egui::{FontData, FontDefinitions, FontFamily, FontId, TextStyle};

/// Phosphor icons, regular weight, drawn with `ic`.
#[allow(dead_code)]
pub mod icon {
    pub const CURSOR: &str = "\u{E1DC}";
    pub const MOVE: &str = "\u{E0A4}";
    pub const ROTATE: &str = "\u{E036}";
    pub const SCALE: &str = "\u{E0A2}";
    pub const CUBE: &str = "\u{E1DA}";
    pub const SPHERE: &str = "\u{EE66}";
    pub const CYLINDER: &str = "\u{E8FC}";
    pub const PLANE: &str = "\u{E45E}";
    pub const LIGHT: &str = "\u{E2DC}";
    pub const SUN: &str = "\u{E472}";
    pub const CAMERA: &str = "\u{E4DA}";
    pub const GROUP: &str = "\u{E25A}";
    pub const IMPORT: &str = "\u{E3D4}";
    pub const FRAME: &str = "\u{E1D6}";
    pub const TRASH: &str = "\u{E4A6}";
    pub const COPY: &str = "\u{E1CA}";
    pub const EYE: &str = "\u{E220}";
    pub const MOUNTAINS: &str = "\u{E7AE}";
    pub const LEFT: &str = "\u{E138}";
    pub const RIGHT: &str = "\u{E13A}";
    pub const UP: &str = "\u{E13C}";
    pub const DOWN: &str = "\u{E136}";
    pub const ZOOM_IN: &str = "\u{E310}";
    pub const ZOOM_OUT: &str = "\u{E30E}";
    pub const FRAME_ALL: &str = "\u{E1D0}";
    pub const PLAY: &str = "\u{E3D0}";
    pub const PAUSE: &str = "\u{E39E}";
    pub const OPEN: &str = "\u{E256}";
    pub const SAVE: &str = "\u{E248}";
    pub const LINK: &str = "\u{E2E2}";
}

/// Icon `glyph` (from `icon`) at `size` points.
pub fn ic(glyph: &str, size: f32) -> egui::RichText {
    egui::RichText::new(glyph).family(FontFamily::Name("icons".into())).size(size)
}

/// Body text size, points.
pub const BODY: f32 = 11.5;

pub fn apply(ctx: &egui::Context) {
    let mut fonts = FontDefinitions::default();
    fonts.font_data.insert("Inter".into(), Arc::new(FontData::from_static(include_bytes!("../../assets/fonts/InterVariable.ttf"))));
    fonts.font_data.insert("Phosphor".into(), Arc::new(FontData::from_static(include_bytes!("../../assets/fonts/Phosphor.ttf"))));
    let prop = fonts.families.entry(FontFamily::Proportional).or_default();
    prop.insert(0, "Inter".into());
    // Icons in a family of their own (`ic`): Inter and the default fonts
    // have glyphs of their own at Phosphor's code points, and Phosphor
    // blank ones for letters.
    fonts.families.insert(FontFamily::Name("icons".into()), vec!["Phosphor".into()]);
    ctx.set_fonts(fonts);
    ctx.set_visuals(egui::Visuals::dark());
    ctx.all_styles_mut(|s| {
        s.text_styles = [
            (TextStyle::Small, FontId::proportional(9.5)),
            (TextStyle::Body, FontId::proportional(BODY)),
            (TextStyle::Button, FontId::proportional(BODY)),
            (TextStyle::Heading, FontId::proportional(13.0)),
            (TextStyle::Monospace, FontId::monospace(10.5)),
        ]
        .into();
        let sp = &mut s.spacing;
        sp.item_spacing = egui::vec2(5.0, 3.0);
        sp.button_padding = egui::vec2(5.0, 1.0);
        sp.interact_size = egui::vec2(32.0, 16.0);
        sp.indent = 12.0;
        sp.slider_width = 110.0;
        sp.combo_width = 90.0;
        sp.text_edit_width = 140.0;
        sp.menu_margin = egui::Margin::same(4);
        sp.window_margin = egui::Margin::same(6);
        s.interaction.selectable_labels = false;
        s.visuals.widgets.noninteractive.corner_radius = egui::CornerRadius::same(2);
        s.visuals.widgets.inactive.corner_radius = egui::CornerRadius::same(2);
        s.visuals.widgets.hovered.corner_radius = egui::CornerRadius::same(2);
        s.visuals.widgets.active.corner_radius = egui::CornerRadius::same(2);
    });
}
