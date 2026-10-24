//! Buttons over the viewport's corner to move the camera without a mouse:
//! three pads to drag on (tumble, track, dolly), as Alt + left, middle and
//! right drag do, for a touchpad, a pen or a touch screen; buttons that
//! move the camera while held, for a single click or tap; and the two
//! framings (F, A).

use super::Editor;

/// Degrees a second the camera tumbles while a button is held, and
/// viewport heights a second it tracks.
const TUMBLE_RATE: f32 = 60.0;
const TRACK_RATE: f32 = 0.5;
/// Dolly steps a second (a step is a tenth of the distance).
const DOLLY_RATE: f32 = 4.0;
/// Drag on the dolly pad: pixels per step.
const DOLLY_PX: f32 = 50.0;

#[derive(Clone, Copy, PartialEq)]
enum Move {
    Tumble(f32, f32),
    Track(f32, f32),
    Dolly(f32),
}

impl Editor {
    /// The navigation buttons, in the viewport's bottom right corner (`rect`,
    /// in points).
    pub(super) fn navigation(&mut self, ctx: &egui::Context, rect: egui::Rect) {
        if !self.navigate {
            return;
        }
        let ppp = ctx.pixels_per_point();
        let dt = ctx.input(|i| i.stable_dt).clamp(0.0, 0.1);
        let mut held: Option<Move> = None;
        let mut dragged: Option<Move> = None;
        let mut frame = None;
        let mut rects: Vec<(&'static str, egui::Rect)> = Vec::new();
        let area = egui::Area::new(egui::Id::unique("navigation")).order(egui::Order::Foreground).pivot(egui::Align2::RIGHT_BOTTOM).fixed_pos(rect.right_bottom() - egui::vec2(8.0, 8.0));
        area.show(ctx, |ui| {
            egui::Frame::popup(ui.style()).show(ui, |ui| {
                ui.spacing_mut().item_spacing = egui::vec2(4.0, 4.0);
                egui::Grid::new("navigation grid").num_columns(5).show(ui, |ui| {
                    // A pad to drag on, then buttons held to move.
                    let pad = |ui: &mut egui::Ui, name: &'static str, hint: &str, rects: &mut Vec<(&'static str, egui::Rect)>| -> egui::Vec2 {
                        let (r, resp) = ui.allocate_exact_size(egui::vec2(68.0, 22.0), egui::Sense::drag());
                        rects.push((name, r));
                        let v = ui.visuals();
                        let fill = if resp.dragged() { v.selection.bg_fill } else if resp.hovered() { v.widgets.hovered.bg_fill } else { v.widgets.inactive.bg_fill };
                        ui.painter().rect_filled(r, 3.0, fill);
                        ui.painter().text(r.center(), egui::Align2::CENTER_CENTER, name, egui::FontId::proportional(12.0), v.strong_text_color());
                        let resp = resp.on_hover_text(hint);
                        if resp.dragged() { resp.drag_delta() * ppp } else { egui::Vec2::ZERO }
                    };
                    let hold = |ui: &mut egui::Ui, text: &'static str, m: Move, held: &mut Option<Move>, rects: &mut Vec<(&'static str, egui::Rect)>| {
                        let b = ui.add(egui::Button::new(text).min_size(egui::vec2(34.0, 22.0)));
                        rects.push((text, b.rect));
                        if b.is_pointer_button_down_on() {
                            *held = Some(m);
                        }
                    };

                    let d = pad(ui, "Tumble", "Drag here to turn round what the camera looks at (Alt + left drag)", &mut rects);
                    if d != egui::Vec2::ZERO {
                        dragged = Some(Move::Tumble(d.x, d.y));
                    }
                    hold(ui, "Left", Move::Tumble(-1.0, 0.0), &mut held, &mut rects);
                    hold(ui, "Right", Move::Tumble(1.0, 0.0), &mut held, &mut rects);
                    hold(ui, "Up", Move::Tumble(0.0, -1.0), &mut held, &mut rects);
                    hold(ui, "Down", Move::Tumble(0.0, 1.0), &mut held, &mut rects);
                    ui.end_row();

                    let d = pad(ui, "Track", "Drag here to slide the view (Alt + middle drag)", &mut rects);
                    if d != egui::Vec2::ZERO {
                        dragged = Some(Move::Track(d.x, d.y));
                    }
                    hold(ui, "Left", Move::Track(1.0, 0.0), &mut held, &mut rects);
                    hold(ui, "Right", Move::Track(-1.0, 0.0), &mut held, &mut rects);
                    hold(ui, "Up", Move::Track(0.0, 1.0), &mut held, &mut rects);
                    hold(ui, "Down", Move::Track(0.0, -1.0), &mut held, &mut rects);
                    ui.end_row();

                    let d = pad(ui, "Dolly", "Drag right or down to come closer (Alt + right drag)", &mut rects);
                    if d != egui::Vec2::ZERO {
                        dragged = Some(Move::Dolly((d.x + d.y) / DOLLY_PX));
                    }
                    hold(ui, "In", Move::Dolly(1.0), &mut held, &mut rects);
                    hold(ui, "Out", Move::Dolly(-1.0), &mut held, &mut rects);
                    let b = ui.add(egui::Button::new("Frame").min_size(egui::vec2(34.0, 22.0))).on_hover_text("Frame what is selected (F)");
                    rects.push(("Frame", b.rect));
                    if b.clicked() {
                        frame = Some(false);
                    }
                    let b = ui.add(egui::Button::new("All").min_size(egui::vec2(34.0, 22.0))).on_hover_text("Frame the subject from afar (A)");
                    rects.push(("All", b.rect));
                    if b.clicked() {
                        frame = Some(true);
                    }
                    ui.end_row();
                });
            });
        });
        self.nav_rects = rects;
        let h = self.app.frame.height as f32 / self.app.render_scale.max(1e-6);
        let cam = &mut self.app.cam;
        match dragged {
            Some(Move::Tumble(dx, dy)) => cam.orbit(dx, dy),
            Some(Move::Track(dx, dy)) => cam.pan(dx, dy, h),
            Some(Move::Dolly(s)) => cam.dolly(s),
            None => {}
        }
        match held {
            // `orbit` turns 0.008 radians a pixel.
            Some(Move::Tumble(x, y)) => {
                let px = TUMBLE_RATE.to_radians() / 0.008 * dt;
                cam.orbit(x * px, y * px);
            }
            Some(Move::Track(x, y)) => {
                let px = TRACK_RATE * h * dt;
                cam.pan(x * px, y * px, h);
            }
            Some(Move::Dolly(s)) => cam.dolly(s * DOLLY_RATE * dt),
            None => {}
        }
        if held.is_some() {
            ctx.request_repaint();
        }
        match frame {
            Some(false) => self.app.frame_selection(),
            Some(true) => {
                self.app.wide = true;
                crate::frame_camera(&mut self.app.cam, &self.app.scene, true);
            }
            None => {}
        }
    }
}
