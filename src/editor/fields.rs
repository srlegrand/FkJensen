//! Assets without a shadow field: every model object casts its shadows
//! through one (`field.rs`), so a scene or an asset that lacks one is
//! turned away when read. This dialog lists what lacks one and builds it,
//! into the assets themselves or for this session only, then does again
//! what was turned away.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver};

use super::Editor;

/// What was turned away, to do again once the fields are built.
#[derive(Clone)]
pub(super) enum Then {
    Open(PathBuf),
    Add(PathBuf),
}

pub(super) struct Wanted {
    then: Then,
    /// The assets without one (none for an old .fkj: built in memory only).
    assets: Vec<PathBuf>,
    /// While building into them: the thread's news, and what it said.
    building: Option<Receiver<Result<String, String>>>,
    log: Vec<String>,
    failed: bool,
}

impl Editor {
    /// Whether `error`, from opening or adding `then`'s file, is about a
    /// missing shadow field; if so the dialog is put up.
    pub(super) fn offer_fields(&mut self, error: &str, then: Then) -> bool {
        if !error.contains(crate::fjs::NO_FIELD) {
            return false;
        }
        let path = match &then {
            Then::Open(p) | Then::Add(p) => p.clone(),
        };
        let assets = assets_without(&path);
        self.fields_wanted = Some(Wanted { then, assets, building: None, log: Vec::new(), failed: false });
        true
    }

    /// The dialog, while there is one.
    pub(super) fn fields_dialog(&mut self, ctx: &egui::Context) {
        let Some(w) = &mut self.fields_wanted else { return };
        // The thread's news.
        let mut finished = false;
        if let Some(rx) = &w.building {
            ctx.request_repaint();
            loop {
                match rx.try_recv() {
                    Ok(Ok(line)) => w.log.push(line),
                    Ok(Err(e)) => {
                        w.log.push(e);
                        w.failed = true;
                    }
                    Err(std::sync::mpsc::TryRecvError::Empty) => break,
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        finished = true;
                        break;
                    }
                }
            }
        }
        if finished {
            w.building = None;
            if !w.failed {
                let then = w.then.clone();
                self.fields_wanted = None;
                self.redo(then, false);
                return;
            }
        }
        let mut close = false;
        let mut build_in = false;
        let mut session = false;
        egui::Window::new("Missing Shadow Fields").collapsible(false).resizable(false).anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0)).show(ctx, |ui| {
            ui.label("Every object casts its shadows through a shadow field baked into its asset. These have none:");
            ui.add_space(4.0);
            if w.assets.is_empty() {
                ui.monospace(match &w.then {
                    Then::Open(p) | Then::Add(p) => p.display().to_string(),
                });
            }
            for a in &w.assets {
                ui.monospace(a.display().to_string());
            }
            ui.add_space(4.0);
            for line in &w.log {
                ui.weak(line);
            }
            ui.add_space(4.0);
            let idle = w.building.is_none();
            ui.horizontal(|ui| {
                if !w.assets.is_empty() {
                    build_in = ui.add_enabled(idle, egui::Button::new("Build and Save into the Assets")).on_hover_text("Bakes each field (seconds a model) and writes it into the asset's file, so it is there from now on.").clicked();
                }
                session = ui.add_enabled(idle, egui::Button::new("Build for This Session")).on_hover_text("Bakes the fields in memory only: they are baked again every time the scene is read.").clicked();
                close = ui.add_enabled(idle, egui::Button::new("Cancel")).clicked();
            });
            if !idle {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label("building...");
                });
            }
        });
        if close {
            self.fields_wanted = None;
        } else if session {
            let then = w.then.clone();
            self.fields_wanted = None;
            self.redo(then, true);
        } else if build_in {
            let (tx, rx) = channel();
            let assets = w.assets.clone();
            w.log.clear();
            w.failed = false;
            w.building = Some(rx);
            std::thread::spawn(move || {
                for a in assets {
                    let _ = tx.send(match crate::fjs::build_field_into(&a) {
                        Ok(r) => Ok(format!("{}: built in {:.1} s ({} opaque, {} cut-out triangles)", a.display(), r.seconds, r.opaque_tris, r.cutout_tris)),
                        Err(e) => Err(e),
                    });
                }
            });
        }
    }

    /// Open or add again, the missing fields built in memory if `session`.
    fn redo(&mut self, then: Then, session: bool) {
        use std::sync::atomic::Ordering;
        let was = crate::fjs::BUILD_MISSING.load(Ordering::Relaxed);
        if session {
            crate::fjs::BUILD_MISSING.store(true, Ordering::Relaxed);
        }
        match then {
            Then::Open(p) => self.open(p),
            Then::Add(p) => self.add_asset(p),
        }
        crate::fjs::BUILD_MISSING.store(was, Ordering::Relaxed);
    }
}

/// The assets `path` (an .fjs, an .fja) reads that have no shadow field;
/// none for an old .fkj, which cannot be written into.
fn assets_without(path: &Path) -> Vec<PathBuf> {
    let ext = path.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
    match ext.as_str() {
        "fjs" => match crate::fjs::read(path) {
            Ok(f) => {
                let dir = path.parent().map(Path::to_path_buf).unwrap_or_default();
                crate::fjs::assets_without_fields(&f, &dir, &crate::assets_dirs())
            }
            Err(_) => Vec::new(),
        },
        "fja" => vec![path.to_path_buf()],
        _ => Vec::new(),
    }
}
