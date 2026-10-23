//! A file browser drawn with egui, for Open, Save As and the importer:
//! folders first, then the files with the extensions asked for.

use std::path::{Path, PathBuf};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    /// Pick an existing file.
    Open,
    /// Name a file to write.
    Save,
    /// Pick a folder.
    Folder,
}

pub enum Outcome {
    /// Still choosing.
    Open,
    Chosen(PathBuf),
    Cancelled,
}

struct Entry {
    name: String,
    dir: bool,
    bytes: u64,
}

pub struct Browser {
    title: String,
    mode: Mode,
    /// Lower case, without the dot. Empty: every file.
    exts: Vec<String>,
    dir: PathBuf,
    /// What the path box holds (edited by hand, Enter to go).
    dir_text: String,
    name: String,
    entries: Vec<Entry>,
    selected: Option<usize>,
    error: Option<String>,
}

/// The user's home folder.
pub fn home() -> PathBuf {
    std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")).map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."))
}

impl Browser {
    /// `exts` without dots, e.g. ["glb", "fbx"]. `name` is the file name
    /// offered when saving.
    pub fn new(title: &str, mode: Mode, exts: &[&str], start: Option<&Path>, name: &str) -> Browser {
        let dir = start.filter(|d| d.is_dir()).map(Path::to_path_buf).or_else(|| std::env::current_dir().ok()).unwrap_or_else(home);
        let mut b = Browser {
            title: title.into(),
            mode,
            exts: exts.iter().map(|e| e.to_lowercase()).collect(),
            dir_text: String::new(),
            dir,
            name: name.into(),
            entries: Vec::new(),
            selected: None,
            error: None,
        };
        b.refresh();
        b
    }

    fn wanted(&self, name: &str) -> bool {
        self.exts.is_empty() || Path::new(name).extension().map(|e| self.exts.contains(&e.to_string_lossy().to_lowercase())).unwrap_or(false)
    }

    fn refresh(&mut self) {
        self.dir_text = self.dir.display().to_string();
        self.entries.clear();
        self.selected = None;
        self.error = None;
        match std::fs::read_dir(&self.dir) {
            Ok(rd) => {
                for e in rd.flatten() {
                    let name = e.file_name().to_string_lossy().into_owned();
                    if name.starts_with('.') {
                        continue;
                    }
                    // Follows links, so a linked folder is a folder.
                    let meta = std::fs::metadata(e.path());
                    let dir = meta.as_ref().map(|m| m.is_dir()).unwrap_or(false);
                    if !dir && (self.mode == Mode::Folder || !self.wanted(&name)) {
                        continue;
                    }
                    self.entries.push(Entry { name, dir, bytes: meta.map(|m| m.len()).unwrap_or(0) });
                }
                self.entries.sort_by(|a, b| b.dir.cmp(&a.dir).then_with(|| a.name.to_lowercase().cmp(&b.name.to_lowercase())));
            }
            Err(e) => self.error = Some(format!("{}: {e}", self.dir.display())),
        }
    }

    fn go(&mut self, dir: PathBuf) {
        if dir.is_dir() {
            self.dir = dir;
            self.refresh();
        } else {
            self.error = Some(format!("{} is not a folder", dir.display()));
        }
    }

    /// Places on the left: home and its usual folders, and the drives on
    /// Windows or the root elsewhere.
    fn places() -> Vec<(String, PathBuf)> {
        let h = home();
        let mut p = vec![("Home".to_string(), h.clone())];
        for sub in ["Desktop", "Documents", "Downloads"] {
            let d = h.join(sub);
            if d.is_dir() {
                p.push((sub.to_string(), d));
            }
        }
        if cfg!(windows) {
            for c in b'A'..=b'Z' {
                let d = PathBuf::from(format!("{}:\\", c as char));
                if d.is_dir() {
                    p.push((format!("{}:", c as char), d));
                }
            }
        } else {
            p.push(("/".to_string(), PathBuf::from("/")));
            for m in ["/media", "/mnt", "/run/media"] {
                if let Ok(rd) = std::fs::read_dir(m) {
                    for e in rd.flatten().filter(|e| e.path().is_dir()) {
                        p.push((e.file_name().to_string_lossy().into_owned(), e.path()));
                    }
                }
            }
        }
        p
    }

    /// Draw the browser as a modal window.
    pub fn show(&mut self, ctx: &egui::Context) -> Outcome {
        let mut outcome = Outcome::Open;
        let screen = ctx.content_rect();
        let size = egui::vec2((screen.width() * 0.7).clamp(420.0, 900.0), (screen.height() * 0.7).clamp(300.0, 640.0));
        let modal = egui::Modal::new(egui::Id::unique("file browser")).show(ctx, |ui| {
            ui.set_width(size.x);
            ui.set_max_width(size.x);
            ui.heading(&self.title);
            ui.horizontal(|ui| {
                if ui.button("Up").clicked() {
                    if let Some(p) = self.dir.parent().map(Path::to_path_buf) {
                        self.go(p);
                    }
                }
                let r = ui.add(egui::TextEdit::singleline(&mut self.dir_text).desired_width(f32::INFINITY));
                if r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    let d = PathBuf::from(self.dir_text.trim());
                    self.go(d);
                }
            });
            ui.separator();
            let list_h = size.y - 130.0;
            // A fixed box: in a modal nothing else bounds the lists.
            ui.allocate_ui_with_layout(egui::vec2(size.x, list_h), egui::Layout::left_to_right(egui::Align::Min), |ui| {
                ui.set_height(list_h);
                ui.vertical(|ui| {
                    ui.set_width(130.0);
                    egui::ScrollArea::vertical().id_salt("places").max_height(list_h).show(ui, |ui| {
                        for (name, path) in Self::places() {
                            if ui.selectable_label(path == self.dir, name).clicked() {
                                self.go(path);
                            }
                        }
                    });
                });
                ui.separator();
                ui.vertical(|ui| {
                    ui.set_width(size.x - 150.0);
                    egui::ScrollArea::vertical().id_salt("files").max_height(list_h).auto_shrink([false, false]).show(ui, |ui| {
                        let mut enter: Option<PathBuf> = None;
                        let mut choose: Option<PathBuf> = None;
                        for (k, e) in self.entries.iter().enumerate() {
                            let label = if e.dir { egui::RichText::new(format!("{}/", e.name)).strong() } else { egui::RichText::new(&e.name) };
                            let r = ui.horizontal(|ui| {
                                let r = ui.selectable_label(self.selected == Some(k), label);
                                if !e.dir {
                                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                        ui.weak(size_text(e.bytes));
                                    });
                                }
                                r
                            });
                            let r = r.inner;
                            if r.clicked() {
                                self.selected = Some(k);
                                if !e.dir {
                                    self.name = e.name.clone();
                                }
                            }
                            if r.double_clicked() {
                                if e.dir {
                                    enter = Some(self.dir.join(&e.name));
                                } else {
                                    choose = Some(self.dir.join(&e.name));
                                }
                            }
                        }
                        if self.entries.is_empty() && self.error.is_none() {
                            ui.weak(if self.exts.is_empty() { "empty folder".to_string() } else { format!("no folders or .{} files here", self.exts.join(" .")) });
                        }
                        if let Some(d) = enter {
                            self.go(d);
                        }
                        if let Some(c) = choose {
                            outcome = Outcome::Chosen(c);
                        }
                    });
                });
            });
            if let Some(e) = &self.error {
                ui.colored_label(ui.visuals().error_fg_color, e);
            }
            ui.separator();
            ui.horizontal(|ui| {
                if self.mode != Mode::Folder {
                    ui.label("File");
                    ui.add(egui::TextEdit::singleline(&mut self.name).desired_width(size.x * 0.5));
                    if !self.exts.is_empty() {
                        ui.weak(format!("*.{}", self.exts.join(" *.")));
                    }
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let ok = match self.mode {
                        Mode::Open => "Open",
                        Mode::Save => "Save",
                        Mode::Folder => "Use this folder",
                    };
                    if ui.button(ok).clicked() {
                        match self.mode {
                            Mode::Folder => {
                                let d = match self.selected {
                                    Some(k) if self.entries[k].dir => self.dir.join(&self.entries[k].name),
                                    _ => self.dir.clone(),
                                };
                                outcome = Outcome::Chosen(d);
                            }
                            Mode::Open if self.selected.is_some_and(|k| self.entries[k].dir) => {
                                let d = self.dir.join(&self.entries[self.selected.unwrap()].name);
                                self.go(d);
                            }
                            _ if !self.name.trim().is_empty() => {
                                let mut p = self.dir.join(self.name.trim());
                                if self.mode == Mode::Save && !self.wanted(self.name.trim()) {
                                    if let Some(e) = self.exts.first() {
                                        p.set_extension(e);
                                    }
                                }
                                if self.mode == Mode::Open && !p.is_file() {
                                    self.error = Some(format!("{} not found", p.display()));
                                } else {
                                    outcome = Outcome::Chosen(p);
                                }
                            }
                            _ => {}
                        }
                    }
                    if ui.button("Cancel").clicked() {
                        outcome = Outcome::Cancelled;
                    }
                });
            });
        });
        if modal.should_close() && matches!(outcome, Outcome::Open) {
            outcome = Outcome::Cancelled;
        }
        outcome
    }
}

pub fn size_text(b: u64) -> String {
    match b {
        b if b >= 1 << 30 => format!("{:.1} GB", b as f64 / (1u64 << 30) as f64),
        b if b >= 1 << 20 => format!("{:.1} MB", b as f64 / (1u64 << 20) as f64),
        b if b >= 1 << 10 => format!("{:.0} KB", b as f64 / 1024.0),
        b => format!("{b} B"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Folders come first, then only the files asked for, hidden ones left
    /// out.
    #[test]
    fn lists_folders_then_wanted_files() {
        let d = std::env::temp_dir().join(format!("fkj-browser-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(d.join("sub")).unwrap();
        for f in ["b.glb", "a.FBX", "notes.txt", ".hidden.glb"] {
            std::fs::write(d.join(f), b"x").unwrap();
        }
        let b = Browser::new("t", Mode::Open, &["glb", "fbx"], Some(&d), "");
        let names: Vec<&str> = b.entries.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(names, ["sub", "a.FBX", "b.glb"]);
        let _ = std::fs::remove_dir_all(&d);
    }
}
