//! Maya's Edit menu: Copy, Paste, Duplicate, Duplicate Special (instance:
//! not yet), Delete, on the selected object (with what is under it), point
//! light or directional light.

use super::{Editor, Inspect};
use crate::Selection;
use crate::raster::DirLight;
use crate::scene::{PointLight, Scene};

/// Where a copied object stood: pivot, rotation, scale, what was frozen
/// and where its translate reads 0.
type Placing = (glam::Vec3, glam::Vec3, glam::Vec3, glam::Affine3A, glam::Vec3);

/// What Copy took.
pub enum Clip {
    /// An object and what is under it, as they were: each a scene of one
    /// object at rest, with where it stood and the copy it is in (an index
    /// into the list, before it).
    Objects(Vec<(Scene, Placing, Option<usize>)>),
    Light(PointLight),
    Directional(DirLight),
    Camera(crate::SceneCamera),
}

impl Editor {
    /// What Edit acts on: the selected object or point light, or the
    /// directional light shown in the attribute editor.
    fn edit_target(&self) -> Option<Target> {
        match (self.app.selection, self.inspect) {
            (_, Inspect::Directional(k)) => Some(Target::Directional(k)),
            (Selection::Object(i), _) if i < self.app.scene.objects.len() => Some(Target::Object(i)),
            (Selection::Light(i), _) if i < self.app.rig.lights.len() => Some(Target::Light(i)),
            (Selection::Camera(k), _) if k < self.app.cameras.len() => Some(Target::Camera(k)),
            _ => None,
        }
    }

    pub fn can_edit(&self) -> bool {
        self.edit_target().is_some()
    }

    /// Delete what is selected; an object goes with everything under it.
    pub fn delete_selected(&mut self) {
        match self.edit_target() {
            Some(Target::Object(i)) => {
                let all = self.selected_objects();
                let all = if all.contains(&i) { all } else { vec![i] };
                let name = if all.len() > 1 { format!("{} objects", all.len()) } else { self.app.scene.objects[i].name.clone() };
                self.app.manip = None;
                self.app.drag = None;
                self.pending_bounce = None;
                self.app.drag_start = None;
                self.app.extra.clear();
                let map = self.app.scene.remove_objects(&all);
                let n = map.iter().filter(|&&m| m == usize::MAX).count();
                // Lights under what went go too, as in Maya; the rest point
                // at where their objects went.
                let rig = &mut self.app.rig;
                for k in (0..rig.lights.len()).rev() {
                    if let Some(mut l) = rig.link(k) {
                        match map.get(l.object).copied() {
                            Some(m) if m != usize::MAX => {
                                l.object = m;
                                rig.set_link(k, Some(l));
                            }
                            _ => {
                                rig.selected = k;
                                rig.remove_selected();
                            }
                        }
                    }
                }
                self.select(Selection::Nothing);
                let under = n.saturating_sub(all.len());
                self.say(format!("deleted {name}{}", if under > 0 { format!(" and {under} under {}", if all.len() > 1 { "them" } else { "it" }) } else { String::new() }), false);
            }
            Some(Target::Light(i)) => {
                self.app.selection = Selection::Light(i);
                self.delete_light();
            }
            Some(Target::Directional(k)) => {
                self.app.settings.dir_lights[k] = DirLight::default();
                self.inspect = Inspect::Selection;
            }
            Some(Target::Camera(k)) => {
                self.app.delete_camera(k);
                self.select(Selection::Nothing);
            }
            None => {}
        }
    }

    /// Copy what is selected, as it is now.
    pub fn copy_selected(&mut self) {
        self.clip = match self.edit_target() {
            Some(Target::Object(i)) => {
                let s = &self.app.scene;
                let tree: Vec<usize> = std::iter::once(i).chain(s.descendants(i)).collect();
                let items = tree
                    .iter()
                    .map(|&j| {
                        let o = &s.objects[j];
                        let mut one = s.extract_object(j);
                        one.objects[0].asset = o.asset.clone();
                        one.objects[0].asset_materials = o.asset_materials.clone();
                        one.objects[0].environment = o.environment;
                        let parent = o.parent.and_then(|p| tree.iter().position(|&t| t == p));
                        (one, (o.pivot, o.rotation, o.scale, o.frozen, o.origin), parent)
                    })
                    .collect();
                Some(Clip::Objects(items))
            }
            Some(Target::Light(i)) => Some(Clip::Light(self.app.rig.lights[i])),
            Some(Target::Directional(k)) => Some(Clip::Directional(self.app.settings.dir_lights[k])),
            Some(Target::Camera(k)) => Some(Clip::Camera(self.app.cameras[k].clone())),
            None => None,
        };
    }

    /// Paste what was copied, where it was, and select it.
    pub fn paste(&mut self) {
        match &self.clip {
            Some(Clip::Objects(items)) => {
                let mut made: Vec<usize> = Vec::new();
                for (one, (pivot, rotation, scale, frozen, origin), parent) in items {
                    let s = &mut self.app.scene;
                    let j = if one.clusters.is_empty() { s.add_group(one.objects[0].name.clone(), *pivot) } else {
                        let j = s.objects.len();
                        s.append(clone_scene(one));
                        j
                    };
                    s.objects[j].parent = parent.map(|p| made[p]);
                    s.set_object(j, *pivot, *rotation, *scale);
                    s.set_frozen(j, *frozen, *origin);
                    made.push(j);
                }
                if let Some(&first) = made.first() {
                    self.select(Selection::Object(first));
                }
            }
            Some(Clip::Light(l)) => {
                if self.app.rig.add(l.position, l.lumens) {
                    let i = self.app.rig.lights.len() - 1;
                    self.app.rig.lights[i] = *l;
                    self.select(Selection::Light(i));
                } else {
                    self.say(format!("the rig holds {} lights at most", crate::lights::MAX_LIGHTS), true);
                }
            }
            Some(Clip::Directional(d)) => match self.app.settings.dir_lights.iter().position(|d| !d.on) {
                Some(k) => {
                    self.app.settings.dir_lights[k] = DirLight { on: true, ..*d };
                    self.inspect = Inspect::Directional(k);
                }
                None => self.say(format!("{} directional lights at most", crate::raster::MAX_DIR_LIGHTS), true),
            },
            Some(Clip::Camera(c)) => {
                let k = self.app.add_camera(false);
                self.app.cameras[k].cam = c.cam;
                self.select(Selection::Camera(k));
            }
            None => {}
        }
    }

    /// Duplicate Instanced, as Maya's: the selected object and what is
    /// under it again, each model an instance of its original (its
    /// geometry, materials and texture tiles shared, not copied), each
    /// group a new group. The new tree stands where the old one does, in
    /// the same group, and is selected.
    pub fn duplicate_instanced(&mut self) {
        let Some(Target::Object(i)) = self.edit_target() else { return };
        let s = &mut self.app.scene;
        let tree: Vec<usize> = std::iter::once(i).chain(s.descendants(i)).collect();
        let mut made: Vec<usize> = Vec::with_capacity(tree.len());
        for (k, &j) in tree.iter().enumerate() {
            let name = next_name(s, &s.objects[j].name);
            let n = s.instance_object(j, name);
            // The group copied too keeps the original's place.
            if s.objects[n].clusters.is_empty() {
                let (r, sc) = (s.objects[j].rotation, s.objects[j].scale);
                s.objects[n].rotation = r;
                s.objects[n].scale = sc;
            }
            let parent = if k == 0 { s.objects[j].parent } else { s.objects[j].parent.and_then(|p| tree.iter().position(|&t| t == p)).map(|p| made[p]) };
            s.objects[n].parent = parent;
            made.push(n);
        }
        self.app.manip = None;
        self.app.drag = None;
        let n = made.len();
        self.select(Selection::Object(made[0]));
        self.say(format!("{} instanced{}", self.app.scene.objects[made[0]].name, if n > 1 { format!(" with {} under it", n - 1) } else { String::new() }), false);
    }

    /// Freeze Transformations on the selected objects: the channels asked
    /// for read 0, 0 and 1 from now on and nothing moves (lights parented
    /// to them keep their places).
    pub fn freeze_selected(&mut self, translate: bool, rotate: bool, scale: bool) {
        let objs = self.selected_objects();
        for &i in &objs {
            self.app.scene.freeze(i, translate, rotate, scale);
            let f = self.app.scene.object_frame(i);
            let rig = &mut self.app.rig;
            for k in 0..rig.lights.len() {
                if let Some(mut l) = rig.link(k).filter(|l| l.object == i) {
                    l.local = f.inverse().transform_point3(rig.lights[k].position);
                    l.last = rig.lights[k].position;
                    rig.set_link(k, Some(l));
                }
            }
        }
        let what: Vec<&str> = [(translate, "translate"), (rotate, "rotate"), (scale, "scale")].iter().filter(|x| x.0).map(|x| x.1).collect();
        self.say(format!("froze {} on {} object{}", what.join(", "), objs.len(), if objs.len() == 1 { "" } else { "s" }), false);
    }

    /// Copy and paste in one, leaving the clipboard as it was.
    pub fn duplicate(&mut self) {
        let kept = self.clip.take();
        self.copy_selected();
        self.paste();
        self.clip = kept;
    }
}

#[derive(Clone, Copy)]
enum Target {
    Object(usize),
    Light(usize),
    Directional(usize),
    Camera(usize),
}

/// `name` with its trailing number the next not taken: tree1 gives tree2,
/// or tree3 if tree2 is.
fn next_name(s: &Scene, name: &str) -> String {
    let stem = name.trim_end_matches(|c: char| c.is_ascii_digit());
    let mut k: usize = name[stem.len()..].parse().unwrap_or(0) + 1;
    while s.objects.iter().any(|o| o.name == format!("{stem}{k}")) {
        k += 1;
    }
    format!("{stem}{k}")
}

/// A scene's copy (its texture tiles still read from where they are).
fn clone_scene(s: &Scene) -> Scene {
    let mut c = s.extract(&[0..s.clusters.len()], glam::Affine3A::IDENTITY);
    c.objects = s.objects.clone();
    c
}
