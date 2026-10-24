//! Maya's Edit menu: Copy, Paste, Duplicate, Duplicate Special (instance:
//! not yet), Delete, on the selected object (with what is under it), point
//! light or directional light.

use super::{Editor, Inspect};
use crate::Selection;
use crate::raster::DirLight;
use crate::scene::{PointLight, Scene};

/// What Copy took.
pub enum Clip {
    /// An object and what is under it, as they were: each a scene of one
    /// object at rest, with where it stood and the copy it is in (an index
    /// into the list, before it).
    Objects(Vec<(Scene, (glam::Vec3, glam::Vec3, f32), Option<usize>)>),
    Light(PointLight),
    Directional(DirLight),
}

impl Editor {
    /// What Edit acts on: the selected object or point light, or the
    /// directional light shown in the attribute editor.
    fn edit_target(&self) -> Option<Target> {
        match (self.app.selection, self.inspect) {
            (_, Inspect::Directional(k)) => Some(Target::Directional(k)),
            (Selection::Object(i), _) if i < self.app.scene.objects.len() => Some(Target::Object(i)),
            (Selection::Light(i), _) if i < self.app.rig.lights.len() => Some(Target::Light(i)),
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
                let name = self.app.scene.objects[i].name.clone();
                self.app.manip = None;
                self.app.drag = None;
                self.pending_bounce = None;
                self.app.drag_start = None;
                let n = self.app.scene.remove_objects(&[i]);
                self.select(Selection::Nothing);
                self.say(format!("deleted {name}{}", if n > 1 { format!(" and {} under it", n - 1) } else { String::new() }), false);
            }
            Some(Target::Light(i)) => {
                self.app.selection = Selection::Light(i);
                self.delete_light();
            }
            Some(Target::Directional(k)) => {
                self.app.settings.dir_lights[k] = DirLight::default();
                self.inspect = Inspect::Selection;
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
                        (one, (o.pivot, o.rotation, o.scale), parent)
                    })
                    .collect();
                Some(Clip::Objects(items))
            }
            Some(Target::Light(i)) => Some(Clip::Light(self.app.rig.lights[i])),
            Some(Target::Directional(k)) => Some(Clip::Directional(self.app.settings.dir_lights[k])),
            None => None,
        };
    }

    /// Paste what was copied, where it was, and select it.
    pub fn paste(&mut self) {
        match &self.clip {
            Some(Clip::Objects(items)) => {
                let mut made: Vec<usize> = Vec::new();
                for (one, (pivot, rotation, scale), parent) in items {
                    let s = &mut self.app.scene;
                    let j = if one.clusters.is_empty() { s.add_group(one.objects[0].name.clone(), *pivot) } else {
                        let j = s.objects.len();
                        s.append(clone_scene(one));
                        j
                    };
                    s.objects[j].parent = parent.map(|p| made[p]);
                    s.set_object(j, *pivot, *rotation, *scale);
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
            None => {}
        }
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
}

/// A scene's copy (its texture tiles still read from where they are).
fn clone_scene(s: &Scene) -> Scene {
    let mut c = s.extract(&[0..s.clusters.len()], glam::Affine3A::IDENTITY);
    c.objects = s.objects.clone();
    c
}
