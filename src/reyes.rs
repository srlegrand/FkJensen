//! Shading the REYES way, kept between frames.
//!
//! Each visible triangle is diced into micropolygons from its own size on
//! screen, not its cluster's or its mesh's: each edge gets a level from its
//! length in pixels (`raster::dice_level`), and the triangle's grid takes
//! the finest of its three. Level L cuts each edge into 2^L parts. The grid
//! is even in the triangle's screen weights, so its points are evenly
//! spaced on screen however the triangle recedes. Points on an edge between
//! that edge's own samples are interpolated along it, so two triangles
//! sharing an edge agree along it. A triangle whose edges are all under
//! the shading rate (`RenderSettings::shading_rate_px`) uses the cluster's
//! shaded vertices.
//!
//! Shading happens on those points, in world space, before the picture is
//! made, and the results are kept (Cook, Carpenter, Catmull 1987). The
//! picture is then resolved by interpolating the shaded points under each
//! pixel.
//!
//! What is kept, and what throws it away:
//!
//! - Diffuse light at each point: kept until a light whose old or new reach
//!   touches the point's cluster changes, or the sky or exposure changes.
//!   Moving a lamp on the chess table re-shades the table and nothing else.
//! - The highlight at each point: depends on the eye, so also redone when
//!   the camera moves.
//! - Visibility: kept while the camera does not move. Dragging a light
//!   skips it entirely.
//! - Shadow maps: kept until their light moves or changes strength.
//!
//! Only points under visible pixels are shaded: a cluster records which of
//! its triangles (or, at level 0, vertices) are needed and which are done.

use std::sync::atomic::{AtomicU64, Ordering};

use glam::{Mat4, Vec3};
use rayon::prelude::*;

use crate::bounce::{self, Bounce};
use crate::raster::{self, Camera, Frame, NO_TRI, NOT_VISIBLE, RenderSettings, Sky, View};
use crate::scene::{PointLight, Scene};
use crate::shadow::CubeShadow;
use crate::sun::SunShadow;
use crate::transfer::SkyShare;

/// Light ids from here on are directional lights (`RenderSettings::
/// dir_lights`); below, point lights.
const DIR_ID: usize = 64;

/// How one triangle was diced, and where its points are kept.
///
/// Its grid lies on the triangle as seen when it was diced: rows along its
/// longest edge on screen, 2^lu + 1 points each, and 2^lv + 1 rows up to the
/// opposite corner, so points are about the shading rate apart
/// both ways on screen however thin or receding the triangle is. The grid
/// covers the rectangle on that edge, twice the triangle; points outside it
/// are shaded on the triangle's plane, for interpolating up to its edges.
#[derive(Clone, Copy)]
struct TriDice {
    /// The finer of lu and lv; 0: the triangle reads the cluster's
    /// vertices; `NOT_VISIBLE`: not diced.
    level: u8,
    lu: u8,
    lv: u8,
    /// The longest edge on screen when diced (from corner `base` to the
    /// next).
    base: u8,
    /// Depth (clip w) and position on screen (pixels from the centre) of
    /// each corner when diced.
    w: [f32; 3],
    px: [glam::Vec2; 3],
    /// Where its points start in `diffuse` and `spec`, and how many fit.
    offset: u32,
    cap: u32,
}

impl Default for TriDice {
    fn default() -> Self {
        Self { level: NOT_VISIBLE, lu: 0, lv: 0, base: 0, w: [1.0; 3], px: [glam::Vec2::ZERO; 3], offset: 0, cap: 0 }
    }
}

impl TriDice {
    fn points(&self) -> usize {
        if self.level == 0 || self.level == NOT_VISIBLE { 0 } else { ((1usize << self.lu) + 1) * ((1usize << self.lv) + 1) }
    }

    /// The grid's origin and its two sides on screen.
    fn frame(&self) -> (glam::Vec2, glam::Vec2, glam::Vec2) {
        let k = self.base as usize;
        let (o, ex) = (self.px[k], self.px[(k + 1) % 3] - self.px[k]);
        let len = ex.length().max(1e-6);
        let up = ex.perp() / len;
        let h = (self.px[(k + 2) % 3] - o).dot(up);
        let ey = if h.abs() < 1e-6 { up * 1e-6 } else { up * h };
        (o, ex, ey)
    }

    /// Screen weights of grid point (u, v), u and v from 0 to 1.
    fn grid_to_screen(&self, u: f32, v: f32) -> [f32; 3] {
        let (o, ex, ey) = self.frame();
        let q = o + ex * u + ey * v - self.px[0];
        let (e1, e2) = (self.px[1] - self.px[0], self.px[2] - self.px[0]);
        let det = e1.perp_dot(e2);
        let det = if det.abs() < 1e-12 { 1e-12 } else { det };
        let (s1, s2) = (q.perp_dot(e2) / det, e1.perp_dot(q) / det);
        [1.0 - s1 - s2, s1, s2]
    }

    /// Screen weights are affine in (u, v): those of (0, 0), and their
    /// change along u and along v, so a grid's points need no `frame`
    /// each.
    fn screen_affine(&self) -> ([f32; 3], [f32; 3], [f32; 3]) {
        let s0 = self.grid_to_screen(0.0, 0.0);
        let su = self.grid_to_screen(1.0, 0.0);
        let sv = self.grid_to_screen(0.0, 1.0);
        (s0, [su[0] - s0[0], su[1] - s0[1], su[2] - s0[2]], [sv[0] - s0[0], sv[1] - s0[1], sv[2] - s0[2]])
    }

    /// Grid position (u, v), each 0 to 1, of screen weights `s`.
    fn screen_to_grid(&self, s: [f32; 3]) -> (f32, f32) {
        let (o, ex, ey) = self.frame();
        let q = self.px[0] * s[0] + self.px[1] * s[1] + self.px[2] * s[2] - o;
        ((q.dot(ex) / ex.length_squared()).clamp(0.0, 1.0), (q.dot(ey) / ey.length_squared()).clamp(0.0, 1.0))
    }
}

/// A triangle diced for the view `vp` on a `size` pixel frame. Corners
/// behind the near plane are taken at it.
fn dice_triangle(vp: Mat4, p: [Vec3; 3], near: f32, size: (usize, usize), rate: f32) -> TriDice {
    let (hw, hh) = (size.0 as f32 * 0.5, size.1 as f32 * 0.5);
    let c = p.map(|q| vp * q.extend(1.0));
    let w = c.map(|q| q.w.max(near));
    let px = [0, 1, 2].map(|k| glam::Vec2::new(c[k].x / w[k] * hw, c[k].y / w[k] * hh));
    let base = (0..3).max_by(|&a, &b| px[a].distance_squared(px[(a + 1) % 3]).total_cmp(&px[b].distance_squared(px[(b + 1) % 3]))).unwrap_or(0);
    let mut d = TriDice { base: base as u8, w, px, ..Default::default() };
    let (_, ex, ey) = d.frame();
    d.lu = raster::dice_level_at(ex.length(), rate);
    d.lv = raster::dice_level_at(ey.length(), rate);
    d.level = d.lu.max(d.lv);
    d
}

/// Widest spacing on screen, under view `vp`, of the grid `d` keeps, from
/// its steps at the triangle's corners; `None` when a corner is behind the
/// near plane.
fn kept_spacing(d: &TriDice, p: [Vec3; 3], vp: Mat4, near: f32, size: (usize, usize)) -> Option<f32> {
    let (hw, hh) = (size.0 as f32 * 0.5, size.1 as f32 * 0.5);
    let (du, dv) = (1.0 / (1u32 << d.lu) as f32, 1.0 / (1u32 << d.lv) as f32);
    let to_px = |u: f32, v: f32| -> Option<glam::Vec2> {
        let b = screen_to_tri(d.grid_to_screen(u, v), d.w);
        let q = vp * (p[0] * b[0] + p[1] * b[1] + p[2] * b[2]).extend(1.0);
        (q.w >= near).then(|| glam::Vec2::new(q.x / q.w * hw, q.y / q.w * hh))
    };
    let mut widest = 0f32;
    for k in 0..3 {
        let mut s = [0.0; 3];
        s[k] = 1.0;
        let (u, v) = d.screen_to_grid(s);
        let (u2, v2) = (if u > 0.5 { u - du } else { u + du }, if v > 0.5 { v - dv } else { v + dv });
        let a = to_px(u, v)?;
        widest = widest.max(a.distance(to_px(u2, v)?)).max(a.distance(to_px(u, v2)?));
    }
    Some(widest)
}

/// A triangle's own weights from its screen weights `s`, for corner
/// depths `w` (perspective-correct interpolation).
#[inline]
fn screen_to_tri(s: [f32; 3], w: [f32; 3]) -> [f32; 3] {
    let b = [s[0] / w[0], s[1] / w[1], s[2] / w[2]];
    let t = b[0] + b[1] + b[2];
    let t = if t.abs() < 1e-30 { 1e-30 } else { t };
    [b[0] / t, b[1] / t, b[2] / t]
}

/// Screen weights from a triangle's own weights `b`.
#[inline]
fn tri_to_screen(b: [f32; 3], w: [f32; 3]) -> [f32; 3] {
    let s = [b[0] * w[0], b[1] * w[1], b[2] * w[2]];
    let t = 1.0 / (s[0] + s[1] + s[2]).max(1e-30);
    [s[0] * t, s[1] * t, s[2] * t]
}

/// Shading kept for one cluster.
#[derive(Default)]
struct ClusterShade {
    /// Finest level kept for any triangle; `NOT_VISIBLE` when none is.
    level: u8,
    /// Per triangle: `UNDICED`, `ON_VERTICES`, or which of `grids` it has.
    state: Vec<u8>,
    /// Grids of triangles diced above level 0; an entry at `NOT_VISIBLE` is
    /// free, its points kept for the next triangle that fits.
    grids: Vec<TriDice>,
    /// Grid points of triangles diced above level 0.
    diffuse: Vec<Vec3>,
    spec: Vec<Vec3>,
    /// Grid points in use, against `diffuse.len()`: the rest is garbage
    /// left by triangles diced again at a larger size.
    live: usize,
    /// Grid triangles whose diffuse light is current.
    done: u128,
    /// Grid triangles whose highlight is current for `spec_eye`.
    spec_done: u128,
    /// Grid triangles seen from their back when last shaded. Shading is
    /// two-sided: normals are turned towards the eye, so a triangle seen
    /// from the other side has its diffuse light redone.
    back: u128,
    /// The cluster's vertices, shaded for triangles at level 0.
    vdiffuse: Vec<Vec3>,
    vspec: Vec<Vec3>,
    vdone: u128,
    vspec_done: u128,
    vback: u128,
    spec_eye: Vec3,
    /// Each point light's share of the kept shading, so a light that
    /// changes has only its own share redone: the lights with a share
    /// (`slots`, the hot light), and per slot, per grid point and per
    /// vertex, its diffuse and highlight.
    slots: Vec<u8>,
    gshare: Vec<Vec<[Vec3; 2]>>,
    vshare: Vec<Vec<[Vec3; 2]>>,
    /// Lights changed since this cluster's shading was kept, one bit each
    /// (`light ids`: point lights from 0, directional lights from
    /// `DIR_ID`).
    light_todo: u128,
    /// The view, rate and triangles needed when its grids were last
    /// checked, and what that found (vertices and grids needed, points
    /// used): while the camera holds still the check is not done again.
    diced_for: Option<(Mat4, f32, u128)>,
    diced: (u128, u128, u64),
    /// The frame in which anything here was last shaded.
    stamp: u64,
    /// The frame in which it was last on screen.
    seen: u64,
}

/// A triangle not diced yet.
const UNDICED: u8 = u8::MAX;
/// A triangle under two pixels, read from the cluster's vertices.
const ON_VERTICES: u8 = u8::MAX - 1;

impl ClusterShade {
    fn empty() -> Self {
        Self { level: NOT_VISIBLE, ..Default::default() }
    }

    /// Bytes held.
    fn bytes(&self) -> usize {
        (self.diffuse.capacity() + self.spec.capacity() + self.vdiffuse.capacity() + self.vspec.capacity()) * 12
            + self.state.capacity()
            + self.grids.capacity() * std::mem::size_of::<TriDice>()
            + self.gshare.iter().chain(&self.vshare).map(|v| v.capacity() * 24).sum::<usize>()
    }

    /// Room for `pts` more grid points (and their shares); where they start.
    fn grow(&mut self, pts: usize) -> u32 {
        let at = self.diffuse.len();
        self.diffuse.resize(at + pts, Vec3::ZERO);
        self.spec.resize(at + pts, Vec3::ZERO);
        for sh in &mut self.gshare {
            sh.resize(at + pts, [Vec3::ZERO; 2]);
        }
        at as u32
    }

    /// The slot of light `l`'s share, made (all zero: it gave nothing) if
    /// it has none.
    fn slot_of(&mut self, l: usize) -> usize {
        if let Some(k) = self.slots.iter().position(|&s| s as usize == l) {
            return k;
        }
        self.slots.push(l as u8);
        self.gshare.push(vec![[Vec3::ZERO; 2]; self.diffuse.len()]);
        self.vshare.push(vec![[Vec3::ZERO; 2]; self.vdiffuse.len()]);
        self.slots.len() - 1
    }

    /// Dicing level of triangle `u` (`NOT_VISIBLE`: not diced).
    fn tri_level(&self, u: usize) -> u8 {
        match self.state.get(u).copied().unwrap_or(UNDICED) {
            UNDICED => NOT_VISIBLE,
            ON_VERTICES => 0,
            i => self.grids[i as usize].level,
        }
    }

    /// Grid of triangle `u`, when it has one.
    fn grid(&self, u: usize) -> Option<&TriDice> {
        match self.state.get(u).copied().unwrap_or(UNDICED) {
            UNDICED | ON_VERTICES => None,
            i => Some(&self.grids[i as usize]),
        }
    }

    fn init(&mut self, tris: usize, verts: usize) {
        self.state = vec![UNDICED; tris];
        self.vdiffuse = vec![Vec3::ZERO; verts];
        self.vspec = vec![Vec3::ZERO; verts];
        self.slots = Vec::new();
        self.gshare = Vec::new();
        self.vshare = Vec::new();
        self.light_todo = 0;
        self.diced_for = None;
        self.forget_grids();
        self.vdone = 0;
        self.vspec_done = 0;
        self.vback = 0;
    }

    /// Light changed: everything kept is shaded again, grids kept.
    fn forget_light(&mut self) {
        self.done = 0;
        self.spec_done = 0;
        self.vdone = 0;
        self.vspec_done = 0;
        self.slots = Vec::new();
        self.gshare = Vec::new();
        self.vshare = Vec::new();
        self.light_todo = 0;
        self.diced_for = None;
    }

    /// Drops every grid and its memory, keeping the vertices.
    fn forget_grids(&mut self) {
        for st in &mut self.state {
            if *st != ON_VERTICES {
                *st = UNDICED;
            }
        }
        self.grids = Vec::new();
        self.diffuse = Vec::new();
        self.spec = Vec::new();
        for sh in &mut self.gshare {
            *sh = Vec::new();
        }
        self.live = 0;
        self.diced_for = None;
        self.done = 0;
        self.spec_done = 0;
        self.back = 0;
    }
}

#[derive(PartialEq, Clone, Copy)]
struct VisKey {
    occlusion: bool,
    tile: usize,
    view: Mat4,
    proj: Mat4,
    size: (usize, usize),
    cull: bool,
    wireframe: bool,
    lod_px: u32,
    /// The clusters drawn coarser while they move, and how much.
    moving: u64,
}

/// A dragged object's layer turns this far (radians) or grows or
/// shrinks this many times on screen before it is captured again.
const LAYER_TURN: f32 = 0.2;
const LAYER_GROW: f32 = 1.3;

/// The picture without the dragged object, for the view `key`.
struct Background {
    key: (Mat4, usize, usize),
    color: Vec<u32>,
    depth: Vec<f32>,
    /// Its triangle ids; the frame keeps the setup records they index.
    tri_id: Vec<u32>,
}

/// The dragged object alone: each pixel it covered as a point in the
/// world and its colour, captured when it had been moved by `ahead` from
/// where the drag began, `distance` from the eye.
struct Layer {
    ahead: glam::Affine3A,
    distance: f32,
    points: Vec<(Vec3, u32)>,
}

/// How long each part of a bake spread over frames may take, ms.
pub const BAKE_MS: f32 = 50.0;

/// Where `render_pass` stops, for a bake spread over frames.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Stop {
    Never,
    /// After the shadow maps, the sky shares and bounce light's surfels.
    AfterBounce,
    /// After shading, until this many ms from the start of the frame.
    Shade(f32),
    /// After gathering bounce light for this many ms (None: all of it).
    Gather(Option<f32>),
    /// Resolve these rows in full, and no others; no bounce gathering.
    Rows(usize, usize),
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum BakeStage {
    Shadows,
    Sky,
    Cloud,
    Bounce,
    Shade,
    /// Bounce light gathered, for up to `BAKE_GATHER_FRAMES` frames; the
    /// rest is gathered after, as usual.
    Gather(u32),
    Resolve,
}

/// Most frames a bake waits for bounce light.
const BAKE_GATHER_FRAMES: u32 = 4;

/// A dragged object being lit where it was let go, a part a frame.
struct Bake {
    /// The last picture of the drag, shown until it is done.
    shown: Vec<u32>,
    stage: BakeStage,
    /// The view it was let go in; another view finishes it at once.
    view: (Mat4, usize, usize),
    /// The next row to resolve, and how many at a time.
    row: usize,
    rows_per: usize,
}

/// An object being dragged (`Renderer::dragged`).
struct Drag {
    clusters: Vec<std::ops::Range<usize>>,
    /// Its box when the drag began, and now.
    start: (Vec3, Vec3),
    now: (Vec3, Vec3),
    /// From where it is to where it was when the drag began, and its turn.
    back: glam::Affine3A,
    turn: glam::Quat,
    background: Option<Background>,
    layer: Option<Layer>,
    /// The last picture drawn while dragging.
    shown: Vec<u32>,
}

/// Cluster `c` taken back by `back` (turning by `turn`): where it was when
/// a drag began.
fn carry(c: &crate::scene::Cluster, back: glam::Affine3A, turn: glam::Quat) -> crate::scene::Cluster {
    let k = back.matrix3.x_axis.length();
    let mut h = *c;
    h.xf = back * c.xf;
    h.turn = (turn * c.turn).normalize();
    h.turned = true;
    h.centre = back.transform_point3(c.centre);
    h.radius = c.radius * k * (1.0 + 1e-5) + 1e-6;
    h.cone_axis = turn * c.cone_axis;
    h
}

static NEXT_STAMP: AtomicU64 = AtomicU64::new(1);

/// Everything kept from one frame to the next.
pub struct Renderer {
    scene_key: (usize, usize, usize),
    cache: Vec<ClusterShade>,
    shadows: Vec<Option<CubeShadow>>,
    prev_lights: Vec<PointLight>,
    prev_reach: Vec<f32>,
    /// Boxes objects were moved from and to since the last frame.
    moves: Vec<(Vec3, Vec3)>,
    /// The light changed last (`HOT_LIGHTS` at most, most recent first):
    /// only it keeps a share of its own in the shading cache, so memory
    /// grows by one share a point.
    hot: Vec<usize>,
    prev_global: Option<(Sky, raster::Sun, f32, bool, bool, bool, usize, bool, bool)>,
    /// The directional lights as last shaded.
    prev_dirs: [raster::DirLight; raster::MAX_DIR_LIGHTS],
    /// The sun's shadow map, and what it was built for (direction, texels,
    /// scene).
    sun_shadow: Option<SunShadow>,
    sun_key: Option<(Vec3, usize, (usize, usize, usize))>,
    /// How point lights' shadow maps were drawn: shadows on, texels, lights.
    shadow_how: Option<(bool, usize, usize)>,
    /// Each directional light's shadow map, and what it was built for.
    dir_shadows: Vec<Option<(SunShadow, (Vec3, usize, (usize, usize, usize)))>>,
    prev_backdrop: Option<raster::Backdrop>,
    /// The surfel cloud and the bounce light at each receiver.
    bounce: Option<Bounce>,
    vis_key: Option<VisKey>,
    vis_stamp: u64,
    levels: Vec<u8>,
    /// Units needed by the current visibility, per cluster (triangles).
    need: Vec<u128>,
    visible: Vec<u32>,
    /// The last resolved picture, before light gizmos are drawn on it.
    base: Vec<u32>,
    /// Which view the base was resolved with.
    base_view: Option<View>,
    frame_no: u64,
    /// The shading rate is multiplied by this to fit the cache budget.
    rate_scale: f32,
    /// The object being dragged, if any (`dragged`).
    drag: Option<Drag>,
    /// A drag being baked over frames (`render_bake`).
    bake: Option<Bake>,
    /// The surfels were made outside a frame and are still to be lit.
    relight_surfels: bool,
    /// Each sample's share of the sky, for the sky contrast it was made for.
    sky_share: Option<(SkyShare, u32)>,
    /// Each surfel's share of the sky, for the shares above.
    surfel_share: Vec<f32>,
}

#[cfg_attr(not(target_pointer_width = "64"), allow(dead_code))]
/// A read-only look at what the renderer keeps, for the visual debug views.
pub struct Peek<'a> {
    pub frame_no: u64,
    cache: &'a [ClusterShade],
    pub shadows: &'a [Option<CubeShadow>],
    pub bounce: Option<&'a Bounce>,
    pub sun_shadow: Option<&'a SunShadow>,
    pub sky_share: Option<&'a SkyShare>,
}

#[cfg_attr(not(target_pointer_width = "64"), allow(dead_code))]
impl Peek<'_> {
    /// Dicing level of the shading kept for cluster `c` (`NOT_VISIBLE`: none).
    pub fn kept_level(&self, c: usize) -> u8 {
        self.cache.get(c).map(|e| e.level).unwrap_or(NOT_VISIBLE)
    }

    /// Dicing level kept for triangle `tri` of cluster `c`.
    pub fn tri_level(&self, c: usize, tri: u32, scene: &Scene) -> u8 {
        self.cache.get(c).map(|e| e.tri_level((tri - scene.clusters[c].first_tri) as usize)).unwrap_or(NOT_VISIBLE)
    }

    /// Where weights `w` of triangle `tri` of cluster `c` fall in its kept
    /// grid, in grid steps, and the grid's level. `None` at level 0 or when
    /// not diced.
    pub fn grid_at(&self, c: usize, tri: u32, scene: &Scene, w: [f32; 3]) -> Option<(f32, f32, u8)> {
        let d = self.tri(c, tri, scene)?;
        if d.level == 0 || d.level == NOT_VISIBLE {
            return None;
        }
        let (u, v) = d.screen_to_grid(tri_to_screen(w, d.w));
        Some((u * (1u32 << d.lu) as f32, v * (1u32 << d.lv) as f32, d.level))
    }

    fn tri(&self, c: usize, tri: u32, scene: &Scene) -> Option<&TriDice> {
        self.cache.get(c)?.grid((tri - scene.clusters[c].first_tri) as usize)
    }

    /// The frame in which cluster `c` was last shaded.
    pub fn shaded_in(&self, c: usize) -> u64 {
        self.cache.get(c).map(|e| e.stamp).unwrap_or(0)
    }

    /// Kept radiance of triangle `tri` of cluster `c` at weights `w`.
    pub fn radiance(&self, scene: &Scene, c: usize, tri: u32, w: [f32; 3], part: Part) -> Vec3 {
        match self.cache.get(c) {
            Some(e) => lookup_part(scene, &scene.clusters[c], e, tri, w, part),
            None => Vec3::ZERO,
        }
    }
}

impl Default for Renderer {
    fn default() -> Self {
        Self::new()
    }
}

impl Renderer {
    pub fn new() -> Self {
        Self {
            scene_key: (0, 0, 0),
            cache: Vec::new(),
            shadows: Vec::new(),
            prev_lights: Vec::new(),
            prev_reach: Vec::new(),
            hot: Vec::new(),
            moves: Vec::new(),
            prev_global: None,
            sun_shadow: None,
            sun_key: None,
            dir_shadows: (0..raster::MAX_DIR_LIGHTS).map(|_| None).collect(),
            shadow_how: None,
            prev_dirs: [raster::DirLight::default(); raster::MAX_DIR_LIGHTS],
            prev_backdrop: None,
            bounce: None,
            vis_key: None,
            vis_stamp: 0,
            levels: Vec::new(),
            need: Vec::new(),
            visible: Vec::new(),
            base: Vec::new(),
            base_view: None,
            frame_no: 0,
            rate_scale: 1.0,
            drag: None,
            bake: None,
            relight_surfels: false,
            sky_share: None,
            surfel_share: Vec::new(),
        }
    }

    /// An object is being dragged, as ZBrush moves a model in 2.5D while
    /// sculpting and bakes it to 3D after: its clusters `clusters` went
    /// from the box `from` to the box `to`, and `back` takes where they
    /// are now to where they were when the drag began. Nothing about light
    /// is redone for it: its shading goes with it, as lit where it began
    /// (points that come into view are lit there too), and the rest of the
    /// scene keeps its shading and shadow maps. The scene without it and
    /// the object alone are drawn once, the object coarser
    /// (`RenderSettings::moving_rate`, `moving_detail`), and each frame
    /// moves the object's picture with it over the rest (`render_drag`).
    /// `drag_ended` bakes it.
    pub fn dragged(&mut self, clusters: Vec<std::ops::Range<usize>>, from: (Vec3, Vec3), to: (Vec3, Vec3), back: glam::Affine3A) {
        let (_, turn, _) = back.to_scale_rotation_translation();
        // A bake still going is overtaken by the new drag: its bounce light
        // is made again with everything else.
        if let Some(b) = self.bake.take() {
            self.finish_bake_resets(b.stage);
        }
        match &mut self.drag {
            Some(d) => {
                d.clusters = clusters;
                d.now = to;
                d.back = back;
                d.turn = turn.normalize();
            }
            None => self.drag = Some(Drag { clusters, start: from, now: to, back, turn: turn.normalize(), background: None, layer: None, shown: Vec::new() }),
        }
        self.vis_key = None;
    }

    /// The materials of clusters `clusters` changed: their shading goes,
    /// and is made again where they are seen.
    pub fn materials_changed(&mut self, clusters: std::ops::Range<usize>) {
        for c in clusters {
            if let Some(e) = self.cache.get_mut(c) {
                *e = ClusterShade::empty();
            }
        }
        // The resolved picture keeps nothing of them.
        self.base_view = None;
    }

    /// The drag ended: the object is lit where it stands, as `moved` from
    /// where the drag began, and bounce light is made again.
    ///
    /// With `over_frames` the bake is spread over the next frames, none
    /// much over `BAKE_MS`, and the last picture of the drag stays on
    /// screen until it is done (`render_bake`).
    pub fn drag_ended(&mut self, over_frames: bool) {
        let Some(d) = self.drag.take() else {
            self.bounce = None;
            return;
        };
        self.vis_key = None;
        let share = self.sky_share.take();
        for r in d.clusters.clone() {
            self.moved(r, d.start, d.now, false);
        }
        match d.background {
            Some(bg) if over_frames && d.shown.len() == bg.color.len() => {
                // The sky shares are made again a frame later.
                self.sky_share = share;
                self.bake = Some(Bake { shown: d.shown, stage: BakeStage::Shadows, view: bg.key, row: 0, rows_per: 64 });
            }
            _ => self.bounce = None,
        }
    }

    /// What a bake left at `stage` had yet to throw away, thrown away now.
    fn finish_bake_resets(&mut self, stage: BakeStage) {
        if stage == BakeStage::Shadows || stage == BakeStage::Sky {
            self.sky_share = None;
        }
        if matches!(stage, BakeStage::Shadows | BakeStage::Sky | BakeStage::Cloud) {
            self.bounce = None;
        }
    }

    /// A frame of the scene, or the next part of a bake. Each part fits in
    /// about `BAKE_MS`, except drawing the shadow maps again and making
    /// the surfel cloud and the sky shares, a part each. The last picture
    /// of the drag is shown until the bake has resolved every row.
    fn render_bake(&mut self, scene: &Scene, cam: &Camera, lights: &[PointLight], settings: &RenderSettings, frame: &mut Frame) {
        let Some(b) = self.bake.as_ref() else {
            self.render_pass(scene, cam, lights, settings, frame, None, Stop::Never);
            return;
        };
        let (w, h) = (frame.width, frame.height);
        if b.view != (cam.proj * cam.view, w, h) || !settings.bake_over_frames {
            // The view changed: the rest at once.
            let b = self.bake.take().unwrap();
            self.finish_bake_resets(b.stage);
            self.vis_key = None;
            self.render_pass(scene, cam, lights, settings, frame, None, Stop::Never);
            return;
        }
        let t = std::time::Instant::now();
        let stage = b.stage;
        let mut next = stage;
        let mut done = false;
        match stage {
            BakeStage::Shadows => {
                // Shadow maps drawn again where it was and is, and what is
                // in its shadows marked.
                self.render_pass(scene, cam, lights, settings, frame, None, Stop::AfterBounce);
                next = BakeStage::Sky;
            }
            BakeStage::Sky => {
                self.sky_share = None;
                self.render_pass(scene, cam, lights, settings, frame, None, Stop::AfterBounce);
                next = BakeStage::Cloud;
            }
            BakeStage::Cloud => {
                // The surfels and receivers where things are now, lit next.
                if settings.bounce {
                    self.bounce = Some(Bounce::new(scene));
                    self.relight_surfels = true;
                } else {
                    self.bounce = None;
                }
                next = BakeStage::Bounce;
            }
            BakeStage::Bounce => {
                self.render_pass(scene, cam, lights, settings, frame, None, Stop::AfterBounce);
                next = BakeStage::Shade;
            }
            BakeStage::Shade => {
                if self.render_pass(scene, cam, lights, settings, frame, None, Stop::Shade(BAKE_MS)) {
                    next = BakeStage::Gather(0);
                }
            }
            BakeStage::Gather(k) => {
                let budget = settings.bounce_budget_ms.map(|ms| ms.max(BAKE_MS));
                let all = self.render_pass(scene, cam, lights, settings, frame, None, Stop::Gather(budget));
                next = if all || k + 1 >= BAKE_GATHER_FRAMES { BakeStage::Resolve } else { BakeStage::Gather(k + 1) };
            }
            BakeStage::Resolve => {
                let (r0, n) = (b.row, b.rows_per);
                let r1 = (r0 + n).min(h);
                self.render_pass(scene, cam, lights, settings, frame, None, Stop::Rows(r0, r1));
                let ms = t.elapsed().as_secs_f32() * 1000.0;
                let b = self.bake.as_mut().unwrap();
                b.row = r1;
                // Rows for the next part, from how long these took.
                b.rows_per = ((n as f32 * BAKE_MS / ms.max(1.0)) as usize).clamp(8, h.max(8));
                done = r1 >= h;
            }
        }
        if done {
            // Every row resolved: the baked picture.
            self.bake = None;
            frame.stats.baking = false;
            return;
        }
        let b = self.bake.as_mut().unwrap();
        b.stage = next;
        frame.color.copy_from_slice(&b.shown);
        frame.stats.baking = true;
    }

    /// For tests: in which frame each cluster's bounce light was last
    /// gathered.
    #[cfg(test)]
    pub fn bounce_stamps(&self) -> Option<&[u64]> {
        self.bounce.as_ref().map(|b| &b.cluster_stamp[..])
    }

    #[cfg_attr(not(target_pointer_width = "64"), allow(dead_code))]
    pub fn peek(&self) -> Peek<'_> {
        Peek { frame_no: self.frame_no, cache: &self.cache, shadows: &self.shadows, bounce: self.bounce.as_ref(), sun_shadow: self.sun_shadow.as_ref(), sky_share: self.sky_share.as_ref().map(|s| &s.0) }
    }

    /// Memory held by the shading cache and the shadow maps, in bytes.
    /// Clusters `moved` were moved (an object's geometry rewritten): their
    /// shading is dropped, everything else is lit again (its shadows may
    /// have changed), and the shadow maps and visibility are made again.
    /// Bounce light is made again too when `bounce`; while an object is
    /// being dragged it is left, and redone when the drag ends.
    /// Clusters `moved` were moved from the box `from` to the box `to`.
    /// Their shading goes; next frame the shadow maps are drawn again
    /// where either box falls on them, and the clusters in the shadow
    /// either box casts from a light or the sun are shaded again (the
    /// rest is kept). `bounce`: make bounce light again too (once the
    /// object settles). The same box twice is the object settling.
    pub fn moved(&mut self, moved: std::ops::Range<usize>, from: (Vec3, Vec3), to: (Vec3, Vec3), bounce: bool) {
        if bounce {
            self.bounce = None;
        }
        if from == to {
            return;
        }
        // Its samples moved and turned.
        self.sky_share = None;
        for c in moved {
            if let Some(e) = self.cache.get_mut(c) {
                *e = ClusterShade::empty();
            }
        }
        for b in [from, to] {
            if !self.moves.contains(&b) {
                self.moves.push(b);
            }
        }
        self.vis_key = None;
    }

    /// Which clusters a pass draws, for the visibility key.
    fn part_key(part: raster::Part<'_>) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        match part {
            raster::Part::All => return 0,
            raster::Part::Without(r) => (1u8, r).hash(&mut h),
            raster::Part::Only(r, k) => (2u8, r, k.to_bits()).hash(&mut h),
        }
        h.finish() | 1
    }

    pub fn memory(&self) -> usize {
        let shade: usize = self.cache.iter().map(|c| c.bytes()).sum();
        let shadows: usize = self.shadows.iter().flatten().map(|s| s.bytes()).sum();
        shade + shadows + self.sun_shadow.as_ref().map(|s| s.bytes()).unwrap_or(0)
    }

    pub fn render(&mut self, scene: &Scene, cam: &Camera, lights: &[PointLight], settings: &RenderSettings, frame: &mut Frame) {
        if self.drag.is_some() && settings.view == View::Shaded {
            self.render_drag(scene, cam, lights, settings, frame);
        } else {
            self.render_bake(scene, cam, lights, settings, frame);
        }
    }

    /// A frame while an object is dragged: the layer of the object, as
    /// captured, moved with it and drawn over the scene without it, by
    /// depth. Both are captured on the drag's first frame; the layer again
    /// when the object has turned or grown on screen too far for it.
    fn render_drag(&mut self, scene: &Scene, cam: &Camera, lights: &[PointLight], settings: &RenderSettings, frame: &mut Frame) {
        let (w, h) = (frame.width, frame.height);
        let vp = cam.proj * cam.view;
        let Some(d) = self.drag.as_ref() else { return };
        let ahead = d.back.inverse();
        let clusters = d.clusters.clone();
        let view_key = (vp, w, h);
        let mut captured = false;
        let mut stats: Option<raster::Stats> = None;
        if d.background.as_ref().is_none_or(|b| b.key != view_key) {
            // The scene without the object.
            self.vis_key = None;
            self.render_pass(scene, cam, lights, settings, frame, Some(raster::Part::Without(&clusters)), Stop::Never);
            let bg = Background { key: view_key, color: frame.color.clone(), depth: frame.depth.clone(), tri_id: frame.tri_id.clone() };
            stats = Some(frame.stats);
            let d = self.drag.as_mut().unwrap();
            d.background = Some(bg);
            d.layer = None;
        }
        let d = self.drag.as_ref().unwrap();
        // How far the object has turned and grown on screen since the
        // layer was captured.
        let centre = (d.now.0 + d.now.1) * 0.5;
        let stale = match &d.layer {
            None => true,
            Some(l) => {
                let rel = ahead * l.ahead.inverse();
                let (k, q, _) = rel.to_scale_rotation_translation();
                let grow = k.x * l.distance / centre.distance(cam.eye).max(1e-4);
                q.angle_between(glam::Quat::IDENTITY) > LAYER_TURN || !(1.0 / LAYER_GROW..=LAYER_GROW).contains(&grow)
            }
        };
        if stale {
            // The object alone, lit as where the drag began.
            self.vis_key = None;
            let mut f = Frame::new(w, h);
            self.render_pass(scene, cam, lights, settings, &mut f, Some(raster::Part::Only(&clusters, settings.moving_detail)), Stop::Never);
            let mut points = Vec::new();
            for (i, &id) in f.tri_id.iter().enumerate() {
                if id == NO_TRI {
                    continue;
                }
                let st = &f.setups[id as usize];
                let c = &scene.clusters[st.cluster as usize];
                let (x, y) = (i % w, i / w);
                let b = raster::barycentrics(st, x as f32 + 0.5, y as f32 + 0.5);
                let bw = [0, 1, 2].map(|k| b[0] * st.bary[0][k] + b[1] * st.bary[1][k] + b[2] * st.bary[2][k]);
                let [i0, i1, i2] = scene.tri(c, st.tri);
                let p = scene.pos(c, i0) * bw[0] + scene.pos(c, i1) * bw[1] + scene.pos(c, i2) * bw[2];
                points.push((p, f.color[i]));
            }
            stats = Some(match stats {
                None => f.stats,
                Some(mut b) => {
                    b.points_shaded += f.stats.points_shaded;
                    b.points_used += f.stats.points_used;
                    b.setup_ms += f.stats.setup_ms;
                    b.raster_ms += f.stats.raster_ms;
                    b.shade_ms += f.stats.shade_ms;
                    b.resolve_ms += f.stats.resolve_ms;
                    b.tris_drawn += f.stats.tris_drawn;
                    b
                }
            });
            let d = self.drag.as_mut().unwrap();
            d.layer = Some(Layer { ahead, distance: centre.distance(cam.eye), points });
            captured = true;
            self.vis_key = None;
        }
        // The scene without it, then the layer moved with it, nearest
        // first by depth.
        let d = self.drag.as_ref().unwrap();
        let (bg, layer) = (d.background.as_ref().unwrap(), d.layer.as_ref().unwrap());
        let t = std::time::Instant::now();
        frame.color.copy_from_slice(&bg.color);
        frame.depth.copy_from_slice(&bg.depth);
        frame.tri_id.copy_from_slice(&bg.tri_id);
        let rel = ahead * layer.ahead.inverse();
        let m = vp * glam::Mat4::from(rel);
        let (hw, hh) = (w as f32 * 0.5, h as f32 * 0.5);
        // The layer's points into a picture of its own, each to the pixel
        // its centre falls in, nearest first.
        let mut lcol = vec![0u32; w * h];
        let mut ldep = vec![0f32; w * h];
        for &(p, col) in &layer.points {
            let q = m * p.extend(1.0);
            if q.w < cam.near {
                continue;
            }
            let inv = 1.0 / q.w;
            let (x, y) = (((q.x * inv + 1.0) * hw).floor(), ((1.0 - q.y * inv) * hh).floor());
            if x < 0.0 || y < 0.0 || x >= w as f32 || y >= h as f32 {
                continue;
            }
            let k = y as usize * w + x as usize;
            if inv > ldep[k] {
                ldep[k] = inv;
                lcol[k] = col;
            }
        }
        // Gaps a pixel wide where it grew or turned: an empty pixel between
        // two of its pixels, across or down, takes the nearer one.
        for _ in 0..2 {
            let (sc, sd) = (lcol.clone(), ldep.clone());
            lcol.par_chunks_mut(w).zip(ldep.par_chunks_mut(w)).enumerate().for_each(|(y, (cr, dr))| {
                if y == 0 || y + 1 >= h {
                    return;
                }
                for x in 1..w - 1 {
                    if dr[x] > 0.0 {
                        continue;
                    }
                    let k = y * w + x;
                    for (a, b) in [(k - 1, k + 1), (k - w, k + w)] {
                        if sd[a] > 0.0 && sd[b] > 0.0 {
                            let near = if sd[a] >= sd[b] { a } else { b };
                            dr[x] = (sd[a] + sd[b]) * 0.5;
                            cr[x] = sc[near];
                            break;
                        }
                    }
                }
            });
        }
        // Over the scene without it, by depth.
        frame.color.par_chunks_mut(w).zip(frame.depth.par_chunks_mut(w)).zip(frame.tri_id.par_chunks_mut(w)).enumerate().for_each(|(y, ((cr, dr), tr))| {
            for x in 0..w {
                let k = y * w + x;
                if ldep[k] > dr[x] {
                    dr[x] = ldep[k];
                    cr[x] = lcol[k];
                    tr[x] = NO_TRI;
                }
            }
        });
        frame.stats = stats.unwrap_or_default();
        frame.stats.clusters_in = scene.clusters.len();
        frame.stats.layer_captured = captured;
        frame.stats.layer_points = layer.points.len();
        frame.stats.layer_ms = t.elapsed().as_secs_f32() * 1000.0;
        frame.vis_stamp = 0;
        let d = self.drag.as_mut().unwrap();
        d.shown.clear();
        d.shown.extend_from_slice(&frame.color);
    }

    /// One frame of the scene, or of `part` of it.
    /// `stop` ends it early, for a bake spread over frames; returns false
    /// when shading was left unfinished for lack of time.
    fn render_pass(&mut self, scene: &Scene, cam: &Camera, lights: &[PointLight], settings: &RenderSettings, frame: &mut Frame, part: Option<raster::Part<'_>>, stop: Stop) -> bool {
        let t_pass = std::time::Instant::now();
        let part = part.unwrap_or(raster::Part::All);
        let n_lights = lights.len().min(64);
        let lights = &lights[..n_lights];
        let key = (scene.qpos.as_ptr() as usize, scene.qpos.len(), scene.clusters.len());
        if key != self.scene_key {
            *self = Self::new();
            self.scene_key = key;
            self.cache = (0..scene.clusters.len()).map(|_| ClusterShade::empty()).collect();
        }
        frame.stats = Default::default();
        self.frame_no += 1;
        let frame_no = self.frame_no;
        // Whether every pixel has to be resolved again, or only those of
        // clusters shaded this frame.
        let mut full_resolve = false;

        // Light reach for this frame.
        let cut = raster::cut_lux(&settings.sky, settings.exposure);
        let reach: Vec<f32> = lights.iter().map(|l| if settings.light_reach { raster::light_reach(l, cut) } else { f32::INFINITY }).collect();

        // What changed since the last frame decides what is thrown away.
        let global = (settings.sky, settings.sun, settings.exposure, settings.light_reach, settings.shadows && settings.shadow_res > 0, settings.bounce, settings.shadow_res, settings.adaptive, settings.sky_occlusion);
        // Without incremental moves (to compare), a move relights all.
        let global_changed = self.prev_global != Some(global) || self.prev_lights.len() != lights.len() || (!settings.incremental_moves && !self.moves.is_empty());
        // Spheres (old and new position, with reach) of lights that changed,
        // and which light.
        let mut spheres: Vec<(Vec3, f32)> = Vec::new();
        let mut changed_lights: Vec<usize> = Vec::new();
        let mut dir_changed = false;
        if global_changed {
            full_resolve = true;
            for c in &mut self.cache {
                c.forget_light();
            }
            self.hot.clear();
            // Point lights' maps stay unless how shadows are drawn changed,
            // a light moved, or its reach grew past what its map holds: the
            // sky, the sun, the exposure and the directional lights do not
            // touch them.
            let how = (settings.shadows, settings.shadow_res, lights.len());
            if self.shadow_how != Some(how) || self.shadows.len() != lights.len() {
                self.shadows = (0..lights.len()).map(|_| None).collect();
                self.shadow_how = Some(how);
            }
            for (i, m) in self.shadows.iter_mut().enumerate() {
                if m.as_ref().is_some_and(|m| m.position != lights[i].position || reach[i] > m.reach) {
                    *m = None;
                }
            }
        } else {
            for (i, l) in lights.iter().enumerate() {
                let old = &self.prev_lights[i];
                if old != l {
                    spheres.push((old.position, self.prev_reach[i]));
                    spheres.push((l.position, reach[i]));
                    changed_lights.push(i);
                    self.hot.retain(|&h| h != i);
                    self.hot.insert(0, i);
                    self.hot.truncate(HOT_LIGHTS);
                    if old.position != l.position || old.lumens != l.lumens {
                        self.shadows[i] = None;
                    }
                }
            }
            // A changed directional light reaches everything: its share is
            // redone everywhere, and it becomes hot.
            let mut dir_bits = 0u128;
            for (k, d) in settings.dir_lights.iter().enumerate() {
                if *d != self.prev_dirs[k] && (d.lights() || self.prev_dirs[k].lights()) {
                    let id = DIR_ID + k;
                    dir_bits |= 1u128 << id;
                    self.hot.retain(|&h| h != id);
                    self.hot.insert(0, id);
                    self.hot.truncate(HOT_LIGHTS);
                    dir_changed = true;
                }
            }
            if dir_bits != 0 {
                full_resolve = true;
                self.cache.par_iter_mut().for_each(|e| e.light_todo |= dir_bits);
            }
            if !spheres.is_empty() {
                // Only the changed lights' shares are redone, where they
                // reach or reached.
                self.cache.par_iter_mut().zip(scene.clusters.par_iter()).for_each(|(e, c)| {
                    for (k, &i) in changed_lights.iter().enumerate() {
                        if spheres[2 * k..2 * k + 2].iter().any(|(p, r)| p.distance(c.centre) <= c.radius + r) {
                            e.light_todo |= 1u128 << i;
                        }
                    }
                });
            }
        }
        self.prev_global = Some(global);
        self.prev_dirs = settings.dir_lights;
        self.prev_lights = lights.to_vec();
        self.prev_reach = reach.clone();

        // Moved objects: shadow maps drawn again where they were and are,
        // and the clusters in the shadows they cast shaded again.
        let ts = std::time::Instant::now();
        let moves = std::mem::take(&mut self.moves);
        if !moves.is_empty() && global_changed {
            // Everything is made again anyway; the sun's map too.
            self.sun_key = None;
            self.dir_shadows.iter_mut().for_each(|d| *d = None);
        }
        if !moves.is_empty() && !global_changed {
            let spheres_m: Vec<(Vec3, f32)> = moves.iter().map(|(lo, hi)| ((*lo + *hi) * 0.5, (*hi - *lo).length() * 0.5)).collect();
            let sun_up = settings.shadows && settings.sun.up();
            // Directions of parallel light: the sun's and the directional
            // lights'.
            let parallel: Vec<Vec3> = (sun_up.then_some(settings.sun.dir)).into_iter().chain(settings.dir_lights.iter().filter(|d| d.lights() && settings.shadows).map(|d| d.dir)).collect();
            for (i, l) in lights.iter().enumerate() {
                let touched: Vec<(Vec3, Vec3)> = moves.iter().zip(&spheres_m).filter(|(_, (c, r))| c.distance(l.position) <= r + reach[i]).map(|(b, _)| *b).collect();
                if touched.is_empty() {
                    continue;
                }
                if let Some(Some(m)) = self.shadows.get_mut(i) {
                    let r = if reach[i].is_finite() { reach[i] } else { scene.radius() * 4.0 + l.position.distance(scene.centre()) };
                    frame.stats.shadow_tris += m.redraw(scene, &touched, r);
                    frame.stats.shadow_maps_built += 1;
                }
            }
            if sun_up {
                if let Some(m) = self.sun_shadow.as_mut() {
                    frame.stats.shadow_tris += m.redraw(scene, &moves);
                    frame.stats.shadow_maps_built += 1;
                }
            }
            for (m, _) in self.dir_shadows.iter_mut().flatten() {
                frame.stats.shadow_tris += m.redraw(scene, &moves);
                frame.stats.shadow_maps_built += 1;
            }
            // Who is in a moved box's shadow, before or after.
            let in_shadow = |c: &crate::scene::Cluster| -> bool {
                spheres_m.iter().any(|&(b, rb)| {
                    // Near enough for its sky occlusion to reach too.
                    let near = c.centre.distance(b) <= c.radius + rb * (1.0 + crate::transfer::INFLUENCE);
                    if near {
                        return true;
                    }
                    let from_light = lights.iter().enumerate().any(|(i, l)| {
                        if l.position.distance(c.centre) > c.radius + reach[i] {
                            return false;
                        }
                        let (to_b, to_c) = (b - l.position, c.centre - l.position);
                        let (db, dc) = (to_b.length(), to_c.length());
                        if db <= rb || dc <= c.radius {
                            return true;
                        }
                        if dc + c.radius < db - rb {
                            return false;
                        }
                        let ang = to_b.dot(to_c) / (db * dc);
                        let spread = (rb / db).min(1.0).asin() + (c.radius / dc).min(1.0).asin();
                        ang.clamp(-1.0, 1.0).acos() <= spread
                    });
                    let from_sun = parallel.iter().any(|&d| {
                        let q = c.centre - b;
                        let along = q.dot(d);
                        along <= rb + c.radius && (q - d * along).length() <= rb + c.radius
                    });
                    from_light || from_sun
                })
            };
            let forgot = AtomicU64::new(0);
            self.cache.par_iter_mut().zip(scene.clusters.par_iter()).for_each(|(e, c)| {
                if in_shadow(c) {
                    e.forget_light();
                    forgot.fetch_add(1, Ordering::Relaxed);
                }
            });
            frame.stats.clusters_relit = forgot.into_inner() as usize;
            full_resolve = true;
        }

        // Shadow maps for lights that have none.
        if settings.shadows {
            let missing: Vec<usize> = (0..lights.len()).filter(|&i| self.shadows[i].is_none()).collect();
            let built: Vec<(usize, CubeShadow)> = missing
                .par_iter()
                .map(|&i| {
                    // With light reach off, a map still has to stop somewhere:
                    // the whole scene.
                    let r = if reach[i].is_finite() { reach[i] } else { scene.radius() * 4.0 + lights[i].position.distance(scene.centre()) };
                    (i, CubeShadow::build(scene, lights[i].position, r, settings.shadow_res))
                })
                .collect();
            frame.stats.shadow_maps_built += built.len();
            for (i, m) in built {
                frame.stats.shadow_tris += m.tris;
                self.shadows[i] = Some(m);
            }
        }
        // The sun's map, when it is up and its direction or the scene changed.
        if settings.shadows && settings.sun.up() {
            let res = (settings.shadow_res * 4).clamp(256, 2048);
            let key = (settings.sun.dir, res, key);
            if self.sun_key != Some(key) {
                let r = scene.focus_radius();
                let m = SunShadow::build(scene, settings.sun.dir, scene.focus_centre(), &[r * 1.6, 600.0, 25_000.0], res);
                frame.stats.shadow_maps_built += 1;
                frame.stats.shadow_tris += m.tris;
                self.sun_shadow = Some(m);
                self.sun_key = Some(key);
            }
        }
        // Each directional light's map, as the sun's.
        if settings.shadows {
            let res = (settings.shadow_res * 4).clamp(256, 2048);
            let focus = (scene.focus_centre(), scene.focus_radius());
            for (i, d) in settings.dir_lights.iter().enumerate() {
                if !d.lights() {
                    self.dir_shadows[i] = None;
                    continue;
                }
                let k = (d.dir, res, key);
                if self.dir_shadows[i].as_ref().is_none_or(|(_, kk)| *kk != k) {
                    let m = SunShadow::build(scene, d.dir, focus.0, &[focus.1 * 1.6, 600.0, 25_000.0], res);
                    frame.stats.shadow_maps_built += 1;
                    frame.stats.shadow_tris += m.tris;
                    self.dir_shadows[i] = Some((m, k));
                }
            }
        }
        frame.stats.shadow_ms = ts.elapsed().as_secs_f32() * 1000.0;
        // Parallel lights: the sun, then the directional lights.
        let mut suns = vec![SunLit {
            lux: settings.sun.lux(settings.exposure),
            dir: settings.sun.dir,
            shadow: if settings.shadows && settings.sun.up() { self.sun_shadow.as_ref() } else { None },
        }];
        for (i, d) in settings.dir_lights.iter().enumerate() {
            if d.lights() {
                suns.push(SunLit { lux: d.color * d.lux, dir: d.dir, shadow: if settings.shadows { self.dir_shadows[i].as_ref().map(|s| &s.0) } else { None } });
            }
        }
        let sun = &suns[..];
        // The sun alone, and each directional light by its slot, for the
        // sum and the shares.
        let sun_main = [SunLit { lux: suns[0].lux, dir: suns[0].dir, shadow: suns[0].shadow }];
        let dir_lit: Vec<Option<SunLit>> = settings
            .dir_lights
            .iter()
            .enumerate()
            .map(|(i, d)| d.lights().then(|| SunLit { lux: d.color * d.lux, dir: d.dir, shadow: if settings.shadows { self.dir_shadows[i].as_ref().map(|s| &s.0) } else { None } }))
            .collect();
        let dir_lit = &dir_lit;
        let sun_main = &sun_main;
        // Whether a light id stands for a light that is there.
        let n_points = lights.len();
        let valid = move |l: usize| l < n_points || (l >= DIR_ID && dir_lit.get(l - DIR_ID).is_some_and(|d| d.is_some()));

        // Bounce: light the surfels again where light changed, then throw
        // away the receivers that bounce could reach from there.
        let tb = std::time::Instant::now();
        let shadows: Vec<Option<&CubeShadow>> = (0..lights.len()).map(|i| if settings.shadows { self.shadows[i].as_ref() } else { None }).collect();
        let sky = settings.sky;
        // Each sample's share of the sky, made again when the sky's shape
        // changes or an object moves.
        let contrast = sky.contrast.to_bits();
        let mut share_changed = false;
        if !(settings.sky_occlusion && sky.on) {
            share_changed = self.sky_share.is_some();
            self.sky_share = None;
        } else if self.sky_share.as_ref().is_none_or(|s| s.1 != contrast) {
            self.sky_share = Some((SkyShare::new(scene, sky.contrast), contrast));
            share_changed = true;
        }
        let sky_share = self.sky_share.as_ref().map(|s| &s.0);
        if settings.bounce {
            let new_cloud = self.bounce.is_none();
            let b = self.bounce.get_or_insert_with(|| Bounce::new(scene));
            if new_cloud || share_changed || self.surfel_share.len() != b.surfels.len() {
                self.surfel_share = match sky_share {
                    Some(s) => b.surfels.par_iter().map(|sf| s.at(sf.cluster as usize, scene, sf.position, sf.normal)[0]).collect(),
                    None => vec![1.0; b.surfels.len()],
                };
            }
            let surfel_share = &self.surfel_share;
            // A new surfel cloud (an object moved) is lit whole.
            let all = global_changed || new_cloud || dir_changed || std::mem::take(&mut self.relight_surfels);
            if all || !spheres.is_empty() {
                let lit = AtomicU64::new(0);
                // The brightest change, as an intensity (cd): how far the
                // change in bounce light can be seen.
                let changed = std::sync::Mutex::new(0f32);
                b.radiance.par_iter_mut().zip(b.surfels.par_iter()).zip(surfel_share.par_iter()).for_each(|((r, sf), &share)| {
                    if !all && !spheres.iter().any(|(p, rr)| p.distance(sf.position) <= rr + (sf.area / std::f32::consts::PI).sqrt()) {
                        return;
                    }
                    let in_reach: Vec<usize> = (0..lights.len()).filter(|&i| lights[i].position.distance(sf.position) <= reach[i]).collect();
                    // Direct light only, without the sky's ground term: the
                    // ground is in the cloud.
                    let new = shade_point(sf.position, sf.normal, sf.albedo, sf.position + sf.normal, Sky { lux: sky.lux * share, ..sky }, false, lights, &in_reach, &reach, &shadows, sun).0;
                    let delta = (new - *r).abs().max_element() * sf.area;
                    *r = new;
                    lit.fetch_add(1, Ordering::Relaxed);
                    if delta > 0.0 {
                        let mut m = changed.lock().unwrap();
                        *m = m.max(delta);
                    }
                });
                frame.stats.surfels_lit = lit.into_inner();
                b.refit();
                let bounce_reach = (changed.into_inner().unwrap() / cut.max(1e-12)).sqrt();
                if all {
                    b.valid.iter_mut().for_each(|v| *v = false);
                } else {
                    for (r, v) in b.valid.iter_mut().enumerate() {
                        let p = b.receivers[r].0;
                        if spheres.iter().any(|(c, rr)| c.distance(p) <= rr + bounce_reach) {
                            *v = false;
                        }
                    }
                }
            }
        }
        frame.stats.bounce_ms = tb.elapsed().as_secs_f32() * 1000.0;
        if stop == Stop::AfterBounce {
            return true;
        }

        // Visibility, unless the camera and the frame are as they were.
        let vk = VisKey { occlusion: settings.occlusion, tile: settings.tile, view: cam.view, proj: cam.proj, size: (frame.width, frame.height), cull: settings.cull_backfaces, wireframe: settings.wireframe, lod_px: settings.lod_px().to_bits(), moving: Self::part_key(part) };
        if self.vis_key == Some(vk) && frame.vis_stamp == self.vis_stamp && self.vis_stamp != 0 {
            frame.stats.visibility_kept = true;
            frame.stats.clusters_in = scene.clusters.len();
            frame.stats.clusters_drawn = self.levels.iter().filter(|l| **l != NOT_VISIBLE).count();
            frame.stats.tris_in = scene.drawable_triangles();
            frame.stats.tris_drawn = frame.setups.len();
        } else {
            full_resolve = true;
            // What owned a pixel last frame is drawn first, to hide the rest
            // behind.
            let prev = (settings.occlusion && self.need.len() == scene.clusters.len()).then_some(&self.need[..]);
            self.levels = raster::visibility(scene, cam, settings.cull_backfaces, settings.wireframe, settings.tile, settings.lod_px(), part, prev, frame);
            self.vis_key = Some(vk);
            self.vis_stamp = NEXT_STAMP.fetch_add(1, Ordering::Relaxed);
            frame.vis_stamp = self.vis_stamp;
            // Which triangles of which clusters are seen.
            self.need.clear();
            self.need.resize(scene.clusters.len(), 0);
            for &id in &frame.tri_id {
                if id != NO_TRI {
                    let s = &frame.setups[id as usize];
                    let c = &scene.clusters[s.cluster as usize];
                    self.need[s.cluster as usize] |= 1u128 << (s.tri - c.first_tri);
                }
            }
            self.visible = (0..scene.clusters.len() as u32).filter(|&c| self.need[c as usize] != 0).collect();
        }

        // Shade what is needed and not already kept.
        let t_shade = std::time::Instant::now();
        let evals = AtomicU64::new(0);
        let shaded = AtomicU64::new(0);
        let used = AtomicU64::new(0);
        let clusters_shaded = AtomicU64::new(0);
        let eye = cam.eye;
        let need = &self.need;
        let ground = !settings.bounce;
        let hot = &self.hot;
        let exposure = settings.exposure.max(1e-12);
        let adaptive = settings.adaptive;
        let interpolated = AtomicU64::new(0);
        let vp = cam.proj * cam.view;
        let (near, size) = (cam.near, (frame.width, frame.height));
        let still_rate = settings.shading_rate_px.max(0.25) * self.rate_scale;
        let moving_rate = still_rate * settings.moving_rate.max(1.0);
        let drag = self.drag.as_ref().map(|d| (&d.clusters[..], d.back, d.turn));
        let shade_budget = match stop {
            Stop::Shade(ms) => Some(ms),
            _ => None,
        };
        let unfinished = std::sync::atomic::AtomicBool::new(false);
        if settings.view == View::Shaded {
            self.cache.par_iter_mut().enumerate().for_each(|(ci, e)| {
                let need_tris = need[ci];
                if need_tris == 0 {
                    return;
                }
                // Out of time: the rest next frame.
                if shade_budget.is_some_and(|ms| t_pass.elapsed().as_secs_f32() * 1000.0 > ms) {
                    unfinished.store(true, Ordering::Relaxed);
                    return;
                }
                let c = &scene.clusters[ci];
                let fv = c.first_vertex as usize;
                // A dragged cluster is diced where it is and shaded where
                // the drag began.
                let carried = drag.filter(|(r, _, _)| r.iter().any(|r| r.contains(&ci))).map(|(_, back, turn)| (carry(c, back, turn), back.transform_point3(eye)));
                let rate = if carried.is_some() { moving_rate } else { still_rate };
                e.seen = frame_no;
                if e.state.len() != c.tri_count as usize {
                    e.init(c.tri_count as usize, c.vertex_count as usize);
                }
                // Too much left by grids that moved to larger places:
                // start the grids over.
                if e.diffuse.len() > 2 * e.live + 4096 {
                    e.forget_grids();
                }
                // A dragged cluster's highlights go with it.
                if e.spec_eye != eye && carried.is_none() {
                    e.spec_done = 0;
                    e.vspec_done = 0;
                    e.spec_eye = eye;
                }
                // Dice each needed triangle from its own size on screen.
                // A kept grid stays while it is at most one level finer,
                // edge by edge, and its spacing on screen within a factor
                // of two.
                let (mut vneed, mut tneed, mut n_used) = (0u128, 0u128, 0u64);
                let still = e.diced_for == Some((vp, rate, need_tris));
                let mut bits = if still { 0 } else { need_tris };
                if still {
                    (vneed, tneed, n_used) = e.diced;
                }
                while bits != 0 {
                    let u = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    let idx = scene.tri(c, c.first_tri + u as u32);
                    let corners = idx.map(|i| scene.pos(c, i));
                    let cur = match e.state[u] {
                        UNDICED | ON_VERTICES => None,
                        i => Some(i as usize),
                    };
                    // A kept grid stays while its points are no further
                    // apart on screen than the rate and no more than a
                    // level finer than a fresh one.
                    let keep = cur.is_some_and(|i| kept_spacing(&e.grids[i], corners, vp, near, size).is_some_and(|s| s <= rate * 1.05 && s > rate * 0.25));
                    if !keep {
                        let fresh = dice_triangle(vp, corners, near, size, rate);
                        if let Some(i) = cur {
                            e.live -= e.grids[i].points();
                            e.grids[i].level = NOT_VISIBLE;
                        }
                        if fresh.level == 0 {
                            e.state[u] = ON_VERTICES;
                            for i in idx {
                                vneed |= 1u128 << (i - fv);
                            }
                            continue;
                        }
                        // Its own entry, or a free one, keeping their points.
                        let i = cur.or_else(|| e.grids.iter().position(|g| g.level == NOT_VISIBLE)).unwrap_or_else(|| {
                            e.grids.push(TriDice::default());
                            e.grids.len() - 1
                        });
                        let pts = fresh.points();
                        let (mut offset, mut cap) = (e.grids[i].offset, e.grids[i].cap);
                        if (cap as usize) < pts {
                            offset = e.grow(pts);
                            cap = pts as u32;
                        }
                        e.grids[i] = TriDice { offset, cap, ..fresh };
                        e.state[u] = i as u8;
                        e.live += pts;
                        e.done &= !(1u128 << u);
                        e.spec_done &= !(1u128 << u);
                    }
                    if e.state[u] == ON_VERTICES {
                        for i in idx {
                            vneed |= 1u128 << (i - fv);
                        }
                        continue;
                    }
                    tneed |= 1u128 << u;
                    n_used += e.grids[e.state[u] as usize].points() as u64;
                }
                if !still {
                    n_used += vneed.count_ones() as u64;
                    e.diced_for = Some((vp, rate, need_tris));
                    e.diced = (vneed, tneed, n_used);
                }
                used.fetch_add(n_used, Ordering::Relaxed);
                e.level = (0..e.state.len()).map(|u| e.tri_level(u)).filter(|l| *l != NOT_VISIBLE).max().unwrap_or(NOT_VISIBLE);
                let held;
                let (c, eye) = match carried {
                    Some((cc, eye)) => {
                        held = cc;
                        (&held, eye)
                    }
                    None => (c, eye),
                };
                let vtodo = vneed & !(e.vdone & e.vspec_done);
                let ttodo = tneed & !(e.done & e.spec_done);
                // Lights changed since the rest was kept: only their shares
                // are redone, on what is on screen; what is kept off screen
                // is shaded in full when it comes back.
                // Lights that reach this cluster.
                // Directional lights reach everything.
                let in_reach: Vec<usize> = (0..lights.len()).filter(|&i| lights[i].position.distance(c.centre) <= c.radius + reach[i]).chain((0..dir_lit.len()).filter(|&k| dir_lit[k].is_some()).map(|k| DIR_ID + k)).collect();
                let lchange = std::mem::take(&mut e.light_todo);
                let (mut vlight, mut tlight) = (0u128, 0u128);
                let (mut vtodo, mut ttodo) = (vtodo, ttodo);
                if lchange != 0 {
                    e.vdone &= vneed;
                    e.vspec_done &= vneed;
                    e.done &= tneed;
                    e.spec_done &= tneed;
                    // A changed light without a share of its own is in the
                    // sum: redo what is on screen in full.
                    let all_kept = (0..128).filter(|l| lchange & (1u128 << l) != 0).all(|l| e.slots.contains(&(l as u8)));
                    if all_kept {
                        vlight = vneed & !vtodo;
                        tlight = tneed & !ttodo;
                    } else {
                        vtodo = vneed;
                        ttodo = tneed;
                    }
                }
                // A light that has just become hot gets a share of its own:
                // what is kept has it in the sum, so what is on screen is
                // redone in full, the rest when it is next on screen.
                let new_slot = hot.iter().any(|&l| valid(l) && in_reach.contains(&l) && !e.slots.contains(&(l as u8)));
                let mut all_diffuse = lchange != 0;
                if new_slot && (e.done != 0 || e.vdone != 0) {
                    e.vdone &= vneed;
                    e.vspec_done &= vneed;
                    e.done &= tneed;
                    e.spec_done &= tneed;
                    vtodo = vneed;
                    ttodo = tneed;
                    vlight = 0;
                    tlight = 0;
                    all_diffuse = true;
                }
                if vtodo == 0 && ttodo == 0 && vlight == 0 && tlight == 0 {
                    return;
                }
                clusters_shaded.fetch_add(1, Ordering::Relaxed);
                e.stamp = frame_no;
                let (mut n_shaded, mut n_evals) = (0u64, 0u64);

                // Texture tiles at the finest level needed, read from the
                // file if not in the cache; grids finer than the finest tile
                // sample it.
                let finest = {
                    let mut l = 0;
                    let mut bits = ttodo | tlight;
                    while bits != 0 {
                        l = l.max(e.tri_level(bits.trailing_zeros() as usize));
                        bits &= bits - 1;
                    }
                    l
                };
                let tiles = if ttodo | tlight != 0 { scene.textures.block(ci, c.tri_count as usize, finest.min(crate::texture::MAX_LEVEL)) } else { None };
                // The points of the vertices `vbits` and grids `tbits` into
                // a batch, normals turned to the side each was last shaded
                // from; their places: vertices, then runs of grid points.
                // A vertex's material: that of the first triangle using it.
                let vmat: [u32; crate::scene::CLUSTER_VERTS] = {
                    let mut m = [u32::MAX; crate::scene::CLUSTER_VERTS];
                    for t in c.first_tri..c.first_tri + c.tri_count {
                        for l in scene.tris[t as usize] {
                            if m[l as usize] == u32::MAX {
                                m[l as usize] = t;
                            }
                        }
                    }
                    m
                };
                let gather = |e: &ClusterShade, vbits: u128, tbits: u128, batch: &mut Batch, vdest: &mut Vec<usize>, gdest: &mut Vec<(usize, usize)>, dims: &mut Vec<(usize, usize)>| {
                    let mut bits = vbits;
                    while bits != 0 {
                        let u = bits.trailing_zeros() as usize;
                        bits &= bits - 1;
                        let v = fv + u;
                        let side = if e.vback & (1u128 << u) != 0 { -1.0 } else { 1.0 };
                        let mat = scene.material(c, if vmat[u] == u32::MAX { c.first_tri } else { vmat[u] });
                        batch.push(scene.pos(c, v), scene.normal(c, v) * side, scene.color(v), mat);
                        vdest.push(u);
                    }
                    let mut bits = tbits;
                    while bits != 0 {
                        let u = bits.trailing_zeros() as usize;
                        bits &= bits - 1;
                        let d = e.grids[e.state[u] as usize];
                        let [i0, i1, i2] = scene.tri(c, c.first_tri + u as u32);
                        let (p0, p1, p2) = (scene.pos(c, i0), scene.pos(c, i1), scene.pos(c, i2));
                        let side = if e.back & (1u128 << u) != 0 { -1.0 } else { 1.0 };
                        let (nu, nv) = (1usize << d.lu, 1usize << d.lv);
                        gdest.push((d.offset as usize, (nu + 1) * (nv + 1)));
                        dims.push((nu, nv));
                        let (s0, su, sv) = d.screen_affine();
                        let (n0, n1, n2) = (scene.normal(c, i0), scene.normal(c, i1), scene.normal(c, i2));
                        let (c0, c1, c2) = (scene.color(i0), scene.color(i1), scene.color(i2));
                        let mat = scene.material(c, c.first_tri + u as u32);
                        // Eight points of a row at a time.
                        use crate::simd::F;
                        let steps = F::from([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
                        let inv_w = [1.0 / d.w[0], 1.0 / d.w[1], 1.0 / d.w[2]];
                        for a in 0..=nu {
                            let fu = a as f32 / nu as f32;
                            let mut b = 0;
                            while b <= nv {
                                let m = (nv + 1 - b).min(8);
                                let fv = (steps + b as f32) * (1.0 / nv as f32);
                                // Screen weights, then the triangle's own
                                // (perspective-correct), as `screen_to_tri`.
                                let q = [0, 1, 2].map(|k| (F::splat(s0[k] + su[k] * fu) + fv * sv[k]) * inv_w[k]);
                                let t = q[0] + q[1] + q[2];
                                let t = t.abs().simd_lt(F::splat(1e-30)).select(F::splat(1e-30), t);
                                let (b0, b1, b2) = (q[0] / t, q[1] / t, q[2] / t);
                                let lerp = |a: f32, bb: f32, c: f32| b0 * a + b1 * bb + b2 * c;
                                let (px, py, pz) = (lerp(p0.x, p1.x, p2.x), lerp(p0.y, p1.y, p2.y), lerp(p0.z, p1.z, p2.z));
                                let (nx, ny, nz) = (lerp(n0.x, n1.x, n2.x), lerp(n0.y, n1.y, n2.y), lerp(n0.z, n1.z, n2.z));
                                let l2 = nx * nx + ny * ny + nz * nz;
                                let ok = l2.simd_gt(F::splat(1e-20));
                                let k = F::splat(side) / l2.max(F::splat(1e-20)).sqrt();
                                let (nx, ny, nz) = (ok.select(nx * k, F::ZERO), ok.select(ny * k, F::splat(side)), ok.select(nz * k, F::ZERO));
                                let (bb1, bb2) = (b1.to_array(), b2.to_array());
                                let alb: [Vec3; 8] = match &tiles {
                                    Some(t) => std::array::from_fn(|j| if j < m { t.sample(u, bb1[j], bb2[j]) } else { Vec3::ZERO }),
                                    None => {
                                        let (r, g, bl) = (lerp(c0.x, c1.x, c2.x).max(F::ZERO).to_array(), lerp(c0.y, c1.y, c2.y).max(F::ZERO).to_array(), lerp(c0.z, c1.z, c2.z).max(F::ZERO).to_array());
                                        std::array::from_fn(|j| Vec3::new(r[j], g[j], bl[j]))
                                    }
                                };
                                batch.push8(m, [px, py, pz, nx, ny, nz], &alb, mat);
                                b += 8;
                            }
                        }
                    }
                };
                // One light's share for every point of a batch.
                let dark = Sky { on: false, ..sky };
                let share_of = |batch: &Batch, l: usize, d: &mut [Vec3], s: &mut [Vec3]| -> u64 {
                    if l >= DIR_ID {
                        // A directional light alone.
                        match dir_lit.get(l - DIR_ID).copied().flatten() {
                            Some(dl) => shade_batch(batch, eye, dark, false, lights, &[], &reach, &shadows, &[dl], d, s, true),
                            None => 0,
                        }
                    } else {
                        shade_batch(batch, eye, dark, false, lights, &[l], &reach, &shadows, &[], d, s, true)
                    }
                };

                // In full: the sky and the sun, then each light's share.
                if vtodo != 0 || ttodo != 0 {
                    // Which side each is seen from now; a flip redoes the
                    // diffuse light too.
                    let mut vdiff = 0u128;
                    let mut bits = vtodo;
                    while bits != 0 {
                        let u = bits.trailing_zeros() as usize;
                        bits &= bits - 1;
                        let v = fv + u;
                        let back = scene.normal(c, v).dot(eye - scene.pos(c, v)) < 0.0;
                        let bit = 1u128 << u;
                        if e.vdone & bit == 0 || ((e.vback & bit != 0) != back) {
                            vdiff |= bit;
                        }
                        if back { e.vback |= bit } else { e.vback &= !bit }
                    }
                    let mut tdiff = 0u128;
                    let mut bits = ttodo;
                    while bits != 0 {
                        let u = bits.trailing_zeros() as usize;
                        bits &= bits - 1;
                        let [i0, i1, i2] = scene.tri(c, c.first_tri + u as u32);
                        let (p0, p1, p2) = (scene.pos(c, i0), scene.pos(c, i1), scene.pos(c, i2));
                        let fnrm = scene.normal(c, i0) + scene.normal(c, i1) + scene.normal(c, i2);
                        let back = fnrm.dot(eye - (p0 + p1 + p2) / 3.0) < 0.0;
                        let bit = 1u128 << u;
                        if e.done & bit == 0 || ((e.back & bit != 0) != back) {
                            tdiff |= bit;
                        }
                        if back { e.back |= bit } else { e.back &= !bit }
                    }
                    // A changed light makes kept diffuse light stale too.
                    if all_diffuse {
                        vdiff = vtodo;
                        tdiff = ttodo;
                    }
                    // Shares of their own for the hot lights; the others are
                    // in the sum, and a share that is no longer hot goes.
                    let mut k = 0;
                    while k < e.slots.len() {
                        if hot.contains(&(e.slots[k] as usize)) {
                            k += 1;
                        } else {
                            e.slots.remove(k);
                            e.gshare.remove(k);
                            e.vshare.remove(k);
                        }
                    }
                    let mut lights_now: Vec<usize> = hot.iter().copied().filter(|&l| in_reach.contains(&l) || e.slots.contains(&(l as u8))).collect();
                    lights_now.retain(|&l| valid(l));
                    // In the sum: point lights in reach and the sun, with the
                    // directional lights that have no share of their own.
                    let in_sum: Vec<usize> = in_reach.iter().copied().filter(|l| *l < DIR_ID && !lights_now.contains(l)).collect();
                    let suns_sum: Vec<SunLit> = sun_main.iter().copied().chain(in_reach.iter().filter(|l| **l >= DIR_ID && !lights_now.contains(l)).filter_map(|l| dir_lit[*l - DIR_ID])).collect();
                    let mut batch = Batch::default();
                    let (mut vdest, mut gdest, mut dims) = (Vec::new(), Vec::new(), Vec::new());
                    gather(e, vtodo, ttodo, &mut batch, &mut vdest, &mut gdest, &mut dims);

                    // The sky, the sun and the lights in the sum, then each
                    // hot light's share (one that no longer reaches gives
                    // nothing): shaded where the picture needs it, the rest
                    // interpolated (`shade_adaptive`).
                    let shares_in: Vec<bool> = lights_now.iter().map(|l| in_reach.contains(l)).collect();
                    let share_here = sky_share.filter(|s| s.covers(ci));
                    let mut eval = |b: &Batch, d: &mut [Vec3], s: &mut [Vec3], sd: &mut [Vec<Vec3>], ss: &mut [Vec<Vec3>]| -> u64 {
                        // Each point shaded takes its share of the sky from
                        // the samples around it (those interpolated need
                        // none).
                        let with_share;
                        let b = match share_here {
                            Some(sh) => {
                                let mut c = b.clone();
                                for k in 0..c.len() {
                                    let v = sh.at(ci, scene, Vec3::new(c.px[k], c.py[k], c.pz[k]), Vec3::new(c.nx[k], c.ny[k], c.nz[k]));
                                    c.sv.push(v[0]);
                                    c.gv.push(v[1]);
                                }
                                with_share = c;
                                &with_share
                            }
                            None => b,
                        };
                        let mut ev = shade_batch(b, eye, sky, ground, lights, &in_sum, &reach, &shadows, &suns_sum, d, s, true);
                        for (k, &l) in lights_now.iter().enumerate() {
                            if shares_in[k] {
                                ev += share_of(b, l, &mut sd[k], &mut ss[k]);
                            }
                        }
                        ev
                    };
                    let a = shade_adaptive(&batch, vdest.len(), &dims, lights_now.len(), exposure, adaptive, &mut eval);
                    n_evals += a.evals;
                    n_shaded += a.shaded as u64;
                    interpolated.fetch_add((batch.len() - a.shaded) as u64, Ordering::Relaxed);
                    let (mut dd, mut ss) = (a.d, a.s);
                    for (li, &l) in lights_now.iter().enumerate() {
                        let k = e.slot_of(l);
                        let (ld, ls) = (&a.sd[li], &a.ss[li]);
                        let mut j = 0;
                        for &u in &vdest {
                            let diff = vdiff & (1u128 << u) != 0;
                            let d = if diff { ld[j] } else { e.vshare[k][u][0] };
                            dd[j] += d;
                            ss[j] += ls[j];
                            e.vshare[k][u] = [d, ls[j]];
                            j += 1;
                        }
                        let mut bits = ttodo;
                        for &(g, count) in &gdest {
                            let u = bits.trailing_zeros() as usize;
                            bits &= bits - 1;
                            let diff = tdiff & (1u128 << u) != 0;
                            for q in g..g + count {
                                let d = if diff { ld[j] } else { e.gshare[k][q][0] };
                                dd[j] += d;
                                ss[j] += ls[j];
                                e.gshare[k][q] = [d, ls[j]];
                                j += 1;
                            }
                        }
                    }
                    for (j, &u) in vdest.iter().enumerate() {
                        if vdiff & (1u128 << u) != 0 {
                            e.vdiffuse[u] = dd[j];
                        }
                        e.vspec[u] = ss[j];
                    }
                    let mut j = vdest.len();
                    let mut bits = ttodo;
                    for &(g, count) in &gdest {
                        let u = bits.trailing_zeros() as usize;
                        bits &= bits - 1;
                        if tdiff & (1u128 << u) != 0 {
                            e.diffuse[g..g + count].copy_from_slice(&dd[j..j + count]);
                        }
                        e.spec[g..g + count].copy_from_slice(&ss[j..j + count]);
                        j += count;
                    }
                    e.vdone |= vtodo;
                    e.vspec_done |= vtodo;
                    e.done |= ttodo;
                    e.spec_done |= ttodo;
                }

                // Changed lights only: the old share out, the new one in.
                if vlight != 0 || tlight != 0 {
                    let mut batch = Batch::default();
                    let (mut vdest, mut gdest) = (Vec::new(), Vec::new());
                    let mut dims = Vec::new();
                    gather(e, vlight, tlight, &mut batch, &mut vdest, &mut gdest, &mut dims);
                    let n = batch.len();
                    let mut ld = vec![Vec3::ZERO; n];
                    let mut ls = vec![Vec3::ZERO; n];
                    let mut bits = lchange;
                    while bits != 0 {
                        let l = bits.trailing_zeros() as usize;
                        bits &= bits - 1;
                        if l < DIR_ID && l >= lights.len() {
                            continue;
                        }
                        let reaches = in_reach.contains(&l);
                        if !reaches && !e.slots.contains(&(l as u8)) {
                            continue;
                        }
                        let k = e.slot_of(l);
                        if reaches {
                            // Every point: one light's lookup costs less than
                            // choosing which points to skip.
                            let a = shade_adaptive(&batch, vdest.len(), &dims, 0, exposure, false, &mut |b: &Batch, d: &mut [Vec3], s: &mut [Vec3], _: &mut [Vec<Vec3>], _: &mut [Vec<Vec3>]| share_of(b, l, d, s));
                            n_evals += a.evals;
                            n_shaded += a.shaded as u64;
                            interpolated.fetch_add((n - a.shaded) as u64, Ordering::Relaxed);
                            ld = a.d;
                            ls = a.s;
                        } else {
                            ld.iter_mut().for_each(|v| *v = Vec3::ZERO);
                            ls.iter_mut().for_each(|v| *v = Vec3::ZERO);
                        }
                        let mut j = 0;
                        for &u in &vdest {
                            let [od, os] = e.vshare[k][u];
                            e.vdiffuse[u] += ld[j] - od;
                            e.vspec[u] += ls[j] - os;
                            e.vshare[k][u] = [ld[j], ls[j]];
                            j += 1;
                        }
                        for &(g, count) in &gdest {
                            for q in g..g + count {
                                let [od, os] = e.gshare[k][q];
                                e.diffuse[q] += ld[j] - od;
                                e.spec[q] += ls[j] - os;
                                e.gshare[k][q] = [ld[j], ls[j]];
                                j += 1;
                            }
                        }
                    }
                }
                evals.fetch_add(n_evals, Ordering::Relaxed);
                shaded.fetch_add(n_shaded, Ordering::Relaxed);
            });
        }
        frame.stats.shade_ms = t_shade.elapsed().as_secs_f32() * 1000.0;
        frame.stats.points_shaded = shaded.into_inner();
        frame.stats.points_interpolated = interpolated.into_inner();
        frame.stats.points_used = used.into_inner();
        frame.stats.clusters_shaded = clusters_shaded.into_inner() as usize;
        frame.stats.light_evals = evals.into_inner();
        frame.stats.light_evals_all = frame.stats.points_shaded * lights.len() as u64;
        if let Stop::Shade(_) = stop {
            return !unfinished.into_inner();
        }

        // Gather bounce light at visible receivers that need it, nearest the
        // camera first, within the time budget.
        let tg = std::time::Instant::now();
        let rows = match stop {
            Stop::Rows(a, b) => Some(a..b),
            _ => None,
        };
        let gather_budget = match stop {
            Stop::Gather(ms) => ms,
            _ => settings.bounce_budget_ms,
        };
        if settings.bounce && settings.view == View::Shaded && rows.is_none() {
            let b = self.bounce.as_mut().expect("made above");
            let bounce_reach = b.reach(cut);
            // The receivers around what is visible that are not current.
            b.prepare(scene, &self.visible);
            let mut todo: Vec<u32> = self.visible.iter().flat_map(|&c| b.near(c as usize).iter().copied()).filter(|&r| !b.valid[r as usize]).collect();
            todo.sort_unstable();
            todo.dedup();
            todo.sort_by(|a, c| b.receivers[*a as usize].0.distance_squared(eye).total_cmp(&b.receivers[*c as usize].0.distance_squared(eye)));
            let mut done = 0;
            for batch in todo.chunks(64) {
                if let Some(ms) = gather_budget {
                    if tg.elapsed().as_secs_f32() * 1000.0 > ms {
                        break;
                    }
                }
                let bref = &*b;
                let results: Vec<(u32, bounce::Sh)> = batch.par_iter().map(|&r| (r, bref.gather(r as usize, bounce_reach))).collect();
                for (r, sh) in results {
                    b.sh[r as usize] = sh;
                    b.valid[r as usize] = true;
                    b.stamp[r as usize] = frame_no;
                }
                done += batch.len();
            }
            // Clusters whose bounce light changed are resolved again.
            if done > 0 {
                for &c in &self.visible {
                    if b.near(c as usize).iter().any(|&r| b.stamp[r as usize] == frame_no) {
                        b.cluster_stamp[c as usize] = frame_no;
                    }
                }
            }
            frame.stats.receivers_updated = done;
            frame.stats.receivers_pending = todo.len() - done;
        }
        frame.stats.gather_ms = tg.elapsed().as_secs_f32() * 1000.0;
        if let Stop::Gather(_) = stop {
            return frame.stats.receivers_pending == 0;
        }

        // Resolve: each pixel interpolates the shaded points under it.
        let t_res = std::time::Instant::now();
        let (width, height) = (frame.width, frame.height);
        let inv_view = cam.view.inverse();
        let exposure = settings.exposure;
        let view = settings.view;
        if self.prev_backdrop != Some(settings.backdrop) {
            full_resolve = true;
            self.prev_backdrop = Some(settings.backdrop);
        }
        if self.base.len() != width * height || self.base_view != Some(view) {
            self.base = vec![0; width * height];
            full_resolve = true;
        }
        self.base_view = Some(view);
        frame.stats.full_resolve = full_resolve;
        let cache = &self.cache;
        let tri_id = &frame.tri_id;
        let setups = &frame.setups;
        let bounce = if settings.bounce { self.bounce.as_ref() } else { None };
        self.base.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
            // A bake resolves its rows in full and leaves the rest.
            let full_resolve = match &rows {
                Some(r) if !r.contains(&y) => return,
                Some(_) => true,
                None => full_resolve,
            };
            // The bounce blend of the pixel to the left, used again by
            // every second pixel of the same cluster: bounce light changes
            // slowly, and blending is the dearest part of a pixel.
            let mut last: Option<(usize, u32, bounce::Sh)> = None;
            for x in 0..width {
                let id = tri_id[y * width + x];
                let changed = |c: usize| cache[c].stamp == frame_no || bounce.is_some_and(|b| b.cluster_stamp[c] == frame_no);
                if !full_resolve && (id == NO_TRI || !changed(setups[id as usize].cluster as usize)) {
                    // Nothing under this pixel changed: keep it.
                    continue;
                }
                if id == NO_TRI {
                    row[x] = match view {
                        View::Shaded => raster::pack(raster::tonemap(raster::backdrop_radiance(&sky, &settings.backdrop, &settings.sun, exposure, raster::sky_dir(cam, width, height, x, y, &inv_view)) * exposure)),
                        View::LightCount => raster::pack(Vec3::splat(0.02)),
                    };
                    continue;
                }
                let s = &setups[id as usize];
                let c = &scene.clusters[s.cluster as usize];
                if view == View::LightCount {
                    let k = (0..lights.len()).filter(|&i| lights[i].position.distance(c.centre) <= c.radius + reach[i]).count();
                    row[x] = raster::pack(raster::heat(k as u32));
                    continue;
                }
                let b = raster::barycentrics(s, x as f32 + 0.5, y as f32 + 0.5);
                let w = [
                    b[0] * s.bary[0][0] + b[1] * s.bary[1][0] + b[2] * s.bary[2][0],
                    b[0] * s.bary[0][1] + b[1] * s.bary[1][1] + b[2] * s.bary[2][1],
                    b[0] * s.bary[0][2] + b[1] * s.bary[1][2] + b[2] * s.bary[2][2],
                ];
                let e = &cache[s.cluster as usize];
                let (diffuse, highlight) = lookup_split(scene, c, e, s.tri, w);
                // A dragged cluster where the drag began, as shaded.
                let ci = s.cluster as usize;
                let held = drag.filter(|(r, _, _)| r.iter().any(|r| r.contains(&ci))).map(|(_, back, turn)| (carry(c, back, turn), back.transform_point3(eye)));
                let (c, eye) = match &held {
                    Some((cc, e)) => (cc, *e),
                    None => (c, eye),
                };
                let [i0, i1, i2] = scene.tri(c, s.tri);
                let p = scene.pos(c, i0) * w[0] + scene.pos(c, i1) * w[1] + scene.pos(c, i2) * w[2];
                // The normal turned to the eye.
                let mut n = (scene.normal(c, i0) * w[0] + scene.normal(c, i1) * w[1] + scene.normal(c, i2) * w[2]).normalize_or(Vec3::Y);
                let to_eye = (eye - p).normalize_or(n);
                if n.dot(to_eye) < 0.0 {
                    n = -n;
                }
                let mat = scene.material(c, s.tri);
                // What the highlight takes from diffuse, seen from here.
                let dw = mat.diffuse_view_weight(n.dot(to_eye));
                let mut radiance = diffuse * dw + highlight;
                if let Some(b) = bounce {
                    // Bounce light, from the cluster's spherical harmonics.
                    let albedo = mat.diffuse(scene.color(i0) * w[0] + scene.color(i1) * w[1] + scene.color(i2) * w[2]) * dw;
                    let sh = match last {
                        Some((lx, lc, sh)) if x % 2 == 1 && lx + 1 == x && lc == s.cluster => sh,
                        _ => {
                            let sh = b.blend(s.cluster as usize, p, n);
                            last = Some((x, s.cluster, sh));
                            sh
                        }
                    };
                    radiance += albedo * bounce::sh_irradiance(&sh, n) * std::f32::consts::FRAC_1_PI;
                }
                row[x] = raster::pack(raster::tonemap(radiance * exposure));
            }
        });
        frame.color.copy_from_slice(&self.base);
        frame.stats.resolve_ms = t_res.elapsed().as_secs_f32() * 1000.0;
        self.fit_budget(settings.cache_budget, frame);
        true
    }

    /// Keeps the shading cache within `budget` bytes. Clusters off screen
    /// go first, least recently seen first, down to three quarters of the
    /// budget. When what is on screen alone does not fit, the shading rate
    /// doubles (up to 16 times) and the grids on screen are dropped, to be
    /// diced coarser next frame; it halves again once the cache is under an
    /// eighth of the budget. Slower or coarser, never out of memory.
    fn fit_budget(&mut self, budget: usize, frame: &mut Frame) {
        let frame_no = self.frame_no;
        let mut total: usize = self.cache.par_iter().map(|e| e.bytes()).sum();
        if total > budget {
            let mut old: Vec<(u64, u32)> = self.cache.iter().enumerate().filter(|(_, e)| e.seen != frame_no && e.bytes() > 0).map(|(i, e)| (e.seen, i as u32)).collect();
            old.sort_unstable();
            for (_, i) in old {
                if total <= budget / 4 * 3 {
                    break;
                }
                let e = &mut self.cache[i as usize];
                total -= e.bytes();
                *e = ClusterShade::empty();
                frame.stats.clusters_evicted += 1;
            }
            if total > budget && self.rate_scale < 16.0 {
                self.rate_scale *= 2.0;
                for e in &mut self.cache {
                    if e.grids.capacity() > 0 {
                        total -= e.bytes();
                        e.forget_grids();
                        total += e.bytes();
                    }
                }
            }
        } else if total < budget / 8 && self.rate_scale > 1.0 {
            self.rate_scale /= 2.0;
        }
        frame.stats.cache_bytes = total;
        frame.stats.rate_scale = self.rate_scale;
    }
}

#[cfg_attr(not(target_pointer_width = "64"), allow(dead_code))]
/// Which part of the kept shading to read.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Part {
    All,
    Highlight,
}

/// Diffuse and highlight radiance at a point inside triangle `tri` with
/// weights `w`, from the kept shading.
fn lookup_split(scene: &Scene, c: &crate::scene::Cluster, e: &ClusterShade, tri: u32, w: [f32; 3]) -> (Vec3, Vec3) {
    let u = (tri - c.first_tri) as usize;
    let state = e.state.get(u).copied().unwrap_or(UNDICED);
    if state == UNDICED {
        return (Vec3::ZERO, Vec3::ZERO);
    }
    if state == ON_VERTICES {
        let l = scene.tris[tri as usize];
        let (mut d, mut s) = (Vec3::ZERO, Vec3::ZERO);
        for k in 0..3 {
            let v = l[k] as usize;
            d += e.vdiffuse[v] * w[k];
            s += e.vspec[v] * w[k];
        }
        return (d, s);
    }
    let g = &e.grids[state as usize];
    let (u, v) = g.screen_to_grid(tri_to_screen(w, g.w));
    let (nu, nv) = (1usize << g.lu, 1usize << g.lv);
    let (fu, fv) = (u * nu as f32, v * nv as f32);
    let (a, b) = ((fu as usize).min(nu - 1), (fv as usize).min(nv - 1));
    let (fu, fv) = (fu - a as f32, fv - b as f32);
    let k = |a: usize, b: usize| g.offset as usize + a * (nv + 1) + b;
    let mix = |arr: &[Vec3]| (arr[k(a, b)] * (1.0 - fu) + arr[k(a + 1, b)] * fu) * (1.0 - fv) + (arr[k(a, b + 1)] * (1.0 - fu) + arr[k(a + 1, b + 1)] * fu) * fv;
    (mix(&e.diffuse), mix(&e.spec))
}

#[cfg_attr(not(target_pointer_width = "64"), allow(dead_code))]
fn lookup_part(scene: &Scene, c: &crate::scene::Cluster, e: &ClusterShade, tri: u32, w: [f32; 3], part: Part) -> Vec3 {
    let u = (tri - c.first_tri) as usize;
    let state = e.state.get(u).copied().unwrap_or(UNDICED);
    if state == UNDICED {
        return Vec3::ZERO;
    }
    if state == ON_VERTICES {
        let l = scene.tris[tri as usize];
        let mut r = Vec3::ZERO;
        for k in 0..3 {
            let v = l[k] as usize;
            r += match part {
                Part::All => e.vdiffuse[v] + e.vspec[v],
                Part::Highlight => e.vspec[v],
            } * w[k];
        }
        return r;
    }
    let d = &e.grids[state as usize];
    let at = |i: usize| match part {
        Part::All => e.diffuse[i] + e.spec[i],
        Part::Highlight => e.spec[i],
    };
    let (u, v) = d.screen_to_grid(tri_to_screen(w, d.w));
    let (nu, nv) = (1usize << d.lu, 1usize << d.lv);
    let (fu, fv) = (u * nu as f32, v * nv as f32);
    let (a, b) = ((fu as usize).min(nu - 1), (fv as usize).min(nv - 1));
    let (fu, fv) = (fu - a as f32, fv - b as f32);
    let p = |a: usize, b: usize| at(d.offset as usize + a * (nv + 1) + b);
    (p(a, b) * (1.0 - fu) + p(a + 1, b) * fu) * (1.0 - fv) + (p(a, b + 1) * (1.0 - fu) + p(a + 1, b + 1) * fu) * fv
}

/// The sun as `shade_point` needs it: illuminance facing it, direction,
/// shadow map.
#[derive(Clone, Copy)]
pub struct SunLit<'a> {
    pub lux: Vec3,
    pub dir: Vec3,
    pub shadow: Option<&'a SunShadow>,
}

/// Diffuse radiance (cd/m²) leaving a point (a surfel: it has no
/// highlight), and how many lights were evaluated. `n` is already turned
/// to the eye's side.
#[allow(clippy::too_many_arguments)]
fn shade_point(
    p: Vec3,
    n: Vec3,
    albedo: Vec3,
    eye: Vec3,
    sky: Sky,
    ground: bool,
    lights: &[PointLight],
    in_reach: &[usize],
    reach: &[f32],
    shadows: &[Option<&CubeShadow>],
    suns: &[SunLit],
) -> (Vec3, u64) {
    let inv_pi = std::f32::consts::FRAC_1_PI;
    let _ = eye;
    let mut diffuse = albedo * sky.irradiance_from(n, ground) * inv_pi;
    let mut evals = 0;
    // The sun and directional lights: everywhere, parallel, each through
    // its own map.
    for sun in suns {
        if sun.lux == Vec3::ZERO {
            continue;
        }
        let ndl = n.dot(sun.dir);
        if ndl > 0.0 {
            let vis = sun.shadow.map(|m| m.lit(p, n)).unwrap_or(1.0);
            if vis > 0.0 {
                let e = sun.lux * (ndl * vis);
                diffuse += e * albedo * inv_pi;
            }
        }
    }
    for &i in in_reach {
        let l = &lights[i];
        let d = l.position - p;
        let d2 = d.length_squared().max(1e-12);
        if d2 > reach[i] * reach[i] {
            continue;
        }
        evals += 1;
        let ldir = d / d2.sqrt();
        let ndl = n.dot(ldir);
        if ndl <= 0.0 {
            continue;
        }
        let vis = match shadows[i] {
            Some(m) => m.lit(p, n),
            None => 1.0,
        };
        if vis <= 0.0 {
            continue;
        }
        let e = l.color * (l.candela() / d2 * ndl * vis);
        diffuse += e * albedo * inv_pi;
    }
    (diffuse, evals)
}

/// Lights that keep a share of their own in the shading cache.
const HOT_LIGHTS: usize = 1;


/// Points to shade, a component per array (structure of arrays), so eight
/// are shaded at once.
#[derive(Default, Clone)]
pub struct Batch {
    pub px: Vec<f32>,
    pub py: Vec<f32>,
    pub pz: Vec<f32>,
    pub nx: Vec<f32>,
    pub ny: Vec<f32>,
    pub nz: Vec<f32>,
    /// What multiplies diffuse light (`Material::diffuse`).
    pub ar: Vec<f32>,
    pub ag: Vec<f32>,
    pub ab: Vec<f32>,
    /// Reflectance facing the eye (`Material::f0`), and roughness.
    pub fr: Vec<f32>,
    pub fg: Vec<f32>,
    pub fb: Vec<f32>,
    pub rough: Vec<f32>,
    /// Each point's share of the sky and of the ground's light
    /// (`transfer.rs`); empty: all of both.
    pub sv: Vec<f32>,
    pub gv: Vec<f32>,
}

impl Batch {
    pub fn len(&self) -> usize {
        self.px.len()
    }

    /// The first `m` lanes of eight points (position and normal, a
    /// component a lane group) with their base colours, of material `mat`.
    #[inline]
    pub fn push8(&mut self, m: usize, pn: [crate::simd::F; 6], albedo: &[Vec3; 8], mat: &crate::scene::Material) {
        let [px, py, pz, nx, ny, nz] = pn.map(|v| v.to_array());
        self.px.extend_from_slice(&px[..m]);
        self.py.extend_from_slice(&py[..m]);
        self.pz.extend_from_slice(&pz[..m]);
        self.nx.extend_from_slice(&nx[..m]);
        self.ny.extend_from_slice(&ny[..m]);
        self.nz.extend_from_slice(&nz[..m]);
        for a in &albedo[..m] {
            self.surface(*a, mat);
        }
    }

    /// A point of base colour `albedo` and material `mat`.
    #[inline]
    pub fn push(&mut self, p: Vec3, n: Vec3, albedo: Vec3, mat: &crate::scene::Material) {
        self.px.push(p.x);
        self.py.push(p.y);
        self.pz.push(p.z);
        self.nx.push(n.x);
        self.ny.push(n.y);
        self.nz.push(n.z);
        self.surface(albedo, mat);
    }

    #[inline]
    fn surface(&mut self, albedo: Vec3, mat: &crate::scene::Material) {
        let d = mat.diffuse(albedo);
        let f = mat.f0(albedo);
        self.ar.push(d.x);
        self.ag.push(d.y);
        self.ab.push(d.z);
        self.fr.push(f.x);
        self.fg.push(f.y);
        self.fb.push(f.z);
        self.rough.push(mat.roughness.clamp(0.02, 1.0));
    }
}

impl Batch {
    /// The points `idx` of this batch, in that order.
    fn select(&self, idx: &[u32]) -> Batch {
        let pick = |v: &Vec<f32>| if v.is_empty() { Vec::new() } else { idx.iter().map(|&i| v[i as usize]).collect() };
        Batch { px: pick(&self.px), py: pick(&self.py), pz: pick(&self.pz), nx: pick(&self.nx), ny: pick(&self.ny), nz: pick(&self.nz), ar: pick(&self.ar), ag: pick(&self.ag), ab: pick(&self.ab), fr: pick(&self.fr), fg: pick(&self.fg), fb: pick(&self.fb), rough: pick(&self.rough), sv: pick(&self.sv), gv: pick(&self.gv) }
    }

    fn albedo(&self, i: usize) -> Vec3 {
        Vec3::new(self.ar[i], self.ag[i], self.ab[i])
    }
}

/// How far apart, in display units before the tone curve, the corners of
/// a cell of four grid points may be for the points between them to be
/// interpolated rather than shaded: about 8 of 255 where it matters most.
pub const ADAPTIVE_TOLERANCE: f32 = 0.03;

/// What `shade_adaptive` made: for every point of the batch the sum
/// (diffuse, highlight), and each share; how many points were shaded and
/// the lights evaluated.
struct Adaptive {
    d: Vec<Vec3>,
    s: Vec<Vec3>,
    sd: Vec<Vec<Vec3>>,
    ss: Vec<Vec<Vec3>>,
    shaded: usize,
    evals: u64,
}

/// Shade a cluster's points where the picture needs them. The first
/// `n_vert` points (vertices) are all shaded; the rest are grids, row by
/// row, of `dims` (nu, nv) parts. Every other point of each grid (each
/// way) is shaded first; a cell of four whose light differs by more than
/// `ADAPTIVE_TOLERANCE` (after `exposure`) has its points between shaded
/// too, and the rest are interpolated from the cell's corners: the light
/// on them (diffuse over albedo, so texture tiles keep their detail) and
/// the highlight. `eval` shades a batch into the sum and each of
/// `shares` shares. Without `adaptive`, every point is shaded.
fn shade_adaptive(
    b: &Batch,
    n_vert: usize,
    dims: &[(usize, usize)],
    shares: usize,
    exposure: f32,
    adaptive: bool,
    eval: &mut dyn FnMut(&Batch, &mut [Vec3], &mut [Vec3], &mut [Vec<Vec3>], &mut [Vec<Vec3>]) -> u64,
) -> Adaptive {
    let n = b.len();
    let mut out = Adaptive { d: vec![Vec3::ZERO; n], s: vec![Vec3::ZERO; n], sd: vec![vec![Vec3::ZERO; n]; shares], ss: vec![vec![Vec3::ZERO; n]; shares], shaded: 0, evals: 0 };
    let mut run = |idx: &[u32], out: &mut Adaptive| {
        if idx.is_empty() {
            return;
        }
        let sub = if idx.len() == n { None } else { Some(b.select(idx)) };
        let sb = sub.as_ref().unwrap_or(b);
        let m = idx.len();
        let (mut d, mut s) = (vec![Vec3::ZERO; m], vec![Vec3::ZERO; m]);
        let (mut sd, mut ss) = (vec![vec![Vec3::ZERO; m]; shares], vec![vec![Vec3::ZERO; m]; shares]);
        out.evals += eval(sb, &mut d, &mut s, &mut sd, &mut ss);
        out.shaded += m;
        for (j, &i) in idx.iter().enumerate() {
            let i = i as usize;
            out.d[i] = d[j];
            out.s[i] = s[j];
            for k in 0..shares {
                out.sd[k][i] = sd[k][j];
                out.ss[k][i] = ss[k][j];
            }
        }
    };
    if !adaptive {
        let all: Vec<u32> = (0..n as u32).collect();
        run(&all, &mut out);
        return out;
    }
    // Pass 1: vertices, and the even points of every grid.
    let coarse_of = |nu: usize, nv: usize, a: usize, bb: usize| (nu < 2 || a % 2 == 0) && (nv < 2 || bb % 2 == 0);
    let mut first: Vec<u32> = (0..n_vert as u32).collect();
    let mut base = n_vert;
    for &(nu, nv) in dims {
        for a in 0..=nu {
            for bb in 0..=nv {
                if coarse_of(nu, nv, a, bb) {
                    first.push((base + a * (nv + 1) + bb) as u32);
                }
            }
        }
        base += (nu + 1) * (nv + 1);
    }
    run(&first, &mut out);
    if first.len() == n {
        return out;
    }
    // The light at each shaded point, for comparing corners: diffuse over
    // albedo (half the light a mid-grey surface shows) and the highlight.
    let lum = Vec3::new(0.2126, 0.7152, 0.0722);
    let mut level = vec![0f32; n];
    for &i in &first {
        let i = i as usize;
        let (mut d, mut s) = (out.d[i], out.s[i]);
        for k in 0..shares {
            d += out.sd[k][i];
            s += out.ss[k][i];
        }
        level[i] = ((d / b.albedo(i).max(Vec3::splat(0.02))).dot(lum) * 0.5 + s.dot(lum)) * exposure;
    }
    // Pass 2: the points between corners that differ; the others are
    // interpolated once those are shaded.
    // 0: shaded in pass 1; 1: to shade; 2: to interpolate from `cell`.
    let mut what = vec![0u8; n];
    let mut cell = vec![0u32; n];
    let mut second: Vec<u32> = Vec::new();
    let mut cells: Vec<([u32; 4], usize, usize, usize, usize, usize)> = Vec::new();
    let mut base = n_vert;
    for &(nu, nv) in dims {
        let at = |a: usize, bb: usize| base + a * (nv + 1) + bb;
        let (su, sv) = (if nu < 2 { 1 } else { 2 }, if nv < 2 { 1 } else { 2 });
        let mut a0 = 0;
        while a0 < nu {
            let a1 = (a0 + su).min(nu);
            let mut b0 = 0;
            while b0 < nv {
                let b1 = (b0 + sv).min(nv);
                let corners = [at(a0, b0), at(a1, b0), at(a0, b1), at(a1, b1)];
                let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
                for &c in &corners {
                    lo = lo.min(level[c]);
                    hi = hi.max(level[c]);
                }
                let smooth = hi - lo <= ADAPTIVE_TOLERANCE;
                let ci = cells.len() as u32;
                cells.push((corners.map(|c| c as u32), a0, a1, b0, b1, nv));
                for a in a0..=a1 {
                    for bb in b0..=b1 {
                        // Corners, and points on an edge the cell before
                        // took.
                        if coarse_of(nu, nv, a, bb) || (a == a0 && a0 > 0) || (bb == b0 && b0 > 0) {
                            continue;
                        }
                        let i = at(a, bb);
                        if smooth {
                            what[i] = 2;
                            cell[i] = ci;
                        } else {
                            what[i] = 1;
                            second.push(i as u32);
                        }
                    }
                }
                b0 = b1;
            }
            a0 = a1;
        }
        base += (nu + 1) * (nv + 1);
    }
    run(&second, &mut out);
    // The corners' light: diffuse over albedo, once a corner.
    let mut light: Vec<Vec<Vec3>> = vec![vec![Vec3::ZERO; n]; 1 + shares];
    for &i in &first {
        let i = i as usize;
        let inv = Vec3::ONE / b.albedo(i).max(Vec3::splat(1e-4));
        light[0][i] = out.d[i] * inv;
        for k in 0..shares {
            light[1 + k][i] = out.sd[k][i] * inv;
        }
    }
    for i in 0..n {
        if what[i] != 2 {
            continue;
        }
        let (cs, a0, a1, b0, b1, nv) = cells[cell[i] as usize];
        let local = i - (cs[0] as usize - (a0 * (nv + 1) + b0));
        let (a, bb) = (local / (nv + 1), local % (nv + 1));
        let fu = (a - a0) as f32 / (a1 - a0).max(1) as f32;
        let fv = (bb - b0) as f32 / (b1 - b0).max(1) as f32;
        let w = [(1.0 - fu) * (1.0 - fv), fu * (1.0 - fv), (1.0 - fu) * fv, fu * fv];
        let cs = cs.map(|c| c as usize);
        let lerp = |v: &[Vec3]| v[cs[0]] * w[0] + v[cs[1]] * w[1] + v[cs[2]] * w[2] + v[cs[3]] * w[3];
        // Diffuse as the corners' light times the point's own albedo.
        let alb = b.albedo(i);
        out.d[i] = lerp(&light[0]) * alb;
        out.s[i] = lerp(&out.s);
        for k in 0..shares {
            out.sd[k][i] = lerp(&light[1 + k]) * alb;
            out.ss[k][i] = lerp(&out.ss[k]);
        }
    }
    out
}

/// `shade_point` for every point of `b`, eight at a time (AVX on the
/// desktop, two halves of four elsewhere, plain arrays on the Pi 1): the
/// diffuse and highlight radiance of point k go to `diffuse[k]` (when
/// `with_diffuse`) and `spec[k]`. Shadow map lookups stay one point at a
/// time, for the points a light faces. Returns the lights evaluated.
#[allow(clippy::too_many_arguments)]
pub fn shade_batch(
    b: &Batch,
    eye: Vec3,
    sky: Sky,
    ground: bool,
    lights: &[PointLight],
    in_reach: &[usize],
    reach: &[f32],
    shadows: &[Option<&CubeShadow>],
    suns: &[SunLit],
    diffuse: &mut [Vec3],
    spec: &mut [Vec3],
    with_diffuse: bool,
) -> u64 {
    use crate::simd::V3;
    use wide::f32x8;
    let n = b.len();
    let zero = f32x8::ZERO;
    let one = f32x8::ONE;
    let inv_pi = std::f32::consts::FRAC_1_PI;
    // The sky's terms that do not depend on the point.
    let sky_c = sky.contrast.clamp(0.0, 1.0);
    let sky_fill = sky.lux * (1.0 - 2.0 * sky_c).exp2();
    let ground_k = if ground { crate::raster::GROUND_TINT * crate::raster::GROUND_BOUNCE } else { Vec3::ZERO };
    let sky_exp = 1.0 + 6.0 * (sky_c - 0.5);
    // The sky's light from direction with up component `y` (0 down, 1
    // up), as its diffuse term shapes it, and the ground's.
    let sky_shape = |up: f32x8| if sky_c <= 0.5 { one + (up - one) * (2.0 * sky_c) } else { (up.max(f32x8::splat(1e-30)).ln() * sky_exp).exp() };
    let norm = |x: f32x8, y: f32x8, z: f32x8| {
        let l2 = x * x + y * y + z * z;
        let inv = f32x8::ONE / l2.max(f32x8::splat(1e-30)).sqrt();
        (x * inv, y * inv, z * inv, l2)
    };
    let mut evals = 0u64;
    let mut k0 = 0;
    while k0 < n {
        let m = (n - k0).min(8);
        // Lanes past the end repeat the last point; their results are not
        // written.
        let ld = |v: &Vec<f32>| -> f32x8 {
            let mut a = [v[k0 + m - 1]; 8];
            a[..m].copy_from_slice(&v[k0..k0 + m]);
            f32x8::from(a)
        };
        let (px, py, pz) = (ld(&b.px), ld(&b.py), ld(&b.pz));
        let (nx, ny, nz) = (ld(&b.nx), ld(&b.ny), ld(&b.nz));
        let (ar, ag, ab) = (ld(&b.ar), ld(&b.ag), ld(&b.ab));
        let (f0r, f0g, f0b) = (ld(&b.fr), ld(&b.fg), ld(&b.fb));
        let rough = ld(&b.rough);
        let (vx, vy, vz, _) = norm(eye.x - px, eye.y - py, eye.z - pz);
        let ndv = (nx * vx + ny * vy + nz * vz).max(f32x8::splat(1e-4));
        // GGX: alpha is roughness squared.
        let a2 = { let a = rough * rough; a * a };
        let alpha = rough * rough;
        // What one bounce of GGX reflects of light from the whole
        // hemisphere (Karis's fit of the split-sum terms A and B): for the
        // sky's highlight, and to give back what single scattering loses
        // on rough surfaces (Kulla and Conty 2017, as Fdez-Agüera 2019).
        let (env_a, env_b) = {
            let r0 = rough * -1.0 + 1.0;
            let r1 = rough * -0.0275 + 0.0425;
            let r2 = rough * -0.572 + 1.04;
            let r3 = rough * 0.022 - 0.04;
            let a004 = (r0 * r0).min((ndv * (-9.28 * std::f32::consts::LN_2)).exp()) * r0 + r1;
            (a004 * -1.04 + r2, a004 * 1.04 + r3)
        };
        let ess = (env_a + env_b).max(f32x8::splat(1e-3));
        let ms = |f0: f32x8| one + f0 * (one / ess - one);
        let (msr, msg, msb) = (ms(f0r), ms(f0g), ms(f0b));
        // Sky, times each point's share of it: diffuse from the normal, the
        // highlight from the mirror direction (towards the normal as it
        // roughens), weighted as the split sum weights it.
        let (mut dr, mut dg, mut db) = (zero, zero, zero);
        let (mut sr, mut sg, mut sb) = (zero, zero, zero);
        if sky.on {
            let up = ny * 0.5 + 0.5;
            let mut f = sky_shape(up);
            let mut g = one - up;
            let (sv, gv) = if b.sv.is_empty() { (one, one) } else { (ld(&b.sv), ld(&b.gv)) };
            f *= sv;
            g *= gv;
            let s = sky_fill * inv_pi;
            dr = ar * (f * (sky.tint.x * s) + g * (ground_k.x * s));
            dg = ag * (f * (sky.tint.y * s) + g * (ground_k.y * s));
            db = ab * (f * (sky.tint.z * s) + g * (ground_k.z * s));
            let k = ndv * 2.0;
            let (rx, ry, rz) = (nx * k - vx, ny * k - vy, nz * k - vz);
            let t = rough * rough;
            let (_, ry, _, _) = norm(rx + (nx - rx) * t, ry + (ny - ry) * t, rz + (nz - rz) * t);
            let rup = ry * 0.5 + 0.5;
            let fs = sky_shape(rup) * sv;
            let gs = (one - rup) * gv;
            let w = |f0: f32x8| f0 * env_a + env_b;
            sr = (fs * (sky.tint.x * s) + gs * (ground_k.x * s)) * w(f0r) * msr;
            sg = (fs * (sky.tint.y * s) + gs * (ground_k.y * s)) * w(f0g) * msg;
            sb = (fs * (sky.tint.z * s) + gs * (ground_k.z * s)) * w(f0b) * msb;
        }
        let lane = |a: f32x8| a.to_array();
        // One light: illuminance per channel `e` (already times n.l and
        // visibility), n.l, direction to it; adds diffuse and highlight
        // (GGX distribution, height-correlated Smith visibility, Schlick's
        // Fresnel).
        let mut add = |e: f32x8, ndl: f32x8, (lx, ly, lz): (f32x8, f32x8, f32x8), col: Vec3, dr: &mut f32x8, dg: &mut f32x8, db: &mut f32x8| {
            let (hx, hy, hz, _) = norm(lx + vx, ly + vy, lz + vz);
            let ndh = (nx * hx + ny * hy + nz * hz).max(zero);
            let vdh = (vx * hx + vy * hy + vz * hz).max(zero);
            let q = ndh * ndh * (a2 - one) + one;
            let d = a2 / (q * q * std::f32::consts::PI).max(f32x8::splat(1e-12));
            let ndl = ndl.max(f32x8::splat(1e-4));
            // Height-correlated Smith, as Hammon (2017) approximates it:
            // no square roots.
            let two_lv = ndl * ndv * 2.0;
            let vis = f32x8::splat(0.5) / (two_lv + (ndl + ndv - two_lv) * alpha).max(f32x8::splat(1e-12));
            let fw = { let x = one - vdh; let x2 = x * x; x2 * x2 * x };
            let dv = d * vis;
            let (er, eg, eb) = (e * col.x, e * col.y, e * col.z);
            *dr += er * ar * inv_pi;
            *dg += eg * ag * inv_pi;
            *db += eb * ab * inv_pi;
            sr += er * dv * (f0r + (one - f0r) * fw) * msr;
            sg += eg * dv * (f0g + (one - f0g) * fw) * msg;
            sb += eb * dv * (f0b + (one - f0b) * fw) * msb;
        };
        // The sun and the directional lights.
        for sun in suns {
            if sun.lux == Vec3::ZERO {
                continue;
            }
            let ndl = nx * sun.dir.x + ny * sun.dir.y + nz * sun.dir.z;
            let facing = ndl.simd_gt(zero).to_bitmask() & ((1u32 << m) - 1);
            if facing != 0 {
                let vis = match sun.shadow {
                    Some(s) => s.lit8(V3 { x: px, y: py, z: pz }, V3 { x: nx, y: ny, z: nz }, facing),
                    None => std::array::from_fn(|j| if facing & (1 << j) != 0 { 1.0 } else { 0.0 }),
                };
                let e = ndl.max(zero) * f32x8::from(vis);
                let dir = (f32x8::splat(sun.dir.x), f32x8::splat(sun.dir.y), f32x8::splat(sun.dir.z));
                // The sun's lux is per channel: colour it in `add`.
                add(e, ndl, dir, sun.lux, &mut dr, &mut dg, &mut db);
            }
        }
        // Point lights in reach of the cluster.
        for &i in in_reach {
            let l = &lights[i];
            let (dx, dy, dz) = (l.position.x - px, l.position.y - py, l.position.z - pz);
            let d2 = (dx * dx + dy * dy + dz * dz).max(f32x8::splat(1e-12));
            let within = d2.simd_le(f32x8::splat(reach[i] * reach[i])).to_bitmask() & ((1u32 << m) - 1);
            evals += within.count_ones() as u64;
            if within == 0 {
                continue;
            }
            let inv_d = f32x8::ONE / d2.sqrt();
            let (lx, ly, lz) = (dx * inv_d, dy * inv_d, dz * inv_d);
            let ndl = nx * lx + ny * ly + nz * lz;
            let lit = within & ndl.simd_gt(zero).to_bitmask();
            if lit == 0 {
                continue;
            }
            let vis = match shadows[i] {
                Some(map) => map.lit8(V3 { x: px, y: py, z: pz }, V3 { x: nx, y: ny, z: nz }, lit),
                None => std::array::from_fn(|j| if lit & (1 << j) != 0 { 1.0 } else { 0.0 }),
            };
            let e = f32x8::splat(l.candela()) / d2 * ndl.max(zero) * f32x8::from(vis);
            add(e, ndl, (lx, ly, lz), l.color, &mut dr, &mut dg, &mut db);
        }
        let (dr, dg, db, sr, sg, sb) = (lane(dr), lane(dg), lane(db), lane(sr), lane(sg), lane(sb));
        for j in 0..m {
            if with_diffuse {
                diffuse[k0 + j] = Vec3::new(dr[j], dg[j], db[j]);
            }
            spec[k0 + j] = Vec3::new(sr[j], sg[j], sb[j]);
        }
        k0 += m;
    }
    evals
}

/// Shading points per second on this machine, all cores, measured for
/// about `seconds`: the real `shade_batch` on synthetic points, the sky and
/// two lights in reach (about what a point of the built-in set evaluates),
/// no shadow maps.
pub fn shading_rate(seconds: f64) -> f64 {
    let lights = [
        PointLight { position: Vec3::new(1.0, 2.0, 1.0), color: Vec3::ONE, lumens: 800.0 },
        PointLight { position: Vec3::new(-2.0, 1.0, 0.5), color: Vec3::new(1.0, 0.8, 0.6), lumens: 400.0 },
    ];
    let in_reach = [0usize, 1];
    let reach = [f32::INFINITY; 2];
    let shadows: [Option<&CubeShadow>; 2] = [None, None];
    let sky = Sky::new(30.0, true);
    let eye = Vec3::new(0.0, 1.0, 5.0);
    // Shaded as a frame shades them: batches of points, eight at a time.
    const CHUNK: usize = 1024;
    let run = |n: usize| -> f64 {
        let t = std::time::Instant::now();
        let s: f32 = (0..n.div_ceil(CHUNK))
            .into_par_iter()
            .map(|c| {
                let mut b = Batch::default();
                for i in c * CHUNK..((c + 1) * CHUNK).min(n) {
                    let a = i as f32 * 0.618;
                    let nn = Vec3::new(a.sin(), (a * 1.7).cos(), (a * 0.3).sin()).normalize_or(Vec3::Y);
                    b.push(nn * 0.5 + Vec3::new(0.0, 0.5, 0.0), nn, Vec3::splat(0.5), &crate::scene::Material::default());
                }
                let mut d = vec![Vec3::ZERO; b.len()];
                let mut sp = vec![Vec3::ZERO; b.len()];
                shade_batch(&b, eye, sky, true, &lights, &in_reach, &reach, &shadows, &[], &mut d, &mut sp, true);
                d.iter().zip(&sp).map(|(d, s)| d.x + s.y).sum::<f32>()
            })
            .sum();
        std::hint::black_box(s);
        t.elapsed().as_secs_f64()
    };
    let mut n = 2000;
    loop {
        let dt = run(n);
        if dt >= seconds || n >= 1 << 26 {
            return n as f64 / dt.max(1e-9);
        }
        n *= 2;
    }
}

/// One frame from nothing: a fresh renderer.
#[cfg(test)]
pub fn render(scene: &Scene, cam: &Camera, lights: &[PointLight], settings: &RenderSettings, frame: &mut Frame) {
    Renderer::new().render(scene, cam, lights, settings, frame);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A ground triangle seen at a grazing angle: wide and thin on screen,
    /// receding. Its grid is fine along its length only, its points no
    /// further apart than the shading rate, and there are about as many as
    /// its area on screen needs.
    #[test]
    fn dicing_is_even_on_screen() {
        let (w, h) = (640, 360);
        let cam = cam_at(0.0, w, h, Vec3::new(0.0, 0.0, -10.0), 30.0, 0.06);
        let vp = cam.proj * cam.view;
        let p = [Vec3::new(-6.0, 0.0, 0.0), Vec3::new(6.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -30.0)];
        let d = dice_triangle(vp, p, cam.near, (w, h), raster::SHADING_RATE_PX);
        let to_px = |b: [f32; 3]| {
            let q = vp * (p[0] * b[0] + p[1] * b[1] + p[2] * b[2]).extend(1.0);
            glam::Vec2::new(q.x / q.w * w as f32 * 0.5, q.y / q.w * h as f32 * 0.5)
        };
        let corner = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]].map(to_px);
        let area = ((corner[1] - corner[0]).perp_dot(corner[2] - corner[0]) * 0.5).abs();
        let (nu, nv) = (1usize << d.lu, 1usize << d.lv);
        let at = |a: usize, b: usize| to_px(screen_to_tri(d.grid_to_screen(a as f32 / nu as f32, b as f32 / nv as f32), d.w));
        let (mut widest, mut closest) = (0f32, f32::MAX);
        for a in 0..=nu {
            for b in 0..=nv {
                for (a2, b2) in [(a + 1, b), (a, b + 1)] {
                    if a2 <= nu && b2 <= nv {
                        let s = at(a, b).distance(at(a2, b2));
                        widest = widest.max(s);
                        closest = closest.min(s);
                    }
                }
            }
        }
        let rate = raster::SHADING_RATE_PX;
        assert!(widest <= rate * 1.001 && closest > rate * 0.5 * 0.999, "{closest} to {widest} px apart");
        // Even micropolygons of the rate would need area / rate^2 points;
        // the grid covers twice the triangle, and rounding to powers of two
        // can double each way.
        let ideal = area / (rate * rate);
        assert!(area > 2000.0 && (d.points() as f32) < 8.0 * ideal, "{} points for {ideal}", d.points());
        // One grid even on the whole triangle, cut as finely as its longest
        // edge needs, would hold far more.
        let longest = (0..3).map(|k| corner[k].distance(corner[(k + 1) % 3])).fold(0f32, f32::max);
        let n = (longest / rate).ceil();
        assert!(n * n / 2.0 > 1.5 * d.points() as f32, "{} against {}", n * n / 2.0, d.points());
    }

    /// Weights to grid and back give the same point.
    #[test]
    fn grid_points_are_found_where_they_were_made() {
        let d = TriDice { lu: 4, lv: 2, level: 4, base: 1, w: [1.0, 7.0, 2.5], px: [glam::Vec2::new(-40.0, 3.0), glam::Vec2::new(10.0, -20.0), glam::Vec2::new(90.0, 11.0)], ..Default::default() };
        for (u, v) in [(0.1, 0.2), (0.5, 0.5), (0.9, 0.3), (0.25, 0.95), (0.02, 0.6)] {
            let (uu, vv) = d.screen_to_grid(d.grid_to_screen(u, v));
            assert!((uu - u).abs() < 1e-4 && (vv - v).abs() < 1e-4, "({u}, {v}) came back ({uu}, {vv})");
        }
        // The triangle lies in the grid: its corners are on the grid's edges.
        for k in 0..3 {
            let mut s = [0.0; 3];
            s[k] = 1.0;
            let (u, v) = d.screen_to_grid(s);
            assert!((0.0..=1.0).contains(&u) && (0.0..=1.0).contains(&v));
        }
        let b = [0.2, 0.3, 0.5];
        let r = screen_to_tri(tri_to_screen(b, d.w), d.w);
        assert!((0..3).all(|k| (r[k] - b[k]).abs() < 1e-5));
    }

    /// Two triangles of very different size on screen share an edge, each
    /// with its own grid. Along the edge they read the same light to well
    /// under an 8-bit step, with a lamp 30 cm off the surface and
    /// micropolygons of about 3 cm. Grids do not share points, so light
    /// changing faster than the micropolygons (a lamp 10 cm off: 2 %)
    /// shows as a seam.
    #[test]
    fn neighbours_agree_along_a_shared_edge() {
        let mut scene = Scene::new();
        // Shared edge 0-2; triangle (0, 1, 2) reaches far down, (0, 2, 3)
        // barely up.
        let pos = [Vec3::new(-0.25, 0.0, 0.0), Vec3::new(0.0, -3.0, 0.0), Vec3::new(0.25, 0.0, 0.0), Vec3::new(0.0, 0.05, 0.0)];
        scene.push_mesh(&pos, &[], &[Vec3::splat(0.6)], &[0, 1, 2, 0, 2, 3]);
        let lights = [PointLight { position: Vec3::new(0.07, 0.0, 0.3), color: Vec3::ONE, lumens: 300.0 }];
        let (w, h) = (320, 320);
        let cam = cam_at(0.0, w, h, Vec3::new(0.0, -1.2, 0.0), 6.0, 0.0);
        let settings = RenderSettings { shadows: false, bounce: false, ..RenderSettings::new(Sky::new(30.0, true)) };
        let mut r = Renderer::new();
        let mut f = Frame::new(w, h);
        r.render(&scene, &cam, &lights, &settings, &mut f);
        let peek = r.peek();
        let (big, small) = (peek.tri_level(0, 0, &scene), peek.tri_level(0, 1, &scene));
        assert!(big >= small + 2, "levels {big} and {small}");
        let mut worst = 0f32;
        for k in 0..=200 {
            let t = k as f32 / 200.0;
            // The same point: 1 - t of corner 0, t of corner 2.
            let a = peek.radiance(&scene, 0, 0, [1.0 - t, 0.0, t], Part::All);
            let b = peek.radiance(&scene, 0, 1, [1.0 - t, t, 0.0], Part::All);
            worst = worst.max((a - b).abs().max_element() / a.max_element().max(1e-6));
        }
        assert!(worst < 0.004, "{worst}");
    }

    use crate::raster::Sky;

    fn cam_at(eye_yaw: f32, w: usize, h: usize, target: Vec3, dist: f32, pitch: f32) -> Camera {
        let c = crate::camera::OrbitCamera { target, distance: dist, yaw: eye_yaw, pitch, fov_y: 45f32.to_radians(), scene_radius: 1.0 };
        Camera { view: c.view(), proj: c.proj(w as f32 / h as f32), eye: c.eye(), near: c.near() }
    }

    /// Behind a wall, a grid of small boxes: once the wall has been seen,
    /// the boxes are skipped without being set up, and the picture and the
    /// pixels' triangles are those of a frame drawn without skipping.
    #[test]
    fn what_is_behind_what_was_seen_is_skipped() {
        let mut scene = Scene::new();
        let quad = |c: Vec3, hx: f32, hy: f32| [c + Vec3::new(-hx, -hy, 0.0), c + Vec3::new(hx, -hy, 0.0), c + Vec3::new(hx, hy, 0.0), c + Vec3::new(-hx, hy, 0.0)];
        // The wall, facing +z, in small quads so it is many clusters.
        for i in 0..8 {
            for j in 0..8 {
                scene.push_mesh(&quad(Vec3::new(-3.5 + i as f32, 0.5 + j as f32 * 0.5, 0.0), 0.5, 0.25), &[], &[], &[0, 1, 2, 0, 2, 3]);
            }
        }
        let behind = scene.clusters.len();
        for i in 0..10 {
            for j in 0..4 {
                scene.push_mesh(&quad(Vec3::new(-2.0 + i as f32 * 0.4, 1.0 + j as f32 * 0.5, -2.0), 0.1, 0.1), &[], &[], &[0, 1, 2, 0, 2, 3]);
            }
        }
        let (w, h) = (160, 90);
        let settings = RenderSettings { bounce: false, ..RenderSettings::new(Sky::new(30.0, true)) };
        let cam = |yaw: f32| cam_at(yaw, w, h, Vec3::new(0.0, 2.0, 0.0), 6.0, 0.0);
        let lights = [PointLight { position: Vec3::new(0.0, 3.0, 3.0), color: Vec3::ONE, lumens: 800.0 }];
        let mut r = Renderer::new();
        let mut frame = Frame::new(w, h);
        r.render(&scene, &cam(0.0), &lights, &settings, &mut frame);
        assert_eq!(frame.stats.clusters_occluded, 0, "the first frame draws everything");
        r.render(&scene, &cam(0.01), &lights, &settings, &mut frame);
        assert!(frame.stats.clusters_occluded >= scene.clusters.len() - behind - 4, "{} skipped", frame.stats.clusters_occluded);
        let mut fresh = Frame::new(w, h);
        render(&scene, &cam(0.01), &lights, &settings, &mut fresh);
        assert!(worst(&frame, &fresh) <= 2, "{}", worst(&frame, &fresh));
        let cluster_of = |f: &Frame, i: usize| (f.tri_id[i] != NO_TRI).then(|| f.setups[f.tri_id[i] as usize].cluster);
        for i in 0..w * h {
            assert_eq!(cluster_of(&frame, i), cluster_of(&fresh, i));
        }
    }

    /// Dragging one light redoes only its share: far fewer light
    /// evaluations than shading in full, and the picture of a fresh render.
    #[test]
    fn a_dragged_light_redoes_only_its_share() {
        let (scene, mut lights) = world();
        let (w, h) = (200, 112);
        // Every point shaded, so the counts compare.
        let settings = RenderSettings { bounce: false, adaptive: false, ..RenderSettings::new(Sky::new(30.0, true)) };
        let cam = cam_at(0.0, w, h, Vec3::new(0.0, 0.85, 0.0), 6.0, 0.07);
        let mut r = Renderer::new();
        let mut frame = Frame::new(w, h);
        r.render(&scene, &cam, &lights, &settings, &mut frame);
        let full = frame.stats.light_evals;
        for _ in 0..4 {
            lights[0].position += Vec3::new(0.15, 0.0, 0.05);
            r.render(&scene, &cam, &lights, &settings, &mut frame);
        }
        assert!(frame.stats.light_evals * 3 < full * 2, "{} evaluations dragging, {} in full", frame.stats.light_evals, full);
        let mut fresh = Frame::new(w, h);
        render(&scene, &cam, &lights, &settings, &mut fresh);
        assert!(worst(&frame, &fresh) <= 2, "{}", worst(&frame, &fresh));
    }

    /// Adaptive shading shades a fraction of the points and gives the
    /// picture of shading them all, within a few levels of 255 at most
    /// pixels.
    #[test]
    fn adaptive_shading_shades_less_for_the_same_picture() {
        let (scene, lights) = world();
        let (w, h) = (240, 135);
        let cam = cam_at(0.3, w, h, Vec3::new(0.0, 0.85, 0.0), 6.0, 0.07);
        let on = RenderSettings { bounce: false, ..RenderSettings::new(Sky::new(30.0, true)) };
        let off = RenderSettings { adaptive: false, ..on };
        let (mut a, mut b) = (Frame::new(w, h), Frame::new(w, h));
        render(&scene, &cam, &lights, &on, &mut a);
        render(&scene, &cam, &lights, &off, &mut b);
        assert!(a.stats.points_shaded * 2 < b.stats.points_shaded, "{} of {}", a.stats.points_shaded, b.stats.points_shaded);
        assert!(a.stats.points_interpolated > 0);
        let mut big = 0;
        for (x, y) in a.color.iter().zip(&b.color) {
            let d = [0, 8, 16].iter().map(|sh| (((x >> sh) & 0xff) as i32 - ((y >> sh) & 0xff) as i32).abs()).max().unwrap();
            if d > 4 {
                big += 1;
            }
        }
        assert!(big * 200 < w * h, "{big} pixels differ by more than 4");
    }

    fn world() -> (Scene, Vec<PointLight>) {
        let s = crate::world::build();
        (s, crate::world::small_lights())
    }

    fn worst(a: &Frame, b: &Frame) -> i32 {
        let mut worst = 0;
        for (x, y) in a.color.iter().zip(&b.color) {
            for sh in [0, 8, 16] {
                worst = worst.max((((x >> sh) & 0xff) as i32 - ((y >> sh) & 0xff) as i32).abs());
            }
        }
        worst
    }

    /// Dragging a light: visibility is kept, only clusters in its reach are
    /// shaded again, and the picture is the one a fresh render gives.
    #[test]
    fn moving_a_light_reshades_only_its_reach() {
        let (scene, mut lights) = world();
        let (w, h) = (240, 135);
        let cam = cam_at(0.0, w, h, Vec3::new(0.0, 0.85, 0.0), 6.0, 0.07);
        let settings = RenderSettings::new(Sky::new(30.0, true));
        let mut r = Renderer::new();
        let mut frame = Frame::new(w, h);
        r.render(&scene, &cam, &lights, &settings, &mut frame);
        let first = frame.stats;
        assert!(!first.visibility_kept);
        // Move the table lamp 20 cm.
        lights[0].position += Vec3::new(0.2, 0.0, 0.0);
        r.render(&scene, &cam, &lights, &settings, &mut frame);
        let s = frame.stats;
        assert!(s.visibility_kept);
        assert!(!s.full_resolve);
        assert_eq!(s.shadow_maps_built, 1);
        assert!(s.clusters_shaded > 0 && s.clusters_shaded * 3 < first.clusters_shaded, "{} of {}", s.clusters_shaded, first.clusters_shaded);
        let mut fresh = Frame::new(w, h);
        render(&scene, &cam, &lights, &settings, &mut fresh);
        // Bounce light kept beyond the reach of the change differs from a
        // fresh gather by less than the cut: at most one 8-bit step.
        assert!(worst(&frame, &fresh) <= 1, "{}", worst(&frame, &fresh));
        // Nothing changed: nothing is shaded.
        r.render(&scene, &cam, &lights, &settings, &mut frame);
        assert_eq!(frame.stats.points_shaded, 0);
        assert_eq!(frame.stats.shadow_maps_built, 0);
    }

    /// Orbiting the camera keeps diffuse shading where the dicing level
    /// holds; the picture stays close to a fresh render.
    #[test]
    fn moving_the_camera_keeps_what_it_can() {
        let (scene, lights) = world();
        let (w, h) = (240, 135);
        let settings = RenderSettings::new(Sky::new(30.0, true));
        let mut r = Renderer::new();
        let mut frame = Frame::new(w, h);
        r.render(&scene, &cam_at(0.0, w, h, Vec3::new(0.0, 0.85, 0.0), 6.0, 0.07), &lights, &settings, &mut frame);
        let cam2 = cam_at(0.02, w, h, Vec3::new(0.0, 0.85, 0.0), 6.0, 0.07);
        r.render(&scene, &cam2, &lights, &settings, &mut frame);
        assert!(!frame.stats.visibility_kept);
        assert_eq!(frame.stats.shadow_maps_built, 0, "shadow maps do not depend on the camera");
        let mut fresh = Frame::new(w, h);
        render(&scene, &cam2, &lights, &settings, &mut fresh);
        assert!(worst(&frame, &fresh) <= 6, "{}", worst(&frame, &fresh));
    }

    /// Orbiting and zooming for a while keeps the shading cache under its
    /// budget: what is off screen is dropped.
    #[test]
    fn the_cache_stays_within_its_budget() {
        let (scene, lights) = world();
        let (w, h) = (320, 180);
        let budget = 24 << 20;
        let settings = RenderSettings { cache_budget: budget, ..RenderSettings::new(Sky::new(30.0, true)) };
        let mut r = Renderer::new();
        let mut f = Frame::new(w, h);
        let mut evicted = 0;
        for i in 0..80 {
            let cam = cam_at(i as f32 * 0.08, w, h, Vec3::new(0.0, 0.5, 0.0), 3.0 + (i as f32 * 0.3).sin() * 2.0, 0.1 + (i as f32 * 0.2).cos() * 0.08);
            r.render(&scene, &cam, &lights, &settings, &mut f);
            assert!(f.stats.cache_bytes <= budget || f.stats.rate_scale > 1.0, "frame {i}: {} bytes", f.stats.cache_bytes);
            evicted += f.stats.clusters_evicted;
        }
        assert!(evicted > 0);
        assert_eq!(f.stats.rate_scale, 1.0, "what is on screen fits");
    }

    /// With a budget too small for what is on screen, micropolygons grow
    /// until it fits, and the picture is still drawn.
    #[test]
    fn a_tiny_budget_makes_micropolygons_coarser_not_a_crash() {
        let (scene, lights) = world();
        let (w, h) = (320, 180);
        let budget = 1 << 20;
        let settings = RenderSettings { cache_budget: budget, ..RenderSettings::new(Sky::new(30.0, true)) };
        let cam = cam_at(0.0, w, h, Vec3::new(0.0, 0.5, 0.0), 2.0, 0.1);
        let mut r = Renderer::new();
        let mut f = Frame::new(w, h);
        for _ in 0..6 {
            r.render(&scene, &cam, &lights, &settings, &mut f);
        }
        assert!(f.stats.rate_scale > 1.0);
        assert!(f.stats.cache_bytes <= budget, "{} bytes", f.stats.cache_bytes);
        let lit = f.color.iter().filter(|p| (**p & 0xff) > 10).count();
        assert!(lit > w * h / 2, "{lit} pixels drawn");
    }

    /// Moving an object, then changing the sky: each time the kept picture
    /// is the one a fresh render gives (the moved clusters are diced again,
    /// everything is lit again, shadows and bounce are made again).
    #[test]
    fn moving_an_object_and_relighting_match_a_fresh_render() {
        let (mut scene, lights) = world();
        let (p, idx) = crate::scene::tests::uv_sphere(24);
        let first = scene.clusters.len();
        let ball: Vec<Vec3> = p.iter().map(|v| *v * 0.3 + Vec3::new(0.0, 0.6, 0.0)).collect();
        scene.push_mesh(&ball, &[], &[Vec3::new(0.8, 0.3, 0.2)], &idx);
        let n = scene.clusters.len();
        let ob = scene.add_object("ball".into(), first..n, Vec3::new(0.0, 0.3, 0.0), Vec3::ZERO);
        // With samples, so its occlusion of the ground's sky moves too.
        scene.scatter_samples(ob);
        crate::transfer::compute(&mut scene.samples);
        let (w, h) = (200, 112);
        let cam = cam_at(0.0, w, h, Vec3::new(0.0, 0.6, 0.0), 4.0, 0.15);
        let mut settings = RenderSettings::new(Sky::new(30.0, true));
        let mut r = Renderer::new();
        let mut f = Frame::new(w, h);
        r.render(&scene, &cam, &lights, &settings, &mut f);
        let before = f.color.clone();
        let from = scene.object_bounds(ob);
        scene.set_object(ob, Vec3::new(0.6, 0.3, 0.2), Vec3::new(0.4, 0.5, -0.3), 1.4);
        let to = scene.object_bounds(ob);
        r.moved(scene.objects[ob].clusters.clone(), from, to, true);
        r.render(&scene, &cam, &lights, &settings, &mut f);
        assert_ne!(f.color, before, "the picture changed");
        let mut fresh = Frame::new(w, h);
        render(&scene, &cam, &lights, &settings, &mut fresh);
        assert!(worst(&f, &fresh) <= 1, "after the move: {}", worst(&f, &fresh));
        settings.sky = Sky::new(300.0, true);
        r.render(&scene, &cam, &lights, &settings, &mut f);
        render(&scene, &cam, &lights, &settings, &mut fresh);
        assert!(worst(&f, &fresh) <= 1, "after the sky changed: {}", worst(&f, &fresh));
    }

    /// While an object is dragged its picture goes with it over the rest:
    /// nothing is shaded or resolved after the first frame, until it has
    /// turned too far, when its picture is captured again. Once let go it
    /// is lit where it stands, as a fresh render.
    #[test]
    fn a_dragged_object_carries_its_light_until_let_go() {
        let (mut scene, lights) = world();
        let (p, idx) = crate::scene::tests::uv_sphere(24);
        let first = scene.clusters.len();
        let ball: Vec<Vec3> = p.iter().map(|v| *v * 0.3 + Vec3::new(0.0, 0.6, 0.0)).collect();
        scene.push_mesh(&ball, &[], &[Vec3::new(0.8, 0.3, 0.2)], &idx);
        let n = scene.clusters.len();
        let ob = scene.add_object("ball".into(), first..n, Vec3::new(0.0, 0.3, 0.0), Vec3::ZERO);
        let (w, h) = (200, 112);
        let cam = cam_at(0.0, w, h, Vec3::new(0.0, 0.6, 0.0), 3.0, 0.15);
        let settings = RenderSettings { adaptive: false, ..RenderSettings::new(Sky::new(30.0, true)) };
        let mut r = Renderer::new();
        let mut f = Frame::new(w, h);
        r.render(&scene, &cam, &lights, &settings, &mut f);
        let full = f.stats.points_shaded;
        // The same drag drawn in 3D, carrying its shading, to compare.
        let mut r3 = Renderer::new();
        let mut f3 = Frame::new(w, h);
        r3.render(&scene, &cam, &lights, &settings, &mut f3);
        let start = scene.objects[ob].clone();
        let mut last = Vec::new();
        for step in 1..=4 {
            let from = scene.object_bounds(ob);
            // The last step turns past what the picture can follow.
            let yaw = if step < 4 { 0.03 * step as f32 } else { 0.6 };
            let (pivot, turn) = (start.pivot + Vec3::new(0.04 * step as f32, 0.0, 0.0), Vec3::new(0.0, yaw, 0.0));
            scene.set_object(ob, pivot, turn, 1.0);
            let to = scene.object_bounds(ob);
            let ahead = glam::Affine3A::from_translation(pivot) * glam::Affine3A::from_rotation_translation(crate::scene::euler_quat(turn), Vec3::ZERO) * glam::Affine3A::from_translation(-start.pivot);
            r.dragged(vec![first..n], from, to, ahead.inverse());
            r.render(&scene, &cam, &lights, &settings, &mut f);
            r3.dragged(vec![first..n], from, to, ahead.inverse());
            r3.render_pass(&scene, &cam, &lights, &settings, &mut f3, None, Stop::Never);
            // Where the ball is in both, the moved picture is the ball as
            // drawn in 3D; it differs only along its outline and where its
            // shadow was.
            let ball_px = f3.tri_id.iter().filter(|&&id| id != NO_TRI && (first..n).contains(&(f3.setups[id as usize].cluster as usize))).count();
            let off = f.color.iter().zip(&f3.color).filter(|(a, b)| (0..3).any(|k| ((**a >> (8 * k)) & 255).abs_diff((**b >> (8 * k)) & 255) > 12)).count();
            assert!(off * 10 < ball_px, "step {step}: {off} pixels off of {ball_px} on the ball");
            assert_eq!(f.stats.layer_captured, step == 1 || step == 4, "step {step}");
            assert!(f.stats.layer_points > 500);
            if step == 2 || step == 3 {
                assert_eq!(f.stats.points_shaded, 0, "step {step}");
                assert!(f.color.iter().zip(&last).filter(|(a, b)| a != b).count() * 3 < w * h, "only round the ball changed");
            }
            if step == 4 {
                assert!(f.stats.points_shaded * 20 < full, "{} points shaded, {full} in full", f.stats.points_shaded);
            }
            last = f.color.clone();
        }
        // Let go.
        r.drag_ended(false);
        r.render(&scene, &cam, &lights, &settings, &mut f);
        let mut fresh = Frame::new(w, h);
        render(&scene, &cam, &lights, &settings, &mut fresh);
        assert!(worst(&f, &fresh) <= 1, "let go: {}", worst(&f, &fresh));
    }

    /// Let go and baked over frames: the drag's last picture stays until
    /// the bake is done, which comes out as a fresh render.
    #[test]
    fn a_bake_over_frames_shows_the_drag_then_the_fresh_render() {
        let (mut scene, lights) = world();
        let (p, idx) = crate::scene::tests::uv_sphere(24);
        let first = scene.clusters.len();
        let ball: Vec<Vec3> = p.iter().map(|v| *v * 0.3 + Vec3::new(0.0, 0.6, 0.0)).collect();
        scene.push_mesh(&ball, &[], &[Vec3::new(0.8, 0.3, 0.2)], &idx);
        let n = scene.clusters.len();
        let ob = scene.add_object("ball".into(), first..n, Vec3::new(0.0, 0.3, 0.0), Vec3::ZERO);
        let (w, h) = (200, 112);
        let cam = cam_at(0.0, w, h, Vec3::new(0.0, 0.6, 0.0), 3.0, 0.15);
        let settings = RenderSettings { adaptive: false, ..RenderSettings::new(Sky::new(30.0, true)) };
        let mut r = Renderer::new();
        let mut f = Frame::new(w, h);
        r.render(&scene, &cam, &lights, &settings, &mut f);
        let start = scene.objects[ob].pivot;
        let from = scene.object_bounds(ob);
        scene.set_object(ob, start + Vec3::new(0.3, 0.0, 0.0), Vec3::ZERO, 1.0);
        let to = scene.object_bounds(ob);
        r.dragged(vec![first..n], from, to, glam::Affine3A::from_translation(-Vec3::new(0.3, 0.0, 0.0)));
        r.render(&scene, &cam, &lights, &settings, &mut f);
        let dragging = f.color.clone();
        r.drag_ended(true);
        let mut frames = 0;
        loop {
            r.render(&scene, &cam, &lights, &settings, &mut f);
            frames += 1;
            if !f.stats.baking {
                break;
            }
            assert_eq!(f.color, dragging, "frame {frames} shows the drag");
            assert!(frames < 100);
        }
        assert!(frames >= 4, "{frames} frames");
        let mut fresh = Frame::new(w, h);
        render(&scene, &cam, &lights, &settings, &mut fresh);
        assert!(worst(&f, &fresh) <= 1, "baked: {}", worst(&f, &fresh));
    }

    /// A dragged cluster is shaded where the drag began.
    #[test]
    fn a_carried_cluster_is_where_the_drag_began() {
        let (p, idx) = crate::scene::tests::uv_sphere(8);
        let mut scene = Scene::new();
        scene.push_mesh(&p, &[], &[Vec3::ONE], &idx);
        let n = scene.clusters.len();
        scene.add_object("ball".into(), 0..n, Vec3::ZERO, Vec3::ZERO);
        let before = scene.clusters.clone();
        let (pivot, turn, k) = (Vec3::new(1.0, 2.0, -0.5), Vec3::new(0.3, 1.1, -0.4), 1.7);
        scene.set_object(0, pivot, turn, k);
        let q = crate::scene::euler_quat(turn);
        let ahead = glam::Affine3A::from_translation(pivot) * glam::Affine3A::from_scale_rotation_translation(Vec3::splat(k), q, Vec3::ZERO);
        let back = ahead.inverse();
        let (_, t, _) = back.to_scale_rotation_translation();
        for (ci, c) in scene.clusters.iter().enumerate() {
            let h = carry(c, back, t);
            for v in c.first_vertex as usize..(c.first_vertex + c.vertex_count) as usize {
                assert!(scene.pos(&h, v).distance(scene.pos(&before[ci], v)) < 1e-4);
                assert!(scene.normal(&h, v).distance(scene.normal(&before[ci], v)) < 1e-3);
            }
            assert!(h.centre.distance(before[ci].centre) < 1e-4);
        }
    }

    /// White under an even sky (a furnace test): a white dielectric of
    /// any roughness, seen from any angle, sends back about what reaches
    /// it and never more; a white metal at most that.
    #[test]
    fn white_under_an_even_sky_conserves_energy() {
        let sky = Sky { contrast: 0.0, ..Sky::new(100.0, true) };
        let l = sky.irradiance_from(Vec3::Y, false).x * std::f32::consts::FRAC_1_PI;
        for metallic in [0.0, 1.0] {
            for rough in [0.05, 0.3, 0.6, 1.0] {
                for deg in [0.0f32, 45.0, 75.0, 85.0] {
                    let mat = crate::scene::Material { tint: Vec3::ONE, roughness: rough, metallic, specular: 0.5, ..Default::default() };
                    let mut b = Batch::default();
                    b.push(Vec3::ZERO, Vec3::Y, Vec3::ONE, &mat);
                    let eye = Vec3::new(deg.to_radians().sin(), deg.to_radians().cos(), 0.0) * 10.0;
                    let (mut d, mut sp) = (vec![Vec3::ZERO], vec![Vec3::ZERO]);
                    shade_batch(&b, eye, sky, false, &[], &[], &[], &[], &[], &mut d, &mut sp, true);
                    // Diffuse as the resolve weighs it for this view.
                    let k = (d[0].x * mat.diffuse_view_weight(deg.to_radians().cos()) + sp[0].x) / l;
                    let lo = if metallic > 0.5 { 0.6 } else { 0.85 };
                    assert!(k <= 1.08 && k >= lo, "metallic {metallic}, roughness {rough}, {deg} degrees: {k} of what arrives");
                }
            }
        }
    }

    /// A metal has no diffuse light, and a rougher surface spreads its
    /// highlight wider and lower.
    #[test]
    fn metal_and_roughness_change_the_light() {
        let light = [PointLight { position: Vec3::new(0.0, 2.0, 0.0), color: Vec3::ONE, lumens: 1000.0 }];
        let shade = |mat: crate::scene::Material, eye: Vec3| {
            let mut b = Batch::default();
            b.push(Vec3::ZERO, Vec3::Y, Vec3::splat(0.5), &mat);
            let (mut d, mut sp) = (vec![Vec3::ZERO], vec![Vec3::ZERO]);
            shade_batch(&b, eye, Sky::new(0.0, false), false, &light, &[0], &[f32::INFINITY], &[None], &[], &mut d, &mut sp, true);
            (d[0], sp[0])
        };
        let metal = crate::scene::Material { metallic: 1.0, ..Default::default() };
        let (d, s) = shade(metal, Vec3::new(0.0, 3.0, 0.0));
        assert_eq!(d, Vec3::ZERO);
        assert!(s.x > 0.0);
        let smooth = crate::scene::Material { roughness: 0.15, ..Default::default() };
        let rough = crate::scene::Material { roughness: 0.8, ..Default::default() };
        // Facing the mirror direction: the smooth one is brighter; well off
        // it, the rough one.
        assert!(shade(smooth.clone(), Vec3::new(0.0, 3.0, 0.0)).1.x > 4.0 * shade(rough.clone(), Vec3::new(0.0, 3.0, 0.0)).1.x);
        assert!(shade(rough, Vec3::new(3.0, 1.0, 0.0)).1.x > shade(smooth, Vec3::new(3.0, 1.0, 0.0)).1.x);
    }

    /// A card between a light and the floor darkens the floor under it.
    #[test]
    fn the_renderer_draws_shadows() {
        let mut scene = Scene::new();
        let quad = |c: Vec3, h: f32| [c + Vec3::new(-h, 0.0, h), c + Vec3::new(h, 0.0, h), c + Vec3::new(h, 0.0, -h), c + Vec3::new(-h, 0.0, -h)];
        scene.push_mesh(&quad(Vec3::ZERO, 2.0), &[], &[Vec3::splat(0.5)], &[0, 1, 2, 0, 2, 3]);
        scene.push_mesh(&quad(Vec3::new(0.0, 0.6, 0.0), 0.25), &[], &[Vec3::splat(0.5)], &[0, 1, 2, 0, 2, 3]);
        let lights = [PointLight { position: Vec3::new(0.0, 1.6, 0.0), color: Vec3::ONE, lumens: 2000.0 }];
        let (w, h) = (160, 120);
        // Looking down from the side, so the floor under the card shows.
        let cam = cam_at(0.0, w, h, Vec3::ZERO, 3.0, 1.0);
        let on = RenderSettings::new(Sky::new(1.0, false));
        // Every point shaded: adaptive shading refines where the shadow
        // makes an edge, so a point there can come out a level above its
        // interpolated self without the shadow.
        let mut a = Frame::new(w, h);
        render(&scene, &cam, &lights, &RenderSettings { exposure: 0.01, adaptive: false, ..on }, &mut a);
        let mut b = Frame::new(w, h);
        render(&scene, &cam, &lights, &RenderSettings { exposure: 0.01, shadows: false, adaptive: false, ..on }, &mut b);
        // Pixels where the floor shows in both and differs: under the card's shadow.
        let darker = a.color.iter().zip(&b.color).filter(|(x, y)| (**x & 0xff) + 20 < (**y & 0xff)).count();
        assert!(darker > 50, "{darker} pixels darker with shadows");
        let brighter: Vec<(usize, u32, u32)> = a.color.iter().zip(&b.color).enumerate().filter(|(_, (x, y))| (**x & 0xff) > (**y & 0xff)).map(|(i, (x, y))| (i, *x, *y)).collect();
        assert!(brighter.is_empty(), "{brighter:x?}");
    }

    /// A textured quad is shaded from its tiles, not its vertex colours.
    #[test]
    fn shading_reads_texture_tiles() {
        use crate::texture::{level_offset, Source, TexEntry};
        use std::sync::Arc;
        let mut scene = Scene::new();
        let quad = [Vec3::new(-1.0, -1.0, 0.0), Vec3::new(1.0, -1.0, 0.0), Vec3::new(1.0, 1.0, 0.0), Vec3::new(-1.0, 1.0, 0.0)];
        scene.push_mesh(&quad, &[], &[Vec3::splat(0.5)], &[0, 1, 2, 0, 2, 3]);
        let c = scene.clusters[0];
        // Pure red tiles at levels 1 to 5.
        let len = level_offset(c.tri_count as usize, 6) as usize;
        let pool: Vec<u8> = (0..len / 3).flat_map(|_| [255u8, 0, 0]).collect();
        scene.textures.entries = vec![TexEntry { level: 5, source: 0, offset: 0, alpha: None }];
        scene.textures.sources = vec![Arc::new(Source::Memory(Arc::new(pool)))];
        let (w, h) = (160, 120);
        let cam = cam_at(0.0, w, h, Vec3::ZERO, 3.0, 0.0);
        let settings = RenderSettings { shadows: false, bounce: false, ..RenderSettings::new(Sky::new(100.0, true)) };
        let mut f = Frame::new(w, h);
        render(&scene, &cam, &[], &settings, &mut f);
        let px = f.color[h / 2 * w + w / 2];
        let (r, g, b) = ((px >> 16) & 255, (px >> 8) & 255, px & 255);
        // Red, with the sky's highlight (4% of it facing) on top.
        assert!(r > 40 && g * 4 < r && b * 4 < r, "red from the tiles, got {r} {g} {b}");
        assert!(scene.textures.stats().reads >= 1);
    }

    /// A card cut out by alpha tiles: where its mask is clear it is not
    /// drawn and casts no shadow; it is seen from both sides.
    #[test]
    fn alpha_tiles_cut_out_a_card_from_both_sides() {
        use crate::texture::{level_offset, tile_points, Source, TexEntry};
        use std::sync::Arc;
        let mut scene = Scene::new();
        let quad = [Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), Vec3::new(1.0, 2.0, 0.0), Vec3::new(-1.0, 2.0, 0.0)];
        scene.push_mesh(&quad, &[], &[Vec3::splat(0.5)], &[0, 1, 2, 0, 2, 3]);
        let c = scene.clusters[0];
        let n = c.tri_count as usize;
        let len = level_offset(n, 4) as usize;
        let pool: Vec<u8> = vec![128; len];
        // The first triangle clear, the second solid, at levels 1 to 3.
        let alpha: Vec<u8> = (1..=3u8).flat_map(|l| (0..n).flat_map(move |t| std::iter::repeat_n(if t == 0 { 0u8 } else { 255 }, tile_points(l)))).collect();
        scene.textures.entries = vec![TexEntry { level: 3, source: 0, offset: 0, alpha: Some(0) }];
        scene.textures.sources = vec![Arc::new(Source::Memory(Arc::new(pool)))];
        scene.textures.alpha_sources = vec![Some(Arc::new(Source::Memory(Arc::new(alpha))))];
        let (w, h) = (160, 120);
        let settings = RenderSettings { shadows: false, bounce: false, ..RenderSettings::new(Sky::new(100.0, true)) };
        let mut counts = Vec::new();
        for yaw in [0.0, std::f32::consts::PI] {
            let cam = cam_at(yaw, w, h, Vec3::new(0.0, 1.0, 0.0), 3.0, 0.0);
            let mut f = Frame::new(w, h);
            render(&scene, &cam, &[], &settings, &mut f);
            let tris: Vec<u32> = f.tri_id.iter().filter(|&&id| id != NO_TRI).map(|&id| f.setups[id as usize].tri).collect();
            assert!(tris.iter().all(|&t| t == c.first_tri + 1), "only the solid triangle is seen");
            counts.push(tris.len());
        }
        assert!(counts.iter().all(|&k| k > 500), "seen from both sides: {counts:?}");
        // Over a floor, under a light: the clear half casts no shadow.
        let floor = [Vec3::new(-3.0, -0.01, -3.0), Vec3::new(3.0, -0.01, -3.0), Vec3::new(3.0, -0.01, 3.0), Vec3::new(-3.0, -0.01, 3.0)];
        let mut lit = Scene::new();
        lit.push_mesh(&floor.map(|p| p + Vec3::new(0.0, 0.0, 0.0)), &[], &[Vec3::splat(0.5)], &[0, 2, 1, 0, 3, 2]);
        let card: Vec<Vec3> = quad.iter().map(|p| Vec3::new(p.x, 1.0, p.y - 1.0)).collect();
        lit.push_mesh(&card, &[], &[Vec3::splat(0.5)], &[0, 1, 2, 0, 2, 3]);
        lit.textures.entries = vec![TexEntry::default(), scene.textures.entries[0]];
        lit.textures.sources = scene.textures.sources.clone();
        lit.textures.alpha_sources = scene.textures.alpha_sources.clone();
        let m = crate::shadow::CubeShadow::build(&lit, Vec3::new(0.0, 3.0, 0.0), 10.0, 256);
        // The card's triangles: (-1,-1) (1,-1) (1,1) clear, (-1,-1) (1,1) (-1,1) solid.
        let under_clear = Vec3::new(0.5, 0.0, -0.5);
        let under_solid = Vec3::new(-0.5, 0.0, 0.5);
        assert!(m.lit(under_clear, Vec3::Y) > 0.5, "the clear half casts nothing");
        assert!(m.lit(under_solid, Vec3::Y) < 0.5, "the solid half casts its shadow");
    }

    /// A floor and a tall wall as one object with its sky transfer: the
    /// floor by the wall gets about half the sky it gets with sky
    /// occlusion off, the floor far from it the same.
    #[test]
    fn the_sky_share_darkens_a_corner_and_nothing_else() {
        let mut scene = Scene::new();
        let grid = |c: Vec3, u: Vec3, v: Vec3, n: usize| {
            let mut p = Vec::new();
            let mut idx = Vec::new();
            for j in 0..=n {
                for i in 0..=n {
                    p.push(c + u * (i as f32 / n as f32 * 2.0 - 1.0) + v * (j as f32 / n as f32 * 2.0 - 1.0));
                }
            }
            let w = (n + 1) as u32;
            for j in 0..n as u32 {
                for i in 0..n as u32 {
                    let a = j * w + i;
                    idx.extend_from_slice(&[a, a + 1, a + w + 1, a, a + w + 1, a + w]);
                }
            }
            (p, idx)
        };
        // Two triangles of floor: the samples, not the mesh, carry the detail.
        let (p, i) = grid(Vec3::ZERO, Vec3::X * 2.0, Vec3::NEG_Z * 2.0, 1);
        scene.push_mesh(&p, &[], &[Vec3::splat(0.5)], &i);
        let (p, i) = grid(Vec3::new(-2.0, 2.0, 0.0), Vec3::NEG_Z * 2.0, Vec3::Y * 2.0, 2);
        scene.push_mesh(&p, &[], &[Vec3::splat(0.5)], &i);
        let n = scene.clusters.len();
        let o = scene.add_object("corner".into(), 0..n, Vec3::ZERO, Vec3::ZERO);
        scene.scatter_samples(o);
        crate::transfer::compute(&mut scene.samples);
        let (w, h) = (160, 120);
        let cam = cam_at(0.4, w, h, Vec3::new(0.0, 0.0, 0.0), 5.0, 0.7);
        let on = RenderSettings { bounce: false, shadows: false, ..RenderSettings::new(Sky::new(300.0, true)) };
        let mut a = Frame::new(w, h);
        render(&scene, &cam, &[], &on, &mut a);
        let mut b = Frame::new(w, h);
        render(&scene, &cam, &[], &RenderSettings { sky_occlusion: false, ..on }, &mut b);
        let (mut near, mut far) = ((0.0, 0.0, 0), (0.0, 0.0, 0));
        for y in 0..h {
            for x in 0..w {
                let Some((p, n)) = raster::unproject(&a, &scene, x, y) else { continue };
                if n.y < 0.9 {
                    continue;
                }
                let (va, vb) = ((a.color[y * w + x] & 255) as f32, (b.color[y * w + x] & 255) as f32);
                let slot = if p.x < -1.85 { &mut near } else if p.x > 1.0 { &mut far } else { continue };
                slot.0 += va;
                slot.1 += vb;
                slot.2 += 1;
            }
        }
        assert!(near.2 > 10 && far.2 > 10, "{near:?} {far:?}");
        let near_ratio = near.0 / near.1;
        assert!(near_ratio < 0.85, "by the wall: {near_ratio}");
        assert!((far.0 / far.1 - 1.0).abs() < 0.05, "far from it: {far:?}");
    }

    /// A ball just over a floor that is no object: the floor under it gets
    /// less sky, the floor two metres away all of it.
    #[test]
    fn an_object_takes_sky_from_the_ground_under_it() {
        let mut scene = Scene::new();
        let quad = |c: Vec3, h: f32| [c + Vec3::new(-h, 0.0, h), c + Vec3::new(h, 0.0, h), c + Vec3::new(h, 0.0, -h), c + Vec3::new(-h, 0.0, -h)];
        scene.push_mesh(&quad(Vec3::ZERO, 3.0), &[], &[Vec3::splat(0.5)], &[0, 1, 2, 0, 2, 3]);
        let (p, idx) = crate::scene::tests::uv_sphere(32);
        let first = scene.clusters.len();
        let ball: Vec<Vec3> = p.iter().map(|v| *v * 0.4 + Vec3::new(0.0, 0.42, 0.0)).collect();
        scene.push_mesh(&ball, &[], &[Vec3::splat(0.5)], &idx);
        let n = scene.clusters.len();
        let o = scene.add_object("ball".into(), first..n, Vec3::ZERO, Vec3::ZERO);
        scene.scatter_samples(o);
        let (w, h) = (160, 120);
        let cam = cam_at(0.3, w, h, Vec3::ZERO, 4.0, 0.9);
        let on = RenderSettings { bounce: false, shadows: false, ..RenderSettings::new(Sky::new(300.0, true)) };
        let mut a = Frame::new(w, h);
        render(&scene, &cam, &[], &on, &mut a);
        let mut b = Frame::new(w, h);
        render(&scene, &cam, &[], &RenderSettings { sky_occlusion: false, ..on }, &mut b);
        let (mut near, mut far) = ((0.0, 0.0, 0), (0.0, 0.0, 0));
        for y in 0..h {
            for x in 0..w {
                let Some((p, n)) = raster::unproject(&a, &scene, x, y) else { continue };
                if n.y < 0.9 || p.y > 0.01 {
                    continue;
                }
                let r = glam::Vec2::new(p.x, p.z).length();
                let slot = if r > 0.42 && r < 0.6 { &mut near } else if r > 2.0 { &mut far } else { continue };
                slot.0 += (a.color[y * w + x] & 255) as f32;
                slot.1 += (b.color[y * w + x] & 255) as f32;
                slot.2 += 1;
            }
        }
        assert!(near.2 > 10 && far.2 > 10, "{near:?} {far:?}");
        // About a quarter of the sky there, cosine weighted, is the
        // ball: 0.75 of the light, about 0.88 in 8-bit sRGB.
        assert!(near.0 / near.1 < 0.93, "by the ball: {near:?}");
        assert!((far.0 / far.1 - 1.0).abs() < 0.02, "far: {far:?}");
    }

    /// A directional light lights a floor and a card casts its shadow;
    /// the backdrop (sky) is the same with it or without.
    #[test]
    fn a_directional_light_lights_the_scene_not_the_sky() {
        let mut scene = Scene::new();
        let quad = |c: Vec3, h: f32| [c + Vec3::new(-h, 0.0, h), c + Vec3::new(h, 0.0, h), c + Vec3::new(h, 0.0, -h), c + Vec3::new(-h, 0.0, -h)];
        scene.push_mesh(&quad(Vec3::ZERO, 3.0), &[], &[Vec3::splat(0.5)], &[0, 1, 2, 0, 2, 3]);
        scene.push_mesh(&quad(Vec3::new(0.0, 0.6, 0.0), 0.3), &[], &[Vec3::splat(0.5)], &[0, 1, 2, 0, 2, 3]);
        let (w, h) = (160, 120);
        let cam = cam_at(0.0, w, h, Vec3::ZERO, 4.0, 0.5);
        let off = RenderSettings { bounce: false, cull_backfaces: false, ..RenderSettings::new(Sky::new(30.0, true)) };
        let mut lit = RenderSettings { ..off };
        lit.dir_lights[0] = raster::DirLight { on: true, dir: Vec3::Y, color: Vec3::ONE, lux: 300.0 };
        let (mut a, mut b) = (Frame::new(w, h), Frame::new(w, h));
        render(&scene, &cam, &[], &off, &mut a);
        render(&scene, &cam, &[], &lit, &mut b);
        let (mut brighter, mut shaded, mut sky_same) = (0, 0, true);
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if a.tri_id[i] == NO_TRI {
                    sky_same &= a.color[i] == b.color[i];
                    continue;
                }
                let Some((p, _)) = raster::unproject(&b, &scene, x, y) else { continue };
                if p.y > 0.01 {
                    continue;
                }
                let (va, vb) = (a.color[i] & 255, b.color[i] & 255);
                if p.x.abs() < 0.2 && p.z.abs() < 0.2 {
                    // Under the card: its shadow, straight down.
                    if vb <= va + 2 {
                        shaded += 1;
                    }
                } else if p.x.abs() > 1.0 && vb > va + 20 {
                    brighter += 1;
                }
            }
        }
        assert!(sky_same, "the backdrop does not change");
        assert!(brighter > 100, "{brighter} pixels lit");
        assert!(shaded > 10, "{shaded} pixels in the card's shadow");
        let (az, el) = raster::DirLight::angles(raster::DirLight::towards(123.0, 37.0));
        assert!((az - 123.0).abs() < 1e-3 && (el - 37.0).abs() < 1e-3);
    }

    /// Turning a directional light redoes only its own share of each point,
    /// and the picture is the one drawn fresh.
    #[test]
    fn a_turned_directional_light_matches_a_fresh_render() {
        let (mut scene, lights) = world();
        let (p, idx) = crate::scene::tests::uv_sphere(24);
        let ball: Vec<Vec3> = p.iter().map(|v| *v * 0.3 + Vec3::new(0.0, 0.6, 0.0)).collect();
        scene.push_mesh(&ball, &[], &[Vec3::new(0.8, 0.3, 0.2)], &idx);
        let (w, h) = (200, 112);
        let cam = cam_at(0.0, w, h, Vec3::new(0.0, 0.6, 0.0), 4.0, 0.15);
        // Every point shaded, as the share is: adaptive shading would
        // interpolate where the fresh render does not.
        let mut settings = RenderSettings { adaptive: false, ..RenderSettings::new(Sky::new(30.0, true)) };
        settings.dir_lights[0] = raster::DirLight { on: true, dir: raster::DirLight::towards(40.0, 50.0), color: Vec3::new(1.0, 0.9, 0.8), lux: 200.0 };
        let mut r = Renderer::new();
        let mut f = Frame::new(w, h);
        r.render(&scene, &cam, &lights, &settings, &mut f);
        for step in 1..=3 {
            settings.dir_lights[0].dir = raster::DirLight::towards(40.0 + 20.0 * step as f32, 50.0 - 5.0 * step as f32);
            r.render(&scene, &cam, &lights, &settings, &mut f);
            assert!(f.stats.shadow_maps_built <= 1, "only its own map is drawn again");
        }
        let mut fresh = Frame::new(w, h);
        render(&scene, &cam, &lights, &settings, &mut fresh);
        assert!(worst(&f, &fresh) <= 1, "{}", worst(&f, &fresh));
        // Off again: as if it had never been on.
        settings.dir_lights[0].on = false;
        r.render(&scene, &cam, &lights, &settings, &mut f);
        render(&scene, &cam, &lights, &settings, &mut fresh);
        assert!(worst(&f, &fresh) <= 1, "off: {}", worst(&f, &fresh));
    }
}
