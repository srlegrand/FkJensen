//! Visibility: which surface each pixel sees.
//!
//! Three passes, each parallel across cores with rayon:
//!
//! 1. **Setup**: clusters are tested against the view frustum (and, with
//!    back-face culling on, against their normal cone) as a whole; the
//!    vertices of each surviving cluster are projected into a small local
//!    array and its triangles are clipped against the near plane, optionally
//!    back-face culled, and given a screen-space record. Each visible
//!    cluster also gets a dicing level from the screen size of its
//!    triangles, for the shading in `reyes.rs`.
//! 2. **Bin**: each record is listed in every 64 x 64 bucket its box touches.
//! 3. **Hide**: each bucket is rasterised by one thread into its own part
//!    of the depth and triangle-id buffers. No atomics, no locks.
//!
//! Nothing is shaded here. Shading happens on diced micropolygons in world
//! space and is kept between frames (`reyes.rs`); when the camera has not
//! moved, this whole module is skipped.
//!
//! Also here: the sky, light reach, and small drawing helpers.

use glam::{Mat4, Vec2, Vec3, Vec4};
use rayon::prelude::*;

use crate::scene::{Cluster, PointLight, Scene};

/// Default bucket edge in pixels; `RenderSettings::tile` sets it per machine.
pub const TILE: usize = 64;
pub const NO_TRI: u32 = u32::MAX;

pub struct Camera {
    pub view: Mat4,
    pub proj: Mat4,
    pub eye: Vec3,
    pub near: f32,
}

pub struct Frame {
    pub width: usize,
    pub height: usize,
    /// 0x00RRGGBB, sRGB, what softbuffer wants.
    pub color: Vec<u32>,
    /// 1/w per pixel, 0 where empty. Larger is closer.
    pub depth: Vec<f32>,
    /// Index into the frame's setup records, or NO_TRI.
    pub tri_id: Vec<u32>,
    /// Per-frame setup records, kept so the app can pick (unproject a pixel).
    pub setups: Vec<TriSetup>,
    pub stats: Stats,
    /// Which renderer's visibility the buffers hold (0: none), so a
    /// renderer only reuses visibility it made itself in this frame.
    pub vis_stamp: u64,
}

#[derive(Default, Clone, Copy, Debug)]
pub struct Stats {
    pub clusters_in: usize,
    pub clusters_drawn: usize,
    /// Point light evaluations done while shading this frame.
    pub light_evals: u64,
    /// What they would have been without light reach: shading points
    /// shaded this frame x lights.
    pub light_evals_all: u64,
    pub tris_in: usize,
    pub tris_drawn: usize,
    /// Whether visibility was kept from the last frame (camera unchanged).
    pub visibility_kept: bool,
    /// While an object is dragged: the points of its picture moved over
    /// the rest, and whether that picture was captured this frame.
    pub layer_points: usize,
    pub layer_captured: bool,
    /// Moving that picture and drawing it over the rest, ms.
    pub layer_ms: f32,
    /// A let-go object is still being baked: the picture is the drag's.
    pub baking: bool,
    /// Shading points shaded this frame (the rest came from the cache).
    pub points_shaded: u64,
    /// Shading points the frame used.
    pub points_used: u64,
    /// Grid points interpolated rather than shaded this frame.
    pub points_interpolated: u64,
    /// Clusters that had any shading done this frame.
    pub clusters_shaded: usize,
    /// Whether every pixel was resolved, or only those of clusters shaded
    /// this frame.
    pub full_resolve: bool,
    /// Surfels lit again, receivers whose bounce was gathered, receivers in
    /// view still waiting, and the time spent.
    pub surfels_lit: u64,
    pub receivers_updated: usize,
    pub receivers_pending: usize,
    pub bounce_ms: f32,
    pub gather_ms: f32,
    /// Shadow maps rebuilt this frame, and triangles drawn into them.
    pub shadow_maps_built: usize,
    pub shadow_tris: u64,
    pub setup_ms: f32,
    pub bin_ms: f32,
    pub raster_ms: f32,
    pub shadow_ms: f32,
    pub shade_ms: f32,
    pub resolve_ms: f32,
    /// Bytes held by the shading cache after the frame, clusters dropped
    /// from it to fit the budget, and the shading rate's multiplier.
    pub cache_bytes: usize,
    pub clusters_evicted: usize,
    pub rate_scale: f32,
    /// Occlusion culling: clusters drawn first (those seen last frame),
    /// clusters then found hidden behind them and skipped.
    pub clusters_first: usize,
    pub clusters_occluded: usize,
    /// Clusters shaded again because an object's shadow moved over them.
    pub clusters_relit: usize,
}

/// A triangle after projection: screen positions, 1/w, and each corner's
/// weights over the source triangle's three vertices (identity unless the
/// triangle was clipped).
#[derive(Clone, Copy)]
pub struct TriSetup {
    pub x: [f32; 3],
    pub y: [f32; 3],
    pub inv_w: [f32; 3],
    pub bary: [[f32; 3]; 3],
    pub tri: u32,
    /// The cluster the triangle belongs to.
    pub cluster: u32,
    /// Its cluster's alpha tiles, an index into the frame's list of them
    /// (`NO_ALPHA`: opaque).
    pub alpha: u32,
    pub min_x: u16,
    pub min_y: u16,
    pub max_x: u16,
    pub max_y: u16,
}

/// The sky: light from every direction above, with nothing between it and
/// any surface, so a mountain 10 km away and a pawn are lit by it alike.
/// This is the sky as light; how it looks behind the scene is `Backdrop`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sky {
    /// Illuminance on an upward-facing surface, in lux.
    pub lux: f32,
    pub on: bool,
    /// Colour of its light on an upward-facing surface, luminance about 1.
    pub tint: Vec3,
    /// Lighting contrast, 0 to 1: at 0 every facing gets the same light
    /// (overcast, flat) and twice as much of it; at 0.5 the light falls off
    /// evenly from up to down; at 1 it comes from overhead only, half as
    /// strong. Changes shading, never the backdrop.
    pub contrast: f32,
}

/// Sky colour seen straight up and at the horizon, luminance about 1.
pub const SKY_ZENITH: Vec3 = Vec3::new(0.55, 0.78, 1.45);
pub const SKY_HORIZON: Vec3 = Vec3::new(1.15, 1.0, 0.85);
/// Light coming back up from the ground, as a share of the sky's.
pub(crate) const GROUND_BOUNCE: f32 = 0.2;
pub(crate) const GROUND_TINT: Vec3 = Vec3::new(1.0, 1.0, 0.9);
/// Colour of the sky's light on an upward-facing surface (luminance 1).
pub const SKY_IRRADIANCE_TINT: Vec3 = Vec3::new(0.82, 0.95, 1.25);

impl Sky {
    pub fn new(lux: f32, on: bool) -> Sky {
        Sky { lux, on, tint: SKY_IRRADIANCE_TINT, contrast: 0.5 }
    }

    /// Illuminance in lux, per colour channel, on a surface facing `n`.
    #[allow(dead_code)]
    pub fn irradiance(&self, n: Vec3) -> Vec3 {
        self.irradiance_from(n, true)
    }

    /// The same, with or without the light the ground sends back up. With
    /// bounce light on, the ground term is left out: the real ground is in
    /// the surfel cloud and bounces its own light.
    pub fn irradiance_from(&self, n: Vec3, ground: bool) -> Vec3 {
        if !self.on {
            return Vec3::ZERO;
        }
        let up = n.y * 0.5 + 0.5;
        let c = self.contrast.clamp(0.0, 1.0);
        // How much of the sky a facing gets, and how strong the sky's
        // light is, by contrast.
        let f = if c <= 0.5 { 1.0 + (up - 1.0) * (2.0 * c) } else { up.powf(1.0 + 6.0 * (c - 0.5)) };
        let fill = (1.0 - 2.0 * c).exp2();
        let g = if ground { GROUND_TINT * GROUND_BOUNCE * (1.0 - up) } else { Vec3::ZERO };
        self.lux * fill * (self.tint * f + g)
    }
}

/// How the sky looks behind the scene: never lights anything, so changing
/// it re-shades nothing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Backdrop {
    pub zenith: Vec3,
    pub horizon: Vec3,
    /// Visible sky contrast, 0 to 1: 0 is one flat colour, 0.5 the gradient
    /// as given, 1 twice as strong; it also sets the sun's glow.
    pub contrast: f32,
}

impl Default for Backdrop {
    fn default() -> Self {
        Backdrop { zenith: SKY_ZENITH, horizon: SKY_HORIZON, contrast: 0.5 }
    }
}

/// Most directional lights.
pub const MAX_DIR_LIGHTS: usize = 4;

/// A directional light: infinitely far, parallel light and parallel
/// shadows, as the sun, but it lights the scene only: the sky, the backdrop
/// and the exposure do not follow it. Its strength is its illuminance in
/// lux on a surface facing it, a physical unit as a point light's lumens.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DirLight {
    pub on: bool,
    /// Unit direction towards the light.
    pub dir: Vec3,
    /// Linear RGB, luminance 1 (`lights::normalise_colour`).
    pub color: Vec3,
    pub lux: f32,
}

/// Brightest a directional light may be: a little over full summer sun.
pub const DIR_MAX_LUX: f32 = 150_000.0;

impl Default for DirLight {
    fn default() -> Self {
        DirLight { on: false, dir: DirLight::towards(210.0, 40.0), color: Vec3::ONE, lux: 2_000.0 }
    }
}

impl DirLight {
    /// Unit direction towards a light at `azimuth` degrees (0 at -Z, the
    /// way the camera looks, turning towards +X) and `elevation` degrees
    /// above the horizon.
    pub fn towards(azimuth: f32, elevation: f32) -> Vec3 {
        let (a, e) = (azimuth.to_radians(), elevation.to_radians());
        Vec3::new(a.sin() * e.cos(), e.sin(), -a.cos() * e.cos()).normalize()
    }

    /// Azimuth and elevation of `dir`, degrees.
    pub fn angles(dir: Vec3) -> (f32, f32) {
        let e = dir.y.clamp(-1.0, 1.0).asin().to_degrees();
        let a = dir.x.atan2(-dir.z).to_degrees().rem_euclid(360.0);
        (a, e)
    }

    /// Whether it lights anything.
    pub fn lights(&self) -> bool {
        self.on && self.lux > 0.0
    }
}

/// The sun: infinitely far, parallel light and parallel shadows, the same
/// on everything in the scene. Its brightness is 0 to 1000: 0 is none, 1000
/// lights a white surface facing it to full white at the current exposure.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sun {
    /// Unit direction towards the sun.
    pub dir: Vec3,
    pub color: Vec3,
    pub brightness: f32,
}

pub const SUN_MAX: f32 = 1000.0;

impl Default for Sun {
    fn default() -> Self {
        Sun { dir: Vec3::new(0.3, 0.7, 0.6).normalize(), color: Vec3::ONE, brightness: 0.0 }
    }
}

impl Sun {
    /// Whether it lights anything: some brightness, above the horizon.
    pub fn up(&self) -> bool {
        self.brightness > 0.0 && self.dir.y > 0.0
    }

    /// Illuminance on a surface facing it, lux per channel, at `exposure`.
    /// At 1000, a white surface facing it shows 4 before the tone curve,
    /// which the curve takes to full white.
    pub fn lux(&self, exposure: f32) -> Vec3 {
        if !self.up() {
            return Vec3::ZERO;
        }
        let at_white = 4.0 * std::f32::consts::PI / exposure.max(1e-12);
        self.color * (self.brightness.clamp(0.0, SUN_MAX) / SUN_MAX * at_white)
    }
}

/// Radiance (cd/m²) of the backdrop seen along `dir`, with the sun's disc
/// and glow.
pub fn backdrop_radiance(sky: &Sky, look: &Backdrop, sun: &Sun, exposure: f32, dir: Vec3) -> Vec3 {
    let mut c = Vec3::ZERO;
    if sky.on {
        let l = sky.lux / std::f32::consts::PI;
        if dir.y >= 0.0 {
            let t = dir.y.sqrt();
            let k = look.contrast.clamp(0.0, 1.0);
            let g = look.horizon.lerp(look.zenith, t);
            let mean = (look.horizon + look.zenith) * 0.5;
            // Below 0.5 the colours flatten towards one; above it the
            // horizon also brightens and the zenith darkens.
            let ramp = 1.0 + (3.2 * (k - 0.5)).max(0.0) * (0.5 - t);
            c = l * ((mean + (g - mean) * (2.0 * k)) * ramp).max(Vec3::ZERO);
        } else {
            c = l * GROUND_BOUNCE * GROUND_TINT;
        }
    }
    if sun.up() && dir.y > -0.02 {
        let cos = dir.dot(sun.dir).clamp(-1.0, 1.0);
        let ang = cos.acos();
        let e = sun.lux(exposure);
        // The disc, 0.27 degrees in radius, drawn at a radius of 0.6 so it
        // reads at viewport sizes: full white.
        if ang < 0.6f32.to_radians() {
            return Vec3::splat(8.0 / exposure.max(1e-12));
        }
        // The glow round it, stronger with the visible sky's contrast.
        c += e / std::f32::consts::PI * (0.12 * look.contrast.clamp(0.0, 1.0) * 2.0 * (-ang / 0.09).exp());
    }
    c
}

/// What the shade pass writes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum View {
    Shaded,
    /// How many point lights each pixel evaluates: shows light reach.
    LightCount,
}

pub struct RenderSettings {
    pub cull_backfaces: bool,
    pub wireframe: bool,
    /// Display value per cd/m², before the tone curve.
    pub exposure: f32,
    pub sky: Sky,
    /// How the sky looks behind the scene.
    pub backdrop: Backdrop,
    pub sun: Sun,
    /// Skip lights at pixels beyond their reach. Off only to compare.
    pub light_reach: bool,
    /// Shadow maps for the point lights.
    pub shadows: bool,
    /// Texels per cube face edge of each shadow map.
    pub shadow_res: usize,
    /// Bucket edge in pixels.
    pub tile: usize,
    /// Bounce light (point-based colour bleeding).
    pub bounce: bool,
    /// Time per frame for gathering bounce light, in ms; None: all of it.
    pub bounce_budget_ms: Option<f32>,
    pub view: View,
    /// Most bytes the shading cache may hold (`reyes::Renderer`).
    pub cache_budget: usize,
    /// Micropolygon spacing on screen, pixels.
    pub shading_rate_px: f32,
    /// Skip clusters hidden behind what was seen last frame. Off only to
    /// compare.
    pub occlusion: bool,
    /// Shade every other grid point first and the rest only where the
    /// light changes (`reyes::shade_adaptive`). Off only to compare.
    pub adaptive: bool,
    /// When an object moves, draw shadow maps again only where it was and
    /// is, and shade again only what is in its shadows. Off only to
    /// compare: everything is lit again.
    pub incremental_moves: bool,
    /// Sky light taken by each object's own shape, from the sky transfer
    /// precomputed at its surface samples (`transfer.rs`). Off only to
    /// compare.
    pub sky_occlusion: bool,
    /// Directional lights: parallel light like the sun's, but only on the
    /// scene, not the sky.
    pub dir_lights: [DirLight; MAX_DIR_LIGHTS],
    /// While an object is dragged its light goes with it, and the rest of
    /// the scene's light stays, until it is let go (`reyes::Renderer::dragged`).
    /// Off: every step of a drag is lit where it is.
    pub drag_preview: bool,
    /// Bake a let-go object over several frames, each about
    /// `reyes::BAKE_MS`, showing the drag's last picture meanwhile.
    pub bake_over_frames: bool,
    /// While an object is being moved, its micropolygons are this many
    /// times further apart than `shading_rate_px` (1: as when still).
    pub moving_rate: f32,
    /// While an object is being moved, the geometric error its level of
    /// detail may show is this many times `lod_px` (1: as when still).
    pub moving_detail: f32,
}

/// Default `RenderSettings::moving_rate`.
pub const MOVING_RATE: f32 = 3.0;
/// Default `RenderSettings::moving_detail`.
pub const MOVING_DETAIL: f32 = 3.0;

/// Shading cache budget when the machine's plan does not set one.
pub const CACHE_BUDGET: usize = if cfg!(target_pointer_width = "64") { 1 << 30 } else { 16 << 20 };

impl RenderSettings {
    /// Geometric error allowed on screen, pixels: as much as the
    /// micropolygons' spacing, since detail finer than that is not shaded.
    pub fn lod_px(&self) -> f32 {
        self.shading_rate_px.max(1.0)
    }

    pub fn new(sky: Sky) -> Self {
        Self { cull_backfaces: true, wireframe: false, exposure: default_exposure(sky.lux), sky, backdrop: Backdrop::default(), sun: Sun::default(), light_reach: true, shadows: true, shadow_res: crate::shadow::SHADOW_RES, tile: TILE, bounce: true, bounce_budget_ms: None, view: View::Shaded, cache_budget: CACHE_BUDGET, shading_rate_px: SHADING_RATE_PX, occlusion: true, adaptive: true, incremental_moves: true, sky_occlusion: true, dir_lights: [DirLight::default(); MAX_DIR_LIGHTS], drag_preview: true, bake_over_frames: true, moving_rate: MOVING_RATE, moving_detail: MOVING_DETAIL }
    }
}

/// An exposure that keeps sky-lit surfaces in the lower half of the
/// display, so lamps read against them.
pub fn default_exposure(sky_lux: f32) -> f32 {
    // Follows the sky three quarters of the way: noon reads brighter than
    // dusk, and night darker, as the eye sees them. Equal to the old rule
    // (a sky-lit white surface at 1/8 of the display) at 30 lux. It stops
    // following at the end of civil twilight (3 lux): past it the night
    // sky goes dark and the lamps carry the picture.
    let lux = sky_lux.max(3.0);
    std::f32::consts::PI / (8.0 * 30f32.powf(0.25) * lux.powf(0.75))
}

/// Largest radiance a surface sends per lux it receives: white diffuse
/// (1/π) plus the highlight's peak.
const BRDF_PEAK: f32 = (1.0 + SPECULAR) / std::f32::consts::PI;
pub const SPECULAR: f32 = 0.12;
/// Steepest slope of the tone curve.
const TONE_SLOPE: f32 = 1.25;

/// Illuminance below which a light's contribution is dropped, in lux.
///
/// With the sky on: 1/64 of what the sky gives the darkest-facing surface
/// (one lit only by ground bounce), a change of about 1.6%, at or under one
/// 8-bit step in the mid tones. Always also: half an 8-bit step at black for
/// the current exposure, so with the sky off a light reaches as far as it
/// can still be seen in the dark.
pub fn cut_lux(sky: &Sky, exposure: f32) -> f32 {
    let black_step = 0.5 / (255.0 * 12.92); // linear value of half an sRGB step at 0
    let floor = black_step / (BRDF_PEAK * TONE_SLOPE * exposure.max(1e-12));
    let sky_part = if sky.on { sky.lux * GROUND_BOUNCE / 64.0 } else { 0.0 };
    floor.max(sky_part)
}

/// How far, in metres, a light's illuminance stays at or above `cut_lux`.
pub fn light_reach(l: &PointLight, cut: f32) -> f32 {
    (l.candela() / cut.max(1e-12)).sqrt()
}

impl Frame {
    pub fn new(width: usize, height: usize) -> Self {
        let n = width * height;
        Self {
            width,
            height,
            color: vec![0; n],
            depth: vec![0.0; n],
            tri_id: vec![NO_TRI; n],
            setups: Vec::new(),
            stats: Stats::default(),
            vis_stamp: 0,
        }
    }

    pub fn resize(&mut self, width: usize, height: usize) {
        if width != self.width || height != self.height {
            *self = Self::new(width, height);
        }
    }
}

struct ClipVert {
    p: Vec4,
    bary: [f32; 3],
}

fn lerp_vert(a: &ClipVert, b: &ClipVert, t: f32) -> ClipVert {
    ClipVert {
        p: a.p.lerp(b.p, t),
        bary: [
            a.bary[0] + (b.bary[0] - a.bary[0]) * t,
            a.bary[1] + (b.bary[1] - a.bary[1]) * t,
            a.bary[2] + (b.bary[2] - a.bary[2]) * t,
        ],
    }
}

/// Clip a triangle against w >= near. Returns 0, 1 or 2 triangles.
fn clip_near(v: [ClipVert; 3], near: f32, out: &mut Vec<[ClipVert; 3]>) {
    let inside: Vec<bool> = v.iter().map(|c| c.p.w >= near).collect();
    let n_in = inside.iter().filter(|b| **b).count();
    match n_in {
        3 => out.push(v),
        0 => {}
        _ => {
            // Sutherland-Hodgman against one plane.
            let mut poly: Vec<ClipVert> = Vec::with_capacity(4);
            for i in 0..3 {
                let j = (i + 1) % 3;
                let (a, b) = (&v[i], &v[j]);
                if inside[i] {
                    poly.push(ClipVert { p: a.p, bary: a.bary });
                }
                if inside[i] != inside[j] {
                    let t = (near - a.p.w) / (b.p.w - a.p.w);
                    poly.push(lerp_vert(a, b, t));
                }
            }
            for k in 1..poly.len().saturating_sub(1) {
                out.push([
                    ClipVert { p: poly[0].p, bary: poly[0].bary },
                    ClipVert { p: poly[k].p, bary: poly[k].bary },
                    ClipVert { p: poly[k + 1].p, bary: poly[k + 1].bary },
                ]);
            }
        }
    }
}

fn setup_tri(tri: u32, v: &[ClipVert; 3], width: usize, height: usize, cull: bool) -> Option<TriSetup> {
    let mut x = [0f32; 3];
    let mut y = [0f32; 3];
    let mut inv_w = [0f32; 3];
    let (w, h) = (width as f32, height as f32);
    for i in 0..3 {
        let iw = 1.0 / v[i].p.w;
        inv_w[i] = iw;
        // NDC to pixels, y down.
        x[i] = (v[i].p.x * iw * 0.5 + 0.5) * w;
        y[i] = (0.5 - v[i].p.y * iw * 0.5) * h;
    }
    // Signed area in screen space (y down). A triangle wound counter-
    // clockwise as seen from its front (right-handed, the USD and OpenGL
    // convention) has negative area here, so positive area is a back face.
    let area = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]);
    if area == 0.0 || !area.is_finite() {
        return None;
    }
    if cull && area > 0.0 {
        return None;
    }
    let min_x = x.iter().cloned().fold(f32::INFINITY, f32::min).floor().max(0.0);
    let min_y = y.iter().cloned().fold(f32::INFINITY, f32::min).floor().max(0.0);
    let max_x = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max).ceil().min(w - 1.0);
    let max_y = y.iter().cloned().fold(f32::NEG_INFINITY, f32::max).ceil().min(h - 1.0);
    if max_x < min_x || max_y < min_y {
        return None;
    }
    let mut s = TriSetup {
        x,
        y,
        inv_w,
        bary: [v[0].bary, v[1].bary, v[2].bary],
        tri,
        cluster: 0,
        alpha: NO_ALPHA,
        min_x: min_x as u16,
        min_y: min_y as u16,
        max_x: max_x as u16,
        max_y: max_y as u16,
    };
    // Make every triangle clockwise on screen so one edge test serves all.
    if area < 0.0 {
        s.x.swap(1, 2);
        s.y.swap(1, 2);
        s.inv_w.swap(1, 2);
        s.bary.swap(1, 2);
    }
    Some(s)
}

/// The frustum planes (left, right, bottom, top, near) of a view-projection
/// matrix, normalised, as (normal, d) with inside meaning `n.dot(p) + d >= 0`.
/// The projection is infinite, so there is no far plane.
pub fn frustum_planes(vp: Mat4, near: f32) -> [Vec4; 5] {
    let (r0, r1, r3) = (vp.row(0), vp.row(1), vp.row(3));
    let norm = |p: Vec4| p / p.truncate().length().max(1e-20);
    [norm(r3 + r0), norm(r3 - r0), norm(r3 + r1), norm(r3 - r1), norm(r3 - Vec4::new(0.0, 0.0, 0.0, near))]
}

/// Whether a box (its eight corners) is wholly outside one of `planes`,
/// by more than `margin` (coarser levels of detail stray a little outside
/// the finest level's box).
pub fn box_outside(corners: &[Vec3; 8], planes: &[Vec4], margin: f32) -> bool {
    planes.iter().any(|p| corners.iter().all(|c| p.truncate().dot(*c) + p.w < -margin))
}

/// Whether a cluster can be skipped: outside a frustum plane, or (with
/// back-face culling) every face turned away from the eye.
fn cluster_hidden(c: &Cluster, planes: &[Vec4; 5], eye: Vec3, cull_backfaces: bool) -> bool {
    for p in planes {
        if p.truncate().dot(c.centre) + p.w < -c.radius {
            return true;
        }
    }
    if cull_backfaces && c.cone_cos > 0.0 {
        let d = c.centre - eye;
        let dist = d.length();
        if dist > c.radius {
            // The cone of view directions to the sphere is asin(r/d) wide.
            // Every face is back-facing when each view direction is within
            // 90 degrees of every face normal.
            let spread = c.cone_cos.clamp(-1.0, 1.0).acos();
            let limit = std::f32::consts::FRAC_PI_2 - spread - (c.radius / dist).asin();
            if limit > 0.0 && (d / dist).dot(c.cone_axis).clamp(-1.0, 1.0).acos() < limit {
                return true;
            }
        }
    }
    false
}

/// Micropolygon spacing on screen, in pixels, when the machine's plan does
/// not set it (`RenderSettings::shading_rate_px`).
pub const SHADING_RATE_PX: f32 = 2.0;
/// Finest dicing level: a triangle is cut into 2^level parts per edge.
/// Finer than the finest texture tile (`texture::MAX_LEVEL`), which grids
/// past it sample.
pub const MAX_DICE_LEVEL: u8 = 8;
/// Dicing level of a cluster that is not on screen.
pub const NOT_VISIBLE: u8 = u8::MAX;

/// Dicing level for an edge `edge_px` pixels long on screen: the smallest
/// level whose micropolygon edges are at most `SHADING_RATE_PX`, up to
/// `MAX_DICE_LEVEL`. Per cluster, from its longest edge, it is the finest
/// level any of its triangles can need.
pub fn dice_level(edge_px: f32) -> u8 {
    dice_level_at(edge_px, SHADING_RATE_PX)
}

/// `dice_level` for micropolygons of `rate` pixels.
pub fn dice_level_at(edge_px: f32, rate: f32) -> u8 {
    if !(edge_px > rate) {
        return 0;
    }
    ((edge_px / rate).log2().ceil() as i32).clamp(0, MAX_DICE_LEVEL as i32) as u8
}

/// Which clusters `visibility` draws.
#[derive(Clone, Copy, Debug)]
pub enum Part<'a> {
    All,
    /// All but these.
    Without(&'a [std::ops::Range<usize>]),
    /// Only these, with their level of detail allowed this many times the
    /// error.
    Only(&'a [std::ops::Range<usize>], f32),
}

/// Pixels per side of a block of the hierarchical depth buffer. 4 culls
/// twice what 8 does in the close view of the built-in set (a block holds
/// its farthest depth, so a block on an outline holds the background).
pub const HIZ_BLOCK: usize = 4;

/// Work out what each pixel sees. Fills `frame.depth`, `frame.tri_id` and
/// `frame.setups`, and returns each cluster's dicing level (`NOT_VISIBLE`
/// for clusters culled whole).
///
/// Of clusters with levels of detail, only those whose error looks no
/// bigger than `lod_px` pixels and whose replacement's does are drawn
/// (`Cluster::lod_drawn`). `part` picks which clusters are drawn at all.
///
/// Occlusion culling in two passes (as Nanite does): the clusters that
/// owned a pixel last frame (`prev`, one entry per cluster, non-zero when
/// seen) are drawn first; the farthest depth of each
/// 8 x 8 block of what they cover is kept; every other cluster whose
/// bounding sphere is behind that depth in all the blocks it covers is
/// skipped. Whatever the first pass misses, the second draws, so nothing
/// is ever wrongly hidden; a camera that moves only makes the first pass
/// less useful. Without `prev` (the first frame) everything is drawn in
/// one pass.
#[allow(clippy::too_many_arguments)]
pub fn visibility(scene: &Scene, cam: &Camera, cull: bool, wireframe: bool, tile: usize, lod_px: f32, part: Part<'_>, prev: Option<&[u128]>, frame: &mut Frame) -> Vec<u8> {
    // The last frame's setup records go before the new ones are made: with
    // the high-resolution T-rex they are about 1 GB.
    frame.setups = Vec::new();
    let tile = tile.max(8);
    let (width, height) = (frame.width, frame.height);
    let vp = cam.proj * cam.view;
    let t0 = std::time::Instant::now();
    let near = cam.near;
    let planes = frustum_planes(vp, near);
    let eye = cam.eye;
    let lod_scale = crate::scene::perspective_scale(eye, cam.proj.y_axis.y * height as f32 * 0.5, near);

    // Objects first: one wholly outside the frustum (its oriented box)
    // takes none of its clusters along, which matters with many instances.
    let mut runs = scene.loose_clusters();
    runs.extend(scene.objects.iter().filter(|o| !o.clusters.is_empty() && !box_outside(&o.corners(), &planes, (o.hi - o.lo).length() * 0.02)).map(|o| o.clusters.clone()));
    // Clusters that can be seen at all: the level of detail for this view,
    // in the frustum, not all turned away.
    let candidates: Vec<u32> = runs
        .into_par_iter()
        .flat_map_iter(|r| r.start as u32..r.end as u32)
        .filter(|&i| {
            let c = &scene.clusters[i as usize];
            let px = match part {
                Part::All => lod_px,
                Part::Without(r) if r.iter().any(|r| r.contains(&(i as usize))) => return false,
                Part::Without(_) => lod_px,
                Part::Only(r, detail) if r.iter().any(|r| r.contains(&(i as usize))) => lod_px * detail.max(1.0),
                Part::Only(..) => return false,
            };
            // A cut-out cluster (leaves on cards) is seen from both sides.
            c.lod_drawn(px, &lod_scale) && !cluster_hidden(c, &planes, eye, cull && !scene.textures.has_alpha(i as usize))
        })
        .collect();
    let prev = prev.filter(|p| p.len() == scene.clusters.len());
    let (first, second): (Vec<u32>, Vec<u32>) = match prev {
        Some(p) => candidates.iter().partition(|&&i| p[i as usize] != 0),
        None => (candidates, Vec::new()),
    };
    let mut level = vec![NOT_VISIBLE; scene.clusters.len()];
    frame.depth.iter_mut().for_each(|d| *d = 0.0);
    frame.tri_id.iter_mut().for_each(|t| *t = NO_TRI);
    let mut setups: Vec<TriSetup> = Vec::new();
    let mut times = [0f32; 3];

    let first_n = first.len();
    let mut alpha: Vec<AlphaTiles> = Vec::new();
    draw_clusters(scene, &first, vp, near, cull, wireframe, tile, frame, &mut setups, &mut alpha, &mut level, &mut times);
    let mut occluded = 0;
    if !second.is_empty() {
        let th = std::time::Instant::now();
        let hiz = hiz_blocks(&frame.depth, width, height);
        let bw = width.div_ceil(HIZ_BLOCK);
        let visible: Vec<u32> = second.par_iter().copied().filter(|&i| !sphere_hidden(&scene.clusters[i as usize], vp, near, width, height, &hiz, bw)).collect();
        occluded = second.len() - visible.len();
        times[0] += th.elapsed().as_secs_f32() * 1000.0;
        draw_clusters(scene, &visible, vp, near, cull, wireframe, tile, frame, &mut setups, &mut alpha, &mut level, &mut times);
    }

    frame.stats.clusters_in = scene.clusters.len();
    frame.stats.clusters_drawn = level.iter().filter(|l| **l != NOT_VISIBLE).count();
    frame.stats.clusters_first = first_n;
    frame.stats.clusters_occluded = occluded;
    frame.stats.tris_in = scene.drawable_triangles();
    frame.stats.tris_drawn = setups.len();
    frame.stats.setup_ms = times[0] + (t0.elapsed().as_secs_f32() * 1000.0 - times.iter().sum::<f32>()).max(0.0);
    frame.stats.bin_ms = times[1];
    frame.stats.raster_ms = times[2];
    frame.setups = setups;
    level
}

/// The farthest depth (smallest 1/w) of each `HIZ_BLOCK` square of the
/// frame, 0 where any pixel is empty.
pub fn hiz_blocks(depth: &[f32], width: usize, height: usize) -> Vec<f32> {
    let bw = width.div_ceil(HIZ_BLOCK);
    let bh = height.div_ceil(HIZ_BLOCK);
    (0..bw * bh)
        .into_par_iter()
        .map(|b| {
            let (bx, by) = (b % bw, b / bw);
            let (x0, y0) = (bx * HIZ_BLOCK, by * HIZ_BLOCK);
            let (x1, y1) = ((x0 + HIZ_BLOCK).min(width), (y0 + HIZ_BLOCK).min(height));
            let mut far = f32::INFINITY;
            for y in y0..y1 {
                for &d in &depth[y * width + x0..y * width + x1] {
                    far = far.min(d);
                }
            }
            far
        })
        .collect()
}

/// Whether a cluster's bounding sphere is behind what is drawn in every
/// block it covers on screen. A sphere reaching the near plane is never
/// hidden.
fn sphere_hidden(c: &Cluster, vp: Mat4, near: f32, width: usize, height: usize, hiz: &[f32], bw: usize) -> bool {
    // The box round the sphere, projected: its corners bound the sphere on
    // screen.
    let r = c.radius;
    let (mut x0, mut y0, mut x1, mut y1) = (f32::INFINITY, f32::INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for k in 0..8 {
        let corner = c.centre + Vec3::new(if k & 1 == 0 { -r } else { r }, if k & 2 == 0 { -r } else { r }, if k & 4 == 0 { -r } else { r });
        let q = vp * corner.extend(1.0);
        if q.w <= near {
            return false;
        }
        let (sx, sy) = ((q.x / q.w * 0.5 + 0.5) * width as f32, (0.5 - q.y / q.w * 0.5) * height as f32);
        x0 = x0.min(sx);
        y0 = y0.min(sy);
        x1 = x1.max(sx);
        y1 = y1.max(sy);
    }
    // The sphere's nearest point, as 1/w (the depth buffer's measure);
    // w along the view grows one for one with distance.
    let w_centre = (vp * c.centre.extend(1.0)).w;
    let nearest = 1.0 / (w_centre - r).max(near);
    let bh = hiz.len() / bw.max(1);
    let clamp = |v: f32, n: usize| (v.max(0.0) as usize / HIZ_BLOCK).min(n.saturating_sub(1));
    if x1 < 0.0 || y1 < 0.0 || x0 >= width as f32 || y0 >= height as f32 {
        return true;
    }
    let (bx0, bx1, by0, by1) = (clamp(x0, bw), clamp(x1, bw), clamp(y0, bh), clamp(y1, bh));
    for by in by0..=by1 {
        for bx in bx0..=bx1 {
            if hiz[by * bw + bx] <= nearest {
                return false;
            }
        }
    }
    true
}

/// Set up, bin and rasterise clusters `list` into the frame's depth and
/// triangle ids (keeping what is there), adding their records to `setups`
/// and their dicing levels to `level`. `times` gains setup, bin and raster
/// milliseconds.
#[allow(clippy::too_many_arguments)]
fn draw_clusters(scene: &Scene, list: &[u32], vp: Mat4, near: f32, cull: bool, wireframe: bool, tile: usize, frame: &mut Frame, setups: &mut Vec<TriSetup>, alpha: &mut Vec<AlphaTiles>, level: &mut [u8], times: &mut [f32; 3]) {
    let (width, height) = (frame.width, frame.height);
    let t0 = std::time::Instant::now();
    // Project each cluster's vertices into a local array and set up its
    // triangles.
    let parts: Vec<(Vec<TriSetup>, Vec<(u32, u8)>, Vec<AlphaTiles>)> = list
        .par_chunks(64)
        .map(|chunk| {
            let mut out: Vec<TriSetup> = Vec::new();
            let mut levels: Vec<(u32, u8)> = Vec::new();
            let mut tiles: Vec<AlphaTiles> = Vec::new();
            let mut clipped: Vec<[ClipVert; 3]> = Vec::new();
            let mut clip = [Vec4::ZERO; crate::scene::CLUSTER_VERTS];
            for &cluster in chunk {
                let c = &scene.clusters[cluster as usize];
                let cut = scene.textures.has_alpha(cluster as usize);
                let cull = cull && !cut;
                let first_setup = out.len();
                for (k, p) in scene.cluster_pos(c).enumerate() {
                    clip[k] = vp * p.extend(1.0);
                }
                // Longest triangle edge on screen, for the dicing level.
                let mut edge_px = 0f32;
                let to_px = |p: Vec4| Vec2::new((p.x / p.w * 0.5 + 0.5) * width as f32, (0.5 - p.y / p.w * 0.5) * height as f32);
                for tri in c.first_tri..c.first_tri + c.tri_count {
                    let l = scene.tris[tri as usize];
                    let (p0, p1, p2) = (clip[l[0] as usize], clip[l[1] as usize], clip[l[2] as usize]);
                    // Trivial rejects against the frustum in clip space.
                    if (p0.w < near && p1.w < near && p2.w < near)
                        || (p0.x > p0.w && p1.x > p1.w && p2.x > p2.w)
                        || (p0.x < -p0.w && p1.x < -p1.w && p2.x < -p2.w)
                        || (p0.y > p0.w && p1.y > p1.w && p2.y > p2.w)
                        || (p0.y < -p0.w && p1.y < -p1.w && p2.y < -p2.w)
                    {
                        continue;
                    }
                    let v = [
                        ClipVert { p: p0, bary: [1.0, 0.0, 0.0] },
                        ClipVert { p: p1, bary: [0.0, 1.0, 0.0] },
                        ClipVert { p: p2, bary: [0.0, 0.0, 1.0] },
                    ];
                    if p0.w >= near && p1.w >= near && p2.w >= near {
                        let (a, b, cc) = (to_px(p0), to_px(p1), to_px(p2));
                        edge_px = edge_px.max(a.distance(b)).max(b.distance(cc)).max(cc.distance(a));
                        if let Some(mut s) = setup_tri(tri, &v, width, height, cull) {
                            s.cluster = cluster;
                            out.push(s);
                        }
                    } else {
                        // Crosses the near plane: as close as it gets, so
                        // dice it as finely as allowed.
                        edge_px = f32::INFINITY;
                        clipped.clear();
                        clip_near(v, near, &mut clipped);
                        for cv in &clipped {
                            if let Some(mut s) = setup_tri(tri, cv, width, height, cull) {
                                s.cluster = cluster;
                                out.push(s);
                            }
                        }
                    }
                }
                levels.push((cluster, dice_level(edge_px)));
                // Cut out: its alpha tiles, a point about a pixel apart.
                if cut && out.len() > first_setup {
                    let l = (edge_px.clamp(1.0, 1e6).log2().ceil() as u8).max(1);
                    if let Some(b) = scene.textures.alpha_block(cluster as usize, c.tri_count as usize, l) {
                        let k = tiles.len() as u32;
                        tiles.push(AlphaTiles { block: b, first_tri: c.first_tri });
                        for s in &mut out[first_setup..] {
                            s.alpha = k;
                        }
                    }
                }
            }
            (out, levels, tiles)
        })
        .collect();
    let base = setups.len();
    setups.reserve(parts.iter().map(|p| p.0.len()).sum());
    for (mut s, l, t) in parts {
        let k = alpha.len() as u32;
        if k > 0 {
            for x in s.iter_mut().filter(|x| x.alpha != NO_ALPHA) {
                x.alpha += k;
            }
        }
        alpha.extend(t);
        setups.extend(s);
        for (c, lv) in l {
            level[c as usize] = lv;
        }
    }
    let alpha: &[AlphaTiles] = alpha;
    let new = &setups[base..];
    let t1 = std::time::Instant::now();

    // Bin into buckets. Each chunk of records writes its own per-bucket
    // lists; the lists are then joined per bucket.
    let tiles_x = width.div_ceil(tile);
    let tiles_y = height.div_ceil(tile);
    let n_tiles = tiles_x * tiles_y;
    let chunk_lists: Vec<Vec<Vec<u32>>> = new
        .par_chunks(8192)
        .enumerate()
        .map(|(ci, chunk)| {
            let mut lists: Vec<Vec<u32>> = vec![Vec::new(); n_tiles];
            let first = (base + ci * 8192) as u32;
            for (k, s) in chunk.iter().enumerate() {
                let id = first + k as u32;
                for ty in s.min_y as usize / tile..=s.max_y as usize / tile {
                    for tx in s.min_x as usize / tile..=s.max_x as usize / tile {
                        lists[ty * tiles_x + tx].push(id);
                    }
                }
            }
            lists
        })
        .collect();
    let tile_lists: Vec<Vec<u32>> = (0..n_tiles)
        .into_par_iter()
        .map(|t| {
            let mut v = Vec::new();
            for cl in &chunk_lists {
                v.extend_from_slice(&cl[t]);
            }
            v
        })
        .collect();
    let t2 = std::time::Instant::now();

    // Hide. Each bucket is rasterised into a scratch of its own, starting
    // from what the frame holds, then copied back, so the buffers are
    // written without sharing.
    let all: &[TriSetup] = setups;
    let (fdepth, fid) = (&frame.depth, &frame.tri_id);
    let tile_results: Vec<Option<(Vec<f32>, Vec<u32>)>> = (0..n_tiles)
        .into_par_iter()
        .map(|t| {
            if tile_lists[t].is_empty() {
                return None;
            }
            let (x0, y0) = ((t % tiles_x) * tile, (t / tiles_x) * tile);
            let (x1, y1) = ((x0 + tile).min(width), (y0 + tile).min(height));
            let (tw, th) = (x1 - x0, y1 - y0);
            let mut depth = Vec::with_capacity(tw * th);
            let mut id = Vec::with_capacity(tw * th);
            for y in y0..y1 {
                depth.extend_from_slice(&fdepth[y * width + x0..y * width + x1]);
                id.extend_from_slice(&fid[y * width + x0..y * width + x1]);
            }
            for &sid in &tile_lists[t] {
                raster_tri(&all[sid as usize], sid, x0, y0, x1, y1, tw, &mut depth, &mut id, wireframe, alpha);
            }
            Some((depth, id))
        })
        .collect();
    for (t, r) in tile_results.into_iter().enumerate() {
        let Some((depth, id)) = r else { continue };
        let (x0, y0) = ((t % tiles_x) * tile, (t / tiles_x) * tile);
        let (x1, y1) = ((x0 + tile).min(width), (y0 + tile).min(height));
        let tw = x1 - x0;
        for (row, y) in (y0..y1).enumerate() {
            let dst = y * width + x0;
            frame.depth[dst..dst + tw].copy_from_slice(&depth[row * tw..row * tw + tw]);
            frame.tri_id[dst..dst + tw].copy_from_slice(&id[row * tw..row * tw + tw]);
        }
    }
    let t3 = std::time::Instant::now();
    times[0] += (t1 - t0).as_secs_f32() * 1000.0;
    times[1] += (t2 - t1).as_secs_f32() * 1000.0;
    times[2] += (t3 - t2).as_secs_f32() * 1000.0;
}

/// The direction of the view ray through pixel (x, y), for the sky behind
/// everything.
pub fn sky_dir(cam: &Camera, width: usize, height: usize, x: usize, y: usize, inv_view: &Mat4) -> Vec3 {
    let ndc = Vec2::new((x as f32 + 0.5) / width as f32 * 2.0 - 1.0, 1.0 - (y as f32 + 0.5) / height as f32 * 2.0);
    inv_view.transform_vector3(Vec3::new(ndc.x / cam.proj.x_axis.x, ndc.y / cam.proj.y_axis.y, -1.0)).normalize_or(Vec3::Y)
}

#[inline]
fn raster_tri(
    s: &TriSetup,
    sid: u32,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    tw: usize,
    depth: &mut [f32],
    id: &mut [u32],
    wireframe: bool,
    alpha: &[AlphaTiles],
) {
    let cut = (s.alpha != NO_ALPHA).then(|| &alpha[s.alpha as usize]);
    let bx0 = (s.min_x as usize).max(x0);
    let bx1 = (s.max_x as usize + 1).min(x1);
    let by0 = (s.min_y as usize).max(y0);
    let by1 = (s.max_y as usize + 1).min(y1);
    if bx0 >= bx1 || by0 >= by1 {
        return;
    }
    // Edge functions, clockwise on screen. e_i(p) = (x_j - x_i)(p.y - y_i) - (y_j - y_i)(p.x - x_i)
    let (ax, ay, bx, by, cx, cy) = (s.x[0], s.y[0], s.x[1], s.y[1], s.x[2], s.y[2]);
    let area = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay);
    if area <= 0.0 {
        return;
    }
    let inv_area = 1.0 / area;
    // Steps per pixel in x and y for each edge.
    let (a0, b0) = (-(cy - by), cx - bx); // edge bc, weight for vertex a
    let (a1, b1) = (-(ay - cy), ax - cx); // edge ca, weight for vertex b
    let (a2, b2) = (-(by - ay), bx - ax); // edge ab, weight for vertex c
    let px = bx0 as f32 + 0.5;
    let py = by0 as f32 + 0.5;
    let mut e0_row = (cx - bx) * (py - by) - (cy - by) * (px - bx);
    let mut e1_row = (ax - cx) * (py - cy) - (ay - cy) * (px - cx);
    let mut e2_row = (bx - ax) * (py - ay) - (by - ay) * (px - ax);
    // 1/w is affine in screen space.
    let dz_dx = (s.inv_w[0] * a0 + s.inv_w[1] * a1 + s.inv_w[2] * a2) * inv_area;
    let dz_dy = (s.inv_w[0] * b0 + s.inv_w[1] * b1 + s.inv_w[2] * b2) * inv_area;
    let mut z_row = (s.inv_w[0] * e0_row + s.inv_w[1] * e1_row + s.inv_w[2] * e2_row) * inv_area;
    // Wireframe: a pixel is on an edge when one edge function is within a
    // pixel's step of zero.
    let edge_w = [
        (a0 * a0 + b0 * b0).sqrt() * 0.75,
        (a1 * a1 + b1 * b1).sqrt() * 0.75,
        (a2 * a2 + b2 * b2).sqrt() * 0.75,
    ];
    for y in by0..by1 {
        let mut e0 = e0_row;
        let mut e1 = e1_row;
        let mut e2 = e2_row;
        let mut z = z_row;
        let row = (y - y0) * tw;
        for x in bx0..bx1 {
            if e0 >= 0.0 && e1 >= 0.0 && e2 >= 0.0 {
                let inside = !wireframe || e0 < edge_w[0] || e1 < edge_w[1] || e2 < edge_w[2];
                let i = row + (x - x0);
                if inside && z > depth[i] && cut.is_none_or(|a| a.solid_at(s, e0, e1, e2)) {
                    depth[i] = z;
                    id[i] = sid;
                }
            }
            e0 += a0;
            e1 += a1;
            e2 += a2;
            z += dz_dx;
        }
        e0_row += b0;
        e1_row += b1;
        e2_row += b2;
        z_row += dz_dy;
    }
}

/// No alpha tiles: an opaque triangle.
pub const NO_ALPHA: u32 = u32::MAX;

/// A cut-out cluster's alpha tiles for a frame, and its first triangle.
pub struct AlphaTiles {
    pub block: std::sync::Arc<crate::texture::Block>,
    pub first_tri: u32,
}

impl AlphaTiles {
    /// Whether triangle `s` is solid where its edge functions are (e0,
    /// e1, e2): the perspective-correct weights, through its setup's
    /// barycentrics (a triangle cut by the near plane) to the stored
    /// triangle's, then its nearest alpha point.
    #[inline]
    pub fn solid_at(&self, s: &TriSetup, e0: f32, e1: f32, e2: f32) -> bool {
        let (l0, l1, l2) = (e0 * s.inv_w[0], e1 * s.inv_w[1], e2 * s.inv_w[2]);
        let sum = l0 + l1 + l2;
        if sum.abs() < 1e-30 {
            return true;
        }
        let inv = 1.0 / sum;
        let (l0, l1, l2) = (l0 * inv, l1 * inv, l2 * inv);
        let b1 = l0 * s.bary[0][1] + l1 * s.bary[1][1] + l2 * s.bary[2][1];
        let b2 = l0 * s.bary[0][2] + l1 * s.bary[1][2] + l2 * s.bary[2][2];
        self.block.solid((s.tri - self.first_tri) as usize, b1, b2)
    }
}

/// Level of alpha tiles shadow maps use: their texels are coarse, and a
/// leaf's shadow needs no finer outline.
pub const SHADOW_ALPHA_LEVEL: u8 = 4;

/// Rasterise a cut-out triangle into a map that keeps the largest value:
/// corners `p` (x, y in texels of `buf`, which is `width` wide, and the
/// value, affine on screen), `inv_w` their 1/w for perspective-correct
/// weights (1 for an orthographic map), `bary` their weights in the stored
/// triangle `tri`, only texels in columns `xs` and rows `ys` (inclusive)
/// where `tiles` say it is solid. Whether it touched any texel.
#[allow(clippy::too_many_arguments)]
pub fn fill_max_cut(buf: &mut [f32], width: usize, xs: (usize, usize), ys: (usize, usize), p: [(f32, f32, f32); 3], inv_w: [f32; 3], bary: [[f32; 3]; 3], tiles: &AlphaTiles, tri: u32) -> bool {
    let [(ax, ay, az), (bx, by, bz), (cx, cy, cz)] = p;
    let area = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay);
    if area == 0.0 || !area.is_finite() {
        return false;
    }
    let inv_area = 1.0 / area;
    let t = (tri - tiles.first_tri) as usize;
    let mut any = false;
    for y in ys.0..=ys.1 {
        let py = y as f32 + 0.5;
        for x in xs.0..=xs.1 {
            let px = x as f32 + 0.5;
            // Weights of a, b and c, all positive inside whichever way
            // the triangle turns.
            let w0 = ((bx - px) * (cy - py) - (cx - px) * (by - py)) * inv_area;
            let w1 = ((cx - px) * (ay - py) - (ax - px) * (cy - py)) * inv_area;
            let w2 = 1.0 - w0 - w1;
            if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                continue;
            }
            let z = w0 * az + w1 * bz + w2 * cz;
            let i = y * width + x;
            if z <= buf[i] {
                continue;
            }
            let (l0, l1, l2) = (w0 * inv_w[0], w1 * inv_w[1], w2 * inv_w[2]);
            let sum = l0 + l1 + l2;
            let (l0, l1, l2) = if sum.abs() > 1e-30 { (l0 / sum, l1 / sum, l2 / sum) } else { (w0, w1, w2) };
            let b1 = l0 * bary[0][1] + l1 * bary[1][1] + l2 * bary[2][1];
            let b2 = l0 * bary[0][2] + l1 * bary[1][2] + l2 * bary[2][2];
            if tiles.block.solid(t, b1, b2) {
                buf[i] = z;
                any = true;
            }
        }
    }
    any
}

/// Perspective-correct barycentric weights of a pixel centre.
#[inline]
pub fn barycentrics(s: &TriSetup, px: f32, py: f32) -> [f32; 3] {
    let (ax, ay, bx, by, cx, cy) = (s.x[0], s.y[0], s.x[1], s.y[1], s.x[2], s.y[2]);
    let area = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay);
    let e0 = (cx - bx) * (py - by) - (cy - by) * (px - bx);
    let e1 = (ax - cx) * (py - cy) - (ay - cy) * (px - cx);
    let e2 = (bx - ax) * (py - ay) - (by - ay) * (px - ax);
    let inv = 1.0 / area;
    let l0 = e0 * inv * s.inv_w[0];
    let l1 = e1 * inv * s.inv_w[1];
    let l2 = e2 * inv * s.inv_w[2];
    let sum = l0 + l1 + l2;
    if sum.abs() < 1e-20 {
        return [1.0, 0.0, 0.0];
    }
    [l0 / sum, l1 / sum, l2 / sum]
}

/// World position under a pixel, from the visibility buffer.
pub fn unproject(frame: &Frame, scene: &Scene, x: usize, y: usize) -> Option<(Vec3, Vec3)> {
    let id = *frame.tri_id.get(y * frame.width + x)?;
    if id == NO_TRI {
        return None;
    }
    let s = &frame.setups[id as usize];
    let b = barycentrics(s, x as f32 + 0.5, y as f32 + 0.5);
    let w0 = b[0] * s.bary[0][0] + b[1] * s.bary[1][0] + b[2] * s.bary[2][0];
    let w1 = b[0] * s.bary[0][1] + b[1] * s.bary[1][1] + b[2] * s.bary[2][1];
    let w2 = b[0] * s.bary[0][2] + b[1] * s.bary[1][2] + b[2] * s.bary[2][2];
    let c = &scene.clusters[s.cluster as usize];
    let [i0, i1, i2] = scene.tri(c, s.tri);
    let p = scene.pos(c, i0) * w0 + scene.pos(c, i1) * w1 + scene.pos(c, i2) * w2;
    let n = (scene.normal(c, i0) * w0 + scene.normal(c, i1) * w1 + scene.normal(c, i2) * w2).normalize_or(Vec3::Y);
    Some((p, n))
}

/// A world point on screen: pixel x, y and 1/w. None when behind the near
/// plane.
pub fn project(cam: &Camera, width: usize, height: usize, p: Vec3) -> Option<(f32, f32, f32)> {
    let c = cam.proj * cam.view * p.extend(1.0);
    if c.w <= cam.near {
        return None;
    }
    let iw = 1.0 / c.w;
    Some(((c.x * iw * 0.5 + 0.5) * width as f32, (0.5 - c.y * iw * 0.5) * height as f32, iw))
}

/// The world-space ray through a pixel centre: (origin at the eye, unit
/// direction).
pub fn pixel_ray(cam: &Camera, width: usize, height: usize, x: f32, y: f32) -> (Vec3, Vec3) {
    let ndc = Vec2::new(x / width as f32 * 2.0 - 1.0, 1.0 - y / height as f32 * 2.0);
    // The direction in view space comes straight from the projection's
    // focal terms; going through the inverse of the whole matrix loses
    // precision with a small near plane.
    let d_view = Vec3::new(ndc.x / cam.proj.x_axis.x, ndc.y / cam.proj.y_axis.y, -1.0);
    let d = cam.view.inverse().transform_vector3(d_view);
    (cam.eye, d.normalize_or(Vec3::NEG_Z))
}

/// Draw a one-pixel line between two world points, without depth test,
/// dashed when `dashed`.
pub fn draw_line(frame: &mut Frame, cam: &Camera, a: Vec3, b: Vec3, color: Vec3, dashed: bool) {
    let (Some(pa), Some(pb)) = (project(cam, frame.width, frame.height, a), project(cam, frame.width, frame.height, b)) else { return };
    let packed = pack(color);
    let steps = (pb.0 - pa.0).abs().max((pb.1 - pa.1).abs()).ceil().min(8192.0) as usize;
    for i in 0..=steps {
        if dashed && (i / 4) % 2 == 1 {
            continue;
        }
        let t = if steps == 0 { 0.0 } else { i as f32 / steps as f32 };
        let x = (pa.0 + (pb.0 - pa.0) * t) as i32;
        let y = (pa.1 + (pb.1 - pa.1) * t) as i32;
        if x >= 0 && y >= 0 && (x as usize) < frame.width && (y as usize) < frame.height {
            frame.color[y as usize * frame.width + x as usize] = packed;
        }
    }
}

/// Draw a filled disc into the colour buffer, depth-tested at its centre.
pub fn draw_disc(frame: &mut Frame, cam: &Camera, centre: Vec3, radius_px: f32, color: Vec3, ring: bool) {
    let Some((sx, sy, iw)) = project(cam, frame.width, frame.height, centre) else { return };
    let (cx, cy) = (sx as i32, sy as i32);
    let occluded = if cx >= 0 && cy >= 0 && (cx as usize) < frame.width && (cy as usize) < frame.height {
        frame.depth[cy as usize * frame.width + cx as usize] > iw
    } else {
        false
    };
    let col = if occluded { color * 0.35 } else { color };
    let packed = pack(col);
    let r = radius_px.ceil() as i32;
    for dy in -r..=r {
        for dx in -r..=r {
            let d2 = (dx * dx + dy * dy) as f32;
            if d2 > radius_px * radius_px {
                continue;
            }
            if ring && d2 < (radius_px - 1.5) * (radius_px - 1.5) {
                continue;
            }
            let (x, y) = (cx + dx, cy + dy);
            if x >= 0 && y >= 0 && (x as usize) < frame.width && (y as usize) < frame.height {
                frame.color[y as usize * frame.width + x as usize] = packed;
            }
        }
    }
}

#[inline]
pub fn tonemap(c: Vec3) -> Vec3 {
    // Simple filmic-ish curve, keeps highlights from clipping to white.
    let c = c.max(Vec3::ZERO);
    c / (Vec3::ONE + c) * TONE_SLOPE
}

/// Colour for a count of lights: dark grey for none, then blue, green,
/// yellow, red, white.
pub fn heat(k: u32) -> Vec3 {
    const RAMP: [Vec3; 6] = [
        Vec3::new(0.012, 0.012, 0.014),
        Vec3::new(0.02, 0.08, 0.5),
        Vec3::new(0.03, 0.45, 0.08),
        Vec3::new(0.7, 0.6, 0.02),
        Vec3::new(0.8, 0.08, 0.02),
        Vec3::new(0.9, 0.9, 0.9),
    ];
    RAMP[(k as usize).min(RAMP.len() - 1)]
}

#[inline]
pub fn pack(c: Vec3) -> u32 {
    let to8 = |v: f32| -> u32 {
        let v = v.clamp(0.0, 1.0);
        // sRGB transfer
        let s = if v <= 0.0031308 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 };
        (s * 255.0 + 0.5) as u32
    };
    (to8(c.x) << 16) | (to8(c.y) << 8) | to8(c.z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sun_at_1000_is_full_white_and_0_is_nothing() {
        for exposure in [0.0001, 0.01, 1.0] {
            let sun = Sun { dir: Vec3::Y, color: Vec3::ONE, brightness: SUN_MAX };
            // A white surface facing it.
            let shown = tonemap(sun.lux(exposure) / std::f32::consts::PI * exposure);
            assert!((shown - Vec3::ONE).abs().max_element() < 1e-4, "{shown}");
            assert_eq!(Sun { brightness: 0.0, ..sun }.lux(exposure), Vec3::ZERO);
            assert_eq!(Sun { dir: Vec3::NEG_Y, ..sun }.lux(exposure), Vec3::ZERO, "below the horizon");
        }
    }

    #[test]
    fn lighting_contrast_shapes_the_sky_light_only() {
        let mut sky = Sky::new(100.0, true);
        let at = |sky: &Sky, n: Vec3| sky.irradiance_from(n, false).y;
        sky.contrast = 0.0;
        assert!((at(&sky, Vec3::Y) - at(&sky, Vec3::NEG_Y)).abs() < 1e-3, "flat: every facing alike");
        sky.contrast = 0.5;
        assert!((at(&sky, Vec3::X) / at(&sky, Vec3::Y) - 0.5).abs() < 1e-3, "default: sideways gets half");
        sky.contrast = 1.0;
        assert!(at(&sky, Vec3::X) / at(&sky, Vec3::Y) < 0.1, "overhead: sideways gets little");
        // The backdrop does not move with it.
        let look = Backdrop::default();
        let sun = Sun::default();
        let a = backdrop_radiance(&Sky { contrast: 0.0, ..sky }, &look, &sun, 0.01, Vec3::new(0.0, 0.5, -1.0).normalize());
        let b = backdrop_radiance(&Sky { contrast: 1.0, ..sky }, &look, &sun, 0.01, Vec3::new(0.0, 0.5, -1.0).normalize());
        assert_eq!(a, b);
    }
    use crate::reyes::render;

    fn quad_scene() -> Scene {
        let mut s = Scene::new();
        let p = [Vec3::new(-1.0, -1.0, 0.0), Vec3::new(1.0, -1.0, 0.0), Vec3::new(1.0, 1.0, 0.0), Vec3::new(-1.0, 1.0, 0.0)];
        s.push_mesh(&p, &[], &[Vec3::splat(0.8)], &[0, 1, 2, 0, 2, 3]);
        s
    }

    fn settings() -> RenderSettings {
        RenderSettings { exposure: 0.05, ..RenderSettings::new(Sky::new(100.0, false)) }
    }

    fn camera(w: usize, h: usize) -> Camera {
        let cam = crate::camera::OrbitCamera { target: Vec3::ZERO, distance: 3.0, yaw: 0.0, pitch: 0.0, fov_y: 45f32.to_radians(), scene_radius: 1.5 };
        Camera { view: cam.view(), proj: cam.proj(w as f32 / h as f32), eye: cam.eye(), near: cam.near() }
    }

    #[test]
    fn quad_covers_centre_and_not_corner() {
        let scene = quad_scene();
        let (w, h) = (320, 200);
        let mut frame = Frame::new(w, h);
        let settings = settings();
        let lights = [PointLight { position: Vec3::new(0.0, 0.0, 3.0), color: Vec3::ONE, lumens: 100.0 }];
        render(&scene, &camera(w, h), &lights, &settings, &mut frame);
        assert_eq!(frame.stats.tris_drawn, 2);
        assert_ne!(frame.tri_id[h / 2 * w + w / 2], NO_TRI);
        assert_eq!(frame.tri_id[0], NO_TRI);
        let (p, n) = unproject(&frame, &scene, w / 2, h / 2).unwrap();
        assert!(p.length() < 0.02, "centre pixel unprojects to the origin, got {p}");
        assert!((n.z.abs() - 1.0).abs() < 1e-3);
    }

    #[test]
    fn triangle_behind_camera_is_clipped_not_drawn_wrong() {
        let mut scene = Scene::new();
        // One triangle crossing the camera plane: two vertices in front, one behind.
        let p = [Vec3::new(-1.0, -1.0, 0.0), Vec3::new(1.0, -1.0, 0.0), Vec3::new(0.0, 1.0, 10.0)];
        scene.push_mesh(&p, &[], &[], &[0, 1, 2]);
        let (w, h) = (160, 100);
        let mut frame = Frame::new(w, h);
        // Either side: this is about clipping.
        let settings = RenderSettings { cull_backfaces: false, ..settings() };
        render(&scene, &camera(w, h), &[], &settings, &mut frame);
        // Clipping yields one or two pieces, and every drawn pixel has finite depth.
        assert!(frame.stats.tris_drawn >= 1 && frame.stats.tris_drawn <= 2);
        assert!(frame.depth.iter().all(|d| d.is_finite()));
    }

    #[test]
    fn lights_fall_off_with_distance_squared() {
        let scene = quad_scene();
        let (w, h) = (160, 100);
        let settings = settings();
        let centre = h / 2 * w + w / 2;
        let mut dark = Frame::new(w, h);
        render(&scene, &camera(w, h), &[], &settings, &mut dark);
        assert_ne!(dark.tri_id[centre], NO_TRI);
        assert_eq!(dark.color[centre], 0, "no lights, no sky: the quad is black");
        let mut lit = Frame::new(w, h);
        let near_light = [PointLight { position: Vec3::new(0.0, 0.0, 0.5), color: Vec3::ONE, lumens: 50.0 }];
        render(&scene, &camera(w, h), &near_light, &settings, &mut lit);
        let far_light = [PointLight { position: Vec3::new(0.0, 0.0, 2.0), color: Vec3::ONE, lumens: 50.0 }];
        let mut dim = Frame::new(w, h);
        render(&scene, &camera(w, h), &far_light, &settings, &mut dim);
        let g = |c: u32| (c >> 8) & 0xff;
        assert!(g(lit.color[centre]) > g(dim.color[centre]), "a closer light is brighter (1/d²)");
        assert!(g(dim.color[centre]) > 0);
    }

    #[test]
    fn clusters_outside_the_view_are_skipped() {
        let mut scene = Scene::new();
        let (p, idx) = crate::scene::tests::uv_sphere(64);
        // One sphere in front of the camera, one far behind it.
        scene.push_mesh(&p, &[], &[], &idx);
        let behind: Vec<Vec3> = p.iter().map(|v| *v * 0.5 + Vec3::new(0.0, 0.0, 20.0)).collect();
        scene.push_mesh(&behind, &[], &[], &idx);
        let (w, h) = (200, 150);
        let mut frame = Frame::new(w, h);
        let settings = RenderSettings { cull_backfaces: false, ..settings() };
        render(&scene, &camera(w, h), &[], &settings, &mut frame);
        assert!(frame.stats.clusters_drawn * 2 <= frame.stats.clusters_in + 1, "{:?}", frame.stats);
        assert!(frame.stats.clusters_drawn > 0);
        // Back-face culling drops roughly the far half of the front sphere too.
        let before = frame.stats.clusters_drawn;
        let settings = RenderSettings { cull_backfaces: true, ..settings };
        let mut culled = Frame::new(w, h);
        render(&scene, &camera(w, h), &[], &settings, &mut culled);
        assert!(culled.stats.clusters_drawn < before, "{} vs {}", culled.stats.clusters_drawn, before);
        // And the picture is the same: no visible pixel is lost.
        let covered = |f: &Frame| f.tri_id.iter().filter(|t| **t != NO_TRI).count();
        assert_eq!(covered(&frame), covered(&culled));
    }

    #[test]
    fn pixel_ray_goes_through_projected_point() {
        let (w, h) = (320, 200);
        let cam = camera(w, h);
        let p = Vec3::new(0.3, -0.2, 0.4);
        let (x, y, _) = project(&cam, w, h, p).unwrap();
        let (o, d) = pixel_ray(&cam, w, h, x, y);
        let closest = o + d * (p - o).dot(d);
        assert!(closest.distance(p) < 1e-3, "{closest} vs {p}");
    }

    #[test]
    fn the_sky_lights_near_and_far_surfaces_alike() {
        // A small card 3 m away and a huge one 5 km away, both facing the
        // camera, same albedo: lit by the sky only they read the same.
        let mut scene = Scene::new();
        let card = |c: Vec3, half: f32| [c + Vec3::new(-half, -half, 0.0), c + Vec3::new(half, -half, 0.0), c + Vec3::new(half, half, 0.0), c + Vec3::new(-half, half, 0.0)];
        scene.push_mesh(&card(Vec3::new(-0.3, 0.0, 0.0), 0.2), &[], &[Vec3::splat(0.5)], &[0, 1, 2, 0, 2, 3]);
        scene.push_mesh(&card(Vec3::new(1500.0, 0.0, -5000.0), 900.0), &[], &[Vec3::splat(0.5)], &[0, 1, 2, 0, 2, 3]);
        let (w, h) = (200, 100);
        let mut frame = Frame::new(w, h);
        let settings = RenderSettings::new(Sky::new(500.0, true));
        render(&scene, &camera(w, h), &[], &settings, &mut frame);
        let near = frame.color[h / 2 * w + w / 2 - 10];
        let far = frame.color[h / 2 * w + w / 2 + 30];
        assert_ne!(frame.tri_id[h / 2 * w + w / 2 - 10], frame.tri_id[h / 2 * w + w / 2 + 30]);
        assert_eq!(near, far);
        assert_ne!(near, 0);
    }

    #[test]
    fn reach_is_where_illuminance_meets_the_cut() {
        let sky = Sky::new(30.0, true);
        let cut = cut_lux(&sky, default_exposure(sky.lux));
        for lm in [12.0, 400.0, 12_000.0, crate::scene::MAX_LUMENS] {
            let l = PointLight { position: Vec3::ZERO, color: Vec3::ONE, lumens: lm };
            let r = light_reach(&l, cut);
            assert!((l.lux_at(r) / cut - 1.0).abs() < 1e-3);
        }
        // Sky off: only the black-level floor, so lights reach further.
        let dark = Sky { on: false, ..sky };
        assert!(cut_lux(&dark, default_exposure(sky.lux)) < cut);
    }

    #[test]
    fn lights_are_skipped_on_clusters_out_of_reach() {
        let mut scene = Scene::new();
        let (p, idx) = crate::scene::tests::uv_sphere(32);
        scene.push_mesh(&p, &[], &[], &idx);
        // Off to the side so the near sphere does not hide it.
        let far: Vec<Vec3> = p.iter().map(|v| *v * 50.0 + Vec3::new(180.0, 0.0, -400.0)).collect();
        scene.push_mesh(&far, &[], &[], &idx);
        let lights = [PointLight { position: Vec3::new(0.0, 1.5, 1.5), color: Vec3::ONE, lumens: 400.0 }];
        let (w, h) = (200, 150);
        let settings = RenderSettings::new(Sky::new(30.0, true));
        let mut culled = Frame::new(w, h);
        render(&scene, &camera(w, h), &lights, &settings, &mut culled);
        let mut full = Frame::new(w, h);
        render(&scene, &camera(w, h), &lights, &RenderSettings { light_reach: false, ..RenderSettings::new(Sky::new(30.0, true)) }, &mut full);
        assert!(culled.stats.light_evals < full.stats.light_evals, "{:?} vs {:?}", culled.stats, full.stats);
        assert_eq!(full.stats.light_evals, full.stats.light_evals_all);
        // The picture does not change by more than one 8-bit step anywhere.
        let mut worst = 0;
        for (a, b) in culled.color.iter().zip(&full.color) {
            for sh in [0, 8, 16] {
                worst = worst.max((((a >> sh) & 0xff) as i32 - ((b >> sh) & 0xff) as i32).abs());
            }
        }
        assert!(worst <= 1, "worst channel difference {worst}");
    }
}
