//! Visual debug views (`--visual-debug`): each stage of the renderer drawn
//! on the frame, for screenshots that explain how it works.
//!
//! Built on 64-bit targets only. These views are for illustration on a
//! capable machine; they read the renderer's caches, the shadow maps and the
//! bounce cloud as they are, may read texture tiles, and allocate freely.
//! Nothing here runs unless a view is chosen, and nothing in the renderer
//! depends on it.

use glam::{Mat4, Vec3};
use rayon::prelude::*;

use crate::bounce::{Bounce, MICRO};
use crate::raster::{self, Camera, Frame, RenderSettings, TriSetup, NO_TRI, NOT_VISIBLE};
use crate::reyes::{Part, Peek};
use crate::scene::{PointLight, Scene};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DebugView {
    Final,
    Clusters,
    Triangles,
    Depth,
    Normals,
    Buckets,
    Dice,
    Micropolygons,
    Reshaded,
    Albedo,
    Tiles,
    Sky,
    Direct,
    Highlight,
    Bounce,
    LightCount,
    Reach,
    Shadow,
    ShadowMap,
    Surfels,
    Receivers,
    MicroRaster,
}

pub const ALL: [DebugView; 22] = [
    DebugView::Final,
    DebugView::Clusters,
    DebugView::Triangles,
    DebugView::Depth,
    DebugView::Normals,
    DebugView::Buckets,
    DebugView::Dice,
    DebugView::Micropolygons,
    DebugView::Reshaded,
    DebugView::Albedo,
    DebugView::Tiles,
    DebugView::Sky,
    DebugView::Direct,
    DebugView::Highlight,
    DebugView::Bounce,
    DebugView::LightCount,
    DebugView::Reach,
    DebugView::Shadow,
    DebugView::ShadowMap,
    DebugView::Surfels,
    DebugView::Receivers,
    DebugView::MicroRaster,
];

impl DebugView {
    pub fn name(self) -> &'static str {
        match self {
            DebugView::Final => "final",
            DebugView::Clusters => "clusters",
            DebugView::Triangles => "triangles",
            DebugView::Depth => "depth",
            DebugView::Normals => "normals",
            DebugView::Buckets => "buckets",
            DebugView::Dice => "dice",
            DebugView::Micropolygons => "micropolygons",
            DebugView::Reshaded => "reshaded",
            DebugView::Albedo => "albedo",
            DebugView::Tiles => "tiles",
            DebugView::Sky => "sky",
            DebugView::Direct => "direct",
            DebugView::Highlight => "highlight",
            DebugView::Bounce => "bounce",
            DebugView::LightCount => "light-count",
            DebugView::Reach => "reach",
            DebugView::Shadow => "shadow",
            DebugView::ShadowMap => "shadow-map",
            DebugView::Surfels => "surfels",
            DebugView::Receivers => "receivers",
            DebugView::MicroRaster => "micro-raster",
        }
    }

    pub fn from_name(s: &str) -> Option<DebugView> {
        ALL.iter().copied().find(|v| v.name() == s)
    }

    pub fn step(self, by: i32) -> DebugView {
        let i = ALL.iter().position(|v| *v == self).unwrap_or(0) as i32;
        let n = ALL.len() as i32;
        ALL[((i + by) % n + n) as usize % ALL.len()]
    }

    /// The caption drawn on the frame: a title and what the colours mean.
    fn caption(self) -> (&'static str, &'static [&'static str]) {
        match self {
            DebugView::Final => ("FINAL", &[]),
            DebugView::Clusters => ("CLUSTERS", &["ONE COLOUR PER CLUSTER: AT MOST 64 VERTICES, 124 TRIANGLES", "THE UNIT OF CULLING, LIGHT REACH, SHADING CACHE AND TEXTURE STREAMING"]),
            DebugView::Triangles => ("TRIANGLES (VISIBILITY BUFFER)", &["ONE COLOUR PER TRIANGLE ID: WHAT EACH PIXEL SEES", "DEPTH AND TRIANGLE ID ARE ALL VISIBILITY STORES; NOTHING IS SHADED THERE"]),
            DebugView::Depth => ("DEPTH", &["DISTANCE FROM THE EYE, LOG SCALE, OVER THE RANGE IN VIEW"]),
            DebugView::Normals => ("NORMALS", &["WORLD-SPACE NORMAL, X Y Z AS RED GREEN BLUE"]),
            DebugView::Buckets => ("BUCKETS", &["EACH BUCKET IS HIDDEN BY ONE THREAD, WITH NO LOCKS", "NUMBER: TRIANGLES BINNED INTO IT; BLUE FEW, RED MANY"]),
            DebugView::Dice => ("DICING LEVEL", &["PER CLUSTER, FROM ITS LONGEST EDGE ON SCREEN", "LEVEL L CUTS A TRIANGLE INTO 2^L PARTS PER EDGE"]),
            DebugView::Micropolygons => ("MICROPOLYGONS", &["THE GRID EACH TRIANGLE WAS SHADED ON, IN WORLD SPACE", "WHITE: TRIANGLE EDGES; COLOUR: MICROPOLYGON EDGES BY LEVEL"]),
            DebugView::Reshaded => ("SHADING CACHE", &["RED: SHADED IN THE LAST FRAMES; GREEN: BOUNCE GATHERED", "GREY: KEPT FROM EARLIER FRAMES. DRAG A LIGHT TO SEE ITS REACH REDONE"]),
            DebugView::Albedo => ("ALBEDO", &["BASE COLOUR: VERTEX COLOURS, OR TEXTURE TILES AT THE KEPT LEVEL"]),
            DebugView::Tiles => ("TEXTURE TILES", &["BRIGHT: THE TILE LEVEL READ FOR WHAT IS SEEN, IN THE CACHE NOW", "DIM: FINER LEVELS LEFT ON DISK; GREY: VERTEX COLOURS ONLY"]),
            DebugView::Sky => ("SKY ONLY", &["LIGHT FROM THE SKY: BY FACING, NEVER BY DISTANCE"]),
            DebugView::Direct => ("DIRECT LIGHT", &["SKY AND POINT LIGHTS FROM THE SHADING CACHE, NO BOUNCE"]),
            DebugView::Highlight => ("HIGHLIGHT ONLY, X4", &["BLINN-PHONG, REDONE WHEN THE CAMERA MOVES"]),
            DebugView::Bounce => ("BOUNCE ONLY, X4", &["ONE BOUNCE, FROM EACH RECEIVER'S SPHERICAL HARMONICS"]),
            DebugView::LightCount => ("LIGHTS EVALUATED", &["POINT LIGHTS WHOSE REACH TOUCHES EACH CLUSTER"]),
            DebugView::Reach => ("LIGHT REACH (SELECTED LIGHT)", &["EACH BAND HALVES THE LUX; THE LAST BAND ENDS AT THE REACH", "OUTSIDE: THE LIGHT IS NEVER EVALUATED"]),
            DebugView::Shadow => ("SHADOW (SELECTED LIGHT)", &["WHITE: LIT; BLACK: IN SHADOW; RED: FACES AWAY; BLUE: BEYOND REACH", "3 X 3 PCF FROM THE CUBE MAP, INSET TOP RIGHT"]),
            DebugView::ShadowMap => ("SHADOW MAP (SELECTED LIGHT)", &["SIX FACES, UNFOLDED; BRIGHTER IS NEARER THE LIGHT", "BLUE: NOTHING THERE WITHIN THE REACH"]),
            DebugView::Surfels => ("SURFELS", &["THE SURFACES AS DISCS, EACH LIT DIRECTLY", "THIS LIT CLOUD IS WHAT BOUNCE LIGHT IS GATHERED FROM"]),
            DebugView::Receivers => ("BOUNCE RECEIVERS", &["GREEN: GATHERED; RED: WAITING FOR ITS TURN"]),
            DebugView::MicroRaster => ("MICRO-RASTER, X4", &["THE CUBE OF 6 X 8 X 8 TEXELS SEEN BY THE RECEIVER UNDER THE CURSOR", "NEAREST SURFELS FILL IT FIRST; THEN IT FOLDS INTO 4 NUMBERS PER COLOUR"]),
        }
    }
}

pub struct Ctx<'a> {
    pub scene: &'a Scene,
    pub cam: &'a Camera,
    pub lights: &'a [PointLight],
    pub selected: Option<usize>,
    pub settings: &'a RenderSettings,
    pub peek: Peek<'a>,
    /// The cursor in frame pixels, for the micro-raster.
    pub cursor: Option<(f32, f32)>,
}

/// What a pixel sees.
struct Hit {
    cluster: usize,
    tri: u32,
    w: [f32; 3],
    p: Vec3,
    /// Shading normal, turned towards the eye.
    n: Vec3,
    /// Vertex colour, linear.
    vcol: Vec3,
}

fn weights(s: &TriSetup, px: f32, py: f32) -> [f32; 3] {
    let b = raster::barycentrics(s, px, py);
    [
        b[0] * s.bary[0][0] + b[1] * s.bary[1][0] + b[2] * s.bary[2][0],
        b[0] * s.bary[0][1] + b[1] * s.bary[1][1] + b[2] * s.bary[2][1],
        b[0] * s.bary[0][2] + b[1] * s.bary[1][2] + b[2] * s.bary[2][2],
    ]
}

fn hit(ctx: &Ctx, frame: &Frame, x: usize, y: usize) -> Option<Hit> {
    let id = frame.tri_id[y * frame.width + x];
    if id == NO_TRI {
        return None;
    }
    let s = &frame.setups[id as usize];
    let w = weights(s, x as f32 + 0.5, y as f32 + 0.5);
    let sc = ctx.scene;
    let t = s.tri as usize * 3;
    let (i0, i1, i2) = (sc.indices[t] as usize, sc.indices[t + 1] as usize, sc.indices[t + 2] as usize);
    let p = sc.positions[i0] * w[0] + sc.positions[i1] * w[1] + sc.positions[i2] * w[2];
    let mut n = (sc.normals[i0] * w[0] + sc.normals[i1] * w[1] + sc.normals[i2] * w[2]).normalize_or(Vec3::Y);
    if n.dot(ctx.cam.eye - p) < 0.0 {
        n = -n;
    }
    let vcol = sc.colors[i0] * w[0] + sc.colors[i1] * w[1] + sc.colors[i2] * w[2];
    Some(Hit { cluster: s.cluster as usize, tri: s.tri, w, p, n, vcol })
}

/// Facing ratio, for giving flat colours some form.
fn facing(ctx: &Ctx, h: &Hit) -> f32 {
    0.35 + 0.65 * h.n.dot((ctx.cam.eye - h.p).normalize_or(Vec3::Z)).abs()
}

fn albedo(ctx: &Ctx, h: &Hit) -> Vec3 {
    let level = ctx.peek.kept_level(h.cluster);
    if level == 0 || level == NOT_VISIBLE {
        return h.vcol;
    }
    let c = &ctx.scene.clusters[h.cluster];
    match ctx.scene.textures.block(h.cluster, c.tri_count as usize, level) {
        Some(b) => b.sample((h.tri - c.first_tri) as usize, h.w[1], h.w[2]),
        None => h.vcol,
    }
}

/// sRGB 0..255 to linear.
fn rgb(r: u8, g: u8, b: u8) -> Vec3 {
    let f = |v: u8| {
        let s = v as f32 / 255.0;
        if s <= 0.04045 { s / 12.92 } else { ((s + 0.055) / 1.055).powf(2.4) }
    };
    Vec3::new(f(r), f(g), f(b))
}

fn level_colour(l: u8) -> Vec3 {
    match l {
        0 => rgb(84, 112, 198),
        1 => rgb(115, 192, 222),
        2 => rgb(145, 204, 117),
        3 => rgb(250, 200, 88),
        4 => rgb(252, 132, 82),
        5 => rgb(238, 102, 102),
        _ => rgb(40, 40, 40),
    }
}

fn hash(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^ (x >> 16)
}

fn hash_colour(k: u32) -> Vec3 {
    let h = hash(k);
    let hue = (h & 0xffff) as f32 / 65535.0 * 6.0;
    let sat = 0.45 + ((h >> 16) & 0xff) as f32 / 255.0 * 0.4;
    let val = 0.75 + ((h >> 24) & 0xff) as f32 / 255.0 * 0.25;
    let c = val * sat;
    let x = c * (1.0 - ((hue % 2.0) - 1.0).abs());
    let (r, g, b) = match hue as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = val - c;
    let s = Vec3::new(r + m, g + m, b + m);
    s.powf(2.2)
}

fn lum(packed: u32) -> f32 {
    let (r, g, b) = ((packed >> 16) & 255, (packed >> 8) & 255, packed & 255);
    let s = (0.2126 * r as f32 + 0.7152 * g as f32 + 0.0722 * b as f32) / 255.0;
    s.powf(2.2)
}

fn shown(ctx: &Ctx, radiance: Vec3) -> Vec3 {
    raster::tonemap(radiance * ctx.settings.exposure)
}

fn cut(ctx: &Ctx) -> f32 {
    raster::cut_lux(&ctx.settings.sky, ctx.settings.exposure)
}

fn selected<'a>(ctx: &'a Ctx) -> Option<(usize, &'a PointLight)> {
    let i = ctx.selected.unwrap_or(0);
    ctx.lights.get(i).map(|l| (i, l))
}

/// Draw `view` over the frame, which holds the final picture.
pub fn draw(view: DebugView, ctx: &Ctx, frame: &mut Frame) {
    if view == DebugView::Final {
        return;
    }
    let base = frame.color.clone();
    let (w, h) = (frame.width, frame.height);
    let dim = |i: usize, k: f32| Vec3::splat(lum(base[i]) * k);
    let mut out = vec![0u32; w * h];
    // Depth: the range of what is in view, 2nd to 98th percentile.
    let (d_lo, d_hi) = depth_range(frame);
    {
        let fr = &*frame;
        let px = |x: usize, y: usize| -> Vec3 {
            let i = y * w + x;
            let Some(hh) = hit(ctx, fr, x, y) else {
                return match view {
                    DebugView::Depth | DebugView::Normals | DebugView::Clusters | DebugView::Triangles | DebugView::Dice | DebugView::Tiles | DebugView::Albedo => Vec3::splat(0.004),
                    _ => dim(i, 0.35),
                };
            };
            match view {
                DebugView::Clusters => hash_colour(hh.cluster as u32) * facing(ctx, &hh),
                DebugView::Triangles => hash_colour(hh.tri.wrapping_mul(2_654_435_761)) * facing(ctx, &hh),
                DebugView::Depth => {
                    let d = 1.0 / fr.depth[i].max(1e-12);
                    depth_ramp(((d.ln() - d_lo.ln()) / (d_hi.ln() - d_lo.ln()).max(1e-6)).clamp(0.0, 1.0))
                }
                DebugView::Normals => (hh.n * 0.5 + Vec3::splat(0.5)).powf(2.2),
                DebugView::Dice => level_colour(ctx.peek.levels.get(hh.cluster).copied().unwrap_or(NOT_VISIBLE)) * facing(ctx, &hh),
                DebugView::Micropolygons => micropolygon(ctx, fr, x, y, &hh, dim(i, 0.55)),
                DebugView::Reshaded => {
                    let now = ctx.peek.frame_no;
                    // Full strength for this frame, fading over the next few.
                    let age = |s: u64| if s == 0 { 0.0 } else { (-((now - s.min(now)) as f32) / 1.5).exp() };
                    let r = age(ctx.peek.shaded_in(hh.cluster));
                    let g = ctx.peek.bounce.map(|b| age(b.stamp[hh.cluster])).unwrap_or(0.0);
                    dim(i, 0.3) * (1.0 - r.max(g)) + rgb(238, 70, 70) * r + rgb(90, 200, 110) * g * (1.0 - r * 0.5)
                }
                DebugView::Albedo => albedo(ctx, &hh),
                DebugView::Tiles => {
                    let e = ctx.scene.textures.entry(hh.cluster);
                    if e.level == 0 {
                        Vec3::splat(0.05) * facing(ctx, &hh)
                    } else {
                        let lv = ctx.peek.kept_level(hh.cluster);
                        let read = if lv == NOT_VISIBLE { 0 } else { lv.min(e.level) };
                        if read > 0 && ctx.scene.textures.resident(hh.cluster, lv) {
                            level_colour(read) * facing(ctx, &hh)
                        } else {
                            level_colour(e.level) * facing(ctx, &hh) * 0.2
                        }
                    }
                }
                DebugView::Sky => shown(ctx, albedo(ctx, &hh) * ctx.settings.sky.irradiance_from(hh.n, !ctx.settings.bounce) * std::f32::consts::FRAC_1_PI),
                DebugView::Direct => shown(ctx, ctx.peek.radiance(ctx.scene, hh.cluster, hh.tri, hh.w, Part::All)),
                DebugView::Highlight => shown(ctx, ctx.peek.radiance(ctx.scene, hh.cluster, hh.tri, hh.w, Part::Highlight) * 4.0),
                DebugView::Bounce => match ctx.peek.bounce {
                    Some(b) => shown(ctx, albedo(ctx, &hh) * b.irradiance_at(ctx.scene, hh.cluster, hh.tri, hh.w, hh.n) * std::f32::consts::FRAC_1_PI * 4.0),
                    None => Vec3::ZERO,
                },
                DebugView::LightCount => {
                    let c = &ctx.scene.clusters[hh.cluster];
                    let cut = cut(ctx);
                    let k = ctx.lights.iter().filter(|l| l.position.distance(c.centre) <= c.radius + raster::light_reach(l, cut)).count();
                    raster::heat(k as u32) * facing(ctx, &hh) * 1.4
                }
                DebugView::Reach => match selected(ctx) {
                    Some((_, l)) => {
                        let d2 = l.position.distance_squared(hh.p).max(1e-6);
                        let e = l.candela() / d2;
                        let cut = cut(ctx);
                        if e < cut {
                            dim(i, 0.18)
                        } else {
                            let k = (e / cut).log2().floor() as u32;
                            let band = reach_colour(k);
                            band * if k % 2 == 0 { 1.0 } else { 0.72 } * facing(ctx, &hh)
                        }
                    }
                    None => dim(i, 0.3),
                },
                DebugView::Shadow => match selected(ctx) {
                    Some((li, l)) => {
                        let reach = raster::light_reach(l, cut(ctx));
                        let to = l.position - hh.p;
                        if to.length() > reach {
                            rgb(20, 30, 70)
                        } else if hh.n.dot(to) <= 0.0 {
                            rgb(110, 25, 25)
                        } else {
                            match ctx.peek.shadows.get(li).and_then(|m| m.as_ref()) {
                                Some(m) => Vec3::splat(0.02 + 0.9 * m.lit(hh.p, hh.n)),
                                None => Vec3::splat(0.9),
                            }
                        }
                    }
                    None => dim(i, 0.3),
                },
                _ => dim(i, 0.45),
            }
        };
        if !matches!(view, DebugView::ShadowMap | DebugView::Surfels) {
            out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
                for (x, o) in row.iter_mut().enumerate() {
                    *o = raster::pack(px(x, y));
                }
            });
        }
    }
    frame.color.copy_from_slice(&out);
    match view {
        DebugView::Buckets => buckets(ctx, frame, &base),
        DebugView::ShadowMap => shadow_map(ctx, frame),
        DebugView::Shadow => {
            if let Some((li, _)) = selected(ctx) {
                if let Some(m) = ctx.peek.shadows.get(li).and_then(|m| m.as_ref()) {
                    let cell = (frame.height / 9).max(24);
                    let x0 = frame.width.saturating_sub(4 * cell + 16);
                    cube_cross(frame, x0, 16, cell, |f, u, v| cube_depth(m, f, u, v));
                }
            }
        }
        DebugView::Surfels => surfels(ctx, frame),
        DebugView::Receivers => receivers(ctx, frame),
        DebugView::MicroRaster => micro_raster(ctx, frame),
        _ => {}
    }
    legend(view, ctx, frame);
}

/// Near and far distances of what the frame sees, metres.
fn depth_range(frame: &Frame) -> (f32, f32) {
    let mut d: Vec<f32> = frame.depth.iter().filter(|v| **v > 0.0).step_by(7).map(|v| 1.0 / v).collect();
    if d.len() < 2 {
        return (0.3, 20000.0);
    }
    d.sort_unstable_by(|a, b| a.total_cmp(b));
    let lo = d[d.len() / 50].max(0.01);
    let hi = d[d.len() * 49 / 50].max(lo * 1.5);
    (lo, hi)
}

fn depth_ramp(t: f32) -> Vec3 {
    // Near warm and bright, far cool and dark.
    let a = rgb(255, 236, 170);
    let b = rgb(220, 90, 60);
    let c = rgb(60, 40, 110);
    let d = rgb(8, 10, 24);
    if t < 0.33 {
        a.lerp(b, t / 0.33)
    } else if t < 0.66 {
        b.lerp(c, (t - 0.33) / 0.33)
    } else {
        c.lerp(d, (t - 0.66) / 0.34)
    }
}

fn reach_colour(k: u32) -> Vec3 {
    const R: [(u8, u8, u8); 8] = [(48, 70, 160), (60, 120, 200), (70, 170, 190), (110, 190, 120), (200, 210, 90), (245, 180, 70), (240, 120, 60), (230, 70, 70)];
    let (r, g, b) = R[(k as usize).min(R.len() - 1)];
    rgb(r, g, b)
}

/// Micropolygon and triangle edges, about a pixel wide.
fn micropolygon(ctx: &Ctx, fr: &Frame, x: usize, y: usize, hh: &Hit, under: Vec3) -> Vec3 {
    let id = fr.tri_id[y * fr.width + x];
    let s = &fr.setups[id as usize];
    let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
    let w0 = hh.w;
    let wx = weights(s, px + 1.0, py);
    let wy = weights(s, px, py + 1.0);
    let level = ctx.peek.kept_level(hh.cluster);
    let level = if level == NOT_VISIBLE { 0 } else { level };
    let n = (1u32 << level) as f32;
    let mut tri_edge = false;
    let mut mp_edge = false;
    for k in 0..3 {
        let g = ((wx[k] - w0[k]).powi(2) + (wy[k] - w0[k]).powi(2)).sqrt().max(1e-9);
        // Triangle edge: this weight near 0.
        if w0[k] / g < 0.8 {
            tri_edge = true;
        }
        if level > 0 {
            let v = w0[k] * n;
            let f = (v - v.round()).abs();
            if f / (g * n) < 0.6 {
                mp_edge = true;
            }
        }
    }
    if tri_edge {
        Vec3::splat(0.95)
    } else if mp_edge {
        level_colour(level)
    } else {
        under
    }
}

fn buckets(ctx: &Ctx, frame: &mut Frame, base: &[u32]) {
    let tile = ctx.settings.tile.max(8);
    let (w, h) = (frame.width, frame.height);
    let (tx, ty) = (w.div_ceil(tile), h.div_ceil(tile));
    let mut counts = vec![0u32; tx * ty];
    for s in &frame.setups {
        for by in s.min_y as usize / tile..=(s.max_y as usize / tile).min(ty - 1) {
            for bx in s.min_x as usize / tile..=(s.max_x as usize / tile).min(tx - 1) {
                counts[by * tx + bx] += 1;
            }
        }
    }
    let max = counts.iter().copied().max().unwrap_or(1).max(1) as f32;
    for y in 0..h {
        for x in 0..w {
            let b = (y / tile) * tx + x / tile;
            let t = ((counts[b] as f32 + 1.0).ln() / (max + 1.0).ln()).clamp(0.0, 1.0);
            let col = Vec3::splat(lum(base[y * w + x]) * 0.3) + reach_colour((t * 7.0).round() as u32) * (0.25 + 0.5 * t);
            let line = x % tile == 0 || y % tile == 0;
            frame.color[y * w + x] = raster::pack(if line { Vec3::splat(0.5) } else { col });
        }
    }
    if tile >= 24 {
        for by in 0..ty {
            for bx in 0..tx {
                let s = counts[by * tx + bx].to_string();
                text(frame, bx * tile + 3, by * tile + 3, if tile >= 64 && frame.height >= 900 { 2 } else { 1 }, &s, Vec3::splat(0.95));
            }
        }
    }
}

/// 1 / depth of a shadow cube face texel, as a grey: nearer the light is
/// brighter. `u`, `v` in 0..1 across the face, `v` up.
fn cube_depth(m: &crate::shadow::CubeShadow, f: usize, u: f32, v: f32) -> Vec3 {
    let (res, d) = m.faces();
    let x = ((u * res as f32) as usize).min(res - 1);
    let y = ((v * res as f32) as usize).min(res - 1);
    let iw = d[f * res * res + y * res + x];
    if iw <= 0.0 {
        return rgb(25, 40, 90);
    }
    let dist = 1.0 / iw;
    let t = ((dist.max(0.05).log10() - 0.05f32.log10()) / (2000f32.log10() - 0.05f32.log10())).clamp(0.0, 1.0);
    Vec3::splat((1.0 - t).powf(1.6) * 0.95 + 0.02)
}

/// Draw a cube unfolded as a cross, `cell` pixels per face, the top left at
/// (x0, y0). Faces are in the order +X, -X, +Y, -Y, +Z, -Z.
fn cube_cross(frame: &mut Frame, x0: usize, y0: usize, cell: usize, f: impl Fn(usize, f32, f32) -> Vec3) {
    const AT: [(usize, usize); 6] = [(2, 1), (0, 1), (1, 0), (1, 2), (1, 1), (3, 1)];
    const NAMES: [&str; 6] = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"];
    for (face, &(cx, cy)) in AT.iter().enumerate() {
        for yy in 0..cell {
            for xx in 0..cell {
                let (x, y) = (x0 + cx * cell + xx, y0 + cy * cell + yy);
                if x >= frame.width || y >= frame.height {
                    continue;
                }
                let u = (xx as f32 + 0.5) / cell as f32;
                let v = 1.0 - (yy as f32 + 0.5) / cell as f32;
                let border = xx == 0 || yy == 0 || xx == cell - 1 || yy == cell - 1;
                frame.color[y * frame.width + x] = raster::pack(if border { Vec3::splat(0.6) } else { f(face, u, v) });
            }
        }
        let u = if cell >= 120 { 2 } else { 1 };
        text(frame, x0 + cx * cell + 4, y0 + cy * cell + 4, u, NAMES[face], rgb(255, 220, 90));
    }
}

fn shadow_map(ctx: &Ctx, frame: &mut Frame) {
    for c in frame.color.iter_mut() {
        *c = raster::pack(Vec3::splat(0.01));
    }
    let Some((li, l)) = selected(ctx) else { return };
    let Some(m) = ctx.peek.shadows.get(li).and_then(|m| m.as_ref()) else {
        text(frame, 16, frame.height / 2, 2, "NO SHADOW MAP: SHADOWS ARE OFF", Vec3::ONE);
        return;
    };
    let top = frame.height / 8;
    let cell = ((frame.width - 32) / 4).min((frame.height - top - 16 - 14 * ui(frame)) / 3);
    let x0 = (frame.width - 4 * cell) / 2;
    cube_cross(frame, x0, top, cell, |f, u, v| cube_depth(m, f, u, v));
    let (res, _) = m.faces();
    let reach = raster::light_reach(l, cut(ctx));
    let s = format!("{} X {} PER FACE, {} TRIANGLES DRAWN, REACH {}", res, res, m.tris, metres(reach));
    let u = ui(frame);
    text(frame, x0, top + 3 * cell + 8, u, &s, Vec3::splat(0.9));
}

fn metres(m: f32) -> String {
    if m >= 1000.0 { format!("{:.1} KM", m / 1000.0) } else if m >= 10.0 { format!("{:.0} M", m) } else { format!("{:.1} M", m) }
}

fn focal_px(cam: &Camera, height: usize) -> f32 {
    cam.proj.y_axis.y * height as f32 * 0.5
}

fn surfels(ctx: &Ctx, frame: &mut Frame) {
    let (w, h) = (frame.width, frame.height);
    for c in frame.color.iter_mut() {
        *c = raster::pack(Vec3::splat(0.004));
    }
    let Some(b) = ctx.peek.bounce else {
        text(frame, 16, h / 2, 2, "NO SURFELS: BOUNCE LIGHT IS OFF", Vec3::ONE);
        return;
    };
    let f = focal_px(ctx.cam, h);
    let mut zbuf = vec![0f32; w * h];
    let vp: Mat4 = ctx.cam.proj * ctx.cam.view;
    for (s, rad) in b.surfels.iter().zip(&b.radiance) {
        if s.normal.dot(ctx.cam.eye - s.position) <= 0.0 {
            continue;
        }
        let c = vp * s.position.extend(1.0);
        if c.w <= ctx.cam.near {
            continue;
        }
        let iw = 1.0 / c.w;
        let (sx, sy) = ((c.x * iw * 0.5 + 0.5) * w as f32, (0.5 - c.y * iw * 0.5) * h as f32);
        let r = (s.area / std::f32::consts::PI).sqrt() * f * iw;
        if r < 0.3 || sx + r < 0.0 || sy + r < 0.0 || sx - r > w as f32 || sy - r > h as f32 {
            continue;
        }
        let r = r.min(400.0);
        // The disc as seen: squashed along its tilt.
        let view_dir = (ctx.cam.eye - s.position).normalize();
        let tilt = s.normal.dot(view_dir).abs().max(0.15);
        let col = raster::tonemap(*rad * ctx.settings.exposure);
        let edge = col * 0.45;
        let ri = r.ceil() as i32;
        for dy in -ri..=ri {
            for dx in -ri..=ri {
                let d2 = (dx * dx + dy * dy) as f32;
                if d2 > r * r * (0.5 + 0.5 * tilt) {
                    continue;
                }
                let (x, y) = (sx as i32 + dx, sy as i32 + dy);
                if x < 0 || y < 0 || x as usize >= w || y as usize >= h {
                    continue;
                }
                let i = y as usize * w + x as usize;
                if zbuf[i] > iw {
                    continue;
                }
                zbuf[i] = iw;
                let rim = r > 3.0 && d2 > (r - 1.2) * (r - 1.2) * (0.5 + 0.5 * tilt);
                frame.color[i] = raster::pack(if rim { edge } else { col });
            }
        }
    }
    let u = ui(frame);
    text(frame, 16, h - 12 * u - 8, u, &format!("{} SURFELS, EACH LIT BY THE SKY AND THE POINT LIGHTS IN REACH", b.surfels.len()), Vec3::splat(0.9));
}

fn receivers(ctx: &Ctx, frame: &mut Frame) {
    let Some(b) = ctx.peek.bounce else {
        text(frame, 16, frame.height / 2, 2, "NO RECEIVERS: BOUNCE LIGHT IS OFF", Vec3::ONE);
        return;
    };
    let (w, h) = (frame.width, frame.height);
    let (mut gathered, mut waiting) = (0usize, 0usize);
    for ci in 0..ctx.scene.clusters.len() {
        let ok = b.valid[ci];
        for r in b.receivers_of(ci) {
            let p = b.receivers[r].0;
            let Some((sx, sy, iw)) = raster::project(ctx.cam, w, h, p) else { continue };
            let (x, y) = (sx as i32, sy as i32);
            if x < 0 || y < 0 || x as usize >= w || y as usize >= h {
                continue;
            }
            if frame.depth[y as usize * w + x as usize] > iw * 1.01 {
                continue;
            }
            if ok { gathered += 1 } else { waiting += 1 }
            let col = raster::pack(if ok { rgb(90, 220, 110) } else { rgb(240, 70, 70) });
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (xx, yy) = (x + dx, y + dy);
                    if xx >= 0 && yy >= 0 && (xx as usize) < w && (yy as usize) < h {
                        frame.color[yy as usize * w + xx as usize] = col;
                    }
                }
            }
        }
    }
    let u = ui(frame);
    text(frame, 16, h - 12 * u - 8, u, &format!("{gathered} GATHERED, {waiting} WAITING, IN VIEW"), Vec3::splat(0.9));
}

/// The receiver whose cluster is under (x, y), nearest the point seen.
fn receiver_at(ctx: &Ctx, frame: &Frame, b: &Bounce, x: usize, y: usize) -> Option<(usize, usize, Vec3)> {
    let hh = hit(ctx, frame, x, y)?;
    let r = b.receivers_of(hh.cluster).min_by(|a, c| b.receivers[*a].0.distance_squared(hh.p).total_cmp(&b.receivers[*c].0.distance_squared(hh.p)))?;
    Some((hh.cluster, r, hh.p))
}

fn micro_raster(ctx: &Ctx, frame: &mut Frame) {
    let (w, h) = (frame.width, frame.height);
    let Some(b) = ctx.peek.bounce else {
        text(frame, 16, h / 2, 2, "NO MICRO-RASTER: BOUNCE LIGHT IS OFF", Vec3::ONE);
        return;
    };
    let (cx, cy) = ctx.cursor.map(|(x, y)| (x.clamp(0.0, w as f32 - 1.0) as usize, y.clamp(0.0, h as f32 - 1.0) as usize)).unwrap_or((w / 2, h / 2));
    let Some((ci, r, _)) = receiver_at(ctx, frame, b, cx, cy) else {
        text(frame, 16, h / 2, 2, "POINT AT A SURFACE", Vec3::ONE);
        return;
    };
    let reach = b.reach(cut(ctx));
    let (rad, cov) = b.raster(r, ci, reach);
    let rp = b.receivers[r].0;
    // Mark the receiver.
    if let Some((sx, sy, _)) = raster::project(ctx.cam, w, h, rp) {
        let (x, y) = (sx as i32, sy as i32);
        for k in -12..=12i32 {
            for (xx, yy) in [(x + k, y), (x, y + k)] {
                if k.abs() > 3 && xx >= 0 && yy >= 0 && (xx as usize) < w && (yy as usize) < h {
                    frame.color[yy as usize * w + xx as usize] = raster::pack(rgb(255, 220, 90));
                }
            }
        }
    }
    let per = MICRO;
    let texel = ((h as f32 * 0.62 / (3 * per) as f32).min(w as f32 * 0.5 / (4 * per) as f32)).max(2.0) as usize;
    let cell = texel * per;
    let x0 = w.saturating_sub(4 * cell + 16);
    let y0 = h.saturating_sub(3 * cell + 16);
    let exposure = ctx.settings.exposure;
    cube_cross(frame, x0, y0, cell, |f, u, v| {
        let tx = ((u * per as f32) as usize).min(per - 1);
        let ty = ((v * per as f32) as usize).min(per - 1);
        let t = f * per * per + ty * per + tx;
        let gx = ((u * cell as f32) as usize) % texel == 0 || ((v * cell as f32) as usize) % texel == 0;
        if gx {
            return Vec3::splat(0.12);
        }
        if cov[t] <= 0.0 {
            // Nothing seen: the sky's share is added separately.
            if (tx + ty) % 2 == 0 { Vec3::splat(0.03) } else { Vec3::splat(0.05) }
        } else {
            raster::tonemap(rad[t] * exposure * 4.0)
        }
    });
    // What the folded spherical harmonics give back: a white ball lit by them.
    let sh = b.gather(r, ci, reach);
    let radius = (cell as f32 * 0.42) as i32;
    let (bx, by) = ((x0 + 3 * cell + cell / 2) as i32, (y0 + 2 * cell + cell / 2) as i32);
    let inv_view = ctx.cam.view.inverse();
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let (fx, fy) = (dx as f32 / radius as f32, -dy as f32 / radius as f32);
            let d2 = fx * fx + fy * fy;
            if d2 > 1.0 {
                continue;
            }
            let nv = Vec3::new(fx, fy, (1.0 - d2).sqrt());
            let n = inv_view.transform_vector3(nv).normalize();
            let e = crate::bounce::sh_irradiance(&sh, n);
            let col = raster::tonemap(e * 0.8 * std::f32::consts::FRAC_1_PI * exposure * 4.0);
            let (x, y) = (bx + dx, by + dy);
            if x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h {
                frame.color[y as usize * w + x as usize] = raster::pack(col);
            }
        }
    }
    text(frame, (x0 + 3 * cell + 4) as usize, (y0 + 2 * cell + 4) as usize, 1, "WHAT 4 SH NUMBERS GIVE", rgb(255, 220, 90));
}

/// Title, notes and keys, top left.
fn legend(view: DebugView, ctx: &Ctx, frame: &mut Frame) {
    let (title, notes) = view.caption();
    let s = ui(frame);
    let idx = ALL.iter().position(|v| *v == view).unwrap_or(0);
    let mut lines: Vec<String> = vec![format!("{}/{}  {}", idx, ALL.len() - 1, title)];
    lines.extend(notes.iter().map(|n| n.to_string()));
    match view {
        DebugView::Reach | DebugView::Shadow | DebugView::ShadowMap => {
            if let Some((i, l)) = selected(ctx) {
                lines.push(format!("LIGHT {}: {:.0} LM, REACH {} (TAB: NEXT LIGHT)", i + 1, l.lumens, metres(raster::light_reach(l, cut(ctx)))));
            }
        }
        _ => {}
    }
    let line_h = 10 * s;
    let width = lines.iter().map(|l| l.len() * 6 * s).max().unwrap_or(0) + 12 * s;
    darken(frame, 8, 8, width, lines.len() * line_h + 8 * s);
    for (k, l) in lines.iter().enumerate() {
        let col = if k == 0 { rgb(255, 220, 90) } else { Vec3::splat(0.92) };
        text(frame, 8 + 6 * s, 8 + 4 * s + k * line_h, if k == 0 { s } else { s.max(1) }, l, col);
    }
    // Swatches for views whose colours are a scale.
    let swatches: Vec<(Vec3, String)> = match view {
        DebugView::Dice | DebugView::Micropolygons => (0..=5u8).map(|l| (level_colour(l), format!("L{} {}", l, ["VERTICES", "<4 PX", "<8 PX", "<16 PX", "<32 PX", "FINEST"][l as usize]))).collect(),
        DebugView::Tiles => (1..=5u8).map(|l| (level_colour(l), format!("L{l} {} PTS", crate::texture::tile_points(l)))).collect(),
        DebugView::LightCount => (0..=5u32).map(|k| (raster::heat(k) * 1.4, if k == 5 { "5+".to_string() } else { k.to_string() })).collect(),
        DebugView::Reach => (0..8u32).map(|k| (reach_colour(k), if k == 0 { "REACH".to_string() } else { format!("X{}", 1u32 << k) })).collect(),
        DebugView::Depth => {
            let (lo, hi) = depth_range(frame);
            (0..5).map(|k| {
                let t = k as f32 / 4.0;
                (depth_ramp(t), metres((lo.ln() + (hi.ln() - lo.ln()) * t).exp()))
            }).collect()
        }
        _ => Vec::new(),
    };
    if !swatches.is_empty() {
        let y = frame.height - 22 * s;
        let mut x = 8;
        let wsum: usize = swatches.iter().map(|(_, n)| 14 * s + n.len() * 6 * s + 10 * s).sum();
        darken(frame, 4, y - 6 * s, wsum + 8, 20 * s);
        for (c, n) in &swatches {
            fill(frame, x + 4, y - 2 * s, 10 * s, 10 * s, *c);
            text(frame, x + 4 + 14 * s, y - s, s, n, Vec3::splat(0.92));
            x += 14 * s + n.len() * 6 * s + 10 * s;
        }
    }
}

/// Text scale for the frame's size.
fn ui(frame: &Frame) -> usize {
    if frame.height >= 600 { 2 } else { 1 }
}

fn darken(frame: &mut Frame, x0: usize, y0: usize, w: usize, h: usize) {
    for y in y0..(y0 + h).min(frame.height) {
        for x in x0..(x0 + w).min(frame.width) {
            let c = &mut frame.color[y * frame.width + x];
            let (r, g, b) = ((*c >> 16) & 255, (*c >> 8) & 255, *c & 255);
            *c = ((r / 4) << 16) | ((g / 4) << 8) | (b / 4);
        }
    }
}

fn fill(frame: &mut Frame, x0: usize, y0: usize, w: usize, h: usize, col: Vec3) {
    let p = raster::pack(col);
    for y in y0..(y0 + h).min(frame.height) {
        for x in x0..(x0 + w).min(frame.width) {
            frame.color[y * frame.width + x] = p;
        }
    }
}

/// Draw `s` in a 5 x 7 pixel font, scaled by `scale`, with a dark shadow.
pub fn text(frame: &mut Frame, x0: usize, y0: usize, scale: usize, s: &str, col: Vec3) {
    let p = raster::pack(col);
    let mut x = x0;
    for ch in s.chars() {
        let g = glyph(ch);
        for (row, bits) in g.iter().enumerate() {
            for c in 0..5 {
                if bits & (0x10 >> c) == 0 {
                    continue;
                }
                for dy in 0..scale {
                    for dx in 0..scale {
                        let (px, py) = (x + c * scale + dx, y0 + row * scale + dy);
                        for (ox, oy, v) in [(1, 1, 0u32), (0, 0, p)] {
                            let (qx, qy) = (px + ox, py + oy);
                            if qx < frame.width && qy < frame.height {
                                frame.color[qy * frame.width + qx] = v;
                            }
                        }
                    }
                }
            }
        }
        x += 6 * scale;
    }
}

fn glyph(c: char) -> [u8; 7] {
    match c.to_ascii_uppercase() {
        'A' => [0x0e, 0x11, 0x11, 0x1f, 0x11, 0x11, 0x11],
        'B' => [0x1e, 0x11, 0x11, 0x1e, 0x11, 0x11, 0x1e],
        'C' => [0x0e, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0e],
        'D' => [0x1e, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1e],
        'E' => [0x1f, 0x10, 0x10, 0x1e, 0x10, 0x10, 0x1f],
        'F' => [0x1f, 0x10, 0x10, 0x1e, 0x10, 0x10, 0x10],
        'G' => [0x0e, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0f],
        'H' => [0x11, 0x11, 0x11, 0x1f, 0x11, 0x11, 0x11],
        'I' => [0x0e, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0e],
        'J' => [0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0c],
        'K' => [0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11],
        'L' => [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1f],
        'M' => [0x11, 0x1b, 0x15, 0x15, 0x11, 0x11, 0x11],
        'N' => [0x11, 0x11, 0x19, 0x15, 0x13, 0x11, 0x11],
        'O' => [0x0e, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0e],
        'P' => [0x1e, 0x11, 0x11, 0x1e, 0x10, 0x10, 0x10],
        'Q' => [0x0e, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0d],
        'R' => [0x1e, 0x11, 0x11, 0x1e, 0x14, 0x12, 0x11],
        'S' => [0x0f, 0x10, 0x10, 0x0e, 0x01, 0x01, 0x1e],
        'T' => [0x1f, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
        'U' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0e],
        'V' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x0a, 0x04],
        'W' => [0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0a],
        'X' => [0x11, 0x11, 0x0a, 0x04, 0x0a, 0x11, 0x11],
        'Y' => [0x11, 0x11, 0x0a, 0x04, 0x04, 0x04, 0x04],
        'Z' => [0x1f, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1f],
        '0' => [0x0e, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0e],
        '1' => [0x04, 0x0c, 0x04, 0x04, 0x04, 0x04, 0x0e],
        '2' => [0x0e, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1f],
        '3' => [0x1f, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0e],
        '4' => [0x02, 0x06, 0x0a, 0x12, 0x1f, 0x02, 0x02],
        '5' => [0x1f, 0x10, 0x1e, 0x01, 0x01, 0x11, 0x0e],
        '6' => [0x06, 0x08, 0x10, 0x1e, 0x11, 0x11, 0x0e],
        '7' => [0x1f, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08],
        '8' => [0x0e, 0x11, 0x11, 0x0e, 0x11, 0x11, 0x0e],
        '9' => [0x0e, 0x11, 0x11, 0x0f, 0x01, 0x02, 0x0c],
        '.' => [0, 0, 0, 0, 0, 0x0c, 0x0c],
        ',' => [0, 0, 0, 0, 0x0c, 0x04, 0x08],
        ':' => [0, 0x0c, 0x0c, 0, 0x0c, 0x0c, 0],
        ';' => [0, 0x0c, 0x0c, 0, 0x0c, 0x04, 0x08],
        '/' => [0, 0x01, 0x02, 0x04, 0x08, 0x10, 0],
        '-' => [0, 0, 0, 0x1f, 0, 0, 0],
        '+' => [0, 0x04, 0x04, 0x1f, 0x04, 0x04, 0],
        '(' => [0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02],
        ')' => [0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08],
        '%' => [0x18, 0x19, 0x02, 0x04, 0x08, 0x13, 0x03],
        '=' => [0, 0, 0x1f, 0, 0x1f, 0, 0],
        '\'' => [0x04, 0x04, 0x08, 0, 0, 0, 0],
        '>' => [0x08, 0x04, 0x02, 0x01, 0x02, 0x04, 0x08],
        '<' => [0x02, 0x04, 0x08, 0x10, 0x08, 0x04, 0x02],
        '^' => [0x04, 0x0a, 0x11, 0, 0, 0, 0],
        '_' => [0, 0, 0, 0, 0, 0, 0x1f],
        '!' => [0x04, 0x04, 0x04, 0x04, 0x04, 0, 0x04],
        '?' => [0x0e, 0x11, 0x01, 0x02, 0x04, 0, 0x04],
        _ => [0; 7],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_round_trip_and_step_wraps() {
        for v in ALL {
            assert_eq!(DebugView::from_name(v.name()), Some(v));
        }
        assert_eq!(DebugView::Final.step(-1), DebugView::MicroRaster);
        assert_eq!(DebugView::MicroRaster.step(1), DebugView::Final);
    }

    #[test]
    fn every_view_draws_on_a_small_scene() {
        use crate::raster::Sky;
        let mut scene = Scene::new();
        let p = [Vec3::new(-2.0, 0.0, -2.0), Vec3::new(2.0, 0.0, -2.0), Vec3::new(2.0, 0.0, 2.0), Vec3::new(-2.0, 0.0, 2.0)];
        scene.push_mesh(&p, &[], &[Vec3::splat(0.6)], &[0, 2, 1, 0, 3, 2]);
        let lights = vec![PointLight { position: Vec3::new(0.0, 1.0, 0.0), color: Vec3::ONE, lumens: 800.0 }];
        let c = crate::camera::OrbitCamera { target: Vec3::ZERO, distance: 5.0, yaw: 0.5, pitch: 0.6, fov_y: 45f32.to_radians(), scene_radius: 3.0 };
        let cam = Camera { view: c.view(), proj: c.proj(1.5), eye: c.eye(), near: c.near() };
        let settings = RenderSettings::new(Sky { lux: 30.0, on: true });
        let mut r = crate::reyes::Renderer::new();
        let mut frame = Frame::new(96, 64);
        r.render(&scene, &cam, &lights, &settings, &mut frame);
        let shaded = frame.color.clone();
        for v in ALL {
            frame.color.copy_from_slice(&shaded);
            let ctx = Ctx { scene: &scene, cam: &cam, lights: &lights, selected: Some(0), settings: &settings, peek: r.peek(), cursor: None };
            draw(v, &ctx, &mut frame);
            if v == DebugView::Final {
                assert_eq!(frame.color, shaded, "the final view leaves the picture alone");
            } else {
                assert_ne!(frame.color, shaded, "{} drew nothing", v.name());
            }
        }
    }

    #[test]
    fn every_caption_character_has_a_glyph() {
        for v in ALL {
            let (t, notes) = v.caption();
            for s in std::iter::once(&t).chain(notes.iter()) {
                for c in s.chars() {
                    assert!(c == ' ' || glyph(c) != [0; 7], "no glyph for {c:?} in {s}");
                }
            }
        }
    }
}
