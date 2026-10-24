//! Sky occlusion kept as spherical harmonics, precomputed per model
//! (precomputed radiance transfer: Sloan, Kautz and Snyder, 2002; Weta's
//! PantaRay for Avatar cached occlusion this way, Pantaleoni et al. 2010).
//!
//! At import, each surface sample (`samples.rs`) looks at the rest of its
//! own model on a small cube raster, as the bounce gather does: the
//! model's samples are discs, near ones laid on one by one, far groups as
//! one disc each, no rays. What stays open, weighted by the cosine to the
//! sample's normal, is folded into nine numbers: order-2 spherical
//! harmonics, enough for the light a diffuse surface gets from any smooth
//! sky to within about 1% (Ramamoorthi and Hanrahan, 2001).
//!
//! At run time the sky is folded into the same nine numbers, and a
//! sample's share of the sky is the dot product of the two over that of
//! an open sample facing the same way. A sky change (contrast, time of
//! day) is one dot product a sample; moving or turning a model turns its
//! nine numbers with it. A shading point takes the share from the samples
//! of its own object around it, so its detail is the samples' spacing,
//! not the mesh's. Occlusion by other objects is not in it.

use std::sync::OnceLock;

use glam::{Quat, Vec3};

use crate::samples::Sample;

/// Order-2 real spherical harmonics at unit direction `d`: Y00, Y1-1, Y10,
/// Y11, Y2-2, Y2-1, Y20, Y21, Y22.
#[inline]
pub fn sh9(d: Vec3) -> [f32; 9] {
    let (x, y, z) = (d.x, d.y, d.z);
    [
        0.282_095,
        0.488_603 * y,
        0.488_603 * z,
        0.488_603 * x,
        1.092_548 * x * y,
        1.092_548 * y * z,
        0.315_392 * (3.0 * z * z - 1.0),
        1.092_548 * x * z,
        0.546_274 * (x * x - y * y),
    ]
}

#[inline]
pub fn dot9(a: &[f32; 9], b: &[f32; 9]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// What an open surface facing `n` gets from a sky with coefficients L,
/// as nine numbers to dot with L: the clamped cosine around `n`.
#[inline]
pub fn open_transfer(n: Vec3) -> [f32; 9] {
    let y = sh9(n);
    let a = [std::f32::consts::PI, 2.0 * std::f32::consts::PI / 3.0, std::f32::consts::PI / 4.0];
    [y[0] * a[0], y[1] * a[1], y[2] * a[1], y[3] * a[1], y[4] * a[2], y[5] * a[2], y[6] * a[2], y[7] * a[2], y[8] * a[2]]
}

/// Five directions in general position, and the inverse of the band-2
/// functions at them: band 2 of any function is recovered from its
/// values there, which is how band 2 is turned.
fn band2() -> &'static ([Vec3; 5], [[f32; 5]; 5]) {
    static B: OnceLock<([Vec3; 5], [[f32; 5]; 5])> = OnceLock::new();
    B.get_or_init(|| {
        let dirs = [Vec3::X, Vec3::Z, Vec3::new(1.0, 1.0, 0.0).normalize(), Vec3::new(0.0, 1.0, 1.0).normalize(), Vec3::new(1.0, 0.0, 1.0).normalize()];
        // m[k][j] = Y2j(dir k), inverted by Gauss-Jordan in f64.
        let mut m = [[0f64; 10]; 5];
        for (k, d) in dirs.iter().enumerate() {
            let y = sh9(*d);
            for j in 0..5 {
                m[k][j] = y[4 + j] as f64;
            }
            m[k][5 + k] = 1.0;
        }
        for c in 0..5 {
            let p = (c..5).max_by(|&a, &b| m[a][c].abs().total_cmp(&m[b][c].abs())).unwrap();
            m.swap(c, p);
            let inv = 1.0 / m[c][c];
            for j in 0..10 {
                m[c][j] *= inv;
            }
            for r in 0..5 {
                if r != c {
                    let f = m[r][c];
                    for j in 0..10 {
                        m[r][j] -= f * m[c][j];
                    }
                }
            }
        }
        let inv: [[f32; 5]; 5] = std::array::from_fn(|r| std::array::from_fn(|c| m[r][5 + c] as f32));
        (dirs, inv)
    })
}

/// Coefficients of the function turned by `q`: g(d) = f(q⁻¹ d).
pub fn rotate(c: &[f32; 9], q: Quat) -> [f32; 9] {
    let mut out = [0f32; 9];
    out[0] = c[0];
    // Band 1 is a vector: (Y11, Y1-1, Y10) go with (x, y, z).
    let v = q * Vec3::new(c[3], c[1], c[2]);
    out[1] = v.y;
    out[2] = v.z;
    out[3] = v.x;
    // Band 2: its values at the five directions, read from the unturned
    // function, then solved for.
    let (dirs, inv) = band2();
    let qi = q.inverse();
    let vals: [f32; 5] = std::array::from_fn(|k| {
        let y = sh9(qi * dirs[k]);
        (0..5).map(|j| c[4 + j] * y[4 + j]).sum()
    });
    for j in 0..5 {
        // inv maps values at the directions to coefficients: c = M⁻¹ v,
        // with M[k][j] = Y2j(d_k), so c_j = sum_k inv[j][k] v_k.
        out[4 + j] = (0..5).map(|k| inv[j][k] * vals[k]).sum();
    }
    out
}

/// Texels per cube face edge of a sample's raster.
const RES: usize = 8;
/// A tree node is taken whole when its radius over its distance is below
/// this.
const FAR: f32 = 0.2;
const LEAF: usize = 4;
/// For occlusion between objects, a group is taken whole when its radius
/// over its distance is below this: soft, so coarser.
const CROSS_FAR: f32 = 0.5;

/// The six cube faces: (axis, right, up).
const FACES: [(Vec3, Vec3, Vec3); 6] = [
    (Vec3::X, Vec3::NEG_Z, Vec3::Y),
    (Vec3::NEG_X, Vec3::Z, Vec3::Y),
    (Vec3::Y, Vec3::X, Vec3::NEG_Z),
    (Vec3::NEG_Y, Vec3::X, Vec3::Z),
    (Vec3::Z, Vec3::X, Vec3::Y),
    (Vec3::NEG_Z, Vec3::NEG_X, Vec3::Y),
];

/// Direction and solid angle of every texel of the raster.
fn texels() -> &'static [(Vec3, f32)] {
    static T: OnceLock<Vec<(Vec3, f32)>> = OnceLock::new();
    T.get_or_init(|| {
        let mut v = Vec::with_capacity(6 * RES * RES);
        let s = 2.0 / RES as f32;
        for (a, r, u) in FACES {
            for y in 0..RES {
                for x in 0..RES {
                    let (fu, fv) = (-1.0 + (x as f32 + 0.5) * s, -1.0 + (y as f32 + 0.5) * s);
                    let d = (a + r * fu + u * fv).normalize();
                    let omega = s * s / (1.0 + fu * fu + fv * fv).powf(1.5);
                    v.push((d, omega));
                }
            }
        }
        v
    })
}

fn texel_of(d: Vec3) -> usize {
    let a = d.abs();
    let f = if a.x >= a.y && a.x >= a.z {
        if d.x >= 0.0 { 0 } else { 1 }
    } else if a.y >= a.z {
        if d.y >= 0.0 { 2 } else { 3 }
    } else if d.z >= 0.0 {
        4
    } else {
        5
    };
    let (ax, r, u) = FACES[f];
    let w = d.dot(ax);
    let m = RES as f32;
    let x = (((d.dot(r) / w) * 0.5 + 0.5) * m).floor().clamp(0.0, m - 1.0) as usize;
    let y = (((d.dot(u) / w) * 0.5 + 0.5) * m).floor().clamp(0.0, m - 1.0) as usize;
    f * RES * RES + y * RES + x
}

#[derive(Clone, Copy, Default)]
struct Node {
    centre: Vec3,
    radius: f32,
    /// Children, or (when `leaf`) the first disc and count.
    a: u32,
    b: u32,
    leaf: bool,
    area: f32,
    n_sum: Vec3,
}

#[derive(Clone, Copy)]
struct Disc {
    p: Vec3,
    n: Vec3,
    area: f32,
    /// Index of the sample it is.
    id: u32,
}

fn build(discs: &mut [Disc], lo: usize, hi: usize, nodes: &mut Vec<Node>) -> u32 {
    let idx = nodes.len() as u32;
    nodes.push(Node::default());
    let s = &discs[lo..hi];
    let (mut bmin, mut bmax) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
    let (mut area, mut n_sum) = (0.0, Vec3::ZERO);
    for d in s {
        bmin = bmin.min(d.p);
        bmax = bmax.max(d.p);
        area += d.area;
        n_sum += d.n * d.area;
    }
    let centre = (bmin + bmax) * 0.5;
    let radius = s.iter().map(|d| d.p.distance(centre) + (d.area / std::f32::consts::PI).sqrt()).fold(0.0, f32::max);
    let mut node = Node { centre, radius, area, n_sum, ..Default::default() };
    if hi - lo <= LEAF {
        node.leaf = true;
        node.a = lo as u32;
        node.b = (hi - lo) as u32;
    } else {
        let ext = bmax - bmin;
        let axis = if ext.x >= ext.y && ext.x >= ext.z { 0 } else if ext.y >= ext.z { 1 } else { 2 };
        let mid = (lo + hi) / 2;
        discs[lo..hi].select_nth_unstable_by(mid - lo, |a, b| a.p[axis].total_cmp(&b.p[axis]));
        node.a = build(discs, lo, mid, nodes);
        node.b = build(discs, mid, hi, nodes);
    }
    nodes[idx as usize] = node;
    idx
}

/// The samples as a tree of discs, to look at from each of them.
pub struct Occluders {
    discs: Vec<Disc>,
    nodes: Vec<Node>,
}

impl Occluders {
    pub fn new(samples: &[Sample]) -> Occluders {
        let mut discs: Vec<Disc> = samples.iter().enumerate().map(|(i, s)| Disc { p: s.position, n: s.normal, area: s.area, id: i as u32 }).collect();
        let mut nodes = Vec::with_capacity(discs.len() / 2 + 1);
        if !discs.is_empty() {
            let n = discs.len();
            build(&mut discs, 0, n, &mut nodes);
        }
        Occluders { discs, nodes }
    }

    /// How much of each texel is covered, seen from sample `me` at `p`
    /// facing `n`, by everything above its tangent plane.
    pub fn coverage(&self, me: u32, p: Vec3, n: Vec3, spacing: f32) -> [f32; 6 * RES * RES] {
        let mut cov = [0f32; 6 * RES * RES];
        if self.nodes.is_empty() {
            return cov;
        }
        let tex = texels();
        // Looked at from a little off the surface.
        let x = p + n * (spacing * 0.25);
        let mut stack: Vec<u32> = vec![0];
        let cover = |q: Vec3, proj: f32, disc: Option<(Vec3, f32)>, cov: &mut [f32; 6 * RES * RES]| {
            let v = q - x;
            let d2 = v.length_squared();
            if d2 < 1e-12 || proj <= 0.0 {
                return;
            }
            let d = d2.sqrt();
            let dir = v / d;
            let omega = (proj / d2).min(2.0 * std::f32::consts::PI);
            let t0 = texel_of(dir);
            if omega <= tex[t0].1 {
                cov[t0] += omega / tex[t0].1;
                return;
            }
            let mut any = false;
            match disc {
                // A large disc: each texel looks along its direction and is
                // covered where that line meets the disc, enlarged by a
                // third so that tiled discs leave no gaps.
                Some((dn, r)) => {
                    let pn = v.dot(dn);
                    let lim = (r * 1.33) * (r * 1.33);
                    for (t, (td, _)) in tex.iter().enumerate() {
                        let den = td.dot(dn);
                        if den.abs() < 1e-6 {
                            continue;
                        }
                        let s = pn / den;
                        if s > 0.0 && (*td * s - v).length_squared() <= lim {
                            cov[t] = 1.0;
                            any = true;
                        }
                    }
                }
                // A large group: the texels inside its cone, widened by half.
                None => {
                    let alpha = (omega / std::f32::consts::PI).sqrt();
                    let cos_a = (alpha * 1.5).min(std::f32::consts::FRAC_PI_2).cos();
                    for (t, (td, _)) in tex.iter().enumerate() {
                        if td.dot(dir) >= cos_a {
                            cov[t] = 1.0;
                            any = true;
                        }
                    }
                }
            }
            if !any {
                cov[t0] = 1.0;
            }
        };
        while let Some(i) = stack.pop() {
            let nd = &self.nodes[i as usize];
            // Wholly below the tangent plane: cannot cover the sky above.
            if n.dot(nd.centre - p) < -nd.radius || nd.area <= 0.0 {
                continue;
            }
            let to = nd.centre - x;
            let d = to.length();
            if !nd.leaf && d > 0.0 && nd.radius / d < FAR {
                // A group as one disc: what it shows is at least a quarter
                // of its area, as a closed shape does from any side. One
                // whose middle is below the horizon is the surface curving
                // away (its cone, widened, would reach above): left out.
                let dir = to / d;
                if dir.dot(n) <= 0.0 {
                    continue;
                }
                let proj = nd.n_sum.dot(dir).abs().max(nd.area * 0.25).min(nd.area);
                cover(nd.centre, proj, None, &mut cov);
                continue;
            }
            if !nd.leaf {
                stack.push(nd.a);
                stack.push(nd.b);
                continue;
            }
            for k in nd.a..nd.a + nd.b {
                let dc = &self.discs[k as usize];
                if dc.id == me || n.dot(dc.p - p) <= 0.0 {
                    continue;
                }
                let v = dc.p - x;
                let d = v.length().max(1e-9);
                // Within a few spacings, a disc low over the horizon is
                // more likely a bump of the same surface than something
                // over it, and as a disc it would cover far more of the
                // sky than the bump does: only those well above count.
                if d < 3.0 * spacing && n.dot(dc.p - p) < 0.5 * d {
                    continue;
                }
                // A disc as near as its own size stands for a piece of
                // curved surface, which covers much less than the flat
                // disc would: its radius is held to half its distance.
                let r = (dc.area / std::f32::consts::PI).sqrt().min(0.5 * d / 1.33);
                let proj = std::f32::consts::PI * r * r * dc.n.dot(v / d).abs();
                cover(dc.p, proj, Some((dc.n, r)), &mut cov);
            }
        }
        for c in cov.iter_mut() {
            *c = c.min(1.0);
        }
        cov
    }

    /// The sky and the ground's light that this model's discs hide from a
    /// point of another object (or of the ground) at `p` facing `n`, each
    /// as irradiance up to the scale of `radiance` (sky, ground). Only
    /// discs facing the point count, so that the far side of the model is
    /// not counted again behind its near side (Bunnell, 2005).
    pub fn hidden(&self, p: Vec3, n: Vec3, radiance: impl Fn(Vec3) -> [f32; 2]) -> [f32; 2] {
        let mut out = [0f32; 2];
        if self.nodes.is_empty() {
            return out;
        }
        let mut add = |q: Vec3, proj: f32, d2: f32| {
            let d = d2.sqrt().max(1e-9);
            let dir = (q - p) / d;
            let cos = n.dot(dir);
            if cos <= 0.0 || proj <= 0.0 {
                return;
            }
            // Solid angle, kept finite as a disc comes near (Bunnell).
            let omega = (proj / (d2 + proj / std::f32::consts::PI)).min(2.0 * std::f32::consts::PI);
            let r = radiance(dir);
            out[0] += r[0] * cos * omega;
            out[1] += r[1] * cos * omega;
        };
        let mut stack: Vec<u32> = vec![0];
        while let Some(i) = stack.pop() {
            let nd = &self.nodes[i as usize];
            // Wholly below the point's horizon.
            if n.dot(nd.centre - p) < -nd.radius || nd.area <= 0.0 {
                continue;
            }
            let to = nd.centre - p;
            let d2 = to.length_squared();
            if !nd.leaf && d2 > 0.0 && nd.radius * nd.radius < CROSS_FAR * CROSS_FAR * d2 {
                // The side it shows the point: at least a quarter of its
                // area, as a closed shape does.
                let proj = (-nd.n_sum.dot(to) / d2.sqrt()).max(nd.area * 0.25).min(nd.area);
                add(nd.centre, proj, d2);
                continue;
            }
            if !nd.leaf {
                stack.push(nd.a);
                stack.push(nd.b);
                continue;
            }
            for k in nd.a..nd.a + nd.b {
                let dc = &self.discs[k as usize];
                let v = p - dc.p;
                let d2 = v.length_squared();
                let facing = dc.n.dot(v);
                if facing <= 0.0 {
                    continue;
                }
                add(dc.p, dc.area * facing / d2.sqrt().max(1e-9), d2);
            }
        }
        out
    }

    /// The nine numbers of sample `me`: the open part of its raster,
    /// cosine weighted. Corrected so that an open sample gives exactly
    /// `open_transfer` (the raster's own error at no occlusion taken out).
    pub fn transfer(&self, me: u32, p: Vec3, n: Vec3, spacing: f32) -> [f32; 9] {
        let cov = self.coverage(me, p, n, spacing);
        let mut t = [0f32; 9];
        let mut open = [0f32; 9];
        for ((d, om), c) in texels().iter().zip(&cov) {
            let w = n.dot(*d);
            if w <= 0.0 {
                continue;
            }
            let y = sh9(*d);
            for k in 0..9 {
                open[k] += y[k] * w * om;
                t[k] += y[k] * w * om * (1.0 - c);
            }
        }
        let exact = open_transfer(n);
        std::array::from_fn(|k| t[k] + (exact[k] - open[k]))
    }
}

/// Fill the sky transfer of every sample (`Sample::sky`), each looking at
/// the others: one model at a time.
pub fn compute(samples: &mut [Sample]) {
    use rayon::prelude::*;
    let occ = Occluders::new(samples);
    let sky: Vec<[f32; 9]> = samples.par_iter().enumerate().map(|(i, s)| occ.transfer(i as u32, s.position, s.normal, s.area.sqrt())).collect();
    for (s, t) in samples.iter_mut().zip(sky) {
        s.sky = t;
    }
}

/// The sky's radiance by direction, up to a scale, as the shading's sky
/// gives light by facing (`raster::Sky::irradiance_from`): at contrast up to
/// one half, an even sphere blended with an even upper half; above, a lobe
/// overhead that narrows. And the ground's: an even lower half.
pub fn sky_coefficients(contrast: f32) -> ([f32; 9], [f32; 9]) {
    let radiance = sky_radiance(contrast);
    let n = 4096;
    let (mut s, mut g) = ([0f32; 9], [0f32; 9]);
    let w = 4.0 * std::f32::consts::PI / n as f32;
    let golden = std::f32::consts::PI * (3.0 - 5f32.sqrt());
    for i in 0..n {
        let y = 1.0 - (i as f32 + 0.5) / n as f32 * 2.0;
        let r = (1.0 - y * y).max(0.0).sqrt();
        let d = Vec3::new((golden * i as f32).cos() * r, y, (golden * i as f32).sin() * r);
        let b = sh9(d);
        let (ls, lg) = (radiance(d), (d.y < 0.0) as u8 as f32);
        for k in 0..9 {
            s[k] += b[k] * ls * w;
            g[k] += b[k] * lg * w;
        }
    }
    (s, g)
}

/// The sky's radiance by direction for `sky_coefficients`, up to a scale.
pub fn sky_radiance(contrast: f32) -> impl Fn(Vec3) -> f32 + Copy {
    let c = contrast.clamp(0.0, 1.0);
    let k = 1.0 + 6.0 * (c - 0.5);
    move |d: Vec3| -> f32 {
        if c <= 0.5 {
            (1.0 - 2.0 * c) + 2.0 * c * (d.y > 0.0) as u8 as f32
        } else if d.y > 0.0 {
            d.y.powf(k - 1.0)
        } else {
            0.0
        }
    }
}

/// A sample's share of the sky and of the ground's light, 0 to 1, for
/// sky and ground coefficients `l`. A sample without transfer gets 1.
pub fn shares(s: &Sample, l: &([f32; 9], [f32; 9])) -> [f32; 2] {
    if s.sky[0].is_nan() {
        return [1.0, 1.0];
    }
    let open = open_transfer(s.normal);
    let share = |l: &[f32; 9]| {
        let o = dot9(&open, l);
        if o <= 1e-6 { 1.0 } else { (dot9(&s.sky, l) / o).clamp(0.0, 1.0) }
    };
    [share(&l.0), share(&l.1)]
}

/// Each sample's share of the sky and of the ground's light for the
/// current sky, and a grid to find the samples around a point.
pub struct SkyShare {
    /// Per sample: share of the sky, share of the ground's light, ambient
    /// occlusion (the open share of the cosine-weighted hemisphere, any
    /// sky), and the bent normal (where it sees most open sky), x, y, z.
    pub share: Vec<[f32; 6]>,
    /// Per cluster: the object whose samples shade it (`u32::MAX`: none).
    cluster_object: Vec<u32>,
    /// Per object: how far a sample counts, metres.
    radius: Vec<f32>,
    /// Per sample: its object.
    sample_object: Vec<u32>,
    cell: f32,
    /// Open-addressed table of cells: key, first and count in `order`.
    table: Vec<(u64, u32, u32)>,
    order: Vec<u32>,
    /// Per object with samples: its discs, and the sphere round them.
    occluders: Vec<Option<(Occluders, Vec3, f32)>>,
    /// Per cluster, the other objects near enough to hide some of its sky:
    /// `near[near_at[c]..near_at[c + 1]]`.
    near_at: Vec<u32>,
    near: Vec<u32>,
    /// Sky and ground coefficients, and the sky's radiance.
    l: ([f32; 9], [f32; 9]),
    contrast: f32,
}

/// How far past its bounding sphere an object hides sky from others, as a
/// share of that sphere's radius; its effect fades out over the second
/// half of that.
pub const INFLUENCE: f32 = 1.0;


#[inline]
fn cell_key(x: i32, y: i32, z: i32) -> u64 {
    let f = |v: i32| (v + (1 << 20)) as u64 & 0x1f_ffff;
    f(x) | f(y) << 21 | f(z) << 42
}

impl SkyShare {
    /// For the scene's samples with transfer, under a sky of `contrast`.
    pub fn new(scene: &crate::scene::Scene, contrast: f32) -> SkyShare {
        let l = sky_coefficients(contrast);
        let samples = &scene.samples;
        let share: Vec<[f32; 6]> = samples
            .iter()
            .map(|s| {
                let [a, b] = shares(s, &l);
                if s.sky[0].is_nan() {
                    return [a, b, 1.0, s.normal.x, s.normal.y, s.normal.z];
                }
                // Band 0 of the transfer over an open one's is the ambient
                // occlusion; band 1 points where the sky is most open.
                let ao = (s.sky[0] / open_transfer(s.normal)[0]).clamp(0.0, 1.0);
                let bent = Vec3::new(s.sky[3], s.sky[1], s.sky[2]).normalize_or(s.normal);
                [a, b, ao, bent.x, bent.y, bent.z]
            })
            .collect();
        // Objects whose samples carry transfer.
        let mut cluster_object = vec![u32::MAX; scene.clusters.len()];
        let mut radius = vec![0f32; scene.objects.len()];
        let mut sample_object = vec![u32::MAX; samples.len()];
        for (i, o) in scene.objects.iter().enumerate() {
            let r = scene.samples_in(o.clusters.clone());
            if r.is_empty() || samples[r.start].sky[0].is_nan() {
                continue;
            }
            radius[i] = 2.0 * samples[r.start].area.sqrt();
            cluster_object[o.clusters.clone()].fill(i as u32);
            sample_object[r].fill(i as u32);
        }
        let cell = radius.iter().copied().fold(0.0, f32::max).max(1e-3);
        let key_of = |p: Vec3| cell_key((p.x / cell).floor() as i32, (p.y / cell).floor() as i32, (p.z / cell).floor() as i32);
        let mut order: Vec<u32> = (0..samples.len() as u32).filter(|&k| sample_object[k as usize] != u32::MAX).collect();
        order.sort_unstable_by_key(|&k| key_of(samples[k as usize].position));
        let mut runs: Vec<(u64, u32, u32)> = Vec::new();
        for (at, &k) in order.iter().enumerate() {
            let key = key_of(samples[k as usize].position);
            match runs.last_mut() {
                Some(r) if r.0 == key => r.2 += 1,
                _ => runs.push((key, at as u32, 1)),
            }
        }
        let cap = (runs.len() * 2).next_power_of_two().max(16);
        let mut table = vec![(u64::MAX, 0, 0); cap];
        for r in runs {
            let mut h = (r.0.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 32) as usize & (cap - 1);
            while table[h].0 != u64::MAX {
                h = (h + 1) & (cap - 1);
            }
            table[h] = r;
        }
        // Every object with samples hides sky from what is near it.
        let occluders: Vec<Option<(Occluders, Vec3, f32)>> = {
            use rayon::prelude::*;
            scene
                .objects
                .par_iter()
                .map(|o| {
                    let r = scene.samples_in(o.clusters.clone());
                    if r.is_empty() {
                        return None;
                    }
                    let ss = &samples[r];
                    let (mut lo, mut hi) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
                    for x in ss {
                        lo = lo.min(x.position);
                        hi = hi.max(x.position);
                    }
                    let c = (lo + hi) * 0.5;
                    let rad = ss.iter().map(|x| x.position.distance(c)).fold(0.0, f32::max);
                    Some((Occluders::new(ss), c, rad))
                })
                .collect()
        };
        let mut near_at = Vec::with_capacity(scene.clusters.len() + 1);
        let mut near = Vec::new();
        let owner = {
            let mut v = vec![u32::MAX; scene.clusters.len()];
            for (i, o) in scene.objects.iter().enumerate() {
                v[o.clusters.clone()].fill(i as u32);
            }
            v
        };
        for (ci, c) in scene.clusters.iter().enumerate() {
            near_at.push(near.len() as u32);
            for (j, oc) in occluders.iter().enumerate() {
                if let Some((_, oc, or)) = oc {
                    if j as u32 != owner[ci] && c.centre.distance(*oc) <= c.radius + or * (1.0 + INFLUENCE) {
                        near.push(j as u32);
                    }
                }
            }
        }
        near_at.push(near.len() as u32);
        SkyShare { share, cluster_object, radius, sample_object, cell, table, order, occluders, near_at, near, l, contrast }
    }

    #[inline]
    fn run(&self, key: u64) -> &[u32] {
        let cap = self.table.len();
        let mut h = (key.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 32) as usize & (cap - 1);
        loop {
            let (k, a, n) = self.table[h];
            if k == key {
                return &self.order[a as usize..(a + n) as usize];
            }
            if k == u64::MAX {
                return &[];
            }
            h = (h + 1) & (cap - 1);
        }
    }

    /// Whether points of cluster `ci` take a share from samples: their
    /// own object's, or other objects' near them.
    #[inline]
    pub fn covers(&self, ci: usize) -> bool {
        self.cluster_object.get(ci).is_some_and(|&o| o != u32::MAX) || self.near_at.get(ci + 1).is_some_and(|&e| e > self.near_at[ci])
    }

    /// The share other objects leave a point of cluster `ci` at `p` facing
    /// `n`: one minus what their discs hide, over what an open point facing
    /// that way gets, for the sky and for the ground's light.
    pub fn others(&self, ci: usize, p: Vec3, n: Vec3) -> [f32; 2] {
        let Some(&end) = self.near_at.get(ci + 1) else { return [1.0, 1.0] };
        let start = self.near_at[ci];
        if start == end {
            return [1.0, 1.0];
        }
        let open = open_transfer(n);
        let (o_sky, o_ground) = (dot9(&open, &self.l.0), dot9(&open, &self.l.1));
        let sky = sky_radiance(self.contrast);
        let mut out = [1f32, 1.0];
        for &j in &self.near[start as usize..end as usize] {
            let Some((occ, c, r)) = &self.occluders[j as usize] else { continue };
            let past = p.distance(*c) - r;
            let reach = r * INFLUENCE;
            if past >= reach {
                continue;
            }
            let fade = if past <= 0.5 * reach { 1.0 } else { 2.0 * (1.0 - past / reach) };
            let h = occ.hidden(p, n, |d| [sky(d), (d.y < 0.0) as u8 as f32]);
            if o_sky > 1e-6 {
                out[0] *= (1.0 - fade * h[0] / o_sky).clamp(0.0, 1.0);
            }
            if o_ground > 1e-6 {
                out[1] *= (1.0 - fade * h[1] / o_ground).clamp(0.0, 1.0);
            }
        }
        out
    }

    /// The shares at a point of cluster `ci` at `p` facing `n`: the samples
    /// of its object within reach, weighted by distance and by how close
    /// their facing is. Points no sample reaches get the whole sky.
    pub fn at(&self, ci: usize, scene: &crate::scene::Scene, p: Vec3, n: Vec3) -> [f32; 2] {
        let own = self.own(ci, scene, p, n);
        let others = self.others(ci, p, n);
        [own[0] * others[0], own[1] * others[1]]
    }

    /// The share an object's own shape leaves a point of it.
    fn own(&self, ci: usize, scene: &crate::scene::Scene, p: Vec3, n: Vec3) -> [f32; 2] {
        match self.own_all(ci, scene, p, n) {
            Some(v) => [v[0], v[1]],
            None => [1.0, 1.0],
        }
    }

    /// Ambient occlusion at a point: what its object's own shape leaves
    /// open of the cosine-weighted hemisphere, times what other objects
    /// leave. 1 where nothing has samples.
    pub fn ao_at(&self, ci: usize, scene: &crate::scene::Scene, p: Vec3, n: Vec3) -> f32 {
        let own = self.own_all(ci, scene, p, n).map(|v| v[2]).unwrap_or(1.0);
        let mut others = 1.0;
        if let Some(&end) = self.near_at.get(ci + 1) {
            for &j in &self.near[self.near_at[ci] as usize..end as usize] {
                let Some((occ, c, r)) = &self.occluders[j as usize] else { continue };
                let past = p.distance(*c) - r;
                let reach = r * INFLUENCE;
                if past >= reach {
                    continue;
                }
                let fade = if past <= 0.5 * reach { 1.0 } else { 2.0 * (1.0 - past / reach) };
                let h = occ.hidden(p, n, |_| [1.0, 0.0]);
                others *= (1.0 - fade * h[0] / std::f32::consts::PI).clamp(0.0, 1.0);
            }
        }
        own * others
    }

    /// The bent normal at a point of an object with sky transfer: where
    /// its own shape leaves the sky most open. None elsewhere.
    pub fn bent_at(&self, ci: usize, scene: &crate::scene::Scene, p: Vec3, n: Vec3) -> Option<Vec3> {
        self.own_all(ci, scene, p, n).map(|v| Vec3::new(v[3], v[4], v[5]).normalize_or(n))
    }

    /// Everything kept per sample, blended at a point of its object.
    fn own_all(&self, ci: usize, scene: &crate::scene::Scene, p: Vec3, n: Vec3) -> Option<[f32; 6]> {
        let o = match self.cluster_object.get(ci) {
            Some(&o) if o != u32::MAX => o,
            _ => return None,
        };
        let r = self.radius[o as usize];
        let r2 = r * r;
        let lo = ((p - Vec3::splat(r)) / self.cell).floor();
        let hi = ((p + Vec3::splat(r)) / self.cell).floor();
        let mut ws = 0f32;
        let mut sum = [0f32; 6];
        for z in lo.z as i32..=hi.z as i32 {
            for y in lo.y as i32..=hi.y as i32 {
                for x in lo.x as i32..=hi.x as i32 {
                    for &k in self.run(cell_key(x, y, z)) {
                        let k = k as usize;
                        if self.sample_object[k] != o {
                            continue;
                        }
                        let sm = &scene.samples[k];
                        let d2 = sm.position.distance_squared(p);
                        let f = sm.normal.dot(n);
                        if d2 >= r2 || f <= 0.2 {
                            continue;
                        }
                        let a = 1.0 - d2 / r2;
                        let w = a * a * (f - 0.2);
                        ws += w;
                        for (a, v) in sum.iter_mut().zip(&self.share[k]) {
                            *a += w * v;
                        }
                    }
                }
            }
        }
        if ws > 0.0 { Some(sum.map(|v| v / ws)) } else { None }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eval(c: &[f32; 9], d: Vec3) -> f32 {
        dot9(c, &sh9(d))
    }

    /// A function turned by `rotate` gives at a turned direction what it
    /// gave at the direction before.
    #[test]
    fn turning_the_coefficients_turns_the_function() {
        let c = [0.7, -0.3, 0.5, 0.2, 0.4, -0.6, 0.3, 0.1, -0.25];
        for q in [Quat::from_rotation_y(0.7), Quat::from_euler(glam::EulerRot::ZYX, 0.3, -1.2, 2.0), Quat::from_rotation_x(std::f32::consts::PI)] {
            let r = rotate(&c, q);
            for d in [Vec3::X, Vec3::new(0.3, 0.8, -0.5).normalize(), Vec3::new(-0.9, -0.1, 0.4).normalize()] {
                assert!((eval(&r, q * d) - eval(&c, d)).abs() < 1e-4, "{q} {d}");
            }
        }
    }

    /// An open sample gets the whole of any sky; the clamped cosine is right.
    #[test]
    fn an_open_sample_sees_the_whole_sky() {
        let s = vec![Sample { position: Vec3::ZERO, normal: Vec3::new(0.3, 0.9, 0.1).normalize(), area: 1e-4, cluster: 0, sky: [f32::NAN; 9] }];
        let mut s2 = s.clone();
        compute(&mut s2);
        for c in [0.0, 0.5, 1.0] {
            let l = sky_coefficients(c);
            let sh = shares(&s2[0], &l);
            assert!((sh[0] - 1.0).abs() < 1e-4 && (sh[1] - 1.0).abs() < 1e-4, "{sh:?}");
        }
        // Irradiance from an even upper half on an upward face is π.
        let up = dot9(&open_transfer(Vec3::Y), &sky_coefficients(0.5).0);
        assert!((up / std::f32::consts::PI - 1.0).abs() < 0.05, "{up}");
    }

    fn plane(c: Vec3, u: Vec3, v: Vec3, n: Vec3, h: f32) -> Vec<Sample> {
        let k = (u.length() * 2.0 / h) as i32;
        let mut out = Vec::new();
        for i in 0..k {
            for j in 0..k {
                let a = (i as f32 + 0.5) / k as f32 * 2.0 - 1.0;
                let b = (j as f32 + 0.5) / k as f32 * 2.0 - 1.0;
                out.push(Sample { position: c + u * a + v * b, normal: n, area: h * h, cluster: 0, sky: [f32::NAN; 9] });
            }
        }
        out
    }

    /// The floor of a corner between two tall walls loses about half its
    /// sky in the corner and keeps all of it far away; under a low roof it
    /// gets little.
    #[test]
    fn walls_and_roofs_take_their_share_of_the_sky() {
        let h = 0.1;
        let mut s = plane(Vec3::new(0.0, 0.0, 0.0), Vec3::X * 4.0, Vec3::Z * 4.0, Vec3::Y, h);
        let floor = s.len();
        s.extend(plane(Vec3::new(-4.0, 4.0, 0.0), Vec3::Y * 4.0, Vec3::Z * 4.0, Vec3::X, h));
        compute(&mut s);
        let l = sky_coefficients(0.5);
        let at = |s: &[Sample], p: Vec3| s[..floor].iter().min_by(|a, b| a.position.distance(p).total_cmp(&b.position.distance(p))).map(|x| shares(x, &l)[0]).unwrap();
        let by_wall = at(&s, Vec3::new(-3.95, 0.0, 0.0));
        let far = at(&s, Vec3::new(3.9, 0.0, 0.0));
        assert!(by_wall > 0.35 && by_wall < 0.65, "{by_wall}");
        assert!(far > 0.8, "{far}");
        let mut r = plane(Vec3::ZERO, Vec3::X * 3.0, Vec3::Z * 3.0, Vec3::Y, h);
        let n = r.len();
        r.extend(plane(Vec3::new(0.0, 0.3, 0.0), Vec3::X * 3.0, Vec3::Z * 3.0, Vec3::NEG_Y, h));
        compute(&mut r);
        let under = r[..n].iter().filter(|x| x.position.length() < 0.5).map(|x| shares(x, &l)[0]).fold(0.0, f32::max);
        assert!(under < 0.1, "{under}");
    }

    /// Turning a model's transfer gives what looking again from the turned
    /// samples gives, up to the raster's own error.
    #[test]
    fn a_turned_transfer_matches_one_made_turned() {
        let h = 0.1;
        let mut s = plane(Vec3::ZERO, Vec3::X * 2.0, Vec3::Z * 2.0, Vec3::Y, h);
        s.extend(plane(Vec3::new(-2.0, 2.0, 0.0), Vec3::Y * 2.0, Vec3::Z * 2.0, Vec3::X, h));
        compute(&mut s);
        let q = Quat::from_euler(glam::EulerRot::ZYX, 0.9, 0.4, -0.3);
        let mut turned: Vec<Sample> = s.iter().map(|x| Sample { position: q * x.position, normal: q * x.normal, sky: [f32::NAN; 9], ..*x }).collect();
        compute(&mut turned);
        let l = sky_coefficients(0.7);
        let mut err = 0.0;
        for (a, b) in s.iter().zip(&turned) {
            let r = Sample { sky: rotate(&a.sky, q), ..*b };
            err += (shares(&r, &l)[0] - shares(b, &l)[0]).abs();
        }
        let mean = err / s.len() as f32;
        assert!(mean < 0.03, "{mean}");
    }
}
