//! Bounce light from point-based colour bleeding (Christensen 2008). No
//! rays.
//!
//! The surfaces are turned into a cloud of small discs, surfels: one per
//! large triangle, or a few per small cluster, binned by facing. Each surfel
//! knows its area, normal and albedo, and is lit directly (sky and point
//! lights through their shadow maps). That lit cloud is the set of micro
//! lights.
//!
//! The surfels sit in a tree whose nodes carry the summed area, the
//! area-weighted normal and the mean radiance of what is under them. To find
//! the bounce light arriving at a receiver (points spread evenly over the
//! surfaces, 15 cm apart on objects, wider on open ground; `samples.rs`),
//! the
//! tree is walked nearest first: near surfels are taken one by one, far
//! groups as one disc each. Each is laid onto a tiny cube raster around
//! the receiver, 6 x 8 x 8 texels: a small or far disc by its solid angle,
//! a large near one by intersecting each texel's direction with the disc.
//! The raster is filled nearest first, so what is behind a nearer surface
//! is hidden: that is the occlusion, without tracing rays through the scene.
//! A surfel seen from behind sends nothing but still hides what is beyond
//! it, so a receiver under the chess board does not see through it. The raster
//! is then folded into spherical harmonics (order 1, four coefficients per
//! colour). A pixel blends the receivers around its point, by distance and
//! facing, and takes the irradiance for its normal from the blend.
//!
//! Light reach applies: nothing farther from a receiver than the reach of
//! the brightest surfel in the scene is visited, and a light that changes
//! only throws away the receivers within its reach plus that bounce reach.
//! Receivers are refreshed nearest the camera first within a time budget per
//! frame, so dragging a light stays interactive and the bounce catches up.

use std::sync::OnceLock;

use crate::simd::F;

use glam::Vec3;

use crate::scene::Scene;

/// A cluster smaller than this (bounding radius, metres) gives a few
/// surfels binned by facing instead of one per triangle.
const SMALL_CLUSTER: f32 = 0.15;
/// Receivers on objects are spread this far apart (metres), whatever their
/// triangles (`samples.rs`).
const RECEIVER_SPACING: f32 = 0.15;
/// Elsewhere (the ground) they grow apart by this share of their distance
/// from the nearest object: bounce light changes near things, and little
/// across open ground.
const RECEIVER_GROWTH: f32 = 0.25;
/// Receivers an object gets at most: a big leafy one (a tree, its
/// thousands of leaf cards) spreads them wider rather than taking more.
/// Bounce light is soft, and gathering is the cost, receiver by receiver,
/// so a forest of instances would otherwise take minutes.
const MAX_OBJECT_RECEIVERS: f32 = 6000.0;
/// An object's surfels come from the level of detail whose error is under
/// this share of its size too (and at least `SURFEL_LOD_ERROR`): a tree
/// gives a coarser cloud than a chess piece.
const SURFEL_LOD_SHARE: f32 = 0.004;
/// A receiver's light is blended over this many of its spacings.
const RECEIVER_REACH: f32 = 2.0;
/// The surfel cloud is made from the level of detail whose error is under
/// this many metres: bounce light is soft, and finer levels would only add
/// surfels.
const SURFEL_LOD_ERROR: f32 = 0.02;
/// Leaf size of the tree.
const LEAF: usize = 4;
/// A node is taken whole when its radius over its distance is below this.
const FAR: f32 = 0.2;
/// Texels per cube face edge of a receiver's raster.
pub const MICRO: usize = 8;

const Y00: f32 = 0.282_095;
const Y1: f32 = 0.488_603;

#[derive(Clone, Copy, Default)]
pub struct Surfel {
    pub position: Vec3,
    pub normal: Vec3,
    pub area: f32,
    pub albedo: Vec3,
    pub cluster: u32,
}

#[derive(Clone, Copy, Default)]
struct Node {
    centre: Vec3,
    radius: f32,
    /// Children, or (when `leaf`) the first surfel and count.
    a: u32,
    b: u32,
    leaf: bool,
    area: f32,
    /// Area-weighted sum of normals.
    n_sum: Vec3,
    /// Mean outgoing radiance, area-weighted.
    radiance: Vec3,
    /// Largest surfel intensity (radiance x area, cd) below.
    max_intensity: f32,
}

/// Incident radiance at a receiver as order-1 spherical harmonics:
/// coefficient 0, then the x, y, z ones.
pub type Sh = [Vec3; 4];

/// Irradiance (lux per channel) on a surface facing `n` from the SH.
pub fn sh_irradiance(sh: &Sh, n: Vec3) -> Vec3 {
    let a0 = std::f32::consts::PI * Y00;
    let a1 = 2.0 * std::f32::consts::PI / 3.0 * Y1;
    (sh[0] * a0 + (sh[1] * n.x + sh[2] * n.y + sh[3] * n.z) * a1).max(Vec3::ZERO)
}

/// How far a receiver sits off its surface, for a cluster of radius `r`.
fn nudge(r: f32) -> f32 {
    (r * 0.02).min(0.002)
}

/// The six cube faces: (axis, right, up).
const FACES: [(Vec3, Vec3, Vec3); 6] = [
    (Vec3::X, Vec3::NEG_Z, Vec3::Y),
    (Vec3::NEG_X, Vec3::Z, Vec3::Y),
    (Vec3::Y, Vec3::X, Vec3::NEG_Z),
    (Vec3::NEG_Y, Vec3::X, Vec3::Z),
    (Vec3::Z, Vec3::X, Vec3::Y),
    (Vec3::NEG_Z, Vec3::NEG_X, Vec3::Y),
];

/// Direction and solid angle of every texel of the receiver raster.
fn texels() -> &'static [(Vec3, f32)] {
    static T: OnceLock<Vec<(Vec3, f32)>> = OnceLock::new();
    T.get_or_init(|| {
        let mut v = Vec::with_capacity(6 * MICRO * MICRO);
        let s = 2.0 / MICRO as f32;
        for (a, r, u) in FACES {
            for y in 0..MICRO {
                for x in 0..MICRO {
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

/// The texels' directions, a component per array, eight to a lane group.
fn texel_dirs() -> &'static (Vec<F>, Vec<F>, Vec<F>) {
    static T: OnceLock<(Vec<F>, Vec<F>, Vec<F>)> = OnceLock::new();
    T.get_or_init(|| {
        let t = texels();
        let group = |f: fn(&Vec3) -> f32| t.chunks(8).map(|c| F::from(std::array::from_fn::<f32, 8, _>(|j| f(&c[j].0)))).collect();
        (group(|d| d.x), group(|d| d.y), group(|d| d.z))
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
    let m = MICRO as f32;
    let x = (((d.dot(r) / w) * 0.5 + 0.5) * m).floor().clamp(0.0, m - 1.0) as usize;
    let y = (((d.dot(u) / w) * 0.5 + 0.5) * m).floor().clamp(0.0, m - 1.0) as usize;
    f * MICRO * MICRO + y * MICRO + x
}

pub struct Bounce {
    pub surfels: Vec<Surfel>,
    /// Outgoing radiance of each surfel from direct light, cd/m².
    pub radiance: Vec<Vec3>,
    nodes: Vec<Node>,
    /// Receivers: position (just off the surface) and normal.
    pub receivers: Vec<(Vec3, Vec3)>,
    /// Per receiver: how far its light is blended, and the cluster it lies
    /// on.
    pub receiver_radius: Vec<f32>,
    owner: Vec<u32>,
    /// A tree over the receivers' reach, to find those around a cluster.
    rnodes: Vec<RNode>,
    rorder: Vec<u32>,
    /// Receivers by the cells their reach meets, a grid per power of two
    /// of reach (`levels`), to find those around a point.
    grid: CellTable,
    levels: Vec<i32>,
    /// Per cluster: the receivers whose reach meets it, found when it is
    /// first seen (`prepare`).
    near: Vec<Option<Box<[u32]>>>,
    /// Per cluster: which of `levels` its near receivers are at, a bit each.
    near_levels: Vec<u64>,
    /// Per receiver: its grid level's place in `levels`.
    level_of: Vec<u8>,
    /// Bounding radius of each cluster.
    cluster_size: Vec<f32>,
    /// Bounce light at each receiver.
    pub sh: Vec<Sh>,
    /// Per receiver: whether it is current, and the frame it was last
    /// gathered in (0: never).
    pub valid: Vec<bool>,
    pub stamp: Vec<u64>,
    /// Per cluster: the frame its bounce light last changed.
    pub cluster_stamp: Vec<u64>,
}

#[derive(Clone, Copy, Default)]
struct RNode {
    centre: Vec3,
    /// Holds every receiver's reach below.
    radius: f32,
    a: u32,
    b: u32,
    leaf: bool,
}

/// A cell of side 2^l.
#[inline]
fn cell_of(p: Vec3, l: i32) -> [i32; 3] {
    let s = (-l as f32).exp2();
    [(p.x * s).floor() as i32, (p.y * s).floor() as i32, (p.z * s).floor() as i32]
}

#[inline]
fn grid_key(l: i32, c: [i32; 3]) -> u64 {
    let f = |v: i32| (v + (1 << 18)) as u64 & 0x7_ffff;
    ((l + 64) as u64 & 0x7f) << 57 | f(c[0]) | f(c[1]) << 19 | f(c[2]) << 38
}

/// Lists of receivers by key, in an open-addressed table.
#[derive(Default)]
struct CellTable {
    table: Vec<(u64, u32, u32)>,
    items: Vec<u32>,
}

impl CellTable {
    fn new(mut entries: Vec<(u64, u32)>) -> CellTable {
        entries.sort_unstable();
        let mut runs: Vec<(u64, u32, u32)> = Vec::new();
        for (at, &(k, _)) in entries.iter().enumerate() {
            match runs.last_mut() {
                Some(r) if r.0 == k => r.2 += 1,
                _ => runs.push((k, at as u32, 1)),
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
        CellTable { table, items: entries.into_iter().map(|e| e.1).collect() }
    }

    #[inline]
    fn run(&self, key: u64) -> &[u32] {
        let cap = self.table.len();
        if cap == 0 {
            return &[];
        }
        let mut h = (key.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 32) as usize & (cap - 1);
        loop {
            let (k, a, n) = self.table[h];
            if k == key {
                return &self.items[a as usize..(a + n) as usize];
            }
            if k == u64::MAX {
                return &[];
            }
            h = (h + 1) & (cap - 1);
        }
    }
}

fn build_rtree(order: &mut [u32], lo: usize, hi: usize, pos: &[(Vec3, Vec3)], reach: &[f32], nodes: &mut Vec<RNode>) -> u32 {
    let idx = nodes.len() as u32;
    nodes.push(RNode::default());
    let (mut bmin, mut bmax) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
    for &r in &order[lo..hi] {
        bmin = bmin.min(pos[r as usize].0);
        bmax = bmax.max(pos[r as usize].0);
    }
    let centre = (bmin + bmax) * 0.5;
    let radius = order[lo..hi].iter().map(|&r| pos[r as usize].0.distance(centre) + reach[r as usize]).fold(0.0, f32::max);
    let mut node = RNode { centre, radius, ..Default::default() };
    if hi - lo <= LEAF * 2 {
        node.leaf = true;
        node.a = lo as u32;
        node.b = (hi - lo) as u32;
    } else {
        let ext = bmax - bmin;
        let axis = if ext.x >= ext.y && ext.x >= ext.z { 0 } else if ext.y >= ext.z { 1 } else { 2 };
        let mid = (lo + hi) / 2;
        order[lo..hi].select_nth_unstable_by(mid - lo, |a, b| pos[*a as usize].0[axis].total_cmp(&pos[*b as usize].0[axis]));
        node.a = build_rtree(order, lo, mid, pos, reach, nodes);
        node.b = build_rtree(order, mid, hi, pos, reach, nodes);
    }
    nodes[idx as usize] = node;
    idx
}

impl Bounce {
    pub fn new(scene: &Scene) -> Bounce {
        let mut surfels = Vec::new();
        let tri = |cl: &crate::scene::Cluster, t: usize| {
            let [a, b, c] = scene.tri(cl, t as u32);
            let (pa, pb, pc) = (scene.pos(cl, a), scene.pos(cl, b), scene.pos(cl, c));
            let f = (pb - pa).cross(pc - pa);
            ((pa + pb + pc) / 3.0, f * 0.5, (scene.color(a) + scene.color(b) + scene.color(c)) / 3.0)
        };
        // Each cluster's surfel error: its object's share, or the least.
        let mut surfel_error = vec![SURFEL_LOD_ERROR; scene.clusters.len()];
        for o in scene.objects.iter().filter(|o| !o.environment && !o.clusters.is_empty()) {
            surfel_error[o.clusters.clone()].fill(((o.hi - o.lo).length() * SURFEL_LOD_SHARE).max(SURFEL_LOD_ERROR));
        }
        for (ci, c) in scene.clusters.iter().enumerate() {
            // One level of detail only, or surfaces would be counted twice.
            if !c.lod_drawn(1.0, |_, _| 1.0 / surfel_error[ci]) {
                continue;
            }
            if c.radius <= SMALL_CLUSTER {
                // Bin the cluster's triangles by the axis they face most.
                let mut bins = [(Vec3::ZERO, Vec3::ZERO, 0f32, Vec3::ZERO); 6];
                for t in c.first_tri..c.first_tri + c.tri_count {
                    let (cen, an, alb) = tri(c, t as usize);
                    let area = an.length();
                    if area <= 0.0 {
                        continue;
                    }
                    let k = {
                        let a = an.abs();
                        if a.x >= a.y && a.x >= a.z {
                            if an.x >= 0.0 { 0 } else { 1 }
                        } else if a.y >= a.z {
                            if an.y >= 0.0 { 2 } else { 3 }
                        } else if an.z >= 0.0 {
                            4
                        } else {
                            5
                        }
                    };
                    bins[k].0 += cen * area;
                    bins[k].1 += an;
                    bins[k].2 += area;
                    bins[k].3 += alb * area;
                }
                for (p, n, area, alb) in bins {
                    if area > 0.0 {
                        surfels.push(Surfel { position: p / area, normal: n.normalize_or(Vec3::Y), area, albedo: alb / area, cluster: ci as u32 });
                    }
                }
            } else {
                for t in c.first_tri..c.first_tri + c.tri_count {
                    let (cen, an, alb) = tri(c, t as usize);
                    let area = an.length();
                    if area > 0.0 {
                        surfels.push(Surfel { position: cen, normal: an / area, area, albedo: alb, cluster: ci as u32 });
                    }
                }
            }
        }
        // Receivers, spread evenly over the surfaces at the surfels' level
        // of detail, whatever their triangles: RECEIVER_SPACING apart on
        // objects; on the rest, wider with distance from the nearest
        // object.
        let at_lod = |ci: usize| scene.clusters[ci].lod_drawn(1.0, |_, _| 1.0 / surfel_error[ci]);
        let mut object_of = vec![u32::MAX; scene.clusters.len()];
        let mut spheres: Vec<(Vec3, f32)> = Vec::new();
        // The ground and the mountains are open ground, objects or not.
        for (i, o) in scene.objects.iter().enumerate() {
            if o.clusters.is_empty() || o.environment {
                continue;
            }
            object_of[o.clusters.clone()].fill(i as u32);
            spheres.push(((o.lo + o.hi) * 0.5, (o.hi - o.lo).length() * 0.5));
        }
        let tris_of = |cs: &[usize]| -> Vec<(u32, u32)> { cs.iter().flat_map(|&c| (scene.clusters[c].first_tri..scene.clusters[c].first_tri + scene.clusters[c].tri_count).map(move |t| (c as u32, t))).collect() };
        let spread = |cs: &[usize], h: f32| -> Vec<(Vec3, Vec3, f32, u32)> {
            let tris = tris_of(cs);
            let get = |k: usize| {
                let (ci, t) = tris[k];
                let c = &scene.clusters[ci as usize];
                let v = scene.tri(c, t);
                (ci, v.map(|v| scene.pos(c, v)), v.map(|v| scene.normal(c, v)))
            };
            let area: f32 = (0..tris.len()).map(|k| crate::samples::tri_area(get(k).1)).sum();
            let h = h.max((area / MAX_OBJECT_RECEIVERS).sqrt());
            let mut out: Vec<(Vec3, Vec3, f32, u32)> = crate::samples::scatter(tris.len(), get, area, h).into_iter().map(|s| (s.position, s.normal, h, s.cluster)).collect();
            if out.is_empty() && area > 0.0 {
                // Smaller than one spacing: one receiver in its middle.
                let (mut p, mut n) = (Vec3::ZERO, Vec3::ZERO);
                for k in 0..tris.len() {
                    let (_, q, m) = get(k);
                    let a = crate::samples::tri_area(q);
                    p += (q[0] + q[1] + q[2]) / 3.0 * a;
                    n += (m[0] + m[1] + m[2]) * a;
                }
                out.push((p / area, n.normalize_or(Vec3::Y), h, cs[0] as u32));
            }
            out
        };
        let mut recv: Vec<(Vec3, Vec3, f32, u32)> = Vec::new();
        for o in scene.objects.iter().filter(|o| !o.environment) {
            let cs: Vec<usize> = o.clusters.clone().filter(|&c| at_lod(c)).collect();
            if !cs.is_empty() {
                recv.extend(spread(&cs, RECEIVER_SPACING));
            }
        }
        {
            // The rest (the ground): spacing grows with distance from the
            // nearest object, whatever the triangles.
            let rest: Vec<usize> = (0..scene.clusters.len()).filter(|&c| object_of[c] == u32::MAX && at_lod(c)).collect();
            let tris = tris_of(&rest);
            let get = |k: usize| {
                let (ci, t) = tris[k];
                let c = &scene.clusters[ci as usize];
                let v = scene.tri(c, t);
                (ci, v.map(|v| scene.pos(c, v)), v.map(|v| scene.normal(c, v)))
            };
            let h = |p: Vec3| {
                let d = spheres.iter().map(|(sc, sr)| (p.distance(*sc) - sr).max(0.0)).fold(f32::INFINITY, f32::min);
                let d = if d.is_finite() { d } else { p.length() };
                RECEIVER_SPACING + RECEIVER_GROWTH * d
            };
            let spread = crate::samples::scatter_varying(tris.len(), get, h);
            // A cluster smaller than its spacing may get none: it gets one
            // in its middle, so that nothing goes without bounce light.
            let mut has = vec![false; scene.clusters.len()];
            for r in &spread {
                has[r.3 as usize] = true;
            }
            recv.extend(spread);
            for &ci in &rest {
                if !has[ci] {
                    let c = &scene.clusters[ci];
                    let (mut p, mut n, mut a) = (Vec3::ZERO, Vec3::ZERO, 0.0);
                    for t in c.first_tri..c.first_tri + c.tri_count {
                        let q = scene.tri_pos(c, t);
                        let ar = crate::samples::tri_area(q);
                        p += (q[0] + q[1] + q[2]) / 3.0 * ar;
                        n += (q[1] - q[0]).cross(q[2] - q[0]);
                        a += ar;
                    }
                    if a > 0.0 {
                        recv.push((p / a, n.normalize_or(Vec3::Y), h(p / a).min(c.radius.max(RECEIVER_SPACING)), ci as u32));
                    }
                }
            }
        }
        let receivers: Vec<(Vec3, Vec3)> = recv.iter().map(|&(p, n, _, c)| (p + n * nudge(scene.clusters[c as usize].radius), n)).collect();
        let receiver_radius: Vec<f32> = recv.iter().map(|r| r.2 * RECEIVER_REACH).collect();
        let owner: Vec<u32> = recv.iter().map(|r| r.3).collect();
        // Each receiver in the cells its reach meets, at the scale of its
        // reach: at most four cells a side.
        let mut entries: Vec<(u64, u32)> = Vec::new();
        let mut levels: Vec<i32> = Vec::new();
        for (r, (p, _)) in receivers.iter().enumerate() {
            let rr = receiver_radius[r].max(1e-4);
            // Cells half the reach: fewer receivers to try a point.
            let l = rr.log2().ceil() as i32 - 1;
            if !levels.contains(&l) {
                levels.push(l);
            }
            let (a, b) = (cell_of(*p - Vec3::splat(rr), l), cell_of(*p + Vec3::splat(rr), l));
            for z in a[2]..=b[2] {
                for y in a[1]..=b[1] {
                    for x in a[0]..=b[0] {
                        entries.push((grid_key(l, [x, y, z]), r as u32));
                    }
                }
            }
        }
        levels.sort_unstable();
        let level_of: Vec<u8> = receiver_radius.iter().map(|&rr| levels.iter().position(|&l| l == rr.max(1e-4).log2().ceil() as i32 - 1).unwrap_or(0) as u8).collect();
        let grid = CellTable::new(entries);
        let mut rorder: Vec<u32> = (0..receivers.len() as u32).collect();
        let mut rnodes = Vec::with_capacity(receivers.len() / LEAF + 1);
        if !receivers.is_empty() {
            let n = rorder.len();
            build_rtree(&mut rorder, 0, n, &receivers, &receiver_radius, &mut rnodes);
        }

        let mut nodes = Vec::with_capacity(surfels.len() / LEAF * 2 + 1);
        let n = surfels.len();
        build(&mut surfels, 0, n, &mut nodes);
        let radiance = vec![Vec3::ZERO; surfels.len()];
        let k = scene.clusters.len();
        let nr = receivers.len();
        let cluster_size = scene.clusters.iter().map(|c| c.radius).collect();
        Bounce {
            surfels,
            radiance,
            nodes,
            receivers,
            receiver_radius,
            owner,
            rnodes,
            rorder,
            grid,
            levels,
            level_of,
            near: vec![None; k],
            near_levels: vec![0; k],
            cluster_size,
            sh: vec![[Vec3::ZERO; 4]; nr],
            valid: vec![false; nr],
            stamp: vec![0; nr],
            cluster_stamp: vec![0; k],
        }
    }

    /// Find, for each of `clusters` not yet done, the receivers whose
    /// reach meets it.
    pub fn prepare(&mut self, scene: &Scene, clusters: &[u32]) {
        use rayon::prelude::*;
        let todo: Vec<u32> = clusters.iter().copied().filter(|&c| self.near[c as usize].is_none()).collect();
        let found: Vec<(u32, Box<[u32]>)> = todo
            .par_iter()
            .map(|&ci| {
                let c = &scene.clusters[ci as usize];
                let mut out = Vec::new();
                if !self.rnodes.is_empty() {
                    let mut stack = vec![0u32];
                    while let Some(i) = stack.pop() {
                        let nd = &self.rnodes[i as usize];
                        if nd.centre.distance(c.centre) > nd.radius + c.radius {
                            continue;
                        }
                        if nd.leaf {
                            for &r in &self.rorder[nd.a as usize..(nd.a + nd.b) as usize] {
                                if self.receivers[r as usize].0.distance(c.centre) <= self.receiver_radius[r as usize] + c.radius {
                                    out.push(r);
                                }
                            }
                        } else {
                            stack.push(nd.a);
                            stack.push(nd.b);
                        }
                    }
                }
                (ci, out.into_boxed_slice())
            })
            .collect();
        for (c, list) in found {
            self.near_levels[c as usize] = list.iter().fold(0u64, |m, &r| m | 1u64 << self.level_of[r as usize].min(63));
            self.near[c as usize] = Some(list);
        }
    }

    /// The receivers around cluster `ci` (empty until `prepare`).
    pub fn near(&self, ci: usize) -> &[u32] {
        self.near.get(ci).and_then(|n| n.as_deref()).unwrap_or(&[])
    }

    /// Bounce irradiance at `p` facing `n` on cluster `ci`: the receivers
    /// around it that have been gathered, weighted by distance within
    /// their reach and by facing, their spherical harmonics blended.
    pub fn irradiance_at(&self, ci: usize, p: Vec3, n: Vec3) -> Vec3 {
        sh_irradiance(&self.blend(ci, p, n), n)
    }

    /// The blended spherical harmonics behind `irradiance_at`, to be used
    /// again for a neighbouring point.
    pub fn blend(&self, ci: usize, p: Vec3, n: Vec3) -> Sh {
        let mut sh = [Vec3::ZERO; 4];
        let mut ws = 0f32;
        let mut best: Option<(f32, usize)> = None;
        let mut take = |r: usize| {
            if self.stamp[r] == 0 {
                return;
            }
            let (rp, rn) = self.receivers[r];
            let f = rn.dot(n);
            if f <= 0.0 {
                return;
            }
            let d2 = rp.distance_squared(p);
            let rr = self.receiver_radius[r];
            if d2 >= rr * rr {
                if best.is_none_or(|(bd, _)| d2 < bd) {
                    best = Some((d2, r));
                }
                return;
            }
            let a = 1.0 - d2 / (rr * rr);
            let w = a * a * f;
            for k in 0..4 {
                sh[k] += self.sh[r][k] * w;
            }
            ws += w;
        };
        let near = self.near(ci);
        if near.len() <= 24 {
            for &r in near {
                take(r as usize);
            }
        } else {
            // Many around: only those whose reach meets the cell the point
            // is in, at each scale of reach.
            let mut m = self.near_levels[ci];
            while m != 0 {
                let l = self.levels[m.trailing_zeros() as usize];
                m &= m - 1;
                let c = cell_of(p, l);
                for &r in self.grid.run(grid_key(l, c)) {
                    take(r as usize);
                }
            }
        }
        if ws > 0.0 {
            sh.map(|v| v / ws)
        } else {
            best.map(|(_, r)| self.sh[r]).unwrap_or([Vec3::ZERO; 4])
        }
    }

    /// Recompute node radiance sums after surfel radiance changed.
    pub fn refit(&mut self) {
        // Nodes are stored parent before children: walk backwards.
        for i in (0..self.nodes.len()).rev() {
            let nd = self.nodes[i];
            let (rad_area, max_i) = if nd.leaf {
                let mut r = Vec3::ZERO;
                let mut m = 0f32;
                for s in nd.a..nd.a + nd.b {
                    let s = s as usize;
                    r += self.radiance[s] * self.surfels[s].area;
                    m = m.max(self.radiance[s].max_element() * self.surfels[s].area);
                }
                (r, m)
            } else {
                let (l, r) = (self.nodes[nd.a as usize], self.nodes[nd.b as usize]);
                (l.radiance * l.area + r.radiance * r.area, l.max_intensity.max(r.max_intensity))
            };
            let n = &mut self.nodes[i];
            n.radiance = if n.area > 0.0 { rad_area / n.area } else { Vec3::ZERO };
            n.max_intensity = max_i;
        }
    }

    /// Distance beyond which no surfel's bounce can be seen, for `cut` lux.
    pub fn reach(&self, cut: f32) -> f32 {
        let m = self.nodes.first().map(|n| n.max_intensity).unwrap_or(0.0);
        (m / cut.max(1e-12)).sqrt()
    }

    /// Bounce light arriving at receiver `r`.
    pub fn gather(&self, r: usize, reach: f32) -> Sh {
        let (rad, cov) = self.raster(r, reach);
        let tex = texels();
        let mut sh = [Vec3::ZERO; 4];
        for (t, (dir, om)) in tex.iter().enumerate() {
            if cov[t] <= 0.0 {
                continue;
            }
            let l = rad[t] * *om;
            sh[0] += l * Y00;
            sh[1] += l * (Y1 * dir.x);
            sh[2] += l * (Y1 * dir.y);
            sh[3] += l * (Y1 * dir.z);
        }
        sh
    }

    #[cfg_attr(not(target_pointer_width = "64"), allow(dead_code))]
    /// The receiver's cube raster before it is folded: radiance and
    /// coverage per texel, faces in the order of `FACES` (+X, -X, +Y, -Y,
    /// +Z, -Z), each `MICRO` x `MICRO`, rows from the face's lower edge.
    pub fn raster(&self, r: usize, reach: f32) -> ([Vec3; 6 * MICRO * MICRO], [f32; 6 * MICRO * MICRO]) {
        let x = self.receivers[r].0;
        let ci = self.owner[r] as usize;
        let own_size = self.cluster_size[ci].min(0.05);
        let nudge = nudge(self.cluster_size[ci]);
        let tex = texels();
        // Every splat is listed with its distance, then the raster is filled
        // nearest first, so what is behind a nearer surface is hidden.
        let mut splats: Vec<(f32, u16, f32, Vec3)> = Vec::with_capacity(1024);
        let mut stack: Vec<u32> = vec![0];
        while let Some(i) = stack.pop() {
            let nd = &self.nodes[i as usize];
            let to = nd.centre - x;
            let d = to.length();
            if d - nd.radius > reach || nd.area <= 0.0 {
                continue;
            }
            let near_whole = !nd.leaf && d > 0.0 && nd.radius / d < FAR;
            if !nd.leaf && !near_whole {
                // Nearer child last, so it is visited first.
                let (l, r) = (nd.a, nd.b);
                let dl = self.nodes[l as usize].centre.distance(x);
                let dr = self.nodes[r as usize].centre.distance(x);
                if dl < dr {
                    stack.push(r);
                    stack.push(l);
                } else {
                    stack.push(l);
                    stack.push(r);
                }
                continue;
            }
            // Discs to splat: the node whole, or each surfel of a leaf.
            // `disc`: the surfel's normal and radius when it is one surfel,
            // so a large one can be laid on the raster by its true shape.
            let mut splat = |p: Vec3, proj_area: f32, l: Vec3, disc: Option<(Vec3, f32)>| {
                let v = p - x;
                let d2 = v.length_squared();
                if d2 < 1e-12 || proj_area <= 0.0 {
                    return;
                }
                let d = d2.sqrt();
                let dir = v / d;
                let omega = (proj_area / d2).min(2.0 * std::f32::consts::PI);
                let alpha = (omega / std::f32::consts::PI).sqrt();
                let t0 = texel_of(dir);
                if omega <= tex[t0].1 {
                    // Smaller than a texel: partial cover of one texel.
                    splats.push((d, t0 as u16, omega / tex[t0].1, l));
                } else if let Some((n, r)) = disc {
                    // A large surfel: each texel looks along its direction
                    // and is covered where that line meets the disc, which
                    // is enlarged by a third so that the discs of a tiled
                    // surface leave no gaps between them.
                    // Eight texels at a time.
                    let mut any = false;
                    let (tx, ty, tz) = texel_dirs();
                    let pn = F::splat((p - x).dot(n));
                    let lim = F::splat((r * 1.33) * (r * 1.33));
                    let q = p - x;
                    for g in 0..tx.len() {
                        let den = tx[g] * n.x + ty[g] * n.y + tz[g] * n.z;
                        let s = pn / den;
                        let (ex, ey, ez) = (tx[g] * s - q.x, ty[g] * s - q.y, tz[g] * s - q.z);
                        let hit = den.simd_lt(F::ZERO) & s.simd_gt(F::ZERO) & (ex * ex + ey * ey + ez * ez).simd_le(lim);
                        let mut m = hit.to_bitmask();
                        if m == 0 {
                            continue;
                        }
                        let s = s.to_array();
                        while m != 0 {
                            let j = m.trailing_zeros() as usize;
                            m &= m - 1;
                            splats.push((s[j], (g * 8 + j) as u16, 1.0, l));
                        }
                        any = true;
                    }
                    if !any {
                        splats.push((d, t0 as u16, 1.0, l));
                    }
                } else {
                    // A large disc covers the texels whose centres lie inside
                    // its cone, widened by half so that neighbouring discs
                    // leave no gaps for light behind them to leak through.
                    // The price: a surface's outline counts a little more
                    // light than it should.
                    let cos_a = (alpha * 1.5).min(std::f32::consts::FRAC_PI_2).cos();
                    let mut any = false;
                    let (tx, ty, tz) = texel_dirs();
                    let ca = F::splat(cos_a);
                    for g in 0..tx.len() {
                        let mut m = (tx[g] * dir.x + ty[g] * dir.y + tz[g] * dir.z).simd_ge(ca).to_bitmask();
                        while m != 0 {
                            let j = m.trailing_zeros() as usize;
                            m &= m - 1;
                            splats.push((d, (g * 8 + j) as u16, 1.0, l));
                            any = true;
                        }
                    }
                    if !any {
                        splats.push((d, t0 as u16, 1.0, l));
                    }
                }
            };
            if nd.leaf {
                for s in nd.a..nd.a + nd.b {
                    let sf = &self.surfels[s as usize];
                    // Distance from the receiver to the surfel's plane, on
                    // its front side when positive.
                    let h = sf.normal.dot(x - sf.position);
                    let d = sf.position.distance(x);
                    let radius = (sf.area / std::f32::consts::PI).sqrt();
                    // The surface the receiver sits on: skip it.
                    if sf.cluster as usize == ci && h.abs() <= nudge * 1.5 {
                        continue;
                    }
                    if h > 0.0 {
                        splat(sf.position, sf.area * h / d, self.radiance[s as usize], Some((sf.normal, radius)));
                    } else if d > own_size * 2.0 {
                        // Seen from behind: the inside of something. It sends
                        // no light this way but hides what is beyond it.
                        // Within twice the receiver's cluster size (at most
                        // 10 cm) it is more likely the curve of the
                        // receiver's own surface, so it is left out there.
                        splat(sf.position, sf.area * -h / d, Vec3::ZERO, Some((-sf.normal, radius)));
                    }
                }
            } else {
                let proj = nd.n_sum.dot(-to / d.max(1e-12));
                splat(nd.centre, proj, nd.radiance, None);
            }
        }
        splats.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));


        let mut rad = [Vec3::ZERO; 6 * MICRO * MICRO];
        let mut cov = [0f32; 6 * MICRO * MICRO];
        for &(_, t, frac, l) in &splats {
            let t = t as usize;
            let free = 1.0 - cov[t];
            if free <= 0.0 {
                continue;
            }
            let add = frac.min(free);
            rad[t] += l * add;
            cov[t] += add;
        }
        (rad, cov)
    }

}

/// Build the tree over `surfels[lo..hi]`, reordering them; returns the node
/// index. Parents come before their children in `nodes`.
fn build(surfels: &mut [Surfel], lo: usize, hi: usize, nodes: &mut Vec<Node>) -> u32 {
    let idx = nodes.len() as u32;
    nodes.push(Node::default());
    let s = &surfels[lo..hi];
    let (mut bmin, mut bmax) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
    let mut area = 0.0;
    let mut n_sum = Vec3::ZERO;
    for sf in s {
        bmin = bmin.min(sf.position);
        bmax = bmax.max(sf.position);
        area += sf.area;
        n_sum += sf.normal * sf.area;
    }
    let centre = (bmin + bmax) * 0.5;
    let radius = s.iter().map(|sf| sf.position.distance(centre) + (sf.area / std::f32::consts::PI).sqrt()).fold(0.0, f32::max);
    let mut node = Node { centre, radius, area, n_sum, ..Default::default() };
    if hi - lo <= LEAF {
        node.leaf = true;
        node.a = lo as u32;
        node.b = (hi - lo) as u32;
    } else {
        let ext = bmax - bmin;
        let axis = if ext.x >= ext.y && ext.x >= ext.z { 0 } else if ext.y >= ext.z { 1 } else { 2 };
        let mid = (lo + hi) / 2;
        surfels[lo..hi].select_nth_unstable_by(mid - lo, |a, b| a.position[axis].total_cmp(&b.position[axis]));
        node.a = build(surfels, lo, mid, nodes);
        node.b = build(surfels, mid, hi, nodes);
    }
    nodes[idx as usize] = node;
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quad(s: &mut Scene, c: Vec3, u: Vec3, v: Vec3, col: Vec3, n: usize) {
        // An n x n grid of quads spanning c +- u, c +- v, facing u x v.
        let mut p = Vec::new();
        let mut idx = Vec::new();
        for j in 0..=n {
            for i in 0..=n {
                let (a, b) = (i as f32 / n as f32 * 2.0 - 1.0, j as f32 / n as f32 * 2.0 - 1.0);
                p.push(c + u * a + v * b);
            }
        }
        let w = (n + 1) as u32;
        for j in 0..n as u32 {
            for i in 0..n as u32 {
                let a = j * w + i;
                idx.extend_from_slice(&[a, a + 1, a + w + 1, a, a + w + 1, a + w]);
            }
        }
        s.push_mesh(&p, &[], &[col], &idx);
    }

    #[test]
    fn sh_of_a_uniform_hemisphere_gives_its_irradiance() {
        // Radiance 1 over the whole sky above: irradiance on an upward
        // surface is π; on a downward one, 0 (order 1 only approximates).
        let mut sh = [Vec3::ZERO; 4];
        for (dir, om) in texels() {
            if dir.y > 0.0 {
                sh[0] += Vec3::ONE * (Y00 * om);
                sh[1] += Vec3::ONE * (Y1 * dir.x * om);
                sh[2] += Vec3::ONE * (Y1 * dir.y * om);
                sh[3] += Vec3::ONE * (Y1 * dir.z * om);
            }
        }
        let up = sh_irradiance(&sh, Vec3::Y).x;
        let down = sh_irradiance(&sh, -Vec3::Y).x;
        assert!((up - std::f32::consts::PI).abs() < 0.3, "{up}");
        assert!(down < 0.3, "{down}");
        let total: f32 = texels().iter().map(|t| t.1).sum();
        assert!((total - 4.0 * std::f32::consts::PI).abs() < 0.05, "{total}");
    }

    /// A bright red wall beside a white floor: the floor near the wall
    /// receives red light; a floor patch behind a second, dark wall does
    /// not see the red one.
    #[test]
    fn red_wall_bleeds_onto_the_floor_and_walls_block_it() {
        let mut s = Scene::new();
        // Floor 4 x 4 m at y = 0, facing up, made of 1 m patches.
        quad(&mut s, Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -2.0), Vec3::splat(0.8), 4);
        // Red wall at x = -2, facing +x.
        quad(&mut s, Vec3::new(-2.0, 1.0, 0.0), Vec3::new(0.0, 0.0, -2.0), Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.9, 0.05, 0.05), 4);
        let mut b = Bounce::new(&s);
        for i in 0..b.surfels.len() {
            // Light the red wall only, as if a lamp shone on it.
            b.radiance[i] = if b.surfels[i].normal.x > 0.9 { Vec3::new(10.0, 0.5, 0.5) } else { Vec3::ZERO };
        }
        b.refit();
        // Floor clusters: receivers near the wall see red from the side.
        // (Receivers at the very foot of the wall are in its plane and see
        // it edge-on; those from half a metre out are checked.)
        let floor: Vec<usize> = (0..b.receivers.len()).filter(|&r| b.receivers[r].0.y < 0.1 && b.receivers[r].1.y > 0.9 && b.receivers[r].0.x > -1.5).collect();
        assert!(!floor.is_empty());
        for &r in &floor {
            let e = sh_irradiance(&b.gather(r, 100.0), Vec3::Y);
            assert!(e.x > 0.5, "floor receiver {r} at {} gets {e}", b.receivers[r].0);
            assert!(e.x > e.y * 5.0, "red: {e}");
        }

        // Put a dark wall between the red wall and a receiver.
        let mut s2 = Scene::new();
        quad(&mut s2, Vec3::new(2.0, 0.0, 0.0), Vec3::new(0.3, 0.0, 0.0), Vec3::new(0.0, 0.0, -0.3), Vec3::splat(0.8), 1);
        quad(&mut s2, Vec3::new(-2.0, 1.0, 0.0), Vec3::new(0.0, 0.0, -2.0), Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.9, 0.05, 0.05), 4);
        quad(&mut s2, Vec3::new(1.0, 1.0, 0.0), Vec3::new(0.0, 0.0, -2.5), Vec3::new(0.0, 1.5, 0.0), Vec3::splat(0.1), 4);
        let mut b2 = Bounce::new(&s2);
        for i in 0..b2.surfels.len() {
            let red = b2.surfels[i].position.x < -1.9;
            b2.radiance[i] = if red { Vec3::new(10.0, 0.5, 0.5) } else { Vec3::ZERO };
        }
        b2.refit();
        let patch = (0..s2.clusters.len()).find(|&c| s2.clusters[c].centre.x > 1.5).unwrap();
        b2.prepare(&s2, &[patch as u32]);
        let at = s2.clusters[patch].centre;
        let e = b2.irradiance_at(patch, at, Vec3::Y);
        assert_eq!(e, Vec3::ZERO, "nothing gathered yet");
        let mine: Vec<u32> = b2.near(patch).to_vec();
        assert!(!mine.is_empty());
        for r in mine {
            b2.sh[r as usize] = b2.gather(r as usize, 100.0);
            b2.stamp[r as usize] = 1;
        }
        let e = b2.irradiance_at(patch, at, Vec3::Y);
        assert!(e.x < 0.05, "the dark wall hides the red one: {e}");
    }

    /// Nothing beyond the reach is visited: a receiver far away gets
    /// nothing, however bright the surfels.
    #[test]
    fn bounce_stops_at_its_reach() {
        let mut s = Scene::new();
        quad(&mut s, Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0), Vec3::splat(0.8), 2);
        quad(&mut s, Vec3::new(0.0, 0.0, -500.0), Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), Vec3::splat(0.8), 1);
        let mut b = Bounce::new(&s);
        for i in 0..b.surfels.len() {
            b.radiance[i] = Vec3::splat(100.0);
        }
        b.refit();
        let cut = 0.1;
        let r = b.reach(cut);
        assert!(r < 400.0, "{r}");
        let far: Vec<usize> = (0..b.receivers.len()).filter(|&k| b.receivers[k].0.z < -400.0).collect();
        assert!(!far.is_empty());
        for k in far {
            assert_eq!(sh_irradiance(&b.gather(k, r), Vec3::Z), Vec3::ZERO);
        }
    }

    /// Energy: a receiver 1 m above a wide floor of radiance 1, facing it,
    /// receives π lux (the whole hemisphere), within 15%.
    #[test]
    fn a_wide_floor_gives_a_full_hemisphere() {
        let mut s = Scene::new();
        quad(&mut s, Vec3::ZERO, Vec3::new(200.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -200.0), Vec3::splat(0.5), 80);
        quad(&mut s, Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.05, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.05), Vec3::splat(0.5), 1);
        let mut b = Bounce::new(&s);
        for i in 0..b.surfels.len() {
            b.radiance[i] = if b.surfels[i].position.y < 0.5 { Vec3::ONE } else { Vec3::ZERO };
        }
        b.refit();
        // A receiver of the card, whose face looks down.
        let r = (0..b.receivers.len()).find(|&r| b.receivers[r].0.y > 0.9 && b.receivers[r].1.y < -0.9).unwrap();
        let e = sh_irradiance(&b.gather(r, 1000.0), -Vec3::Y).x;
        assert!((e / std::f32::consts::PI - 1.0).abs() < 0.15, "{e}");
        // And facing up, it gets almost nothing from below.
        let up = sh_irradiance(&b.gather(r, 1000.0), Vec3::Y).x;
        assert!(up < 0.35, "{up}");
    }

    /// Receivers follow area, not triangles: an object made of two
    /// triangles gets them 15 cm apart, as one of thousands does; ground
    /// that is no object gets them closer near an object than far from it.
    #[test]
    fn receivers_are_spread_by_area_not_triangles() {
        let count_on = |n: usize| {
            let mut s = Scene::new();
            quad(&mut s, Vec3::new(0.0, 1.0, 0.0), Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0), Vec3::splat(0.5), n);
            let k = s.clusters.len();
            s.add_object("card".into(), 0..k, Vec3::ZERO, Vec3::ZERO);
            Bounce::new(&s).receivers.len()
        };
        let (coarse, fine) = (count_on(1), count_on(40));
        assert!(coarse > 60, "{coarse}");
        assert!((coarse as f32 / fine as f32 - 1.0).abs() < 0.1, "{coarse} on 2 triangles, {fine} on 3,200");
        // Ground of 20 m quads round a small object at the origin.
        let mut s = Scene::new();
        quad(&mut s, Vec3::ZERO, Vec3::new(60.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -60.0), Vec3::splat(0.5), 6);
        let first = s.clusters.len();
        quad(&mut s, Vec3::new(0.0, 0.5, 0.0), Vec3::new(0.3, 0.0, 0.0), Vec3::new(0.0, 0.0, -0.3), Vec3::splat(0.5), 1);
        let k = s.clusters.len();
        s.add_object("box".into(), first..k, Vec3::ZERO, Vec3::ZERO);
        let b = Bounce::new(&s);
        let ground = |lo: f32, hi: f32| (0..b.receivers.len()).filter(|&r| b.receivers[r].0.y < 0.1 && { let d = b.receivers[r].0.length(); d >= lo && d < hi }).count() as f32 / (std::f32::consts::PI * (hi * hi - lo * lo));
        let (near, far) = (ground(0.0, 8.0), ground(40.0, 55.0));
        assert!(near > far * 4.0, "{near} per m² near, {far} far");
    }
}
