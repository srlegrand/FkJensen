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
//! the bounce light arriving at a receiver (one per vertex of a large
//! cluster, one for a small one), the
//! tree is walked nearest first: near surfels are taken one by one, far
//! groups as one disc each. Each is laid onto a tiny cube raster around
//! the receiver, 6 x 8 x 8 texels: a small or far disc by its solid angle,
//! a large near one by intersecting each texel's direction with the disc.
//! The raster is filled nearest first, so what is behind a nearer surface
//! is hidden: that is the occlusion, without tracing rays through the scene.
//! A surfel seen from behind sends nothing but still hides what is beyond
//! it, so a receiver under the chess board does not see through it. The raster
//! is then folded into spherical harmonics (order 1, four coefficients per
//! colour), from which the irradiance for any normal of the cluster comes
//! cheaply when the pixel is resolved.
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
/// Mean triangle area (m²) from which a cluster's receivers are its
/// vertices.
const LARGE_TRIANGLE: f32 = 0.01;
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

/// Which of the six axis directions `n` is closest to.
fn facing(n: Vec3) -> usize {
    let a = n.abs();
    if a.x >= a.y && a.x >= a.z {
        if n.x >= 0.0 { 0 } else { 1 }
    } else if a.y >= a.z {
        if n.y >= 0.0 { 2 } else { 3 }
    } else if n.z >= 0.0 {
        4
    } else {
        5
    }
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
    /// Receivers, cluster by cluster: position and mean normal.
    pub receivers: Vec<(Vec3, Vec3)>,
    /// Cluster `c`'s receivers are `first_receiver[c]..first_receiver[c + 1]`.
    pub first_receiver: Vec<u32>,
    /// Whether a cluster's receivers are its vertices (else one per facing).
    per_vertex: Vec<bool>,
    /// For vertices of such clusters, their receiver.
    vertex_receiver: Vec<u32>,
    /// Bounding radius of each cluster.
    cluster_size: Vec<f32>,
    /// Bounce light at each receiver.
    pub sh: Vec<Sh>,
    /// Per cluster: whether its receivers are current, and the frame they
    /// were last gathered in.
    pub valid: Vec<bool>,
    pub stamp: Vec<u64>,
}

impl Bounce {
    pub fn new(scene: &Scene) -> Bounce {
        let mut surfels = Vec::new();
        let tri = |t: usize| {
            let i = t * 3;
            let (a, b, c) = (scene.indices[i] as usize, scene.indices[i + 1] as usize, scene.indices[i + 2] as usize);
            let (pa, pb, pc) = (scene.positions[a], scene.positions[b], scene.positions[c]);
            let f = (pb - pa).cross(pc - pa);
            ((pa + pb + pc) / 3.0, f * 0.5, (scene.colors[a] + scene.colors[b] + scene.colors[c]) / 3.0)
        };
        // Every cluster's surface binned by the axis it faces most: its
        // receivers, and, for small clusters, its surfels.
        let bins_of = |c: &crate::scene::Cluster| {
            let mut bins = [(Vec3::ZERO, Vec3::ZERO, 0f32, Vec3::ZERO); 6];
            for t in c.first_tri..c.first_tri + c.tri_count {
                let (cen, an, alb) = tri(t as usize);
                let area = an.length();
                if area <= 0.0 {
                    continue;
                }
                let k = facing(an);
                bins[k].0 += cen * area;
                bins[k].1 += an;
                bins[k].2 += area;
                bins[k].3 += alb * area;
            }
            bins
        };
        let all_bins: Vec<[(Vec3, Vec3, f32, Vec3); 6]> = scene.clusters.iter().map(bins_of).collect();
        for (ci, c) in scene.clusters.iter().enumerate() {
            // One level of detail only, or surfaces would be counted twice.
            if !c.lod_drawn(1.0, |_, _| 1.0 / SURFEL_LOD_ERROR) {
                continue;
            }
            if c.radius <= SMALL_CLUSTER {
                // Bin the cluster's triangles by the axis they face most.
                let mut bins = [(Vec3::ZERO, Vec3::ZERO, 0f32, Vec3::ZERO); 6];
                for t in c.first_tri..c.first_tri + c.tri_count {
                    let (cen, an, alb) = tri(t as usize);
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
                    let (cen, an, alb) = tri(t as usize);
                    let area = an.length();
                    if area > 0.0 {
                        surfels.push(Surfel { position: cen, normal: an / area, area, albedo: alb, cluster: ci as u32 });
                    }
                }
            }
        }
        // Receivers sit off their surface along its normal by 2% of the
        // cluster's size, at most 2 mm.
        let acc: Vec<[(Vec3, Vec3, f32); 6]> = all_bins.iter().map(|b| b.map(|(p, n, a, _)| (p, n.normalize_or(Vec3::Y) * a, a))).collect();
        // A large cluster (big triangles: a floor, a wall, a table top) gets
        // a receiver at each of its vertices instead, and a pixel blends the
        // three of its triangle, so bounce light varies across the surface.
        let mut receivers: Vec<(Vec3, Vec3)> = Vec::new();
        let mut first_receiver = Vec::with_capacity(scene.clusters.len() + 1);
        let mut per_vertex = Vec::with_capacity(scene.clusters.len());
        let mut vertex_receiver = vec![u32::MAX; scene.positions.len()];
        for (bins, c) in acc.iter().zip(&scene.clusters) {
            first_receiver.push(receivers.len() as u32);
            // Large triangles (a floor, a wall, a table top) get a receiver
            // per vertex; a cluster of small ones (a model) gets one.
            let area: f32 = (c.first_tri..c.first_tri + c.tri_count)
                .map(|t| {
                    let i = t as usize * 3;
                    let (a, b, cc) = (scene.positions[scene.indices[i] as usize], scene.positions[scene.indices[i + 1] as usize], scene.positions[scene.indices[i + 2] as usize]);
                    (b - a).cross(cc - a).length() * 0.5
                })
                .sum::<f32>()
                / c.tri_count.max(1) as f32;
            let big = c.radius > SMALL_CLUSTER && area >= LARGE_TRIANGLE;
            per_vertex.push(big);
            if big {
                // Vertices that share a position and normal (the corners of
                // neighbouring quads) share a receiver.
                let start = receivers.len();
                for v in c.first_vertex..c.first_vertex + c.vertex_count {
                    let n = scene.normals[v as usize];
                    let p = scene.positions[v as usize] + n * nudge(c.radius);
                    let found = (start..receivers.len()).find(|&r| receivers[r].0 == p && receivers[r].1 == n);
                    vertex_receiver[v as usize] = match found {
                        Some(r) => r as u32,
                        None => {
                            receivers.push((p, n));
                            (receivers.len() - 1) as u32
                        }
                    };
                }
            } else {
                // A small cluster: one receiver at the centre of its surfels.
                // Its spherical harmonics hold every direction, so the
                // surface's own curve picks the right light at each pixel.
                let (mut p, mut n, mut a) = (Vec3::ZERO, Vec3::ZERO, 0.0);
                for (bp, bn, ba) in bins {
                    p += *bp;
                    n += *bn;
                    a += *ba;
                }
                if a > 0.0 {
                    let n = n.normalize_or(Vec3::Y);
                    receivers.push((p / a + n * nudge(c.radius), n));
                }
            }
        }
        first_receiver.push(receivers.len() as u32);

        let mut nodes = Vec::with_capacity(surfels.len() / LEAF * 2 + 1);
        let n = surfels.len();
        build(&mut surfels, 0, n, &mut nodes);
        let radiance = vec![Vec3::ZERO; surfels.len()];
        let k = scene.clusters.len();
        let nr = receivers.len();
        let cluster_size = scene.clusters.iter().map(|c| c.radius).collect();
        Bounce { surfels, radiance, nodes, receivers, first_receiver, per_vertex, vertex_receiver, cluster_size, sh: vec![[Vec3::ZERO; 4]; nr], valid: vec![false; k], stamp: vec![0; k] }
    }

    pub fn receivers_of(&self, ci: usize) -> std::ops::Range<usize> {
        self.first_receiver[ci] as usize..self.first_receiver[ci + 1] as usize
    }

    /// Bounce irradiance at a point of triangle `tri` of cluster `ci` with
    /// weights `w` over its vertices, facing `n`: blended from the
    /// triangle's vertex receivers, or from the cluster's receiver whose
    /// facing is closest.
    pub fn irradiance_at(&self, scene: &Scene, ci: usize, tri: u32, w: [f32; 3], n: Vec3) -> Vec3 {
        if self.per_vertex[ci] {
            let t = tri as usize * 3;
            let mut e = Vec3::ZERO;
            for k in 0..3 {
                let r = self.vertex_receiver[scene.indices[t + k] as usize] as usize;
                e += sh_irradiance(&self.sh[r], n) * w[k];
            }
            return e;
        }
        self.irradiance(ci, n)
    }

    /// Bounce irradiance on a surface of cluster `ci` facing `n`, from the
    /// receiver of that cluster whose facing is closest.
    pub fn irradiance(&self, ci: usize, n: Vec3) -> Vec3 {
        let mut best: Option<(f32, usize)> = None;
        for r in self.receivers_of(ci) {
            let d = self.receivers[r].1.dot(n);
            if best.is_none_or(|(bd, _)| d > bd) {
                best = Some((d, r));
            }
        }
        best.map(|(_, r)| sh_irradiance(&self.sh[r], n)).unwrap_or(Vec3::ZERO)
    }

    /// Gather bounce light at every receiver of cluster `ci`.
    pub fn gather_cluster(&self, ci: usize, reach: f32) -> Vec<Sh> {
        self.receivers_of(ci).map(|r| self.gather(r, ci, reach)).collect()
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

    /// Bounce light arriving at receiver `r` of cluster `ci`.
    pub fn gather(&self, r: usize, ci: usize, reach: f32) -> Sh {
        let (rad, cov) = self.raster(r, ci, reach);
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
    pub fn raster(&self, r: usize, ci: usize, reach: f32) -> ([Vec3; 6 * MICRO * MICRO], [f32; 6 * MICRO * MICRO]) {
        let x = self.receivers[r].0;
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
            let ci = (0..s.clusters.len()).find(|&c| b.receivers_of(c).contains(&r)).unwrap();
            let e = sh_irradiance(&b.gather(r, ci, 100.0), Vec3::Y);
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
        let patch = (0..s2.clusters.len()).find(|&c| b2.receivers[b2.receivers_of(c).start].0.x > 1.5).unwrap();
        let e = b2.irradiance(patch, Vec3::Y);
        let mut b2 = b2;
        let sh = b2.gather_cluster(patch, 100.0);
        for (k, r) in b2.receivers_of(patch).enumerate() {
            b2.sh[r] = sh[k];
        }
        assert_eq!(e, Vec3::ZERO, "nothing gathered yet");
        let e = b2.irradiance(patch, Vec3::Y);
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
        let far = (0..s.clusters.len()).find(|&c| b.receivers[b.receivers_of(c).start].0.z < -400.0).unwrap();
        for sh in b.gather_cluster(far, r) {
            assert_eq!(sh_irradiance(&sh, Vec3::Z), Vec3::ZERO);
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
        let ci = (0..s.clusters.len()).find(|&c| b.receivers[b.receivers_of(c).start].0.y > 0.9).unwrap();
        // The receiver of the card's face that looks down.
        let r = b.receivers_of(ci).find(|&r| b.receivers[r].1.y < -0.9).unwrap();
        let e = sh_irradiance(&b.gather(r, ci, 1000.0), -Vec3::Y).x;
        assert!((e / std::f32::consts::PI - 1.0).abs() < 0.15, "{e}");
        // And facing up, it gets almost nothing from below.
        let up = sh_irradiance(&b.gather(r, ci, 1000.0), Vec3::Y).x;
        assert!(up < 0.35, "{up}");
    }
}
