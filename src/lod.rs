//! Levels of detail for clustered meshes, built once at import (the cluster
//! hierarchy of Nanite: Karis, Stubbs, Wihlidal, SIGGRAPH 2021, after
//! Cignoni et al.'s Batched Multi-Triangulation, 2005).
//!
//! Level 0 is the mesh cut into clusters. To make level 1, clusters are
//! gathered into groups of about four neighbours; each group is simplified
//! to about half its triangles with its outer edges locked, so it still
//! meets its neighbours whatever level they are drawn at; and the result is
//! cut into clusters again. Repeat until a level has one group or nothing
//! simplifies further. Every cluster then carries two errors: its own (how
//! far its group strays from the mesh as made, plus what its children
//! strayed) and its replacement's. The renderer draws a cluster when its own
//! error looks small enough and its replacement's does not
//! (`Cluster::lod_drawn`), which picks exactly one level for each stretch of
//! surface, crack-free.
//!
//! Simplification collapses edges by quadric error (Garland, Heckbert
//! 1997), moving a vertex onto a neighbour, so every vertex of every level
//! is a vertex of the mesh as made, with its normal, colour and texture
//! coordinates. A collapse is refused when it would flip or crush a
//! triangle or tear the surface. Vertices shared with other groups or on
//! open edges never move. Seams (one position, several vertices: texture
//! coordinate or normal seams) may move along themselves, both sides
//! together, never off.
//!
//! Quadrics only order the collapses. The error of a group is measured
//! after: the largest distance, in metres, from a vertex of the group
//! before simplifying to the nearest triangle after.

#![allow(dead_code)]

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use glam::Vec3;
use rayon::prelude::*;

use crate::scene::build_clusters;

/// Clusters gathered into one group.
pub const GROUP: usize = 4;
/// A group must lose at least this share of its triangles to make a level.
const MIN_REDUCTION: f32 = 0.15;

/// One cluster of the hierarchy: triangles as vertex ids of the mesh as
/// made, the input triangle each came from, and its level of detail.
#[derive(Clone, Debug)]
pub struct LodCluster {
    pub tris: Vec<[u32; 3]>,
    pub source: Vec<u32>,
    pub level: u8,
    pub error: f32,
    pub sphere: (Vec3, f32),
    pub parent_error: f32,
    pub parent_sphere: (Vec3, f32),
}

/// What the build did.
#[derive(Debug, Default)]
pub struct LodStats {
    /// Triangles made by each pass (the first: the mesh as made).
    pub tris: Vec<usize>,
    /// Clusters made by each pass.
    pub clusters: Vec<usize>,
}

/// Build the hierarchy of a mesh. `tris` index `positions`.
pub fn build(positions: &[Vec3], tris: &[[u32; 3]]) -> (Vec<LodCluster>, LodStats) {
    build_logged(positions, tris, &|_| {})
}

/// `build`, telling `log` how each level went.
pub fn build_logged(positions: &[Vec3], tris: &[[u32; 3]], log: &dyn Fn(String)) -> (Vec<LodCluster>, LodStats) {
    let t0 = std::time::Instant::now();
    let mut stats = LodStats::default();
    if tris.is_empty() {
        return (Vec::new(), stats);
    }
    // One id per distinct position, so that seams (vertices split for
    // texture coordinates or normals) still join for adjacency, and are
    // locked.
    let (pos_id, _) = weld(positions);

    let flat: Vec<u32> = tris.iter().flatten().copied().collect();
    let mut out: Vec<LodCluster> = build_clusters(positions, &flat)
        .into_iter()
        .map(|b| {
            let ctris: Vec<[u32; 3]> = b.local_tris.iter().map(|t| t.map(|l| b.vertices[l as usize])).collect();
            let sphere = bounding_sphere(positions, &ctris);
            LodCluster { tris: ctris, source: b.source_tris, level: 0, error: 0.0, sphere, parent_error: f32::INFINITY, parent_sphere: (Vec3::ZERO, 0.0) }
        })
        .collect();
    stats.tris.push(tris.len());
    stats.clusters.push(out.len());

    // Clusters of the current level still to be grouped, as indices into
    // `out`.
    let mut level: Vec<usize> = (0..out.len()).collect();
    let mut lvl = 0u8;
    while level.len() > 1 && lvl < 64 {
        let tl = std::time::Instant::now();
        let groups = make_groups(&out, &level, positions, &pos_id);
        let t_group = tl.elapsed().as_secs_f32();
        // Which group uses each position at this level: a position used by
        // two groups is on a border and stays put.
        let mut owner: HashMap<u32, u32> = HashMap::new();
        let mut border: HashMap<u32, bool> = HashMap::new();
        for (g, members) in groups.iter().enumerate() {
            for &ci in members {
                for t in &out[ci].tris {
                    for &v in t {
                        let p = pos_id[v as usize];
                        match owner.get(&p) {
                            Some(&o) if o != g as u32 => {
                                border.insert(p, true);
                            }
                            Some(_) => {}
                            None => {
                                owner.insert(p, g as u32);
                            }
                        }
                    }
                }
            }
        }
        let results: Vec<Option<(Vec<[u32; 3]>, Vec<u32>, f32)>> = groups
            .par_iter()
            .map(|members| {
                let mut gt: Vec<[u32; 3]> = Vec::new();
                let mut gs: Vec<u32> = Vec::new();
                for &ci in members {
                    gt.extend_from_slice(&out[ci].tris);
                    gs.extend_from_slice(&out[ci].source);
                }
                let locked = |v: u32| border.contains_key(&pos_id[v as usize]);
                let target = gt.len() / 2;
                let (kept, error) = simplify(positions, &pos_id, &gt, &locked, target);
                if (kept.len() as f32) > gt.len() as f32 * (1.0 - MIN_REDUCTION) {
                    return None;
                }
                let new_tris: Vec<[u32; 3]> = kept.iter().map(|&(t, _)| t).collect();
                let new_src: Vec<u32> = kept.iter().map(|&(_, k)| gs[k as usize]).collect();
                Some((new_tris, new_src, error))
            })
            .collect();

        let mut next = Vec::new();
        // Clusters whose group would not simplify go round again, grouped
        // with other neighbours next time. Were they left as they are, the
        // groups beside them could move the vertices they share.
        let mut carried = Vec::new();
        let mut level_tris = 0;
        for (members, res) in groups.iter().zip(results) {
            let Some((gt, gs, err)) = res else {
                carried.extend_from_slice(members);
                continue;
            };
            let new_level = members.iter().map(|&ci| out[ci].level).max().unwrap_or(0) + 1;
            let child_error = members.iter().map(|&ci| out[ci].error).fold(0.0f32, f32::max);
            // Errors only grow up the hierarchy, and a group's sphere holds
            // its children's: what is drawn changes monotonically with
            // distance.
            let error = (err + child_error).max(child_error * 1.0001).max(f32::MIN_POSITIVE);
            let sphere = merge_spheres(members.iter().map(|&ci| out[ci].sphere));
            for &ci in members {
                out[ci].parent_error = error;
                out[ci].parent_sphere = sphere;
            }
            // Cut the simplified group into clusters again.
            let flat: Vec<u32> = gt.iter().flatten().copied().collect();
            for b in build_clusters(positions, &flat) {
                let ctris: Vec<[u32; 3]> = b.local_tris.iter().map(|t| t.map(|l| b.vertices[l as usize])).collect();
                let src: Vec<u32> = b.source_tris.iter().map(|&k| gs[k as usize]).collect();
                level_tris += ctris.len();
                next.push(out.len());
                out.push(LodCluster { tris: ctris, source: src, level: new_level, error, sphere, parent_error: f32::INFINITY, parent_sphere: (Vec3::ZERO, 0.0) });
            }
        }
        if next.is_empty() {
            break;
        }
        lvl += 1;
        stats.tris.push(level_tris);
        log(format!(
            "pass {lvl}: {level_tris} triangles in {} new clusters from {} groups, {} clusters carried over ({:.1} s grouping, {:.1} s in all; {:.0} s so far)",
            next.len(),
            groups.len(),
            carried.len(),
            t_group,
            tl.elapsed().as_secs_f32(),
            t0.elapsed().as_secs_f32()
        ));
        stats.clusters.push(next.len());
        next.extend(carried);
        level = next;
    }
    (out, stats)
}

/// One id per distinct position (exact match), and whether each vertex
/// shares its position with another vertex.
fn weld(positions: &[Vec3]) -> (Vec<u32>, Vec<bool>) {
    let mut map: HashMap<[u32; 3], u32> = HashMap::with_capacity(positions.len());
    let mut count: Vec<u32> = Vec::new();
    let ids: Vec<u32> = positions
        .iter()
        .map(|p| {
            let key = [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()];
            let n = map.len() as u32;
            let id = *map.entry(key).or_insert(n);
            if id as usize == count.len() {
                count.push(0);
            }
            count[id as usize] += 1;
            id
        })
        .collect();
    let seam = ids.iter().map(|&i| count[i as usize] > 1).collect();
    (ids, seam)
}

/// Gather a level's clusters into groups of about `GROUP` neighbours: in
/// space-filling order, each group starts at the next free cluster and takes
/// the free neighbour sharing the most edges with it, then the next.
fn make_groups(out: &[LodCluster], level: &[usize], positions: &[Vec3], pos_id: &[u32]) -> Vec<Vec<usize>> {
    // Edge (by position ids) -> clusters using it.
    let mut edges: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
    for (k, &ci) in level.iter().enumerate() {
        for t in &out[ci].tris {
            for e in 0..3 {
                let (a, b) = (pos_id[t[e] as usize], pos_id[t[(e + 1) % 3] as usize]);
                let key = (a.min(b), a.max(b));
                let list = edges.entry(key).or_default();
                if list.last() != Some(&(k as u32)) {
                    list.push(k as u32);
                }
            }
        }
    }
    // Shared edges between clusters of the level.
    let mut adj: Vec<HashMap<u32, u32>> = vec![HashMap::new(); level.len()];
    for list in edges.values() {
        for i in 0..list.len() {
            for j in 0..list.len() {
                if i != j {
                    *adj[list[i] as usize].entry(list[j]).or_insert(0) += 1;
                }
            }
        }
    }
    // Space-filling order of the cluster centres.
    let centres: Vec<Vec3> = level.iter().map(|&ci| out[ci].sphere.0).collect();
    let (mut lo, mut hi) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
    for c in &centres {
        lo = lo.min(*c);
        hi = hi.max(*c);
    }
    let ext = (hi - lo).max(Vec3::splat(1e-20));
    let mut order: Vec<(u64, usize)> = centres
        .iter()
        .enumerate()
        .map(|(k, c)| {
            let q = ((*c - lo) / ext * 1023.0).clamp(Vec3::ZERO, Vec3::splat(1023.0));
            (morton10(q.x as u32, q.y as u32, q.z as u32), k)
        })
        .collect();
    order.sort_unstable();
    let _ = positions;
    let mut taken = vec![false; level.len()];
    let mut groups = Vec::new();
    for &(_, seed) in &order {
        if taken[seed] {
            continue;
        }
        taken[seed] = true;
        let mut g = vec![seed];
        while g.len() < GROUP {
            // The free neighbour with the most edges shared with the group.
            let mut best: Option<(u32, usize)> = None;
            for &m in &g {
                for (&n, &w) in &adj[m] {
                    let n = n as usize;
                    if taken[n] {
                        continue;
                    }
                    let shared: u32 = g.iter().map(|&x| adj[x].get(&(n as u32)).copied().unwrap_or(0)).sum();
                    let _ = w;
                    if best.is_none_or(|(s, b)| shared > s || (shared == s && n < b)) {
                        best = Some((shared, n));
                    }
                }
            }
            match best {
                Some((_, n)) => {
                    taken[n] = true;
                    g.push(n);
                }
                None => break,
            }
        }
        groups.push(g.into_iter().map(|k| level[k]).collect());
    }
    groups
}

fn morton10(x: u32, y: u32, z: u32) -> u64 {
    let spread = |mut v: u64| {
        v &= 0x3ff;
        v = (v | v << 16) & 0x30000ff;
        v = (v | v << 8) & 0x300f00f;
        v = (v | v << 4) & 0x30c30c3;
        v = (v | v << 2) & 0x9249249;
        v
    };
    spread(x as u64) | spread(y as u64) << 1 | spread(z as u64) << 2
}

/// A sphere around a set of triangles' vertices.
fn bounding_sphere(positions: &[Vec3], tris: &[[u32; 3]]) -> (Vec3, f32) {
    let (mut lo, mut hi) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
    for t in tris {
        for &v in t {
            lo = lo.min(positions[v as usize]);
            hi = hi.max(positions[v as usize]);
        }
    }
    let c = (lo + hi) * 0.5;
    let r = tris.iter().flatten().map(|&v| positions[v as usize].distance(c)).fold(0.0, f32::max);
    (c, r)
}

/// A sphere holding every sphere given.
fn merge_spheres(spheres: impl Iterator<Item = (Vec3, f32)>) -> (Vec3, f32) {
    let s: Vec<(Vec3, f32)> = spheres.collect();
    let (mut lo, mut hi) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
    for (c, r) in &s {
        lo = lo.min(*c - Vec3::splat(*r));
        hi = hi.max(*c + Vec3::splat(*r));
    }
    let c = (lo + hi) * 0.5;
    let r = s.iter().map(|(sc, sr)| sc.distance(c) + sr).fold(0.0, f32::max);
    (c, r)
}

/// A symmetric 4 x 4 quadric, upper triangle, and the total area of the
/// planes in it.
#[derive(Clone, Copy, Default)]
struct Quadric([f64; 10], f64);

impl Quadric {
    /// Squared distance to the plane n.x + d = 0 (`n` unit length),
    /// weighted by `area`.
    fn plane(n: Vec3, d: f32, area: f32) -> Self {
        let (a, b, c, d, w) = (n.x as f64, n.y as f64, n.z as f64, d as f64, area as f64);
        Quadric([a * a * w, a * b * w, a * c * w, a * d * w, b * b * w, b * c * w, b * d * w, c * c * w, c * d * w, d * d * w], w)
    }
    fn add(&mut self, o: &Quadric) {
        for k in 0..10 {
            self.0[k] += o.0[k];
        }
        self.1 += o.1;
    }
    /// Mean squared distance from `p` to the planes, by area.
    fn eval(&self, p: Vec3) -> f64 {
        let (x, y, z) = (p.x as f64, p.y as f64, p.z as f64);
        let q = &self.0;
        (q[0] * x * x + 2.0 * q[1] * x * y + 2.0 * q[2] * x * z + 2.0 * q[3] * x + q[4] * y * y + 2.0 * q[5] * y * z + 2.0 * q[6] * y + q[7] * z * z + 2.0 * q[8] * z + q[9]) / self.1.max(1e-30)
    }
}

#[derive(PartialEq)]
struct Candidate {
    cost: f64,
    from: u32,
    to: u32,
    stamp: (u32, u32),
}

impl Eq for Candidate {}

impl Ord for Candidate {
    fn cmp(&self, o: &Self) -> Ordering {
        // Cheapest first.
        o.cost.partial_cmp(&self.cost).unwrap_or(Ordering::Equal).then_with(|| (o.from, o.to).cmp(&(self.from, self.to)))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

/// Collapse edges of `tris` until at most `target` triangles are left or no
/// collapse is allowed. Vertices for which `locked` holds never move.
///
/// Edges join positions, not vertices, so a seam (one position, several
/// vertices with their own texture coordinates or normals) is one edge
/// like any other. Each triangle corner keeps its own vertex: when a
/// position moves onto a neighbour, every corner there takes the vertex
/// across the edge on its own side of the seam, and a collapse with a
/// corner that has none (it would drag a seam off itself) is refused.
///
/// Returns the triangles left as vertices (with the index of the input
/// triangle each one was), and the error: the largest distance between the
/// surface before and after.
pub fn simplify(positions: &[Vec3], pos_id: &[u32], tris: &[[u32; 3]], locked: &dyn Fn(u32) -> bool, target: usize) -> (Vec<([u32; 3], u32)>, f32) {
    // Local positions, and for each one a vertex there (for locking and
    // the position itself).
    let mut local: HashMap<u32, u32> = HashMap::new();
    let mut at: Vec<u32> = Vec::new();
    let mut t: Vec<[u32; 3]> = tris
        .iter()
        .map(|tri| {
            tri.map(|v| {
                *local.entry(pos_id[v as usize]).or_insert_with(|| {
                    at.push(v);
                    (at.len() - 1) as u32
                })
            })
        })
        .collect();
    // The vertex at each corner.
    let mut c: Vec<[u32; 3]> = tris.to_vec();
    let nv = at.len();
    let p = |l: u32| positions[at[l as usize] as usize];
    // The surface before, for measuring the error after.
    let before: Vec<[Vec3; 3]> = t.iter().map(|tri| tri.map(|x| p(x))).collect();
    let mut alive: Vec<bool> = t.iter().map(|x| x[0] != x[1] && x[1] != x[2] && x[2] != x[0]).collect();
    let mut live = alive.iter().filter(|a| **a).count();
    let mut around: Vec<Vec<u32>> = vec![Vec::new(); nv];
    for (k, tri) in t.iter().enumerate() {
        if alive[k] {
            for &v in tri {
                around[v as usize].push(k as u32);
            }
        }
    }
    // Open edges (one triangle) lock their positions, as do the positions
    // `locked` names.
    let mut edge_count: HashMap<(u32, u32), u32> = HashMap::new();
    for (k, tri) in t.iter().enumerate() {
        if !alive[k] {
            continue;
        }
        for e in 0..3 {
            let (a, b) = (tri[e], tri[(e + 1) % 3]);
            *edge_count.entry((a.min(b), a.max(b))).or_insert(0) += 1;
        }
    }
    let mut fixed: Vec<bool> = (0..nv).map(|l| locked(at[l])).collect();
    for (k, tri) in t.iter().enumerate() {
        if !alive[k] {
            continue;
        }
        for e in 0..3 {
            let (a, b) = (tri[e], tri[(e + 1) % 3]);
            if edge_count[&(a.min(b), a.max(b))] != 2 {
                fixed[a as usize] = true;
                fixed[b as usize] = true;
            }
        }
    }
    // Quadrics.
    let mut q = vec![Quadric::default(); nv];
    for (k, tri) in t.iter().enumerate() {
        if !alive[k] {
            continue;
        }
        let (a, b, cc) = (p(tri[0]), p(tri[1]), p(tri[2]));
        let n = (b - a).cross(cc - a);
        if n.length_squared() <= 0.0 {
            continue;
        }
        let area = n.length() * 0.5;
        let n = n.normalize();
        let pq = Quadric::plane(n, -n.dot(a), area);
        for &v in tri {
            q[v as usize].add(&pq);
        }
    }
    let mut stamp = vec![0u32; nv];
    let mut heap = BinaryHeap::new();
    let push = |heap: &mut BinaryHeap<Candidate>, q: &[Quadric], stamp: &[u32], fixed: &[bool], from: u32, to: u32| {
        if fixed[from as usize] {
            return;
        }
        let mut s = q[from as usize];
        s.add(&q[to as usize]);
        heap.push(Candidate { cost: s.eval(p(to)).max(0.0), from, to, stamp: (stamp[from as usize], stamp[to as usize]) });
    };
    for (k, tri) in t.iter().enumerate() {
        if !alive[k] {
            continue;
        }
        for e in 0..3 {
            let (a, b) = (tri[e], tri[(e + 1) % 3]);
            push(&mut heap, &q, &stamp, &fixed, a, b);
            push(&mut heap, &q, &stamp, &fixed, b, a);
        }
    }
    while live > target {
        let Some(cand) = heap.pop() else { break };
        let (u, v) = (cand.from as usize, cand.to as usize);
        if cand.stamp != (stamp[u], stamp[v]) || fixed[u] {
            continue;
        }
        // Still an edge?
        let shared: Vec<u32> = around[u].iter().copied().filter(|&k| alive[k as usize] && t[k as usize].contains(&(v as u32))).collect();
        if shared.is_empty() {
            continue;
        }
        // Link condition: the positions next to both u and v are exactly
        // the third corners of the triangles on the edge.
        let (ru, rv) = (ring(&around, &alive, &t, u), ring(&around, &alive, &t, v));
        let common = ru.iter().filter(|w| rv.binary_search(w).is_ok()).count();
        if common != shared.len() {
            continue;
        }
        // Each vertex at u, the vertex at v across the edge on its side.
        let mut across: Vec<(u32, u32)> = Vec::new();
        for &k in &shared {
            let k = k as usize;
            let (iu, iv) = ((0..3).find(|&i| t[k][i] as usize == u).unwrap(), (0..3).find(|&i| t[k][i] as usize == v).unwrap());
            if !across.iter().any(|(a, _)| *a == c[k][iu]) {
                across.push((c[k][iu], c[k][iv]));
            }
        }
        // No triangle may flip or collapse to a sliver, and every corner at
        // u must have a vertex to go to.
        let pv = p(v as u32);
        let mut ok = true;
        for &k in &around[u] {
            let k = k as usize;
            if !alive[k] || t[k].contains(&(v as u32)) {
                continue;
            }
            let iu = (0..3).find(|&i| t[k][i] as usize == u).unwrap();
            if !across.iter().any(|(a, _)| *a == c[k][iu]) {
                ok = false;
                break;
            }
            let tri = t[k];
            let pts = tri.map(|x| p(x));
            let old = (pts[1] - pts[0]).cross(pts[2] - pts[0]);
            let new_pts = tri.map(|x| if x as usize == u { pv } else { p(x) });
            let new = (new_pts[1] - new_pts[0]).cross(new_pts[2] - new_pts[0]);
            let longest = (new_pts[0] - new_pts[1]).length_squared().max((new_pts[1] - new_pts[2]).length_squared()).max((new_pts[2] - new_pts[0]).length_squared());
            if new.dot(old) <= 0.0 || new.length() < 1e-3 * longest || new.normalize_or_zero().dot(old.normalize_or_zero()) < 0.2 {
                ok = false;
                break;
            }
        }
        if !ok {
            continue;
        }
        // Collapse u onto v.
        for &k in &shared {
            alive[k as usize] = false;
            live -= 1;
        }
        let moved: Vec<u32> = around[u].iter().copied().filter(|&k| alive[k as usize]).collect();
        for &k in &moved {
            let k = k as usize;
            for i in 0..3 {
                if t[k][i] as usize == u {
                    t[k][i] = v as u32;
                    c[k][i] = across.iter().find(|(a, _)| *a == c[k][i]).unwrap().1;
                }
            }
            around[v].push(k as u32);
        }
        around[u].clear();
        around[v].retain(|&k| alive[k as usize]);
        around[v].sort_unstable();
        around[v].dedup();
        let qu = q[u];
        q[v].add(&qu);
        stamp[u] = stamp[u].wrapping_add(1);
        stamp[v] = stamp[v].wrapping_add(1);
        for w in ring(&around, &alive, &t, v) {
            push(&mut heap, &q, &stamp, &fixed, v as u32, w);
            push(&mut heap, &q, &stamp, &fixed, w, v as u32);
        }
    }
    let kept: Vec<([u32; 3], u32)> = (0..t.len()).filter(|&k| alive[k]).map(|k| (c[k], k as u32)).collect();
    // Measured error, both ways: every position before to the nearest
    // triangle after, and points across each triangle after (centre and
    // edge middles) to the nearest triangle before.
    let after: Vec<[Vec3; 3]> = kept.iter().map(|(tri, _)| tri.map(|v| positions[v as usize])).collect();
    let one_way = (0..nv).map(|l| nearest(p(l as u32), &after)).fold(0.0f32, f32::max);
    let other_way = after
        .iter()
        .flat_map(|[a, b, c]| [(*a + *b + *c) / 3.0, (*a + *b) * 0.5, (*b + *c) * 0.5, (*c + *a) * 0.5])
        .map(|q| nearest(q, &before))
        .fold(0.0f32, f32::max);
    let error = one_way.max(other_way);
    (kept, if error.is_finite() { error } else { 0.0 })
}

/// Distance from `p` to the nearest of `tris`, skipping those whose box is
/// already further than the best so far.
fn nearest(p: Vec3, tris: &[[Vec3; 3]]) -> f32 {
    let mut best = f32::INFINITY;
    for t in tris {
        let lo = t[0].min(t[1]).min(t[2]);
        let hi = t[0].max(t[1]).max(t[2]);
        let gap = (lo - p).max(p - hi).max(Vec3::ZERO);
        if gap.length_squared() >= best * best {
            continue;
        }
        best = best.min(point_triangle_distance(p, t));
    }
    best
}

/// Distance from `p` to the triangle `t` (Ericson, Real-Time Collision
/// Detection, 5.1.5).
fn point_triangle_distance(p: Vec3, t: &[Vec3; 3]) -> f32 {
    let (a, b, c) = (t[0], t[1], t[2]);
    let (ab, ac, ap) = (b - a, c - a, p - a);
    let (d1, d2) = (ab.dot(ap), ac.dot(ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return p.distance(a);
    }
    let bp = p - b;
    let (d3, d4) = (ab.dot(bp), ac.dot(bp));
    if d3 >= 0.0 && d4 <= d3 {
        return p.distance(b);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return p.distance(a + ab * v);
    }
    let cp = p - c;
    let (d5, d6) = (ab.dot(cp), ac.dot(cp));
    if d6 >= 0.0 && d5 <= d6 {
        return p.distance(c);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return p.distance(a + ac * w);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return p.distance(b + (c - b) * w);
    }
    let denom = 1.0 / (va + vb + vc);
    let (v, w) = (vb * denom, vc * denom);
    p.distance(a + ab * v + ac * w)
}

/// Vertices next to `x` through live triangles, sorted.
fn ring(around: &[Vec<u32>], alive: &[bool], t: &[[u32; 3]], x: usize) -> Vec<u32> {
    let mut r: Vec<u32> = around[x].iter().filter(|&&k| alive[k as usize]).flat_map(|&k| t[k as usize]).filter(|&w| w as usize != x).collect();
    r.sort_unstable();
    r.dedup();
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::tests::uv_sphere;

    fn as_tris(idx: &[u32]) -> Vec<[u32; 3]> {
        idx.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
    }

    /// A flat grid simplifies to almost nothing with no error; its border
    /// stays where it was.
    #[test]
    fn a_flat_grid_simplifies_without_error() {
        let n = 20;
        let mut p = Vec::new();
        for z in 0..=n {
            for x in 0..=n {
                p.push(Vec3::new(x as f32, 0.0, z as f32));
            }
        }
        let mut idx = Vec::new();
        for z in 0..n {
            for x in 0..n {
                let a = (z * (n + 1) + x) as u32;
                idx.extend_from_slice(&[a, a + n as u32 + 1, a + 1, a + 1, a + n as u32 + 1, a + n as u32 + 2]);
            }
        }
        let tris = as_tris(&idx);
        let (ids, _) = weld(&p);
        let (kept, err) = simplify(&p, &ids, &tris, &|_| false, 10);
        assert!(kept.len() < tris.len() / 4, "{} of {}", kept.len(), tris.len());
        assert!(err < 1e-4, "{err}");
        // Every border vertex is still used.
        let used: std::collections::HashSet<u32> = kept.iter().flat_map(|(t, _)| *t).collect();
        for (i, q) in p.iter().enumerate() {
            if q.x == 0.0 || q.z == 0.0 || q.x == n as f32 || q.z == n as f32 {
                assert!(used.contains(&(i as u32)), "border vertex {i} at {q} moved");
            }
        }
    }

    /// A grid split down the middle by a seam (the middle column's
    /// positions twice, one vertex for each side): it simplifies across the
    /// seam's length, and no triangle ever mixes the two sides' vertices.
    #[test]
    fn seams_simplify_along_themselves_and_stay_whole() {
        let n = 20usize;
        let mut p = Vec::new();
        // Left side columns 0..=10, right side columns 10..=20: column 10 twice.
        let mut id = vec![vec![[0u32; 2]; n + 1]; n + 1];
        for z in 0..=n {
            for x in 0..=n {
                let pos = Vec3::new(x as f32, (x as f32 * 0.3).sin() * 0.2, z as f32);
                if x <= 10 {
                    p.push(pos);
                    id[z][x][0] = (p.len() - 1) as u32;
                }
                if x >= 10 {
                    p.push(pos);
                    id[z][x][1] = (p.len() - 1) as u32;
                }
            }
        }
        let side = |x: usize| if x < 10 { 0 } else { 1 };
        let mut tris = Vec::new();
        for z in 0..n {
            for x in 0..n {
                let s = side(x);
                let (a, b, c, d) = (id[z][x][s], id[z][x + 1][s], id[z + 1][x][s], id[z + 1][x + 1][s]);
                tris.push([a, c, b]);
                tris.push([b, c, d]);
            }
        }
        let left: std::collections::HashSet<u32> = (0..=n).flat_map(|z| (0..=10).map(move |x| (z, x))).map(|(z, x)| id[z][x][0]).collect();
        let (ids, _) = weld(&p);
        let (kept, _) = simplify(&p, &ids, &tris, &|_| false, tris.len() / 8);
        assert!(kept.len() < tris.len() / 4, "{} of {}", kept.len(), tris.len());
        for (t, _) in &kept {
            let l = t.iter().filter(|v| left.contains(v)).count();
            assert!(l == 0 || l == 3, "a triangle mixes the two sides: {t:?}");
        }
        // The seam is still a line of positions shared by both sides.
        let used: std::collections::HashSet<u32> = kept.iter().flat_map(|(t, _)| *t).collect();
        let seam_left = (0..=n).filter(|&z| used.contains(&id[z][10][0])).count();
        let seam_right = (0..=n).filter(|&z| used.contains(&id[z][10][1])).count();
        assert_eq!(seam_left, seam_right);
        assert!(seam_left < n, "the seam simplified along itself: {seam_left} of {} positions left", n + 1);
    }

    /// A sphere's hierarchy: each level about halves, errors grow up the
    /// levels, every cluster stays within the cluster limits, and the
    /// coarsest level still looks like the sphere.
    #[test]
    fn a_sphere_builds_levels_that_halve() {
        let (p, idx) = uv_sphere(128);
        let (out, stats) = build(&p, &as_tris(&idx));
        assert!(stats.tris.len() >= 4, "{:?}", stats.tris);
        // Each pass makes about half of what went in, and the coarsest cut
        // (everything nothing replaces) under a tenth of the mesh (its poles,
        // where many vertices share a position, never move).
        assert!(stats.tris[1] < stats.tris[0] * 3 / 5, "{:?}", stats.tris);
        let roots: usize = out.iter().filter(|c| c.parent_error.is_infinite()).map(|c| c.tris.len()).sum();
        assert!(roots * 10 < idx.len() / 3, "{roots} triangles at the top, {:?}", stats.tris);
        for c in &out {
            assert!(c.tris.len() <= crate::scene::CLUSTER_TRIS);
            assert!(c.parent_error > c.error || c.parent_error.is_infinite());
            // The replacement's sphere holds this one's.
            if c.parent_error.is_finite() {
                assert!(c.sphere.0.distance(c.parent_sphere.0) + c.sphere.1 <= c.parent_sphere.1 * 1.0001 + 1e-6);
            }
        }
        let top = out.iter().map(|c| c.level).max().unwrap();
        let coarse: Vec<&LodCluster> = out.iter().filter(|c| c.level == top).collect();
        for c in coarse {
            for t in &c.tris {
                for &v in t {
                    assert!((p[v as usize].length() - 1.0).abs() < 1e-5, "vertices stay on the sphere");
                }
            }
        }
        // Each level's error bounds how much deeper below the sphere its
        // triangles sit than level 0's do.
        let depth = |tris: &[[u32; 3]]| {
            tris.iter()
                .flat_map(|t| {
                    let (a, b, cc) = (p[t[0] as usize], p[t[1] as usize], p[t[2] as usize]);
                    [(a + b + cc) / 3.0, (a + b) * 0.5, (b + cc) * 0.5, (cc + a) * 0.5]
                })
                .map(|q| 1.0 - q.length())
                .fold(0.0f32, f32::max)
        };
        let base = out.iter().filter(|c| c.level == 0).map(|c| depth(&c.tris)).fold(0.0f32, f32::max);
        // Clusters of one group share its error and sphere: compare per group.
        let mut groups: HashMap<(u8, u32, u32), f32> = HashMap::new();
        for c in out.iter().filter(|c| c.level > 0) {
            let g = groups.entry((c.level, c.error.to_bits(), c.sphere.1.to_bits())).or_insert(0.0);
            *g = g.max(depth(&c.tris));
        }
        for c in &out {
            if c.level == 0 {
                continue;
            }
            let sag = (groups[&(c.level, c.error.to_bits(), c.sphere.1.to_bits())] - base).max(0.0);
            assert!(c.error >= sag * 0.8, "level {}: error {} under the real {}", c.level, c.error, sag);
        }
    }

    /// Picking one cut through the hierarchy, at any distance, covers the
    /// sphere exactly once: the area drawn is the sphere's area, never
    /// double, never missing a group.
    #[test]
    fn every_cut_covers_the_surface_once() {
        let (p, idx) = uv_sphere(96);
        let (out, _) = build(&p, &as_tris(&idx));
        let area = |c: &LodCluster| c.tris.iter().map(|t| (p[t[1] as usize] - p[t[0] as usize]).cross(p[t[2] as usize] - p[t[0] as usize]).length() * 0.5).sum::<f32>();
        let full: f32 = out.iter().filter(|c| c.level == 0).map(area).sum();
        for dist in [1.5f32, 3.0, 10.0, 40.0, 200.0, 5000.0] {
            let eye = Vec3::new(0.0, 0.0, dist);
            let scale = |c: Vec3, r: f32| 1000.0 / (c.distance(eye) - r).max(0.01);
            let drawn: f32 = out
                .iter()
                .filter(|c| {
                    let cl = crate::scene::Cluster { lod_error: c.error, lod_centre: c.sphere.0, lod_radius: c.sphere.1, parent_error: c.parent_error, parent_centre: c.parent_sphere.0, parent_radius: c.parent_sphere.1, ..Default::default() };
                    cl.lod_drawn(1.0, scale)
                })
                .map(area)
                .sum();
            // Never twice. Simplified levels lose a little area where they
            // cut corners; the coarsest, a sphere of a few hundred
            // triangles, loses more.
            let floor = if dist <= 40.0 { 0.97 } else { 0.0 };
            assert!(drawn > full * floor && drawn < full * 1.001, "at {dist} m: {drawn} of {full}");
        }
    }
}
