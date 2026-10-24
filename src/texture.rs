//! Textures as per-triangle tiles, streamed on demand.
//!
//! The idea is Ptex's (Burley and Lacewell 2008): no UV atlas at render
//! time, each face carries its own small texture. Here a face is a
//! triangle, and its texture at level L is the colour at exactly the points
//! the REYES dicer shades at level L: the (2^L + 1)(2^L + 2) / 2 grid points
//! of the triangle cut into 2^L parts per edge. A shading point of a
//! triangle diced at level L reads its colour straight from the tile at
//! level L, with no filtering and no UV maths at render time. Level 0 is the
//! triangle's three corners, which the vertex colours already hold.
//!
//! The tiles are baked at import (`tools/fkj-import`), with the source
//! textures filtered down to each level, and stored in the `.fkj` file's
//! `TEXR` section, cluster by cluster, level by level. Nothing of them is
//! read at load. When a cluster is shaded at level L, the block of its
//! triangles' tiles at that level is read from the file and kept in a
//! memory-capped cache. Only visible triangles' textures, at the size they
//! are seen at, ever reach memory.

use std::collections::HashMap;
use std::fs::File;
use std::sync::{Arc, Mutex};

use glam::Vec3;

/// Finest tile level. The dicer goes finer (`raster::MAX_DICE_LEVEL`) and
/// samples this level there.
pub const MAX_LEVEL: u8 = 5;

/// Points in a triangle tile at level `l` (the dicer's grid points).
pub fn tile_points(l: u8) -> usize {
    let n = 1usize << l;
    (n + 1) * (n + 2) / 2
}

/// Index of grid point (i, j) of a tile with `n` parts per edge: i steps
/// towards the triangle's second vertex, j towards its third.
#[inline]
pub fn tile_index(n: usize, i: usize, j: usize) -> usize {
    i * (n + 1) - i * i.saturating_sub(1) / 2 + j
}

/// Bytes of one cluster's tiles at levels 1 to `l - 1`, so the offset of
/// level `l` inside the cluster's data.
pub fn level_offset(tris: usize, l: u8) -> u64 {
    (1..l).map(|k| (tris * tile_points(k) * 3) as u64).sum()
}

/// Where a cluster's tiles are.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TexEntry {
    /// Finest level stored; 0 means no tiles (vertex colours only).
    pub level: u8,
    pub source: u16,
    /// Offset of its level-1 block in the source's pool.
    pub offset: u64,
    /// Offset of its cutout tiles in the source's alpha pool: the same
    /// levels and points, one byte each (0 to 255, opaque from 128). None
    /// for an opaque cluster. A cluster with them is drawn from both sides.
    pub alpha: Option<u64>,
}

/// Where tile data comes from.
pub enum Source {
    /// A `.fkj` file, read with positioned reads, and where in it the pool
    /// starts.
    File { file: File, pool: u64 },
    /// Tiles already in memory (the importer, tests).
    Memory(Arc<Vec<u8>>),
}

impl Source {
    pub fn read(&self, offset: u64, len: usize) -> std::io::Result<Vec<u8>> {
        match self {
            Source::File { file, pool } => {
                let mut buf = vec![0u8; len];
                read_exact_at(file, &mut buf, pool + offset)?;
                Ok(buf)
            }
            Source::Memory(m) => {
                let a = offset as usize;
                m.get(a..a + len).map(|s| s.to_vec()).ok_or_else(|| std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "tile outside the pool"))
            }
        }
    }
}

#[cfg(unix)]
fn read_exact_at(f: &File, buf: &mut [u8], at: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    f.read_exact_at(buf, at)
}

#[cfg(windows)]
fn read_exact_at(f: &File, buf: &mut [u8], mut at: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    let mut done = 0;
    while done < buf.len() {
        let n = f.seek_read(&mut buf[done..], at)?;
        if n == 0 {
            return Err(std::io::ErrorKind::UnexpectedEof.into());
        }
        done += n;
        at += n as u64;
    }
    Ok(())
}

/// A block of one cluster's tiles at one level: sRGB8 RGB, triangle by
/// triangle, `tile_points(level)` points each (one byte each for alpha).
pub struct Block {
    pub level: u8,
    pub data: Vec<u8>,
}

/// From an alpha byte up, a point is solid.
pub const ALPHA_CUT: u8 = 128;

impl Block {
    /// Whether triangle `t` (index within the cluster) is solid at
    /// barycentric weights (b1, b2): its nearest alpha point.
    #[inline]
    pub fn solid(&self, t: usize, b1: f32, b2: f32) -> bool {
        let n = 1usize << self.level;
        let nf = n as f32;
        let i = ((b1 * nf + 0.5) as isize).clamp(0, n as isize) as usize;
        let j = ((b2 * nf + 0.5) as isize).clamp(0, (n - i) as isize) as usize;
        self.data.get(t * tile_points(self.level) + tile_index(n, i, j)).is_none_or(|&a| a >= ALPHA_CUT)
    }

    /// Share of triangle `t`'s alpha points that are solid.
    pub fn coverage(&self, t: usize) -> f32 {
        let k = tile_points(self.level);
        let pts = self.data.get(t * k..t * k + k).unwrap_or(&[]);
        if pts.is_empty() {
            return 1.0;
        }
        pts.iter().filter(|&&a| a >= ALPHA_CUT).count() as f32 / pts.len() as f32
    }

    /// Linear colour at grid point (i, j) of triangle `t` (index within the
    /// cluster).
    #[inline]
    pub fn point(&self, t: usize, i: usize, j: usize) -> Vec3 {
        let n = 1usize << self.level;
        let k = (t * tile_points(self.level) + tile_index(n, i, j)) * 3;
        let lut = srgb_lut();
        Vec3::new(lut[self.data[k] as usize], lut[self.data[k + 1] as usize], lut[self.data[k + 2] as usize])
    }

    /// Linear colour at barycentric weights (b1, b2) of triangle `t`,
    /// interpolated between the tile's points (used when a triangle is
    /// diced finer than its tile).
    pub fn sample(&self, t: usize, b1: f32, b2: f32) -> Vec3 {
        let n = 1usize << self.level;
        let nf = n as f32;
        let (u1, u2) = (b1.clamp(0.0, 1.0) * nf, b2.clamp(0.0, 1.0) * nf);
        let i = (u1.floor() as usize).min(n - 1);
        let j = (u2.floor() as usize).min(n - 1 - i);
        let (fi, fj) = (u1 - i as f32, u2 - j as f32);
        if fi + fj <= 1.0 || i + j + 1 >= n {
            let (fi, fj) = if fi + fj > 1.0 { (fi / (fi + fj), fj / (fi + fj)) } else { (fi, fj) };
            self.point(t, i, j) * (1.0 - fi - fj) + self.point(t, i + 1, j) * fi + self.point(t, i, j + 1) * fj
        } else {
            self.point(t, i + 1, j + 1) * (fi + fj - 1.0) + self.point(t, i, j + 1) * (1.0 - fi) + self.point(t, i + 1, j) * (1.0 - fj)
        }
    }
}

fn srgb_lut() -> &'static [f32; 256] {
    static LUT: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();
    LUT.get_or_init(|| {
        std::array::from_fn(|i| {
            let s = i as f32 / 255.0;
            if s <= 0.04045 { s / 12.92 } else { ((s + 0.055) / 1.055).powf(2.4) }
        })
    })
}

/// A block's key: where its data is (source, offset), its level, and
/// whether it is alpha, so that instances, whose clusters point at the
/// same tiles, share blocks.
type Key = (u16, u64, u8, bool);

struct Cache {
    blocks: HashMap<Key, (Arc<Block>, u64)>,
    bytes: usize,
    clock: u64,
}

/// What the cache did, for the status line.
#[derive(Clone, Copy, Debug, Default)]
pub struct TexStats {
    pub reads: u64,
    pub bytes_read: u64,
    pub resident: usize,
}

/// All of a scene's tiles: an entry per cluster, the sources, and the cache.
pub struct Textures {
    pub entries: Vec<TexEntry>,
    pub sources: Vec<Arc<Source>>,
    /// Each source's alpha pool, if it has one (same order as `sources`).
    pub alpha_sources: Vec<Option<Arc<Source>>>,
    cache: Mutex<Cache>,
    /// Most bytes the cache holds before it drops the least recently used.
    pub budget: Mutex<usize>,
    stats: Mutex<TexStats>,
}

impl Default for Textures {
    fn default() -> Self {
        Textures {
            entries: Vec::new(),
            sources: Vec::new(),
            alpha_sources: Vec::new(),
            cache: Mutex::new(Cache { blocks: HashMap::new(), bytes: 0, clock: 0 }),
            budget: Mutex::new(64 << 20),
            stats: Mutex::new(TexStats::default()),
        }
    }
}

impl Textures {
    pub fn is_empty(&self) -> bool {
        self.entries.iter().all(|e| e.level == 0)
    }

    pub fn entry(&self, cluster: usize) -> TexEntry {
        self.entries.get(cluster).copied().unwrap_or_default()
    }

    #[cfg_attr(not(target_pointer_width = "64"), allow(dead_code))]
    /// Whether cluster `c`'s tiles at `level` (or its finest, if coarser)
    /// are in the cache now.
    pub fn resident(&self, c: usize, level: u8) -> bool {
        let e = self.entry(c);
        if e.level == 0 || level == 0 {
            return false;
        }
        let key = (e.source, e.offset, level.min(e.level), false);
        self.cache.lock().unwrap().blocks.contains_key(&key)
    }

    pub fn set_budget(&self, bytes: usize) {
        *self.budget.lock().unwrap() = bytes.max(1 << 20);
    }

    pub fn stats(&self) -> TexStats {
        let mut s = *self.stats.lock().unwrap();
        s.resident = self.cache.lock().unwrap().bytes;
        s
    }

    /// Append another scene's tiles for clusters numbered from
    /// `cluster_base`; clusters without an entry get none.
    pub fn append(&mut self, other: &Textures, cluster_base: usize, other_clusters: usize) {
        self.entries.resize(cluster_base, TexEntry::default());
        let base = self.sources.len() as u16;
        self.alpha_sources.resize(self.sources.len(), None);
        self.sources.extend(other.sources.iter().cloned());
        self.alpha_sources.extend((0..other.sources.len()).map(|k| other.alpha_sources.get(k).cloned().flatten()));
        for c in 0..other_clusters {
            let mut e = other.entry(c);
            if e.level > 0 {
                e.source += base;
            }
            self.entries.push(e);
        }
    }

    /// The block of cluster `c`'s tiles at `level` (at most the stored
    /// level), from the cache or read from the source. None when the
    /// cluster has no tiles or the read failed.
    pub fn block(&self, c: usize, tri_count: usize, level: u8) -> Option<Arc<Block>> {
        let e = self.entry(c);
        if e.level == 0 || level == 0 {
            return None;
        }
        let src = self.sources.get(e.source as usize)?;
        self.fetch(src, (e.source, e.offset, level.min(e.level), false), tri_count, 3)
    }

    /// Whether cluster `c` is cut out by alpha tiles.
    #[inline]
    pub fn has_alpha(&self, c: usize) -> bool {
        self.entries.get(c).is_some_and(|e| e.alpha.is_some() && e.level > 0)
    }

    /// The block of cluster `c`'s alpha tiles at `level` (at most the
    /// stored level). None for an opaque cluster.
    pub fn alpha_block(&self, c: usize, tri_count: usize, level: u8) -> Option<Arc<Block>> {
        let e = self.entry(c);
        let a = e.alpha?;
        if e.level == 0 {
            return None;
        }
        let src = self.alpha_sources.get(e.source as usize)?.as_ref()?;
        self.fetch(src, (e.source, a, level.clamp(1, e.level), true), tri_count, 1)
    }

    /// A block from the cache, or read from `src`: `bytes` per point.
    fn fetch(&self, src: &Source, key: Key, tri_count: usize, bytes: usize) -> Option<Arc<Block>> {
        let (_, offset, level, _) = key;
        {
            let mut cache = self.cache.lock().unwrap();
            cache.clock += 1;
            let now = cache.clock;
            if let Some((b, used)) = cache.blocks.get_mut(&key) {
                *used = now;
                return Some(b.clone());
            }
        }
        let len = tri_count * tile_points(level) * bytes;
        let data = src.read(offset + level_offset(tri_count, level) / 3 * bytes as u64, len).ok()?;
        let block = Arc::new(Block { level, data });
        {
            let mut st = self.stats.lock().unwrap();
            st.reads += 1;
            st.bytes_read += len as u64;
        }
        let budget = *self.budget.lock().unwrap();
        let mut cache = self.cache.lock().unwrap();
        cache.clock += 1;
        let now = cache.clock;
        if cache.blocks.insert(key, (block.clone(), now)).is_none() {
            cache.bytes += len;
        }
        // Over budget: drop the least recently used blocks.
        while cache.bytes > budget && cache.blocks.len() > 1 {
            let (&old, _) = cache.blocks.iter().filter(|(k, _)| **k != key).min_by_key(|(_, (_, used))| *used).unwrap();
            if let Some((b, _)) = cache.blocks.remove(&old) {
                cache.bytes -= b.data.len();
            }
        }
        Some(block)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One cluster of two triangles with tiles at levels 1 and 2, where each
    /// point's colour encodes (triangle, level, i, j).
    fn sample_textures() -> (Textures, Vec<u8>) {
        let mut pool = Vec::new();
        for l in 1..=2u8 {
            let n = 1usize << l;
            for t in 0..2u8 {
                for i in 0..=n {
                    for j in 0..=(n - i) {
                        pool.extend_from_slice(&[t * 100 + l, i as u8 * 10, j as u8 * 10]);
                    }
                }
            }
        }
        let tex = Textures { entries: vec![TexEntry { level: 2, source: 0, offset: 0, alpha: None }], sources: vec![Arc::new(Source::Memory(Arc::new(pool.clone())))], ..Default::default() };
        (tex, pool)
    }

    #[test]
    fn blocks_hold_each_triangles_grid_points() {
        let (tex, _) = sample_textures();
        let lut = srgb_lut();
        for l in 1..=2u8 {
            let b = tex.block(0, 2, l).unwrap();
            let n = 1usize << l;
            for t in 0..2 {
                for i in 0..=n {
                    for j in 0..=(n - i) {
                        let c = b.point(t, i, j);
                        assert_eq!(c, Vec3::new(lut[t * 100 + l as usize], lut[i * 10], lut[j * 10]));
                    }
                }
            }
        }
        // Asking finer than stored gives the finest stored.
        assert_eq!(tex.block(0, 2, 5).unwrap().level, 2);
        assert!(tex.block(1, 2, 1).is_none(), "no entry, no tiles");
    }

    /// Two clusters over the same tiles (instances) share one block: the
    /// second asks for it and nothing more is read.
    #[test]
    fn instances_share_blocks() {
        let (mut tex, _) = sample_textures();
        tex.entries.push(tex.entries[0]);
        let a = tex.block(0, 2, 2).unwrap();
        let b = tex.block(1, 2, 2).unwrap();
        assert!(Arc::ptr_eq(&a, &b));
        assert_eq!(tex.stats().reads, 1);
    }

    #[test]
    fn sampling_between_points_interpolates_and_hits_points_exactly() {
        let (tex, _) = sample_textures();
        let b = tex.block(0, 2, 2).unwrap();
        // A grid point.
        assert!((b.sample(1, 0.25, 0.5) - b.point(1, 1, 2)).length() < 1e-6);
        // Halfway between (0,0) and (1,0).
        let mid = b.sample(0, 0.125, 0.0);
        assert!((mid - (b.point(0, 0, 0) + b.point(0, 1, 0)) * 0.5).length() < 1e-5);
    }

    #[test]
    fn the_cache_stays_within_budget_and_counts_reads() {
        let (tex, _) = sample_textures();
        *tex.budget.lock().unwrap() = 1;
        tex.block(0, 2, 1).unwrap();
        tex.block(0, 2, 2).unwrap();
        // Budget below one block: only the newest is kept.
        assert_eq!(tex.cache.lock().unwrap().blocks.len(), 1);
        tex.block(0, 2, 2).unwrap();
        assert_eq!(tex.stats().reads, 2, "the second request for level 2 came from the cache");
    }

    #[test]
    fn grid_index_matches_the_dicer() {
        for l in 0..=5u8 {
            let n = 1usize << l;
            let mut k = 0;
            for i in 0..=n {
                for j in 0..=(n - i) {
                    assert_eq!(tile_index(n, i, j), k);
                    k += 1;
                }
            }
            assert_eq!(k, tile_points(l));
        }
    }
}
