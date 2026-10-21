//! What machine this is, and what to render on it.
//!
//! At start-up FkJensen reads what the operating system reports (CPU model,
//! cores, SIMD features, cache sizes, memory) and measures what it does not
//! report: memory read bandwidth, and how many shading points per second the
//! cores actually manage. From those it chooses its own resolution, whether
//! to load the chess set, shadow map size, bounce light and bucket size.
//! Every choice can be overridden from the command line.
//!
//! The rates are measured, not guessed from the model name: a Raspberry Pi 1
//! and a Threadripper run the same code, and the numbers decide.

use std::time::Instant;

use rayon::prelude::*;

#[derive(Clone, Debug, Default)]
pub struct Cache {
    pub level: u32,
    /// "Data", "Instruction" or "Unified".
    pub kind: String,
    pub bytes: u64,
}

#[derive(Clone, Debug, Default)]
pub struct Hardware {
    pub arch: &'static str,
    pub model: String,
    /// The board, where the system says (a Raspberry Pi does).
    pub board: Option<String>,
    pub cores: usize,
    pub max_mhz: Option<f32>,
    /// SIMD and floating-point features the CPU reports.
    pub features: Vec<String>,
    pub caches: Vec<Cache>,
    pub mem_total: Option<u64>,
    pub mem_available: Option<u64>,
    /// Measured read bandwidth from memory, GB/s, one core and all cores.
    pub bandwidth_one: f64,
    pub bandwidth_all: f64,
    /// Whether the bandwidth buffer was larger than the last cache (if
    /// not, the figures include cache).
    pub bandwidth_past_cache: bool,
    /// Measured shading points per second, all cores, by
    /// `reyes::shading_rate` (shading alone, without dicing and lookup).
    pub shading_rate: f64,
    /// How long detection took, ms.
    pub detect_ms: f64,
}

/// What FkJensen chooses for this machine.
#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    /// The window renders at this fraction of its size and is scaled up
    /// (for a 1280 x 720 window; any other output follows `pixel_budget`).
    pub render_scale: f32,
    /// Pixels a frame can have to re-shade in the interactive time.
    pub pixel_budget: f64,
    /// Size of a `--png` render when `--size` is not given.
    pub png_size: (usize, usize),
    pub with_chess: bool,
    pub shadows: bool,
    pub shadow_res: usize,
    pub bounce: bool,
    /// Bounce gathering time per window frame, ms.
    pub bounce_budget_ms: f32,
    /// Bucket edge in pixels.
    pub tile: usize,
    /// Why, one line per choice.
    pub reasons: Vec<String>,
}

/// A frame's shading pass does about half the points per second the
/// shading benchmark does (dicing, interpolation, the cache), measured on
/// the machine FkJensen is written on: 17 M/s in the benchmark, 8.8 M/s in
/// the wide view.
const FRAME_RATE_SHARE: f64 = 0.5;
/// Benchmark rate of that machine (2 cores of a Xeon at 2.1 GHz).
const REFERENCE_RATE: f64 = 17.0e6;

/// Shading points per pixel in a typical view of the built-in set
/// (measured: 2.2 M points for 0.92 M pixels in the wide view, 1.1 M in the
/// close view; up to 5 on low-resolution renders).
const POINTS_PER_PIXEL: f64 = 3.0;
/// The window aims to re-shade a whole frame in this time when the camera
/// moves.
const INTERACTIVE_S: f64 = 0.15;
/// A `--png` render with no `--size` aims to take this long to shade.
const STILL_S: f64 = 2.0;
/// Largest automatic `--png` size.
const PNG_MAX: (usize, usize) = (1280, 720);
/// Memory the built-in set needs with the chess set, all features on, at
/// 1280 x 720 (measured peak 473 MB on x86-64, 20/10/2026), with margin.
const CHESS_MEMORY: u64 = 600 << 20;
/// Bytes read and written per triangle each time visibility runs: vertex
/// positions and the setup record. The chess set has 1.43 M triangles.
const CHESS_TRIANGLES: f64 = 1.43e6;
const BYTES_PER_TRIANGLE: f64 = 100.0;
/// Longest the chess set's per-frame memory traffic may take.
const CHESS_TRAFFIC_S: f64 = 0.25;
/// Benchmark rate below which bounce light is left off: an eighth of the
/// reference machine, where the first gather of the built-in set takes
/// 1.6 s (so about 13 s here).
const BOUNCE_MIN_RATE: f64 = REFERENCE_RATE / 8.0;
/// Share of available memory shadow maps may take.
const SHADOW_MEMORY_SHARE: f64 = 0.1;
/// Lights the built-in set has.
const BUILT_IN_LIGHTS: usize = 17;

fn read(path: &str) -> Option<String> {
    std::fs::read_to_string(path).ok()
}

fn cpuinfo_field(info: &str, key: &str) -> Option<String> {
    info.lines().find_map(|l| {
        let (k, v) = l.split_once(':')?;
        (k.trim() == key).then(|| v.trim().to_string())
    })
}

fn meminfo_bytes(info: &str, key: &str) -> Option<u64> {
    info.lines().find_map(|l| {
        let rest = l.strip_prefix(key)?.strip_prefix(':')?;
        let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
        Some(kb * 1024)
    })
}

/// Parse a sysfs cache size such as "32K" or "8192K" or "2M".
fn parse_size(s: &str) -> Option<u64> {
    let s = s.trim();
    let (num, mult) = match s.chars().last()? {
        'K' | 'k' => (&s[..s.len() - 1], 1024),
        'M' | 'm' => (&s[..s.len() - 1], 1024 * 1024),
        'G' | 'g' => (&s[..s.len() - 1], 1024 * 1024 * 1024),
        _ => (s, 1),
    };
    num.parse::<u64>().ok().map(|n| n * mult)
}

#[cfg(target_arch = "x86_64")]
fn features(_cpuinfo: &str) -> Vec<String> {
    let mut f = Vec::new();
    macro_rules! check {
        ($($name:tt),*) => { $( if std::is_x86_feature_detected!($name) { f.push($name.to_string()); } )* };
    }
    check!("sse2", "sse4.1", "avx", "avx2", "fma", "avx512f");
    f
}

#[cfg(not(target_arch = "x86_64"))]
fn features(cpuinfo: &str) -> Vec<String> {
    cpuinfo_features(cpuinfo, cfg!(target_arch = "aarch64"))
}

/// SIMD and floating-point features from an ARM /proc/cpuinfo.
#[cfg_attr(target_arch = "x86_64", allow(dead_code))]
fn cpuinfo_features(cpuinfo: &str, aarch64: bool) -> Vec<String> {
    let mut f = Vec::new();
    let list = cpuinfo_field(cpuinfo, "Features").or_else(|| cpuinfo_field(cpuinfo, "flags")).unwrap_or_default();
    for want in ["vfp", "vfpv3", "vfpv4", "neon", "asimd", "sve", "sve2"] {
        if list.split_whitespace().any(|x| x == want) {
            f.push(want.to_string());
        }
    }
    // Every aarch64 CPU has Advanced SIMD (NEON), reported or not.
    if aarch64 && !f.iter().any(|x| x == "asimd") {
        f.push("asimd".into());
    }
    f
}

fn caches() -> Vec<Cache> {
    let mut out = Vec::new();
    for i in 0..8 {
        let base = format!("/sys/devices/system/cpu/cpu0/cache/index{i}");
        let Some(size) = read(&format!("{base}/size")).and_then(|s| parse_size(&s)) else { continue };
        let level = read(&format!("{base}/level")).and_then(|s| s.trim().parse().ok()).unwrap_or(0);
        let kind = read(&format!("{base}/type")).map(|s| s.trim().to_string()).unwrap_or_default();
        out.push(Cache { level, kind, bytes: size });
    }
    out
}

/// Read bandwidth over a buffer of `bytes`, one thread or all, GB/s.
fn bandwidth(buf: &[u64], all: bool) -> f64 {
    let sum = |b: &[u64]| -> u64 {
        // Four accumulators so the adds do not wait on each other.
        let mut a = [0u64; 4];
        for c in b.chunks_exact(4) {
            a[0] = a[0].wrapping_add(c[0]);
            a[1] = a[1].wrapping_add(c[1]);
            a[2] = a[2].wrapping_add(c[2]);
            a[3] = a[3].wrapping_add(c[3]);
        }
        a[0] ^ a[1] ^ a[2] ^ a[3]
    };
    let bytes = (buf.len() * 8) as f64;
    let mut best = 0f64;
    let start = Instant::now();
    let mut passes = 0;
    // Best of several passes, for at least 30 ms or 3 passes.
    while passes < 3 || (start.elapsed().as_secs_f64() < 0.03 && passes < 50) {
        let t = Instant::now();
        let s = if all { buf.par_chunks(1 << 16).map(sum).reduce(|| 0, |a, b| a ^ b) } else { sum(buf) };
        std::hint::black_box(s);
        best = best.max(bytes / t.elapsed().as_secs_f64().max(1e-9));
        passes += 1;
    }
    best / 1e9
}

impl Hardware {
    /// Read the system and run the two measurements. Takes about 0.1 to
    /// 0.5 s.
    pub fn detect() -> Hardware {
        let t0 = Instant::now();
        let cpuinfo = read("/proc/cpuinfo").unwrap_or_default();
        let meminfo = read("/proc/meminfo").unwrap_or_default();
        let model = cpuinfo_field(&cpuinfo, "model name")
            .or_else(|| cpuinfo_field(&cpuinfo, "Processor"))
            .or_else(|| cpuinfo_field(&cpuinfo, "Hardware"))
            .unwrap_or_else(|| "unknown CPU".into());
        let board = cpuinfo_field(&cpuinfo, "Model").or_else(|| read("/sys/firmware/devicetree/base/model").map(|s| s.trim_end_matches('\0').to_string()));
        let max_mhz = read("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq")
            .and_then(|s| s.trim().parse::<f32>().ok())
            .map(|khz| khz / 1000.0)
            .or_else(|| cpuinfo_field(&cpuinfo, "cpu MHz").and_then(|s| s.parse().ok()));
        let mem_total = meminfo_bytes(&meminfo, "MemTotal");
        let mem_available = meminfo_bytes(&meminfo, "MemAvailable");
        let caches = caches();
        let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);

        // Bandwidth over a buffer twice the last cache, so it measures
        // memory and not cache, within an eighth of free memory and 512 MB.
        let last_cache = caches.iter().map(|c| c.bytes).max().unwrap_or(1 << 20);
        let free = mem_available.unwrap_or(1 << 30);
        let size = (last_cache * 2).max(32 << 20).min(512 << 20).min(free / 8).max(1 << 20);
        let past_cache = size > last_cache;
        let buf: Vec<u64> = (0..size / 8).map(|i| i.wrapping_mul(0x9e3779b97f4a7c15)).collect();
        let bandwidth_one = bandwidth(&buf, false);
        let bandwidth_all = bandwidth(&buf, true);
        drop(buf);

        let shading_rate = crate::reyes::shading_rate(0.05);
        Hardware {
            arch: std::env::consts::ARCH,
            model,
            board,
            cores,
            max_mhz,
            features: features(&cpuinfo),
            caches,
            mem_total,
            mem_available,
            bandwidth_one,
            bandwidth_all,
            bandwidth_past_cache: past_cache,
            shading_rate,
            detect_ms: t0.elapsed().as_secs_f64() * 1000.0,
        }
    }

    pub fn cache(&self, level: u32) -> Option<u64> {
        self.caches.iter().filter(|c| c.level == level && c.kind != "Instruction").map(|c| c.bytes).max()
    }

    /// Choose settings for the built-in set on this machine.
    pub fn plan(&self) -> Plan {
        let mut reasons = Vec::new();
        let bench = self.shading_rate.max(1.0);
        let rate = bench * FRAME_RATE_SHARE;

        // Window: shade a whole frame in INTERACTIVE_S. The scale applies
        // to each side of a 1280 x 720 window.
        let px = rate * INTERACTIVE_S / POINTS_PER_PIXEL;
        let pixel_budget = px;
        let render_scale = ((px / (1280.0 * 720.0)).sqrt() as f32).clamp(0.125, 1.0);
        let render_scale = (render_scale * 8.0).round().max(1.0) / 8.0;
        reasons.push(format!(
            "frames of about {:.0} K pixels ({:.0}% of a 1280 x 720 window): about {:.2} M shading points/s in a frame, {} ms per frame at {POINTS_PER_PIXEL} points per pixel",
            pixel_budget / 1e3,
            render_scale * 100.0,
            rate / 1e6,
            (INTERACTIVE_S * 1000.0) as u32
        ));

        // Still renders.
        let px = rate * STILL_S / POINTS_PER_PIXEL;
        let w = ((px * 16.0 / 9.0).sqrt() as usize).clamp(160, PNG_MAX.0) / 16 * 16;
        let png_size = (w, w * 9 / 16);
        reasons.push(format!("--png without --size renders {}x{} (about {} s of shading)", png_size.0, png_size.1, STILL_S as u32));

        // The chess set: memory, and memory traffic per frame.
        let traffic_s = CHESS_TRIANGLES * BYTES_PER_TRIANGLE / (self.bandwidth_all.max(0.01) * 1e9);
        let mem_ok = self.mem_available.is_none_or(|m| m >= CHESS_MEMORY);
        let with_chess = mem_ok && traffic_s <= CHESS_TRAFFIC_S;
        reasons.push(match (mem_ok, traffic_s <= CHESS_TRAFFIC_S) {
            (true, true) => format!(
                "chess set loaded: {} available, its triangles take {:.0} ms of memory traffic per frame",
                self.mem_available.map(|m| format!("{} MB", m >> 20)).unwrap_or("memory size unknown, assumed".into()),
                traffic_s * 1000.0
            ),
            (false, _) => format!("chess set left out: {} MB available, it needs about {} MB", self.mem_available.unwrap_or(0) >> 20, CHESS_MEMORY >> 20),
            (true, false) => format!("chess set left out: its triangles would take {:.0} ms of memory traffic per frame at {:.2} GB/s", traffic_s * 1000.0, self.bandwidth_all),
        });

        // Shadow maps: as fine as the speed allows, within a share of memory.
        let speed = bench / REFERENCE_RATE;
        let mut shadow_res = if speed >= 0.5 {
            512
        } else if speed >= 0.1 {
            256
        } else {
            128
        };
        let budget = (self.mem_available.unwrap_or(4 << 30) as f64 * SHADOW_MEMORY_SHARE) as usize;
        while shadow_res > 32 && BUILT_IN_LIGHTS * 6 * shadow_res * shadow_res * 4 > budget {
            shadow_res /= 2;
        }
        let shadows = shadow_res >= 64;
        reasons.push(if shadows {
            format!("shadow maps {shadow_res} x {shadow_res} per face ({:.1} MB for {BUILT_IN_LIGHTS} lights)", (BUILT_IN_LIGHTS * 6 * shadow_res * shadow_res * 4) as f64 / 1e6)
        } else {
            "shadow maps off: not enough memory for 64 x 64".into()
        });

        let bounce = bench >= BOUNCE_MIN_RATE;
        let bounce_budget_ms = (30.0 * speed.min(1.0)).max(10.0) as f32;
        reasons.push(if bounce {
            format!("bounce light on, gathered {bounce_budget_ms:.0} ms per window frame")
        } else {
            format!("bounce light off: {:.2} M benchmark points/s, it needs {:.1} M", bench / 1e6, BOUNCE_MIN_RATE / 1e6)
        });

        // Buckets: depth and triangle id take 8 bytes a pixel; a bucket
        // should fit in half the L2 cache. At most 64, at least 16.
        let l2 = self.cache(2).or(self.cache(1).map(|c| c * 4)).unwrap_or(256 << 10);
        let mut tile = 64;
        while tile > 16 && (tile * tile * 8) as u64 > l2 / 2 {
            tile /= 2;
        }
        reasons.push(format!("buckets {tile} x {tile} ({} KB of depth and ids, L2 {} KB)", tile * tile * 8 / 1024, l2 >> 10));

        Plan { render_scale, pixel_budget, png_size, with_chess, shadows, shadow_res, bounce, bounce_budget_ms, tile, reasons }
    }

    pub fn report(&self) -> String {
        let mut s = String::new();
        let gb = |b: u64| format!("{:.2} GB", b as f64 / (1u64 << 30) as f64);
        s += &format!("CPU: {} ({}), {} cores", self.model, self.arch, self.cores);
        if let Some(m) = self.max_mhz {
            s += &format!(", up to {m:.0} MHz");
        }
        s += "\n";
        if let Some(b) = &self.board {
            s += &format!("board: {b}\n");
        }
        s += &format!("features: {}\n", if self.features.is_empty() { "none reported".into() } else { self.features.join(" ") });
        if self.caches.is_empty() {
            s += "caches: not reported\n";
        } else {
            let list: Vec<String> = self
                .caches
                .iter()
                .map(|c| format!("L{} {} {} KB", c.level, c.kind.to_lowercase(), c.bytes >> 10))
                .collect();
            s += &format!("caches (per core or shared): {}\n", list.join(", "));
        }
        s += &format!(
            "memory: {} total, {} available\n",
            self.mem_total.map(gb).unwrap_or("unknown".into()),
            self.mem_available.map(gb).unwrap_or("unknown".into())
        );
        s += &format!(
            "measured: memory read {:.2} GB/s on one core, {:.2} GB/s on all{}; shading benchmark {:.2} M points/s on all cores ({:.0} ms to measure)",
            self.bandwidth_one,
            self.bandwidth_all,
            if self.bandwidth_past_cache { "" } else { " (buffer within the last cache: includes cache)" },
            self.shading_rate / 1e6,
            self.detect_ms
        );
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn machine(rate: f64, mem_mb: u64, bw: f64, l2_kb: u64) -> Hardware {
        Hardware {
            cores: 1,
            mem_available: Some(mem_mb << 20),
            bandwidth_all: bw,
            bandwidth_one: bw,
            shading_rate: rate,
            caches: vec![Cache { level: 2, kind: "Unified".into(), bytes: l2_kb << 10 }],
            ..Default::default()
        }
    }

    #[test]
    fn sizes_parse() {
        assert_eq!(parse_size("32K"), Some(32 << 10));
        assert_eq!(parse_size("8M\n"), Some(8 << 20));
        assert_eq!(parse_size("512"), Some(512));
        assert_eq!(meminfo_bytes("MemTotal:         177564 kB\nMemAvailable:      60000 kB\n", "MemAvailable"), Some(60000 * 1024));
        assert_eq!(cpuinfo_field("model name\t: ARMv6-compatible processor rev 7 (v6l)\n", "model name").as_deref(), Some("ARMv6-compatible processor rev 7 (v6l)"));
        // A Raspberry Pi 1: VFP, no NEON.
        let pi1 = "processor\t: 0\nmodel name\t: ARMv6-compatible processor rev 7 (v6l)\nFeatures\t: half thumb fastmult vfp edsp java tls\nHardware\t: BCM2835\nModel\t\t: Raspberry Pi Model B Rev 1\n";
        assert_eq!(cpuinfo_features(pi1, false), vec!["vfp".to_string()]);
        assert_eq!(cpuinfo_field(pi1, "Model").as_deref(), Some("Raspberry Pi Model B Rev 1"));
        let a72 = "Features\t: fp asimd evtstrm crc32 cpuid\n";
        assert_eq!(cpuinfo_features(a72, true), vec!["asimd".to_string()]);
    }

    /// The Raspberry Pi 1 Model B Rev 1 as measured on 20/10/2026: about
    /// 80 K shading points/s in a frame (so about 160 K in the benchmark),
    /// 59 MB available.
    #[test]
    fn a_pi_one_gets_a_small_light_plan() {
        let p = machine(160e3, 59, 0.3, 128).plan();
        assert!(!p.with_chess);
        assert!(!p.bounce);
        assert_eq!(p.render_scale, 0.125);
        assert!(p.png_size.0 <= 320, "{:?}", p.png_size);
        assert!(p.shadows && p.shadow_res <= 128, "{}", p.shadow_res);
    }

    /// The machine FkJensen is written on: 2 cores, 17 M benchmark points/s.
    #[test]
    fn a_desktop_gets_everything() {
        let p = machine(17e6, 8000, 8.0, 1024).plan();
        assert!(p.with_chess && p.bounce && p.shadows);
        assert_eq!(p.shadow_res, 512);
        assert_eq!(p.png_size, PNG_MAX);
        assert!(p.render_scale >= 0.5);
        assert_eq!(p.tile, 64);
    }

    #[test]
    fn little_memory_shrinks_shadow_maps() {
        let p = machine(17e6, 40, 8.0, 1024).plan();
        assert!(p.shadow_res < 512);
        assert!((BUILT_IN_LIGHTS * 6 * p.shadow_res * p.shadow_res * 4) as f64 <= 40.0 * 1048576.0 * SHADOW_MEMORY_SHARE || !p.shadows);
    }

    #[test]
    fn detection_runs_and_measures() {
        let h = Hardware::detect();
        assert!(h.cores >= 1);
        assert!(h.bandwidth_one > 0.0 && h.shading_rate > 0.0);
        let p = h.plan();
        assert!(p.render_scale > 0.0 && p.render_scale <= 1.0);
        assert!(!p.reasons.is_empty());
    }
}
