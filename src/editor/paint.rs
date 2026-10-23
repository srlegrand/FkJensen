//! Draws egui on the CPU into a 0x00RRGGBB buffer, the same one the
//! renderer presents through softbuffer. No GPU anywhere.
//!
//! egui hands over triangles in points with premultiplied sRGB vertex
//! colours and texture coordinates; its own edge feathering does the
//! antialiasing, so one sample per pixel centre is enough. Blending is
//! premultiplied "over" in sRGB space, as egui's own backends do.
//!
//! Textures: egui's own (the font atlas and any images it manages) arrive
//! as deltas; user textures (the importer's preview) are set directly with
//! `set_user`. Bands of rows are painted in parallel.

use std::collections::HashMap;

use egui::epaint::{textures::TexturesDelta, ClippedPrimitive, Primitive, TextureId, Vertex};
use egui::{ImageData, TextureFilter};
use rayon::prelude::*;

/// Rows per parallel band.
const BAND: usize = 32;

struct Texture {
    w: usize,
    h: usize,
    /// Premultiplied sRGBA.
    px: Vec<[u8; 4]>,
    linear: bool,
}

impl Texture {
    fn texel(&self, x: i32, y: i32) -> [f32; 4] {
        let x = x.clamp(0, self.w as i32 - 1) as usize;
        let y = y.clamp(0, self.h as i32 - 1) as usize;
        let p = self.px[y * self.w + x];
        [p[0] as f32, p[1] as f32, p[2] as f32, p[3] as f32]
    }

    /// Premultiplied colour at (u, v) in 0..1, channels 0..255.
    fn sample(&self, u: f32, v: f32) -> [f32; 4] {
        let x = u * self.w as f32 - 0.5;
        let y = v * self.h as f32 - 0.5;
        if !self.linear {
            return self.texel(x.round() as i32, y.round() as i32);
        }
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let (x0, y0) = (x0 as i32, y0 as i32);
        let a = self.texel(x0, y0);
        let b = self.texel(x0 + 1, y0);
        let c = self.texel(x0, y0 + 1);
        let d = self.texel(x0 + 1, y0 + 1);
        let mut o = [0.0; 4];
        for k in 0..4 {
            let top = a[k] + (b[k] - a[k]) * fx;
            let bot = c[k] + (d[k] - c[k]) * fx;
            o[k] = top + (bot - top) * fy;
        }
        o
    }
}

#[derive(Default)]
pub struct Painter {
    textures: HashMap<TextureId, Texture>,
}

impl Painter {
    pub fn new() -> Painter {
        Painter::default()
    }

    /// Apply egui's new or changed textures (before painting).
    pub fn set_textures(&mut self, delta: &TexturesDelta) {
        for (id, deltas) in &delta.set {
            for d in deltas {
                let ImageData::Color(img) = &d.image;
                let [w, h] = img.size;
                let linear = d.options.magnification == TextureFilter::Linear;
                let px: Vec<[u8; 4]> = img.pixels.iter().map(|c| c.to_array()).collect();
                match d.pos {
                    None => {
                        self.textures.insert(*id, Texture { w, h, px, linear });
                    }
                    Some([x0, y0]) => {
                        let Some(t) = self.textures.get_mut(id) else { continue };
                        t.linear = linear;
                        for y in 0..h {
                            for x in 0..w {
                                let (tx, ty) = (x0 + x, y0 + y);
                                if tx < t.w && ty < t.h {
                                    t.px[ty * t.w + tx] = px[y * w + x];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Drop the textures egui is done with (after painting).
    pub fn free_textures(&mut self, delta: &TexturesDelta) {
        for id in &delta.free {
            self.textures.remove(id);
        }
    }

    /// Set user texture `id` from a 0x00RRGGBB picture.
    pub fn set_user(&mut self, id: u64, w: usize, h: usize, rgb: &[u32]) {
        let px = rgb.iter().map(|&c| [(c >> 16) as u8, (c >> 8) as u8, c as u8, 255]).collect();
        self.textures.insert(TextureId::User(id), Texture { w, h, px, linear: true });
    }

    /// Paint `prims` over `buf` (`w` x `h`, 0x00RRGGBB). Positions are in
    /// points; `ppp` is pixels per point.
    pub fn paint(&self, prims: &[ClippedPrimitive], ppp: f32, buf: &mut [u32], w: usize, h: usize) {
        // Triangles in pixels, with their clip rectangle and texture, so the
        // bands only look at what reaches them.
        let mut tris: Vec<Tri> = Vec::new();
        for p in prims {
            let Primitive::Mesh(mesh) = &p.primitive else { continue };
            let Some(tex) = self.textures.get(&mesh.texture_id) else { continue };
            let clip = [
                (p.clip_rect.min.x * ppp).round().max(0.0),
                (p.clip_rect.min.y * ppp).round().max(0.0),
                (p.clip_rect.max.x * ppp).round().min(w as f32),
                (p.clip_rect.max.y * ppp).round().min(h as f32),
            ];
            if clip[0] >= clip[2] || clip[1] >= clip[3] {
                continue;
            }
            for t in mesh.indices.chunks_exact(3) {
                let v = |i: u32| mesh.vertices.get(i as usize).copied();
                let (Some(a), Some(b), Some(c)) = (v(t[0]), v(t[1]), v(t[2])) else { continue };
                if let Some(tri) = Tri::new([a, b, c], ppp, clip, tex) {
                    tris.push(tri);
                }
            }
        }
        // Each band's triangles, in paint order.
        let bands = h.div_ceil(BAND);
        let mut bins: Vec<Vec<u32>> = vec![Vec::new(); bands];
        for (k, t) in tris.iter().enumerate() {
            let b0 = (t.lo[1] as usize / BAND).min(bands - 1);
            let b1 = ((t.hi[1] as usize).saturating_sub(1) / BAND).min(bands - 1);
            for b in &mut bins[b0..=b1] {
                b.push(k as u32);
            }
        }
        buf[..w * h].par_chunks_mut(BAND * w).zip(bins.par_iter()).enumerate().for_each(|(band, (rows, bin))| {
            let y0 = band * BAND;
            let y1 = y0 + rows.len() / w;
            for &k in bin {
                tris[k as usize].fill(rows, w, y0, y1);
            }
        });
    }
}

/// A triangle ready to fill: edge functions and attribute planes in pixels.
struct Tri<'a> {
    /// Pixel bounds, already clipped.
    lo: [f32; 2],
    hi: [f32; 2],
    /// Edge i: e(x, y) = a*x + b*y + c, >= 0 inside.
    edges: [[f32; 3]; 3],
    /// Top-left rule: an edge with e == 0 counts as inside.
    owns: [bool; 3],
    /// Plane per attribute: value = base + dx*x + dy*y. r, g, b, a, u, v.
    planes: [[f32; 3]; 6],
    tex: &'a Texture,
    /// One colour, one texel (most of a UI): skip the texture per pixel.
    flat: Option<[f32; 4]>,
}

impl<'a> Tri<'a> {
    fn new(v: [Vertex; 3], ppp: f32, clip: [f32; 4], tex: &'a Texture) -> Option<Tri<'a>> {
        let p: [[f32; 2]; 3] = v.map(|v| [v.pos.x * ppp, v.pos.y * ppp]);
        let area = (p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[2][0] - p[0][0]) * (p[1][1] - p[0][1]);
        if area.abs() < 1e-6 || !area.is_finite() {
            return None;
        }
        // egui's winding varies: orient so the inside is positive.
        let (p, v) = if area < 0.0 { ([p[0], p[2], p[1]], [v[0], v[2], v[1]]) } else { (p, v) };
        let area = area.abs();
        let lo = [p.iter().map(|q| q[0]).fold(f32::INFINITY, f32::min).floor().max(clip[0]), p.iter().map(|q| q[1]).fold(f32::INFINITY, f32::min).floor().max(clip[1])];
        let hi = [p.iter().map(|q| q[0]).fold(f32::NEG_INFINITY, f32::max).ceil().min(clip[2]), p.iter().map(|q| q[1]).fold(f32::NEG_INFINITY, f32::max).ceil().min(clip[3])];
        if lo[0] >= hi[0] || lo[1] >= hi[1] {
            return None;
        }
        let mut edges = [[0.0; 3]; 3];
        let mut owns = [false; 3];
        for i in 0..3 {
            // Edge opposite vertex i, from j to k.
            let (j, k) = ((i + 1) % 3, (i + 2) % 3);
            let a = p[j][1] - p[k][1];
            let b = p[k][0] - p[j][0];
            let c = p[j][0] * p[k][1] - p[k][0] * p[j][1];
            edges[i] = [a / area, b / area, c / area];
            // In y-down screen space with this orientation: top edges are
            // horizontal going left, left edges go down.
            owns[i] = (a == 0.0 && b < 0.0) || a > 0.0;
        }
        let attr = |f: &dyn Fn(&Vertex) -> f32| -> [f32; 3] {
            let s = v.map(|v| f(&v));
            // value = sum s_i * e_i(x, y)
            let mut pl = [0.0; 3];
            for i in 0..3 {
                for k in 0..3 {
                    pl[k] += s[i] * edges[i][k];
                }
            }
            pl
        };
        let planes = [
            attr(&|v| v.color.r() as f32),
            attr(&|v| v.color.g() as f32),
            attr(&|v| v.color.b() as f32),
            attr(&|v| v.color.a() as f32),
            attr(&|v| v.uv.x),
            attr(&|v| v.uv.y),
        ];
        let same_colour = v[0].color == v[1].color && v[1].color == v[2].color;
        let texel = |v: &Vertex| ((v.uv.x * tex.w as f32) as i32, (v.uv.y * tex.h as f32) as i32);
        let one_texel = texel(&v[0]) == texel(&v[1]) && texel(&v[1]) == texel(&v[2]);
        let flat = (same_colour && one_texel).then(|| {
            let t = tex.sample(v[0].uv.x, v[0].uv.y);
            let c = v[0].color.to_array();
            [0, 1, 2, 3].map(|k| t[k] * c[k] as f32 / 255.0)
        });
        Some(Tri { lo, hi, edges, owns, planes, tex, flat })
    }

    /// The pixels of row centre `py` inside every edge: a span from the
    /// edge equations, one pixel wider each side, where the exact test is
    /// done.
    fn span(&self, py: f32) -> (usize, usize) {
        let (mut lo, mut hi) = (self.lo[0], self.hi[0]);
        for e in &self.edges {
            // a*x + k >= 0
            let k = e[1] * py + e[2];
            if e[0] > 0.0 {
                lo = lo.max((-k / e[0] - 0.5).floor() - 1.0);
            } else if e[0] < 0.0 {
                hi = hi.min((-k / e[0] - 0.5).ceil() + 2.0);
            } else if k < 0.0 {
                return (0, 0);
            }
        }
        let lo = lo.max(self.lo[0]);
        let hi = hi.min(self.hi[0]);
        if lo >= hi { (0, 0) } else { (lo as usize, hi as usize) }
    }

    fn fill(&self, rows: &mut [u32], w: usize, band_y0: usize, band_y1: usize) {
        let y_lo = (self.lo[1] as usize).max(band_y0);
        let y_hi = (self.hi[1] as usize).min(band_y1);
        let at = |pl: &[f32; 3], x: f32, y: f32| pl[0] * x + pl[1] * y + pl[2];
        // An opaque flat colour is written, not blended.
        let solid = self.flat.filter(|c| c[3] >= 254.5).map(|c| {
            let q = |v: f32| (v + 0.5).clamp(0.0, 255.0) as u32;
            (q(c[0]) << 16) | (q(c[1]) << 8) | q(c[2])
        });
        // Premultiplied source times 256 per channel (b, g, r by shift / 8)
        // and what is kept of the destination, of 256.
        let flat_blend = self.flat.filter(|_| solid.is_none()).map(|c| {
            let q = |v: f32| (v * 256.0).round().clamp(0.0, 65280.0) as u32;
            ([q(c[2]), q(c[1]), q(c[0])], 256 - ((c[3] / 255.0 * 256.0).round() as u32).min(256))
        });
        for y in y_lo..y_hi {
            let py = y as f32 + 0.5;
            let (x_lo, x_hi) = self.span(py);
            if x_lo >= x_hi {
                continue;
            }
            let row = &mut rows[(y - band_y0) * w..(y - band_y0 + 1) * w];
            let inside = |x: usize| {
                let px = x as f32 + 0.5;
                (0..3).all(|i| {
                    let e = at(&self.edges[i], px, py);
                    e > 0.0 || (e == 0.0 && self.owns[i])
                })
            };
            // A triangle's row is one run: find its ends, then fill it.
            let Some(first) = (x_lo..x_hi).find(|&x| inside(x)) else { continue };
            let last = (first..x_hi).rev().find(|&x| inside(x)).unwrap_or(first);
            if let Some((add, keep)) = flat_blend {
                // One translucent colour (shadows, the dimmed backdrop):
                // integer blend, the same for every pixel.
                for d in &mut row[first..=last] {
                    let ch = |sh: u32| -> u32 { ((add[sh as usize / 8] + ((*d >> sh) & 255) * keep + 128) >> 8).min(255) };
                    *d = (ch(16) << 16) | (ch(8) << 8) | ch(0);
                }
                continue;
            }
            for x in first..=last {
                let px = x as f32 + 0.5;
                if let Some(c) = solid {
                    row[x] = c;
                    continue;
                }
                let src = match self.flat {
                    Some(c) => c,
                    None => {
                        let t = self.tex.sample(at(&self.planes[4], px, py), at(&self.planes[5], px, py));
                        [0, 1, 2, 3].map(|k| t[k] * at(&self.planes[k], px, py).clamp(0.0, 255.0) / 255.0)
                    }
                };
                let a = src[3];
                if a <= 0.0 && src[0] <= 0.0 && src[1] <= 0.0 && src[2] <= 0.0 {
                    continue;
                }
                let keep = 1.0 - a / 255.0;
                let d = row[x];
                let ch = |s: f32, sh: u32| -> u32 { (s + ((d >> sh) & 255) as f32 * keep + 0.5).clamp(0.0, 255.0) as u32 };
                row[x] = (ch(src[0], 16) << 16) | (ch(src[1], 8) << 8) | ch(src[2], 0);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A full egui frame with text and widgets paints, and its panel colour
    /// and text both reach the buffer.
    #[test]
    fn egui_frame_paints() {
        let ctx = egui::Context::default();
        let mut painter = Painter::new();
        let (w, h) = (320usize, 200usize);
        let raw = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(w as f32, h as f32))),
            ..Default::default()
        };
        let mut out = None;
        // Two passes: the first lays out, fonts arrive with it.
        for _ in 0..2 {
            out = Some(ctx.run_ui(raw.clone(), |ui| {
                egui::CentralPanel::default().show(ui, |ui| {
                    ui.heading("FkJensen");
                    let _ = ui.button("Import...");
                });
            }));
            let o = out.as_ref().unwrap();
            painter.set_textures(&o.textures_delta);
        }
        let o = out.unwrap();
        let prims = ctx.tessellate(o.shapes, o.pixels_per_point);
        let mut buf = vec![0x00ff00ffu32; w * h];
        painter.paint(&prims, o.pixels_per_point, &mut buf, w, h);
        // The panel covers the magenta.
        assert!(buf.iter().all(|&c| c != 0x00ff00ff), "unpainted pixels");
        // Text: some pixels much brighter than the panel.
        let bg = buf[(h - 2) * w + w - 2];
        let lum = |c: u32| ((c >> 16) & 255) + ((c >> 8) & 255) + (c & 255);
        let bright = buf.iter().filter(|&&c| lum(c) > lum(bg) + 150).count();
        assert!(bright > 50, "{bright} text pixels");
    }

    /// Two triangles sharing an edge cover each pixel exactly once: a
    /// half-transparent quad comes out even, no seam down its diagonal.
    #[test]
    fn shared_edges_blend_once() {
        let mut painter = Painter::new();
        painter.set_user(7, 1, 1, &[0xffffff]);
        let c = egui::Color32::from_rgba_premultiplied(100, 100, 100, 128);
        let v = |x: f32, y: f32| Vertex { pos: egui::pos2(x, y), uv: egui::pos2(0.5, 0.5), color: c };
        let mesh = egui::epaint::Mesh {
            indices: vec![0, 1, 2, 0, 2, 3],
            vertices: vec![v(0.0, 0.0), v(16.0, 0.0), v(16.0, 16.0), v(0.0, 16.0)],
            texture_id: TextureId::User(7),
        };
        let prims = [ClippedPrimitive { clip_rect: egui::Rect::EVERYTHING, primitive: Primitive::Mesh(mesh) }];
        let mut buf = vec![0u32; 16 * 16];
        painter.paint(&prims, 1.0, &mut buf, 16, 16);
        assert!(buf.iter().all(|&p| p == buf[0]), "seam");
        assert_eq!(buf[0] & 255, 100);
    }
}
