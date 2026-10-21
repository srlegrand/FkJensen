//! Output straight to the Linux framebuffer, with no display server.
//!
//! When neither X nor Wayland is running (a Raspberry Pi on its text
//! console), frames are written to `/dev/fb0` and the keyboard and mouse are
//! read from `/dev/input/event*`. No window system, no GPU: the frame is
//! copied into the framebuffer's memory, converted to its pixel format.
//!
//! The framebuffer's size, stride and pixel format come from the
//! `FBIOGET_VSCREENINFO` and `FBIOGET_FSCREENINFO` ioctls. The user needs to
//! be in the `video` group for the framebuffer and the `input` group for
//! the keyboard and mouse.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::io::AsRawFd;

use winit::keyboard::KeyCode;

/// How the framebuffer stores a pixel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Format {
    pub bits: u32,
    /// Bit offset and length of each channel.
    pub red: (u32, u32),
    pub green: (u32, u32),
    pub blue: (u32, u32),
}

impl Format {
    /// Convert one 0x00RRGGBB pixel.
    #[inline]
    pub fn encode(&self, c: u32) -> u32 {
        let ch = |v: u32, (off, len): (u32, u32)| -> u32 {
            if len == 0 {
                return 0;
            }
            (v >> (8 - len.min(8))) << off
        };
        ch((c >> 16) & 0xff, self.red) | ch((c >> 8) & 0xff, self.green) | ch(c & 0xff, self.blue)
    }

    pub fn bytes(&self) -> usize {
        self.bits.div_ceil(8) as usize
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Geometry {
    pub width: usize,
    pub height: usize,
    /// Bytes per row.
    pub stride: usize,
    /// Byte offset of the visible area (from x/y offset in a virtual screen).
    pub offset: usize,
    pub format: Format,
}

pub struct Framebuffer {
    file: File,
    pub geometry: Geometry,
    line: Vec<u8>,
}

const FBIOGET_VSCREENINFO: libc::c_ulong = 0x4600;
const FBIOGET_FSCREENINFO: libc::c_ulong = 0x4602;

/// Read the framebuffer's geometry with the two ioctls.
fn query(fd: libc::c_int) -> Result<Geometry, String> {
    // fb_var_screeninfo is 40 u32s; fb_fix_screeninfo is under 128 bytes on
    // 32 and 64 bit.
    let mut var = [0u32; 40];
    let mut fix = [0u8; 128];
    // SAFETY: the buffers are larger than the kernel structs.
    unsafe {
        if libc::ioctl(fd, FBIOGET_VSCREENINFO as _, var.as_mut_ptr()) != 0 {
            return Err(format!("FBIOGET_VSCREENINFO: {}", std::io::Error::last_os_error()));
        }
        if libc::ioctl(fd, FBIOGET_FSCREENINFO as _, fix.as_mut_ptr()) != 0 {
            return Err(format!("FBIOGET_FSCREENINFO: {}", std::io::Error::last_os_error()));
        }
    }
    // fb_fix_screeninfo: char id[16]; unsigned long smem_start; u32 smem_len,
    // type, type_aux, visual; u16 xpanstep, ypanstep, ywrapstep; u32
    // line_length (aligned to 4).
    let ulong = std::mem::size_of::<libc::c_ulong>();
    let line_at = (16 + ulong + 16 + 6).div_ceil(4) * 4;
    let stride = u32::from_ne_bytes(fix[line_at..line_at + 4].try_into().unwrap()) as usize;
    let format = Format { bits: var[6], red: (var[8], var[9]), green: (var[11], var[12]), blue: (var[14], var[15]) };
    let (xoff, yoff) = (var[4] as usize, var[5] as usize);
    Ok(Geometry {
        width: var[0] as usize,
        height: var[1] as usize,
        stride,
        offset: yoff * stride + xoff * format.bytes(),
        format,
    })
}

impl Framebuffer {
    pub fn open(path: &str) -> Result<Framebuffer, String> {
        let file = OpenOptions::new().read(true).write(true).open(path).map_err(|e| {
            let hint = if e.kind() == std::io::ErrorKind::PermissionDenied { " (add the user to the video group)" } else { "" };
            format!("no display server, and cannot open {path}: {e}{hint}")
        })?;
        let geometry = query(file.as_raw_fd())?;
        if !matches!(geometry.format.bits, 16 | 24 | 32) {
            return Err(format!("{path}: {} bits per pixel is not handled", geometry.format.bits));
        }
        Ok(Framebuffer { file, line: vec![0; geometry.stride], geometry })
    }

    /// A framebuffer over any file, with the geometry given (for tests).
    #[cfg(test)]
    pub fn with_geometry(file: File, geometry: Geometry) -> Framebuffer {
        Framebuffer { file, line: vec![0; geometry.stride], geometry }
    }

    /// Show `src` (0x00RRGGBB, `sw` x `sh`) scaled to the whole screen,
    /// nearest pixel, and draw a cursor at (cx, cy) screen pixels. A screen
    /// row is converted once per source row and written again for the rows
    /// that repeat it, so a frame rendered small costs little to show.
    pub fn show(&mut self, src: &[u32], sw: usize, sh: usize, cursor: Option<(usize, usize)>) -> std::io::Result<()> {
        let g = self.geometry;
        let bpp = g.format.bytes();
        let xs: Vec<usize> = (0..g.width).map(|x| (x * sw / g.width).min(sw - 1)).collect();
        self.file.seek(SeekFrom::Start(g.offset as u64))?;
        let mut built: Option<usize> = None;
        for y in 0..g.height {
            let sy = (y * sh / g.height).min(sh - 1);
            let near = cursor.is_some_and(|(_, cy)| (y as i64 - cy as i64).abs() <= 8);
            if near || built != Some(sy) {
                let row = &src[sy * sw..sy * sw + sw];
                for x in 0..g.width {
                    let mut c = row[xs[x]];
                    if near {
                        let (cx, cy) = cursor.unwrap();
                        // A small white cross with a dark outline.
                        let (dx, dy) = (x as i64 - cx as i64, y as i64 - cy as i64);
                        if (dx == 0 && dy.abs() <= 7) || (dy == 0 && dx.abs() <= 7) {
                            c = 0xffffff;
                        } else if (dx.abs() == 1 && dy.abs() <= 8) || (dy.abs() == 1 && dx.abs() <= 8) {
                            c = 0x000000;
                        }
                    }
                    let v = g.format.encode(c);
                    let at = x * bpp;
                    self.line[at..at + bpp].copy_from_slice(&v.to_le_bytes()[..bpp]);
                }
                built = if near { None } else { Some(sy) };
            }
            self.file.write_all(&self.line[..g.width * bpp])?;
            if g.stride > g.width * bpp {
                self.file.seek(SeekFrom::Current((g.stride - g.width * bpp) as i64))?;
            }
        }
        Ok(())
    }
}

/// What the input devices reported since the last poll.
#[derive(Default, Debug, PartialEq)]
pub struct Events {
    pub keys: Vec<(KeyCode, bool)>,
    /// Mouse buttons 0 left, 1 middle, 2 right.
    pub buttons: Vec<(usize, bool)>,
    pub motion: (i32, i32),
    pub wheel: i32,
}

pub struct Input {
    devices: Vec<File>,
}

const EV_KEY: u16 = 1;
const EV_REL: u16 = 2;

/// Linux key codes (input-event-codes.h) to winit's.
fn key_code(code: u16) -> Option<KeyCode> {
    use KeyCode::*;
    Some(match code {
        1 => Escape,
        2 => Digit1,
        3 => Digit2,
        11 => Digit0,
        12 => Minus,
        13 => Equal,
        14 => Backspace,
        15 => Tab,
        16 => KeyQ,
        17 => KeyW,
        18 => KeyE,
        19 => KeyR,
        23 => KeyI,
        25 => KeyP,
        26 => BracketLeft,
        27 => BracketRight,
        29 => ControlLeft,
        30 => KeyA,
        31 => KeyS,
        32 => KeyD,
        33 => KeyF,
        34 => KeyG,
        35 => KeyH,
        37 => KeyK,
        38 => KeyL,
        42 => ShiftLeft,
        44 => KeyZ,
        45 => KeyX,
        46 => KeyC,
        48 => KeyB,
        50 => KeyM,
        51 => Comma,
        52 => Period,
        54 => ShiftRight,
        78 => NumpadAdd,
        74 => NumpadSubtract,
        97 => ControlRight,
        103 => ArrowUp,
        104 => PageUp,
        105 => ArrowLeft,
        106 => ArrowRight,
        108 => ArrowDown,
        109 => PageDown,
        111 => Delete,
        _ => return None,
    })
}

/// Parse raw `struct input_event`s: a timeval (two C longs), then u16 type,
/// u16 code, i32 value.
pub fn parse(bytes: &[u8], long_bytes: usize, out: &mut Events) {
    let size = 2 * long_bytes + 8;
    for ev in bytes.chunks_exact(size) {
        let t = 2 * long_bytes;
        let kind = u16::from_ne_bytes([ev[t], ev[t + 1]]);
        let code = u16::from_ne_bytes([ev[t + 2], ev[t + 3]]);
        let value = i32::from_ne_bytes([ev[t + 4], ev[t + 5], ev[t + 6], ev[t + 7]]);
        match kind {
            EV_KEY if value != 2 => {
                // value 2 is auto-repeat: ignored, as in the window.
                let down = value == 1;
                match code {
                    0x110 => out.buttons.push((0, down)),
                    0x111 => out.buttons.push((2, down)),
                    0x112 => out.buttons.push((1, down)),
                    c => {
                        if let Some(k) = key_code(c) {
                            out.keys.push((k, down));
                        }
                    }
                }
            }
            EV_REL => match code {
                0 => out.motion.0 += value,
                1 => out.motion.1 += value,
                8 => out.wheel += value,
                _ => {}
            },
            _ => {}
        }
    }
}

impl Input {
    /// Open every `/dev/input/event*` that can be read.
    pub fn open() -> Input {
        let mut devices = Vec::new();
        if let Ok(dir) = std::fs::read_dir("/dev/input") {
            for e in dir.flatten() {
                let name = e.file_name().to_string_lossy().into_owned();
                if name.starts_with("event") {
                    if let Ok(f) = OpenOptions::new().read(true).custom_flags(libc::O_NONBLOCK).open(e.path()) {
                        devices.push(f);
                    }
                }
            }
        }
        Input { devices }
    }

    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    pub fn poll(&mut self) -> Events {
        let mut out = Events::default();
        let long = std::mem::size_of::<libc::c_long>();
        let mut buf = [0u8; 4096];
        for d in &mut self.devices {
            while let Ok(n) = d.read(&mut buf) {
                if n == 0 {
                    break;
                }
                parse(&buf[..n], long, &mut out);
            }
        }
        out
    }
}

/// Whether there is no display server to open a window on.
pub fn no_display_server() -> bool {
    std::env::var_os("DISPLAY").is_none() && std::env::var_os("WAYLAND_DISPLAY").is_none() && std::env::var_os("WAYLAND_SOCKET").is_none()
}

/// Hide the text console's cursor and clear it, and put it back on drop.
pub struct Console;

impl Console {
    pub fn take() -> Console {
        print!("\x1b[?25l\x1b[2J");
        let _ = std::io::stdout().flush();
        Console
    }
}

impl Drop for Console {
    fn drop(&mut self) {
        print!("\x1b[?25h\x1b[2J\x1b[H");
        let _ = std::io::stdout().flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pixel_formats() {
        let rgb565 = Format { bits: 16, red: (11, 5), green: (5, 6), blue: (0, 5) };
        assert_eq!(rgb565.encode(0xff0000), 0xf800);
        assert_eq!(rgb565.encode(0x00ff00), 0x07e0);
        assert_eq!(rgb565.encode(0x0000ff), 0x001f);
        assert_eq!(rgb565.encode(0xffffff), 0xffff);
        let xrgb = Format { bits: 32, red: (16, 8), green: (8, 8), blue: (0, 8) };
        assert_eq!(xrgb.encode(0x123456), 0x123456);
        let xbgr = Format { bits: 32, red: (0, 8), green: (8, 8), blue: (16, 8) };
        assert_eq!(xbgr.encode(0x123456), 0x563412);
    }

    #[test]
    fn events_parse_on_32_and_64_bit() {
        for long in [4usize, 8] {
            let mut bytes = Vec::new();
            let mut push = |kind: u16, code: u16, value: i32| {
                bytes.extend(std::iter::repeat_n(0u8, 2 * long));
                bytes.extend_from_slice(&kind.to_ne_bytes());
                bytes.extend_from_slice(&code.to_ne_bytes());
                bytes.extend_from_slice(&value.to_ne_bytes());
            };
            push(EV_KEY, 17, 1); // W down
            push(EV_KEY, 17, 2); // repeat, ignored
            push(EV_KEY, 0x110, 1); // left button
            push(EV_REL, 0, 5);
            push(EV_REL, 1, -3);
            push(EV_REL, 0, 2);
            push(EV_REL, 8, -1);
            push(EV_KEY, 1, 0); // Esc up
            let mut e = Events::default();
            parse(&bytes, long, &mut e);
            assert_eq!(e.keys, vec![(KeyCode::KeyW, true), (KeyCode::Escape, false)]);
            assert_eq!(e.buttons, vec![(0, true)]);
            assert_eq!(e.motion, (7, -3));
            assert_eq!(e.wheel, -1);
        }
    }

    /// A frame shown on a 16-bit "framebuffer" that is a plain file, with a
    /// stride wider than the picture, scaled up 2 x.
    #[test]
    fn frames_land_in_the_framebuffer() {
        let dir = std::env::temp_dir().join(format!("fkjensen-fb-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("fb");
        let g = Geometry { width: 4, height: 2, stride: 12, offset: 0, format: Format { bits: 16, red: (11, 5), green: (5, 6), blue: (0, 5) } };
        std::fs::write(&path, vec![0xaau8; g.stride * g.height]).unwrap();
        let f = OpenOptions::new().read(true).write(true).open(&path).unwrap();
        let mut fb = Framebuffer::with_geometry(f, g);
        fb.show(&[0xff0000, 0x0000ff], 2, 1, None).unwrap();
        let out = std::fs::read(&path).unwrap();
        let px = |row: usize, x: usize| u16::from_le_bytes([out[row * 12 + x * 2], out[row * 12 + x * 2 + 1]]);
        for row in 0..2 {
            assert_eq!([px(row, 0), px(row, 1), px(row, 2), px(row, 3)], [0xf800, 0xf800, 0x001f, 0x001f]);
            // The padding past the visible row is untouched.
            assert_eq!(&out[row * 12 + 8..row * 12 + 12], &[0xaa; 4]);
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
