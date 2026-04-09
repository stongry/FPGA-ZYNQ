#!/usr/bin/env python3
"""
Stream local Linux desktop to FZ3A over TCP for real-time DP display.

Usage:
    python3 stream_desktop.py                  # default 192.168.6.192:5000 @ 30fps
    python3 stream_desktop.py <host> [fps]     # custom target

Must be run from a terminal that has access to your desktop session
(NOT from an SSH/TTY without DISPLAY/WAYLAND_DISPLAY set).
"""
import os, sys, subprocess, socket, struct, time, shutil

HOST = sys.argv[1] if len(sys.argv) > 1 else '192.168.6.192'
FPS  = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
PORT = 5000
W, H = 1280, 720
FRAME_SIZE = W * H * 4
HEADER = b'IMG\x00' + struct.pack('<III', W, H, 0)

# --- detect capture method ---
display = os.environ.get('DISPLAY', '')
wl      = os.environ.get('WAYLAND_DISPLAY', '')
xdg     = os.environ.get('XDG_SESSION_TYPE', '')

print(f"[env] DISPLAY={display!r}  WAYLAND_DISPLAY={wl!r}  XDG_SESSION_TYPE={xdg!r}")

ffmpeg_input = None
if display:
    # X11 (or XWayland) — use x11grab, simplest and fastest
    print(f"[mode] X11 capture via x11grab on {display}")
    ffmpeg_input = ['-f', 'x11grab', '-framerate', str(int(FPS)), '-i', display]
elif wl and shutil.which('wf-recorder'):
    print("[mode] Wayland capture via wf-recorder (pipe to ffmpeg)")
    # not supported inline yet
    print("       wf-recorder mode: unsupported in this script, use X11")
    sys.exit(1)
elif wl:
    # Try kmsgrab fallback (needs root) or pipewire
    print("[mode] Wayland detected but no wf-recorder; trying kmsgrab (needs sudo)")
    ffmpeg_input = ['-f', 'kmsgrab', '-framerate', str(int(FPS)), '-i', '-']
else:
    print("ERROR: neither DISPLAY nor WAYLAND_DISPLAY is set.")
    print("Run this from a terminal emulator on your desktop, not from a raw TTY.")
    sys.exit(1)

# --- build ffmpeg command ---
# Scale to 1280x720 with letterbox, output raw RGBA
vf = f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=black"
cmd = [
    'ffmpeg', '-hide_banner', '-loglevel', 'warning',
] + ffmpeg_input + [
    '-vf', vf,
    '-pix_fmt', 'rgba',
    '-f', 'rawvideo',
    'pipe:1',
]
print(f"[ffmpeg] {' '.join(cmd)}")

p = subprocess.Popen(cmd, stdout=subprocess.PIPE)

print(f"[tcp] connecting to {HOST}:{PORT} ...")
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8 * 1024 * 1024)
s.connect((HOST, PORT))
print(f"[tcp] connected. streaming {W}x{H}@{FPS}fps... Ctrl-C to stop")

n = 0
t0 = time.time()
last_report = t0
try:
    while True:
        buf = b''
        while len(buf) < FRAME_SIZE:
            chunk = p.stdout.read(FRAME_SIZE - len(buf))
            if not chunk:
                raise RuntimeError("ffmpeg stdout closed")
            buf += chunk
        s.sendall(HEADER + buf)
        n += 1
        now = time.time()
        if now - last_report >= 2.0:
            dt = now - t0
            print(f"[stats] frames={n}  fps={n/dt:.1f}  bw={n*(FRAME_SIZE+16)/dt/1e6:.0f} MB/s")
            last_report = now
except KeyboardInterrupt:
    print("\n[stop] user interrupted")
except Exception as e:
    print(f"[err] {e}")
finally:
    s.close()
    p.terminate()
    dt = time.time() - t0
    print(f"[done] {n} frames in {dt:.1f}s = {n/dt:.1f} fps avg")
