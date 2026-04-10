#!/usr/bin/env python3
"""
Stream a synthetic 30 fps test pattern to FZ3A over TCP for real DP display.
Uses ffmpeg lavfi testsrc2 (colorful moving test chart with timecode) so we
can visually confirm smooth 30 fps motion on the monitor.

Usage:
    python3 stream_test.py                  # default 192.168.6.192:5000 @ 30 fps
    python3 stream_test.py <host> [fps] [pattern]
      pattern: testsrc2 | smptebars | mandelbrot | life
"""
import sys, socket, struct, subprocess, time

HOST    = sys.argv[1] if len(sys.argv) > 1 else '192.168.6.192'
FPS     = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
PATTERN = sys.argv[3] if len(sys.argv) > 3 else 'testsrc2'
PORT = 5000
W, H = 1280, 720
FRAME_SIZE = W * H * 4
HEADER = b'IMG\x00' + struct.pack('<III', W, H, 0)

# lavfi source string: testsrc2 draws a colorful test pattern with a moving
# sweep and a frame counter -- great for spotting dropped frames or tearing.
src = f"{PATTERN}=size={W}x{H}:rate={int(FPS)}"
if PATTERN == 'mandelbrot':
    src = f"mandelbrot=size={W}x{H}:rate={int(FPS)}"
elif PATTERN == 'life':
    src = f"life=size={W}x{H}:rate={int(FPS)}:mold=10:r={int(FPS)}:ratio=0.5:death_color=#c83232:life_color=#00ff00"

cmd = [
    'ffmpeg', '-hide_banner', '-loglevel', 'warning',
    '-f', 'lavfi', '-i', src,
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
print(f"[tcp] connected. Streaming {PATTERN} {W}x{H}@{FPS} fps.  Ctrl-C to stop.")

n = 0
t0 = time.time()
last_report = t0
try:
    while True:
        buf = b''
        while len(buf) < FRAME_SIZE:
            chunk = p.stdout.read(FRAME_SIZE - len(buf))
            if not chunk:
                raise RuntimeError("ffmpeg closed")
            buf += chunk
        s.sendall(HEADER + buf)
        n += 1
        now = time.time()
        if now - last_report >= 1.0:
            dt = now - t0
            inst_fps = n / dt
            inst_bw  = n * (FRAME_SIZE + 16) / dt / 1e6
            print(f"[stats] frames={n}  avg_fps={inst_fps:.1f}  bw={inst_bw:.0f} MB/s")
            last_report = now
except KeyboardInterrupt:
    print("\n[stop] user interrupted")
except Exception as e:
    print(f"[err] {e}")
finally:
    s.close()
    p.terminate()
    dt = time.time() - t0
    if n:
        print(f"[done] {n} frames in {dt:.1f}s = {n/dt:.1f} fps avg")
