#!/usr/bin/env python3
"""
Pull an RTSP H.264 stream, decode + scale to 1280x720 RGBA, and push to
the FZ3A board for DP display.

Usage:
    python3 stream_rtsp.py                              # default source + 192.168.6.192
    python3 stream_rtsp.py <rtsp_url>
    python3 stream_rtsp.py <rtsp_url> <fz3a_host>
    python3 stream_rtsp.py <rtsp_url> <fz3a_host> <fps>
"""
import sys, socket, struct, subprocess, time

RTSP = sys.argv[1] if len(sys.argv) > 1 else 'rtsp://192.168.6.162:8554/live'
HOST = sys.argv[2] if len(sys.argv) > 2 else '192.168.6.192'
FPS  = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0
PORT = 5000
W, H = 1280, 720
FRAME_SIZE = W * H * 4
HEADER = b'IMG\x00' + struct.pack('<III', W, H, 0)

# Letterbox-scale video to 1280x720 preserving aspect ratio, cap to FPS.
vf = (f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
      f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=black,"
      f"fps={int(FPS)}")

cmd = [
    'ffmpeg', '-hide_banner', '-loglevel', 'warning',
    '-rtsp_transport', 'tcp',
    '-fflags', 'nobuffer', '-flags', 'low_delay',
    '-i', RTSP,
    '-an',  # drop audio
    '-vf', vf,
    '-pix_fmt', 'rgba',
    '-f', 'rawvideo',
    'pipe:1',
]
print(f"[rtsp] {RTSP}")
print(f"[target] {HOST}:{PORT}  {W}x{H}@{int(FPS)} RGBA")
print(f"[ffmpeg] {' '.join(cmd)}")
p = subprocess.Popen(cmd, stdout=subprocess.PIPE)

print(f"[tcp] connecting ...")
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8 * 1024 * 1024)
s.connect((HOST, PORT))
print(f"[tcp] connected. Relaying... Ctrl-C to stop.")

n = 0
t0 = time.time()
last = t0
try:
    while True:
        buf = b''
        while len(buf) < FRAME_SIZE:
            chunk = p.stdout.read(FRAME_SIZE - len(buf))
            if not chunk:
                raise RuntimeError("ffmpeg stdout closed (stream ended?)")
            buf += chunk
        s.sendall(HEADER + buf)
        n += 1
        now = time.time()
        if now - last >= 1.0:
            dt = now - t0
            print(f"[stats] frames={n}  avg_fps={n/dt:.1f}  "
                  f"bw={n*(FRAME_SIZE+16)/dt/1e6:.0f} MB/s")
            last = now
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
