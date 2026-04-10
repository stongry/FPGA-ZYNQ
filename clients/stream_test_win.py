#!/usr/bin/env python3
"""Windows version of stream_test.py - synthetic test pattern to FZ3A."""
import os, sys, socket, struct, subprocess, time

FFMPEG = r'C:\Users\huye\fz3a\ffmpeg\bin\ffmpeg.exe'

HOST    = sys.argv[1] if len(sys.argv) > 1 else '192.168.6.192'
FPS     = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
PATTERN = sys.argv[3] if len(sys.argv) > 3 else 'testsrc2'
PORT = 5000
W, H = 1280, 720
FRAME_SIZE = W * H * 4
HEADER = b'IMG\x00' + struct.pack('<III', W, H, 0)

src = f"{PATTERN}=size={W}x{H}:rate={int(FPS)}"
cmd = [
    FFMPEG, '-hide_banner', '-loglevel', 'warning',
    '-f', 'lavfi', '-i', src,
    '-pix_fmt', 'rgba',
    '-f', 'rawvideo',
    'pipe:1',
]
print(f"[cmd ] {' '.join(cmd)}")
p = subprocess.Popen(cmd, stdout=subprocess.PIPE)

print(f"[tcp ] connecting to {HOST}:{PORT} ...")
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8 * 1024 * 1024)
s.connect((HOST, PORT))
print(f"[tcp ] connected. {W}x{H}@{FPS}. Ctrl-C to stop.")

n = 0
t0 = time.time()
last = t0
out = bytearray(len(HEADER) + FRAME_SIZE)
out[:len(HEADER)] = HEADER
view_payload = memoryview(out)[len(HEADER):]
view_all     = memoryview(out)
try:
    while True:
        pos = 0
        while pos < FRAME_SIZE:
            nread = p.stdout.readinto(view_payload[pos:])
            if not nread:
                raise RuntimeError("ffmpeg closed")
            pos += nread
        s.sendall(view_all)
        n += 1
        now = time.time()
        if now - last >= 1.0:
            dt = now - t0
            print(f"[stats] frames={n}  avg_fps={n/dt:.1f}  "
                  f"bw={n*(FRAME_SIZE+16)/dt/1e6:.0f} MB/s")
            last = now
except KeyboardInterrupt:
    print("\n[stop]")
except Exception as e:
    print(f"[err ] {e}")
finally:
    try: s.close()
    except Exception: pass
    p.terminate()
    dt = time.time() - t0
    if n:
        print(f"[done] {n} frames in {dt:.1f}s = {n/dt:.1f} fps avg")
