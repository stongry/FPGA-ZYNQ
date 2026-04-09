#!/usr/bin/env python3
"""Stream a video file to FZ3A over persistent TCP for real-time DP display.
Usage: stream_video.py <video_file> [host] [target_fps]"""
import subprocess, socket, struct, sys, time

if len(sys.argv) < 2:
    print("usage: stream_video.py <video.mp4> [host=192.168.6.192] [fps=30]")
    sys.exit(1)

path = sys.argv[1]
host = sys.argv[2] if len(sys.argv) > 2 else '192.168.6.192'
fps  = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0
PORT = 5000
W, H = 1280, 720
FRAME_SIZE = W * H * 4
HEADER = b'IMG\x00' + struct.pack('<III', W, H, 0)

print(f"target {W}x{H} @ {fps} fps, {FRAME_SIZE} bytes/frame")

# ffmpeg: decode video, resize to 1280x720, output raw RGBA frames to stdout
cmd = [
    'ffmpeg', '-hide_banner', '-loglevel', 'error',
    '-re',                     # real-time playback rate
    '-stream_loop', '-1',      # loop forever
    '-i', path,
    '-vf', f'scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=black,fps={fps}',
    '-pix_fmt', 'rgba',
    '-f', 'rawvideo',
    'pipe:1'
]
p = subprocess.Popen(cmd, stdout=subprocess.PIPE)

print(f"Connecting to {host}:{PORT}...")
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((host, PORT))
s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8*1024*1024)
print("Connected. Streaming...")

n = 0
t_start = time.time()
try:
    while True:
        raw = b''
        while len(raw) < FRAME_SIZE:
            chunk = p.stdout.read(FRAME_SIZE - len(raw))
            if not chunk: break
            raw += chunk
        if len(raw) < FRAME_SIZE: break
        s.sendall(HEADER + raw)
        n += 1
        if n % 30 == 0:
            elapsed = time.time() - t_start
            actual_fps = n / elapsed
            mbps = (n * (FRAME_SIZE + len(HEADER))) / elapsed / 1e6
            print(f"[frame {n}] actual={actual_fps:.1f} fps, {mbps:.0f} MB/s")
except KeyboardInterrupt:
    print("\nStopped by user")
finally:
    s.close()
    p.terminate()
    print(f"Total: {n} frames")
