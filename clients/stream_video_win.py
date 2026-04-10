#!/usr/bin/env python3
"""
Windows -> FZ3A DP: stream a local video file via ffmpeg + TCP.

Uses the ffmpeg we installed at C:\\Users\\huye\\fz3a\\ffmpeg\\bin\\ffmpeg.exe
to decode any common format (mp4/mkv/avi/mov/webm/...) down to raw RGBA
1280x720, then pipes each frame to the FZ3A image server over TCP.

Usage:
    python stream_video_win.py <path\\to\\video>
    python stream_video_win.py <path\\to\\video> <host>
    python stream_video_win.py <path\\to\\video> <host> <fps>
    python stream_video_win.py <path\\to\\video> <host> <fps> loop

    loop = repeat forever (ffmpeg -stream_loop -1)
"""
import os, sys, socket, struct, subprocess, time, shutil

FFMPEG_CANDIDATES = [
    r'C:\Users\huye\fz3a\ffmpeg\bin\ffmpeg.exe',
    r'ffmpeg',
]

def find_ffmpeg():
    for c in FFMPEG_CANDIDATES:
        if os.path.isabs(c) and os.path.isfile(c):
            return c
        if not os.path.isabs(c):
            found = shutil.which(c)
            if found:
                return found
    print("ERROR: ffmpeg.exe not found")
    sys.exit(1)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    video = sys.argv[1]
    host  = sys.argv[2] if len(sys.argv) > 2 else '192.168.6.192'
    fps   = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0
    loop  = len(sys.argv) > 4 and sys.argv[4].lower() == 'loop'

    PORT  = 5000
    W, H  = 1280, 720
    FRAME_SIZE = W * H * 4
    HEADER = b'IMG\x00' + struct.pack('<III', W, H, 0)

    if not os.path.isfile(video):
        print(f"ERROR: file not found: {video}")
        sys.exit(1)

    ffmpeg = find_ffmpeg()
    print(f"[ff  ] {ffmpeg}")
    print(f"[file] {video}")

    # Letterbox to 1280x720, force target fps, strip audio.
    vf = (f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
          f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=black,"
          f"fps={int(fps)}")

    cmd = [ffmpeg, '-hide_banner', '-loglevel', 'warning']
    if loop:
        cmd += ['-stream_loop', '-1']
    cmd += [
        '-i', video,
        '-an',                   # no audio
        '-vf', vf,
        '-pix_fmt', 'rgba',
        '-f', 'rawvideo',
        'pipe:1',
    ]
    print(f"[cmd ] {' '.join(cmd)}")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)

    print(f"[tcp ] connecting to {host}:{PORT} ...")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8 * 1024 * 1024)
    s.connect((host, PORT))
    print(f"[tcp ] connected.  Ctrl-C to stop.")

    n = 0
    t0 = time.time()
    last = t0
    # Preallocate a single buffer: [HEADER | FRAME_SIZE payload]
    out = bytearray(len(HEADER) + FRAME_SIZE)
    out[:len(HEADER)] = HEADER
    view_payload = memoryview(out)[len(HEADER):]  # direct pipe-read target
    view_all     = memoryview(out)                # direct socket-send source
    try:
        while True:
            # Fill payload in place
            pos = 0
            while pos < FRAME_SIZE:
                nread = p.stdout.readinto(view_payload[pos:])
                if not nread:
                    raise RuntimeError("ffmpeg stdout closed (video ended)")
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
        print("\n[stop] interrupted")
    except Exception as e:
        print(f"[err ] {e}")
    finally:
        try: s.close()
        except Exception: pass
        p.terminate()
        dt = time.time() - t0
        if n:
            print(f"[done] {n} frames in {dt:.1f}s = {n/dt:.1f} fps avg")


if __name__ == '__main__':
    main()
