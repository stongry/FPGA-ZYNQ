#!/usr/bin/env python3
"""
Windows DirectShow webcam -> FZ3A DP display, via ffmpeg.

Uses ffmpeg's dshow input (much faster/more reliable than OpenCV on
Windows), scales + letterboxes to 1280x720 RGBA, and pipes each frame
over TCP to the FZ3A image server at 192.168.6.192:5000.

Usage:
    python cam_ff_to_fz3a.py                                 # auto default cam
    python cam_ff_to_fz3a.py list                            # list devices
    python cam_ff_to_fz3a.py "<dshow device name>"
    python cam_ff_to_fz3a.py "<device name>" <host>
    python cam_ff_to_fz3a.py "<device name>" <host> <fps>
    python cam_ff_to_fz3a.py "<device name>" <host> <fps> <cap_w> <cap_h>

Tip: run  cam_ff_to_fz3a.py list  first to copy the exact camera name
     (dshow friendly names can include Chinese characters, quote them).
"""
import os, sys, socket, struct, subprocess, time, shutil

# Preferred locations for a local ffmpeg.exe
FFMPEG_CANDIDATES = [
    r'C:\Users\huye\fz3a\ffmpeg\bin\ffmpeg.exe',
    r'C:\ffmpeg\bin\ffmpeg.exe',
    r'ffmpeg',  # from PATH
]

def find_ffmpeg():
    for c in FFMPEG_CANDIDATES:
        if os.path.isabs(c) and os.path.isfile(c):
            return c
        if not os.path.isabs(c):
            found = shutil.which(c)
            if found:
                return found
    print("ERROR: ffmpeg.exe not found. Set FFMPEG_CANDIDATES or add to PATH.")
    sys.exit(1)


def list_devices(ffmpeg):
    print("ffmpeg -list_devices true -f dshow -i dummy")
    proc = subprocess.run(
        [ffmpeg, '-hide_banner', '-list_devices', 'true', '-f', 'dshow', '-i', 'dummy'],
        capture_output=True, text=True, encoding='utf-8', errors='replace'
    )
    # ffmpeg writes device list to stderr
    print(proc.stderr)


def main():
    ffmpeg = find_ffmpeg()
    print(f"[ff  ] {ffmpeg}")

    if len(sys.argv) > 1 and sys.argv[1] == 'list':
        list_devices(ffmpeg)
        return

    dev   = sys.argv[1] if len(sys.argv) > 1 else None
    host  = sys.argv[2] if len(sys.argv) > 2 else '192.168.6.192'
    fps   = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0
    cw    = int(sys.argv[4]) if len(sys.argv) > 4 else 640
    ch    = int(sys.argv[5]) if len(sys.argv) > 5 else 480
    PORT  = 5000
    W, H  = 1280, 720
    FRAME_SIZE = W * H * 4
    HEADER = b'IMG\x00' + struct.pack('<III', W, H, 0)

    if dev is None:
        print("ERROR: missing device name. Run with 'list' to see devices.")
        sys.exit(1)

    vf = (f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
          f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=black")

    cmd = [
        ffmpeg, '-hide_banner', '-loglevel', 'warning',
        '-f', 'dshow',
        '-video_size', f'{cw}x{ch}',
        '-framerate', str(int(fps)),
        '-i', f'video={dev}',
        '-vf', vf,
        '-pix_fmt', 'rgba',
        '-f', 'rawvideo',
        'pipe:1',
    ]
    print(f"[ff  ] {' '.join(cmd)}")
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
