#!/usr/bin/env python3
"""
Windows webcam -> FZ3A DP display, over TCP.

Captures from the default (or chosen) DirectShow camera with OpenCV,
letterboxes to 1280x720, converts to RGBA, and sends each frame to
the FZ3A image server at 192.168.6.192:5000.

Usage:
    python cam_to_fz3a.py                           # camera 0, 192.168.6.192
    python cam_to_fz3a.py <cam_idx>
    python cam_to_fz3a.py <cam_idx> <host>
    python cam_to_fz3a.py <cam_idx> <host> <target_fps>
    python cam_to_fz3a.py list                      # enumerate cameras
"""
import sys, socket, struct, time
import cv2
import numpy as np

PORT = 5000
W, H = 1280, 720
FRAME_SIZE = W * H * 4
HEADER = b'IMG\x00' + struct.pack('<III', W, H, 0)


def list_cameras(max_idx=8):
    print("scanning DirectShow cameras (0..{})".format(max_idx - 1))
    for i in range(max_idx):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                h, w = frame.shape[:2]
                print(f"  cam {i}: {w}x{h}  OK")
            else:
                print(f"  cam {i}: opened, no frame")
            cap.release()
        else:
            print(f"  cam {i}: unavailable")


def letterbox_to_720p(bgr):
    """Scale bgr frame to fit inside 1280x720, pad with black."""
    ih, iw = bgr.shape[:2]
    r = min(W / iw, H / ih)
    nw, nh = int(iw * r), int(ih * r)
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    x, y = (W - nw) // 2, (H - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'list':
        list_cameras()
        return

    cam_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    host    = sys.argv[2] if len(sys.argv) > 2 else '192.168.6.192'
    tgt_fps = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0
    # Optional native capture resolution (4th/5th args).  Default 640x480
    # because USB2 webcams in YUY2 can only do 30 fps at this size.
    cap_w   = int(sys.argv[4]) if len(sys.argv) > 4 else 640
    cap_h   = int(sys.argv[5]) if len(sys.argv) > 5 else 480
    period  = 1.0 / tgt_fps

    # Try Media Foundation first (fast), fall back to DirectShow if not.
    backends = [('MSMF', cv2.CAP_MSMF), ('DSHOW', cv2.CAP_DSHOW)]
    cap = None
    for name, backend in backends:
        print(f"[cam ] trying {name} backend for camera {cam_idx}...")
        cap = cv2.VideoCapture(cam_idx, backend)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                print(f"[cam ] using {name}")
                break
        cap.release()
        cap = None
    if cap is None or not cap.isOpened():
        print(f"ERROR: cannot open camera {cam_idx}")
        sys.exit(1)

    # Avoid touching camera properties if possible (MSMF reinits the stream
    # on each set() call, causing "Failed to select stream 0" warnings).
    # Only change resolution if the caller explicitly asked for non-default.
    if cap_w and cap_h:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_h)
    cw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ch = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cfps = cap.get(cv2.CAP_PROP_FPS)
    fcc  = int(cap.get(cv2.CAP_PROP_FOURCC))
    fcc_s = ''.join(chr((fcc >> (8*i)) & 0xff) for i in range(4))
    print(f"[cam ] got {cw}x{ch}@{cfps:.0f}fps fourcc={fcc_s} "
          f"(will letterbox to {W}x{H})")

    print(f"[tcp ] connecting to {host}:{PORT} ...")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8 * 1024 * 1024)
    s.connect((host, PORT))
    print(f"[tcp ] connected. Streaming... Ctrl-C to stop.")

    n = 0
    t0 = time.time()
    last = t0
    next_deadline = t0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                print("[cam ] read failed, retrying...")
                time.sleep(0.01)
                continue
            # letterbox if needed
            if frame_bgr.shape[0] != H or frame_bgr.shape[1] != W:
                frame_bgr = letterbox_to_720p(frame_bgr)
            # BGR -> RGBA  (FZ3A DP uses RGBA8888, R in low byte)
            rgba = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA)
            s.sendall(HEADER + rgba.tobytes())
            n += 1

            now = time.time()
            if now - last >= 1.0:
                dt = now - t0
                print(f"[stats] frames={n}  avg_fps={n/dt:.1f}  "
                      f"bw={n*(FRAME_SIZE+16)/dt/1e6:.0f} MB/s")
                last = now

            # cap frame rate (camera may produce faster than target)
            next_deadline += period
            sleep_for = next_deadline - time.time()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_deadline = time.time()
    except KeyboardInterrupt:
        print("\n[stop] interrupted")
    except Exception as e:
        print(f"[err ] {e}")
    finally:
        cap.release()
        s.close()
        dt = time.time() - t0
        if n:
            print(f"[done] {n} frames in {dt:.1f}s = {n/dt:.1f} fps avg")


if __name__ == '__main__':
    main()
