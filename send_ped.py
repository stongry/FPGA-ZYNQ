#!/usr/bin/env python3
"""
Send a 320x240 grayscale image to the FZ3A pedestrian detection server
on port 5002, print detection results.

Usage:
    python3 send_ped.py <image_path> [host]
    python3 send_ped.py test             # send synthetic test pattern

Protocol:
    client -> board:  "PED\0" + w(4)=320 + h(4)=240 + fmt(4) + 76800 bytes u8
    board  -> client: "DET\0" + n(4) + n * {pos(4) + score(4)}
"""
import sys, socket, struct, time, os
import numpy as np

HOST = '192.168.6.192'
PORT = 5002
W, H = 320, 240

def load_image(path):
    if path == 'test':
        # Synthetic test: vertical gradient with a "pedestrian-like" rectangle
        img = np.zeros((H, W), dtype=np.uint8)
        # Background gradient
        for y in range(H):
            img[y, :] = int(y * 255 / H)
        # Dark rectangle in center (simulating a figure)
        img[56:184, 128:192] = 200
        return img
    try:
        from PIL import Image
        im = Image.open(path).convert('L').resize((W, H))
        return np.array(im, dtype=np.uint8)
    except ImportError:
        sys.exit("Need `pip install pillow`")

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <image_or_test> [host]")
        sys.exit(1)

    target = sys.argv[1]
    host = sys.argv[2] if len(sys.argv) > 2 else HOST

    img = load_image(target)
    assert img.shape == (H, W), f"Expected {H}x{W}, got {img.shape}"
    pixels = img.tobytes()

    # Build header: "PED\0" + w + h + fmt
    hdr = b'PED\x00' + struct.pack('<III', W, H, 0)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.settimeout(10)
    sock.connect((host, PORT))

    t0 = time.time()
    sock.sendall(hdr + pixels)

    # Read reply: "DET\0" + n(4) + n * 8
    reply_hdr = b''
    while len(reply_hdr) < 8:
        reply_hdr += sock.recv(8 - len(reply_hdr))

    magic = reply_hdr[:4]
    n_dets = struct.unpack('<I', reply_hdr[4:8])[0]

    dets = []
    if n_dets > 0:
        det_data = b''
        need = n_dets * 8
        while len(det_data) < need:
            det_data += sock.recv(need - len(det_data))
        for i in range(n_dets):
            pos_word = struct.unpack('<I', det_data[i*8:i*8+4])[0]
            score = struct.unpack('<i', det_data[i*8+4:i*8+8])[0]
            x = pos_word & 0xFF
            y = (pos_word >> 8) & 0xFF
            dets.append((x, y, score))

    dt = time.time() - t0
    sock.close()

    print(f"[ped] {n_dets} detections in {dt*1000:.1f}ms")
    for i, (x, y, score) in enumerate(dets):
        print(f"  [{i}] pos=({x},{y}) window=64x128 score={score}")

if __name__ == '__main__':
    main()
