#!/usr/bin/env python3
"""
Send a 28x28 grayscale digit image to the FZ3A MNIST inference server
on port 5001, print the prediction + probabilities.

Usage:
    python3 send_digit.py <image_path> [host]
    python3 send_digit.py digit_pngs/digit_7_3.png
    python3 send_digit.py all                     # send every PNG in digit_pngs/
    python3 send_digit.py all 192.168.6.192

Protocol:
    client -> board :  "MNI\\0" + w(4) + h(4) + fmt(4) + 784 bytes u8 grayscale
    board  -> client:  "CLS\\0" + pred(1) + pad(3) + 10 * float32 probs
"""
import os, sys, socket, struct, glob, time

HOST_DEFAULT = '192.168.6.192'
PORT = 5001
W, H = 28, 28

def to_784(path):
    """Load any image, resize to 28x28 grayscale u8.  Auto-invert if
    background looks light (MNIST convention: black bg, white digit)."""
    try:
        from PIL import Image, ImageOps
    except ImportError:
        sys.exit("ERROR: need `pip install pillow` (or install python-pillow via pacman)")
    img = Image.open(path).convert('L')
    if img.size != (W, H):
        img = img.resize((W, H), Image.LANCZOS)
    data = bytes(img.tobytes())
    # Heuristic: MNIST is black background, white digit (mean < 128).
    # If mean is high, assume inverted and flip.
    mean = sum(data) / len(data)
    if mean > 127:
        data = bytes(255 - b for b in data)
    return data

def send_one(host, img_bytes, label_hint=None):
    assert len(img_bytes) == W * H
    header = b'MNI\x00' + struct.pack('<III', W, H, 0)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    s.settimeout(5.0)
    s.connect((host, PORT))
    s.sendall(header + img_bytes)
    # Wait for 4+4+40 = 48 byte reply
    buf = b''
    t0 = time.time()
    while len(buf) < 48:
        chunk = s.recv(48 - len(buf))
        if not chunk:
            break
        buf += chunk
    rtt = (time.time() - t0) * 1000
    s.close()
    if len(buf) != 48 or buf[:4] != b'CLS\x00':
        print(f'[err] bad reply: len={len(buf)} magic={buf[:4]!r}')
        return None
    pred = buf[4]
    probs = struct.unpack('<10f', buf[8:48])
    conf = probs[pred]
    # nice progress bar of top-3
    idxs = sorted(range(10), key=lambda k: -probs[k])[:3]
    top3 = '  '.join(f'{d}={probs[d]*100:5.1f}%' for d in idxs)
    hint = f' (truth={label_hint})' if label_hint is not None else ''
    print(f'[cnn] pred={pred} conf={conf*100:5.1f}%  top3: {top3}  {rtt:.1f}ms{hint}')
    return pred

def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    target = sys.argv[1]
    host = sys.argv[2] if len(sys.argv) > 2 else HOST_DEFAULT

    if target == 'all':
        pngs = sorted(glob.glob('/tmp/fz3a_dp/digit_pngs/digit_*.png'))
        if not pngs:
            sys.exit('no digit_pngs/*.png found - run mnist_train_export.py first')
        print(f'[send] {len(pngs)} images to {host}:{PORT}')
        correct = 0
        for p in pngs:
            name = os.path.basename(p)
            # digit_<truth>_<idx>.png
            try:
                truth = int(name.split('_')[1])
            except Exception:
                truth = None
            data = to_784(p)
            pred = send_one(host, data, truth)
            if truth is not None and pred == truth:
                correct += 1
        print(f'[summary] {correct}/{len(pngs)} correct = {correct*100/len(pngs):.1f}%')
    else:
        data = to_784(target)
        send_one(host, data)

if __name__ == '__main__':
    main()
