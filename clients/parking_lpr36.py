#!/usr/bin/env python3
"""Parking LPR pipeline with letters (36-class: 0-9, A-Z).
Board must be in mode 5 (LPR36). Switch via UART 'M' if needed."""
import socket, struct, numpy as np, os, json, time
from PIL import Image

BOARD = "192.168.6.191"
PORT = 5001
CHARS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

def send_char(sock, img_28x28):
    hdr = b'MNI\x00' + struct.pack('<III', 28, 28, 0)
    sock.sendall(hdr + img_28x28.astype(np.uint8).tobytes())
    resp = b''
    while len(resp) < 48:
        chunk = sock.recv(48 - len(resp))
        if not chunk: raise ConnectionError
        resp += chunk
    return resp[4]  # 0..35

def otsu_threshold(gray):
    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
    total = gray.size; sum_total = np.sum(np.arange(256) * hist)
    sum_b = 0; w_b = 0; max_var = 0; th = 0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0: continue
        w_f = total - w_b
        if w_f == 0: break
        sum_b += t * hist[t]
        mean_b = sum_b / w_b; mean_f = (sum_total - sum_b) / w_f
        var = w_b * w_f * (mean_b - mean_f)**2
        if var > max_var: max_var = var; th = t
    return th

def connected_components(binary):
    h, w = binary.shape
    label = np.zeros((h, w), dtype=np.int32); nl = 1
    for y in range(h):
        for x in range(w):
            if binary[y, x] and label[y, x] == 0:
                stack = [(y, x)]
                while stack:
                    cy, cx = stack.pop()
                    if cy < 0 or cy >= h or cx < 0 or cx >= w: continue
                    if not binary[cy, cx] or label[cy, cx] != 0: continue
                    label[cy, cx] = nl
                    stack.extend([(cy+1,cx),(cy-1,cx),(cy,cx+1),(cy,cx-1)])
                nl += 1
    boxes = []
    for k in range(1, nl):
        ys, xs = np.where(label == k)
        if len(ys) < 40: continue  # stricter filter for letters
        boxes.append((xs.min(), ys.min(), xs.max(), ys.max(), len(ys)))
    return boxes

def extract_28x28(patch):
    h, w = patch.shape
    target = 20
    if h >= w:
        new_h = target; new_w = max(1, int(w * target / h))
    else:
        new_w = target; new_h = max(1, int(h * target / w))
    img = Image.fromarray(patch).resize((new_w, new_h), Image.BILINEAR)
    arr = np.array(img); out = np.zeros((28, 28), dtype=np.uint8)
    y0 = (28 - new_h) // 2; x0 = (28 - new_w) // 2
    out[y0:y0+new_h, x0:x0+new_w] = arr
    return out

def recognize(image_path, sock):
    img = np.array(Image.open(image_path).convert('L'))
    img_inv = 255 - img if img.mean() > 127 else img
    t = otsu_threshold(img_inv)
    binary = (img_inv > t).astype(np.uint8)
    boxes = connected_components(binary)
    boxes.sort(key=lambda b: b[0])
    result = ''
    for (x0, y0, x1, y1, _) in boxes:
        patch = img_inv[y0:y1+1, x0:x1+1].astype(np.uint8)
        patch_bin = np.where(patch > t, patch, 0).astype(np.uint8)
        norm = extract_28x28(patch_bin)
        pred = send_char(sock, norm)
        if pred < 36: result += CHARS[pred]
        else: result += '?'
    return result

labels = json.load(open('/tmp/lpr36_plates/labels.json'))
files = sorted(os.listdir('/tmp/lpr36_plates'))
files = [f for f in files if f.endswith('.png')]

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(10); s.connect((BOARD, PORT))

correct_plate = 0; correct_char = 0; total_char = 0
per_char_stat = {c: [0, 0] for c in CHARS}  # [correct, total]
failures = []
t0 = time.time()

for i, fn in enumerate(files):
    true_s = fn.split('_')[-1].split('.')[0]
    pred_s = recognize(f'/tmp/lpr36_plates/{fn}', s)
    if pred_s == true_s:
        correct_plate += 1; status = "✓"
    else:
        status = "✗"
        failures.append((fn, true_s, pred_s))
    if len(pred_s) == len(true_s):
        for p, t in zip(pred_s, true_s):
            total_char += 1
            if p == t: correct_char += 1
            if t in per_char_stat:
                per_char_stat[t][1] += 1
                if p == t: per_char_stat[t][0] += 1
    print(f"  {i+1:2d}. {fn[:26]} → '{pred_s}' (true '{true_s}') {status}")

dt = time.time() - t0
s.close()

print(f"\n{'='*60}")
print(f"Plate-level accuracy: {correct_plate}/{len(files)} = {correct_plate/len(files)*100:.1f}%")
if total_char > 0:
    print(f"Char-level accuracy:  {correct_char}/{total_char} = {correct_char/total_char*100:.1f}%")
print(f"Avg time: {dt/len(files)*1000:.1f}ms per plate")

# Split stats: digits vs letters
digit_c = sum(per_char_stat[c][0] for c in '0123456789')
digit_t = sum(per_char_stat[c][1] for c in '0123456789')
letter_c = sum(per_char_stat[c][0] for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ')
letter_t = sum(per_char_stat[c][1] for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ')
print(f"\nDigit chars:  {digit_c}/{digit_t} = {100*digit_c/max(digit_t,1):.1f}%")
print(f"Letter chars: {letter_c}/{letter_t} = {100*letter_c/max(letter_t,1):.1f}%")

if failures:
    print(f"\nFailures ({len(failures)}):")
    for fn, t, p in failures[:15]:
        print(f"  {fn[:30]}: pred='{p}' true='{t}'")
