#!/usr/bin/env python3
"""Parking-lot-number LPR pipeline using FPGA board CNN.

Pipeline:
  1. Load plate image (color/gray)
  2. Grayscale + Otsu binarize
  3. Invert if needed (MNIST expects white-on-black)
  4. Connected component analysis: find digit bounding boxes
  5. For each box: extract, resize to 20x20, center in 28x28
  6. Send each 28x28 patch to board CNN (TCP 5001)
  7. Assemble result string
"""
import socket, struct, numpy as np, os, json, time
from PIL import Image

BOARD = "192.168.6.191"
PORT = 5001

def send_digit(sock, img_28x28):
    hdr = b'MNI\x00' + struct.pack('<III', 28, 28, 0)
    sock.sendall(hdr + img_28x28.astype(np.uint8).tobytes())
    resp = b''
    while len(resp) < 48:
        chunk = sock.recv(48 - len(resp))
        if not chunk: raise ConnectionError
        resp += chunk
    return resp[4]  # pred

def otsu_threshold(gray):
    """Simple Otsu's method to find optimal threshold."""
    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
    total = gray.size
    sum_total = np.sum(np.arange(256) * hist)
    sum_b = 0; w_b = 0; max_var = 0; threshold = 0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0: continue
        w_f = total - w_b
        if w_f == 0: break
        sum_b += t * hist[t]
        mean_b = sum_b / w_b
        mean_f = (sum_total - sum_b) / w_f
        var = w_b * w_f * (mean_b - mean_f)**2
        if var > max_var: max_var = var; threshold = t
    return threshold

def connected_components(binary):
    """Simple 4-connectivity CC labeling. Returns list of bounding boxes."""
    h, w = binary.shape
    label = np.zeros((h, w), dtype=np.int32)
    next_label = 1
    # Flood fill with iterative stack
    for y in range(h):
        for x in range(w):
            if binary[y, x] and label[y, x] == 0:
                stack = [(y, x)]
                while stack:
                    cy, cx = stack.pop()
                    if cy < 0 or cy >= h or cx < 0 or cx >= w: continue
                    if not binary[cy, cx] or label[cy, cx] != 0: continue
                    label[cy, cx] = next_label
                    stack.extend([(cy+1,cx), (cy-1,cx), (cy,cx+1), (cy,cx-1)])
                next_label += 1
    # Get bounding boxes
    boxes = []
    for lbl in range(1, next_label):
        ys, xs = np.where(label == lbl)
        if len(ys) < 30: continue  # filter tiny noise
        boxes.append((xs.min(), ys.min(), xs.max(), ys.max(), len(ys)))
    return boxes

def extract_centered_28x28(patch):
    """Resize digit patch to 20x20 preserving aspect, center in 28x28."""
    h, w = patch.shape
    target = 20
    if h > w:
        new_h = target; new_w = max(1, int(w * target / h))
    else:
        new_w = target; new_h = max(1, int(h * target / w))
    img = Image.fromarray(patch).resize((new_w, new_h), Image.BILINEAR)
    arr = np.array(img)
    out = np.zeros((28, 28), dtype=np.uint8)
    y0 = (28 - new_h) // 2; x0 = (28 - new_w) // 2
    out[y0:y0+new_h, x0:x0+new_w] = arr
    return out

def recognize_plate(image_path, sock, debug=False):
    """Full pipeline: image file → string of digits."""
    img = np.array(Image.open(image_path).convert('L'))
    # Check if digits are black on white (most common) and invert
    mean_val = img.mean()
    if mean_val > 127:  # mostly light background
        img_inv = 255 - img
    else:
        img_inv = img.copy()
    # Otsu binarize
    t = otsu_threshold(img_inv)
    binary = (img_inv > t).astype(np.uint8)
    # Find digit components
    boxes = connected_components(binary)
    # Sort by x (left to right)
    boxes.sort(key=lambda b: b[0])
    if debug:
        print(f"  Found {len(boxes)} components, threshold={t}")
        for b in boxes: print(f"    box: x={b[0]}-{b[2]} y={b[1]}-{b[3]} area={b[4]}")
    # For each box: extract, normalize, classify
    result = ''
    for (x0, y0, x1, y1, area) in boxes:
        # Extract patch from inverted binary
        patch = img_inv[y0:y1+1, x0:x1+1].astype(np.uint8)
        # Apply binarize to patch for cleanliness
        patch_bin = np.where(patch > t, patch, 0).astype(np.uint8)
        # Center in 28x28
        norm28 = extract_centered_28x28(patch_bin)
        # Classify
        pred = send_digit(sock, norm28)
        result += str(pred)
    return result, len(boxes)

# === Main: test on 50 synthetic parking plates ===
plate_dir = '/tmp/parking_test/plates'
labels = json.load(open('/tmp/parking_test/labels.json'))
plate_files = sorted(os.listdir(plate_dir))

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(10)
s.connect((BOARD, PORT))

correct_full = 0  # exact string match
correct_chars = 0
total_chars = 0
per_digit_stat = {d: [0, 0] for d in range(10)}
failures = []
t0 = time.time()

for i, fn in enumerate(plate_files):
    true_str = fn.split('_')[-1].split('.')[0]  # "plate_003_26" → "26"
    path = os.path.join(plate_dir, fn)
    try:
        pred_str, n_dets = recognize_plate(path, s)
    except Exception as e:
        print(f"  ERROR on {fn}: {e}")
        continue

    if pred_str == true_str:
        correct_full += 1
        status = "✓"
    else:
        status = "✗"
        failures.append((fn, true_str, pred_str))

    # char-level accuracy (only count if same length)
    if len(pred_str) == len(true_str):
        for p, t in zip(pred_str, true_str):
            total_chars += 1
            if p == t: correct_chars += 1
            if t.isdigit():
                per_digit_stat[int(t)][1] += 1
                if p == t: per_digit_stat[int(t)][0] += 1

    print(f"  {i+1:2d}. {fn[:24]} → '{pred_str}' (true '{true_str}') {status}")

dt = time.time() - t0
s.close()

print(f"\n{'='*60}")
print(f"Plate-level accuracy: {correct_full}/{len(plate_files)} = {correct_full/len(plate_files)*100:.1f}%")
if total_chars > 0:
    print(f"Char-level accuracy:  {correct_chars}/{total_chars} = {correct_chars/total_chars*100:.1f}%")
print(f"Avg time: {dt/len(plate_files)*1000:.1f}ms per plate (segmentation + CNN)")
print(f"\nPer-digit stats:")
for d in range(10):
    c, t = per_digit_stat[d]
    if t > 0: print(f"  digit {d}: {c}/{t} = {c/t*100:.0f}%")
if failures:
    print(f"\nFailures ({len(failures)}):")
    for fn, true_s, pred_s in failures[:10]:
        print(f"  {fn[:30]}: pred='{pred_s}' true='{true_s}'")
