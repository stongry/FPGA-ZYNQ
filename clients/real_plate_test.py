#!/usr/bin/env python3
"""Test LPR on REAL plate images (blue background, white text, standard Chinese format)."""
import socket, struct, numpy as np, os, time, subprocess, re
from PIL import Image

BOARD = "192.168.6.191"
PORT = 5001
WIN_SSH = ["ssh", "-p", "2222", "huye@192.168.6.244"]
PROVINCES = '京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼'
LPR36 = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

def send_uart_M(n=1):
    for _ in range(n):
        cmd = 'powershell -Command "$p=[System.IO.Ports.SerialPort]::new(\'COM9\',115200); $p.Open(); $p.Write(\'M\'); Start-Sleep -Milliseconds 150; $p.Close()"'
        subprocess.run(WIN_SSH + [cmd], capture_output=True, timeout=15)
        time.sleep(0.25)

def send_char(sock, img):
    hdr = b'MNI\x00' + struct.pack('<III', 28, 28, 0)
    sock.sendall(hdr + img.astype(np.uint8).tobytes())
    resp = b''
    while len(resp) < 48:
        chunk = sock.recv(48 - len(resp))
        if not chunk: raise ConnectionError
        resp += chunk
    return resp[4]

def otsu(g):
    hist, _ = np.histogram(g.flatten(), bins=256, range=(0, 256))
    total = g.size; st = np.sum(np.arange(256) * hist)
    sb = 0; wb = 0; mv = 0; th = 0
    for t in range(256):
        wb += hist[t]
        if wb == 0: continue
        wf = total - wb
        if wf == 0: break
        sb += t * hist[t]
        v = wb * wf * (sb/wb - (st-sb)/wf)**2
        if v > mv: mv = v; th = t
    return th

def cc_boxes(binary, min_area=30):
    h, w = binary.shape
    label = np.zeros((h, w), dtype=np.int32); nl = 1
    for y in range(h):
        for x in range(w):
            if binary[y, x] and label[y, x] == 0:
                stk = [(y, x)]
                while stk:
                    cy, cx = stk.pop()
                    if cy < 0 or cy >= h or cx < 0 or cx >= w: continue
                    if not binary[cy, cx] or label[cy, cx]: continue
                    label[cy, cx] = nl
                    stk.extend([(cy+1,cx),(cy-1,cx),(cy,cx+1),(cy,cx-1)])
                nl += 1
    boxes = []
    for k in range(1, nl):
        ys, xs = np.where(label == k)
        if len(ys) < min_area: continue
        boxes.append((xs.min(), ys.min(), xs.max(), ys.max(), len(ys)))
    return boxes

def norm28(patch):
    h, w = patch.shape
    if h == 0 or w == 0: return np.zeros((28,28), dtype=np.uint8)
    target = 22
    if h >= w: nh=target; nw=max(1,int(w*target/h))
    else: nw=target; nh=max(1,int(h*target/w))
    img = Image.fromarray(patch).resize((nw, nh), Image.BILINEAR)
    out = np.zeros((28, 28), dtype=np.uint8)
    y0 = (28-nh)//2; x0 = (28-nw)//2
    out[y0:y0+nh, x0:x0+nw] = np.array(img)
    return out

def segment_real_plate(image_path, n_chars=7):
    """Segment real plate: handle blue bg, dot separator, border.
    Strategy:
      1. Grayscale, detect if needs inversion
      2. Otsu binarize
      3. Find all CC boxes
      4. Filter by height (char-sized, not border/dot)
      5. Take leftmost cluster as Chinese (merge), next 6 as alphanum
    """
    img = np.array(Image.open(image_path).convert('L'))
    H, W = img.shape
    # Real plates: blue bg + white text. After grayscale: dark bg + light text
    # If mean > 127 (whitish bg synthetic plates), invert
    img_work = 255 - img if img.mean() > 127 else img.copy()
    t = otsu(img_work)
    binary = (img_work > t).astype(np.uint8)

    # Find all CC boxes
    boxes = cc_boxes(binary, min_area=20)
    if not boxes:
        return [np.zeros((28,28),dtype=np.uint8)]*n_chars

    # Filter by character-like height (at least 40% of plate height)
    min_h = int(H * 0.35)
    chars = [b for b in boxes if (b[3]-b[1]+1) >= min_h and (b[3]-b[1]+1) <= H*0.95]
    chars.sort(key=lambda b: b[0])

    if len(chars) == 0:
        return [np.zeros((28,28),dtype=np.uint8)]*n_chars

    # Compute overall active bbox
    x0a = min(b[0] for b in chars); x1a = max(b[2] for b in chars)
    total_w = x1a - x0a + 1
    avg_cw = total_w / n_chars

    # Adaptive Chinese boundary: try different ratios
    best = None; best_score = 1e9
    for ratio in [0.9, 1.0, 1.1, 1.2, 1.3, 1.4]:
        cn_b = x0a + int(avg_cw * ratio)
        cn_list = [b for b in chars if b[0] < cn_b]
        oth_list = [b for b in chars if b[0] >= cn_b]
        score = abs(len(oth_list) - (n_chars - 1))
        if len(cn_list) == 0: score += 100
        if len(cn_list) > 5: score += 10
        if score < best_score:
            best_score = score; best = (cn_list, oth_list)
    cn_boxes, other_boxes = best

    # Build Chinese patch by merging left cluster
    if cn_boxes:
        cx0 = min(b[0] for b in cn_boxes); cx1 = max(b[2] for b in cn_boxes)
        cy0 = min(b[1] for b in cn_boxes); cy1 = max(b[3] for b in cn_boxes)
        p = img_work[cy0:cy1+1, cx0:cx1+1].astype(np.uint8)
        p = np.where(p > t, p, 0).astype(np.uint8)
        cn_patch = norm28(p)
    else:
        cn_patch = np.zeros((28,28), dtype=np.uint8)

    patches = [cn_patch]
    for (x0, y0, x1, y1, _) in other_boxes[:n_chars-1]:
        p = img_work[y0:y1+1, x0:x1+1].astype(np.uint8)
        p = np.where(p > t, p, 0).astype(np.uint8)
        patches.append(norm28(p))
    while len(patches) < n_chars:
        patches.append(np.zeros((28,28), dtype=np.uint8))
    return patches[:n_chars]

def extract_label(fn):
    """Get ground truth from filename."""
    # Format: "_N_xxx.jpg" or "陕xxx_suffix.jpg"
    m = re.match(r'_\d+_([^.]+)\.\w+', fn)
    if m: return m.group(1)
    # e.g. "陕CQ3TP_1.jpg" → "陕CQ3TP" or "陕CQ3TP1"
    # The "_1" is ambiguous; try both
    base = fn.split('.')[0]
    if '_convert' in base:
        return base.split('_')[0]  # "新AU3006"
    # "陕CQ3TP_1" → "陕CQ3TP1" (treat suffix as last digit)
    parts = base.rsplit('_', 1)
    if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) <= 2:
        return parts[0] + parts[1]
    return base

# === Main ===
DIR = '/tmp/real_plates'
files = sorted([f for f in os.listdir(DIR) if f.endswith(('.jpg','.png'))])
print(f"Found {len(files)} real plates")

# Get ground truth from filenames
true_labels = {f: extract_label(f) for f in files}
for f in files:
    lbl = true_labels[f]
    print(f"  {f:40s} → label '{lbl}' (len={len(lbl)})")

# Reset board + switch to mode 6 (CN31)
print("\n[Reset] Hotdow firmware...")
subprocess.run(WIN_SSH + ["C:\\\\Xilinx\\\\Vivado\\\\2024.2\\\\bin\\\\xsdb.bat C:\\\\Users\\\\huye\\\\fz3a\\\\dp\\\\hotdow.tcl"],
               capture_output=True, timeout=60)
subprocess.run(WIN_SSH + ["C:\\\\Xilinx\\\\Vivado\\\\2024.2\\\\bin\\\\xsdb.bat C:\\\\Users\\\\huye\\\\fz3a\\\\dp\\\\continue_cpu.tcl"],
               capture_output=True, timeout=30)
for i in range(15):
    try:
        sp = socket.socket(); sp.settimeout(1); sp.connect((BOARD, PORT)); sp.close(); break
    except: time.sleep(1)
print("[Reset] OK, switching to mode 6 (CN31)")
send_uart_M(5)  # 1→6

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(15); s.connect((BOARD, PORT))

# Segment all plates (filter 7-char only)
all_patches = []
for pi, fn in enumerate(files):
    lbl = true_labels[fn]
    if len(lbl) != 7:
        print(f"  SKIP {fn} (label len={len(lbl)}, not standard 7)")
        continue
    patches = segment_real_plate(os.path.join(DIR, fn), 7)
    for ci, p in enumerate(patches):
        all_patches.append((pi, ci, p, fn, lbl))

# Classify Chinese chars (mode 6)
cn_preds = {}
for pi, ci, p, fn, lbl in all_patches:
    if ci == 0:
        pred = send_char(s, p)
        cn_preds[pi] = PROVINCES[pred] if pred < 31 else '?'
# Switch to mode 5 (LPR36): 6→0→1→2→3→4→5 (+6 presses)
send_uart_M(6)
al_preds = {}
for pi, ci, p, fn, lbl in all_patches:
    if ci >= 1:
        pred = send_char(s, p)
        al_preds[(pi, ci)] = LPR36[pred] if pred < 36 else '?'
s.close()

# Assemble + score
correct_plate = 0; total_plate = 0
correct_char = 0; total_char = 0
prov_c = 0; prov_t = 0; aln_c = 0; aln_t = 0
print("\n=== RESULTS ===")
for pi, fn in enumerate(files):
    lbl = true_labels[fn]
    if len(lbl) != 7: continue
    if pi not in cn_preds: continue
    pred = cn_preds[pi] + ''.join(al_preds.get((pi, ci), '?') for ci in range(1, 7))
    total_plate += 1
    if pred == lbl:
        correct_plate += 1; status = "✓"
    else: status = "✗"
    if len(pred) == len(lbl):
        for idx, (p, t) in enumerate(zip(pred, lbl)):
            total_char += 1
            if p == t: correct_char += 1
            if idx == 0:
                prov_t += 1; prov_c += (1 if p==t else 0)
            else:
                aln_t += 1; aln_c += (1 if p==t else 0)
    print(f"  {fn:35s} '{pred}' vs '{lbl}' {status}")

print(f"\n{'='*60}")
print(f"Plate-level: {correct_plate}/{total_plate} = {100*correct_plate/max(total_plate,1):.1f}%")
print(f"Char-level:  {correct_char}/{total_char} = {100*correct_char/max(total_char,1):.1f}%")
print(f"  Province: {prov_c}/{prov_t} = {100*prov_c/max(prov_t,1):.1f}%")
print(f"  Alphanum: {aln_c}/{aln_t} = {100*aln_c/max(aln_t,1):.1f}%")
