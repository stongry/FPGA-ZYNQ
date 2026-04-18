#!/usr/bin/env python3
"""PS+PL 协同 test on real plates.
3-pass inference:
  Pass 1 (mode 6, CN31 PS): classify first char (Chinese province)
  Pass 2 (mode 5, LPR36 PS): classify all alphanum chars
  Pass 3 (mode 1, PL CNN):  VERIFY chars that LPR36 predicts as digit (0-9)
                             PL CNN at 98.79% on MNIST digits >> PS LPR36
                             If PL CNN agrees → confirm digit
                             If disagree → use higher-confidence prediction
"""
import socket, struct, numpy as np, os, time, subprocess, json, re
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
    """Send 28x28 char, return (pred_u8, probs[10] as numpy)."""
    hdr = b'MNI\x00' + struct.pack('<III', 28, 28, 0)
    sock.sendall(hdr + img.astype(np.uint8).tobytes())
    resp = b''
    while len(resp) < 48:
        chunk = sock.recv(48 - len(resp))
        if not chunk: raise ConnectionError
        resp += chunk
    pred = resp[4]
    probs = np.frombuffer(resp[8:48], dtype=np.float32)
    return pred, probs

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

def cc_boxes(binary, min_area=20):
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
    if h >= w: nh, nw = 22, max(1, int(w*22/h))
    else: nw, nh = 22, max(1, int(h*22/w))
    img = Image.fromarray(patch).resize((nw, nh), Image.BILINEAR)
    out = np.zeros((28,28), dtype=np.uint8)
    y0c = (28-nh)//2; x0c = (28-nw)//2
    out[y0c:y0c+nh, x0c:x0c+nw] = np.array(img)
    return out

def segment_plate_v2(image_path, n_chars=7):
    """Resize to 280x80, filter borders + dot, adaptive CN boundary."""
    try:
        img_pil = Image.open(image_path).convert('L').resize((280, 80), Image.BILINEAR)
    except Exception:
        return None
    img = np.array(img_pil)
    H, W = img.shape
    img_work = 255 - img if img.mean() > 127 else img.copy()
    t = otsu(img_work)
    binary = (img_work > t).astype(np.uint8)
    boxes = cc_boxes(binary, min_area=25)
    if not boxes: return None
    char_boxes = []
    for b in boxes:
        bh = b[3]-b[1]+1; bw = b[2]-b[0]+1
        if bh < 25 or bh > 76: continue
        if bw < 5 or bw > 70: continue
        if b[1] == 0 and bh < 40: continue
        if b[3] >= H-1 and bh < 40: continue
        char_boxes.append(b)
    char_boxes.sort(key=lambda b: b[0])
    non_dot = []
    for b in char_boxes:
        bw = b[2]-b[0]+1; bh = b[3]-b[1]+1; area = b[4]
        if area < 100 and bw < 15 and bh < 25:
            continue
        non_dot.append(b)
    if len(non_dot) < 4: return None
    x0a = min(b[0] for b in non_dot); x1a = max(b[2] for b in non_dot)
    total_w = x1a - x0a + 1
    avg_cw = total_w / n_chars
    best_cn = None; best_other = None; best_score = 1e9
    for ratio in [0.9, 1.0, 1.1, 1.2]:
        cn_b = x0a + int(avg_cw * ratio)
        cn_l = [b for b in non_dot if b[0] < cn_b]
        oth_l = [b for b in non_dot if b[0] >= cn_b]
        score = abs(len(oth_l) - (n_chars - 1))
        if len(cn_l) == 0: score += 100
        if score < best_score:
            best_score = score; best_cn = cn_l; best_other = oth_l
    if not best_cn or len(best_other) < 3: return None
    cx0 = min(b[0] for b in best_cn); cx1 = max(b[2] for b in best_cn)
    cy0 = min(b[1] for b in best_cn); cy1 = max(b[3] for b in best_cn)
    p = img_work[cy0:cy1+1, cx0:cx1+1].astype(np.uint8)
    p = np.where(p > t, p, 0).astype(np.uint8)
    patches = [norm28(p)]
    for (x0, y0, x1, y1, _) in best_other[:n_chars-1]:
        p = img_work[y0:y1+1, x0:x1+1].astype(np.uint8)
        p = np.where(p > t, p, 0).astype(np.uint8)
        patches.append(norm28(p))
    while len(patches) < n_chars:
        patches.append(np.zeros((28,28), dtype=np.uint8))
    return patches[:n_chars]

def extract_label(fn):
    m = re.match(r'_\d+_([^.]+)\.\w+', fn)
    if m: return m.group(1)
    base = fn.split('.')[0]
    if '_convert' in base: return base.split('_')[0]
    parts = base.rsplit('_', 1)
    if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) <= 2:
        return parts[0] + parts[1]
    return base

# === Main ===
DIR = '/tmp/real_plates'
files = sorted([f for f in os.listdir(DIR) if f.endswith(('.jpg','.png'))])
print(f"Testing {len(files)} REAL plates with PS+PL cooperation\n")

# Segment all plates
print("[Seg] Segmenting all plates (PS Python)...")
all_patches = []  # [(plate_idx, char_idx, patch, fn, label)]
skip_list = []
for pi, fn in enumerate(files):
    lbl = extract_label(fn)
    if len(lbl) != 7:
        skip_list.append(fn); continue
    patches = segment_plate_v2(os.path.join(DIR, fn), 7)
    if patches is None or len(patches) != 7:
        skip_list.append(fn); continue
    for ci, p in enumerate(patches):
        all_patches.append((pi, ci, p, fn, lbl))
n_plates = len([f for f in files if f not in skip_list])
print(f"  {n_plates} plates × 7 chars = {len(all_patches)} patches")

# Hotdow to reset board state (mode 1 default)
print("[Reset] Hotdow firmware...")
subprocess.run(WIN_SSH + ["C:\\\\Xilinx\\\\Vivado\\\\2024.2\\\\bin\\\\xsdb.bat C:\\\\Users\\\\huye\\\\fz3a\\\\dp\\\\hotdow.tcl"],
               capture_output=True, timeout=60)
subprocess.run(WIN_SSH + ["C:\\\\Xilinx\\\\Vivado\\\\2024.2\\\\bin\\\\xsdb.bat C:\\\\Users\\\\huye\\\\fz3a\\\\dp\\\\continue_cpu.tcl"],
               capture_output=True, timeout=30)
for i in range(15):
    try:
        sp = socket.socket(); sp.settimeout(1); sp.connect((BOARD, PORT)); sp.close(); break
    except: time.sleep(1)

s = socket.socket(); s.settimeout(15); s.connect((BOARD, PORT))

# Pass 1: mode 6 (CN31) for Chinese chars
print("[Pass 1] Switch to mode 6 (CN31 PS)")
send_uart_M(5)  # 1→6
cn_preds = {}; cn_conf = {}
for pi, ci, p, fn, lbl in all_patches:
    if ci == 0:
        pred, probs = send_char(s, p)
        cn_preds[pi] = (PROVINCES[pred] if pred < 31 else '?')
        cn_conf[pi] = float(probs[:10].max())  # approximate confidence
print(f"  Classified {len(cn_preds)} Chinese chars")

# Pass 2: mode 5 (LPR36) for alphanum
print("[Pass 2] Switch to mode 5 (LPR36 PS)")
send_uart_M(6)  # 6→5
ps_preds = {}; ps_probs = {}
for pi, ci, p, fn, lbl in all_patches:
    if ci >= 1:
        pred, probs = send_char(s, p)
        ps_preds[(pi, ci)] = pred
        ps_probs[(pi, ci)] = probs

# Pass 3: mode 1 (PL CNN) to verify chars PS predicts as digits
print("[Pass 3] Switch to mode 1 (PL CNN) for digit verification")
send_uart_M(3)  # 5→6→0→1
pl_preds = {}; pl_probs = {}
verify_count = 0
for pi, ci, p, fn, lbl in all_patches:
    if ci >= 1 and ps_preds[(pi, ci)] < 10:  # only verify if PS thinks it's a digit
        pred, probs = send_char(s, p)
        pl_preds[(pi, ci)] = pred
        pl_probs[(pi, ci)] = probs
        verify_count += 1
print(f"  Verified {verify_count} digit candidates with PL CNN")

s.close()

# Combine predictions
final_preds = {}
ps_to_pl_changes = 0; ps_to_pl_agree = 0
for pi, ci, p, fn, lbl in all_patches:
    if ci == 0:
        final_preds[(pi, ci)] = cn_preds.get(pi, '?')
    else:
        ps_pred = ps_preds[(pi, ci)]
        if ps_pred < 10 and (pi, ci) in pl_preds:
            pl_pred = pl_preds[(pi, ci)]
            ps_conf = float(ps_probs[(pi, ci)][ps_pred])
            pl_conf = float(pl_probs[(pi, ci)][pl_pred])
            # If PL more confident, use PL
            if pl_pred < 10 and pl_conf > ps_conf:
                final_preds[(pi, ci)] = LPR36[pl_pred]  # digit
                if pl_pred != ps_pred: ps_to_pl_changes += 1
                else: ps_to_pl_agree += 1
            else:
                final_preds[(pi, ci)] = LPR36[ps_pred] if ps_pred < 36 else '?'
                if pl_pred == ps_pred: ps_to_pl_agree += 1
        else:
            final_preds[(pi, ci)] = LPR36[ps_pred] if ps_pred < 36 else '?'

print(f"  PS↔PL agreements: {ps_to_pl_agree}, PS→PL changes: {ps_to_pl_changes}")

# Score
print("\n=== RESULTS (PS+PL 协同) ===")
correct_plate = 0; total_plate = 0; correct_char = 0; total_char = 0
prov_c = 0; prov_t = 0; aln_c = 0; aln_t = 0
for pi, fn in enumerate(files):
    if fn in skip_list: continue
    lbl = extract_label(fn)
    pred = ''.join(final_preds.get((pi, ci), '?') for ci in range(7))
    total_plate += 1
    if pred == lbl:
        correct_plate += 1; status = "✓"
    else: status = "✗"
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
