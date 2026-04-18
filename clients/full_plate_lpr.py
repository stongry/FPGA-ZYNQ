#!/usr/bin/env python3
"""FAST Chinese LPR — batch process + hotdow reset for deterministic starting mode.

Strategy:
  1. Hotdow ELF → board at default mode 1 (PL-CNN)
  2. Segment ALL plates upfront
  3. Press 'M' 5 times → mode 6 (CN31)
  4. Classify all Chinese chars (1 per plate)
  5. Press 'M' 6 times → mode 5 (LPR36)
  6. Classify all alphanum chars (6 per plate)
  7. Assemble
"""
import socket, struct, numpy as np, os, json, time, subprocess
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

def send_char(sock, img_28x28):
    hdr = b'MNI\x00' + struct.pack('<III', 28, 28, 0)
    sock.sendall(hdr + img_28x28.astype(np.uint8).tobytes())
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

def norm28(patch):
    h, w = patch.shape
    if h == 0 or w == 0: return np.zeros((28,28), dtype=np.uint8)
    target = 22
    if h >= w:
        nh = target; nw = max(1, int(w * target / h))
    else:
        nw = target; nh = max(1, int(h * target / w))
    img = Image.fromarray(patch).resize((nw, nh), Image.BILINEAR)
    out = np.zeros((28, 28), dtype=np.uint8)
    y0 = (28 - nh) // 2; x0 = (28 - nw) // 2
    out[y0:y0+nh, x0:x0+nw] = np.array(img)
    return out

def cc_boxes(binary, min_area=60):
    h, w = binary.shape
    label = np.zeros((h, w), dtype=np.int32); nl = 1
    for y in range(h):
        for x in range(w):
            if binary[y, x] and label[y, x] == 0:
                stk = [(y, x)]
                while stk:
                    cy, cx = stk.pop()
                    if cy<0 or cy>=h or cx<0 or cx>=w: continue
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

def segment_hybrid(image_path, n_chars=7):
    """Adaptive segmentation: try multiple CN/alphanum boundary ratios,
       pick the one giving exactly 6 well-separated alphanum components."""
    img = np.array(Image.open(image_path).convert('L'))
    img_inv = 255 - img if img.mean() > 127 else img
    t = otsu(img_inv)
    binary = (img_inv > t).astype(np.uint8)

    col_sum = binary.sum(axis=0); row_sum = binary.sum(axis=1)
    ac = np.where(col_sum > 0)[0]; ar = np.where(row_sum > 0)[0]
    if len(ac) == 0 or len(ar) == 0: return []
    x0a, x1a = ac[0], ac[-1]; y0a, y1a = ar[0], ar[-1]
    total_w = x1a - x0a + 1
    avg_char_w = total_w / n_chars

    # Get all CC boxes sorted by x
    boxes = cc_boxes(binary, min_area=40)
    boxes.sort(key=lambda b: b[0])
    if len(boxes) == 0: return [np.zeros((28,28),dtype=np.uint8)] * n_chars

    # Adaptive boundary: try several ratios, pick one yielding close to 6 alphanum boxes
    # Sometimes Chinese char is narrow (1 box), sometimes wide (3+ boxes)
    best_cn_boxes = None; best_other = None; best_score = 1e9
    for ratio in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
        cn_b = x0a + int(avg_char_w * ratio)
        cn_list = [b for b in boxes if b[0] < cn_b]
        oth_list = [b for b in boxes if b[0] >= cn_b]
        # Score: prefer having exactly n-1=6 alphanum boxes
        score = abs(len(oth_list) - (n_chars - 1))
        # Penalty: fewer CN boxes is fine, but CN should have at least 1
        if len(cn_list) == 0: score += 100
        # Penalty: too many CN boxes (ratio>1.5) unlikely
        if len(cn_list) > 5: score += 10
        if score < best_score:
            best_score = score; best_cn_boxes = cn_list; best_other = oth_list
    cn_boxes = best_cn_boxes; other_boxes = best_other

    # Merge CN boxes into single bbox
    if cn_boxes:
        cx0 = min(b[0] for b in cn_boxes); cx1 = max(b[2] for b in cn_boxes)
        cy0 = min(b[1] for b in cn_boxes); cy1 = max(b[3] for b in cn_boxes)
        p = img_inv[cy0:cy1+1, cx0:cx1+1].astype(np.uint8)
        p = np.where(p > t, p, 0).astype(np.uint8)
        cn_patch = norm28(p)
    else:
        cn_patch = np.zeros((28,28), dtype=np.uint8)

    # If we have MORE than 6 alphanum boxes, merge adjacent small ones (possible noise or split chars)
    # If we have FEWER than 6 boxes, likely merged chars - hard to fix without over-splitting
    # For now: just take first 6
    patches = [cn_patch]
    for (x0, y0, x1, y1, _) in other_boxes[:n_chars-1]:
        p = img_inv[y0:y1+1, x0:x1+1].astype(np.uint8)
        p = np.where(p > t, p, 0).astype(np.uint8)
        patches.append(norm28(p))
    while len(patches) < n_chars:
        patches.append(np.zeros((28,28), dtype=np.uint8))
    return patches[:n_chars]

# === Main ===
labels = json.load(open('/tmp/full_plates/labels.json'))
files = sorted([f for f in os.listdir('/tmp/full_plates') if f.endswith('.png')])
N = len(files)
print(f"Testing {N} full Chinese plates (FAST batched + hotdow reset)...\n")

# Phase 0: Segment all plates (no board needed)
t_seg0 = time.time()
all_patches = []
for pi, fn in enumerate(files):
    patches = segment_hybrid(f'/tmp/full_plates/{fn}', 7)
    if len(patches) == 7:
        for ci, p in enumerate(patches):
            all_patches.append((pi, ci, p))
t_seg = time.time() - t_seg0
print(f"[Seg] {N} plates → {len(all_patches)} chars in {t_seg:.1f}s")

# Phase 1: Hotdow firmware to reset to default mode 1 (deterministic)
print("[Reset] Hotdow firmware (forces mode 1 default)...")
t_rst0 = time.time()
subprocess.run(WIN_SSH + ["C:\\\\Xilinx\\\\Vivado\\\\2024.2\\\\bin\\\\xsdb.bat C:\\\\Users\\\\huye\\\\fz3a\\\\dp\\\\hotdow.tcl"],
               capture_output=True, timeout=60)
subprocess.run(WIN_SSH + ["C:\\\\Xilinx\\\\Vivado\\\\2024.2\\\\bin\\\\xsdb.bat C:\\\\Users\\\\huye\\\\fz3a\\\\dp\\\\continue_cpu.tcl"],
               capture_output=True, timeout=30)
for i in range(15):
    try:
        sp = socket.socket(); sp.settimeout(1); sp.connect((BOARD, PORT)); sp.close(); break
    except: time.sleep(1)
t_rst = time.time() - t_rst0
print(f"[Reset] Board back online in {t_rst:.1f}s, at mode 1 (PL-CNN)")
# Phase 2: Switch to mode 6 (from mode 1: +5 presses)
t_sw0 = time.time()
send_uart_M(5)  # 1→2→3→4→5→6
t_sw1 = time.time() - t_sw0
print(f"[Sw] Mode 1→6 in {t_sw1:.1f}s")

# Phase 3: Classify all Chinese chars
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(15); s.connect((BOARD, PORT))
t_cn0 = time.time()
cn_preds = {}
for pi, ci, p in all_patches:
    if ci == 0:
        pred = send_char(s, p)
        cn_preds[pi] = PROVINCES[pred] if pred < 31 else '?'
t_cn = time.time() - t_cn0
print(f"[CN]  {N} chars classified in {t_cn:.2f}s ({t_cn/N*1000:.1f}ms/char)")

# Phase 4: Switch to mode 5 (6→0→1→2→3→4→5: +6 presses)
t_sw2a = time.time()
send_uart_M(6)
t_sw2 = time.time() - t_sw2a
print(f"[Sw] Mode 6→5 in {t_sw2:.1f}s")

# Phase 5: Classify all alphanum chars
t_al0 = time.time()
al_preds = {}
for pi, ci, p in all_patches:
    if ci >= 1:
        pred = send_char(s, p)
        al_preds[(pi, ci)] = LPR36[pred] if pred < 36 else '?'
t_al = time.time() - t_al0
print(f"[AL]  {N*6} chars classified in {t_al:.2f}s ({t_al/(N*6)*1000:.1f}ms/char)")
s.close()

# Phase 6: Assemble + score
correct_plate = 0; correct_char = 0; total_char = 0
prov_c = 0; prov_t = 0; aln_c = 0; aln_t = 0
failures = []
for pi, fn in enumerate(files):
    true_s = fn.split('_', 2)[-1].replace('.png', '')
    if pi not in cn_preds:
        pred_s = '?'*7
    else:
        pred_s = cn_preds[pi] + ''.join(al_preds.get((pi, ci), '?') for ci in range(1, 7))
    ok = pred_s == true_s
    status = "✓" if ok else "✗"
    if ok: correct_plate += 1
    else: failures.append((fn, true_s, pred_s))
    if len(pred_s) == len(true_s):
        for idx, (p, t) in enumerate(zip(pred_s, true_s)):
            total_char += 1
            if p == t: correct_char += 1
            if idx == 0:
                prov_t += 1
                if p == t: prov_c += 1
            else:
                aln_t += 1
                if p == t: aln_c += 1
    print(f"  {pi+1:2d}. '{pred_s}'  (true '{true_s}') {status}")

t_total = time.time() - t_seg0
print(f"\n{'='*60}")
print(f"Plate-level:   {correct_plate}/{N} = {correct_plate/N*100:.1f}%")
print(f"Char-level:    {correct_char}/{total_char} = {100*correct_char/max(total_char,1):.1f}%")
print(f"  Province (CN 31):  {prov_c}/{prov_t} = {100*prov_c/max(prov_t,1):.1f}%")
print(f"  Alphanum (LPR 36): {aln_c}/{aln_t} = {100*aln_c/max(aln_t,1):.1f}%")
print(f"Timing breakdown:")
print(f"  Segment:       {t_seg:.1f}s")
print(f"  Hotdow reset:  {t_rst:.1f}s")
print(f"  Mode switch:   {t_sw1+t_sw2:.1f}s")
print(f"  CN classify:   {t_cn:.1f}s ({t_cn/N*1000:.0f}ms/char)")
print(f"  AL classify:   {t_al:.1f}s ({t_al/(N*6)*1000:.0f}ms/char)")
print(f"  TOTAL: {t_total:.1f}s for {N} plates ({t_total/N*1000:.0f}ms/plate)")
if failures:
    print(f"\nFailures ({len(failures)}):")
    for fn, t, p in failures[:12]:
        print(f"  {fn[:35]}: '{p}' vs '{t}'")

json.dump({
    'plate_acc': correct_plate/N, 'char_acc': correct_char/max(total_char,1),
    'province_acc': prov_c/max(prov_t,1), 'alnum_acc': aln_c/max(aln_t,1),
    'total_time_s': t_total, 'per_plate_ms': t_total/N*1000,
}, open('/tmp/full_plate_fast_results.json','w'), ensure_ascii=False, indent=2)
