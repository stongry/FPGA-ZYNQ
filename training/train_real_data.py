"""Train LPR36 + CN31 on REAL plate character crops from richjjj/chinese_license_plate_rec dataset.
Parallel segmentation across 40 cores, dual-GPU training.
"""
import numpy as np, os, random, time, argparse
from PIL import Image
from multiprocessing import Pool

PROVINCES = '京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼'
LPR36 = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
DATA_ROOT = '/tmp/clpr/images'
LABEL_FILE = '/tmp/clpr/plate_labels/balanced_base_lpr_3000_train.txt'
VAL_LABEL_FILE = '/tmp/clpr/plate_labels/balanced_base_lpr_3000_val.txt'

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
    if h >= w: nh, nw = 22, max(1, int(w*22/h))
    else: nw, nh = 22, max(1, int(h*22/w))
    img = Image.fromarray(patch).resize((nw, nh), Image.BILINEAR)
    out = np.zeros((28,28), dtype=np.uint8)
    y0c = (28-nh)//2; x0c = (28-nw)//2
    out[y0c:y0c+nh, x0c:x0c+nw] = np.array(img)
    return out

def segment_plate_to_chars(image_path, n_chars=7):
    """Segment real plate into char patches using height-filtered CC + adaptive CN boundary."""
    try:
        img = np.array(Image.open(image_path).convert('L'))
    except Exception:
        return None
    H, W = img.shape
    if H < 20 or W < 50: return None
    img_work = 255 - img if img.mean() > 127 else img.copy()
    t = otsu(img_work)
    binary = (img_work > t).astype(np.uint8)
    boxes = cc_boxes(binary, min_area=15)
    if not boxes: return None
    min_h = int(H * 0.35); max_h = int(H * 0.95)
    chars = [b for b in boxes if min_h <= (b[3]-b[1]+1) <= max_h]
    chars.sort(key=lambda b: b[0])
    if len(chars) == 0: return None
    x0a = min(b[0] for b in chars); x1a = max(b[2] for b in chars)
    total_w = x1a - x0a + 1
    avg_cw = total_w / n_chars
    best = None; best_score = 1e9
    for ratio in [0.9, 1.0, 1.1, 1.2, 1.3]:
        cn_b = x0a + int(avg_cw * ratio)
        cn_list = [b for b in chars if b[0] < cn_b]
        oth_list = [b for b in chars if b[0] >= cn_b]
        score = abs(len(oth_list) - (n_chars - 1))
        if len(cn_list) == 0: score += 100
        if score < best_score:
            best_score = score; best = (cn_list, oth_list)
    cn_boxes, other_boxes = best
    if not cn_boxes or len(other_boxes) < 3: return None
    cx0 = min(b[0] for b in cn_boxes); cx1 = max(b[2] for b in cn_boxes)
    cy0 = min(b[1] for b in cn_boxes); cy1 = max(b[3] for b in cn_boxes)
    p = img_work[cy0:cy1+1, cx0:cx1+1].astype(np.uint8)
    p = np.where(p > t, p, 0).astype(np.uint8)
    patches = [norm28(p)]
    for (x0, y0, x1, y1, _) in other_boxes[:n_chars-1]:
        p = img_work[y0:y1+1, x0:x1+1].astype(np.uint8)
        p = np.where(p > t, p, 0).astype(np.uint8)
        patches.append(norm28(p))
    while len(patches) < n_chars:
        patches.append(np.zeros((28,28), dtype=np.uint8))
    return patches[:n_chars]

def _process_label(entry):
    """Worker: given (rel_path, plate_text), return list of (patch, char_label, position).
    Skip if label is not standard 7-char format, or if segmentation fails."""
    rel_path, plate_text = entry
    # Handle 8-char NEV plates: skip for now
    if len(plate_text) != 7: return []
    # Validate each char is in our alphabet
    if plate_text[0] not in PROVINCES: return []
    for c in plate_text[1:]:
        if c not in LPR36: return []
    img_path = os.path.join(DATA_ROOT, rel_path)
    if not os.path.exists(img_path): return []
    patches = segment_plate_to_chars(img_path, 7)
    if patches is None or len(patches) != 7: return []
    # Pair each patch with its char label + position
    results = []
    for pos, patch in enumerate(patches):
        char = plate_text[pos]
        if pos == 0:  # Chinese province
            label_idx = PROVINCES.index(char)
        else:  # Alphanum
            label_idx = LPR36.index(char)
        results.append((patch.flatten(), label_idx, pos, char))
    return results

def load_and_segment(label_file, max_samples=None, n_workers=40):
    """Load labels, segment in parallel, return (patches, labels, positions)."""
    entries = []
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ')
            if len(parts) >= 2:
                entries.append((parts[0], parts[1]))
    if max_samples: entries = entries[:max_samples]
    print(f"  Processing {len(entries)} plates across {n_workers} cores...")
    t0 = time.time()
    with Pool(n_workers) as p:
        all_results = p.map(_process_label, entries)
    # Flatten
    patches = []; labels = []; positions = []; chars = []
    for plate_results in all_results:
        for patch, label, pos, char in plate_results:
            patches.append(patch)
            labels.append(label)
            positions.append(pos)
            chars.append(char)
    print(f"  → {len(patches)} chars from {sum(1 for r in all_results if len(r)==7)} valid plates in {time.time()-t0:.1f}s")
    return np.array(patches, dtype=np.float32), np.array(labels), np.array(positions), chars

def train_classifier(X_tr, y_tr, X_te, y_te, hidden, num_classes, device, epochs=25, batch=1024, name=""):
    import torch, torch.nn as nn
    class MLP(nn.Module):
        def __init__(s):
            super().__init__()
            s.fc1 = nn.Linear(784, hidden); s.fc2 = nn.Linear(hidden, num_classes)
        def forward(s, x): return s.fc2(torch.relu(s.fc1(x)))
    mlp = MLP().to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
    loss_fn = nn.CrossEntropyLoss()
    Xt = torch.tensor(X_tr/255.0).to(device); yt = torch.tensor(y_tr, dtype=torch.long).to(device)
    Xte = torch.tensor(X_te/255.0).to(device); yte = torch.tensor(y_te, dtype=torch.long).to(device)
    for ep in range(epochs):
        perm = torch.randperm(len(Xt)); tl = 0
        for i in range(0, len(Xt), batch):
            idx = perm[i:i+batch]
            opt.zero_grad()
            loss = loss_fn(mlp(Xt[idx]), yt[idx])
            loss.backward(); opt.step(); tl += loss.item()
        sched.step()
        mlp.eval()
        with torch.no_grad():
            acc = (mlp(Xte).argmax(1) == yte).float().mean().item()
        mlp.train()
        if ep % 3 == 0 or ep == epochs-1:
            print(f"  [{name}] Ep{ep+1}/{epochs}: loss={tl:.1f} acc={acc*100:.2f}%")
    mlp.eval()
    with torch.no_grad():
        pred = mlp(Xte).argmax(1).cpu().numpy()
        fp32_acc = (pred == y_te).mean()
    return mlp, float(fp32_acc)

def export_header(mlp, X_te, y_te, prefix, guard, hidden, n_cls, hidden_name, classes_name, chars_decl, out_file):
    W1 = mlp.fc1.weight.data.cpu().numpy(); b1 = mlp.fc1.bias.data.cpu().numpy()
    W2 = mlp.fc2.weight.data.cpu().numpy(); b2 = mlp.fc2.bias.data.cpu().numpy()
    def q(W):
        s = float(np.abs(W).max()/127)
        return np.clip(np.round(W/s),-128,127).astype(np.int8), s
    W1q, s1 = q(W1); W2q, s2 = q(W2)
    X_te_u8 = np.clip(np.round(X_te),0,255).astype(np.uint8)
    acc1 = X_te_u8.astype(np.int32) @ W1q.T.astype(np.int32)
    z1 = acc1.astype(np.float32)*(s1/255.0) + b1
    h1 = np.maximum(z1, 0)
    acc2 = h1 @ W2q.T.astype(np.float32)*s2 + b2
    int8_acc = (acc2.argmax(1) == y_te).mean()
    h = f"""/* {prefix} v8 REAL data trained (richjjj/chinese_license_plate_rec) */
/* INT8={int8_acc*100:.2f}% */
#ifndef {guard}
#define {guard}
#define {hidden_name} {hidden}
#define {classes_name} {n_cls}
static const float {prefix}_s1 = {s1:.9e}f;
static const float {prefix}_s2 = {s2:.9e}f;
static const int8_t {prefix}_W1[{hidden}][784] = {{
"""
    for j in range(hidden):
        h += "  {" + ",".join(str(int(v)) for v in W1q[j]) + "},\n"
    h += "};\n"
    h += f"static const float {prefix}_b1[{hidden}] = {{" + ",".join(f"{v:.6e}f" for v in b1) + "};\n"
    h += f"static const int8_t {prefix}_W2[{n_cls}][{hidden}] = {{\n"
    for j in range(n_cls):
        h += "  {" + ",".join(str(int(v)) for v in W2q[j]) + "},\n"
    h += "};\n"
    h += f"static const float {prefix}_b2[{n_cls}] = {{" + ",".join(f"{v:.6e}f" for v in b2) + "};\n"
    h += chars_decl + "#endif\n"
    with open(out_file, 'w', encoding='utf-8') as f: f.write(h)
    print(f"  [{prefix}] INT8: {int8_acc*100:.2f}%  saved {out_file}")
    return float(int8_acc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--classes', choices=['lpr36', 'cn31'], required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch', type=int, default=1024)
    parser.add_argument('--max-train', type=int, default=10000)
    args = parser.parse_args()

    print(f"=== Training {args.classes} on REAL data ({args.device}) ===")
    t_all = time.time()

    print("[Data] Loading training plates...")
    X_tr, y_tr, pos_tr, chars_tr = load_and_segment(LABEL_FILE, max_samples=args.max_train)
    print("[Data] Loading validation plates...")
    X_te, y_te, pos_te, chars_te = load_and_segment(VAL_LABEL_FILE, max_samples=2000)

    # Filter by classifier type
    if args.classes == 'lpr36':
        mask_tr = pos_tr >= 1; mask_te = pos_te >= 1
        HIDDEN, N_CLS = 256, 36
        OUT = '/tmp/lpr36_weights.h'
        prefix, guard = 'lpr36', 'LPR36_WEIGHTS_H'
        hidden_name, classes_name = 'LPR36_HIDDEN', 'LPR36_CLASSES'
        chars_decl = 'static const char lpr36_chars[36] = {' + ','.join(f"'{c}'" for c in LPR36) + '};\n'
    else:
        mask_tr = pos_tr == 0; mask_te = pos_te == 0
        HIDDEN, N_CLS = 256, 31
        OUT = '/tmp/lpr_cn31_weights.h'
        prefix, guard = 'cn31', 'LPR_CN31_WEIGHTS_H'
        hidden_name, classes_name = 'CN31_HIDDEN', 'CN31_CLASSES'
        chars_decl = 'static const char * const cn31_provinces[31] = {' + ','.join(f'"{c}"' for c in PROVINCES) + '};\n'

    X_tr_f = X_tr[mask_tr]; y_tr_f = y_tr[mask_tr]
    X_te_f = X_te[mask_te]; y_te_f = y_te[mask_te]
    print(f"  Train: {len(X_tr_f)} samples, Test: {len(X_te_f)}")
    # Print class distribution for validation
    unique, counts = np.unique(y_tr_f, return_counts=True)
    print(f"  Classes covered: {len(unique)}/{N_CLS}")
    if len(unique) < N_CLS:
        missing = set(range(N_CLS)) - set(unique.tolist())
        names = [LPR36 if args.classes=='lpr36' else PROVINCES]
        char_list = LPR36 if args.classes=='lpr36' else PROVINCES
        print(f"  MISSING classes: {[char_list[i] for i in missing]}")

    print(f"[Train] Training {HIDDEN}-hidden MLP on {args.device}...")
    mlp, fp32 = train_classifier(X_tr_f, y_tr_f, X_te_f, y_te_f, HIDDEN, N_CLS, args.device,
                                   epochs=args.epochs, batch=args.batch, name=args.classes)

    print(f"[Export] FP32 test acc: {fp32*100:.2f}%")
    int8_acc = export_header(mlp, X_te_f, y_te_f, prefix, guard, HIDDEN, N_CLS,
                              hidden_name, classes_name, chars_decl, OUT)
    print(f"\nTotal: {time.time()-t_all:.1f}s")
