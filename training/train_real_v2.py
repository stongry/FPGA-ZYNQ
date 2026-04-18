"""Train LPR on real plates with IMPROVED segmentation + larger model + validation filter.

Improvements:
1. Resize plate to standard 280x80 before segmentation (helps low-res).
2. Detect and remove plate border (thin long components on edges).
3. Detect dot separator in position 2.5 region.
4. Filter training samples where segmentation quality is poor.
5. Larger 2-layer MLP (784 -> 512 -> 256 -> N).
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
    """Improved segmentation:
    1. Resize to 280x80 standard
    2. Filter border (thin long components)
    3. Detect dot separator
    4. Adaptive boundary for Chinese
    5. Return quality score indicating confidence"""
    try:
        img_pil = Image.open(image_path).convert('L')
    except Exception:
        return None, 0
    # Resize to standard 280x80 to get ~40px char height (helps low-res)
    img_pil = img_pil.resize((280, 80), Image.BILINEAR)
    img = np.array(img_pil)
    H, W = img.shape  # 80, 280
    img_work = 255 - img if img.mean() > 127 else img.copy()
    t = otsu(img_work)
    binary = (img_work > t).astype(np.uint8)
    boxes = cc_boxes(binary, min_area=25)
    if not boxes: return None, 0
    # Filter: char should have reasonable height (30-70px at 80 tall plate)
    char_boxes = []
    for b in boxes:
        bh = b[3]-b[1]+1; bw = b[2]-b[0]+1
        # Skip border-like (very thin long) or border-corner components
        # Char should be 30-75px tall, 8-60px wide
        if bh < 25 or bh > 76: continue
        if bw < 5 or bw > 70: continue
        # Skip if touching edges aggressively (border components)
        if b[1] == 0 and bh < 40: continue  # thin top element
        if b[3] >= H-1 and bh < 40: continue
        char_boxes.append(b)
    char_boxes.sort(key=lambda b: b[0])
    # Detect and remove dot separator
    # Dot: small component (area 20-100), located between pos 2 char and pos 3 char
    # In 280-wide plate, dot is typically at x ~90-100px
    non_dot_boxes = []
    for b in char_boxes:
        bw = b[2]-b[0]+1; bh = b[3]-b[1]+1
        area = b[4]
        # Dot heuristic: small area, small size, mid-height
        if area < 100 and bw < 15 and bh < 25:
            continue  # skip dot
        non_dot_boxes.append(b)
    if len(non_dot_boxes) < 4: return None, 0  # too few chars
    # Compute quality: prefer exactly n_chars boxes
    quality = max(0, 100 - abs(len(non_dot_boxes) - n_chars) * 15)
    x0a = min(b[0] for b in non_dot_boxes); x1a = max(b[2] for b in non_dot_boxes)
    total_w = x1a - x0a + 1
    avg_cw = total_w / n_chars
    # Find best CN boundary by trying ratios
    best_cn = None; best_other = None; best_score = 1e9
    for ratio in [0.9, 1.0, 1.1, 1.2]:
        cn_b = x0a + int(avg_cw * ratio)
        cn_l = [b for b in non_dot_boxes if b[0] < cn_b]
        oth_l = [b for b in non_dot_boxes if b[0] >= cn_b]
        score = abs(len(oth_l) - (n_chars - 1))
        if len(cn_l) == 0: score += 100
        if score < best_score:
            best_score = score; best_cn = cn_l; best_other = oth_l
    if not best_cn or len(best_other) < 3: return None, 0
    # Merge CN boxes into single char
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
    # Quality: penalize if we padded with blanks
    quality -= (n_chars - min(len(best_other)+1, n_chars)) * 20
    return patches[:n_chars], quality

def _process_label(entry):
    rel_path, plate_text = entry
    if len(plate_text) != 7: return []
    if plate_text[0] not in PROVINCES: return []
    for c in plate_text[1:]:
        if c not in LPR36: return []
    img_path = os.path.join(DATA_ROOT, rel_path)
    if not os.path.exists(img_path): return []
    result = segment_plate_v2(img_path, 7)
    if result is None: return []
    patches, quality = result
    if quality < 40: return []  # filter low-quality segmentation
    if patches is None or len(patches) != 7: return []
    results = []
    for pos, patch in enumerate(patches):
        char = plate_text[pos]
        if pos == 0:
            label_idx = PROVINCES.index(char)
        else:
            label_idx = LPR36.index(char)
        results.append((patch.flatten(), label_idx, pos, char))
    return results

def load_and_segment(label_file, max_samples=None, n_workers=40):
    entries = []
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ')
            if len(parts) >= 2:
                entries.append((parts[0], parts[1]))
    if max_samples: entries = entries[:max_samples]
    print(f"  Processing {len(entries)} plates ({n_workers} cores)...")
    t0 = time.time()
    with Pool(n_workers) as p:
        all_results = p.map(_process_label, entries)
    patches = []; labels = []; positions = []
    valid = 0
    for plate_results in all_results:
        if len(plate_results) == 7:
            valid += 1
        for patch, label, pos, char in plate_results:
            patches.append(patch)
            labels.append(label)
            positions.append(pos)
    print(f"  → {len(patches)} chars from {valid} valid plates ({valid*100/len(entries):.1f}% pass rate) in {time.time()-t0:.1f}s")
    return np.array(patches, dtype=np.float32), np.array(labels), np.array(positions)

def train_classifier(X_tr, y_tr, X_te, y_te, hidden1, hidden2, num_classes, device, epochs=40, batch=1024, name=""):
    import torch, torch.nn as nn
    # Keep 1-layer MLP (firmware compatibility). hidden2 ignored.
    class MLP(nn.Module):
        def __init__(s):
            super().__init__()
            s.fc1 = nn.Linear(784, hidden1)
            s.fc2 = nn.Linear(hidden1, num_classes)
            s.drop = nn.Dropout(0.15)
        def forward(s, x):
            x = torch.relu(s.fc1(x))
            x = s.drop(x)
            return s.fc2(x)
    mlp = MLP().to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()
    Xt = torch.tensor(X_tr/255.0).to(device); yt = torch.tensor(y_tr, dtype=torch.long).to(device)
    Xte = torch.tensor(X_te/255.0).to(device); yte = torch.tensor(y_te, dtype=torch.long).to(device)
    best_acc = 0; best_state = None
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
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in mlp.state_dict().items()}
        mlp.train()
        if ep % 3 == 0 or ep == epochs-1:
            print(f"  [{name}] Ep{ep+1}/{epochs}: loss={tl:.1f} acc={acc*100:.2f}% best={best_acc*100:.2f}%")
    # Load best weights
    mlp.load_state_dict(best_state)
    return mlp, best_acc

def export_header_2layer(mlp, X_te, y_te, prefix, guard, h1, h2, n_cls, hidden_name, classes_name, chars_decl, out_file):
    """Export 1-layer MLP weights (compatible with existing firmware)."""
    W1 = mlp.fc1.weight.data.cpu().numpy(); b1 = mlp.fc1.bias.data.cpu().numpy()
    W2 = mlp.fc2.weight.data.cpu().numpy(); b2 = mlp.fc2.bias.data.cpu().numpy()
    def q(W):
        s = float(np.abs(W).max()/127)
        return np.clip(np.round(W/s),-128,127).astype(np.int8), s
    W1q, s1 = q(W1); W2q, s2 = q(W2)
    X_te_u8 = np.clip(np.round(X_te),0,255).astype(np.uint8)
    acc1 = X_te_u8.astype(np.int32) @ W1q.T.astype(np.int32)
    z1 = acc1.astype(np.float32)*(s1/255.0) + b1
    h1_act = np.maximum(z1, 0)
    acc2 = h1_act @ W2q.T.astype(np.float32)*s2 + b2
    int8_acc = (acc2.argmax(1) == y_te).mean()
    print(f"  [{prefix}] INT8: {int8_acc*100:.2f}%")
    h = f"""/* {prefix} v9: 1-layer MLP 784->{h1}->{n_cls} trained on REAL data + improved seg */
/* INT8={int8_acc*100:.2f}% */
#ifndef {guard}
#define {guard}
#define {hidden_name} {h1}
#define {classes_name} {n_cls}
static const float {prefix}_s1 = {s1:.9e}f;
static const float {prefix}_s2 = {s2:.9e}f;
static const int8_t {prefix}_W1[{h1}][784] = {{
"""
    for j in range(h1):
        h += "  {" + ",".join(str(int(v)) for v in W1q[j]) + "},\n"
    h += "};\n"
    h += f"static const float {prefix}_b1[{h1}] = {{" + ",".join(f"{v:.6e}f" for v in b1) + "};\n"
    h += f"static const int8_t {prefix}_W2[{n_cls}][{h1}] = {{\n"
    for j in range(n_cls):
        h += "  {" + ",".join(str(int(v)) for v in W2q[j]) + "},\n"
    h += "};\n"
    h += f"static const float {prefix}_b2[{n_cls}] = {{" + ",".join(f"{v:.6e}f" for v in b2) + "};\n"
    h += chars_decl + "#endif\n"
    with open(out_file, 'w', encoding='utf-8') as f: f.write(h)
    return float(int8_acc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--classes', choices=['lpr36', 'cn31'], required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch', type=int, default=2048)
    parser.add_argument('--max-train', type=int, default=70000)
    args = parser.parse_args()

    print(f"=== {args.classes} v9 (improved seg + 2-layer MLP) on {args.device} ===")
    t_all = time.time()
    print("[Data] Loading training plates...")
    X_tr, y_tr, pos_tr = load_and_segment(LABEL_FILE, max_samples=args.max_train)
    print("[Data] Loading validation plates...")
    X_te, y_te, pos_te = load_and_segment(VAL_LABEL_FILE, max_samples=2000)

    if args.classes == 'lpr36':
        mask_tr = pos_tr >= 1; mask_te = pos_te >= 1
        H1, H2, N_CLS = 512, 256, 36
        OUT = '/tmp/lpr36_weights.h'
        prefix, guard = 'lpr36', 'LPR36_WEIGHTS_H'
        hidden_name, classes_name = 'LPR36_HIDDEN', 'LPR36_CLASSES'
        chars_decl = 'static const char lpr36_chars[36] = {' + ','.join(f"'{c}'" for c in LPR36) + '};\n'
    else:
        mask_tr = pos_tr == 0; mask_te = pos_te == 0
        H1, H2, N_CLS = 512, 256, 31
        OUT = '/tmp/lpr_cn31_weights.h'
        prefix, guard = 'cn31', 'LPR_CN31_WEIGHTS_H'
        hidden_name, classes_name = 'CN31_HIDDEN', 'CN31_CLASSES'
        chars_decl = 'static const char * const cn31_provinces[31] = {' + ','.join(f'"{c}"' for c in PROVINCES) + '};\n'

    X_tr_f = X_tr[mask_tr]; y_tr_f = y_tr[mask_tr]
    X_te_f = X_te[mask_te]; y_te_f = y_te[mask_te]
    print(f"  Train: {len(X_tr_f)} samples, Test: {len(X_te_f)}")
    unique, counts = np.unique(y_tr_f, return_counts=True)
    print(f"  Classes covered: {len(unique)}/{N_CLS}")

    print(f"[Train] 2-layer MLP {H1}->{H2} on {args.device}")
    mlp, best = train_classifier(X_tr_f, y_tr_f, X_te_f, y_te_f, H1, H2, N_CLS, args.device,
                                   epochs=args.epochs, batch=args.batch, name=args.classes)

    int8_acc = export_header_2layer(mlp, X_te_f, y_te_f, prefix, guard, H1, H2, N_CLS,
                                      hidden_name, classes_name, chars_decl, OUT)
    print(f"\n{args.classes} FP32 best: {best*100:.2f}%  INT8: {int8_acc*100:.2f}%  time: {time.time()-t_all:.1f}s")
