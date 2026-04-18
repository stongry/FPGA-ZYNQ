"""End-to-end CNN for LPR: input 128x32 plate image → 7 char predictions.
No segmentation! CNN learns spatial layout directly.
"""
import numpy as np, os, random, time, argparse
from PIL import Image
from multiprocessing import Pool

PROVINCES = '京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼'
LPR36 = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
DATA_ROOT = '/tmp/clpr/images'
LABEL_FILE = '/tmp/clpr/plate_labels/balanced_base_lpr_3000.txt'
VAL_LABEL_FILE = '/tmp/clpr/plate_labels/balanced_base_lpr_3000_val.txt'

PLATE_H, PLATE_W = 32, 128

def _load_plate(entry):
    """Worker: load & resize one plate, return (img, label_indices or None).
    label_indices: [prov_idx, alnum_idx[0..5]] = 7 ints."""
    rel_path, txt = entry
    if len(txt) != 7 or txt[0] not in PROVINCES: return None
    if any(c not in LPR36 for c in txt[1:]): return None
    path = os.path.join(DATA_ROOT, rel_path)
    if not os.path.exists(path): return None
    try:
        img = Image.open(path).convert('L').resize((PLATE_W, PLATE_H), Image.BILINEAR)
    except: return None
    arr = np.array(img, dtype=np.uint8)
    labels = [PROVINCES.index(txt[0])] + [LPR36.index(c) for c in txt[1:]]
    return arr, labels

def load_data(label_file, max_n=None):
    entries = []
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ')
            if len(parts) >= 2:
                entries.append((parts[0], parts[1]))
    if max_n: entries = entries[:max_n]
    print(f"  Loading {len(entries)} plates...")
    t0 = time.time()
    with Pool(40) as p:
        results = p.map(_load_plate, entries)
    imgs = []; labels = []
    for r in results:
        if r is None: continue
        imgs.append(r[0]); labels.append(r[1])
    imgs_arr = np.array(imgs, dtype=np.float32)
    labels_arr = np.array(labels, dtype=np.int64)
    print(f"  → {len(imgs)} plates loaded in {time.time()-t0:.1f}s, shape {imgs_arr.shape}")
    return imgs_arr, labels_arr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--max-train', type=int, default=87000)
    args = parser.parse_args()

    print(f"=== End-to-end CNN LPR ({args.device}) ===")
    t_all = time.time()
    X_tr, Y_tr = load_data(LABEL_FILE, max_n=args.max_train)
    X_te, Y_te = load_data(VAL_LABEL_FILE, max_n=2000)
    print(f"  Train: {len(X_tr)}  Test: {len(X_te)}")

    import torch, torch.nn as nn, torch.nn.functional as F
    class PlateCNN(nn.Module):
        def __init__(s):
            super().__init__()
            s.conv1 = nn.Conv2d(1, 32, 3, padding=1); s.bn1 = nn.BatchNorm2d(32)
            s.conv2 = nn.Conv2d(32, 64, 3, padding=1); s.bn2 = nn.BatchNorm2d(64)
            s.conv3 = nn.Conv2d(64, 128, 3, padding=1); s.bn3 = nn.BatchNorm2d(128)
            s.conv4 = nn.Conv2d(128, 256, 3, padding=1); s.bn4 = nn.BatchNorm2d(256)
            # After 4x Pool2x2: 32→2, 128→8
            s.fc = nn.Linear(256 * 2 * 8, 512)
            s.drop = nn.Dropout(0.3)
            s.head_cn = nn.Linear(512, 31)  # province
            s.heads_al = nn.ModuleList([nn.Linear(512, 36) for _ in range(6)])  # 6 alnum heads
        def forward(s, x):
            x = F.max_pool2d(F.relu(s.bn1(s.conv1(x))), 2)  # 16x64
            x = F.max_pool2d(F.relu(s.bn2(s.conv2(x))), 2)  # 8x32
            x = F.max_pool2d(F.relu(s.bn3(s.conv3(x))), 2)  # 4x16
            x = F.max_pool2d(F.relu(s.bn4(s.conv4(x))), 2)  # 2x8
            x = x.flatten(1)
            x = s.drop(F.relu(s.fc(x)))
            out_cn = s.head_cn(x)
            outs_al = [h(x) for h in s.heads_al]
            return out_cn, outs_al

    net = PlateCNN().to(args.device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"  Model params: {n_params/1e6:.2f}M")
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()

    Xt = torch.tensor(X_tr/255.0).unsqueeze(1).to(args.device)  # (N,1,32,128)
    Yt = torch.tensor(Y_tr, dtype=torch.long).to(args.device)   # (N,7)
    Xte = torch.tensor(X_te/255.0).unsqueeze(1).to(args.device)
    Yte = torch.tensor(Y_te, dtype=torch.long).to(args.device)

    best_plate_acc = 0; best_state = None
    for ep in range(args.epochs):
        perm = torch.randperm(len(Xt)); tl = 0
        net.train()
        for i in range(0, len(Xt), args.batch):
            idx = perm[i:i+args.batch]
            x = Xt[idx]; y = Yt[idx]
            # Slight aug: random brightness shift
            x = x + torch.randn_like(x) * 0.05
            x = torch.clamp(x, 0, 1)
            opt.zero_grad()
            out_cn, outs_al = net(x)
            loss = loss_fn(out_cn, y[:, 0])
            for j, o in enumerate(outs_al):
                loss = loss + loss_fn(o, y[:, j+1])
            loss.backward(); opt.step(); tl += loss.item()
        sched.step()
        # Eval
        net.eval()
        with torch.no_grad():
            out_cn, outs_al = net(Xte)
            pred_cn = out_cn.argmax(1)
            preds_al = [o.argmax(1) for o in outs_al]
            # Char level
            char_correct = (pred_cn == Yte[:, 0]).sum().item()
            for j, p in enumerate(preds_al):
                char_correct += (p == Yte[:, j+1]).sum().item()
            char_acc = char_correct / (len(Yte) * 7)
            # Plate level (all 7 correct)
            all_correct = (pred_cn == Yte[:, 0])
            for j, p in enumerate(preds_al):
                all_correct = all_correct & (p == Yte[:, j+1])
            plate_acc = all_correct.float().mean().item()
        if plate_acc > best_plate_acc:
            best_plate_acc = plate_acc
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        if ep % 2 == 0 or ep == args.epochs - 1:
            print(f"  Ep{ep+1}/{args.epochs}: loss={tl:.1f}  char={char_acc*100:.2f}%  plate={plate_acc*100:.2f}%  best_plate={best_plate_acc*100:.2f}%")

    print(f"\n=== BEST: plate={best_plate_acc*100:.2f}% ===")
    net.load_state_dict(best_state)
    net.eval()

    # Save model state
    torch.save({'state_dict': best_state, 'plate_acc': best_plate_acc},
               '/tmp/plate_cnn_e2e.pt')

    # Test on real plates (our 6 test samples)
    print("\n=== Testing on real test plates ===")
    import re
    test_dir = '/tmp/real_plates'
    if os.path.exists(test_dir):
        for fn in sorted(os.listdir(test_dir)):
            if not fn.endswith(('.jpg','.png')): continue
            # Get GT
            m = re.match(r'_\d+_([^.]+)\.\w+', fn)
            if m: gt = m.group(1)
            else:
                base = fn.split('.')[0]
                if '_convert' in base: gt = base.split('_')[0]
                else:
                    p = base.rsplit('_', 1)
                    gt = p[0]+p[1] if len(p)==2 and p[1].isdigit() else base
            if len(gt) != 7: continue
            path = os.path.join(test_dir, fn)
            img = Image.open(path).convert('L').resize((PLATE_W, PLATE_H), Image.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0
            x = torch.tensor(arr).unsqueeze(0).unsqueeze(0).to(args.device)
            with torch.no_grad():
                out_cn, outs_al = net(x)
                pred_cn = out_cn.argmax(1).item()
                preds_al = [o.argmax(1).item() for o in outs_al]
            pred = PROVINCES[pred_cn] + ''.join(LPR36[p] for p in preds_al)
            status = '✓' if pred == gt else '✗'
            print(f"  {fn:35s} '{pred}' vs '{gt}' {status}")

    print(f"\nTotal time: {time.time()-t_all:.1f}s")
