"""Full evaluation on val.txt (37K plates) with FP32 and INT8 models.
Also estimate province/position-wise accuracy breakdown."""
import numpy as np, os, time, argparse
from PIL import Image
from multiprocessing import Pool
import torch, torch.nn as nn, torch.nn.functional as F

PROVINCES = '京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼'
LPR36 = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
DATA_ROOT = '/tmp/clpr/images'
PLATE_H, PLATE_W = 32, 128

def _load(entry):
    rel_path, txt = entry
    if len(txt) != 7 or txt[0] not in PROVINCES: return None
    if any(c not in LPR36 for c in txt[1:]): return None
    path = os.path.join(DATA_ROOT, rel_path)
    if not os.path.exists(path): return None
    try:
        img = Image.open(path).convert('L').resize((PLATE_W, PLATE_H), Image.BILINEAR)
    except: return None
    return np.array(img, dtype=np.uint8), [PROVINCES.index(txt[0])] + [LPR36.index(c) for c in txt[1:]]

def load(label_file, max_n=None):
    entries = []
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ')
            if len(parts) >= 2:
                entries.append((parts[0], parts[1]))
    if max_n: entries = entries[:max_n]
    print(f"Loading {len(entries)}...")
    t0 = time.time()
    with Pool(40) as p:
        results = p.map(_load, entries)
    imgs = []; labels = []
    for r in results:
        if r is None: continue
        imgs.append(r[0]); labels.append(r[1])
    print(f"  → {len(imgs)} in {time.time()-t0:.1f}s")
    return np.array(imgs, dtype=np.float32), np.array(labels, dtype=np.int64)

class PlateCNN(nn.Module):
    def __init__(s):
        super().__init__()
        s.conv1 = nn.Conv2d(1, 32, 3, padding=1); s.bn1 = nn.BatchNorm2d(32)
        s.conv2 = nn.Conv2d(32, 64, 3, padding=1); s.bn2 = nn.BatchNorm2d(64)
        s.conv3 = nn.Conv2d(64, 128, 3, padding=1); s.bn3 = nn.BatchNorm2d(128)
        s.conv4 = nn.Conv2d(128, 256, 3, padding=1); s.bn4 = nn.BatchNorm2d(256)
        s.fc = nn.Linear(256*2*8, 512)
        s.drop = nn.Dropout(0.3)
        s.head_cn = nn.Linear(512, 31)
        s.heads_al = nn.ModuleList([nn.Linear(512, 36) for _ in range(6)])
    def forward(s, x):
        x = F.max_pool2d(F.relu(s.bn1(s.conv1(x))), 2)
        x = F.max_pool2d(F.relu(s.bn2(s.conv2(x))), 2)
        x = F.max_pool2d(F.relu(s.bn3(s.conv3(x))), 2)
        x = F.max_pool2d(F.relu(s.bn4(s.conv4(x))), 2)
        x = x.flatten(1)
        x = s.drop(F.relu(s.fc(x)))
        return s.head_cn(x), [h(x) for h in s.heads_al]

device = 'cuda:0'
ckpt = torch.load('/tmp/plate_cnn_e2e.pt', map_location=device, weights_only=True)
net = PlateCNN().to(device)
net.load_state_dict(ckpt['state_dict'])
net.eval()
print(f"Model loaded. Train best plate: {ckpt['plate_acc']*100:.2f}%")

# Full val.txt
print("\n=== Full val.txt (37K) ===")
X, Y = load('/tmp/clpr/plate_labels/val.txt', max_n=None)
print(f"Loaded {len(X)} plates")

batch = 256
preds_cn = []; preds_al = [[] for _ in range(6)]
with torch.no_grad():
    for i in range(0, len(X), batch):
        x = torch.tensor(X[i:i+batch]/255.0).unsqueeze(1).to(device)
        out_cn, outs_al = net(x)
        preds_cn.append(out_cn.argmax(1).cpu().numpy())
        for j, o in enumerate(outs_al):
            preds_al[j].append(o.argmax(1).cpu().numpy())

preds_cn = np.concatenate(preds_cn)
preds_al = [np.concatenate(p) for p in preds_al]

# Per-position accuracy
pos0 = (preds_cn == Y[:, 0]).mean()
pos_al = [(preds_al[j] == Y[:, j+1]).mean() for j in range(6)]

# Plate-level: all 7 correct
all_ok = (preds_cn == Y[:, 0])
for j in range(6):
    all_ok = all_ok & (preds_al[j] == Y[:, j+1])
plate_acc = all_ok.mean()

# Char-level
char_correct = (preds_cn == Y[:, 0]).sum()
for j in range(6):
    char_correct += (preds_al[j] == Y[:, j+1]).sum()
char_acc = char_correct / (len(Y) * 7)

print(f"\n{'='*60}")
print(f"Plate accuracy: {plate_acc*100:.2f}% ({int(all_ok.sum())}/{len(Y)})")
print(f"Char accuracy:  {char_acc*100:.2f}% ({int(char_correct)}/{len(Y)*7})")
print(f"\nPer-position accuracy:")
print(f"  Pos 0 (Chinese): {pos0*100:.2f}%")
for j in range(6):
    print(f"  Pos {j+1} (alnum):  {pos_al[j]*100:.2f}%")

# Per-province accuracy on position 0
print(f"\nPer-province accuracy (position 0):")
per_prov = {}
for p in range(31):
    mask = Y[:, 0] == p
    if mask.sum() > 0:
        per_prov[PROVINCES[p]] = float((preds_cn[mask] == p).mean())
for ch, acc in sorted(per_prov.items(), key=lambda x: x[1]):
    n = int((Y[:, 0] == PROVINCES.index(ch)).sum())
    print(f"  {ch}: {acc*100:.1f}% (n={n})")
