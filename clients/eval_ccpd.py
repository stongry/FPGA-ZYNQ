"""Evaluate v3 CNN on CCPD 30K subset (完全独立数据集).
CCPD filename: [area]-[tilt]-[bbox]-[vertices]-[plate_indices]-[bright]-[blur]_...jpg
  bbox: x1,y1_x2,y2 (plate bounding box in full car image)
  plate_indices: prov_idx_letter_idx_...5 alnum indices
"""
import numpy as np, os, time
from PIL import Image
from multiprocessing import Pool
import torch, torch.nn as nn, torch.nn.functional as F

# Our model's output mapping
OUR_PROVINCES = '京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼'
OUR_LPR36 = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

# CCPD's character mapping (for parsing ground truth indices)
CCPD_PROVINCES = ['皖','沪','津','渝','冀','晋','蒙','辽','吉','黑','苏','浙','京','闽','赣','鲁',
                   '豫','鄂','湘','粤','桂','琼','川','贵','云','藏','陕','甘','青','宁','新','警','学']
CCPD_ALNUM = ['A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V',
               'W','X','Y','Z','0','1','2','3','4','5','6','7','8','9']

def parse_ccpd_filename(fn):
    """Extract bbox + plate text from CCPD filename."""
    base = os.path.basename(fn)
    # Strip suffix after last underscore (ccpd_base_xxxxx.jpg etc.)
    # Core format: area-tilt-bbox-vertices-plate_indices-bright-blur
    # Example: 0189655172414-91_85-233,591_457,681-463,678_245,663_232,585_450,600-0_3_15_27_28_29_30-134-129_ccpd_base_005128.jpg
    # Split by '-' first
    core = base.split('_ccpd')[0]  # remove _ccpd_xxx.jpg suffix
    parts = core.split('-')
    if len(parts) < 5: return None
    try:
        bbox_str = parts[2]  # "x1,y1_x2,y2"
        x1y1, x2y2 = bbox_str.split('_')
        x1, y1 = map(int, x1y1.split(','))
        x2, y2 = map(int, x2y2.split(','))
        plate_idx = parts[4].split('_')  # ["0", "3", "15", "27", ...]
        if len(plate_idx) != 7: return None
        # Parse ground truth
        prov_idx = int(plate_idx[0])
        if prov_idx >= len(CCPD_PROVINCES): return None
        prov_ch = CCPD_PROVINCES[prov_idx]
        # Skip special chars 警(31)/学(32) - not in our model
        if prov_ch in ['警', '学']: return None
        alnum_idxs = [int(x) for x in plate_idx[1:]]
        if any(i >= len(CCPD_ALNUM) for i in alnum_idxs): return None
        plate_chars = [prov_ch] + [CCPD_ALNUM[i] for i in alnum_idxs]
        # Check all alnum chars are in our LPR36 (should be since CCPD excludes I/O)
        for c in plate_chars[1:]:
            if c not in OUR_LPR36: return None
        return (x1, y1, x2, y2), ''.join(plate_chars)
    except Exception:
        return None

def load_and_crop(task):
    """Worker: load image, crop plate using bbox, resize."""
    full_path, info = task
    (x1, y1, x2, y2), plate_text = info
    try:
        img = Image.open(full_path).convert('L')
        # Crop plate
        crop = img.crop((x1, y1, x2, y2))
        crop = crop.resize((128, 32), Image.BILINEAR)
        arr = np.array(crop, dtype=np.uint8)
        # Labels: prov_idx (0-30) + 6 alnum idx (0-35)
        labels = [OUR_PROVINCES.index(plate_text[0])] + [OUR_LPR36.index(c) for c in plate_text[1:]]
        return arr, labels, plate_text
    except Exception as e:
        return None

# Parse all filenames first (fast)
print("Parsing CCPD filenames...")
root = '/tmp/ccpd30k'
files = [f for f in os.listdir(root) if f.endswith('.jpg')]
print(f"  {len(files)} CCPD images found")

tasks = []; skip = 0
for fn in files:
    info = parse_ccpd_filename(fn)
    if info is None: skip += 1; continue
    tasks.append((os.path.join(root, fn), info))
print(f"  Valid (parseable) plates: {len(tasks)}, skipped: {skip}")

print(f"\nLoading + cropping {len(tasks)} plates in parallel (40 cores)...")
t0 = time.time()
with Pool(40) as p:
    results = p.map(load_and_crop, tasks)
imgs = []; labels = []; texts = []
for r in results:
    if r is None: continue
    imgs.append(r[0]); labels.append(r[1]); texts.append(r[2])
X = np.array(imgs, dtype=np.float32)
Y = np.array(labels, dtype=np.int64)
print(f"  → {len(X)} plates ready in {time.time()-t0:.1f}s")

# Load model
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
net = PlateCNN().to(device); net.load_state_dict(ckpt['state_dict']); net.eval()
print(f"\nModel loaded (train val best plate: {ckpt['plate_acc']*100:.2f}%)")

# Inference
print("\n=== Running inference on CCPD 30K ===")
t0 = time.time()
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
print(f"  Inference time: {time.time()-t0:.1f}s")

# Stats
pos0 = (preds_cn == Y[:, 0]).mean()
pos_al = [(preds_al[j] == Y[:, j+1]).mean() for j in range(6)]
all_ok = (preds_cn == Y[:, 0])
for j in range(6):
    all_ok = all_ok & (preds_al[j] == Y[:, j+1])
plate_acc = all_ok.mean()
char_correct = (preds_cn == Y[:, 0]).sum()
for j in range(6):
    char_correct += (preds_al[j] == Y[:, j+1]).sum()
char_acc = char_correct / (len(Y) * 7)

print(f"\n{'='*60}")
print(f"CCPD 30K (INDEPENDENT dataset) evaluation:")
print(f"  Plate accuracy: {plate_acc*100:.2f}% ({int(all_ok.sum())}/{len(Y)})")
print(f"  Char accuracy:  {char_acc*100:.2f}% ({int(char_correct)}/{len(Y)*7})")
print(f"\nPer-position accuracy:")
print(f"  Pos 0 (Chinese): {pos0*100:.2f}%")
for j in range(6):
    print(f"  Pos {j+1}:          {pos_al[j]*100:.2f}%")

# Show some failure examples
print(f"\nRandom failure examples:")
wrong = np.where(~all_ok)[0]
for idx in wrong[:10]:
    pred = OUR_PROVINCES[preds_cn[idx]] + ''.join(OUR_LPR36[preds_al[j][idx]] for j in range(6))
    gt = texts[idx]
    print(f"  {gt} → {pred}")
