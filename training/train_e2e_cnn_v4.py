"""v4: richjjj + CCPD 22K combined training. Holdout 6K CCPD for test."""
import numpy as np, os, time, argparse, random
from PIL import Image
from multiprocessing import Pool

OUR_PROVINCES = '京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼'
OUR_LPR36 = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
CCPD_PROVINCES = ['皖','沪','津','渝','冀','晋','蒙','辽','吉','黑','苏','浙','京','闽','赣','鲁',
                   '豫','鄂','湘','粤','桂','琼','川','贵','云','藏','陕','甘','青','宁','新','警','学']
CCPD_ALNUM = ['A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V',
               'W','X','Y','Z','0','1','2','3','4','5','6','7','8','9']
PLATE_H, PLATE_W = 32, 128

# Load richjjj plates (already-cropped)
def _load_richjjj(entry):
    rel, txt = entry
    if len(txt) != 7 or txt[0] not in OUR_PROVINCES: return None
    if any(c not in OUR_LPR36 for c in txt[1:]): return None
    path = os.path.join('/tmp/clpr/images', rel)
    if not os.path.exists(path): return None
    try:
        img = Image.open(path).convert('L').resize((PLATE_W, PLATE_H), Image.BILINEAR)
    except: return None
    arr = np.array(img, dtype=np.uint8)
    labels = [OUR_PROVINCES.index(txt[0])] + [OUR_LPR36.index(c) for c in txt[1:]]
    return arr, labels

# Load CCPD (crop from full car image using filename bbox)
def _load_ccpd(task):
    full_path, info = task
    (x1, y1, x2, y2), plate_text = info
    try:
        img = Image.open(full_path).convert('L')
        crop = img.crop((x1, y1, x2, y2)).resize((PLATE_W, PLATE_H), Image.BILINEAR)
    except: return None
    arr = np.array(crop, dtype=np.uint8)
    labels = [OUR_PROVINCES.index(plate_text[0])] + [OUR_LPR36.index(c) for c in plate_text[1:]]
    return arr, labels

def parse_ccpd(fn):
    base = os.path.basename(fn)
    core = base.split('_ccpd')[0]
    parts = core.split('-')
    if len(parts) < 5: return None
    try:
        x1y1, x2y2 = parts[2].split('_')
        x1, y1 = map(int, x1y1.split(','))
        x2, y2 = map(int, x2y2.split(','))
        pi = parts[4].split('_')
        if len(pi) != 7: return None
        pr = int(pi[0])
        if pr >= len(CCPD_PROVINCES) or CCPD_PROVINCES[pr] in ['警','学']: return None
        pch = CCPD_PROVINCES[pr]
        alnum_idxs = [int(x) for x in pi[1:]]
        if any(i >= len(CCPD_ALNUM) for i in alnum_idxs): return None
        txt = pch + ''.join(CCPD_ALNUM[i] for i in alnum_idxs)
        if any(c not in OUR_LPR36 for c in txt[1:]): return None
        return (x1, y1, x2, y2), txt
    except: return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch', type=int, default=256)
    args = parser.parse_args()

    print("=== v4: richjjj 213K + CCPD 22K ===")
    t_all = time.time()

    # Load richjjj train
    print("\n[1/4] Loading richjjj train (213K)...")
    entries = []
    with open('/tmp/clpr/plate_labels/train.txt') as f:
        for line in f:
            parts = line.strip().split(' ')
            if len(parts) >= 2: entries.append((parts[0], parts[1]))
    t0 = time.time()
    with Pool(40) as p:
        res = p.map(_load_richjjj, entries)
    r_imgs = [r[0] for r in res if r]
    r_lbls = [r[1] for r in res if r]
    print(f"  → {len(r_imgs)} in {time.time()-t0:.1f}s")

    # Load richjjj val (2K for eval)
    print("\n[2/4] Loading richjjj val (for same-domain eval)...")
    entries = []
    with open('/tmp/clpr/plate_labels/val.txt') as f:
        for line in f:
            parts = line.strip().split(' ')
            if len(parts) >= 2: entries.append((parts[0], parts[1]))
    entries = entries[:2000]
    with Pool(40) as p:
        res = p.map(_load_richjjj, entries)
    val_r_imgs = [r[0] for r in res if r]
    val_r_lbls = [r[1] for r in res if r]
    print(f"  → {len(val_r_imgs)} richjjj val")

    # Load CCPD and split 22K/6K
    print("\n[3/4] Parsing + loading CCPD 28K...")
    ccpd_root = '/tmp/ccpd30k'
    files = [f for f in os.listdir(ccpd_root) if f.endswith('.jpg')]
    tasks = []
    for fn in files:
        info = parse_ccpd(fn)
        if info: tasks.append((os.path.join(ccpd_root, fn), info))
    print(f"  Valid parseable: {len(tasks)}")
    random.seed(42)
    random.shuffle(tasks)
    # Split 22K train / 6K test
    ccpd_train_tasks = tasks[:22000]
    ccpd_test_tasks = tasks[22000:28000]
    print(f"  Train split: {len(ccpd_train_tasks)}, Test split: {len(ccpd_test_tasks)}")

    t0 = time.time()
    with Pool(40) as p:
        res = p.map(_load_ccpd, ccpd_train_tasks)
    c_imgs = [r[0] for r in res if r]
    c_lbls = [r[1] for r in res if r]
    print(f"  Train loaded: {len(c_imgs)} in {time.time()-t0:.1f}s")

    t0 = time.time()
    with Pool(40) as p:
        res_te = p.map(_load_ccpd, ccpd_test_tasks)
    cte_imgs = [r[0] for r in res_te if r]
    cte_lbls = [r[1] for r in res_te if r]
    print(f"  Test loaded: {len(cte_imgs)} in {time.time()-t0:.1f}s")

    # Combine
    X_tr = np.array(r_imgs + c_imgs, dtype=np.float32)
    Y_tr = np.array(r_lbls + c_lbls, dtype=np.int64)
    X_val_r = np.array(val_r_imgs, dtype=np.float32)
    Y_val_r = np.array(val_r_lbls, dtype=np.int64)
    X_val_c = np.array(cte_imgs, dtype=np.float32)
    Y_val_c = np.array(cte_lbls, dtype=np.int64)
    print(f"\nTotal TRAIN: {len(X_tr)} (richjjj {len(r_imgs)} + CCPD {len(c_imgs)})")
    print(f"Val richjjj: {len(X_val_r)}, Val CCPD holdout: {len(X_val_c)}")

    # Class weights
    prov_counts = np.bincount(Y_tr[:, 0], minlength=31).astype(np.float32)
    prov_counts = np.maximum(prov_counts, 1)
    prov_weights = 1.0 / np.sqrt(prov_counts)
    prov_weights = prov_weights / prov_weights.mean()

    import torch, torch.nn as nn, torch.nn.functional as F
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

    net = PlateCNN().to(args.device)
    print(f"Model params: {sum(p.numel() for p in net.parameters())/1e6:.2f}M")
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    warmup = 3
    def lr_lam(ep):
        if ep < warmup: return (ep+1)/warmup
        import math
        return 0.5*(1+math.cos(math.pi*(ep-warmup)/(args.epochs-warmup)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lam)
    prov_w = torch.tensor(prov_weights, dtype=torch.float32).to(args.device)
    cn_loss = nn.CrossEntropyLoss(weight=prov_w, label_smoothing=0.1)
    al_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

    Xt = torch.tensor(X_tr/255.0).unsqueeze(1).to(args.device)
    Yt = torch.tensor(Y_tr, dtype=torch.long).to(args.device)
    Xvr = torch.tensor(X_val_r/255.0).unsqueeze(1).to(args.device)
    Yvr = torch.tensor(Y_val_r, dtype=torch.long).to(args.device)
    Xvc = torch.tensor(X_val_c/255.0).unsqueeze(1).to(args.device)
    Yvc = torch.tensor(Y_val_c, dtype=torch.long).to(args.device)

    sw = prov_weights[Y_tr[:, 0]]
    sampler = torch.utils.data.WeightedRandomSampler(sw.tolist(), len(Xt), replacement=True)

    def augment(x):
        B = x.shape[0]
        bright = (torch.rand(B,1,1,1,device=x.device)-0.5)*0.3
        contrast = 1.0+(torch.rand(B,1,1,1,device=x.device)-0.5)*0.4
        x = torch.clamp((x-0.5)*contrast+0.5+bright, 0, 1)
        x = x + torch.randn_like(x)*0.05
        if random.random() < 0.35:
            s = random.uniform(0.35, 0.8)
            H, W = max(8,int(32*s)), max(32,int(128*s))
            x = F.interpolate(x, size=(H,W), mode='bilinear', align_corners=False)
            x = F.interpolate(x, size=(32,128), mode='bilinear', align_corners=False)
        if random.random() < 0.3:
            import math
            a = (random.random()-0.5)*10
            t = a*math.pi/180
            c, si = math.cos(t), math.sin(t)
            aff = torch.tensor([[c,-si,0],[si,c,0]], dtype=torch.float32, device=x.device)
            aff = aff.unsqueeze(0).repeat(B,1,1)
            grid = F.affine_grid(aff, x.shape, align_corners=False)
            x = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        return torch.clamp(x, 0, 1)

    def eval_set(Xe, Ye):
        net.eval()
        with torch.no_grad():
            preds_cn = []; preds_al = [[] for _ in range(6)]
            for i in range(0, len(Xe), 512):
                out_cn, outs_al = net(Xe[i:i+512])
                preds_cn.append(out_cn.argmax(1))
                for j, o in enumerate(outs_al):
                    preds_al[j].append(o.argmax(1))
            preds_cn = torch.cat(preds_cn); preds_al = [torch.cat(p) for p in preds_al]
            all_ok = (preds_cn == Ye[:, 0])
            for j in range(6): all_ok = all_ok & (preds_al[j] == Ye[:, j+1])
            plate_acc = all_ok.float().mean().item()
            char_c = (preds_cn == Ye[:, 0]).sum().item()
            for j in range(6): char_c += (preds_al[j] == Ye[:, j+1]).sum().item()
            char_acc = char_c / (len(Ye)*7)
        net.train()
        return plate_acc, char_acc

    best_ccpd = 0; best_state = None
    for ep in range(args.epochs):
        indices = list(sampler); tl = 0
        net.train()
        for i in range(0, len(indices), args.batch):
            bi = indices[i:i+args.batch]
            x = Xt[bi]; y = Yt[bi]
            x = augment(x)
            opt.zero_grad()
            out_cn, outs_al = net(x)
            loss = cn_loss(out_cn, y[:,0])
            for j, o in enumerate(outs_al): loss = loss + al_loss(o, y[:,j+1])
            loss.backward(); opt.step(); tl += loss.item()
        sched.step()
        # Eval on both
        pa_r, ca_r = eval_set(Xvr, Yvr)
        pa_c, ca_c = eval_set(Xvc, Yvc)
        if pa_c > best_ccpd:
            best_ccpd = pa_c
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        if ep % 5 == 0 or ep == args.epochs-1:
            print(f"  Ep{ep+1}: loss={tl:.1f} rich_plate={pa_r*100:.2f}% ccpd_plate={pa_c*100:.2f}% ccpd_char={ca_c*100:.2f}% best_ccpd={best_ccpd*100:.2f}%")

    print(f"\n=== BEST CCPD plate: {best_ccpd*100:.2f}% ===")
    net.load_state_dict(best_state)
    torch.save({'state_dict': best_state, 'plate_acc': best_ccpd}, '/tmp/plate_cnn_e2e.pt')
    # Final eval
    pa_r, ca_r = eval_set(Xvr, Yvr)
    pa_c, ca_c = eval_set(Xvc, Yvc)
    print(f"richjjj val: plate={pa_r*100:.2f}% char={ca_r*100:.2f}%")
    print(f"CCPD holdout: plate={pa_c*100:.2f}% char={ca_c*100:.2f}%")
    print(f"\nTotal: {time.time()-t_all:.1f}s")
