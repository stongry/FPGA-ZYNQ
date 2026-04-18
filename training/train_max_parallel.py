"""Max-utilization training template: 40-core data gen + dual-GPU + large batch.

Uses:
- multiprocessing.Pool(40): parallel image rendering (~10x faster data gen)
- torch.cuda.device('cuda:0'): RTX 5090 (32GB) for main model
- torch.cuda.device('cuda:1'): Quadro RTX 5000 (16GB) for second model in parallel
- Large batch 1024 to saturate GPU memory bandwidth
"""
import numpy as np, os, json, random, argparse
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from multiprocessing import Pool
import functools, time, sys

def _init_worker(seed):
    """Per-worker init: seed random."""
    random.seed(seed + os.getpid())
    np.random.seed((seed + os.getpid()) % (2**32))

def _render_batch(args):
    """Worker function: render N samples of class c_idx.
    args = (c_idx, ch, n_samples, fonts, segment_style)"""
    c_idx, ch, n_samples, fonts, segment_style = args
    results = []
    for _ in range(n_samples):
        if segment_style:
            font_size = random.randint(60, 80)
            font = ImageFont.truetype(random.choice(fonts), font_size)
            big = Image.new('L', (120, 120), 0)
            draw = ImageDraw.Draw(big)
            bbox = draw.textbbox((0, 0), ch, font=font)
            tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
            x = (120-tw)//2 - bbox[0]; y = (120-th)//2 - bbox[1]
            draw.text((x, y), ch, fill=255, font=font)
            if random.random() < 0.3:
                big = big.rotate(random.uniform(-6, 6), resample=Image.BILINEAR, fillcolor=0)
            arr = np.array(big)
            if random.random() < 0.25:
                arr = np.array(Image.fromarray(arr).filter(ImageFilter.GaussianBlur(random.uniform(0.5, 1.0))))
            arr = np.clip(arr.astype(np.float32) + np.random.normal(0, 5, arr.shape), 0, 255).astype(np.uint8)
            binary = arr > max(40, int(arr.mean()*0.5))
            ca = np.where(binary.sum(0) > 0)[0]; ra = np.where(binary.sum(1) > 0)[0]
            if len(ca)==0 or len(ra)==0:
                out = np.zeros((28,28), dtype=np.uint8)
            else:
                x0, x1 = ca[0], ca[-1]; y0, y1 = ra[0], ra[-1]
                patch = np.where(binary[y0:y1+1, x0:x1+1], arr[y0:y1+1, x0:x1+1], 0).astype(np.uint8)
                h, w = patch.shape
                if h >= w: nh, nw = 22, max(1, int(w*22/h))
                else: nw, nh = 22, max(1, int(h*22/w))
                img = Image.fromarray(patch).resize((nw, nh), Image.BILINEAR)
                out = np.zeros((28, 28), dtype=np.uint8)
                y0c = (28-nh)//2; x0c = (28-nw)//2
                out[y0c:y0c+nh, x0c:x0c+nw] = np.array(img)
                if random.random() < 0.15:
                    out = np.clip(out.astype(np.float32) + np.random.normal(0, 4, out.shape), 0, 255).astype(np.uint8)
        results.append((out.flatten(), c_idx))
    return results

def generate_dataset(chars, n_per_class, fonts, n_workers=40, segment_style=True):
    """Parallel dataset generation across 40 cores."""
    # Split work: each (class, chunk) pair is one task
    chunk_size = max(50, n_per_class // n_workers)
    tasks = []
    for c_idx, ch in enumerate(chars):
        remaining = n_per_class
        while remaining > 0:
            take = min(chunk_size, remaining)
            tasks.append((c_idx, ch, take, fonts, segment_style))
            remaining -= take
    print(f"  [DataGen] {len(tasks)} tasks across {n_workers} workers...")
    t0 = time.time()
    with Pool(n_workers, initializer=_init_worker, initargs=(42,)) as pool:
        all_results = pool.map(_render_batch, tasks)
    X = np.concatenate([np.array([r[0] for r in batch]) for batch in all_results])
    y = np.concatenate([np.array([r[1] for r in batch]) for batch in all_results])
    print(f"  [DataGen] Generated {len(X)} samples in {time.time()-t0:.1f}s")
    return X.astype(np.float32), y

def train_model(X_tr, y_tr, X_te, y_te, hidden, num_classes, epochs, batch, device, lr=1e-3):
    import torch, torch.nn as nn
    class MLP(nn.Module):
        def __init__(s):
            super().__init__()
            s.fc1 = nn.Linear(784, hidden); s.fc2 = nn.Linear(hidden, num_classes)
        def forward(s, x): return s.fc2(torch.relu(s.fc1(x)))

    mlp = MLP().to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    Xt = torch.tensor(X_tr/255.0).to(device); yt = torch.tensor(y_tr, dtype=torch.long).to(device)
    Xte_t = torch.tensor(X_te/255.0).to(device); yte_t = torch.tensor(y_te, dtype=torch.long).to(device)

    for ep in range(epochs):
        perm = torch.randperm(len(Xt)); tl = 0
        for i in range(0, len(Xt), batch):
            idx = perm[i:i+batch]
            opt.zero_grad()
            loss = loss_fn(mlp(Xt[idx]), yt[idx])
            loss.backward(); opt.step(); tl += loss.item()
        mlp.eval()
        with torch.no_grad():
            acc = (mlp(Xte_t).argmax(1) == yte_t).float().mean().item()
        mlp.train()
        print(f"  Ep{ep+1}: loss={tl:.1f}  acc={acc*100:.2f}%")
    return mlp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--classes', choices=['lpr36', 'cn31'], required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch', type=int, default=1024)
    parser.add_argument('--train-per-class', type=int, default=3000)
    parser.add_argument('--test-per-class', type=int, default=300)
    args = parser.parse_args()

    CHARS_LPR36 = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    CHARS_CN31 = '京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼'
    FONTS_EN = [f for f in ['/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
                             '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                             '/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf',
                             '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf'] if os.path.exists(f)]
    FONTS_CN = [f for f in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc',
                             '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
                             '/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc',
                             '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc'] if os.path.exists(f)]

    if args.classes == 'lpr36':
        CHARS, FONTS, N_CLS, HIDDEN = CHARS_LPR36, FONTS_EN, 36, 128
    else:
        CHARS, FONTS, N_CLS, HIDDEN = CHARS_CN31, FONTS_CN, 31, 128

    print(f"=== Training {args.classes} on {args.device} ===")
    print(f"  Hidden: {HIDDEN}, Classes: {N_CLS}, Epochs: {args.epochs}, Batch: {args.batch}")

    t_all = time.time()
    print(f"[DataGen] Training set: {args.train_per_class} × {N_CLS} = {args.train_per_class*N_CLS}")
    X_tr, y_tr = generate_dataset(CHARS, args.train_per_class, FONTS, 40, segment_style=True)
    print(f"[DataGen] Test set: {args.test_per_class} × {N_CLS} = {args.test_per_class*N_CLS}")
    X_te, y_te = generate_dataset(CHARS, args.test_per_class, FONTS, 40, segment_style=True)

    print(f"[Train] Starting on {args.device}...")
    mlp = train_model(X_tr, y_tr, X_te, y_te, HIDDEN, N_CLS, args.epochs, args.batch, args.device)

    # INT8 export
    import torch
    W1 = mlp.fc1.weight.data.cpu().numpy(); b1 = mlp.fc1.bias.data.cpu().numpy()
    W2 = mlp.fc2.weight.data.cpu().numpy(); b2 = mlp.fc2.bias.data.cpu().numpy()
    def q(W):
        s = float(np.abs(W).max()/127)
        return np.clip(np.round(W/s),-128,127).astype(np.int8), s
    W1q, s1 = q(W1); W2q, s2 = q(W2)
    X_te_u8 = np.clip(np.round(X_te), 0, 255).astype(np.uint8)
    acc1 = X_te_u8.astype(np.int32) @ W1q.T.astype(np.int32)
    z1 = acc1.astype(np.float32)*(s1/255.0) + b1
    h1 = np.maximum(z1, 0)
    acc2 = h1 @ W2q.T.astype(np.float32)*s2 + b2
    int8_acc = (acc2.argmax(1) == y_te).mean()
    fp32_acc = float((mlp(torch.tensor(X_te/255.0).to(args.device)).argmax(1).cpu().numpy() == y_te).mean())
    print(f"\n=== RESULT ({args.classes}) ===")
    print(f"FP32: {fp32_acc*100:.2f}%  INT8: {int8_acc*100:.2f}%")
    print(f"Total time: {time.time()-t_all:.1f}s")

    # Export header (minimal, name depends on class set)
    if args.classes == 'lpr36':
        prefix = 'lpr36'; guard = 'LPR36_WEIGHTS_H'; hidden_name = 'LPR36_HIDDEN'
        classes_name = 'LPR36_CLASSES'; chars_decl = 'static const char lpr36_chars[36] = {' + ','.join(f"'{c}'" for c in CHARS) + '};\n'
        out_file = '/tmp/lpr36_weights.h'
    else:
        prefix = 'cn31'; guard = 'LPR_CN31_WEIGHTS_H'; hidden_name = 'CN31_HIDDEN'
        classes_name = 'CN31_CLASSES'; chars_decl = 'static const char * const cn31_provinces[31] = {' + ','.join(f'"{c}"' for c in CHARS) + '};\n'
        out_file = '/tmp/lpr_cn31_weights.h'

    h = f"""/* {args.classes} v5 max-parallel training */
/* FP32={fp32_acc*100:.2f}% INT8={int8_acc*100:.2f}% */
#ifndef {guard}
#define {guard}
#define {hidden_name} {HIDDEN}
#define {classes_name} {N_CLS}
static const float {prefix}_s1 = {s1:.9e}f;
static const float {prefix}_s2 = {s2:.9e}f;
static const int8_t {prefix}_W1[{HIDDEN}][784] = {{
"""
    for j in range(HIDDEN):
        h += "  {" + ",".join(str(int(v)) for v in W1q[j]) + "},\n"
    h += "};\n"
    h += f"static const float {prefix}_b1[{HIDDEN}] = {{" + ",".join(f"{v:.6e}f" for v in b1) + "};\n"
    h += f"static const int8_t {prefix}_W2[{N_CLS}][{HIDDEN}] = {{\n"
    for j in range(N_CLS):
        h += "  {" + ",".join(str(int(v)) for v in W2q[j]) + "},\n"
    h += "};\n"
    h += f"static const float {prefix}_b2[{N_CLS}] = {{" + ",".join(f"{v:.6e}f" for v in b2) + "};\n"
    h += chars_decl + "#endif\n"
    with open(out_file, 'w', encoding='utf-8') as f: f.write(h)
    print(f"Saved {out_file} ({len(h.encode('utf-8'))} bytes)")
