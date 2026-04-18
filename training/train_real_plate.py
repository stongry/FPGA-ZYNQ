"""Train LPR36 + CN31 v6 with REAL plate font + realistic augmentation.
Uses: platech.ttf (car plate font) + blue-bg + low resolution + blur.
Parallel data gen (40 cores), GPU training.
"""
import numpy as np, os, random
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from multiprocessing import Pool
import time, sys, argparse

PLATE_FONT = '/tmp/ccpd_test/HyperLPR/resource/font/platech.ttf'

def _init_worker(seed):
    random.seed(seed + os.getpid())
    np.random.seed((seed + os.getpid()) % (2**32))

def _render_char(args):
    """Render char like real plate: blue bg or white bg, then segment extraction."""
    c_idx, ch, n_samples = args
    font = ImageFont.truetype(PLATE_FONT, random.randint(40, 60))
    results = []
    for _ in range(n_samples):
        # Random orientation: real plates are blue-bg white-text
        use_blue = random.random() < 0.7  # 70% realistic, 30% synthetic style

        # Step 1: Render large
        canvas_w, canvas_h = 120, 120
        if use_blue:
            # Blue bg (darker) with white text
            fs = random.randint(70, 90)
            font = ImageFont.truetype(PLATE_FONT, fs)
            big = Image.new('L', (canvas_w, canvas_h), random.randint(30, 80))  # dark blue
            draw = ImageDraw.Draw(big)
            bbox = draw.textbbox((0,0), ch, font=font)
            tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
            x = (canvas_w-tw)//2 - bbox[0]; y = (canvas_h-th)//2 - bbox[1]
            draw.text((x,y), ch, fill=random.randint(220, 255), font=font)  # white
        else:
            # White bg with black text (synthetic style)
            fs = random.randint(70, 90)
            font = ImageFont.truetype(PLATE_FONT, fs)
            big = Image.new('L', (canvas_w, canvas_h), 255)
            draw = ImageDraw.Draw(big)
            bbox = draw.textbbox((0,0), ch, font=font)
            tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
            x = (canvas_w-tw)//2 - bbox[0]; y = (canvas_h-th)//2 - bbox[1]
            draw.text((x,y), ch, fill=0, font=font)

        # Step 2: Augmentations simulating real plate degradations
        if random.random() < 0.5:
            big = big.rotate(random.uniform(-10, 10), resample=Image.BILINEAR, fillcolor=big.getpixel((0,0)))
        # EXTREME low resolution: real plates char is only 15-30px, we render at 90px
        # → ratio 0.17-0.33. Train with wide range [0.15, 0.8]
        if random.random() < 0.8:  # Almost always downsample
            scale = random.uniform(0.15, 0.8)
            small = big.resize((max(4,int(canvas_w*scale)), max(4,int(canvas_h*scale))), Image.BILINEAR)
            big = small.resize((canvas_w, canvas_h), Image.BILINEAR)
        if random.random() < 0.5:
            big = big.filter(ImageFilter.GaussianBlur(random.uniform(0.5, 2.0)))
        # Add random border strokes (simulate plate frame getting into char bbox)
        if random.random() < 0.25:
            draw2 = ImageDraw.Draw(big)
            bg_color = big.getpixel((0,0))
            fg_color = 255 if bg_color < 127 else 0
            side = random.choice(['top','bottom','left','right'])
            if side == 'top': draw2.rectangle([0,0,canvas_w,random.randint(2,6)], fill=fg_color)
            elif side == 'bottom': draw2.rectangle([0,canvas_h-random.randint(2,6),canvas_w,canvas_h], fill=fg_color)
            elif side == 'left': draw2.rectangle([0,0,random.randint(2,6),canvas_h], fill=fg_color)
            else: draw2.rectangle([canvas_w-random.randint(2,6),0,canvas_w,canvas_h], fill=fg_color)
        arr = np.array(big)
        # Noise + contrast variation
        if random.random() < 0.5:
            arr = arr.astype(np.float32) + np.random.normal(0, random.uniform(3, 12), arr.shape)
            arr = np.clip(arr, 0, 255)
        if random.random() < 0.3:
            arr = np.clip(arr.astype(np.int16) + random.randint(-30, 30), 0, 255)
        arr = arr.astype(np.uint8)

        # Step 3: Invert if needed (normalize to dark bg + light text like after otsu)
        if arr.mean() > 127:
            arr = 255 - arr

        # Step 4: Binarize (simulate Otsu)
        threshold = max(50, int(arr.mean() * 0.7))
        binary = arr > threshold

        # Step 5: Find bbox like segmentation does
        ca = np.where(binary.sum(0) > 0)[0]
        ra = np.where(binary.sum(1) > 0)[0]
        if len(ca) == 0 or len(ra) == 0:
            out = np.zeros((28,28), dtype=np.uint8)
        else:
            x0, x1 = ca[0], ca[-1]
            y0, y1 = ra[0], ra[-1]
            patch = np.where(binary[y0:y1+1, x0:x1+1], arr[y0:y1+1, x0:x1+1], 0).astype(np.uint8)
            # Resize to 22x22 keeping aspect, center in 28x28 (matches segmentation output)
            h, w = patch.shape
            if h >= w: nh, nw = 22, max(1, int(w*22/h))
            else: nw, nh = 22, max(1, int(h*22/w))
            img = Image.fromarray(patch).resize((nw, nh), Image.BILINEAR)
            out = np.zeros((28,28), dtype=np.uint8)
            y0c = (28-nh)//2; x0c = (28-nw)//2
            out[y0c:y0c+nh, x0c:x0c+nw] = np.array(img)

        results.append((out.flatten(), c_idx))
    return results

def gen_parallel(chars, n_per_class, n_workers=40):
    chunk = max(100, n_per_class // n_workers)
    tasks = []
    for c_idx, ch in enumerate(chars):
        rem = n_per_class
        while rem > 0:
            take = min(chunk, rem)
            tasks.append((c_idx, ch, take))
            rem -= take
    t0 = time.time()
    with Pool(n_workers, initializer=_init_worker, initargs=(42,)) as p:
        results = p.map(_render_char, tasks)
    X = np.concatenate([np.array([r[0] for r in batch]) for batch in results])
    y = np.concatenate([np.array([r[1] for r in batch]) for batch in results])
    print(f"  Generated {len(X)} in {time.time()-t0:.1f}s")
    return X.astype(np.float32), y

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--classes', choices=['lpr36', 'cn31'], required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch', type=int, default=1024)
    parser.add_argument('--train-per-class', type=int, default=5000)
    args = parser.parse_args()

    if args.classes == 'lpr36':
        CHARS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        HIDDEN, N_CLS = 128, 36
        OUT = '/tmp/lpr36_weights.h'
        prefix, guard = 'lpr36', 'LPR36_WEIGHTS_H'
        hidden_name, classes_name = 'LPR36_HIDDEN', 'LPR36_CLASSES'
        chars_decl = 'static const char lpr36_chars[36] = {' + ','.join(f"'{c}'" for c in CHARS) + '};\n'
    else:
        CHARS = '京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼'
        HIDDEN, N_CLS = 128, 31
        OUT = '/tmp/lpr_cn31_weights.h'
        prefix, guard = 'cn31', 'LPR_CN31_WEIGHTS_H'
        hidden_name, classes_name = 'CN31_HIDDEN', 'CN31_CLASSES'
        chars_decl = 'static const char * const cn31_provinces[31] = {' + ','.join(f'"{c}"' for c in CHARS) + '};\n'

    print(f"=== {args.classes} v6 (real plate font + blue bg aug) ===")
    t_all = time.time()
    print(f"[DataGen] Train: {args.train_per_class}×{N_CLS}")
    X_tr, y_tr = gen_parallel(CHARS, args.train_per_class, 40)
    print(f"[DataGen] Test: 300×{N_CLS}")
    X_te, y_te = gen_parallel(CHARS, 300, 40)

    import torch, torch.nn as nn
    class MLP(nn.Module):
        def __init__(s):
            super().__init__()
            s.fc1 = nn.Linear(784, HIDDEN); s.fc2 = nn.Linear(HIDDEN, N_CLS)
        def forward(s, x): return s.fc2(torch.relu(s.fc1(x)))

    mlp = MLP().to(args.device)
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    Xt = torch.tensor(X_tr/255.0).to(args.device); yt = torch.tensor(y_tr, dtype=torch.long).to(args.device)
    Xte_t = torch.tensor(X_te/255.0).to(args.device); yte_t = torch.tensor(y_te, dtype=torch.long).to(args.device)

    for ep in range(args.epochs):
        perm = torch.randperm(len(Xt)); tl = 0
        for i in range(0, len(Xt), args.batch):
            idx = perm[i:i+args.batch]
            opt.zero_grad()
            loss = loss_fn(mlp(Xt[idx]), yt[idx])
            loss.backward(); opt.step(); tl += loss.item()
        mlp.eval()
        with torch.no_grad():
            acc = (mlp(Xte_t).argmax(1) == yte_t).float().mean().item()
        mlp.train()
        if ep % 3 == 0 or ep == args.epochs-1:
            print(f"  Ep{ep+1}: loss={tl:.1f} acc={acc*100:.2f}%")

    mlp.eval()
    with torch.no_grad():
        pred = mlp(Xte_t).argmax(1).cpu().numpy()
        fp32 = (pred == y_te).mean()

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
    print(f"FP32: {fp32*100:.2f}%  INT8: {int8_acc*100:.2f}%  time: {time.time()-t_all:.1f}s")

    h = f"""/* {args.classes} v6 real-plate-font trained */
/* FP32={fp32*100:.2f}% INT8={int8_acc*100:.2f}% */
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
    with open(OUT, 'w', encoding='utf-8') as f: f.write(h)
    print(f"Saved {OUT}")
