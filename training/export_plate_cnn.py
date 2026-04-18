"""Export trained PlateCNN to C header for firmware deployment.
BN folded into Conv, weights quantized to INT8.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

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

ckpt = torch.load('/tmp/plate_cnn_e2e.pt', map_location='cpu', weights_only=True)
net = PlateCNN()
net.load_state_dict(ckpt['state_dict'])
net.eval()
print(f"Model loaded. Validation plate acc: {ckpt['plate_acc']*100:.2f}%")

def fold_bn(conv, bn):
    """Fold BatchNorm into preceding Conv2d.
    Returns (W_eff, b_eff) where:
      y = conv(x, W_eff) + b_eff  replaces  bn(conv(x, W, b_conv))
    """
    W = conv.weight.data.numpy()        # (O, I, K, K)
    b = conv.bias.data.numpy()          # (O,)
    gamma = bn.weight.data.numpy()      # (O,)
    beta = bn.bias.data.numpy()         # (O,)
    mean = bn.running_mean.data.numpy() # (O,)
    var = bn.running_var.data.numpy()   # (O,)
    eps = bn.eps
    scale = gamma / np.sqrt(var + eps)  # (O,)
    W_eff = W * scale.reshape(-1, 1, 1, 1)   # broadcast
    b_eff = (b - mean) * scale + beta
    return W_eff, b_eff

# Fold BN into each conv
c1_w, c1_b = fold_bn(net.conv1, net.bn1)
c2_w, c2_b = fold_bn(net.conv2, net.bn2)
c3_w, c3_b = fold_bn(net.conv3, net.bn3)
c4_w, c4_b = fold_bn(net.conv4, net.bn4)

# FC weights
fc_w = net.fc.weight.data.numpy()   # (512, 4096)
fc_b = net.fc.bias.data.numpy()     # (512,)
hcn_w = net.head_cn.weight.data.numpy()  # (31, 512)
hcn_b = net.head_cn.bias.data.numpy()
hal_w = [h.weight.data.numpy() for h in net.heads_al]  # 6x (36, 512)
hal_b = [h.bias.data.numpy() for h in net.heads_al]

# Quantize to INT8 (per-tensor symmetric)
def q8(W):
    s = float(np.abs(W).max() / 127)
    if s == 0: s = 1e-6
    return np.clip(np.round(W/s), -128, 127).astype(np.int8), s

# Verify INT8 inference matches FP32
def int8_sim(x_u8):
    """Simulate INT8 inference exactly as firmware will do."""
    c1q, s_c1 = q8(c1_w); c2q, s_c2 = q8(c2_w); c3q, s_c3 = q8(c3_w); c4q, s_c4 = q8(c4_w)
    fcq, s_fc = q8(fc_w); hcnq, s_hcn = q8(hcn_w)
    halq = [q8(w) for w in hal_w]

    # Conv+ReLU+Pool pipeline in INT8 with float biases for accuracy
    def conv_int8(x, W_q, b, scale_w):
        """Returns float32 output after conv + ReLU + maxpool2x2.
        x: float32 HxWxC (or CxHxW) — we use CxHxW convention.
        """
        O, I, K, K2 = W_q.shape
        H, Wd = x.shape[1], x.shape[2]
        # Pad input by 1 on each side
        x_pad = np.pad(x, ((0,0),(1,1),(1,1)), constant_values=0)
        # Compute conv output (no quantization of x — use float for now to verify)
        out = np.zeros((O, H, Wd), dtype=np.float32)
        for oc in range(O):
            acc = np.zeros((H, Wd), dtype=np.float32)
            for ic in range(I):
                for kh in range(3):
                    for kw in range(3):
                        w_v = W_q[oc, ic, kh, kw] * scale_w
                        acc += x_pad[ic, kh:kh+H, kw:kw+Wd] * w_v
            out[oc] = acc + b[oc]
        # ReLU
        out = np.maximum(out, 0)
        # MaxPool 2x2
        out = out.reshape(O, H//2, 2, Wd//2, 2).max(axis=(2,4))
        return out

    # Conv1: input (1, 32, 128) — float normalized
    x = x_u8.astype(np.float32) / 255.0
    x = x.reshape(1, 32, 128)
    c1_out = conv_int8(x, c1q, c1_b, s_c1)  # (32, 16, 64)
    c2_out = conv_int8(c1_out, c2q, c2_b, s_c2)  # (64, 8, 32)
    c3_out = conv_int8(c2_out, c3q, c3_b, s_c3)  # (128, 4, 16)
    c4_out = conv_int8(c3_out, c4q, c4_b, s_c4)  # (256, 2, 8)
    # Flatten
    flat = c4_out.flatten()  # 4096
    # FC
    fc_out = (flat @ fcq.astype(np.float32).T) * s_fc + fc_b
    fc_out = np.maximum(fc_out, 0)  # ReLU
    # Heads
    cn_out = (fc_out @ hcnq.astype(np.float32).T) * s_hcn + hcn_b
    al_outs = [(fc_out @ wq.astype(np.float32).T) * sw + hal_b[i]
               for i, (wq, sw) in enumerate(halq)]
    return int(cn_out.argmax()), [int(a.argmax()) for a in al_outs]

# Quick verification on a few test plates
print("\n=== INT8 verification on real plates ===")
import os
from PIL import Image
PROVINCES = '京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼'
LPR36 = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
import re
if os.path.exists('/tmp/real_plates'):
    for fn in sorted(os.listdir('/tmp/real_plates')):
        if not fn.endswith(('.jpg','.png')): continue
        m = re.match(r'_\d+_([^.]+)\.\w+', fn)
        if m: gt = m.group(1)
        else:
            base = fn.split('.')[0]
            if '_convert' in base: gt = base.split('_')[0]
            else:
                p = base.rsplit('_', 1)
                gt = p[0]+p[1] if len(p)==2 and p[1].isdigit() else base
        if len(gt) != 7: continue
        img = Image.open(f'/tmp/real_plates/{fn}').convert('L').resize((128, 32), Image.BILINEAR)
        arr = np.array(img, dtype=np.uint8)
        # FP32 check
        with torch.no_grad():
            x = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
            out_cn, outs_al = net(x)
            pred_fp = PROVINCES[out_cn.argmax().item()] + ''.join(LPR36[o.argmax().item()] for o in outs_al)
        # INT8 sim (slow but verifies)
        cn_i, al_i = int8_sim(arr)
        pred_i8 = PROVINCES[cn_i] + ''.join(LPR36[a] for a in al_i)
        match_fp = '✓' if pred_fp == gt else '✗'
        match_i8 = '✓' if pred_i8 == gt else '✗'
        print(f"  {fn:35s} FP32: '{pred_fp}' {match_fp}  INT8: '{pred_i8}' {match_i8}")

# Actual quantization + export to C
print("\n=== Exporting quantized weights ===")
c1q, s_c1 = q8(c1_w); c2q, s_c2 = q8(c2_w); c3q, s_c3 = q8(c3_w); c4q, s_c4 = q8(c4_w)
fcq, s_fc = q8(fc_w); hcnq, s_hcn = q8(hcn_w)
halq = [q8(w) for w in hal_w]

# Write header file
out = '/tmp/plate_cnn_weights.h'
with open(out, 'w', encoding='utf-8') as f:
    f.write(f"""/* PlateCNN weights (INT8). Trained on 87K real plates.
 * Validation plate acc: {ckpt['plate_acc']*100:.2f}%
 * Architecture: 4x Conv3x3 + BN + ReLU + MaxPool2x2 + FC(4096→512) + 7 heads
 * Input: 128x32 grayscale (uint8 0-255)
 * Output: 1 province (31 classes) + 6 alphanum (36 classes)
 */
#ifndef PLATE_CNN_WEIGHTS_H
#define PLATE_CNN_WEIGHTS_H
#include <stdint.h>

#define PCN_IN_H 32
#define PCN_IN_W 128
#define PCN_CN_CLASSES 31
#define PCN_AL_CLASSES 36
#define PCN_N_AL_HEADS 6

/* BN-folded Conv weight scales */
static const float pcn_s_c1 = {s_c1:.9e}f;
static const float pcn_s_c2 = {s_c2:.9e}f;
static const float pcn_s_c3 = {s_c3:.9e}f;
static const float pcn_s_c4 = {s_c4:.9e}f;
static const float pcn_s_fc = {s_fc:.9e}f;
static const float pcn_s_hcn = {s_hcn:.9e}f;
static const float pcn_s_hal[6] = {{{','.join(f'{s:.9e}f' for _,s in halq)}}};
""")

    # Conv1: (32, 1, 3, 3) = 288
    f.write("/* conv1: 32*1*3*3 = 288 */\n")
    f.write("static const int8_t pcn_c1W[32][9] = {\n")
    for o in range(32):
        vals = c1q[o, 0].flatten()
        f.write("  {" + ",".join(str(int(v)) for v in vals) + "},\n")
    f.write("};\n")
    f.write(f"static const float pcn_c1b[32] = {{{','.join(f'{v:.6e}f' for v in c1_b)}}};\n\n")

    # Conv2: (64, 32, 3, 3) = 18432
    f.write("/* conv2: 64*32*3*3 = 18432 */\n")
    f.write("static const int8_t pcn_c2W[64][32][9] = {\n")
    for o in range(64):
        f.write("  {")
        for i in range(32):
            vals = c2q[o, i].flatten()
            f.write("{" + ",".join(str(int(v)) for v in vals) + "},")
        f.write("},\n")
    f.write("};\n")
    f.write(f"static const float pcn_c2b[64] = {{{','.join(f'{v:.6e}f' for v in c2_b)}}};\n\n")

    # Conv3: (128, 64, 3, 3) = 73728
    f.write("/* conv3: 128*64*3*3 = 73728 */\n")
    f.write("static const int8_t pcn_c3W[128][64][9] = {\n")
    for o in range(128):
        f.write("  {")
        for i in range(64):
            vals = c3q[o, i].flatten()
            f.write("{" + ",".join(str(int(v)) for v in vals) + "},")
        f.write("},\n")
    f.write("};\n")
    f.write(f"static const float pcn_c3b[128] = {{{','.join(f'{v:.6e}f' for v in c3_b)}}};\n\n")

    # Conv4: (256, 128, 3, 3) = 294912
    f.write("/* conv4: 256*128*3*3 = 294912 */\n")
    f.write("static const int8_t pcn_c4W[256][128][9] = {\n")
    for o in range(256):
        f.write("  {")
        for i in range(128):
            vals = c4q[o, i].flatten()
            f.write("{" + ",".join(str(int(v)) for v in vals) + "},")
        f.write("},\n")
    f.write("};\n")
    f.write(f"static const float pcn_c4b[256] = {{{','.join(f'{v:.6e}f' for v in c4_b)}}};\n\n")

    # FC: (512, 4096) = 2M
    f.write("/* fc: 512*4096 = 2097152 */\n")
    f.write("static const int8_t pcn_fcW[512][4096] = {\n")
    for o in range(512):
        f.write("  {" + ",".join(str(int(v)) for v in fcq[o]) + "},\n")
    f.write("};\n")
    f.write(f"static const float pcn_fcb[512] = {{{','.join(f'{v:.6e}f' for v in fc_b)}}};\n\n")

    # Head CN: (31, 512)
    f.write("/* head_cn: 31*512 */\n")
    f.write("static const int8_t pcn_hcnW[31][512] = {\n")
    for o in range(31):
        f.write("  {" + ",".join(str(int(v)) for v in hcnq[o]) + "},\n")
    f.write("};\n")
    f.write(f"static const float pcn_hcnb[31] = {{{','.join(f'{v:.6e}f' for v in hcn_b)}}};\n\n")

    # Heads AL: 6 × (36, 512)
    f.write("/* heads_al: 6 × 36*512 */\n")
    f.write("static const int8_t pcn_halW[6][36][512] = {\n")
    for h in range(6):
        f.write("  {\n")
        for o in range(36):
            f.write("    {" + ",".join(str(int(v)) for v in halq[h][0][o]) + "},\n")
        f.write("  },\n")
    f.write("};\n")
    f.write("static const float pcn_halb[6][36] = {\n")
    for h in range(6):
        f.write("  {" + ",".join(f'{v:.6e}f' for v in hal_b[h]) + "},\n")
    f.write("};\n\n")

    f.write(f"#endif\n")

import os as _os
size = _os.path.getsize(out)
print(f"Saved {out}: {size/1024/1024:.2f} MB")
