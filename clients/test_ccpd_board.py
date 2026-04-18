#!/usr/bin/env python3
"""Test 150 CCPD samples on FPGA board (v5 INT8 CNN)."""
import socket, struct, os, time
import numpy as np
from PIL import Image

BOARD = "192.168.6.191"; PORT = 5003
OUR_PROVINCES = '京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼'
OUR_LPR36 = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
CCPD_PROVINCES = ['皖','沪','津','渝','冀','晋','蒙','辽','吉','黑','苏','浙','京','闽','赣','鲁',
                   '豫','鄂','湘','粤','桂','琼','川','贵','云','藏','陕','甘','青','宁','新','警','学']
CCPD_ALNUM = ['A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V',
               'W','X','Y','Z','0','1','2','3','4','5','6','7','8','9']

def predict_plate(sock, img):
    hdr = b'PLT\x00' + struct.pack('<III', 128, 32, 0)
    sock.sendall(hdr + img.tobytes())
    resp = b''
    while len(resp) < 11:
        chunk = sock.recv(11 - len(resp))
        if not chunk: raise ConnectionError
        resp += chunk
    prov = resp[4]
    al = [resp[5+i] for i in range(6)]
    return (OUR_PROVINCES[prov] if prov < 31 else '?') + ''.join(OUR_LPR36[a] if a < 36 else '?' for a in al)

def parse_ccpd(fn):
    base = os.path.basename(fn)
    core = base.split('_ccpd')[0]
    parts = core.split('-')
    if len(parts) < 5: return None
    try:
        x1y1, x2y2 = parts[2].split('_')
        x1, y1 = map(int, x1y1.split(',')); x2, y2 = map(int, x2y2.split(','))
        pi = parts[4].split('_')
        if len(pi) != 7: return None
        pr = int(pi[0])
        if pr >= len(CCPD_PROVINCES) or CCPD_PROVINCES[pr] in ['警','学']: return None
        pch = CCPD_PROVINCES[pr]
        if pch not in OUR_PROVINCES: return None
        alnum = [CCPD_ALNUM[int(x)] for x in pi[1:]]
        if any(c not in OUR_LPR36 for c in alnum): return None
        return (x1, y1, x2, y2), pch + ''.join(alnum)
    except: return None

root = '/tmp/ccpd_sample'
files = []
for dirpath, _, fnames in os.walk(root):
    for f in fnames:
        if f.endswith('.jpg'): files.append(os.path.join(dirpath, f))
print(f"Testing {len(files)} CCPD samples on BOARD v5\n")

s = socket.socket(); s.settimeout(30); s.connect((BOARD, PORT))
correct_plate = 0; total = 0
correct_char = 0; total_char = 0
t0 = time.time()
failures = []
for fp in files:
    info = parse_ccpd(fp)
    if info is None: continue
    (x1, y1, x2, y2), gt = info
    try:
        img = Image.open(fp).convert('L').crop((x1, y1, x2, y2)).resize((128, 32), Image.BILINEAR)
        arr = np.array(img, dtype=np.uint8)
        pred = predict_plate(s, arr)
    except Exception as e:
        continue
    total += 1
    ok = pred == gt
    if ok: correct_plate += 1
    else: failures.append((gt, pred))
    for p, t in zip(pred, gt):
        total_char += 1
        if p == t: correct_char += 1
    if total <= 20:
        status = '✓' if ok else '✗'
        print(f"  {total:3d}. {gt} → {pred} {status}")
    elif total % 30 == 0:
        print(f"  [progress {total}/{len(files)}] plate_acc={correct_plate*100/total:.1f}% char_acc={correct_char*100/total_char:.1f}%")
s.close()
dt = time.time() - t0

print(f"\n{'='*60}")
print(f"CCPD independent test ON BOARD (v5):")
print(f"  Plate: {correct_plate}/{total} = {100*correct_plate/max(total,1):.2f}%")
print(f"  Char:  {correct_char}/{total_char} = {100*correct_char/max(total_char,1):.2f}%")
print(f"  Time: {dt:.1f}s ({dt/max(total,1)*1000:.0f}ms/plate)")
print(f"\nFailures ({len(failures)}/{total}):")
for gt, pred in failures[:10]:
    print(f"  {gt} → {pred}")
