#!/usr/bin/env python3
"""Test end-to-end plate CNN on FPGA board (PS software inference).
Protocol: PLT\0 + w(4)=128 + h(4)=32 + fmt(4) + 4096 bytes u8 grayscale
Reply:    PRD\0 + prov(1) + al[6] (7 total char indices)
"""
import socket, struct, os, time, re
import numpy as np
from PIL import Image

BOARD = "192.168.6.191"
PORT = 5003
PROVINCES = '京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼'
LPR36 = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

def predict_plate(sock, plate_img):
    """plate_img: 2D uint8 array (32, 128). Returns predicted string."""
    hdr = b'PLT\x00' + struct.pack('<III', 128, 32, 0)
    sock.sendall(hdr + plate_img.tobytes())
    resp = b''
    while len(resp) < 11:
        chunk = sock.recv(11 - len(resp))
        if not chunk: raise ConnectionError
        resp += chunk
    assert resp[:4] == b'PRD\x00', f"Bad magic: {resp[:4]}"
    prov = resp[4]
    al = [resp[5+i] for i in range(6)]
    pred = (PROVINCES[prov] if prov < 31 else '?') + \
           ''.join(LPR36[a] if a < 36 else '?' for a in al)
    return pred

def get_label(fn):
    m = re.match(r'_\d+_([^.]+)\.\w+', fn)
    if m: return m.group(1)
    base = fn.split('.')[0]
    if '_convert' in base: return base.split('_')[0]
    p = base.rsplit('_', 1)
    return p[0]+p[1] if len(p)==2 and p[1].isdigit() else base

def main():
    DIR = '/tmp/real_plates'
    files = sorted([f for f in os.listdir(DIR) if f.endswith(('.jpg','.png'))])
    print(f"Testing {len(files)} real plates on BOARD CNN (PS software)\n")

    s = socket.socket()
    s.settimeout(30)
    s.connect((BOARD, PORT))

    correct_plate = 0; total_plate = 0
    correct_char = 0; total_char = 0
    total_time = 0
    for fn in files:
        lbl = get_label(fn)
        if len(lbl) != 7:
            print(f"  SKIP {fn} (len={len(lbl)})")
            continue
        img = Image.open(os.path.join(DIR, fn)).convert('L').resize((128, 32), Image.BILINEAR)
        arr = np.array(img, dtype=np.uint8)
        t0 = time.time()
        pred = predict_plate(s, arr)
        dt = time.time() - t0
        total_time += dt
        total_plate += 1
        ok = pred == lbl
        if ok: correct_plate += 1
        for idx, (p, t) in enumerate(zip(pred, lbl)):
            total_char += 1
            if p == t: correct_char += 1
        status = "✓" if ok else "✗"
        print(f"  {fn:35s} '{pred}' vs '{lbl}' {status}  ({dt*1000:.0f}ms)")

    s.close()
    print(f"\n{'='*60}")
    print(f"Plate-level: {correct_plate}/{total_plate} = {100*correct_plate/max(total_plate,1):.1f}%")
    print(f"Char-level:  {correct_char}/{total_char} = {100*correct_char/max(total_char,1):.1f}%")
    print(f"Avg inference: {total_time/max(total_plate,1)*1000:.0f}ms/plate")

if __name__ == '__main__':
    main()
