#!/usr/bin/env python3
"""Send an image file to FZ3A over TCP port 5000 for DP display.
Usage: send_image.py <image_file> [host]
Image is resized to 1280x720 and sent as raw RGBA8888 with a small header.
"""
import socket
import struct
import sys
from PIL import Image

if len(sys.argv) < 2:
    print("usage: send_image.py <file.jpg|png> [host]")
    sys.exit(1)

path = sys.argv[1]
host = sys.argv[2] if len(sys.argv) > 2 else '192.168.6.192'
PORT = 5000
W, H = 1280, 720

print(f"Loading {path}...")
img = Image.open(path)
img = img.resize((W, H), Image.LANCZOS).convert('RGBA')
raw = img.tobytes()  # PIL RGBA = bytes[0]=R, bytes[1]=G, bytes[2]=B, bytes[3]=A
# That matches what our DP expects (R in low byte of 32-bit word, little-endian)
print(f"Resized to {W}x{H}, raw size = {len(raw)} bytes")

print(f"Connecting to {host}:{PORT}...")
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(10.0)
s.connect((host, PORT))

hdr = b'IMG\x00' + struct.pack('<III', W, H, 0)  # fmt=0 RGBA8888
print(f"Sending header + {len(raw)} bytes...")
t0 = __import__('time').time()
s.sendall(hdr + raw)
t1 = __import__('time').time()
print(f"Sent {len(hdr)+len(raw)} bytes in {t1-t0:.2f}s = {(len(hdr)+len(raw))/(t1-t0)/1e6:.1f} MB/s")
s.close()
print("Done. Check monitor.")
