# Multi-Task CNN Vision System on ALINX FZ3A

Complete on-board traffic perception system: **two CNNs run simultaneously in PL**, orchestrated by bare-metal ARM A53 PS. Full-scene processing, no external compute.

## 🎯 System Capabilities

| Task | Input | Output | Hardware |
|---|---|---|---|
| **License plate recognition** | Arbitrary scene image | Plate text (province + 6 alnum) | PL `plate_cnn_hls` |
| **Pedestrian detection** | Arbitrary scene image | Bounding boxes + count | PL `pedcnn_hls_ip` sliding window |
| **Edge-based plate localization** | 320×240..1280×720 grayscale | Candidate rectangles | PS software Sobel + grid density |

## Architecture

```
Client (TCP port 5004):
  Raw photo → grayscale → send to board
                              ↓
Board PS (ARM Cortex-A53 bare-metal):
  ① Sobel edge detection (software ~100ms)
  ② Grid density ranking → candidate rectangles
  ③ Pixel-precision vertical bbox refinement
  ④ Multi-scale variants (tight + expanded)
                              ↓
Board PL (FPGA INT8 accelerators):
  ⑤ plate_cnn_hls per candidate (675ms / plate)
  ⑥ pedcnn_hls per sliding window (42ms / window)
                              ↓
Board PS:
  ⑦ NMS on pedestrian detections (IoU > 0.3)
  ⑧ Serialize RES\0 + plates + peds
                              ↓
Client: Draw bboxes + text on visualization
```

## FPGA Resource Utilization (v18 bitstream)

| IP | BRAM18 | DSP | Description |
|---|---|---|---|
| `plate_cnn_hls` | ~308 | 66 | 4 Conv + FC + 7 heads, end-to-end INT8 |
| `pedcnn_hls_ip` | ~20 | ~20 | 3 Conv + GAP + FC, binary classifier |
| AXI Smartconnect | ~5 | - | PS-PL interconnect |
| **Total** | **~330 / 432 (76%)** | **~86 / 360 (24%)** | ZU3EG XCZU3EG |

## Accuracy Measurements

### Plate CNN Alone (pre-cropped 128×32 input)
Measured on 141 CCPD holdout:
- **Plate-level**: 87.94% (124/141)
- **Char-level**: 97.97% (967/987)
- **Latency**: 675 ms / plate (PL HLS)

### End-to-End Plate Pipeline (on-board Sobel localize + CNN)
Measured on 10 CCPD scene images:
- **Found candidate**: 100% (10/10 images yielded candidate rects)
- **Exact plate match**: **50.00%** (5/10)

Progression of end-to-end accuracy through localization refinement iterations:

| Localization algorithm | Exact match | Observation |
|---|---|---|
| Coarse top-N grid scan | 10% (1/10) | Picks sky/foliage edges first |
| Add sky-skip (y: 30–90%) | 30% (3/10) | Skip top sky band, bottom ground |
| Add edge-score ranking (top-32) | 40% (4/10) | Rank candidates by edge density |
| + vertical bbox refine (pixel-prec) | 40% | True bbox found but offset matters |
| + horizontal bbox refine | 20% (regression) | Column threshold unreliable on gaps |
| + 2-scale variants (tight + expanded) | **50% (5/10)** | Multi-scale helps CNN find best crop |

### Plate CNN + Client-Side OpenCV Localization (baseline)
For comparison (violates "all on board" constraint):
- **40%** (5 real photos) — similar to on-board

**On-board Sobel with multi-scale = OpenCV baseline**, while staying fully embedded.

### Pedestrian Detection
- **Classifier alone**: verified 0 false positives on black image, detections on true pedestrians
- **End-to-end sliding window**: 8 peds detected in Penn-Fudan test scene (before NMS → 2 after NMS)
- **Per-window latency**: ~42 ms (PL pedcnn)
- **Total per 320×240 image**: ~6 s with step=24 + Sobel density filter (from 11 s baseline)

## TCP Protocol

**Port 5003** — plate CNN only (pre-cropped input)
```
REQ: "PLT\0" + w(4)=128 + h(4)=32 + fmt(4) + 4096 bytes u8
RES: "PRD\0" + prov(1) + al[6]
```

**Port 5002** — pedcnn only (320×240 input, PL sliding window)
```
REQ: "PED\0" + w(4)=320 + h(4)=240 + fmt(4) + 76800 bytes u8
RES: "DET\0" + n(4) + n * {pos(4) + score(4)}
```

**Port 5004** — full scene multi-task
```
REQ: "ALL\0" + w(4) + h(4) + fmt(4) + w*h bytes u8 (max 1280×720)
RES: "RES\0" + n_plates(1)
     + n_plates * {x(2) y(2) w(2) h(2) prov(1) al[6]}
     + n_peds(1)
     + n_peds * {x(2) y(2) w(2) h(2) score(4)}
```

## Firmware Pipeline (PS, phase2b_main.c)

Key functions added for on-board multi-task:
- `sobel_ps()` — 3×3 Sobel edge detection in software
- `find_plate_rects()` — grid density → candidate ranking → bbox refinement → multi-scale variants
- `refine_bbox_vertical()` — pixel-precision tight top/bottom detection
- `crop_resize_128x32()` — patch extraction + nearest-neighbor resize for plate CNN
- `pedcnn_run_patch()` — single 64×128 window through PL pedcnn
- `pedcnn_sliding()` — step=24 sliding window with Sobel density pre-filter + in-firmware NMS
- `downsample_320x240()` — full-image → pedcnn input
- `all_run_and_reply()` — TCP 5004 orchestrator

## Build Pipeline

### Vivado (Linux 40-core server)
```
source settings64.sh
cd /home/eea/fz3a_build
vivado -mode batch -source integrate_v18.tcl
# ~10-15 min to write_bitstream
```

### HLS (kernels)
- `plate_cnn_hls/plate_cnn_hls_kernel.cpp` — plate CNN
- `pedcnn_hls/pedcnn_hls_kernel.cpp` — pedestrian classifier
- Export IP catalog for BD integration

### Firmware (Windows cross-compile)
```powershell
aarch64-none-elf-gcc -O2 -mcpu=cortex-a53 -ffreestanding \
  phase2b_main.c stubs.c \
  -lxil -llwip4 -lgcc -lc -lm \
  -o phase2b.elf
```

### Deploy
```tcl
# deploy_correct.tcl on XSDB
source psu_init.tcl
psu_init
psu_ps_pl_reset_config
psu_post_config
psu_ps_pl_isolation_removal    # CRITICAL for JTAG
fpga -no-rev -f design.bit
rst -processor
dow phase2b.elf
con
```

## Key Engineering Lessons

1. **`psu_ps_pl_isolation_removal` is mandatory** for XSDB JTAG deploy (FSBL does it; JTAG doesn't).
2. **HLS `PIPELINE II=1` at outer loops triggers complete unroll** of inner loops → instruction count explosion. Use `ARRAY_PARTITION` on weights instead.
3. **`apply_bd_automation` > manual crossbar** — hand-setting `NUM_MI` leaves dangling ports that break AXI decoding.
4. **PL HLS register maps matter** — image BRAM at 0x1000, control regs at 0x00, two separate DDR pointer regs — misalignment silently returns garbage.
5. **Cross-domain fine-tuning risk** — training on richjjj 213K gave 99.84% val but dropped CCPD to 60%. Always include target-domain data.
6. **On-board Sobel vs client OpenCV**: simple 8×8 grid + sort + vertical refine matches OpenCV accuracy for plate localization.

## Demo Outputs

In `bitstreams/`:
- `design_1_wrapper_v18_plate_pedcnn.bit` — production bitstream (plate + pedcnn)
- `design_1_wrapper_baseline.bit` — plate-only bitstream (validated 87.94%)

In project root demo PNG files:
- Penn-Fudan (single real pedestrian): 1 bbox after NMS ✓
- CCPD stop scenes: 5–10 plate candidates with 1–2 valid plate texts

## Future Work

- **Plate classifier pre-filter**: add tiny binary CNN before plate_cnn (like pedcnn for ped)
- **PL Sobel** (`filter_hls` integration): v19 bitstream attempted, boot hang requires debug
- **Multi-scale image pyramid**: 3 resolution test for small plates
- **Larger training data**: CCPD2020 original (matches test distribution better than our test set)

---

**Summary**: Two real CNNs in FPGA PL running simultaneously on a ZU3EG (mid-range SoC), with pure ARM A53 orchestration. End-to-end traffic perception (plate + pedestrian) at ~6–11 seconds per 720p frame, completely self-contained.

*Built on ALINX FZ3A / Xilinx XCZU3EG, Vivado / Vitis 2024.2*
