# PL Plate CNN on ALINX FZ3A (Zynq UltraScale+ XCZU3EG)

End-to-end Chinese license plate recognition CNN deployed as FPGA Programmable Logic (PL) HLS kernel, accessed from bare-metal ARM A53 firmware via AXI-Lite.

## Final Results

| Metric | Value |
|---|---|
| **Plate accuracy** | **87.94%** (124/141 CCPD) |
| **Char accuracy** | **97.97%** (967/987) |
| **PL latency** | **675 ms/plate** |
| **PS baseline** | 1170 ms/plate |
| **Speedup** | 1.73x |
| Target device | XCZU3EG (ALINX FZ3A) |
| BRAM usage | 75% (326/432) |
| DSP usage | 36% (131/360) |

## Architecture

```
Input:  128×32 grayscale plate (4096 bytes, uint8)
  │
  ├─ Conv1 (1→32)   + ReLU + MaxPool2x2 → (32, 16, 64)
  ├─ Conv2 (32→64)  + ReLU + MaxPool2x2 → (64, 8, 32)
  ├─ Conv3 (64→128) + ReLU + MaxPool2x2 → (128, 4, 16)
  ├─ Conv4 (128→256) + ReLU + MaxPool2x2 → (256, 2, 8) = 4096 features
  ├─ FC (4096→512) + ReLU        [weights streamed from DDR via m_axi]
  ├─ Head CN (512→31) → argmax → province index
  └─ Heads AL ×6 (512→36) → argmax → 6 alnum indices

Output: 7 bytes [province, alnum×6]
```

- INT8 per-channel quantization
- Conv + heads weights baked in BRAM (~500KB)
- FC weights streamed from DDR (2MB)
- Host-side activation scales (per-layer 99.99% percentile)

## Key Engineering Lessons

### 1. `psu_ps_pl_isolation_removal` is mandatory for XSDB JTAG deploy

FSBL boot automatically releases the AIB (Autonomous Interface Bridge) isolation between PS and PL. XSDB `fpga -file` does NOT. Without calling `psu_ps_pl_isolation_removal` in the deploy TCL, every AXI-Lite transaction from PS CPU to PL times out silently (AXI AP transaction error, DAP status 0x30000021).

```tcl
# Correct deploy sequence
source psu_init.tcl
targets -set -filter {name =~ "PSU"}
psu_init
psu_ps_pl_reset_config
psu_post_config
psu_ps_pl_isolation_removal    ;# CRITICAL - missing from most docs
fpga -no-rev -f design.bit
targets -set -filter {name =~ "Cortex-A53 #0"}
rst -processor
dow firmware.elf
con
```

### 2. Vivado BD `apply_bd_automation` pitfalls

Do NOT manually `set_property CONFIG.NUM_MI N` on `ps8_0_axi_periph` after deleting IPs. Stale crossbar ports (dangling masters) make AXI address decoder route addresses to nonexistent slaves → CPU hangs forever waiting for BRESP.

Working approach: delete the offending IP via `delete_bd_objs`, let `apply_bd_automation -rule xilinx.com:bd_rule:axi4` manage `NUM_MI` automatically.

### 3. HLS `PIPELINE II=1` pragma placement

Placing `#pragma HLS PIPELINE II=1` at the outer loop (e.g. `OW`) forces complete unroll of inner loops — `32 ic × 3 ky × 3 kx × 4 pool = 1152 MACs` physically realized. ZU3EG has 360 DSPs → 3.6M-instruction HLS IR explosion, synthesis hangs or exceeds resources.

Solution: place PIPELINE at a level where the loop body is small enough (or use `UNROLL factor=N` + `ARRAY_PARTITION cyclic factor=N` on weights + activations).

### 4. HLS register layout of BRAM arrays

`s_axilite port=image` where `image[4096]` creates a 4KB BRAM slave. Firmware cannot write byte-by-byte via AXI-Lite (each 32-bit transaction ≈ 300ns) — that’s 1.2 ms minimum just to load the image. Generated register map:

```
0x0000 ap_ctrl
0x0010/0x0014 fc_weights_addr[63:0]
0x001c/0x0020 fc_weights_ddr[63:0]   ← HLS generates TWO pointer params
0x0028 ~ 0x002f predictions[7]
0x1000 ~ 0x1fff image[4096]
```

Firmware must write BOTH `fc_weights_addr` and `fc_weights_ddr` to the same DDR address — they’re the same logical parameter but HLS generates two scalar registers.

### 5. Firmware weight-mismatch debugging

If PL mode returns 0/141 but bitstream programmed successfully + AXI-Lite works: the likely cause is Conv weights in the HLS IP (baked in bitstream) don’t match the FC weights in firmware. Both come from the same training checkpoint; updating only one side produces random outputs.

Tip: keep baseline and FT weights in clearly labeled files, verify md5 before deploying.

### 6. Fine-tuning cross-domain

Fine-tuned on `richjjj/chinese_license_plate_rec` (213K plates, val=99.84%) but CCPD cross-domain eval gave **60.28%** plate accuracy (down from baseline 87.94%). richjjj and CCPD are different distributions. Need CCPD training data to preserve CCPD accuracy.

### 7. HLS latency optimization limits

Tried ARRAY_PARTITION factor=4/9 and various UNROLL strategies — achieved at most **3-5% latency reduction** (663→630ms). Real 10-20x speedup needs kernel rewrite with:

- Padded buffers to eliminate `if (iy<0||iy>=H) continue` boundary checks (breaks II=1)
- Dataflow streaming between Conv layers (no buf_a/buf_b ping-pong)
- `ic` loop unrolled with adequate DSP (parallelism)
- Weight ARRAY_PARTITION along parallel-read axis

## Deploy Workflow

1. **Linux server** (40 cores + RTX 5090) → Vivado synth + bitstream
2. **Windows host** (Vitis) → XSDB JTAG deploy bitstream + ELF to board
3. **Board (FZ3A)** → runs firmware serving TCP 5003 (plate CNN)

### Compile firmware (Windows)

```powershell
# dp/build.ps1
aarch64-none-elf-gcc -O2 -mcpu=cortex-a53 -ffreestanding \
  -I"$bsp\include" -T lscript.ld "-L$bsp\lib" \
  -Wl,--start-group phase2b_main.c stubs.c \
  -lxil -llwip4 -lgcc -lc -lm -Wl,--end-group \
  -o phase2b.elf
```

### Deploy via XSDB

```tcl
# See dp/deploy_correct.tcl - key bit:
connect -url tcp:localhost:3121
source psu_init.tcl
targets -set -filter {name =~ "PSU"}
psu_init
psu_ps_pl_reset_config
psu_post_config
psu_ps_pl_isolation_removal
fpga -no-rev -f design_1_wrapper.bit
targets -set -filter {name =~ "Cortex-A53 #0"}
rst -processor
dow phase2b.elf
con
```

### Test from Linux host

```bash
# Full CCPD 141-plate test
python3 clients/test_ccpd_board.py
# Expected: Plate 124/141 = 87.94%, 675ms/plate

# Toggle PS↔PL via UART 'V' key
# PS mode: 1170 ms/plate, same accuracy (same model, different backend)
```

## Files

| Path | Description |
|---|---|
| `hls/plate_cnn_hls_kernel.cpp` | HLS C++ kernel (baseline, 663ms) |
| `hls/plate_cnn_hls_weights.h` | INT8 Conv + Head weights (baked into bitstream) |
| `firmware/phase2b_main.c` | A53 bare-metal firmware |
| `firmware/plate_cnn_fc_weights.h` | INT8 FC weights (embedded in ELF) |
| `firmware/plate_cnn_weights.h` | PS-mode CNN weights (different format) |
| `bitstreams/design_1_wrapper_baseline.bit` | **PRODUCTION** bitstream (verified 87.94%) |
| `scripts/integrate_v15.tcl` | Vivado BD integration (baseline + clean crossbar) |
| `scripts/run_plate_cnn_hls.tcl` | Vitis HLS synthesis script |
| `clients/test_ccpd_board.py` | 141-sample CCPD CCPD verification |
| `clients/test_plate_cnn_board.py` | 5-sample quick smoke test |
| `.../deploy_correct.tcl` (Windows) | XSDB deploy script with isolation removal |

## Memory Map

| Address | Size | Function |
|---|---|---|
| `0xA0000000` | 64 KB | plate_cnn_hls AXI-Lite slave (v15 BD) |
| `0x0012XXXX` | 2 MB | FC weights in DDR `.rodata` section |
| `0x00000000 – 0x7FFFFFFF` | 2 GB | DDR (code + data) |

## Known Issues

1. **HLS FSM residual state** — after power cycle without `rst -processor`, HLS IP can have `ap_start=1` latched. Firmware must `(void)PLT_CNN_AP_CTRL` read to clear `ap_done`, then wait for `ap_idle=1`.

2. **Ethernet PHY link delay** — KSZ9031 takes 5-15 seconds to link up after power cycle. Wait before TCP tests.

3. **Cross-domain FT gap** — richjjj FT drops CCPD accuracy 27%. Don’t fine-tune on single domain if testing cross-domain.

## Failure Analysis (17/141)

Most errors are visually similar character confusions:
- D↔0, Q↔0 (×3)
- Y↔T (×2)
- 3↔8 (×2)
- 7↔2/1/6 (×3)
- S↔A, A↔F, U↔0

All format-valid — post-processing rules (no I/O letters, position-1 must be letter) provide zero improvement because model already respects format. Need training-side fix (confusion-targeted FT, Focal loss, or deeper model).

---

_Built for ALINX FZ3A, Zynq UltraScale+ XCZU3EG. Vivado/Vitis 2024.2._
