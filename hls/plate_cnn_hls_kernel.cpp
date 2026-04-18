/* PlateCNN HLS Accelerator — Full PL End-to-End CNN
 *
 * Architecture:
 *   Input:  128x32 u8 image (4096 bytes)
 *   Conv1 (1→32 ch, 3x3) + ReLU + MaxPool2x2 → (32, 16, 64)
 *   Conv2 (32→64)         + ReLU + MaxPool2x2 → (64, 8, 32)
 *   Conv3 (64→128)        + ReLU + MaxPool2x2 → (128, 4, 16)
 *   Conv4 (128→256)       + ReLU + MaxPool2x2 → (256, 2, 8) = 4096 features
 *   FC (4096→512) + ReLU  (weights from DDR via m_axi)
 *   Head CN (512→31) → argmax → prov (0..30)
 *   Heads AL × 6 (512→36) → argmax → al[0..5] (0..35 each)
 *
 * Output: 7 bytes (prov + 6 alnum indices)
 *
 * Per-channel INT8 weights + FP32 scales for accurate quantization.
 * Conv+Heads weights in BRAM (~500KB), FC weights streamed from DDR (2MB).
 */

#include <stdint.h>
#include <ap_int.h>
#include "plate_cnn_hls_weights.h"

#define IN_H 32
#define IN_W 128

#define C1_OC 32
#define C1_OH 16
#define C1_OW 64

#define C2_OC 64
#define C2_OH 8
#define C2_OW 32

#define C3_OC 128
#define C3_OH 4
#define C3_OW 16

#define C4_OC 256
#define C4_OH 2
#define C4_OW 8

#define FC_IN 4096
#define FC_OUT 512

#define CN_CLASSES 31
#define AL_CLASSES 36
#define N_AL_HEADS 6

/* Quantize + ReLU helpers */
static inline int8_t quant_relu_int8(float v, float scale) {
    if (v < 0.0f) v = 0.0f;
    float q = v / scale;
    if (q > 127.0f) q = 127.0f;
    if (q < -128.0f) q = -128.0f;
    return (int8_t)(int)(q + 0.5f);
}

static inline int8_t quant_int8(float v, float scale) {
    float q = v / scale;
    if (q > 127.0f) q = 127.0f;
    if (q < -128.0f) q = -128.0f;
    return (int8_t)(int)(q + (v >= 0 ? 0.5f : -0.5f));
}

extern "C" void plate_cnn_hls(
    uint8_t  image[IN_H * IN_W],             // AXI-Lite write (4096 bytes)
    uint64_t fc_weights_addr,                 // AXI-Lite: DDR address of FC weights (int8 512x4096)
    const int8_t *fc_weights_ddr,             // m_axi pointer to DDR FC weights
    uint8_t  predictions[7]                   // AXI-Lite read: [prov, al0..al5]
) {
#pragma HLS INTERFACE s_axilite port=image            bundle=ctrl
#pragma HLS INTERFACE s_axilite port=fc_weights_addr  bundle=ctrl
#pragma HLS INTERFACE s_axilite port=predictions      bundle=ctrl
#pragma HLS INTERFACE m_axi     port=fc_weights_ddr   offset=slave bundle=gmem depth=2097152 max_read_burst_length=256
#pragma HLS INTERFACE s_axilite port=fc_weights_ddr   bundle=ctrl
#pragma HLS INTERFACE s_axilite port=return           bundle=ctrl

    /* Activation buffers (INT8 intermediate). Use ping-pong where sizes differ. */
    /* Largest is conv1 output: 32*16*64 = 32K int8 = 32KB */
    static int8_t buf_a[32 * 16 * 64];
    static int8_t buf_b[32 * 16 * 64];
#pragma HLS BIND_STORAGE variable=buf_a type=RAM_S2P impl=BRAM
#pragma HLS BIND_STORAGE variable=buf_b type=RAM_S2P impl=BRAM

    /* ============================================================
     * Conv1: uint8 image → int8 (32, 16, 64)
     * Input scale: 1/255 (image treated as [0,1])
     * ============================================================ */
    CONV1_OC: for (int oc = 0; oc < C1_OC; oc++) {
        float bi = pcn_c1_b[oc];
        float sw = pcn_c1_s[oc];
        CONV1_OH: for (int oh = 0; oh < C1_OH; oh++) {
            CONV1_OW: for (int ow = 0; ow < C1_OW; ow++) {
#pragma HLS PIPELINE II=1
                /* Compute 4 conv outputs for 2x2 pool window, take max after ReLU */
                float best = 0.0f;
                for (int dy = 0; dy < 2; dy++) {
                    for (int dx = 0; dx < 2; dx++) {
                        int y = 2*oh + dy;
                        int x = 2*ow + dx;
                        int32_t acc = 0;
                        for (int ky = 0; ky < 3; ky++) {
                            int iy = y + ky - 1;
                            if (iy < 0 || iy >= IN_H) continue;
                            for (int kx = 0; kx < 3; kx++) {
#pragma HLS UNROLL
                                int ix = x + kx - 1;
                                if (ix < 0 || ix >= IN_W) continue;
                                int k = ky * 3 + kx;
                                acc += (int32_t)image[iy * IN_W + ix] * (int32_t)pcn_c1_w[oc][k];
                            }
                        }
                        /* Real value = acc * (1/255) * sw + bias */
                        float real = (float)acc * (pcn_s_in * sw) + bi;
                        if (real < 0.0f) real = 0.0f;
                        if (real > best) best = real;
                    }
                }
                buf_a[oc * (C1_OH * C1_OW) + oh * C1_OW + ow] = quant_int8(best, pcn_s_a1);
            }
        }
    }

    /* ============================================================
     * Conv2: int8 (32, 16, 64) → int8 (64, 8, 32)
     * ============================================================ */
    CONV2_OC: for (int oc = 0; oc < C2_OC; oc++) {
        float bi = pcn_c2_b[oc];
        float sw = pcn_c2_s[oc];
        float scale_comb = pcn_s_a1 * sw;
        CONV2_OH: for (int oh = 0; oh < C2_OH; oh++) {
            CONV2_OW: for (int ow = 0; ow < C2_OW; ow++) {
                float best = 0.0f;
                for (int dy = 0; dy < 2; dy++) {
                    for (int dx = 0; dx < 2; dx++) {
                        int y = 2*oh + dy;
                        int x = 2*ow + dx;
                        int32_t acc = 0;
                        for (int ic = 0; ic < C1_OC; ic++) {
#pragma HLS PIPELINE II=1
                            for (int ky = 0; ky < 3; ky++) {
                                int iy = y + ky - 1;
                                if (iy < 0 || iy >= C1_OH) continue;
                                for (int kx = 0; kx < 3; kx++) {
#pragma HLS UNROLL
                                    int ix = x + kx - 1;
                                    if (ix < 0 || ix >= C1_OW) continue;
                                    int k = ky * 3 + kx;
                                    acc += (int32_t)buf_a[ic * (C1_OH * C1_OW) + iy * C1_OW + ix] *
                                           (int32_t)pcn_c2_w[oc][ic][k];
                                }
                            }
                        }
                        float real = (float)acc * scale_comb + bi;
                        if (real < 0.0f) real = 0.0f;
                        if (real > best) best = real;
                    }
                }
                buf_b[oc * (C2_OH * C2_OW) + oh * C2_OW + ow] = quant_int8(best, pcn_s_a2);
            }
        }
    }

    /* ============================================================
     * Conv3: int8 (64, 8, 32) → int8 (128, 4, 16). Output to buf_a.
     * ============================================================ */
    CONV3_OC: for (int oc = 0; oc < C3_OC; oc++) {
        float bi = pcn_c3_b[oc];
        float sw = pcn_c3_s[oc];
        float scale_comb = pcn_s_a2 * sw;
        CONV3_OH: for (int oh = 0; oh < C3_OH; oh++) {
            CONV3_OW: for (int ow = 0; ow < C3_OW; ow++) {
                float best = 0.0f;
                for (int dy = 0; dy < 2; dy++) {
                    for (int dx = 0; dx < 2; dx++) {
                        int y = 2*oh + dy;
                        int x = 2*ow + dx;
                        int32_t acc = 0;
                        for (int ic = 0; ic < C2_OC; ic++) {
#pragma HLS PIPELINE II=1
                            for (int ky = 0; ky < 3; ky++) {
                                int iy = y + ky - 1;
                                if (iy < 0 || iy >= C2_OH) continue;
                                for (int kx = 0; kx < 3; kx++) {
#pragma HLS UNROLL
                                    int ix = x + kx - 1;
                                    if (ix < 0 || ix >= C2_OW) continue;
                                    int k = ky * 3 + kx;
                                    acc += (int32_t)buf_b[ic * (C2_OH * C2_OW) + iy * C2_OW + ix] *
                                           (int32_t)pcn_c3_w[oc][ic][k];
                                }
                            }
                        }
                        float real = (float)acc * scale_comb + bi;
                        if (real < 0.0f) real = 0.0f;
                        if (real > best) best = real;
                    }
                }
                buf_a[oc * (C3_OH * C3_OW) + oh * C3_OW + ow] = quant_int8(best, pcn_s_a3);
            }
        }
    }

    /* ============================================================
     * Conv4: int8 (128, 4, 16) → int8 (256, 2, 8) = 4096 features.
     * Output to buf_b (treated as flat 4096 array for FC).
     * ============================================================ */
    CONV4_OC: for (int oc = 0; oc < C4_OC; oc++) {
        float bi = pcn_c4_b[oc];
        float sw = pcn_c4_s[oc];
        float scale_comb = pcn_s_a3 * sw;
        CONV4_OH: for (int oh = 0; oh < C4_OH; oh++) {
            CONV4_OW: for (int ow = 0; ow < C4_OW; ow++) {
                float best = 0.0f;
                for (int dy = 0; dy < 2; dy++) {
                    for (int dx = 0; dx < 2; dx++) {
                        int y = 2*oh + dy;
                        int x = 2*ow + dx;
                        int32_t acc = 0;
                        for (int ic = 0; ic < C3_OC; ic++) {
#pragma HLS PIPELINE II=1
                            for (int ky = 0; ky < 3; ky++) {
                                int iy = y + ky - 1;
                                if (iy < 0 || iy >= C3_OH) continue;
                                for (int kx = 0; kx < 3; kx++) {
#pragma HLS UNROLL
                                    int ix = x + kx - 1;
                                    if (ix < 0 || ix >= C3_OW) continue;
                                    int k = ky * 3 + kx;
                                    acc += (int32_t)buf_a[ic * (C3_OH * C3_OW) + iy * C3_OW + ix] *
                                           (int32_t)pcn_c4_w[oc][ic][k];
                                }
                            }
                        }
                        float real = (float)acc * scale_comb + bi;
                        if (real < 0.0f) real = 0.0f;
                        if (real > best) best = real;
                    }
                }
                /* Output to buf_b in C4 layout: [oc][oh][ow] flattened = (256*2*8) = 4096 */
                int idx = oc * (C4_OH * C4_OW) + oh * C4_OW + ow;
                buf_b[idx] = quant_int8(best, pcn_s_a4);
            }
        }
    }

    /* ============================================================
     * FC: int8 [4096] → int8 [512]
     * Weights (int8 [512][4096]) STREAMED from DDR via m_axi burst.
     * buf_b has conv4 output as int8 [4096].
     * Result stored in fc_out[512] (float32 for head computation).
     *
     * IMPORTANT: FC weight layout in DDR file `plate_cnn_fc_weights.bin`:
     *   Header (16 bytes): "PCFC" + OC(u32) + IC(u32)
     *   Data:    int8 [512][4096] row-major
     *   Scales:  float32 [512]
     *   Biases:  float32 [512]
     *
     * Firmware loads this file to DDR and passes base addr via fc_weights_addr.
     * Scales and biases are embedded here (since they fit in BRAM).
     *
     * For simplicity: assume scales come after the 2MB weights at offset 2MB,
     *   biases at offset 2MB + 2KB. Firmware sets pointer to start of weights.
     *   Scales/biases are handled as part of the DDR data stream.
     *
     * Actually: to keep HLS simple, store FC scales+biases in BRAM header.
     * Firmware provides only raw weights pointer.
     */
    /* pcn_fc_s[512] and pcn_fc_b[512] are in weights header (BRAM) */
    float fc_out[FC_OUT];

    /* For each output neuron: burst-read one row of weights, compute dot product */
    FC_O: for (int o = 0; o < FC_OUT; o++) {
        int8_t w_row[FC_IN];
#pragma HLS BIND_STORAGE variable=w_row type=RAM_1P impl=BRAM
        /* Burst read 4096 int8 = 4KB */
        FC_LOAD: for (int i = 0; i < FC_IN; i++) {
#pragma HLS PIPELINE II=1
            w_row[i] = fc_weights_ddr[o * FC_IN + i];
        }
        /* Dot product with int8 activation buf_b */
        int32_t acc = 0;
        FC_DOT: for (int i = 0; i < FC_IN; i++) {
#pragma HLS PIPELINE II=1
            acc += (int32_t)buf_b[i] * (int32_t)w_row[i];
        }
        float real = (float)acc * pcn_s_a4 * pcn_fc_s[o] + pcn_fc_b[o];
        if (real < 0.0f) real = 0.0f;  // ReLU
        fc_out[o] = real;
    }

    /* ============================================================
     * Head CN: 512 → 31 classes, argmax
     * ============================================================ */
    float cn_best = -1e30f;
    int cn_idx = 0;
    HEAD_CN: for (int o = 0; o < CN_CLASSES; o++) {
        float acc = pcn_hcn_b[o];
        float sw = pcn_hcn_s[o];
        for (int i = 0; i < FC_OUT; i++) {
#pragma HLS PIPELINE II=1
            /* FC output is float, weights are int8 + per-channel scale */
            acc += fc_out[i] * ((float)pcn_hcn_w[o][i] * sw);
        }
        if (acc > cn_best) { cn_best = acc; cn_idx = o; }
    }
    predictions[0] = (uint8_t)cn_idx;

    /* ============================================================
     * Heads AL × 6: each 512 → 36 classes, argmax
     * ============================================================ */
    HEAD_AL: for (int h = 0; h < N_AL_HEADS; h++) {
        float best = -1e30f;
        int best_idx = 0;
        for (int o = 0; o < AL_CLASSES; o++) {
            float acc = pcn_hal_b[h][o];
            float sw = pcn_hal_s[h][o];
            for (int i = 0; i < FC_OUT; i++) {
#pragma HLS PIPELINE II=1
                acc += fc_out[i] * ((float)pcn_hal_w[h][o][i] * sw);
            }
            if (acc > best) { best = acc; best_idx = o; }
        }
        predictions[1 + h] = (uint8_t)best_idx;
    }
}
