// PlateCNN HLS Accelerator for FZ3A
// Architecture: 4x (Conv3x3 + ReLU + MaxPool2x2)
//   Input:  128x32 u8 (4096 bytes)
//   Conv1: (1,32,128) -> (32,32,128) -> pool -> (32,16,64)
//   Conv2: (32,16,64) -> (64,16,64)  -> pool -> (64,8,32)
//   Conv3: (64,8,32)  -> (128,8,32)  -> pool -> (128,4,16)
//   Conv4: (128,4,16) -> (256,4,16)  -> pool -> (256,2,8)
//   Output: 256x2x8 = 4096 int8 features
//
// All weights INT8, fold BN into conv bias.
// AXI-Lite: PS writes image, reads output features.
// Expected resource: ~110 BRAM18, ~40 DSP48, ~15K LUT
// Expected latency: ~5-15ms @ 100MHz

#include <stdint.h>
#include <ap_int.h>
#include "plate_cnn_hls_weights.h"

#define IN_H 32
#define IN_W 128

// Dimensions per layer output (after pool)
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

#define OUT_SIZE (C4_OC * C4_OH * C4_OW)  // 4096

/* Bias/weight scale shift for requantization (keeps int32 intermediate in range) */
#define REQUANT_SHIFT 10

static int16_t clamp16(int32_t v) {
    if (v > 32767) return 32767;
    if (v < -32768) return -32768;
    return (int16_t)v;
}

static int8_t clamp8(int32_t v) {
    if (v > 127) return 127;
    if (v < -128) return -128;
    return (int8_t)v;
}

/* Conv3x3 + ReLU + MaxPool2x2 pipeline layer template.
 * Input:  in[Cin][Hi][Wi] (int16)
 * Output: out[Co][Ho][Wo] (int16) where Ho = Hi/2, Wo = Wi/2
 * Weights: W[Co][Cin][3][3] (int8), bias[Co] (int32 pre-scaled)
 */

extern "C" void plate_cnn_hls(
    uint8_t  image[IN_H * IN_W],       // 4096 bytes, AXI-Lite
    int8_t   out_features[OUT_SIZE]    // 4096 int8, AXI-Lite
) {
#pragma HLS INTERFACE s_axilite port=image        bundle=ctrl
#pragma HLS INTERFACE s_axilite port=out_features bundle=ctrl
#pragma HLS INTERFACE s_axilite port=return       bundle=ctrl

    /* Activation buffers (ping-pong). Sizes match max needed. */
    static int16_t buf_a[C1_OC][C1_OH][C1_OW];  // 32*16*64 = 32K int16 = 64KB
    static int16_t buf_b[C2_OC][C2_OH][C2_OW];  // 64*8*32 = 16K
#pragma HLS BIND_STORAGE variable=buf_a type=RAM_S2P impl=BRAM
#pragma HLS BIND_STORAGE variable=buf_b type=RAM_S2P impl=BRAM

    /* ===================== Conv1: (1, 32, 128) -> (32, 16, 64) ===================== */
    /* Directly process input uint8; output to buf_a */
    CONV1_OC: for (int oc = 0; oc < C1_OC; oc++) {
        CONV1_OH: for (int oh = 0; oh < C1_OH; oh++) {
            CONV1_OW: for (int ow = 0; ow < C1_OW; ow++) {
                /* Compute 2x2 conv outputs, take max (fused ReLU+Pool) */
                int16_t best = 0;
                for (int dy = 0; dy < 2; dy++) {
                    for (int dx = 0; dx < 2; dx++) {
                        int y = 2*oh + dy;
                        int x = 2*ow + dx;
                        int32_t acc = conv1_b[oc];
                        for (int ky = 0; ky < 3; ky++) {
                            int iy = y + ky - 1;
                            if (iy < 0 || iy >= IN_H) continue;
                            for (int kx = 0; kx < 3; kx++) {
                                int ix = x + kx - 1;
                                if (ix < 0 || ix >= IN_W) continue;
                                int w_idx = oc*9 + ky*3 + kx;
                                acc += (int32_t)image[iy*IN_W + ix] * (int32_t)conv1_w[w_idx];
                            }
                        }
                        int16_t v = clamp16(acc >> REQUANT_SHIFT);
                        if (v > best) best = v;  /* ReLU+Max fused */
                    }
                }
                buf_a[oc][oh][ow] = best;
            }
        }
    }

    /* ===================== Conv2: (32, 16, 64) -> (64, 8, 32) ===================== */
    CONV2_OC: for (int oc = 0; oc < C2_OC; oc++) {
        CONV2_OH: for (int oh = 0; oh < C2_OH; oh++) {
            CONV2_OW: for (int ow = 0; ow < C2_OW; ow++) {
                int16_t best = 0;
                for (int dy = 0; dy < 2; dy++) {
                    for (int dx = 0; dx < 2; dx++) {
                        int y = 2*oh + dy;
                        int x = 2*ow + dx;
                        int32_t acc = conv2_b[oc];
                        for (int ic = 0; ic < C1_OC; ic++) {
                            for (int ky = 0; ky < 3; ky++) {
                                int iy = y + ky - 1;
                                if (iy < 0 || iy >= C1_OH) continue;
                                for (int kx = 0; kx < 3; kx++) {
                                    int ix = x + kx - 1;
                                    if (ix < 0 || ix >= C1_OW) continue;
                                    int w_idx = oc*(C1_OC*9) + ic*9 + ky*3 + kx;
                                    acc += (int32_t)buf_a[ic][iy][ix] * (int32_t)conv2_w[w_idx];
                                }
                            }
                        }
                        int16_t v = clamp16(acc >> REQUANT_SHIFT);
                        if (v > best) best = v;
                    }
                }
                buf_b[oc][oh][ow] = best;
            }
        }
    }

    /* ===================== Conv3: (64, 8, 32) -> (128, 4, 16) ===================== */
    /* Store output back in buf_a (reuse for pingpong). Need to extend buf_a to 128*4*16 */
    /* Since C3_OC=128 > C1_OC=32, need bigger buffer. Use separate buf_c. */
    static int16_t buf_c[C3_OC][C3_OH][C3_OW];  // 128*4*16 = 8K int16 = 16KB
#pragma HLS BIND_STORAGE variable=buf_c type=RAM_S2P impl=BRAM

    CONV3_OC: for (int oc = 0; oc < C3_OC; oc++) {
        CONV3_OH: for (int oh = 0; oh < C3_OH; oh++) {
            CONV3_OW: for (int ow = 0; ow < C3_OW; ow++) {
                int16_t best = 0;
                for (int dy = 0; dy < 2; dy++) {
                    for (int dx = 0; dx < 2; dx++) {
                        int y = 2*oh + dy;
                        int x = 2*ow + dx;
                        int32_t acc = conv3_b[oc];
                        for (int ic = 0; ic < C2_OC; ic++) {
                            for (int ky = 0; ky < 3; ky++) {
                                int iy = y + ky - 1;
                                if (iy < 0 || iy >= C2_OH) continue;
                                for (int kx = 0; kx < 3; kx++) {
                                    int ix = x + kx - 1;
                                    if (ix < 0 || ix >= C2_OW) continue;
                                    int w_idx = oc*(C2_OC*9) + ic*9 + ky*3 + kx;
                                    acc += (int32_t)buf_b[ic][iy][ix] * (int32_t)conv3_w[w_idx];
                                }
                            }
                        }
                        int16_t v = clamp16(acc >> REQUANT_SHIFT);
                        if (v > best) best = v;
                    }
                }
                buf_c[oc][oh][ow] = best;
            }
        }
    }

    /* ===================== Conv4: (128, 4, 16) -> (256, 2, 8) ===================== */
    /* Output directly to out_features array (flattened) */
    CONV4_OC: for (int oc = 0; oc < C4_OC; oc++) {
        CONV4_OH: for (int oh = 0; oh < C4_OH; oh++) {
            CONV4_OW: for (int ow = 0; ow < C4_OW; ow++) {
                int32_t best = 0;
                for (int dy = 0; dy < 2; dy++) {
                    for (int dx = 0; dx < 2; dx++) {
                        int y = 2*oh + dy;
                        int x = 2*ow + dx;
                        int32_t acc = conv4_b[oc];
                        for (int ic = 0; ic < C3_OC; ic++) {
                            for (int ky = 0; ky < 3; ky++) {
                                int iy = y + ky - 1;
                                if (iy < 0 || iy >= C3_OH) continue;
                                for (int kx = 0; kx < 3; kx++) {
                                    int ix = x + kx - 1;
                                    if (ix < 0 || ix >= C3_OW) continue;
                                    int w_idx = oc*(C3_OC*9) + ic*9 + ky*3 + kx;
                                    acc += (int32_t)buf_c[ic][iy][ix] * (int32_t)conv4_w[w_idx];
                                }
                            }
                        }
                        int32_t v = acc >> REQUANT_SHIFT;
                        if (v < 0) v = 0;
                        if (v > best) best = v;
                    }
                }
                /* Output as int8 for PS side use */
                out_features[oc*(C4_OH*C4_OW) + oh*C4_OW + ow] = clamp8(best);
            }
        }
    }
}
