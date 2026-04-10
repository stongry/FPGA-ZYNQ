// TinyLeNet CNN HLS Accelerator for FZ3A
// Architecture: Conv1(1->8,5x5)->ReLU->Pool->Conv2(8->16,5x5)->ReLU->Pool->FC1(256->64)->ReLU->FC2(64->10)->Argmax
// Weights embedded as constants, INT8 quantized
// AXI-Lite interface: PS writes 784-byte image, triggers start, reads 10 scores + prediction
#include <stdint.h>
#include <ap_int.h>
#include "cnn_hls_weights.h"

#define IN_W  28
#define IN_H  28

// Conv1 output: 8 x 24 x 24
#define C1_OC 8
#define C1_K  5
#define C1_OW 24
#define C1_OH 24

// Pool1 output: 8 x 12 x 12
#define P1_OW 12
#define P1_OH 12

// Conv2 output: 16 x 8 x 8
#define C2_OC 16
#define C2_IC 8
#define C2_K  5
#define C2_OW 8
#define C2_OH 8

// Pool2 output: 16 x 4 x 4 = 256
#define P2_OW 4
#define P2_OH 4
#define FC1_IN 256

#define FC1_OUT 64
#define FC2_OUT 10

#define REQUANT_SHIFT 8

static int16_t clamp16(int32_t v) {
    if (v > 32767) return 32767;
    if (v < -32768) return -32768;
    return (int16_t)v;
}

extern "C" void cnn_tinylenet_hls(
    uint8_t image[784],
    int32_t scores_out[10],
    int32_t *pred_out
) {
#pragma HLS INTERFACE s_axilite port=image      bundle=ctrl
#pragma HLS INTERFACE s_axilite port=scores_out bundle=ctrl
#pragma HLS INTERFACE s_axilite port=pred_out   bundle=ctrl
#pragma HLS INTERFACE s_axilite port=return     bundle=ctrl

    // ---- Layer buffers ----
    int16_t conv1_out[C1_OC][C1_OH][C1_OW];
    int16_t pool1_out[C1_OC][P1_OH][P1_OW];
    int16_t conv2_out[C2_OC][C2_OH][C2_OW];
    int16_t pool2_out[C2_OC][P2_OH][P2_OW];
    int16_t fc1_out[FC1_OUT];
    int32_t fc2_raw[FC2_OUT];

    // ============================================================
    // Conv1: 1 x 28 x 28 -> 8 x 24 x 24
    // ============================================================
    CONV1_OC: for (int oc = 0; oc < C1_OC; oc++) {
        CONV1_OH: for (int oh = 0; oh < C1_OH; oh++) {
            CONV1_OW: for (int ow = 0; ow < C1_OW; ow++) {
// no forced pipeline - let HLS decide
                int32_t acc = conv1_b[oc];
                for (int kh = 0; kh < C1_K; kh++) {
                    for (int kw = 0; kw < C1_K; kw++) {
                        int idx_in = (oh + kh) * IN_W + (ow + kw);
                        int idx_w = oc * (C1_K * C1_K) + kh * C1_K + kw;
                        acc += (int32_t)image[idx_in] * (int32_t)conv1_w[idx_w];
                    }
                }
                int16_t val = clamp16(acc >> REQUANT_SHIFT);
                conv1_out[oc][oh][ow] = (val > 0) ? val : (int16_t)0; // ReLU
            }
        }
    }

    // ============================================================
    // MaxPool1: 8 x 24 x 24 -> 8 x 12 x 12
    // ============================================================
    POOL1_OC: for (int oc = 0; oc < C1_OC; oc++) {
        POOL1_OH: for (int oh = 0; oh < P1_OH; oh++) {
            POOL1_OW: for (int ow = 0; ow < P1_OW; ow++) {
// no forced pipeline - let HLS decide
                int16_t mx = conv1_out[oc][oh*2][ow*2];
                if (conv1_out[oc][oh*2][ow*2+1] > mx) mx = conv1_out[oc][oh*2][ow*2+1];
                if (conv1_out[oc][oh*2+1][ow*2] > mx) mx = conv1_out[oc][oh*2+1][ow*2];
                if (conv1_out[oc][oh*2+1][ow*2+1] > mx) mx = conv1_out[oc][oh*2+1][ow*2+1];
                pool1_out[oc][oh][ow] = mx;
            }
        }
    }

    // ============================================================
    // Conv2: 8 x 12 x 12 -> 16 x 8 x 8
    // ============================================================
    CONV2_OC: for (int oc = 0; oc < C2_OC; oc++) {
        CONV2_OH: for (int oh = 0; oh < C2_OH; oh++) {
            CONV2_OW: for (int ow = 0; ow < C2_OW; ow++) {
// no forced pipeline - let HLS decide
                int32_t acc = conv2_b[oc];
                for (int ic = 0; ic < C2_IC; ic++) {
                    for (int kh = 0; kh < C2_K; kh++) {
                        for (int kw = 0; kw < C2_K; kw++) {
                            int16_t pixel = pool1_out[ic][oh + kh][ow + kw];
                            int idx_w = ((oc * C2_IC + ic) * C2_K + kh) * C2_K + kw;
                            acc += (int32_t)pixel * (int32_t)conv2_w[idx_w];
                        }
                    }
                }
                int16_t val = clamp16(acc >> REQUANT_SHIFT);
                conv2_out[oc][oh][ow] = (val > 0) ? val : (int16_t)0; // ReLU
            }
        }
    }

    // ============================================================
    // MaxPool2: 16 x 8 x 8 -> 16 x 4 x 4
    // ============================================================
    POOL2_OC: for (int oc = 0; oc < C2_OC; oc++) {
        POOL2_OH: for (int oh = 0; oh < P2_OH; oh++) {
            POOL2_OW: for (int ow = 0; ow < P2_OW; ow++) {
// no forced pipeline - let HLS decide
                int16_t mx = conv2_out[oc][oh*2][ow*2];
                if (conv2_out[oc][oh*2][ow*2+1] > mx) mx = conv2_out[oc][oh*2][ow*2+1];
                if (conv2_out[oc][oh*2+1][ow*2] > mx) mx = conv2_out[oc][oh*2+1][ow*2];
                if (conv2_out[oc][oh*2+1][ow*2+1] > mx) mx = conv2_out[oc][oh*2+1][ow*2+1];
                pool2_out[oc][oh][ow] = mx;
            }
        }
    }

    // ============================================================
    // FC1: 256 -> 64 + ReLU
    // ============================================================
    FC1: for (int j = 0; j < FC1_OUT; j++) {
// no forced pipeline - let HLS decide
        int32_t acc = fc1_b[j];
        for (int i = 0; i < FC1_IN; i++) {
            // Flatten pool2_out: [oc][oh][ow] -> index i
            int oc = i / (P2_OH * P2_OW);
            int rem = i % (P2_OH * P2_OW);
            int oh = rem / P2_OW;
            int ow = rem % P2_OW;
            int16_t pixel = pool2_out[oc][oh][ow];
            acc += (int32_t)pixel * (int32_t)fc1_w[j * FC1_IN + i];
        }
        int16_t val = clamp16(acc >> REQUANT_SHIFT);
        fc1_out[j] = (val > 0) ? val : (int16_t)0; // ReLU
    }

    // ============================================================
    // FC2: 64 -> 10 (no ReLU)
    // ============================================================
    FC2: for (int k = 0; k < FC2_OUT; k++) {
// no forced pipeline - let HLS decide
        int32_t acc = fc2_b[k];
        for (int j = 0; j < FC1_OUT; j++) {
            acc += (int32_t)fc1_out[j] * (int32_t)fc2_w[k * FC1_OUT + j];
        }
        fc2_raw[k] = acc;
        scores_out[k] = acc;
    }

    // ============================================================
    // Argmax
    // ============================================================
    int best = 0;
    for (int k = 1; k < FC2_OUT; k++) {
        if (fc2_raw[k] > fc2_raw[best]) best = k;
    }
    *pred_out = best;
}
