// HOG + Linear SVM Pedestrian Detector HLS Accelerator v2
// Uses m_axi for image input (DDR DMA), s_axilite for control + results
// Input: 320x240 grayscale image in DDR
// Output: up to 16 detection boxes (x, y, score)
#include <stdint.h>
#include <string.h>
#include "ped_hls_weights.h"

#define IMG_W  320
#define IMG_H  240
#define CELL   8
#define BINS   9

#define N_CX  (IMG_W / CELL)   // 40
#define N_CY  (IMG_H / CELL)   // 30

// Detection window: 64x128 in cells = 8x16
#define WIN_CX  8
#define WIN_CY  16
#define BLOCK   2

#define WIN_BX  (WIN_CX - BLOCK + 1)  // 7
#define WIN_BY  (WIN_CY - BLOCK + 1)  // 15
#define FEAT_DIM (WIN_BX * WIN_BY * BLOCK * BLOCK * BINS) // 3780

#define N_WIN_X (N_CX - WIN_CX + 1)   // 33
#define N_WIN_Y (N_CY - WIN_CY + 1)   // 15

#define MAX_DETS 16

static int16_t abs16(int16_t v) { return v < 0 ? -v : v; }

extern "C" void ped_detect_hls(
    const uint8_t *image,              // m_axi: read from DDR
    int32_t det_out[MAX_DETS * 2],     // s_axilite: detection results
    int32_t *num_dets_out,             // s_axilite: number of detections
    int32_t threshold                  // s_axilite: SVM threshold
) {
#pragma HLS INTERFACE m_axi port=image offset=slave bundle=gmem depth=76800
#pragma HLS INTERFACE s_axilite port=image bundle=ctrl
#pragma HLS INTERFACE s_axilite port=det_out bundle=ctrl
#pragma HLS INTERFACE s_axilite port=num_dets_out bundle=ctrl
#pragma HLS INTERFACE s_axilite port=threshold bundle=ctrl
#pragma HLS INTERFACE s_axilite port=return bundle=ctrl

    // Local copy of image from DDR
    uint8_t img_local[IMG_H * IMG_W];
    memcpy(img_local, image, IMG_H * IMG_W);

    // ================================================================
    // Step 1: Compute gradients and cell histograms
    // ================================================================
    int32_t hist[N_CY][N_CX][BINS];

    for (int cy = 0; cy < N_CY; cy++)
        for (int cx = 0; cx < N_CX; cx++)
            for (int b = 0; b < BINS; b++)
                hist[cy][cx][b] = 0;

    GRAD_Y: for (int y = 1; y < IMG_H - 1; y++) {
        GRAD_X: for (int x = 1; x < IMG_W - 1; x++) {
            int16_t gx = (int16_t)img_local[y * IMG_W + x + 1] - (int16_t)img_local[y * IMG_W + x - 1];
            int16_t gy = (int16_t)img_local[(y + 1) * IMG_W + x] - (int16_t)img_local[(y - 1) * IMG_W + x];

            int16_t mag = abs16(gx) + abs16(gy);
            int16_t ax = abs16(gx);
            int16_t ay = abs16(gy);

            int bin;
            if (mag < 4) {
                bin = 0;
            } else if (ax > (ay << 1)) {
                bin = (gx > 0) ? 0 : 8;
            } else if (ay > (ax << 1)) {
                bin = 4;
            } else if (gx > 0) {
                bin = (gy > 0) ? 2 : 6;
            } else {
                bin = (gy > 0) ? 6 : 2;
            }

            int cy = y / CELL;
            int cx = x / CELL;
            hist[cy][cx][bin] += mag;
        }
    }

    // ================================================================
    // Step 2: Sliding window SVM
    // ================================================================
    int n_dets = 0;
    int32_t det_buf[MAX_DETS * 2];

    WIN_Y: for (int wy = 0; wy < N_WIN_Y; wy++) {
        WIN_X: for (int wx = 0; wx < N_WIN_X; wx++) {
            int32_t svm_acc = SVM_BIAS_Q;
            int feat_idx = 0;

            BLK_Y: for (int by = 0; by < WIN_BY; by++) {
                BLK_X: for (int bx = 0; bx < WIN_BX; bx++) {
                    for (int dy = 0; dy < BLOCK; dy++) {
                        for (int dx = 0; dx < BLOCK; dx++) {
                            for (int b = 0; b < BINS; b++) {
                                int32_t cell_val = hist[wy + by + dy][wx + bx + dx][b];
                                int32_t norm_val = cell_val >> 4;
                                if (norm_val > 255) norm_val = 255;
                                svm_acc += norm_val * (int32_t)svm_weights[feat_idx];
                                feat_idx++;
                            }
                        }
                    }
                }
            }

            if (svm_acc > threshold && n_dets < MAX_DETS) {
                int px = wx * CELL;
                int py = wy * CELL;
                det_buf[n_dets * 2]     = (py << 8) | px;
                det_buf[n_dets * 2 + 1] = svm_acc;
                n_dets++;
            }
        }
    }

    for (int i = 0; i < MAX_DETS * 2; i++)
        det_out[i] = (i < n_dets * 2) ? det_buf[i] : 0;
    *num_dets_out = n_dets;
}
