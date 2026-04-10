// Video Filter HLS Accelerator for FZ3A
// Reads 1280x720 RGBA frame from DDR, applies filter, writes back
// Filters: 0=pass, 1=gray, 2=negative, 3=Sobel, 4=Laplacian, 5=dilate, 6=erode
// Uses m_axi for DDR access, s_axilite for control
#include <stdint.h>
#include <string.h>

#define FB_W    1280
#define FB_H    720
#define FB_SIZE (FB_W * FB_H)  // pixels (each 4 bytes RGBA)

// Filter IDs matching firmware
#define FLT_NONE      0
#define FLT_GRAY      1
#define FLT_NEGATIVE  2
#define FLT_SOBEL     3
#define FLT_LAPLACIAN 4
#define FLT_DILATE    5
#define FLT_ERODE     6

static inline uint8_t sat8(int v) {
    return v < 0 ? 0 : (v > 255 ? 255 : (uint8_t)v);
}

static inline uint8_t rgb2y(uint8_t r, uint8_t g, uint8_t b) {
    return (uint8_t)(((uint32_t)r * 76 + (uint32_t)g * 150 + (uint32_t)b * 29) >> 8);
}

extern "C" void filter_hls(
    uint32_t *src,       // m_axi: source RGBA frame in DDR
    uint32_t *dst,       // m_axi: destination RGBA frame in DDR
    int32_t   filter_id  // s_axilite: filter type
) {
#pragma HLS INTERFACE m_axi port=src offset=slave bundle=gmem0 depth=921600
#pragma HLS INTERFACE m_axi port=dst offset=slave bundle=gmem1 depth=921600
#pragma HLS INTERFACE s_axilite port=src bundle=ctrl
#pragma HLS INTERFACE s_axilite port=dst bundle=ctrl
#pragma HLS INTERFACE s_axilite port=filter_id bundle=ctrl
#pragma HLS INTERFACE s_axilite port=return bundle=ctrl

    // Line buffers for 3x3 kernels (3 rows of grayscale)
    uint8_t line[3][FB_W];
#pragma HLS ARRAY_PARTITION variable=line dim=1 complete

    if (filter_id == FLT_NONE) {
        // Passthrough: burst copy
        memcpy(dst, src, FB_SIZE * 4);
        return;
    }

    if (filter_id == FLT_GRAY) {
        GRAY_Y: for (int y = 0; y < FB_H; y++) {
            GRAY_X: for (int x = 0; x < FB_W; x++) {
                uint32_t px = src[y * FB_W + x];
                uint8_t r = px & 0xFF, g = (px >> 8) & 0xFF, b = (px >> 16) & 0xFF, a = (px >> 24) & 0xFF;
                uint8_t Y = rgb2y(r, g, b);
                dst[y * FB_W + x] = Y | (Y << 8) | (Y << 16) | ((uint32_t)a << 24);
            }
        }
        return;
    }

    if (filter_id == FLT_NEGATIVE) {
        NEG_Y: for (int y = 0; y < FB_H; y++) {
            NEG_X: for (int x = 0; x < FB_W; x++) {
                uint32_t px = src[y * FB_W + x];
                uint8_t a = (px >> 24) & 0xFF;
                dst[y * FB_W + x] = (px ^ 0x00FFFFFF) | ((uint32_t)a << 24);
            }
        }
        return;
    }

    // 3x3 kernel filters: load grayscale line buffers
    // Preload first two rows
    for (int x = 0; x < FB_W; x++) {
        uint32_t px0 = src[x];
        line[0][x] = rgb2y(px0 & 0xFF, (px0 >> 8) & 0xFF, (px0 >> 16) & 0xFF);
        uint32_t px1 = src[FB_W + x];
        line[1][x] = rgb2y(px1 & 0xFF, (px1 >> 8) & 0xFF, (px1 >> 16) & 0xFF);
    }
    // Copy first row as-is
    for (int x = 0; x < FB_W; x++) dst[x] = src[x];

    KERN_Y: for (int y = 1; y < FB_H - 1; y++) {
        int r0 = (y - 1) % 3, r1 = y % 3, r2 = (y + 1) % 3;
        // Load next row
        for (int x = 0; x < FB_W; x++) {
            uint32_t px = src[(y + 1) * FB_W + x];
            line[r2][x] = rgb2y(px & 0xFF, (px >> 8) & 0xFF, (px >> 16) & 0xFF);
        }

        // Copy border pixels
        dst[y * FB_W] = src[y * FB_W];
        dst[y * FB_W + FB_W - 1] = src[y * FB_W + FB_W - 1];

        KERN_X: for (int x = 1; x < FB_W - 1; x++) {
            uint8_t v = 0;

            if (filter_id == FLT_SOBEL) {
                int gx = -(int)line[r0][x-1] - 2*(int)line[r1][x-1] - (int)line[r2][x-1]
                        +(int)line[r0][x+1] + 2*(int)line[r1][x+1] + (int)line[r2][x+1];
                int gy =  (int)line[r0][x-1] + 2*(int)line[r0][x] + (int)line[r0][x+1]
                        - (int)line[r2][x-1] - 2*(int)line[r2][x] - (int)line[r2][x+1];
                int m = (gx < 0 ? -gx : gx) + (gy < 0 ? -gy : gy);
                v = sat8(m);
            } else if (filter_id == FLT_LAPLACIAN) {
                int c = line[r1][x];
                int lap = 5 * c - (int)line[r0][x] - (int)line[r2][x]
                        - (int)line[r1][x-1] - (int)line[r1][x+1];
                v = sat8(c + (lap << 1));
            } else if (filter_id == FLT_DILATE) {
                uint8_t mx = 0;
                for (int dy = -1; dy <= 1; dy++)
                    for (int dx = -1; dx <= 1; dx++) {
                        uint8_t val = line[(y + dy + 3) % 3][x + dx];
                        if (val > mx) mx = val;
                    }
                v = mx;
            } else if (filter_id == FLT_ERODE) {
                uint8_t mn = 255;
                for (int dy = -1; dy <= 1; dy++)
                    for (int dx = -1; dx <= 1; dx++) {
                        uint8_t val = line[(y + dy + 3) % 3][x + dx];
                        if (val < mn) mn = val;
                    }
                v = mn;
            }

            dst[y * FB_W + x] = v | (v << 8) | (v << 16) | 0xFF000000;
        }
    }
    // Copy last row
    for (int x = 0; x < FB_W; x++) dst[(FB_H-1) * FB_W + x] = src[(FB_H-1) * FB_W + x];
}
