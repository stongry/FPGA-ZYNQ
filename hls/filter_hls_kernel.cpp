// Video Filter HLS Accelerator v2 - Burst-optimized
// Key optimizations:
//   1. Burst read entire rows (1280 pixels) for better DDR bandwidth
//   2. Pipeline inner pixel loop II=1
//   3. Line buffer partitioned for parallel access
#include <stdint.h>
#include <string.h>

#define FB_W    1280
#define FB_H    720

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

extern "C" void filter_hls(
    uint32_t *src,
    uint32_t *dst,
    int32_t   filter_id
) {
#pragma HLS INTERFACE m_axi port=src offset=slave bundle=gmem0 depth=921600 max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=dst offset=slave bundle=gmem1 depth=921600 max_write_burst_length=64
#pragma HLS INTERFACE s_axilite port=src bundle=ctrl
#pragma HLS INTERFACE s_axilite port=dst bundle=ctrl
#pragma HLS INTERFACE s_axilite port=filter_id bundle=ctrl
#pragma HLS INTERFACE s_axilite port=return bundle=ctrl

    if (filter_id == FLT_NONE) {
        memcpy(dst, src, FB_W * FB_H * 4);
        return;
    }

    // Row buffers for burst read
    uint32_t row_buf[FB_W];
#pragma HLS BIND_STORAGE variable=row_buf type=ram_1p

    if (filter_id == FLT_GRAY || filter_id == FLT_NEGATIVE) {
        // Simple per-pixel: burst read row, process, burst write
        SIMPLE_Y: for (int y = 0; y < FB_H; y++) {
            // Burst read
            memcpy(row_buf, &src[y * FB_W], FB_W * 4);
            // Process
            SIMPLE_X: for (int x = 0; x < FB_W; x++) {
#pragma HLS PIPELINE II=1
                uint32_t px = row_buf[x];
                if (filter_id == FLT_GRAY) {
                    uint8_t r = px & 0xFF, g = (px >> 8) & 0xFF, b = (px >> 16) & 0xFF;
                    uint8_t Y = (uint8_t)(((uint32_t)r * 76 + (uint32_t)g * 150 + (uint32_t)b * 29) >> 8);
                    row_buf[x] = Y | (Y << 8) | (Y << 16) | (px & 0xFF000000);
                } else {
                    row_buf[x] = (px ^ 0x00FFFFFF) | (px & 0xFF000000);
                }
            }
            // Burst write
            memcpy(&dst[y * FB_W], row_buf, FB_W * 4);
        }
        return;
    }

    // 3x3 kernel filters with line buffers
    uint8_t line[3][FB_W];
#pragma HLS ARRAY_PARTITION variable=line dim=1 complete

    uint32_t out_row[FB_W];

    // Preload row 0 and 1
    memcpy(row_buf, &src[0], FB_W * 4);
    INIT0: for (int x = 0; x < FB_W; x++) {
#pragma HLS PIPELINE II=1
        uint32_t px = row_buf[x];
        line[0][x] = (uint8_t)(((px & 0xFF) * 76 + ((px >> 8) & 0xFF) * 150 + ((px >> 16) & 0xFF) * 29) >> 8);
    }
    memcpy(row_buf, &src[FB_W], FB_W * 4);
    INIT1: for (int x = 0; x < FB_W; x++) {
#pragma HLS PIPELINE II=1
        uint32_t px = row_buf[x];
        line[1][x] = (uint8_t)(((px & 0xFF) * 76 + ((px >> 8) & 0xFF) * 150 + ((px >> 16) & 0xFF) * 29) >> 8);
    }
    // Copy first row as-is
    memcpy(out_row, &src[0], FB_W * 4);
    memcpy(&dst[0], out_row, FB_W * 4);

    KERN_Y: for (int y = 1; y < FB_H - 1; y++) {
        int r0 = (y - 1) % 3, r1 = y % 3, r2 = (y + 1) % 3;

        // Burst read next row + convert to gray
        memcpy(row_buf, &src[(y + 1) * FB_W], FB_W * 4);
        LOAD: for (int x = 0; x < FB_W; x++) {
#pragma HLS PIPELINE II=1
            uint32_t px = row_buf[x];
            line[r2][x] = (uint8_t)(((px & 0xFF) * 76 + ((px >> 8) & 0xFF) * 150 + ((px >> 16) & 0xFF) * 29) >> 8);
        }

        // Border pixels
        out_row[0] = src[y * FB_W];
        out_row[FB_W - 1] = src[y * FB_W + FB_W - 1];

        // 3x3 kernel
        KERN_X: for (int x = 1; x < FB_W - 1; x++) {
#pragma HLS PIPELINE II=1
            uint8_t v;
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
            } else { // ERODE
                uint8_t mn = 255;
                for (int dy = -1; dy <= 1; dy++)
                    for (int dx = -1; dx <= 1; dx++) {
                        uint8_t val = line[(y + dy + 3) % 3][x + dx];
                        if (val < mn) mn = val;
                    }
                v = mn;
            }
            out_row[x] = v | (v << 8) | (v << 16) | 0xFF000000;
        }

        // Burst write processed row
        memcpy(&dst[y * FB_W], out_row, FB_W * 4);
    }

    // Copy last row
    memcpy(row_buf, &src[(FB_H-1) * FB_W], FB_W * 4);
    memcpy(&dst[(FB_H-1) * FB_W], row_buf, FB_W * 4);
}
