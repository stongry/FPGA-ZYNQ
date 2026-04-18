/* Minimal testbench for PlateCNN HLS kernel (used for C-sim validation).
 * Just tests that the function compiles and runs without crashing. */
#include <stdio.h>
#include <string.h>
#include <stdint.h>

extern "C" void plate_cnn_hls(
    uint8_t image[32 * 128],
    uint64_t fc_weights_addr,
    const int8_t *fc_weights_ddr,
    uint8_t predictions[7]
);

int main() {
    uint8_t image[32 * 128];
    uint8_t predictions[7];
    /* Dummy FC weights (2MB zeros for testbench) */
    static int8_t fc_weights[512 * 4096];
    memset(fc_weights, 0, sizeof(fc_weights));
    memset(image, 128, sizeof(image));
    memset(predictions, 0, sizeof(predictions));

    plate_cnn_hls(image, 0, fc_weights, predictions);

    printf("Predictions: [%d, %d, %d, %d, %d, %d, %d]\n",
           predictions[0], predictions[1], predictions[2], predictions[3],
           predictions[4], predictions[5], predictions[6]);
    return 0;
}
