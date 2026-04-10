//=============================================================================
// cnn_params.vh - TinyLeNet CNN Accelerator Parameters
//=============================================================================
// Network: Conv1(1->8,5x5) → ReLU → MaxPool2 →
//          Conv2(8->16,5x5) → ReLU → MaxPool2 →
//          FC1(256→64) → ReLU → FC2(64→10) → Argmax
//
// Data types:
//   Weights:      INT8  (symmetric quantization per layer)
//   Activations:  INT16 (after requantization + clip)
//   Accumulators: INT32
//=============================================================================

`ifndef CNN_PARAMS_VH
`define CNN_PARAMS_VH

// === Input dimensions ===
`define IN_W        28
`define IN_H        28
`define IN_C        1
`define IN_SIZE     (`IN_W * `IN_H * `IN_C)   // 784

// === Conv1: 1→8, 5x5, stride 1 ===
`define C1_OC       8
`define C1_IC       1
`define C1_K        5
`define C1_OW       (`IN_W  - `C1_K + 1)      // 24
`define C1_OH       (`IN_H  - `C1_K + 1)      // 24
`define C1_OUT_SIZE (`C1_OC * `C1_OW * `C1_OH) // 4608
`define C1_W_COUNT  (`C1_OC * `C1_IC * `C1_K * `C1_K)  // 200

// === MaxPool 2×2 after Conv1 ===
`define P1_OW       (`C1_OW / 2)              // 12
`define P1_OH       (`C1_OH / 2)              // 12
`define P1_OUT_SIZE (`C1_OC * `P1_OW * `P1_OH) // 1152

// === Conv2: 8→16, 5x5, stride 1 ===
`define C2_OC       16
`define C2_IC       8
`define C2_K        5
`define C2_OW       (`P1_OW - `C2_K + 1)      // 8
`define C2_OH       (`P1_OH - `C2_K + 1)      // 8
`define C2_OUT_SIZE (`C2_OC * `C2_OW * `C2_OH) // 1024
`define C2_W_COUNT  (`C2_OC * `C2_IC * `C2_K * `C2_K)  // 3200

// === MaxPool 2×2 after Conv2 ===
`define P2_OW       (`C2_OW / 2)              // 4
`define P2_OH       (`C2_OH / 2)              // 4
`define P2_OUT_SIZE (`C2_OC * `P2_OW * `P2_OH) // 256

// === FC1: 256 → 64 ===
`define FC1_IN      `P2_OUT_SIZE              // 256
`define FC1_OUT     64
`define FC1_W_COUNT (`FC1_IN * `FC1_OUT)      // 16384

// === FC2: 64 → 10 ===
`define FC2_IN      `FC1_OUT                  // 64
`define FC2_OUT     10
`define FC2_W_COUNT (`FC2_IN * `FC2_OUT)      // 640

// === Bit widths ===
`define W_BITS      8       // weight width
`define A_BITS      16      // activation width (feature maps)
`define ACC_BITS    32      // accumulator width
`define U8_BITS     8       // input image bits

// === Requantization shift (weight_scale * input_scale → activation_scale)
// For simplicity we shift right by REQUANT_SHIFT after accumulation
`define REQUANT_SHIFT 8

// === AXI-Lite register map ===
// 0x00: CTRL         W/R [0]=start, self-clearing
// 0x04: STATUS       R   [0]=done, [1]=busy
// 0x08..0x0C: reserved
// 0x10..0x37: PROBS[10]  R int32 (probability * 1000)
// 0x40..0x47: PRED_CLASS R u32
// 0x80..0x03FF: INPUT image memory (784 bytes, word-aligned)
`define REG_CTRL         16'h0000
`define REG_STATUS       16'h0004
`define REG_PROBS_BASE   16'h0010
`define REG_PRED         16'h0040
`define REG_INPUT_BASE   16'h0100   // offset for input image

`endif // CNN_PARAMS_VH
