//=============================================================================
// conv_layer.v - Generic 2D Convolution Engine
//=============================================================================
// Performs: out[oc][oy][ox] = ReLU( bias[oc] +
//                                    sum_{ic,ky,kx}(in[ic][oy+ky][ox+kx] * w[oc][ic][ky][kx]) )
// Sequential implementation: one output pixel per inner loop completion.
// For each output pixel: 5*5*IC MACs, executed one per cycle.
//
// Control: start_i → busy_o (high during compute) → done_o (pulse at end)
// Input memory: addressed as [ic*IH*IW + iy*IW + ix]
// Weight memory: addressed as [oc*IC*K*K + ic*K*K + ky*K + kx]
// Output memory: addressed as [oc*OH*OW + oy*OW + ox]
//=============================================================================
`timescale 1ns / 1ps

module conv_layer #(
    parameter IN_W   = 28,
    parameter IN_H   = 28,
    parameter IC     = 1,
    parameter OC     = 8,
    parameter K      = 5,
    parameter OUT_W  = 24,
    parameter OUT_H  = 24,
    parameter A_BITS = 16,   // activation width
    parameter W_BITS = 8,    // weight width
    parameter B_BITS = 32,   // bias width
    parameter ACC_BITS = 32,
    parameter REQUANT_SHIFT = 8,
    parameter IN_ADDR_W  = 14,
    parameter OUT_ADDR_W = 14,
    parameter W_ADDR_W   = 14
)(
    input  wire                     clk,
    input  wire                     rst_n,

    // Control
    input  wire                     start_i,
    output reg                      busy_o,
    output reg                      done_o,

    // Input feature map (read)
    output reg  [IN_ADDR_W-1:0]     in_addr_o,
    input  wire [A_BITS-1:0]        in_data_i,

    // Weight ROM (read)
    output reg  [W_ADDR_W-1:0]      w_addr_o,
    input  wire signed [W_BITS-1:0] w_data_i,

    // Bias ROM (read)
    output reg  [$clog2(OC)-1:0]    b_addr_o,
    input  wire signed [B_BITS-1:0] b_data_i,

    // Output feature map (write)
    output reg                      out_we_o,
    output reg  [OUT_ADDR_W-1:0]    out_addr_o,
    output reg  signed [A_BITS-1:0] out_data_o
);

    // State machine
    localparam S_IDLE    = 3'd0;
    localparam S_LOAD_B  = 3'd1;
    localparam S_READ    = 3'd2;
    localparam S_MAC     = 3'd3;
    localparam S_WRITE   = 3'd4;
    localparam S_DONE    = 3'd5;

    reg [2:0] state;

    // Loop counters
    reg [$clog2(OC)-1:0]    oc_cnt;
    reg [$clog2(OUT_H)-1:0] oy_cnt;
    reg [$clog2(OUT_W)-1:0] ox_cnt;
    reg [$clog2(IC):0]      ic_cnt;
    reg [$clog2(K):0]       ky_cnt;
    reg [$clog2(K):0]       kx_cnt;

    // Accumulator
    reg signed [ACC_BITS-1:0] acc;

    // Pipeline regs for MAC
    reg signed [A_BITS-1:0]   in_reg;
    reg signed [W_BITS-1:0]   w_reg;
    reg                       mac_valid;

    // Addr computations (combinational)
    wire [IN_ADDR_W-1:0] in_addr_calc =
        ic_cnt * (IN_H * IN_W) +
        (oy_cnt + ky_cnt) * IN_W +
        (ox_cnt + kx_cnt);

    wire [W_ADDR_W-1:0] w_addr_calc =
        oc_cnt * (IC * K * K) +
        ic_cnt * (K * K) +
        ky_cnt * K +
        kx_cnt;

    wire [OUT_ADDR_W-1:0] out_addr_calc =
        oc_cnt * (OUT_H * OUT_W) +
        oy_cnt * OUT_W +
        ox_cnt;

    // Main FSM
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= S_IDLE;
            busy_o    <= 1'b0;
            done_o    <= 1'b0;
            oc_cnt    <= 0;
            oy_cnt    <= 0;
            ox_cnt    <= 0;
            ic_cnt    <= 0;
            ky_cnt    <= 0;
            kx_cnt    <= 0;
            acc       <= 0;
            out_we_o  <= 1'b0;
            mac_valid <= 1'b0;
        end else begin
            done_o   <= 1'b0;
            out_we_o <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start_i) begin
                        oc_cnt <= 0;
                        oy_cnt <= 0;
                        ox_cnt <= 0;
                        busy_o <= 1'b1;
                        state  <= S_LOAD_B;
                    end
                end

                S_LOAD_B: begin
                    // Load bias for this output channel
                    b_addr_o <= oc_cnt;
                    acc      <= 0;
                    ic_cnt   <= 0;
                    ky_cnt   <= 0;
                    kx_cnt   <= 0;
                    state    <= S_READ;
                end

                S_READ: begin
                    // Issue read for in_data and w_data
                    in_addr_o <= in_addr_calc;
                    w_addr_o  <= w_addr_calc;
                    mac_valid <= 1'b1;
                    state     <= S_MAC;
                end

                S_MAC: begin
                    // BRAM output is now valid
                    if (mac_valid) begin
                        acc <= acc + $signed(in_data_i) * $signed(w_data_i);
                        mac_valid <= 1'b0;
                    end

                    // Advance kernel/channel counters
                    if (kx_cnt == K - 1) begin
                        kx_cnt <= 0;
                        if (ky_cnt == K - 1) begin
                            ky_cnt <= 0;
                            if (ic_cnt == IC - 1) begin
                                // Done all MACs for this output pixel
                                ic_cnt <= 0;
                                // Add bias (already loaded) and write
                                state <= S_WRITE;
                            end else begin
                                ic_cnt <= ic_cnt + 1;
                                state  <= S_READ;
                            end
                        end else begin
                            ky_cnt <= ky_cnt + 1;
                            state  <= S_READ;
                        end
                    end else begin
                        kx_cnt <= kx_cnt + 1;
                        state  <= S_READ;
                    end
                end

                S_WRITE: begin
                    // Bias add + requantize + ReLU
                    begin : write_block
                        reg signed [ACC_BITS-1:0] biased;
                        reg signed [ACC_BITS-1:0] shifted;
                        reg signed [A_BITS-1:0] clipped;
                        biased  = acc + $signed(b_data_i);
                        shifted = biased >>> REQUANT_SHIFT;
                        // ReLU + clip to A_BITS
                        if (shifted < 0) begin
                            clipped = 0;
                        end else if (shifted > ((1 <<< (A_BITS - 1)) - 1)) begin
                            clipped = (1 <<< (A_BITS - 1)) - 1;
                        end else begin
                            clipped = shifted[A_BITS-1:0];
                        end
                        out_data_o <= clipped;
                    end
                    out_addr_o <= out_addr_calc;
                    out_we_o   <= 1'b1;

                    // Advance output position
                    if (ox_cnt == OUT_W - 1) begin
                        ox_cnt <= 0;
                        if (oy_cnt == OUT_H - 1) begin
                            oy_cnt <= 0;
                            if (oc_cnt == OC - 1) begin
                                state <= S_DONE;
                            end else begin
                                oc_cnt <= oc_cnt + 1;
                                state  <= S_LOAD_B;
                            end
                        end else begin
                            oy_cnt <= oy_cnt + 1;
                            state  <= S_LOAD_B;
                        end
                    end else begin
                        ox_cnt <= ox_cnt + 1;
                        state  <= S_LOAD_B;
                    end
                end

                S_DONE: begin
                    busy_o <= 1'b0;
                    done_o <= 1'b1;
                    state  <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
