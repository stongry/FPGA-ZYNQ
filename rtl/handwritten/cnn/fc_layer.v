//=============================================================================
// fc_layer.v - Fully Connected Layer Engine
//=============================================================================
// Computes: out[o] = bias[o] + sum_i(in[i] * w[o][i])
// Optionally applies ReLU activation.
//=============================================================================
`timescale 1ns / 1ps

module fc_layer #(
    parameter N_IN      = 256,
    parameter N_OUT     = 64,
    parameter APPLY_RELU = 1,
    parameter A_BITS    = 16,
    parameter W_BITS    = 8,
    parameter B_BITS    = 32,
    parameter ACC_BITS  = 32,
    parameter REQUANT_SHIFT = 8,
    parameter IN_ADDR_W  = 9,
    parameter OUT_ADDR_W = 7,
    parameter W_ADDR_W   = 16
)(
    input  wire                     clk,
    input  wire                     rst_n,

    input  wire                     start_i,
    output reg                      busy_o,
    output reg                      done_o,

    // Input vector (read)
    output reg  [IN_ADDR_W-1:0]     in_addr_o,
    input  wire signed [A_BITS-1:0] in_data_i,

    // Weight ROM (read)
    output reg  [W_ADDR_W-1:0]      w_addr_o,
    input  wire signed [W_BITS-1:0] w_data_i,

    // Bias ROM (read)
    output reg  [$clog2(N_OUT)-1:0] b_addr_o,
    input  wire signed [B_BITS-1:0] b_data_i,

    // Output (write)
    output reg                      out_we_o,
    output reg  [OUT_ADDR_W-1:0]    out_addr_o,
    output reg  signed [A_BITS-1:0] out_data_o
);

    localparam S_IDLE  = 3'd0;
    localparam S_START = 3'd1;
    localparam S_READ  = 3'd2;
    localparam S_MAC   = 3'd3;
    localparam S_WR    = 3'd4;
    localparam S_DONE  = 3'd5;

    reg [2:0] state;
    reg [$clog2(N_OUT)-1:0] o_cnt;
    reg [$clog2(N_IN):0]    i_cnt;
    reg signed [ACC_BITS-1:0] acc;
    reg mac_valid;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state    <= S_IDLE;
            busy_o   <= 1'b0;
            done_o   <= 1'b0;
            out_we_o <= 1'b0;
            o_cnt    <= 0;
            i_cnt    <= 0;
            acc      <= 0;
            mac_valid <= 1'b0;
        end else begin
            done_o   <= 1'b0;
            out_we_o <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start_i) begin
                        o_cnt  <= 0;
                        busy_o <= 1'b1;
                        state  <= S_START;
                    end
                end

                S_START: begin
                    // Load bias for this output
                    b_addr_o <= o_cnt;
                    acc      <= 0;
                    i_cnt    <= 0;
                    state    <= S_READ;
                end

                S_READ: begin
                    in_addr_o <= i_cnt;
                    w_addr_o  <= o_cnt * N_IN + i_cnt;
                    mac_valid <= 1'b1;
                    state     <= S_MAC;
                end

                S_MAC: begin
                    if (mac_valid) begin
                        acc <= acc + $signed(in_data_i) * $signed(w_data_i);
                        mac_valid <= 1'b0;
                    end

                    if (i_cnt == N_IN - 1) begin
                        state <= S_WR;
                    end else begin
                        i_cnt <= i_cnt + 1;
                        state <= S_READ;
                    end
                end

                S_WR: begin
                    begin : wr_block
                        reg signed [ACC_BITS-1:0] biased;
                        reg signed [ACC_BITS-1:0] shifted;
                        reg signed [A_BITS-1:0] clipped;
                        biased  = acc + $signed(b_data_i);
                        shifted = biased >>> REQUANT_SHIFT;
                        if (APPLY_RELU && shifted < 0) begin
                            clipped = 0;
                        end else if (shifted > ((1 <<< (A_BITS - 1)) - 1)) begin
                            clipped = (1 <<< (A_BITS - 1)) - 1;
                        end else if (shifted < -(1 <<< (A_BITS - 1))) begin
                            clipped = -(1 <<< (A_BITS - 1));
                        end else begin
                            clipped = shifted[A_BITS-1:0];
                        end
                        out_data_o <= clipped;
                    end
                    out_addr_o <= o_cnt;
                    out_we_o   <= 1'b1;

                    if (o_cnt == N_OUT - 1) begin
                        state <= S_DONE;
                    end else begin
                        o_cnt <= o_cnt + 1;
                        state <= S_START;
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
