//=============================================================================
// pool_layer.v - 2x2 Max Pooling Engine
//=============================================================================
// For each output pixel, reads 4 input pixels (2x2 window) and writes the max.
// Input:  [C][IH][IW]
// Output: [C][IH/2][IW/2]
//=============================================================================
`timescale 1ns / 1ps

module pool_layer #(
    parameter IN_W   = 24,
    parameter IN_H   = 24,
    parameter C      = 8,
    parameter OUT_W  = 12,
    parameter OUT_H  = 12,
    parameter A_BITS = 16,
    parameter IN_ADDR_W  = 14,
    parameter OUT_ADDR_W = 14
)(
    input  wire                     clk,
    input  wire                     rst_n,

    input  wire                     start_i,
    output reg                      busy_o,
    output reg                      done_o,

    // Input feature map (read)
    output reg  [IN_ADDR_W-1:0]     in_addr_o,
    input  wire signed [A_BITS-1:0] in_data_i,

    // Output feature map (write)
    output reg                      out_we_o,
    output reg  [OUT_ADDR_W-1:0]    out_addr_o,
    output reg  signed [A_BITS-1:0] out_data_o
);

    localparam S_IDLE = 3'd0;
    localparam S_R0   = 3'd1;
    localparam S_R1   = 3'd2;
    localparam S_R2   = 3'd3;
    localparam S_R3   = 3'd4;
    localparam S_WR   = 3'd5;
    localparam S_DONE = 3'd6;

    reg [2:0] state;
    reg [$clog2(C):0]     c_cnt;
    reg [$clog2(OUT_H):0] oy_cnt;
    reg [$clog2(OUT_W):0] ox_cnt;

    reg signed [A_BITS-1:0] val0, val1, val2, val3;

    wire [IN_ADDR_W-1:0] base = c_cnt * (IN_H * IN_W) + (oy_cnt*2) * IN_W + (ox_cnt*2);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state    <= S_IDLE;
            busy_o   <= 1'b0;
            done_o   <= 1'b0;
            out_we_o <= 1'b0;
            c_cnt    <= 0;
            oy_cnt   <= 0;
            ox_cnt   <= 0;
        end else begin
            done_o   <= 1'b0;
            out_we_o <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start_i) begin
                        c_cnt  <= 0;
                        oy_cnt <= 0;
                        ox_cnt <= 0;
                        busy_o <= 1'b1;
                        state  <= S_R0;
                    end
                end

                S_R0: begin
                    in_addr_o <= base;           // top-left
                    state     <= S_R1;
                end

                S_R1: begin
                    // Capture val0 (BRAM latency = 1)
                    val0      <= in_data_i;
                    in_addr_o <= base + 1;       // top-right
                    state     <= S_R2;
                end

                S_R2: begin
                    val1      <= in_data_i;
                    in_addr_o <= base + IN_W;    // bottom-left
                    state     <= S_R3;
                end

                S_R3: begin
                    val2      <= in_data_i;
                    in_addr_o <= base + IN_W + 1; // bottom-right
                    state     <= S_WR;
                end

                S_WR: begin
                    val3 = in_data_i;
                    begin : max_calc
                        reg signed [A_BITS-1:0] m01, m23, mall;
                        m01  = (val0 > val1) ? val0 : val1;
                        m23  = (val2 > val3) ? val2 : val3;
                        mall = (m01 > m23) ? m01 : m23;
                        out_data_o <= mall;
                    end
                    out_addr_o <= c_cnt * (OUT_H * OUT_W) + oy_cnt * OUT_W + ox_cnt;
                    out_we_o   <= 1'b1;

                    // Advance output position
                    if (ox_cnt == OUT_W - 1) begin
                        ox_cnt <= 0;
                        if (oy_cnt == OUT_H - 1) begin
                            oy_cnt <= 0;
                            if (c_cnt == C - 1) begin
                                state <= S_DONE;
                            end else begin
                                c_cnt <= c_cnt + 1;
                                state <= S_R0;
                            end
                        end else begin
                            oy_cnt <= oy_cnt + 1;
                            state  <= S_R0;
                        end
                    end else begin
                        ox_cnt <= ox_cnt + 1;
                        state  <= S_R0;
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
