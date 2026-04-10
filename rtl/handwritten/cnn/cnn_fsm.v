//=============================================================================
// cnn_fsm.v - Top-level CNN sequencer for TinyLeNet
//=============================================================================
// Sequences: Conv1 → Pool1 → Conv2 → Pool2 → FC1 → FC2 → Argmax
// Generates start pulses and waits for each layer's done.
//=============================================================================
`timescale 1ns / 1ps

module cnn_fsm (
    input  wire clk,
    input  wire rst_n,

    input  wire start_i,
    output reg  done_o,
    output reg  busy_o,

    // Per-layer start/done handshakes
    output reg  c1_start_o,
    input  wire c1_done_i,
    output reg  p1_start_o,
    input  wire p1_done_i,
    output reg  c2_start_o,
    input  wire c2_done_i,
    output reg  p2_start_o,
    input  wire p2_done_i,
    output reg  fc1_start_o,
    input  wire fc1_done_i,
    output reg  fc2_start_o,
    input  wire fc2_done_i,

    // Current stage debug
    output reg [3:0] stage_o
);

    localparam S_IDLE    = 4'd0;
    localparam S_C1_GO   = 4'd1;
    localparam S_C1_WAIT = 4'd2;
    localparam S_P1_GO   = 4'd3;
    localparam S_P1_WAIT = 4'd4;
    localparam S_C2_GO   = 4'd5;
    localparam S_C2_WAIT = 4'd6;
    localparam S_P2_GO   = 4'd7;
    localparam S_P2_WAIT = 4'd8;
    localparam S_FC1_GO  = 4'd9;
    localparam S_FC1_WAIT = 4'd10;
    localparam S_FC2_GO  = 4'd11;
    localparam S_FC2_WAIT = 4'd12;
    localparam S_DONE    = 4'd13;

    reg [3:0] state;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= S_IDLE;
            busy_o      <= 1'b0;
            done_o      <= 1'b0;
            c1_start_o  <= 1'b0;
            p1_start_o  <= 1'b0;
            c2_start_o  <= 1'b0;
            p2_start_o  <= 1'b0;
            fc1_start_o <= 1'b0;
            fc2_start_o <= 1'b0;
            stage_o     <= 4'd0;
        end else begin
            // default: deassert all pulses
            c1_start_o  <= 1'b0;
            p1_start_o  <= 1'b0;
            c2_start_o  <= 1'b0;
            p2_start_o  <= 1'b0;
            fc1_start_o <= 1'b0;
            fc2_start_o <= 1'b0;
            done_o      <= 1'b0;

            case (state)
                S_IDLE: begin
                    busy_o <= 1'b0;
                    if (start_i) begin
                        busy_o <= 1'b1;
                        state  <= S_C1_GO;
                    end
                end

                S_C1_GO:   begin c1_start_o <= 1'b1; stage_o <= 4'd1; state <= S_C1_WAIT; end
                S_C1_WAIT: if (c1_done_i) state <= S_P1_GO;

                S_P1_GO:   begin p1_start_o <= 1'b1; stage_o <= 4'd2; state <= S_P1_WAIT; end
                S_P1_WAIT: if (p1_done_i) state <= S_C2_GO;

                S_C2_GO:   begin c2_start_o <= 1'b1; stage_o <= 4'd3; state <= S_C2_WAIT; end
                S_C2_WAIT: if (c2_done_i) state <= S_P2_GO;

                S_P2_GO:   begin p2_start_o <= 1'b1; stage_o <= 4'd4; state <= S_P2_WAIT; end
                S_P2_WAIT: if (p2_done_i) state <= S_FC1_GO;

                S_FC1_GO:   begin fc1_start_o <= 1'b1; stage_o <= 4'd5; state <= S_FC1_WAIT; end
                S_FC1_WAIT: if (fc1_done_i) state <= S_FC2_GO;

                S_FC2_GO:   begin fc2_start_o <= 1'b1; stage_o <= 4'd6; state <= S_FC2_WAIT; end
                S_FC2_WAIT: if (fc2_done_i) state <= S_DONE;

                S_DONE: begin
                    done_o  <= 1'b1;
                    busy_o  <= 1'b0;
                    stage_o <= 4'd15;
                    state   <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
