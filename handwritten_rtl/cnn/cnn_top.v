//=============================================================================
// cnn_top.v - TinyLeNet CNN Accelerator Top-Level
//=============================================================================
// Integrates:
//   - AXI-Lite slave (control + input image + output readout)
//   - Feature-map BRAMs for each layer
//   - Weight ROMs initialized from .hex files
//   - Conv1, Pool1, Conv2, Pool2, FC1, FC2 engines
//   - Argmax on FC2 output → pred_class register
//
// Packaged as a Vivado IP via scripts/package_ip.tcl
//=============================================================================
`timescale 1ns / 1ps
`include "cnn_params.vh"

module cnn_top (
    // AXI-Lite (32-bit)
    input  wire         s_axi_aclk,
    input  wire         s_axi_aresetn,

    input  wire [11:0]  s_axi_awaddr,
    input  wire [2:0]   s_axi_awprot,
    input  wire         s_axi_awvalid,
    output wire         s_axi_awready,

    input  wire [31:0]  s_axi_wdata,
    input  wire [3:0]   s_axi_wstrb,
    input  wire         s_axi_wvalid,
    output wire         s_axi_wready,

    output wire [1:0]   s_axi_bresp,
    output wire         s_axi_bvalid,
    input  wire         s_axi_bready,

    input  wire [11:0]  s_axi_araddr,
    input  wire [2:0]   s_axi_arprot,
    input  wire         s_axi_arvalid,
    output wire         s_axi_arready,

    output wire [31:0]  s_axi_rdata,
    output wire [1:0]   s_axi_rresp,
    output wire         s_axi_rvalid,
    input  wire         s_axi_rready,

    // Debug LED
    output wire [3:0]   stage_led
);

    wire clk   = s_axi_aclk;
    wire rst_n = s_axi_aresetn;

    // ============================================================
    // AXI-Lite slave
    // ============================================================
    wire start_w;
    wire busy_w;
    wire done_w;
    wire [3:0] stage_w;
    wire [31:0] pred_w;
    wire [31:0] probs_w [0:9];

    wire        in_bram_we;
    wire [9:0]  in_bram_waddr;
    wire [7:0]  in_bram_din;

    axi_lite_slave u_axi (
        .s_axi_aclk    (clk),
        .s_axi_aresetn (rst_n),

        .s_axi_awaddr  (s_axi_awaddr),
        .s_axi_awprot  (s_axi_awprot),
        .s_axi_awvalid (s_axi_awvalid),
        .s_axi_awready (s_axi_awready),

        .s_axi_wdata   (s_axi_wdata),
        .s_axi_wstrb   (s_axi_wstrb),
        .s_axi_wvalid  (s_axi_wvalid),
        .s_axi_wready  (s_axi_wready),

        .s_axi_bresp   (s_axi_bresp),
        .s_axi_bvalid  (s_axi_bvalid),
        .s_axi_bready  (s_axi_bready),

        .s_axi_araddr  (s_axi_araddr),
        .s_axi_arprot  (s_axi_arprot),
        .s_axi_arvalid (s_axi_arvalid),
        .s_axi_arready (s_axi_arready),

        .s_axi_rdata   (s_axi_rdata),
        .s_axi_rresp   (s_axi_rresp),
        .s_axi_rvalid  (s_axi_rvalid),
        .s_axi_rready  (s_axi_rready),

        .start_o       (start_w),
        .busy_i        (busy_w),
        .done_i        (done_w),
        .stage_i       (stage_w),
        .pred_i        (pred_w),
        .prob0_i       (probs_w[0]), .prob1_i (probs_w[1]),
        .prob2_i       (probs_w[2]), .prob3_i (probs_w[3]),
        .prob4_i       (probs_w[4]), .prob5_i (probs_w[5]),
        .prob6_i       (probs_w[6]), .prob7_i (probs_w[7]),
        .prob8_i       (probs_w[8]), .prob9_i (probs_w[9]),

        .in_bram_we_o  (in_bram_we),
        .in_bram_addr_o(in_bram_waddr),
        .in_bram_din_o (in_bram_din)
    );

    assign stage_led = stage_w;

    // ============================================================
    // Feature map BRAMs (one per layer output)
    // ============================================================
    // Input: 784 bytes stored as 16-bit activations (center+scale done upstream by PS)
    // For simplicity, PS writes already-normalized int16 data through AXI into in_bram
    // (PS driver handles the float-to-int16 conversion).

    // Input BRAM: 1024 x 16 (bytes from PS → zero-extended)
    wire [9:0]  c1_in_raddr;
    wire signed [15:0] c1_in_rdata;

    reg  [9:0]  in_waddr_reg;
    always @(posedge clk) in_waddr_reg <= in_bram_waddr;

    bram_dp #(.DATA_W(16), .ADDR_W(10), .DEPTH(1024)) u_in_bram (
        .clka  (clk),
        .wea   (in_bram_we),
        .addra (in_bram_waddr),
        .dina  ({8'd0, in_bram_din}),   // PS writes u8, extend to 16-bit
        .clkb  (clk),
        .addrb (c1_in_raddr),
        .doutb (c1_in_rdata)
    );

    // ============================================================
    // Weight ROMs
    // ============================================================
    wire [8:0]  c1_w_addr;   // 200 weights → 8 bits
    wire signed [7:0] c1_w_data;
    weight_rom #(.DATA_W(8), .ADDR_W(9), .DEPTH(512),
                 .INIT_FILE("C:/Users/huye/fz3a/cnn/weights/conv1_weights.hex")) u_c1_wrom (
        .clk(clk), .addr(c1_w_addr), .dout(c1_w_data)
    );

    wire [2:0] c1_b_addr;
    wire signed [31:0] c1_b_data;
    bias_rom #(.ADDR_W(3), .INIT_FILE("C:/Users/huye/fz3a/cnn/weights/conv1_bias.hex")) u_c1_brom (
        .clk(clk), .addr(c1_b_addr), .dout(c1_b_data)
    );

    wire [12:0] c2_w_addr;  // 3200 → 12 bits
    wire signed [7:0] c2_w_data;
    weight_rom #(.DATA_W(8), .ADDR_W(12), .DEPTH(4096),
                 .INIT_FILE("C:/Users/huye/fz3a/cnn/weights/conv2_weights.hex")) u_c2_wrom (
        .clk(clk), .addr(c2_w_addr[11:0]), .dout(c2_w_data)
    );

    wire [3:0] c2_b_addr;
    wire signed [31:0] c2_b_data;
    bias_rom #(.ADDR_W(4), .INIT_FILE("C:/Users/huye/fz3a/cnn/weights/conv2_bias.hex")) u_c2_brom (
        .clk(clk), .addr(c2_b_addr), .dout(c2_b_data)
    );

    wire [15:0] fc1_w_addr;  // 16384 → 14 bits
    wire signed [7:0] fc1_w_data;
    weight_rom #(.DATA_W(8), .ADDR_W(14), .DEPTH(16384),
                 .INIT_FILE("C:/Users/huye/fz3a/cnn/weights/fc1_weights.hex")) u_fc1_wrom (
        .clk(clk), .addr(fc1_w_addr[13:0]), .dout(fc1_w_data)
    );

    wire [5:0] fc1_b_addr;
    wire signed [31:0] fc1_b_data;
    bias_rom #(.ADDR_W(6), .INIT_FILE("C:/Users/huye/fz3a/cnn/weights/fc1_bias.hex")) u_fc1_brom (
        .clk(clk), .addr(fc1_b_addr), .dout(fc1_b_data)
    );

    wire [9:0] fc2_w_addr;  // 640 → 10 bits
    wire signed [7:0] fc2_w_data;
    weight_rom #(.DATA_W(8), .ADDR_W(10), .DEPTH(1024),
                 .INIT_FILE("C:/Users/huye/fz3a/cnn/weights/fc2_weights.hex")) u_fc2_wrom (
        .clk(clk), .addr(fc2_w_addr), .dout(fc2_w_data)
    );

    wire [3:0] fc2_b_addr;
    wire signed [31:0] fc2_b_data;
    bias_rom #(.ADDR_W(4), .INIT_FILE("C:/Users/huye/fz3a/cnn/weights/fc2_bias.hex")) u_fc2_brom (
        .clk(clk), .addr(fc2_b_addr), .dout(fc2_b_data)
    );

    // ============================================================
    // Conv1: 1→8, 5×5, output 24×24×8 = 4608
    // ============================================================
    wire c1_start, c1_done, c1_busy;
    wire signed [15:0] c1_out_din;
    wire        c1_out_we;
    wire [12:0] c1_out_waddr;
    wire [12:0] c1_out_raddr;
    wire signed [15:0] c1_out_rdata;

    assign c1_in_raddr = c1_addr_int[9:0];
    wire [13:0] c1_addr_int;

    conv_layer #(
        .IN_W(28), .IN_H(28), .IC(1), .OC(8), .K(5),
        .OUT_W(24), .OUT_H(24),
        .A_BITS(16), .W_BITS(8), .B_BITS(32),
        .IN_ADDR_W(14), .OUT_ADDR_W(13), .W_ADDR_W(9)
    ) u_conv1 (
        .clk(clk), .rst_n(rst_n),
        .start_i   (c1_start), .busy_o(c1_busy), .done_o(c1_done),
        .in_addr_o (c1_addr_int), .in_data_i(c1_in_rdata),
        .w_addr_o  (c1_w_addr), .w_data_i(c1_w_data),
        .b_addr_o  (c1_b_addr), .b_data_i(c1_b_data),
        .out_we_o  (c1_out_we), .out_addr_o(c1_out_waddr),
        .out_data_o(c1_out_din)
    );

    bram_dp #(.DATA_W(16), .ADDR_W(13), .DEPTH(4608)) u_c1_bram (
        .clka(clk), .wea(c1_out_we), .addra(c1_out_waddr), .dina(c1_out_din),
        .clkb(clk), .addrb(c1_out_raddr), .doutb(c1_out_rdata)
    );

    // ============================================================
    // Pool1: 8×24×24 → 8×12×12 = 1152
    // ============================================================
    wire p1_start, p1_done, p1_busy;
    wire signed [15:0] p1_out_din;
    wire        p1_out_we;
    wire [10:0] p1_out_waddr;
    wire [10:0] p1_out_raddr;
    wire signed [15:0] p1_out_rdata;

    pool_layer #(
        .IN_W(24), .IN_H(24), .C(8),
        .OUT_W(12), .OUT_H(12),
        .A_BITS(16), .IN_ADDR_W(13), .OUT_ADDR_W(11)
    ) u_pool1 (
        .clk(clk), .rst_n(rst_n),
        .start_i(p1_start), .busy_o(p1_busy), .done_o(p1_done),
        .in_addr_o(c1_out_raddr), .in_data_i(c1_out_rdata),
        .out_we_o(p1_out_we), .out_addr_o(p1_out_waddr),
        .out_data_o(p1_out_din)
    );

    bram_dp #(.DATA_W(16), .ADDR_W(11), .DEPTH(1152)) u_p1_bram (
        .clka(clk), .wea(p1_out_we), .addra(p1_out_waddr), .dina(p1_out_din),
        .clkb(clk), .addrb(p1_out_raddr), .doutb(p1_out_rdata)
    );

    // ============================================================
    // Conv2: 8→16, 5×5, output 8×8×16 = 1024
    // ============================================================
    wire c2_start, c2_done, c2_busy;
    wire signed [15:0] c2_out_din;
    wire        c2_out_we;
    wire [9:0]  c2_out_waddr;
    wire [9:0]  c2_out_raddr;
    wire signed [15:0] c2_out_rdata;

    conv_layer #(
        .IN_W(12), .IN_H(12), .IC(8), .OC(16), .K(5),
        .OUT_W(8), .OUT_H(8),
        .A_BITS(16), .W_BITS(8), .B_BITS(32),
        .IN_ADDR_W(11), .OUT_ADDR_W(10), .W_ADDR_W(12)
    ) u_conv2 (
        .clk(clk), .rst_n(rst_n),
        .start_i   (c2_start), .busy_o(c2_busy), .done_o(c2_done),
        .in_addr_o (p1_out_raddr), .in_data_i(p1_out_rdata),
        .w_addr_o  (c2_w_addr[11:0]), .w_data_i(c2_w_data),
        .b_addr_o  (c2_b_addr), .b_data_i(c2_b_data),
        .out_we_o  (c2_out_we), .out_addr_o(c2_out_waddr),
        .out_data_o(c2_out_din)
    );
    assign c2_w_addr[12] = 1'b0;

    bram_dp #(.DATA_W(16), .ADDR_W(10), .DEPTH(1024)) u_c2_bram (
        .clka(clk), .wea(c2_out_we), .addra(c2_out_waddr), .dina(c2_out_din),
        .clkb(clk), .addrb(c2_out_raddr), .doutb(c2_out_rdata)
    );

    // ============================================================
    // Pool2: 16×8×8 → 16×4×4 = 256
    // ============================================================
    wire p2_start, p2_done, p2_busy;
    wire signed [15:0] p2_out_din;
    wire        p2_out_we;
    wire [7:0]  p2_out_waddr;
    wire [7:0]  p2_out_raddr;
    wire signed [15:0] p2_out_rdata;

    pool_layer #(
        .IN_W(8), .IN_H(8), .C(16),
        .OUT_W(4), .OUT_H(4),
        .A_BITS(16), .IN_ADDR_W(10), .OUT_ADDR_W(8)
    ) u_pool2 (
        .clk(clk), .rst_n(rst_n),
        .start_i(p2_start), .busy_o(p2_busy), .done_o(p2_done),
        .in_addr_o(c2_out_raddr), .in_data_i(c2_out_rdata),
        .out_we_o(p2_out_we), .out_addr_o(p2_out_waddr),
        .out_data_o(p2_out_din)
    );

    bram_dp #(.DATA_W(16), .ADDR_W(8), .DEPTH(256)) u_p2_bram (
        .clka(clk), .wea(p2_out_we), .addra(p2_out_waddr), .dina(p2_out_din),
        .clkb(clk), .addrb(p2_out_raddr), .doutb(p2_out_rdata)
    );

    // ============================================================
    // FC1: 256 → 64
    // ============================================================
    wire fc1_start, fc1_done, fc1_busy;
    wire signed [15:0] fc1_out_din;
    wire        fc1_out_we;
    wire [5:0]  fc1_out_waddr;
    wire [5:0]  fc1_out_raddr;
    wire signed [15:0] fc1_out_rdata;

    fc_layer #(
        .N_IN(256), .N_OUT(64), .APPLY_RELU(1),
        .A_BITS(16), .W_BITS(8), .B_BITS(32),
        .IN_ADDR_W(8), .OUT_ADDR_W(6), .W_ADDR_W(14)
    ) u_fc1 (
        .clk(clk), .rst_n(rst_n),
        .start_i(fc1_start), .busy_o(fc1_busy), .done_o(fc1_done),
        .in_addr_o(p2_out_raddr), .in_data_i(p2_out_rdata),
        .w_addr_o(fc1_w_addr[13:0]), .w_data_i(fc1_w_data),
        .b_addr_o(fc1_b_addr), .b_data_i(fc1_b_data),
        .out_we_o(fc1_out_we), .out_addr_o(fc1_out_waddr),
        .out_data_o(fc1_out_din)
    );
    assign fc1_w_addr[15:14] = 2'b0;

    bram_dp #(.DATA_W(16), .ADDR_W(6), .DEPTH(64)) u_fc1_bram (
        .clka(clk), .wea(fc1_out_we), .addra(fc1_out_waddr), .dina(fc1_out_din),
        .clkb(clk), .addrb(fc1_out_raddr), .doutb(fc1_out_rdata)
    );

    // ============================================================
    // FC2: 64 → 10
    // ============================================================
    wire fc2_start, fc2_done, fc2_busy;
    wire signed [15:0] fc2_out_din;
    wire        fc2_out_we;
    wire [3:0]  fc2_out_waddr;

    reg signed [15:0] fc2_scores [0:9];
    always @(posedge clk) if (fc2_out_we) fc2_scores[fc2_out_waddr] <= fc2_out_din;

    fc_layer #(
        .N_IN(64), .N_OUT(10), .APPLY_RELU(0),
        .A_BITS(16), .W_BITS(8), .B_BITS(32),
        .IN_ADDR_W(6), .OUT_ADDR_W(4), .W_ADDR_W(10)
    ) u_fc2 (
        .clk(clk), .rst_n(rst_n),
        .start_i(fc2_start), .busy_o(fc2_busy), .done_o(fc2_done),
        .in_addr_o(fc1_out_raddr), .in_data_i(fc1_out_rdata),
        .w_addr_o(fc2_w_addr), .w_data_i(fc2_w_data),
        .b_addr_o(fc2_b_addr), .b_data_i(fc2_b_data),
        .out_we_o(fc2_out_we), .out_addr_o(fc2_out_waddr),
        .out_data_o(fc2_out_din)
    );

    // ============================================================
    // Expose scores directly as probabilities (simplified, no softmax)
    // PS can do softmax if needed; argmax is done here
    // ============================================================
    genvar g;
    generate
        for (g = 0; g < 10; g = g + 1) begin : gen_probs
            assign probs_w[g] = {{16{fc2_scores[g][15]}}, fc2_scores[g]};
        end
    endgenerate

    // ============================================================
    // Argmax on final scores
    // ============================================================
    reg [3:0] argmax;
    reg signed [15:0] maxv;
    integer k;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            argmax <= 0;
            maxv   <= -16'sd32768;
        end else if (done_w) begin
            argmax <= 0;
            maxv   <= fc2_scores[0];
            for (k = 1; k < 10; k = k + 1) begin
                if (fc2_scores[k] > maxv) begin
                    maxv   <= fc2_scores[k];
                    argmax <= k[3:0];
                end
            end
        end
    end

    assign pred_w = {28'd0, argmax};

    // ============================================================
    // Main sequencer
    // ============================================================
    cnn_fsm u_fsm (
        .clk(clk), .rst_n(rst_n),
        .start_i(start_w),
        .busy_o (busy_w),
        .done_o (done_w),
        .c1_start_o (c1_start), .c1_done_i (c1_done),
        .p1_start_o (p1_start), .p1_done_i (p1_done),
        .c2_start_o (c2_start), .c2_done_i (c2_done),
        .p2_start_o (p2_start), .p2_done_i (p2_done),
        .fc1_start_o(fc1_start), .fc1_done_i(fc1_done),
        .fc2_start_o(fc2_start), .fc2_done_i(fc2_done),
        .stage_o    (stage_w)
    );

endmodule
