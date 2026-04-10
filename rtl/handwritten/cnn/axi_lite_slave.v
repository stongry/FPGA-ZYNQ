//=============================================================================
// axi_lite_slave.v - Simple AXI4-Lite slave for CNN accelerator control
//=============================================================================
// Address map (byte addresses, 32-bit words):
//   0x000          CTRL    W   [0]=start (self-clearing)
//   0x004          STATUS  R   [0]=done, [1]=busy, [7:4]=stage
//   0x010..0x037   PROBS   R   10 x int32 probability (fixed-point Q16)
//   0x040          PRED    R   u32 predicted class
//   0x100..0x3FF   INPUT   W   input image memory (784 bytes, packed into words)
//=============================================================================
`timescale 1ns / 1ps

module axi_lite_slave #(
    parameter ADDR_W = 12,   // 4KB register space
    parameter DATA_W = 32
)(
    // AXI-Lite clock/reset
    input  wire                  s_axi_aclk,
    input  wire                  s_axi_aresetn,

    // Write address channel
    input  wire [ADDR_W-1:0]     s_axi_awaddr,
    input  wire [2:0]            s_axi_awprot,
    input  wire                  s_axi_awvalid,
    output reg                   s_axi_awready,

    // Write data channel
    input  wire [DATA_W-1:0]     s_axi_wdata,
    input  wire [DATA_W/8-1:0]   s_axi_wstrb,
    input  wire                  s_axi_wvalid,
    output reg                   s_axi_wready,

    // Write response channel
    output reg  [1:0]            s_axi_bresp,
    output reg                   s_axi_bvalid,
    input  wire                  s_axi_bready,

    // Read address channel
    input  wire [ADDR_W-1:0]     s_axi_araddr,
    input  wire [2:0]            s_axi_arprot,
    input  wire                  s_axi_arvalid,
    output reg                   s_axi_arready,

    // Read data channel
    output reg  [DATA_W-1:0]     s_axi_rdata,
    output reg  [1:0]            s_axi_rresp,
    output reg                   s_axi_rvalid,
    input  wire                  s_axi_rready,

    // --- User interface (to CNN logic) ---
    output reg                   start_o,          // single-cycle pulse
    input  wire                  busy_i,
    input  wire                  done_i,
    input  wire [3:0]            stage_i,
    input  wire [31:0]           pred_i,
    input  wire [31:0]           prob0_i, prob1_i, prob2_i, prob3_i, prob4_i,
    input  wire [31:0]           prob5_i, prob6_i, prob7_i, prob8_i, prob9_i,

    // Input image write port (to input BRAM)
    output reg                   in_bram_we_o,
    output reg  [9:0]            in_bram_addr_o,   // 1024 words → 4KB
    output reg  [7:0]            in_bram_din_o
);

    // Debug scratch register (read/write at 0x008)
    reg [31:0] scratch;
    // Count of start pulses generated (debug)
    reg [31:0] start_count;

    // Latched done flag (cleared on read)
    reg done_latched;
    always @(posedge s_axi_aclk or negedge s_axi_aresetn) begin
        if (!s_axi_aresetn) done_latched <= 1'b0;
        else if (done_i)    done_latched <= 1'b1;
        // cleared when CTRL.start written
        else if (start_o)   done_latched <= 1'b0;
    end

    // --- Write handshake (fixed: latch AW+W together, then decode) ---
    reg aw_done, w_done;
    reg [ADDR_W-1:0] waddr_lat;
    reg [31:0]       wdata_lat;
    reg              wr_fire;   // single-cycle pulse when both AW+W received

    always @(posedge s_axi_aclk or negedge s_axi_aresetn) begin
        if (!s_axi_aresetn) begin
            s_axi_awready <= 1'b0;
            s_axi_wready  <= 1'b0;
            s_axi_bvalid  <= 1'b0;
            s_axi_bresp   <= 2'b00;
            start_o       <= 1'b0;
            scratch       <= 32'hCAFE0000;
            start_count   <= 32'd0;
            in_bram_we_o  <= 1'b0;
            in_bram_addr_o<= 10'd0;
            in_bram_din_o <= 8'd0;
            waddr_lat     <= 0;
            wdata_lat     <= 0;
            aw_done       <= 1'b0;
            w_done        <= 1'b0;
            wr_fire       <= 1'b0;
        end else begin
            start_o      <= 1'b0;
            in_bram_we_o <= 1'b0;
            s_axi_awready <= 1'b0;
            s_axi_wready  <= 1'b0;
            wr_fire       <= 1'b0;

            // Latch AW when valid and not yet latched
            if (s_axi_awvalid && !aw_done) begin
                s_axi_awready <= 1'b1;
                waddr_lat     <= s_axi_awaddr;
                aw_done       <= 1'b1;
            end

            // Latch W when valid and not yet latched
            if (s_axi_wvalid && !w_done) begin
                s_axi_wready <= 1'b1;
                wdata_lat    <= s_axi_wdata;
                w_done       <= 1'b1;
            end

            // When both AW and W received, fire write
            if ((aw_done || (s_axi_awvalid && !aw_done)) &&
                (w_done  || (s_axi_wvalid  && !w_done))) begin
                if (!wr_fire && !s_axi_bvalid) begin
                    wr_fire <= 1'b1;
                    // Use combinational mux: if just latched this cycle, use input directly
                    // Otherwise use latched value
                end
            end

            // Execute write on fire (one cycle after both latched)
            if (aw_done && w_done && !s_axi_bvalid) begin
                case (waddr_lat[11:0])
                    12'h000: begin // CTRL
                        if (wdata_lat[0]) begin
                            start_o <= 1'b1;
                            start_count <= start_count + 1;
                        end
                    end
                    12'h008: scratch <= wdata_lat;
                    default: begin
                        if (waddr_lat >= 12'h100 && waddr_lat < 12'h500) begin
                            in_bram_we_o   <= 1'b1;
                            in_bram_addr_o <= waddr_lat[11:2] - 10'd64;
                            in_bram_din_o  <= wdata_lat[7:0];
                        end
                    end
                endcase
                s_axi_bvalid <= 1'b1;
                s_axi_bresp  <= 2'b00;
                aw_done <= 1'b0;
                w_done  <= 1'b0;
            end

            // Clear bvalid when master accepts response
            if (s_axi_bvalid && s_axi_bready) begin
                s_axi_bvalid <= 1'b0;
            end
        end
    end

    // --- Read handshake ---
    reg [ADDR_W-1:0] raddr_lat;
    always @(posedge s_axi_aclk or negedge s_axi_aresetn) begin
        if (!s_axi_aresetn) begin
            s_axi_arready <= 1'b0;
            s_axi_rvalid  <= 1'b0;
            s_axi_rdata   <= 32'd0;
            s_axi_rresp   <= 2'b00;
            raddr_lat     <= 0;
        end else begin
            if (!s_axi_arready && s_axi_arvalid) begin
                s_axi_arready <= 1'b1;
                raddr_lat     <= s_axi_araddr;
            end else begin
                s_axi_arready <= 1'b0;
            end

            if (s_axi_arready && s_axi_arvalid && !s_axi_rvalid) begin
                s_axi_rvalid <= 1'b1;
                s_axi_rresp  <= 2'b00;
                case (raddr_lat[11:0])
                    12'h000: s_axi_rdata <= start_count;    // CTRL read: start count
                    12'h004: s_axi_rdata <= {24'd0, stage_i, 2'd0, busy_i, done_latched};
                    12'h008: s_axi_rdata <= scratch;        // scratch register
                    12'h00C: s_axi_rdata <= {31'd0, busy_i}; // raw busy signal
                    12'h010: s_axi_rdata <= prob0_i;
                    12'h014: s_axi_rdata <= prob1_i;
                    12'h018: s_axi_rdata <= prob2_i;
                    12'h01C: s_axi_rdata <= prob3_i;
                    12'h020: s_axi_rdata <= prob4_i;
                    12'h024: s_axi_rdata <= prob5_i;
                    12'h028: s_axi_rdata <= prob6_i;
                    12'h02C: s_axi_rdata <= prob7_i;
                    12'h030: s_axi_rdata <= prob8_i;
                    12'h034: s_axi_rdata <= prob9_i;
                    12'h040: s_axi_rdata <= pred_i;
                    default: s_axi_rdata <= 32'hDEAD_BEEF;
                endcase
            end else if (s_axi_rvalid && s_axi_rready) begin
                s_axi_rvalid <= 1'b0;
            end
        end
    end

endmodule
