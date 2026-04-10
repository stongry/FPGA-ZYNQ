//=============================================================================
// weight_rom.v - ROM wrappers for each layer's weights and biases
// Initialized from .hex files produced by scripts/quantize.py
//=============================================================================
`timescale 1ns / 1ps

module weight_rom #(
    parameter DATA_W   = 8,
    parameter ADDR_W   = 12,
    parameter DEPTH    = 1 << ADDR_W,
    parameter INIT_FILE = "weights.hex"
)(
    input  wire                clk,
    input  wire [ADDR_W-1:0]   addr,
    output reg  signed [DATA_W-1:0] dout
);

    (* ROM_STYLE = "BLOCK" *)
    reg [DATA_W-1:0] mem [0:DEPTH-1];

    initial begin
        $readmemh(INIT_FILE, mem);
    end

    always @(posedge clk) begin
        dout <= mem[addr];
    end

endmodule

//=============================================================================
// bias_rom - same as weight_rom but with 32-bit biases
//=============================================================================
module bias_rom #(
    parameter ADDR_W   = 4,
    parameter DEPTH    = 1 << ADDR_W,
    parameter INIT_FILE = "bias.hex"
)(
    input  wire                clk,
    input  wire [ADDR_W-1:0]   addr,
    output reg  signed [31:0]  dout
);

    (* ROM_STYLE = "BLOCK" *)
    reg [31:0] mem [0:DEPTH-1];

    initial begin
        $readmemh(INIT_FILE, mem);
    end

    always @(posedge clk) begin
        dout <= mem[addr];
    end

endmodule
