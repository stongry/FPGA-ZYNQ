//=============================================================================
// bram_sp.v - Simple single-port synchronous BRAM (inferrable)
//=============================================================================
`timescale 1ns / 1ps

module bram_sp #(
    parameter DATA_W = 16,
    parameter ADDR_W = 10,
    parameter DEPTH  = 1 << ADDR_W,
    parameter INIT_FILE = ""
)(
    input  wire                clk,
    input  wire                we,
    input  wire [ADDR_W-1:0]   addr,
    input  wire [DATA_W-1:0]   din,
    output reg  [DATA_W-1:0]   dout
);

    reg [DATA_W-1:0] mem [0:DEPTH-1];

    initial begin
        if (INIT_FILE != "") begin
            $readmemh(INIT_FILE, mem);
        end
    end

    always @(posedge clk) begin
        if (we) begin
            mem[addr] <= din;
        end
        dout <= mem[addr];
    end

endmodule

//=============================================================================
// bram_dp - True dual-port BRAM (write port A, read port B)
//=============================================================================
module bram_dp #(
    parameter DATA_W = 16,
    parameter ADDR_W = 10,
    parameter DEPTH  = 1 << ADDR_W,
    parameter INIT_FILE = ""
)(
    input  wire                clka,
    input  wire                wea,
    input  wire [ADDR_W-1:0]   addra,
    input  wire [DATA_W-1:0]   dina,

    input  wire                clkb,
    input  wire [ADDR_W-1:0]   addrb,
    output reg  [DATA_W-1:0]   doutb
);

    reg [DATA_W-1:0] mem [0:DEPTH-1];

    initial begin
        if (INIT_FILE != "") begin
            $readmemh(INIT_FILE, mem);
        end
    end

    always @(posedge clka) begin
        if (wea) mem[addra] <= dina;
    end

    always @(posedge clkb) begin
        doutb <= mem[addrb];
    end

endmodule
