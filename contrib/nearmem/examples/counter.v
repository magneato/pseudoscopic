// Simple test counter - 4-bit binary counter
// This file can be loaded by gpufpga_load_verilog()

module counter(
    input clk,
    input rst,
    output [3:0] count
);

wire [3:0] next;
reg [3:0] q;

// Increment logic
assign next[0] = ~q[0];
assign next[1] = q[0] ^ q[1];
assign next[2] = (q[0] & q[1]) ^ q[2];
assign next[3] = (q[0] & q[1] & q[2]) ^ q[3];

// Output
assign count = q;

endmodule
