# Vitis HLS TCL script for PlateCNN
# Run on Windows: vitis_hls -f run_plate_cnn_hls.tcl
#
# Output: plate_cnn_hls_prj/solution1/impl/ip/... (packaged IP)
# Or export as Verilog: plate_cnn_hls_prj/solution1/impl/verilog/

open_project plate_cnn_hls_prj
set_top plate_cnn_hls

# Source files
add_files plate_cnn_hls_kernel.cpp
add_files -tb plate_cnn_hls_tb.cpp

# Part: ZU3EG
open_solution solution1 -flow_target vivado
set_part xczu3eg-sfvc784-1-i
create_clock -period 10.0 -name default ; # 100 MHz

# Configure memory interfaces
config_interface -m_axi_addr64
config_interface -m_axi_max_read_burst_length 256
config_interface -m_axi_max_write_burst_length 256
config_interface -m_axi_latency 64

# C simulation (optional, slow)
# csim_design

# High-level synthesis
csynth_design

# RTL co-simulation (optional, slow)
# cosim_design

# Export IP for Vivado BD
export_design -format ip_catalog -description "PlateCNN end-to-end INT8 accelerator"

exit
