# Vivado TCL script: add PlateCNN HLS IP to existing design_1 block design
# Usage: vivado -mode batch -source integrate_plate_cnn_hls.tcl
#
# Prerequisites:
# - fz3a Vivado project open with design_1 block design
# - PlateCNN HLS IP repo added
#
# Outputs: new bitstream with PlateCNN IP at AXI addr 0xA0040000

set proj_dir "C:/Users/huye/fz3a/xazu3eg_2G_release"
set bd_name design_1
set hls_ip_path "C:/Users/huye/fz3a/plate_cnn_hls/plate_cnn_hls_prj/solution1/impl/ip"

open_project $proj_dir/xazu3eg_2G_release.xpr

# Add HLS IP to repo
set_property ip_repo_paths [concat [get_property ip_repo_paths [current_project]] $hls_ip_path] [current_project]
update_ip_catalog

open_bd_design $proj_dir/xazu3eg_2G_release.srcs/sources_1/bd/$bd_name/$bd_name.bd

# Add PlateCNN HLS IP
create_bd_cell -type ip -vlnv xilinx.com:hls:plate_cnn_hls:1.0 plate_cnn_hls_0

# Connect clock + reset (same as other HLS IPs)
connect_bd_net [get_bd_pins plate_cnn_hls_0/ap_clk] [get_bd_pins /pl_clk0_net]
connect_bd_net [get_bd_pins plate_cnn_hls_0/ap_rst_n] [get_bd_pins /pl_rst0_net]

# Connect s_axi_ctrl to AXI Interconnect (from PS M_AXI_HPM0_FPD)
# Find existing axi_interconnect_0 master ports and add new one
set axi_intc [get_bd_cells axi_interconnect_0]
set current_masters [get_property CONFIG.NUM_MI $axi_intc]
set new_masters [expr {$current_masters + 1}]
set_property CONFIG.NUM_MI $new_masters $axi_intc
set new_master_idx [format "%02d" [expr {$new_masters - 1}]]
connect_bd_intf_net [get_bd_intf_pins plate_cnn_hls_0/s_axi_ctrl] \
    [get_bd_intf_pins axi_interconnect_0/M${new_master_idx}_AXI]
connect_bd_net [get_bd_pins axi_interconnect_0/M${new_master_idx}_ACLK] [get_bd_pins /pl_clk0_net]
connect_bd_net [get_bd_pins axi_interconnect_0/M${new_master_idx}_ARESETN] [get_bd_pins /pl_rst0_net]

# Connect m_axi_gmem to SmartConnect (HP port to DDR)
# SmartConnect should already exist from PED HLS; add a new slave port
set sc [get_bd_cells axi_smc]
if {$sc == ""} {
    puts "ERROR: SmartConnect axi_smc not found"
    exit 1
}
set current_slaves [get_property CONFIG.NUM_SI $sc]
set new_slaves [expr {$current_slaves + 1}]
set_property CONFIG.NUM_SI $new_slaves $sc
set new_slave_idx [format "%02d" [expr {$new_slaves - 1}]]
connect_bd_intf_net [get_bd_intf_pins plate_cnn_hls_0/m_axi_gmem] \
    [get_bd_intf_pins axi_smc/S${new_slave_idx}_AXI]
connect_bd_net [get_bd_pins axi_smc/aclk${new_slave_idx}] [get_bd_pins /pl_clk0_net]

# Assign address for s_axi_ctrl (0xA0040000)
assign_bd_address -target_address_space /zynq_ultra_ps_e_0/Data \
    [get_bd_addr_segs plate_cnn_hls_0/s_axi_ctrl/Reg] -offset 0xA0040000 -range 64K

# Assign address for m_axi_gmem (full DDR range for FC weights access)
assign_bd_address -target_address_space /plate_cnn_hls_0/Data_m_axi_gmem \
    [get_bd_addr_segs zynq_ultra_ps_e_0/SAXIGP2/HP0_DDR_LOW] -offset 0x00000000 -range 2G

# Validate + save BD
validate_bd_design
save_bd_design

# Generate wrapper + synthesize + implement + bitstream
make_wrapper -files [get_files $bd_name.bd] -top
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1

puts "Done. New bitstream at $proj_dir/xazu3eg_2G_release.runs/impl_1/design_1_wrapper.bit"
