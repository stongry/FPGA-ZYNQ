puts "==== DP TEST with NEW psu_init from regen BD ===="
connect -url tcp:localhost:3121
source C:/Users/huye/fz3a/xazu3eg_2G_release/xazu3eg_2G_release.srcs/sources_1/bd/design_1/ip/design_1_zynq_ultra_ps_e_0_0/psu_init.tcl
targets -set -nocase -filter {name =~ "PSU"}
catch {psu_init} ie
puts "psu_init: $ie"
catch {psu_post_config}
catch {psu_ps_pl_isolation_removal}
catch {psu_ps_pl_reset_config}
after 500

puts "==== program PL bit (existing) ===="
catch {fpga -file C:/Users/huye/fz3a/xazu3eg_2G_release/xazu3eg_2G_release.runs/impl_1/design_1_wrapper.bit}

puts "==== select Cortex-A53 #0 ===="
targets -set -nocase -filter {name =~ "Cortex-A53 #0"}
catch {rst -processor}
after 200
catch {stop}

puts "==== dow dp_test.elf ===="
dow C:/Users/huye/fz3a/dp/dp_test.elf

puts "==== con ===="
con
after 18000
catch {stop}
disconnect
exit
