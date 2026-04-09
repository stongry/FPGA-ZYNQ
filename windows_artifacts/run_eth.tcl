connect -url tcp:localhost:3121
source C:/Users/huye/fz3a/xazu3eg_2G_release/xazu3eg_2G_release.srcs/sources_1/bd/design_1/ip/design_1_zynq_ultra_ps_e_0_0/psu_init.tcl
targets -set -nocase -filter {name =~ "PSU"}
catch {psu_init}
catch {psu_post_config}
after 300
targets -set -nocase -filter {name =~ "Cortex-A53 #0"}
catch {rst -processor}
after 300
catch {stop}
dow C:/Users/huye/fz3a/dp/eth_test.elf
con
after 50000
catch {stop}
disconnect
exit
