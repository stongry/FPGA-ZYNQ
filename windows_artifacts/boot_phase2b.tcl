puts "== boot_phase2b.tcl =="
connect
# Full system reset with register clear to recover from wedged A53 debug.
targets -set -nocase -filter {name =~ "PSU"}
catch { rst -system -clear-registers }
after 500
# PSU init (DDR, clocks, MIO) using Vivado-exported tcl
source C:/Users/huye/fz3a/vitis_ws/fz3a_plat/hw/psu_init.tcl
puts "running psu_init..."
catch {psu_init}
after 500
catch {psu_ps_pl_isolation_removal}
after 100
catch {psu_ps_pl_reset_config}
puts "psu_init complete"
# Halt A53#0 and load our app (DDR is now valid)
targets -set -nocase -filter {name =~ "Cortex-A53*0"}
catch {stop}
catch { rst -processor }
after 200
puts "loading phase2b.elf..."
dow C:/Users/huye/fz3a/dp/phase2b.elf
con
puts "== DONE =="
disconnect
