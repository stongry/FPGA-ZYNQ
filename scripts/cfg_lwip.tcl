puts "==== configure lwIP DHCP + timers ===="
setws C:/Users/huye/fz3a/vitis_ws
platform active fz3a_plat
domain active standalone_domain
puts "--- list lwip options ---"
catch {bsp listparams -lib lwip220} lp
puts $lp
puts "--- enable dhcp/timers ---"
catch {bsp config lwip_dhcp true} r1; puts "lwip_dhcp: $r1"
catch {bsp config no_sys_no_timers false} r2; puts "no_sys_no_timers: $r2"
puts "--- bsp write/regenerate/platform generate ---"
bsp write
bsp regenerate
platform generate
puts "==== DONE ===="
exit
