puts "==== add lwip220 via bsp setlib ===="
setws C:/Users/huye/fz3a/vitis_ws
platform active fz3a_plat
domain active standalone_domain
puts "==== bsp setlib ===="
bsp setlib -name lwip220 -ver 1.1
puts "==== bsp write ===="
bsp write
puts "==== bsp regenerate ===="
bsp regenerate
puts "==== platform generate ===="
platform generate
puts "==== DONE ===="
exit
