setws C:/Users/huye/fz3a/vitis_ws
platform active fz3a_plat
domain active standalone_domain
puts "==== bsp help ===="
catch {help bsp} out
puts $out
puts "==== library help ===="
catch {help library} out2
puts $out2
exit
