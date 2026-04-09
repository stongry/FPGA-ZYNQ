connect -url tcp:localhost:3121
targets -set -nocase -filter {name =~ "PSU"}
puts "=== GEM3 MAC registers ==="
puts [format "NWCTRL       = 0x%08X" [lindex [mrd -value 0xFF0E0000] 0]]
puts [format "NWCFG        = 0x%08X (bit0=speed100, bit10=gigabit)" [lindex [mrd -value 0xFF0E0004] 0]]
puts [format "Frames TX OK = 0x%08X" [lindex [mrd -value 0xFF0E0108] 0]]
puts [format "Frames RX OK = 0x%08X" [lindex [mrd -value 0xFF0E015C] 0]]
puts [format "Broadcast TX = 0x%08X" [lindex [mrd -value 0xFF0E010C] 0]]
puts [format "Broadcast RX = 0x%08X" [lindex [mrd -value 0xFF0E0160] 0]]
puts [format "Alignment err= 0x%08X" [lindex [mrd -value 0xFF0E0190] 0]]
puts [format "FCS err      = 0x%08X" [lindex [mrd -value 0xFF0E0194] 0]]
puts [format "SymbErr      = 0x%08X" [lindex [mrd -value 0xFF0E01A8] 0]]
puts [format "Resource err = 0x%08X" [lindex [mrd -value 0xFF0E019C] 0]]
puts [format "RX OvrRun    = 0x%08X" [lindex [mrd -value 0xFF0E01A4] 0]]
puts "=== IOU_SLCR GEM3 RGMII control ==="
puts [format "GEM_CLK_CTRL (0xFF180308) = 0x%08X" [lindex [mrd -value 0xFF180308] 0]]
puts [format "GEM_CTRL     (0xFF180360) = 0x%08X" [lindex [mrd -value 0xFF180360] 0]]
disconnect
exit
