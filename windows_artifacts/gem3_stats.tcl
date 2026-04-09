connect -url tcp:localhost:3121
targets -set -nocase -filter {name =~ "PSU"}
puts "=== GEM3 0xFF0E0000 stats dump ==="
puts [format "  NWCTRL (+0x000)  = 0x%08X" [lindex [mrd -value 0xFF0E0000] 0]]
puts [format "  NWCFG  (+0x004)  = 0x%08X" [lindex [mrd -value 0xFF0E0004] 0]]
puts [format "  NWSR   (+0x008)  = 0x%08X" [lindex [mrd -value 0xFF0E0008] 0]]
puts [format "  DMACR  (+0x010)  = 0x%08X" [lindex [mrd -value 0xFF0E0010] 0]]
puts [format "  TXSR   (+0x014)  = 0x%08X" [lindex [mrd -value 0xFF0E0014] 0]]
puts [format "  RXSR   (+0x020)  = 0x%08X" [lindex [mrd -value 0xFF0E0020] 0]]
puts [format "  ISR    (+0x024)  = 0x%08X" [lindex [mrd -value 0xFF0E0024] 0]]
puts [format "  TxQ0 BASE(+0x01C)= 0x%08X" [lindex [mrd -value 0xFF0E001C] 0]]
puts [format "  RxQ0 BASE(+0x018)= 0x%08X" [lindex [mrd -value 0xFF0E0018] 0]]
puts "=== STATS counters (+0x100+) ==="
puts [format "  Octets TX lo    = 0x%08X" [lindex [mrd -value 0xFF0E0104] 0]]
puts [format "  Frames TX OK    = 0x%08X" [lindex [mrd -value 0xFF0E0108] 0]]
puts [format "  Broadcast TX    = 0x%08X" [lindex [mrd -value 0xFF0E010C] 0]]
puts [format "  Octets RX lo    = 0x%08X" [lindex [mrd -value 0xFF0E0158] 0]]
puts [format "  Frames RX OK    = 0x%08X" [lindex [mrd -value 0xFF0E015C] 0]]
puts [format "  Broadcast RX    = 0x%08X" [lindex [mrd -value 0xFF0E0160] 0]]
puts [format "  Alignment errs  = 0x%08X" [lindex [mrd -value 0xFF0E0190] 0]]
puts [format "  FCS errs        = 0x%08X" [lindex [mrd -value 0xFF0E0194] 0]]
disconnect
exit
