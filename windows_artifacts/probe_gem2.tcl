connect -url tcp:localhost:3121
targets -set -nocase -filter {name =~ "PSU"}
puts [format "Frames TX OK = 0x%08X" [lindex [mrd -value 0xFF0E0108] 0]]
puts [format "Frames RX OK = 0x%08X" [lindex [mrd -value 0xFF0E015C] 0]]
puts [format "Broadcast RX = 0x%08X" [lindex [mrd -value 0xFF0E0160] 0]]
puts [format "Multicast RX = 0x%08X" [lindex [mrd -value 0xFF0E0164] 0]]
puts [format "RXSR         = 0x%08X" [lindex [mrd -value 0xFF0E0020] 0]]
puts [format "RxQ0 BASE    = 0x%08X" [lindex [mrd -value 0xFF0E0018] 0]]
puts [format "RX buffer QSize = 0x%08X" [lindex [mrd -value 0xFF0E0024] 0]]
puts [format "ISR          = 0x%08X" [lindex [mrd -value 0xFF0E0024] 0]]
disconnect
exit
