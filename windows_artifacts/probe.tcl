puts "==== DP register probe ===="
connect -url tcp:localhost:3121
targets -set -nocase -filter {name =~ "PSU"}
catch {source C:/Users/huye/fz3a/psu_init.tcl}
catch {psu_init}
catch {psu_post_config}
after 200

# DP_PSU register base = 0xFD4A0000
# XDPPSU_INTERRUPT_SIGNAL_STATE = offset 0x130, bit 0 = HPD raw
# XDPPSU_VERSION = offset 0x800
puts ">>> DP register dump:"
puts [format "DP base 0xFD4A0000 = 0x%08X" [lindex [mrd -value 0xFD4A0000] 0]]
puts [format "DP VERSION (+0x800) = 0x%08X" [lindex [mrd -value 0xFD4A0800] 0]]
puts [format "DP INTR_SIGNAL_STATE (+0x130) = 0x%08X" [lindex [mrd -value 0xFD4A0130] 0]]
puts [format "DP INTR_STATUS (+0x140) = 0x%08X" [lindex [mrd -value 0xFD4A0140] 0]]
puts [format "DP INTR_MASK (+0x144) = 0x%08X" [lindex [mrd -value 0xFD4A0144] 0]]
puts [format "DP TRANSMITTER_ENABLE (+0x80) = 0x%08X" [lindex [mrd -value 0xFD4A0080] 0]]
puts [format "DP MAIN_STREAM_ENABLE (+0x84) = 0x%08X" [lindex [mrd -value 0xFD4A0084] 0]]

# Also check IOU_SLCR MIO 28 config (DP HPD pin)
# IOU_SLCR base = 0xFF180000, MIO_PIN_28 offset = 0x70 (= 4 * 28)
puts [format "IOU MIO_PIN_28 (0xFF180070) = 0x%08X (function bits 1:5)" [lindex [mrd -value 0xFF180070] 0]]

puts {>>> done}
disconnect
exit
