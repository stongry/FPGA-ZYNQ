connect -url tcp:localhost:3121
targets -set -nocase -filter {name =~ "PSU"}
puts "=== GEM3 RX Queue base address ==="
puts [format "RxQ0 BASE = 0x%08X" [lindex [mrd -value 0xFF0E0018] 0]]
puts [format "NWCTRL = 0x%08X (bit3=RX_EN)" [lindex [mrd -value 0xFF0E0000] 0]]
puts [format "NWCFG = 0x%08X" [lindex [mrd -value 0xFF0E0004] 0]]
puts [format "Frames RX OK  = 0x%08X" [lindex [mrd -value 0xFF0E015C] 0]]
puts [format "Frames TX OK  = 0x%08X" [lindex [mrd -value 0xFF0E0108] 0]]

puts "=== RX BD ring at 0x00600000 (first 16 BDs, each 8 bytes) ==="
for {set i 0} {$i < 16} {incr i} {
    set addr [expr {0x00600000 + $i*8}]
    set w0 [lindex [mrd -value $addr] 0]
    set w1 [lindex [mrd -value [expr {$addr+4}]] 0]
    set used [expr {$w0 & 1}]
    set buf_addr [expr {$w0 & 0xFFFFFFFC}]
    set len [expr {$w1 & 0x1FFF}]
    puts [format "  BD%02d @ 0x%08X: w0=0x%08X w1=0x%08X  used=%d buf=0x%08X len=%d" $i $addr $w0 $w1 $used $buf_addr $len]
}
disconnect
exit
