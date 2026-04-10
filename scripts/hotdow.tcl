connect
targets -set -nocase -filter {name =~ "Cortex-A53*0"}
catch {stop}
rst -processor
dow C:/Users/huye/fz3a/dp/phase2b.elf
con
disconnect
