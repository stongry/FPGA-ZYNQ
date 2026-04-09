$ErrorActionPreference = 'Continue'
Set-Location C:\Users\huye\fz3a\dp
$bsp = 'C:\Users\huye\fz3a\vitis_ws\fz3a_plat\psu_cortexa53_0\standalone_domain\bsp\psu_cortexa53_0'
$gcc = 'C:\Xilinx\Vitis\2024.2\gnu\aarch64\nt\aarch64-none\bin\aarch64-none-elf-gcc.exe'
$incl = "$bsp\include"
$lib  = "$bsp\lib"
& $gcc -O2 -mcpu=cortex-a53 -ffreestanding `
  "-I$incl" `
  -T lscript.ld `
  "-L$lib" `
  '-Wl,--start-group' `
  phase2b_main.c stubs.c `
  -lxil -llwip4 -lgcc -lc `
  '-Wl,--end-group' `
  -o phase2b.elf 2>&1 | Tee-Object build.log
Write-Host "EXIT=$LASTEXITCODE"
if (Test-Path phase2b.elf) {
  Get-ChildItem phase2b.elf | Select-Object Name,Length,LastWriteTime
}
