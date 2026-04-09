/* FZ3A DP test pattern - DDR-based linker script
 * Code/data at 0x00100000, framebuffer at 0x10000000 (256MB)
 * Uses BSP boot.S which provides _vector_table entry point
 */
_STACK_SIZE = DEFINED(_STACK_SIZE) ? _STACK_SIZE : 0x10000;  /* 64K stack */
_HEAP_SIZE  = DEFINED(_HEAP_SIZE)  ? _HEAP_SIZE  : 0x10000;  /* 64K heap */

MEMORY
{
   ddr_low  : ORIGIN = 0x00100000, LENGTH = 0x0FF00000  /* 255 MB code/data */
   ddr_fb   : ORIGIN = 0x10000000, LENGTH = 0x00800000  /* 8 MB framebuffer */
}

ENTRY(_vector_table)

SECTIONS
{
.text : {
   KEEP (*(.vectors))
   *(.boot)
   *(.text)
   *(.text.*)
   *(.gnu.linkonce.t.*)
   *(.plt)
   *(.gnu_warning)
   *(.gcc_execpt_table)
   *(.glue_7)
   *(.glue_7t)
   *(.ARM.extab)
   *(.gnu.linkonce.armextab.*)
} > ddr_low

.note.gnu.build-id : {
   KEEP (*(.note.gnu.build-id))
} > ddr_low

.init (ALIGN(64)): { KEEP (*(.init)) } > ddr_low
.fini (ALIGN(64)): { KEEP (*(.fini)) } > ddr_low

.rodata (ALIGN(64)): {
   __rodata_start = .;
   *(.rodata) *(.rodata.*) *(.gnu.linkonce.r.*)
   __rodata_end = .;
} > ddr_low

.rodata1 (ALIGN(64)): { *(.rodata1) *(.rodata1.*) } > ddr_low
.eh_frame    : { *(.eh_frame) } > ddr_low
.eh_framehdr : { *(.eh_framehdr) } > ddr_low
.gcc_except_table : { *(.gcc_except_table) } > ddr_low

.mmu_tbl0 (ALIGN(4096)) : {
   __mmu_tbl0_start = .;
   *(.mmu_tbl0)
   __mmu_tbl0_end = .;
} > ddr_low

.mmu_tbl1 (ALIGN(4096)) : {
   __mmu_tbl1_start = .;
   *(.mmu_tbl1)
   __mmu_tbl1_end = .;
} > ddr_low

.mmu_tbl2 (ALIGN(4096)) : {
   __mmu_tbl2_start = .;
   *(.mmu_tbl2)
   __mmu_tbl2_end = .;
} > ddr_low

.data (ALIGN(64)): {
   __data_start = .;
   *(.data) *(.data.*) *(.gnu.linkonce.d.*)
   *(.jcr) *(.got) *(.got.plt)
   __data_end = .;
} > ddr_low

.tdata (ALIGN(64)): {
   __tdata_start = .;
   *(.tdata) *(.tdata.*) *(.gnu.linkonce.td.*)
   __tdata_end = .;
} > ddr_low

.fixup : {
   __fixup_start = .;
   *(.fixup)
   __fixup_end = .;
} > ddr_low

.ARM.exidx : {
   __exidx_start = .;
   *(.ARM.exidx*)
   __exidx_end = .;
} > ddr_low

.preinit_array (ALIGN(64)): {
   __preinit_array_start = .;
   KEEP (*(SORT(.preinit_array.*))) KEEP (*(.preinit_array))
   __preinit_array_end = .;
} > ddr_low

.init_array (ALIGN(64)): {
   __init_array_start = .;
   KEEP (*(SORT(.init_array.*))) KEEP (*(.init_array))
   __init_array_end = .;
} > ddr_low

.fini_array (ALIGN(64)): {
   __fini_array_start = .;
   KEEP (*(SORT(.fini_array.*))) KEEP (*(.fini_array))
   __fini_array_end = .;
} > ddr_low

.sdata (ALIGN(64)): {
   __sdata_start = .;
   *(.sdata) *(.sdata.*) *(.gnu.linkonce.s.*)
   __sdata_end = .;
} > ddr_low

.sbss (NOLOAD) : {
   __sbss_start = .;
   . = ALIGN(64);
   *(.sbss) *(.sbss.*) *(.gnu.linkonce.sb.*)
   . = ALIGN(64);
   __sbss_end = .;
} > ddr_low

.tbss (ALIGN(64)): {
   __tbss_start = .;
   *(.tbss) *(.tbss.*) *(.gnu.linkonce.tb.*)
   __tbss_end = .;
} > ddr_low

.bss (NOLOAD) : {
   . = ALIGN(64);
   __bss_start__ = .;
   *(.bss) *(.bss.*) *(.gnu.linkonce.b.*)
   *(COMMON)
   . = ALIGN(64);
   __bss_end__ = .;
} > ddr_low

_SDA_BASE_  = __sdata_start  + ((__sbss_end - __sdata_start) / 2);

.heap (NOLOAD) : {
   . = ALIGN(64);
   _heap = .;
   HeapBase = .;
   _heap_start = .;
   . += _HEAP_SIZE;
   _heap_end = .;
   HeapLimit = .;
} > ddr_low

.stack (NOLOAD) : {
   . = ALIGN(64);
   _el3_stack_end = .;
   . += _STACK_SIZE;
   __el3_stack = .;
   __el2_stack = .;
   __el1_stack = .;
   __el0_stack = .;
} > ddr_low

/* Framebuffer placed at fixed DDR address, 8 MB */
.framebuffer (NOLOAD) : {
   . = ALIGN(4096);
   *(.framebuffer)
   *(.framebuffer.*)
} > ddr_fb

_end = .;
}
