// Minimal newlib syscall stubs for bare-metal
#include <stddef.h>
#include <sys/stat.h>
#include <errno.h>
#include <unistd.h>

int errno;

void _exit(int status) { (void)status; while (1) __asm__ volatile("wfe"); }
int _close(int fd) { (void)fd; return -1; }
int _fstat(int fd, struct stat *st) { (void)fd; if (st) st->st_mode = S_IFCHR; return 0; }
int _isatty(int fd) { (void)fd; return 1; }
off_t _lseek(int fd, off_t off, int w) { (void)fd; (void)off; (void)w; return 0; }
int _read(int fd, void *buf, size_t cnt) { (void)fd; (void)buf; (void)cnt; return 0; }
int _write(int fd, const char *buf, size_t cnt) {
    extern void outbyte(char c);
    (void)fd;
    for (size_t i = 0; i < cnt; i++) outbyte(buf[i]);
    return (int)cnt;
}
int _kill(int pid, int sig) { (void)pid; (void)sig; errno = EINVAL; return -1; }
int _getpid(void) { return 1; }
int _open(const char *p, int f, int m) { (void)p; (void)f; (void)m; return -1; }
int _link(const char *o, const char *n) { (void)o; (void)n; errno = EMLINK; return -1; }
int _unlink(const char *n) { (void)n; errno = ENOENT; return -1; }
int _stat(const char *f, struct stat *st) { (void)f; if (st) st->st_mode = S_IFCHR; return 0; }
int _times(void *buf) { (void)buf; return -1; }

extern char _heap_start, _heap_end;
static char *heap_ptr = &_heap_start;
void *_sbrk(intptr_t inc) {
    char *prev = heap_ptr;
    if (heap_ptr + inc > &_heap_end) { errno = ENOMEM; return (void*)-1; }
    heap_ptr += inc;
    return prev;
}

/* lwIP requires sys_now() returning milliseconds since start (NO_SYS mode) */
#include <stdint.h>
uint32_t sys_now(void) {
    uint64_t cnt, freq;
    __asm__ volatile("mrs %0, cntpct_el0" : "=r"(cnt));
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(freq));
    /* Convert counter ticks to milliseconds */
    return (uint32_t)((cnt * 1000ULL) / freq);
}
