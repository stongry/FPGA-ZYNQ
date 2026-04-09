// FZ3A Ethernet + lwIP TCP echo server (Phase 2a)
// - Init GEM3 via XEmacPs
// - Init lwIP 2.2.0 raw API
// - DHCP client, fallback to static 192.168.1.100 after 15s
// - TCP server on port 5000
// - Callback: print packet info to UART, echo back a short ACK
#include <stdint.h>
#include <string.h>
#include "xparameters.h"
#include "xil_printf.h"
#include "xil_cache.h"
#include "sleep.h"

#include "lwip/err.h"
#include "lwip/tcp.h"
#include "lwip/udp.h"
#include "lwip/inet.h"
#include "lwip/init.h"
#include "lwip/dhcp.h"
#include "lwip/timeouts.h"
#include "netif/xadapter.h"
#include "xscugic.h"
#include "xil_exception.h"

// === stdout to UART1 -> CP2102 -> COM9 ===
#define UART1_BASE   0xFF010000UL
#define UART_SR      (*(volatile uint32_t*)(UART1_BASE + 0x2C))
#define UART_FIFO    (*(volatile uint32_t*)(UART1_BASE + 0x30))
#define UART_TXFULL  (1U << 4)
void outbyte(char c) {
    while (UART_SR & UART_TXFULL) { }
    UART_FIFO = (uint32_t)(unsigned char)c;
}

// === ethernet config ===
#define PLATFORM_EMAC_BASEADDR XPAR_XEMACPS_0_BASEADDR  /* GEM3 = 0xFF0E0000 */
#define TCP_PORT 5000

static struct netif server_netif;
static unsigned char mac_ethernet_address[6] = { 0x00, 0x0A, 0x35, 0x00, 0xFC, 0x3A };

/* TCP server recv callback */
static err_t tcp_recv_cb(void *arg, struct tcp_pcb *tpcb, struct pbuf *p, err_t err) {
    (void)arg;
    if (!p) {
        xil_printf("[tcp] client closed\r\n");
        tcp_close(tpcb);
        return ERR_OK;
    }
    if (err != ERR_OK) {
        xil_printf("[tcp] recv err=%d\r\n", err);
        pbuf_free(p);
        return err;
    }
    /* Acknowledge bytes at the TCP layer */
    tcp_recved(tpcb, p->tot_len);
    xil_printf("[tcp] recv %d bytes from %d.%d.%d.%d port %d: ",
               p->tot_len,
               ip4_addr1(&tpcb->remote_ip), ip4_addr2(&tpcb->remote_ip),
               ip4_addr3(&tpcb->remote_ip), ip4_addr4(&tpcb->remote_ip),
               tpcb->remote_port);
    /* Dump first up to 32 bytes as hex */
    int dump = (p->tot_len > 32) ? 32 : p->tot_len;
    uint8_t buf[32];
    pbuf_copy_partial(p, buf, dump, 0);
    for (int i = 0; i < dump; i++) xil_printf("%02X ", buf[i]);
    if (p->tot_len > dump) xil_printf("... (%d total)", p->tot_len);
    xil_printf("\r\n");

    /* Echo back a short ACK */
    const char *ack = "ACK\r\n";
    tcp_write(tpcb, ack, 5, TCP_WRITE_FLAG_COPY);
    tcp_output(tpcb);
    pbuf_free(p);
    return ERR_OK;
}

/* TCP server accept callback */
static err_t tcp_accept_cb(void *arg, struct tcp_pcb *newpcb, err_t err) {
    (void)arg; (void)err;
    xil_printf("[tcp] accept from %d.%d.%d.%d port %d\r\n",
               ip4_addr1(&newpcb->remote_ip), ip4_addr2(&newpcb->remote_ip),
               ip4_addr3(&newpcb->remote_ip), ip4_addr4(&newpcb->remote_ip),
               newpcb->remote_port);
    tcp_recv(newpcb, tcp_recv_cb);
    return ERR_OK;
}

static void start_tcp_server(void) {
    struct tcp_pcb *pcb = tcp_new();
    if (!pcb) { xil_printf("tcp_new failed\r\n"); return; }
    err_t err = tcp_bind(pcb, IP_ADDR_ANY, TCP_PORT);
    if (err != ERR_OK) { xil_printf("tcp_bind failed: %d\r\n", err); return; }
    pcb = tcp_listen(pcb);
    if (!pcb) { xil_printf("tcp_listen failed\r\n"); return; }
    tcp_accept(pcb, tcp_accept_cb);
    xil_printf("[tcp] server listening on port %d\r\n", TCP_PORT);
}

static void print_ip(const char *label, const ip_addr_t *ip) {
    xil_printf("%s: %d.%d.%d.%d\r\n", label,
               ip4_addr1(ip), ip4_addr2(ip), ip4_addr3(ip), ip4_addr4(ip));
}

int main(void) {
    Xil_DCacheDisable();  /* Simplest for emacps DMA coherency */
    Xil_ICacheDisable();

    xil_printf("\r\n===========================================\r\n");
    xil_printf("  FZ3A ETHERNET + lwIP TCP echo (Phase 2a)\r\n");
    xil_printf("===========================================\r\n");
    xil_printf("EMAC base = 0x%08X (GEM3)\r\n", PLATFORM_EMAC_BASEADDR);
    xil_printf("MAC       = %02X:%02X:%02X:%02X:%02X:%02X\r\n",
               mac_ethernet_address[0], mac_ethernet_address[1],
               mac_ethernet_address[2], mac_ethernet_address[3],
               mac_ethernet_address[4], mac_ethernet_address[5]);

    /* ===== GIC SETUP (required by Xilinx lwIP adapter) ===== */
    xil_printf("GIC setup...\r\n");
    Xil_ExceptionInit();
    XScuGic_DeviceInitialize(XPAR_SCUGIC_0_DEVICE_ID);
    Xil_ExceptionRegisterHandler(XIL_EXCEPTION_ID_IRQ_INT,
        (Xil_ExceptionHandler)XScuGic_DeviceInterruptHandler,
        (void*)XPAR_SCUGIC_0_DEVICE_ID);
    Xil_ExceptionEnable();
    xil_printf("GIC enabled, IRQ unmasked\r\n");

    xil_printf("lwip_init...\r\n");
    lwip_init();

    /* Register EMAC with lwIP. Start with 0.0.0.0 for DHCP. */
    ip_addr_t ip_zero, nm_zero, gw_zero;
    IP4_ADDR(&ip_zero, 0, 0, 0, 0);
    IP4_ADDR(&nm_zero, 0, 0, 0, 0);
    IP4_ADDR(&gw_zero, 0, 0, 0, 0);

    xil_printf("xemac_add...\r\n");
    if (!xemac_add(&server_netif, &ip_zero, &nm_zero, &gw_zero,
                   mac_ethernet_address, PLATFORM_EMAC_BASEADDR)) {
        xil_printf("xemac_add FAILED\r\n");
        while (1) __asm__ volatile("wfe");
    }
    netif_set_default(&server_netif);
    netif_set_up(&server_netif);
    xil_printf("netif up\r\n");

    xil_printf("dhcp_start...\r\n");
    dhcp_start(&server_netif);

    /* Poll for DHCP to bind */
    int dhcp_tries = 0;
    while (dhcp_supplied_address(&server_netif) == 0 && dhcp_tries < 150) {
        xemacif_input(&server_netif);
        sys_check_timeouts();
        usleep(100000);
        dhcp_tries++;
        if (dhcp_tries % 10 == 0) xil_printf("  waiting DHCP (%d)...\r\n", dhcp_tries);
    }

    if (dhcp_supplied_address(&server_netif)) {
        xil_printf("DHCP OK\r\n");
        print_ip("  IP ", &server_netif.ip_addr);
        print_ip("  GW ", &server_netif.gw);
        print_ip("  NM ", &server_netif.netmask);
    } else {
        xil_printf("DHCP timeout - falling back to static 192.168.6.210\r\n");
        ip_addr_t ip, nm, gw;
        IP4_ADDR(&ip, 192, 168, 6, 210);
        IP4_ADDR(&nm, 255, 255, 255, 0);
        IP4_ADDR(&gw, 192, 168, 6, 1);
        netif_set_addr(&server_netif, &ip, &nm, &gw);
        xil_printf("  ping me at 192.168.6.210 from Windows to test RGMII\r\n");
    }

    start_tcp_server();

    xil_printf("entering main loop (IRQ-driven RX + poll drain)\r\n");
    uint32_t tick = 0;
    uint32_t pkts_before = 0;
    uint32_t last_reported = 0;
    while (1) {
        int n = xemacif_input(&server_netif);
        if (n > 0) pkts_before += n;
        sys_check_timeouts();
        tick++;
        if (tick % 500000 == 0) {
            if (pkts_before != last_reported) {
                xil_printf("rx=%d\r\n", pkts_before);
                last_reported = pkts_before;
            }
        }
    }
    return 0;
}
