// FZ3A DisplayPort test pattern (Phase 1)
// Bare-metal Cortex-A53 #0, links against Xilinx BSP libxil.a
// Outputs 1920x1080@60 RGBA8888 framebuffer through PS DP -> miniDP
//
// Polling-based, no interrupts, no GIC. Polls XDpPsu_IsConnected() in main loop
// and runs link training when monitor is detected.

#include <stdint.h>
#include "xparameters.h"
#include "xdpdma.h"
#include "xdppsu.h"
#include "xavbuf.h"
#include "xavbuf_clk.h"
#include "xil_printf.h"
#include "xil_cache.h"
#include "sleep.h"

// === stdout override: send xil_printf to UART1 (CP2102 -> COM9) ===
#define UART1_BASE   0xFF010000UL
#define UART_SR      (*(volatile uint32_t*)(UART1_BASE + 0x2C))
#define UART_FIFO    (*(volatile uint32_t*)(UART1_BASE + 0x30))
#define UART_TXFULL  (1U << 4)

void outbyte(char c) {
    while (UART_SR & UART_TXFULL) { }
    UART_FIFO = (uint32_t)(unsigned char)c;
}

// === Configuration ===
typedef enum {
    LANE_COUNT_1 = 1,
    LANE_COUNT_2 = 2,
} LaneCount_t;

typedef enum {
    LINK_RATE_162GBPS = 0x06,
    LINK_RATE_270GBPS = 0x0A,
    LINK_RATE_540GBPS = 0x14,
} LinkRate_t;

typedef struct {
    XDpPsu  *DpPsuPtr;
    XAVBuf  *AVBufPtr;
    XDpDma  *DpDmaPtr;
    XVidC_VideoMode      VideoMode;
    XVidC_ColorDepth     Bpc;
    XDpPsu_ColorEncoding ColorEncode;
    u8 UseMaxLaneCount;
    u8 UseMaxLinkRate;
    u8 LaneCount;
    u8 LinkRate;
    u8 EnSynchClkMode;
    u32 PixClkHz;
} Run_Config;

/* Framebuffer allocated big enough for 1920x1080 but we use 1280x720 */
#define FB_ALLOC_SIZE (1920 * 1080 * 4)
#define FB_W         1280
#define FB_H         720
#define BUFFERSIZE   (FB_W * FB_H * 4)
#define LINESIZE     (FB_W * 4)
#define STRIDE       LINESIZE

// Place framebuffer at fixed DDR address far from code (.text @ 0x100000)
// Code is < 1MB, weights are bss, framebuffer at 16 MB safe distance
__attribute__((aligned(256), section(".framebuffer")))
u8 Frame[FB_ALLOC_SIZE];

XDpPsu DpPsu;
XAVBuf AVBuf;
XDpDma DpDma;
Run_Config RunCfg;
XDpDma_FrameBuffer FrameBuffer;

// === Init the run config ===
static void InitRunConfig(Run_Config *cfg) {
    cfg->DpPsuPtr        = &DpPsu;
    cfg->AVBufPtr        = &AVBuf;
    cfg->DpDmaPtr        = &DpDma;
    cfg->VideoMode       = XVIDC_VM_1280x720_60_P;
    cfg->Bpc             = XVIDC_BPC_8;
    cfg->ColorEncode     = XDPPSU_CENC_RGB;
    cfg->UseMaxLaneCount = 1;
    cfg->UseMaxLinkRate  = 1;
    cfg->LaneCount       = LANE_COUNT_2;
    cfg->LinkRate        = LINK_RATE_540GBPS;
    cfg->EnSynchClkMode  = 0;
}

// === Init the DP / DPDMA / AVBuf subsystem (mirrors Xilinx example) ===
static int InitDpDmaSubsystem(Run_Config *cfg) {
    XDpPsu_Config       *DpPsuCfgPtr;
    XDpDma_Config       *DpDmaCfgPtr;
    XDpPsu *DpPsuPtr = cfg->DpPsuPtr;
    XAVBuf *AVBufPtr = cfg->AVBufPtr;
    XDpDma *DpDmaPtr = cfg->DpDmaPtr;
    int Status;

    DpPsuCfgPtr = XDpPsu_LookupConfig(XPAR_PSU_DP_DEVICE_ID);
    XDpPsu_CfgInitialize(DpPsuPtr, DpPsuCfgPtr, DpPsuCfgPtr->BaseAddr);

    XAVBuf_CfgInitialize(AVBufPtr, DpPsuPtr->Config.BaseAddr, XPAR_PSU_DP_DEVICE_ID);

    DpDmaCfgPtr = XDpDma_LookupConfig(XPAR_XDPDMA_0_DEVICE_ID);
    XDpDma_CfgInitialize(DpDmaPtr, DpDmaCfgPtr);

    Status = XDpPsu_InitializeTx(DpPsuPtr);
    if (Status != XST_SUCCESS) { xil_printf("XDpPsu_InitializeTx FAILED\r\n"); return XST_FAILURE; }

    Status = XDpDma_SetGraphicsFormat(DpDmaPtr, RGBA8888);
    if (Status != XST_SUCCESS) { xil_printf("XDpDma_SetGraphicsFormat FAILED\r\n"); return XST_FAILURE; }

    Status = XAVBuf_SetInputNonLiveGraphicsFormat(AVBufPtr, RGBA8888);
    if (Status != XST_SUCCESS) { xil_printf("XAVBuf_SetInputNonLiveGraphicsFormat FAILED\r\n"); return XST_FAILURE; }

    XDpDma_SetQOS(DpDmaPtr, 11);
    XAVBuf_EnableGraphicsBuffers(AVBufPtr, 1);
    XAVBuf_SetOutputVideoFormat(AVBufPtr, RGB_8BPC);
    XAVBuf_InputVideoSelect(AVBufPtr, XAVBUF_VIDSTREAM1_NONE, XAVBUF_VIDSTREAM2_NONLIVE_GFX);
    XAVBuf_ConfigureGraphicsPipeline(AVBufPtr);
    XAVBuf_ConfigureOutputVideo(AVBufPtr);
    XAVBuf_SetBlenderAlpha(AVBufPtr, 0, 0);  /* Match example: use pixel alpha */
    XDpPsu_CfgMsaEnSynchClkMode(DpPsuPtr, cfg->EnSynchClkMode);
    XAVBuf_SetAudioVideoClkSrc(AVBufPtr, XAVBUF_PS_CLK, XAVBUF_PS_CLK);
    XAVBuf_SoftReset(AVBufPtr);
    return XST_SUCCESS;
}

// === Wake up the monitor (DPCD power-up) ===
static u32 DpPsu_Wakeup(Run_Config *cfg) {
    u8 AuxData = 0x1;
    u32 Status = XDpPsu_AuxWrite(cfg->DpPsuPtr, XDPPSU_DPCD_SET_POWER_DP_PWR_VOLTAGE, 1, &AuxData);
    if (Status != XST_SUCCESS) xil_printf("Wakeup1 FAIL\r\n");
    Status = XDpPsu_AuxWrite(cfg->DpPsuPtr, XDPPSU_DPCD_SET_POWER_DP_PWR_VOLTAGE, 1, &AuxData);
    if (Status != XST_SUCCESS) xil_printf("Wakeup2 FAIL\r\n");
    return Status;
}

// === Link training ===
static u32 DpPsu_HpdTrain(Run_Config *cfg) {
    XDpPsu *Dp = cfg->DpPsuPtr;
    XDpPsu_LinkConfig *Lc = &Dp->LinkConfig;
    u32 Status;
    Status = XDpPsu_GetRxCapabilities(Dp);
    if (Status != XST_SUCCESS) { xil_printf("GetRxCaps FAIL\r\n"); return XST_FAILURE; }
    Status = XDpPsu_SetEnhancedFrameMode(Dp, Lc->SupportEnhancedFramingMode ? 1 : 0);
    if (Status != XST_SUCCESS) { xil_printf("EFM FAIL\r\n"); return XST_FAILURE; }
    Status = XDpPsu_SetLaneCount(Dp, cfg->UseMaxLaneCount ? Lc->MaxLaneCount : cfg->LaneCount);
    if (Status != XST_SUCCESS) { xil_printf("LaneCount FAIL\r\n"); return XST_FAILURE; }
    Status = XDpPsu_SetLinkRate(Dp, cfg->UseMaxLinkRate ? Lc->MaxLinkRate : cfg->LinkRate);
    if (Status != XST_SUCCESS) { xil_printf("LinkRate FAIL\r\n"); return XST_FAILURE; }
    Status = XDpPsu_SetDownspread(Dp, Lc->SupportDownspreadControl);
    if (Status != XST_SUCCESS) { xil_printf("Downspread FAIL\r\n"); return XST_FAILURE; }
    xil_printf("Lanes=%d, LinkRate=%d, training...\r\n", Dp->LinkConfig.LaneCount, Dp->LinkConfig.LinkRate);
    Status = XDpPsu_EstablishLink(Dp);
    if (Status == XST_SUCCESS) xil_printf("Training OK\r\n");
    else                       xil_printf("Training FAIL\r\n");
    return Status;
}

// === Setup MSA + start video stream ===
static void DpPsu_SetupVideoStream(Run_Config *cfg) {
    XDpPsu *Dp = cfg->DpPsuPtr;
    XDpPsu_MainStreamAttributes *Msa = &Dp->MsaConfig;
    XDpPsu_SetColorEncode(Dp, cfg->ColorEncode);
    XDpPsu_CfgMsaSetBpc(Dp, cfg->Bpc);
    XDpPsu_CfgMsaUseStandardVideoMode(Dp, cfg->VideoMode);
    cfg->PixClkHz = Msa->PixelClockHz;
    XAVBuf_SetPixelClock(cfg->PixClkHz);
    XDpPsu_WriteReg(Dp->Config.BaseAddr, XDPPSU_SOFT_RESET, 0x1);
    usleep(10);
    XDpPsu_WriteReg(Dp->Config.BaseAddr, XDPPSU_SOFT_RESET, 0x0);
    XDpPsu_SetMsaValues(Dp);
    XDpPsu_WriteReg(Dp->Config.BaseAddr, 0xB124, 0x3);
    usleep(10);
    XDpPsu_WriteReg(Dp->Config.BaseAddr, 0xB124, 0x0);
    XDpPsu_EnableMainLink(Dp, 1);
    xil_printf("Video stream up\r\n");
}

// === Generate a super-simple test pattern: 3 horizontal bands R/G/B ===
// At 1280x720 the layout is:
//   y =   0..239  -> solid RED    (0xFFFF0000 in RGBA where A=FF,B=00,G=00,R=FF)
//   y = 240..479  -> solid GREEN
//   y = 480..719  -> solid BLUE
// In memory as u32 little-endian RGBA8888: byte0=R byte1=G byte2=B byte3=A
// So u32 value = (A<<24)|(B<<16)|(G<<8)|R
static void GenerateTestPattern(u8 *fb) {
    u32 *p = (u32*)fb;
    const u32 RED   = 0xFF0000FF;  /* A=FF B=00 G=00 R=FF */
    const u32 GREEN = 0xFF00FF00;  /* A=FF B=00 G=FF R=00 */
    const u32 BLUE  = 0xFFFF0000;  /* A=FF B=FF G=00 R=00 */
    const u32 WHITE = 0xFFFFFFFF;
    const int W = 1280, H = 720;
    for (int y = 0; y < H; y++) {
        u32 c;
        if      (y < H / 3)      c = RED;
        else if (y < 2 * H / 3)  c = GREEN;
        else                     c = BLUE;
        for (int x = 0; x < W; x++) p[y * W + x] = c;
    }
    /* Big white border */
    for (int x = 0; x < W; x++) {
        for (int t = 0; t < 10; t++) {
            p[t * W + x]         = WHITE;
            p[(H - 1 - t) * W + x] = WHITE;
        }
    }
    for (int y = 0; y < H; y++) {
        for (int t = 0; t < 10; t++) {
            p[y * W + t]         = WHITE;
            p[y * W + (W - 1 - t)] = WHITE;
        }
    }
}

int main(void) {
    Xil_DCacheDisable();
    Xil_ICacheDisable();
    xil_printf("\r\n===========================================\r\n");
    xil_printf("  FZ3A DP TEST PATTERN (bare-metal, no GIC)\r\n");
    xil_printf("===========================================\r\n");
    xil_printf("Frame buffer @ %p, size=%d bytes\r\n", Frame, BUFFERSIZE);
    xil_printf("Generating test pattern (%dx%d)...\r\n", FB_W, FB_H);
    GenerateTestPattern(Frame);
    /* Verify readback: sample a few pixels to confirm they're really in DDR */
    u32 *fb = (u32*)Frame;
    xil_printf("Readback: p[0]=0x%08X p[mid]=0x%08X p[near_bottom]=0x%08X\r\n",
               fb[0], fb[FB_W * (FB_H/2)], fb[FB_W * (FB_H - 20)]);
    /* Flush D-cache just in case the BSP boot left dirty cache lines */
    Xil_DCacheFlushRange((INTPTR)Frame, BUFFERSIZE);
    xil_printf("Cache flushed. Initializing DP subsystem...\r\n");
    InitRunConfig(&RunCfg);
    if (InitDpDmaSubsystem(&RunCfg) != XST_SUCCESS) {
        xil_printf("InitDpDmaSubsystem FAIL\r\n");
        while (1) __asm__ volatile("wfe");
    }
    xil_printf("DP subsystem init OK\r\n");
    FrameBuffer.Address  = (INTPTR)Frame;
    FrameBuffer.Stride   = STRIDE;
    FrameBuffer.LineSize = LINESIZE;
    FrameBuffer.Size     = BUFFERSIZE;
    /* DP register raw debug */
    xil_printf("DP raw regs:\r\n");
    xil_printf("  LINK_BW_SET (+0x000) = 0x%08X\r\n", XDpPsu_ReadReg(0xFD4A0000, 0x000));
    xil_printf("  TRANSMITTER_EN (+0x080) = 0x%08X\r\n", XDpPsu_ReadReg(0xFD4A0000, 0x080));
    xil_printf("  INTR_SIG_STATE (+0x130) = 0x%08X (bit0=HPD)\r\n", XDpPsu_ReadReg(0xFD4A0000, 0x130));
    xil_printf("  IsConnected = %d\r\n", XDpPsu_IsConnected(RunCfg.DpPsuPtr));

    /* Wait up to 5 seconds for HPD to come up, then proceed regardless */
    xil_printf("Waiting up to 5s for HPD...\r\n");
    int hpd = 0;
    for (int i = 0; i < 50; i++) {
        u32 sig = XDpPsu_ReadReg(0xFD4A0000, 0x130);
        if (sig & 1) { hpd = 1; break; }
        usleep(100000);
    }
    if (hpd) {
        xil_printf("HPD HIGH detected\r\n");
    } else {
        xil_printf("HPD never went high after 5s. PROCEEDING ANYWAY (forced)\r\n");
    }

    xil_printf("Disabling main link before training...\r\n");
    XDpPsu_EnableMainLink(RunCfg.DpPsuPtr, 0);
    xil_printf("Wakeup monitor via DPCD aux...\r\n");
    u32 wakerc = DpPsu_Wakeup(&RunCfg);
    xil_printf("Wakeup rc = %d\r\n", wakerc);

    for (int retry = 0; retry < 5; retry++) {
        usleep(200000);
        xil_printf("[retry %d] DpPsu_HpdTrain...\r\n", retry);
        u32 trc = DpPsu_HpdTrain(&RunCfg);
        if (trc != XST_SUCCESS) { xil_printf("  train rc=%d, retry\r\n", trc); continue; }
        xil_printf("Training OK -> Display framebuffer\r\n");
        /* Explicitly enable the graphics DMA channel */
        XDpDma_SetChannelState(RunCfg.DpDmaPtr, GraphicsChan, XDPDMA_ENABLE);
        XDpDma_DisplayGfxFrameBuffer(RunCfg.DpDmaPtr, &FrameBuffer);
        DpPsu_SetupVideoStream(&RunCfg);
        /* Flush again */
        Xil_DCacheFlushRange((INTPTR)Frame, BUFFERSIZE);
        /* POLLING MODE: bypass the VSYNC-interrupt path entirely.
         * Directly call the public SetupChannel + Trigger functions
         * that the ISR would normally call. */
        xil_printf("Directly calling SetupChannel + Trigger for graphics...\r\n");
        extern void XDpDma_SetupChannel(XDpDma *InstancePtr, XDpDma_ChannelType Channel);
        XDpDma_SetupChannel(RunCfg.DpDmaPtr, GraphicsChan);
        XDpDma_SetChannelState(RunCfg.DpDmaPtr, GraphicsChan, XDPDMA_ENABLE);
        int trc2 = XDpDma_Trigger(RunCfg.DpDmaPtr, GraphicsChan);
        xil_printf("XDpDma_Trigger rc = %d\r\n", trc2);
        xil_printf("--- POST-SETUP REGISTER DUMP ---\r\n");
        xil_printf("DP LINK_BW_SET        = 0x%08X\r\n", XDpPsu_ReadReg(0xFD4A0000, 0x000));
        xil_printf("DP LANE_COUNT_SET     = 0x%08X\r\n", XDpPsu_ReadReg(0xFD4A0000, 0x004));
        xil_printf("DP MAIN_STREAM_ENABLE = 0x%08X (bit0)\r\n", XDpPsu_ReadReg(0xFD4A0000, 0x084));
        xil_printf("DP MAIN_STREAM_HTOTAL = 0x%08X\r\n", XDpPsu_ReadReg(0xFD4A0000, 0x180));
        xil_printf("DP MAIN_STREAM_VTOTAL = 0x%08X\r\n", XDpPsu_ReadReg(0xFD4A0000, 0x184));
        xil_printf("DP MAIN_STREAM_HRES   = 0x%08X\r\n", XDpPsu_ReadReg(0xFD4A0000, 0x190));
        xil_printf("DP MAIN_STREAM_VRES   = 0x%08X\r\n", XDpPsu_ReadReg(0xFD4A0000, 0x194));
        /* DPDMA CH3 (Graphics) base = 0xFD4C0500 */
        xil_printf("DPDMA CH3 DSCR_STRT_ADDR = 0x%08X\r\n", *(volatile u32*)0xFD4C0504);
        xil_printf("DPDMA CH3 DSCR_NEXT_ADDR = 0x%08X\r\n", *(volatile u32*)0xFD4C050C);
        xil_printf("DPDMA CH3 PYLD_CUR_ADDR  = 0x%08X\r\n", *(volatile u32*)0xFD4C0514);
        xil_printf("DPDMA CH3 CNTL (+0x518)  = 0x%08X (bit0=en)\r\n", *(volatile u32*)0xFD4C0518);
        xil_printf("DPDMA GLOBAL CNTL        = 0x%08X\r\n", *(volatile u32*)0xFD4C0100);
        xil_printf("DP video output running. Pattern should be visible on monitor.\r\n");
        int tickcount = 0;
        while (1) {
            XDpDma_InterruptHandler(RunCfg.DpDmaPtr);
            usleep(16666); /* ~60 Hz */
            tickcount++;
            if (tickcount % 60 == 0) {
                u32 isr = *(volatile u32*)0xFD4C0004;
                u32 pyld = *(volatile u32*)0xFD4C0514;
                xil_printf("t=%d ISR=0x%08X CH3.CNTL=0x%08X PYLD_CUR=0x%08X\r\n",
                    tickcount, isr,
                    *(volatile u32*)0xFD4C0518, pyld);
                /* Clear ISR bits that were set so we can see new events */
                *(volatile u32*)0xFD4C0004 = isr;
            }
        }
    }
    xil_printf("All retries failed. Halting.\r\n");
    while (1) __asm__ volatile("wfe");
    return 0;
}
