# FZ3A 裸机视频流 DP 显示项目 —— 存档 2026-04-09

ALINX FZ3A (Xilinx XCZU3EG ZynqMP) 裸机 bare-metal 项目，从 0 做到
**千兆 TCP → DPDMA → mini DP @ 30 fps** 实时视频流，并在 A53 CPU 侧
实现 17 个图像处理滤镜，UART 串口实时切换。

## 最终达成指标

| 项 | 值 |
|---|---|
| DP 输出分辨率 | 1280×720 @ 60 Hz RGBA8888 |
| 网络吞吐（TCP raw RGBA） | **108 MB/s ≈ 864 Mbps**（千兆效率 ≈ 91%） |
| 合成源帧率（testsrc2, Linux 侧）| **29.3 fps** |
| Windows MP4 文件流 | **26.8 fps**（IMG_2039.MOV H.264 1920×1072）|
| RTSP 中转 | 24-25 fps |
| 双缓冲无撕裂 | ✅（Trigger-per-frame mode）|
| FPGA PL 资源占用 | < 4%（DP/GEM3/DPDMA 都走 PS 硬核） |

## 项目阶段

1. **Phase 1**: 板载 JTAG 连接验证 → Vivado BSP 导出 → Vitis 裸机 DP 1280×720 彩条显示
2. **Phase 1.5**: MNIST 手写识别 PS+PL 分离（HLS 加速器）
3. **Phase 2a**: lwIP 2.2.0 + xemacps_v3_21 Ethernet bring-up，KSZ9031 PHY read-only 模式，DHCP，TCP echo server
4. **Phase 2b**: 单帧网络图像 → DP 显示（`send_image.py`）
5. **Phase 2c**: 实时视频流（`stream_test.py` / `stream_desktop.py` / `stream_rtsp.py` / `stream_video_win.py`）
6. **Phase 2d**: 双缓冲无撕裂 + UART 串口控制 + **17 种图像处理滤镜**（移植自 b3/v22v Verilog RTL 工程）

## 关键技术突破

### 1. D-cache + DPDMA 协同工作（Xilinx 官方 example 没做的事）

Xilinx 的 `xdpdma_video_example.c` 粗暴地 `Xil_DCacheDisable()`，因为 DPDMA
描述符是嵌在 `XDpDma` struct 里的，D-cache 开启时 CPU 写描述符到 cache 不
同步到 DDR，DPDMA 硬件读到陈旧数据 → 黑屏。

正确解法（phase2b_main.c 开头）：

```c
/* 1. Flush 启动残留 cache lines */
Xil_DCacheFlush();

/* 2. 把 8 MB framebuffer 区标成 NORM_NONCACHE */
for (uint64_t a = 0x10000000; a < 0x10800000; a += 0x200000)
    Xil_SetTlbAttributes(a, NORM_NONCACHE);

/* 3. 全屏障 */
__asm__ volatile("dsb sy; isb" ::: "memory");

/* 4. memset DpDma/FrameBuffers 清零（.framebuffer NOLOAD 不会自动清） */
memset(&DpDma, 0, sizeof(DpDma));
memset(&FrameBuffers, 0, sizeof(FrameBuffers));

/* 5. 继续 DP 驱动初始化 —— 描述符写 DDR 直达，lwIP/memcpy 继续享用 cache */
```

把 `XDpDma DpDma` + `XDpDma_FrameBuffer FrameBuffers[2]` + `u8 Frames[2][...]`
全部放进 `.framebuffer` section（linker 映射到 `ddr_fb @ 0x10000000`），
整个 DP 数据路径绕开 cache，lwIP/TCP/memcpy 全速享受 cache。

### 2. 双缓冲消撕裂

`Frames[2]` + `FrameBuffers[2]`：CPU 写 back 时 DPDMA 读 front。
帧写完调用 `XDpDma_DisplayGfxFrameBuffer + SetupChannel + Trigger` 切换描述符。
三调用缺一不可（Trigger 不调会撕裂，SetupChannel 不调 SRC_ADDR 不更新）。

```
front = g_front (DPDMA reading)
back  = g_back  (CPU writing via memcpy)
frame complete → Trigger swap → front := back, back := 1 - back
```

### 3. KSZ9031 PHY read-only 模式

Xilinx xemacpsif 默认不识别 Micrel OUI，写 MDIO 会打断已 Linux-init 好
的 link。解决：加 Micrel ID `0x0022`，只从 reg 0x0A bit 11 读
1000BASE-T FD 结果，**不写任何 PHY 寄存器**。

参见 `xemacpsif_physpeed.c` 的 `get_Micrel_phy_speed()`。

## 文件清单

### 板侧源码（/ 开发主机 Linux）

```
phase2b_main.c        # 主应用（DP+lwIP+TCP+双缓冲+17 滤镜+UART 控制）
stubs.c               # newlib syscalls + sys_now() for lwIP
lscript.ld            # 链接器：ddr_low 0x00100000 + ddr_fb 0x10000000/8MB
xemacpsif_physpeed.c  # 打过补丁的 PHY 检测（KSZ9031 read-only 模式）
lwipopts.h            # lwIP 调优（TCP_WND=64K, MEM_SIZE=512K, 256 desc）
build.ps1             # Windows 侧 aarch64-none-elf-gcc 编译脚本
```

### xsdb 启动/下载脚本

```
boot_phase2b.tcl      # 冷启动：rst -system + psu_init + dow elf + con
hotdow.tcl            # 热下载：targets 9 + rst -processor + dow + con
list_targets.tcl      # 列 xsdb targets
regen_bsp.tcl         # 重建 BSP
```

### 发送端脚本

```
Linux 侧:
  send_image.py          # 发单张图
  stream_test.py         # ffmpeg lavfi testsrc2/smptebars/mandelbrot
  stream_desktop.py      # x11grab 桌面流
  stream_rtsp.py         # RTSP 中转（`rtsp://192.168.6.162:8554/live`）
  stream_video.py        # 本地视频文件

Windows 侧:
  stream_video_win.py    # 视频文件流（零拷贝 bytearray 优化）
  stream_test_win.py     # testsrc2 合成
  cam_ff_to_fz3a.py      # dshow 摄像头（被 NVIDIA Broadcast 挡了，未跑通）
  cam_to_fz3a.py         # OpenCV MSMF 摄像头备用
  play_2039.bat          # 启动器（task scheduler 用）
  run_video.bat / run_cam.bat / run_mandel.bat
```

### Windows 产物

```
windows_artifacts/
  phase2b.elf            # 最终编译产物 938920 字节
  boot_phase2b.tcl       # 冷启动脚本（带 rst -system -clear-registers）
  hotdow.tcl             # 热下载脚本
  play_2039.bat          # IMG_2039.MOV loop 播放
  *.tcl                  # 各种调试脚本
```

## 滤镜清单（phase2b_main.c）

所有算法移植自 `E:\Linux\nanjing\b3\v22v\video640 19fps\UDP_EG4_6.2\source_code\rtl\`
中的 Verilog RTL 模块。按 UART 串口单字符触发：

| Key | 滤镜 | 路径 | 对应 RTL |
|---|---|---|---|
| `0`/`n` | 直通 | direct memcpy | - |
| `i` | 反色 negative | inline | - |
| `g` | 灰度（RGB→Y 76/150/29）| inline w/ R/G buffering | 通用 |
| `b` | 固定阈值二值化 | inline | - |
| `h` | 16 色热力图 | inline LUT | `threshold_region_segment.v` |
| `l` | 低光增强 gamma LUT | inline | `lowlight_enhance.v` |
| `r`/`x`/`w` | 红/绿/蓝通道 only | inline | - |
| `+`/`-` | 亮度 ±64 | inline | - |
| `y` | R↔B channel swap | inline | - |
| `u` | 蓝色区域高亮 | inline | `blue_region_highlight.v` (简化) |
| `s` | Sobel 3×3 | post-pass | `sobel_filter.v` |
| `p` | Laplacian 锐化 | post-pass strength=2 | `laplacian_sharpen.v` |
| `d` | 3×3 膨胀 | post-pass | `dilation_filter_gray.v` |
| `e` | 3×3 腐蚀 | post-pass | `erosion_filter_gray.v` |
| `o` | Otsu 自动阈值 | two-pass | `otsu_binarize.v`（真 Otsu 算法） |
| `?` | 打印帮助 | | |

**inline 滤镜**（前 12 个）：每帧 25-26 fps，几乎无性能损失
**post-pass 3×3 核**：每帧 10-12 fps（读写非缓存 framebuffer 开销）
**Otsu**：8-10 fps（两遍扫描 + 直方图）

## 运行步骤（冷启动到播放）

1. **编译**（Windows cmd/PowerShell）：
   ```cmd
   powershell -ExecutionPolicy Bypass -File C:\Users\huye\fz3a\dp\build.ps1
   ```

2. **冷启动板子**（Windows）：
   ```cmd
   C:\Xilinx\Vitis\2024.2\bin\xsdb.bat C:\Users\huye\fz3a\dp\boot_phase2b.tcl
   ```
   板子启动后自动跑到 "entering main loop"，DHCP 拿 192.168.6.192，
   TCP listen on 5000，DP 显示 waiting pattern (蓝+黄)。

3. **推视频**（两种路径）：
   - **Linux 主机**: `python3 /tmp/fz3a_dp/stream_test.py 192.168.6.192 30 testsrc2`
   - **Windows**: `python -u C:\Users\huye\fz3a\dp\stream_video_win.py "D:\dw\IMG_2039.MOV" 192.168.6.192 30 loop`

4. **切换滤镜** — 打开 COM9 @ 115200（PuTTY / Vitis Serial Monitor / MobaXterm），
   按 `?` 看菜单，按 `i`/`g`/`s` 等字符实时切换。

## 踩过的坑（按时间顺序）

1. **JTAG 连接**: hw_server tcp:localhost:3121 + `targets 9` (A53#0)
2. **BSP 重建**: Vitis 每次改 lwipopts/xlwipconfig 都要 `regen_bsp.tcl` 重编库
3. **KSZ9031 PHY**: 默认 xemacpsif 不识别，写 MDIO 会杀 link → read-only 补丁
4. **lwIP GIC**: 必须 `Xil_ExceptionInit + XScuGic_DeviceInitialize + ExceptionRegisterHandler + ExceptionEnable`，否则 ISR 不跑、BD 不 drain、`frames_rx=0`
5. **DHCP 100 次循环**: fallback 192.168.6.210，但常规拿 192.168.6.192
6. **D-cache 黑屏**: Xilinx 官方 example 直接禁 cache。正解 = 放 .framebuffer NOLOAD + NORM_NONCACHE TLB + memset + dsb/isb 屏障
7. **.framebuffer NOLOAD 不清零**: 移 DpDma 进去后必须手动 memset，否则驱动看到垃圾
8. **Windows 百兆限速**: 网卡协商到 100 Mbps 导致只有 13 MB/s，换线到 1 Gbps 后 94 MB/s
9. **Windows Python sendall 慢**: `HEADER + buf` 拼接每帧 3.5 MB 内存 alloc+copy。优化：预分配 bytearray + memoryview + `readinto`，18.9 fps → 25 fps
10. **NVIDIA Broadcast 独占摄像头**: 杀掉进程也自动重启，ffmpeg dshow 拿不到帧，最终放弃摄像头走视频文件路线
11. **双缓冲的 Trigger**: 仅调 SetupChannel 不调 Trigger → 撕裂。三调用缺一不可
12. **EDITR not ready**: recv_cb 死循环/竞态导致 A53 debug 接口卡死，只能物理 reset 板子
13. **xsdb 的 UTF-8**: PowerShell 传参多层 escape，最终用 `.bat` 包装器更稳

## 板卡 / 工具链

- **板卡**: ALINX FZ3A (XCZU3EG-SFVC784-1-I ZynqMP), KSZ9031 GigE PHY, mini DP OUT, 2 GB DDR4
- **开发主机**: Linux（我），远程 Windows（huye@192.168.6.244:2222）+ Vivado 2024.2 + Vitis 2024.2
- **JTAG**: Digilent JTAG-SMT2 板载，FT4232H → tcp:localhost:3121
- **工具链**: `C:\Xilinx\Vitis\2024.2\gnu\aarch64\nt\aarch64-none\bin\aarch64-none-elf-gcc.exe`
- **BSP 路径**: `C:\Users\huye\fz3a\vitis_ws\fz3a_plat\psu_cortexa53_0\standalone_domain\bsp\psu_cortexa53_0\{lib,include}`
- **FSBL/PMU 不需要**：直接用 xsdb `psu_init.tcl` 手动 bring-up DDR/clock/MIO

## 性能对比

| 场景 | FPS | 带宽 |
|---|---:|---:|
| Linux → testsrc2 | 29.3 | 108 MB/s |
| Linux → RTSP 中转 | 24.5 | 87 MB/s |
| Windows → testsrc2 | 26.5 | 98 MB/s |
| Windows → 本地 MP4 | 26.8 | 99 MB/s |
| 反色 inline | 25-26 | 93 MB/s |
| 灰度 inline | 24-25 | 92 MB/s |
| 热力图 inline | 24-25 | 92 MB/s |
| 低光增强 inline | 22-24 | 88 MB/s |
| 双缓冲 + Trigger-per-frame | 21-22 | 80 MB/s |
| Sobel post-pass | 10-12 | 40 MB/s |
| Laplacian post-pass | 10-12 | 40 MB/s |
| Otsu two-pass | 8-10 | 35 MB/s |

## 后续可做

- **PL 硬件加速**: 把 3×3 核放进 FPGA HLS IP，消除 CPU post-pass 瓶颈
- **DPDMA IRQ 驱动**: 真正的 VSYNC 同步 swap，彻底消撕裂 @ 30 fps
- **H.264 解码 IP**: Windows 送压缩流，板子解码，10 MB/s 即可 1080p
- **JPEG 压缩传输**: 减少 90% 带宽
- **摄像头路径**: 用非 NVIDIA Broadcast 的 Windows 或 Linux 主机抓 cam

---
归档时间: 2026-04-09 23:53 GMT+8
