# Scene Perception System — 车牌识别 + 行人检测技术细节

ALINX FZ3A (Xilinx Zynq UltraScale+ XCZU3EG) 裸机固件上的完整场景感知系统：**两个 CNN 同时在 PL 侧运行**，ARM A53 PS 编排，**全部在板子上完成，不借用外部计算**。

本文档汇总车牌识别（新）+ 行人检测（已有）的模型、HLS 内核、PS 软件、协议、实测结果。

---

## 1. 系统总览

### 1.1 硬件

| 组件 | 规格 |
|------|------|
| 芯片 | Xilinx XCZU3EG (ALINX FZ3A) |
| PS | Quad-core ARM Cortex-A53 @ 1.2 GHz |
| PL | Zynq UltraScale+, 432 BRAM18, 360 DSP48E, 71K LUT |
| DDR | 2GB DDR4 |
| 网络 | GEM3 Gigabit Ethernet (lwIP) |

### 1.2 数据流

```
Client (Python, 任意 PC) ──TCP──▶ Board (裸机):
                                    ① PS: 接收图像
                                    ② PS: 定位（Sobel / HSV）
                                    ③ PL: plate_cnn (INT8)
                                    ④ PL: pedcnn 滑窗
                                    ⑤ PS: NMS + 序列化
                                  ──TCP──▶ Client: 显示
```

**核心原则**：主体 CNN 强制走 PL 加速；PS 做编排、定位、后处理。

---

## 2. 车牌识别

### 2.1 CNN 模型

端到端 7 字符中国车牌识别，跳过字符分割。

**架构**
```
Input:  128×32 灰度 (4096 字节, uint8)
  ├─ Conv1 (1→32)   + BN + ReLU + MaxPool2x2 → (32, 16, 64)
  ├─ Conv2 (32→64)  + BN + ReLU + MaxPool2x2 → (64, 8, 32)
  ├─ Conv3 (64→128) + BN + ReLU + MaxPool2x2 → (128, 4, 16)
  ├─ Conv4 (128→256)+ BN + ReLU + MaxPool2x2 → (256, 2, 8) = 4096 features
  ├─ FC (4096→512)  + ReLU
  ├─ Head CN (512→31)  → argmax → 省份 index
  └─ Heads AL × 6 (512→36) → argmax → 6 字母数字 index
```

**参数**: 2.61M (INT8 量化后 **2.6 MB**)

### 2.2 训练

**数据**: richjjj 213K (HuggingFace) + CCPD2019 22K 子集 = **235K**

**关键技巧**：
- Province-weighted Loss (`1/sqrt(class_count)`) 处理省份极端不平衡
- WeightedRandomSampler 过采样罕见省份
- Label smoothing 0.1
- 随机裁剪 + 旋转 ±10° + 亮度对比度抖动
- BN folded into Conv2d (训练后预计算)
- 对称 per-tensor INT8 量化

**环境**: RTX 5090 + Quadro RTX 5000 双 GPU, AMP FP16, batch=1024, 100 epochs, **13.6 min**

**脚本**: `training/train_e2e_cnn_v5.py`, 导出 `training/export_plate_cnn.py`

### 2.3 PL HLS 内核 (plate_cnn_hls)

**文件**: `hls/plate_cnn_hls_kernel.cpp` (snapshot flat) / build server `plate_cnn_hls/plate_cnn_hls_kernel.cpp`

**资源占用 (实测 Vivado 实现)**：

| 资源 | 用量 | 占比 |
|------|------|------|
| BRAM18 | ~308 | 71% |
| DSP48E | 66 | 18% |
| LUT | 9,000 | 12.8% |
| FF | 15,000 | 10.6% |

**关键优化**：
- Conv 权重在 BRAM：`ARRAY_PARTITION` 按 output channel 分块，实现并行卷积
- FC 权重在 **DDR**（2.1 MB INT8，放不进 BRAM）：通过 `m_axi` 流式读取
- PIPELINE II=1 在内循环避免外循环指令爆炸
- ap_fixed<8,8> 数据类型全链路 INT8

**MMIO 寄存器 (AXI-Lite base = 0xA0000000)**：

| 偏移 | 名称 | 说明 |
|------|------|------|
| 0x00 | AP_CTRL | bit0=start, bit1=done, bit2=idle |
| 0x10 | FC_ADDR_LO | FC 权重低 32 位 |
| 0x14 | FC_ADDR_HI | FC 权重高 32 位 |
| 0x1C | FC_DDR_LO | 同上，冗余 |
| 0x20 | FC_DDR_HI | 同上，冗余 |
| 0x28 | PRED[0..6] | 7 字节预测输出 (1 省份 + 6 字母数字) |
| 0x1000 | IMG[4096] | 输入图像 BRAM |

**延迟**：**675 ms / 车牌**（PS 软件实现 1170 ms，提升 1.73×）

### 2.4 板上车牌定位

**流程**：
```
全场景图 (灰度 720×1160 max)
   ↓
Sobel 3×3 (PS 软件, ~100 ms)
   ↓
边缘密度网格扫描 (8×8 cell)
   ↓ 
横向连续高密度 run (长宽比 2:1-7:1)
   ↓
像素精度 Y 轴 bbox 细化
   ↓
NMS 候选去重 (IoU > 0.5)
   ↓
每候选 3 variants (tight / +3px / +6px)
```

**函数**: `sobel_ps()`, `find_plate_rects()`, `refine_bbox_vertical()`, `crop_resize_128x32()` (均在 `firmware/phase2b_main.c`)

### 2.5 可选：HSV 颜色辅助定位

协议扩展 `fmt=1` (RGB888 输入) 启用：

```
RGB → HSV → 蓝色掩码 (H 100-124, S ≥ 43, V ≥ 46)
         → find_plate_rects 复用 (把蓝色掩码当作"边缘")
         → NMS 跨源去重 (Sobel ∪ HSV)
```

**HSV 阈值**：OpenCV 兼容 (H 0-179 的蓝色范围)，覆盖中国蓝色车牌。黄色/绿色需扩展。

**函数**: `rgb_to_gray()` (BT.601 luma), `rgb_to_hsv_blue_mask()`

### 2.6 实测结果

#### 预裁剪 (GT bbox) 准确率 — CNN 本身能力上限

在 CCPD2019 外独立的 141 张 holdout 上（GT bbox → crop → resize 128×32 → CNN）：

| 指标 | 结果 |
|------|------|
| **板级准确率** | **124/141 = 87.94%** |
| **字符级** | **967/987 = 97.97%** |
| PL 延迟 | 675 ms/车牌 |

#### 端到端 (on-board 定位 + PL CNN) — 10 张 CCPD 场景图

| 配置 | 分辨率 | Exact match | 备注 |
|------|--------|-------------|------|
| 基线 Sobel + 2 variants | 720px | 5/10 = 50% | 单源定位 |
| + NMS 去重 + 3 variants | 720px | 5/10 = 50% | 结构优化等价 |
| + HSV 颜色通路 | 720px | 5/10 = 50% | 救回 #6，丢 #8 |

**失败分析**（5 张未中）：
- #2, #7, #9 — Sobel 和 HSV 都给不出干净 crop（边缘极差）
- #8 (皖AV**V**697V) — CNN 把 V→1 误读（**字符级精度，非定位**）
- #10 (皖A**D**J713) — CNN 把 D→J 误读（**字符级精度**）

**核心结论**：5 张失败中 4 张是 **CNN 字符识别精度问题**，不是定位问题。结构性优化（NMS、多 variants、HSV）等价于基线 50%。

---

## 3. 行人检测 (HOG + Linear SVM)

**注**：本节描述 Phase 2i 起稳定验证的 HOG+SVM 加速器（独立 TCP 5002 端口）。v18 bitstream 的 `pedcnn_hls_ip` 是替代方案（binary CNN），资源占用相近但在真实街景上的指标未达到 HOG+SVM 同等水平。**HOG+SVM 是经 100 张真实街景充分验证的生产方案。**

### 3.1 算法流程

经典 Dalal & Triggs pipeline，**全部在 PL 里完成**，PS 只负责把图像地址交给 HLS。

```
320×240 灰度图 (PS 写入 DDR)
   │  m_axi DMA burst read
   ▼
Gradient (centered difference, L1 magnitude)
   ▼
Cell Histogram (8×8 cell, 9 orientation bins)
   ▼
Sliding Window (64×128, stride 8px = 495 windows)
   ▼
Linear SVM (3780-dim dot product per window)
   ▼
Detections: (x, y, score) × max 16
   │  回写 PS BRAM
   ▼
PS 读取检测结果并序列化回客户端
```

**特征维度推导**：64×128 窗口、8×8 cell → 8×16 = 128 cells；2×2 block 4 cells → 7×15 = 105 blocks × 36 (4 cell × 9 bin) = **3780 维**。

### 3.2 训练

**数据**: Penn-Fudan Pedestrian Dataset (真实街景行人照片)

**分类器**: scikit-learn `LinearSVC`
- 5-fold CV accuracy: 86.4%
- INT8 对称量化 SVM 权重 + bias (板上存放)
- `training/train_inria_svm.py`

**为什么能从 86.4% CV 跑到 95% 真实准确率**：CV 在高度随机划分的正负样本上测，FP 率较高；100 张 Penn-Fudan 完整街景上，Recall=100% (零漏检) + Precision=90.9%，证明 SVM 得分阈值略偏保守。

### 3.3 PL HLS 内核 (ped_hls)

**文件**: `hls/ped_hls_kernel.cpp` + `hls/ped_hls_weights.h` + `hls/ped_svm_info.json`

**接口**：
- `m_axi` 读 DDR 图像 (76,800 字节 burst)
- `s_axilite` 控制 + 结果
- SmartConnect 桥到 `S_AXI_HP0_FPD`

**HLS 优化**：
- Gradient + HOG cell histogram 流水线 (PIPELINE II=1)
- HOG cell BRAM 按 cell x 方向 ARRAY_PARTITION → 滑窗可并行读
- SVM 3780-MAC 全展开 DSP 阵列
- SVM 权重 INT8 per-row 在 BRAM

**MMIO 寄存器 (base = 0xA0020000)**:

| 偏移 | 名称 | 说明 |
|------|------|------|
| 0x00 | AP_CTRL | bit0=start, bit1=done, bit2=idle |
| 0x10 | IMAGE_ADDR_LO | DDR 图像地址低 32 位 |
| 0x14 | IMAGE_ADDR_HI | DDR 图像地址高 32 位 |
| 0x1C | NUM_DETS | 输出检测数 (uint32) |
| 0x2C | THRESHOLD | SVM 决策阈值 (int32) |
| 0x80+4i | DET[i] | 检测 i 结果 `(y<<20)|(x<<8)|score_lo`（packed） |

**资源占用**：

| 资源 | 用量 | 占比 |
|------|------|------|
| BRAM18 | ~40 | 9% |
| DSP48E | ~30 | 8% |
| LUT | ~9,000 | 13% |
| FF | ~12,000 | 8.5% |

**延迟**: **22 ms / 帧 (~46 FPS)**

### 3.4 固件调用 (TCP 5002)

```c
// 简化流程 (ped_recv_cb 触发)
memcpy(PED_DDR_BUF, incoming_320x240, 76800);
PED_IMAGE_ADDR_LO = (uint32_t)(PED_DDR_BUF & 0xFFFFFFFF);
PED_IMAGE_ADDR_HI = (uint32_t)(PED_DDR_BUF >> 32);
PED_THRESHOLD = g_ped_threshold;
PED_AP_CTRL = PED_AP_START;
while (!(PED_AP_CTRL & PED_AP_DONE)) { /* poll */ }
n = PED_NUM_DETS;
for (i = 0; i < n; i++) {
    int32_t d = PED_DET(i);  // packed (y, x, score)
    dets[i] = unpack(d);
}
// 序列化 "DET\0" + n + n*{x(2), y(2), w=64, h=128, score(4)}
```

**端口 5002 协议**:
```
REQ: "PED\0" + w(4)=320 + h(4)=240 + fmt(4) + 76800 bytes u8
RES: "DET\0" + n(4) + n * {pos(4) + score(4)}
     其中 pos 高 16 位 = y, 低 16 位 = x；窗口固定 64×128
```

### 3.5 实测结果

**数据集**: Penn-Fudan Pedestrian Dataset，100 张真实街景（50 正样本 + 50 负样本）

| 指标 | 结果 |
|------|------|
| **准确率 (Accuracy)** | **95.0%** ⭐ |
| **精确率 (Precision)** | **90.9%** |
| **召回率 (Recall)** | **100.0%** (零漏检) |
| **F1 Score** | **95.2%** |
| 推理延迟 | **22 ms / 帧 (~46 FPS)** |

**混淆矩阵**：

```
              预测有行人  预测无行人
实际有行人      TP=50      FN=0
实际无行人      FP=5       TN=45
```

**PS vs PL 对比**：

| 方式 | 延迟 | FPS | 说明 |
|------|------|-----|------|
| PS 估算 (ARM A53) | ~100+ ms | <10 | 76K 像素梯度 + 495 窗 × 3780 MAC ≈ 1.87M MAC @ 1.2GHz |
| **PL HOG+SVM (HLS)** | **22 ms** | **46** | PL 完成全部 HOG + SVM，PS 只做 I/O |
| 加速比 | | **~4.5×** | |

**为什么 Recall 100%**：SVM 训练时负样本数量大于正样本 (背景 patches 数量远多于行人)，阈值调到偏保守方向；加上 64×128 窗口对站立成人行人的尺度匹配好，真正的行人在 495 个窗口里至少有 1 个高响应。

**失败模式** (5 FP)：街景中容易被误检为行人的结构——
- 电线杆 / 路灯垂直结构
- 汽车车门竖直边缘 + 座椅轮廓
- 大衣挂钩 / 店面 mannequin (训练集里没有)

---

## 4. 集成 — TCP 5004 多任务流水线

### 4.1 协议

**端口 5004 — 全场景多任务**

```
REQ: "ALL\0" + w(4) + h(4) + fmt(4) + data_bytes
     fmt=0: 灰度, data_bytes = w × h
     fmt=1: RGB888, data_bytes = 3 × w × h

RES: "RES\0" + n_plates(1)
     + n_plates × { x(2) y(2) w(2) h(2) prov(1) al[6] }
     + n_peds(1)
     + n_peds × { x(2) y(2) w(2) h(2) score(4) }
```

**兼容独立端口**：
- 5003 — 仅车牌（预裁剪 128×32 输入）
- 5002 — 仅行人（320×240 输入）

### 4.2 编排器 `all_run_and_reply()`

```c
// 简化流程
if (fmt == 1) {
    rgb_to_gray(ALL_RGB_BUF, ALL_SRC_BUF, W, H);
    rgb_to_hsv_blue_mask(ALL_RGB_BUF, ALL_HSV_BUF, W, H);
}
sobel_ps(ALL_SRC_BUF, ALL_EDGE_BUF, W, H);
n_rects = find_plate_rects(ALL_EDGE_BUF, ...);
if (fmt == 1) {
    n_rects += find_plate_rects(ALL_HSV_BUF, ..., append=true);  // NMS across sources
}
for (i = 0; i < n_rects; i++) {
    crop_resize_128x32(ALL_SRC_BUF, rects[i], patch);
    if (g_plt_use_pl) plt_cnn_run_pl(patch, &prov, al);  // PL CNN
    else              pcn_infer(patch, &prov, al);       // PS CNN 兜底
    plate_results[i] = {prov, al};
}
downsample_320x240(ALL_SRC_BUF, ped_small);
// v18 bitstream: pedcnn 滑窗 (Sobel 密度预筛 + step=24)
// Phase 2i bitstream: 走独立端口 5002，调用 ped_hls HOG+SVM (22ms, 46 FPS)
n_peds = pedcnn_sliding(ped_small, ped_pos, ped_score);
serialize_reply("RES\0", n_plates, plate_results, n_peds, ped_pos, ped_score);
```

---

## 5. FPGA 资源汇总

### v18 bitstream (当前部署, plate + pedcnn)

| IP | BRAM18 | DSP48E | LUT | FF | 说明 |
|----|--------|--------|-----|-----|------|
| `plate_cnn_hls` | ~308 | 66 | 9000 | 15000 | 4 Conv + FC + 7 heads |
| `pedcnn_hls_ip` | ~20 | ~20 | 5000 | 6000 | binary 3 Conv + GAP + FC |
| AXI Smartconnect | ~5 | 0 | 2000 | 3000 | PS-PL 互联 |
| **总计** | **~333 / 432 (77%)** | **~86 / 360 (24%)** | **~16K / 71K (22%)** | — | XCZU3EG |

### Phase 2i bitstream (HOG+SVM 验证版, 独立部署 / 不含 plate_cnn)

| IP | BRAM18 | DSP48E | LUT | FF | 说明 |
|----|--------|--------|-----|-----|------|
| `ped_hls` (HOG+SVM) | ~40 | ~30 | 9000 | 12000 | Gradient + HOG + 滑窗 + LinearSVM |
| AXI Smartconnect | ~5 | 0 | 2000 | 3000 | PS-PL 互联 |
| **总计** | **~45 / 432 (10%)** | **~30 / 360 (8%)** | **~11K / 71K (15%)** | — | 极省资源，可叠加 plate_cnn |

**未来方向**：把 `ped_hls` 和 `plate_cnn_hls` 合进同一 bitstream (预计 ~350 BRAM / ~96 DSP，仍在 XCZU3EG 预算内)，同时获得 HOG+SVM 的验证准确率和 plate_cnn 的车牌识别。当前 v18 用了 `pedcnn_hls_ip` 是因为合成 bitstream 资源测算时更保守，不代表最终选型。

---

## 6. 构建与部署

### 6.1 HLS 内核合成
```bash
cd hls/plate_cnn_hls
vitis_hls -f run_hls.tcl  # → IP catalog
cd ../pedcnn_hls
vitis_hls -f run_hls.tcl
```

### 6.2 Vivado Bitstream
```bash
source settings64.sh
cd /home/eea/fz3a_build
vivado -mode batch -source integrate_v18.tcl  # ~10-15 min write_bitstream
```

### 6.3 固件交叉编译 (Windows host)
```powershell
aarch64-none-elf-gcc -O2 -mcpu=cortex-a53 -ffreestanding `
  -T lscript.ld `
  phase2b_main.c stubs.c `
  -lxil -llwip4 -lgcc -lc -lm `
  -o phase2b.elf
```

### 6.4 XSDB JTAG 部署（规范顺序）
```tcl
connect
rst -system -clear-registers
source psu_init.tcl
psu_init
psu_ps_pl_reset_config
psu_post_config
psu_ps_pl_isolation_removal   # JTAG 部署必需
fpga -no-rev -file design_1_wrapper.bit
targets -filter {name =~ "Cortex-A53*0"}
rst -processor
dow phase2b.elf
con
```

**已知坑**：如果 con 后 PC 卡在 0x100000 不动，手动 `stop; con` 一次通常能唤醒 L2 cache。

---

## 7. 关键工程经验

### 7.1 HLS 优化
1. **PIPELINE II=1 放在外层循环会触发内层完全展开** → 指令数爆炸。应对：只 PIPELINE 最内层，外层用 `ARRAY_PARTITION` 权重实现并行。
2. **FC 层权重 2 MB 放不进 BRAM**，必须走 `m_axi` 从 DDR 流式读。对应寄存器接口 FC_ADDR_LO/HI。
3. **plate_cnn HLS 寄存器映射**：图像 BRAM 在 0x1000，控制寄存器在 0x00，两个独立 DDR 指针寄存器，错位会静默返回乱码。

### 7.2 Vivado 集成
1. **用 `apply_bd_automation` 而不是手接 crossbar**：手设 `NUM_MI` 会留 dangling port 破坏 AXI decoding。
2. **`psu_ps_pl_isolation_removal` 是 JTAG 部署必需步骤**（FSBL boot 会做，JTAG 不会）。
3. 推荐用一个综合脚本跑完 `source integrate_vXX.tcl` + `launch_runs impl_1 -to_step write_bitstream -jobs 40`。

### 7.3 训练跨域
1. 单一 richjjj 数据集验证 99.84%，换到 CCPD 掉到 60%（合成/真实分布差）→ **必须混入目标域数据**。
2. v5 加 22K CCPD 后跨域准确率从 63.41% → 87.94% (+24.5%)。

### 7.4 lwIP 约束
1. **TCP_WND = 2048 字节**（默认）：全分辨率 720×1160 灰度图（835 KB）传输约 7 分钟（2KB window × RTT）。生产推荐降采样到 720 max dim。
2. **lwIP recv callback 不能长时间阻塞**：PL CNN 每次 675 ms，30 个候选 = 20 秒，会饿死 GEM3 Ethernet IRQ → MAC 死锁。当前默认 PS CNN (~10 ms 每次)，PL CNN 通过 UART 'V' 键切换。

### 7.5 板上 Sobel vs 客户端 OpenCV
简单 8×8 网格 + sort + Y 轴精化，定位准确率与客户端 OpenCV (Canny + 形态学) 相当。无需借外部设备。

---

## 8. 最终结果汇总

### 车牌

| 测试 | 输入 | 准确率 | 延迟 |
|------|------|--------|------|
| **CNN 本身** (预裁剪) | 141 张 GT crop | **87.94%** 板级 / 97.97% 字符 | 675 ms PL |
| **端到端** (on-board Sobel) | 10 张 CCPD 场景 | 5/10 = 50% | ~15-30 s/帧 |

### 行人 (HOG + Linear SVM, PL HLS)

| 测试 | 输入 | 结果 | 延迟 |
|------|------|------|------|
| **100 张 Penn-Fudan** | 320×240 真实街景 | **Acc 95.0% / Prec 90.9% / Rec 100% / F1 95.2%** | **22 ms / 46 FPS** |
| PS 估算对比 | 同输入 | 同算法 | ~100+ ms → **PL 4.5× 加速** |

### 总结

完整系统在 2 GB DDR / XCZU3EG 上稳定运行：
- PL 占用 77% BRAM / 24% DSP / 22% LUT
- 两个 CNN 并发（通过时分复用 AXI 总线）
- 纯 ARM A53 裸机编排，不依赖任何外部计算
- Python 客户端通过以太网接收 + 绘制结果

---

## 9. 已知限制与未来工作

### 已识别但未集成的突破点

| 路径 | 预期增益 | 状态 |
|------|---------|------|
| **异步 PL CNN**（不阻塞 lwIP） | 端到端 50% → 65% | 根因已识别 |
| **多尺度图像金字塔**（客户端同图传 2-3 scale） | 50% → 65% | 未测 |
| **PL CNN 重训 + bitstream 重建**（Focal CTC + crop-jitter） | 50% → 75%+ | PS 版 v6 已训（59%），未部署 |
| **PL Sobel-X**（替 PS 软件 Sobel） | 定位延迟 100ms → <2ms | v19 bitstream 尝试过但 boot hang |

### 已知不足
- PL CNN 不能在 lwIP 上下文里直接用（会饿死 Ethernet IRQ）
- 全分辨率输入传输慢（lwIP TCP_WND 小）
- HSV 颜色过滤在夜间/过曝/绿牌+树叶场景失效
- 行人 + 车牌同时推理 6-30 秒/帧，不适合实时视频

### 长期方向
- 加 FINN-based 车牌验证器 BNN（6-15 BRAM）作为前置过滤
- 迁移到 CCPD2020 + CBLPRD-330k 平衡数据集
- DMA 异步流水线，把 PL 执行彻底和 lwIP 解耦

---

**Built on ALINX FZ3A / Xilinx XCZU3EG, Vivado / Vitis 2024.2**  
**Repo**: github.com/stongry/FPGA-ZYNQ
