# add_rms_norm_bias 融合算子 Ascend 性能优化报告

## 1. 背景

RMSNorm 是 Transformer 每层 decoder 的必经算子；带残差连接的推理引擎里，归一化前还要先把 attention/FFN 的输出与 residual 相加。把「残差相加 → RMSNorm → bias」融为一个 kernel，可以把中间 x 的往返读写省掉，并将三次 launch 合为一次。vLLM Ascend 插件据此实现了自研算子 **`add_rms_norm_bias`**：一次调用产出三份输出（归一化结果 y、统计量 rstd、相加后的残差 x），由 `AscendRMSNorm.forward_oot` 在每层 decoder 调用，是比同族算子 rms_norm_cast 更热的生产路径。

本文档总结该算子在 Ascend 910B3 上的性能优化过程。优化在姊妹战役（rms_norm_cast，另有报告）方法论的基础上展开——事件四件套、行间双缓冲流水、Gather+stride-0 广播等范式被直接复用——并新增了三条本算子特有的经验：**先实证 tiling 变体映射再动手**、**以未动路径作噪声金丝雀**、**严格断言是性能优化的前提**。同时完整记录了一轮负结果（reduce 依赖链重构被设备实验证伪）与两条 c220 硬件实证事实。

### 测试环境

| 项目 | 值 |
|------|-----|
| 设备 | Ascend 910B3（dav_c220 向量核），40 AIV，UB 192KB/核 |
| CANN | 9.1.0 |
| 模型场景 | DeepSeek 类 hidden=7168，bf16 主 / fp16 对照 |
| 测试矩阵 | tokens = 1/4/16/64/128/512/1024/2048/4096 × bf16/fp16 |
| 计时口径 | NPUGraph capture+replay（≈背靠背 serving，5 样本取中位）；msprof op 做管线归因 |

### 算子功能

```text
x     = bf16(fp32(x1) + fp32(x2))                 # 残差相加（c220 无 bf16 元级加法，须在 fp32 域）
rstd  = 1/sqrt(mean(x²) + eps)                    # fp32
y     = bf16( fp32(bf16(x_norm)) × fp32(γ) + fp32(β) )   # 双舍入契约
返回 (y, rstd, x) 三份输出
```

### 结论摘要

| 项目 | 结论 |
|---|---|
| 生产主路径 | bf16、hidden=7168、rows≥41 → tiling key 30（NORMAL 变体）；decode（rows≤40）→ SINGLE_N |
| 基线瓶颈 | 发射线程停等（scalar_wait 占墙钟 85%）+ 行间零流水 + 15 遍全宽 pass 中 3 遍冗余 |
| 核心方案 | 消 V_S/S_V 往返（rstd 全程 V 域）、双缓冲行间流水、γ/β cast 上提每核 |
| 最终收益 | bf16 @2048 -24.8%（119.4→89.8µs）、@4096 -26.8%；fp16 @2048 -4.1% |
| 最终状态 | 有效带宽约 1.31TB/s（名义峰值 ~1.6TB/s 的 82%），接近带宽/向量流水转折区 |
| 正确性 | 原 126 用例 + 新增严格 150 用例全绿（y=3ulp / rstd=1e-3 / x 位级） |
| 已回退方案 | Reduce 多链交错重构：收益噪声级且存在 dst==src1 交错 fault 风险（见 §4.6） |

---

## 2. 算子在大模型中的位置与数据

### 2.1 位置

```text
token embeddings
  → N × decoder layer（每层两次归一化，带残差时都走本算子）
       attention/FFN 输出 x1 + residual x2 → RMSNorm → bias → y
  → logits

调用方：AscendRMSNorm.forward_oot；prefill（大 tokens）与 decode（1~4 tokens）都命中
```

### 2.2 变体分发——本算子的第一课：先确认优化对象

算子按 `(dtype_key×10 + mode_key)` 分发 5 个变体（NORMAL/SPLIT_D/MERGE_N/SINGLE_N/MULTI_N）。任务简报认定「hidden=7168 走 SPLIT_D」，但静态读 tiling 推出的是另一回事，最终以设备日志实证（`ASCEND_GLOBAL_LOG_LEVEL=1` 采集 tiling 输出）：

| shape (hidden=7168) | tiling key | 变体 | 说明 |
|---|---|---|---|
| bf16, rows ≤ 40 | 33 | SINGLE_N | decode 档 |
| **bf16, rows ≥ 41** | **30** | **NORMAL** | **生产 prefill 主路径** |
| fp16, rows ≤ 40 | 13 | SINGLE_N | |
| fp16, rows ≥ 41 | 14 | MULTI_N | fp16 prefill（双缓冲、仓内"优等生"） |
| col > 11264 | *1 | SPLIT_D | **生产无此 shape**（简报误判即源于此） |

SPLIT_D 的触发条件是 `numCol > ubFactor`（有 β 时 bf16 上限 11264 列），7168 远够不着。**如果按简报直接优化 SPLIT_D，全部工作量会落在零收益的路径上。**教训：分发条件要用实测 tiling 日志钉死，不要信文档/口口相传。

### 2.3 输入输出与片上数据流

| 张量 | 形状 | 类型 | 说明 |
|---|---|---|---|
| x1 / x2 | `[tokens, 7168]` | bf16 | 输入与残差 |
| γ / β | `[7168]` | bf16 | 每核只载一次 |
| y / x | `[tokens, 7168]` | bf16 | 输出 |
| rstd | `[tokens, 1]` | fp32 | 统计量输出 |

原始实现（NORMAL 变体）**每行全串行**：单缓冲队列使下一行的 MTE2 装载必须等上一行的 V 消费完；每行末尾一次 `V_S→GetValue→S_V` 标量往返把发射线程钉在行尾。bf16 的 y 链在 fp32 域做数学（c220 无 bf16 元级乘加），每行 **15 遍全宽 pass**——其中 3 遍冗余（γ/β 的 cast 每行重做、1/N 在 reduce 前全宽乘）。

优化后的片上数据流（R2 流水化后）：

```text
每行 i（槽位 s = i&1）：
  MTE2: x1[i+1]/x2[i+1] 预取进另一槽     ← 与本行 V 链重叠
  V:    cast x1/x2 → add → rint = x_out（落 x1[s]）
        mul 平方 → reduce → 标量链(1 elem) → rstd 留在 V 域
        Gather 广播 rstd → mul → rint = y-mid → widen → ×γ → +β → rint = y（落 x2[s]）
  MTE3: x_out 与 y 落盘                  ← 与下一行 V 链重叠
  rstd: 1 elem Adds 进 8-lane 累积块 → 块末 Gather 压缩 → 一次 DataCopyPad 落 GM
```

零 `V_S/S_V` 往返；发射线程（scalar）只负责发指令，永不等待。

---

## 3. 优化总览

| 轮次 | 手段 | 针对的问题 | 参考来源 |
|---|---|---|---|
| 第一轮·减法 | 消 V_S/S_V 标量往返；γ/β cast 上提每核；1/N 折叠进标量；tiling ubFactor 11264→7168 | 15 遍全宽 pass 中 3 遍冗余 + 发射线程停等 | 代码走读 + msprof |
| 第二轮·流水 | x[2]/x2[2] 双缓冲（局部行号奇偶）、事件四件套三方向单在途、rstd 全程 V 域 | 行间零流水（单缓冲串行）、scalar_wait 85% | rms_norm_cast R2 模板 + msprof |
| 第三轮·广度 | NORMAL 序言重叠；MULTI_N 折叠 1/N；SINGLE_N 消往返 | 64-128 档序言占比、fp16/decode 路径 | 前两轮方法平移 |
| 检视修复 | 外部检视 7 条全修（UB 成本模型、rstd 压缩、溢出域、host 校验）+ 新增严格测试 | 正确性（宽松断言兜不住的行混叠/越界） | 外部检视 + 设备复现 |
| 负结果 | Reduce 依赖链重构（8 链交错） | 假设的 reduce 延迟链 | 设备实验证伪 |

**经验法则**：先实证 tiling 命中的变体（优化错变体 = 零收益）→ 消标量往返与冗余 pass（等价变换，风险最低）→ 行间流水（收益最大，风险最高，用已验证模板）→ 每步以「未动路径」作噪声金丝雀校准收益真伪 → HBM 带宽墙校准终局预期。

**代码定位**（仓库 vllm-ascend，分支 add-rms-norm-bias-perf）：

| 轮次 | 文件 / 函数 | commit |
|---|---|---|
| 第一轮 | `csrc/moe/add_rms_norm_bias/op_kernel/add_rms_norm_bias.h`：`MulByRstd`、三 dtype `Compute`；`op_host/add_rms_norm_bias_tiling.cpp`：`DetermineModeParameters` | 02392101f |
| 第二轮 | 同 kernel 文件：`ProcessPipelined` / `ProcessPipelinedRow` | 7484fc0f1 |
| 第三轮 | 同文件序言重排；`add_rms_norm_bias_multi_n.h` `ComputeRstd`；`add_rms_norm_bias_single_n.h` `ProcessFp16/Fp32/Bf16` | 8e50acd56 |
| 检视修复 | tiling 成本模型与校验；三 kernel 文件；`tests/e2e/.../test_add_rms_norm_bias_strict.py`（新增） | d7132b0d5 |

---

## 4. 逐轮优化过程

### 4.1 初始版本：每行全串行，双瓶颈叠加

**瓶颈判定**（msprof op，2048×7168 bf16，40 核均值，墙钟 124.4µs）：

| 指标 | 值 | 占墙钟 | 判读 |
|---|---:|---:|---|
| vec busy | 87.9µs | 73% | 最大占用者但未饱和（1.69µs/行 ÷ 15 pass ≈ 0.11µs/pass） |
| mte2 / mte3 busy | 27.0 / 18.7µs | 22% / 15% | 未与 V 充分重叠 |
| **scalar_wait** | **106.0µs** | **85%** | 发射线程几乎全程停等 |

两个信号拼出双瓶颈结构：V 是最大占用者但空转 27%；scalar_wait 85% 说明发射线程被每行的标量往返与单缓冲队列的串行点钉死——它停着，装载/落盘的重叠就无从谈起。注意口径：**各 pipe 的 busy/wait 统计在时间上可能重叠，不是互斥占比，不能相加得到 100%**。

**代码走读量化清单**（bf16 每行，按类别计数）：

| 类别 | 基线 | 第一轮后 | 变化 | 明细 |
|---|---:|---:|---|---|
| 全宽向量 pass（≥64 lane 宽度） | 15 | 12 | **-3** | cast x1、cast x2、add、rint=x 输出、mul 平方、**muls ×1/N（折叠）**、reduce 树、muls ×rstd、rint y-mid、重新 widen、**cast γ（上提）**、mul γ、**cast β（上提）**、add β、rint y 输出 |
| 单元素向量/标量操作 | ~6 | ~6 | 不变 | Adds(eps)/Sqrt/Duplicate(ONE)/Div + rstd 累积 |
| V/S 跨流水同步 | 2 次/行 | 0 | **-2 次/行** | V_S→GetValue→S_V→SetValue 整链消除（rstd 留在 V 域） |
| GM 读写量/行 | 5 份 | 5 份 | 不变 | 读 x1/x2，写 x/y（+rstd 每 64 行 1 次） |

（表中编号即上表行序；"15 遍全宽"只计全宽向量遍历，不含单元素操作与同步步骤。）

### 4.2 第一轮：消标量往返与冗余 pass

#### 问题分析

三处冗余 + 一处停等：γ/β 每核只载一次却每行 cast；1/N 在 reduce 前全宽乘；rstd 经 V_S/GetValue/S_V/SetValue 绕行标量域。每行 15 遍全宽 → 12 遍，且消掉每行 2 次跨 pipe 同步。

#### 关键变化

```cpp
// ① rstd 广播留在向量域（位级一致）：Gather 复制 sqx[0] 成 8-lane 块，
//    stride-0 块广播 Mul 替代标量操作数的 Muls——rstd 不再走 S pipe
__aicore__ inline void MulByRstd(dst, src, sqx, count) {
    Gather(rstd8, sqx, zeroOff8, 0, 8);
    PipeBarrier<PIPE_V>();
    Mul(dst, src, rstd8, 64, repeatTimes, {1, 1, 0, 8, 8, 0});   // src1RepStride=0
    ...   // 尾块同参
}

// ② γ/β 每核 widen 一次进常驻缓冲（+57.4KB，靠 tiling 收紧腾出）
// ③ 1/N 折叠：reduce 前全宽 Muls → reduce 后 1 元素 Muls
// ④ tiling：NORMAL 模式 ubFactor 11264 → numColAlign（原常量是 16B/列×11264
//    =180KB 的全宽安全设计；收紧后缓冲随 numCol 比例，为 cast 常驻腾 76KB）
```

#### 结果

NPUGraph：@2048 119.4→109.3µs（**-8.4%**），@512-4096 -7~8%；msprof wall 124.4→114.7µs、vec busy 87.9→77.8µs，mte2/mte3 不变——「纯减 vec 工作」的定位被数据确认。未动代码的 fp16 路径同轮波动 ±1µs → 该档噪声带，**收益判定以 ≥512 档为准**。

![第一轮各档位提升率](r1_improvement.svg)

### 4.3 第二轮：行间流水（收益最大的一轮）

#### 问题分析

R1 后 vec busy 仍只占 70%、scalar_wait 84%：单缓冲队列使行 i+1 的装载等行 i 的 V 消费，发射线程被逐行钉死。**消除标量往返只是松绑，行间流水才是把 vec 喂饱的手段。**

#### 关键变化

按 rms_norm_cast R2 已验证模板重写 bf16 NORMAL（该模板自带三条避坑结论：局部行号奇偶索引、AllocEventID 四件套、退出零悬挂标志）：

```text
槽位：x1[2]/x2[2]（装载 + 兼作 x/y 输出），fp32 工作区单缓冲（仅 V 触碰，
      V 有序天然安全）；γ/β bf16 暂存借道 row-0 槽位，一次 V_MTE2 守卫释放
事件：MTE2_V（装载→V）、V_MTE2（槽位释放→装载）、MTE3_MTE2（落盘→装载）
      各单在途 + 每行一对 V_MTE3；wait 全落在消费 pipe（发射线程零阻塞）；
      发射条件与消费条件严格一致（i+2 < rowWork），退出零悬挂
rstd：全程 V 域——reduce 后 1 elem Adds 进 8-lane/行累积块（32B 对齐向量写），
      每 rowFactor 行 Gather 压缩 + 单次 DataCopyPad 落 GM
```

#### 结果

NPUGraph：@2048 109.3→**86.7µs**（**-20.7%**），@4096 -21.9%，@512 -7.8%，≤128 档持平（2-4 行/核流水不回本，属预期）。msprof：wall **86.4µs**（-30.5% vs 基线），vec busy 89.5%，**scalar_wait 106→1.86µs（85%→2%）**——发射线程完全解放，三 pipe 忙碌和/墙钟 = 1.32。

指令级流水图（本机 msprof `TimelineDetail`，前后各一份存 traces/）：

| 指标 | 流水前 | 流水后 |
|---|---:|---:|
| wall | 101.4µs | **80.7µs** |
| MTE2 忙碌 | 50.8% | **94.2%** |
| SCALAR 忙碌 | 90.0% | **2.5%** |
| 三线并行度 | 2.27 | **2.87** |

![行间流水示意](pipeline_timeline.svg)

精度：126/126 一次通过（零挂死零 NaN）——模板纪律（对齐已验证实现、消费条件守卫发射、奇偶槽位）把 rms_norm_cast 首次尝试踩过的三个坑全部绕开。

### 4.4 第三轮：广度轮——序言重叠与旁支变体

三处独立小改动，一轮闭环：① NORMAL 序言里 γ/β 的 MTE2 装载提前到 iota 预计算之前（64 次 Duplicate ≈1.6µs 发射与装载重叠），64-128 档受益；② MULTI_N 折叠 1/N（fp16+bf16 共用 ComputeRstd）；③ SINGLE_N 三 dtype 消 V_S/S_V（Gather 广播，复用已死的 reduce 工作区尾块）+ 折叠。

结果：fp16 @512-2048 **-4.2~-5.3%**（MULTI_N 主收益）；bf16 decode 1-4 tokens **-2.6~-3.9%**（SINGLE_N）；bf16 大档持平（序言已摊薄）。各档收益方向与机理一一对应——「改动在哪，收益就在哪」本身就是回归验证。

![第三轮各档位提升率](r3_improvement.svg)

### 4.5 检视修复轮：宽松断言兜不住的错误

外部静态检视 7 条，全部采纳，其中三条是**真缺陷**且两条为本轮优化引入：

| # | 问题 | 处置与教训 |
|---|---|---|
| 1 | rstd 压缩 Gather 索引按 `[b×32]×8` 生成——偏移张量**按字节逐元素**消费，n>8 时行值重复 8 份 | iota 改连续 `[0,32,...]`，**标量 SetValue** 写入（逐元素 V-store 4B 非对齐当场 fault——修复引入又当场修复）；126 用例竟全绿，因为原断言 rtol/atol=100 |
| 2 | bf16 NORMAL 缓冲随 numCol 比例增长，hidden=8192 时 UB 197KB > 192KB（原 11264 常量实为 16B/列全宽安全设计） | tiling 按 dtype/β 的**每列字节成本模型**驱动，超限回退 SPLIT_D；回退保持 colTileNum≥2（单 chunk 是 SPLIT_D 从未验证过的路径，设备实测每核一整行垃圾） |
| 3 | 1/N 后移扩大溢出域：sum(x²) 在 x≳3e17 时 Inf，golden 恒有限 | fp32/bf16 **回退折叠**（@2048 +4.5% 为正确性代价）；fp16 保留（max raw sum 5.3e13 ≪ fp32 max，有界证明入注释） |
| 4-5,7 | beta 无 shape/dtype 校验；epsilon 放过 NaN（NaN<0=false，且**调用点丢弃返回值**）；numel 无 32 位防护 | host 侧补齐；epsilon 修复要点是让返回值真正被检查 |
| 6 | 断言 rtol/atol=100 形同虚设 | **新增独立严格测试**（不动原文件）：y=3 ulp（双舍入链放大 rstd 求和顺序差异，实测最大 2 ulp）、rstd=1e-3（行间独立随机数据，行混叠必挂）、x 位级相等；补 UB 边界 col=8192、尾块 rows=313/2064/2600、无 β、epsilon/beta 拒绝路径 |

验证：原 126 + 严格 150 全绿。代价：@2048 89.8µs（85.9 → +4.5%）。**"126/126 通过"在宽松断言下不是精度证据**——严格断言先行，性能优化才有裁判。

**优化后 NORMAL 路径 UB 预算表**（bf16，hidden=7168；单位统一 KiB=1024B，可用 UB = 192KiB − 1KiB 系统保留 = 191KiB = 195,584B）：

| 缓冲 | 每列成本 | hidden=7168 占用 |
|---|---:|---:|
| x1[2] / x2[2] 双缓冲（bf16，各 2×7168×2B） | 8 B | 28 KiB ×2 |
| xFp32Buf + sqxBuf（fp32 工作区） | 8 B | 28 KiB ×2 |
| γ/β fp32 常驻 | 8 B | 28 KiB ×2 |
| rstdAcc（8-lane/行 × rowFactor=64） | — | 2 KiB |
| iota + compBuf + reduceFp32 | — | 0.75 KiB |
| one / zeroOff / rstdBcast | — | 0.09 KiB |
| **合计** | **24 B/列** | **170.8 KiB**（余量 20.2 KiB） |

hidden=8192 推导：24 × 8192 + 3200 = 199,808B > 195,584B → 超限。固定项 3200B 的构成：
实际缓冲 2912B（2KiB + 0.75KiB + 96B）+ 288B 对齐/备用预留。回退阈值公式（与代码
`DetermineModeParameters` 一致）：`numColAlign ≤ (195584 − 3200) / 24 = 8016 列`
——对齐到 16 的倍数后 7168 走 NORMAL、8192 回退 SPLIT_D（7168 的编码常量余量
19.9 KiB）。host 侧以同式实装为成本模型，kernel 不再自行假设预算。

**严格正确性测试矩阵**（test_add_rms_norm_bias_strict.py，golden 为 numpy/torch fp32 参考实现，随机输入 uniform[1,10]、seed=45）：

| 验证维度 | 用例 | 结果 |
|---|---|---|
| 数值契约（原 126 矩阵，宽松断言仅作回归） | 7 rows × 6 cols × 3 dtype | 126/126 |
| y 容差 = 3 输出 ulp（双舍入链放大 rstd 求和顺序差异；实测最大 2 fp16 ulp / 2 bf16 ulp） | 覆盖全部严格用例 | 通过 |
| rstd 行混叠——随机行（rtol/atol=1e-3） | 全部多行/核 shape | 通过 |
| rstd 行混叠——**构造行**（幅度按 1/100/0.01 循环，相邻行 rstd 差 ~100×，任何行混叠必以数个量级失败；随机行不能保证这一点） | rows=2600/128 × bf16 | 通过 |
| **极值溢出域**（x≈3e17，numCol=7168：逐元素 ×1/N 有限而 raw sum 会 Inf——正是折叠回退的边界；断言 rstd 有限且匹配 golden） | bf16 / fp32 | 通过 |
| x 位级一致（双方同为一次 fp32 加 + 一次舍入） | 同上 | 通过 |
| 压缩尾块长度（rows=313/2064/2600 → n=8/52/64+1） | bf16/fp16/fp32 | 通过 |
| UB 边界（col=8192 → SPLIT_D 回退、colTileNum≥2） | 3 dtype | 通过 |
| 无 β 路径（nullptrBeta kernel 分支与成本模型） | 4 组 | 通过 |
| epsilon NaN / +Inf 拒绝 | 2 用例 | 通过 |
| beta 短 shape 拒绝（越界读防护） | 1 用例 | 通过 |
| 未覆盖（留档）：全零/Inf 输入、输入输出别名/原地约束 | — | — |

合计 **154/154**（150 + 极值 2 + 行对比 2）。

![检视修复的正确性代价与累计收益](r4_correctness_cost.svg)

### 4.6 负结果：Reduce 依赖链重构被设备证伪

按固定模板记录（假设 → 实验变量 → 控制变量 → 设备观测 → 结果 → 结论 → 回退）：

1. **假设**：legacy 归约的单指令多 repeat `Add(..., dstRepStride=0)` 在同一累加块上形成 112 深 RAW 链，顺序 V 管线应完整暴露该延迟（估 0.3-0.6µs/行）；8 链交错发射可消除。
2. **实验变量**：新增 `ReduceSumFP32Parallel`（8×64-float 累加器、交错发射、显式两两折叠 + WRS），仅切换流水化 bf16 路径的 reduce 调用，其余不动；过程中按二分法变换发射顺序与操作数位次（V1/V2a/V2b，见下表）。
3. **控制变量**：数据、shape、设备、测量口径全部不变；每轮 clean build + md5 自检。
4. **设备观测**：

| 实验 | 发射顺序 / 操作数位次 | 观测 |
|---|---|---|
| V1 | 交错发射，累加器在 src1（dst==src1） | **vector core exception**（col=7168 无尾也崩 → 与尾路径无关） |
| V2a | 链主体顺序发射（其余同 V1） | 通过 → 故障定位于交错本身 |
| V2b | 交错发射，累加器在 **src0**（dst==src0） | 通过 → 可用形式 |
| V2b 最终版 | 150 严格 + 126 全绿 | @2048 89.09 vs 89.81µs（**-0.8%，噪声级**）；vec busy 77.4µs = 加回 avgFactor pass 后的预期值，分毫未降 |

5. **性能与正确性结果**：正确性零回退；性能收益 -0.8% 为噪声级——**当前配置下未观测到可测的依赖延迟收益**。观测与解释分开记录：设备观测是"交错 8 链与单指令 112-repeat 链等速"；对此的候选解释（未做微架构级验证）包括硬件对 repeat 间累加内部转发/按吞吐执行；静态分析发现的 0.44µs/行"未解释缺口"同样存在多个未验证候选（cast 半吞吐、barrier 开销等），不归因于单一因素。
6. **结论**（按强度分级）：**实测**——交错 8 链重构在当前配置下收益为噪声级；**确定性事实**——交错发射的单 repeat `Add`，dst 轮转且 dst==src1 时触发 vector core exception，累加器放 src0 位（dst==src0）则安全（顺序发射时 dst==src1 无恙）；**待验证假设**——多 repeat 累加的依赖延迟是否在任何配置下可观测、V 时间缺口的构成。
7. **回退及原因**：整体回退——收益噪声级、112 条标量发射/行 vs legacy 1 条（issue 侧反而更重）、+1.75KB UB；负结果与机理留档，避免未来重复推导。

### 4.7 总收益与最终状态

**累计优化（NPUGraph，hidden=7168，µs）**：

| 版本 | @512 | @1024 | @2048 | @4096 |
|---|---:|---:|---:|---:|
| 基线 | 33.2 | 59.0 | 119.4 | 250.9 |
| + 第一轮（消往返/冗余 pass） | 30.8 | 54.2 | 109.3 | 231.1 |
| + 第二轮（行间流水） | 28.4 | 47.9 | 86.7 | 180.6 |
| + 第三轮（广度） | 27.7 | 47.1 | 85.9 | 180.3 |
| + 检视修复（溢出域回退） | **29.4** | **49.5** | **89.8** | **183.7** |
| **累计** | **-11.4%** | **-16.2%** | **-24.8%** | **-26.8%** |

![各版本时延演进](latency_evolution.svg)

fp16（MULTI_N 折叠保留）：@2048 63.0→60.4µs（-4.1%）。vs unfused ref：bf16 @2048 快 29%（基线仅快 6.5%）。

**最终状态**（msprof，2048×7168 bf16，40 核均值，墙钟 89.4µs）：vec busy 77.4µs（占 aiv 活跃时间 89.3%、占任务墙钟 86.6%——两个分母不同，下文百分比均为前者口径）、scalar_wait 2%、三线并行度 2.87。MTE2/MTE3 搬运大部分与向量计算重叠，未单独落入最终关键路径（各 pipe 统计时间上重叠，非互斥占比）。

**关键路径分解**（2048×7168 bf16，十进制单位）：

| 项 | 值 | 推导 |
|---|---:|---|
| 总搬运量 | 117.4 MB | 读 x1+x2 + 写 x+y = 4 × 2048×7168×2B = 117.4 MB（rstd 8KB 可忽略） |
| 理论 HBM 下限 | 73.4 µs | 117.4 MB ÷ 名义峰值 1.6 TB/s（设备规格值，非实测） |
| 实测向量占用 | 77.4 µs | msprof vec busy（占墙钟 86.6%）；R5 实验证明其中依赖延迟成分在当前测量精度下不可见（见 §4.6） |
| 实际墙钟 | 89.4 µs | 重叠后的关键路径 |

有效带宽 = 117.4 MB ÷ 89.4 µs ≈ **1.31 TB/s**（十进制），为名义峰值的 ~82%——带宽与向量流水接近转折区。**继续削减 V 指令为何没有等比例兑现**：R5 实验直接证伪——把 reduce 依赖"延迟"消掉后 vec busy 分毫未降（多 repeat 累加硬件内部按吞吐执行），同时墙钟受带宽下限托底。两条下限与墙钟的差分别核对：**带宽 17.9%**（89.4 − 73.4 = 16.0µs）、**向量占用 13.4%**（89.4 − 77.4 = 12.0µs）；差距由 ramp/尾行排空、装载/落盘非完美重叠与 DataCopyPad 短粒度构成，V 侧微优化的可兑现空间收敛到 decode 预取等 issue/延迟方向。

**decode 与 prefill 分列**（量级与机制不同，不共用一张纵轴）：

| 档位 | 机制 | 基线 → 终值 | 说明 |
|---|---|---|---|
| decode 1/4/16 tok（SINGLE_N） | 发射+延迟混合瓶颈 | 3.36→3.46 / 3.70→3.87 / 5.32→5.69 | µs 级噪声带内；剩余候选=γ/β 装载并行预取（§4.4③ 的 SINGLE_N 部分） |
| prefill 512-4096 tok（NORMAL/MULTI_N） | 带宽+向量流水 | 上表 | 本报告主体 |

---

## 5. 归因速查表与关键经验

### 5.1 归因速查表

（本项目实际走过的归因路径；"看什么"是 msprof 字段、tiling 日志或代码检视点）

| 症状 / 问题 | 看什么 | 本例读数 → 结论 |
|---|---|---|
| 不确定优化对象是否命中 | `ASCEND_GLOBAL_LOG_LEVEL=1` 采 tiling 日志的 key/ub_factor | 简报说 SPLIT_D，实测 key 30=NORMAL → 重定优化对象 |
| scalar_wait 占比异常高 | `aiv_scalar_wait_time / wall` | 85% → 发射线程被标量往返+单缓冲串行钉死，先消往返再行间流水 |
| vec busy 不饱和 + scalar_wait 高 | 双指标并列读 | 73% + 85% → 双瓶颈：减 pass 与流水都要做，顺序是先减（等价）后流水（重构） |
| 收益是真还是噪声 | **未动路径作金丝雀**，同轮对照 | fp16（未动）波动 ±1µs/±10% → bf16 的 -8.4%/-20.7% 判真，128 档 ±1.2µs 判噪声 |
| 数字看着对但不敢信 | 断言强度审计 | rtol=100 形同虚设 → rstd 行混叠（压缩 bug）漏网；分级严格断言（y=3ulp/rstd=1e-3/x 位级）是前提 |
| 大 shape 收益到顶了？ | 搬运字节 ÷ 墙钟 vs HBM 峰值 | 117.4MB/89.4µs = 1.31TB/s ≈ 82% 峰值（名义 1.6TB/s）→ 剩余空间 ~18%，V 侧优化被带宽墙截胡 |
| "合理"的优化上板无效 | 不纠结理论，直接测 | reduce 拆链 vec busy 分毫未降 → 硬件内部已按吞吐执行，负结果留档 |
| 测试结果离奇（超时/崩溃） | `/proc/loadavg`、设备日志 | 共享宿主 loadavg 200+（192 核）→ CPU 饥饿，结果不可信，先排环境 |

### 5.2 关键经验

1. **先实证分发，再动手**：多变体算子的第一件事是用 tiling 日志钉死每个生产 shape 命中的变体与参数（key、ub_factor、block_factor）。本例简报的变体归属是错的——按简报优化将获得精确的零收益；
2. **发射线程的停等是最优先消除项**：`V_S` 是唯一阻塞发射线程的跨 pipe 等待；scalar_wait 占比是它的直接读数。消除标量往返（Gather+stride-0 广播、rstd 全程 V 域）比削减 V 指令更先做——发射线程解放后，流水与减 pass 的收益才兑现；
3. **bf16 的 pass 下限先算清**：c220 无 bf16 元级乘加 → 所有数学在 fp32 域，加宽/舍入/再宽都是语义必需。可削减的只有"重复计算"（每行重做的 cast、reduce 前的全宽 1/N）；把语义 pass 数数清楚，才知道优化空间还剩多少；
4. **未动路径是天然噪声金丝雀**：同轮测量中代码未变的路径波动多少，噪声带就是多少——中小 shape 的收益真伪全靠它裁决（±10% 波动与 -8.4% 收益的判别实例）；
5. **严格断言是性能优化的前提**：rtol/atol=100 的测试下"126 全绿"不含信息量。y/rstd/x 分级容差（3 ulp / 1e-3 / 位级）+ 行间独立随机数据 + 尾块/边界 shape，才能让"全绿"重新成为精度证据；rstd 行混叠这类错误 y 都会带偏，但幅度恰在宽松容差内；
6. **UB 预算与 tiling 必须成本模型化**：常量 ubFactor 往往是"按最坏列宽 × 每列字节"的全宽安全设计；一旦按 numCol 收紧，缓冲总量就随 shape 比例增长——host 侧必须按 dtype/β 的每列成本 + 固定开销核算，超限回退到已验证路径（并保证回退配置本身落在被验证过的参数域内——本例 colTileNum=1 即回退引入的新雷）；
7. **溢出域也是契约**：数学等价变换要区分"舍入路径 ~1ulp"与"溢出域改变"两类——前者可接受（golden 自身求和顺序也不同），后者必须回退或有界证明（fp16 的 max raw sum 可静态证明远离 fp32 上限，故折叠保留）；
8. **硬件行为要实验裁定，直觉会让路**：reduce 依赖链重构是教科书式优化，上板零收益（多 repeat 累加硬件内部按吞吐执行）；而"acc 在 src1 位的交错累加会 fault、放 src0 安全"这种文档没有的雷，只有设备实验能暴露。负结果与其机理同样留档；
9. **环境先于归因**：共享宿主机 loadavg 飙到 200+ 时，测试超时、vector core 异常、离谱波动都可能是 CPU 饥饿的产物——`/proc/loadavg` 一条命令先排除环境，再怀疑 kernel。

---

## 6. 附录

### 6.1 复现信息

| 项 | 值 |
|---|---|
| 仓库 / 分支 | vllm-ascend @ `add-rms-norm-bias-perf`（基线 7dcbbe56a → 检视修复 d7132b0d5 → 测试补充 621cff8c9） |
| CANN / 驱动 | CANN 9.1.0；npu-smi 25.5.1（驱动/固件随包） |
| 编译 | `bash csrc/build_aclnn.sh $(pwd) ascend910b` → `pip install -e . --no-build-isolation`；**改 kernel 前必须清 build 树**（`csrc/build/binary/ascend910b/{src,bin}/add_rms_norm_bias` + `gen/*_ascend910b_*.done`，src_copy 不追踪源文件改动） |
| 安装自检 | 仓库源与 `csrc/build/.../src/add_rms_norm_bias/op_kernel/*.h` md5 一致，.o mtime 晚于源 |
| 设备 | 910B3，40 AIV，默认频率（OpBasicInfo.csv 记录 1800MHz）；共享宿主机，**loadavg 需 <100**（本战役 200+ 时两组数据作废） |
| 计时口径 | NPUGraph capture+replay：warmup 3 次 → capture（自适应 batch）→ replay 5 样本取中位；脚本 `benchmarks/add_rms_norm_bias.py` |
| 管线归因 | `env -u ASCEND_RT_VISIBLE_DEVICES msprof op --warm-up=10 --launch-count=1 --kernel-name=AddRmsNormBias --aic-metrics=PipeUtilization python benchmarks/add_rms_norm_bias_msprof.py <tokens>`（指令级流水图换 `--aic-metrics=TimelineDetail`） |
| 精度 | `pytest tests/e2e/nightly/single_node/ops/singlecard_ops/test_add_rms_norm_bias{,_strict}.py -q`（**代码与测试在本报告仓之外**，位于 vllm-ascend 仓上述分支；严格测试脚本副本见本目录 `原始数据/`） |
| 原始数据 | vllm-ascend 仓内 `profiling_addnorm_*/`（PipeUtilization CSV）、`traces/`（指令级 TimelineDetail + README）、`ADD_RMS_NORM_BIAS_OPT_NOTES.md`（逐轮全量记录）；四个关键采集点的 CSV 副本见本目录 `原始数据/`（baseline/r1/r2/r5 @2048） |

一条命令复测关键数据：`python benchmarks/add_rms_norm_bias.py`（输出 9 档 × bf16/fp16 的 fused 与 ref 两列 JSON）。

### 6.2 统计口径与噪声

- 表中数值为 5 样本中位数（开发期口径）；关键档位均有 ≥2 次独立复测，**≥512 档轮间差 <1%**，1-128 档绝对波动 ±0.1~1.2µs（±1~10%）——因此小档结论以"噪声带内/持平"表述，不报百分比收益；
- 未动代码的 fp16 路径为噪声金丝雀：其同轮波动即噪声带定义；
- 各 pipe busy/wait 统计时间上重叠，非互斥占比，不可相加。

### 6.3 最终变体覆盖矩阵

（"性能结论"一律为**终值 vs 基线**的当前状态，不含中间轮次的历史读数；分发为检视修复后的最终分发。）

| dtype | shape 范围 | 变体（key） | 是否修改 | 性能结论（终值） | 严格测试 |
|---|---|---|---|---|---|
| bf16 | rows≤40 | SINGLE_N（33） | 是（消 V_S+折叠 1/N*） | 1/4 tok +3~4.6%、16 tok +0.5%——**均属 ±1µs 噪声带，无可靠收益** | 通过 |
| bf16 | rows≥41 | NORMAL（30） | 是（流水化重写） | @2048 -24.8%、@4096 -26.8% | 通过 |
| fp16 | rows≤40 | SINGLE_N（13） | 是（同 bf16 改动） | 噪声带内 | 通过 |
| fp16 | rows≥41 | MULTI_N（14） | 是（折叠 1/N*，fp16 有界证明成立故保留） | @2048 -4.1% | 通过 |
| fp16/fp32 | 中宽非对齐 col | NORMAL（10/20） | 部分（第一轮减法，未流水化） | 测试形状覆盖，无生产 shape | 通过 |
| bf16 | col≤5120 rows≥41 | MULTI_N（34） | 是（共用 ComputeRstd；1/N 折叠已随 bf16 回退） | 测试形状覆盖 | 通过 |
| 全 dtype | col≤2000 | MERGE_N（*2） | 否 | 小列宽，无生产 shape | 通过（未优化） |
| 全 dtype | **col>8016（成本模型超限）或 col>11264** | SPLIT_D（*1，colTileNum≥2） | 否（休眠；8192 为新增回退入口） | 回退路径已验证（8192 边界用例） | 边界通过 |

\* 折叠的保留/回退判据见 §4.5 #3：fp16 有界证明成立故保留；bf16/fp32 的 1/N 已回退为 reduce 前逐元素缩放。
分发说明：**基线分发** col≤11264 时按行数/列宽在 SINGLE_N/MERGE_N/MULTI_N/NORMAL 间选择；**最终分发**仅新增一条规则——NORMAL 前先过 UB 成本模型，超限（如 col=8192 bf16+β）改派 SPLIT_D（colTileNum≥2）。

### 6.4 实现约束与维护红线

1. 事件发射条件必须与消费条件严格一致（`i+2 < rowWork` 一类守卫），kernel 退出时不得遗留在途事件（悬挂标志毒化同核下一个 kernel）；
2. 事件一律 `AllocEventID/SetFlag/WaitFlag/ReleaseEventID` 四件套，禁止 `FetchEventID` 承担多在途语义（连续 fetch 返回同 ID）；
3. 双缓冲槽位一律使用**局部行号**奇偶（`i & 1`），绝对行号会错位产生 NaN；
4. 归约的累加器若用于交错发射，必须放 **src0** 操作数位（dst==src1 的交错轮转在 c220 上 fault）；
5. 修改 rstd 压缩逻辑必须覆盖 rows>8 与全部尾块场景（Gather 偏移按字节逐元素消费）；
6. UB 预算由 host tiling 成本模型计算（`DetermineModeParameters`），kernel 不自行假设；新增/删除缓冲必须同步更新 fixedBytes 常量并用 col=8192 边界用例验证回退；
7. 逐元素 V 操作的 dst/src 必须天然 32B 对齐（对不上用标量 SetValue + S_V 同步，不要硬上 V store）；
8. c220 实证行为（§4.6 两条）不得直接外推到其他架构（A5 regbase 路径另有实现）；修改融合算子数值路径前，先核对 golden 的溢出域与舍入契约（§4.5 #3）。
