# dequant_swiglu_quant 融合算子 Ascend 性能优化报告

## 1. 背景

W8A8 动态量化推理里，MoE/共享专家的 FFN 前半段是一串「量化矩阵乘 → 反量化 → SwiGLU → 动态量化」：矩阵乘输出 int32，必须先乘回浮点域、过激活、再量化成 int8 喂给下一层矩阵乘。vLLM Ascend 插件的自研算子 **`dequant_swiglu_quant`** 把后三步融为一个 kernel，减少中间张量的 HBM 往返。它用于每层共享专家；路由专家仅在未走默认融合路径时调用。

本文档总结该算子在 Ascend 910B3 上的性能优化：先以设备上的 tiling key 确认生产命中的 kernel，再通过等价变换、tiling 校真与 group 核分派改善性能。双缓冲实验无可测收益，也暴露出 tiling 预算与 kernel 缓冲布局必须同步核对。

### 测试环境

| 项目 | 值 |
|------|-----|
| 设备 | Ascend 910B3（dav_c220 向量核），40 AIV，UB 192KB/核 |
| CANN | 9.1.0 |
| 模型场景 | 本次 DeepSeek W8A8 配置：共享专家 2H=4096/TP 度（swiglu_mode=1）；测试的 MoE w1 输出为 2H=4096 |
| 测试矩阵 | tokens = 1/4/16/64/128/512/1024/2048/4096 × 2H ∈ {4096,2048,1024,512}；group 路径 g64/g256 |
| 计时口径 | NPUGraph capture+replay（≈背靠背 serving，5 样本取中位）作性能记分板；msprof op 作管线归因，两者绝对耗时不直接混用 |

### 算子契约

```text
f     = fp32(x_int32) × weight_scale[2H] × activation_scale[row]   # 反量化
gate/up = f 左/右半（activate_left 可翻转）
swiglu_mode=1（SwiGLUGate）: clamp(limit) → up += β → silu(α·gate) × up   # gpt-oss 风格
swiglu_mode=0（普通）      : silu(gate) × up
scale = max|swiglu行| / 127 (fp32)
y     = int8( RINT(swiglu/scale) 经 half 中转 TRUNC )              # 动态量化
```

数值链 fp32 → RINT → int32 → half(ROUND, deq 1.0) → int8(TRUNC)；half 中转对
[-127,127] 的整数值无损。

### 结论摘要

| 项目 | 结论 |
|---|---|
| 生产调用路径 | 共享专家与 MoE MC2 回退路径命中 `DequantSwigluQuantBase` / `DequantSwigluQuantGroup`，tiling key 分别为 200000000 / 100000000 / 110000000；所列变体头文件未见这两个调用方命中 |
| 基线画像 | vec busy 约占墙钟 80%；共享专家仅用 36/40 个 AIV，非 cut group 路径存在核分派失衡；其余非 vec 墙钟尚不能仅凭 pipe busy 归因 |
| 核心方案 | 等价 pass 消除（-6.6%）、tiling 校真（40 核 + 公式 db=1 + 饱和守卫）、group 环形核分派 |
| 最终收益 | 共享专家 @2048 **-17.6%**（57.1→47.0µs）、@4096 **-20.1%**（106.0→84.7µs）；MC2 group prefill **-14.0%** |
| 最终状态 | msprof wall 48.5µs、vec busy 36.4µs；有效带宽 779GB/s（名义峰值约 49%）。vec 是主要 pipe 占用，但仅凭这些指标不能排除局部访存或同步限制 |
| 正确性 | 精度测试通过 |
| 回退风险 | decode 1-4 token 观测到 +0.1~0.35µs；MC2 cut-group g256@512 观测到 +11~25%，但宿主负载漂移明显，需受控复测 |
| 已回退方案 | tile 双缓冲（db=2）：本次测试未测得收益，保留负结果 |

---

## 2. 算子在大模型中的位置与数据

### 2.1 位置

```text
token embeddings
  → N × decoder layer
       attention 输出 → npu_dynamic_quant → W8A8 矩阵乘(int32 out)
       共享专家: npu_quant_matmul → 【dequant_swiglu_quant】→ int8 ↓proj   ← 每层必调
       路由专家: grouped_matmul（fusion 默认）或
                 gmm1 → 【dequant_swiglu_quant(group_index)】（MC2/fusion 关闭时）
```

两个调用方（`shared_experts.py`、`w8a8_dynamic.py`）的参数画像一致：x=int32、
dynamic 量化、bias/quant_scale 均为 None——都落入 DskTiling 的能力域。

### 2.2 分派实证：先确认生产命中的 kernel

该算子有多套 tiling 模板和 kernel 变体。静态走读给出分派条件后，
再以 msprof `OpBasicInfo.csv` 记录的 kernel 名后缀（tiling key）核对
两个生产调用方的实际命中情况：

| 调用方 | 配置 | tiling key（设备实证） | kernel |
|---|---|---|---|
| 共享专家（每层） | x=int32、无 group、swiglu_mode=1 | **200000000**（blockDim 36→40） | `DequantSwigluQuantBase` |
| MoE MC2 回退 | x=int32、group_index=每组行数 | **100000000**（普通）/ **110000000**（cut：组数≥32 且均摊≤16 行） | Base / `DequantSwigluQuantGroup` |
| 所列两个调用方未命中 | bf16/fp16 x 直入等 | 10000-30013 | 变体 kernel |

**DskTiling 的能力域 = group_index 存在，或（x=int32 且 swiglu_mode=1）**——
两个生产调用方均落在该能力域，因此本轮重点优化它们实际命中的
`DequantSwigluQuantBase` / `DequantSwigluQuantGroup`。其他变体仍需保留
正确性回归，不能仅凭文件新旧推断性能热点。

两个调用契约陷阱（留档）：group 路径要求 weight_scale 为 2D [组数, 2H]；
`group_index` 语义是**每组行数**（`cumsum_group_list(..., 1)` 的
dst_list_type=1），传 cumsum 会让 kernel 按递增值当行数累加越界，表象是
aivec 崩溃（MTE 非法 GM 地址），极像 kernel bug。

### 2.3 基线画像（msprof @2048×4096，36 核）

| 指标 | 值 | 判读 |
|---|---:|---|
| wall | 59.1µs | NPUGraph 57.1 一致 |
| vec busy | 47.3µs（约 80% 墙钟） | 各 pipe 中占用最高，值得优先减向量工作量 |
| mte2 / mte3 busy | 15.7 / 5.7µs | 与其他 pipe 的时间可重叠，不能相加归因 |
| scalar busy / vector_stall | 16.5µs / 42.5µs | 表明向量相关等待显著，仍需结合 trace 定位来源 |
| 有效带宽 | 639GB/s（名义峰值约 40%） | 未显示带宽饱和，但不足以独立排除访存限制 |

每行（H=2048）约 22 遍全宽向量 pass，其中 4 遍可等价消除（weight scale
广播拷贝每 tile 重复、两次去交错拷贝、β=0 时的无效 Adds）；36 核上限闲置
4 核；20.5KB 死分配（tmpBuf2，
全仓无引用）。

---

## 3. 优化总览

| 轮次 | 手段 | 针对的问题 | 结果 |
|---|---|---|---|
| R0 | 完善精度回归 | 覆盖主路径、边界与异常输入 | ✅ 精度测试通过 |
| R1 | 等价 pass 消除：ws 行乘、去交错消除（原地 strided 半区运算）、β=0 跳过、死分配删除 | 22 遍全宽 pass 中 4 遍冗余 | ✅ @2048 -6.6% |
| R2 | tile 双缓冲（db=2）+ 队列握手合并 | 假设的 MTE2 串行等待 | ⛔ **零收益**（负结果，全回退） |
| R3 | tiling 校真：36→40 核、公式 db=1、饱和守卫、删 mode0 强制 tile | 核数闲置、tile 尺寸被历史公式压小、小批量核饥饿 | ✅ 累计 -17.6% |
| R4 | group 环形核分派 | 每组都从 core 0 分派，尾核闲死 | ✅ 受害 shape -14.0% |

**经验法则**：先用设备实证钉死分派 → 精度回归先行 → 等价变换减 pass →
tiling 与 kernel 缓冲布局联动校真 → 分派均衡 → 每步用未动路径 + eager ref
双金丝雀校准噪声（共享宿主 loadavg 33→123 的漂移曾使小档读数 ±10%）。

---

## 4. 逐轮优化过程

### 4.1 R0：精度回归

在性能改动前完善回归覆盖；基线及后续各轮精度测试通过。

### 4.2 R1：等价 pass 消除（-6.6%）

四项位级等价变换：

```cpp
// ① weight scale 行均匀 → 逐行直乘，消掉每 tile 的 [p,2H] 广播拷贝
for (row = 0; row < proDimsx; ++row)
    Mul(xRow, xRow, weightScale, 2H);
// ② SwiGLU 直接在交错半区上原地进行（clamp/β/silu in place），
//    仅最终乘法物化 [p,H] 结果供量化段 → 消掉两次去交错拷贝
// ③ β==0 时跳过 Adds（x+0.0 恒等；-0.0→+0.0 翻转不影响 int8 乘积）
// ④ 删除 tmpBuf2 死分配（swigluMode==1 时分配 5B/输出元素、全仓无引用）
```

**门槛是本轮的主课**：per-row 形态用少量指令换向量工作量，只在 vec-bound
形状赚回。初版门槛 `proDimsx ≤ 8` 使 2H=2048（p=5）回退 +8.6%、cut-group
decode（每组单 tile）回退 +5.8%；修正为
`rowPath_ = (UbFactorDimx ≤ 4) && (ubDimxLoop ≥ 2)`——**判据必须是运行时
形态（每核 tile 数）而非静态 tile 参数**，单 tile 核是 issue-bound，+3 条
指令直接上关键路径。

![各版本时延演进](latency_evolution.svg)

### 4.3 R2：tile 双缓冲——一个完整的负结果

按固定模板记录（假设 → 实验变量 → 控制变量 → 设备观测 → 结果 → 结论 → 回退）：

1. **假设**：xActQueue 单缓冲使下一 tile 的 MTE2 装载串行在 V 计算之后；
   DB_BUFFER=2（tiling 公式本就按 db=2 预留）应隐藏每 tile ~0.5µs 的
   装载等待，预期大档 -10~15%。
2. **实验变量**：DB_BUFFER 1→2 与 outQueue 双缓冲（kernel 缓冲数）、
   ComputeDequant→ComputeSwiGLU 之间的 EnQue/DeQue 握手合并（单次
   DeQue/Free）；过程中按二分法拆出溢出修复（删 mode0 强制 tile 特例、
   outQueue 回 db=1）与纯增益测量两个阶段。
3. **控制变量**：保持数据、shape、设备与测量口径一致；每轮 clean build、
   核对构建产物，并进行精度回归。
4. **设备观测**：
   - 首版**崩溃**（VEC 指令 UB 越界）：mode0 特例强制 p=4@2H=4096 是
     db=1 时代的调优，db=2 下 xActQueue 翻倍后总量 232KB > 192KB；
   - 修复溢出后：NPUGraph @2048 为 53.85µs，R1 为 53.33µs，差距约 1%；
     独立的 msprof 采集显示 MTE2 busy 从基线 15.7µs 到本轮 19.2µs，
     vec/wall 仍约 80%。busy 时间会重叠，这组数据不能直接证明装载已被隐藏。
5. **性能与正确性结果**：崩溃版本被回归测试发现；修复后精度测试通过。
   本次 NPUGraph 测量未见性能收益。非 vec 墙钟可能包含队列事件、流水
   启停和访存等待；现有 pipe 汇总不足以量化各项占比。
6. **结论**（按强度分级）：**实测**——本次 db=2 实验未测得收益；
   **确定性事实**——tiling 公式（db 预算、强制 tile）与 kernel 缓冲布局
   失配可压小 tile，严重时导致 UB 溢出；**待验证假设**——减少队列事件
   可能改善大档耗时，需用隔离实验或指令级 trace 验证。
7. **回退及原因**：整体回退（db=2、outQueue 双缓冲、握手合并全部还原），
   保持与 R1 最小 diff；记录本次配置和测量结果，供后续复查。

### 4.4 R3：tiling 校真（累计 -17.6%）

三项 tiling 改动（kernel 与 R1 相同）：

1. **核数 36→40**：36 来自 DeepSeek V4 移植，无文档理由；设备 40 AIV。
2. **公式 db 校真**：公式按 db=2 预算而 kernel 实为 db=1（历史失配），
   tile 被无声压小——校真后 p 取真实上限（2H=4096：2→4）。
3. **饱和守卫**：大 tile 减边界开销，但小批量下会减核数（ceil(T/p)），
   丢并行比省 tile 更贵。规则 `p = min(p_max, max(p_legacy, ceil(T/40)))`：
   只在所有核饱和时才用更大的 tile；p_legacy 按历史公式复算以精确复现
   基线小批量行为。

**两次守卫迭代（负结果留档）**：
- 初版反向 cap（p ≤ ceil(T/40) 保核数）：**方向全错**——小 2H 的每 tile
  固定开销远大于并行度收益，2H=512/64 行：32 核 × 2 行 tile 比 3 核 ×
  22 行 tile **慢 72%**。大 tile 优先，守卫只在大 tile 开始吞核数时介入。
- 公式 db 失配期间 T=128 因 p=3 产生 3+1 尾 tile（+17%）——db 校真后
  p=4 整除恢复。

### 4.5 R4：group 环形核分派（受害 shape -14.0%）

审计发现的真缺陷：非 cut group 路径里每组都从 core 0 重新分派
（`if (blockIdx_ < realCoreDim)`），组间不复用核序——64 组 × 32 行时全部
落在 core 0-31、32-39 闲死，比均衡的 cut 路径同工作量慢 1.5×
（93.0µs vs 61.4µs）。修复：`Process()` 维护环形核偏移，每组的核区间从上
一组结束处轮转（mod blockDim），组内行块按环内位置分配。适用面：组数
< 32 或均摊 > 16 行/组的全部非 cut 形态（EP 每卡专家数 < 32、MC2 prefill）。

![分档收益](gains_by_shape.svg)

### 4.6 回退风险与适用范围

| 场景 | 本次观测 | 目前可解释到什么程度 | 后续判断 |
|---|---|---|---|
| decode 1-4 token（全 2H） | +0.1~0.35µs（约 3.1→3.4µs） | T=1 的 msprof 中 scalar busy 2.27µs、vec busy 1.07µs，提示标量/调度开销值得检查；尚未隔离出代码生成、分支或测量噪声的贡献 | 与大档收益权衡；需在固定负载下重复测量，并检查模型级 decode 时延，不能仅凭算子旁流推断影响已被掩盖 |
| g256@512（MC2 cut-group 回退路径） | +11~25%，同日读数随宿主负载漂移 | 本轮未直接修改 cut-group 分派逻辑，但尚不能将回退归因于代码生成；该形态仅在未走默认 fusion 时启用 | 先受控复测并与未动路径对照，再决定是否按 shape 拆分 kernel |

### 4.7 总收益与最终状态

**累计优化（NPUGraph，共享专家主路径 2H=4096，µs）**：

| tokens | 基线 | +R1 | +R3/R4 最终 | 累计 |
|---:|---:|---:|---:|---:|
| 512 | 20.98 | 20.72 | **18.38** | **-12.4%** |
| 1024 | 32.96 | 31.86 | **27.93** | **-15.3%** |
| 2048 | 57.06 | 53.33 | **47.02** | **-17.6%** |
| 4096 | 106.01 | 98.46 | **84.73** | **-20.1%** |

2H=2048 @4096 -11.5%、2H=1024 @4096 -12.6%、2H=512 @4096 -4.9%；
group 路径 g64@2048 **-14.0%**（93.0→80.0µs）。其余未列形状的收益需按
各自 2H 与 tokens 查看，不能把所有中档概括为持平。

![瓶颈归因](attribution.svg)

**最终态管线画像**（@2048×4096，msprof）：wall 48.5µs，vec busy 36.4µs
（约为 wall 的 75%），两者相差约 12.1µs。各 pipe busy 可重叠，这个差值
不是已测得的“队列开销”，不能拆成各项相加。有效带宽 779GB/s（名义峰值
约 49%）尚不足以单独判定内存带宽是否处于关键路径。若继续优化，应对
队列事件、流水启停和访存等待分别做隔离测量。

---

## 5. 归因速查表与关键经验

### 5.1 归因速查表

| 症状 / 问题 | 看什么 | 本例读数 → 结论 |
|---|---|---|
| 不确定优化对象 | msprof `OpBasicInfo.csv` 的 kernel 名后缀（tiling key） | 本轮两个生产调用方命中 key 200000000/100000000/110000000 对应的 Base/Group kernel |
| op 调用即崩（MTE 非法 GM 地址） | 先查调用契约再怀疑 kernel | group_index 要**每组行数**（非 cumsum）、ws 要 2D；传错语义=越界崩溃 |
| 装了新 kernel 行为没变 / 空跑一轮 | 构建产物校验与实际加载版本 | 先排除缓存或旧产物；rebase 后陈旧 CMake 缓存也可能引用已删目标 |
| 双缓冲后墙钟不降 | NPUGraph 主指标与 msprof pipe busy 分开看 | NPUGraph 53.33→53.85µs；MTE2 busy 的变化不能单独证明装载是否处于关键路径 |
| 双缓冲后崩溃（VEC 越界 UB） | tiling 公式 vs kernel InitBuffer 逐项对账 | 公式 db=2、kernel db=1、特例 p=4 三者失配 → 232KB > 192KB |
| 大 tile 反而更慢 | ceil(T/p) 核数 vs 每 tile 固定开销 | 2H=512/64 行：32 核×2 行比 3 核×22 行慢 72%——**小 2H 大 tile 优先** |
| 收益真假（共享宿主） | 未动路径 + eager ref 对照，跨时段复测 | loadavg 33→123 时小档曾漂移约 ±10%；单次差异不宜当成稳定回退 |
| 小 shape 变慢但代码路径相同 | msprof @T=1 的 scalar/vec busy | scalar 2.27µs、vec 1.07µs，提示应检查标量与调度开销，具体原因待隔离 |

### 5.2 关键经验

1. **分派实证先于一切**：多变体算子先用设备上的 tiling key 钉死生产命中的
   kernel 文件；"文件新=路径热"的直觉在本算子上完全失效；
2. **调用契约即坑**：group_index 语义（行数 vs cumsum）与 ws 维度传错都
   表现为设备端崩溃，先证契约再查 kernel；
3. **精度回归先行**：性能改动前先确认主路径和边界条件均通过；
4. **per-row/双路径等价变换有门槛**：用指令数换向量工作量只在 vec-bound
   赚回，判据要用运行时形态（每核 tile 数）；
5. **tiling 公式与 kernel 缓冲布局是一份账**：db 数、死缓冲、强制 tile
   任何一方改动都要重新对账，否则要么无声压小 tile、要么直接溢出崩溃；
6. **流水缺口先归因再动手**：本次 db=2 未测得收益，但 pipe busy 汇总
   不足以确定 MTE2 是否处于关键路径，更不能直接把剩余墙钟归为队列开销；
7. **负结果完整留档**：db=2、反向 cap、int32→int8 直转（classic API 无
   此类型对，仅 MicroAPI 有）——三个负结果各带机制，避免未来重推。
