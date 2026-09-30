---
name: ascend-operator-optimization-ppt
description: Create or revise technical PowerPoint decks that explain Ascend/AscendC operator optimization results, kernel dataflow, profiling evidence, correctness, and benchmark comparisons. Use for PPT/PPTX deliverables based on operator work.
---

# Ascend 算子优化 PPT

把代码、测试、profiler 和优化报告中的证据转成面向指定听众的技术演示文稿。先检查仓库 `AGENTS.md`、源报告、测试数据和用户给的 PPT 模板；遵循用户指定的页数、时长、受众、语言和视觉风格。

写作前读取共享指南：

- `../../../skills/ascend-operator-optimization-docs/references/presentation-writing.md`

交付单文件演示稿或同时导出 PDF/HTML 时，再读 `../../../skills/ascend-operator-optimization-docs/references/document-export.md` 的独立交付与对应格式检查。路径相对本 `SKILL.md` 所在目录解析；Codex 与 Claude Code 的同名入口保持一致。

## 工作方式

1. 先列出演示目标和主结论，再选择页面顺序。一般从算子/场景、实际路径、基线瓶颈、优化机制、逐轮证据、正确性、最终收益、限制与经验展开；按时长裁剪，不机械套用固定页数。
2. 每页只承担一个主要信息。用明确标题、可读图表和少量解释文字表达；实现细节可放讲稿备注，仅在用户允许时增加附录。关键数据口径仍需在图表附近可见。
3. 性能曲线、结果数字和正确性结论必须回溯到同一份源报告/数据。标注 shape、dtype、硬件、单位、对照版本和测量口径；展示失败或回退时解释原因。
4. 数据图、表格和需要编辑的流程图尽可能使用可编辑对象。图形编码清晰，注明系列和单位；不要用示意图冒充实测 trace。
5. 使用当前环境可用的演示文稿工具，不假定两个 agent 有相同插件。完成后导出 PPTX，检查资源已打包与对象可编辑性，并逐页渲染检查文本溢出、图例/数据标签、错位、字体替换和内容遗漏。修正后复查受影响页面。

## 边界

- 不编造测量值、引用或结论；数据不全时显式标注或暂留占位，并在交付说明里指出。
- 区分单 shape 收益、全场景结论和理论推导；不将计数器百分比当成互斥占比直接相加。
- 保留用户要求的页数和模板。如无指定，选择足以讲清楚证据的紧凑页数，不增加与主线无关的封面或装饰页。
- 若用户先要求报告后要求 PPT，以报告和原始数据为事实来源；发现冲突时先标出并说明采用的口径。
- 独立演示稿在稿内解释优化机制，不要求听众先读姊妹报告或访问私仓；正确性结论以确认的测试证据为准，按用户要求简述。
- 只有用户要求提交/推送时才执行，并且只包含本任务的技能或成品文件，不带入其他未提交内容。
- 最终说明交付位置、渲染检查情况及仍影响使用的限制。
