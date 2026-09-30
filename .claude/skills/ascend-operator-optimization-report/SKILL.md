---
name: ascend-operator-optimization-report
description: Write, review, revise, or export evidence-based Ascend/AscendC operator optimization reports as Markdown, self-contained HTML, or PDF. Use for kernel performance documentation, not implementation-only changes.
---

# Ascend 算子优化报告

面向算子开发、性能复盘和评审，产出有证据支撑的独立优化报告，按用户要求交付 Markdown、单文件 HTML 或 PDF。先读仓库及目录中的 `AGENTS.md`、用户提供的模板和相关实现/测试/测量材料。沿用用户明确指定的格式。

开始写作前读取共享指南：

- `../../../skills/ascend-operator-optimization-docs/references/report-writing.md`

生成 HTML/PDF，或交付需脱离仓库独立阅读时，再读取：

- `../../../skills/ascend-operator-optimization-docs/references/document-export.md`

以上路径相对本 `SKILL.md` 所在目录解析，不是相对终端当前目录。Codex 与 Claude Code 的同名入口共享这些指南；修改入口时检查两者一致。

## 工作方式

1. 盘点可用证据：算子契约、分发/tiling、生产 shape、代码版本、基准脚本、profiler、正确性测试、历史报告。对用户正在编辑的文件做增量处理，避免覆盖其未提交内容。
2. 建立事实表：将数字标记为设备实测、可复算推导、假设或待补；验证单位、shape、dtype、版本和计时口径。存在矛盾时先在文档中标明，能从原始数据核实则核实。
3. 围绕“命中路径 → 基线瓶颈 → 优化假设 → 代码变化 → 性能与正确性证据 → 剩余瓶颈”组织报告。每轮写清有效范围和是否回退。
4. 将真实材料转化为结果表、分发矩阵、内存预算或数据流图。图应支持结论，不能装饰或暗示未测量的结果。
5. 完成后核对数字、图表和引用。对报告中明确引用的本地图片检查文件存在且可读；生成/修改图后实际渲染检查，无法渲染时说明未完成视觉验收。
6. 多格式交付使用同一份已确认正文。单文件 HTML 内嵌图片，PDF 检查完整分页、中文字体和正文完整性；不把导出成功等同于显示正确。

## 边界

- 不编造 benchmark、硬件规格、commit、测试结果或 profiler 结论。缺数据时标“待补/未测”，提供需要补的字段。
- 不把一次特定设备/shape 的性能结果推广为普遍结论。明确芯片、软件栈和有效 shape 范围。
- 本仓库的独立报告不依赖姊妹报告、私仓路径/commit 或外部附件；仅在用户需要且读者可访问时加入这些引用。精度结果按用户要求简述，不为写报告修改测试断言。
- 用户只要求撰写/检视报告时，不擅自改算子实现；用户明确要求生成 PPT 时，继续读取 `../../../skills/ascend-operator-optimization-docs/references/presentation-writing.md` 并按 PPT 交付。
- 创建导出文件不等于获准提交/推送。用户要求推送时只暂存本任务文件，验证提交范围和远端结果，不暴露远端 URL 中的凭据。
- 结尾简要列出新增/修改文件、关键数据缺口和已完成的核对项。
