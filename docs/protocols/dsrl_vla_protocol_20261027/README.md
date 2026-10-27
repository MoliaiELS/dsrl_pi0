# DSRL 与 VLA 实验执行包

这是 2026-10-27 的研究规约与 Codex 交接包。新实验尚未运行。

## 文件

- EXPERIMENT_SPEC.md：完整研究问题、所有默认与条件实验设置、实现要求、统计、结果解释、验收与来源。
- protocol.v1.json：结构化科学协议。它不是现有 outcome_probe 或 RLinf 可直接加载的配置；Codex 应按此生成实际运行配置。
- CODEX_HANDOFF.md：可以直接粘贴给 Codex 的执行指令。
- validate_protocol.py：Python 标准库静态检查器，只检查种子、矩阵与预算。
- VALIDATION.txt：本交付包静态检查的实际输出，不是机器人仿真结果。

## 怎么用

1. 将整个目录交给能访问你 dsrl_pi0 本地项目与远端工作流的 Codex。
2. 让它先读 EXPERIMENT_SPEC.md 与 protocol.v1.json。
3. 将 CODEX_HANDOFF.md 的任务提示词粘贴给它。
4. 默认执行 first_batch；它不包含新增训练。
5. 根据实际诊断报告选择已写明设置的后续模块。

## 静态自检

~~~bash
python validate_protocol.py protocol.v1.json
~~~

默认矩阵应输出最多 1,010 分支、408,800 策略执行环境步。总物理步硬上限 450,000，还包含构造、重放、重试。把这个检查通过当作仿真实验通过是不正确的。

## 本轮研究的判定顺序

先问语言是否控制目标，再问目标选对后能否执行，再问阶段信息是否有额外帮助。允许微调 VLA，但每次主要比较应保持共同底座、参数范围、任务、预算与评测定义一致。

