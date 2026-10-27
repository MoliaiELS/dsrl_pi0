# 给 Codex 的执行指令

把本文件与 EXPERIMENT_SPEC.md、protocol.v1.json、validate_protocol.py 放到 Codex 可读的工作区。下面整段可直接作为任务提示词。

---

请在现有 MoliaiELS/dsrl_pi0 项目中实施并执行本实验规约。先完整阅读 EXPERIMENT_SPEC.md 和 protocol.v1.json，再读当前仓库 AGENTS.md、已有 remote/job 配置。不要只返回研究建议或待办清单。

## 目标

完成默认 first_batch：E0 工程核验、E1 开发与辅助文本、E1 独立确认、E2 小规模开发诊断。输出可复核的原始结果、统计、预算和 REPORT.md，回答失败主要来自目标选择、动作执行还是阶段衔接。

本次 first_batch 不包含新模型训练。后续 SFT、RLinf PPO、阶段估计与恢复的参数已经写进规约，先根据首批实际证据确定下一批。

允许未来微调 VLA，包括 action expert 或 LoRA。“冻结 VLA”不再是研究不可变约束；本次评估保持现有权重固定，是为了诊断与可比。

## 先做这些具体工作

1. 读取当前分支、dirty status、submodule 状态和 AGENTS.md；保留用户改动。在独立分支/worktree 实施，记录实际 source commit，不强制回退版本。
2. 按既有 Windows 编辑、git 同步、远端固定 commit worktree、Slurm 执行流程工作。远端命令前按现有约定 source bash.sh；从当前配置解析 SSH alias、setup script、模型和结果路径。
3. 运行交付包内的静态检查：
   
   ~~~bash
   python validate_protocol.py protocol.v1.json
   ~~~
   
   这只验证设计、种子和预算，不表示仿真或训练已验证。
4. 审计并复用 tools/control_probe.py、tools/outcome_probe.py 的模型/controller恢复、噪声生成和精确重放。新增完整双目标 rollout 与 evaluator；不要只替换旧 outcome_probe 的 instructions。
5. 实现 tools/language_goal_probe.py、tools/stage_probe.py、tools/research_runner.py、tools/research_stats.py，或实现职责等价且更符合仓库风格的少量文件。若文件名不同，在报告给出明确映射。
6. 将新模式接入已有 Slurm、smoke、commit/config hash gate 与断点续跑。协议 JSON 是科学规约，尚不是现有代码的原生配置；需要解析、校验并生成 resolved config。
7. 完成 E0 的有效性测试，再跑开发集。确认 prompt token 未截断、B 指令始终保持 B、A/B目标正确、400步不被原goal done终止。
8. 完成E1开发后，再完成E2 global开发、固定时间校准与fixed/oracle开发；随后统一锁定整个协议、case manifest、所有hash，再跑E1确认集。按 reset cluster 统计并产出完整报告，确认后不再修改E2校准字段。

## 不能改错的设置

- E1 固定同一个奶酪任务物理场景，只更换全回合文本。
- A=cream_cheese_1 入托盘；B=butter_1 入同一托盘。
- 当前工具会在40步后切回原文、用原奶酪目标评分并提前done，三处都必须处理。
- E1固定400步，warmup=0，query=20，prediction horizon=50，denoise=10。
- 同时报告首次稳定抓取对象与对应官方目标曾经成功；不能混成一个“理解成功率”。
- 五策略为 Base、bounded-random、三个已有DSRL controller。空/无关文本只做开发辅助，requested_goal_id=null。
- 不按策略失败筛掉初态，不根据结果更换B、改提示词、追加确认样本或读取保留集。
- E2两个官方drawer任务都保留；global/fixed/oracle共三条件。oracle只看当前真值，标为信息上界。
- 三种seed分开：reset、sampling、controller training。
- 所有评估不更新权重、不写训练replay。
- 首批≤408,800策略执行步；含构造/重放/重试的总物理步≤450,000；累计分配GPU时间≤24小时。

## 执行行为

遇到普通工程错误，定位并修复，继续已授权的当前批次。每个case最多2次技术重试，保留原始错误；不能用新随机seed重跑到成功。

如果环境、数据、资源或访问能力确实阻断，完成不受影响的代码、配置、静态验证和可运行部分，写清 BLOCKED 原因与精确恢复命令。不要称作已完成实验，不要为“有结果”而改变目标、horizon或基座。

到预算边界写 PARTIAL_BUDGET，保存可恢复状态和已完成结果。不要把输出截短后标 COMPLETE，也不要在未知GPU可用性下启动全部后续训练。

## 最终交付

至少输出：

- 实际代码commit、已解析配置、模型/controller/norm stats hash；
- 完整case manifest、逐回合和逐步事件、动作与噪声记录；
- 两目标混淆矩阵、官方active-goal成功率、配对cluster区间；
- E2三条件对比、阶段失败分类；
- 缺失与工程失败分母、总物理步和GPU成本；
- REPORT.md、decision.json；
- 下一批只能推荐一个首要干预方向，并解释它针对哪种已观察到的错误。

最终回复不要只说“任务已提交”。说明哪些实验实际完成、关键数值、证据路径、限制和下一步；保留负结果与不确定结论。

