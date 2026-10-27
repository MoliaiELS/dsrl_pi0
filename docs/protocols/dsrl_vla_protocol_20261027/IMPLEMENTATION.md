# 执行映射与恢复入口

原始六文件保持交付内容；`configs/research_protocol_v1.json` 是输入规约副本，运行时解析任务注册表、checkpoint、norm stats、实际提示token与环境路径，再保存 `resolved_protocol.json`。JSON规范化哈希与原始文件SHA分别记录，不混用。

| 职责 | 实现 |
| --- | --- |
| 完整双目标回合、冻结模型/controller恢复、官方谓词、输入/动作接口 | `tools/language_goal_probe.py` |
| 阶段文本、当前复合谓词滞回、固定时序校准 | `tools/stage_probe.py` |
| case计划、首次稳定抓取状态机、原子结果、预算与哈希 | `tools/research_core.py` |
| 顺序、E0、单case重试、确认前统一锁定、Slurm运行入口 | `tools/research_runner.py` |
| reset-cluster bootstrap、缺失分母、混淆矩阵、报告与决策 | `tools/research_stats.py` |
| 无GPU科学语义检查 | `tools/test_research.py` |

本轮 `research-smoke` 为10个40步技术分支，额外两次40步原动作重放及2个单步真值witness，低于原20短分支/800步上限。E1 dev/aux/confirm仍为100/30/800；E2 dev为60。本轮无新优化器、训练replay或梯度更新。

```powershell
powershell -ExecutionPolicy Bypass -File tools/remote.ps1 sync
powershell -ExecutionPolicy Bypass -File tools/remote.ps1 research-smoke
powershell -ExecutionPolicy Bypass -File tools/remote.ps1 status <E0_JOB_ID>
powershell -ExecutionPolicy Bypass -File tools/remote.ps1 research -SmokeJobId <E0_JOB_ID>
```

正式模式提交前与运行时均核验E0的commit、协议和成功标记。全程仅允许一个research GPU作业。E0可按集群实际资源调整到node1，必须保留20分钟上限；完整评估使用计算节点，资源配置通过 `-ConfigPath` 指定。

同版本恢复使用 `research -SmokeJobId <E0_JOB_ID> -RestorePath <原始结果目录>`；它保留已完成case，拒绝输入配置/commit/case manifest或产物哈希变化。技术失败保留attempt目录，最多两次重试（共三次尝试），正常任务失败不重跑。修复代码后不能冒充原版本续接，必须新建版本和运行并保留原记录。

原子case提交包含动作/噪声/key、逐步事件、逐query提示、终态和文件SHA。预算在物理执行前预留，崩溃遗留预留以不确定上界计入，防止漏算。`POLICY_FAILURE` 是有效观察，不能与工程异常合并。

统计只用完整reset cluster作主比较，缺失最坏/最好界另列。控制器seed不当作额外独立初始状态。开发/确认不混池；E2开发信号不能当独立确认结论。保留集不展开、不读结果。
