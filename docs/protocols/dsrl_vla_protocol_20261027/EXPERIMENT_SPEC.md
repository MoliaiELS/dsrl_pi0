# DSRL 与 VLA 指令条件化研究实验规约

版本：v1.0，2026-10-27  
用途：交给 Codex，在现有 dsrl_pi0 工作流中实现、运行、验证并报告。  
状态：**研究设计与实施说明，尚未执行本文件提出的新实验。**

## 1 研究问题与本轮交付

本研究的问题是：**在视觉场景本身足以提示某个常见任务时，VLA 是否仍能由语言指定目标；在目标与局部技能明确之后，当前阶段信息是否能改善多阶段任务的执行与强化学习适配？**

研究允许修改 VLA 参数。现有 DSRL 继续作为有价值的已训练基线；后续 action expert PPO、LoRA 等适配方式按诊断结果使用。“冻结 VLA”只属于部分对照的设置，不是研究的永久限制。

本规约把工作拆成两层：

1. **默认首批**：E0 工程核验、E1 双目标指令诊断、E2 多阶段小规模诊断。把错误定位清楚，产出可复核结果。
2. **条件后续**：E2 独立确认、必要的共同 SFT、RLinf 参数微调、可部署阶段估计与恢复。每个模块都有触发条件与预算，不把所有组合一开始全部运行。

Codex 默认执行 first_batch，完成后提交结果报告与决策文件。后续模块已经给出设置，但不属于 first_batch 的运行清单。选择后续模块属于下一批研究任务的范围界定，不需要在首批内部反复请求例行确认。

## 2 已有证据与本规约的边界

以下是旧报告与源码已支持的事实，不是本轮预期结果：

| 已有结果 | 能说明什么 | 不能说明什么 |
|---|---|---|
| 奶酪入托盘任务 Base 35/100；三份最终 DSRL 控制器 99/100、100/100、98/100 | 单任务潜噪声 RL 的实现与适配有效 | 整个 LIBERO benchmark 复现完成；语言策略已有独立收益 |
| 部分开发状态上 regrasp 文本有改善；独立确认中收益不稳定 | 文本干预值得继续诊断 | 已经学会 Instruction RL 或 Joint RL |
| C2 v6 四个有效参考上，原指令 Base 与 DSRL 都 4/4 | 这些条件恢复状态可被现有方法完成 | 恢复选择器存在成功率提升空间；覆盖所有掉落情况 |
| 当前 SAC 只读外部图像与 proprio；不读文本 | 同一个初始观测下，噪声控制器不能直接按 A/B 文本选择不同噪声 | 基础 VLA 完全不使用语言 |
| 现有实现更新外部 SAC controller，原 π0 权重冻结 | 目前复现对应标准 DSRL 的参数边界 | 后续研究必须保持 VLA 冻结 |

旧报告是本规约的事实起点：[实验报告][S1]。标准 DSRL 冻结基础策略的设计见原论文 [S2]。RLinf 的 action expert PPO 是另一条可用的适配路线 [S8]。

**特别区分**：模型无视空文本、不相关文本、错误目标文本，是不同现象。空文本仍有 tokenizer 特殊 token；不相关文本没有可评分的机器人目标。只有“相同场景、两个合法且不同的任务指令”的对照，才能直接检验目标是否受语言控制。

## 3 版本、环境与资源约定

### 3.1 固定来源

| 项目 | 审计版本 |
|---|---|
| 用户仓库 | https://github.com/MoliaiELS/dsrl_pi0 |
| 用户分支 | experiment/libero-baseline-20261018 |
| 本次审计的分支 commit | 4774febfa18cc64f3c28e6f39d63bee28676a8fb |
| 原基线实验 commit | c227be8c48353115b7c5de0b3e99fb17d388462f |
| LIBERO 子模块 | 87edbd1b823e464bc66557d213cc45e9f88aa543 |
| OpenPI 子模块 | a6d2400d2534ce32e7bdf8747709b97aaef8ec04 |
| 条件后续 RLinf 审计 commit | fab7f407fbc611085eb163962e3c89c0ab424c1a |

先读当前工作区 AGENTS.md 与远端执行入口，检查分支是否已前进。若前进，保留用户改动，记录差异，在独立分支或 worktree 实施。不要为了回到本表而重置、覆盖用户现有代码。

AGENTS.md 中旧“冻结 VLA”研究背景由本轮用户允许微调的明确指示更新；其版本记录、工作区保护和 Slurm 工作流仍适用。[S3]

### 3.2 执行环境

沿用 Windows 本地修改与 git 同步、远端固定 commit worktree、Slurm 执行的既有流程。远端命令前按现有约定 source bash.sh；实际路径从当前机器配置解析。SSH alias 的历史提示为 idac_sever，实际使用前读取现有配置验证。

已知运行线索：

- 模型配置 pi0_libero，来源 s3://openpi-assets/checkpoints/pi0_libero。
- 模型与结果历史数据根目录为 /mnt/slurmfs-4090node3/user_data/yguo704/dsrl_pi0/。
- 控制器来自 runs/job-18046、runs/job-18103、runs/job-18123 的 checkpoint500000。
- 由既有配置与 manifest 解析模型、norm stats、controller 的绝对路径，不拼猜路径。
- 每个 GPU job 使用 1 GPU、8 CPU；首批最多并行 1 个 GPU job。
- 长评估与训练提交 Slurm；登录节点只做轻量准备与检查。
- 数据和 checkpoint 放现有数据目录，避免复制多份模型进代码 worktree。
- 记录 GPU 型号、显存容量、CUDA/EGL、依赖版本、实际 Slurm 资源；不要凭旧故障永久排除某节点，也不要覆盖系统 GPU 分配。

首批总物理控制步硬上限为 450,000；累计已分配 GPU 时间上限 24 小时。预测超额时停止在可恢复边界，写 PARTIAL_BUDGET，保留已完成结果；不改短 horizon 来凑 COMPLETE。预算包含重放、构造与技术重试，不能只统计有效结果。

学校 A100 作为后续可选资源。以实际可获得的 40GB/80GB 型号做 profile；不把单张 A100 等同于“全参数在线 RL 一定能跑”。

### 3.3 共同模型与控制设置

以下为本规约锁定设置。与当前运行环境发生不一致时，先记录并解决，再启动确认集。

| 设置 | 数值 |
|---|---|
| 模型 | 同一份 π0_libero |
| 模型 latent/action prediction | 50 × 32 latent；50 × 7 环境动作 |
| 每次实际执行 | 前 20 个动作；之后重新观察与推理 |
| 去噪步数 | 10；运行时断言实际调用未被覆盖 |
| 控制频率 | 20 Hz |
| RGB | 外部、腕部两路；渲染 256 × 256，沿用原预处理至 224 × 224 |
| proprio | 8 维，保持现有顺序、单位、姿态表示 |
| DSRL 输入 | 原有 64 × 64 外部图像 + proprio，不增加语言输入 |
| 初始化 | 有 seed 的 env.reset；不偷偷改成官方 pruned_init |
| warmup | 0 个显式环境控制步 |
| E1 budget | 400 个 env.step |
| E2 budget | 600 个 env.step |
| prompt 最大长度 | 48 token；必须检查未截断 |
| 评估更新 | 0；不写训练 replay |
| 状态重放容差 | physics 1e-8；proprio 1e-7；两路图像像素差 0 |
| 同实现动作重复容差 | 1e-5 |

原 OpenPI 的 sample_actions 默认去噪步数是 10 [S5]。不要把 RLinf 的其他默认采样值直接带进比较。

每次只执行 chunk 的前 20 步。换文本发生在下一个 query boundary，旧预测未执行的 30 步直接丢弃。不得保留旧文本生成的动作队列后，却把其执行结果标为新文本效果。每个 rollout 从空队列开始。

新探针复用 obs_to_pi_zero_input、obs_to_img、obs_to_qpos 与原 output transform。原代码将 infer 返回的动作直接传入 env.step；不能额外夹爪二值化、翻转符号、重复归一化或新增动作裁剪。E0在同一observation、prompt、显式z下比较新旧infer前20×7动作，atol=1e-5，不增加环境步。

## 4 实验矩阵与预算

### 4.1 默认首批

| 阶段 | 矩阵 | 分支数 | 每支上限 | 策略执行步上限 |
|---|---|---:|---:|---:|
| E0 工程短测 | 最多 20 个短分支，含重复核验 | ≤20 | 40 | ≤800 |
| E1 开发 | 10 reset × 2 指令 × 1 sampling seed × 5 策略 | 100 | 400 | 40,000 |
| E1 辅助文本 | 5 reset × 3 文本 × 1 sampling seed × 2 策略 | 30 | 400 | 12,000 |
| E1 独立确认 | 40 reset × 2 指令 × 2 sampling seeds × 5 策略 | 800 | 400 | 320,000 |
| E2 开发 | 2 任务 × 10 reset × 3 调度条件 × 1 sampling seed | 60 | 600 | 36,000 |
| **首批合计** | 不含条件后续 | **≤1,010** | — | **≤408,800** |

上述不是包含一切准备开销的“总交互”。450,000 的首批总物理步上限，为构造、重放、predicate 核验、技术重试留下最多 41,200 步。E1 从 raw reset 开始，无策略参考前缀。

E1 固定跑满 400 步，因此完整矩阵的执行步数固定。E2 可在官方完整目标达成后结束，所以表中为上限。

### 4.2 条件后续预算

| 模块 | 明确上限 |
|---|---|
| E2 独立确认 | 480 分支，≤288,000 策略步 |
| E1 布局交换附加条件 | 新增 800 分支，≤320,000 策略步 |
| 局部衔接探针 | ≤40 分支，≤24,000 续跑步，前缀另外统计 |
| 共同 SFT | 1 个 seed，500 updates pilot，最多 3,000 updates |
| PPO pilot | G/GS 各 1 run，各 100,000 有效环境步；总约 200,000 |
| PPO formal | G/GS × 3 seeds，6 runs，各 500,000；总目标 3,000,000 |
| 可部署阶段估计验证 | 1,440 分支，≤864,000 策略步 |
| 可选 stage-ID 训练对照 | 3 runs × 500,000；另计 1,500,000 |
| 条件恢复确认 | 最多 1,920 续跑分支；每支实际剩余步数小于 600 |

不要把 formal PPO 的 3,000,000 训练步当成整个后续项目总成本：验证、最终测试、保留能力评估、SFT 演示数量和构造重放需要另外计数。第 12 节给出训练评估频率。

## 5 种子、配对与预注册

### 5.1 种子划分

| 用途 | reset seeds |
|---|---|
| E0 技术核验 | 9000–9009 |
| E1 开发 | 3000–3009 |
| E1 辅助 | 3000–3004，开发集子集 |
| E1 确认 | 3100–3139 |
| E1 保留 | 3200–3239，首批不读取 |
| E2 开发 | 每任务 4000–4009 |
| E2 确认 | 每任务 4100–4139 |
| E2 保留 | 每任务 4200–4239 |
| SFT 后新目标检查 | 3300–3319 |
| SFT 后新阶段检查 | 每任务 4300–4319 |
| PPO 训练验证 | 每任务 6100–6109 |
| PPO 最终测试 | 每任务 6200–6239 |
| 可部署阶段最终测试 | 每任务 6300–6339 |
| 恢复开发／确认 | 每任务 7000–7009／7100–7139 |

旧报告保留的 200–209 继续保留。

先扫描已有 manifests 是否已经使用新种子。若冲突，在任何新结果产生前整体平移该阶段开发、确认、保留命名空间，更新 resolved protocol 与哈希。不能看到表现后换种子。

sampling seed 在开发中为 0，确认中为 0、1。现有 controller_training_seed 为 0、1、2。PPO pilot 使用 17，正式训练从共同底座重新开始，使用 0、1、2。

噪声 key 沿用：

~~~python
base_key = jax.random.PRNGKey(73000 + reset_seed + 100000 * sampling_seed)
query_key = jax.random.fold_in(base_key, query_index)
~~~

不同指令在相同 reset、sampling seed、query index 下复用 key。各噪声模式的分布不同，不能声称噪声数组都相同。DSRL 轨迹分叉后观测不同，之后噪声也可以不同；第一帧相同观测和 key 的噪声必须相同。

### 5.2 锁定时点

E0 与开发阶段允许修复代码、token 截断、目标 evaluator 和状态恢复错误。不能根据成功率更换 A/B 对象、搜索提示词或改变主要指标。首批统一顺序为 E0→E1 dev/aux→E2 dev与时间校准→锁定整个resolved protocol→E1 confirm，避免确认后再改E2字段导致协议哈希变化。

确认集开始前，保存：

- resolved_protocol.json；
- source commit、子模块、模型与 controller hashes；
- 完整 case manifest 与固定运行顺序；
- prompt 原文、token/mask 长度及 hash；
- 固定时间调度的切换时刻；
- 统计脚本与协议 SHA-256。

确认集执行过程中，不根据中途结果改变样本量、方法、文本或评判规则。发现实质错误时，将该批标为无效版本，修复后使用新的确认命名空间；保留原失败记录，不能把修复前后结果混成一个确认。

## 6 E0 工程核验

### 6.1 必须通过的核验

1. **模型与 controller 恢复**：定位三份指定 checkpoint500000；验证 source run、training seed、task 与文件 hash。现有 loader 可能只检查 latest checkpoint，不能用任意 latest 替代指定文件。
2. **输入链路**：同一 observation 下，A/B 的图像与 proprio 完全相同，prompt/token 不同。空文本必须保留为空字符串，不能回填默认任务。
3. **文本持续性**：E0 的 40 步短测检查两个 query 的 B 文本与 token 均保持 B；随后在 E1 开发的完整 400 步分支检查所有 20 次 query，禁止第 40 步后切回 A。该完整检查计入 E1 开发预算。
4. **双目标 evaluator**：明确验证 A 成立/B 不成立、B 成立/A 不成立、两者都不成立、两者都成立四类 predicate witness。构造 witness 的仿真真值只用于测试，不进入评估策略输入。
5. **原 done 不提前结束**：在 A 已成立的 witness 中继续合法 step，真实仿真短支仍最多40步；用mock runner核验完整400步计数，再由E1开发真实400步分支确认。记录 native_done，E0仿真仍不超800步。
6. **抓取类别**：检测所有可抓取的 movable objects，覆盖 A/B/OTHER/MULTIPLE/NONE，不能只检查奶酪。
7. **精确共同状态**：fresh env + 相同 seed + 记录动作重放；验证 physics、proprio、双图像、XML。直接复制 sim state 不足以替代控制器历史。
8. **无学习泄漏**：评估期间无 optimizer.step、无 replay 插入；旧 controller、VLA 参数 hash 不变。
9. **边界计数**：400 步包含准确 20 次 query；40 步短测包含 2 次。异常退出不能伪装成成功完成。
10. **断点续跑**：已完成 case 不重复执行或覆盖；结果原子写入。manifest 与现有 output hash 不匹配时拒绝拼接。

这些是会直接改变科学结论的测试，值得实现。无需为日志字符串、简单配置读取等低风险行为扩写大量镜像测试。

### 6.2 无需运行训练

E0 不训练新 controller，不做 100 回合基线重现。已有三次复现结果仍保留。E0 的目标是确认新探针没有破坏输入、计分和运行语义。

## 7 E1 相同场景的双目标诊断

### 7.1 场景与目标

固定实例化：

LIVING_ROOM_SCENE3_pick_up_the_cream_cheese_and_put_it_in_the_tray。

所有分支使用同一份物理环境定义。不要随 prompt 切换 task_id、BDDL、obj_of_interest 或视觉输入。

| 目标 | 指令原文 | 目标谓词 |
|---|---|---|
| A | pick up the cream cheese and put it in the tray | In cream_cheese_1 wooden_tray_1_contain_region |
| B | pick up the butter and put it in the tray | In butter_1 wooden_tray_1_contain_region |

已核对官方 A/B BDDL 的 problem、domain、regions、fixtures、objects、init 完全相同；仅 language、obj_of_interest、goal 不同。这给出了无需新增物体的合法同场景任务对。[S6]

字母汤目标也有相同物理定义，可保留为未来扩展，但不因黄油失败而自动替换。这里“合法任务”不等于已证明当前 π0 有能力完成它。

### 7.2 五个策略

| ID | 设置 | 作用 |
|---|---|---|
| base | 每个 query 完整 50×32 独立标准高斯 | 原始 VLA |
| bounded_random | 32 维 U(-1,1)，时间维重复 50 次 | 匹配 DSRL 范围、维度与重复形式的未学习对照 |
| dsrl_s0 | job-18046，training seed 0，step 500000 | 已训练单任务 controller |
| dsrl_s1 | job-18103，training seed 1，step 500000 | 同上 |
| dsrl_s2 | job-18123，training seed 2，step 500000 | 同上 |

不为每个 DSRL 重复复制 Base 数据并将其当作独立重复。

这五组还不能完全分解噪声幅度与时间相关性的每一项影响。如 E1 显示主要差异可能来自噪声参数化，再以独立附加规约加入 repeated Gaussian；首批不增加此维度。

### 7.3 指令与终止语义

E1 的 A/B 指令**从头到尾保持不变**。全支固定 400 步，完整记录 A/B 两个目标的 first passage 与终帧状态。

当前 outcome_probe.py 在 intervention_steps 后切回 instructions[0]，并使用原目标成功信号；因此不能仅改 JSON 字符串复用原 rollout。[S4]

最小新逻辑：

~~~python
# 包装层级必须以实际对象结构验证；下列是接口意图。
goal_A = bool(env.env._eval_predicate(
    ["in", "cream_cheese_1", "wooden_tray_1_contain_region"]))
goal_B = bool(env.env._eval_predicate(
    ["in", "butter_1", "wooden_tray_1_contain_region"]))
~~~

环境构造 ignore_done=True；runner 把返回值命名 native_done，记录但不因它 break。锁定版本 LIBERO 的 step 会把返回 done 改为原 _check_success；只设置 ignore_done=True 不会改变这个返回信号。[S7]

固定 400 步忽略的是已知原任务成功终止。非有限动作、数值崩溃、渲染失败等仍写工程错误，不强行继续。

### 7.4 两个并列主轴

**主轴一：目标选择。**

对每个可移动对象分别累计连续 grasp 步数。第一次有对象达到3步时锁定首抓分类：该步仅一个对象达到阈值，按A/B/OTHER记录；同一步有多个对象首次达到阈值，记MULTIPLE；全程没有则NONE。后续换抓别的物体不改写首抓分类。保存完整逐步 contact/grasp 事件，以便检查物体滑动、提前接触等边界。

报告：

- 每条指令的 correct_first_grasp；
- 指令 × 首抓对象混淆矩阵；
- pair_both_correct：同一 reset 和 sampling seed 下 A/B 两支均首先选对；
- B 指令下仍首抓 A 的 wrong_target_bias；
- NONE、OTHER、MULTIPLE 比例；
- 语言干预效应 L：

~~~text
L = 0.5 * [
  P(first_grasp=A | prompt_A) - P(first_grasp=A | prompt_B)
  + P(first_grasp=B | prompt_B) - P(first_grasp=B | prompt_A)
]
~~~

L 接近 1 表示明显随语言选择；无论指令如何都抓 A 时接近 0。若大部分根本没有抓取，低 L 不能单独归因于语言理解。

**主轴二：任务执行。**

active_goal_ever_success = 当前指令对应的官方 BDDL 谓词在 400 步内是否曾经成立。A 看 A，B 看 B。同时报告两个目标的 ever、first_step、final。

官方 In 判据并不额外要求松手或连续稳定 3 步。因此这两个条件只能另列次指标，不能悄悄替代官方成功语义。

**次指标：**

- active_goal_stable_released_success：active predicate 为真且目标未被抓持的联合条件连续成立3步；
- active_goal_first_completion_without_prior_wrong_goal：active raw first step存在，wrong raw first step不存在或严格更晚；同时完成不算选择性完成；
- 两目标都曾完成；
- 目标后来是否被破坏；
- 首次接触、稳定抓取、抬升与成功时间；
- restricted time-to-success：失败按 400 计入。只对成功回合取平均另列，避免幸存者偏差。

### 7.5 辅助文本

仅在 3000–3004，用 Base 与固定 dsrl_s0 各跑：

| ID | 文本 | requested_goal_id |
|---|---|---|
| A_paraphrase | place the cream cheese in the tray | A |
| empty | 空字符串 | null |
| unrelated | the weather is pleasant today | null |

empty/unrelated 只报告行为与 A/B goal 的发生率，不能写成“空指令任务成功率”。这些对照回答默认行为、提示敏感性，不能单独证明记忆、训练数据泄漏或完全忽略语言。

### 7.6 解释与下一步

| 结果模式 | 当前可支持的解释 | 后续干预 |
|---|---|---|
| Base 随 A/B 选目标，DSRL 多数仍偏 A | 当前单任务 controller 可能把动作分布拉向训练目标 | 研究目标条件化适配；直接参数微调亦可 |
| Base 与 DSRL 都首抓同一对象，且其他基本运动正常 | 当前场景存在默认目标偏置或语言 grounding 不足 | 多目标 SFT/LoRA；不能直接称“记住全部任务” |
| 首抓正确，运输/放置失败 | 执行能力是主要瓶颈 | action expert RL |
| A 能做，B 连抓取都做不到 | 无法区分目标条件化问题与 B 局部技能不足 | 受控能力补齐/正确示范；不换掉 B 来找正结果 |
| Base 与 DSRL 都能按目标完成 | 指令接口可用 | E2 阶段衔接与调度 |
| DSRL 和 bounded_random 的偏置相近 | 噪声参数化可能解释部分差异 | 有限的噪声分解附加对照 |
| 多数 NONE、差异混合或区间很宽 | INCONCLUSIVE | 保留不确定性，定位具体失败，不自动扩大搜索 |

DSRL 在 B 上变差，是对现有 A-only controller 的迁移诊断，不能当作 DSRL 训练失败。因为基础 VLA 冻结，也不能把它写成基础权重遗忘。

## 8 E2 多阶段诊断

### 8.1 预先固定两个任务

| ID | 官方任务 | 完整目标 |
|---|---|---|
| T1 | KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it | Open top drawer AND In bowl top drawer |
| T2 | KITCHEN_SCENE10_put_the_butter_at_the_front_in_the_top_drawer_of_the_cabinet_and_close_it | In front butter top drawer AND Close top drawer |

T2 的顶层抽屉在初态已打开。目标对象是 butter_1，即 front butter，不是 butter_2。[S9]

任务 ID 由精确名字在当前 registry 中解析并保存，不猜编号。两个任务都保留，即使一个没有改善；不事后只报告有利任务。

### 8.2 三个条件

都使用原 Base π0、完整高斯、query=20、horizon=600、warmup=0。

| 条件 | 输入与切换 |
|---|---|
| global | 全程只有完整 goal |
| fixed_stage | 完整 goal + 局部阶段；按预先锁定时间切换 |
| oracle_stage | 完整 goal + 相同局部阶段；按当前仿真 predicate 切换 |

模板：

~~~text
global: Goal: {complete_goal}.
staged: Goal: {complete_goal}. Next: {stage_text}.
~~~

| 任务 | 阶段 0 文本 | 阶段 1 文本 | 阶段 0 完成判据 |
|---|---|---|---|
| T1 | Open the top drawer | Place the bowl inside it | 官方 Open(top_region) 连续 3 步 |
| T2 | Place the front butter inside it | Close the top drawer | 官方 In(butter_1, top_region) 且未抓持，连续 3 步 |

oracle 在每个低层步更新阶段0完整判据的连续真假计数，只在query boundary使用**过去已观察到的当前状态**修改文本。阶段0判据连续3步为真才前进，连续3步为假才回退；T2前进与回退都使用In且未抓持这一完整复合判据。完整官方goal成立时优先结束episode。不用未来轨迹、未来成功与hindsight-best选择。

JSON中的嵌套AND只是目标声明。LIBERO的_eval_predicate不是递归AST解释器；E2完整成功直接调用该任务env.check_success()，或逐个原子谓词求AND。阶段判据只向_eval_predicate传入open/in等原子表达式。

保持完整 goal 可以避免局部提示丢失最终意图。若 48 token 审计失败，先缩短预定义模板与阶段措辞，保证目标与阶段都保留，在确认集前锁定；不能只让 staged 条件获得更长上下文。

### 8.3 固定时序的校准

先完成每任务 E2-dev 的 10 个 global 分支。从其中首次通过阶段 0 判据的时间取中位数，向上取整为 20 的倍数，截断到 [100,300]。

没有通过事件时，T1 切换时刻固定为 200，T2 为 300。之后完成同一开发集的 fixed/oracle 分支。校准只用开发 global 数据；确认前固定两个时间，不在确认中重新估计。

这比随意规定一个明显很差的固定时序更能检验“是否需要读当前状态”。

### 8.4 记录什么

除完整官方成功与时间外，记录：

- 第一阶段是否完成、何时完成；
- 是否已经具备第二阶段前提；
- 首次转换到第二阶段之后是否成功；
- 重复开抽屉、提前关抽屉、选错 butter、抓取后掉落、放入但关不上、超时等事件；
- 每次切换的条件、输入文本、query index；
- 在自然衔接状态中的失败，而非只在单独干净 reset 的技能成功。

如需要局部衔接探针：取 global 开发分支中按 reset 排序、最多 10 个第一阶段有效完成前缀，每任务从相同 prefix 重放后比较 global_continue 与 local_stage_continue。没有构造出的参考也记录。最多 40 个续跑分支，≤24,000 续跑步，构造与重放另外统计。不得悄悄将机械臂回到 home；如果加入统一 reset 动作，须对所有方法一致并计入预算。

### 8.5 确认 gate 与结果解释

开发仅用于投入决策。建议 gate：

- 至少一个任务 oracle 完整成功 ≥4/10；
- 相对 global 至少 2 个净救回状态；
- 有具体阶段证据说明失败如何被改变。

满足后，两个任务都进入独立确认：40 reset × 2 sampling seeds × 3 条件，共 480 分支，≤288,000 步。

| 确认结果 | 解释 |
|---|---|
| oracle 优于 global，和 fixed 相当 | 阶段拆分可能有价值，尚未显示状态相关切换的必要性 |
| oracle 同时优于 global 与 fixed | 当前阶段状态有利用空间；下一步检验可部署估计器 |
| 三者都高 | 当前任务的调度提升空间有限；不训练复杂选择器 |
| 三者都低，局部技能也失败 | 先适配执行能力或建立语言接口 |
| 局部技能可完成，自然衔接却失败 | 状态分布、过渡动作或阶段切换是重点 |
| 结果差异小且不稳定 | INCONCLUSIVE；零样本探针不构成微调后假设的否定 |

oracle读取仿真真值，是获得额外状态信息的对照；它不保证最优调度，其分数也不是所有方法的数学上界。它的优势不能直接当作可部署方法的最终结果。

## 9 统计与继续标准

### 9.1 配对聚类统计

独立采样单位是 task 内的 reset/common state。E1 有 40 个 cluster，每个含两条指令、两个 sampling seeds 和五个策略；不是 800 个独立场景。

- 固定 bootstrap RNG=99021。
- 10,000 次 paired cluster bootstrap。
- 每次重采样 reset，完整保留同一 reset 内所有分支配对。
- E2 按任务分别抽样 reset；先逐任务，再给两任务等权宏平均。
- 记录效应量、95% 区间和配对 wins/losses。
- E1 三个 controller 分别报告；其均值称“这三个固定控制器的均值”。
- PPO 三训练 seed 分别报告成功率和差值，再报告均值、训练 seed 样本标准差；不把三个种子当充分的总体稳定性证据。
- 全部观测相同时，bootstrap 会退化；必须标记，不能把零宽区间当作一般性确定性。

预注册 E1 主要方法对比为 DSRL 三 controller 均值减 Base、减 bounded_random，围绕目标选择与对应目标成功两个主轴报告。其余逐目标、逐 seed、辅助文本比较完整展示并标明探索性。若另报 p 值，统一对主比较实施 Holm，不从一堆未校正结果中挑显著者。

### 9.2 缺失与失败

保存 planned、attempted、valid 三个分母。政策动作导致失败，计入政策失败；工程崩溃不能随意计为成功或普通任务失败。

主配对统计使用完整 cluster，另给缺失分支按最好/最坏情况填充的敏感性界。确认少于 32/40 完整 cluster 时不做主要机制判定，先修复工程或报告不足。不能按哪个策略失败来排除初态。

### 9.3 工程去留阈值

下列是预注册的资源投入阈值，不是统计显著性的替代物：

- 语言接口可用候选：Base 两目标正确首抓率均 ≥0.60，pair_both_correct ≥0.50，L ≥0.20，两个目标官方成功率均 ≥0.20。
- controller 目标偏置候选：至少两个 DSRL 在 B 指令下首抓 A 的比例相对 Base 增加 ≥0.20；同时检查 bounded_random 与对应任务成功率。
- 执行瓶颈候选：首抓正确率 ≥0.60，但对应目标成功率 <0.20。
- 其他模式：INCONCLUSIVE，明确列出仍混杂的解释。

达到阈值表示有理由投入某个后续实验；是否支持科学论点仍需看效应量、区间、失败视频、独立确认和跨任务一致性。

## 10 条件分支 SFT 与 LoRA

### 10.1 两种用途分开

**SFT-A：修复目标接口或 B 技能。**

- A/B 官方同场景演示各 50 条。
- 每目标按完整轨迹划分 40 train、10 validation；固定 split seed=99023。
- 训练按目标均衡。
- 只用轨迹真实执行目标的 global 标签。
- 不能把抓奶酪轨迹换成“抓黄油”文本，伪造语言监督。
- 行为复核用新的 3300–3319；不能重复在 E1 确认集上调参数。

**SFT-S：建立多阶段输入接口。**

- T1/T2 各最多 25 条正确演示：20 train、5 validation。
- 同一状态动作数据以 50% global、50% global+stage 输入采样；global/stage 使用同一动作监督。
- stage 标签来自该状态当时的 predicate/人工核验，不用未来成功倒推为当前阶段。
- 默认只接受后续完整 50 个真实动作都处于同一阶段、且未跨轨迹末尾的窗口；global 与 staged 变体使用完全相同的窗口。统计每个阶段的窗口数量与过滤比例。
- 当前 native SFT loss 不提供本规约所需的逐样本阶段边界 mask。因此首版不依赖该功能，也不把 padding 当真实动作。如果某阶段没有完整 50 步窗口，标记该接口训练数据不足；不能静默缩短模型 horizon 或让未来阶段动作混入标签。
- 如后续确实要利用跨阶段窗口，需另行实现 collator→loss 的有效动作 mask、按有效元素归一化，并阻止被屏蔽未来动作通过 noised action/attention 泄漏至有效监督；该扩展不属于本版 SFT-S。
- 后续 G 与 GS 使用同一个共同适配 checkpoint。

SFT-S 是共同 warmstart。单靠它无法说明 stage 标签比纯 global 数据更有价值；如果论文要回答标签价值，需要另做 equal-data 的 global-only SFT 对照。

共同SFT总上限为3,000 updates。默认按诊断只启用一个profile；若开始训练前已确定两者均需要，则按SFT-A→SFT-S顺序各最多1,500 updates，SFT-S从SFT-A选定并验证的checkpoint开始。每段500步pilot包含在各自预算内，不另加。已经耗完3,000后不得自动再开第二个3,000，需将其列为另一个研究批次。

### 10.2 具体训练设置

| 参数 | 值 |
|---|---:|
| 训练 seed | 17 |
| 模型 | native openpi、π0 |
| LoRA | rank=8，alpha=8，dropout=0 |
| target scope | all_linear；必须实际审计覆盖 |
| global batch / micro batch | 8 / 1 |
| learning rate | 1e-4 |
| warmup | 100 updates |
| gradient clip | 1.0 |
| pilot | 500 optimizer updates |
| 总上限 | 3,000 updates |
| 保存 | 每 500 updates |
| 选择 checkpoint | held-out trajectory validation loss 最低；逐目标 loss 也报告 |

保留已锁定官方模板的 Adam betas、eps、weight decay、scheduler 具体设置，全部展开写进 resolved YAML；不要只记录“default”。行为测试不用于循环搜索 checkpoint。

源码中的 all_linear 并不保证每一层都适配。必须列出 VLM attention 的 .0 与 action expert 的 .1 实际 LoRA 参数、视觉与 action projection 的覆盖、全部 trainable 参数量。部分 native Gemma MLP 不是 nn.Linear，不会自动获得 LoRA。[S12]

SFT 的 train_expert_only 并不自动触发 RL 分支的冻结逻辑。采用本 LoRA 路线时仍须验证原权重冻结。若想做 action-expert-only SFT，应显式实现冻结并测试，不仅改一个 YAML flag。

真实入口模板：

~~~bash
bash examples/sft/run_vla_sft.sh fyp_ab_shared_lora_sft
~~~

这是创建相应新 YAML 后的命令；现有仓库不自带这个 fyp 配置。

后续若改用 is_lora=False 的 action expert PPO，先完成 LoRA 合并/导出并验证同输入、同显式噪声的输出等价；不要把普通 checkpoint 格式转换当作自动合并。

## 11 RLinf 迁移与资源 profile

### 11.1 先对齐底座

必须从现有 π0_libero 转换，而非换成另一个 π0.5 或 few-shot checkpoint 后比较。

RLinf 的真实转换器示例：

~~~bash
python -m rlinf.utils.ckpt_convertor.openpi.convert \
  --mode jax_to_openpi \
  --input-model /resolved/current_pi0_checkpoint \
  --input-norm-stats /resolved/current_norm_stats.json \
  --output-model /resolved/converted_pi0 \
  --output-norm-stats /resolved/converted_pi0/physical-intelligence/libero/norm_stats.json \
  --no-pi05 \
  --action-horizon 50
~~~

以上 /resolved 路径由 Codex 从本地 manifest 解析后生成实际命令，不能原样运行。[S10]

对齐清单：

- 原权重与 norm stats hash；
- 两路图像翻转、resize、裁剪与通道顺序；
- proprio 顺序、单位与姿态表示；
- 动作反归一化、坐标系、夹爪符号；
- prediction horizon 50、execution chunk 20、denoise steps 10；
- prompt/token 长度与含义；
- reset 分布、warmup、奖励、成功、终止与 timeout。

PPO也必须实现raw_seeded_reset适配器，明确绕过官方有限pruned_init库；specific_reset_id=null本身不证明这一点。保存每个episode的reset seed、初态生成器版本、初态hash与是否调用set_init_state。训练seeds使用协议的独立区间，验证/测试命名空间不得进入训练。如果改用有限初态库，必须按库内state ID划分训练/验证/测试并另立协议，不能只用不同RNG seed宣称初态隔离。

同一数字 RNG seed 不会让 JAX/PyTorch 自动生成同一噪声。静态对齐必须显式传入同一个 z 数组。

至少用 10 个开发观测 × A/B/阶段文本 × 2 个固定 z 做动作误差报告，并记录两框架内部重复误差作为基准。原始 normalized-action RMSE >0.01 或最大绝对误差 >0.1 时触发检查；这些是工程报警线，不是允许忽略较小系统性误差的证明。低于报警线仍需查错维度、相机、夹爪等结构性问题，并做开发集闭环对齐。

若跨框架无法合理对齐，先停止跨框架 DSRL 数值归因。仍可把同一 RLinf 底座上的 G/GS 作为新的、明确标注的内部实验；不能混称原框架只有一种变量变化。

### 11.2 单卡 profile

在实际分配的 4090/A100 上：

1. 加载模型、8 env 与 rollout/actor。
2. 至少完整跨过一个 200 步 fragment 边界，确认 episode 继续。
3. 执行真实 PPO backward/update，验证梯度、参数 delta、显存。
4. 保存并在新进程恢复 checkpoint。
5. 在开发状态评估一次，验证 tokenizer、输出和终止未变化。

初始 micro batch=1，开启既有 offload。若 profile 允许，可统一改为 2、4、8；两主组保持相同设置。显存调整不改变任务、chunk、模型、reward 或总有效交互。

profile 记录峰值显存、CPU 内存、每 1,000 有效环境步耗时、一次 update 耗时与预计整轮成本。profile 失败的先修环境/资源，不把失败当成算法无效。

## 12 VLA 参数微调主实验

### 12.1 只训练两个主组

| 组别 | 输入 | 训练范围 |
|---|---|---|
| G | 完整 global goal | action expert + value head |
| GS | 相同 global goal + 当前 oracle stage | 相同 action expert + value head |

共同条件：

- 同一原始或共同 SFT checkpoint。
- 两任务 T1/T2 均衡训练。
- 相同奖励、学习率、batch、采样设置、有效环境步预算。
- 相同训练初态分布与 seed schedule。
- VLM 冻结；记录全部 trainable/frozen 参数清单与更新前后 hash。

RL 在线轨迹会因策略不同而不同。只能说“相同初态分布、数据来源与预算”，不能承诺“完全相同在线数据”。

这一步检验阶段信息是否改善参数适配。DSRL 是参照思想与旧基线，不在本实验同时训练噪声 SAC。

### 12.2 PPO 设置

从官方 libero_spatial_ppo_openpi_quickstart.yaml 复制新配置，改成指定两个 LIBERO-90 任务。

| 参数 | 值 |
|---|---|
| model_type / pi05 / is_lora | openpi / False / False |
| openpi.task | rl |
| train_expert_only | True |
| add_value_head | True |
| noise_method | flow_sde |
| detach_critic_input | True |
| action_horizon / action_chunk | 50 / 20 |
| num_action_chunks / num_steps | 20 / 10 |
| train env 数量 | 8，每任务 4 |
| episode horizon | 600 |
| rollout fragment | 200 低层步 |
| rollout_epoch | 1 |
| auto_reset / ignore_terminations | True / False |
| specific_reset_id | null；不能固定一个 trial |
| global batch / micro batch | 80 / 1 |
| PPO update_epoch | 4 |
| actor lr / value lr | 1e-6 / 1e-4 |
| gamma / GAE lambda | 0.99 / 0.95，按当前 chunk 级实现解释 |
| clip low / high / clip_ratio_c | 0.2 / 0.2 / 3.0 |
| value clip | 0.2 |
| entropy bonus / KL beta | 0 / 0 |
| gradient clip | 1.0 |
| advantage normalization | True |
| actor/rollout offload | True |
| rollout pipeline stages | 1 |

每轮名义 8×200/20=80 个 chunk 样本，global batch 80 合法。不要为凑 256 随意改变 rollout 长度。[S11]

**奖励目标**：完整官方 goal 首次达成 +1，其余 0。use_rel_reward=True、use_step_penalty=False、reward_coef=1 只是配置起点，不能单凭这三个字段认为已经实现。当前 relative reward 为相邻状态奖励之差；chunk 内若成功后继续执行并失去 goal，可能出现负奖励。Codex 必须在第一次成功处锁定 terminal，聚合前屏蔽本 episode 后续动作和奖励，或实现等价的一次性成功 wrapper；用“先成功、后被破坏”的测试验证有效奖励总和仍为 +1。两组使用同一实现，且不额外加阶段 shaping 奖励。若以后尝试 shaping，作为另一个锁定协议，不能只给 GS 使用。

**长任务语义**：

- 第 200 步 fragment 边界：bootstrap 并继续同一 episode。
- 任务成功：true terminal，不 bootstrap。
- 600 步外部 TimeLimit：truncation，使用 reset 前的 terminal observation bootstrap；评估中仍计为未完成。
- 所有 padding、done 后多余动作与 auto-reset 要正确 mask。
- 通过 episode_id、elapsed_steps、reset_reason 断言没有每 200 步偷偷 reset。
- 显式设置 algorithm.bootstrap_type=standard，并核验当前代码在 truncation 上使用 terminal observation，真实 success terminal 不 bootstrap。
- rollout buffer 保存每个 chunk 实际使用的 prompt、token IDs/mask、阶段、global goal 以及 next observation 对应阶段。actor 重算 log probability 必须使用当时保存的条件；critic/GAE 使用对应的下一状态条件，不从当前环境或回合最终阶段重新构造历史 prompt。
- 同时保存old log-prob；episode reset时重置阶段与观测历史。首次成功后的额外动作必须在奖励聚合与trajectory构造中被排除，而不仅是在预算计数中mask。PPO保留既有完整20动作chunk的联合log-prob定义，old/new一致；成功发生在第k<20步时回报仅含前k步，不通过任意删去action维度假称计算了前缀边缘概率。总env.step调用与有效前缀步数分开记录。

当前 EnvWorker 的 auto_reset=False 路径会在 fragment 起点重新 reset，不能仅设 episode horizon=600 就以为得到 600 步任务。[S11]

### 12.3 Pilot 与正式预算

**Pilot**：

- seed=17。
- G、GS 都用两个任务均衡训练；不按先验哪个任务更好而只保留一个。
- 每组 100,000 个有效底层环境步。
- 0、25k、50k、75k、100k 附近在验证集评估。
- 每次每任务 10 reset × 1 sampling seed，即 20 episodes。
- 两组 × 5 checkpoints × 20 episodes =200 验证 episodes，≤120,000 验证步。
- 每 run 最多 80 次收集迭代、24 已分配 GPU 小时。若到 cap 未达预算，PARTIAL_BUDGET。
- 约 63 次名义迭代为 100,800 步，但不能用名义值替代真实计数。

**Formal**：

- 训练 seeds=0、1、2，全部从共同底座重新开始。
- G/GS 共 6 个 runs，每 run 500,000 有效底层步。
- 每 100k 附近验证；保留最后预算 checkpoint 为主要终点评价。
- 每 run 最多 400 次收集迭代、72 已分配 GPU 小时。
- 主要终点是首次达到目标预算的完整 rollout checkpoint，允许至多 1,600 有效步 overshoot，并显示实际步数；未达到目标即触发 cap 时为 PARTIAL_BUDGET，不能标为完成 500k。
- 训练曲线横轴为有效环境步；另报 env.step 实调用数、optimizer updates、GPU 小时。
- 不用 test 挑最佳 checkpoint，不让其中一组获得额外 tuning budget。

formal gate：迁移/profile 通过，两组 pilot 完整无数值错误，出现可解释的奖励与学习/接口信号。若都没有奖励，先共同 SFT 或定位技能缺失，停止盲开 3 seeds formal。

正式验证上限：6 runs × 6 checkpoints × 20 episodes =720 episodes，≤432,000 步。共同零训练结果可缓存，但不得把重复缓存算成独立重复。

正式最终测试：6 runs × 2 tasks × 40 resets × 2 sampling seeds =960 episodes，≤576,000 步。共同底座 G/GS 两种输入在同一测试集另做零训练参照，共 320 episodes，≤192,000 步。

保留能力检查：E1 保留集 3200–3239，所有微调后模型都以同一 global A/B 文本测试，不加阶段文本；6 models × 40 resets × 2 goals × 2 sampling seeds =960 episodes，≤384,000 步。共同底座一次 160 episodes，≤64,000 步。这检查 drawer 微调后是否损害原有目标条件化；不是训练目标之一。

### 12.4 真正的 launcher 形式

RLinf run_embodiment.sh 的第二个位置参数是 ROBOT_PLATFORM，不是任意 Hydra overrides。先生成每组/seed 的完整 YAML，然后：

~~~bash
bash examples/embodiment/run_embodiment.sh fyp_global_seed0 LIBERO
bash examples/embodiment/run_embodiment.sh fyp_global_stage_seed0 LIBERO
~~~

这些 fyp YAML 是待创建文件。不要写成 launcher 后跟 actor.seed=0 并假定会生效。[S13]

runner.max_steps 是收集+更新的迭代数，不是低层环境步。需要复用 valid_action_counts/mask 实现预算 gate，并展开全部配置。

### 12.5 为什么不直接解冻原 DSRL

旧 SAC 的动作是 latent z，其 replay 中的转移依赖冻结的 VLA 解码映射。若同步改变解码器参数，旧 z 对应的动作与状态转移会变化，replay 非平稳；原 SAC actor loss 也不会自动为 VLA 提供正确的训练目标。

因此不要仅删除 freeze 就命名为 Joint RL。先用已有 PPO action-expert 路线，或者先 SFT、再固定新 VLA 重训外部 controller。若未来做真正联合优化，需要单独给出目标函数、梯度路径、replay 处理和稳定性对照。

## 13 从 oracle 变为可部署阶段输入

仅在阶段条件有可解释收益后启用。

### 13.1 阶段估计器

输入只包含当前与前 2 个 query 的两路 RGB 特征、proprio、已知 global goal。禁止 simulator object pose、官方 predicate、未来帧进入推理。

采用冻结当前 VLA 视觉 encoder 的每视角 token 均值池化特征；记录实际张量来源与维度。如果实现只能抽取含语言的 prefix 特征，必须用仅含 global goal 的文本重新编码；训练、验证、部署都不能把 oracle-stage token 或当前真值阶段送进 encoder。不能直接缓存 GS rollout 中已经混入 oracle stage 的 prefix 来训练阶段估计器。

轻量 MLP：

- hidden=[256,256]，GELU，dropout=0.1；
- Adam learning rate=1e-3，weight decay=1e-4；
- batch=256；
- 最多 20 epochs 或 10,000 updates，先到即止；
- 按完整训练 episode 划分 80%/20% train/validation；
- class balance，训练 seed=17；
- 用 validation macro-F1 选择 checkpoint。

训练标签可以由训练轨迹的 oracle 提供；验证和最终测试轨迹不能回流。每个 GS 训练 seed 配一个估计器，数据量逐任务记录。

每 20 步预测一次；连续两次 argmax 为同一新阶段才切换，允许回退。初始阶段按任务定义设定。记录标签延迟、抖动、预测准确率以及 rollout 成功，不能用分类准确率代替任务收益。

### 13.2 端到端评测

固定三个条件：

- G-global；
- GS-oracle；
- GS-predicted。

3 training seeds × 2 tasks × 40 new resets × 2 sampling seeds × 3 conditions =1,440 分支，≤864,000 步。

若 oracle 优势明显而 predicted 消失，说明阶段估计/观测鲁棒性是当前瓶颈；不能宣称已得到可部署提升。

可做少量错误阶段/打乱阶段评估定位依赖，但这是失配诊断。只有要主张“自然语言表示本身优于其他阶段表示”时，才新增 equal-budget 的 stage-ID 训练组：同样 3 seeds × 500k，ID 也必须经过同样适配，不能拿陌生零样本 ID 当公平对照。

## 14 条件恢复实验

旧 C2 只有四个有效参考且出现天花板，因此不直接扩大旧 C2 来追求正结果。新恢复在指定 T1/T2 中使用预注册物理事件，作为阶段方法的压力测试。

### 14.1 共同参考构造

- 用共同训练前 checkpoint + global 提示作为固定 reference policy。
- 每任务 reset 7000–7009 开发，7100–7139 确认。
- 最多 300 步寻找目标物体稳定抓取：连续 3 步 grasp，抬升 ≥4 cm。
- 保持固定世界 EE 目标稳定 20–60 步。
- 同一参考创建两个 20 步分支：sham 保持夹爪闭合；drop 物理张开夹爪。
- 不传送物体；从已有 recovery_probe 的物理释放逻辑改造。
- drop 条件验证落差 ≥2 cm、桌面支撑接触、可达、无严重穿透、水平漂移 ≤8 cm；sham 条件验证仍稳定抓持、未发生真实掉落。不要用drop的落地要求筛掉所有sham。
- 所有阈值先在开发构造中固定，不依据 G/GS 恢复成败调难度。

同一参考的动作前缀与物理条件重放给全部方法。600 总步包括参考前缀、稳定、释放与恢复；剩余预算不足则按同一规则处理。

主配对恢复分析只纳入sham与drop都通过各自物理验证的共同reference；按reference记录任一条件无效的原因，不根据后续策略恢复成败决定配对有效性。

### 14.2 分母与评测

分别报告：

1. 参考 trigger coverage；
2. 物理构造 valid coverage；
3. 有效共同状态上的条件恢复率；
4. 如果另做各策略自身轨迹触发的端到端实验，单独报告其完整成功率。

未触发不算已经失败的“条件恢复回合”，也不能从覆盖率消失。共同参考可能选择了特定状态分布，不能外推整个任务恢复能力。

确认最多：

2 tasks × 40 references × 2 physical conditions × 2 methods × 3 training seeds × 2 sampling seeds =1,920 续跑分支。

方法为 G-global 与 GS-predicted。若可部署阶段估计尚未完成，只能以 oracle 做诊断并明确命名，不能替代最终方法。按 reset/reference 聚类；保存所有无效构造与未触发原因。

## 15 必须实现的代码与接口

### 15.1 复用与新增

| 文件 | 任务 |
|---|---|
| tools/control_probe.py | 复用模型加载、输入处理、精确重放；不直接当完整双目标 runner |
| tools/outcome_probe.py | 复用 SAC controller 恢复与噪声构造；避免其40步切回原指令和原goal逻辑 |
| 新 tools/language_goal_probe.py | E1 全回合 prompt、独立 A/B evaluator、fixed horizon、全部对象事件 |
| 新 tools/stage_probe.py | E2 global/fixed/oracle、阶段事件与完整目标 |
| 新 tools/research_runner.py | 验证协议、展开case、阶段依赖、预算与断点续跑 |
| 新 tools/research_stats.py | 配对cluster统计、缺失敏感性界、表格 |
| 新 configs/research_protocol_v1.json | 从交付协议生成的已解析运行配置 |
| tests 或现有测试目录 | 第6节的科学有效性测试 |
| tools/job.sh、remote.ps1、remote_jobs.ps1 | 将新增模式接入已存在的smoke、commit与config hash检查流程 |

不要为了新增 E1 修改旧报告结果含义。既有 baseline configs 与 runs 留存。submodule 修改如果确有必要，要单独记录 diff；当前 E1 fixed-horizon 方案不需要 monkeypatch LIBERO 的 success。

### 15.2 待实现 CLI 合约

以下是**要求 Codex 实现的目标接口**，不是声称仓库已有这些命令：

~~~bash
python -m tools.research_runner --protocol configs/research_protocol_v1.json --mode validate
python -m tools.research_runner --protocol configs/research_protocol_v1.json --mode plan
python -m tools.research_runner --protocol configs/research_protocol_v1.json --phase e0 --output RUN_DIR
python -m tools.research_runner --protocol configs/research_protocol_v1.json --preset first_batch --output RUN_DIR --resume
python -m tools.research_stats --input RUN_DIR --protocol configs/research_protocol_v1.json
~~~

plan 必须只展开 manifests、统计矩阵与预算，不运行模型。first_batch 的顺序为 E0→E1 dev/aux→E2 global dev→固定时序校准→E2 fixed/oracle dev→锁定整个协议→E1 confirm。可以复用相同 GPU job，但要保存阶段边界；确认开始后不再修改已锁定的E2字段。

所有长运行在 Slurm job 内执行，不能让上述例子绕过既有资源提交入口。

remote.ps1 的 ConfigPath 是机器/SSH/Slurm 配置，ExperimentConfig 才是实验 JSON。现有模式中的 SmokeJobId、commit/config hash gate 应扩展到新模式，不伪造成功 smoke ID。

### 15.3 每个 case 的最小记录

~~~json
{
  "case_id": "immutable ID derived from protocol and factors",
  "protocol_hash": "...",
  "task_name": "...",
  "reset_seed": 3100,
  "sampling_seed": 0,
  "controller_training_seed": null,
  "policy_id": "base",
  "prompt_id": "B",
  "requested_goal_id": "B",
  "common_state_id": "...",
  "first_grasp_class": "B",
  "first_grasp_step": 54,
  "active_goal_ever_success": true,
  "goal_A_ever": false,
  "goal_B_ever": true,
  "goal_A_first_step": null,
  "goal_B_first_step": 172,
  "goal_A_final": false,
  "goal_B_final": true,
  "executed_env_steps": 400,
  "vla_calls": 20,
  "termination_reason": "fixed_horizon",
  "status": "COMPLETE"
}
~~~

这是 schema 示例，不是真实结果。实际另存动作数组、query noise/key、逐步 predicate/grasp、prompt schedule、final physics hash、工程错误及重试记录。缺失首次事件用 null，不用 0。

状态分三层：execution_status表示COMPLETE、PARTIAL_BUDGET、BLOCKED_ENVIRONMENT、BLOCKED_DATA或ENGINEERING_ERROR；case_status区分有效执行、政策失败、无效构造和工程异常；scientific_decision记录接口可用、执行瓶颈、阶段收益候选或INCONCLUSIVE。完整跑完但结论不确定可同时是execution_status=COMPLETE与scientific_decision=INCONCLUSIVE。

### 15.4 必须交付的运行文件

- resolved_protocol.json、protocol.sha256；
- software_manifest.json、checkpoint_manifest.json、trainable_manifest.json；
- prompt_audit.json、run_manifest.json；
- common_states/、actions/、episodes.jsonl、events.jsonl；
- summary_by_condition.csv、paired_cluster_statistics.json；
- budget.json：训练/评估/构造/重放/重试、有效步/总step调用、VLA调用、GPU时间分开；
- decision.json：gate数值、支持的解释、仍混杂的解释、建议下一模块；
- REPORT.md：完整结果表、负结果、区间、失败类别、实现差异与实际限制。

保存全部数值结果。视频采用事先固定的前2个reset，以及按case_id排序的首个“赢/输/均失败”代表，不人工挑最漂亮案例。若补渲染，只重放已存动作，不重新采样政策。

## 16 结果报告必须回答的问题

报告开头按以下顺序写结论，数据不足可以明确回答“不确定”：

1. 模型是否在完全相同场景中按 A/B 改变目标？
2. 已有 DSRL 是否改变这种目标控制，随机噪声能否解释变化？
3. 失败主要发生在选目标、执行、还是阶段衔接？
4. global、fixed、oracle 的差异支持哪一种机制？
5. 若微调，收益来自训练本身，还是阶段信息在相同预算下的额外价值？
6. oracle 收益能否由可部署观测获得？
7. 哪些结论只适用于当前两个/三个任务，哪些仍缺泛化证据？

不得使用以下推断：

- 动作 RMSE 变了 → 一定理解了指令；
- 空指令也成功 → 一定记住任务；
- Joint/GS 优于 Base → 一定优于 G 或 DSRL；
- oracle 成功 → 已有可部署控制器；
- 训练 seed 共3个、分支很多 → 训练稳定性已经充分证明；
- 当前两个任务有效 → 整个 LIBERO 或实机有效；
- 零样本阶段文本无收益 → 微调后阶段接口必然无收益。

## 17 Codex 验收标准

首批达到以下条件才写 COMPLETE；政策失败是一个有效完成的实验结果，工程缺失不是：

- 代码与已解析配置已经保存，可复现实验命令。
- E0 核验通过；确认集前的协议、token与checkpoint记录完整。
- 首批全部计划 cases 完成并留存结果。若存在未完成矩阵，仍交付阶段报告与精确计数，但总状态必须为 PARTIAL_BUDGET、BLOCKED 或 ENGINEERING_ERROR，不能写 COMPLETE。
- 所有 A/B 分支全程保持对应文本；目标评分正确，native_done 未造成错停。
- 统计按 reset 聚类；控制器 seed、sampling seed、reset seed 分开。
- 输出成功率与首抓对象两个主轴、负结果和缺失分母。
- GPU/step预算与checkpoint没有虚报。
- REPORT.md 和 decision.json 能决定下一批属于目标接口、执行适配、还是阶段调度。

源码审计与计划文件已经完成，不等于这些运行验收已经通过。Codex 必须填入实际运行证据。

## 18 来源

事实来源与本规约建议设置分开：所有 seeds、预算、阈值、额外指标、模型比较与新增 CLI 均是本规约的设计；引用用于支持现有实现、已完成结果与官方任务/接口事实。

[S1]: https://app.notion.com/p/DSRL-LIBERO-2026-10-21-400d26c2767d810a875ede5797067b5b
[S2]: https://arxiv.org/html/2506.15799v2
[S3]: https://github.com/MoliaiELS/dsrl_pi0/blob/4774febfa18cc64f3c28e6f39d63bee28676a8fb/AGENTS.md
[S4]: https://github.com/MoliaiELS/dsrl_pi0/blob/4774febfa18cc64f3c28e6f39d63bee28676a8fb/tools/outcome_probe.py
[S5]: https://github.com/nakamotoo/openpi/blob/a6d2400d2534ce32e7bdf8747709b97aaef8ec04/src/openpi/models/pi0.py
[S6]: https://github.com/Lifelong-Robot-Learning/LIBERO/blob/87edbd1b823e464bc66557d213cc45e9f88aa543/libero/libero/bddl_files/libero_90/LIVING_ROOM_SCENE3_pick_up_the_butter_and_put_it_in_the_tray.bddl
[S7]: https://github.com/nakamotoo/LIBERO/blob/87edbd1b823e464bc66557d213cc45e9f88aa543/libero/libero/envs/bddl_base_domain.py
[S8]: https://github.com/RLinf/RLinf/blob/fab7f407fbc611085eb163962e3c89c0ab424c1a/rlinf/models/embodiment/openpi/__init__.py
[S9]: https://github.com/Lifelong-Robot-Learning/LIBERO/blob/87edbd1b823e464bc66557d213cc45e9f88aa543/libero/libero/bddl_files/libero_90/KITCHEN_SCENE10_put_the_butter_at_the_front_in_the_top_drawer_of_the_cabinet_and_close_it.bddl
[S10]: https://github.com/RLinf/RLinf/blob/fab7f407fbc611085eb163962e3c89c0ab424c1a/rlinf/utils/ckpt_convertor/openpi/README.md
[S11]: https://github.com/RLinf/RLinf/blob/fab7f407fbc611085eb163962e3c89c0ab424c1a/rlinf/workers/env/env_worker.py
[S12]: https://github.com/RLinf/RLinf/blob/fab7f407fbc611085eb163962e3c89c0ab424c1a/rlinf/models/__init__.py
[S13]: https://github.com/RLinf/RLinf/blob/fab7f407fbc611085eb163962e3c89c0ab424c1a/examples/embodiment/run_embodiment.sh

其他可核对入口：

- [官方 A 目标 BDDL](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/87edbd1b823e464bc66557d213cc45e9f88aa543/libero/libero/bddl_files/libero_90/LIVING_ROOM_SCENE3_pick_up_the_cream_cheese_and_put_it_in_the_tray.bddl)
- [官方 T1 BDDL](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/87edbd1b823e464bc66557d213cc45e9f88aa543/libero/libero/bddl_files/libero_90/KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it.bddl)
- [已有共同状态探针](https://github.com/MoliaiELS/dsrl_pi0/blob/4774febfa18cc64f3c28e6f39d63bee28676a8fb/tools/control_probe.py)
- [已有确认配置](https://github.com/MoliaiELS/dsrl_pi0/blob/4774febfa18cc64f3c28e6f39d63bee28676a8fb/configs/libero_confirm_seed0.json)
- [RLinf PPO 配置](https://github.com/RLinf/RLinf/blob/fab7f407fbc611085eb163962e3c89c0ab424c1a/examples/embodiment/config/libero_spatial_ppo_openpi_quickstart.yaml)
- [RLinf LiberoEnv](https://github.com/RLinf/RLinf/blob/fab7f407fbc611085eb163962e3c89c0ab424c1a/rlinf/envs/sim/libero/libero_env.py)
- [RLinf SFT 配置](https://github.com/RLinf/RLinf/blob/fab7f407fbc611085eb163962e3c89c0ab424c1a/examples/sft/config/libero_sft_openpi.yaml)
- [RLinf π0 说明](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/pi0.html)

