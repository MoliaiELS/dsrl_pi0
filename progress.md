# DSRL π₀ / LIBERO 基线进度

更新时间：2026-10-19 12:19（Asia/Shanghai）。本文件记录事实状态；提交成功不等于运行成功。

## 当前轮次（2026-10-19 11:55 启动）

- `18102`：seed 0 最终控制器独立评估已完成，COMPLETED/0、SUCCESS；新进程从 `runs/job-18046` 恢复 checkpoint500000，100 回合成功 99 次（99%），15,580 环境步、802 chunk，平均回合长度 155.8，耗时 10:34。与 Base 35/100 相比，观测成功率提高 64 个百分点；目前仍仅 task 57、seed 0，未验证多任务／多 seed／受扰恢复。
- `18103`：seed 1 正式 DSRL-SAC，使用同一冻结 VLA、配置、代码和 500,000 次更新预算；SAC 从头初始化，不从 seed 0 续训。通过同 commit 的短测 18044／Base 18045 检查；账户作业数限制排队后，约 12:05 在 4090node2 开始运行，Slurm 分配 CUDA_VISIBLE_DEVICES=2（未硬编码）。
- 12:19 早期运行检查通过：已核对 500／1000／1500／2000 次更新的 actor loss、critic loss、temperature 均有限，temperature 为正；所有已记录训练标量未出现 NaN／Inf。快照为 2,040 次更新、12,171 训练环境步、612 chunk、33 回合；持续更新与采集均正常。检查证据 `runs/job-18103/startup_check.json`，这不是收敛或最终性能结论。
- seed 1 warmup 为 10,067 环境步、505 chunk、27 回合；训练内更新前评估另计 10 回合、3,753 环境步、188 chunk，成功 2/10。训练／评估交互预算分别记录。
- seed 1 最近采样显存 9,221 MiB（9.0 GiB），进程主存约 8.0 GiB，启动 RSS 峰值 22,483 MiB（22.0 GiB）；单张 4090、8 CPU，未见资源异常增长。后续由 Slurm 后台继续运行，本轮不持续监控完整训练。
- 两个作业各申请单张 4090、8 CPU，由 Slurm 分配 GPU；独立输出目录 `runs/job-18102`／`runs/job-18103`。代码与依赖没有变化，复用已验证环境与模型缓存，不重新安装。
- 本轮只推进 seed 1 训练；seed 2 留待检查 seed 1 后再启动。
- 评估 18102 的日志字段 gradient_updates=1 是原代码选择学习后控制器的模式标记，不代表评估期间执行了一次更新；实际恢复的是第 500,000 次更新后的控制器。评估不采集训练 replay、不更新参数。

## seed 0 训练结束记录（2026-10-19 11:45；历史）

- 作业 `18046` 正常完成，Slurm COMPLETED、ExitCode 0:0、SUCCESS.json 存在；2026-10-18 16:26 启动，2026-10-19 00:55 完成，调度器耗时 8:28:46。
- 完成 **500,000 次梯度更新**；训练交互 **496,291 环境步、25,504 chunk、2,663 回合**；训练内评估另有 98,535 环境步、5,062 chunk、510 回合，未混入训练计数。
- Base VLA 独立评估为 **35/100（35%）**；最终更新 500,000 处的训练内评估为 **10/10（100%）**，平均回合长度 144.3。两者评估样本量和 reset 序列不同，不能当作已完成的等样本配对比较。
- 50,000 更新处首次取得 10/10；400,000–500,000 更新的 11 次评估合计 105/110（95.45%），来自多个不同控制器版本，不能作为最终 checkpoint 的独立 110 回合成绩。整体显示明显学习改善，但目前仅 task 57、seed 0。
- 已记录的训练标量未发现 NaN／Inf；末次 actor loss 3.86460、critic loss 0.019421、temperature 0.0071692。损失只作数值健康检查，效果以任务成功率评估为准。
- 10 个 checkpoint 均有非空数据；最终 `runs/job-18046/checkpoint500000` 含 10 个文件、34,353,853 字节，含 OCDBT manifest 和 checkpoint metadata。该简报阶段尚未重新加载终评；本轮 18102 已验证恢复并完成 100 回合，见本文顶部。
- 资源：单张 RTX 4090、8 CPU，1007 次采样；峰值显存 9,221 MiB（9.0 GiB），主存 RSS 峰值 25,057 MiB（24.5 GiB）、中位约 7.4 GiB。整卡 GPU 利用率采样均值约 14.9%、中位 0%、峰值 100%；交互／更新交替造成利用率波动，不等同于持续满卡计算。
- 汇总与曲线：远端 `runs/job-18046/result_summary.json`、`learning_curve.png`；本地副本在忽略目录 `tmp/results-18046/`。原始指标、资源和视频仍留在原运行目录。
- 修复 `status` 对已从活动队列移除作业的查询：squeue 失败后继续读取 sacct 与持久化结果；对 18046 的完成状态查询已验证成功。
- 上一轮简报时未启动新实验；本轮已提交最终 checkpoint 独立评估 18102 与 seed 1 训练 18103，具体状态见本文顶部。

## 当前目标

- 固定 LIBERO-90 task 57：拿起奶油奶酪并放入托盘。
- 冻结 `s3://openpi-assets/checkpoints/pi0_libero`；训练官方 DSRL-SAC 潜噪声控制器。
- seed 0 的完整短测试、Base 评估、500,000 次训练更新与独立终评均已完成；当前推进同配置的 seed 1 复验。
- seed 1、2 在 seed 0 的运行与结果检查后推进；本阶段不加入 Instruction RL、Joint RL 或扰动训练。

## 已验证的基础

- 环境修复提交：`fba905524dfba5bc05f6343dfe539a7c251bc46d`。
- OpenPI：`a6d2400d2534ce32e7bdf8747709b97aaef8ec04`；LIBERO：`87edbd1b823e464bc66557d213cc45e9f88aa543`。
- Python 3.11.11、JAX 0.5.0、TensorFlow 2.19.0、protobuf 5.29.6、Torch 2.6.0+cpu。
- 安装与导入、GPU 枚举、EGL/GL 加载均通过；模型加载、仿真动作、参数更新、保存恢复由 18044 短测与正式运行验证，最终控制器恢复由 18102 独立终评验证。
- 本地已有的 `bash.sh` 未提交改动保留。

## 本轮工作

| 工作 | 状态 | 证据／待完成事项 |
| --- | --- | --- |
| 核对远端与 Slurm | 已检查 | 分区 GPU，账户 iadc-acco，QOS phd；单卡 4090，长作业排除登录节点 |
| LIBERO 运行版本锁 | 已验证渲染 | MuJoCo 3.3.1、gym-aloha 0.1.5、dm-control 1.0.15；pip check 通过；job 18041 在 4090node2 渲染通过 |
| 权重与 tokenizer 缓存 | 已验证 | tokenizer 官方 MD5/SHA256 已核对；checkpoint 19 文件、12,014,131,888 字节，全部通过原 S3 ETag／multipart ETag；清单保存在模型目录 download_manifest.json |
| 固定入口与本地指标／视频 | 已实现 | sync、env、smoke、eval、train、status、logs；正式提交要求短测和 Base 作业成功 |
| 模型／动作／更新／保存恢复短测试 | 已验证 18044 | COMPLETED/0，batch 256、16 次更新；参数变化 L2=0.571960；冻结 VLA 输出不变、独立控制器保存恢复一致 |
| Base VLA 评估 | 已完成 18045 | COMPLETED/0；35/100 成功，成功率 35%；34,786 环境步、1,755 chunk；不更新参数、不写入训练 replay |
| seed 0 正式训练 | 已完成 18046 | COMPLETED/0，500,000 更新；496,291 训练环境步；最终训练内评估 10/10；checkpoint500000 已保存 |
| seed 0 最终控制器独立评估 | 已完成 18102 | checkpoint500000 新进程恢复；99/100 成功（99%）；对照 Base 35/100，观测增益 +64 pp |
| seed 1 正式训练 | 运行中 18103，早期检查通过 | 已确认至少 2,000 次更新；4 个日志检查点损失有限；目标 500,000 更新 |

阶段区别：`18044` 是已通过的 16 次更新短测试；`18045` 是正式训练前的 Base 评估；后续正式训练才会先采集 warmup replay、再交替采集与更新 SAC 噪声控制器。三个阶段都使用冻结 VLA，只有短测试和正式训练更新 SAC。

## 固定协议与资源

- 实际运行 commit：`c227be8c48353115b7c5de0b3e99fb17d388462f`，独立 worktree；后续文档提交不改变已提交作业代码。
- SAC：batch 256，UTD 20，warmup 500 个 chunk（原实现要求 buffer 长度 > 500）；每个 chunk 最多执行 20 个环境步，每回合最多 400 步。
- 训练预算：500,000 次梯度更新，每 10,000 次更新评估 10 回合，每 50,000 次保存。原始 Base 先独立评估 100 回合；最终控制器另作 100 回合评估。
- 分别记录梯度更新、训练 chunk／环境步和评估 chunk／环境步；warmup 交互计入训练交互。保留原实现的初始采集噪声和 Base 评估噪声差异。
- Slurm：GPU 分区，iadc-acco/phd，单张 RTX 4090、8 CPU，排除 4090node1；正式训练时间上限 24 小时。集群未配置可调度内存容量，省略内存资源请求，以进程 RSS 实测记录占用。
- `resources.jsonl` 每 30 秒记录当前 Python 进程 CPU/RSS、GPU 显存及所在 GPU 的利用率／功耗／温度；GPU 利用率是整卡采样，不能当作进程独占指标。
- 官方 checkpoint 19 文件共 12,014,131,888 字节；本地完整下载已通过 GCS MD5 与原 S3 ETag 来源检查，远端继续核对原 S3 multipart ETag。tokenizer SHA256：`8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6`。

## 运行记录

- `18041`：渲染探针，commit `b61d4ac`，4090node2，36 秒，COMPLETED/0；结果 `runs/job-18041/environment.png`。
- `18042`：渲染 + SAC 更新／恢复探针，commit `92da8e3`，4090node2，FAILED/1。3 次更新均有限，恢复测试发现原保存前缀 `checkpoint` 与恢复默认前缀 `checkpoint_` 不一致；修复后复测，不把该失败称为训练成功。
- `18043`：修复后的渲染 + SAC 探针，commit `cb0a06c`，4090node2，76 秒，COMPLETED/0；3 次更新有限，actor 参数变化 L2=0.151459，独立初始化控制器恢复后输出一致。
- `18044`：完整 VLA 集成短测，commit `c227be8`，4090node2，8 分 28 秒，COMPLETED/0；16 次更新均有限，参数变化 L2=0.571960，冻结 VLA 输出不变，保存恢复后输出一致。日志 `logs/slurm/18044.log`，结果 `runs/job-18044`。40 步短回合仅用于流程检查，其成功率不作为基线成绩。
- `18045`：Base VLA 100 回合评估，commit `c227be8`、seed 0，4090node2，22 分 51 秒，COMPLETED/0；35/100 成功（35%），34,786 环境步、1,755 chunk，平均回合长度 347.86。结果 `runs/job-18045`，视频 `media/0-eval_video_*.mp4`。这是当前 task 57 单 seed 的 Base 成绩，不是完整论文复现结果。
- `18046`：正式 DSRL-SAC，commit `c227be8`、seed 0，2026-10-18 16:26 CST 提交；已通过 18044／18045 同 commit 成功检查。日志 `logs/slurm/18046.log`，结果 `runs/job-18046`。
- `18046` 启动检查历史（10-18 约 16:37）：当时 Slurm RUNNING，已记录第 500 次更新；actor loss=-15.1141、critic loss=0.00090554、temperature=0.86494，均有限。训练交互 10,800 环境步、540 chunk、27 回合；其中 warmup 为 10,400 步、520 chunk、26 回合（严格 >500 才更新）。最终结果见本文顶部。
- 训练内更新前评估：2/10 成功、3,760 环境步、189 chunk；与独立 Base 100 回合的 reset 随机状态不同，不能据此称性能下降。最终控制器的独立 100 回合对照现已完成，结果 99/100。
- `18046` 启动资源采样历史：单卡 RTX 4090，分配 8 CPU；显存 9,221 MiB（约 9.0 GiB），当时进程 RSS 约 7.0 GiB、峰值约 24.5 GiB。完整资源汇总见本文顶部；连续记录在 resources.jsonl。
- 完整短测采样峰值：GPU 9,219 MiB，主存 RSS 24,701.6 MiB；编译与加载阶段占大部分耗时，不能直接据此推算训练吞吐。
- 完整短测保留正式 batch size 256，只缩短 episode／更新预算；修复短测初始 replay 容量为 0 的边界。作业记录权重校验清单，并分别累计训练／评估的环境步数与 chunk 数。
- `doctor` 已确认依赖与所有模块导入通过；源码检查改为核对项目环境脚本选定的固定工作目录。共享 editable 安装仍指向安装工作目录，作业通过 PYTHONPATH 显式绑定当前固定目录，避免修改运行中作业。
- 新增日志与执行包装；保留原 SAC 损失和模型采样逻辑。修复更新预算越界和末步重复保存，增加显式最终保存、RNG 存储、参数变化／冻结 VLA 输出检查。
- 当前 checkpoint 仅支持控制器权重、优化器与控制器 RNG 恢复；尚不支持 replay buffer／环境随机状态的完整续训。

每次运行应记录：commit、子模块、完整配置、seed、模型清单、依赖、Slurm job ID、节点、结果目录、GPU/CPU/内存采样、训练与评估指标。

运行目录使用固定 commit，不在运行中同步代码。大文件位于 `/mnt/slurmfs-4090node3/user_data/yguo704/dsrl_pi0`。

## 下次对话

先读取本文件，再使用 `tools/remote.ps1 status 18103` / `logs 18103`，以及 `status 18102` 查看 seed 0 独立评估；不重复安装未变化的环境，不重启仍在运行的作业。

当前训练目录：`/mnt/slurmfs-4090node3/user_data/yguo704/dsrl_pi0/runs/job-18103`。
seed 0 训练目录 `runs/job-18046`；独立评估目录 `runs/job-18102`。18102 汇总在 result_summary.json；唯一失败为从 0 编号的 rollout 80，视频为 media/1-eval_video_80.mp4。
seed 0 完成状态、SUCCESS、500,000 更新和最终 checkpoint 已核对；独立终评 18102 已完成，99/100 成功。seed 1 训练 18103 已通过早期运行检查；后续先检查 18103 的最终状态／指标／checkpoint，再推进 seed 1 最终评估和 seed 2，不重复启动已有作业。

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File tools/remote.ps1 status 18103
powershell.exe -NoProfile -ExecutionPolicy Bypass -File tools/remote.ps1 logs 18103
powershell.exe -NoProfile -ExecutionPolicy Bypass -File tools/remote.ps1 status 18102
```

训练每 50,000 次更新保存 checkpoint，时限 24 小时。会话／SSH 断开不影响 Slurm 运行。若失败或超时，先读 FAILED.json 和日志；当前没有 replay／模拟器状态的完整续训能力，不应直接宣称可无损恢复训练。
