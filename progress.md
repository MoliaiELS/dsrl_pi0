# DSRL π₀ / LIBERO 基线进度

更新时间：2026-10-18（Asia/Shanghai）。本文件记录事实状态；提交成功不等于运行成功。

## 当前目标

- 固定 LIBERO-90 task 57：拿起奶油奶酪并放入托盘。
- 冻结 `s3://openpi-assets/checkpoints/pi0_libero`；训练官方 DSRL-SAC 潜噪声控制器。
- 先完成短测试和 Base VLA 评估，再提交 seed 0、500,000 次梯度更新的正式训练。
- seed 1、2 在 seed 0 的运行与结果检查后推进；本阶段不加入 Instruction RL、Joint RL 或扰动训练。

## 已验证的基础

- 环境修复提交：`fba905524dfba5bc05f6343dfe539a7c251bc46d`。
- OpenPI：`a6d2400d2534ce32e7bdf8747709b97aaef8ec04`；LIBERO：`87edbd1b823e464bc66557d213cc45e9f88aa543`。
- Python 3.11.11、JAX 0.5.0、TensorFlow 2.19.0、protobuf 5.29.6、Torch 2.6.0+cpu。
- 安装与导入、GPU 枚举、EGL/GL 动态库加载已通过；截至本轮开始尚未验证模型加载、仿真动作、参数更新、保存恢复。
- 本地已有的 `bash.sh` 未提交改动保留。

## 本轮工作

| 工作 | 状态 | 证据／待完成事项 |
| --- | --- | --- |
| 核对远端与 Slurm | 已检查 | 分区 GPU，账户 iadc-acco，QOS phd；单卡 4090，长作业排除登录节点 |
| LIBERO 运行版本锁 | 进行中 | 原安装 MuJoCo 2.3.7，官方 LIBERO 入口使用 3.3.1，需受控验证 |
| 权重与 tokenizer 缓存 | 待完成 | 只下载官方来源，记录文件大小与 SHA256 |
| 固定入口与本地指标／视频 | 进行中 | sync、smoke、eval、train、status、logs |
| 模型／动作／更新／保存恢复短测试 | 待完成 | 必须通过后才正式训练 |
| Base VLA 评估 | 待完成 | 原始高斯先验，固定配置与 seed |
| seed 0 正式训练 | 未提交 | 500,000 梯度更新；chunk 和环境步数另计 |

## 运行记录

每次运行应记录：commit、子模块、完整配置、seed、模型清单、依赖、Slurm job ID、节点、结果目录、GPU/CPU/内存采样、训练与评估指标。

运行目录使用固定 commit，不在运行中同步代码。大文件位于 `/mnt/slurmfs-4090node3/user_data/yguo704/dsrl_pi0`。

## 下次对话

先读取本文件，再使用 `tools/remote.ps1 status <job_id>` / `logs <job_id>`；不重复安装未变化的环境，不重启仍在运行的作业。
