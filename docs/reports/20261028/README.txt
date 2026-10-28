本地实验证据报告（2026-10-28）

用 Chrome / Edge 打开 report.html，可离线阅读并播放9段真实仿真视频。
移动或分享时复制整个文件夹，或解压本目录的 evidence_report_20261028.zip。
media/ 是本次整理复用的历史视频，不含新推理；data/ 是小型来源副本和聚合结果。
figures/ 的图片与对应实验文字穿插展示。没有下载模型或原始演示数据集。

SFT截止快照：2026-10-28 13:25:31 CST，训练job18658、累计270更新。
这是报告截止状态，不是实时仪表盘。SFT闭环效果尚未完成评估。
训练及自动终评交接由远端独立Slurm/CPU工作流继续运行。

实时查询（在项目根目录PowerShell执行）：
powershell -ExecutionPolicy Bypass -File tools/remote.ps1 status 18658

source_manifest.json 记录实际读取来源、SHA256与52项内容/引用检查。
报告生成器 tools/reports/build_evidence_report_20261028.py 只读本地缓存，不访问远端，
在项目已有 tmp/experiment-report-20261021/.venv Python环境下可重建。
report_body.html 是带占位符的作者源文件，阅读请打开已展开的 report.html。
