"""Local-only training completion note; deliberately reads no behavior outcomes."""
import csv
from datetime import datetime,timezone,timedelta
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'tmp/language-stage-20261027/sft_a_completion_20261028'
OUT=ROOT/'docs/reports/20261028/sft-a-18658'
TZ=timezone(timedelta(hours=8))


def main():
    bundle=json.loads((SOURCE/'snapshot_bundle.json').read_text(encoding='utf-8-sig'))
    outdata=OUT/'data';outdata.mkdir(parents=True,exist_ok=True)
    sources={}
    for name,item in bundle['files'].items():
        raw=item['text'].encode('utf-8')
        assert hashlib.sha256(raw).hexdigest()==item['sha256']
        target=outdata/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        sources[name]=dict(sha256=item['sha256'],bytes=len(raw))
    load=lambda name:json.loads(bundle['files'][name]['text'])
    done=load('train/SFT_TRAINING_COMPLETE.json');audit=load('train/checkpoint_audit/checkpoint_audit.json')
    assert done['passed'] and done['updates']==3000 and audit['passed'] and audit['checked_checkpoints']==8
    rows=audit['checkpoints'];selected=next(r for r in rows if r['step']==done['selected']['step'])
    assert selected['step']==500 and selected['sha256']=='7747e4c31aff07f887034c64e1bacb855f9fe592b28f2e66afa116890d6db224'
    assert min(rows,key=lambda r:r['validation']['loss'])['step']==500
    metrics=[json.loads(s) for s in bundle['files']['train/metrics.jsonl']['text'].splitlines() if s.strip()]
    train=[r for r in metrics if r.get('phase')=='sft_training']
    assert [r['gradient_updates'] for r in train]==list(range(3,3001))
    resources=[json.loads(s) for s in bundle['files']['train/resources.jsonl']['text'].splitlines() if s.strip()]
    assert resources and load('evaluation/SCHEDULER_RESOURCE_OVERRIDE.json')['scientific_config_unchanged']
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
        'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'font.size':10})
    def save(fig,name):
        fig.savefig(OUT/name,dpi=165,bbox_inches='tight',facecolor='white');plt.close(fig)
    steps=np.array([r['step'] for r in rows])
    fig,axes=plt.subplots(1,2,figsize=(12,4.4))
    values=np.array([r['sft/train_loss'] for r in train]);updates=np.array([r['gradient_updates'] for r in train])
    axes[0].plot(updates,values,color='#bad9d4',lw=.3,label='逐更新训练loss')
    axes[0].plot(updates[49:],np.convolve(values,np.ones(50)/50,mode='valid'),color='#178f87',lw=1.7,label='训练50更新均值')
    axes[0].plot(steps,[r['validation']['loss'] for r in rows],'o-',color='#b4484b',lw=1.8,label='固定held-out均值')
    axes[0].set(xlabel='累计更新（含profile2次）',ylabel='32维flow-matching loss',title='训练完成；验证最优在500次')
    axes[0].legend(frameon=False,fontsize=8)
    for g,color in [('A','#178f87'),('B','#d08a35')]:
        axes[1].plot(steps,[r['validation'][g]['real7'] for r in rows],'o-',label=f'{g}真实7维',color=color)
    axes[1].set(xlabel='累计更新',ylabel='固定held-out真实7维loss',title='A/B分别呈现，不能替代行为成功率')
    axes[1].legend(frameon=False,fontsize=8)
    for ax in axes:ax.axvline(500,color='#7c91ac',ls='--',lw=1)
    fig.tight_layout();save(fig,'training_validation.png')

    fig,axes=plt.subplots(1,2,figsize=(11,4.1))
    axes[0].plot(steps,[r['adapter_l2_delta_from_initial'] for r in rows],'o-',color='#178f87')
    axes[0].set(xlabel='累计更新',ylabel='LoRA参数L2变化（相对初始化）',title='继续训练确有参数变化')
    axes[1].bar(['500次','3000次'],[r['changed_adapter_tensors'] for r in [selected,rows[-1]]],color='#178f87',label='已改变')
    axes[1].bar(['500次','3000次'],[462-r['changed_adapter_tensors'] for r in [selected,rows[-1]]],bottom=[r['changed_adapter_tensors'] for r in [selected,rows[-1]]],color='#d9e2ed',label='未改变')
    axes[1].set(ylim=(0,540),ylabel='LoRA张量数（非标量参数量）',title='404/462张量改变；不假称全覆盖')
    axes[1].legend(frameon=False,ncol=2,fontsize=8)
    fig.tight_layout();save(fig,'adapter_delta.png')

    hours=np.array([r['time_unix']-resources[0]['time_unix'] for r in resources])/3600
    gpu=np.array([max([float(g['process_memory_mib']) for g in r.get('gpus',[])]+[0]) for r in resources])/1024
    rss=np.array([float(r['cpu_percent_rss_kib'].split()[1])/1024**2 for r in resources])
    peak_rss=np.array([r['process_max_rss_mib']/1024 for r in resources])
    fig,ax=plt.subplots(figsize=(10,3.5));ax.plot(hours,gpu,label='GPU进程显存',color='#178f87');ax.plot(hours,rss,label='CPU当前RSS',color='#7c91ac')
    ax.plot(hours,peak_rss,label='CPU历史峰值RSS（进程记录）',color='#9d8b7b',ls='--')
    ax.set(xlabel='训练作业采样时长（小时）',ylabel='GiB',title='资源实录：当前占用与进程记录峰值');ax.legend(frameon=False)
    fig.tight_layout();save(fig,'resources.png')
    snapshot=datetime.fromisoformat(bundle['captured_utc']).astimezone(TZ).strftime('%Y-%m-%d %H:%M:%S CST')
    table=[]
    for r in rows:
        v=r['validation'];table.append('<tr>'+''.join(f'<td>{html.escape(str(x))}</td>' for x in [r['step'],f"{v['loss']:.6f}",f"{v['A']['real7']:.6f}",f"{v['B']['real7']:.6f}",r['changed_adapter_tensors'],f"{r['adapter_l2_delta_from_initial']:.4f}"] )+'</tr>')
    css=(OUT.parent/'report.css').read_text(encoding='utf-8')
    final_gap=(rows[-1]['validation']['loss']/selected['validation']['loss']-1)*100
    report=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SFT-A训练完成与评估交接</title><style>{css}</style><main>
    <div class="kicker">SFT-A / 训练完成记录</div><h1>3000次更新已完成；<br>选定500次权重，闭环终评排队中</h1>
    <p class="meta">数据快照：{snapshot}。训练job18658，COMPLETED/0，用时04:59:21。评估job18674，快照时PENDING/Resources。<a href="../report.html">此前完整证据报告</a>。</p>
    <div class="banner">原计划的3000次更新全部完成，未追加训练。固定held-out loss在500次最低，严格使用该权重进入80回合配对评估。CPU审计8份checkpoint通过；本文没有新的VLA调用、仿真或行为结果。</div>
    <h2>1. 完整训练与验证曲线</h2><figure><img src="training_validation.png" alt="完整训练与验证loss"><figcaption>固定验证窗口/噪声与同一整轨迹划分；横轴是梯度更新，不是环境步。训练为离线监督，新增仿真交互0。</figcaption></figure>
    <table><thead><tr><th>更新</th><th>验证32维均值</th><th>A真实7维</th><th>B真实7维</th><th>改变的LoRA张量</th><th>参数L2变化</th></tr></thead><tbody>{''.join(table)}</tbody></table>
    <p>最优点500：均值0.029195、A真实7维0.112833、B真实7维0.152391；最终3000：均值0.032350，较最优高{final_gap:.1f}%。这提示后续拟合没有继续改善固定验证集；不足以单独断定内部原因或闭环退化。A真实7维在1000略低，但选择标准是预先固定的双目标32维等权均值，仍选500。</p>
    <h2>2. 参数变化与保存产物核验</h2><figure><img src="adapter_delta.png" alt="实际LoRA参数变化及张量覆盖"><figcaption>6,043,136个名义LoRA参数、462张量；404张量从初始化发生非零变化。部分native路径未被all_linear有效适配，原profile的覆盖限制保留。</figcaption></figure>
    <p>0、2、500、1000、1500、2000、2500、3000共8份checkpoint已在CPU实际读取：逐文件SHA256、配置/数据/底座身份、参数shape与有限性、优化器step和RNG状态结构均通过；最优点按所有保存验证loss重新核对为500。训练中保存的原参数冻结hash一致。此次审计未重新实例化VLA，也未重新计算内存中完整VLA权重hash，不能作更强冻结声明。</p>
    <p>选定权重SHA256：<code>{selected['sha256']}</code>。详见<a href="data/train/checkpoint_audit/checkpoint_audit.json">CPU审计证据</a>。原训练代码<code>1ede832071e9de25cc92566b241f80916717d675</code>；比较范围仍是同native底座内部，不声称与旧JAX数值等价。</p>
    <h2>3. 资源与自动推进</h2><figure><img src="resources.png" alt="训练资源占用"><figcaption>GPU进程显存峰值{gpu.max():.2f}GiB；Torch分配峰值6.57GiB是不同口径。CPU加载峰值约24.8GiB，常驻RSS中位约{np.median(rss):.2f}GiB。</figcaption></figure>
    <p>18674自17:57:42排队。计算节点GPU均被占用；账户提交限额阻止额外node1短测。已将同一作业GPU请求从限定3090放宽为<code>gpu:1</code>，计算节点3090/4090均可匹配，继续排除登录节点；保留原job ID、提交时间、代码、权重、种子与预算。两臂在同一个实际分配的GPU上评估，最终报告记录硬件。</p>
    <p>训练完成 → 18674执行同native未适配/适配两臂 × A/B × reset3300–3319（80回合，最多32000物理步）→ 独立CPU审计/配对统计/8段预定录像/HTML与ZIP。两个交接进程均已脱离会话并持续更新心跳；当前尚无SFT闭环成功率或仿真视频，待评估生成，不用演示录像冒充策略效果。</p>
    <pre>powershell -ExecutionPolicy Bypass -File tools/remote.ps1 status 18674
powershell -ExecutionPolicy Bypass -File tools/remote.ps1 status 18658</pre>
    <h2>4. 下一步判断</h2><p>先等80回合完成，分开看B正确首抓、B选择性完成、A保留和执行链路，再选一个研究分支。离线loss下降不足以启动新的SFT-S/PPO或再次3000更新；本轮总SFT预算已用完。保留集和旧确认集未用于选权重。资源等待时间由调度器决定，当前无可靠开始时间承诺。</p>
    </main></html>'''
    (OUT/'report.html').write_text(report,encoding='utf-8')
    manifest=dict(snapshot=snapshot,source_sha256=sources,training_updates=3000,selected_step=500,
        checked_checkpoints=8,behavior_results_available=False,report_sha256=hashlib.sha256(report.encode()).hexdigest(),
        cpu_audit_passed=True,local_report_simulation_steps=0,local_report_model_forwards=0)
    (OUT/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'report':str(OUT/'report.html'),'snapshot':snapshot,'selected_step':500,'checkpoints':8,'resources_samples':len(resources)},ensure_ascii=False))


if __name__=='__main__':main()
