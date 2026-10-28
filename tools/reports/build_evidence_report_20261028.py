"""Build a local evidence report from cached artifacts; never contacts the server.

Run with the existing report Python environment. Raw runs, videos and snapshots
remain outside Git. The generated manifest records precisely which files were read.
"""
from __future__ import annotations

import csv
import hashlib
import html
import json
import shutil
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/reports/20261028"
OLD = ROOT / "tmp/experiment-report-20261021"
FIRST = ROOT / "tmp/language-stage-20261027/job-18584"
SNAP = ROOT / "tmp/language-stage-20261027/report_snapshot_20261028"
TZ = timezone(timedelta(hours=8))
SOURCES = {}
CHECKS = []


def record(path):
    path = Path(path)
    raw = path.read_bytes()
    SOURCES[path.relative_to(ROOT).as_posix()] = {
        "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)
    }
    return raw


def read_json(path):
    return json.loads(record(path).decode("utf-8-sig"))


def read_csv(path):
    return list(csv.DictReader(record(path).decode("utf-8-sig").splitlines()))


def copy(source, destination):
    record(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def table(headers, rows):
    def cell(x):
        return html.escape(str(x))
    return '<div class="table-wrap"><table><thead><tr>' + ''.join(
        f'<th>{cell(x)}</th>' for x in headers
    ) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join(f'<td>{cell(x)}</td>' for x in row) + '</tr>' for row in rows
    ) + '</tbody></table></div>'


def check(name, condition):
    if not condition:
        raise AssertionError(name)
    CHECKS.append(name)


def finish(fig, name):
    fig.savefig(OUT / "figures" / name, dpi=165, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def success(row):
    return str(row["success"]).lower() == "true"


def main():
    for name in ["figures", "media", "data"]:
        (OUT / name).mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False, "font.size": 10, "axes.spines.top": False,
        "axes.spines.right": False, "axes.titleweight": "bold",
    })
    facts = read_json(OLD / "facts.json")
    baseline = read_csv(OLD / "data_tables/baseline_final_summary.csv")
    dev = read_csv(OLD / "data_tables/C1_development_360_cases.csv")
    confirm = read_csv(OLD / "data_tables/confirmation_300_cases.csv")
    c2 = read_csv(OLD / "data_tables/C2_v6_96_cases.csv")
    summary = read_csv(FIRST / "summary_by_condition.csv")
    stats = read_json(FIRST / "paired_cluster_statistics.json")
    protocol = read_json(FIRST / "resolved_protocol.json")
    calibration = read_json(FIRST / "calibration.json")
    audit = read_json(FIRST / "trajectory_audit.json")
    closure = read_json(FIRST / "closure_check.json")
    allocation = read_json(FIRST / "allocation_accounting.json")
    static = read_json(SNAP / "job-18655/static_alignment.json")
    native_cases = read_json(SNAP / "job-18655/closed_loop/results.json")
    latest = read_json(SNAP / "job-18658/latest.json")
    train_run = read_json(SNAP / "job-18658/run.json")
    config = read_json(ROOT / "configs/sft_a_v1.json")
    migration = read_json(ROOT / "configs/sft_a_migration_review.json")
    profile = read_json(SNAP/'job-18657/SFT_PROFILE_PASS.json')
    coverage = read_json(SNAP/'job-18657/lora_coverage.json')
    gradients = read_json(SNAP/'job-18657/gradient-coverage-000002.json')
    demos = read_json(SNAP/'research/DEMONSTRATION_ORDER_AUDIT.json')
    cross_gpu = read_json(SNAP/'research/JAX_CROSS_GPU_NUMERICS.json')
    workflow = read_json(SNAP/'workflow/workflow.json')
    training_metrics = [json.loads(line) for line in record(SNAP/'job-18658/metrics.jsonl').decode('utf-8').splitlines() if line.strip()]
    for relative,expected in migration['source_artifact_sha256'].items():
        source=SNAP/'job-18655'/relative
        if source.is_file():
            check('migration locked hash: '+relative,hashlib.sha256(record(source)).hexdigest()==expected)
    data_manifest = read_json(ROOT / "tmp/language-stage-20261027/job-18643/data_manifest.json")
    env = read_json(ROOT / "tmp/language-stage-20261027/sft_a_environment/NATIVE_ENV_READY.json")
    # Archive the small inputs behind the report, without importing runtime code.
    for source in [
        OLD / "data_tables/baseline_final_summary.csv",
        OLD / "data_tables/C1_development_360_cases.csv",
        OLD / "data_tables/confirmation_300_cases.csv",
        OLD / "data_tables/C2_v6_96_cases.csv",
        FIRST / "summary_by_condition.csv", FIRST / "paired_cluster_statistics.json",
        FIRST / "resolved_protocol.json", FIRST / "calibration.json",
        ROOT / "configs/sft_a_v1.json", ROOT / "configs/sft_a_migration_review.json",
        SNAP / "job-18658/latest.json", SNAP / "job-18658/run.json",
        SNAP / "job-18657/SFT_PROFILE_PASS.json", SNAP / "job-18657/lora_coverage.json",
        SNAP / "job-18657/gradient-coverage-000002.json", SNAP / "job-18657/adapter_deltas.json",
        SNAP / "job-18657/checkpoint-000000.json", SNAP / "job-18657/checkpoint-000002.json",
        SNAP / "research/DEMONSTRATION_ORDER_AUDIT.json", SNAP / "research/JAX_CROSS_GPU_NUMERICS.json",
        SNAP / "workflow/workflow.json", SNAP / "workflow/launch.json",
    ]:
        copy(source, OUT / "data" / source.name)

    old_figures = [
        "01_task_scene.png", "02_algorithm.png", "03_learning_curves.png",
        "06_prompt_schedule.png", "07_probe_matrix.png", "08_c1_common_states.png",
        "09_c1_results.png", "13_c2_common_states.png", "17_trigger_and_budget.png",
    ]
    for name in old_figures:
        copy(OLD / "figures" / name, OUT / "figures" / name)
    for name in ["e1_confusion.png", "e1_auxiliary_behavior.png", "e2_stages.png", "E2_4000_contact_sheet.png"]:
        copy(FIRST / "figures" / name, OUT / "figures" / name)
    for name in [
        "V02_baseline_four_methods.mp4", "V05_normal_probe_instruction.mp4",
        "V06_c1_noise_and_conditions.mp4", "V07_c1_four_instructions.mp4",
        "V09_confirmation_gain_and_regression.mp4", "V10_c2_held_vs_drop.mp4",
        "V12_c2_random_failure_and_recovery.mp4",
    ]:
        copy(OLD / "videos" / name, OUT / "media" / name)
    for name in ["E1_dev3000_all_policies.mp4", "E2_dev4000_all_conditions.mp4"]:
        copy(FIRST / "media" / name, OUT / "media" / name)
    for name in ["A_image_data_vs_render.png", "B_wrist_data_vs_render.png"]:
        copy(ROOT / "tmp/language-stage-20261027/job-18643" / name, OUT / "figures" / name)
    # Decode an existing frame locally for each video poster; no simulator replay.
    import imageio.v2 as imageio
    from PIL import Image
    for video in sorted((OUT/'media').glob('*.mp4')):
        with imageio.get_reader(video) as reader:
            frame=reader.get_data(0)
        Image.fromarray(frame).save(OUT/'figures'/f'{video.stem}_poster.jpg',quality=88)

    check("baseline successes are 35/99/100/98 out of 100", [int(x["successes"]) for x in baseline] == [35,99,100,98])
    check("C1 development has 360 branches and 10 reset clusters", len(dev) == 360 and len({r['reset_seed'] for r in dev}) == 10)
    check("C1 confirmation has 300 branches and 10 reset clusters", len(confirm) == 300 and len({r['reset_seed'] for r in confirm}) == 10)
    check("C2 contains 96 branches and four triggered references", len(c2) == 96 and len({r['reset_seed'] for r in c2}) == 4)
    check("static migration has 120 cases and failed original gate", len(static['cases']) == 120 and static['passed'] is False)
    check("native closed-loop contains 20 complete 400-step cases", len(native_cases) == 20 and all(x['executed_env_steps'] == 400 for x in native_cases))
    check("SFT runtime is explicitly within-native", train_run['comparison_scope'] == 'within_native_runtime_only' and migration['cross_framework_equivalence_claim'] is False)
    check("SFT split is 40/10 complete trajectories per goal", config['train_trajectories_per_goal'] == 40 and config['validation_trajectories_per_goal'] == 10)
    check('real profile updated adapters and restored exactly in a fresh process',profile['passed'] and profile['updates']==2 and profile['adapter_max_delta']>0 and profile['fresh_process_restore_max_error']==0)
    check('LoRA parameter/tensor count and actual unused paths',coverage['all_base_frozen'] and coverage['trainable_parameters']==6043136 and len(coverage['trainable_tensors'])==462 and len(gradients['unused_trainable_tensors'])==56)
    check('genuine demonstrations first grasp the requested object for A/B',demos['completed'] and demos['summary']['A']['first_grasp']=={'A':50} and demos['summary']['B']['first_grasp']=={'B':50})
    check('cross-GPU JAX sensitivity is not called framework equivalence',cross_gpu['input_arrays_identical'] and cross_gpu['alarm_cases']==33 and not migration['cross_framework_equivalence_claim'])
    check('automatic evaluation workflow is waiting for this training, not duplicated',workflow['phase']=='WAIT_TRAIN' and workflow['train_job']=='18658' and workflow['evaluation_job'] is None)

    # Baseline Wilson intervals describe evaluation trials, not training-seed variance.
    fig, ax = plt.subplots(figsize=(9.2,4.5))
    k = np.array([int(r['successes']) for r in baseline]); n = 100; z=1.96
    p=k/n; center=(p+z*z/(2*n))/(1+z*z/n)
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    colors=['#75889e','#178f87','#178f87','#178f87']
    ax.bar(range(4), p*100, color=colors, width=.56)
    ax.errorbar(range(4), p*100, yerr=np.maximum(0,np.vstack([p-center+half,center+half-p]))*100, fmt='none',ecolor='#172b4d',capsize=5)
    for i,v in enumerate(k): ax.text(i, (center[i]+half[i])*100+2, f'{v}/100', ha='center', weight='bold')
    ax.set(xticks=range(4),xticklabels=['Base VLA','DSRL seed 0','DSRL seed 1','DSRL seed 2'],ylim=(0,113),ylabel='独立终评成功率（%）', title='同一单任务：三个训练 seed 均明显高于 Base')
    ax.text(.5,-.20,'误差线：每次100回合的Wilson 95%区间；Base仅评估一组，未作严格逐回合配对。',ha='center',transform=ax.transAxes,fontsize=9,color='#52637a')
    finish(fig,'baseline_endpoint.png')

    train_rows=[]
    for seed,tid,eid in [(0,'18046','18102'),(1,'18103','18122'),(2,'18123','18125')]:
        t=facts['job-'+tid]['latest']; e=facts['job-'+eid]['latest']
        train_rows.append([seed,tid,eid,f"{int(t['gradient_updates']):,}",f"{int(t['env_steps']):,}",f"{int(t['chunk_transitions']):,}",f"{int(t['evaluation/total_env_steps']):,}"])

    # Recompute the paired confirmation gains/losses and exploratory cluster intervals.
    labels=['Base','有界随机','DSRL s0','DSRL s1','DSRL s2']; identifiers=[('base','0'),('bounded_random','0'),('dsrl','0'),('dsrl','1'),('dsrl','2')]
    conf_rows=[]; conf_original=[]; conf_regrasp=[]; paired_export=[]
    rng=np.random.default_rng(20261028)
    for label,(noise,seed) in zip(labels,identifiers):
        rows=[r for r in confirm if r['noise_mode']==noise and r['controller_training_seed']==seed]
        pairs={}
        for r in rows: pairs.setdefault((r['reset_seed'],r['condition']),{})[r['instruction']]=success(r)
        check(f'{label}: 30 complete original/regrasp confirmation pairs', len(pairs)==30 and all(set(x)=={'original','regrasp'} for x in pairs.values()))
        a=sum(p['original'] for p in pairs.values()); b=sum(p['regrasp'] for p in pairs.values())
        wins=sum(not p['original'] and p['regrasp'] for p in pairs.values()); losses=sum(p['original'] and not p['regrasp'] for p in pairs.values())
        cluster={}
        for (reset,condition),v in pairs.items(): cluster.setdefault(reset,[]).append(int(v['regrasp'])-int(v['original']))
        values=np.array([np.mean(cluster[s]) for s in sorted(cluster)])
        boot=values[rng.integers(0,len(values),size=(10000,len(values)))].mean(axis=1)
        lo,hi=np.quantile(boot,[.025,.975]); delta=(b-a)/30
        conf_rows.append([label,f'{a}/30',f'{b}/30',f'{wins} / {losses}',f'{delta*100:+.1f} pp',f'[{lo*100:+.1f}, {hi*100:+.1f}] pp'])
        paired_export.append(dict(policy=label,original=a,regrasp=b,n=30,wins=wins,losses=losses,reset_clusters=10,delta=delta,ci95=[lo,hi]))
        conf_original.append(a); conf_regrasp.append(b)
    check('confirmation counts match archived run summaries',conf_original==[12,14,28,26,27] and conf_regrasp==[18,15,29,28,27])
    fig,ax=plt.subplots(figsize=(10,4.6)); x=np.arange(5)
    a=ax.bar(x-.18,np.array(conf_original)/30*100,.35,label='original',color='#7c91ac')
    b=ax.bar(x+.18,np.array(conf_regrasp)/30*100,.35,label='regrasp（前40步）',color='#d08a35')
    for bars,counts in [(a,conf_original),(b,conf_regrasp)]:
        for bar,count in zip(bars,counts): ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,f'{count}/30',ha='center',fontsize=9)
    ax.set(xticks=x,xticklabels=labels,ylim=(0,118),ylabel='原A目标完成率（%）',title='独立确认：改写收益比开发集更弱，且存在退化分支')
    ax.legend(frameon=False,loc='upper left',ncol=2)
    ax.text(.5,-.19,'每柱30分支 = 10个reset × 3种共同状态；统计推断以10个reset为cluster。',ha='center',transform=ax.transAxes,fontsize=9,color='#52637a')
    finish(fig,'confirmation_recomputed.png')

    c2_rows=[]
    for instruction in ['original','regrasp','paraphrase','pick']:
        line=[instruction]
        for mode in ['base','bounded_random','dsrl']:
            rows=[r for r in c2 if r['condition']=='C2_drop' and r['instruction']==instruction and r['noise_mode']==mode]
            check(f'C2 {instruction}/{mode}: four actual dropped states',len(rows)==4)
            line.append(f'{sum(success(r) for r in rows)}/4')
        c2_rows.append(line)
    check('C2 only failure is original/bounded_random',c2_rows==[['original','4/4','3/4','4/4'],['regrasp','4/4','4/4','4/4'],['paraphrase','4/4','4/4','4/4'],['pick','4/4','4/4','4/4']])

    # Exact confirmation counts, never reconstructed from rounded percentages.
    e1_rows=[]; e1_export=[]
    policies=['base','bounded_random','dsrl_s0','dsrl_s1','dsrl_s2']
    for label,policy in zip(labels,policies):
        arms={r['prompt']:r for r in summary if r['phase']=='e1_confirm' and r['policy']==policy}
        check(policy+' E1 has 80 valid trials per goal',all(int(arms[g]['valid'])==80 for g in ['A','B']))
        a,b=arms['A'],arms['B']
        def count(row,key): return round(float(row[key])*int(row['valid']))
        values=[int(a['first_grasp_A']),int(b['first_grasp_B']),count(a,'active_goal_ever_success'),count(b,'active_goal_ever_success'),int(b['first_grasp_A']),count(b,'active_goal_first_completion_without_prior_wrong_goal')]
        e1_rows.append([label]+[f'{v}/80' for v in values])
        e1_export.append(dict(policy=policy,correct_first_A=values[0],correct_first_B=values[1],ever_A=values[2],ever_B=values[3],wrong_A_under_B=values[4],selective_B=values[5],n=80))
    check('three DSRL controllers never first-grasp B in E1 confirmation',all(x['correct_first_B']==0 for x in e1_export[2:]))
    check('pooled DSRL B-ever 56/240 but selective-first 7/240',sum(x['ever_B'] for x in e1_export[2:])==56 and sum(x['selective_B'] for x in e1_export[2:])==7)
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    x=np.arange(5); w=.24
    for ax,series,title in [
        (axes[0],[('correct_first_A','A指令正确首抓','#178f87'),('correct_first_B','B指令正确首抓','#d08a35')],'目标选择：A强，B首抓几乎消失'),
        (axes[1],[('ever_B','B曾经完成','#7c91ac'),('selective_B','B先于错误目标完成','#b4484b')],'B目标：ever不能替代选择性完成'),
    ]:
        for j,(key,label,color) in enumerate(series):
            vals=[r[key] for r in e1_export]
            bars=ax.bar(x+(j-.5)*w,np.array(vals)/80*100,w,label=label,color=color)
            for bar,k in zip(bars,vals): ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+2,str(k),ha='center',fontsize=8)
        ax.set(xticks=x,xticklabels=['Base','随机','s0','s1','s2'],ylim=(0,132),ylabel='回合比例（%）')
        ax.set_title(title,pad=15,fontsize=11)
        ax.legend(frameon=False,fontsize=8,loc='upper left')
    fig.subplots_adjust(wspace=.28,top=.8)
    fig.text(.5,-.02,'柱顶数字为次数（每策略/目标80回合）；40个reset × 2个采样seed，以reset成簇。',ha='center',color='#52637a',fontsize=9)
    finish(fig,'e1_goal_fidelity.png')

    # Compare the new runtime with the exact old cases; seed values alone are not parity.
    old_cases=[json.loads(line) for line in record(FIRST/'episodes.jsonl').decode('utf-8').splitlines() if line.strip()]
    old_by_id={r['case']['id']:r for r in old_cases}
    grasp_agreement=sum(r['first_grasp_class']==old_by_id[r['case']['id']]['first_grasp_class'] for r in native_cases)/20
    goal_agreement=sum(r['goal_ever']==old_by_id[r['case']['id']]['goal_ever'] for r in native_cases)/20
    check('migration closed-loop agreement remains 60% grasp / 70% goal-vector',grasp_agreement==.6 and goal_agreement==.7)
    native_rows=[]
    for goal in ['A','B']:
        rows=[r for r in native_cases if r['case']['requested_goal_id']==goal]
        counts=Counter(r['first_grasp_class'] for r in rows)
        native_rows.append([goal,len(rows),counts['A'],counts['B'],counts['OTHER'],counts['NONE'],sum(r['active_goal_ever_success'] for r in rows)])
    rmse=np.array([r['rmse'] for r in static['cases']]); maximum=np.array([r['maximum'] for r in static['cases']])
    alarms=(rmse>.01)|(maximum>.1)
    check('static alarm count is 23 of 120',int(alarms.sum())==23)
    fig,axes=plt.subplots(1,2,figsize=(11.3,4.5))
    axes[0].scatter(range(120),rmse,c=np.where(alarms,'#b4484b','#178f87'),s=17)
    axes[0].axhline(.01,color='#b4484b',ls='--',lw=1,label='RMSE报警线0.01')
    axes[0].set(yscale='log',xlabel='固定观测 × 文本 × 显式z（120组）',ylabel='normalized action RMSE',title='静态转换仍未通过工程门禁'); axes[0].legend(frameon=False,fontsize=8)
    axes[1].barh(['首抓类别一致','双目标ever向量一致'],[grasp_agreement*100,goal_agreement*100],color=['#d08a35','#7c91ac'])
    axes[1].axvline(90,color='#b4484b',ls='--',label='预设门槛90%')
    axes[1].set(xlim=(0,103),xlabel='原JAX vs native的一致率（%）',title='匹配初态/z仍不足以建立行为等价')
    for i,v in enumerate([grasp_agreement*100,goal_agreement*100]): axes[1].text(v+1,i,f'{v:.0f}%',va='center')
    axes[1].legend(frameon=False,loc='lower right',fontsize=8)
    fig.tight_layout(); finish(fig,'migration_limits.png')

    # A snapshot is a snapshot: draw no invented training curve.
    stamp=datetime.fromtimestamp(latest['time_unix'],TZ).strftime('%Y-%m-%d %H:%M:%S CST')
    fig,axes=plt.subplots(1,2,figsize=(11.3,3.6))
    for i,(cap,label) in enumerate([(500,'pilot（500更新）'),(3000,'有条件上限（3000更新）')]):
        axes[0].barh(i,cap,color='#e4ebf2',height=.45)
        axes[0].barh(i,latest['gradient_updates'],color='#178f87',height=.45)
        axes[0].text(cap+35,i,f"{latest['gradient_updates']}/{cap}",va='center')
    axes[0].set(yticks=[0,1],yticklabels=['pilot 500','max 3000'],xlim=(0,3550),xlabel='梯度更新次数',title='SFT-A进度快照；不代表最终训练量')
    axes[1].axis('off')
    lines=[f"已记录更新：{latest['gradient_updates']}",f"train flow loss：{latest['sft/train_loss']:.5f}",f"梯度范数：{latest['sft/gradient_norm']:.4f}",f"本次update：{latest['sft/update_seconds']:.2f}秒",f"Torch峰值：{latest['sft/peak_gpu_mib']/1024:.2f} GiB",'验证/行为结果：待完成']
    axes[1].text(.05,.96,'\n'.join(lines),va='top',fontsize=12,linespacing=1.6,color='#172b4d')
    fig.suptitle(stamp,fontsize=10,color='#52637a'); fig.tight_layout(); finish(fig,'sft_snapshot.png')
    selected=[m for m in training_metrics if m.get('phase')=='sft_training' and m.get('gradient_updates',0)<=latest['gradient_updates']]
    updates=np.array([m['gradient_updates'] for m in selected]); loss=np.array([m['sft/train_loss'] for m in selected])
    check('training curve uses actual saved updates through the snapshot',len(selected)>100 and updates[-1]==latest['gradient_updates'] and np.isfinite(loss).all())
    fig,axes=plt.subplots(1,2,figsize=(11.3,4.1))
    axes[0].plot(updates,loss,color='#a5cfc9',lw=.7,label='每次更新实录')
    axes[0].plot(updates[19:],np.convolve(loss,np.ones(20)/20,mode='valid'),color='#178f87',lw=2,label='20次更新滑动均值')
    axes[0].set(xlabel='累计梯度更新（含profile 2次）',ylabel='训练flow-matching loss',title='训练目标下降；尚不能替代闭环评估'); axes[0].legend(frameon=False,fontsize=8)
    axes[1].plot(updates,[m['sft/learning_rate'] for m in selected],color='#7c91ac',lw=2)
    axes[1].axvline(100,color='#d08a35',ls='--',lw=1,label='warmup 100')
    axes[1].set(xlabel='累计梯度更新',ylabel='实际学习率',title='warmup已跨过，进入cosine阶段'); axes[1].ticklabel_format(axis='y',style='sci',scilimits=(0,0)); axes[1].legend(frameon=False,fontsize=8)
    fig.tight_layout(); finish(fig,'sft_training_curve.png')

    replacements={
        'BASELINE_TABLE':table(['方法','训练seed','独立终评job','成功/回合','平均完成/超时步数'],[[r['method'],r['seed'],r['job'],f"{r['successes']}/{r['episodes']}",r['mean_episode_steps'] or '未在汇总中提供'] for r in baseline]),
        'TRAIN_BUDGET_TABLE':table(['seed','训练job','终评job','梯度更新','训练物理步','训练chunk','训练内评估步'],train_rows),
        'CONFIRM_TABLE':table(['策略','original','regrasp','改写救回 / 损失','净变化','探索性95% cluster区间'],conf_rows),
        'C2_TABLE':table(['掉落后前40步指令','Base','有界随机','DSRL seed0'],c2_rows),
        'E1_TABLE':table(['策略','A正确首抓','B正确首抓','A ever','B ever','B指令误抓A','B选择性首次完成'],e1_rows),
        'NATIVE_TABLE':table(['目标指令','回合','首抓A','首抓B','首抓OTHER','无首抓','请求目标ever'],native_rows),
        'SNAPSHOT_TIME':html.escape(stamp),
        'TRAIN_UPDATES':str(latest['gradient_updates']),
        'TRAIN_LOSS':f"{latest['sft/train_loss']:.5f}",
        'UPDATE_SECONDS':f"{latest['sft/update_seconds']:.2f}",
        'RMSE_MEAN':f'{rmse.mean():.8f}',
        'RMSE_MAX':f'{rmse.max():.8f}',
        'MAX_ABS':f'{maximum.max():.6f}',
        'GENERATED_TIME':datetime.now(TZ).strftime('%Y-%m-%d %H:%M CST'),
    }
    for source in [ROOT/'docs/protocols/dsrl_vla_protocol_20261027/EXPERIMENT_SPEC.md',ROOT/'docs/protocols/dsrl_vla_protocol_20261027/SFT_A_EXECUTION.md']:
        record(source)
    body=record(OUT/'report_body.html').decode('utf-8')
    for key,value in replacements.items(): body=body.replace('@@'+key+'@@',value)
    import re
    body=re.sub(r'<video controls preload="none" src="media/([^\"]+)\.mp4"',lambda m: f'<video controls preload="none" poster="figures/{m[1]}_poster.jpg" src="media/{m[1]}.mp4"',body)
    check('all report template slots resolved','@@' not in body)
    css=record(OUT/'report.css').decode('utf-8')
    document='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DSRL π₀ / LIBERO 实验证据与下一步方案 · 2026-10-28</title><style>'+css+'</style></head><body>'+body+'</body></html>'
    (OUT/'report.html').write_text(document,encoding='utf-8')
    derived=dict(confirmation_paired=paired_export,e1_confirmation=e1_export,c2_drop=c2_rows,native_development=native_rows,snapshot=latest,snapshot_time=stamp)
    (OUT/'data/derived_results.json').write_text(json.dumps(derived,ensure_ascii=False,indent=2),encoding='utf-8')
    # Validate all portable report references, including video links and posters.
    from html.parser import HTMLParser
    class Links(HTMLParser):
        def __init__(self): super().__init__(); self.refs=[]; self.ids=set(); self.videos=0
        def handle_starttag(self,tag,attrs):
            a=dict(attrs)
            if 'id' in a: self.ids.add(a['id'])
            if tag=='video': self.videos+=1
            self.refs += [v for k,v in attrs if k in ('href','src','poster')]
    parser=Links(); parser.feed(document)
    missing=[]
    for ref in parser.refs:
        if ref.startswith(('http:','https:','mailto:','data:')): continue
        if ref.startswith('#'):
            if ref[1:] not in parser.ids: missing.append(ref)
        elif ref!='source_manifest.json' and not (OUT/ref.split('#')[0]).exists(): missing.append(ref)
    check('all report local references and section anchors exist',not missing)
    manifest={'generated_at':replacements['GENERATED_TIME'],'data_cutoff':stamp,'source_policy':'Only cached local files read; no SSH, GPU, simulator or training environment used by this builder.','sources':SOURCES,'checks':CHECKS,'report_references':len(parser.refs),'embedded_videos':parser.videos,'scope':'completed original-JAX studies + failed migration diagnostic + ongoing within-native SFT snapshot'}
    (OUT/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    assert (OUT/'source_manifest.json').is_file()
    print(json.dumps({'report':str(OUT/'report.html'),'checks':len(CHECKS),'sources':len(SOURCES),'references':len(parser.refs),'videos':parser.videos,'snapshot':stamp},ensure_ascii=False))


if __name__=='__main__':
    main()
