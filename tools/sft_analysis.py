"""CPU-only closure for the locked SFT-A paired behavior check.

Reads completed traces and checkpoint metadata; never loads a model, steps an
environment, submits a GPU job, or selects checkpoints by behavior outcomes.
"""
import argparse
import copy
import csv
import html
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import zipfile

import numpy as np

from tools.research_core import GraspTracker, atomic, completed, file_sha, read, sha

ARMS = ('native_unadapted', 'adapted')
METRICS = ('correct_first_grasp', 'active_goal_ever_success', 'active_goal_final',
           'active_goal_stable_released_success',
           'active_goal_first_completion_without_prior_wrong_goal')


def audit_episode(directory, record, protocol, evaluation_commit):
    """Recompute every reported goal/selection metric from saved low-level events."""
    directory = Path(directory)
    c = record['case']
    assert completed(directory, c) == record
    assert record['attempt'] == 0 and record['code_commit'] == evaluation_commit
    assert record['input_protocol_sha256'] == sha(protocol)
    assert c['task'] == 'E1' and c['horizon'] == 400 and c['sampling_seed'] == 0
    files = record['artifact_hashes']
    events = read(directory / next(k for k in files if k.endswith('/events.json')))
    prompts = read(directory / next(k for k in files if k.endswith('/prompts.json')))
    with np.load(directory / next(k for k in files if k.endswith('/arrays.npz')), allow_pickle=False) as arrays:
        actions, noise, keys = [arrays[k].copy() for k in ('executed_actions','noise_sequence','sampling_keys')]
        assert actions.shape == (400,7) and np.isfinite(actions).all()
        assert noise.shape == (20,1,50,32) and np.isfinite(noise).all()
        assert keys.shape == (20,2) and np.isfinite(arrays['final_physics']).all()
    assert record['executed_env_steps'] == 400 and record['vla_queries'] == 20
    assert [e['step'] for e in events] == list(range(401))
    assert [p['step'] for p in prompts] == list(range(0,400,20))
    goal = c['requested_goal_id']; wrong = 'B' if goal == 'A' else 'A'
    target = next(g for g in protocol['e1']['goals'] if g['id'] == goal)
    assert all(p['text'] == target['text'] and len(p['tokens']) == len(p['mask']) == 48 for p in prompts)
    assert all(e['active_text'] == target['text'] for e in events[1:])
    assert not any(events[0]['goals'].values())
    assert not any(events[0]['grasp'][g['object']] for g in protocol['e1']['goals'])
    tracker = GraspTracker(events[0]['grasp'])
    for e in events[1:]: tracker.update(e['grasp'], e['step'])
    assert record['first_grasp_class'] == tracker.first_class
    assert record['first_grasp_step'] == tracker.first_step
    assert record['first_grasp_objects'] == tracker.first_objects
    assert record['correct_first_grasp'] == (tracker.first_class == goal)
    for g in ('A','B'):
        hits = [e['step'] for e in events if e['goals'][g]]
        assert record['goal_ever'][g] == bool(hits)
        assert record['first_goal_step'][g] == (hits[0] if hits else None)
        assert record['goal_final'][g] == events[-1]['goals'][g]
        reversal = any(a['goals'][g] and not b['goals'][g] for a,b in zip(events,events[1:]))
        assert record['goal_reversal'][g] == reversal
    first, other = record['first_goal_step'][goal], record['first_goal_step'][wrong]
    assert record['active_goal_ever_success'] == record['goal_ever'][goal]
    assert record['active_goal_final'] == record['goal_final'][goal]
    assert record['active_goal_first_step'] == first
    assert record['restricted_time_to_success'] == (first if first is not None else 400)
    assert record['both_goals_ever'] == all(record['goal_ever'].values())
    assert record['active_goal_first_completion_without_prior_wrong_goal'] == bool(first is not None and (other is None or first < other))
    count = 0; stable = False
    for event in events[1:]:
        count = count + 1 if event['goals'][goal] and not event['grasp'][target['object']] else 0
        stable |= count >= 3
    assert record['active_goal_stable_released_success'] == stable
    assert record['termination_reason'] == 'fixed_horizon'
    assert record['status'] == ('COMPLETE' if record['active_goal_ever_success'] else 'POLICY_FAILURE')
    common = directory / 'common_states' / (record['common_state']['id'] + '.npz')
    assert file_sha(common) == record['common_state']['file_sha256']
    assert not list((directory / 'attempts' / c['id']).glob('*/error.json'))
    return dict(noise=noise, keys=keys, tokens=[p['tokens'] for p in prompts],
                masks=[p['mask'] for p in prompts],
                native_done_continued=sum(e['native_done'] for e in events[1:-1]))


def interval(values, seed=99025):
    values = np.asarray(values, dtype=float)
    assert values.ndim == 1 and len(values) > 1 and np.isfinite(values).all()
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0,len(values),size=(10000,len(values)))].mean(axis=1)
    return dict(estimate=float(values.mean()), ci95=np.quantile(draws,[.025,.975]).tolist(),
                reset_clusters=len(values), degenerate=bool(np.ptp(values)==0),
                warning='Degenerate empirical bootstrap does not establish population certainty' if np.ptp(values)==0 else None)


def paired_statistics(index, seeds):
    summary = {}; contrasts = {}
    for arm in ARMS:
        summary[arm] = {}
        for goal in ('A','B'):
            rows = [index[arm,seed,goal] for seed in seeds]
            confusion = {k:sum(r['first_grasp_class']==k for r in rows) for k in ('A','B','OTHER','MULTIPLE','NONE')}
            summary[arm][goal] = dict(n=len(rows),counts={m:sum(bool(r[m]) for r in rows) for m in METRICS},first_grasp=confusion)
        both = []; language = []
        for seed in seeds:
            a,b = index[arm,seed,'A'],index[arm,seed,'B']
            both.append(a['correct_first_grasp'] and b['correct_first_grasp'])
            language.append(.5*(int(a['first_grasp_class']=='A')-int(b['first_grasp_class']=='A')+
                                 int(b['first_grasp_class']=='B')-int(a['first_grasp_class']=='B')))
        summary[arm]['pair_both_correct'] = dict(count=sum(both), n=len(seeds))
        summary[arm]['L'] = interval(language)
    for goal in ('A','B'):
        contrasts[goal] = {}
        for metric in METRICS:
            a=np.array([bool(index['native_unadapted',s,goal][metric]) for s in seeds])
            b=np.array([bool(index['adapted',s,goal][metric]) for s in seeds])
            contrasts[goal][metric] = dict(**interval(b.astype(int)-a.astype(int)),
                gained=int((~a & b).sum()),lost=int((a & ~b).sum()),baseline=int(a.sum()),adapted=int(b.sum()))
    language_delta=[]; both_delta=[]
    for seed in seeds:
        def stats(arm):
            a,b=index[arm,seed,'A'],index[arm,seed,'B']
            return (.5*(int(a['first_grasp_class']=='A')-int(b['first_grasp_class']=='A')+int(b['first_grasp_class']=='B')-int(a['first_grasp_class']=='B')),
                    int(a['correct_first_grasp'] and b['correct_first_grasp']))
        a,b=stats(ARMS[0]),stats(ARMS[1]);language_delta.append(b[0]-a[0]);both_delta.append(b[1]-a[1])
    contrasts['L']=interval(language_delta);contrasts['pair_both_correct']=interval(both_delta)
    return dict(summary=summary,adapted_minus_unadapted=contrasts,
                interpretation='New-reset development check, 20 paired reset clusters, one SFT seed; no formal confirmation or OOD claim',
                bootstrap=dict(samples=10000,seed=99025,unit='whole reset with both goals and both arms',multiplicity_adjusted=False))


def analyze(evaluation, training, protocol_file):
    evaluation,training = Path(evaluation).resolve(),Path(training).resolve()
    output=evaluation/'analysis';output.mkdir(exist_ok=True)
    run=read(evaluation/'run.json');cfg=read(training/'data_manifest.json')['config']
    finish=read(evaluation/'SFT_EVAL_COMPLETE.json');train=read(training/'SFT_TRAINING_COMPLETE.json')
    assert finish['passed'] and train['passed'] and finish['selected']==train['selected']
    assert finish['episodes']==80 and finish['physical_steps']==32000
    assert run['commit']==read(training/'run.json')['commit']
    assert read(evaluation/'comparison_scope.json')['comparison_scope']=='within_native_runtime_only'
    assert file_sha(evaluation/'comparison_scope.json')==file_sha(training/'comparison_scope.json')
    assert run['config_sha256']==read(training/'run.json')['config_sha256']
    seeds=cfg['behavior_reset_seeds'];assert seeds==list(range(3300,3320))
    protocol=copy.deepcopy(read(protocol_file)['protocol'])
    protocol['seed_splits']['e1_development']=seeds
    index={};traces={};sources={};videos=[]
    for arm in ARMS:
        folder=evaluation/arm;plan=read(folder/'case_manifest.json');rows=read(folder/'results.json')
        assert len(plan)==len(rows)==40 and {r['case']['id'] for r in rows}=={c['id'] for c in plan}
        assert {p.stem for p in (folder/'cases').glob('*.json')}=={c['id'] for c in plan}
        assert not list(folder.glob('attempts/*/*/error.json'))
        assert read(folder/'SIMULATION_COMPLETE.json')['passed']
        budget=read(folder/'budget.json')
        assert budget['policy_steps']==budget['recorded_physical_steps']==16000
        assert budget['replay_steps']==budget['construction_steps']==budget['uncertain_steps']==0 and not budget['reservations']
        for r in rows:
            c=r['case'];seed=c['reset_seed'];goal=c['requested_goal_id']
            assert seed in seeds and goal in ('A','B') and c['policy']=='base' and c['prompt_id']==goal
            assert r['common_state']['id']==f'E1__{seed}'
            assert r['common_state']['called_set_init_state'] is False
            assert not any(r['common_state']['errors'].values())
            without_id={k:v for k,v in c.items() if k!='id'}
            assert c['id']=='sft_behavior__'+sha(without_id)[:20]
            key=(arm,seed,goal);assert key not in index;index[key]=r
            traces[key]=audit_episode(folder,r,protocol,run['commit'])
            sources[f'{arm}/cases/{c["id"]}.json']=file_sha(folder/'cases'/(c['id']+'.json'))
            if seed in seeds[:2]:
                video=next((k for k in r['artifact_hashes'] if k.endswith('/rollout.mp4')),None)
                assert video is not None;videos.append(dict(arm=arm,goal=goal,seed=seed,path=f'../{arm}/{video}'))
        for goal in ('A','B'):
            actual=finish['results'][arm][goal]
            selected_rows=[index[arm,s,goal] for s in seeds]
            assert actual==dict(episodes=20,first_correct=sum(r['correct_first_grasp'] for r in selected_rows),
                goal_ever=sum(r['active_goal_ever_success'] for r in selected_rows),
                selective_first_completion=sum(r['active_goal_first_completion_without_prior_wrong_goal'] for r in selected_rows))
    for seed in seeds:
        with np.load(evaluation/ARMS[0]/'common_states'/f'E1__{seed}.npz',allow_pickle=False) as a,np.load(evaluation/ARMS[1]/'common_states'/f'E1__{seed}.npz',allow_pickle=False) as b:
            assert set(a.files)==set(b.files)
            for k in a.files:np.testing.assert_array_equal(a[k],b[k])
        for goal in ('A','B'):
            a,b=[traces[arm,seed,goal] for arm in ARMS]
            for k in ('noise','keys','tokens','masks'):np.testing.assert_array_equal(a[k],b[k])
        np.testing.assert_array_equal(traces[ARMS[0],seed,'A']['noise'],traces[ARMS[0],seed,'B']['noise'])
        assert traces[ARMS[0],seed,'A']['tokens']!=traces[ARMS[0],seed,'B']['tokens']
    checkpoint=Path(train['selected']['path']);meta=read(checkpoint.with_suffix('.json'))
    assert meta['step']==train['selected']['step'] and meta['sha256']==file_sha(checkpoint)
    assert meta['validation']['loss']==train['selected']['loss']
    candidates=list(training.glob('checkpoint-*.json'))+list(Path(read(training/'run.json')['gate_dir']).glob('checkpoint-*.json'))
    best=min((read(p)['validation']['loss'],read(p)['step']) for p in candidates)
    assert (train['selected']['loss'],train['selected']['step'])==best
    pilot=read(training/'PILOT_DECISION.json')
    assert bool(pilot['passed'])==bool(pilot['current']['loss']<pilot['initial']['loss'] and pilot['current']['B']['real7']<pilot['initial']['B']['real7'])
    assert train['updates']==(3000 if pilot['passed'] else 500)
    stats=paired_statistics(index,seeds);atomic(output/'paired_statistics.json',stats)
    audit=dict(passed=True,episodes=80,reset_clusters=20,physical_steps=32000,technical_errors=0,
        input_protocol_sha256=sha(protocol),evaluation_commit=run['commit'],analysis_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        analysis_code_sha256=file_sha(__file__),source_case_sha256=sources,selected_checkpoint_sha256=meta['sha256'],
        common_states_and_explicit_noise_equal=True,all_reported_metrics_recomputed=True,
        checkpoint_selection_verified_from_validation=True,simulation_steps_added=0,vla_forwards_added=0,
        native_done_continued=sum(t['native_done_continued'] for t in traces.values()),
        frozen_parameter_evidence='Training asserts original base hash at each save; profile validates fresh-process restore. In-memory evaluation weights were not hashed after rollout; no stronger claim is made.')
    atomic(output/'trajectory_closure.json',audit)
    make_report(output,evaluation,training,stats,train,videos)
    atomic(output/'video_index.json',videos)
    atomic(output/'SFT_ANALYSIS_COMPLETE.json',dict(passed=True,episodes=80,reset_clusters=20,report=str(output/'REPORT.html'),
        statistics_sha256=file_sha(output/'paired_statistics.json'),closure_sha256=file_sha(output/'trajectory_closure.json'),
        comparison_scope='within_native_runtime_only',simulation_steps_added=0,vla_forwards_added=0))
    archive=output/'result_bundle.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as bundle:
        for path in sorted(output.iterdir()):
            if path.is_file() and path!=archive:bundle.write(path,str(path.relative_to(evaluation)))
        for video in videos:
            path=(output/video['path']).resolve();assert path.is_relative_to(evaluation)
            bundle.write(path,str(path.relative_to(evaluation)))
        for name in ('run.json','SFT_EVAL_COMPLETE.json','selected_checkpoint.json','comparison_scope.json'):
            bundle.write(evaluation/name,name)
    with zipfile.ZipFile(archive) as bundle:assert bundle.testzip() is None
    flag=read(output/'SFT_ANALYSIS_COMPLETE.json');flag.update(archive=str(archive),archive_sha256=file_sha(archive))
    atomic(output/'SFT_ANALYSIS_COMPLETE.json',flag)
    return read(output/'SFT_ANALYSIS_COMPLETE.json')


def make_report(output,evaluation,training,stats,train,videos):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4));x=np.arange(2)
    for ax,metric,title in zip(axes,METRICS[:2],['Correct first grasp','Requested goal ever']):
        for j,arm in enumerate(ARMS):
            values=[stats['summary'][arm][g]['counts'][metric] for g in ('A','B')]
            bars=ax.bar(x+(j-.5)*.34,np.array(values)/20,.34,label=arm,color=['#7c91ac','#178f87'][j])
            for bar,value in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.02,f'{value}/20',ha='center',fontsize=9)
        ax.set(xticks=x,xticklabels=['A: cream cheese','B: butter'],ylim=(0,1.24),title=title,ylabel='Episode fraction')
        ax.legend(frameon=False,fontsize=8)
    fig.tight_layout();fig.savefig(output/'paired_behavior.png',dpi=160);plt.close(fig)
    rows=[]
    for goal in ('A','B'):
        for metric in METRICS:
            v=stats['adapted_minus_unadapted'][goal][metric]
            rows.append([goal,metric,f"{v['baseline']}/20",f"{v['adapted']}/20",v['gained'],v['lost'],v['estimate'],v['ci95']])
    with (output/'paired_summary.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(['goal','metric','unadapted','adapted','gained','lost','delta','cluster_ci95']);writer.writerows(rows)
    header='<tr><th>目标</th><th>指标</th><th>未适配</th><th>适配</th><th>救回/损失</th><th>差值及95%区间</th></tr>'
    lines=[]
    for goal,metric,a,b,w,l,delta,ci in rows:
        lines.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [goal,metric,a,b,f'{w}/{l}',f'{delta*100:+.1f}pp [{ci[0]*100:+.1f},{ci[1]*100:+.1f}]'])+'</tr>')
    media=[]
    for v in videos:
        media.append(f'<figure><video controls preload="none" src="{html.escape(v["path"])}"></video><figcaption>{v["arm"]} / goal {v["goal"]} / reset {v["seed"]}：预先指定案例，非按成功筛选。</figcaption></figure>')
    text=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>SFT-A 配对行为复核</title>
    <style>body{{max-width:1100px;margin:40px auto;padding:24px;font:16px/1.8 sans-serif;color:#172b4d}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:9px;border-bottom:1px solid #ddd;text-align:left}}img,video{{max-width:100%}}figure{{margin:28px 0}}pre{{white-space:pre-wrap}}.note{{background:#fff5e6;padding:16px}}figcaption{{font-size:13px;color:#52637a}}</style>
    <h1>SFT-A：同native底座的目标接口/技能复核</h1>
    <p>训练job {training.name}；评估job {evaluation.name}。真实训练{train['updates']}次更新，选定step {train['selected']['step']}，held-out flow loss {train['selected']['loss']:.6f}。checkpoint仅由验证loss选取。</p>
    <p class="note">本结果只支持同native runtime内部比较。原JAX/native迁移门禁未通过，不能跨框架单变量归因。20个新reset×A/B×两臂＝80回合；一份SFT训练seed，属于开发复核，不是正式多seed确认或OOD。</p>
    <p>全程400步、chunk20、显式z与初态配对；全部80条动作、文本、事件、目标指标及checkpoint选择经CPU审计。新增环境步/模型调用均0。首抓、ever、最终、稳定释放及先于错误目标完成分开列出。</p>
    <img src="paired_behavior.png" alt="两臂逐目标首抓与完成统计">
    <table>{header}{''.join(lines)}</table>
    <p>区间：10,000次整reset成簇bootstrap，seed99025。退化区间不证明总体确定性，未校正多重比较。完整L、双目标均正确及混淆计数见<a href="paired_statistics.json">配对统计</a>；完整核验见<a href="trajectory_closure.json">轨迹闭环</a>。</p>
    <h2>预定reset3300/3301仿真录像</h2>{''.join(media)}
    <h2>后续解释边界</h2><p>B正确首抓/选择性完成提高且A保留，才支持共同warm start候选；loss下降但B仍错选，应继续诊断接口/覆盖；B选对但不完成则研究执行链路；A退化需明确报告权衡。后续方向需要人工审阅，不自动启动SFT-S、PPO或新Noise RL。</p></html>'''
    (output/'REPORT.html').write_text(text,encoding='utf-8')


def watch(args):
    import fcntl
    directory=Path(args.workflow_dir).resolve();training=Path(args.training).resolve()
    lock=(directory/'analysis.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(pid=os.getpid(),phase='WAITING',simulation_steps_added=0,vla_forwards_added=0)
    def publish():
        state['updated_unix']=time.time();atomic(directory/'analysis_workflow.json',state)
        atomic(training/'SFT_ANALYSIS_WORKFLOW.json',state)
        if state.get('evaluation_directory'):atomic(Path(state['evaluation_directory'])/'SFT_ANALYSIS_WORKFLOW.json',state)
    try:
        while True:
            flow=read(directory/'workflow.json');state['handoff_phase']=flow['phase']
            if flow.get('evaluation_job'):
                state['evaluation_directory']=str(training.parent/('job-'+flow['evaluation_job']))
            if flow['phase']=='COMPLETE':
                state['phase']='AUDITING';publish()
                result=analyze(state['evaluation_directory'],training,args.protocol)
                state.update(phase='COMPLETE',completion=result);publish();return
            if flow['phase'].startswith(('FAILED','WORKFLOW_ERROR','SUBMIT_REJECTED','SUBMISSION_UNCERTAIN')):
                state.update(phase='STOPPED_UPSTREAM',reason=flow.get('error',flow['phase']));publish();return
            if time.time()-flow['updated_unix']>300:
                os.kill(flow['pid'],0)
                state['warning']='Handoff heartbeat older than 300 seconds; process still exists'
            state['phase']='WAITING_FOR_EVALUATION';publish();time.sleep(60)
    except BaseException as error:
        state.update(phase='ANALYSIS_ERROR',error=str(error),traceback=traceback.format_exc());publish();raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--training',required=True);p.add_argument('--protocol',required=True)
    p.add_argument('--evaluation');p.add_argument('--workflow-dir')
    args=p.parse_args()
    if args.workflow_dir:watch(args)
    else:
        assert args.evaluation
        print(json.dumps(analyze(args.evaluation,args.training,args.protocol)))
