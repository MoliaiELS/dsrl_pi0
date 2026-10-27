"""Paired reset-cluster statistics and a reproducible first-batch report."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from tools.research_core import atomic, completed, read


def interval(values, seed=99021):
    x=np.asarray(values,dtype=float)
    if not len(x): return dict(estimate=None,ci95=[None,None],clusters=0,degenerate=False)
    rng=np.random.default_rng(seed)
    boot=x[rng.integers(len(x),size=(10000,len(x)))].mean(axis=1)
    return dict(estimate=float(x.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),clusters=len(x),
                degenerate=bool(np.ptp(x)==0),warning='Degenerate observed cluster distribution does not establish certainty' if np.ptp(x)==0 else None)


def paired_contrasts(records,planned,phase):
    by_seed=defaultdict(list)
    for c in planned:
        if c['phase']==phase: by_seed[c['reset_seed']].append(c)
    lookup={r['case']['id']:r for r in records if r['status'] in ['COMPLETE','POLICY_FAILURE']}
    complete=[s for s,cs in by_seed.items() if all(c['id'] in lookup for c in cs)]
    contrasts={}
    for metric in ['correct_first_grasp','active_goal_ever_success']:
        for reference in ['base','bounded_random']:
            effects=[];bounds=[]
            for seed,cs in by_seed.items():
                def method(method,fill):
                    values=[float(lookup[c['id']][metric]) if c['id'] in lookup else fill for c in cs if c['policy']==method]
                    return float(np.mean(values))
                if seed in complete:
                    effects.append(np.mean([method('dsrl_s'+str(i),0) for i in range(3)])-method(reference,0))
                low=np.mean([method('dsrl_s'+str(i),0) for i in range(3)])-method(reference,1)
                high=np.mean([method('dsrl_s'+str(i),1) for i in range(3)])-method(reference,0)
                bounds.append((float(low),float(high)))
            contrasts[metric+'__dsrl_mean_minus_'+reference]=dict(**interval(effects),
                missing_sensitivity=[float(np.mean([v[i] for v in bounds])) for i in (0,1)] if bounds else [None,None])
    return dict(complete_clusters=complete,planned_clusters=sorted(by_seed),contrasts=contrasts,
                dsrl_mean_scope='mean of these three fixed controllers, not independent environment replication')


def generate(output,p,execution_status=None):
    output=Path(output)
    plan=read(output/'case_manifest.json')
    records=[r for c in plan if (r:=completed(output,c))]
    with (output/'episodes.jsonl').open('w',encoding='utf-8') as stream:
        for r in records: stream.write(json.dumps(r,ensure_ascii=False)+'\n')
    with (output/'events.jsonl').open('w',encoding='utf-8') as stream:
        for r in records:
            path=next((f for f in r['artifact_hashes'] if f.endswith('events.json')),None)
            if path:
                for event in read(output/path): stream.write(json.dumps(dict(case_id=r['case']['id'],**event))+'\n')
    groups=defaultdict(list)
    for r in records:
        c=r['case'];groups[(c['phase'],c['task'],c['policy'],c['prompt_id'])].append(r)
    summary=[]
    for (phase,task,policy,prompt),rs in sorted(groups.items()):
        cs=[c for c in plan if (c['phase'],c['task'],c['policy'],c['prompt_id'])==(phase,task,policy,prompt)]
        valid=[r for r in rs if r['status'] in ['COMPLETE','POLICY_FAILURE']]
        attempts=sum((output/'attempts'/c['id']).exists() or any(r['case']['id']==c['id'] for r in rs) for c in cs)
        row=dict(phase=phase,task=task,policy=policy,prompt=prompt,planned=len(cs),attempted=attempts,
            completed=len(rs),valid=len(valid),invalid=sum(r['status']=='INVALID_CONSTRUCTION' for r in rs),
            engineering_missing=len(cs)-len(rs))
        for metric in ['correct_first_grasp','active_goal_ever_success','both_goals_ever','active_goal_final',
                       'active_goal_stable_released_success','active_goal_first_completion_without_prior_wrong_goal',
                       'restricted_time_to_success']:
            values=[r[metric] for r in valid if r.get(metric) is not None]
            row[metric]=float(np.mean(values)) if values else None
        for klass in p['e1']['first_grasp_classes']:
            row['first_grasp_'+klass]=sum(r['first_grasp_class']==klass for r in valid)
        for g in ['A','B']:
            row['goal_'+g+'_ever']=float(np.mean([r['goal_ever'][g] for r in valid if g in r['goal_ever']])) if any(g in r['goal_ever'] for r in valid) else None
        summary.append(row)
    if summary:
        with (output/'summary_by_condition.csv').open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    paired={phase:paired_contrasts(records,plan,phase) for phase in ['e1_dev','e1_confirm']}
    valid=[r for r in records if r['status'] in ['COMPLETE','POLICY_FAILURE']]
    interfaces={}
    confirmation_clusters=paired['e1_confirm']['complete_clusters']
    for policy in [x['id'] for x in p['e1']['policies']]:
        rs=[r for r in valid if r['case']['phase']=='e1_confirm' and r['case']['policy']==policy
            and r['case']['reset_seed'] in confirmation_clusters]
        bygoal={g:[r for r in rs if r['case']['requested_goal_id']==g] for g in ['A','B']}
        def rate(goal,metric):
            values=bygoal[goal];return float(np.mean([r[metric] for r in values])) if values else None
        confusion={g:{k:sum(r['first_grasp_class']==k for r in bygoal[g]) for k in p['e1']['first_grasp_classes']} for g in ['A','B']}
        def grasp_prob(g,k): return confusion[g][k]/len(bygoal[g]) if bygoal[g] else 0
        both=[]
        for seed in paired['e1_confirm']['complete_clusters']:
            for sample in [0,1]:
                branches=[r for r in rs if r['case']['reset_seed']==seed and r['case']['sampling_seed']==sample]
                both.append(all(r['correct_first_grasp'] for r in branches) and len(branches)==2)
        interfaces[policy]=dict(correct_first={g:rate(g,'correct_first_grasp') for g in bygoal},
            success={g:rate(g,'active_goal_ever_success') for g in bygoal},confusion=confusion,
            L=.5*(grasp_prob('A','A')-grasp_prob('B','A')+grasp_prob('B','B')-grasp_prob('A','B')) if rs else None,
            pair_both_correct=float(np.mean(both)) if both else None,wrong_A_under_B=grasp_prob('B','A') if rs else None)
        cluster_values=defaultdict(list)
        for seed in confirmation_clusters:
            selected=[r for r in rs if r['case']['reset_seed']==seed]
            by_goal={g:[r for r in selected if r['case']['prompt_id']==g] for g in ['A','B']}
            def probability(goal,klass):
                return float(np.mean([r['first_grasp_class']==klass for r in by_goal[goal]]))
            cluster_values['L'].append(.5*(probability('A','A')-probability('B','A')+probability('B','B')-probability('A','B')))
            cluster_values['wrong_A_under_B'].append(probability('B','A'))
            for goal in ['A','B']:
                for metric in ['correct_first_grasp','active_goal_ever_success']:
                    cluster_values[goal+'__'+metric].append(float(np.mean([r[metric] for r in by_goal[goal]])))
            pairs=[]
            for sample in [0,1]:
                branches=[r for r in selected if r['case']['sampling_seed']==sample]
                pairs.append(all(r['correct_first_grasp'] for r in branches))
            cluster_values['pair_both_correct'].append(float(np.mean(pairs)))
        interfaces[policy]['complete_cluster_statistics']={key:interval(values) for key,values in cluster_values.items()}
    paired['e1_goal_bias_contrasts']={}
    for policy in ['dsrl_s0','dsrl_s1','dsrl_s2']:
        for reference in ['base','bounded_random']:
            effects=[]
            for seed in confirmation_clusters:
                selected=[r for r in valid if r['case']['phase']=='e1_confirm' and r['case']['reset_seed']==seed and r['case']['prompt_id']=='B']
                a=[r['first_grasp_class']=='A' for r in selected if r['case']['policy']==policy]
                b=[r['first_grasp_class']=='A' for r in selected if r['case']['policy']==reference]
                effects.append(float(np.mean(a)-np.mean(b)))
            paired['e1_goal_bias_contrasts'][policy+'_minus_'+reference]=interval(effects)
    stage={}
    for task in p['e2']['tasks']:
        rs=[r for r in valid if r['case']['phase']=='e2_dev' and r['case']['task']==task['id']]
        by={(r['case']['reset_seed'],r['case']['condition']):r for r in rs}
        complete=[s for s in p['seed_splits']['e2_development'] if all((s,c) in by for c in p['e2']['conditions'])]
        stage[task['id']]=dict(complete_clusters=complete,conditions={},contrasts={})
        for cond in p['e2']['conditions']:
            cr=[r for r in rs if r['case']['condition']==cond]
            stage[task['id']]['conditions'][cond]=dict(n=len(cr),successes=sum(r['active_goal_ever_success'] for r in cr),
                first_stage=sum(r['first_stage_completion'] is not None for r in cr),
                failure_diagnostics={k:sum(bool(r['failure_diagnostics'][k]) for r in cr) for k in cr[0]['failure_diagnostics']} if cr else {})
        for ref in ['global','fixed_stage']:
            stage[task['id']]['contrasts']['oracle_minus_'+ref]=interval([
                float(by[(s,'oracle_stage')]['active_goal_ever_success'])-float(by[(s,ref)]['active_goal_ever_success']) for s in complete])
    paired['interfaces']=interfaces;paired['e2_development']=stage
    # Macro bootstrap resamples independently within each task, then weights tasks equally.
    macro={}
    for ref in ['global','fixed_stage']:
        values=[]
        rng=np.random.default_rng(99021)
        for task in stage.values():
            rs=[r for r in valid if r['case']['task']==next(k for k,v in stage.items() if v is task)]
            by={(r['case']['reset_seed'],r['case']['condition']):r for r in rs}
            x=np.array([float(by[(s,'oracle_stage')]['active_goal_ever_success'])-float(by[(s,ref)]['active_goal_ever_success']) for s in task['complete_clusters']])
            if len(x): values.append((float(x.mean()),x[rng.integers(len(x),size=(10000,len(x)))].mean(axis=1)))
        if len(values)==2:
            macro[ref]=dict(estimate=float(np.mean([v[0] for v in values])),ci95=np.quantile(np.mean([v[1] for v in values],axis=0),[.025,.975]).tolist())
    paired['e2_macro_oracle_contrasts']=macro
    atomic(output/'paired_cluster_statistics.json',paired)
    if len(paired['e1_confirm']['complete_clusters'])>=32:
        from tools.research_media import figures
        figures(output,summary,paired)
    usable=interfaces.get('base',{})
    enough=len(paired['e1_confirm']['complete_clusters'])>=32
    label='INCONCLUSIVE';next_intervention='collect missing planned evidence before choosing a training intervention'
    if enough:
        correct=usable['correct_first'];success=usable['success']
        next_intervention='goal interface repair SFT' if any(correct[g]<.6 for g in ['A','B']) else 'action expert execution repair'
        if all(correct[g]>=.6 and success[g]>=.2 for g in ['A','B']) and usable['pair_both_correct']>=.5 and usable['L']>=.2:
            label='LANGUAGE_INTERFACE_USABLE';next_intervention='action expert execution repair'
        if any(correct[g]>=.6 and success[g]<.2 for g in ['A','B']):
            label='EXECUTION_BOTTLENECK';next_intervention='action expert execution repair'
        affected=[k for k,v in interfaces.items() if k.startswith('dsrl') and v['wrong_A_under_B']-usable['wrong_A_under_B']>=.2]
        if len(affected)>=2:
            label='GOAL_BIAS_CANDIDATE';next_intervention='goal interface repair SFT'
    gates={}
    for task,value in stage.items():
        conditions=value['conditions'];o=conditions['oracle_stage'];g=conditions['global']
        net=o['successes']-g['successes']
        localized=o['first_stage']>0 and g['first_stage']>g['successes']
        gates[task]=dict(passes=o['successes']>=4 and net>=2 and localized,net_saved=net,localized_evidence=localized)
    if enough and label in ['INCONCLUSIVE','LANGUAGE_INTERFACE_USABLE'] and any(g['passes'] for g in gates.values()):
        label='STAGE_SIGNAL_CANDIDATE';next_intervention='confirm the stage information effect on both tasks before stage estimator training'
    required=[c for c in plan if c['phase']!='e0']
    done={r['case']['id'] for r in records}
    all_done=all(c['id'] in done for c in required) and (output/'E0_PASS.json').exists()
    execution_status=execution_status or ('COMPLETE' if all_done else 'ENGINEERING_ERROR')
    decision=dict(execution_status=execution_status,scientific_decision=label,minimum_complete_clusters_met=enough,
        next_priority=next_intervention,e2_future_confirmation_gate=gates,
        later_training_started=False,limitations=['Scientific gates are not significance tests','E2 is development evidence',
            'No reserved-set results read','Three controllers are fixed checkpoint comparisons'],
        planned_cases=len(required),completed_cases=sum(c['id'] in done for c in required),
        valid_cases=sum(r['case']['phase']!='e0' for r in valid))
    import subprocess
    from tools.research_core import file_sha
    try: analysis_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    except (OSError,subprocess.SubprocessError):analysis_commit=None
    atomic(output/'analysis_manifest.json',dict(analysis_commit=analysis_commit,analysis_file_sha256=file_sha(__file__),
        evaluation_code_commit=read(output/'run_manifest.json')['code_commit'],protocol_metrics_unchanged=True,
        independent_cluster='reset within task',confirmation_primary_subset='complete clusters across all paired conditions'))
    atomic(output/'decision.json',decision)
    lines=['# 语言目标与阶段诊断报告','',f"执行状态：{execution_status}；科学判断：{label}。",
        f"代码：`{read(output/'run_manifest.json')['code_commit']}`。完整配置见 resolved_protocol.json，逐案例见 episodes.jsonl，逐步见 events.jsonl。",
        '',f"计划 {len(required)}，完成 {decision['completed_cases']}，有效 {decision['valid_cases']}；确认集完整 reset clusters {len(paired['e1_confirm']['complete_clusters'])}/40。",
        '', 'E1 固定奶酪任务物理场景，400步；A 奶酪入托盘，B 黄油入同一托盘；所有20次查询保持指定文本。',
        'Base、bounded random 与三份固定500000-update DSRL控制器共享reset和sampling key。首次稳定抓取与官方目标曾经成功分开报告。',
        '', '| E1 text | requested goal | exact prompt |','| --- | --- | --- |']
    for prompt in p['e1']['goals']+p['e1']['auxiliary_prompts']:
        lines.append(f"| {prompt['id']} | {prompt.get('requested_goal_id',prompt['id'] if prompt['id'] in ['A','B'] else None)} | `{prompt['text']}` |")
    lines += [
        '', '![两条独立指标轴](figures/e1_two_axes.png)', '![首次稳定抓取混淆矩阵](figures/e1_confusion.png)',
        '', '| split/task/policy/prompt | valid/planned | correct first grasp | active goal ever |', '| --- | --- | --- | --- |']
    for row in summary:
        lines.append(f"| {row['phase']}/{row['task']}/{row['policy']}/{row['prompt']} | {row['valid']}/{row['planned']} | {row['correct_first_grasp']} | {row['active_goal_ever_success']} |")
    lines+=['','## 辅助文本行为对照','',
        '辅助文本仅开发集，每条件最多5个reset。空文本/无关文本没有requested goal；下表是行为频率，不称任务成功率。',
        '![辅助文本行为](figures/e1_auxiliary_behavior.png)',
        '', '| policy / text | valid/planned | grasp A/B/OTHER/MULTIPLE/NONE | goal A ever | goal B ever |',
        '| --- | --- | --- | --- | --- |']
    for row in summary:
        if row['phase']=='e1_aux':
            counts='/'.join(str(row['first_grasp_'+k]) for k in p['e1']['first_grasp_classes'])
            lines.append(f"| {row['policy']} / {row['prompt']} | {row['valid']}/{row['planned']} | {counts} | {row['goal_A_ever']} | {row['goal_B_ever']} |")
    lines += ['', '## 配对证据','', '统计单位是reset，保留文本/采样/控制器分支配对。10000次cluster bootstrap，seed99021；退化区间不代表确定性。',
              '![配对效应区间](figures/e1_paired_effects.png)',
              '效应量、95%区间和缺失值最坏/最好界见 paired_cluster_statistics.json。辅助空文本/无关文本没有requested goal，不计算任务成功率。',
              '', '## 阶段诊断','', 'T1：打开上抽屉再放入碗；T2：前方黄油入初始打开的上抽屉再关闭。',
              'global完整目标、fixed按开发global首阶段完成时间中位数取整并限制100–300步、oracle按当前复合谓词连续3步并允许回退，仅查询边界更新。',
              'Oracle为真值信息上界，不是可部署阶段估计器。全部阶段文本、实际切换和失败标签已逐步保存。',
              '![阶段与完整目标](figures/e2_stages.png)',
              '```json',json.dumps(stage,ensure_ascii=False,indent=2),'```','', '## 成本与下一步','',
              '物理步含短测、重放与重试；budget.json区分策略/重放/构造/不确定上界。resources.jsonl记录RSS/显存/GPU利用率。',
              f'首要下一步：{next_intervention}。本批没有SFT、PPO、Instruction RL或Joint RL训练。']
    # Exact stage prompts and schedule, alongside the corresponding diagnostic text.
    stage_lines=['','| task | condition/phase | exact prompt |','| --- | --- | --- |']
    for task in p['e2']['tasks']:
        stage_lines.append(f"| {task['id']} | global | `{p['e2']['prompt_global_template'].format(goal=task['global_goal'])}` |")
        for i in [0,1]:
            stage_lines.append(f"| {task['id']} | stage {i} | `{p['e2']['prompt_stage_template'].format(goal=task['global_goal'],stage=task['stage_'+str(i)])}` |")
    if (output/'calibration.json').exists():stage_lines+=['', '`calibration`: '+json.dumps(read(output/'calibration.json')['switch_steps'])]
    index=lines.index('## 阶段诊断')+1
    lines[index:index]=stage_lines
    # Link the first-two-reset videos for every configuration, including ceiling cases
    # with no win/loss representative; filenames stay tied to immutable case artifacts.
    video_groups=defaultdict(list)
    for record in records:
        c=record['case']
        for filename in record['artifact_hashes']:
            if filename.endswith('rollout.mp4'):
                video_groups[(c['phase'],c['task'],c['policy'],c['prompt_id'])].append((c['reset_seed'],c['sampling_seed'],filename))
    for phases,anchor,title in [(['e1_dev','e1_confirm'],'## 辅助文本行为对照','E1各配置预定前两个reset录像'),
                               (['e1_aux'],'## 配对证据','辅助配置预定前两个reset录像'),
                               (['e2_dev'],'## 成本与下一步','E2各配置预定前两个reset录像')]:
        index_lines=['',title+'：','', '| phase / task / policy / text | fixed reset videos |','| --- | --- |']
        for key,values in sorted(video_groups.items()):
            if key[0] in phases:
                links='；'.join(f'[reset {seed}, sampling {sample}]({filename})' for seed,sample,filename in sorted(values))
                index_lines.append('| '+' / '.join(key)+' | '+links+' |')
        index=lines.index(anchor);lines[index:index]=index_lines+['']
    selection_path=output/'media/selection.json'
    if selection_path.exists():
        selection=read(selection_path)
        for task,anchor in [('E1','## 配对证据'),('T1','## 成本与下一步'),('T2','## 成本与下一步')]:
            media_lines=['',f'{task} 预定规则选择的保存动作重放（非重新采样）：']
            for row in selection:
                if row['task']==task:
                    labels=[]
                    lookup={r['case']['id']:r['case'] for r in records}
                    for ident in row['cases']:
                        c=lookup[ident]
                        labels.append(f"[{c['policy']} / {c['prompt_id']} / reset {c['reset_seed']} / sampling {c['sampling_seed']}](media/{ident}.mp4)")
                    media_lines.append(f"- {row['comparison']} / {row.get('goal','')} / {row['stratum']}: "+'；'.join(labels))
            index=lines.index(anchor)
            lines[index:index]=media_lines+['']
    if (output/'budget.json').exists(): lines+=['','```json',json.dumps(read(output/'budget.json'),ensure_ascii=False,indent=2),'```']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return decision


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input',required=True);parser.add_argument('--protocol',default='configs/research_protocol_v1.json')
    args=parser.parse_args();print(json.dumps(generate(args.input,read(args.protocol)),indent=2))
