"""Predetermined representative selection; replay saved actions, never resample policy."""
from pathlib import Path

import numpy as np

from tools.research_core import atomic, completed, file_sha, lock


def select_representatives(records):
    valid=[r for r in records if r['status'] in ['COMPLETE','POLICY_FAILURE']]
    e1={(r['case']['reset_seed'],r['case']['sampling_seed'],r['case']['prompt_id'],r['case']['policy']):r
        for r in valid if r['case']['phase']=='e1_confirm'}
    e2={(r['case']['task'],r['case']['reset_seed'],r['case']['condition']):r
        for r in valid if r['case']['phase']=='e2_dev'}
    selection=[]
    for policy in ['dsrl_s0','dsrl_s1','dsrl_s2']:
        for goal in ['A','B']:
            candidates=[]
            for key,base in e1.items():
                seed,sampling,prompt,method=key
                if prompt!=goal or method!='base' or (seed,sampling,prompt,policy) not in e1:continue
                other=e1[(seed,sampling,prompt,policy)]
                a,b=base['active_goal_ever_success'],other['active_goal_ever_success']
                label='win' if b and not a else 'loss' if a and not b else 'both_fail' if not a and not b else 'both_success'
                candidates.append((base['case']['id'],label,base,other))
            for label in ['win','loss','both_fail']:
                matches=sorted(x for x in candidates if x[1]==label)
                if matches:
                    _,_,base,other=matches[0]
                    selection.append(dict(comparison=policy+'_vs_base',task='E1',goal=goal,stratum=label,
                                          cases=[base['case']['id'],other['case']['id']]))
    for task in ['T1','T2']:
        candidates=[]
        for (t,seed,cond),base in e2.items():
            if t!=task or cond!='global' or (t,seed,'oracle_stage') not in e2:continue
            other=e2[(t,seed,'oracle_stage')]
            a,b=base['active_goal_ever_success'],other['active_goal_ever_success']
            label='win' if b and not a else 'loss' if a and not b else 'both_fail' if not a and not b else 'both_success'
            candidates.append((base['case']['id'],label,base,other))
        for label in ['win','loss','both_fail']:
            matches=sorted(x for x in candidates if x[1]==label)
            if matches:
                _,_,base,other=matches[0]
                selection.append(dict(comparison='oracle_vs_global',task=task,stratum=label,
                                      cases=[base['case']['id'],other['case']['id']]))
    return selection


def replay_representatives(b,records,budget):
    import imageio.v2 as imageio
    output=b.output/'media';output.mkdir(exist_ok=True)
    selection=select_representatives(records)
    lock(output/'selection.json',selection)
    lookup={r['case']['id']:r for r in records}
    verified=[]
    for ident in sorted(set(c for selection_row in selection for c in selection_row['cases'])):
        record=lookup[ident];case=record['case']
        video=output/(ident+'.mp4');check=output/(ident+'.json')
        if check.exists():
            from tools.research_core import read
            previous=read(check)
            assert file_sha(video)==previous['video_sha256']
            verified.append(previous);continue
        arrays_path=next(f for f in record['artifact_hashes'] if f.endswith('arrays.npz'))
        with np.load(b.output/arrays_path) as saved:
            actions=saved['executed_actions'].copy();expected=saved['final_physics'].copy()
        env,obs=b.reset(case['task'],case['reset_seed'])
        frames=[obs['agentview_image'][::-1,::-1].copy()]
        try:
            common=b.common(case,env,obs)
            for start in range(0,len(actions),20):
                key='media:'+ident+':'+str(start)
                budget.reserve(key,min(20,len(actions)-start),'replay');actual=0
                try:
                    for offset,action in enumerate(actions[start:start+20]):
                        obs,*_=env.step(action);actual+=1
                        if (start+offset+1)%4==0:frames.append(obs['agentview_image'][::-1,::-1].copy())
                finally:budget.settle(key,actual)
            np.testing.assert_allclose(env.get_sim_state(),expected,rtol=0,atol=1e-8)
            imageio.mimsave(video,frames,fps=5,codec='libx264')
            value=dict(case_id=ident,common_state=common,replayed_steps=len(actions),
                final_physics_max_error=float(np.max(np.abs(env.get_sim_state()-expected))),
                vla_calls=0,gradient_updates=0,video_sha256=file_sha(video))
            atomic(check,value);verified.append(value)
            b.logger.log({'phase':'rendering_recorded_actions','research/current_case':ident,
                          'research/physical_steps':budget.state['recorded_physical_steps']},0)
        finally:env.close()
    atomic(output/'replay_verification.json',verified)


def figures(output,summary,paired):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);folder=output/'figures';folder.mkdir(exist_ok=True)
    policies=['base','bounded_random','dsrl_s0','dsrl_s1','dsrl_s2']
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,metric,title in zip(axes,['correct_first_grasp','active_goal_ever_success'],['Correct first stable grasp','Official active goal ever satisfied']):
        for j,goal in enumerate(['A','B']):
            ys=[next((r[metric] for r in summary if r['phase']=='e1_confirm' and r['policy']==policy and r['prompt']==goal),None) for policy in policies]
            ax.bar(np.arange(5)+(j-.5)*.36,[v if v is not None else np.nan for v in ys],width=.36,label=goal)
        ax.set(ylim=(0,1.08),title=title,ylabel='Proportion')
        ax.set_xticks(np.arange(5),policies,rotation=15);ax.legend()
    fig.savefig(folder/'e1_two_axes.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,5,figsize=(16,3),layout='constrained')
    classes=['A','B','OTHER','MULTIPLE','NONE']
    for ax,policy in zip(axes,policies):
        data=np.array([[paired['interfaces'][policy]['confusion'][g][k] for k in classes] for g in ['A','B']])
        ax.imshow(data,cmap='Blues',vmin=0,vmax=80,aspect='auto')
        for i in range(2):
            for j in range(5):ax.text(j,i,str(data[i,j]),ha='center',va='center',fontsize=8)
        ax.set_title(policy);ax.set_xticks(range(5),classes,rotation=60,fontsize=8);ax.set_yticks([0,1],['prompt A','prompt B'])
    fig.savefig(folder/'e1_confusion.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    effects=paired['e1_confirm']['contrasts']
    for i,(name,value) in enumerate(effects.items()):
        if value['estimate'] is not None:
            mean=value['estimate'];low,high=value['ci95']
            ax.errorbar(mean,i,xerr=[[max(0,mean-low)],[max(0,high-mean)]],fmt='o')
    ax.set_yticks(range(len(effects)),list(effects),fontsize=8);ax.axvline(0,color='gray',linestyle='--')
    ax.set(xlabel='Paired effect (95% reset-cluster bootstrap)',title='Three fixed DSRL controllers mean contrasts')
    fig.savefig(folder/'e1_paired_effects.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,task in zip(axes,['T1','T2']):
        conditions=paired['e2_development'][task]['conditions']
        for j,key in enumerate(['successes','first_stage']):
            ax.bar(np.arange(3)+(j-.5)*.36,[conditions[c][key]/max(1,conditions[c]['n']) for c in ['global','fixed_stage','oracle_stage']],width=.36,label=key)
        ax.set_xticks(range(3),['global','fixed','oracle']);ax.set(ylim=(0,1.08),title=task+' (development n<=10)',ylabel='Proportion');ax.legend()
    fig.savefig(folder/'e2_stages.png',dpi=180);plt.close(fig)
    aux=[r for r in summary if r['phase']=='e1_aux']
    if aux:
        fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
        prompts=['A_paraphrase','empty','unrelated']
        for ax,key,title in zip(axes,['first_grasp_A','goal_A_ever','goal_B_ever'],
                                ['First stable grasp is A','Official goal A ever true','Official goal B ever true']):
            for j,policy in enumerate(['base','dsrl_s0']):
                rows=[next((r for r in aux if r['policy']==policy and r['prompt']==prompt),None) for prompt in prompts]
                values=[r[key]/r['valid'] if key.startswith('first_grasp') and r and r['valid'] else r[key] if r else np.nan for r in rows]
                ax.bar(np.arange(3)+(j-.5)*.36,values,width=.36,label=policy)
            ax.set_xticks(range(3),prompts,rotation=20);ax.set(ylim=(0,1.08),title=title,ylabel='Behavior frequency (n<=5)');ax.legend()
        fig.savefig(folder/'e1_auxiliary_behavior.png',dpi=180);plt.close(fig)
