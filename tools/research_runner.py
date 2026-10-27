"""Locked, resumable first_batch scientific evaluations; no learning operations."""
import argparse
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np

from tools.research_core import (Budget, BudgetExceeded, GraspTracker, StageTracker, atomic,
    canonical, case_plan, completed, file_sha, lock, read, sha, validate)
from tools.stage_probe import prompt_for, calibrate


def commit():
    return subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()


def scan_seeds(p,root,output):
    """Inspect existing configuration/state manifests, never reserved case outcomes."""
    target=set(v for k,values in p['seed_splits'].items() if k.startswith(('e0_','e1_','e2_'))
               for v in values)
    collisions=[]
    inspected=[]
    def walk(value,path):
        if isinstance(value,dict):
            for k,v in value.items():
                if k in ['reset_seed','reset_seeds','eval_reset_seeds']:
                    values=v if isinstance(v,list) else [v]
                    overlap=target.intersection(x for x in values if isinstance(x,int))
                    if overlap: collisions.append(dict(path=str(path),field=k,seeds=sorted(overlap)))
                elif isinstance(v,(dict,list)): walk(v,path)
        elif isinstance(value,list):
            for v in value: walk(v,path)
    for directory in sorted((root/'runs').glob('job-*')):
        if directory.resolve()==output.resolve(): continue
        files={directory/'config.json',directory/'run.json'}
        for pattern in ['common_states/manifest.json','*/common_states/manifest.json','state_manifest.json','*/config.json']:
            files.update(directory.glob(pattern))
        for path in sorted(files):
            if path.is_file():
                walk(read(path),path)
                inspected.append(dict(path=str(path),sha256=file_sha(path)))
    result=dict(scanned=inspected,collisions=collisions,passed=not collisions,
                reserved_outcomes_read=False,requested_reset_seeds=sorted(target))
    atomic(output/'seed_collision_audit.json',result)
    assert not collisions, f'Seed namespace collision before new results; resolve whole blocks first: {collisions}'


def rollout(b,case,budget,calibration,attempt):
    from tools.language_goal_probe import InvalidConstruction
    env,obs=b.reset(case['task'],case['reset_seed'])
    actions_out,noises,keys,trace,schedule=[],[],[],[],[]
    frames=[]
    path=b.output/'attempts'/case['id']/str(attempt)
    path.mkdir(parents=True,exist_ok=False)
    first={}; ever={}; last={}; stable={}; stable_ever={}; first_contact={}; first_lift={}
    tracker=StageTracker()
    result=dict(case=case,attempt=attempt,input_protocol_sha256=sha(b.p),code_commit=commit())
    try:
        result['common_state']=b.common(case,env,obs)
        initial=b.events(env,case['task'])
        trace.append(dict(step=0,**initial))
        anchor=initial['positions']
        grasp=GraspTracker(initial['grasp'])
        for goal in initial['goals']:
            first[goal]=None;ever[goal]=False;stable[goal]=0;stable_ever[goal]=False
        # Every prediction uses only RGB/proprio/text and explicit noise; no truth enters infer/actor.
        for start in range(0,case['horizon'],20):
            text,phase=prompt_for(b.p,case,tracker,start,calibration)
            audit=b.audit_prompt(text)
            noise,key=b.noise(obs,case['policy'],case['reset_seed'],case['sampling_seed'],start//20)
            proposed=b.infer(obs,text,noise)
            noises.append(noise);keys.append(key)
            schedule.append(dict(step=start,text=text,phase=phase,tokens=audit['tokens'],mask=audit['mask']))
            reservation=case['id']+':'+str(attempt)+':'+str(start)
            budget.reserve(reservation,min(20,case['horizon']-start),'policy')
            actual=0
            try:
                for j in range(min(20,case['horizon']-start)):
                    action=proposed[j].copy()
                    obs,_,native_done,_=env.step(action)
                    actual+=1;actions_out.append(action)
                    if not np.isfinite(env.get_sim_state()).all(): raise FloatingPointError('Nonfinite simulator state')
                    step=len(actions_out)
                    event=b.events(env,case['task'])
                    event.update(step=step,native_done=bool(native_done),active_text=text,phase=phase)
                    trace.append(event)
                    grasp.update(event['grasp'],step)
                    for obj,value in event['contact'].items():
                        if value and obj not in first_contact: first_contact[obj]=step
                        if event['positions'][obj][2] > anchor[obj][2]+.02 and obj not in first_lift: first_lift[obj]=step
                    for goal,value in event['goals'].items():
                        if value and first[goal] is None: first[goal]=step
                        ever[goal] |= value
                        last[goal]=value
                        obj=next((g['object'] for g in b.p['e1']['goals'] if g['id']==goal),None)
                        released=value and (obj is None or not event['grasp'][obj])
                        stable[goal]=stable[goal]+1 if released else 0
                        stable_ever[goal] |= stable[goal]>=3
                    video_seeds=b.p['seed_splits']['e0_technical'][:2] if case['phase']=='e0' else (
                        b.p['seed_splits']['e2_development'][:2] if case['task']!='E1' else
                        b.p['seed_splits']['e1_confirmation' if case['phase']=='e1_confirm' else 'e1_development'][:2])
                    if case['reset_seed'] in video_seeds and step%4==0:
                        frames.append(obs['agentview_image'][::-1,::-1].copy())
                    if case['task']!='E1':
                        # Complete official goal terminates before any oracle update.
                        if event['official_success']: break
                        tracker.observe(event['stage_0_done'],step)
                # E1 deliberately ignores both native goal and TimeLimit done.
            finally:
                budget.settle(reservation,actual)
            b.logger.log(dict(phase=case['phase'],**{'research/current_case':case['id'],
                'research/current_steps':len(actions_out),'research/physical_steps':budget.state['recorded_physical_steps']}),0)
            if case['task']!='E1' and event['official_success']: break
        if case['task']=='E1':
            assert len(actions_out)==case['horizon'] and len(schedule)==case['horizon']//20
            assert all(s['text']==schedule[0]['text'] for s in schedule), 'Prompt returned to original'
        active=case['requested_goal_id'] if case['task']=='E1' else 'global'
        active_success=ever.get(active) if active else None
        result.update(status='COMPLETE' if active_success or active is None else 'POLICY_FAILURE',
            executed_env_steps=len(actions_out),vla_queries=len(noises),
            first_grasp_class=grasp.first_class,first_grasp_step=grasp.first_step,first_grasp_objects=grasp.first_objects,
            correct_first_grasp=(grasp.first_class==active) if active and case['task']=='E1' else None,
            first_goal_step=first,goal_ever=ever,goal_final=last,active_goal_ever_success=active_success,
            active_goal_final=last.get(active) if active else None,active_goal_first_step=first.get(active) if active else None,
            restricted_time_to_success=first.get(active) or case['horizon'] if active else None,
            active_goal_stable_released_success=stable_ever.get(active) if active else None,
            both_goals_ever=all(ever.values()) if case['task']=='E1' else None,
            first_contact=first_contact,first_lift=first_lift,first_stage_completion=tracker.first_complete,
            oracle_transitions=tracker.transitions,
            termination_reason='official_goal' if case['task']!='E1' and active_success else 'fixed_horizon',
            goal_reversal={g:any(e['goals'][g] and not n['goals'][g] for e,n in zip(trace,trace[1:])) for g in ever})
        if active and case['task']=='E1':
            wrong='B' if active=='A' else 'A'
            result['active_goal_first_completion_without_prior_wrong_goal']=bool(first[active] is not None and
                (first[wrong] is None or first[active]<first[wrong]))
        if case['task']!='E1':
            result['failure_diagnostics']=dict(
                wrong_butter_grasp=any(e['grasp'].get('butter_2',False) for e in trace),
                premature_close=any(e.get('drawer_closed',False) and not e.get('stage_0_done',False) for e in trace[1:]) if case['task']=='T2' else False,
                stage_0_reached=tracker.first_complete is not None,
                placed_not_closed=bool(case['task']=='T2' and trace[-1].get('stage_0_done') and not active_success),
                repeated_stage_opening=sum(not a.get('drawer_open',False) and z.get('drawer_open',False) for a,z in zip(trace,trace[1:])),
                drop_events=sum(a['grasp'][obj] and not z['grasp'][obj] and z['positions'][obj][2]<a['positions'][obj][2]-.02
                    for a,z in zip(trace,trace[1:]) for obj in a['grasp']),timeout=not active_success)
        np.savez_compressed(path/'arrays.npz',executed_actions=np.asarray(actions_out),noise_sequence=np.asarray(noises),
            sampling_keys=np.asarray(keys),final_physics=env.get_sim_state().copy())
        atomic(path/'events.json',trace)
        atomic(path/'prompts.json',schedule)
        if frames:
            import imageio.v2 as imageio
            imageio.mimsave(path/'rollout.mp4',frames,fps=5,codec='libx264')
        result['artifact_hashes']={str(f.relative_to(b.output)):file_sha(f) for f in sorted(path.iterdir()) if f.is_file()}
        atomic(b.output/'cases'/(case['id']+'.json'),result)
        return result
    except BaseException as error:
        atomic(path/'error.json',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),
                                     executed_steps=len(actions_out)))
        if actions_out:
            np.savez_compressed(path/'partial.npz',executed_actions=actions_out,noise_sequence=noises,sampling_keys=keys)
        atomic(path/'events.json',trace)
        raise
    finally:
        env.close()


def run_case(b,c,budget,calibration):
    from tools.language_goal_probe import InvalidConstruction
    previous=completed(b.output,c)
    if previous: return previous
    folder=b.output/'attempts'/c['id']
    attempts=sorted(int(p.name) for p in folder.glob('*') if p.is_dir())
    next_attempt=max(attempts,default=-1)+1
    for attempt in range(next_attempt,3):
        try:
            return rollout(b,c,budget,calibration,attempt)
        except InvalidConstruction as error:
            value=dict(case=c,status='INVALID_CONSTRUCTION',reason=str(error),artifact_hashes={},executed_env_steps=0)
            atomic(b.output/'cases'/(c['id']+'.json'),value)
            return value
        except BudgetExceeded:
            raise
        except Exception:
            if attempt==2: raise
            b.logger.log(dict(phase='technical_retry',**{'research/current_case':c['id'],'research/attempt':attempt+1}),0)
    raise RuntimeError('Technical retries exhausted for '+c['id'])


def e0_checks(b,cases,budget):
    subprocess.run([__import__('sys').executable,'-m','unittest','tools.test_research','-q'],check=True)
    # Same-observation language transformation and original action pipeline parity.
    env,obs=b.reset('E1',9000)
    try:
        a,btext=[g['text'] for g in b.p['e1']['goals']]
        inputs=[b.policy._input_transform(b.probe.to_input(copy.deepcopy(obs),
                __import__('types').SimpleNamespace(env='libero',task_description=text))) for text in [a,btext,'']]
        for key in ['state']:
            np.testing.assert_array_equal(inputs[0][key],inputs[1][key])
        for key in inputs[0]['image']:
            np.testing.assert_array_equal(inputs[0]['image'][key],inputs[1]['image'][key])
        assert not np.array_equal(inputs[0]['tokenized_prompt'],inputs[1]['tokenized_prompt'])
        assert not np.array_equal(inputs[0]['tokenized_prompt'],inputs[2]['tokenized_prompt'])
        noise,_=b.noise(obs,'base',9000,0,0)
        reference=np.asarray(b.policy.infer(b.probe.to_input(copy.deepcopy(obs),
            __import__('types').SimpleNamespace(env='libero',task_description=a)),noise=b.jax.numpy.asarray(noise))['actions'])
        np.testing.assert_allclose(reference[:20],b.infer(obs,a,noise)[:20],atol=1e-5,rtol=0)
        for policy in b.actors:
            n1,k1=b.noise(copy.deepcopy(obs),policy,9000,0,0)
            n2,k2=b.noise(copy.deepcopy(obs),policy,9000,0,0)
            np.testing.assert_array_equal(n1,n2)
        snap=b.snapshot(env,obs)
    finally: env.close()
    env,obs=b.reset('E1',9000)
    try: b.check_snapshot(snap,b.snapshot(env,obs))
    finally: env.close()
    records=[run_case(b,c,budget,{}) for c in cases]
    assert all(r['executed_env_steps']==40 and r['vla_queries']==2 for r in records)
    # Fresh reset plus recorded action replay, including controller history and both cameras.
    c=next(c for c in cases if c['policy']=='base' and c['reset_seed']==9000)
    saved=completed(b.output,c)
    arrays=np.load(b.output/next(f for f in saved['artifact_hashes'] if f.endswith('arrays.npz')))
    env,obs=b.reset('E1',9000)
    try:
        budget.reserve('e0_replay',40,'replay');actual=0
        try:
            for action in arrays['executed_actions']:
                obs,*_=env.step(action);actual+=1
        finally: budget.settle('e0_replay',actual)
        np.testing.assert_allclose(env.get_sim_state(),arrays['final_physics'],rtol=0,atol=1e-8)
        replay_snap=b.snapshot(env,obs)
    finally:env.close()
    env,obs=b.reset('E1',9000)
    try:
        budget.reserve('e0_replay_images',40,'replay');actual=0
        try:
            for action in arrays['executed_actions']:
                obs,*_=env.step(action);actual+=1
        finally:budget.settle('e0_replay_images',actual)
        b.check_snapshot(replay_snap,b.snapshot(env,obs))
    finally:env.close()
    # Predicate truth table: evaluation-only object placement, never a policy input.
    witness=[]
    for inside_a,inside_b in [(False,False),(True,False),(False,True),(True,True)]:
        env,obs=b.reset('E1',9001)
        try:
            region=env.env.object_states_dict['wooden_tray_1_contain_region'].get_geom_state()['pos'].copy()
            for i,(goal,inside) in enumerate(zip(b.p['e1']['goals'],[inside_a,inside_b])):
                if inside:
                    obj=env.env.objects_dict[goal['object']]
                    q=env.sim.data.get_joint_qpos(obj.joints[0]).copy()
                    q[:3]=region+np.array([(-1 if i==0 else 1)*.025,0,0])
                    env.sim.data.set_joint_qpos(obj.joints[0],q)
                    env.sim.data.set_joint_qvel(obj.joints[0],np.zeros(6))
            env.sim.forward()
            actual=b.events(env,'E1')['goals']
            assert actual==dict(A=inside_a,B=inside_b), (region,actual,inside_a,inside_b)
            witness.append(dict(expected=dict(A=inside_a,B=inside_b),observed=actual,
                                physics_sha256=sha(env.get_sim_state().tobytes())))
            if inside_a:
                budget.reserve('truth_step_'+str(inside_b),1,'construction');n=0
                try:
                    obs,_,done,_=env.step(np.zeros(7));n=1
                    assert done and env.env.ignore_done
                finally:budget.settle('truth_step_'+str(inside_b),n)
        finally:env.close()
    b.verify_frozen()
    result=dict(passed=True,short_branches=len(records),policy_steps=sum(r['executed_env_steps'] for r in records),
        action_pipeline_parity=True,prompt_observation_parity=True,prompt_tokens_different=True,empty_prompt_preserved=True,
        exact_reset_replay=True,exact_action_replay=True,all_objects_grasp_contact=True,controller_sampling_repeatable=True,
        persistent_B_both_queries=True,native_done_true_can_continue=True,predicate_truth_witnesses=witness,
        immutable_atomic_cases=True,resume_hash_validation=True,eval_updates=0,replay_insertions=0)
    lock(b.output/'E0_PASS.json',result)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',default='configs/research_protocol_v1.json')
    parser.add_argument('--mode',choices=['validate','plan','run'],default='run')
    parser.add_argument('--phase',choices=['e0'])
    parser.add_argument('--preset',choices=['first_batch'],default='first_batch')
    parser.add_argument('--output')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--gate-job')
    args=parser.parse_args()
    p=read(args.protocol)
    plan=case_plan(p)
    check=validate(p)
    if args.mode in ['validate','plan']:
        print(json.dumps(dict(protocol_id=p['protocol_id'],protocol_sha256=sha(p),**check,
            **({'cases':plan} if args.mode=='plan' else {})),ensure_ascii=False,indent=2))
        return
    assert args.output
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    # A shared advisory lock protects the one-GPU protocol even for direct CLI launches.
    import fcntl
    guard_path=Path(os.environ['DSRL_DATA_ROOT'])/'research'/p['protocol_id']/'allocation.lock'
    guard_path.parent.mkdir(parents=True,exist_ok=True)
    allocation_guard=guard_path.open('a')
    fcntl.flock(allocation_guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not args.resume: assert not (output/'run_manifest.json').exists(), 'Use --resume for an existing run'
    from tools.run_logger import RunLogger
    from tools.language_goal_probe import Backend
    code=commit()
    run=dict(commit=code,mode='research-smoke' if args.phase else 'research',protocol_file_sha256=file_sha(args.protocol),
        protocol_sha256=sha(p),model_source='s3://openpi-assets/checkpoints/pi0_libero',learning=False,
        python=__import__('sys').executable,hostname=os.uname().nodename,job_id=os.environ.get('SLURM_JOB_ID'))
    lock(output/'run_manifest.json',dict(code_commit=code,protocol_sha256=sha(p),case_manifest_sha256=sha(plan)))
    atomic(output/'run.json',run)
    lock(output/'case_manifest.json',plan)
    lock(output/'input_protocol.json',p)
    (output/'protocol.sha256').write_text(sha(p)+'\n')
    logger=RunLogger(output)
    shared=Path(os.environ['DSRL_DATA_ROOT'])/'research'/p['protocol_id']/'budget.json'
    shared.parent.mkdir(parents=True,exist_ok=True)
    # Bootstrap early engineering allocations made before the shared ledger was introduced.
    if not shared.exists():
        old=[]
        for runfile in (Path(os.environ['DSRL_DATA_ROOT'])/'runs').glob('job-*/run.json'):
            previous=read(runfile)
            if previous.get('protocol_sha256')==sha(p) and (runfile.parent/'budget.json').exists():
                old.append(read(runfile.parent/'budget.json'))
        if old:
            merged=copy.deepcopy(old[0])
            for item in old[1:]:
                assert not set(s['job_id'] for s in merged['sessions']).intersection(s['job_id'] for s in item['sessions']), 'Overlapping legacy budget; reconcile explicitly'
                for k in ['recorded_physical_steps','uncertain_steps','policy_steps','replay_steps','construction_steps']:merged[k]+=item[k]
                merged['sessions']+=item['sessions']
            atomic(shared,merged)
    budget=Budget(output/'budget.json',shared_path=shared)
    try:
        if not (output/'seed_collision_audit.json').exists(): scan_seeds(p,Path(os.environ['DSRL_DATA_ROOT']),output)
        assert read(output/'seed_collision_audit.json')['passed']
        if args.gate_job and not args.phase:
            gate=Path(os.environ['DSRL_DATA_ROOT'])/'runs'/('job-'+args.gate_job)
            gr=read(gate/'run.json')
            assert gr['commit']==code and gr['protocol_sha256']==sha(p) and read(gate/'E0_PASS.json')['passed']
            assert read(gate/'SUCCESS.json')['passed']
            atomic(output/'gate_check.json',dict(passed=True,job=args.gate_job,commit=code,protocol_sha256=sha(p)))
            # Shared budget already includes prerequisite and failed engineering allocations.
            gb=read(gate/'budget.json')
            lock(output/'prerequisite_budget.json',gb)
            assert {s['job_id'] for s in gb['sessions']}.issubset({s['job_id'] for s in budget.state['sessions']})
        b=Backend(p,output,logger)
        if args.gate_job and not args.phase:
            smoke_resolved=read(gate/'resolved_protocol.json')
            for field in ['model_hashes','controller_paths','prompt_token_audit','sampling_settings','task_name_to_id','bash_setup_sha256']:
                assert b.resolved[field]==smoke_resolved[field], f'E0 gate mismatch: {field}'
            assert file_sha(output/'checkpoint_manifest.json')==file_sha(gate/'checkpoint_manifest.json')
        atomic(output/'software_manifest.json',dict(code_commit=code,submodules=subprocess.check_output(
            ['git','submodule','status','--recursive'],text=True),dependencies={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}))
        resolved=dict(protocol=p,code_commit=code,**b.resolved)
        if args.phase:
            e0_checks(b,[c for c in plan if c['phase']=='e0'],budget)
            lock(output/'resolved_protocol.json',resolved)
            (output/'protocol.sha256').write_text(sha(resolved)+'\n')
        else:
            if args.gate_job:
                lock(output/'E0_PASS.json',read(gate/'E0_PASS.json'))
            else:
                e0_checks(b,[c for c in plan if c['phase']=='e0'],budget)
            for phase in ['e1_dev','e1_aux']:
                for c in [c for c in plan if c['phase']==phase]: run_case(b,c,budget,{})
            for c in [c for c in plan if c['phase']=='e2_dev' and c['condition']=='global']: run_case(b,c,budget,{})
            e2records=[completed(output,c) for c in plan if c['phase']=='e2_dev' and c['condition']=='global']
            calibration,evidence=calibrate(p,e2records)
            lock(output/'calibration.json',dict(switch_steps=calibration,evidence=evidence))
            for c in [c for c in plan if c['phase']=='e2_dev' and c['condition']!='global']: run_case(b,c,budget,calibration)
            b.verify_frozen()
            resolved.update(calibration=read(output/'calibration.json'),case_manifest_sha256=sha(plan),
                checkpoints_sha256=file_sha(output/'checkpoint_manifest.json'),prompt_audit_sha256=file_sha(output/'prompt_audit.json'),
                seed_audit_sha256=file_sha(output/'seed_collision_audit.json'))
            lock(output/'resolved_protocol.json',resolved)
            (output/'protocol.sha256').write_text(sha(resolved)+'\n')
            lock(output/'confirmation_lock.json',dict(resolved_protocol_sha256=sha(resolved),
                manifest_sha256=sha(plan),code_commit=code))
            for c in [c for c in plan if c['phase']=='e1_confirm']: run_case(b,c,budget,calibration)
            b.verify_frozen()
            from tools.research_media import replay_representatives
            replay_representatives(b,[r for c in plan if (r:=completed(output,c))],budget)
            from tools.research_stats import generate
            generate(output,p)
        budget.flush()
        atomic(output/'SUCCESS.json',dict(passed=True,mode=run['mode'],execution_status='COMPLETE'))
        logger.log(dict(phase='completed',**{'research/cases_completed':len(list((output/'cases').glob('*.json')))}),0)
    except BaseException as error:
        status='PARTIAL_BUDGET' if isinstance(error,BudgetExceeded) else 'ENGINEERING_ERROR'
        atomic(output/'FAILED.json',dict(type=type(error).__name__,error=str(error),execution_status=status,traceback=traceback.format_exc()))
        if not args.phase:
            from tools.research_stats import generate
            generate(output,p,execution_status=status)
        raise
    finally:
        budget.flush();logger.close()


if __name__=='__main__': main()
