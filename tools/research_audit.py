"""Read-only semantic closure of saved trajectories; no simulation/model forward."""
import argparse
from pathlib import Path

import numpy as np

from tools.research_core import GraspTracker, atomic, completed, file_sha, read, sha


def audit(output,p,require_complete=False,verify_weights=False):
    output=Path(output)
    plan=read(output/'case_manifest.json')
    manifest=read(output/'run_manifest.json')
    assert sha(plan)==manifest['case_manifest_sha256'] and sha(p)==manifest['protocol_sha256']
    records=[r for c in plan if (r:=completed(output,c))]
    checked=[];native_done_continued=0;common_hashes={}
    for r in records:
        c=r['case']
        if r['status']=='INVALID_CONSTRUCTION':continue
        assert r['code_commit']==manifest['code_commit']
        common=r['common_state'];name=common['id']
        if name not in common_hashes:common_hashes[name]=file_sha(output/'common_states'/(name+'.npz'))
        assert common_hashes[name]==common['file_sha256']
        files=r['artifact_hashes']
        events=read(output/next(k for k in files if k.endswith('events.json')))
        prompts=read(output/next(k for k in files if k.endswith('prompts.json')))
        with np.load(output/next(k for k in files if k.endswith('arrays.npz'))) as saved:
            actions=saved['executed_actions'];noise=saved['noise_sequence'];keys=saved['sampling_keys']
            steps=len(actions);queries=(steps+19)//20
            assert actions.shape==(steps,7) and np.isfinite(actions).all()
            assert noise.shape==(queries,1,50,32) and np.isfinite(noise).all()
            assert keys.shape==(queries,2)
            assert np.isfinite(saved['final_physics']).all()
            if c['policy']!='base':
                assert np.max(np.abs(noise))<=1.00001
                assert np.array_equal(noise,np.repeat(noise[:,:,:1,:],50,axis=2))
        assert len(events)==steps+1 and len(prompts)==queries
        assert r['executed_env_steps']==steps and r['vla_queries']==queries
        assert [e['step'] for e in events]==list(range(steps+1))
        assert [s['step'] for s in prompts]==list(range(0,steps,20))
        assert all(len(s['tokens'])==48 and len(s['mask'])==48 for s in prompts)
        tracker=GraspTracker(events[0]['grasp'])
        for event in events[1:]:tracker.update(event['grasp'],event['step'])
        assert r['first_grasp_class']==tracker.first_class
        assert r['first_grasp_step']==tracker.first_step
        assert r['first_grasp_objects']==tracker.first_objects
        for goal in events[0]['goals']:
            successes=[e['step'] for e in events if e['goals'][goal]]
            assert not events[0]['goals'][goal]
            assert r['goal_ever'][goal]==bool(successes)
            assert r['first_goal_step'][goal]==(successes[0] if successes else None)
            assert r['goal_final'][goal]==events[-1]['goals'][goal]
        if c['task']=='E1':
            assert steps==c['horizon'] and queries==c['horizon']//20
            text=next(x['text'] for x in p['e1']['goals']+p['e1']['auxiliary_prompts'] if x['id']==c['prompt_id'])
            assert all(s['text']==text for s in prompts)
            assert all(e['active_text']==text for e in events[1:])
            assert r['termination_reason']=='fixed_horizon'
            native_done_continued+=sum(e['native_done'] for e in events[1:-1])
            goal=c['requested_goal_id']
            if goal:
                assert r['correct_first_grasp']==(r['first_grasp_class']==goal)
                assert r['active_goal_ever_success']==r['goal_ever'][goal]
                wrong='B' if goal=='A' else 'A'
                a,b=r['first_goal_step'][goal],r['first_goal_step'][wrong]
                assert r['active_goal_first_completion_without_prior_wrong_goal']==bool(a is not None and (b is None or a<b))
                stable=0;passed=False
                obj=next(x['object'] for x in p['e1']['goals'] if x['id']==goal)
                for event in events[1:]:
                    stable=stable+1 if event['goals'][goal] and not event['grasp'][obj] else 0
                    passed|=stable>=3
                assert passed==r['active_goal_stable_released_success']
            else:
                assert r['correct_first_grasp'] is None and r['active_goal_ever_success'] is None
        else:
            success_steps=[e['step'] for e in events if e['official_success']]
            if success_steps:
                assert success_steps==[steps] and r['termination_reason']=='official_goal'
            else:assert steps==600 and r['termination_reason']=='fixed_horizon'
            assert r['active_goal_ever_success']==bool(success_steps)
            task=next(t for t in p['e2']['tasks'] if t['id']==c['task'])
            true=false=phase=0;first=None
            for start in range(steps):
                if start%20==0:
                    if c['condition']=='oracle_stage':
                        phase=1 if true>=3 else 0 if false>=3 else phase
                    elif c['condition']=='fixed_stage':
                        phase=int(start>=read(output/'calibration.json')['switch_steps'][c['task']])
                    else:phase=None
                    expected=(p['e2']['prompt_global_template'].format(goal=task['global_goal']) if phase is None else
                        p['e2']['prompt_stage_template'].format(goal=task['global_goal'],stage=task['stage_'+str(phase)]))
                    assert prompts[start//20]['text']==expected and prompts[start//20]['phase']==phase
                event=events[start+1]
                if event['official_success']:break
                value=event['stage_0_done'];true=true+1 if value else 0;false=false+1 if not value else 0
                if true>=3 and first is None:first=start+1
            assert first==r['first_stage_completion']
        assert r['status']==('POLICY_FAILURE' if r['active_goal_ever_success'] is False else 'COMPLETE')
        checked.append(c['id'])
    required=[c for c in plan if c['phase']!='e0']
    count=sum(c['id'] in {r['case']['id'] for r in records} for c in required)
    if require_complete:
        assert count==len(required) and read(output/'SUCCESS.json')['passed']
        assert read(output/'confirmation_lock.json')['code_commit']==read(output/'run_manifest.json')['code_commit']
    if verify_weights:
        resolved=read(output/'resolved_protocol.json')
        for item in resolved['model_hashes']['files']:
            assert file_sha(Path(resolved['model_path'])/item['path'])==item['sha256']
        for controller in read(output/'checkpoint_manifest.json').values():
            for relative,expected in controller['file_hashes'].items():assert file_sha(Path(controller['path'])/relative)==expected
    budget=read(output/'budget.json')
    assert budget['recorded_physical_steps']==sum(budget[k] for k in ['policy_steps','replay_steps','construction_steps'])
    assert budget['physical_steps_upper_bound']<=450000 and budget['allocated_gpu_hours_lower_bound']<=24
    before=read(output/'trainable_manifest.json')
    frozen=read(output/'frozen_check.json') if (output/'frozen_check.json').exists() else None
    if frozen:
        assert before['controllers_before']==frozen['controllers'] and before['vla_state_before']==frozen['vla_state']
    result=dict(passed=True,checked_valid_cases=len(checked),completed_main_cases=count,planned_main_cases=len(required),
        all_main_cases_present=count==len(required),native_done_steps_followed_by_further_steps=native_done_continued,
        saved_actions_and_noises_finite=True,prompts_and_goal_metrics_match_saved_trace=True,
        simulation_steps_added=0,vla_forwards_added=0,weight_files_verified=verify_weights,
        frozen_full_state_verified=True if frozen else None,
        frozen_check_status='VERIFIED_AT_LATEST_BOUNDARY' if frozen else 'PENDING_PHASE_BOUNDARY',
        audit_code_sha256=file_sha(__file__))
    atomic(output/('closure_check.json' if require_complete else 'trajectory_audit.json'),result)
    # Preserve the original artifact and explicitly correct its imprecise field name.
    atomic(output/'optimizer_restore_audit.json',dict(
        source='jaxrl2/agents/pixel_sac/pixel_sac_learner.py:187,204,213; tools/language_goal_probe.py',
        restoration_constructs_train_states_with_optimizer_state=True,optimizer_steps_during_evaluation=0,
        legacy_field_correction='trainable_manifest.optimizer_created=false is imprecise: SAC initialization constructs optimizer state before restoring the original checkpoint. No training update is performed.',
        original_manifest_sha256=file_sha(output/'trainable_manifest.json'),full_restored_state_unchanged=True if frozen else None))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input',required=True);parser.add_argument('--protocol',required=True)
    parser.add_argument('--require-complete',action='store_true');parser.add_argument('--verify-weights',action='store_true')
    a=parser.parse_args()
    print(__import__('json').dumps(audit(a.input,read(a.protocol),a.require_complete,a.verify_weights),indent=2))
