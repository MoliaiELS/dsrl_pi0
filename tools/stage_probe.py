"""Query-boundary stage conditions, with current-state hysteresis and rollback."""
from tools.research_core import StageTracker, fixed_schedule


def prompt_for(p, case, tracker, step, calibration):
    if case['task'] == 'E1':
        choices = p['e1']['goals'] + p['e1']['auxiliary_prompts']
        return next(x['text'] for x in choices if x['id'] == case['prompt_id']), None
    task=next(t for t in p['e2']['tasks'] if t['id'] == case['task'])
    if case['condition'] == 'global':
        return p['e2']['prompt_global_template'].format(goal=task['global_goal']), None
    phase = tracker.query(step) if case['condition'] == 'oracle_stage' else int(step >= calibration[task['id']])
    return p['e2']['prompt_stage_template'].format(goal=task['global_goal'],stage=task['stage_'+str(phase)]), phase


def calibrate(p, records):
    result={}
    evidence={}
    for task in p['e2']['tasks']:
        selected=[r for r in records if r['case']['task']==task['id'] and r['case']['condition']=='global']
        assert len(selected)==10, (task['id'],len(selected))
        times=[r.get('first_stage_completion') for r in selected if r['status'] in ['COMPLETE','POLICY_FAILURE']]
        result[task['id']]=fixed_schedule(times,task['fixed_fallback_step'])
        evidence[task['id']]=dict(first_stage_times=times,source_cases=[r['case']['id'] for r in selected],
                                fallback_used=not any(t is not None for t in times))
    return result,evidence
