"""Offline first-object/goal audit from recorded physics, without env stepping."""
import argparse
from collections import Counter
from pathlib import Path
import os
import h5py
import numpy as np
from tools.research_core import GraspTracker,atomic,read
from tools.sft_data import bddl_for,fix_xml


def main(args):
    from libero.libero.envs.env_wrapper import ControlEnv
    from tools.run_logger import RunLogger
    data=read(args.manifest);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    logger=RunLogger(out);rows=[];restores=0
    goals=read(Path(os.environ['DSRL_DATA_ROOT'])/'runs/job-18584/resolved_protocol.json')['protocol']['e1']['goals']
    for goal in ['A','B']:
        env=ControlEnv(bddl_file_name=str(bddl_for(goal)),use_camera_obs=False,has_renderer=False,
                       has_offscreen_renderer=False,ignore_done=True,control_freq=20)
        try:
            env.seed(99030);env.reset()
            with h5py.File(data['source_files'][goal]['path'],'r') as f:
                for row in [r for r in data['trajectories'] if r['goal']==goal]:
                    ep=f['data'][row['trajectory']];env.reset_from_xml_string(fix_xml(ep.attrs['model_file']))
                    objects=env.env.objects_dict;tracker=GraspTracker(objects)
                    first={g['id']:None for g in goals}
                    for step,state in enumerate(ep['states']):
                        env.set_init_state(state);restores+=1
                        grasp={k:bool(env.env._check_grasp(env.robots[0].gripper,v.contact_geoms)) for k,v in objects.items()}
                        tracker.update(grasp,step)
                        for g in goals:
                            if first[g['id']] is None and env.env._eval_predicate(g['predicate']):first[g['id']]=step
                    rows.append(dict(goal=goal,trajectory=row['trajectory'],split=row['split'],frames=row['frames'],
                        first_grasp=tracker.first_class,first_grasp_step=tracker.first_step,first_goal=first,
                        terminal_native_success=bool(env.check_success())))
                    logger.log({'phase':'offline_demo_order_audit','sft/demos_audited':len(rows),'sft/state_restorations':restores},0)
        finally:env.close()
    summary={g:dict(first_grasp=dict(Counter(r['first_grasp'] for r in rows if r['goal']==g)),
        terminal_success=sum(r['terminal_native_success'] for r in rows if r['goal']==g)) for g in ['A','B']}
    atomic(out/'DEMONSTRATION_ORDER_AUDIT.json',dict(completed=True,trajectories=rows,summary=summary,
        state_restorations=restores,simulation_steps=0,model_forwards=0,gradient_updates=0,
        limitation='contacts recomputed at saved physics states; no claim of bitwise historical sensor or action replay'))
    logger.close();print('DEMONSTRATION_ORDER_AUDIT',summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    main(p.parse_args())
