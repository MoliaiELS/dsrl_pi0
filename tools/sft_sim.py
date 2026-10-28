"""Old-environment simulation worker. Only RGB/proprio/text/noise cross IPC."""
import argparse
import copy
import os
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np

from tools.research_core import Budget, atomic, read, file_sha


def run(args):
    import jax
    from libero.libero.envs import OffScreenRenderEnv
    from examples.train_utils_sim import obs_to_qpos
    from openpi_client import image_tools
    from openpi.models.tokenizer import PaligemmaTokenizer
    from tools.control_probe import Probe
    from tools.language_goal_probe import Backend
    from tools.research_runner import rollout
    from tools.run_logger import RunLogger
    from tools.sft_data import bddl_for
    root=Path(os.environ['DSRL_DATA_ROOT'])/'runs/job-18584'
    p=read(root/'resolved_protocol.json')['protocol']
    output=args.output;output.mkdir(parents=True,exist_ok=True)
    logger=RunLogger(output)
    b=Backend.__new__(Backend);b.p=p;b.output=output;b.logger=logger;b.jax=jax;b.reference=None
    b.paths={'E1':bddl_for('A')};b.actors={}
    b.probe=SimpleNamespace(env_class=OffScreenRenderEnv,to_proprio=obs_to_qpos,variant=SimpleNamespace(env='libero'))
    b.probe.snapshot=lambda env,obs,prefix:Probe.snapshot(b.probe,env,obs,prefix)
    tokenizer=PaligemmaTokenizer(max_len=48)
    b.audit_prompt=lambda text:dict(zip(['tokens','mask'],[v.tolist() for v in tokenizer.tokenize(text)]))
    request=os.fdopen(args.request_fd,'wb',buffering=0);response=os.fdopen(args.response_fd,'rb',buffering=0)
    def infer(obs,text,noise):
        message=dict(state=obs_to_qpos(copy.deepcopy(obs),b.probe.variant),prompt=text,noise=noise,
            image=np.asarray(image_tools.resize_with_pad(obs['agentview_image'][::-1,::-1],224,224)),
            wrist=np.asarray(image_tools.resize_with_pad(obs['robot0_eye_in_hand_image'][::-1,::-1],224,224)))
        pickle.dump(message,request,protocol=5)
        result=pickle.load(response)
        assert result.shape==(50,7) and np.isfinite(result).all()
        return result
    b.infer=infer
    if args.kind=='alignment':
        cases=[c for c in read(root/'case_manifest.json') if c['phase']=='e1_dev' and c['policy']=='base']
        assert len(cases)==20
        # Replay the actual saved z arrays, rather than assuming CPU/GPU JAX
        # random.normal produces byte-identical floating-point values.
        sequences={}
        for case in cases:
            original=read(root/'cases'/(case['id']+'.json'))
            artifact=next(k for k in original['artifact_hashes'] if k.endswith('/arrays.npz'))
            assert file_sha(root/artifact)==original['artifact_hashes'][artifact]
            with np.load(root/artifact) as arrays:
                key=(case['reset_seed'],case['sampling_seed'])
                value=(arrays['noise_sequence'].copy(),arrays['sampling_keys'].copy())
                if key in sequences:np.testing.assert_array_equal(sequences[key][0],value[0])
                sequences[key]=value
        b.noise=lambda obs,policy,reset,sampling,query:tuple(v[query] for v in sequences[(reset,sampling)])
        save_common=b.common
        def common(case,env,obs):
            record=save_common(case,env,obs)
            with np.load(root/'common_states'/f"E1__{case['reset_seed']}.npz") as original:
                b.check_snapshot(original,b.snapshot(env,obs))
            return record
        b.common=common
    else:
        from tools.research_core import sha
        cases=[]
        for seed in read(args.config)['behavior_reset_seeds']:
            for goal in ['A','B']:
                case=dict(condition='constant',horizon=400,phase='e1_dev',policy='base',prompt_id=goal,
                    requested_goal_id=goal,reset_seed=seed,sampling_seed=0,task='E1')
                case['id']='sft_behavior__'+sha(case)[:20];cases.append(case)
        p['seed_splits']['e1_development']=read(args.config)['behavior_reset_seeds']
    atomic(output/'case_manifest.json',cases)
    budget=Budget(output/'budget.json',cap=400*len(cases),hours=2)
    results=[]
    try:
        for case in sorted(cases,key=lambda c:c['id']):
            result=rollout(b,case,budget,{},0);results.append(result)
            if args.kind=='alignment':
                name=f"E1__{case['reset_seed']}.npz"
                with np.load(root/'common_states'/name) as original,np.load(output/'common_states'/name) as actual:
                    b.check_snapshot(original,actual)
        atomic(output/'results.json',results)
        atomic(output/'SIMULATION_COMPLETE.json',dict(passed=True,episodes=len(results),physical_steps=budget.state['recorded_physical_steps']))
    finally:
        request.close();response.close();logger.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--request-fd',type=int,required=True);p.add_argument('--response-fd',type=int,required=True)
    p.add_argument('--kind',choices=['alignment','behavior'],required=True);p.add_argument('--config',required=True)
    run(p.parse_args())
