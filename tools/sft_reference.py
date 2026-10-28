"""Original JAX model reference, in the preserved simulation environment."""
import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np

from tools.research_core import atomic, file_sha, read


def static_reference(output):
    import jax
    import jax.numpy as jnp
    from openpi.training import config
    from openpi.policies import policy_config
    from openpi.models.model import Observation
    from openpi_client import image_tools
    assert jax.default_backend() == 'gpu', jax.devices()
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    root=Path(os.environ['DSRL_DATA_ROOT'])/'runs/job-18584'
    p=read(root/'protocol.resolved.json') if (root/'protocol.resolved.json').exists() else read(root/'resolved_protocol.json')
    p=p.get('protocol',p)
    texts=[g['text'] for g in p['e1']['goals']]
    for t in p['e2']['tasks']:
        texts.extend(p['e2']['prompt_stage_template'].format(goal=t['global_goal'],stage=t['stage_'+str(i)]) for i in (0,1))
    assert len(texts)==6
    policy=policy_config.create_trained_policy(config.get_config('pi0_libero'),Path(os.environ['DSRL_MODEL_PATH']))
    policy._sample_kwargs={'num_steps':10}
    names=[f'E1__{i}' for i in range(3000,3004)]+[f'{task}__{i}' for task in ['T1','T2'] for i in range(4000,4003)]
    rng=np.random.default_rng(99025);zs=rng.normal(size=(2,1,50,32)).astype(np.float32)
    rows=[]
    for ni,name in enumerate(names):
        with np.load(root/'common_states'/(name+'.npz')) as raw:
            state=raw['proprio'].copy()
            images={k:np.asarray(image_tools.resize_with_pad(raw[src][::-1,::-1],224,224))
                    for k,src in [('image','agentview_image'),('wrist_image','wrist_image')]}
        for ti,text in enumerate(texts):
            inputs=policy._input_transform({'observation/state':state,'observation/image':images['image'],
                                          'observation/wrist_image':images['wrist_image'],'prompt':text})
            batched=jax.tree.map(lambda x:jnp.asarray(x)[None],inputs)
            observation=Observation.from_dict(batched)
            if ni in [0,4,7] and ti==0:
                from openpi.shared import nnx_utils
                raw_model=policy._sample_actions.__wrapped__.__self__
                prefix,pmask,par=nnx_utils.module_jit(raw_model.embed_prefix)(observation)
                np.savez_compressed(output/f'prefix_{ni}.npz',prefix=np.asarray(prefix).astype(np.float32),
                                    mask=np.asarray(pmask),ar_mask=np.asarray(par))
            for zi,z in enumerate(zs):
                action=np.asarray(policy._sample_actions(observation,noise=jnp.asarray(z),num_steps=10))[0]
                repeat=np.asarray(policy._sample_actions(observation,noise=jnp.asarray(z),num_steps=10))[0]
                arrays={'state':np.asarray(inputs['state']), 'tokens':np.asarray(inputs['tokenized_prompt']),
                        'mask':np.asarray(inputs['tokenized_prompt_mask']), 'noise':z[0], 'actions':action,'repeat':repeat}
                for key,value in inputs['image'].items():arrays['image__'+key]=np.asarray(value)
                for key,value in inputs['image_mask'].items():arrays['image_mask__'+key]=np.asarray(value)
                name_out=f'{ni:02d}-{ti}-{zi}.npz';np.savez_compressed(output/name_out,**arrays)
                rows.append(dict(file=name_out,observation=name,text=text,noise_index=zi,sha256=file_sha(output/name_out),
                                 internal_max=float(np.max(np.abs(action-repeat)))))
                print('REFERENCE',len(rows),120,flush=True)
    atomic(output/'manifest.json',dict(cases=rows,backend=str(jax.devices()),device_kind=jax.devices()[0].device_kind,gradient_updates=0,
           original_checkpoint=os.environ['DSRL_MODEL_PATH'],original_protocol_sha256=file_sha(root/('protocol.resolved.json' if (root/'protocol.resolved.json').exists() else 'resolved_protocol.json'))))
    import subprocess,sys
    atomic(output.parent/'run.json',dict(mode='sft-reference',learning=False,job_id=os.environ.get('SLURM_JOB_ID'),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),python=sys.executable))
    atomic(output.parent/'SFT_REFERENCE_PASS.json',dict(passed=True,cases=120,gradient_updates=0,
           manifest_sha256=file_sha(output/'manifest.json'),internal_max=max(r['internal_max'] for r in rows)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    args=parser.parse_args();static_reference(args.output)
