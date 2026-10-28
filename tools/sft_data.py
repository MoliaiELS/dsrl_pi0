"""Audited HDF5 trajectory splits and the original pi0_libero data convention."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import numpy as np

from tools.research_core import atomic, file_sha, read


PROMPTS = {'A': 'pick up the cream cheese and put it in the tray',
           'B': 'pick up the butter and put it in the tray'}


def source_rows(config, data_root):
    spec = read(config['data_source_manifest'])
    return [(('A' if 'cream_cheese' in r['path'] else 'B'),
             Path(data_root) / 'datasets/libero/sft_a_raw' / Path(r['path']).name, r) for r in spec['files']]


def fix_xml(text):
    """Relocate assets only; leave MJCF geometry, controller and state unchanged."""
    import robosuite
    import libero.libero
    roots = [Path(robosuite.__file__).parent / 'models/assets', Path(libero.libero.__file__).parent / 'assets']
    tree = ET.fromstring(text)
    for node in tree.iter():
        name = node.get('file')
        if not name or Path(name).is_file():
            continue
        suffix = name.split('/assets/', 1)[-1]
        matches = [root / suffix for root in roots if (root / suffix).is_file()]
        assert len(matches) == 1, name
        node.set('file', str(matches[0]))
    return ET.tostring(tree, encoding='unicode')


def bddl_for(goal):
    from libero.libero import get_libero_path
    filename = 'LIVING_ROOM_SCENE3_pick_up_the_' + ('cream_cheese' if goal == 'A' else 'butter') + '_and_put_it_in_the_tray.bddl'
    return Path(get_libero_path('bddl_files')) / 'libero_90' / filename


def original_transforms(model_root):
    from openpi.training import config, checkpoints
    from openpi import transforms
    c = config.get_config('pi0_libero')
    d = c.data.create(c.assets_dirs, c.model)
    stats = checkpoints.load_norm_stats(Path(model_root) / 'assets', d.asset_id)
    assert not d.use_quantile_norm
    normalise = transforms.compose([*d.data_transforms.inputs, transforms.Normalize(stats), *d.model_transforms.inputs])
    return normalise, stats


def prepare(config, data_root, output, render=False):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    from tools.run_logger import RunLogger
    logger = RunLogger(output)
    import subprocess
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    atomic(output / 'run.json', dict(commit=commit, mode='sft-data', job_id=os.environ.get('SLURM_JOB_ID'),
                                  learning=False, config_sha256=file_sha(config)))
    cfg = read(config)
    model_root = Path(data_root) / 'checkpoints/pi0_libero'
    pipeline, stats = original_transforms(model_root)
    norm_file = model_root / 'assets/physical-intelligence/libero/norm_stats.json'
    rng = np.random.default_rng(cfg['split_seed'])
    entries=[]; cameras=[]; source_files={}; tokens={}; reference={}
    try:
        for goal, path, spec in sorted(source_rows(cfg, data_root)):
            assert path.stat().st_size == spec['size'] and file_sha(path) == spec['lfs']['oid']
            source_files[goal] = dict(path=str(path), sha256=spec['lfs']['oid'])
            with h5py.File(path, 'r') as f:
                data=f['data']; names=sorted(data, key=lambda s:int(s.rsplit('_',1)[1]))
                assert len(names)==50
                problem=json.loads(data.attrs['problem_info'])
                assert problem['language_instruction'].lower().strip()==PROMPTS[goal]
                order=rng.permutation(len(names)); training=set(order[:40].tolist())
                from libero.libero.envs.env_wrapper import ControlEnv
                env=ControlEnv(bddl_file_name=str(bddl_for(goal)), use_camera_obs=False,
                               has_renderer=False, has_offscreen_renderer=False, ignore_done=True, control_freq=20)
                try:
                    env.seed(99030);env.reset()
                    for i,name in enumerate(names):
                        ep=data[name]; t=len(ep['actions'])
                        assert ep['actions'].shape==(t,7) and t>=50
                        assert int(ep['dones'][-1])==1 and int(ep['rewards'][-1])==1
                        assert np.isfinite(ep['actions'][...]).all() and np.isfinite(ep['states'][...]).all()
                        env.reset_from_xml_string(fix_xml(ep.attrs['model_file']))
                        env.set_init_state(ep['states'][-1])
                        success=bool(env.check_success())
                        assert success, (goal,name,'recorded terminal state does not satisfy official goal')
                        val_starts=np.unique(np.rint(np.linspace(0,t-50,min(cfg['training']['validation_windows_per_trajectory'],t-49))).astype(int)).tolist()
                        entries.append(dict(goal=goal, trajectory=name, split='train' if i in training else 'validation',
                                            frames=t, real_windows=t-49, validation_starts=val_starts, terminal_goal_verified=success))
                        logger.log({'phase':'auditing_demonstrations','sft/demonstrations_checked':len(entries)},0)
                finally: env.close()
                if render:
                    cameras.extend(audit_cameras(goal,data,names[:2],output))
                example=data[names[0]]
                raw=dict(state=np.concatenate([example['obs/ee_states'][0],example['obs/gripper_states'][0]]),
                         image=example['obs/agentview_rgb'][0], wrist=example['obs/eye_in_hand_rgb'][0],
                         actions=example['actions'][:50])
                inp={'observation/state':raw['state'], 'observation/image':raw['image'],
                     'observation/wrist_image':raw['wrist'],'prompt':PROMPTS[goal], 'actions':raw['actions'].copy()}
                check=pipeline(copy.deepcopy(inp))
                tokens[goal]=dict(ids=check['tokenized_prompt'].tolist(),mask=check['tokenized_prompt_mask'].tolist())
                for k,v in raw.items(): reference[goal+'__raw_'+k]=v
                reference[goal+'__state']=np.asarray(check['state'],dtype=np.float32)
                reference[goal+'__actions']=np.asarray(check['actions'],dtype=np.float32)
                reference[goal+'__image']=np.asarray(check['image']['base_0_rgb'])
                reference[goal+'__wrist']=np.asarray(check['image']['left_wrist_0_rgb'])
        image_ops={}
        if render:
            for camera in ['image','wrist']:
                choices={tuple(r[camera]['best_flip']) for r in cameras}
                assert len(choices)==1, (camera,cameras)
                image_ops[camera]=list(next(iter(choices)))
        manifest=dict(protocol_sha256=file_sha(config), config=cfg, code_commit=commit,
                      source_files=source_files, trajectories=entries, tokens=tokens, prompts=PROMPTS,
                      normalisation={k:dict(mean=v.mean.tolist(), std=v.std.tolist()) for k,v in stats.items()},
                      norm_stats_sha256=file_sha(norm_file), extra_delta_actions_first_six=True,
                      camera_audit=cameras, image_flip_axes=image_ops, camera_alignment_passed=render,
                      real_action_horizon=50, simulation_steps=0, model_forwards=0, gradient_updates=0)
        for goal in ['A','B']:
            verification=copy.deepcopy(manifest);verification['image_flip_axes']={}
            actual=transform_sample(verification,goal,reference[goal+'__raw_state'],
                dict(image=reference[goal+'__raw_image'],wrist=reference[goal+'__raw_wrist']),
                reference[goal+'__raw_actions'])
            for key in ['state','actions']:
                np.testing.assert_allclose(actual[key],reference[goal+'__'+key],atol=1e-6,rtol=0)
            for key in ['image','wrist']:
                np.testing.assert_array_equal(actual[key],reference[goal+'__'+key])
        manifest['original_data_transform_interop_passed']=True
        atomic(output/'data_manifest.json',manifest)
        np.savez_compressed(output/'data_transform_reference.npz',**reference)
        atomic(output/'SFT_DATA_PASS.json',dict(passed=True, manifest=str(output/'data_manifest.json'),
              manifest_sha256=file_sha(output/'data_manifest.json'), trajectories=100, train=80, validation=20,
              camera_alignment_passed=render, simulation_steps=0, model_forwards=0))
        logger.log({'phase':'completed','sft/demonstrations_checked':100},0)
        return manifest
    finally: logger.close()


def audit_cameras(goal,data,names,output):
    from libero.libero.envs import OffScreenRenderEnv
    from examples.train_utils_sim import obs_to_qpos
    from types import SimpleNamespace
    from PIL import Image
    rows=[]
    env=OffScreenRenderEnv(bddl_file_name=str(bddl_for(goal)), camera_heights=128,camera_widths=128)
    try:
        env.seed(99030);env.reset()
        for name in names:
            ep=data[name];env.reset_from_xml_string(fix_xml(ep.attrs['model_file']))
            obs=env.set_init_state(ep['states'][0])
            state=np.concatenate([ep['obs/ee_states'][0],ep['obs/gripper_states'][0]])
            np.testing.assert_allclose(obs_to_qpos(obs,SimpleNamespace(env='libero')),state,atol=1e-5,rtol=0)
            row=dict(goal=goal,trajectory=name)
            for key,obs_key,raw_key in [('image','agentview_image','agentview_rgb'),('wrist','robot0_eye_in_hand_image','eye_in_hand_rgb')]:
                target=obs[obs_key][::-1,::-1].astype(float); raw=ep['obs'][raw_key][0]
                choices=[]
                for y,x in [(False,False),(True,False),(False,True),(True,True)]:
                    img=raw[::-1 if y else 1,::-1 if x else 1]
                    choices.append((float(np.abs(target-img).mean()),[y,x]))
                choices.sort(key=lambda v:v[0]);best=choices[0]
                assert best[0]<15 and choices[1][0]-best[0]>3,(goal,name,key,choices)
                row[key]=dict(best_flip=best[1],mean_absolute_pixel_error=best[0],all_candidates=choices)
                if name==names[0]:
                    fixed=raw[::-1 if best[1][0] else 1,::-1 if best[1][1] else 1]
                    Image.fromarray(np.concatenate([fixed,target.astype(np.uint8)],axis=1)).save(output/(goal+'_'+key+'_data_vs_render.png'))
            rows.append(row)
    finally: env.close()
    return rows


def transform_sample(manifest,goal,state,images,actions=None):
    """NumPy/JAX CPU adapter matching the frozen author's OpenPI convention."""
    import jax
    import jax.numpy as jnp
    state=np.pad(np.asarray(state), (0,32-len(state)))
    raw_state=state.copy();norm=manifest['normalisation']
    state=(state-np.asarray(norm['state']['mean']))/(np.asarray(norm['state']['std'])+1e-6)
    result={'state':state.astype(np.float32)}
    for key,img in images.items():
        flips=manifest['image_flip_axes'].get(key,[False,False])
        img=np.ascontiguousarray(img[::-1 if flips[0] else 1,::-1 if flips[1] else 1])
        if img.shape[:2]!=(224,224):
            img=np.asarray(jnp.round(jax.image.resize(jnp.asarray(img),(224,224,3),method='linear')).clip(0,255).astype(jnp.uint8))
        result[key]=img
    result['tokens']=np.asarray(manifest['tokens'][goal]['ids'],dtype=np.int64)
    result['mask']=np.asarray(manifest['tokens'][goal]['mask'],dtype=bool)
    if actions is not None:
        action=np.pad(np.asarray(actions),((0,0),(0,32-actions.shape[-1])))
        action[:,:6]-=raw_state[None,:6]
        result['actions']=((action-np.asarray(norm['actions']['mean']))/(np.asarray(norm['actions']['std'])+1e-6)).astype(np.float32)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--render',action='store_true')
    args=parser.parse_args()
    prepare(args.config,os.environ['DSRL_DATA_ROOT'],args.output,args.render)
