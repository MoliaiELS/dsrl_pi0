"""Project single-GPU adapter around the pinned RLinf native Pi0 implementation.

This is not the stock RLinf distributed SFT pipeline. Original simulation runs
in its preserved environment; model, native loss and selective dtypes are RLinf.
"""
import argparse
import hashlib
import json
import os
import pickle
import re
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

from tools.research_core import atomic, file_sha, read


def converted_path():
    return Path(os.environ['DSRL_DATA_ROOT'])/'checkpoints/pi0_libero_native_fab7'


def convert(logger):
    target=converted_path();source=Path(os.environ['DSRL_MODEL_PATH'])
    if (target/'CONVERSION_COMPLETE.json').exists():
        record=read(target/'CONVERSION_COMPLETE.json')
        assert file_sha(target/'model.safetensors')==record['converted_sha256']
        assert file_sha(source/'assets/physical-intelligence/libero/norm_stats.json')==record['norm_sha256']
        return
    assert not (target/'model.safetensors').exists(),'Unvalidated partial conversion; inspect before retry'
    logger.log({'phase':'converting_original_checkpoint'},0)
    norm=source/'assets/physical-intelligence/libero/norm_stats.json'
    subprocess.run([sys.executable,'-m','tools.sft_convert','--source',str(source),'--target',str(target)],check=True)
    assert file_sha(norm)==file_sha(target/'physical-intelligence/libero/norm_stats.json')
    atomic(target/'CONVERSION_COMPLETE.json',dict(converted_sha256=file_sha(target/'model.safetensors'),
           norm_sha256=file_sha(norm),rlinf_commit='fab7f407fbc611085eb163962e3c89c0ab424c1a',
           source_manifest_sha256=file_sha(source/'download_manifest.json')))


def load_model():
    import torch
    from omegaconf import OmegaConf
    from rlinf.models.embodiment.openpi import get_model
    assert torch.cuda.is_available(),'Allocated GPU cannot initialize CUDA'
    weights=converted_path()/'model.safetensors'
    signature=(weights.stat().st_size,weights.stat().st_mtime_ns)
    if getattr(load_model,'verified_signature',None)!=signature:
        assert file_sha(weights)==read(converted_path()/'CONVERSION_COMPLETE.json')['converted_sha256'],'Converted base weight hash changed'
        load_model.verified_signature=signature
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    cfg=OmegaConf.create(dict(model_path=str(converted_path()),precision='bf16',pi05=False,action_dim=7,
        num_action_chunks=20,num_steps=10,openpi=dict(task='sft',config_name='pi0_libero',action_horizon=50,
        action_chunk=20,model_action_dim=32,paligemma_variant='gemma_2b',action_expert_variant='gemma_300m',
        max_token_len=48,discrete_state_input=False,torch_compile=False)))
    model=get_model(cfg)
    from tools.sft_compat import apply_original_siglip_activation,preserve_original_policy_load
    apply_original_siglip_activation(model)
    preserve_original_policy_load(model)
    return model.to(device='cuda')


def observation(sample,device='cuda'):
    import torch
    from rlinf.models.embodiment.openpi.modules.model import Observation
    # ascontiguousarray promotes scalar masks to shape (1,), which would add
    # a second batch dimension. array(copy=True) preserves zero-dimensional masks.
    tensor=lambda a:torch.as_tensor(np.array(a,copy=True,order='C'),device=device)[None]
    keys=[k[7:] for k in sample if k.startswith('image__')]
    result=Observation.from_dict(dict(state=tensor(sample['state']).float(),
        image={k:tensor(sample['image__'+k]) for k in keys},
        image_mask={k:tensor(sample['image_mask__'+k]).bool() for k in keys},
        tokenized_prompt=tensor(sample['tokens']).long(),tokenized_prompt_mask=tensor(sample['mask']).bool()))
    assert result.state.shape==(1,32) and result.tokenized_prompt.shape==(1,48)
    assert all(v.shape==(1,) for v in result.image_masks.values())
    return result


def packed(sample):
    return dict(state=sample['state'],tokens=sample['tokens'],mask=sample['mask'],
        image__base_0_rgb=sample['image'],image__left_wrist_0_rgb=sample['wrist'],
        image__right_wrist_0_rgb=np.zeros_like(sample['image']),
        image_mask__base_0_rgb=np.asarray(True),image_mask__left_wrist_0_rgb=np.asarray(True),
        image_mask__right_wrist_0_rgb=np.asarray(False))


def simulation(model,manifest,args,kind,output,logger=None):
    import torch
    from tools.sft_data import transform_sample, PROMPTS
    req_r,req_w=os.pipe();res_r,res_w=os.pipe()
    env=dict(os.environ,JAX_PLATFORMS='cpu')
    if kind=='alignment':
        # Retain the original absolute asset paths as well as identical code:
        # model XML hashes include asset filenames, not just scene geometry.
        env['LIBERO_CONFIG_PATH']=str(Path(os.environ['DSRL_DATA_ROOT'])/'runs/job-18584/libero_config')
        original=read(Path(os.environ['DSRL_DATA_ROOT'])/'runs/job-18584/resolved_protocol.json')['code_commit']
        env['DSRL_SFT_SIM_LIBERO_ROOT']=str(Path.cwd().parent/original[:12]/'LIBERO')
        assert Path(env['DSRL_SFT_SIM_LIBERO_ROOT']).is_dir()
    child=subprocess.Popen(['bash','tools/sft_old_env.sh','tools.sft_sim','--output',str(output),
        '--request-fd',str(req_w),'--response-fd',str(res_r),'--kind',kind,'--config',args.config],
        env=env,pass_fds=(req_w,res_r))
    os.close(req_w);os.close(res_r)
    runtime=dict(manifest,image_flip_axes={})
    queries=0
    if logger:logger.log({'phase':'native_'+kind+'_simulation','sft/simulation_arm':output.name},0)
    try:
        with os.fdopen(req_r,'rb',buffering=0) as requests,os.fdopen(res_w,'wb',buffering=0) as responses:
            while True:
                try:message=pickle.load(requests)
                except EOFError:break
                goal=next(g for g,t in PROMPTS.items() if t==message['prompt'])
                sample=transform_sample(runtime,goal,message['state'],dict(image=message['image'],wrist=message['wrist']))
                with torch.inference_mode():
                    actions=model.sample_actions(observation(packed(sample)),num_steps=10,
                        noise=torch.as_tensor(message['noise'],device='cuda')).float().cpu().numpy()[0]
                norm=manifest['normalisation']['actions']
                actions=actions*(np.asarray(norm['std'])+1e-6)+np.asarray(norm['mean'])
                actions[:,:6]+=message['state'][None,:6]
                pickle.dump(actions[:,:7],responses,protocol=5)
                queries+=1
                if logger and queries%20==0:
                    latest=read(output/'latest.json') if (output/'latest.json').exists() else {}
                    logger.log({'phase':'native_'+kind+'_simulation','sft/vla_queries_served':queries,
                        'sft/physical_steps':latest.get('research/physical_steps'),
                        'sft/current_case':latest.get('research/current_case')},0)
        assert child.wait(timeout=60)==0,'Simulator worker failed'
    finally:
        if child.poll() is None:child.terminate();child.wait(timeout=20)
    return read(output/'results.json')


def align(args,logger):
    import torch
    assert torch.cuda.is_available(),'Allocated GPU cannot initialize CUDA'
    logger.log({'phase':'native_gpu_ready','sft/gpu_name':torch.cuda.get_device_name(),'sft/torch_cuda':torch.version.cuda},0)
    gate=read(Path(args.gate_dir)/'SFT_DATA_PASS.json');assert gate['camera_alignment_passed']
    # Exit the JAX process before constructing the Torch model to release VRAM.
    references=Path(args.reference_dir) if args.reference_dir else args.output/'jax_reference'
    if not (references/'manifest.json').exists():
        env=dict(os.environ,JAX_PLATFORMS='cuda')
        subprocess.run(['bash','tools/sft_old_env.sh','tools.sft_reference','--output',str(references)],check=True,env=env)
    convert(logger)
    model=load_model().eval();rows=[]
    reuse=Path(args.restore) if args.restore else None
    if reuse:
        previous=read(reuse/'static_alignment.json')
        assert not previous['passed'],'This continuation only reuses a still-failed static report, never creates a parity pass'
        assert read(reuse/'run.json')['config_sha256']==file_sha(args.config)
        assert file_sha(reuse/'native-requirements.txt')==file_sha(Path(os.environ['DSRL_SFT_PREFIX'])/'requirements-resolved.txt')
        with torch.inference_mode():
            for name in ['00-0-0.npz','04-0-0.npz','07-0-0.npz']:
                with np.load(references/name) as sample,np.load(reuse/('torch-'+name)) as expected:
                    value=model.sample_actions(observation(sample),num_steps=10,noise=torch.as_tensor(sample['noise'],device='cuda')[None]).float().cpu().numpy()[0]
                    np.testing.assert_allclose(value,expected['actions'],atol=1e-5,rtol=0)
        rows=previous['cases']
        logger.log({'phase':'continued_failed_static_diagnostic','sft/static_source':str(reuse),'sft/alignment_cases':len(rows)},0)
    prefix_rows=[]
    with torch.inference_mode():
        for ni in [0,4,7]:
            source=references/f'prefix_{ni}.npz'
            if not source.exists():continue
            with np.load(source) as target,np.load(references/f'{ni:02d}-0-0.npz') as sample:
                value,mask,ar=model.embed_prefix(observation(sample));value=value.float().cpu().numpy()
                delta=value-target['prefix']
                np.testing.assert_array_equal(mask.cpu().numpy(),target['mask'])
                np.testing.assert_array_equal(ar.cpu().numpy(),target['ar_mask'])
                prefix_rows.append(dict(observation=ni,rmse=float(np.sqrt(np.mean(delta**2))),maximum=float(np.abs(delta).max()),
                    token_groups={name:dict(rmse=float(np.sqrt(np.mean(delta[:,a:b]**2))),maximum=float(np.abs(delta[:,a:b]).max()))
                        for name,a,b in [('base',0,256),('wrist',256,512),('masked_right',512,768),('text',768,816)]}))
    atomic(args.output/'prefix_alignment.json',prefix_rows)
    with torch.inference_mode():
        for row in ([] if reuse else read(references/'manifest.json')['cases']):
            assert file_sha(references/row['file'])==row['sha256'],'Corrupt original reference'
            with np.load(references/row['file']) as sample:
                obs=observation(sample);noise=torch.as_tensor(sample['noise'],device='cuda')[None]
                actual=model.sample_actions(obs,num_steps=10,noise=noise)[0].float().cpu().numpy()
                again=model.sample_actions(obs,num_steps=10,noise=noise)[0].float().cpu().numpy()
                delta=actual-sample['actions'];np.savez_compressed(args.output/('torch-'+row['file']),actions=actual)
                rows.append(dict(**row,rmse=float(np.sqrt(np.mean(delta**2))),maximum=float(np.abs(delta).max()),
                    real7_rmse=float(np.sqrt(np.mean(delta[:,:7]**2))),per_dimension_rmse=np.sqrt(np.mean(delta**2,axis=0)).tolist(),
                    torch_internal_max=float(np.abs(actual-again).max())))
            logger.log({'phase':'static_alignment','sft/alignment_cases':len(rows),'sft/alignment_rmse':rows[-1]['rmse']},0)
    assert len(rows)==120
    report=dict(cases=rows,passed=all(r['rmse']<=.01 and r['maximum']<=.1 and r['torch_internal_max']<=1e-5 and r['internal_max']<=1e-5 for r in rows),
                engineering_alarms=dict(rmse=.01,maximum=.1),gradient_updates=0,closed_loop_passed=False)
    if reuse:report.update(reused_from=str(reuse),source_report_sha256=file_sha(reuse/'static_alignment.json'),
        reuse_scope='failed static evidence only; three unchanged-output probes passed; continue simulator diagnostics, training remains closed')
    atomic(args.output/'static_alignment.json',report)
    # Alarms keep training closed, but the planned development-only closed loop
    # still runs to diagnose whether numeric differences change behavior.
    manifest=read(Path(args.gate_dir)/'data_manifest.json')
    atomic(args.output/'data_manifest.json',manifest)
    results=simulation(model,manifest,args,'alignment',args.output/'closed_loop',logger)
    root=Path(os.environ['DSRL_DATA_ROOT'])/'runs/job-18584/cases'
    pairs=[]
    for r in results:
        original=read(root/(r['case']['id']+'.json'))
        pairs.append(dict(case=r['case']['id'],first_grasp_match=r['first_grasp_class']==original['first_grasp_class'],
            goals_match=r['goal_ever']==original['goal_ever'],original_first=original['first_grasp_class'],
            native_first=r['first_grasp_class'],original_goals=original['goal_ever'],native_goals=r['goal_ever']))
    first_agreement=float(np.mean([p['first_grasp_match'] for p in pairs]));goal_agreement=float(np.mean([p['goals_match'] for p in pairs]))
    closed=dict(pairs=pairs,first_grasp_agreement=first_agreement,goal_agreement=goal_agreement,
                passed=first_agreement>=.9 and goal_agreement>=.9,physical_steps=8000,
                gate='predeclared engineering agreement >=90% on each outcome; not a scientific equivalence test')
    atomic(args.output/'closed_loop_alignment.json',closed)
    assert report['passed'] and closed['passed'],'Static/closed-loop conversion differences require review before SFT'
    atomic(args.output/'data_manifest.json',manifest)
    atomic(args.output/'SFT_ALIGNMENT_PASS.json',dict(passed=True,static_cases=len(rows),closed_loop=closed,
           data_manifest_sha256=file_sha(args.output/'data_manifest.json'),gradient_updates=0))


class Demonstrations:
    def __init__(self,manifest):
        import h5py
        self.m=manifest;self.files={g:h5py.File(s['path'],'r') for g,s in manifest['source_files'].items()}
        self.rng=np.random.default_rng(manifest['config']['training_seed'])
        self.training={g:[t for t in manifest['trajectories'] if t['goal']==g and t['split']=='train'] for g in ['A','B']}

    def sample(self,row,start):
        from tools.sft_data import transform_sample
        ep=self.files[row['goal']]['data'][row['trajectory']]
        state=np.concatenate([ep['obs/ee_states'][start],ep['obs/gripper_states'][start]])
        offset=self.m['config']['data_action_offset']
        return transform_sample(self.m,row['goal'],state,dict(image=ep['obs/agentview_rgb'][start],
            wrist=ep['obs/eye_in_hand_rgb'][start]),ep['actions'][start+offset:start+offset+50])

    def batch(self):
        goals=['A','B']*4;self.rng.shuffle(goals)
        for goal in goals:
            row=self.training[goal][int(self.rng.integers(40))]
            yield self.sample(row,int(self.rng.integers(row['real_windows'])))

    def close(self):
        for f in self.files.values():f.close()


def adapters(model,cfg,output):
    import torch
    from peft import LoraConfig,inject_adapter_in_model
    for p in model.parameters():p.requires_grad_(False)
    model=inject_adapter_in_model(LoraConfig(r=cfg['lora_rank'],lora_alpha=cfg['lora_alpha'],lora_dropout=cfg['lora_dropout'],
        target_modules='all-linear',init_lora_weights='gaussian'),model)
    params={n:p for n,p in model.named_parameters() if p.requires_grad}
    assert params and all('lora_' in n for n in params)
    for p in params.values():p.data=p.data.float()
    groups={str(i):[n for n in params if re.search(r'\.attn\.[qkvo]_proj\.'+str(i)+r'\.lora_',n)] for i in [0,1]}
    assert all(groups.values()),'Missing VLM or action-expert attention adapters'
    assert any(n.startswith('img.') for n in params) and any(n.startswith('action_out_proj.') for n in params)
    atomic(output/'lora_coverage.json',dict(trainable_parameters=sum(p.numel() for p in params.values()),
        trainable_tensors=[dict(name=n,shape=list(p.shape),dtype=str(p.dtype)) for n,p in params.items()],attention_experts=groups,
        gemma_custom_mlp_adapted=any('mlps.' in n for n in params),all_base_frozen=True))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    return model,params


def frozen_hash(model):
    h=hashlib.sha256()
    for n,p in model.named_parameters():
        if p.requires_grad:continue
        assert p.grad is None,n
        h.update(n.encode());h.update(str(p.dtype).encode());h.update(p.detach().cpu().view(__import__('torch').uint8).numpy().tobytes())
    return h.hexdigest()


def validate_loss(model,data,seed):
    import torch
    model.eval();groups={g:[] for g in ['A','B']};rng=np.random.default_rng(seed)
    with torch.inference_mode():
        for row in data.m['trajectories']:
            if row['split']!='validation':continue
            values=[]
            for start in row['validation_starts']:
                sample=data.sample(row,start)
                noise=torch.as_tensor(rng.normal(size=(1,50,32)).astype(np.float32),device='cuda')
                t=torch.as_tensor((rng.beta(1.5,1,size=1)*.999+.001).astype(np.float32),device='cuda')
                loss=model.compute_loss(observation(packed(sample)),torch.as_tensor(sample['actions'],device='cuda')[None],train=False,noise=noise,time=t)
                assert torch.isfinite(loss).all()
                values.append([float(loss.mean()),float(loss[:,:,:7].mean())])
            groups[row['goal']].append(np.mean(values,axis=0).tolist())
    result={g:dict(loss=float(np.mean(groups[g],axis=0)[0]),real7=float(np.mean(groups[g],axis=0)[1])) for g in groups}
    result['loss']=(result['A']['loss']+result['B']['loss'])/2
    return result


def save_checkpoint(path,model,params,optimizer,scheduler,data,step,state):
    import torch
    value=dict(step=step,adapters={n:p.detach().cpu() for n,p in params.items()},optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(),sampler_rng=data.rng.bit_generator.state,cpu_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all(),config_sha256=file_sha(state['config_path']),
        data_manifest_sha256=state['manifest_sha256'],base_sha256=read(converted_path()/'CONVERSION_COMPLETE.json')['converted_sha256'],
        state=state)
    temporary=path.with_suffix('.tmp');torch.save(value,temporary);temporary.replace(path)
    atomic(path.with_suffix('.json'),dict(step=step,sha256=file_sha(path),validation=state['current_validation'],
        best_step=state['best_step'],best_loss=state['best_loss'],config_sha256=value['config_sha256']))


def restore_checkpoint(path,params,optimizer=None,scheduler=None,data=None):
    import torch
    value=torch.load(path,map_location='cpu',weights_only=False)
    assert value['base_sha256']==read(converted_path()/'CONVERSION_COMPLETE.json')['converted_sha256']
    assert set(params)==set(value['adapters'])
    with torch.no_grad():
        for n,p in params.items():p.copy_(value['adapters'][n])
    if optimizer is not None:
        optimizer.load_state_dict(value['optimizer']);scheduler.load_state_dict(value['scheduler'])
        data.rng.bit_generator.state=value['sampler_rng'];torch.set_rng_state(value['cpu_rng']);torch.cuda.set_rng_state_all(value['cuda_rng'])
    return value


def train(args,logger):
    import torch
    from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup
    cfg=read(args.config);tr=cfg['training'];gate=Path(args.gate_dir)
    manifest=read(gate/'data_manifest.json');assert file_sha(args.config)==manifest['protocol_sha256']
    torch.manual_seed(cfg['training_seed']);torch.cuda.manual_seed_all(cfg['training_seed'])
    model,params=adapters(load_model(),tr,args.output);baseline=frozen_hash(model)
    data=Demonstrations(manifest)
    optimizer=torch.optim.AdamW(list(params.values()),lr=tr['learning_rate'],betas=(tr['adam_beta1'],tr['adam_beta2']),
        eps=tr['adam_epsilon'],weight_decay=tr['weight_decay'])
    scheduler=get_cosine_with_min_lr_schedule_with_warmup(optimizer,num_warmup_steps=tr['warmup_updates'],
        num_training_steps=tr['max_updates'],num_cycles=.5,min_lr=0.)
    # The same public HF scheduler is called by pinned RLinf get_lr_scheduler('cosine').
    state=dict(config_path=str(Path(args.config).resolve()),manifest_sha256=file_sha(gate/'data_manifest.json'),
        best_loss=float('inf'),best_step=0,best_checkpoint='',initial_validation=None,current_validation=None,frozen_sha256=baseline)
    step=0
    if args.restore:
        saved=restore_checkpoint(Path(args.restore),params,optimizer,scheduler,data)
        assert saved['config_sha256']==file_sha(args.config) and saved['data_manifest_sha256']==state['manifest_sha256']
        assert saved['state']['frozen_sha256']==baseline
        state=saved['state'];step=saved['step']
    else:
        initial=validate_loss(model,data,tr['validation_noise_seed']);state['initial_validation']=initial
        state['current_validation']=initial;state['best_loss']=initial['loss']
        state['best_checkpoint']=str(args.output/'checkpoint-000000.pt')
        save_checkpoint(args.output/'checkpoint-000000.pt',model,params,optimizer,scheduler,data,0,state)
        logger.log({'phase':'initial_validation','sft/validation_loss':initial['loss'],'sft/B_real7':initial['B']['real7']},0)
    atomic(args.output/'data_manifest.json',manifest)
    start_step=step;limit=2 if args.mode=='profile' else tr['max_updates']
    first_params={n:p.detach().cpu().clone() for n,p in params.items()}
    started=time.monotonic()
    try:
        while step<limit:
            model.train();optimizer.zero_grad(set_to_none=True);losses=[]
            update_started=time.monotonic();used_lr=optimizer.param_groups[0]['lr']
            for sample in data.batch():
                loss=model.compute_loss(observation(packed(sample)),torch.as_tensor(sample['actions'],device='cuda')[None],train=True).mean()
                assert torch.isfinite(loss),'Nonfinite training loss'
                (loss/tr['global_batch_size']).backward();losses.append(float(loss.detach()))
            norm=torch.nn.utils.clip_grad_norm_(list(params.values()),tr['gradient_clip'],error_if_nonfinite=True)
            assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
            if args.mode=='profile':
                gradients={n:(None if p.grad is None else float(p.grad.float().norm())) for n,p in params.items()}
                active_experts={str(i):[n for n,v in gradients.items() if v is not None and v>0 and re.search(r'\.attn\.[qkvo]_proj\.'+str(i)+r'\.lora_',n)] for i in [0,1]}
                assert all(active_experts.values()),'Missing actual VLM/action-expert adapter gradients'
                atomic(args.output/f'gradient-coverage-{step+1:06d}.json',dict(gradients=gradients,active_experts=active_experts,
                    unused_trainable_tensors=[n for n,v in gradients.items() if v is None],
                    note='MHA functional projections may bypass nested Linear adapters; presence does not prove effective coverage'))
            optimizer.step();scheduler.step();step+=1
            logger.log({'phase':'sft_training','sft/train_loss':float(np.mean(losses)),'sft/gradient_norm':float(norm),
                'sft/learning_rate':used_lr,'sft/update_seconds':time.monotonic()-update_started,
                'sft/peak_gpu_mib':torch.cuda.max_memory_allocated()/2**20},step)
            if step%tr['save_every']==0 or step==limit:
                val=validate_loss(model,data,tr['validation_noise_seed']);state['current_validation']=val
                improved=val['loss']<state['best_loss']
                if improved:
                    state['best_loss']=val['loss'];state['best_step']=step
                    state['best_checkpoint']=str(args.output/f'checkpoint-{step:06d}.pt')
                assert frozen_hash(model)==baseline,'Original base parameters changed'
                save_checkpoint(args.output/f'checkpoint-{step:06d}.pt',model,params,optimizer,scheduler,data,step,state)
                if improved:
                    atomic(args.output/'best_checkpoint.json',dict(step=step,path=str(args.output/f'checkpoint-{step:06d}.pt'),loss=val['loss']))
                logger.log({'phase':'validation','sft/validation_loss':val['loss'],'sft/A_real7':val['A']['real7'],'sft/B_real7':val['B']['real7']},step)
                if step==tr['pilot_updates']:
                    passed=val['loss']<state['initial_validation']['loss'] and val['B']['real7']<state['initial_validation']['B']['real7']
                    atomic(args.output/'PILOT_DECISION.json',dict(passed=passed,step=step,initial=state['initial_validation'],current=val))
                    if not passed:break
        delta=max(float((p.detach().cpu()-first_params[n]).abs().max()) for n,p in params.items())
        assert delta>0,'No adapter parameter changed'
        if args.mode=='profile':
            atomic(args.output/'adapter_deltas.json',{n:float((p.detach().cpu()-first_params[n]).abs().max()) for n,p in params.items()})
            model.eval();sample=data.sample(data.training['A'][0],0);arrays=packed(sample)
            arrays['noise']=np.random.default_rng(99026).normal(size=(1,50,32)).astype(np.float32)
            with torch.inference_mode():
                arrays['expected']=model.sample_actions(observation(arrays),noise=torch.as_tensor(arrays['noise'],device='cuda'),num_steps=10).float().cpu().numpy()
            np.savez_compressed(args.output/'profile_reference.npz',**arrays)
            atomic(args.output/'PROFILE_UPDATES_COMPLETE.json',dict(passed=True,updates=step,adapter_max_delta=delta,
                frozen_sha256=baseline,seconds=time.monotonic()-started,checkpoint=str(args.output/f'checkpoint-{step:06d}.pt')))
        else:
            selected=dict(step=state['best_step'],path=state['best_checkpoint'],loss=state['best_loss'])
            atomic(args.output/'SFT_TRAINING_COMPLETE.json',dict(passed=True,updates=step,pilot_only=step<tr['max_updates'],selected=selected,
                initial_validation=state['initial_validation'],final_validation=state['current_validation'],frozen_sha256=baseline,
                adapter_max_delta=delta,simulation_steps=0))
    finally:data.close()


def verify_restore(args,logger):
    import torch
    cfg=read(args.config);profile=read(args.output/'PROFILE_UPDATES_COMPLETE.json')
    model,params=adapters(load_model(),cfg['training'],args.output)
    saved=restore_checkpoint(Path(profile['checkpoint']),params)
    assert frozen_hash(model)==profile['frozen_sha256']
    model.eval()
    with np.load(args.output/'profile_reference.npz') as sample,torch.inference_mode():
        actual=model.sample_actions(observation(sample),noise=torch.as_tensor(sample['noise'],device='cuda'),num_steps=10).float().cpu().numpy()
        error=float(np.abs(actual-sample['expected']).max())
        np.testing.assert_allclose(actual,sample['expected'],atol=1e-5,rtol=0)
    atomic(args.output/'SFT_PROFILE_PASS.json',dict(passed=True,updates=saved['step'],fresh_process_restore_max_error=error,
           checkpoint=profile['checkpoint'],frozen_sha256=profile['frozen_sha256'],adapter_max_delta=profile['adapter_max_delta']))
    logger.log({'phase':'profile_restore_passed','sft/restore_max_error':error},saved['step'])


def evaluate(args,logger):
    import gc
    import torch
    cfg=read(args.config);gate=Path(args.gate_dir);manifest=read(gate/'data_manifest.json')
    selected=read(gate/'SFT_TRAINING_COMPLETE.json')['selected'];atomic(args.output/'selected_checkpoint.json',selected)
    model=load_model().eval()
    baseline=simulation(model,manifest,args,'behavior',args.output/'native_unadapted',logger)
    del model;gc.collect();torch.cuda.empty_cache()
    model,params=adapters(load_model(),cfg['training'],args.output)
    restore_checkpoint(Path(selected['path']),params);model.eval()
    adapted=simulation(model,manifest,args,'behavior',args.output/'adapted',logger)
    for seed in cfg['behavior_reset_seeds']:
        name=f'E1__{seed}.npz'
        with np.load(args.output/'native_unadapted/common_states'/name) as base,np.load(args.output/'adapted/common_states'/name) as tuned:
            assert set(base.files)==set(tuned.files)
            for key in base.files:np.testing.assert_array_equal(base[key],tuned[key])
    summary={}
    for name,results in [('native_unadapted',baseline),('adapted',adapted)]:
        summary[name]={g:dict(episodes=20,first_correct=sum(r['correct_first_grasp'] for r in results if r['case']['requested_goal_id']==g),
            goal_ever=sum(r['active_goal_ever_success'] for r in results if r['case']['requested_goal_id']==g),
            selective_first_completion=sum(r['active_goal_first_completion_without_prior_wrong_goal'] for r in results if r['case']['requested_goal_id']==g)) for g in ['A','B']}
    atomic(args.output/'SFT_EVAL_COMPLETE.json',dict(passed=True,selected=selected,results=summary,episodes=80,physical_steps=32000,
        interpretation='paired new-reset development behavior check; no E1 confirmation or reserved tuning'))
    logger.log({'phase':'sft_behavior_complete','sft/episodes':80,'sft/physical_steps':32000},0)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',required=True,choices=['align','profile','train','evaluate','verify'])
    parser.add_argument('--config',required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gate-dir',required=True);parser.add_argument('--restore',default='')
    parser.add_argument('--reference-dir',default=os.environ.get('DSRL_SFT_REFERENCE_DIR',''));args=parser.parse_args()
    from tools.run_logger import RunLogger
    args.output.mkdir(parents=True,exist_ok=True);logger=RunLogger(args.output)
    if args.mode!='verify':atomic(args.output/'run.json',dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        mode='sft-'+args.mode,learning=args.mode in ['profile','train'],job_id=os.environ.get('SLURM_JOB_ID'),
        config_sha256=file_sha(args.config),python=sys.executable,hostname=os.uname().nodename,
        original_model=os.environ['DSRL_MODEL_PATH'],converted_model=str(converted_path()),gate_dir=args.gate_dir,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        rlinf_commit=subprocess.check_output(['git','-C',os.environ['DSRL_RLINF_ROOT'],'rev-parse','HEAD'],text=True).strip(),
        slurm={k:v for k,v in os.environ.items() if k in ['SLURM_JOB_PARTITION','SLURM_CPUS_PER_TASK','SLURM_JOB_NODELIST']}))
    if args.mode!='verify':atomic(args.output/'numerical_compatibility.json',dict(
        siglip_and_gemma_gelu='original JAX typed elementary tanh formula, including bf16 intermediate rounding; CPU 1001-point bf16 reference exact',
        source_policy_load='all checkpoint leaves bf16; native fp32 islands hold bf16-rounded values upcast to fp32',
        single_gpu_loss_state_dtype='fp32, identical to inference input; no unnecessary FSDP bf16 input downcast',
        checkpoint_files_modified=False,tf32_matmul=False,tf32_cudnn=False,bf16_reduced_precision_reduction=False,
        upstream_rlinf_unmodified=True))
    try:
        if args.mode=='align':align(args,logger)
        elif args.mode in ['profile','train']:train(args,logger)
        elif args.mode=='verify':verify_restore(args,logger)
        else:evaluate(args,logger)
    except BaseException as error:
        atomic(args.output/'FAILED.json',dict(type=type(error).__name__,error=str(error),traceback=traceback.format_exc()))
        raise
    finally:logger.close()


if __name__=='__main__':main()
