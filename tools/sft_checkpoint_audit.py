"""Inspect the completed SFT artifacts on CPU without loading the VLA model."""
import argparse
import math
import os
from pathlib import Path
import subprocess

from tools.research_core import atomic, file_sha, read


def audit(training):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','Run with CUDA_VISIBLE_DEVICES empty'
    import torch
    torch.set_num_threads(2)
    training=Path(training).resolve()
    run=read(training/'run.json');done=read(training/'SFT_TRAINING_COMPLETE.json')
    assert done['passed'] and done['updates']==3000
    assert run['comparison_scope']=='within_native_runtime_only'
    assert read(training/'PILOT_DECISION.json')['passed']
    profile=Path(run['gate_dir'])
    coverage=read(training/'lora_coverage.json')
    config_sha=run['config_sha256'];manifest_sha=file_sha(training/'data_manifest.json')
    scope_sha=file_sha(training/'comparison_scope.json')
    base=read(Path(run['converted_model'])/'CONVERSION_COMPLETE.json')['converted_sha256']
    original=torch.load(profile/'checkpoint-000000.pt',map_location='cpu',weights_only=False)
    initial=original['adapters']
    assert len(initial)==462 and sum(p.numel() for p in initial.values())==6043136
    assert len(initial)==len(coverage['trainable_tensors']) and coverage['all_base_frozen']
    checkpoints=sorted(list(profile.glob('checkpoint-*.json'))+list(training.glob('checkpoint-*.json')),
        key=lambda p:read(p)['step'])
    rows=[]
    for metadata in checkpoints:
        row=read(metadata);path=metadata.with_suffix('.pt')
        digest=file_sha(path);assert digest==row['sha256']
        saved=torch.load(path,map_location='cpu',weights_only=False)
        assert saved['step']==row['step']
        assert saved['config_sha256']==config_sha
        assert saved['data_manifest_sha256']==manifest_sha
        assert saved['base_sha256']==base
        assert saved['state']['frozen_sha256']==done['frozen_sha256']
        assert saved['state']['comparison_scope_sha256']==scope_sha
        assert file_sha(Path(saved['state']['config_path']))==config_sha
        assert saved['state']['current_validation']==row['validation']
        assert set(saved['adapters'])==set(initial)
        assert isinstance(saved['sampler_rng'],dict)
        assert saved['cpu_rng'].dtype==torch.uint8 and saved['cpu_rng'].ndim==1
        assert all(s.dtype==torch.uint8 and s.ndim==1 for s in saved['cuda_rng'])
        maximum=0.;squared=0.;changed=0
        for name,p in saved['adapters'].items():
            assert 'lora_' in name and p.device.type=='cpu'
            assert p.shape==initial[name].shape and torch.isfinite(p).all()
            delta=(p.float()-initial[name].float())
            m=float(delta.abs().max());maximum=max(maximum,m);changed+=m>0
            squared+=float(delta.double().square().sum())
        states=saved['optimizer']['state'];steps=[]
        for item in states.values():
            for value in item.values():
                if isinstance(value,torch.Tensor):assert value.device.type=='cpu' and torch.isfinite(value).all()
            steps.append(int(item['step']))
        if saved['step']:
            assert maximum>0 and steps and min(steps)==max(steps)==saved['step']
        else:assert maximum==0 and not steps
        val=row['validation']
        assert all(math.isfinite(val[g][k]) for g in ('A','B') for k in ('loss','real7'))
        assert math.isclose(val['loss'],(val['A']['loss']+val['B']['loss'])/2,rel_tol=1e-12)
        rows.append(dict(step=saved['step'],checkpoint=str(path),sha256=digest,
            bytes=path.stat().st_size,validation=val,adapter_max_delta_from_initial=maximum,
            adapter_l2_delta_from_initial=math.sqrt(squared),changed_adapter_tensors=changed,
            total_adapter_tensors=len(initial),optimizer_states=len(states),optimizer_step_min=min(steps) if steps else 0,
            optimizer_step_max=max(steps) if steps else 0))
        del saved
    assert [r['step'] for r in rows]==[0,2,500,1000,1500,2000,2500,3000]
    best=min(rows,key=lambda r:(r['validation']['loss'],r['step']))
    assert best['step']==done['selected']['step']==500
    assert best['checkpoint']==done['selected']['path']
    assert best['validation']['loss']==done['selected']['loss']
    assert not torch.cuda.is_initialized(),'CPU audit initialized CUDA'
    output=training/'checkpoint_audit';output.mkdir(exist_ok=True)
    result=dict(passed=True,training_job=run['job_id'],training_updates=done['updates'],
        training_commit=run['commit'],audit_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        audit_code_sha256=file_sha(__file__),checked_checkpoints=len(rows),selected_step=500,
        selection='minimum fixed equal-goal/trajectory held-out 32D loss; no behavior read',
        adapters_finite=True,optimizer_states_finite=True,checkpoint_sha256_verified=True,
        config_sha256=config_sha,data_manifest_sha256=manifest_sha,comparison_scope_sha256=scope_sha,
        stored_frozen_parameter_hash=done['frozen_sha256'],base_weight_files_rehashed=False,
        limitation='Saved base identity and frozen hashes checked; VLA parameters were not instantiated or rehashed in memory by this CPU audit.',
        behavior_outcomes_read=False,simulation_steps_added=0,vla_forwards_added=0,
        gradient_updates_added=0,cuda_initialized=False,checkpoints=rows)
    atomic(output/'checkpoint_audit.json',result)
    atomic(training/'SFT_CHECKPOINT_AUDIT.json',dict(passed=True,checked_checkpoints=len(rows),
        selected_step=500,training_updates=3000,detail=str(output/'checkpoint_audit.json'),
        detail_sha256=file_sha(output/'checkpoint_audit.json'),simulation_steps_added=0,
        vla_forwards_added=0,gradient_updates_added=0,cuda_initialized=False))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--training',required=True);args=p.parse_args()
    result=audit(args.training)
    print({k:v for k,v in result.items() if k!='checkpoints'})
