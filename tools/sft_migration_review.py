"""Evidence-locked review for protocol 11.1's within-native comparison fallback.

Never creates an alignment-pass flag or changes a failed numerical-parity result.
An explicit decision file must be committed after the complete diagnostic run.
"""
import argparse
from collections import Counter
import math
import os
from pathlib import Path
import subprocess
import numpy as np
from tools.research_core import atomic,canonical,file_sha,read


def main(args):
    source=Path(args.run);root=Path(os.environ['DSRL_DATA_ROOT']);decision=read(args.decision)
    assert decision['source_job']==source.name.removeprefix('job-')
    assert decision['comparison_scope']=='within_native_runtime_only'
    assert decision['approve_sft_a_profile'] is True and decision['cross_framework_equivalence_claim'] is False
    assert decision['protocol_basis']=='EXPERIMENT_SPEC.md section 11.1 fallback'
    assert read(source/'run.json')['config_sha256']==file_sha(args.config)==decision['config_sha256']
    for name,expected in decision['source_artifact_sha256'].items():assert file_sha(source/name)==expected,name
    failure=read(source/'FAILED.json')
    assert failure['type']=='AssertionError' and failure['error']=='Static/closed-loop conversion differences require review before SFT'
    static=read(source/'static_alignment.json');assert len(static['cases'])==120 and not static['passed']
    assert all(r['internal_max']<=1e-5 and r['torch_internal_max']<=1e-5 and math.isfinite(r['rmse']) for r in static['cases'])
    complete=read(source/'closed_loop/SIMULATION_COMPLETE.json')
    assert complete['passed'] and complete['episodes']==20 and complete['physical_steps']==8000
    budget=read(source/'closed_loop/budget.json');assert budget['recorded_physical_steps']==8000 and budget['uncertain_steps']==0 and not budget['reservations']
    results=read(source/'closed_loop/results.json');assert len(results)==20
    cases=read(source/'closed_loop/case_manifest.json')
    original=[c for c in read(root/'runs/job-18584/case_manifest.json') if c['phase']=='e1_dev' and c['policy']=='base']
    assert sorted(map(canonical,cases))==sorted(map(canonical,original))
    reset_errors={}
    for seed in sorted({c['reset_seed'] for c in cases}):
        name=f'E1__{seed}.npz'
        with np.load(source/'closed_loop/common_states'/name) as actual,np.load(root/'runs/job-18584/common_states'/name) as reference:
            np.testing.assert_array_equal(actual['model_xml_sha256'],reference['model_xml_sha256'])
            values={}
            for key,tolerance in [('physics',1e-8),('proprio',1e-7),('agentview_image',0),('wrist_image',0)]:
                error=float(np.abs(actual[key].astype(np.float64)-reference[key].astype(np.float64)).max())
                assert error<=tolerance,(seed,key,error)
                values[key]=error
            reset_errors[str(seed)]=values
    for result in results:
        assert result['status'] in ['COMPLETE','POLICY_FAILURE'] and result['executed_env_steps']==400
        assert all(v==0 for v in result['common_state']['errors'].values())
        for name,expected in result['artifact_hashes'].items():assert file_sha(source/'closed_loop'/name)==expected,name
    assert not list((source/'closed_loop/attempts').glob('*/0/error.json'))
    prefix=read(source/'prefix_alignment.json')
    assert len(prefix)==3 and all(r['token_groups']['text']['maximum']==0 for r in prefix)
    demo_path=root/'research/sft-a-v1-20261028/demo_order/DEMONSTRATION_ORDER_AUDIT.json'
    demo=read(demo_path);assert len(demo['trajectories'])==100
    for goal in ['A','B']:
        assert demo['summary'][goal]['first_grasp']=={goal:50} and demo['summary'][goal]['terminal_success']==50
    assert file_sha(demo_path)==decision['demo_order_sha256']
    cross=root/'research/sft-a-v1-20261028/JAX_CROSS_GPU_NUMERICS.json'
    assert file_sha(cross)==decision['jax_cross_gpu_sha256'] and read(cross)['input_arrays_identical']
    manifest=read(source/'data_manifest.json');assert manifest['protocol_sha256']==file_sha(args.config)
    assert sum(t['split']=='train' for t in manifest['trajectories'])==80
    assert sum(t['split']=='validation' for t in manifest['trajectories'])==20
    summary={g:dict(first_grasp=dict(Counter(r['first_grasp_class'] for r in results if r['case']['requested_goal_id']==g)),
        goal_ever=sum(r['active_goal_ever_success'] for r in results if r['case']['requested_goal_id']==g)) for g in ['A','B']}
    scope=dict(comparison_scope='within_native_runtime_only',cross_framework_numerical_equivalence=False,
        original_jax_dsrl_single_variable_attribution=False,approved_intervention='SFT-A LoRA, paired native unadapted vs adapted only',
        protocol_basis=decision['protocol_basis'],source_diagnostic=str(source),decision_sha256=file_sha(args.decision),
        reviewer_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        static_parity_passed=False,closed_loop_alignment=read(source/'closed_loop_alignment.json'),
        native_development_summary=summary,reason=decision['reason'],max_optimizer_updates_including_profile=3000,
        exact_reset_errors=reset_errors,
        stage_or_ppo_authorized=False)
    atomic(source/'comparison_scope.json',scope)
    atomic(source/'SFT_BASELINE_READY.json',dict(passed=True,comparison_scope=scope['comparison_scope'],
        cross_framework_numerical_equivalence=False,comparison_scope_sha256=file_sha(source/'comparison_scope.json'),
        config_sha256=file_sha(args.config),data_manifest_sha256=file_sha(source/'data_manifest.json'),
        reviewed_diagnostic_job=decision['source_job'],gradient_updates=0))
    print('WITHIN_NATIVE_BASELINE_READY',summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--decision',required=True);p.add_argument('--config',required=True)
    main(p.parse_args())
