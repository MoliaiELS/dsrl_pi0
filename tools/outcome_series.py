"""Sequential isolated evaluation processes within one Slurm allocation."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

from tools.run_logger import write_json


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def run_series(args, config, folder, protocol_path):
    started = time.monotonic()
    stages = config['stages']
    assert len({s['id'] for s in stages}) == len(stages)
    assert all(s['id'].replace('_', '').isalnum() for s in stages)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    stage_configs = [read(Path(s['config'])) for s in stages]
    assert all(stage_configs) and all(c.get('kind') != 'outcome_series' for c in stage_configs)
    planned = 0
    for cfg in stage_configs:
        if args.mode == 'outcome-smoke':
            planned += len(cfg['smoke_eval_reset_seeds']) * len(cfg['conditions']) * len(cfg['smoke_pairs'])
        else:
            planned += len(cfg['reset_seeds']) * len(cfg['conditions']) * len(cfg['instructions']) * len(cfg['noise_modes'])
    write_json(folder / 'config.json', config)
    write_json(folder / 'run.json', {
        'commit': commit, 'submodules': subprocess.check_output(['git', 'submodule', 'status'], text=True),
        'mode': args.mode, 'seed': args.seed, 'job_id': os.environ.get('SLURM_JOB_ID'),
        'hostname': os.uname().nodename, 'learning': False, 'series': True,
        'model_source': 's3://openpi-assets/checkpoints/pi0_libero',
        'model_path': os.environ['DSRL_MODEL_PATH'], 'python': sys.executable,
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'protocol_file_sha256': hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        'stage_protocol_hashes': {s['id']: hashlib.sha256(Path(s['config']).read_bytes()).hexdigest() for s in stages},
        'split': config['split']})
    completed = []

    def refresh(current, phase='evaluating_outcome_series'):
        keys = ['cases_completed', 'cases_successful', 'prefix_env_steps', 'replay_env_steps',
                'continuation_env_steps', 'settling_env_steps', 'vla_calls', 'controller_calls',
                'states_built', 'validation_rollouts']
        counts = {'probe/' + k: 0 for k in keys}
        resources = []
        for stage in stages:
            directory = folder / stage['id']
            latest = read(directory / 'latest.json')
            for key in counts:
                counts[key] += latest.get(key, 0)
            if stage['id'] == current:
                counts['probe/current_case'] = latest.get('probe/current_case', latest.get('phase', 'starting'))
            samples = directory / 'resources.jsonl'
            if samples.exists():
                resources.extend(dict(json.loads(line), stage=stage['id'])
                                 for line in samples.read_text().splitlines() if line.strip())
        event = {'phase': phase, 'gradient_updates': 0, 'time_unix': time.time(),
                 'wall_seconds': time.monotonic() - started, **counts, 'probe/cases_planned': planned,
                 'workflow/current_stage': current, 'workflow/stages_completed': len(completed),
                 'workflow/stages_planned': len(stages)}
        write_json(folder / 'latest.json', event)
        with (folder / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(event) + '\n')
        temp = folder / 'resources.jsonl.tmp'
        temp.write_text(''.join(json.dumps(r) + '\n' for r in resources))
        temp.replace(folder / 'resources.jsonl')
        print('SERIES', json.dumps(event), flush=True)

    try:
        for stage in stages:
            current = stage['id']
            directory = folder / current
            assert not directory.exists(), f'Refusing to overwrite stage {directory}'
            refresh(current)
            command = [sys.executable, '-u', '-m', 'tools.outcome_probe', '--mode', args.mode,
                       '--config', stage['config'], '--seed', str(args.seed), '--output', str(directory)]
            with (folder / (current + '.log')).open('w') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                try:
                    while True:
                        try:
                            code = child.wait(timeout=30)
                            break
                        except subprocess.TimeoutExpired:
                            refresh(current)
                    assert code == 0, f'Stage {current} exited {code}; see {log.name}'
                finally:
                    if child.poll() is None:
                        child.terminate()
                        child.wait(timeout=20)
            assert read(directory / 'SUCCESS.json').get('passed'), current
            summary = read(directory / 'outcome_summary.json')
            assert summary['gradient_updates'] == 0 and summary['controller_unchanged']
            assert summary['frozen_vla_output_unchanged'] and summary['reference_rollouts_reproducible']
            assert summary['counts']['cases_completed'] == summary['counts']['cases_planned']
            group = stage.get('common_state_group', 'all' if config.get('require_common_states') else None)
            previous = [s for s in completed if group and s.get('common_state_group') == group]
            if previous:
                reference = folder / previous[0]['id'] / 'common_states'
                witnesses = sorted((directory / 'common_states').glob('*.npz'))
                assert [p.name for p in witnesses] == sorted(p.name for p in reference.glob('*.npz'))
                for witness in witnesses:
                    with np.load(witness) as actual, np.load(reference / witness.name) as expected:
                        assert set(actual.files) == set(expected.files)
                        for key in actual.files:
                            np.testing.assert_array_equal(actual[key], expected[key], err_msg=f'{current}/{witness.name}/{key}')
            completed.append({'id': current, 'directory': str(directory), 'common_state_group': group,
                              'run': read(directory / 'run.json'), 'summary': summary})
            verified_groups = {g: sum(s.get('common_state_group') == g for s in completed)
                               for g in {s.get('common_state_group') for s in completed} if g}
            write_json(folder / 'series_summary.json', {'stages': completed, 'learning': False,
                       'common_state_groups_verified': verified_groups,
                       'cross_controller_common_states_verified': any(n > 1 for n in verified_groups.values())})
            refresh(current)
        assert read(folder / 'latest.json')['probe/cases_completed'] == planned
        write_json(folder / 'SUCCESS.json', {'mode': args.mode, 'passed': True, 'stages_completed': len(completed)})
        refresh(stages[-1]['id'], 'completed')
    except BaseException as error:
        write_json(folder / 'FAILED.json', {'type': type(error).__name__, 'error': str(error),
                   'completed_stages': [s['id'] for s in completed], 'traceback': traceback.format_exc()})
        raise
