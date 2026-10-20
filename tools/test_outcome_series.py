"""CPU regression checks for orchestration; fixture children load no VLA/simulator."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import outcome_series as series


CHILD = '''
import argparse, json, os
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser()
p.add_argument('--mode'); p.add_argument('--config'); p.add_argument('--seed'); p.add_argument('--output')
a=p.parse_args(); cfg=json.loads(Path(a.config).read_text()); out=Path(a.output); out.mkdir()
def write(name,value): (out/name).write_text(json.dumps(value))
write('run.json', {'fixture_only':True})
write('latest.json', {'probe/cases_completed':1,'probe/cases_successful':1,
                     'probe/vla_calls':2,'probe/current_case':'fixture/'+cfg['id']})
(out/'resources.jsonl').write_text(json.dumps({'pid':os.getpid(),'process_max_rss_mib':10,'gpus':[]})+'\\n')
(out/'common_states').mkdir()
np.savez(out/'common_states/state.npz', physics=np.array([cfg.get('witness',0.0)]))
if cfg.get('fail'): raise SystemExit(3)
write('outcome_summary.json', {'gradient_updates':0,'controller_unchanged':True,
      'frozen_vla_output_unchanged':True,'reference_rollouts_reproducible':True,
      'counts':{'cases_completed':1,'cases_planned':1}})
write('SUCCESS.json', {'passed':True})
'''


class SeriesRegression(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='dsrl-series-regression-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / 'run'
        self.output.mkdir()
        self.child = self.root / 'fixture_child.py'
        self.child.write_text(CHILD)
        stages = []
        for i in range(4):
            name = f'seed{i}' if i < 3 else 'recovery'
            cfg = {'id': name, 'smoke_eval_reset_seeds': [10], 'conditions': [{}],
                   'smoke_pairs': [['original','dsrl']], 'reset_seeds': [100],
                   'instructions': [{}], 'noise_modes': ['dsrl'], 'witness': 0 if i < 3 else 9}
            path = self.root / (name + '.json')
            path.write_text(json.dumps(cfg))
            stages.append({'id': name, 'config': str(path),
                           'common_state_group': 'confirmation' if i < 3 else 'recovery'})
        self.config = {'split': 'CPU fixture, no policy execution', 'stages': stages}
        self.protocol = self.root / 'protocol.json'
        self.protocol.write_text(json.dumps(self.config))

    def execute(self, mode='outcome-smoke'):
        real_popen = subprocess.Popen
        fixture = self.child

        def launch(command, *args, **kwargs):
            if command[:4] == [sys.executable, '-u', '-m', 'tools.outcome_probe']:
                child = real_popen([sys.executable, str(fixture), *command[4:]], *args, **kwargs)
                real_wait = child.wait
                first = True

                def wait(timeout=None):
                    nonlocal first
                    if first:
                        first = False
                        return real_wait(timeout=0.001)
                    return real_wait(timeout=timeout)
                child.wait = wait
                return child
            return real_popen(command, *args, **kwargs)

        args = SimpleNamespace(mode=mode, seed=0)
        with patch.dict(os.environ, {'DSRL_MODEL_PATH': '/fixture/no-model'}), \
                patch.object(series.subprocess, 'Popen', side_effect=launch), \
                contextlib.redirect_stdout(io.StringIO()):
            series.run_series(args, self.config, self.output, self.protocol)

    def test_four_stages_keep_case_names_out_of_numeric_totals(self):
        self.execute()
        latest = series.read(self.output / 'latest.json')
        self.assertEqual(latest['probe/cases_completed'], 4)
        self.assertEqual(latest['probe/vla_calls'], 8)
        self.assertEqual(latest['probe/current_case'], 'fixture/recovery')
        self.assertEqual(latest['phase'], 'completed')
        summary = series.read(self.output / 'series_summary.json')
        self.assertEqual(summary['common_state_groups_verified'], {'confirmation': 3, 'recovery': 1})
        self.assertTrue(summary['cross_controller_common_states_verified'])
        self.assertEqual(series.read(self.output / 'SUCCESS.json')['stages_completed'], 4)
        self.assertEqual(len((self.output / 'resources.jsonl').read_text().splitlines()), 4)

    def test_child_failure_stops_remaining_stages_and_preserves_successful_outputs(self):
        path = Path(self.config['stages'][1]['config'])
        path.write_text(json.dumps(dict(series.read(path), fail=True)))
        with self.assertRaisesRegex(AssertionError, 'Stage seed1 exited 3'):
            self.execute()
        self.assertEqual(series.read(self.output / 'FAILED.json')['completed_stages'], ['seed0'])
        self.assertTrue((self.output / 'seed0/SUCCESS.json').exists())
        self.assertFalse((self.output / 'seed2').exists())
        self.assertFalse((self.output / 'SUCCESS.json').exists())

    def test_changed_common_state_is_rejected(self):
        path = Path(self.config['stages'][1]['config'])
        path.write_text(json.dumps(dict(series.read(path), witness=1)))
        with self.assertRaises(AssertionError):
            self.execute('outcome')
        self.assertEqual(series.read(self.output / 'FAILED.json')['completed_stages'], ['seed0'])
        self.assertFalse((self.output / 'seed2').exists())
        self.assertFalse((self.output / 'SUCCESS.json').exists())


if __name__ == '__main__':
    unittest.main()
