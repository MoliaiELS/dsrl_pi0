"""CPU checks for recorded reference settling and rejection of unstable holds."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from tools.recovery_probe import RecoveryProbe


class ReferenceStabilization(unittest.TestCase):
    def fixture(self, speed=0, lose_grasp=False):
        temp = tempfile.TemporaryDirectory(prefix='dsrl-reference-test-')
        self.addCleanup(temp.cleanup)
        actions = []
        data = SimpleNamespace(get_joint_qvel=lambda _: np.array([speed, 0, 0, 0, 0, 0]))
        env = SimpleNamespace(sim=SimpleNamespace(data=data), closed=False)

        def step(action):
            actions.append(np.array(action))
            return {}, 0, False, {}

        env.step = step
        env.close = lambda: setattr(env, 'closed', True)
        probe = RecoveryProbe.__new__(RecoveryProbe)
        probe.cfg = {'query_freq': 20, 'reference_max_steps': 4, 'reference_prefix_steps': 1,
                     'reference_lift_m': .04, 'reference_translation_m': .01,
                     'reference_grasp_hold_steps': 1, 'reference_stabilize_steps': 3,
                     'max_reference_speed_mps': .02, 'controller': {'run_dir': 'fixture'}}
        probe.counts = {'prefix_env_steps': 0}
        probe.directory = Path(temp.name)
        probe.task = SimpleNamespace(language='original goal')
        probe.frozen_reference = None
        probe.log = lambda *a, **kw: None
        probe.reset_seed = lambda _: (env, {})
        probe.target = lambda _: SimpleNamespace(joints=['object_freejoint'])
        probe.noise = lambda *a: np.zeros((1, 50, 32))
        probe.infer = lambda *a: np.tile([.5, 0, 0, 0, 0, 0, 1], (20, 1))

        def events(*_):
            held = bool(actions) and not (lose_grasp and len(actions) > 1)
            pos = [0, 0, .445] if not actions else [.1, 0, .5]
            return {'position': np.array(pos), 'grasped': held, 'success': False}

        probe.events = events
        return probe, env, actions

    def test_stabilizing_actions_are_part_of_the_replay_prefix_and_budget(self):
        probe, env, actions = self.fixture()
        prefix = probe.prefix(21)
        self.assertEqual(prefix.shape, (4, 7))
        self.assertEqual(probe.counts['prefix_env_steps'], 4)
        np.testing.assert_array_equal(prefix, np.asarray(actions))
        np.testing.assert_array_equal(prefix[1:, :6], 0)
        np.testing.assert_array_equal(prefix[1:, -1], 1)
        record = json.loads((probe.directory / 'reference_prefixes/seed021.json').read_text())
        self.assertEqual(record['trigger_env_steps'], 1)
        self.assertEqual(record['stabilization_env_steps'], 3)
        self.assertEqual(record['env_steps'], 4)
        self.assertTrue(env.closed)

    def test_reference_still_moving_is_rejected(self):
        probe, env, _ = self.fixture(speed=.1)
        with self.assertRaisesRegex(AssertionError, 'Reference still moving for seed 21'):
            probe.prefix(21)
        self.assertTrue(env.closed)

    def test_reference_that_loses_grasp_is_rejected(self):
        probe, env, _ = self.fixture(lose_grasp=True)
        with self.assertRaisesRegex(AssertionError, 'Unstable held reference for seed 21'):
            probe.prefix(21)
        self.assertTrue(env.closed)


if __name__ == '__main__':
    unittest.main()
