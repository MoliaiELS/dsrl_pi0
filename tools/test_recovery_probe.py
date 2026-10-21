"""CPU checks for recorded reference settling and rejection of unstable holds."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from tools.recovery_probe import RecoveryProbe


class ReferenceStabilization(unittest.TestCase):
    def fixture(self, speed=0, lose_grasp=False, speeds=None):
        temp = tempfile.TemporaryDirectory(prefix='dsrl-reference-test-')
        self.addCleanup(temp.cleanup)
        actions = []
        def velocity(_):
            value = speed if speeds is None else speeds[min(len(actions) - 4, len(speeds) - 1)]
            return np.array([value, 0, 0, 0, 0, 0])

        data = SimpleNamespace(get_joint_qvel=velocity)
        env = SimpleNamespace(sim=SimpleNamespace(data=data), closed=False)

        def step(action):
            actions.append(np.array(action))
            return {}, 0, False, {}

        env.step = step
        env.close = lambda: setattr(env, 'closed', True)
        probe = RecoveryProbe.__new__(RecoveryProbe)
        probe.cfg = {'query_freq': 20, 'reference_max_steps': 4, 'reference_prefix_steps': 1,
                     'reference_lift_m': .04, 'reference_translation_m': .01,
                     'reference_grasp_hold_steps': 3, 'reference_stabilize_steps': 3,
                     'reference_stabilize_max_steps': 8, 'episode_budget': 400,
                     'min_remaining_steps': 200, 'settle_steps': 20,
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
            held = bool(actions) and not (lose_grasp and len(actions) > 3)
            pos = [0, 0, .445] if not actions else [.1, 0, .5]
            return {'position': np.array(pos), 'grasped': held, 'success': False}

        probe.events = events
        return probe, env, actions

    def test_stabilizing_actions_are_part_of_the_replay_prefix_and_budget(self):
        probe, env, actions = self.fixture()
        prefix = probe.prefix(21)
        self.assertEqual(prefix.shape, (6, 7))
        self.assertEqual(probe.counts['prefix_env_steps'], 6)
        np.testing.assert_array_equal(prefix, np.asarray(actions))
        np.testing.assert_array_equal(prefix[3:, :6], 0)
        np.testing.assert_array_equal(prefix[3:, -1], 1)
        record = json.loads((probe.directory / 'reference_prefixes/seed021.json').read_text())
        self.assertEqual(record['trigger_env_steps'], 3)
        self.assertEqual(record['stabilization_env_steps'], 3)
        self.assertEqual(record['env_steps'], 6)
        self.assertTrue(env.closed)

    def test_reference_still_moving_is_rejected(self):
        probe, env, actions = self.fixture(speed=.1)
        with self.assertRaisesRegex(AssertionError, 'Reference still moving for seed 21'):
            probe.prefix(21)
        self.assertTrue(env.closed)
        self.assertEqual(len(actions), 11)
        record = json.loads((probe.directory / 'reference_prefixes/seed021_failure.json').read_text())
        self.assertEqual(record['stabilization_env_steps'], 8)

    def test_reference_that_loses_grasp_is_rejected(self):
        probe, env, _ = self.fixture(lose_grasp=True)
        with self.assertRaisesRegex(AssertionError, 'Unstable held reference for seed 21'):
            probe.prefix(21)
        self.assertTrue(env.closed)

    def test_waits_past_the_old_endpoint_until_the_full_window_is_stable(self):
        probe, env, actions = self.fixture(speeds=[.03109985, .01065194, .01386708, .01])
        prefix = probe.prefix(21)
        record = json.loads((probe.directory / 'reference_prefixes/seed021.json').read_text())
        self.assertEqual(record['trigger_env_steps'], 3)
        self.assertEqual(record['stabilization_env_steps'], 4)
        self.assertEqual(len(prefix), 7)
        self.assertEqual(probe.counts['prefix_env_steps'], 7)
        self.assertTrue(env.closed)

    def test_already_slow_reference_still_observes_the_minimum_wait(self):
        probe, env, actions = self.fixture()
        probe.cfg['reference_stabilize_steps'] = 5
        probe.prefix(21)
        self.assertEqual(len(actions), 8)
        record = json.loads((probe.directory / 'reference_prefixes/seed021.json').read_text())
        self.assertEqual(record['stabilization_env_steps'], 5)
        self.assertTrue(env.closed)

    def test_never_exceeds_the_common_episode_budget(self):
        probe, env, actions = self.fixture(speed=.1)
        probe.cfg.update(episode_budget=9, min_remaining_steps=1, settle_steps=2)
        with self.assertRaisesRegex(AssertionError, 'after 3 steps'):
            probe.prefix(21)
        self.assertEqual(len(actions), 6)
        self.assertTrue(env.closed)


if __name__ == '__main__':
    unittest.main()
