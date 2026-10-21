"""CPU checks for recorded reference settling and rejection of unstable holds."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from tools.recovery_probe import RecoveryProbe
from tools.outcome_probe import OutcomeProbe, ReferenceNotTriggered


class ReferenceStabilization(unittest.TestCase):
    def fixture(self, speed=0, lose_grasp=False, speeds=None):
        temp = tempfile.TemporaryDirectory(prefix='dsrl-reference-test-')
        self.addCleanup(temp.cleanup)
        actions = []
        def velocity(_):
            value = speed if speeds is None else speeds[min(len(actions) - 4, len(speeds) - 1)]
            return np.array([value, 0, 0, 0, 0, 0])

        data = SimpleNamespace(get_joint_qvel=velocity, get_site_xpos=lambda _: np.zeros(3))
        controller = SimpleNamespace(eef_name='grip_site', use_delta=True, control_dim=6,
                                     input_min=-np.ones(6), input_max=np.ones(6),
                                     output_min=-np.array([.05] * 3 + [.5] * 3),
                                     output_max=np.array([.05] * 3 + [.5] * 3), goal_pos=np.zeros(3))
        env = SimpleNamespace(sim=SimpleNamespace(data=data), closed=False,
                              env=SimpleNamespace(control_timestep=.05),
                              robots=[SimpleNamespace(controller=controller)])

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
                     'max_sham_displacement_m': .015,
                     'reference_speed_measure': 'control_interval_displacement',
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
            if len(actions) > 3:
                values = [speed] * (len(actions) - 3) if speeds is None else [
                    speeds[min(i, len(speeds) - 1)] for i in range(len(actions) - 3)]
                pos[0] += .05 * sum(values)
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

    def test_hold_feedback_keeps_one_world_position_target(self):
        probe, env, _ = self.fixture()
        current = np.array([.01, -.02, .005])
        env.sim.data.get_site_xpos = lambda _: current
        action = probe.position_hold_action(env, np.zeros(3), 1)
        np.testing.assert_allclose(current + action[:3] * .05, 0, atol=1e-8)
        np.testing.assert_array_equal(action[3:6], 0)
        self.assertEqual(action[-1], 1)

    def test_hold_feedback_is_bounded_and_preserves_release_command(self):
        probe, env, _ = self.fixture()
        action = probe.position_hold_action(env, np.array([1, -1, 1]), -1)
        np.testing.assert_array_equal(action[:3], [1, -1, 1])
        self.assertEqual(action[-1], -1)

    def test_hold_feedback_rejects_an_incompatible_controller(self):
        probe, env, _ = self.fixture()
        env.robots[0].controller.use_delta = False
        with self.assertRaisesRegex(AssertionError, 'delta OSC_POSE'):
            probe.position_hold_action(env, np.zeros(3), 1)

    def test_contact_velocity_jitter_is_logged_separately_from_observed_motion(self):
        probe, env, _ = self.fixture()
        env.sim.data.get_joint_qvel = lambda _: np.array([.03, 0, 0, 0, 0, 0])
        probe.prefix(21)
        record = json.loads((probe.directory / 'reference_prefixes/seed021.json').read_text())
        self.assertEqual(record['speed_measure'], 'control_interval_displacement')
        self.assertEqual(record['stabilization_tail'][-1]['linear_speed_mps'], 0)
        self.assertEqual(record['stabilization_tail'][-1]['instantaneous_linear_speed_mps'], .03)

    def test_position_motion_cannot_pass_due_to_a_low_instantaneous_sample(self):
        probe, env, _ = self.fixture(speed=.1)
        env.sim.data.get_joint_qvel = lambda _: np.zeros(6)
        with self.assertRaisesRegex(AssertionError, 'Reference still moving'):
            probe.prefix(21)

    def test_large_window_motion_cannot_pass_on_three_quiet_endpoint_samples(self):
        probe, env, _ = self.fixture(speeds=[0, .4, 0, 0, 0])
        probe.cfg.update(reference_stabilize_steps=5, reference_stabilize_max_steps=5)
        with self.assertRaisesRegex(AssertionError, 'window span=0.020000'):
            probe.prefix(21)

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

    def test_untriggered_reference_records_all_attempted_actions_and_reason(self):
        probe, env, actions = self.fixture()
        probe.cfg['reference_max_steps'] = 2
        with self.assertRaises(ReferenceNotTriggered) as caught:
            probe.prefix(24)
        record = caught.exception.record
        self.assertEqual(record['reset_seed'], 24)
        self.assertEqual(record['reason'], 'reference_prefix_budget_exhausted')
        self.assertEqual(record['prefix_env_steps'], 2)
        saved = np.load(probe.directory / record['actions_file'])['actions']
        np.testing.assert_array_equal(saved, np.asarray(actions))
        self.assertTrue(env.closed)


class TriggerCohortAccounting(unittest.TestCase):
    def fixture(self):
        temp = tempfile.TemporaryDirectory(prefix='dsrl-trigger-cohort-')
        self.addCleanup(temp.cleanup)
        probe = OutcomeProbe.__new__(OutcomeProbe)
        probe.directory = Path(temp.name)
        probe.cfg = {'reset_seeds': [24, 20], 'conditions': [{'id': 'C0'}],
                     'allow_untriggered_references': True, 'episode_budget': 400, 'settle_steps': 20}
        probe.counts, probe.state_manifest = {'states_built': 0}, []
        probe.log = lambda *a, **kw: None

        def prefix(seed):
            if seed == 24:
                raise ReferenceNotTriggered({'reset_seed': 24, 'reason': 'reference_prefix_budget_exhausted'})
            return np.zeros((1, 7))

        env = SimpleNamespace(closed=False)
        env.close = lambda: setattr(env, 'closed', True)
        probe.prefix = prefix
        probe.replay_prefix = lambda *_: (env, {})
        probe.apply_condition = lambda *a, **kw: ({}, {'triggered': False})
        probe.snapshot = lambda *_: {'physics': np.zeros(3), 'proprio': np.zeros(8),
            'agentview_image': np.zeros((2, 2, 3), dtype=np.uint8),
            'wrist_image': np.zeros((2, 2, 3), dtype=np.uint8), 'model_xml_sha256': 'fixture'}
        return probe, env

    def test_nontrigger_is_reported_without_removing_the_requested_seed(self):
        probe, env = self.fixture()
        states = probe.make_outcome_states()
        self.assertEqual([state['reset_seed'] for state in states], [20])
        self.assertEqual(probe.cfg['reset_seeds'], [24, 20])
        self.assertEqual(probe.counts['reference_attempts'], 2)
        self.assertEqual(probe.counts['reference_triggered'], 1)
        self.assertEqual(probe.counts['reference_not_triggered'], 1)
        manifest = json.loads((probe.directory / 'common_states/manifest.json').read_text())
        self.assertEqual(manifest['untriggered_references'][0]['reset_seed'], 24)
        self.assertTrue(env.closed)

    def test_no_eligible_reference_cannot_be_reported_as_success(self):
        probe, _ = self.fixture()
        probe.cfg['reset_seeds'] = [24]
        with self.assertRaisesRegex(AssertionError, 'No eligible common reference states'):
            probe.make_outcome_states()

    def test_physically_invalid_condition_still_stops_the_evaluation(self):
        probe, env = self.fixture()
        probe.cfg['reset_seeds'] = [20]

        def invalid(*args, **kwargs):
            raise AssertionError('Dropped target is not supported by the table')

        probe.apply_condition = invalid
        with self.assertRaisesRegex(AssertionError, 'not supported'):
            probe.make_outcome_states()
        self.assertEqual(probe.counts.get('reference_not_triggered', 0), 0)
        self.assertTrue(env.closed)

    def test_nontrigger_handling_requires_explicit_protocol_opt_in(self):
        probe, _ = self.fixture()
        probe.cfg['allow_untriggered_references'] = False
        with self.assertRaises(ReferenceNotTriggered):
            probe.make_outcome_states()


if __name__ == '__main__':
    unittest.main()
