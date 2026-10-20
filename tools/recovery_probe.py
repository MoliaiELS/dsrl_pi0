"""Development pilot: physical gripper release after verified object lift."""
import copy
import json

import numpy as np

from tools.outcome_probe import OutcomeProbe
from tools.run_logger import write_json


class RecoveryProbe(OutcomeProbe):
    def prefix(self, seed):
        env, obs = self.reset_seed(seed)
        actions_out, stable = [], 0
        initial = self.events(env, 0)['position'].copy()
        zero = np.zeros((1, 50, 32), dtype=np.float32)
        try:
            for step in range(self.cfg['reference_max_steps']):
                if step % self.cfg['query_freq'] == 0:
                    if self.frozen_reference is None:
                        saved = copy.deepcopy(obs)
                        self.frozen_reference = (saved, self.infer(obs, self.task.language, zero).copy())
                    # Separate namespace from continuation RNG. Same recorded prefix for all methods.
                    noise = self.noise(obs, 'dsrl', seed, 100000 + step // self.cfg['query_freq'])
                    actions = self.infer(obs, self.task.language, noise)
                action = actions[step % self.cfg['query_freq']]
                obs, _, done, _ = env.step(action)
                self.counts['prefix_env_steps'] += 1
                actions_out.append(action.copy())
                assert not done, 'Reference reached goal before the held-object trigger'
                event = self.events(env, initial[2])
                eligible = (event['grasped'] and event['position'][2] - initial[2] >= self.cfg['reference_lift_m']
                            and np.linalg.norm(event['position'][:2] - initial[:2]) >= self.cfg['reference_translation_m'])
                stable = stable + 1 if eligible else 0
                if step + 1 >= self.cfg['reference_prefix_steps'] and stable >= self.cfg['reference_grasp_hold_steps']:
                    folder = self.directory / 'reference_prefixes'
                    folder.mkdir(exist_ok=True)
                    write_json(folder / f'seed{seed:03d}.json', {
                        'reset_seed': seed, 'env_steps': len(actions_out),
                        'initial_target_position': initial.tolist(), 'held_target_position': event['position'].tolist(),
                        'grasped': event['grasped'], 'stable_trigger_steps': stable,
                        'reference_controller_run': self.cfg['controller']['run_dir'],
                        'reference_rng_query_offset': 100000,
                        'trigger_truth_scope': 'evaluation-only trigger; controller reads images and proprioception'})
                    return np.asarray(actions_out)
            raise AssertionError(f'No stable lifted-object trigger within {self.cfg["reference_max_steps"]} steps for seed {seed}')
        finally:
            env.close()

    def apply_condition(self, env, condition, hold_action, *, ik_screen=False):
        before = self.events(env, 0)
        assert before['grasped'] and not before['success'], 'Reference must hold the target before release/sham'
        release = condition['release']
        hold = np.zeros(7)
        hold[-1] = -1 if release else 1
        for _ in range(self.cfg['settle_steps']):
            obs, _, done, _ = env.step(hold)
            self.counts['settling_env_steps'] += 1
            assert not done, 'Original goal completed while constructing the common state'
        after = self.events(env, before['position'][2])
        delta = after['position'] - before['position']
        velocity = env.sim.data.get_joint_qvel(self.target(env).joints[0]).copy()
        if release:
            assert not after['grasped'], 'Release did not separate target and gripper'
            assert self.contacts(env), 'Dropped target is not supported by the table'
            assert -delta[2] >= self.cfg['min_drop_height_m'], delta
            assert np.linalg.norm(delta[:2]) <= self.cfg['max_drop_horizontal_m'], delta
            assert np.linalg.norm(velocity[:3]) <= self.cfg['max_settled_speed_mps'], velocity
        else:
            assert after['grasped'], 'C0 held-object sham lost the target'
            assert np.linalg.norm(delta) <= self.cfg['max_sham_displacement_m'], delta
        validation = {'triggered': bool(release), 'release_gripper_command': float(hold[-1]),
                      'initial_target_position': after['position'].tolist(),
                      'held_position_before': before['position'].tolist(), 'drop_displacement_m': delta.tolist(),
                      'target_not_grasped': not after['grasped'], 'supported': bool(release),
                      'no_forbidden_penetration': True if release else None,
                      'settled_linear_speed_mps': float(np.linalg.norm(velocity[:3])),
                      'scope': 'C2 physical release after lift; C0 keeps holding for identical settling duration'}
        if ik_screen:
            validation['position_ik_residual_m'] = self.position_ik_screen(env, after['position'])
            validation['ik_scope'] = 'position-only above-object screen; no recovery guarantee'
        return obs, validation

    def make_outcome_states(self):
        states = super().make_outcome_states()
        for state in states:
            assert state['remaining_budget'] >= self.cfg['min_remaining_steps'], state['id']
        path = self.directory / 'common_states' / 'manifest.json'
        manifest = json.loads(path.read_text())
        manifest.update(method=self.cfg['common_state_method'],
                        C0='gripper held closed after the same lifted-object prefix and settling duration',
                        trigger='stable grasp plus lift and horizontal movement; physical gripper release for C2',
                        scope=self.cfg['split'])
        write_json(path, manifest)
        return states
