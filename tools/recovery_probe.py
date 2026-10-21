"""Development pilot: physical gripper release after verified object lift."""
import copy
import json

import numpy as np

from tools.outcome_probe import OutcomeProbe
from tools.run_logger import write_json


class RecoveryProbe(OutcomeProbe):
    @staticmethod
    def eef_position(env):
        controller = env.robots[0].controller
        return np.asarray(env.sim.data.get_site_xpos(controller.eef_name)).copy()

    @classmethod
    def position_hold_action(cls, env, goal_position, gripper):
        controller = env.robots[0].controller
        assert controller.use_delta and controller.control_dim == 6, 'Expected delta OSC_POSE controller'
        lower, upper = controller.input_min[:3], controller.input_max[:3]
        out_lower, out_upper = controller.output_min[:3], controller.output_max[:3]
        scale = (out_upper - out_lower) / (upper - lower)
        assert np.all(scale > 0)
        # A zero delta re-anchors the OSC target to its current position on every
        # call, allowing drift. Invert the existing action scale to keep one fixed
        # world-position target; retain the previous orientation goal with zeros.
        delta = np.asarray(goal_position) - cls.eef_position(env)
        action = np.zeros(7, dtype=np.float32)
        action[:3] = np.clip((delta - (out_upper + out_lower) / 2) / scale
                             + (upper + lower) / 2, lower, upper)
        action[-1] = gripper
        return action

    def prefix(self, seed):
        self.log('building_reference_prefix', **{'probe/current_case': f'reference_seed{seed:03d}'})
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
                    trigger_steps = len(actions_out)
                    trigger_position = event['position'].copy()
                    hold_position = self.eef_position(env)
                    settling = []
                    minimum = self.cfg.get('reference_stabilize_steps', 0)
                    requested_maximum = self.cfg.get('reference_stabilize_max_steps', minimum)
                    window = self.cfg['reference_grasp_hold_steps']
                    budget_cap = (self.cfg['episode_budget'] - self.cfg['min_remaining_steps']
                                  - self.cfg['settle_steps'] - trigger_steps)
                    maximum = min(requested_maximum, budget_cap)
                    assert 0 < window <= minimum <= maximum, (
                        f'Insufficient stabilization budget for seed {seed}: '
                        f'min={minimum}, max={maximum}, remaining_cap={budget_cap}')
                    # Stop the moving reference before either intervention. These actions
                    # belong to the shared replay prefix and consume the episode budget.
                    for _ in range(maximum):
                        hold = self.position_hold_action(env, hold_position, 1)
                        obs, _, done, _ = env.step(hold)
                        self.counts['prefix_env_steps'] += 1
                        actions_out.append(hold.copy())
                        assert not done, f'Reference stabilization completed goal for seed {seed}'
                        event = self.events(env, initial[2])
                        velocity = env.sim.data.get_joint_qvel(self.target(env).joints[0])
                        settling.append({'grasped': event['grasped'],
                                         'linear_speed_mps': float(np.linalg.norm(velocity[:3])),
                                         'target_position_m': event['position'].tolist(),
                                         'eef_position_m': self.eef_position(env).tolist()})
                        tail = settling[-window:]
                        stable_hold = (len(tail) == window and all(s['grasped'] for s in tail)
                                       and max(s['linear_speed_mps'] for s in tail)
                                       <= self.cfg['max_reference_speed_mps'])
                        if len(settling) >= minimum and stable_hold:
                            break
                    folder = self.directory / 'reference_prefixes'
                    folder.mkdir(exist_ok=True)
                    if not stable_hold:
                        write_json(folder / f'seed{seed:03d}_failure.json', {
                            'reset_seed': seed, 'trigger_env_steps': trigger_steps,
                            'stabilization_env_steps': len(settling), 'stabilization_maximum': maximum,
                            'fixed_eef_goal_m': hold_position.tolist(),
                            'stabilization_samples': settling, 'prefix_env_steps': len(actions_out)})
                        self.log('reference_stabilization_failed', **{
                            'probe/current_case': f'reference_seed{seed:03d}',
                            'probe/reference_stabilization_steps': len(settling)})
                        assert all(s['grasped'] for s in tail), f'Unstable held reference for seed {seed}: {tail}'
                        raise AssertionError(f'Reference still moving for seed {seed} after {maximum} steps: {tail}')
                    write_json(folder / f'seed{seed:03d}.json', {
                        'reset_seed': seed, 'env_steps': len(actions_out),
                        'trigger_env_steps': trigger_steps, 'stabilization_env_steps': len(settling),
                        'stabilization_minimum': minimum, 'stabilization_maximum': maximum,
                        'stabilization_budget_cap': budget_cap, 'stable_hold_window': window,
                        'stabilization_samples': settling,
                        'fixed_eef_goal_m': hold_position.tolist(),
                        'held_position_before_stabilization': trigger_position.tolist(),
                        'stabilization_tail': settling[-self.cfg['reference_grasp_hold_steps']:],
                        'initial_target_position': initial.tolist(), 'held_target_position': event['position'].tolist(),
                        'grasped': event['grasped'], 'stable_trigger_steps': stable,
                        'reference_controller_run': self.cfg['controller']['run_dir'],
                        'reference_rng_query_offset': 100000,
                        'trigger_truth_scope': 'evaluation-only trigger; controller reads images and proprioception'})
                    self.log('reference_stabilized', **{
                        'probe/current_case': f'reference_seed{seed:03d}',
                        'probe/reference_stabilization_steps': len(settling)})
                    return np.asarray(actions_out)
            raise AssertionError(f'No stable lifted-object trigger within {self.cfg["reference_max_steps"]} steps for seed {seed}')
        finally:
            env.close()

    def apply_condition(self, env, condition, hold_action, *, ik_screen=False):
        before = self.events(env, 0)
        assert before['grasped'] and not before['success'], 'Reference must hold the target before release/sham'
        release = condition['release']
        hold_position = np.array(env.robots[0].controller.goal_pos).copy()
        gripper = -1 if release else 1
        for _ in range(self.cfg['settle_steps']):
            hold = self.position_hold_action(env, hold_position, gripper)
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
            assert np.linalg.norm(delta) <= self.cfg['max_sham_displacement_m'], (
                f'{condition["id"]} drift {np.linalg.norm(delta):.6f} m exceeds '
                f'{self.cfg["max_sham_displacement_m"]:.6f} m; delta={delta.tolist()}')
        validation = {'triggered': bool(release), 'release_gripper_command': float(gripper),
                      'fixed_eef_goal_m': hold_position.tolist(),
                      'initial_target_position': after['position'].tolist(),
                      'held_position_before': before['position'].tolist(), 'drop_displacement_m': delta.tolist(),
                      'target_not_grasped': not after['grasped'], 'supported': bool(release),
                      'no_forbidden_penetration': True if release else None,
                      'settled_linear_speed_mps': float(np.linalg.norm(velocity[:3])),
                      'scope': 'C2 release / C0 closed gripper at the same fixed EEF position for identical settling duration'}
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
