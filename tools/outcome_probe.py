"""Paired full-budget instruction interventions with a frozen learned noise actor."""
import argparse
from collections import defaultdict
import copy
import hashlib
import importlib.metadata as metadata
import itertools
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import traceback
from types import SimpleNamespace

import numpy as np
from PIL import Image

from tools.control_probe import Probe, digest
from tools.run_logger import RunLogger, write_json


class OutcomeProbe(Probe):
    def __init__(self, config, directory, logger):
        super().__init__(config, directory, logger)
        import jax
        from flax.training import checkpoints
        from jaxrl2.agents.pixel_sac.pixel_sac_learner import PixelSACLearner
        from examples.train_utils_sim import obs_to_img

        self.jax, self.to_actor_image = jax, obs_to_img
        self.counts.update(settling_env_steps=0, states_built=0, controller_calls=0,
                           validation_rollouts=0, cases_successful=0)
        self.results, self.state_manifest = [], []
        self.frozen_reference = None
        data_root = Path(os.environ['DSRL_DATA_ROOT']).resolve()
        source = (data_root / config['controller']['run_dir']).resolve()
        assert source.is_relative_to(data_root), source
        source_cfg = json.loads((source / 'config.json').read_text())
        assert json.loads((source / 'SUCCESS.json').read_text())['passed']
        assert source_cfg['mode'] == 'train' and source_cfg['seed'] == config['controller'].get('training_seed', 0)
        assert source_cfg['task_id'] == config['task_id'] and source_cfg['task_suite'] == config['task_suite']
        expected_step = config['controller']['checkpoint_step']
        checkpoint = checkpoints.latest_checkpoint(str(source), prefix='checkpoint')
        assert checkpoint and Path(checkpoint).name == f'checkpoint{expected_step}', checkpoint
        self.actor_variant = SimpleNamespace(env='libero', resize_image=source_cfg['resize_image'])
        side = source_cfg['resize_image']
        observations = {'pixels': np.zeros((1, side, side, 3, 1), dtype=np.uint8),
                        'state': np.zeros((1, 8, 1), dtype=np.float32)}
        kwargs = copy.deepcopy(source_cfg['train_kwargs'])
        if kwargs.pop('cosine_decay', False):
            kwargs['decay_steps'] = source_cfg['max_steps']
        self.log('loading_noise_controller')
        self.actor = PixelSACLearner(0, observations, np.zeros((1, 1, 32), dtype=np.float32), **kwargs)
        self.actor.restore_checkpoint(str(source))
        assert int(np.asarray(self.actor._actor.step)) == expected_step
        self.controller_fingerprint = self.fingerprint()
        record = json.loads((directory / 'run.json').read_text())
        record.update(controller_checkpoint=str(checkpoint), controller_training_updates=expected_step,
                      controller_training_seed=source_cfg['seed'],
                      controller_fingerprint=self.controller_fingerprint,
                      controller_source_run=json.loads((source / 'run.json').read_text()))
        write_json(directory / 'run.json', record)
        self.log('noise_controller_loaded')

    def fingerprint(self):
        value = hashlib.sha256()
        for leaf in self.jax.tree_util.tree_leaves(self.actor._save_dict):
            array = np.asarray(leaf)
            value.update(str((array.shape, str(array.dtype))).encode())
            value.update(array.tobytes())
        return value.hexdigest()

    def reset_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        env = self.env_class(bddl_file_name=str(self.bddl), camera_heights=self.cfg['resolution'],
                             camera_widths=self.cfg['resolution'])
        try:
            env.seed(seed)
            return env, env.reset()
        except BaseException:
            env.close()
            raise

    def target(self, env):
        return env.env.objects_dict[self.cfg['target_object']]

    def events(self, env, anchor_z):
        body = env.env.obj_body_id[self.cfg['target_object']]
        pos = np.array(env.sim.data.body_xpos[body])
        grasped = bool(env.env._check_grasp(env.robots[0].gripper, self.target(env).contact_geoms))
        return {'position': pos, 'grasped': grasped, 'lifted': bool(pos[2] > anchor_z + 0.02),
                'success': bool(env.check_success())}

    def contacts(self, env):
        target = {env.sim.model.geom_name2id(n) for n in self.target(env).contact_geoms}
        model = env.sim.model._model
        root = env.sim.model.body_name2id(self.cfg['support_body'])
        support = set()
        for geom, body in enumerate(model.geom_bodyid):
            while body != 0 and body != root:
                body = model.body_parentid[body]
            if body == root:
                support.add(geom)
        assert target and support
        data = env.sim.data._data
        forbidden, supported = [], False
        for contact in data.contact[:data.ncon]:
            pair = {int(contact.geom1), int(contact.geom2)}
            if not pair.intersection(target):
                continue
            other = pair - target
            if other.intersection(support):
                supported = True
            elif float(contact.dist) < -self.cfg['penetration_tolerance_m']:
                forbidden.append({'geoms': sorted(pair), 'distance_m': float(contact.dist)})
        assert not forbidden, f'Target penetrates a non-support geometry: {forbidden}'
        return supported

    def position_ik_screen(self, env, position):
        """Evaluation-only geometric screen in separate MjData; no policy input."""
        import mujoco
        model = env.sim.model._model
        data = mujoco.MjData(model)
        data.qpos[:] = env.sim.data.qpos
        joints = env.robots[0].robot_model.joints
        qids = [env.sim.model.get_joint_qpos_addr(n) for n in joints]
        vids = [env.sim.model.get_joint_qvel_addr(n) for n in joints]
        ids = [env.sim.model.joint_name2id(n) for n in joints]
        limits = model.jnt_range[ids]
        site = env.sim.model.site_name2id(env.robots[0].gripper.important_sites['grip_site'])
        goal = position + np.array([0, 0, self.cfg['ik_screen_height_m']])
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        for _ in range(100):
            mujoco.mj_forward(model, data)
            error = goal - data.site_xpos[site]
            if np.linalg.norm(error) <= self.cfg['ik_screen_tolerance_m']:
                return float(np.linalg.norm(error))
            mujoco.mj_jacSite(model, data, jp, jr, site)
            jac = jp[:, vids]
            delta = jac.T @ np.linalg.solve(jac @ jac.T + 0.001 * np.eye(3), error)
            data.qpos[qids] = np.clip(data.qpos[qids] + np.clip(delta, -0.12, 0.12),
                                      limits[:, 0] + 1e-4, limits[:, 1] - 1e-4)
        raise AssertionError(f'Position IK screen failed: residual {np.linalg.norm(error)}')

    def apply_condition(self, env, condition, hold_action, *, ik_screen=False):
        before = self.events(env, 0)
        assert not before['grasped'] and not before['success'], 'Common state must precede target grasp/success'
        offset = np.asarray(condition['offset_m'], dtype=np.float64)
        assert offset.shape == (3,) and offset[2] == 0 and np.linalg.norm(offset) <= 0.05
        joint = self.target(env).joints[0]
        qpos = env.sim.data.get_joint_qpos(joint).copy()
        assert qpos.shape == (7,)
        qpos[:3] += offset
        env.sim.data.set_joint_qpos(joint, qpos)
        env.sim.data.set_joint_qvel(joint, np.zeros(6))
        env.sim.forward()
        self.contacts(env)
        for _ in range(self.cfg['settle_steps']):
            obs, _, done, _ = env.step(hold_action)
            self.counts['settling_env_steps'] += 1
            assert not done, 'State construction completed the original task'
        after = self.events(env, before['position'][2])
        assert not after['grasped'], 'Target grasped during common settling'
        assert self.contacts(env), 'Target has no contact with its support fixture'
        realized = after['position'] - before['position']
        assert np.linalg.norm(realized[:2] - offset[:2]) <= self.cfg['offset_tolerance_m'], realized
        assert abs(realized[2]) <= self.cfg['height_tolerance_m'], realized
        velocity = env.sim.data.get_joint_qvel(joint).copy()
        assert np.linalg.norm(velocity[:3]) <= self.cfg['max_settled_speed_mps'], velocity
        info = {'triggered': bool(np.linalg.norm(offset) > 0), 'requested_offset_m': offset.tolist(),
                'realized_offset_m': realized.tolist(), 'initial_target_position': after['position'].tolist(),
                'supported': True, 'no_forbidden_penetration': True, 'target_not_grasped': True,
                'settled_linear_speed_mps': float(np.linalg.norm(velocity[:3]))}
        if ik_screen:
            info['position_ik_residual_m'] = self.position_ik_screen(env, after['position'])
            info['ik_scope'] = 'position-only above-object reachability screen; not a recovery guarantee'
        return obs, info

    def prefix(self, seed):
        env, obs = self.reset_seed(seed)
        actions_out = []
        zero = np.zeros((1, 50, 32), dtype=np.float32)
        try:
            for step in range(self.cfg['reference_prefix_steps']):
                if step % self.cfg['query_freq'] == 0:
                    saved_obs = copy.deepcopy(obs)
                    actions = self.infer(obs, self.task.language, zero)
                    if self.frozen_reference is None:
                        self.frozen_reference = (saved_obs, actions.copy())
                action = actions[step % self.cfg['query_freq']]
                obs, _, done, _ = env.step(action)
                self.counts['prefix_env_steps'] += 1
                actions_out.append(action.copy())
                assert not done
            return np.asarray(actions_out)
        finally:
            env.close()

    def replay_prefix(self, seed, prefix):
        env, obs = self.reset_seed(seed)
        try:
            for action in prefix:
                obs, _, done, _ = env.step(action)
                self.counts['replay_env_steps'] += 1
                assert not done
            return env, obs
        except BaseException:
            env.close()
            raise

    def make_outcome_states(self):
        states = []
        folder = self.directory / 'common_states'
        folder.mkdir()
        for seed in self.cfg['reset_seeds']:
            prefix = self.prefix(seed)
            hold = np.zeros(7)
            hold[-1] = prefix[-1, -1]
            for condition in self.cfg['conditions']:
                env, _ = self.replay_prefix(seed, prefix)
                try:
                    obs, validation = self.apply_condition(env, condition, hold, ik_screen=True)
                    witness = self.snapshot(env, obs, prefix)
                finally:
                    env.close()
                name = f'seed{seed:03d}__{condition["id"]}'
                state = {'id': name, 'reset_seed': seed, 'condition': condition, 'prefix': prefix,
                         'hold_action': hold, 'witness': witness, 'validation': validation,
                         'remaining_budget': self.cfg['episode_budget'] - len(prefix) - self.cfg['settle_steps']}
                states.append(state)
                recipe = {k: v for k, v in state.items() if k not in ['prefix', 'hold_action', 'witness']}
                recipe.update(prefix_actions_sha256=digest(prefix), hold_action=hold.tolist(),
                              model_xml_sha256=witness['model_xml_sha256'])
                self.state_manifest.append(recipe)
                write_json(folder / (name + '.json'), recipe)
                np.savez_compressed(folder / (name + '.npz'), hold_action=hold,
                                    **{k: v for k, v in witness.items() if k != 'model_xml_sha256'})
                Image.fromarray(witness['agentview_image'][::-1, ::-1]).save(folder / (name + '.png'))
                self.counts['states_built'] += 1
                self.log('building_outcome_states', **{'probe/current_case': name})
        write_json(folder / 'manifest.json', {'states': self.state_manifest,
                   'method': 'seeded reset, recorded prefix, recorded offset, zero target velocity, common settling',
                   'C0': 'zero-offset sham, same velocity reset and settling as C1',
                   'scope': self.cfg.get('split', 'development states; no OOD claim')})
        return states

    def replay_outcome(self, state):
        env, _ = self.replay_prefix(state['reset_seed'], state['prefix'])
        try:
            obs, _ = self.apply_condition(env, state['condition'], state['hold_action'])
            actual = self.snapshot(env, obs, state['prefix'])
            expected = state['witness']
            assert actual['model_xml_sha256'] == expected['model_xml_sha256']
            errors = {}
            for key, tol in [('physics', self.cfg['physics_atol']), ('proprio', self.cfg['proprio_atol']),
                             ('agentview_image', self.cfg['image_atol']), ('wrist_image', self.cfg['image_atol'])]:
                errors[key] = float(np.max(np.abs(actual[key].astype(np.float64) - expected[key].astype(np.float64))))
                assert errors[key] <= tol, (state['id'], key, errors[key])
            self.restore_errors.append(errors)
            return env, obs
        except BaseException:
            env.close()
            raise

    def noise(self, obs, mode, reset_seed, query):
        base = self.cfg['noise_seed_base'] + reset_seed + 100000 * self.cfg['seed']
        key = self.jax.random.fold_in(self.jax.random.PRNGKey(base), query)
        if mode == 'base':
            return np.asarray(self.jax.random.normal(key, (1, 50, 32)))
        if mode == 'bounded_random':
            value = np.asarray(self.jax.random.uniform(key, (1, 1, 32), minval=-1, maxval=1))
        elif mode == 'dsrl':
            observations = {'pixels': self.to_actor_image(obs, self.actor_variant)[None, ..., None],
                            'state': self.to_proprio(obs, self.variant)[None, ..., None]}
            previous_rng = self.actor._rng
            try:
                self.actor._rng = key
                value = self.actor.sample_actions(observations).reshape(1, 1, 32)
                self.counts['controller_calls'] += 1
            finally:
                self.actor._rng = previous_rng
        else:
            raise ValueError(mode)
        assert np.isfinite(value).all() and np.max(np.abs(value)) <= 1.00001
        return np.repeat(value, 50, axis=1)

    def rollout(self, state, instruction, mode, *, save=True):
        env, obs = self.replay_outcome(state)
        name = '__'.join([state['id'], instruction['id'], mode])
        self.log('evaluating_outcomes', **{'probe/current_case': name})
        executed, noises, trace, frames, schedule = [], [], [], [], []
        first_grasp = first_lift = first_success = None
        total_return = 0.0
        save_video = (save and state['reset_seed'] in self.cfg['video_reset_seeds'] and
                      (instruction['id'] == 'original' and mode in ['base', 'dsrl']))
        try:
            for step in range(state['remaining_budget']):
                if step % self.cfg['query_freq'] == 0:
                    active = instruction if step < self.cfg['intervention_steps'] else self.cfg['instructions'][0]
                    noise = self.noise(obs, mode, state['reset_seed'], step // self.cfg['query_freq'])
                    actions = self.infer(obs, active['text'], noise)
                    noises.append(noise)
                    schedule.append({'step': step, 'instruction': active['id']})
                action = actions[step % self.cfg['query_freq']]
                obs, reward, done, _ = env.step(action)
                self.counts['continuation_env_steps'] += 1
                executed.append(action.copy())
                total_return += float(reward)
                event = self.events(env, state['validation']['initial_target_position'][2])
                if event['grasped'] and first_grasp is None:
                    first_grasp = step + 1
                if event['lifted'] and first_lift is None:
                    first_lift = step + 1
                if event['success'] and first_success is None:
                    first_success = step + 1
                trace.append([*self.to_proprio(obs, self.variant)[:3], *event['position'],
                              int(event['grasped']), int(event['lifted']), int(event['success'])])
                assert np.isfinite(trace[-1]).all()
                if save_video and step % self.cfg['video_stride'] == 0:
                    frames.append(Image.fromarray(obs['agentview_image'][::-1, ::-1]))
                if done:
                    break
            used = len(executed)
            assert len(state['prefix']) + self.cfg['settle_steps'] + used <= self.cfg['episode_budget']
            if instruction['id'] != 'original' and used > self.cfg['intervention_steps']:
                assert any(e['step'] == self.cfg['intervention_steps'] and e['instruction'] == 'original' for e in schedule)
            result = {'id': name, 'state_id': state['id'], 'reset_seed': state['reset_seed'],
                      'condition': state['condition']['id'], 'perturbation_triggered': state['validation']['triggered'],
                      'instruction': instruction['id'], 'noise_mode': mode, 'success': bool(env.check_success()),
                      'remaining_budget': state['remaining_budget'], 'continuation_steps': used,
                      'total_episode_steps': len(state['prefix']) + self.cfg['settle_steps'] + used,
                      'return': total_return, 'first_grasp_step': first_grasp, 'first_lift_step': first_lift,
                      'first_success_step': first_success, 'prompt_schedule': schedule,
                      'rng_recipe': {'base': self.cfg['noise_seed_base'] + state['reset_seed'] + 100000 * self.cfg['seed'],
                                     'fold_in': 'query index'},
                      'policy_kind': 'fixed instruction intervention with frozen noise policy; no learned selector'}
            arrays = {'executed_actions': np.asarray(executed), 'noise_sequence': np.asarray(noises),
                      'trace': np.asarray(trace), 'final_physics': env.get_sim_state().copy()}
        finally:
            env.close()
        if save:
            folder = self.directory / 'cases'
            folder.mkdir(exist_ok=True)
            write_json(folder / (name + '.json'), result)
            np.savez_compressed(folder / (name + '.npz'), **arrays)
            if frames:
                frames[0].save(folder / (name + '.gif'), save_all=True, append_images=frames[1:],
                               duration=50 * self.cfg['video_stride'], loop=0)
            self.results.append(result)
            self.counts['cases_completed'] += 1
            self.counts['cases_successful'] += int(result['success'])
            write_json(self.directory / 'outcome_results.json', self.results)
            self.log('evaluating_outcomes', **{'probe/current_case': name})
        return result, arrays

    def aggregate(self):
        groups = defaultdict(list)
        paired = {}
        for row in self.results:
            groups[(row['condition'], row['instruction'], row['noise_mode'])].append(row)
            paired[(row['state_id'], row['instruction'], row['noise_mode'])] = row
        summaries, comparisons = [], []
        for (condition, instruction, mode), rows in groups.items():
            triggered = rows[0]['perturbation_triggered']
            rate = sum(r['success'] for r in rows) / len(rows)
            summaries.append({'condition': condition, 'instruction': instruction, 'noise_mode': mode,
                              'episodes': len(rows), 'successes': sum(r['success'] for r in rows),
                              'success_rate': rate, 'recovery_rate_if_triggered': rate if triggered else None,
                              'mean_continuation_steps': float(np.mean([r['continuation_steps'] for r in rows]))})
            if instruction != 'original':
                original = [paired[(r['state_id'], 'original', mode)] for r in rows]
                comparisons.append({'condition': condition, 'instruction': instruction, 'noise_mode': mode,
                                    'paired_states': len(rows),
                                    'wins': sum(r['success'] and not o['success'] for r, o in zip(rows, original)),
                                    'losses': sum(o['success'] and not r['success'] for r, o in zip(rows, original))})
        return {'groups': summaries, 'paired_instruction_comparisons': comparisons}

    def run(self):
        self.log('building_outcome_states')
        states = self.make_outcome_states()
        evaluated = [s for s in states if s['reset_seed'] in self.cfg['eval_reset_seeds']]
        pairs = self.cfg['eval_pairs']
        self.counts['cases_planned'] = len(evaluated) * len(pairs)
        write_json(self.directory / 'config.json', self.cfg)
        instructions = {i['id']: i for i in self.cfg['instructions']}
        for state in evaluated:
            for instruction_id, mode in pairs:
                result, arrays = self.rollout(state, instructions[instruction_id], mode)
                if (instruction_id == 'original' and mode == 'dsrl' and
                        state['reset_seed'] in self.cfg['repeat_validation_reset_seeds'] and
                        state['condition']['id'] in self.cfg.get('repeat_validation_conditions', ['C0', 'C1_plus_x'])):
                    again, repeated = self.rollout(state, instructions[instruction_id], mode, save=False)
                    np.testing.assert_allclose(arrays['executed_actions'], repeated['executed_actions'], rtol=0, atol=self.cfg['action_atol'])
                    np.testing.assert_allclose(arrays['final_physics'], repeated['final_physics'], rtol=0, atol=self.cfg['physics_atol'])
                    assert result['success'] == again['success'] and result['prompt_schedule'] == again['prompt_schedule']
                    self.counts['validation_rollouts'] += 1
        obs, expected = self.frozen_reference
        actions = self.infer(obs, self.task.language, np.zeros((1, 50, 32), dtype=np.float32))
        np.testing.assert_allclose(actions, expected, rtol=0, atol=self.cfg['action_atol'])
        assert self.fingerprint() == self.controller_fingerprint, 'Controller state changed during inference'
        assert self.counts['cases_completed'] == self.counts['cases_planned']
        summary = {'scope': self.cfg.get('split', 'development') + '; fixed interventions, not Instruction RL or Joint RL',
                   'gradient_updates': 0, 'counts': self.counts, **self.aggregate(),
                   'max_replay_errors': {k: max(e[k] for e in self.restore_errors) for k in self.restore_errors[0]},
                   'controller_unchanged': True, 'frozen_vla_output_unchanged': True,
                   'reference_rollouts_reproducible': True,
                   'interpretation': 'C0 success and triggered C1 recovery are separate. Seeds/conditions are paired, not independent trials. No OOD claim.'}
        write_json(self.directory / 'outcome_summary.json', summary)
        self.log('completed', **{'probe/replay_verified': True, 'probe/controller_unchanged': True,
                                  'probe/frozen_vla_unchanged': True})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['outcome-smoke', 'outcome'], required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    folder = Path(args.output).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    protocol_path = Path(args.config)
    cfg = json.loads(protocol_path.read_text())
    if cfg.get('kind') == 'outcome_series':
        from tools.outcome_series import run_series
        run_series(args, cfg, folder, protocol_path)
        return
    cfg.update(seed=args.seed, mode=args.mode)
    if args.mode == 'outcome-smoke' and 'smoke_reset_seeds' in cfg:
        cfg['planned_reset_seeds'] = cfg['reset_seeds']
        cfg['reset_seeds'] = cfg['smoke_reset_seeds']
        cfg['planned_split'] = cfg.get('split', 'development')
        cfg['split'] = 'development smoke; confirmation states not evaluated'
    assert cfg['intervention_steps'] % cfg['query_freq'] == 0
    assert cfg['reference_prefix_steps'] > 0 and cfg['settle_steps'] > 0
    assert cfg['reference_prefix_steps'] + cfg['settle_steps'] + cfg['intervention_steps'] < cfg['episode_budget']
    assert set(cfg['reset_seeds']).isdisjoint(cfg['reserved_test_reset_seeds'])
    assert cfg['instructions'][0]['id'] == 'original'
    assert len(set(cfg['reset_seeds'])) == len(cfg['reset_seeds'])
    assert len({i['id'] for i in cfg['instructions']}) == len(cfg['instructions'])
    assert len({c['id'] for c in cfg['conditions']}) == len(cfg['conditions'])
    assert 'dsrl' in cfg['noise_modes'] and set(cfg['noise_modes']) <= {'base', 'bounded_random', 'dsrl'}
    if args.mode == 'outcome-smoke':
        cfg.update(eval_reset_seeds=cfg['smoke_eval_reset_seeds'], eval_pairs=cfg['smoke_pairs'])
    else:
        cfg.update(eval_reset_seeds=cfg['reset_seeds'],
                   eval_pairs=list(itertools.product([i['id'] for i in cfg['instructions']], cfg['noise_modes'])))
    assert set(cfg['eval_reset_seeds']).issubset(cfg['reset_seeds'])
    write_json(folder / 'config.json', cfg)
    write_json(folder / 'dependencies.json', {d.metadata['Name']: d.version for d in metadata.distributions()})
    shutil.copy2(Path(os.environ['DSRL_MODEL_PATH']) / 'download_manifest.json', folder / 'model_manifest.json')
    write_json(folder / 'run.json', {
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'submodules': subprocess.check_output(['git', 'submodule', 'status'], text=True),
        'mode': args.mode, 'seed': args.seed, 'job_id': os.environ.get('SLURM_JOB_ID'),
        'hostname': os.uname().nodename, 'python': sys.executable,
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'model_source': 's3://openpi-assets/checkpoints/pi0_libero', 'model_path': os.environ['DSRL_MODEL_PATH'],
        'learning': False, 'protocol_file_sha256': hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        'split': ('development smoke; confirmation states not evaluated' if args.mode == 'outcome-smoke'
                  and 'smoke_reset_seeds' in cfg else cfg.get('split', 'development; reserved test seeds are not evaluated')),
        'common_state_method': 'seeded fresh reset, recorded actions, recorded offset and common settling'})
    logger = RunLogger(folder)
    try:
        OutcomeProbe(cfg, folder, logger).run()
        write_json(folder / 'SUCCESS.json', {'mode': args.mode, 'passed': True})
    except BaseException as error:
        write_json(folder / 'FAILED.json', {'type': type(error).__name__, 'error': str(error),
                                          'traceback': traceback.format_exc()})
        raise
    finally:
        logger.close()


if __name__ == '__main__':
    main()
