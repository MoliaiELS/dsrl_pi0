"""Paired instruction/noise probes with verified reset-and-action replay; no RL."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import random
import shutil
from types import SimpleNamespace
import subprocess
import sys
import traceback

# Match the verified baseline's XLA setting before importing JAX/OpenPI.
if '--xla_gpu_triton_gemm_any' not in os.environ.get('XLA_FLAGS', ''):
    os.environ['XLA_FLAGS'] = os.environ.get('XLA_FLAGS', '') + ' --xla_gpu_triton_gemm_any=True'

import numpy as np
from PIL import Image

from tools.run_logger import RunLogger, write_json


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def noise_sequence(condition, queries, horizon=50, dim=32):
    """The same explicit noise sequence is reused for every instruction/state."""
    rng = np.random.default_rng(condition['seed'])
    kind = condition['kind']
    if kind == 'gaussian_full':
        return rng.normal(size=(queries, 1, horizon, dim)).astype(np.float32)
    if kind == 'zero_repeat':
        values = np.zeros((queries, 1, 1, dim), dtype=np.float32)
    elif kind == 'uniform_repeat':
        values = rng.uniform(-1, 1, (queries, 1, 1, dim)).astype(np.float32)
    elif kind == 'gaussian_repeat':
        # Match the first time position of gaussian_full at every query.
        values = rng.normal(size=(queries, 1, horizon, dim)).astype(np.float32)[:, :, :1, :]
    else:
        raise ValueError(f'Unknown noise condition: {kind}')
    return np.repeat(values, horizon, axis=2)


class Probe:
    def __init__(self, config, directory, logger):
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        from examples.train_utils_sim import obs_to_pi_zero_input, obs_to_qpos
        from openpi.training import config as openpi_config
        from openpi.policies import policy_config

        self.cfg, self.directory, self.logger = config, directory, logger
        self.env_class = OffScreenRenderEnv
        self.to_input, self.to_proprio = obs_to_pi_zero_input, obs_to_qpos
        self.task = benchmark.get_benchmark_dict()[config['task_suite']]().get_task(config['task_id'])
        self.bddl = Path(get_libero_path('bddl_files')) / self.task.problem_folder / self.task.bddl_file
        self.variant = SimpleNamespace(env='libero', task_description=self.task.language)
        self.counts = {'prefix_env_steps': 0, 'replay_env_steps': 0,
                       'continuation_env_steps': 0, 'vla_calls': 0, 'cases_completed': 0}
        self.restore_errors = []
        self.cfg['original_goal'] = self.task.language
        for instruction in self.cfg['instructions']:
            instruction['text'] = instruction['text'].replace('{task}', self.task.language)
        self.log('loading_frozen_vla')
        self.policy = policy_config.create_trained_policy(
            openpi_config.get_config(config['model_config']), Path(os.environ['DSRL_MODEL_PATH']))
        assert (self.policy.action_horizon, self.policy.action_dim) == (50, 32)
        self.log('frozen_vla_loaded')

    def log(self, phase, **values):
        self.logger.log({'phase': phase, **{'probe/' + k: v for k, v in self.counts.items()},
                         **values}, 0)

    def reset(self):
        # Fresh environment reconstructs controller, observable and task history too.
        random.seed(self.cfg['seed'])
        np.random.seed(self.cfg['seed'])
        env = self.env_class(bddl_file_name=str(self.bddl),
                             camera_heights=self.cfg['resolution'], camera_widths=self.cfg['resolution'])
        try:
            env.seed(self.cfg['seed'])
            return env, env.reset()
        except BaseException:
            env.close()
            raise

    def infer(self, obs, prompt, noise):
        import jax.numpy as jnp
        variant = SimpleNamespace(env='libero', task_description=prompt)
        actions = np.asarray(self.policy.infer(self.to_input(obs, variant), noise=jnp.asarray(noise))['actions'])
        self.counts['vla_calls'] += 1
        assert actions.ndim == 2 and actions.shape[0] >= self.cfg['query_freq'] and actions.shape[1] == 7
        assert np.isfinite(actions).all(), 'Nonfinite VLA actions'
        return actions

    def snapshot(self, env, obs, prefix):
        return {'prefix_actions': np.asarray(prefix).reshape(-1, 7),
                'physics': env.get_sim_state().copy(),
                'proprio': self.to_proprio(obs, self.variant).copy(),
                'agentview_image': obs['agentview_image'].copy(),
                'wrist_image': obs['robot0_eye_in_hand_image'].copy(),
                'model_xml_sha256': hashlib.sha256(env.sim.model.get_xml().encode()).hexdigest()}

    def make_states(self):
        self.log('building_common_states')
        states, prefix = {}, []
        env, obs = self.reset()
        zero = np.zeros((1, 50, 32), dtype=np.float32)
        try:
            for step in range(max(self.cfg['prefix_steps']) + 1):
                if step in self.cfg['prefix_steps']:
                    name = f'normal_step_{step:03d}'
                    states[name] = self.snapshot(env, obs, prefix)
                if step == max(self.cfg['prefix_steps']):
                    break
                if step % self.cfg['query_freq'] == 0:
                    actions = self.infer(obs, self.task.language, zero)
                action = actions[step % self.cfg['query_freq']]
                obs, _, done, _ = env.step(action)
                self.counts['prefix_env_steps'] += 1
                prefix.append(action.copy())
                assert not done, 'Reference prefix terminated before the selected state; revise the protocol'
        finally:
            env.close()
        folder = self.directory / 'common_states'
        folder.mkdir()
        manifest = []
        for name, state in states.items():
            np.savez_compressed(folder / (name + '.npz'), **{k: v for k, v in state.items() if k != 'model_xml_sha256'})
            Image.fromarray(state['agentview_image'][::-1, ::-1]).save(folder / (name + '.png'))
            manifest.append({'id': name, 'reset_seed': self.cfg['seed'],
                             'prefix_steps': len(state['prefix_actions']),
                             'remaining_episode_budget': self.cfg['episode_budget'] - len(state['prefix_actions']),
                             'prefix_actions_sha256': digest(state['prefix_actions']),
                             'physics_sha256': digest(state['physics']),
                             'agentview_sha256': digest(state['agentview_image']),
                             'model_xml_sha256': state['model_xml_sha256']})
        write_json(folder / 'manifest.json', {
            'method': 'fresh seeded environment reset plus exact recorded action-prefix replay',
            'scope': 'normal states only; simulator vector is a verification witness, not a full controller serialization',
            'reference_policy': 'original instruction and zero noise, not the Gaussian Base benchmark',
            'states': manifest})
        return states

    def replay(self, state):
        env, obs = self.reset()
        try:
            for action in state['prefix_actions']:
                obs, _, done, _ = env.step(action)
                self.counts['replay_env_steps'] += 1
                assert not done, 'Replayed prefix terminated'
            actual = self.snapshot(env, obs, state['prefix_actions'])
            errors = {}
            assert actual['model_xml_sha256'] == state['model_xml_sha256'], 'Model XML changed across replay'
            for key, tol in [('physics', self.cfg['physics_atol']), ('proprio', self.cfg['proprio_atol']),
                             ('agentview_image', self.cfg['image_atol']), ('wrist_image', self.cfg['image_atol'])]:
                errors[key] = float(np.max(np.abs(actual[key].astype(np.float64) - state[key].astype(np.float64))))
                assert errors[key] <= tol, f'Common-state replay mismatch: {key} {errors[key]} > {tol}'
            self.restore_errors.append(errors)
            return env, obs
        except BaseException:
            env.close()
            raise

    def case(self, state, instruction, condition, name, *, save=True):
        env, obs = self.replay(state)
        limit = min(self.cfg['continuation_steps'], self.cfg['episode_budget'] - len(state['prefix_actions']))
        noises = noise_sequence(condition, (limit + self.cfg['query_freq'] - 1) // self.cfg['query_freq'])
        frames, executed, rewards, first = [], [], [], None
        start_proprio = self.to_proprio(obs, self.variant).copy()
        try:
            for step in range(limit):
                if step % self.cfg['query_freq'] == 0:
                    actions = self.infer(obs, instruction['text'], noises[step // self.cfg['query_freq']])
                    if first is None:
                        first = actions[:min(limit, self.cfg['query_freq'])].copy()
                action = actions[step % self.cfg['query_freq']]
                obs, reward, done, _ = env.step(action)
                self.counts['continuation_env_steps'] += 1
                executed.append(action.copy())
                rewards.append(float(reward))
                if save and instruction['id'] in ['original', 'paraphrase'] and condition['id'] == 'zero':
                    frames.append(Image.fromarray(obs['agentview_image'][::-1, ::-1]))
                if done:
                    break
            result = {'id': name, 'instruction': instruction, 'noise_condition': condition,
                      'prefix_steps': len(state['prefix_actions']), 'continuation_steps': len(executed),
                      'original_goal_success': bool(env.check_success()), 'return': sum(rewards),
                      'noise_rms': float(np.sqrt(np.mean(noises ** 2))),
                      'noise_max_abs': float(np.max(np.abs(noises))),
                      'noise_within_actor_bounds': bool(np.max(np.abs(noises)) <= 1),
                      'eef_displacement': (self.to_proprio(obs, self.variant)[:3] - start_proprio[:3]).tolist()}
            arrays = {'first_actions': first, 'executed_actions': np.asarray(executed),
                      'final_physics': env.get_sim_state().copy(), 'noise_sequence': noises}
        finally:
            env.close()
        if save:
            folder = self.directory / 'cases'
            folder.mkdir(exist_ok=True)
            np.savez_compressed(folder / (name + '.npz'), **arrays)
            write_json(folder / (name + '.json'), result)
            if frames:
                frames[0].save(folder / (name + '.gif'), save_all=True, append_images=frames[1:], duration=50, loop=0)
            self.counts['cases_completed'] += 1
            self.log('probing', **{'probe/current_case': name})
        return result, arrays

    def run(self):
        states = self.make_states()
        write_json(self.directory / 'config.json', self.cfg)
        all_results, effects = [], []
        original, zero = self.cfg['instructions'][0], self.cfg['noise_conditions'][0]
        frozen_reference = None
        for state_name, state in states.items():
            first_actions = {}
            for instruction in self.cfg['instructions']:
                for condition in self.cfg['noise_conditions']:
                    case_name = '__'.join([state_name, instruction['id'], condition['id']])
                    result, arrays = self.case(state, instruction, condition, case_name)
                    all_results.append(result)
                    first_actions[(instruction['id'], condition['id'])] = arrays['first_actions']
                    if instruction == original and condition == zero:
                        repeated_result, repeated = self.case(state, instruction, condition, case_name + '__repeat', save=False)
                        np.testing.assert_allclose(arrays['first_actions'], repeated['first_actions'], rtol=0, atol=self.cfg['action_atol'])
                        np.testing.assert_allclose(arrays['executed_actions'], repeated['executed_actions'], rtol=0, atol=self.cfg['action_atol'])
                        np.testing.assert_allclose(arrays['final_physics'], repeated['final_physics'], rtol=0, atol=self.cfg['physics_atol'])
                        assert result['original_goal_success'] == repeated_result['original_goal_success']
                        if frozen_reference is None:
                            frozen_reference = (state, arrays['first_actions'].copy())
            base = first_actions[(original['id'], zero['id'])]
            for instruction in self.cfg['instructions']:
                for condition in self.cfg['noise_conditions']:
                    a = first_actions[(instruction['id'], condition['id'])]
                    interaction = a - first_actions[(instruction['id'], zero['id'])] - first_actions[(original['id'], condition['id'])] + base
                    effects.append({'state': state_name, 'instruction': instruction['id'], 'instruction_role': instruction['role'], 'noise': condition['id'],
                                    'first_chunk_rmse_vs_reference': float(np.sqrt(np.mean((a - base) ** 2))),
                                    'translation_rmse_vs_reference': float(np.sqrt(np.mean((a[:, :3] - base[:, :3]) ** 2))),
                                    'rotation_rmse_vs_reference': float(np.sqrt(np.mean((a[:, 3:6] - base[:, 3:6]) ** 2))),
                                    'gripper_rmse_vs_reference': float(np.sqrt(np.mean((a[:, 6:] - base[:, 6:]) ** 2))),
                                    'joint_nonadditivity_rmse': float(np.sqrt(np.mean(interaction ** 2)))})
        state, expected = frozen_reference
        # Recover the full visible observation through the same replay recipe.
        env, obs = self.replay(state)
        try:
            actions = self.infer(obs, original['text'], noise_sequence(zero, 1)[0])
            np.testing.assert_allclose(actions[:len(expected)], expected, rtol=0, atol=self.cfg['action_atol'])
        finally:
            env.close()
        summary = {'scope': 'exploratory normal-state short-horizon probes; not RL, OOD or recovery evidence',
                   'original_goal': self.task.language, 'gradient_updates': 0,
                   'action_metric_scope': 'first proposed action chunk at the common state; may include actions after early termination',
                   'counts': self.counts, 'states': list(states), 'cases': all_results, 'action_effects': effects,
                   'repeated_reference_continuations_match': True, 'frozen_vla_output_unchanged': True,
                   'max_replay_errors': {key: max(e[key] for e in self.restore_errors) for key in self.restore_errors[0]}}
        write_json(self.directory / 'probe_summary.json', summary)
        self.log('completed', **{'probe/replay_verified': True, 'probe/frozen_vla_unchanged': True})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['probe-smoke', 'probe'], required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    folder = Path(args.output).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(args.config).read_text())
    cfg.update(mode=args.mode, seed=args.seed)
    assert cfg['instructions'][0]['id'] == 'original' and cfg['noise_conditions'][0]['kind'] == 'zero_repeat'
    assert len({i['id'] for i in cfg['instructions']}) == len(cfg['instructions'])
    assert len({i['id'] for i in cfg['noise_conditions']}) == len(cfg['noise_conditions'])
    assert cfg['continuation_steps'] > 0 and min(cfg['prefix_steps']) == 0
    assert max(cfg['prefix_steps']) < cfg['episode_budget'] and cfg['query_freq'] > 0
    if args.mode == 'probe-smoke':
        cfg['instructions'] = cfg['instructions'][:2]
        cfg['noise_conditions'] = cfg['noise_conditions'][:2]
        cfg['continuation_steps'] = 4
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
        'learning': False, 'common_state_method': 'fresh seeded reset and recorded action-prefix replay'})
    logger = RunLogger(folder)
    try:
        Probe(cfg, folder, logger).run()
        write_json(folder / 'SUCCESS.json', {'mode': args.mode, 'passed': True})
    except BaseException as error:
        write_json(folder / 'FAILED.json', {'type': type(error).__name__, 'error': str(error),
                                          'traceback': traceback.format_exc()})
        raise
    finally:
        logger.close()


if __name__ == '__main__':
    main()
