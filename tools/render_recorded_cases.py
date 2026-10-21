"""Render recorded outcome actions without loading VLA or SAC weights.

This creates presentation artifacts, not new evaluation outcomes. Each video is
accepted only after common-state replay and final physics/success checks pass.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tools.recovery_probe import RecoveryProbe
from tools.outcome_probe import OutcomeProbe


def read(path):
    return json.loads(Path(path).read_text())


class RecordedReplay(RecoveryProbe):
    def __init__(self, cfg, source, case, destination):
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        from examples.train_utils_sim import obs_to_qpos

        self.cfg, self.source, self.case = cfg, source, case
        self.variant = SimpleNamespace(env='libero')
        self.to_proprio = obs_to_qpos
        self.env_class = OffScreenRenderEnv
        task = benchmark.get_benchmark_dict()[cfg['task_suite']]().get_task(cfg['task_id'])
        self.bddl = Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
        self.counts = {'replay_env_steps': 0, 'settling_env_steps': 0}
        self.restore_errors = []
        self.frames = 0
        self.steps = 0
        self.phase = 'reference prefix'
        self.active = 'original'
        self.font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
        self.small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12)
        self.writer = subprocess.Popen([
            'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo',
            '-pix_fmt', 'rgb24', '-s', '512x368', '-r', '10', '-i', '-',
            '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '21',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(destination)
        ], stdin=subprocess.PIPE)

    def capture(self, obs):
        canvas = Image.new('RGB', (512, 368), '#101c2c')
        draw = ImageDraw.Draw(canvas)
        row = self.case
        draw.text((10, 6), f"{self.source.name} | reset {row['reset_seed']} | {row['condition']}",
                  font=self.font, fill='white')
        draw.text((10, 27), f"{row['instruction']} + {row['noise_mode']} | recorded result: "
                  + ('SUCCESS' if row['success'] else 'FAILURE'), font=self.small, fill='#a7d7f9')
        draw.text((10, 46), f"env step {self.steps:03d} | {self.phase} | active: {self.active}",
                  font=self.small, fill='white')
        prompts = {item['id']: item['text'] for item in self.cfg['instructions']}
        prompt = prompts.get(self.active, self.active)
        for i, line in enumerate(textwrap.wrap(prompt, width=70)[:2]):
            draw.text((10, 64 + 15 * i), line, font=self.small, fill='#e4edf5')
        for i, key in enumerate(['agentview_image', 'robot0_eye_in_hand_image']):
            frame = Image.fromarray(np.ascontiguousarray(obs[key][::-1, ::-1])).resize((256, 256))
            canvas.paste(frame, (256 * i, 112))
        self.writer.stdin.write(np.asarray(canvas).tobytes())
        self.frames += 1

    def replay_prefix(self, seed, prefix):
        env, obs = self.reset_seed(seed)
        original_step = env.step

        def recorded_step(action):
            result = original_step(action)
            self.steps += 1
            if self.steps % 2 == 0:
                self.capture(result[0])
            return result

        env.step = recorded_step
        self.capture(obs)
        try:
            for action in prefix:
                obs, _, done, _ = env.step(action)
                self.counts['replay_env_steps'] += 1
                assert not done, 'Recorded reference unexpectedly terminated'
            return env, obs
        except BaseException:
            env.close()
            raise

    def apply_condition(self, env, condition, hold_action, *, ik_screen=False):
        self.phase = 'matched hold / physical release' if self.cfg.get('state_builder') == 'held_object_drop' else 'offset / matched settling'
        if self.cfg.get('state_builder') == 'held_object_drop':
            return RecoveryProbe.apply_condition(self, env, condition, hold_action, ik_screen=ik_screen)
        return OutcomeProbe.apply_condition(self, env, condition, hold_action, ik_screen=ik_screen)

    def close(self):
        self.writer.stdin.close()
        assert self.writer.wait(timeout=60) == 0, 'Video encoder failed'


def render(source, name, output):
    cfg = read(source / 'config.json')
    case = read(source / 'cases' / (name + '.json'))
    recipe = read(source / 'common_states' / (case['state_id'] + '.json'))
    with np.load(source / 'common_states' / (case['state_id'] + '.npz')) as raw:
        state = dict(recipe)
        state['prefix'] = raw['prefix_actions'].copy()
        state['hold_action'] = raw['hold_action'].copy()
        state['witness'] = {key: raw[key].copy() for key in raw.files if key != 'hold_action'}
        state['witness']['model_xml_sha256'] = recipe['model_xml_sha256']
    destination = output / (source.name + '__' + name + '.mp4')
    replay = RecordedReplay(cfg, source, case, destination)
    env = None
    try:
        env, obs = replay.replay_outcome(state)
        replay.phase = 'recorded continuation'
        with np.load(source / 'cases' / (name + '.npz')) as saved:
            actions = saved['executed_actions']
            for i, action in enumerate(actions):
                replay.active = case['instruction'] if i < cfg['intervention_steps'] else 'original'
                obs, _, done, _ = env.step(action)
                assert not done or i == len(actions) - 1, 'Unexpected early termination'
            final_error = float(np.max(np.abs(env.get_sim_state() - saved['final_physics'])))
            np.testing.assert_allclose(env.get_sim_state(), saved['final_physics'], rtol=0, atol=cfg['physics_atol'])
            assert bool(env.check_success()) == case['success'], 'Recorded task outcome changed'
            assert replay.steps == case['total_episode_steps']
            replay.capture(obs)
        return {'source': str(source), 'case_id': name, 'file': destination.name,
                'success': case['success'], 'env_steps': replay.steps,
                'video_frames': replay.frames, 'fps': 10, 'control_timestep_s': env.env.control_timestep,
                'common_state_errors': replay.restore_errors,
                'final_physics_max_error': final_error, 'vla_calls': 0, 'gradient_updates': 0,
                'scope': 'replayed saved actions with verified original outcome; not a fresh policy evaluation'}
    finally:
        if env is not None:
            env.close()
        replay.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    plan = read(args.plan)
    # XML contains absolute asset paths. Import the locked LIBERO checkout used
    # by each source experiment, and isolate different checkouts in processes.
    grouped = {}
    for item in plan:
        commit = read(Path(item['source']) / 'run.json')['commit']
        grouped.setdefault(commit, []).append(item)
    if len(grouped) > 1:
        records = []
        for commit, items in grouped.items():
            folder = output / commit[:12]
            folder.mkdir(exist_ok=True)
            worker_plan = folder / 'plan.json'
            worker_plan.write_text(json.dumps(items, indent=2) + '\n')
            subprocess.run([sys.executable, '-u', '-m', 'tools.render_recorded_cases',
                            '--plan', str(worker_plan), '--output', str(folder)], check=True)
            for record in read(folder / 'render_manifest.json'):
                record['file'] = commit[:12] + '/' + record['file']
                records.append(record)
            (output / 'render_manifest.json').write_text(json.dumps(records, indent=2) + '\n')
        (output / 'SUCCESS.json').write_text(json.dumps({'passed': True, 'videos': len(records),
            'vla_calls': 0, 'gradient_updates': 0}) + '\n')
        return
    import yaml
    source = Path(plan[0]['source'])
    candidates = [parent / 'libero_config' / 'config.yaml' for parent in [source, *source.parents]]
    original_config = next((path for path in candidates if path.is_file()), None)
    assert original_config is not None, source
    settings = yaml.safe_load(original_config.read_text())
    original_libero = Path(settings['benchmark_root']).parents[1]
    sys.path.insert(0, str(original_libero))
    os.environ['LIBERO_CONFIG_PATH'] = str(original_config.parent)
    records = []
    for item in plan:
        record = render(Path(item['source']), item['case_id'], output)
        records.append(record)
        (output / 'render_manifest.json').write_text(json.dumps(records, indent=2) + '\n')
        print(json.dumps({'completed': len(records), 'planned': len(plan), **record}), flush=True)
    (output / 'SUCCESS.json').write_text(json.dumps({'passed': True, 'videos': len(records),
        'vla_calls': 0, 'gradient_updates': 0}) + '\n')


if __name__ == '__main__':
    main()
