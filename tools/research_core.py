"""Dependency-light scientific design, event state machines and durable budgets."""
import hashlib
import itertools
import json
from pathlib import Path
import random
import time


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()


def sha(value):
    return hashlib.sha256(value if isinstance(value, bytes) else canonical(value)).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_bytes(canonical(value))
    tmp.replace(path)


def lock(path, value):
    path = Path(path)
    if path.exists():
        assert read(path) == value, f'Immutable manifest mismatch: {path}'
    else:
        atomic(path, value)


def case_plan(p):
    cases = []
    def add(phase, task, reset, sampling, policy, prompt, goal, horizon, condition='constant'):
        c = dict(phase=phase, task=task, reset_seed=reset, sampling_seed=sampling,
                 policy=policy, prompt_id=prompt, requested_goal_id=goal, horizon=horizon, condition=condition)
        c['id'] = phase + '__' + sha([p['protocol_id'], c])[:20]
        cases.append(c)
    splits = p['seed_splits']
    for phase, split, samplings in [('e1_dev','e1_development',[0]),('e1_confirm','e1_confirmation',[0,1])]:
        for seed, sample, policy, goal in itertools.product(splits[split], samplings, p['e1']['policies'], p['e1']['goals']):
            add(phase, 'E1', seed, sample, policy['id'], goal['id'], goal['id'], 400)
    for seed, policy, prompt in itertools.product(splits['e1_auxiliary'], p['e1']['auxiliary_policies'], p['e1']['auxiliary_prompts']):
        add('e1_aux', 'E1', seed, 0, policy, prompt['id'], prompt['requested_goal_id'], 400)
    for task, seed, cond in itertools.product(p['e2']['tasks'], splits['e2_development'], p['e2']['conditions']):
        add('e2_dev', task['id'], seed, 0, 'base', cond, None, 600, cond)
    # E0 is fixed, independent of development observations; 10 branches, 400 steps.
    for seed in splits['e0_technical'][:2]:
        for policy in p['e1']['policies']:
            add('e0', 'E1', seed, 0, policy['id'], 'A' if seed == splits['e0_technical'][0] else 'B',
                'A' if seed == splits['e0_technical'][0] else 'B', 40)
    random.Random(splits['case_order_seed']).shuffle(cases)
    return cases


def validate(p):
    assert p['protocol_id'] == 'dsrl-vla-language-stage-v1-20261027'
    assert p['shared_eval']['execute_action_chunk'] == 20
    assert p['shared_eval']['denoise_steps'] == 10
    assert p['shared_eval']['warmup_steps'] == 0
    assert len(p['e1']['goals']) == 2 and len(p['e1']['policies']) == 5
    cases = case_plan(p)
    counts = {s: sum(c['phase'] == s for c in cases) for s in ['e0','e1_dev','e1_aux','e1_confirm','e2_dev']}
    assert counts == dict(e0=10, e1_dev=100, e1_aux=30, e1_confirm=800, e2_dev=60), counts
    assert sum(c['horizon'] for c in cases) <= 408800
    assert len({c['id'] for c in cases}) == len(cases)
    reserved = set(s for k,v in p['seed_splits'].items() if 'reserved' in k for s in v)
    assert not reserved.intersection(c['reset_seed'] for c in cases)
    return dict(counts=counts, max_policy_steps=sum(c['horizon'] for c in cases), case_manifest_sha256=sha(cases))


class GraspTracker:
    def __init__(self, objects):
        self.count = dict.fromkeys(objects, 0)
        self.first_class, self.first_step, self.first_objects = 'NONE', None, []

    def update(self, grasp, step):
        for obj in self.count:
            self.count[obj] = self.count[obj] + 1 if grasp[obj] else 0
        reached = sorted(k for k,v in self.count.items() if v == 3)
        if self.first_step is None and reached:
            self.first_step, self.first_objects = step, reached
            self.first_class = ('MULTIPLE' if len(reached) > 1 else
                                {'cream_cheese_1':'A','butter_1':'B'}.get(reached[0], 'OTHER'))


class StageTracker:
    def __init__(self):
        self.true = self.false = self.phase = 0
        self.first_complete = None
        self.transitions = []

    def observe(self, predicate, step):
        self.true = self.true + 1 if predicate else 0
        self.false = self.false + 1 if not predicate else 0
        if self.true >= 3 and self.first_complete is None:
            self.first_complete = step

    def query(self, step):
        target = 1 if self.true >= 3 else 0 if self.false >= 3 else self.phase
        if target != self.phase:
            self.transitions.append(dict(step=step, previous=self.phase, phase=target))
        self.phase = target
        return self.phase


def fixed_schedule(times, fallback):
    import numpy as np
    usable = [x for x in times if x is not None]
    return int(min(300, max(100, 20 * __import__('math').ceil(float(np.median(usable)) / 20)))) if usable else fallback


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    """Reserve before stepping; interrupted reservations count as uncertain upper bounds."""
    def __init__(self, path, cap=450000, hours=24):
        self.path, self.cap, self.hours = Path(path), cap, hours
        self.state = read(path) if self.path.exists() else dict(recorded_physical_steps=0, uncertain_steps=0,
            policy_steps=0, replay_steps=0, construction_steps=0, reservations={}, sessions=[])
        for value in self.state['reservations'].values():
            self.state['uncertain_steps'] += value['steps']
        self.state['reservations'] = {}
        self.started = time.time()
        self.session = dict(job_id=__import__('os').environ.get('SLURM_JOB_ID'), started=self.started, seconds=0)
        self.state['sessions'].append(self.session)
        self.flush()

    def flush(self):
        self.session['seconds'] = time.time() - self.started
        self.state['physical_steps_upper_bound'] = self.state['recorded_physical_steps'] + self.state['uncertain_steps']
        self.state['allocated_gpu_hours_lower_bound'] = sum(s['seconds'] for s in self.state['sessions']) / 3600
        atomic(self.path, self.state)

    def reserve(self, key, steps, kind):
        self.flush()
        upper = self.state['physical_steps_upper_bound'] + sum(v['steps'] for v in self.state['reservations'].values())
        if upper + steps > self.cap or sum(s['seconds'] for s in self.state['sessions']) >= self.hours * 3600:
            raise BudgetExceeded('First-batch physical/GPU budget exhausted')
        assert key not in self.state['reservations']
        self.state['reservations'][key] = dict(steps=steps, kind=kind)
        self.flush()

    def settle(self, key, actual):
        value = self.state['reservations'].pop(key)
        assert 0 <= actual <= value['steps']
        self.state['recorded_physical_steps'] += actual
        self.state[value['kind'] + '_steps'] += actual
        self.flush()


def completed(directory, case):
    p = Path(directory) / 'cases' / (case['id'] + '.json')
    if not p.exists():
        return None
    value = read(p)
    assert value['case'] == case, f'Case manifest changed: {p}'
    for relative, expected in value['artifact_hashes'].items():
        assert file_sha(Path(directory) / relative) == expected, f'Corrupt result: {relative}'
    return value
