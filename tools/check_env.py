"""Installation checks only: never instantiate a model, policy, or simulator."""

import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    print('Python:', sys.version, flush=True)
    print('Executable:', sys.executable, flush=True)
    subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, check=True)
    subprocess.run(['git', 'submodule', 'status'], cwd=root, check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)
    from packaging.requirements import Requirement

    for filename in ['requirements.txt', 'constraints/remote.txt']:
        for line in (root / filename).read_text().splitlines():
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            requirement = Requirement(line)
            version = metadata.version(requirement.name)
            if version not in requirement.specifier:
                raise RuntimeError(f'{requirement}: installed {version}')
    print('Project requirements and compatibility pins: PASS', flush=True)
    for name in ['jax', 'jaxlib', 'jax-cuda12-plugin', 'tensorflow', 'protobuf',
                 'torch', 'pillow', 'wandb', 'mujoco', 'libero', 'openpi']:
        print(f'{name}=={metadata.version(name)}', flush=True)

    # Force CPU only in the child processes. Preserve Slurm GPU visibility.
    child_env = dict(os.environ, JAX_PLATFORMS='cpu', TF_CPP_MIN_LOG_LEVEL='3',
                     WANDB_MODE='disabled', HF_HUB_OFFLINE='1',
                     TRANSFORMERS_OFFLINE='1')
    modules = [
        'tensorflow', 'tf_keras', 'jax', 'flax', 'distrax', 'torch',
        'dm_control.mujoco', 'robosuite', 'libero.libero.benchmark',
        'libero.libero.envs', 'openpi.models.pi0',
        'openpi.policies.policy_config', 'openpi.training.config',
        'openpi_client.image_tools', 'gcsfs', 's3fs',
        'google.cloud.storage', 'google.cloud.storage_control_v2',
        'jaxrl2.agents.pixel_sac.pixel_sac_learner',
        'examples.train_sim',
    ]
    failures = []
    for module in modules:
        code = (
            'import importlib, json; '
            f'm = importlib.import_module({module!r}); '
            'print(json.dumps({"module": m.__name__, '
            '"file": getattr(m, "__file__", None)}))'
        )
        try:
            result = subprocess.run(
                [sys.executable, '-c', code], cwd=root, env=child_env,
                stdin=subprocess.DEVNULL, capture_output=True, text=True,
                timeout=90,
            )
            if result.returncode:
                failures.append(module)
                print(f'FAIL {module}\n{result.stdout}\n{result.stderr}', flush=True)
            else:
                print(f'PASS {module}: {result.stdout.strip()}', flush=True)
        except subprocess.TimeoutExpired:
            failures.append(module)
            print(f'FAIL {module}: import timed out', flush=True)

    # Check each editable package's actual source, even when invoked elsewhere.
    expected = {'openpi': str(root / 'openpi/src/openpi'),
                'openpi_client': str(root / 'openpi/packages/openpi-client/src/openpi_client'),
                'jaxrl2': str(root / 'jaxrl2'),
                'libero.libero': str(root / 'LIBERO/libero/libero')}
    source_env = dict(child_env)
    source_env.pop('PYTHONPATH', None)
    code = (
        'import importlib, json; from pathlib import Path; '
        f'expected = {expected!r}; '
        'actual = {name: str(Path(importlib.import_module(name).__file__).resolve().parent) '
        'for name in expected}; '
        'print(json.dumps(actual)); assert actual == expected, (actual, expected)'
    )
    result = subprocess.run(
        [sys.executable, '-c', code], env=source_env,
        cwd=os.environ['TMPDIR'], stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=90,
    )
    print('Installed sources without PYTHONPATH:', result.stdout, flush=True)
    if result.returncode:
        failures.append('editable:sources')
        print(result.stderr, flush=True)
    print(json.dumps({'failed': failures, 'training_started': False,
                      'inference_started': False}), flush=True)
    return bool(failures)


if __name__ == '__main__':
    sys.exit(main())
