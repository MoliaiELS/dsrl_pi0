"""A short allocated-GPU render check; no VLA weights are loaded."""
import json
from pathlib import Path
import sys

import jax
import mujoco
import numpy as np
from PIL import Image
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

directory = Path(sys.argv[1])
print('JAX_GPU', jax.devices('gpu'), 'MUJOCO', mujoco.__version__, flush=True)
task = benchmark.get_benchmark_dict()['libero_90']().get_task(57)
env = OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files')) /
                                          task.problem_folder / task.bddl_file),
                         camera_heights=256, camera_widths=256)
env.seed(0)
obs = env.reset()
for _ in range(5):
    obs, reward, done, info = env.step(np.zeros(7))
Image.fromarray(obs['agentview_image'][::-1, ::-1]).save(directory / 'environment.png')
print('RENDER_OK', obs['agentview_image'].shape, list(obs), flush=True)
env.close()
(directory / 'SUCCESS.json').write_text(json.dumps({'env_probe': True, 'mujoco': mujoco.__version__}))
