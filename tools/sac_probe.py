"""Exercise the actual SAC update and checkpoint before downloading a VLA."""
import json
from pathlib import Path
import sys
import gym
import jax
import numpy as np
from flax.core import freeze

from jaxrl2.agents.pixel_sac.pixel_sac_learner import PixelSACLearner

directory = Path(sys.argv[1])
config = json.loads(Path('configs/libero_baseline.json').read_text())
kwargs = config['train_kwargs']
np.random.seed(0)
obs = {'pixels': np.zeros((1, 64, 64, 3, 1), np.uint8),
       'state': np.zeros((1, 8, 1), np.float32)}
actions = np.zeros((1, 1, 32), np.float32)
agent = PixelSACLearner(0, obs, actions, **kwargs)
before = [np.array(x) for x in jax.tree_util.tree_leaves(agent._actor.params)]
batch = freeze({
    'observations': {key: np.repeat(value, 16, axis=0) for key, value in obs.items()},
    'next_observations': {key: np.repeat(value, 16, axis=0) for key, value in obs.items()},
    'actions': np.zeros((16, 1, 32), np.float32), 'rewards': -np.ones(16, np.float32),
    'masks': np.ones(16, np.float32), 'discount': np.full(16, 0.999**20, np.float32),
})
for step in range(3):
    metrics = jax.device_get(agent.update(batch))
    assert all(np.isfinite(v).all() for v in metrics.values()), metrics
    print('SAC_UPDATE', step + 1, {k: float(v) for k, v in metrics.items() if np.ndim(v) == 0}, flush=True)
delta = sum(float(np.sum((np.array(x)-y)**2)) for x,y in zip(
    jax.tree_util.tree_leaves(agent._actor.params), before)) ** 0.5
assert delta > 0
expected = agent.eval_actions(obs)
agent.save_checkpoint(str(directory / 'sac-checkpoint'), 3, 3)
restored = PixelSACLearner(1, obs, actions, **kwargs)
restored.restore_checkpoint(str(directory / 'sac-checkpoint'))
np.testing.assert_allclose(expected, restored.eval_actions(obs), rtol=1e-5, atol=1e-5)
(directory / 'SUCCESS.json').write_text(json.dumps({'render': True, 'sac_updates': 3,
                                                 'actor_delta_l2': delta, 'checkpoint_restore': True}))
print('SAC_PROBE_PASSED', flush=True)
