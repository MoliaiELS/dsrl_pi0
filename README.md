<div align="center">

# DSRL for π₀: Diffusion Steering via Reinforcement Learning

## [[website](https://diffusion-steering.github.io)]      [[paper](https://arxiv.org/abs/2506.15799)]

</div>


## Overview
This repository provides the official implementation for our paper: [Steering Your Diffusion Policy with Latent Space Reinforcement Learning](https://arxiv.org/abs/2506.15799) (CoRL 2025).

Specifically, it contains a JAX-based implementation of DSRL (Diffusion Steering via Reinforcement Learning) for steering a pre-trained generalist policy, [π₀](https://github.com/Physical-Intelligence/openpi), across various environments, including:

- **Simulation:** Libero, Aloha  
- **Real Robot:** Franka

If you find this repository useful for your research, please cite:

```
@article{wagenmaker2025steering,
  author    = {Andrew Wagenmaker and Mitsuhiko Nakamoto and Yunchu Zhang and Seohong Park and Waleed Yagoub and Anusha Nagabandi and Abhishek Gupta and Sergey Levine},
  title     = {Steering Your Diffusion Policy with Latent Space Reinforcement Learning},
  journal   = {Conference on Robot Learning (CoRL)},
  year      = {2025},
}
```

## Installation
1. Create a conda environment:
```
conda create -n dsrl_pi0 python=3.11.11
conda activate dsrl_pi0
```

2. Clone this repo with all submodules
```
git clone git@github.com:nakamotoo/dsrl_pi0.git --recurse-submodules
cd dsrl_pi0
```

3. Install all packages and dependencies
```
pip install -e .
pip install -r requirements.txt
pip install "jax[cuda12]==0.5.0"

# install openpi
pip install -e openpi
pip install -e openpi/packages/openpi-client

# install Libero
pip install -e LIBERO
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu # needed for libero
```

### IADC remote environment repair (installation checks only)

The remote environment tools do not start training or inference:

```powershell
.\tools\remote.ps1 install-env
.\tools\remote.ps1 doctor
```

Connection, Conda, code and storage paths are in `tools/remote.json`. Every call
initializes Conda and sources `tools/remote-env.sh`. Use that script for manual
sessions as well; the repository's original `bash.sh` launches experiments.
The installation is validated in the separate `dsrl_pi0-envfix` worktree;
editable installs must point to that directory. Original remote files are retained.

`constraints/remote.txt` keeps TensorFlow 2.19 / protobuf 5.29.6 and compatible
Google/gRPC packages together. Install all four editable projects in one pip
transaction with those constraints. LIBERO requires setuptools
`--config-settings editable_mode=compat` because its namespace package layout
produces an empty finder with the default editable mode. Pillow is pinned to
11.1.0 to satisfy the locked OpenPI dependency (`>=11.0.0`).

Torch follows the original README's CPU 2.6.0 build. If the server cannot reach
the official PyTorch index, transfer its CPython 3.11 Linux wheel to
`$DSRL_DATA_ROOT/wheels/torch-2.6.0+cpu-cp311-cp311-linux_x86_64.whl` and rerun
`install-env`. Verify its SHA256 against the official wheel index before use.

Import checks force JAX to CPU, disable W&B, and set offline model access. They
import the simulation training module without calling `main`, and do not load
weights, construct a simulator, generate actions, or update parameters. These
checks establish installation health only. Environment snapshots, commit IDs,
submodule versions and logs are saved under
`$DSRL_DATA_ROOT/logs/environment/`; data and caches stay outside Git.

## Training (Simulation)
Libero
```
bash examples/scripts/run_libero.sh
```
Aloha
```
bash examples/scripts/run_aloha.sh
```
### Training Logs
We provide sample W&B runs and logs: https://wandb.ai/mitsuhiko/DSRL_pi0_public

## Training (Real)
For real-world experiments, we use the remote hosting feature from pi0 (see [here](https://github.com/Physical-Intelligence/openpi/blob/main/docs/remote_inference.md)) which enables us to host the pi0 model on a higher-spec remote server, in case the robot's client machine is not powerful enough. 

0. Setup Franka robot and install DROID package [[link](https://github.com/droid-dataset/droid.git)]

1. [On the remote server] Host pi0 droid model on your remote server
```
cd openpi && python scripts/serve_policy.py --env=DROID
```
2. [On your robot client machine] Run DSRL
```
bash examples/scripts/run_real.sh
```


## Credits
This repository is built upon [jaxrl2](https://github.com/ikostrikov/jaxrl2) and [PTR](https://github.com/Asap7772/PTR) repositories. 
In case of any questions, bugs, suggestions or improvements, please feel free to contact me at nakamoto\[at\]berkeley\[dot\]edu 
