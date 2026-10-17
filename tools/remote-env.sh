#!/usr/bin/env bash
# Source this file after activating Conda. No training or inference is started.
# Retains the original remote bash.sh's PYTHONPATH, XLA and EGL setup.
export DSRL_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export DSRL_DATA_ROOT="${DSRL_DATA_ROOT:-/mnt/slurmfs-4090node3/user_data/yguo704/dsrl_pi0}"
export PYTHONPATH="$DSRL_REPO_ROOT:$DSRL_REPO_ROOT/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export XDG_CACHE_HOME="$DSRL_DATA_ROOT/cache"
export PIP_CACHE_DIR="$DSRL_DATA_ROOT/cache/pip"
export HF_HOME="$DSRL_DATA_ROOT/cache/huggingface"
export OPENPI_DATA_HOME="$DSRL_DATA_ROOT/cache/openpi"
export NUMBA_CACHE_DIR="$DSRL_DATA_ROOT/cache/numba"
export JAX_COMPILATION_CACHE_DIR="$DSRL_DATA_ROOT/cache/jax"
export LIBERO_CONFIG_PATH="$DSRL_DATA_ROOT/config/libero"
export EXP="$DSRL_DATA_ROOT/outputs"
export WANDB_DIR="$DSRL_DATA_ROOT/logs/wandb"
export TMPDIR="$DSRL_DATA_ROOT/tmp"
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME" "$OPENPI_DATA_HOME" "$NUMBA_CACHE_DIR" \
    "$JAX_COMPILATION_CACHE_DIR" "$LIBERO_CONFIG_PATH" "$EXP" "$WANDB_DIR" "$TMPDIR"
