#!/usr/bin/env bash
set -euo pipefail
allocated_cuda=${CUDA_VISIBLE_DEVICES-}
fixed_repo=$PWD
fixed_libero_config=${LIBERO_CONFIG_PATH-}
source "$(dirname "$(dirname "$(dirname "$(dirname "$DSRL_OLD_PYTHON")")")")/etc/profile.d/conda.sh"
conda activate dsrl_pi0
source "$DSRL_BASH_SETUP"
export CUDA_VISIBLE_DEVICES="$allocated_cuda"
cd "$fixed_repo"
source tools/remote-env.sh
if [[ -n "$fixed_libero_config" ]]; then export LIBERO_CONFIG_PATH="$fixed_libero_config"; fi
exec "$DSRL_OLD_PYTHON" -u -m "$@"
