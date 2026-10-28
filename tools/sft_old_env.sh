#!/usr/bin/env bash
set -euo pipefail
allocated_cuda=${CUDA_VISIBLE_DEVICES-}
fixed_repo=$PWD
source "$(dirname "$(dirname "$(dirname "$(dirname "$DSRL_OLD_PYTHON")")")")/etc/profile.d/conda.sh"
conda activate dsrl_pi0
source "$DSRL_BASH_SETUP"
export CUDA_VISIBLE_DEVICES="$allocated_cuda"
cd "$fixed_repo"
source tools/remote-env.sh
exec "$DSRL_OLD_PYTHON" -u -m "$@"
