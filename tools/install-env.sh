#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source tools/remote-env.sh
log_dir="$DSRL_DATA_ROOT/logs/environment/install-$(date +%Y%m%dT%H%M%S)"
mkdir -p "$log_dir"
exec > >(tee "$log_dir/install.log") 2>&1
echo "Environment record: $log_dir"
python -m pip freeze --all > "$log_dir/before.txt"
git rev-parse HEAD > "$log_dir/commit.txt"
git submodule status > "$log_dir/submodules.txt"

# Install CPU torch first so OpenPI cannot pull the latest CUDA torch build.
torch_wheel="$DSRL_DATA_ROOT/wheels/torch-2.6.0+cpu-cp311-cp311-linux_x86_64.whl"
if [[ -f "$torch_wheel" ]]; then
    echo "5b6ae523bfb67088a17ca7734d131548a2e60346c622621e4248ed09dd0790cc  $torch_wheel" | sha256sum -c -
    python -m pip install "$torch_wheel" --index-url https://pypi.tuna.tsinghua.edu.cn/simple
else
    python -m pip install 'torch==2.6.0+cpu' --timeout 20 --retries 1 \
        --index-url https://download.pytorch.org/whl/cpu
fi
# LIBERO lacks an outer __init__.py: setuptools' default editable finder is empty.
# compat mode exposes the source directory without modifying the locked submodule.
python -m pip install -c constraints/remote.txt -r requirements.txt \
    -e . -e ./LIBERO -e ./openpi -e ./openpi/packages/openpi-client \
    --config-settings editable_mode=compat \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple

python - <<'PY'
import os
from pathlib import Path
import yaml

root = Path(os.environ['DSRL_REPO_ROOT']) / 'LIBERO/libero/libero'
datasets = Path(os.environ['DSRL_DATA_ROOT']) / 'datasets/libero'
datasets.mkdir(parents=True, exist_ok=True)
config = Path(os.environ['LIBERO_CONFIG_PATH']) / 'config.yaml'
if not config.exists():
    config.write_text(yaml.safe_dump({
        'benchmark_root': str(root),
        'bddl_files': str(root / 'bddl_files'),
        'init_states': str(root / 'init_files'),
        'assets': str(root / 'assets'),
        'datasets': str(datasets),
    }), encoding='utf-8')
print('LIBERO config:', config)
PY
python -m pip check
python tools/check_env.py
python -m pip freeze --all > "$log_dir/after.txt"
conda list --explicit > "$log_dir/conda-explicit.txt"
echo "Environment installation and import checks passed: $log_dir"
