#!/usr/bin/env bash
set -euo pipefail
repo="$1"; conda_root="$2"; conda_env="$3"; data_root="$4"
mode="$5"; config="$6"; seed="$7"; restore_path="${8:-}"
gate_job="${9:-}"
source "$conda_root/etc/profile.d/conda.sh"
conda activate "$conda_env"
if [[ "$mode" == 'research' || "$mode" == 'research-smoke' ]]; then
    export DSRL_BASH_SETUP="${DSRL_BASH_SETUP:-$(python -c 'import json; print(json.load(open("'"$repo"'/tools/remote.json"))["BashSetupPath"])')}"
    source "$DSRL_BASH_SETUP"
fi
cd "$repo"
export DSRL_DATA_ROOT="$data_root"
source tools/remote-env.sh
run_dir="$DSRL_DATA_ROOT/runs/job-$SLURM_JOB_ID"
mkdir -p "$run_dir/libero_config"
if [[ -n "$gate_job" ]]; then
    if [[ "$mode" == 'outcome' ]]; then
    python - "$gate_job" "$config" "$run_dir" <<'PY'
import hashlib, json, os, subprocess, sys
from pathlib import Path
job, protocol, output = sys.argv[1:]
assert job.isdigit()
source = Path(os.environ['DSRL_DATA_ROOT']) / 'runs' / ('job-' + job)
run = json.loads((source / 'run.json').read_text())
success = json.loads((source / 'SUCCESS.json').read_text())
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
digest = hashlib.sha256(Path(protocol).read_bytes()).hexdigest()
assert run['commit'] == commit and run['mode'] == 'outcome-smoke' and success['passed'], (run, success)
assert run['protocol_file_sha256'] == digest, 'Smoke protocol differs from scheduled outcome protocol'
check = {'smoke_job_id': job, 'passed': True, 'commit': commit, 'protocol_file_sha256': digest}
(Path(output) / 'gate_check.json').write_text(json.dumps(check, indent=2))
print('DEFERRED_SMOKE_GATE', json.dumps(check), flush=True)
PY
    fi
fi
export LIBERO_CONFIG_PATH="$run_dir/libero_config"
export DSRL_MODEL_PATH="$DSRL_DATA_ROOT/checkpoints/pi0_libero"
export WANDB_MODE=disabled
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TF_CPP_MIN_LOG_LEVEL=3
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1
python - <<'PY'
import os
from pathlib import Path
import yaml
root = Path(os.environ['DSRL_REPO_ROOT']) / 'LIBERO/libero/libero'
config = Path(os.environ['LIBERO_CONFIG_PATH']) / 'config.yaml'
config.write_text(yaml.safe_dump({
    'benchmark_root': str(root), 'bddl_files': str(root / 'bddl_files'),
    'init_states': str(root / 'init_files'), 'assets': str(root / 'assets'),
    'datasets': str(Path(os.environ['DSRL_DATA_ROOT']) / 'datasets/libero'),
}))
PY
echo "JOB=$SLURM_JOB_ID MODE=$mode OUTPUT=$run_dir NODE=$(hostname)"
git rev-parse HEAD
git submodule status
python -m pip freeze --all > "$run_dir/pip-freeze.txt"
python - <<'PY'
import importlib, os
from pathlib import Path
root = Path(os.environ['DSRL_REPO_ROOT'])
for name in ['jaxrl2', 'libero.libero', 'openpi', 'openpi_client']:
    path = Path(importlib.import_module(name).__file__).resolve()
    assert path.is_relative_to(root), (name, path, root)
    print('SOURCE', name, path)
PY
if [[ "$mode" == 'env' ]]; then
    python -u tools/env_probe.py "$run_dir"
    python -u tools/sac_probe.py "$run_dir"
else
    test -f "$DSRL_MODEL_PATH/download_manifest.json"
    if [[ "$mode" == 'research' || "$mode" == 'research-smoke' ]]; then
        research_output="${restore_path:-$run_dir}"
        research_args=(--protocol "$config" --output "$research_output" --resume)
        if [[ "$mode" == 'research-smoke' ]]; then research_args+=(--phase e0); fi
        if [[ -n "$gate_job" ]]; then research_args+=(--gate-job "$gate_job"); fi
        python -u -m tools.research_runner "${research_args[@]}"
    elif [[ "$mode" == 'outcome' || "$mode" == 'outcome-smoke' ]]; then
        python -u -m tools.outcome_probe --mode "$mode" --config "$config" --seed "$seed" --output "$run_dir"
    elif [[ "$mode" == 'probe' || "$mode" == 'probe-smoke' ]]; then
        python -u -m tools.control_probe --mode "$mode" --config "$config" --seed "$seed" --output "$run_dir"
    else
        python -u -m tools.libero_run --mode "$mode" --config "$config" --seed "$seed" \
            --output "$run_dir" --restore-path "$restore_path"
    fi
fi
