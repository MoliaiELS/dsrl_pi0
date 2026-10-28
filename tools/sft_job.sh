#!/usr/bin/env bash
set -euo pipefail
repo=$1; conda_root=$2; conda_env=$3; data_root=$4; stage=$5; config=$6
sft_prefix=$7; rlinf_root=$8; gate_job=${9:-}; restore=${10:-}; setup=${11}
allocated_cuda=${CUDA_VISIBLE_DEVICES-}
source "$conda_root/etc/profile.d/conda.sh"
conda activate "$conda_env"
source "$setup"
export CUDA_VISIBLE_DEVICES="$allocated_cuda"
cd "$repo"
export DSRL_DATA_ROOT="$data_root"
source tools/remote-env.sh
export DSRL_BASH_SETUP="$setup"
export DSRL_SFT_PREFIX="$sft_prefix"
export DSRL_RLINF_ROOT="$rlinf_root"
export DSRL_OLD_PYTHON="$conda_root/envs/$conda_env/bin/python"
export DSRL_MODEL_PATH="$data_root/checkpoints/pi0_libero"
run_dir="$data_root/runs/job-$SLURM_JOB_ID"
mkdir -p "$run_dir/libero_config"
export LIBERO_CONFIG_PATH="$run_dir/libero_config"
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONWARNINGS=ignore
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export USE_TF=0
python - <<'PY'
from pathlib import Path
import os,yaml
p=Path(os.environ['DSRL_REPO_ROOT'])/'LIBERO/libero/libero'
(Path(os.environ['LIBERO_CONFIG_PATH'])/'config.yaml').write_text(yaml.safe_dump(dict(benchmark_root=str(p),bddl_files=str(p/'bddl_files'),init_states=str(p/'init_files'),assets=str(p/'assets'),datasets=os.environ['DSRL_DATA_ROOT']+'/datasets/libero')))
PY
if [[ "$stage" == data ]]; then
  python -u -m tools.sft_data --config "$config" --output "$run_dir" --render
else
  export DSRL_SFT_GATE_DIR="$data_root/runs/job-$gate_job"
  python - "$config" "$stage" <<'PY'
import os,json,hashlib,sys
from pathlib import Path
config,stage=sys.argv[1:];p=Path(os.environ['DSRL_SFT_GATE_DIR'])
flag={'align':'SFT_DATA_PASS.json','profile':'SFT_ALIGNMENT_PASS.json','train':'SFT_PROFILE_PASS.json','evaluate':'SFT_TRAINING_COMPLETE.json'}[stage]
assert json.loads((p/flag).read_text())['passed'],flag
assert json.loads((p/'run.json').read_text())['config_sha256']==hashlib.sha256(Path(config).read_bytes()).hexdigest(),'Prerequisite config differs'
PY
  test -f "$sft_prefix/requirements-resolved.txt"
  source "$sft_prefix/bin/activate"
  export PYTHONPATH="$repo:$rlinf_root:$PYTHONPATH"
  python -u -m tools.sft_native --mode "$stage" --config "$config" --output "$run_dir" --gate-dir "$DSRL_SFT_GATE_DIR" --restore "$restore"
fi
