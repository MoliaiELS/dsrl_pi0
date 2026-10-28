#!/usr/bin/env bash
set -euo pipefail
prefix=$1
rlinf_root=$2
project_root=$3
commit=fab7f407fbc611085eb163962e3c89c0ab424c1a
export CUDA_VISIBLE_DEVICES=
export JAX_PLATFORMS=cpu
export USE_TF=0
export HF_HUB_DISABLE_XET=1
if [[ ! -d "$rlinf_root/.git" ]]; then
  mkdir -p "$(dirname "$rlinf_root")"
  git -c http.lowSpeedLimit=1024 -c http.lowSpeedTime=30 clone --filter=blob:none --no-checkout https://github.com/RLinf/RLinf.git "$rlinf_root"
fi
test -z "$(git -C "$rlinf_root" status --porcelain)"
git -C "$rlinf_root" fetch origin "$commit"
git -C "$rlinf_root" checkout --detach "$commit"
if [[ ! -x "$prefix/bin/python" ]]; then
  python -m venv "$prefix"
fi
"$prefix/bin/python" -m pip install 'uv==0.9.5' -i https://pypi.tuna.tsinghua.edu.cn/simple
"$prefix/bin/python" -m uv pip install --python "$prefix/bin/python" -r "$project_root/configs/sft_a_requirements.txt" -e "$rlinf_root" -i https://pypi.tuna.tsinghua.edu.cn/simple
"$prefix/bin/python" -m pip check
"$prefix/bin/python" -m pip freeze > "$prefix/requirements-resolved.txt"
git -C "$rlinf_root" rev-parse HEAD > "$prefix/rlinf-commit.txt"
