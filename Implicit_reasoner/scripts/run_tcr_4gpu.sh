#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
export IVQA_ROOT=${IVQA_ROOT:-${REPO_ROOT}}
cd "${SCRIPT_DIR}/.."
export PYTHONPATH=$PWD:${PYTHONPATH:-}

python tools/audit_tcr_repo.py
python -m py_compile dataset/tcr_video_sampling.py dataset/it_dataset_mistral.py models/tcr_modules.py models/videochat_mistra/videochat2_it_mistral.py tasks/train_it.py inference_icr_mc_tcr.py inference_icr_open_tcr.py tools/score_mc.py tools/check_no_duration_leak.py tools/audit_tcr_repo.py

torchrun --nnodes=1 --nproc_per_node=4 --master_port=29501 \
  tasks/train_it.py scripts/videochat_mistral/config_7b_stage3_tcr.py
