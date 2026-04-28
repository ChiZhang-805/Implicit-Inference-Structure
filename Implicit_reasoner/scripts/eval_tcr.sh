#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
export IVQA_ROOT=${IVQA_ROOT:-${REPO_ROOT}}
cd "${SCRIPT_DIR}/.."
export PYTHONPATH=$PWD:${PYTHONPATH:-}

python inference_icr_mc_tcr.py \
  --config scripts/videochat_mistral/config_7b_stage3_tcr.py \
  --checkpoint result/tcr_irmpp/ckpt_latest.pth \
  --inference_file ../dataset/ICR/mini_testing.json \
  --video_root ../dataset/video_data \
  --output_file result/tcr_mc_eval.json \
  --diagnostics

python tools/score_mc.py --pred_path result/tcr_mc_eval.json
python tools/check_no_duration_leak.py --result_json result/tcr_mc_eval.json --annotation_json ../dataset/ICR/mini_testing.json

python inference_icr_open_tcr.py \
  --config scripts/videochat_mistral/config_7b_stage3_tcr.py \
  --checkpoint result/tcr_irmpp/ckpt_latest.pth \
  --inference_file ../dataset/ICR/test_open_ended.json \
  --video_root ../dataset/video_data \
  --output_file result/tcr_open_eval.json \
  --diagnostics

python tools/gpt_score.py --pred_file result/tcr_open_eval.json --gt_file ../dataset/ICR/test_open_ended.json
