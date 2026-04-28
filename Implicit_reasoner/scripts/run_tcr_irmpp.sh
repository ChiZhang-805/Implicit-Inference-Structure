cd /workspace/Implicit-VideoQA/Implicit_reasoner

python -m py_compile \
  dataset/tcr_video_sampling.py \
  models/tcr_modules.py \
  dataset/it_dataset_mistral.py \
  models/videochat_mistra/videochat2_it_mistral.py \
  tasks/train_it.py \
  inference_icr_mc_tcr.py \
  inference_icr_open_tcr.py

OMP_NUM_THREADS=2 torchrun --nnodes=1 --nproc_per_node=1 \
  tasks/train_it.py \
  scripts/videochat_mistral/config_7b_stage3_tcr.py

python inference_icr_mc_tcr.py \
  --config scripts/videochat_mistral/config_7b_stage3_tcr.py \
  --checkpoint result/tcr_irmpp/ckpt_best.pth \
  --inference_file ../dataset/ICR/mini_testing.json \
  --video_root ../dataset/video_data \
  --output_file result/tcr_mc_result.json \
  --diagnostics

python tools/score_mc.py --pred_path result/tcr_mc_result.json

python inference_icr_open_tcr.py \
  --config scripts/videochat_mistral/config_7b_stage3_tcr.py \
  --checkpoint result/tcr_irmpp/ckpt_best.pth \
  --inference_file ../dataset/ICR/test_open_ended.json \
  --video_root ../dataset/video_data \
  --output_file result/tcr_open_result.json \
  --diagnostics

python eval_GPT_score.py \
  --pred_path result/tcr_open_result.json \
  --output_json result/tcr_open_score.json
