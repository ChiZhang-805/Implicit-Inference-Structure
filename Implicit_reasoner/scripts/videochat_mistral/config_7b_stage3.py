import os
from pathlib import Path
from configs.instruction_data import *

ROOT = os.environ.get("IVQA_ROOT", str(Path(__file__).resolve().parents[3]))
WEIGHTS = os.path.join(ROOT, "weights")

train_corpus = "videochat2_instruction_new"
train_file = os.path.join(ROOT, "dataset/ICR/mini_training.json")
test_file = dict()
test_types = []
num_workers = 4
stop_key = None
w_ce = 3.0
num_frames = 8
num_frames_test = 8
batch_size = 4
max_txt_l = 512
pre_text = False

inputs = dict(image_res=224, video_input=dict(num_frames="${num_frames}", sample_type="rand", num_frames_test="${num_frames_test}", sample_type_test="middle", random_aug=False), max_txt_l=dict(image="${max_txt_l}", video="${max_txt_l}"), batch_size=dict(image="${batch_size}", video="${batch_size}"), batch_size_test=dict(image="${batch_size}", video="${batch_size}"))

model = dict(model_cls="VideoChat2_it_mistral", vit_blip_model_path=os.path.join(WEIGHTS, "umt_l16_qformer.pth"), mistral_model_path=os.path.join(WEIGHTS, "Mistral-7B-Instruct-v0.2"), videochat2_model_path=os.path.join(WEIGHTS, "videochat2_mistral_7b_stage3.pth"), freeze_vit=True, freeze_qformer=False, max_txt_len="${max_txt_l}", low_resource=False, vision_encoder=dict(name="vit_l14", img_size=224, patch_size=16, d_model=1024, encoder_embed_dim=1024, encoder_depth=24, encoder_num_heads=16, drop_path_rate=0.0, num_frames="${num_frames}", tubelet_size=1, use_checkpoint=True, checkpoint_num=18, pretrained="", return_index=-2, vit_add_ln=True, ckpt_num_frame=4), num_query_token=32, qformer_hidden_dropout_prob=0.1, qformer_attention_probs_dropout_prob=0.1, qformer_drop_path_rate=0.2, extra_num_query_token=64, qformer_text_input=True, system="The question involves implicit visual information and hidden key evidence. ", start_token="<Video>", end_token="</Video>", add_second_msg=True, img_start_token="<Image>", img_end_token="</Image>", random_shuffle=True, use_flash_attention=True, use_lora=True, lora_r=16, lora_alpha=32, lora_dropout=0.1)

optimizer = dict(opt="adamW", lr=2.5e-5, opt_betas=[0.9, 0.999], weight_decay=0.02, max_grad_norm=-1, different_lr=dict(enable=False, module_names=[], lr=1e-3))
scheduler = dict(sched="cosine", epochs=25, min_lr_multi=0.25, warmup_epochs=0.6)
evaluate = False
deep_fusion = False
evaluation = dict(eval_frame_ensemble="concat", eval_x_only=False, k_test=128, eval_offload=True)
fp16 = True
gradient_checkpointing = True
wandb = dict(enable=False, entity="", project="videochat2")
dist_url = "env://"
device = "cuda"
mode = "it_mistral"
output_dir = os.path.join(ROOT, "Implicit_reasoner/result/stage3")
resume = False
debug = False
log_freq = 10
seed = 42
save_latest = True
auto_resume = False
pretrained_path = ""
