import os
from pathlib import Path
from configs.instruction_data import *

ROOT = os.environ.get("IVQA_ROOT", str(Path(__file__).resolve().parents[3]))
DATASET_ROOT = os.path.join(ROOT, "dataset")
WEIGHTS = os.path.join(ROOT, "weights")
VIDEO_ROOT = os.path.join(DATASET_ROOT, "video_data")

train_file = os.path.join(DATASET_ROOT, "NExT-GQA/ICR/training_4k.json")
if not os.path.exists(train_file):
    if os.environ.get("IVQA_ALLOW_MINI_FALLBACK", "0") == "1":
        train_file = os.path.join(DATASET_ROOT, "ICR/mini_training.json")
    else:
        raise FileNotFoundError(f"Missing training file: {train_file}. Set IVQA_ALLOW_MINI_FALLBACK=1 only for debugging.")

available_corpus["icr_tcr"] = [train_file, VIDEO_ROOT]
train_corpus = "icr_tcr"
train_file = [available_corpus[train_corpus]]

test_file = dict()
test_types = []
num_workers = 4
num_frames = 8
num_frames_test = 8
batch_size = 2
max_txt_l = 768

inputs = dict(
    image_res=224,
    video_input=dict(
        num_frames="${num_frames}",
        sample_type="rand",
        num_frames_test="${num_frames_test}",
        sample_type_test="middle",
        random_aug=False,
    ),
    max_txt_l=dict(image="${max_txt_l}", video="${max_txt_l}"),
    batch_size=dict(image="${batch_size}", video="${batch_size}"),
    batch_size_test=dict(image="${batch_size}", video="${batch_size}"),
)

model = dict(
    model_cls="VideoChat2_it_mistral",
    vit_blip_model_path=os.path.join(WEIGHTS, "umt_l16_qformer.pth"),
    mistral_model_path=os.path.join(WEIGHTS, "Mistral-7B-Instruct-v0.2"),
    videochat2_model_path=os.path.join(WEIGHTS, "videochat2_mistral_7b_stage3.pth"),
    freeze_vit=True,
    freeze_qformer=False,
    max_txt_len="${max_txt_l}",
    low_resource=False,
    vision_encoder=dict(
        name="vit_l14", img_size=224, patch_size=16, d_model=1024,
        encoder_embed_dim=1024, encoder_depth=24, encoder_num_heads=16,
        drop_path_rate=0.0, num_frames="${num_frames}", tubelet_size=1,
        use_checkpoint=True, checkpoint_num=18, pretrained="", return_index=-2,
        vit_add_ln=True, ckpt_num_frame=4,
    ),
    num_query_token=32,
    qformer_hidden_dropout_prob=0.1,
    qformer_attention_probs_dropout_prob=0.1,
    qformer_drop_path_rate=0.2,
    extra_num_query_token=64,
    qformer_text_input=True,
    system="The explicit visual evidence is hidden. Answer the implicit video question by reasoning from contextual visual information, action-intent memory, and context QA memory. ",
    start_token="<Video>",
    end_token="</Video>",
    add_second_msg=True,
    img_start_token="<Image>",
    img_end_token="</Image>",
    random_shuffle=True,
    use_clip_text_matching=False,
    use_flash_attention=os.environ.get("DISABLE_FLASH_ATTN", "0") != "1",
    use_lora=True,
    lora_r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    tcr_multitask=True,
    tcr_topk_clues=3,              # backward-compatible MC default
    tcr_topk_clues_open=1,
    tcr_topk_clues_mc=3,
    tcr_clue_threshold_open=0.68,
    tcr_clue_threshold_mc=0.55,
    w_open_lm=1.0,
    # Open-first schedule. Re-enable stronger MC auxiliary weights only after
    # open-ended accuracy recovers near the original baseline.
    w_mc_lm=0.2,
    w_option_ce=0.2,
    w_relation_ce=0.5,
    w_answer_verify=0.05,
    w_answer_align=0.0,
    clue_dropout_prob=0.25,
    no_clue_prob=0.10,
    num_refine_steps=1,
)

optimizer = dict(
    opt="adamW",
    lr=1.5e-5,
    opt_betas=[0.9, 0.999],
    weight_decay=0.02,
    max_grad_norm=1.0,
    different_lr=dict(enable=False, module_names=[], lr=1e-3),
)
scheduler = dict(sched="cosine", epochs=18, min_lr_multi=0.20, warmup_epochs=1.0)

fp16 = True
gradient_checkpointing = True
evaluate = False
dist_url = "env://"
device = "cuda"
mode = "it_mistral"
output_dir = os.path.join(ROOT, "Implicit_reasoner/result/tcr_irmpp")
resume = False
debug = False
log_freq = 10
seed = 42
save_latest = True
auto_resume = False
pretrained_path = ""
wandb = dict(enable=False, entity="", project="videochat2")
