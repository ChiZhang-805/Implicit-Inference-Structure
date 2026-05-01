import argparse
import json
import os
import re
from pathlib import Path

import torch
from decord import VideoReader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from utils.config import Config
from models.videochat_mistra.videochat2_it_mistral import VideoChat2_it_mistral
from dataset.tcr_video_sampling import sample_tcr_multi_views

VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".avi")
PREFERRED_OPEN_VIEW_ORDER = ["boundary", "query", "global"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--inference_file", required=True)
    p.add_argument("--video_root", required=True)
    p.add_argument("--output_file", required=True)
    p.add_argument("--num_frames", type=int, default=8)
    p.add_argument("--views", default="global,boundary,query")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--diagnostics", action="store_true")
    return p.parse_args()


def build_video_index(video_root):
    idx = {}
    for p in Path(video_root).rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            idx[p.name.lower()] = str(p)
    return idx


def resolve_video(video_id, index):
    vid = str(video_id)
    base = vid[:-4] if Path(vid).suffix.lower() in VIDEO_EXTS else vid
    cands = [base, base[2:]] if base.startswith("v_") else [base, f"v_{base}"]
    for b in cands:
        for e in VIDEO_EXTS:
            p = index.get(f"{b}{e}".lower())
            if p:
                return p
    raise FileNotFoundError(video_id)


def transform_frames(frames):
    tfm = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        transforms.Lambda(lambda x: x.float().div(255.0)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return tfm(frames)


def clean_candidate(text):
    t = (text or "").strip()
    t = re.sub(r"^(Answer:|The answer is:)\s*", "", t, flags=re.I)
    t = t.replace("</s>", "").strip()
    if "[INST]" in t:
        t = t.split("[INST]")[-1].strip()
    if "." in t:
        t = t.split(".")[0].strip() + "."
    words = t.split()
    if len(words) > 30:
        t = " ".join(words[:30])
        if not t.endswith("."):
            t += "."
    return t


def build_context_text(sample):
    context = sample.get("context", {}).get("context_question", "")
    if isinstance(context, dict):
        return "; ".join([
            f"Q:{v.get('question', '')} A:{v.get('answer', '')}"
            for v in context.values()
            if isinstance(v, dict)
        ])
    if isinstance(context, list):
        lines = []
        for v in context:
            if isinstance(v, dict):
                lines.append(f"Q:{v.get('question', '')} A:{v.get('answer', '')}")
            elif isinstance(v, str):
                lines.append(v)
        return "; ".join([x for x in lines if x.strip()])
    return str(context) if context else ""


def main():
    args = parse_args()
    cfg = Config.from_file(args.config)
    model = VideoChat2_it_mistral(config=cfg.model)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    msg = model.load_state_dict(state, strict=False)
    print(f"missing_keys={len(msg.missing_keys)}, unexpected_keys={len(msg.unexpected_keys)}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    prompt_instruction = "Please answer the question directly and concisely. Do not mention options."

    data = json.load(open(args.inference_file))
    if args.limit:
        data = data[:args.limit]
    views = [x.strip() for x in args.views.split(",") if x.strip()]
    vindex = build_video_index(args.video_root)

    outs = []
    for sample in data:
        qa = sample.get("qa", sample)
        q = qa.get("question", sample.get("question", ""))
        video_id = sample.get("video_id", qa.get("video_id"))
        path = resolve_video(video_id, vindex)
        pred_relation = sample.get("action_relation_intent", "")
        context_text = build_context_text(sample)

        vr = VideoReader(path, num_threads=1)
        sampled = sample_tcr_multi_views(
            vr,
            args.num_frames,
            q,
            mask_duration=sample.get("duration"),
            all_duration=sample.get("all_duration"),
        )

        all_candidates, frame_diag = [], {}
        clue_diag, ctx_diag = {}, {}
        with torch.no_grad():
            for v in views:
                if v not in sampled:
                    continue
                idxs = sampled[v]["indices"]
                secs = [float(s) for s in sampled[v]["seconds"]]
                frame_diag[v] = secs
                video = transform_frames(vr.get_batch(idxs).permute(0, 3, 1, 2)).unsqueeze(0).to(device)
                img_embeds, use_image = model.encode_img(video, [f"Extract useful clues for answering: {q}"])
                selected_clues, selected_ctx = model.tcr_select_memory_for_inference(
                    img_embeds[0], q, json.dumps([]), pred_relation, context_text, mode="open"
                )
                clue_diag[v] = selected_clues
                ctx_diag[v] = selected_ctx
                mem = model._build_tcr_memory_prompt(selected_clues, selected_ctx)
                mem_block = f"{mem}\n" if mem else ""
                prefix = (
                    f"{model.human_start} {model.start_token}{model.end_token} {model.human_end} "
                    f"{model.human_start} {mem_block}Question: {q}\n"
                )
                prompt = prefix + prompt_instruction + f" {model.human_end}"
                gen_embeds = model.build_tcr_generation_embeds(img_embeds, prompt, use_image)
                attn = torch.ones(gen_embeds.shape[:2], dtype=torch.long, device=device)
                out = model.mistral_model.generate(
                    inputs_embeds=gen_embeds,
                    attention_mask=attn,
                    max_new_tokens=96,
                    num_beams=1,
                    do_sample=False,
                    pad_token_id=model.mistral_tokenizer.eos_token_id,
                )
                text = model.mistral_tokenizer.decode(out[0], skip_special_tokens=True)
                all_candidates.append((v, "open", clean_candidate(text), img_embeds[0], q, selected_clues))

        candidates = [c for c in all_candidates if c[2]]
        if not candidates:
            pred = "The visible context suggests an implied event."
        else:
            ordered = []
            for view_name in PREFERRED_OPEN_VIEW_ORDER:
                ordered.extend([c for c in candidates if c[0] == view_name])
            ordered.extend([c for c in candidates if c not in ordered])
            pred = ordered[0][2]

        if len(pred.split()) < 5:
            ql = q.lower()
            if ql.startswith("why"):
                pred = f"The reason is that {pred.rstrip('.')} .".replace("  ", " ")
            elif ql.startswith("how"):
                pred = f"It is done by {pred.rstrip('.')} .".replace("  ", " ")
            else:
                pred = f"The likely answer is that {pred.rstrip('.')} .".replace("  ", " ")
            pred = pred.replace(" .", ".")

        out_item = {
            "question_id": sample.get("question_id", qa.get("question_id", "")),
            "video_id": video_id,
            "question": q,
            "answer": qa.get("answer", sample.get("answer", "")),
            "pred": pred,
            "selected_clues": clue_diag,
            "selected_context": ctx_diag,
            "frame_seconds_by_view": frame_diag,
            "candidate_preds": [c[2] for c in all_candidates if c[2]],
        }
        if not args.diagnostics:
            for k in ["selected_clues", "selected_context", "frame_seconds_by_view", "candidate_preds"]:
                out_item.pop(k)
        outs.append(out_item)

    out_dir = os.path.dirname(args.output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    json.dump(outs, open(args.output_file, "w"), indent=2)
    print(f"Saved {len(outs)} outputs -> {args.output_file}")


if __name__ == "__main__":
    main()
