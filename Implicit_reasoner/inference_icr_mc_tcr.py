import argparse
import json
import os
import time
from pathlib import Path

import torch
from tqdm import tqdm
from decord import VideoReader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from utils.config import Config
from models.videochat_mistra.videochat2_it_mistral import VideoChat2_it_mistral
from dataset.tcr_video_sampling import sample_tcr_multi_views

VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".avi")
LETTERS = ["A", "B", "C", "D", "E"]


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
    p.add_argument("--no_text_nll", action="store_true", help="Skip option-text likelihood scoring for faster MC inference.")
    p.add_argument("--save_every", type=int, default=20, help="Incrementally save predictions every N samples.")
    return p.parse_args()


def norm_letter(x, n=5):
    if x is None:
        return ""
    t = str(x).strip().upper()
    if t in LETTERS[:n]:
        return t
    if t.isdigit():
        i = int(t)
        if 0 <= i < n:
            return LETTERS[i]
    return ""


def build_video_index(video_root):
    idx = {}
    for p in Path(video_root).rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            idx[p.name.lower()] = str(p)
    return idx


def resolve_video(video_id, index):
    vid = str(video_id)
    base = vid[:-4] if Path(vid).suffix.lower() in VIDEO_EXTS else vid
    cands = [base]
    if base.startswith("v_"):
        cands.append(base[2:])
    else:
        cands.append(f"v_{base}")
    for b in cands:
        for e in VIDEO_EXTS:
            p = index.get(f"{b}{e}".lower())
            if p:
                return p
    raise FileNotFoundError(f"video not found: {video_id}")


def transform_frames(frames):
    tfm = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        transforms.Lambda(lambda x: x.float().div(255.0)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return tfm(frames)


def parse_options(sample):
    qa = sample.get("qa", sample)
    options = sample.get("options") or qa.get("options") or []
    ans = sample.get("ans", qa.get("ans", qa.get("answer")))
    return qa, [str(o) for o in options], norm_letter(ans, n=len(options) if options else 5)


def normalize_scores(vals):
    t = torch.tensor(vals, dtype=torch.float32)
    if t.numel() == 0:
        return t
    if torch.max(torch.abs(t)).item() < 1e-8:
        return torch.zeros_like(t)
    t = (t - t.mean()) / (t.std(unbiased=False) + 1e-6)
    return t


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

    data = json.load(open(args.inference_file))
    if args.limit:
        data = data[:args.limit]
    view_names = [x.strip() for x in args.views.split(",") if x.strip()]
    video_index = build_video_index(args.video_root)

    results = []
    t0 = time.time()
    for sample_idx, sample in enumerate(tqdm(data, desc="MC inference")):
        qa, options, answer = parse_options(sample)
        q = qa.get("question", sample.get("question", ""))
        video_id = sample.get("video_id", qa.get("video_id"))
        pred_relation = sample.get("action_relation_intent", "")
        context = sample.get("context", {}).get("context_question", "")
        context_text = ""
        if isinstance(context, dict):
            context_text = "; ".join([f"Q:{v.get('question','')} A:{v.get('answer','')}" for v in context.values() if isinstance(v, dict)])

        path = resolve_video(video_id, video_index)
        vr = VideoReader(path, num_threads=1)
        multi = sample_tcr_multi_views(vr, args.num_frames, q, mask_duration=sample.get("duration"), all_duration=sample.get("all_duration"))

        agg = torch.zeros(len(options), dtype=torch.float32)
        clue_diag, ctx_diag = {}, {}
        frame_diag, option_diag = {}, {}

        with torch.no_grad():
            for v in view_names:
                if v not in multi:
                    continue
                idxs = multi[v]["indices"]
                secs = [float(s) for s in multi[v]["seconds"]]
                frame_diag[v] = secs
                frames = vr.get_batch(idxs).permute(0, 3, 1, 2)
                video = transform_frames(frames).unsqueeze(0).to(device)
                instruction = [f"Extract useful clues for answering: {q}"]
                img_embeds, use_image = model.encode_img(video, instruction)
                selected_clues, selected_ctx = model.tcr_select_memory_for_inference(
                    img_embeds[0], q, json.dumps(options), pred_relation, context_text, mode="mc"
                )
                clue_diag[v] = selected_clues
                ctx_diag[v] = selected_ctx
                mem_prompt = model._build_tcr_memory_prompt(selected_clues, selected_ctx)
                option_block = "\n".join([f"({LETTERS[i]}) {opt}" for i, opt in enumerate(options)])
                prompt_prefix = (
                    f"{model.human_start} {model.start_token}{model.end_token} {model.human_end} "
                    f"{model.human_start} {mem_prompt}\nQuestion: {q}\n{option_block}\nReply with one option letter. {model.human_end}"
                )
                letter_targets = [f"The answer is ({LETTERS[i]}). {model.assist_end}" for i in range(len(options))]
                text_targets = [f"({LETTERS[i]}) {options[i]} {model.assist_end}" for i in range(len(options))]
                letter_nll = model.score_option_likelihood(img_embeds, prompt_prefix, letter_targets, use_image)
                if args.no_text_nll:
                    text_nll = [0.0 for _ in range(len(options))]
                else:
                    text_nll = model.score_option_likelihood(img_embeds, prompt_prefix, text_targets, use_image)
                cls_logits = model.score_options_classifier(img_embeds[0], q, options).detach().float().cpu()

                ln = normalize_scores(letter_nll)
                tn = normalize_scores(text_nll)
                cn = normalize_scores(cls_logits.tolist())
                if args.no_text_nll:
                    final = 0.60 * (-ln) + 0.40 * cn
                else:
                    final = 0.45 * (-ln) + 0.25 * (-tn) + 0.30 * cn
                agg += final
                option_diag[v] = {LETTERS[i]: float(final[i].item()) for i in range(len(options))}

        avg = agg / max(1, len(view_names))
        pred_idx = int(torch.argmax(avg).item()) if len(options) > 0 else 0
        pred = LETTERS[pred_idx] if options else ""
        out = {
            "question_id": sample.get("question_id", qa.get("question_id", "")),
            "video_id": video_id,
            "question": q + ("\n" + "\n".join([f"({LETTERS[i]}) {o}" for i, o in enumerate(options)]) if options else ""),
            "answer": answer,
            "pred": pred,
            "selected_clues": clue_diag,
            "selected_context": ctx_diag,
            "frame_seconds_by_view": frame_diag,
            "option_scores": {LETTERS[i]: float(avg[i].item()) for i in range(len(options))},
        }
        if not args.diagnostics:
            out.pop("selected_clues")
            out.pop("selected_context")
            out.pop("frame_seconds_by_view")
            out.pop("option_scores")
        results.append(out)
        if args.save_every > 0 and len(results) % args.save_every == 0:
            out_dir = os.path.dirname(args.output_file)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            json.dump(results, open(args.output_file, "w"), indent=2)
            elapsed = time.time() - t0
            print(f"[partial] Saved {len(results)}/{len(data)} results -> {args.output_file}; elapsed={elapsed:.1f}s", flush=True)

    out_dir = os.path.dirname(args.output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    json.dump(results, open(args.output_file, "w"), indent=2)
    print(f"Saved {len(results)} results -> {args.output_file}")


if __name__ == "__main__":
    main()
