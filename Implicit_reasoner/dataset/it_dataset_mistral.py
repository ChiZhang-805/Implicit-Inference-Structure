import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader

from dataset.base_dataset import ImageVideoBaseDataset
from dataset.video_utils import VIDEO_READER_FUNCS
from dataset.tcr_video_sampling import sample_tcr_frame_indices

logger = logging.getLogger(__name__)

VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".avi")


def _repo_root():
    return Path(os.environ.get("IVQA_ROOT", str(Path(__file__).resolve().parents[2])))


class ITImgTrainDataset_mistral(ImageVideoBaseDataset):
    media_type = "image"

    def __init__(self, ann_file, transform, system="", start_token="<Image>", end_token="</Image>",
                 random_shuffle=True, return_question_instruction=False, dynamic_config=None):
        super().__init__()
        self.media_type = "video"
        self.label_file, self.data_root = self._parse_ann_file(ann_file)
        logger.info("Load json file")
        with open(self.label_file, "r") as f:
            self.anno = json.load(f)
        self.num_examples = len(self.anno)
        self.transform = transform
        self.dynamic_config = dynamic_config

        if system:
            assert system[-1] == " ", "' ' should be add in the end of system."
        self.human_start = "[INST]"
        self.human_end = "[/INST]"
        self.assist_end = "</s>"
        self.start_token = start_token
        self.end_token = end_token
        self.system = system
        self.random_shuffle = random_shuffle
        self.return_question_instruction = return_question_instruction

    def _parse_ann_file(self, ann_file):
        root = _repo_root()
        default_video_root = str(root / "dataset" / "video_data")
        if isinstance(ann_file, (list, tuple)):
            label_file = ann_file[0]
            data_root = ann_file[1] if len(ann_file) > 1 and ann_file[1] else default_video_root
        else:
            label_file = ann_file
            data_root = default_video_root
        return str(label_file), str(data_root)

    def _build_video_index(self):
        index = {}
        base = Path(self.data_root)
        if not base.exists():
            return index
        for p in base.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in VIDEO_EXTS:
                continue
            index[p.name.lower()] = str(p)
        return index

    def _resolve_video_path(self, sample):
        video_id = str(sample.get("video_id", sample.get("video", ""))).strip()
        if not video_id:
            raise FileNotFoundError("Missing video_id in sample")
        vid = video_id[:-4] if Path(video_id).suffix.lower() in VIDEO_EXTS else video_id

        candidates = []
        base_ids = [vid]
        if vid.startswith("v_"):
            base_ids.append(vid[2:])
        else:
            base_ids.append(f"v_{vid}")

        for b in base_ids:
            for ext in VIDEO_EXTS:
                candidates.append(f"{b}{ext}")

        for c in candidates:
            path = self._video_index.get(c.lower())
            if path:
                return path
        raise FileNotFoundError(f"Cannot resolve video {video_id} under {self.data_root}")

    def __len__(self):
        return self.num_examples

    def _extract_qa(self, sample):
        qa = sample.get("qa", sample)
        if isinstance(qa, list):
            qa = qa[0] if qa else {}
        return qa if isinstance(qa, dict) else {}

    def _normalize_answer_letter(self, ans, options_len):
        if ans is None:
            return ""
        if isinstance(ans, int):
            idx = ans
        else:
            t = str(ans).strip().upper()
            if t in ["A", "B", "C", "D", "E"]:
                idx = ord(t) - ord("A")
            else:
                try:
                    idx = int(t)
                except Exception:
                    idx = -1
        return chr(ord("A") + idx) if 0 <= idx < options_len else ""

    def _parse_options_and_answer(self, sample, qa):
        options = sample.get("options") or qa.get("options") or []
        options = [str(o).strip() for o in options]
        ans = sample.get("ans", qa.get("ans", qa.get("answer_idx", qa.get("answer_id"))))
        answer_letter = self._normalize_answer_letter(ans, len(options))
        return options, answer_letter

    def get_correct_option_text(self, sample):
        qa = self._extract_qa(sample)
        options, answer_letter = self._parse_options_and_answer(sample, qa)
        if answer_letter:
            idx = ord(answer_letter) - ord("A")
            if 0 <= idx < len(options):
                return options[idx]
        return ""

    def build_context_text(self, context):
        if isinstance(context, dict):
            context = list(context.values())
        lines = []
        for x in context or []:
            if isinstance(x, dict):
                q = str(x.get("question", "")).strip()
                a = str(x.get("answer", "")).strip()
                if q or a:
                    lines.append(f"Q: {q} A: {a}")
            elif isinstance(x, str) and x.strip():
                lines.append(x.strip())
        text = "Context QA memory: " + "; ".join(lines) if lines else ""
        words = text.split()
        return " ".join(words[:300]) if len(words) > 300 else text

    def expand_open_answer(self, question, short_answer, correct_option_text):
        # Open-ended supervision must stay on the original dataset answer.
        # Do NOT replace it with the correct MC option text: that makes the
        # decoder learn an option-style answer distribution and hurts GPT-based
        # open-ended evaluation.
        del question, correct_option_text
        base = (short_answer or "").strip()
        if not base:
            return ""
        if not base.endswith("."):
            base = base + "."
        return base

    def build_open_conversation(self, question, rich_open_answer, context_text, msg):
        payload = "The explicit evidence is hidden. Answer the question directly and concisely using context QA memory and visual clues."
        if context_text:
            payload += f"\n{context_text}"
        payload += f"\nQuestion: {question}"
        convo = self.system
        convo += f"{self.human_start} {self.start_token}{self.end_token}{msg.rstrip()} {self.human_end}"
        convo += f" {self.human_start} {payload} {self.human_end} {rich_open_answer} {self.assist_end}"
        return convo.strip()

    def build_mc_conversation(self, question, options, answer_letter, context_text, msg):
        option_block = "\n".join([f"({chr(ord('A') + i)}) {o}" for i, o in enumerate(options[:5])])
        payload = "The explicit evidence is hidden. Select the best option using contextual clues."
        if context_text:
            payload += f"\n{context_text}"
        payload += f"\nQuestion: {question}\n{option_block}\nReply with one option letter."
        convo = self.system
        convo += f"{self.human_start} {self.start_token}{self.end_token}{msg.rstrip()} {self.human_end}"
        convo += f" {self.human_start} {payload} {self.human_end} The answer is ({answer_letter}). {self.assist_end}"
        return convo.strip()

    def get_anno(self, index):
        sample = self.anno[index]
        qa = self._extract_qa(sample)
        context = sample.get("context", {}).get("context_question", sample.get("context", []))
        pred_rel = sample.get("action_relation_intent", "")
        if isinstance(pred_rel, str):
            pred_rel_list = [x[2:] if x.startswith("- ") else x for x in pred_rel.split("\n") if x.strip()]
        else:
            pred_rel_list = [str(x).strip() for x in pred_rel]
        return {
            "image": self._resolve_video_path(sample),
            "qa": qa,
            "whole_duration": sample.get("all_duration"),
            "mask_duration": sample.get("duration"),
            "gt_action_relation_intent": context,
            "pred_action_relation_intent": pred_rel_list,
            "action_label": sample.get("action_mathing_label", "[-1]"),
            "intent_label": sample.get("intent_mathing_label", "[-1]"),
            "question_id": sample.get("question_id", qa.get("question_id", index)),
        }


class ITVidTrainDataset_mistral(ITImgTrainDataset_mistral):
    media_type = "video"

    def __init__(self, ann_file, transform, num_frames=4, video_reader_type="decord", sample_type="rand", num_tries=3,
                 system="", start_token="<Video>", end_token="</Video>", add_second_msg=False,
                 random_shuffle=True, return_question_instruction=False, dynamic_config=None):
        super().__init__(ann_file, transform, system=system, start_token=start_token, end_token=end_token,
                         random_shuffle=random_shuffle, return_question_instruction=return_question_instruction,
                         dynamic_config=dynamic_config)
        self.num_frames = num_frames
        self.video_reader_type = video_reader_type
        self.video_reader = VIDEO_READER_FUNCS[video_reader_type]
        self.sample_type = sample_type
        self.num_tries = num_tries
        self.add_second_msg = add_second_msg
        self.tcr_multitask = bool(getattr(dynamic_config.model, "tcr_multitask", False)) if dynamic_config is not None and hasattr(dynamic_config, "model") else False
        self._video_index = self._build_video_index()

    def load_and_transform_tcr_video(self, index, video_path, question, mask_duration, all_duration, view="query"):
        del all_duration
        vr = VideoReader(video_path, num_threads=1)
        view_seed = {"query": 17, "global": 31, "boundary": 47}.get(str(view), 59)
        seed = (index * 997 + view_seed) & 0xFFFFFFFF
        frame_indices, seconds = sample_tcr_frame_indices(vr, self.num_frames, question, mask_duration=mask_duration, mode=view, seed=seed)
        frames = vr.get_batch(frame_indices).permute(0, 3, 1, 2)
        if self.dynamic_config:
            from dataset.hd_utils import HD_transform_padding, HD_transform_no_padding
            local_size = self.dynamic_config["local_size"]
            hd_num = self.dynamic_config["hd_num"]
            padding = self.dynamic_config["padding"]
            frames = HD_transform_padding(frames.float(), image_size=local_size, hd_num=hd_num) if padding else HD_transform_no_padding(frames.float(), image_size=local_size, hd_num=hd_num)
        video = self.transform(frames)
        return video, [f"{s:.3f}" for s in seconds]

    def __getitem__(self, index):
        try:
            ann = self.get_anno(index)
            qa = ann["qa"]
            question = qa.get("question", "")
            short_answer = qa.get("answer", "")
            options, answer_letter = self._parse_options_and_answer(ann, qa)
            correct_option_text = self.get_correct_option_text({**ann, "qa": qa, "options": options, "ans": answer_letter})
            rich_open_answer = self.expand_open_answer(question, short_answer, correct_option_text)

            if self.tcr_multitask:
                video, sec = self.load_and_transform_tcr_video(index, ann["image"], question, ann["mask_duration"], ann["whole_duration"], view="query")
            else:
                video, _, sec = self.load_and_transform_media_data_video(index, ann["image"], return_fps=True, clip=ann["mask_duration"], dynamic_config=self.dynamic_config)

            msg = f" The video contains {len(sec)} frames sampled at {', '.join(sec)} seconds. " if self.add_second_msg and sec else ""
            context_text = self.build_context_text(ann.get("gt_action_relation_intent", []))
            gt_ctx = ann.get("gt_action_relation_intent", [])
            if isinstance(gt_ctx, dict):
                gt_ctx = list(gt_ctx.values())
            gt_action_relation_intent = ". ".join([f"{x.get('question','')}: {x.get('answer','')}" if isinstance(x, dict) else str(x) for x in gt_ctx if x])
            pred_action_relation_intent = " ".join(ann.get("pred_action_relation_intent", []))
            instruction = ("Extract parts of the contextual visual information that are beneficial for answering the question: " + question).strip()

            if self.tcr_multitask:
                open_conversation = self.build_open_conversation(question, rich_open_answer, context_text, msg)
                mc_conversation = self.build_mc_conversation(question, options, answer_letter, context_text, msg) if options else ""
                options_json = json.dumps(options, ensure_ascii=False)
                sampled_seconds = ",".join(sec)
                return (
                    video,
                    open_conversation,
                    instruction,
                    gt_action_relation_intent,
                    pred_action_relation_intent,
                    ann["action_label"],
                    ann["intent_label"],
                    index,
                    mc_conversation,
                    options_json,
                    answer_letter,
                    context_text,
                    question,
                    rich_open_answer,
                    sampled_seconds,
                )

            conversation = self.build_open_conversation(question, short_answer, "", msg)
            return video, conversation, instruction, gt_action_relation_intent, pred_action_relation_intent, ann["action_label"], ann["intent_label"], index
        except Exception as e:
            logger.warning(f"Caught exception {e} when loading video")
            return self.__getitem__(np.random.randint(0, len(self)))


class ITTextTrainDataset_mistral(ImageVideoBaseDataset):
    media_type = "text"

    def __init__(self, ann_file, transform, system="", start_token=None, end_token=None,
                 random_shuffle=True, return_question_instruction=False, dynamic_config=None):
        super().__init__()
        self.media_type = "text"
        self.label_file, self.data_root = ann_file[:2]
        with open(self.label_file, 'r') as f:
            self.anno = json.load(f)
        self.num_examples = len(self.anno)
        self.human_start = "[INST]"
        self.human_end = "[/INST]"
        self.assist_end = "</s>"
        self.system = system
        self.random_shuffle = random_shuffle
        self.return_question_instruction = return_question_instruction

    def get_anno(self, index):
        return {"qa": self.anno[index]["QA"]}

    def __len__(self):
        return self.num_examples

    def process_qa(self, qa):
        if self.random_shuffle and len(qa) > 1:
            random.shuffle(qa)
        cur_instruction = qa[0].get("i", "") + " " if qa and qa[0].get("i", "") else ""
        conversation = self.system + cur_instruction
        for sentence in qa:
            q = sentence.get("q", "")
            a = sentence.get("a", "")
            if q:
                conversation += (" " + self.human_start + " " + q + " " + self.human_end)
            conversation += (" " + a + " " + self.assist_end)
        if self.return_question_instruction and cur_instruction:
            cur_instruction += qa[0].get("q", "")
        return conversation.strip(), cur_instruction.strip()

    def __getitem__(self, index):
        ann = self.get_anno(index)
        conversation, instruction = self.process_qa(ann["qa"])
        return torch.zeros(1), conversation, instruction, index
