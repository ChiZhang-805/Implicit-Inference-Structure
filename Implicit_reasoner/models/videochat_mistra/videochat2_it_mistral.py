import random
import logging
import ast
import json
import os
from pathlib import Path

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType
from ..blip2.blip2 import Blip2Base, disabled_train
from ..icr_modules import Causal_intent_RelationHead, Vision_clue_enhancement, Vision_action_enhancement
from ..tcr_modules import TCRMemorySelector, TCROptionClassifier, TCRAnswerVerifier, TCRContextQARetriever, mean_pool_text_embeds
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

logger = logging.getLogger(__name__)


class VideoChat2_it_mistral(Blip2Base):
    """
    VideoChat2 model.
    """

    def __init__(self, config):
        super().__init__()
        # pretrained_path
        vit_blip_model_path = config.get("vit_blip_model_path", None)
        mistral_model_path = config.get("mistral_model_path")
        videochat2_model_path = config.get("videochat2_model_path", "")
        freeze_vit = config.get("freeze_vit", True)
        freeze_qformer = config.get("freeze_qformer", True)
        # vit
        low_resource = config.get("low_resource", False)  # use 8 bit and put vit in cpu
        # qformer
        num_query_token = config.get("num_query_token")
        qformer_hidden_dropout_prob = config.get("qformer_hidden_dropout_prob", 0.1)
        qformer_attention_probs_dropout_prob = config.get("qformer_attention_probs_dropout_prob", 0.1)
        qformer_drop_path_rate = config.get("qformer_drop_path_rate", 0.1)
        extra_num_query_token = config.get("extra_num_query_token", 32)
        self.qformer_text_input = config.get("qformer_text_input", False)
        # prompt
        max_txt_len = config.get("max_txt_len", 32)
        self.w_ce = config.get("w_ce", 3.0)
        self.tcr_multitask = config.get("tcr_multitask", False)
        self.tcr_topk_clues = config.get("tcr_topk_clues", 3)
        self.tcr_clue_threshold = config.get("tcr_clue_threshold", 0.55)
        self.w_open_lm = config.get("w_open_lm", 1.0)
        self.w_mc_lm = config.get("w_mc_lm", 0.7)
        self.w_option_ce = config.get("w_option_ce", 1.0)
        self.w_relation_ce = config.get("w_relation_ce", 3.0)
        self.w_answer_verify = config.get("w_answer_verify", 0.5)
        self.w_answer_align = config.get("w_answer_align", 0.05)
        self.clue_dropout_prob = config.get("clue_dropout_prob", 0.25)
        self.no_clue_prob = config.get("no_clue_prob", 0.10)
        self.human_start = "[INST]"
        self.human_end = "[/INST]"
        self.assist_end = "</s>"
        self.start_token = config.get("start_token", "<Video>")
        self.end_token = config.get("end_token", "</Video>")
        self.img_start_token = config.get("img_start_token", "<Image>")
        self.img_end_token = config.get("img_end_token", "</Image>")
        logger.info(f"Add instruction in qformer: {self.qformer_text_input}")
        self.CE_loss = torch.nn.CrossEntropyLoss()
        self.debug = config.get("debug", False)
        use_flash_attention = config.get("use_flash_attention", False)
        self.use_lora = config.get("use_lora", False)
        lora_r = config.get("lora_r", 8)
        lora_alpha = config.get("lora_alpha", 32)
        lora_dropout = config.get("lora_dropout", 0.05)
        self.clip_matching = bool(config.get("use_clip_text_matching", False))
        self.clip_tokenizer = None
        self.clip_model = None
        self.tokenizer = self.init_tokenizer(truncation_side="left")
        self.tokenizer.padding_side = "left"
        self.low_resource = low_resource
        self.vision_encoder, self.vision_layernorm = self.init_vision_encoder_umt(config)
        self.qformer, self.query_tokens = self.init_Qformer(
            num_query_token, config.vision_encoder.encoder_embed_dim,
            qformer_hidden_dropout_prob=qformer_hidden_dropout_prob,
            qformer_attention_probs_dropout_prob=qformer_attention_probs_dropout_prob,
            qformer_drop_path_rate=qformer_drop_path_rate,
        )

        if not self.qformer_text_input:
            self.qformer.bert.embeddings.word_embeddings = None
            self.qformer.bert.embeddings.position_embeddings = None
            for layer in self.qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        else:
            self.qformer.resize_token_embeddings(len(self.tokenizer))
        self.qformer.cls = None

        if vit_blip_model_path:
            logger.info(f"Load ViT and QFormer from {vit_blip_model_path}")
            state_dict = torch.load(vit_blip_model_path, map_location="cpu")
            msg = self.load_state_dict(state_dict, strict=False)
            logger.info(msg)
            logger.info('Loading ViT and Q-Former Done')

        self.extra_num_query_token = extra_num_query_token
        if extra_num_query_token > 0:
            logger.info(f"Add extra {extra_num_query_token} tokens in QFormer")
            self.extra_query_tokens = nn.Parameter(
                torch.zeros(1, extra_num_query_token, self.query_tokens.shape[-1])
            )

        if freeze_vit:
            logger.info("freeze vision encoder")
            for _, param in self.vision_encoder.named_parameters():
                param.requires_grad = False
            self.vision_encoder = self.vision_encoder.eval()
            self.vision_encoder.train = disabled_train
            for _, param in self.vision_layernorm.named_parameters():
                param.requires_grad = False
            self.vision_layernorm = self.vision_layernorm.eval()
            self.vision_layernorm.train = disabled_train

        if freeze_qformer:
            logger.info("freeze Qformer")
            for _, param in self.qformer.named_parameters():
                param.requires_grad = False
            self.qformer = self.qformer.eval()
            self.qformer.train = disabled_train
            self.query_tokens.requires_grad = False

        logger.info('Loading Mistral')
        self.mistral_tokenizer = AutoTokenizer.from_pretrained(mistral_model_path)
        self.mistral_tokenizer.padding_side = "left"
        if not self.mistral_tokenizer.pad_token:
            logger.info("Set pad_token")
            self.mistral_tokenizer.pad_token = self.mistral_tokenizer.eos_token

        if self.debug:
            logger.info("Debug mode, build small Mistral")
            mistral_config = AutoConfig.from_pretrained(mistral_model_path)
            mistral_config.hidden_size = 512
            mistral_config.intermediate_size = 2048
            mistral_config.num_attention_heads = 8
            mistral_config.num_hidden_layers = 12
            mistral_config.torch_dtype = torch.float16
            self.mistral_model = AutoModelForCausalLM.from_config(mistral_config)
        else:
            if use_flash_attention:
                self.mistral_model = AutoModelForCausalLM.from_pretrained(
                    mistral_model_path,
                    torch_dtype=torch.float16,
                    # use_flash_attention_2=True,
                    attn_implementation="flash_attention_2",
                )
            else:
                self.mistral_model = AutoModelForCausalLM.from_pretrained(
                    mistral_model_path,
                    torch_dtype=torch.float16,
                )

        logger.info("freeze Mistral")
        for _, param in self.mistral_model.named_parameters():
            param.requires_grad = False
        logger.info('Loading Mistral Done')

        if self.use_lora:
            logger.info("Use lora to finetune mistral")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, inference_mode=False,
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj", "lm_head"]
            )
            self.mistral_model = get_peft_model(self.mistral_model, peft_config)
            print("for mistral model:")
            self.mistral_model.print_trainable_parameters()

        self.relation_head = Causal_intent_RelationHead(config)
        self.clue_enhance = Vision_clue_enhancement(config)
        self.vision_enhance = Vision_action_enhancement(config)
        self.mistral_proj = nn.Linear(
            self.qformer.config.hidden_size, self.mistral_model.config.hidden_size
        )
        self.max_txt_len = max_txt_len
        self.tcr_selector = TCRMemorySelector(self.mistral_model.config.hidden_size)
        self.option_classifier = TCROptionClassifier(self.mistral_model.config.hidden_size)
        self.answer_verifier = TCRAnswerVerifier(self.mistral_model.config.hidden_size)
        self.context_retriever = TCRContextQARetriever()

        if self.clip_matching:
            from transformers import CLIPTokenizer, CLIPTextModel
            clip_path = config.get("clip_model_path", None)
            if clip_path is None:
                root = Path(os.environ.get("IVQA_ROOT", str(Path(__file__).resolve().parents[3])))
                fallback = root / "weights" / "clip-vit-base-patch32"
                clip_path = str(fallback) if fallback.exists() else None
            if clip_path:
                self.clip_tokenizer = CLIPTokenizer.from_pretrained(clip_path)
                self.clip_model = CLIPTextModel.from_pretrained(clip_path)
            else:
                logger.warning("CLIP text matching enabled but no valid clip_model_path found; disabling.")
                self.clip_matching = False

        # load weights of VideoChat2
        if videochat2_model_path:
            logger.info(f"Load VideoChat2 from: {videochat2_model_path}")
            ckpt = torch.load(videochat2_model_path, map_location="cpu")
            if 'model' in ckpt.keys():
                msg = self.load_state_dict(ckpt['model'], strict=False)
            else:
                msg = self.load_state_dict(ckpt, strict=False)
            logger.info(msg)

        # causal_ckpt_path = '...'
        # causal_ckpt = torch.load(causal_ckpt_path, map_location="cpu")
        # relation_head_weights = {k: v for k, v in causal_ckpt["model"].items() if 'relation_head' in k}
        # relation_head_weights_new, clue_enhance_weights_new, vision_enhance_weights_new = {}, {}, {}
        # for key, value in relation_head_weights.items():
        #     new_key = key
        #     if 'relation_head.' in key:
        #         new_key = key.replace('relation_head.', '')  # 移除 relation_head 前缀
        #     relation_head_weights_new[new_key] = value
        # relation_head_msg = self.relation_head.load_state_dict(relation_head_weights_new, strict=False)
        #
        # clue_enhance_weights = {k: v for k, v in causal_ckpt["model"].items() if 'clue_enhance' in k}
        # for key, value in clue_enhance_weights.items():
        #     new_key = key
        #     if 'clue_enhance.' in key:
        #         new_key = key.replace('clue_enhance.', '')  # 移除 relation_head 前缀
        #     clue_enhance_weights_new[new_key] = value
        # clue_enhance_msg = self.clue_enhance.load_state_dict(clue_enhance_weights_new, strict=False)
        #
        # vision_enhance_weights = {k: v for k, v in causal_ckpt["model"].items() if 'vision_enhance' in k}
        # for key, value in vision_enhance_weights.items():
        #     new_key = key
        #     if 'vision_enhance.' in key:
        #         new_key = key.replace('vision_enhance.', '')  # 移除 relation_head 前缀
        #     vision_enhance_weights_new[new_key] = value
        # vision_enhance_msg = self.vision_enhance.load_state_dict(vision_enhance_weights_new, strict=False)
        # print("causal weight successfully loaded!!!")


    def vit_to_cpu(self):
        self.vision_layernorm.to("cpu")
        self.vision_layernorm.float()
        self.vision_encoder.to("cpu")
        self.vision_encoder.float()

    def encode_img(self, image, instruction):
        device = image.device
        if self.low_resource:
            self.vit_to_cpu()
            image = image.to("cpu")

        with self.maybe_autocast():
            T = image.shape[1]
            use_image = True if T == 1 else False
            image = image.permute(0, 2, 1, 3, 4)  # [B,T,C,H,W] -> [B,C,T,H,W]

            image_embeds = self.vision_encoder(image, use_image)
            B, T, L, C = image_embeds.shape
            image_embeds = image_embeds.reshape(B, -1, C)
            image_embeds = self.vision_layernorm(image_embeds).to(device)  # [B, T*L, C]

            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            if self.extra_num_query_token > 0:
                query_tokens = torch.cat([self.query_tokens, self.extra_query_tokens], dim=1)
            else:
                query_tokens = self.query_tokens
            query_tokens = query_tokens.expand(image_embeds.shape[0], -1, -1)
            if self.qformer_text_input:
                text_Qformer = self.tokenizer(
                    instruction,
                    padding='longest',
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(image_embeds.device)
                query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image_embeds.device)
                Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

                query_output = self.qformer.bert(
                    text_Qformer.input_ids,
                    attention_mask=Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
            else:
                query_output = self.qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )

            inputs_mistral = self.mistral_proj(query_output.last_hidden_state[:, :query_tokens.size(1), :])
        return inputs_mistral, use_image

    def _get_text_len(self, text):
        return self.mistral_tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.shape[1]

    def _text_embed_mean(self, texts, max_length=64):
        if len(texts) == 0:
            return torch.zeros(0, self.mistral_model.config.hidden_size, device=self.query_tokens.device)
        tok = self.mistral_tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(self.query_tokens.device)
        if self.use_lora:
            embedder = self.mistral_model.base_model.model.model.embed_tokens
        else:
            embedder = self.mistral_model.model.embed_tokens
        emb = embedder(tok.input_ids).float()
        return mean_pool_text_embeds(emb, tok.attention_mask)

    def _parse_labels(self, action_label, intent_label, n_candidates):
        def parse(x):
            if isinstance(x, str):
                try:
                    x = ast.literal_eval(x)
                except Exception:
                    x = []
            if not isinstance(x, (list, tuple)):
                x = []
            return list(x)
        a = parse(action_label)
        b = parse(intent_label)
        labels, valid = [], []
        for i in range(n_candidates):
            av = a[i] if i < len(a) else -1
            bv = b[i] if i < len(b) else -1
            if av == -1 and bv == -1:
                labels.append(0)
                valid.append(False)
            else:
                labels.append(1 if (av == 1 or bv == 1) else 0)
                valid.append(True)
        return torch.tensor(labels, device=self.query_tokens.device), torch.tensor(valid, device=self.query_tokens.device)

    def _parse_pred_pairs(self, pred_relation):
        pairs = []
        for r in str(pred_relation).split('. '):
            if ':' in r:
                left, right = r.split(':', 1)
                act = left.strip().lstrip('-').strip()
                it = right.strip().rstrip('.')
                if act or it:
                    pairs.append((act, it))
        return pairs

    def _select_tcr_clues(self, img_embeds_single, question, options_json, pred_relation, action_label, intent_label, context_text, mode):
        pairs = self._parse_pred_pairs(pred_relation)
        if not pairs:
            return [], [], torch.tensor(0.0, device=img_embeds_single.device)
        memory_texts = [f"{a} -> {i}" for a, i in pairs]
        mem = self._text_embed_mean(memory_texts)
        q = self._text_embed_mean([question])[0]
        opt = None
        options = []
        try:
            options = json.loads(options_json) if options_json else []
        except Exception:
            options = []
        if options:
            opt = self._text_embed_mean([str(x) for x in options]).mean(dim=0)
        vis = img_embeds_single.mean(dim=0)
        logits, probs = self.tcr_selector(mem, q, opt, vis)
        labels, valid_mask = self._parse_labels(action_label, intent_label, len(pairs))
        if valid_mask.any():
            relation_loss = self.CE_loss(logits[valid_mask], labels[valid_mask])
        else:
            relation_loss = torch.tensor(0.0, device=img_embeds_single.device)

        kept = []
        idx = torch.argsort(probs, descending=True).tolist()
        for i in idx:
            p = probs[i].item()
            if p >= self.tcr_clue_threshold:
                kept.append((pairs[i][0], pairs[i][1], p))
            if len(kept) >= self.tcr_topk_clues:
                break
        if mode == 'mc' and len(kept) == 0 and len(idx) > 0 and probs[idx[0]].item() >= 0.45:
            i = idx[0]
            kept.append((pairs[i][0], pairs[i][1], probs[i].item()))

        ctx_lines = []
        if context_text:
            pieces = [x.strip() for x in context_text.replace('Context QA memory:', '').split(';') if x.strip()]
            ctx_lines = self.context_retriever.rank(question, pieces, topk=2)
        return kept, ctx_lines, relation_loss

    def _build_tcr_memory_prompt(self, selected_pairs, selected_context_lines):
        lines = ["Relevant context memory:"]
        for a, i, _ in selected_pairs:
            lines.append(f"- Action: {a} Intent: {i}")
        for c in selected_context_lines:
            lines.append(f"- Context QA: {c}")
        text = "\n".join(lines)
        w = text.split()
        if len(w) > 220:
            text = ' '.join(w[:220])
        return text

    def _inject_memory_prompt(self, prompt, mem_prompt):
        if not mem_prompt or self.human_start not in prompt:
            return prompt
        return prompt.replace(self.human_start + ' ', self.human_start + ' ' + mem_prompt + "\n", 1)

    def _lm_loss_for_prompts(self, img_embeds, prompts, use_image):
        if len(prompts) == 0:
            return torch.tensor(0.0, device=img_embeds.device)
        loss = self.forward_legacy_with_embeds(img_embeds, prompts, use_image)
        return loss

    def forward_legacy_with_embeds(self, img_embeds, text_input, use_image):
        batch_size, img_len, _ = img_embeds.shape
        max_len = 0
        input_embed_list, p_before_len_list, target_list = [], [], []
        for idx, prompt in enumerate(text_input):
            tmp_img_embeds = img_embeds[idx].unsqueeze(0)
            end_token = self.img_end_token if use_image else self.end_token
            p_before, p_after = prompt.split(end_token)
            p_after = end_token + p_after
            p_before_tokens = self.mistral_tokenizer(p_before, return_tensors='pt', add_special_tokens=False).to(tmp_img_embeds.device)
            p_after_tokens = self.mistral_tokenizer(p_after, return_tensors='pt', add_special_tokens=False).to(tmp_img_embeds.device)
            emb_fn = self.mistral_model.base_model.model.model.embed_tokens if self.use_lora else self.mistral_model.model.embed_tokens
            p_before_embeds = emb_fn(p_before_tokens.input_ids)
            p_after_embeds = emb_fn(p_after_tokens.input_ids)
            input_embeds = torch.cat([p_before_embeds, tmp_img_embeds, p_after_embeds], dim=1)
            sep1 = self.human_start + ' '
            sep2 = ' ' + self.human_end + ' '
            raw_text = p_after.split(sep2)
            for j in range(0, len(raw_text)-1):
                raw_text[j] = raw_text[j] + sep2
            answer_targets = p_after_tokens.input_ids.clone()
            cur_len = self._get_text_len(raw_text[0].rstrip())
            answer_targets[:, :cur_len] = -100
            for text in raw_text[1:-1]:
                total_len = self._get_text_len(text.rstrip())
                ans_len = self._get_text_len((text.split(sep1)[0]).rstrip())
                answer_targets[:, (cur_len + ans_len):(cur_len + total_len)] = -100
                cur_len += total_len
            cur_len += self._get_text_len(raw_text[-1].rstrip())
            max_len = max(max_len, input_embeds.shape[1])
            input_embed_list.append(input_embeds)
            p_before_len_list.append(p_before_tokens.input_ids.shape[1])
            target_list.append(answer_targets)
        txt_len = min(max_len + 1, self.max_txt_len + img_len)
        inp = torch.ones([batch_size, txt_len], dtype=torch.long, device=img_embeds.device) * self.mistral_tokenizer.pad_token_id
        emb_fn = self.mistral_model.base_model.model.model.embed_tokens if self.use_lora else self.mistral_model.model.embed_tokens
        inputs_embeds = emb_fn(inp)
        attention_mask = torch.zeros([batch_size, txt_len], dtype=torch.long, device=img_embeds.device)
        targets = torch.ones([batch_size, txt_len], dtype=torch.long, device=img_embeds.device).fill_(-100)
        bos_ids = torch.full((batch_size, 1), self.mistral_tokenizer.bos_token_id, device=img_embeds.device, dtype=torch.long)
        bos_embeds = emb_fn(bos_ids)
        inputs_embeds[:, :1] = bos_embeds
        for idx in range(batch_size):
            input_len = min(input_embed_list[idx].shape[1], txt_len - 1)
            inputs_embeds[idx, 1:(input_len+1)] = input_embed_list[idx][:, :input_len]
            attention_mask[idx, :(input_len+1)] = 1
            p_before_len = p_before_len_list[idx]
            targets[idx, (p_before_len + img_len + 1):(input_len + 1)] = target_list[idx][0, :(input_len - p_before_len - img_len)]
        outputs = self.mistral_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, return_dict=True, labels=targets, use_cache=False)
        return outputs.loss

    def _option_classification_loss(self, img_embeds, raw_questions, options_jsons, answer_letters):
        losses = []
        for i, q in enumerate(raw_questions):
            try:
                options = json.loads(options_jsons[i]) if options_jsons[i] else []
            except Exception:
                options = []
            if not options:
                continue
            letter = (answer_letters[i] or '').strip().upper()
            if letter not in ['A','B','C','D','E']:
                continue
            target = ord(letter)-ord('A')
            if target >= len(options):
                continue
            q_emb = self._text_embed_mean([q])[0]
            o_emb = self._text_embed_mean([str(x) for x in options])
            v_emb = img_embeds[i].mean(dim=0)
            logits = self.option_classifier(v_emb, q_emb, o_emb)
            losses.append(self.CE_loss(logits.unsqueeze(0), torch.tensor([target], device=img_embeds.device)))
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=img_embeds.device)

    def _answer_verifier_loss(self, img_embeds, raw_questions, rich_open_answers, options_jsons, answer_letters):
        losses = []
        b = len(raw_questions)
        for i in range(b):
            pos = (rich_open_answers[i] or '').strip()
            if not pos:
                continue
            cands = [pos]
            try:
                opts = json.loads(options_jsons[i]) if options_jsons and options_jsons[i] else []
            except Exception:
                opts = []
            letter = (answer_letters[i] or '').strip().upper()
            gt_idx = ord(letter)-ord('A') if letter in ['A','B','C','D','E'] else -1
            for j, o in enumerate(opts):
                if j != gt_idx:
                    cands.append(str(o))
            if b > 1:
                cands.append(rich_open_answers[(i+1) % b])
            q_emb = self._text_embed_mean([raw_questions[i]])[0]
            v_emb = img_embeds[i].mean(dim=0)
            a_emb = self._text_embed_mean(cands)
            logits = torch.stack([self.answer_verifier(v_emb, q_emb, a_emb[k]) for k in range(a_emb.shape[0])])
            losses.append(self.CE_loss(logits.unsqueeze(0), torch.tensor([0], device=img_embeds.device)))
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=img_embeds.device)

    def tcr_select_memory_for_inference(self, img_embeds_single, question, options_json, pred_relation, context_text, mode):
        action_label = "[-1]"
        intent_label = "[-1]"
        selected_pairs, selected_context, _ = self._select_tcr_clues(img_embeds_single, question, options_json, pred_relation, action_label, intent_label, context_text, mode)
        return selected_pairs, selected_context

    def build_tcr_generation_embeds(self, img_embeds, prompt, use_image):
        end_token = self.img_end_token if use_image else self.end_token
        if end_token in prompt:
            p_before, p_after = prompt.split(end_token, 1)
            p_after = end_token + p_after
        else:
            p_before, p_after = prompt, ""
        tok_before = self.mistral_tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        tok_after = self.mistral_tokenizer(p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        emb_fn = self.mistral_model.base_model.model.model.embed_tokens if self.use_lora else self.mistral_model.model.embed_tokens
        return torch.cat([emb_fn(tok_before.input_ids), img_embeds, emb_fn(tok_after.input_ids)], dim=1)

    def score_option_likelihood(self, img_embeds, prompt_prefix, option_targets, use_image):
        scores = []
        for target in option_targets:
            prompt = f"{prompt_prefix} {target}"
            loss = self.forward_legacy_with_embeds(img_embeds, [prompt], use_image)
            scores.append(float(loss.detach().item()))
        return scores

    def score_options_classifier(self, img_embeds, question, options):
        if not options:
            return torch.zeros(0, device=img_embeds.device)
        q_emb = self._text_embed_mean([question])[0]
        o_emb = self._text_embed_mean([str(x) for x in options])
        v_emb = img_embeds.mean(dim=0)
        return self.option_classifier(v_emb, q_emb, o_emb)

    def verify_open_candidates(self, img_embeds, question, candidates):
        if not candidates:
            return torch.zeros(0, device=img_embeds.device)
        q_emb = self._text_embed_mean([question])[0]
        v_emb = img_embeds.mean(dim=0)
        a_emb = self._text_embed_mean([str(c) for c in candidates])
        logits = [self.answer_verifier(v_emb, q_emb, a_emb[i]) for i in range(a_emb.shape[0])]
        return torch.stack(logits)

    def _answer_alignment_loss(self, img_embeds, rich_open_answer, options_json=None, answer_letter=None):
        if not rich_open_answer:
            return torch.tensor(0.0, device=img_embeds.device)
        losses = []
        bsz = len(rich_open_answer)
        for i in range(bsz):
            pos = (rich_open_answer[i] or "").strip()
            if not pos:
                continue
            negs = []
            try:
                opts = json.loads(options_json[i]) if options_json and options_json[i] else []
            except Exception:
                opts = []
            gt = (answer_letter[i] or "").strip().upper() if answer_letter else ""
            gt_idx = ord(gt) - ord("A") if gt in ["A", "B", "C", "D", "E"] else -1
            for j, o in enumerate(opts):
                if j != gt_idx:
                    negs.append(str(o))
            if bsz > 1 and rich_open_answer[(i+1) % bsz]:
                negs.append(str(rich_open_answer[(i+1) % bsz]))
            if not negs:
                continue
            v = torch.nn.functional.normalize(img_embeds[i].mean(dim=0), dim=0)
            pos_e = torch.nn.functional.normalize(self._text_embed_mean([pos])[0], dim=0)
            neg_e = torch.nn.functional.normalize(self._text_embed_mean(negs), dim=1)
            pos_sim = torch.matmul(v, pos_e)
            neg_sim = torch.matmul(neg_e, v)
            logits = torch.cat([pos_sim.unsqueeze(0), neg_sim], dim=0).unsqueeze(0) / 0.07
            losses.append(self.CE_loss(logits, torch.tensor([0], device=img_embeds.device)))
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=img_embeds.device)

    def _forward_tcr(self, image, text_input, instruction, gt_relation, pred_relation, action_label, intent_label, epoch,
                     mc_text_input=None, options_json=None, answer_letter=None, context_text=None, raw_question=None,
                     rich_open_answer=None, sampled_seconds=None):
        img_embeds, use_image = self.encode_img(image, instruction)
        open_prompts = list(text_input)
        mc_prompts = list(mc_text_input) if mc_text_input is not None else []
        rel_losses = []
        for i in range(len(open_prompts)):
            q = raw_question[i] if raw_question else ''
            oj = options_json[i] if options_json else '[]'
            ctx = context_text[i] if context_text else ''
            sp, sc, rl = self._select_tcr_clues(img_embeds[i], q, oj, pred_relation[i], action_label[i], intent_label[i], ctx, mode='open')
            mem = self._build_tcr_memory_prompt(sp, sc)
            open_prompts[i] = self._inject_memory_prompt(open_prompts[i], mem)
            rel_losses.append(rl)
            if i < len(mc_prompts) and mc_prompts[i]:
                sp2, sc2, rl2 = self._select_tcr_clues(img_embeds[i], q, oj, pred_relation[i], action_label[i], intent_label[i], ctx, mode='mc')
                mem2 = self._build_tcr_memory_prompt(sp2, sc2)
                mc_prompts[i] = self._inject_memory_prompt(mc_prompts[i], mem2)
                rel_losses.append(rl2)

        loss_open_lm = self._lm_loss_for_prompts(img_embeds, open_prompts, use_image)
        loss_mc_lm = self._lm_loss_for_prompts(img_embeds, [p for p in mc_prompts if p], use_image) if mc_prompts else torch.tensor(0.0, device=img_embeds.device)
        loss_option_ce = self._option_classification_loss(img_embeds, raw_question or ['']*len(open_prompts), options_json or ['[]']*len(open_prompts), answer_letter or ['']*len(open_prompts))
        loss_relation_ce = torch.stack(rel_losses).mean() if rel_losses else torch.tensor(0.0, device=img_embeds.device)
        loss_answer_verify = self._answer_verifier_loss(img_embeds, raw_question or ['']*len(open_prompts), rich_open_answer or ['']*len(open_prompts), options_json or ['[]']*len(open_prompts), answer_letter or ['']*len(open_prompts))
        loss_answer_align = self._answer_alignment_loss(img_embeds, rich_open_answer or ["" for _ in open_prompts], options_json=options_json, answer_letter=answer_letter)
        relation_weight = max(0.5, self.w_relation_ce - 0.09 * epoch)
        loss = self.w_open_lm*loss_open_lm + self.w_mc_lm*loss_mc_lm + self.w_option_ce*loss_option_ce + relation_weight*loss_relation_ce + self.w_answer_verify*loss_answer_verify + self.w_answer_align*loss_answer_align
        return dict(loss=loss, loss_open_lm=loss_open_lm, loss_mc_lm=loss_mc_lm, loss_option_ce=loss_option_ce, loss_relation_ce=loss_relation_ce, loss_answer_verify=loss_answer_verify, loss_answer_align=loss_answer_align)

    def forward(self, image, text_input, instruction, gt_relation, pred_relation, action_label, intent_label, epoch, mc_text_input=None, options_json=None, answer_letter=None, context_text=None, raw_question=None, rich_open_answer=None, sampled_seconds=None):
        if self.tcr_multitask:
            return self._forward_tcr(image, text_input, instruction, gt_relation, pred_relation, action_label, intent_label, epoch, mc_text_input=mc_text_input, options_json=options_json, answer_letter=answer_letter, context_text=context_text, raw_question=raw_question, rich_open_answer=rich_open_answer, sampled_seconds=sampled_seconds)

        global action_list, intent_list, loss_ce
        img_embeds, use_image = self.encode_img(image, instruction)
        batch_size, img_len, _ = img_embeds.shape
        loss_ce = 0
        # mark the largest length
        # when padding, the attention mask will be 0
        max_len = 0
        input_embed_list = []
        p_before_len_list = []
        target_list = []

        gt_pairs, pred_pairs = [], []
        for idx, gt_prompt in enumerate(gt_relation):
            gt_pairs.append(gt_prompt.split(". "))
            pred_pairs.append(pred_relation[idx].split(". "))

        # match the pairs while training
        # similarity matching which is not for backward propagation
        pairs_cross_examples_wo_backward = []
        for batch_id, gt_pair in enumerate(gt_pairs):
            pairs = []
            pred_pair = pred_pairs[batch_id]
            gt_actions = [pair.split(":")[0] for pair in gt_pair]
            pred_actions = [pair.split(":")[0] for pair in pred_pair]
            gt_actions = [s.lstrip() for s in gt_actions]
            pred_actions = [s.lstrip() for s in pred_actions]
            # similarity matching
            for pred_index, pred in enumerate(pred_actions):
                max_similarity = 0
                best_match_index = None
                if self.clip_matching and self.clip_model is not None and self.clip_tokenizer is not None:
                    self.clip_model = self.clip_model.to(img_embeds.device)
                    pred_inputs = self.clip_tokenizer(pred, return_tensors="pt", padding=True, truncation=True).to(img_embeds.device)
                    with torch.no_grad():
                        pred_embedding = self.clip_model(**pred_inputs).last_hidden_state.mean(dim=1)
                    for gt_index, gt in enumerate(gt_actions):
                        gt_inputs = self.clip_tokenizer(gt, return_tensors="pt", padding=True, truncation=True).to(img_embeds.device)
                        with torch.no_grad():
                            gt_embedding = self.clip_model(**gt_inputs).last_hidden_state.mean(dim=1)
                        similarity = torch.nn.functional.cosine_similarity(pred_embedding, gt_embedding).item()
                        if similarity > 0.65 and similarity > max_similarity:
                            max_similarity = similarity
                            best_match_index = gt_index

                if best_match_index is not None:
                    pairs.append({
                        "pred_action": pred_pair[pred_index].split(": ")[0][1:].lstrip(),
                        "pred_intent": pred_pair[pred_index].split(": ")[1].lstrip() if pred_pair[pred_index].
                        split(": ")[1].lstrip().endswith(".") else pred_pair[pred_index].split(": ")[1].lstrip() + "."
                    })
            pairs_cross_examples_wo_backward.append(pairs)

        # gpt4_matching which is for backward propagation
        pairs_cross_examples = []
        for idx, gt_relation_sample in enumerate(gt_relation):
            # gt_relation_sample = gt_relation_sample.split(". ")
            pred_relation_sample = pred_relation[idx].split(". ")
            action_label_sample = ast.literal_eval(action_label[idx])
            # intent_label_sample = ast.literal_eval(intent_label[idx])
            # gt_actions = [relation.split(": ")[0] for relation in gt_relation_sample]
            # gt_intents = [relation.split(": ")[1] for relation in gt_relation_sample]
            pred_actions = [relation.split(": ")[0][1:] for relation in pred_relation_sample]
            # print(pred_relation_sample)
            # print(pred_relation_sample)
            # for sammple in pred_relation_sample:
            #     print(sammple.split(": ")[1])
            pred_intents = [relation.split(": ")[1] for relation in pred_relation_sample]
            # print(pred_intents)
            pred_actions = [s.lstrip() for s in pred_actions]
            pred_intents = [s.lstrip() for s in pred_intents]
            pairs = []
            for index, pred_action in enumerate(pred_actions):
                if action_label_sample[index] != -1:
                    pairs.append({
                        "pred_action": pred_action,
                        "pred_intent": pred_intents[index] if pred_intents[index].endswith(".")
                        else pred_intents[index] + "."
                    })
            pairs_cross_examples.append(pairs)

        # backward + non-backward -> complete pairs
        complete_pairs = [a + b for a, b in zip(pairs_cross_examples, pairs_cross_examples_wo_backward)]

        # calculate the text embeddings of all action-intent pairs
        all_action_list, all_intent_list = [], []
        for example_pairs in complete_pairs:
            action_list, intent_list = [], []
            for pair in example_pairs:
                pred_action = pair["pred_action"]
                pred_action_tokens = (
                    self.mistral_tokenizer(pred_action, return_tensors="pt", padding='max_length', truncation=True,
                                           max_length=15).
                    to(img_embeds.device))
                pred_intent = pair["pred_intent"]
                pred_intent_tokens = (
                    self.mistral_tokenizer(pred_intent, return_tensors="pt", padding='max_length', truncation=True,
                                           max_length=15).
                    to(img_embeds.device))

                if self.use_lora:
                    pred_action_embeds = self.mistral_model.base_model.model.model.embed_tokens(
                        pred_action_tokens.input_ids)
                    pred_intent_embeds = self.mistral_model.base_model.model.model.embed_tokens(
                        pred_intent_tokens.input_ids)
                else:
                    pred_action_embeds = self.mistral_model.model.embed_tokens(pred_action_tokens.input_ids)
                    pred_intent_embeds = self.mistral_model.model.embed_tokens(pred_intent_tokens.input_ids)

                action_list.append(pred_action_embeds)
                intent_list.append(pred_intent_embeds)

            all_action_list.append(action_list)
            all_intent_list.append(intent_list)

        # action_enhancement + relation deduction
        # calculate the pair relation
        # 0 for negative, 1 for positive
        self.clue_enhance = self.clue_enhance.to(img_embeds.device)
        self.relation_head = self.relation_head.to(img_embeds.device)

        results = []
        for index, actions in enumerate(all_action_list):
            if actions:
                intent_pred_query = all_intent_list[index]
                enhanced_intent = self.clue_enhance(actions, img_embeds[index, :, :])
                relation_pred = self.relation_head(intent_pred_query, enhanced_intent)
                relation_pred = [relation.squeeze(0)[0, :] for relation in relation_pred]
                relation_pred = [t.unsqueeze(0) for t in relation_pred]
                relation_pred = torch.cat(relation_pred, dim=0)
                result = [1 if t[1] > t[0] else 0 for t in relation_pred]
                results.append(result)
                # auxiliary loss calculation
                if intent_label[index] != '[-1]':
                    gt_list = ast.literal_eval(intent_label[index])
                    gt_length = len(gt_list)
                    gt_label = torch.tensor(gt_list).to(img_embeds.device)
                    loss_ce += self.CE_loss(relation_pred[:gt_length], gt_label)
            else:
                result = []
                results.append(result)
        text_input = list(text_input)
        actions_for_refine_qformers = []
        for index, result in enumerate(results):
            if not (all((item == 0) for item in result) or (len(result) == 0)):
                # ensuring there is at least one positive pair
                clue_prompt_str = "Context clues are as following: "
                actions_for_refine_qformer = "Context key action clues: "
                last_index = len(result) - 1 - result[::-1].index(1)  # find the last positive position index
                for label_idx, label in enumerate(result):
                    if label == 1:
                        clue_prompt_dict = complete_pairs[index][label_idx]
                        if label_idx != last_index:
                            # not the final pair
                            actions_for_refine_qformer += clue_prompt_dict["pred_action"] + ", "
                            clue_prompt_str += clue_prompt_dict["pred_action"] + " to " + clue_prompt_dict["pred_intent"][:-1] + ", "
                        else:
                            # the final pair
                            actions_for_refine_qformer += clue_prompt_dict["pred_action"] + "."
                            clue_prompt_str += clue_prompt_dict["pred_action"] + " to " + clue_prompt_dict["pred_intent"] + " "
                output_str = text_input[index].replace("[INST]", clue_prompt_str + "[INST]", 1)
                text_input[index] = output_str
            else:
                actions_for_refine_qformer = ""
            actions_for_refine_qformers.append(actions_for_refine_qformer)
        text_input = tuple(text_input)

        origin_img_embeds_list = list(torch.split(img_embeds, 1, dim=0))

        # enhance the vision input
        for idx, action in enumerate(actions_for_refine_qformers):
            if action != "":
                img_embeds_refine, _ = self.encode_img(image[idx].unsqueeze(0), action)
                refined_output = self.vision_enhance(img_embeds_refine, origin_img_embeds_list[idx])
                origin_img_embeds_list[idx] = refined_output
        img_embeds = torch.cat(origin_img_embeds_list, dim=0)

        # handle each prompt individually (written by videochat2.)
        for idx, prompt in enumerate(text_input):
            tmp_img_embeds = img_embeds[idx].unsqueeze(0)
            # split the prompt via END_TOKEN
            end_token = self.img_end_token if use_image else self.end_token
            p_before, p_after = prompt.split(end_token)
            p_after = end_token + p_after
            p_before_tokens = self.mistral_tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(
                tmp_img_embeds.device)
            p_after_tokens = self.mistral_tokenizer(p_after, return_tensors="pt", add_special_tokens=False).to(
                tmp_img_embeds.device)
            if self.use_lora:
                p_before_embeds = self.mistral_model.base_model.model.model.embed_tokens(p_before_tokens.input_ids)
                p_after_embeds = self.mistral_model.base_model.model.model.embed_tokens(p_after_tokens.input_ids)
            else:
                p_before_embeds = self.mistral_model.model.embed_tokens(p_before_tokens.input_ids)
                p_after_embeds = self.mistral_model.model.embed_tokens(p_after_tokens.input_ids)
            input_embeds = torch.cat([p_before_embeds, tmp_img_embeds, p_after_embeds], dim=1)

            # extract the answers and mask the target
            # the answers are only in the p_after
            sep1 = self.human_start + " "
            sep2 = " " + self.human_end + " "
            raw_text = p_after.split(sep2)
            for idx in range(0, len(raw_text) - 1):
                raw_text[idx] = raw_text[idx] + sep2
            # the first raw_text contains system and question
            # the last raw_text only contains answer
            # rstrip() for the extra " "
            answer_targets = p_after_tokens.input_ids.clone()
            # [target] "xxxxx. </s>"
            cur_len = self._get_text_len(raw_text[0].rstrip())
            answer_targets[:, :cur_len] = -100
            for text in raw_text[1:-1]:
                total_len = self._get_text_len(text.rstrip())
                ans_len = self._get_text_len((text.split(sep1)[0]).rstrip())
                answer_targets[:, (cur_len + ans_len):(cur_len + total_len)] = -100
                cur_len += total_len
            cur_len += self._get_text_len(raw_text[-1].rstrip())

            if self.debug:  # Inspect and check the correctness of masking
                z = answer_targets[0].clone()
                z = torch.where(z == -100, self.mistral_tokenizer.unk_token_id, z)
                logger.info(self.mistral_tokenizer.decode(z))

            assert cur_len == answer_targets.shape[
                1], f"The final length ({cur_len}) is not equal to the original prompt ({answer_targets.shape[1]}): {prompt}"

            max_len = max(max_len, input_embeds.shape[1])
            input_embed_list.append(input_embeds)
            p_before_len_list.append(p_before_tokens.input_ids.shape[1])
            target_list.append(answer_targets)

        # plus one for bos
        # max_txt_len plus num_query_token is the max len
        txt_len = min(max_len + 1, self.max_txt_len + img_len)
        inputs_embeds = torch.ones([batch_size, txt_len], dtype=torch.long).to(
            img_embeds.device) * self.mistral_tokenizer.pad_token_id
        if self.use_lora:
            inputs_embeds = self.mistral_model.base_model.model.model.embed_tokens(inputs_embeds)
        else:
            inputs_embeds = self.mistral_model.model.embed_tokens(inputs_embeds)
        attention_mask = torch.zeros([batch_size, txt_len], dtype=torch.long).to(img_embeds.device)
        targets = torch.ones([batch_size, txt_len], dtype=torch.long).to(img_embeds.device).fill_(-100)
        # set bos_token
        bos_ids = torch.full((batch_size, 1), self.mistral_tokenizer.bos_token_id, device=img_embeds.device, dtype=torch.long)
        bos_embeds = emb_fn(bos_ids)
        inputs_embeds[:, :1] = bos_embeds
        for idx in range(batch_size):
            input_len = min(input_embed_list[idx].shape[1], txt_len - 1)
            # if less than txt_len, the input will be padding
            # if more than txt_len, the input will be truncated
            inputs_embeds[idx, 1:(input_len + 1)] = input_embed_list[idx][:, :input_len]
            # the attention_mask is 0 when padding
            attention_mask[idx, :(input_len + 1)] = 1
            # the target is -100 when padding
            p_before_len = p_before_len_list[idx]
            targets[idx, (p_before_len + img_len + 1):(input_len + 1)] = target_list[idx][0,
                                                                         :(input_len - p_before_len - img_len)]

        with self.maybe_autocast():
            outputs = self.mistral_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
                use_cache=False,  # current flash_attn2 dows not support padding=right for mistral
            )

        relation_weight = max(0.5, self.w_ce - 0.09 * epoch)
        total = outputs.loss + relation_weight * loss_ce
        return dict(
            loss=total,
            loss_open_lm=outputs.loss,
            loss_mc_lm=torch.tensor(0.0, device=outputs.loss.device),
            loss_option_ce=torch.tensor(0.0, device=outputs.loss.device),
            loss_relation_ce=loss_ce if isinstance(loss_ce, torch.Tensor) else torch.tensor(float(loss_ce), device=outputs.loss.device),
            loss_answer_verify=torch.tensor(0.0, device=outputs.loss.device),
        )
