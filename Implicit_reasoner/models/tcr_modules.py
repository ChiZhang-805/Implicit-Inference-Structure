import re
from typing import Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def mean_pool_text_embeds(token_embeds, attention_mask=None):
    if attention_mask is None:
        return token_embeds.mean(dim=1)
    mask = attention_mask.unsqueeze(-1).to(token_embeds.dtype)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return (token_embeds * mask).sum(dim=1) / denom


class TCRMemorySelector(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        in_dim = hidden_size * 6
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, memory_embeds, question_embed, option_embed, visual_embed, relation_logits=None):
        n, h = memory_embeds.shape
        q = question_embed.view(1, h).expand(n, -1)
        v = visual_embed.view(1, h).expand(n, -1)
        if option_embed is None:
            o = torch.zeros_like(q)
        else:
            o = option_embed.view(1, h).expand(n, -1)
        x = torch.cat([memory_embeds, q, o, v, memory_embeds * q, memory_embeds * v], dim=-1)
        logits = self.mlp(x)
        if relation_logits is not None:
            logits = logits + relation_logits
        probs = torch.softmax(logits, dim=-1)[:, 1]
        return logits, probs


class TCRContextQARetriever:
    def _tok(self, s):
        return set(re.findall(r"[a-z0-9]+", (s or "").lower()))

    def rank(self, question: str, context_lines: List[str], context_embeds: Optional[torch.Tensor] = None, question_embed: Optional[torch.Tensor] = None, topk: int = 2):
        if not context_lines:
            return []
        qtok = self._tok(question)
        lexical_scores = []
        for c in context_lines:
            ctok = self._tok(c)
            lexical_scores.append(len(qtok & ctok) / max(1.0, len(qtok | ctok)))
        scores = torch.tensor(lexical_scores, dtype=torch.float)
        if context_embeds is not None and question_embed is not None and len(context_lines) == context_embeds.shape[0]:
            cos = F.cosine_similarity(context_embeds, question_embed.unsqueeze(0), dim=-1)
            scores = scores + 0.5 * cos.cpu()
        topk = max(1, min(topk, len(context_lines)))
        idx = torch.topk(scores, k=topk).indices.tolist()
        return [context_lines[i] for i in idx]


class TCROptionClassifier(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        in_dim = hidden_size * 5
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, visual_embed, question_embed, option_embeds):
        m = option_embeds.shape[0]
        v = visual_embed.view(1, -1).expand(m, -1)
        q = question_embed.view(1, -1).expand(m, -1)
        x = torch.cat([v, q, option_embeds, q * option_embeds, v * option_embeds], dim=-1)
        return self.mlp(x).squeeze(-1)


class TCRAnswerVerifier(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        in_dim = hidden_size * 5
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, visual_embed, question_embed, candidate_answer_embed):
        x = torch.cat([
            visual_embed,
            question_embed,
            candidate_answer_embed,
            question_embed * candidate_answer_embed,
            visual_embed * candidate_answer_embed,
        ], dim=-1)
        return self.mlp(x).squeeze(-1)
