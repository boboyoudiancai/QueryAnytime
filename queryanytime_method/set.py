from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class SETOutput:
    logits: Tensor
    token_evidence: Tensor
    query_weights: Tensor
    top_indices: Tensor


class QueryIndependentSET(nn.Module):
    """Compress one dense frame into fixed-capacity, query-independent tokens."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        num_tokens: int = 32,
        num_layers: int = 2,
        num_heads: int = 8,
        coordinate_scale: float = 50.0,
    ) -> None:
        super().__init__()
        if num_tokens <= 0 or num_layers <= 0:
            raise ValueError("SET token and layer counts must be positive.")
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens
        self.coordinate_scale = coordinate_scale
        self.seeds = nn.Parameter(torch.randn(1, num_tokens, hidden_dim) * 0.02)
        self.coordinate_proj = nn.Sequential(
            nn.Linear(3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.modality = nn.Embedding(2, hidden_dim)
        self.attention = nn.ModuleList(
            [
                nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        visual: Tensor,
        coords: Tensor,
        modality: Tensor,
        valid: Tensor | None = None,
    ) -> Tensor:
        if visual.ndim != 3 or coords.shape != (*visual.shape[:2], 3):
            raise ValueError("visual must be [B,N,D] and coords must be [B,N,3].")
        if modality.shape != visual.shape[:2]:
            raise ValueError("modality must be [B,N].")
        if valid is None:
            valid = torch.ones(visual.shape[:2], dtype=torch.bool, device=visual.device)
        if valid.shape != visual.shape[:2]:
            raise ValueError("valid must be [B,N].")
        context = visual
        context = context + self.coordinate_proj(torch.tanh(coords / self.coordinate_scale))
        context = context + self.modality(modality.long())
        context = context * valid.unsqueeze(-1).to(context.dtype)
        empty = ~valid.any(dim=1)
        if empty.any():
            context = context.clone()
            context[empty, 0] = 0
            valid = valid.clone()
            valid[empty, 0] = True
        tokens = self.seeds.expand(visual.shape[0], -1, -1)
        for layer in self.attention:
            tokens, _ = layer(tokens, context, context, key_padding_mask=~valid)
        return self.norm(tokens)


class SETActivationHead(nn.Module):
    """Late-interaction temporal presence classifier over cached SET tokens."""

    def __init__(self, hidden_dim: int, *, topk: int = 4, temperature: float = 0.1) -> None:
        super().__init__()
        if topk <= 0 or temperature <= 0:
            raise ValueError("topk and temperature must be positive.")
        self.topk = topk
        self.temperature = temperature
        self.query_proj = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim))
        self.set_proj = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim))
        self.token_gate = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
        self.logit_scale = nn.Parameter(torch.tensor(2.3025851))
        self.logit_bias = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        text_tokens: Tensor,
        set_tokens: Tensor,
        text_valid: Tensor | None = None,
        set_valid: Tensor | None = None,
    ) -> SETOutput:
        if text_tokens.ndim == 2:
            text_tokens = text_tokens.unsqueeze(1)
        if text_tokens.ndim != 3 or set_tokens.ndim != 3:
            raise ValueError("text_tokens and set_tokens must be [B,L,D] and [B,K,D].")
        if text_tokens.shape[0] != set_tokens.shape[0]:
            raise ValueError("Text and SET batches must match.")
        slots = set_tokens.shape[1]
        text_valid = text_valid if text_valid is not None else torch.ones(
            text_tokens.shape[:2], dtype=torch.bool, device=text_tokens.device
        )
        set_valid = set_valid if set_valid is not None else torch.ones(
            set_tokens.shape[:2], dtype=torch.bool, device=set_tokens.device
        )
        query = F.normalize(self.query_proj(text_tokens).float(), dim=-1)
        slots_projected = F.normalize(self.set_proj(set_tokens).float(), dim=-1)
        similarity = torch.einsum("bld,bkd->blk", query, slots_projected)
        similarity = similarity.masked_fill(
            ~set_valid.unsqueeze(1), torch.finfo(similarity.dtype).min
        )
        k = min(self.topk, slots)
        top_values, top_indices = torch.topk(similarity, k=k, dim=-1)
        evidence = self.temperature * torch.logsumexp(
            top_values / self.temperature, dim=-1
        ) - self.temperature * torch.log(torch.tensor(float(k), device=top_values.device))
        gate = self.token_gate(text_tokens.float()).squeeze(-1)
        gate = gate.masked_fill(~text_valid, torch.finfo(gate.dtype).min)
        weights = torch.softmax(gate, dim=-1)
        pooled = (weights * evidence).sum(dim=-1)
        scale = self.logit_scale.float().clamp(max=4.6052).exp()
        logits = scale * pooled + self.logit_bias.float()
        token_evidence = self._readout_evidence(query, slots_projected, set_valid)
        return SETOutput(logits, token_evidence, weights, top_indices)

    @staticmethod
    def _readout_evidence(query: Tensor, slots: Tensor, valid: Tensor) -> Tensor:
        similarity = torch.einsum("bld,bkd->blk", query, slots)
        similarity = similarity.masked_fill(~valid.unsqueeze(1), torch.finfo(similarity.dtype).min)
        weights = torch.softmax(similarity, dim=-1)
        weights = weights * valid.unsqueeze(1).to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return torch.einsum("blk,bkd->bld", weights, slots)


def set_presence_loss(logits: Tensor, labels: Tensor, valid: Tensor | None = None) -> Tensor:
    if valid is None:
        valid = torch.ones_like(labels, dtype=torch.bool)
    if logits.shape != labels.shape or valid.shape != labels.shape:
        raise ValueError("SET logits, labels, and valid mask must have the same shape.")
    return F.binary_cross_entropy_with_logits(logits[valid], labels.float()[valid])


def dense_evidence_kd_loss(
    teacher_evidence: Tensor,
    student_evidence: Tensor,
    valid_text: Tensor | None = None,
) -> Tensor:
    if teacher_evidence.shape != student_evidence.shape:
        raise ValueError("Teacher and student evidence must have the same shape.")
    cosine = (F.normalize(student_evidence.float(), dim=-1) * F.normalize(
        teacher_evidence.float().detach(), dim=-1
    )).sum(dim=-1)
    if valid_text is not None:
        cosine = cosine[valid_text]
    return (1.0 - cosine).mean() if cosine.numel() else teacher_evidence.sum() * 0.0
