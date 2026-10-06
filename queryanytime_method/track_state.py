from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class GeometryState:
    center: Tensor
    velocity: Tensor
    extent: Tensor
    orientation: Tensor
    confidence: Tensor
    age: int = 0
    frame_id: str | None = None


@dataclass
class SemanticGeometricTrackState:
    semantic: Tensor
    geometry: GeometryState | None


def _frame_delta(previous: str | None, current: str | None) -> int:
    if previous is None or current is None:
        return 1
    previous_sequence, _, previous_frame = str(previous).rpartition("/")
    current_sequence, _, current_frame = str(current).rpartition("/")
    try:
        if previous_sequence != current_sequence:
            return 1
        return max(1, int(current_frame) - int(previous_frame))
    except ValueError:
        return 1


def predict_geometry(state: GeometryState, frame_id: str | None) -> GeometryState:
    delta = _frame_delta(state.frame_id, frame_id)
    return GeometryState(
        center=state.center + delta * state.velocity,
        velocity=state.velocity,
        extent=state.extent,
        orientation=state.orientation,
        confidence=state.confidence,
        age=state.age + delta,
        frame_id=frame_id,
    )


def fit_geometry(
    coords: Tensor,
    point_scores: Tensor,
    previous: GeometryState | None,
    frame_id: str | None,
    *,
    score_threshold: float = 0.5,
    min_points: int = 2,
    velocity_ema: float = 0.8,
    update_strength: float = 1.0,
) -> GeometryState | None:
    """Fit center, extent, orientation, and velocity from predicted point evidence."""
    scores = point_scores.float().clamp_min(0.0)
    keep = torch.isfinite(scores) & (scores >= score_threshold)
    if int(keep.sum()) < min_points:
        return predict_geometry(previous, frame_id) if previous is not None else None
    selected = coords[keep].float()
    weights = scores[keep].clamp_min(1e-4)
    confidence = weights.mean().clamp(0.0, 1.0).reshape(1)
    weights = weights / weights.sum().clamp_min(1e-6)
    center = (selected * weights[:, None]).sum(dim=0, keepdim=True)
    centered = selected[:, :2] - center[:, :2]
    covariance = (centered * weights[:, None]).T @ centered
    covariance = covariance + torch.eye(2, device=coords.device) * 1e-4
    values, vectors = torch.linalg.eigh(covariance)
    order = torch.argsort(values, descending=True)
    values = values[order].clamp_min(1e-4)
    vectors = vectors[:, order]
    extent = values.sqrt().unsqueeze(0)
    orientation = torch.atan2(vectors[1, 0], vectors[0, 0]).reshape(1)
    if previous is None:
        velocity = torch.zeros_like(center)
        age = 0
    else:
        delta = _frame_delta(previous.frame_id, frame_id)
        predicted = previous.center + delta * previous.velocity
        observed_velocity = (center - previous.center) / float(delta)
        velocity = velocity_ema * previous.velocity + (1.0 - velocity_ema) * observed_velocity
        alpha = torch.as_tensor(update_strength, device=center.device).clamp(0.0, 1.0)
        center = predicted + alpha * (center - predicted)
        velocity = previous.velocity + alpha * (velocity - previous.velocity)
        extent = previous.extent + alpha * (extent - previous.extent)
        angle_delta = torch.atan2(
            torch.sin(orientation - previous.orientation),
            torch.cos(orientation - previous.orientation),
        )
        orientation = previous.orientation + alpha * angle_delta
        confidence = previous.confidence + alpha.reshape(-1) * (confidence - previous.confidence)
        age = 0 if float(alpha.item()) > 0.0 else previous.age + delta
    return GeometryState(center, velocity, extent, orientation, confidence, age, frame_id)


class CausalSemanticMemory(nn.Module):
    """Four-token semantic identity memory with confidence-gated residual update."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        tokens: int = 4,
        update_probability: float = 0.05,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.tokens = tokens
        self.offsets = nn.Parameter(torch.randn(tokens, hidden_dim) * 0.02)
        self.attention = nn.MultiheadAttention(hidden_dim, 8, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim + 2),
            nn.Linear(2 * hidden_dim + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, torch.logit(torch.tensor(update_probability)))

    def forward(
        self,
        decoder_query: Tensor,
        point_tokens: Tensor,
        valid: Tensor,
        previous: Tensor | None,
        mask_confidence: Tensor,
        mask_margin: Tensor,
    ) -> tuple[Tensor, Tensor]:
        seed = self.offsets.unsqueeze(0).expand(decoder_query.shape[0], -1, -1) + decoder_query
        query = seed if previous is None else previous
        retrieved, _ = self.attention(query, point_tokens, point_tokens, key_padding_mask=~valid)
        candidate = self.norm(query + retrieved)
        candidate = candidate + self.ffn(candidate)
        if previous is None:
            return self.norm(candidate), torch.ones(
                decoder_query.shape[0], 1, device=decoder_query.device
            )
        quality = torch.stack(
            [mask_confidence.flatten(), torch.tanh(mask_margin.flatten())], dim=-1
        )
        gate = torch.sigmoid(
            self.gate(torch.cat([query.mean(1), candidate.mean(1), quality], dim=-1))
        )
        return self.norm(query + gate.unsqueeze(-1) * (candidate - query)), gate


class TrackStateUpdater(nn.Module):
    """Combine semantic memory and geometry state for the spatial decoder."""

    def __init__(self, hidden_dim: int, *, velocity_ema: float = 0.8) -> None:
        super().__init__()
        self.semantic_memory = CausalSemanticMemory(hidden_dim)
        self.velocity_ema = velocity_ema
        self.geometry_bias = nn.Sequential(
            nn.Linear(4, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )

    def forward(
        self,
        state: SemanticGeometricTrackState | None,
        decoder_query: Tensor,
        point_tokens: Tensor,
        coords: Tensor,
        point_scores: Tensor,
        mask_confidence: Tensor,
        mask_margin: Tensor,
        frame_id: str | None,
    ) -> tuple[SemanticGeometricTrackState, Tensor]:
        previous_semantic = None if state is None else state.semantic
        semantic, gate = self.semantic_memory(
            decoder_query,
            point_tokens,
            torch.ones(point_tokens.shape[:2], dtype=torch.bool, device=point_tokens.device),
            previous_semantic,
            mask_confidence,
            mask_margin,
        )
        previous_geometry = None if state is None else state.geometry
        geometry = fit_geometry(
            coords,
            point_scores,
            previous_geometry,
            frame_id,
            velocity_ema=self.velocity_ema,
            update_strength=float(gate.mean().detach().item()),
        )
        bias = self.geometry_attention_bias(coords, geometry)
        return SemanticGeometricTrackState(semantic, geometry), bias

    def geometry_attention_bias(self, coords: Tensor, state: GeometryState | None) -> Tensor:
        if state is None:
            return coords.new_zeros(coords.shape[:-1])
        delta = coords[..., :2] - state.center[..., :2].unsqueeze(-2)
        cosine = torch.cos(state.orientation).unsqueeze(-1)
        sine = torch.sin(state.orientation).unsqueeze(-1)
        local = torch.stack(
            [cosine * delta[..., 0] + sine * delta[..., 1],
             -sine * delta[..., 0] + cosine * delta[..., 1]],
            dim=-1,
        )
        local = local / state.extent.unsqueeze(-2).clamp_min(1e-4)
        confidence = state.confidence.unsqueeze(-1).unsqueeze(-1).expand_as(local[..., :1])
        features = torch.cat(
            [local, local.norm(dim=-1, keepdim=True), confidence],
            dim=-1,
        )
        return self.geometry_bias(features).squeeze(-1) * state.confidence.unsqueeze(-1)
