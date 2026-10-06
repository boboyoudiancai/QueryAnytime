from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .set import SETActivationHead, QueryIndependentSET
from .track_state import SemanticGeometricTrackState, TrackStateUpdater


@dataclass
class CachedFrame:
    frame_id: str
    dense_tokens: Tensor
    coords: Tensor
    set_tokens: Tensor


@dataclass
class SearchResult:
    active_frame_ids: list[str]
    presence_logits: Tensor
    frame_ids: list[str]


@dataclass
class SpatialConditioning:
    dense_tokens: Tensor
    semantic_state: Tensor
    geometry_bias: Tensor
    track_state: SemanticGeometricTrackState


class QueryAnytimeMethod(nn.Module):
    """Small composition of the paper's SET and TrackState mechanisms."""

    def __init__(self, hidden_dim: int, *, num_set_tokens: int = 32, topk: int = 4) -> None:
        super().__init__()
        self.set_tokenizer = QueryIndependentSET(hidden_dim, num_tokens=num_set_tokens)
        self.activation = SETActivationHead(hidden_dim, topk=topk)
        self.track_state = TrackStateUpdater(hidden_dim)
        self.history: list[CachedFrame] = []

    @torch.no_grad()
    def ingest_frame(
        self,
        frame_id: str,
        dense_tokens: Tensor,
        coords: Tensor,
        modality: Tensor,
        valid: Tensor | None = None,
    ) -> None:
        """Encode one physical frame once and retain dense values plus SET tokens."""
        set_tokens = self.set_tokenizer(dense_tokens, coords, modality, valid)
        self.history.append(CachedFrame(frame_id, dense_tokens, coords, set_tokens))

    def search(self, query_tokens: Tensor, *, threshold: float = 0.0) -> SearchResult:
        if not self.history:
            raise RuntimeError("Cannot search before ingesting at least one frame.")
        set_tokens = torch.cat([item.set_tokens for item in self.history], dim=0)
        query_batch = query_tokens.expand(len(self.history), -1, -1)
        result = self.activation(query_batch, set_tokens)
        logits = result.logits.reshape(-1)
        active = [
            frame.frame_id
            for frame, logit in zip(self.history, logits)
            if float(logit.detach()) > threshold
        ]
        return SearchResult(active, logits, [item.frame_id for item in self.history])

    def condition_spatial_decoder(
        self,
        query_tokens: Tensor,
        frame_index: int,
        state: SemanticGeometricTrackState | None,
        decoder_query: Tensor,
        point_scores: Tensor,
        mask_confidence: Tensor,
        mask_margin: Tensor,
    ) -> SpatialConditioning:
        """Update TrackState and return the conditioning consumed by a mask decoder."""
        if not 0 <= frame_index < len(self.history):
            raise IndexError("frame_index is outside the ingested history.")
        frame = self.history[frame_index]
        next_state, geometry_bias = self.track_state(
            state,
            decoder_query,
            frame.dense_tokens,
            frame.coords,
            point_scores,
            mask_confidence,
            mask_margin,
            frame.frame_id,
        )
        return SpatialConditioning(
            dense_tokens=frame.dense_tokens,
            semantic_state=next_state.semantic,
            geometry_bias=geometry_bias,
            track_state=next_state,
        )

    def clear(self) -> None:
        self.history.clear()
