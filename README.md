# QueryAnytime Method Reference

This repository is a small, method-only reference implementation of
**QueryAnytime: Find When Before Where for Online 4D Point Cloud Segmentation**.
It exposes the two paper contributions without dataset adapters, Slurm launchers,
feature caches, checkpoint migration, or production training infrastructure.

```text
incoming dense visual tokens
    -> query-independent SET tokens
    -> query-time SET retrieval: when
    -> query-conditioned spatial decoder inputs: where
    -> semantic-geometric TrackState propagation
```

## Components

`queryanytime_method/set.py` contains:

- `QueryIndependentSET`: fixed-capacity tokenization with coordinate and modality
  embeddings followed by seed-to-visual cross-attention.
- `SETActivationHead`: text-token to SET-token cosine retrieval, SmoothTopK
  aggregation, query-token weighting, and presence logits.
- `dense_evidence_kd_loss`: cosine distillation from dense query-conditioned
  evidence to compact SET evidence.

`queryanytime_method/track_state.py` contains:

- four-token causal semantic memory with confidence/margin-gated residual update;
- pose-transformed geometry state with center, velocity, extent, orientation, and
  age-decayed confidence;
- orientation-aware local-coordinate geometry bias;
- constant-velocity propagation for missed activations.

`queryanytime_method/method.py` composes the two stages. It encodes each frame
once, stores only the compact SET tokens for temporal search, selects activated
frames after query arrival, and returns the dense features plus TrackState
conditioning needed by a spatial decoder.

## Minimal example

```python
import torch

from queryanytime_method import QueryAnytimeMethod

method = QueryAnytimeMethod(hidden_dim=256, set_tokens=32)
visual = torch.randn(1, 512, 256)
coords = torch.randn(1, 512, 3)
modality = torch.zeros(1, 512, dtype=torch.long)
method.ingest_frame("08/000001", visual, coords, modality)

query_tokens = torch.randn(1, 12, 256)
result = method.search(query_tokens, threshold=0.0)
print(result.active_frame_ids)
```

This package deliberately does not prescribe a dataset format or a mask decoder.
The returned `SpatialConditioning` object is the interface between the method
and an application-specific spatial segmentation head.

## Training losses

The reference exposes the method-level objectives only:

- frame-presence BCE over valid query-frame pairs;
- optional SET token diversity regularization;
- dense teacher/student cosine evidence distillation on target-present pairs.

Stage scheduling, data sampling, checkpoint loading, and distributed execution
belong to the surrounding experiment code rather than this reference package.

## License

The authors should add the project license and third-party model/data notices
before public release.
