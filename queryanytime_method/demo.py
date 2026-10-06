from __future__ import annotations

import torch

from .method import QueryAnytimeMethod


def main() -> None:
    method = QueryAnytimeMethod(hidden_dim=64, set_tokens=8)
    method.ingest_frame(
        "08/000001",
        torch.randn(1, 32, 64),
        torch.randn(1, 32, 3),
        torch.zeros(1, 32, dtype=torch.long),
    )
    query = torch.randn(1, 6, 64)
    result = method.search(query)
    print(
        {
            "active_frame_ids": result.active_frame_ids,
            "logit_shape": tuple(result.presence_logits.shape),
        }
    )


if __name__ == "__main__":
    main()
