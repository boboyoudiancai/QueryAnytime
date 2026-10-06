import torch

from queryanytime_method import QueryAnytimeMethod, QueryIndependentSET


def test_set_is_query_independent():
    module = QueryIndependentSET(hidden_dim=16, num_tokens=4)
    visual = torch.randn(1, 10, 16)
    coords = torch.randn(1, 10, 3)
    modality = torch.zeros(1, 10, dtype=torch.long)
    first = module(visual, coords, modality)
    second = module(visual, coords, modality)
    assert first.shape == (1, 4, 16)
    assert torch.equal(first, second)


def test_method_search_uses_cached_frames():
    method = QueryAnytimeMethod(hidden_dim=16, num_set_tokens=4)
    method.ingest_frame(
        "08/000002",
        torch.randn(1, 10, 16),
        torch.randn(1, 10, 3),
        torch.zeros(1, 10, dtype=torch.long),
    )
    result = method.search(torch.randn(1, 5, 16))
    assert result.frame_ids == ["08/000002"]
    assert result.presence_logits.shape == (1,)
