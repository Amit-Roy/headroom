"""The CPU policy preserves structural output while avoiding redundant inference."""

import pytest

from headroom.transforms.content_router import (
    CompressionStrategy,
    ContentRouter,
    ContentRouterConfig,
    _estimate_tokens,
)
from headroom.transforms.lossless_compaction import search_unheading


@pytest.mark.parametrize(
    ("original", "residual", "expected"),
    [
        (1000, 100, False),
        (2000, 900, False),
        (1000, 700, True),
        (1000, 256, False),
        (400, 256, True),
        (0, 0, False),
    ],
)
def test_admission_balances_residual_size_and_existing_savings(original, residual, expected):
    router = ContentRouter(ContentRouterConfig())
    assert router._should_layer_ml(original, residual) is expected


def test_operator_can_restore_upstream_chaining(monkeypatch):
    router = ContentRouter(ContentRouterConfig())
    monkeypatch.setenv("HEADROOM_SKIP_REDUNDANT_ML", "0")
    assert router._should_layer_ml(1000, 10)
    monkeypatch.delenv("HEADROOM_SKIP_REDUNDANT_ML")
    router.config.skip_redundant_ml = False
    assert router._should_layer_ml(1000, 10)


def test_lossless_fold_skips_inference_and_preserves_exact_recovery(monkeypatch):
    content = (
        "\n".join(
            f"src/services/payments/processing/settlement.py:{i}:    do_not_retry(order_id)"
            for i in range(1, 100)
        )
        + "\n"
    )
    router = ContentRouter(ContentRouterConfig(lossless_then_lossy=True))
    calls = []

    def neural(text, *_args):
        calls.append(text)
        return "unexpected neural output", 3

    monkeypatch.setattr(router, "_try_ml_compressor", neural)
    folded, tokens, strategies = router._apply_strategy_to_content(
        content, CompressionStrategy.SEARCH, ""
    )
    assert not calls
    assert search_unheading(folded) == content
    assert tokens < _estimate_tokens(content)
    assert CompressionStrategy.KOMPRESS.value not in strategies


@pytest.mark.parametrize("residual_words,should_call", [(100, False), (450, False), (700, True)])
def test_structural_log_result_only_gets_useful_neural_passes(
    monkeypatch, residual_words, should_call
):
    original = " ".join(f"original{i}" for i in range(1000))
    residual = " ".join(f"remaining{i}" for i in range(residual_words))
    router = ContentRouter(ContentRouterConfig(lossless_then_lossy=True))
    monkeypatch.setattr(router, "_lossless_first", lambda text, strategy: (text, None))
    monkeypatch.setattr(router, "_get_log_compressor", lambda: object())
    monkeypatch.setattr(router, "_registry_compress_content", lambda *args: residual)
    calls = []

    def neural(text, *_args):
        calls.append(text)
        return text, _estimate_tokens(text)

    monkeypatch.setattr(router, "_try_ml_compressor", neural)
    result, _, _ = router._apply_strategy_to_content(original, CompressionStrategy.LOG, "")
    assert bool(calls) is should_call
    assert result == residual
