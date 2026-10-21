"""Default compression is preserved; inference-skipping requires explicit opt-in."""

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
    router = ContentRouter(ContentRouterConfig(skip_redundant_ml=True))
    assert router._should_layer_ml(original, residual) is expected


def test_operator_can_restore_upstream_chaining(monkeypatch):
    router = ContentRouter(ContentRouterConfig(skip_redundant_ml=True))
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
    router = ContentRouter(ContentRouterConfig(lossless_then_lossy=True, skip_redundant_ml=True))
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
    router = ContentRouter(ContentRouterConfig(lossless_then_lossy=True, skip_redundant_ml=True))
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


def test_default_keeps_neural_chaining_even_with_environment_opt_in(monkeypatch):
    monkeypatch.setenv("HEADROOM_SKIP_REDUNDANT_ML", "1")
    router = ContentRouter(ContentRouterConfig())
    assert not router.config.skip_redundant_ml
    assert router._should_layer_ml(1000, 10)


def test_default_lossless_fold_runs_neural_second_pass(monkeypatch):
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
        return "do_not_retry(order_id)", 3

    monkeypatch.setattr(router, "_try_ml_compressor", neural)
    result, _, _ = router._apply_strategy_to_content(content, CompressionStrategy.SEARCH, "")
    assert len(calls) == 1
    assert search_unheading(calls[0]) == content
    assert result == "do_not_retry(order_id)"


@pytest.mark.parametrize("residual_words", [100, 450])
def test_default_structural_log_runs_neural_second_pass(monkeypatch, residual_words):
    original = " ".join(f"original{i}" for i in range(1000))
    residual = " ".join(f"remaining{i}" for i in range(residual_words))
    router = ContentRouter(ContentRouterConfig(lossless_then_lossy=True))
    monkeypatch.setattr(router, "_lossless_first", lambda text, strategy: (text, None))
    monkeypatch.setattr(router, "_get_log_compressor", lambda: object())
    monkeypatch.setattr(router, "_registry_compress_content", lambda *args: residual)
    calls = []

    def neural(text, *_args):
        calls.append(text)
        return "important result", 2

    monkeypatch.setattr(router, "_try_ml_compressor", neural)
    result, _, _ = router._apply_strategy_to_content(original, CompressionStrategy.LOG, "")
    assert calls == [residual]
    assert result == "important result"
