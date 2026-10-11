"""
ARENA 1.2 — TransformerLens Exercise 03
ActivationCache basics

Goal:
    Run a HookedTransformer with `run_with_cache` and retrieve specific
    internal activations.

What is already known:
    - residual streams
    - Q/K/V
    - attention patterns
    - parameters vs activations

New skill:
    Using TransformerLens' ActivationCache.

Current TODOs:
    1. Run the model with cache.
    2. Retrieve resid_pre at layer 5.
    3. Retrieve q at layer 5.
    4. Retrieve the attention pattern at layer 5.

Important TransformerLens shapes:
    residual stream:
        (batch, sequence, d_model)

    q / k / v:
        (batch, sequence, n_heads, d_head)

    attention pattern:
        (batch, n_heads, query_position, key_position)

Run:
    python arena_1_2_transformerlens_03_activation_cache.py
"""

from __future__ import annotations

import torch as t
from transformer_lens import HookedTransformer


MODEL_NAME = "gpt2-small"
PROMPT = "The cat sat on the mat. The cat"
LAYER = 5


def get_cached_activations(
    model: HookedTransformer,
    tokens: t.Tensor,
) -> tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]:
    """
    Return:
        logits, resid_pre, q, pattern

    TODO 1:
        Run the model with `run_with_cache`.

    TODO 2:
        Retrieve `resid_pre` for LAYER.

    TODO 3:
        Retrieve `q` for LAYER.

    TODO 4:
        Retrieve `pattern` for LAYER.

    Do not change the tests.
    """

    # TODO 1
    logits, cache = model.run_with_cache(tokens)

    # TODO 2
    resid_pre = cache["resid_pre",LAYER]

    # TODO 3
    q = cache["q",LAYER]

    # TODO 4
    pattern = cache["pattern",LAYER]

    return logits, resid_pre, q, pattern


def run_tests(
    model: HookedTransformer,
    tokens: t.Tensor,
    logits: t.Tensor,
    resid_pre: t.Tensor,
    q: t.Tensor,
    pattern: t.Tensor,
) -> None:
    """Local correctness and parity tests."""

    B, T = tokens.shape
    H = model.cfg.n_heads
    D = model.cfg.d_model
    d_head = model.cfg.d_head
    V = model.cfg.d_vocab

    # logits: (B, T, vocab)
    assert logits.shape == (B, T, V), (
        f"Expected logits shape {(B, T, V)}, got {tuple(logits.shape)}"
    )

    # resid_pre: (B, T, D)
    assert resid_pre.shape == (B, T, D), (
        f"Expected resid_pre shape {(B, T, D)}, "
        f"got {tuple(resid_pre.shape)}"
    )

    # q: (B, T, H, d_head)
    assert q.shape == (B, T, H, d_head), (
        f"Expected q shape {(B, T, H, d_head)}, got {tuple(q.shape)}"
    )

    # pattern: (B, H, query_pos, key_pos)
    assert pattern.shape == (B, H, T, T), (
        f"Expected pattern shape {(B, H, T, T)}, "
        f"got {tuple(pattern.shape)}"
    )

    assert t.isfinite(logits).all()
    assert t.isfinite(resid_pre).all()
    assert t.isfinite(q).all()
    assert t.isfinite(pattern).all()

    # Attention rows sum to 1.
    row_sums = pattern.sum(dim=-1)
    assert t.allclose(row_sums, t.ones_like(row_sums), atol=1e-5), (
        "Attention pattern rows should sum to 1."
    )

    # Causal masking means no future key gets attention.
    future_mask = t.triu(
        t.ones(T, T, dtype=t.bool, device=pattern.device),
        diagonal=1,
    )
    future_attention = pattern.masked_select(
        future_mask.unsqueeze(0).unsqueeze(0)
    )
    assert t.allclose(
        future_attention,
        t.zeros_like(future_attention),
        atol=1e-7,
    ), "Found nonzero attention to a future position."

    # Determinism: same model + same input -> same activations.
    logits2, cache2 = model.run_with_cache(tokens)

    assert t.allclose(logits, logits2, atol=1e-6, rtol=0.0)
    assert t.allclose(
        resid_pre,
        cache2["resid_pre", LAYER],
        atol=1e-6,
        rtol=0.0,
    )
    assert t.allclose(
        q,
        cache2["q", LAYER],
        atol=1e-6,
        rtol=0.0,
    )
    assert t.allclose(
        pattern,
        cache2["pattern", LAYER],
        atol=1e-6,
        rtol=0.0,
    )

    print("LOCAL TESTS PASSED")
    print("ActivationCache retrieval and shape conventions are correct.")


def main() -> None:
    print("=" * 72)
    print("ARENA 1.2 — TransformerLens Exercise 03")
    print("=" * 72)
    print()
    print("Goal:")
    print(f"  Run with cache and inspect layer {LAYER}.")
    print()
    print("Shape conventions:")
    print("  resid_pre : (batch, sequence, d_model)")
    print("  q         : (batch, sequence, heads, d_head)")
    print("  pattern   : (batch, heads, query_position, key_position)")
    print()
    print("Prompt:")
    print(f'  "{PROMPT}"')
    print()

    print(f"Loading {MODEL_NAME}...")
    model = HookedTransformer.from_pretrained(MODEL_NAME)
    model.eval()
    print("Model loaded.")
    print()

    tokens = model.to_tokens(PROMPT)

    logits, resid_pre, q, pattern = get_cached_activations(model, tokens)

    print("Retrieved shapes:")
    print(f"  tokens    : {tuple(tokens.shape)}")
    print(f"  logits    : {tuple(logits.shape)}")
    print(f"  resid_pre : {tuple(resid_pre.shape)}")
    print(f"  q         : {tuple(q.shape)}")
    print(f"  pattern   : {tuple(pattern.shape)}")
    print()

    print("Example:")
    print(f"  pattern[0, 0, -1, :].shape = {tuple(pattern[0, 0, -1, :].shape)}")
    print(
        "  This is head 0's attention distribution over key positions "
        "for the final query position."
    )
    print()

    run_tests(
        model=model,
        tokens=tokens,
        logits=logits,
        resid_pre=resid_pre,
        q=q,
        pattern=pattern,
    )

    print()
    print("Exercise 03 PASSED.")
    print("Next: navigating cache names and inspecting activations.")


if __name__ == "__main__":
    main()
