"""
ARENA 1.2 — Exercise 14
Canonical L0H7 QK previous-token circuit

Exact curriculum target:
    Model: callummcdougall/attn_only_2L_half
    Head:  layer 0, head 7

Goal:
    Reverse-engineer why L0H7 is a previous-token head using its WEIGHT-LEVEL
    QK circuit and the model's learned positional embeddings.

The relevant matrix is:
    W_pos @ W_QK @ W_pos.T

where:
    W_QK = W_Q @ W_K.T

This gives a [n_ctx, n_ctx] matrix of position-to-position attention scores
before causal masking, scaling, and softmax.

Current TODOs:
    1. Construct W_QK for L0H7.
    2. Construct the position-by-position score matrix.
    3. Turn the scores into a causal attention pattern (mask + scale + softmax).
    4. Compute the mean attention on the previous-token diagonal.

Run:
    python arena_1_2_transformerlens_14_qk_prev_token_circuit.py
"""

from __future__ import annotations

import torch as t
from huggingface_hub import hf_hub_download
from transformer_lens import HookedTransformer, HookedTransformerConfig
import math


REPO_ID = "callummcdougall/attn_only_2L_half"
WEIGHTS_FILE = "attn_only_2L_half.pth"
LAYER = 0
HEAD = 7


def load_canonical_model() -> HookedTransformer:
    """Build and load the exact ARENA toy model."""
    cfg = HookedTransformerConfig(
        n_layers=2,
        d_model=768,
        n_ctx=2048,
        d_head=64,
        model_name="attn_only_2L_half",
        n_heads=12,
        d_vocab=50278,
        act_fn=None,
        normalization_type=None,
        positional_embedding_type="shortformer",
        attention_dir="causal",
        attn_only=True,
        device="cpu",
    )

    model = HookedTransformer(cfg)
    weights_path = hf_hub_download(repo_id=REPO_ID, filename=WEIGHTS_FILE)
    state_dict = t.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def build_qk_pattern(model: HookedTransformer) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    """
    Return W_QK, raw position-by-position scores, and normalized pattern.

    TODO 1:
        Construct W_QK for L0H7.

    TODO 2:
        Construct pos_by_pos_scores using W_pos @ W_QK @ W_pos.T.

    TODO 3:
        Apply causal masking, divide by sqrt(d_head), and softmax over keys.
    """

    W_QK = model.W_Q[LAYER,HEAD] @ model.W_K[LAYER,HEAD].T
    pos_by_pos_scores = model.W_pos @ W_QK @ model.W_pos.T
    mask = t.tril(t.ones_like(pos_by_pos_scores,dtype=t.bool))
    pos_by_pos_pattern = pos_by_pos_scores.masked_fill(~mask,float('-inf'))
    pos_by_pos_pattern = pos_by_pos_pattern/math.sqrt(model.cfg.d_head)
    pos_by_pos_pattern = t.softmax(pos_by_pos_pattern,dim=-1)
    return W_QK, pos_by_pos_scores, pos_by_pos_pattern


def previous_token_score(pattern: t.Tensor) -> t.Tensor:
    """
    Return the mean attention weight on diag(-1).

    Input shape:
        (n_ctx, n_ctx)

    Output:
        scalar tensor
    """

    # TODO 4
    score = pattern.diagonal(-1).mean()
    return score


def run_tests(
    model: HookedTransformer,
    W_QK: t.Tensor,
    pos_by_pos_scores: t.Tensor,
    pos_by_pos_pattern: t.Tensor,
    prev_score: t.Tensor,
) -> None:
    """Structural and behavioral checks."""

    n_ctx = model.cfg.n_ctx
    d_model = model.cfg.d_model
    d_head = model.cfg.d_head

    assert W_QK.shape == (d_model, d_model)
    assert pos_by_pos_scores.shape == (n_ctx, n_ctx)
    assert pos_by_pos_pattern.shape == (n_ctx, n_ctx)

    assert t.isfinite(W_QK).all()
    assert t.isfinite(pos_by_pos_scores).all()
    assert t.isfinite(pos_by_pos_pattern).all()
    assert t.isfinite(prev_score)

    # Causal attention: future key positions receive zero probability.
    future_mask = t.triu(
        t.ones((n_ctx, n_ctx), dtype=t.bool), diagonal=1
    )
    future_values = pos_by_pos_pattern[future_mask]
    assert t.allclose(future_values, t.zeros_like(future_values), atol=1e-7)

    # Every query-position row is a probability distribution.
    row_sums = pos_by_pos_pattern.sum(dim=-1)
    assert t.allclose(row_sums, t.ones_like(row_sums), atol=1e-5)

    # Previous-token score should be positive and meaningfully larger than
    # an approximately uniform causal baseline for this trained head.
    uniform_causal_baseline = 1.0 / n_ctx
    assert prev_score.item() > 10 * uniform_causal_baseline, (
        f"Previous-token score is unexpectedly weak: {prev_score.item():.6f}"
    )

    # Parity spot-check against the direct QK weights.
    direct_q = model.W_pos @ model.W_Q[LAYER, HEAD]
    direct_k = model.W_pos @ model.W_K[LAYER, HEAD]
    direct_scores = direct_q @ direct_k.T
    max_error = (pos_by_pos_scores - direct_scores).abs().max().item()
    assert max_error < 1e-3, (
        f"Associative QK computation differs by {max_error:.3e}; "
        "this is larger than expected floating-point error."
    )

    print(f"QK associativity max error: {max_error:.3e}")
    print("LOCAL TESTS PASSED")
    print("QK construction, masking, normalization, and score are valid.")


def main() -> None:
    print("=" * 72)
    print("ARENA 1.2 — Exercise 14")
    print("Canonical L0H7 QK previous-token circuit")
    print("=" * 72)
    print()
    print("Exact model: callummcdougall/attn_only_2L_half")
    print("Target head: L0H7")
    print()
    print("We are computing:")
    print("  W_QK = W_Q @ W_K.T")
    print("  pos_by_pos = W_pos @ W_QK @ W_pos.T")
    print("  pattern = causal_softmax(pos_by_pos / sqrt(d_head))")
    print()

    print("Loading canonical ARENA checkpoint...")
    model = load_canonical_model()
    print("Model loaded.")
    print()

    W_QK, pos_by_pos_scores, pos_by_pos_pattern = build_qk_pattern(model)
    prev_score = previous_token_score(pos_by_pos_pattern)

    print("MODEL CONFIG")
    print("-" * 72)
    print(f"n_layers = {model.cfg.n_layers}")
    print(f"n_heads  = {model.cfg.n_heads}")
    print(f"d_model  = {model.cfg.d_model}")
    print(f"d_head   = {model.cfg.d_head}")
    print(f"n_ctx    = {model.cfg.n_ctx}")
    print()

    print("QK CIRCUIT")
    print("-" * 72)
    print(f"W_QK shape           = {tuple(W_QK.shape)}")
    print(f"pos_by_pos shape     = {tuple(pos_by_pos_scores.shape)}")
    print(f"pattern shape        = {tuple(pos_by_pos_pattern.shape)}")
    print(f"Mean diag(-1) score  = {prev_score.item():.6f}")
    print()

    # Inspect a few query positions. We expect a strong preference for q-1.
    print("SELECTED QUERY POSITIONS")
    print("-" * 72)
    for q in [1, 5, 10, 50, 100]:
        row = pos_by_pos_pattern[q]
        k = int(row.argmax())
        print(
            f"query {q:4d} -> top key {k:4d} "
            f"(expected previous key {q-1:4d}), "
            f"attention={row[k].item():.4f}"
        )

    print()
    run_tests(
        model=model,
        W_QK=W_QK,
        pos_by_pos_scores=pos_by_pos_scores,
        pos_by_pos_pattern=pos_by_pos_pattern,
        prev_score=prev_score,
    )

    print()
    print("EXERCISE 14 PASSED")
    print("Next: K-composition between L0H7's OV circuit and L1H4's QK circuit.")


if __name__ == "__main__":
    main()
