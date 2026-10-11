"""
ARENA 1.2 — TransformerLens Exercise 10B
Detecting copying behavior from the OV circuit

Correction to Exercise 10:
    The previous exercise used mean diagonal strength as a copying score.
    That is a useful diagnostic, but it is NOT the formal copying statistic
    used in the current ARENA / Transformer Circuits material.

The canonical working statistic is positive eigenvalue mass:

    copying_score = sum(real(eigenvalues)) / sum(abs(eigenvalues))

for the relevant low-rank OV operator.

Why this is computationally feasible:
    The full token-to-token OV matrix is ~50k x 50k, but the head is rank
    <= d_head (=64 in GPT-2 Small). The nonzero eigenvalues of AB and BA are
    identical, so we can compute the relevant eigenvalues using a 64 x 64
    matrix instead of materializing a 50,257 x 50,257 matrix.

Current TODOs:
    1. Construct the 64 x 64 matrix whose eigenvalues match the nonzero
       eigenvalues of the full OV vocabulary map.
    2. Compute the copying score from those eigenvalues.

Run:
    python arena_1_2_transformerlens_10b_copying_eigenvalue_score.py
"""

from __future__ import annotations

import torch as t
from transformer_lens import HookedTransformer


MODEL_NAME = "gpt2-small"
TARGET_LAYER = 7
TARGET_HEAD = 10


# ---------------------------------------------------------------------------
# Core exercise
# ---------------------------------------------------------------------------

def make_small_ov_operator(
    model: HookedTransformer,
    layer: int,
    head: int,
) -> t.Tensor:
    """
    Construct the small matrix whose eigenvalues equal the nonzero
    eigenvalues of the full token-to-token OV map.

    Weight shapes:
        W_E : (d_vocab, d_model)
        W_V : (d_model, d_head)
        W_O : (d_head, d_model)
        W_U : (d_model, d_vocab)

    Full vocabulary map:
        M = W_E @ W_V @ W_O @ W_U

    This is 50,257 x 50,257 and must NOT be materialized.

    Write it as M = A @ B where:
        A = W_E @ W_V       -> (d_vocab, d_head)
        B = W_O @ W_U       -> (d_head, d_vocab)

    Then use the AB/BA eigenvalue identity.

    TODO 1:
        Return the d_head x d_head BA matrix.
    """

    W_E = model.W_E
    W_V = model.W_V[layer, head]
    W_O = model.W_O[layer, head]
    W_U = model.W_U

    # TODO 1
    small_operator = W_O @ W_U @ W_E @ W_V

    return small_operator


def copying_score_from_operator(operator: t.Tensor) -> float:
    """
    Compute the canonical eigenvalue-based copying score.

    For eigenvalues lambda_i:
        score = sum_i Re(lambda_i) / sum_i |lambda_i|

    Score is theoretically in [-1, 1]. Positive values indicate positive
    eigenvalue mass, which is the working formalization of copying.

    TODO 2:
        Compute the eigenvalues and then the score.
    """

    # TODO 2
    num = operator.diagonal().sum()
    ev = t.linalg.eigvals(operator)
    ev = abs(ev)
    den = ev.sum()
    score = num / den
    return float(score)


# ---------------------------------------------------------------------------
# Synthetic tests
# ---------------------------------------------------------------------------

def run_synthetic_tests() -> None:
    """Test the eigenvalue score on matrices with known behavior."""

    # All eigenvalues positive -> score +1.
    positive = t.diag(t.tensor([1.0, 2.0, 3.0]))
    positive_score = copying_score_from_operator(positive)
    assert abs(positive_score - 1.0) < 1e-6

    # All eigenvalues negative -> score -1.
    negative = t.diag(t.tensor([-1.0, -2.0, -3.0]))
    negative_score = copying_score_from_operator(negative)
    assert abs(negative_score + 1.0) < 1e-6

    # Balanced positive / negative mass -> approximately zero.
    mixed = t.diag(t.tensor([1.0, -1.0, 2.0, -2.0]))
    mixed_score = copying_score_from_operator(mixed)
    assert abs(mixed_score) < 1e-6

    print("SYNTHETIC TESTS PASSED")


def run_model_test(model: HookedTransformer) -> None:
    """Inspect the target induction head and nearby heads."""

    print()
    print("TARGET HEAD")
    print("=" * 72)
    print(f"Target: L{TARGET_LAYER}H{TARGET_HEAD}")

    target_operator = make_small_ov_operator(
        model, TARGET_LAYER, TARGET_HEAD
    )

    print(f"Small operator shape: {tuple(target_operator.shape)}")

    target_score = copying_score_from_operator(target_operator)
    print(f"Copying score: {target_score:.4f}")

    assert t.isfinite(target_operator).all()
    assert -1.000001 <= target_score <= 1.000001

    print()
    print("ALL-HEAD SCAN")
    print("=" * 72)

    results: list[tuple[float, int, int]] = []

    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            operator = make_small_ov_operator(model, layer, head)
            score = copying_score_from_operator(operator)
            results.append((score, layer, head))

    results.sort(reverse=True)

    print("Top 10 heads by eigenvalue-based copying score:")
    for rank, (score, layer, head) in enumerate(results[:10], start=1):
        print(f"  {rank:2d}. L{layer}H{head}: {score:.4f}")

    target_rank = next(
        rank
        for rank, (_, layer, head) in enumerate(results, start=1)
        if layer == TARGET_LAYER and head == TARGET_HEAD
    )

    print()
    print(
        f"L{TARGET_LAYER}H{TARGET_HEAD} rank: {target_rank}/"
        f"{len(results)}"
    )

    print()
    print("INTERPRETATION")
    print("=" * 72)
    print("A strongly positive copying score means positive eigenvalue mass")
    print("dominates the OV operator: directions are preferentially mapped")
    print("toward corresponding output directions.")
    print()
    print("This is evidence of copying behavior in the OV circuit, not proof")
    print("that the head performs only copying in every context.")


def main() -> None:
    print("=" * 72)
    print("ARENA 1.2 — TransformerLens Exercise 10B")
    print("Detecting copying from OV eigenvalues")
    print("=" * 72)
    print()
    print("We are correcting the previous copying statistic.")
    print("Mean diagonal strength is descriptive; the formal working score")
    print("here is normalized positive eigenvalue mass.")
    print()

    run_synthetic_tests()

    print()
    print(f"Loading {MODEL_NAME}...")
    model = HookedTransformer.from_pretrained(MODEL_NAME)
    model.eval()
    print("Model loaded.")

    run_model_test(model)

    print()
    print("EXERCISE 10B COMPLETE")
    print("Next: use QK/OV composition to connect previous-token heads to")
    print("induction heads at the weight level.")


if __name__ == "__main__":
    main()
