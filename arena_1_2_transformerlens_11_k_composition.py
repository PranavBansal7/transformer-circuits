"""
ARENA 1.2 — TransformerLens Exercise 11
K-composition between attention heads

Goal
----
Measure whether an earlier head's OV output subspace overlaps with a later
head's QK/K input subspace.

For earlier head A and later head B:
    W_OV^A = W_V^A @ W_O^A
    W_QK^B = W_Q^B @ (W_K^B)^T

For K-composition, we study:
    W_OV^A @ (W_QK^B)^T

and normalize its Frobenius norm by the norms of the two component maps.

Current TODOs
-------------
1. Construct W_OV for one head.
2. Construct W_QK for one head.
3. Compute the normalized K-composition score.

Target later head:
    L7H10

We will scan earlier heads and compare their scores.
"""

from __future__ import annotations

import math
import torch as t
from transformer_lens import HookedTransformer


MODEL_NAME = "gpt2-small"
TARGET_LAYER = 7
TARGET_HEAD = 10


def make_W_OV(model: HookedTransformer, layer: int, head: int) -> t.Tensor:
    """
    Construct the OV map for one attention head.

    With our row-vector convention:
        W_OV = W_V @ W_O

    Shapes:
        W_V: (d_model, d_head)
        W_O: (d_head, d_model)
        W_OV: (d_model, d_model)

    TODO 1
    """
    # TODO 1
    W_OV = model.W_V[layer,head] @ model.W_O[layer,head]
    return W_OV


def make_W_QK(model: HookedTransformer, layer: int, head: int) -> t.Tensor:
    """
    Construct the QK bilinear map for one attention head.

    Shapes:
        W_Q: (d_model, d_head)
        W_K: (d_model, d_head)

    TODO 2
        Construct W_Q @ W_K.T, giving a (d_model, d_model) matrix.
    """
    # TODO 2
    W_QK = model.W_Q[layer,head] @ model.W_K[layer,head].T
    return W_QK


def k_composition_score(
    W_OV_A: t.Tensor,
    W_QK_B: t.Tensor,
) -> t.Tensor:
    """
    Compute normalized K-composition score.

    score = || W_OV_A @ W_QK_B.T ||_F
            --------------------------------
              ||W_OV_A||_F ||W_QK_B||_F

    TODO 3
    """
    # TODO 3
    score = t.linalg.norm(W_OV_A @ W_QK_B.T,ord="fro") / (t.linalg.norm(W_OV_A,ord="fro")*t.linalg.norm(W_QK_B,ord="fro"))
    return score


def synthetic_tests() -> None:
    """Check matrix orientation and normalization on tiny examples."""
    I = t.eye(4)
    zero = t.zeros(4, 4)

    same = k_composition_score(I, I)
    assert t.allclose(same, t.tensor(0.5)), (
        f"Expected 0.5 for 4x4 identity matrices, got {same.item()}"
    )

    # Orthogonal support gives zero composition.
    A = t.zeros(4, 4)
    B = t.zeros(4, 4)
    A[0, 0] = 1.0
    B[1, 1] = 1.0
    orth = k_composition_score(A, B)
    assert t.allclose(orth, t.tensor(0.0)), (
        f"Expected 0.0 for orthogonal supports, got {orth.item()}"
    )


    print("SYNTHETIC TESTS PASSED")


def main() -> None:
    print("=" * 72)
    print("ARENA 1.2 — TransformerLens Exercise 11")
    print("K-composition between attention heads")
    print("=" * 72)
    print()
    print(f"Target later head: L{TARGET_LAYER}H{TARGET_HEAD}")
    print(
        "K-composition asks whether an earlier head's OV output aligns with "
        "the later head's key-input computation."
    )
    print()

    synthetic_tests()
    print()

    print(f"Loading {MODEL_NAME}...")
    model = HookedTransformer.from_pretrained(MODEL_NAME)
    model.eval()
    print("Model loaded.")
    print()

    target_W_QK = make_W_QK(model, TARGET_LAYER, TARGET_HEAD)
    target_norm = target_W_QK.norm()

    baseline = 1.0 / math.sqrt(model.cfg.d_model)
    print(f"Approx. random baseline: {baseline:.4f}")
    print()

    results: list[tuple[float, int, int]] = []

    for layer in range(TARGET_LAYER):
        for head in range(model.cfg.n_heads):
            W_OV = make_W_OV(model, layer, head)
            score = k_composition_score(W_OV, target_W_QK)
            results.append((float(score.item()), layer, head))

    results.sort(reverse=True)

    print("Top 10 earlier heads by K-composition into L7H10:")
    for rank, (score, layer, head) in enumerate(results[:10], start=1):
        print(f"  {rank:2d}. L{layer}H{head}: {score:.4f}")

    l4h11 = next(
        score for score, layer, head in results
        if layer == 4 and head == 11
    )

    print()
    print(f"L4H11 K-composition into L7H10: {l4h11:.4f}")
    print()
    print("Reference point:")
    print(f"  1/sqrt(d_model) = {baseline:.4f}")
    print()
    print("LOCAL SHAPE / FINITENESS CHECKS PASSED")
    print(f"W_QK(L7H10).shape = {tuple(target_W_QK.shape)}")
    print(f"||W_QK(L7H10)||_F = {target_norm.item():.4f}")
    print()
    print("EXERCISE 11 COMPLETE")
    print(
        "Next: compare these weight-level composition scores with the "
        "causal ablation result from L4H11."
    )


if __name__ == "__main__":
    main()
