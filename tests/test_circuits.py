"""
Unit and Synthetic Regression Tests for Transformer Circuits Pipeline.
Verifies mathematical soundness and numerical precision of:
1. Low-rank K-composition Gram matrix computations
2. Bilinear pre-softmax QK component decomposition
3. Targeted activation patching hooks
4. Low-rank OV operator eigenvalue-based copying score
"""

import math
import torch as t
import pytest

from arena_1_2_transformerlens_10b_copying_eigenvalue_score import (
    copying_score_from_operator,
)
from arena_1_2_transformerlens_18_targeted_ablation import (
    value_ablation_hook,
    induction_pattern_score_hook,
)
from arena_1_2_transformerlens_19_qk_component_decomposition import (
    pairwise_qk_score_contributions,
    sum_pairwise_scores,
)
from arena_1_2_transformerlens_20_canonical_k_composition import (
    k_composition_score,
    direct_k_composition_score,
)


def test_copying_eigenvalue_extremes():
    """Verify that pure positive, pure negative, and balanced operators yield exact scores."""
    # All positive eigenvalues -> +1.0
    pos_mat = t.diag(t.tensor([1.0, 2.0, 3.0]))
    assert abs(copying_score_from_operator(pos_mat) - 1.0) < 1e-6

    # All negative eigenvalues -> -1.0
    neg_mat = t.diag(t.tensor([-1.0, -2.0, -3.0]))
    assert abs(copying_score_from_operator(neg_mat) - (-1.0)) < 1e-6

    # Balanced -> ~0.0
    balanced_mat = t.diag(t.tensor([1.0, -1.0, 2.0, -2.0]))
    assert abs(copying_score_from_operator(balanced_mat)) < 1e-6


def test_k_composition_low_rank_vs_direct():
    """Verify low-rank Gram matrix K-composition matches direct Frobenius norm within 1e-8."""
    t.manual_seed(2026)
    d_model, d_head = 16, 4
    V = t.randn(d_model, d_head, dtype=t.float64)
    O = t.randn(d_head, d_model, dtype=t.float64)
    Q = t.randn(d_model, d_head, dtype=t.float64)
    K = t.randn(d_model, d_head, dtype=t.float64)

    fast_score = k_composition_score(V, O, Q, K)
    direct_score = direct_k_composition_score(V, O, Q, K)

    assert fast_score.ndim == 0
    assert t.isfinite(fast_score)
    assert t.allclose(fast_score, direct_score, atol=1e-8, rtol=1e-7)


def test_targeted_value_ablation():
    """Verify that zeroing head h leaves all other heads bitwise identical."""
    t.manual_seed(42)
    v = t.randn(2, 7, 12, 64)
    original = v.clone()
    ablated = value_ablation_hook(v.clone(), None, head_index_to_ablate=7)

    assert ablated.shape == original.shape
    assert t.count_nonzero(ablated[:, :, 7, :]) == 0
    keep_mask = t.ones(12, dtype=t.bool)
    keep_mask[7] = False
    assert t.equal(ablated[:, :, keep_mask, :], original[:, :, keep_mask, :])


def test_induction_stripe_diagonal_indexing():
    """Verify the offset -(S - 1) diagonal extraction correctly isolates the stripe."""
    S = 5
    T = 1 + 2 * S
    fake_pattern = t.zeros(1, 12, T, T)
    fake_pattern[0, 4].diagonal(offset=-(S - 1)).fill_(1.0)

    store = {}
    induction_pattern_score_hook(fake_pattern, None, 4, S, store)
    assert t.allclose(store["score"], t.tensor(1.0))


def test_qk_component_bilinear_reconstruction():
    """Verify that summing pairwise QK component dot products reconstructs unmasked scores."""
    t.manual_seed(123)
    q_comps = t.randn(2, 1, 5, 4)
    k_comps = t.randn(3, 1, 5, 4)

    pair_scores = pairwise_qk_score_contributions(q_comps, k_comps, d_head=4)
    assert pair_scores.shape == (2, 3, 1, 5, 5)

    reconstructed = sum_pairwise_scores(pair_scores)
    q_sum = q_comps.sum(dim=0)
    k_sum = k_comps.sum(dim=0)
    expected = t.einsum("bqd,bkd->bqk", q_sum, k_sum) / math.sqrt(4)

    assert t.allclose(reconstructed, expected, atol=1e-6, rtol=1e-6)
