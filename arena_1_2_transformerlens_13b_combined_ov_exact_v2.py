"""
ARENA 1.2 — Exact canonical combined OV circuit (fixed loader)

Canonical model:
    callummcdougall/attn_only_2L_half

Canonical heads:
    L1H4 and L1H10

This exercise follows the current ARENA Section 4 exercise:
    W_O_both = concat(W_O[1, 4], W_O[1, 10]) along the head dimension
    W_V_both = concat(W_V[1, 4], W_V[1, 10]) along the d_head dimension
    W_OV_eff = W_E @ FactoredMatrix(W_V_both, W_O_both) @ W_U

The giant (d_vocab, d_vocab) matrix is never materialized.
"""

from __future__ import annotations

import torch as t
import einops
from huggingface_hub import hf_hub_download
from transformer_lens import FactoredMatrix, HookedTransformer, HookedTransformerConfig


REPO_ID = "callummcdougall/attn_only_2L_half"
FILENAME = "attn_only_2L_half.pth"

LAYER = 1
HEADS = [4, 10]
BATCH_SIZE = 1000
DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")


def load_canonical_model() -> HookedTransformer:
    """Build the exact ARENA toy model and load its custom checkpoint."""
    cfg = HookedTransformerConfig(
        d_model=768,
        d_head=64,
        n_heads=12,
        n_layers=2,
        n_ctx=2048,
        d_vocab=50278,
        attention_dir="causal",
        attn_only=True,
        tokenizer_name="EleutherAI/gpt-neox-20b",
        seed=398,
        use_attn_result=True,
        normalization_type=None,
        positional_embedding_type="shortformer",
    )

    model = HookedTransformer(cfg)
    weights_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    state_dict = t.load(weights_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def make_combined_ov_circuit(model: HookedTransformer) -> FactoredMatrix:
    """
    Build the effective OV circuit for L1H4 + L1H10.

    TODO:
        Extract W_O and W_V for the two layer-1 heads and concatenate them
        into the factors of one FactoredMatrix.

    Target factor shapes:
        W_V_both: (d_model, 2*d_head)
        W_O_both: (2*d_head, d_model)

    Return:
        W_E @ FactoredMatrix(W_V_both, W_O_both) @ W_U
    """

    # TODO: your implementation here
    W_O_both = t.cat([model.W_O[LAYER,head] for head in HEADS],dim=0)

    W_V_both = t.cat([model.W_V[LAYER,head] for head in HEADS],dim=1)

    W_OV_eff = FactoredMatrix(W_V_both,W_O_both)

    return model.W_E@W_OV_eff@model.W_U


def top_1_acc(full_ov_circuit: FactoredMatrix, batch_size: int = BATCH_SIZE) -> float:
    """Fraction of vocabulary rows whose top logit is the same token."""
    total = 0
    n = full_ov_circuit.shape[0]
    indices = t.arange(n, device=full_ov_circuit.A.device)

    for batch_indices in t.split(indices, batch_size):
        # Only materialize this small [batch, vocab] slice.
        ab_slice = full_ov_circuit[batch_indices].AB
        predicted = t.argmax(ab_slice, dim=1)
        total += (predicted == batch_indices).float().sum().item()

    return total / n


def synthetic_test() -> None:
    """Test the concatenation logic on small matrices."""
    t.manual_seed(0)
    d_model, d_head = 5, 2

    v1 = t.randn(d_model, d_head)
    v2 = t.randn(d_model, d_head)
    o1 = t.randn(d_head, d_model)
    o2 = t.randn(d_head, d_model)

    v_both = t.cat([v1, v2], dim=1)
    o_both = t.cat([o1, o2], dim=0)

    expected = v1 @ o1 + v2 @ o2
    actual = v_both @ o_both

    assert v_both.shape == (d_model, 2 * d_head)
    assert o_both.shape == (2 * d_head, d_model)
    assert t.allclose(actual, expected, atol=1e-6)

    print("SYNTHETIC TESTS PASSED")


def main() -> None:
    print("=" * 72)
    print("ARENA 1.2 — Exact canonical combined OV circuit (fixed loader)")
    print("=" * 72)
    print()
    print(f"Model: {REPO_ID}")
    print(f"Target heads: L1H4 + L1H10")
    print(f"Device: {DEVICE}")
    print()
    print("Important: this is the custom ARENA checkpoint, so it is loaded")
    print("from its explicit HookedTransformerConfig + state dict.")
    print()

    synthetic_test()
    print()

    print("Loading canonical ARENA checkpoint...")
    model = load_canonical_model()
    print("Model loaded.")
    print()

    print("MODEL CONFIG")
    print("-" * 72)
    print(f"n_layers = {model.cfg.n_layers}")
    print(f"n_heads  = {model.cfg.n_heads}")
    print(f"d_model  = {model.cfg.d_model}")
    print(f"d_head   = {model.cfg.d_head}")
    print(f"d_vocab  = {model.cfg.d_vocab}")
    print()

    full_ov = make_combined_ov_circuit(model)

    print("FACTORED EFFECTIVE OV CIRCUIT")
    print("-" * 72)
    print(f"Logical shape = {tuple(full_ov.shape)}")
    print(f"A shape       = {tuple(full_ov.A.shape)}")
    print(f"B shape       = {tuple(full_ov.B.shape)}")
    print(f"Hidden dim    = {full_ov.A.shape[1]}")
    print()

    score = top_1_acc(full_ov)
    print(f"Combined L1H4 + L1H10 copying accuracy: {score:.4f}")
    print("ARENA reference: about 0.9560")
    print()

    assert 0.94 < score < 0.97, (
        f"Unexpected copying accuracy {score:.4f}. "
        "Check the combined W_V/W_O concatenation."
    )

    print("LOCAL TESTS PASSED")
    print("Combined OV circuit matches the expected ARENA behavior.")
    print()
    print("EXERCISE 13B PASSED")
    print("Next: canonical L0H7 QK previous-token circuit.")


if __name__ == "__main__":
    main()
