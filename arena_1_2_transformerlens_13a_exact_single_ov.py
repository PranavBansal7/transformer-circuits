"""
ARENA 1.2 — Exact canonical Exercise: OV copying circuit for L1H4
===================================================================

This is the current ARENA 1.2 Section 4 exercise, translated into a
standalone .py file.

MODEL
-----
This is the canonical 2-layer attention-only toy model used by ARENA:
    repo: callummcdougall/attn_only_2L_half
    file: attn_only_2L_half.pth
    layers: 2
    heads: 12
    d_model: 768
    d_head: 64
    d_vocab: 50278
    context: 2048
    normalization: none
    positional embeddings: shortformer

CURRENT EXERCISE
----------------
Compute the full OV circuit for L1H4:

    W_E @ W_V[1,4] @ W_O[1,4] @ W_U

using TransformerLens' FactoredMatrix, then measure the fraction of input
vocabulary tokens for which the strongest output logit is on the diagonal.

ARENA's reference result is about 30.79% top-1 accuracy for L1H4 alone.

IMPORTANT
---------
Do NOT materialize the full (50278, 50278) matrix.
Use FactoredMatrix indexing to evaluate manageable row batches.

Run:
    python arena_1_2_transformerlens_13a_exact_single_ov.py
"""

from __future__ import annotations

import torch as t
from huggingface_hub import hf_hub_download
from transformer_lens import FactoredMatrix, HookedTransformer, HookedTransformerConfig


REPO_ID = "callummcdougall/attn_only_2L_half"
WEIGHTS_FILE = "attn_only_2L_half.pth"
LAYER = 1
HEAD = 4
BATCH_SIZE = 1000


def load_model() -> HookedTransformer:
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
    weights_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=WEIGHTS_FILE,
    )
    pretrained_weights = t.load(
        weights_path,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(pretrained_weights)
    model.eval()
    return model


def make_full_ov_circuit(model: HookedTransformer) -> FactoredMatrix:
    """
    Compute the full OV circuit for L1H4 as a FactoredMatrix.

    Mathematically:

        W_E @ W_V[1,4] @ W_O[1,4] @ W_U

    TODO:
        Construct and return the FactoredMatrix.

    Hint:
        You can construct the middle OV circuit first and then multiply by
        W_E and W_U, all while keeping the representation factored.
    """

    # TODO: implement
    a = model.W_E @ model.W_V[1,4]
    b = model.W_O[1,4] @ model.W_U
    return FactoredMatrix(a,b)


def top_1_acc(full_ov: FactoredMatrix, batch_size: int = BATCH_SIZE) -> float:
    """
    Return the fraction of vocabulary tokens for which the maximum output
    logit lies on the diagonal.

    IMPORTANT:
        full_ov[indices].AB has shape (batch_size, d_vocab), where each row
        corresponds to one input token and columns correspond to output-token
        logits. Therefore argmax is over dim=1.

    This function is provided so the exercise stays focused on constructing
    the circuit.
    """
    total_correct = 0

    indices = t.arange(full_ov.shape[0], dtype=t.long)
    for batch_indices in t.split(indices, batch_size):
        logits = full_ov[batch_indices].AB
        predicted = t.argmax(logits, dim=1)
        total_correct += int((predicted == batch_indices).sum())

    return total_correct / full_ov.shape[0]


def run_tests(model: HookedTransformer, full_ov: FactoredMatrix) -> None:
    assert model.cfg.n_layers == 2
    assert model.cfg.n_heads == 12
    assert model.cfg.d_model == 768
    assert model.cfg.d_head == 64
    assert model.cfg.d_vocab == 50278

    assert full_ov.shape == (model.cfg.d_vocab, model.cfg.d_vocab)
    assert full_ov.A.shape[0] == model.cfg.d_vocab
    assert full_ov.B.shape[1] == model.cfg.d_vocab
    assert full_ov.mdim == model.cfg.d_head

    # Direct-vs-factored parity on a tiny set of rows.
    rows = t.tensor([0, 1, 7, 42, 12345, 50277], dtype=t.long)

    direct = (
        (model.W_E[rows] @ model.W_V[LAYER, HEAD])
        @ (model.W_O[LAYER, HEAD] @ model.W_U)
    )
    factored = full_ov[rows].AB

    assert t.allclose(direct, factored, atol=1e-5, rtol=1e-5)

    score = top_1_acc(full_ov)
    print("LOCAL TESTS PASSED")
    print(f"L1H4 top-1 copying accuracy: {score:.4f}")

    # ARENA reports approximately 30.79%.
    assert 0.25 < score < 0.40, (
        f"Unexpected L1H4 copying accuracy: {score:.4f}. "
        "Expected roughly 0.3079."
    )


def main() -> None:
    print("=" * 72)
    print("ARENA 1.2 — Exercise 13A")
    print("Exact canonical OV copying circuit for L1H4")
    print("=" * 72)
    print()
    print("Model: attn_only_2L_half")
    print("Target: L1H4")
    print()
    print("Conceptual circuit:")
    print("  input token")
    print("      ↓")
    print("    W_E")
    print("      ↓")
    print("    W_V")
    print("      ↓")
    print("    W_O")
    print("      ↓")
    print("    W_U")
    print("      ↓")
    print("  output logits")
    print()
    print("The logical matrix is (d_vocab, d_vocab), but we keep it factored.")
    print()

    print("Loading canonical ARENA checkpoint...")
    model = load_model()
    print("Model loaded.")
    print()

    print("MODEL CONFIG")
    print("-" * 72)
    print(f"n_layers = {model.cfg.n_layers}")
    print(f"n_heads  = {model.cfg.n_heads}")
    print(f"d_model  = {model.cfg.d_model}")
    print(f"d_head   = {model.cfg.d_head}")
    print(f"d_vocab  = {model.cfg.d_vocab}")
    print(f"n_ctx    = {model.cfg.n_ctx}")
    print()

    full_ov = make_full_ov_circuit(model)

    print("FACTORED OV CIRCUIT")
    print("-" * 72)
    print(f"Logical shape = {full_ov.shape}")
    print(f"A shape       = {tuple(full_ov.A.shape)}")
    print(f"B shape       = {tuple(full_ov.B.shape)}")
    print(f"Hidden dim    = {full_ov.mdim}")
    print()

    score = top_1_acc(full_ov)
    print(f"L1H4 copying accuracy: {score:.4f}")
    print("ARENA reference: about 0.3079")
    print()

    run_tests(model, full_ov)

    print()
    print("EXERCISE 13A PASSED")
    print("Next: combine L1H4 + L1H10 into the effective OV circuit.")


if __name__ == "__main__":
    main()
