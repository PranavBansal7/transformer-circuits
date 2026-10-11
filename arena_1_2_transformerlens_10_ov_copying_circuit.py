"""
ARENA 1.2 — TransformerLens Exercise 10
OV copying circuit

Goal
----
Use TransformerLens model weights to analyze the OV circuit of an attention
head, rather than relying on a particular prompt.

For one head, the content path is approximately:

    token embedding
        -> W_V
        -> W_O
        -> residual-stream update
        -> W_U
        -> vocabulary logits

With GPT-2's row-vector convention, the effective vocabulary-to-vocabulary
map can be written as:

    W_E @ W_V @ W_O @ W_U

This produces a matrix where entry [input_token, output_token] measures the
raw logit effect of the input token's embedding flowing through the head's
OV circuit into the output vocabulary.

For a copying head, we'd expect relatively strong diagonal entries:

    input token A -> output token A
    input token B -> output token B
    ...

CURRENT TODOs
-------------
1. Compute the effective OV vocabulary matrix for one layer/head.
2. Compute the copying score: the mean diagonal value of that matrix.
3. Compute the mean of the strongest off-diagonal value in each row. This is
   a simple baseline telling us how much the head maps tokens to some OTHER
   token instead of copying themselves.

Important
---------
This is a weight-level hypothesis test. It does NOT prove that the head uses
this map on every prompt, and it ignores the input-dependent attention pattern.
Later we'll combine QK and OV evidence.

Run
---
    python arena_1_2_transformerlens_10_ov_copying_circuit.py
"""

from __future__ import annotations

import torch as t
from transformer_lens import HookedTransformer


MODEL_NAME = "gpt2-small"
TARGET_LAYER = 7
TARGET_HEAD = 10


def effective_ov_vocab_matrix(
    model: HookedTransformer,
    layer: int,
    head: int,
) -> t.Tensor:
    """
    Return the effective vocabulary-to-vocabulary OV map for one head.

    Expected output shape:
        (d_vocab, d_vocab)

    TODO 1:
        Extract the embedding rows and this head's W_V / W_O weights, then
        compose them with W_U.

    Use the row-vector convention:
        W_E @ W_V @ W_O @ W_U

    Shapes:
        W_E          : (d_vocab, d_model)
        W_V[layer, h]: (d_model, d_head)
        W_O[layer, h]: (d_head, d_model)
        W_U          : (d_model, d_vocab)
    """
    # TODO 1
    return model.W_E @ model.W_V[layer,head] @ model.W_O[layer,head] @ model.W_U


def copying_score(ov_vocab: t.Tensor) -> t.Tensor:
    """
    Return a scalar copying score.

    TODO 2:
        Take the main diagonal of the vocabulary-to-vocabulary matrix and
        average it.
    """
    # TODO 2
    return ov_vocab.diagonal().mean()


def mean_strongest_off_diagonal(ov_vocab: t.Tensor) -> t.Tensor:
    """
    Return the mean strongest *non-diagonal* value in each input-token row.

    TODO 3:
        Mask out the diagonal, then take the maximum over output tokens for
        each input token, then average across input tokens.
    """
    # TODO 3
    ov_vocab.fill_diagonal_(0)
    mx, _ = ov_vocab.max(dim=-1)
    return mx.mean()


def run_synthetic_tests() -> None:
    """Test the statistics independently of GPT-2 / TransformerLens."""

    # Input-token rows, output-token columns.
    matrix = t.tensor(
        [
            [5.0, 0.5, 0.2, 0.1],
            [0.3, 4.0, 0.2, 0.1],
            [0.1, 0.2, 3.0, 0.4],
            [0.2, 0.1, 0.7, 2.0],
        ]
    )

    score = copying_score(matrix)
    off_diag = mean_strongest_off_diagonal(matrix)

    assert t.allclose(score, t.tensor(3.5)), (
        f"Expected copying score 3.5, got {score.item()}"
    )

    expected_off_diag = t.tensor((0.5 + 0.3 + 0.4 + 0.7) / 4)
    assert t.allclose(off_diag, expected_off_diag), (
        f"Unexpected strongest off-diagonal score: {off_diag.item()}"
    )

    print("SYNTHETIC TESTS PASSED")


def inspect_weight_shapes(model: HookedTransformer) -> None:
    """Print the relevant shapes once so the composition is concrete."""
    print("RELEVANT WEIGHT SHAPES")
    print("=" * 72)
    print(f"W_E                  : {tuple(model.W_E.shape)}")
    print(f"W_V                  : {tuple(model.W_V.shape)}")
    print(f"W_O                  : {tuple(model.W_O.shape)}")
    print(f"W_U                  : {tuple(model.W_U.shape)}")
    print()
    print("Single-head slices:")
    print(f"W_V[{TARGET_LAYER}, {TARGET_HEAD}] : {tuple(model.W_V[TARGET_LAYER, TARGET_HEAD].shape)}")
    print(f"W_O[{TARGET_LAYER}, {TARGET_HEAD}] : {tuple(model.W_O[TARGET_LAYER, TARGET_HEAD].shape)}")
    print()


def main() -> None:
    print("=" * 72)
    print("ARENA 1.2 — TransformerLens Exercise 10")
    print("OV copying circuit")
    print("=" * 72)
    print()
    print(f"Target head: L{TARGET_LAYER}H{TARGET_HEAD}")
    print()
    print("We are analyzing the weight-level map:")
    print("  token embedding -> W_V -> W_O -> unembedding")
    print()
    print("For a copying head, diagonal entries of the effective")
    print("vocabulary-to-vocabulary map should be comparatively large.")
    print()

    run_synthetic_tests()
    print()

    print(f"Loading {MODEL_NAME}...")
    model = HookedTransformer.from_pretrained(MODEL_NAME)
    model.eval()
    print("Model loaded.")
    print()

    inspect_weight_shapes(model)

    ov_vocab = effective_ov_vocab_matrix(
        model,
        layer=TARGET_LAYER,
        head=TARGET_HEAD,
    )

    print(f"Effective OV vocabulary matrix shape: {tuple(ov_vocab.shape)}")
    print()

    score = copying_score(ov_vocab)
    off_diag = mean_strongest_off_diagonal(ov_vocab)

    print(f"Copying score (mean diagonal)       : {score.item():.4f}")
    print(f"Mean strongest off-diagonal value   : {off_diag.item():.4f}")
    print(f"Diagonal minus off-diagonal baseline: {(score - off_diag).item():.4f}")
    print()

    # Inspect a few vocabulary rows using actual token strings.
    example_token_ids = [
        model.to_single_token(" the"),
        model.to_single_token(" cat"),
        model.to_single_token(" dog"),
    ]

    print("EXAMPLE OV ROWS")
    print("=" * 72)
    for token_id in example_token_ids:
        row = ov_vocab[token_id].clone()
        top_values, top_ids = row.topk(5)

        input_token = model.to_single_str_token(token_id)
        print(f"Input token {token_id}: {input_token!r}")
        for value, out_id in zip(top_values.tolist(), top_ids.tolist()):
            output_token = model.to_single_str_token(out_id)
            print(f"  -> {out_id:<6} {output_token!r:<16} effect={value:>9.4f}")
        print()

    # Shape and finiteness checks.
    assert ov_vocab.shape == (model.cfg.d_vocab, model.cfg.d_vocab)
    assert t.isfinite(ov_vocab).all()

    print("LOCAL TESTS PASSED")
    print("OV composition and summary statistics are valid.")
    print()
    print("EXERCISE 10 PASSED")
    print("Next: connect QK selection to OV writing and study composition.")


if __name__ == "__main__":
    main()
