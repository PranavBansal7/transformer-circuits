"""
ARENA 1.2 — TransformerLens Exercise 02
Tokenization + forward pass

What this teaches:
    text
      ↓
    tokenizer
      ↓
    token IDs
      ↓
    HookedTransformer
      ↓
    logits

You already know what a transformer does mathematically. The new skill is
learning how TransformerLens exposes the model's input and output interface.

CURRENT TODOs:
    1. Convert PROMPT into token IDs with the model tokenizer.
    2. Convert those token IDs back into readable string tokens.
    3. Run those token IDs through the model to obtain logits.
    4. Extract the logits for the final sequence position.

Do NOT implement caching or hooks yet. Those come in later exercises.

Run:
    python arena_1_2_transformerlens_02_tokenize_and_forward.py
"""

from __future__ import annotations

import torch as t
from transformer_lens import HookedTransformer


MODEL_NAME = "gpt2-small"

# We deliberately use a simple repeated-token prompt because later mech-interp
# exercises will use repeated sequences too.
PROMPT = "The cat sat on the mat. The cat"


def tokenize_prompt(model: HookedTransformer, text: str) -> t.Tensor:
    """
    Convert text -> token IDs.

    Expected shape:
        (batch, sequence_length)

    TODO 1:
        Use the TransformerLens tokenization API.
    """
    tokens = model.to_tokens(text)
    return tokens


def decode_tokens(model: HookedTransformer, tokens: t.Tensor) -> list[str]:
    """
    Convert token IDs -> readable token strings.

    TODO 2:
        Use the TransformerLens API which returns one string per token.
    """
    str_tokens = model.to_str_tokens(tokens)
    return str_tokens


def run_model(model: HookedTransformer, tokens: t.Tensor) -> t.Tensor:
    """
    Run the transformer and return logits.

    Expected shape:
        (batch, sequence_length, d_vocab)

    TODO 3:
        Run a normal forward pass.
    """
    logits = model(tokens)
    return logits


def get_final_position_logits(logits: t.Tensor) -> t.Tensor:
    """
    Return logits for the final sequence position.

    Input:
        (batch, sequence_length, d_vocab)

    Output:
        (batch, d_vocab)

    TODO 4:
        Select the final sequence position while preserving batch and vocab
        dimensions.
    """
    final_logits = logits[:,-1,:]
    return final_logits


def run_tests(
    model: HookedTransformer,
    tokens: t.Tensor,
    str_tokens: list[str],
    logits: t.Tensor,
    final_logits: t.Tensor,
) -> None:
    """Local correctness tests."""

    # ---------- tokenization ----------
    assert isinstance(tokens, t.Tensor), "tokens must be a torch.Tensor"
    assert tokens.dtype in (t.int64, t.int32), (
        f"tokens should contain integer IDs, got dtype={tokens.dtype}"
    )
    assert tokens.ndim == 2, (
        f"tokens should have shape (batch, seq), got {tuple(tokens.shape)}"
    )
    assert tokens.shape[0] == 1, "This exercise uses one input sequence."

    # ---------- decoding ----------
    assert len(str_tokens) == tokens.shape[1], (
        "There should be exactly one string token per token position."
    )
    assert all(isinstance(tok, str) for tok in str_tokens)

    # ---------- forward pass ----------
    assert isinstance(logits, t.Tensor), "logits must be a torch.Tensor"
    assert logits.ndim == 3, (
        "logits should have shape (batch, seq, vocab)"
    )
    assert logits.shape[0] == tokens.shape[0]
    assert logits.shape[1] == tokens.shape[1]
    assert logits.shape[2] == model.cfg.d_vocab
    assert t.isfinite(logits).all(), "Logits contain non-finite values."

    # ---------- final-position slice ----------
    assert final_logits.shape == (1, model.cfg.d_vocab), (
        f"Expected (1, d_vocab), got {tuple(final_logits.shape)}"
    )

    # This must exactly match the corresponding slice of the full logits.
    expected_final = logits[:, -1, :]
    assert t.allclose(final_logits, expected_final, rtol=0.0, atol=1e-6), (
        "Final-position logits do not match logits[:, -1, :]."
    )

    print("LOCAL TESTS PASSED")


def main() -> None:
    print("=" * 72)
    print("ARENA 1.2 — TransformerLens Exercise 02")
    print("=" * 72)
    print()
    print("Task:")
    print("  text → token IDs → string tokens → logits")
    print()
    print("Conceptual shapes:")
    print("  tokens:       (batch, sequence_length)")
    print("  logits:       (batch, sequence_length, d_vocab)")
    print("  final_logits: (batch, d_vocab)")
    print()
    print("Prompt:")
    print(f'  "{PROMPT}"')
    print()

    print(f"Loading {MODEL_NAME}...")
    model = HookedTransformer.from_pretrained(MODEL_NAME)
    print("Model loaded.")
    print()

    tokens = tokenize_prompt(model, PROMPT)
    str_tokens = decode_tokens(model, tokens)
    logits = run_model(model, tokens)
    final_logits = get_final_position_logits(logits)

    print("Token IDs:")
    print(tokens)
    print()

    print("String tokens:")
    print(str_tokens)
    print()

    print(f"tokens.shape       = {tuple(tokens.shape)}")
    print(f"logits.shape       = {tuple(logits.shape)}")
    print(f"final_logits.shape = {tuple(final_logits.shape)}")
    print()

    # Show the model's top 5 candidates at the final sequence position.
    top_values, top_ids = final_logits[0].topk(5)

    print("Top 5 next-token candidates at the final position:")
    for rank, (value, token_id) in enumerate(
        zip(top_values.tolist(), top_ids.tolist()), start=1
    ):
        token_text = model.to_single_str_token(token_id)
        print(f"  {rank}. id={token_id:<6} logit={value:>9.4f} token={token_text!r}")
    print()

    run_tests(model, tokens, str_tokens, logits, final_logits)

    print()
    print("Exercise 02 PASSED.")
    print("Next: run_with_cache and the ActivationCache.")


if __name__ == "__main__":
    main()
