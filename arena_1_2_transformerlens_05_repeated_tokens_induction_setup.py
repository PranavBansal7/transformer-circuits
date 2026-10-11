"""
ARENA 1.2 — TransformerLens Exercise 05
Repeated-token sequences: setting up the induction experiment

Goal:
    Create a sequence of random tokens repeated twice, run GPT-2 on it,
    and compare next-token performance on the first and second halves.

Why:
    A model with induction heads should do much better on the repeated half
    because it can use the first occurrence of a sequence to predict its
    continuation in the second occurrence.

Current TODOs:
    1. Generate [BOS, random_sequence, same_random_sequence].
    2. Run that sequence through the model with a cache.

The token sequence should have shape:
    (batch_size, 1 + 2 * seq_len)

The logits should have shape:
    (batch_size, 1 + 2 * seq_len, d_vocab)

The cache is an ActivationCache.

Run:
    python arena_1_2_transformerlens_05_repeated_tokens_induction_setup.py
"""

from __future__ import annotations

import torch as t
from transformer_lens import ActivationCache, HookedTransformer


MODEL_NAME = "gpt2-small"
SEQ_LEN = 50
BATCH_SIZE = 1


def generate_repeated_tokens(
    model: HookedTransformer,
    seq_len: int,
    batch_size: int = 1,
) -> t.Tensor:
    """
    Generate:

        [BOS, A, B, C, ..., X, A, B, C, ..., X]

    where A...X is one random sequence repeated exactly twice.

    TODO 1:
        Implement this function.

    Requirements:
        - use model.tokenizer.bos_token_id for the prefix
        - generate random token IDs from the model vocabulary
        - make the second half exactly equal to the first half
        - return shape (batch_size, 1 + 2 * seq_len)
        - use torch.long / int64 token IDs
        - make the result reproducible with torch.manual_seed(0)
    """

    # TODO 1
    t.manual_seed(0)
    bos_id = model.tokenizer.bos_token_id
    bos_tensor = t.full(size=(batch_size,1),fill_value=bos_id,dtype=t.long)
    random = t.randint(low=0,high=model.cfg.d_vocab,size=(batch_size,seq_len),dtype=t.long)
    rep_tokens = t.cat([bos_tensor,random,random],dim=1)

    return rep_tokens


def run_and_cache_model_repeated_tokens(
    model: HookedTransformer,
    seq_len: int,
    batch_size: int = 1,
) -> tuple[t.Tensor, t.Tensor, ActivationCache]:
    """
    Generate the repeated-token sequence and run the model with a cache.

    TODO 2:
        Use your generate_repeated_tokens function, then call
        model.run_with_cache(...).

    Return:
        rep_tokens
        rep_logits
        rep_cache
    """

    # TODO 2
    rep_tokens = generate_repeated_tokens(model,seq_len,batch_size)
    rep_logits,rep_cache = model.run_with_cache(rep_tokens)

    return rep_tokens, rep_logits, rep_cache


def get_next_token_log_probs(
    logits: t.Tensor,
    tokens: t.Tensor,
) -> t.Tensor:
    """
    Return the log-probability assigned to the true next token at each
    prediction position.

    Input:
        logits: (batch, seq, vocab)
        tokens: (batch, seq)

    Output:
        log_probs: (batch, seq - 1)

    This function is provided so the exercise stays focused on constructing
    and running the repeated sequence.
    """

    logprobs = logits.log_softmax(dim=-1)
    next_token_ids = tokens[:, 1:]
    next_token_logprobs = logprobs[:, :-1, :].gather(
        dim=-1,
        index=next_token_ids.unsqueeze(-1),
    ).squeeze(-1)

    return next_token_logprobs


def run_tests(
    model: HookedTransformer,
    tokens: t.Tensor,
    logits: t.Tensor,
    cache: ActivationCache,
) -> None:
    """Local correctness tests."""

    B = BATCH_SIZE
    full_seq_len = 1 + 2 * SEQ_LEN
    V = model.cfg.d_vocab

    assert tokens.shape == (B, full_seq_len), (
        f"Expected tokens shape {(B, full_seq_len)}, "
        f"got {tuple(tokens.shape)}"
    )
    assert tokens.dtype == t.long

    expected_bos = int(model.tokenizer.bos_token_id)
    assert int(tokens[0, 0]) == expected_bos, (
        "First token must be GPT-2's BOS token."
    )

    first_half = tokens[:, 1 : 1 + SEQ_LEN]
    second_half = tokens[:, 1 + SEQ_LEN : 1 + 2 * SEQ_LEN]
    assert t.equal(first_half, second_half), (
        "The second random-token half must exactly equal the first half."
    )

    assert t.all((first_half >= 0) & (first_half < V))

    assert logits.shape == (B, full_seq_len, V), (
        f"Expected logits shape {(B, full_seq_len, V)}, "
        f"got {tuple(logits.shape)}"
    )
    assert t.isfinite(logits).all()

    assert cache["resid_pre", 0].shape[0] == B
    assert cache["resid_pre", 0].shape[1] == full_seq_len
    assert cache["pattern", 0].shape[-2:] == (full_seq_len, full_seq_len)

    logits2, _ = model.run_with_cache(tokens)
    assert t.allclose(logits, logits2, atol=1e-6, rtol=0.0)

    log_probs = get_next_token_log_probs(logits, tokens)
    assert log_probs.shape == (B, full_seq_len - 1)
    assert t.isfinite(log_probs).all()

    print("LOCAL TESTS PASSED")


def main() -> None:
    print("=" * 72)
    print("ARENA 1.2 — TransformerLens Exercise 05")
    print("Repeated-token induction setup")
    print("=" * 72)
    print()
    print(f"Sequence length (repeated half): {SEQ_LEN}")
    print(f"Batch size: {BATCH_SIZE}")
    print()
    print("Target structure:")
    print("  [BOS, A, B, C, ..., X, A, B, C, ..., X]")
    print()
    print("The first half is new/random.")
    print("The second half repeats the first half exactly.")
    print()

    print(f"Loading {MODEL_NAME}...")
    model = HookedTransformer.from_pretrained(MODEL_NAME)
    model.eval()
    print("Model loaded.")
    print()

    tokens, logits, cache = run_and_cache_model_repeated_tokens(
        model,
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE,
    )

    print(f"tokens.shape = {tuple(tokens.shape)}")
    print(f"logits.shape = {tuple(logits.shape)}")
    print(f"type(cache) = {type(cache).__name__}")
    print()

    string_tokens = model.to_str_tokens(tokens)
    print("First 12 string tokens:")
    print(string_tokens[:12])
    print()

    log_probs = get_next_token_log_probs(logits, tokens).squeeze(0)

    first_half_mean = log_probs[:SEQ_LEN].mean()
    second_half_mean = log_probs[SEQ_LEN:].mean()

    print(f"Mean next-token log-prob, first half : {first_half_mean.item():.3f}")
    print(f"Mean next-token log-prob, second half: {second_half_mean.item():.3f}")
    print()

    print("What we expect:")
    print("  The second-half score should be substantially better")
    print("  if induction behavior is present.")
    print()

    run_tests(model, tokens, logits, cache)

    print()
    print("EXERCISE 05 PASSED")
    print("Next: inspect the repeated-sequence attention patterns and identify")
    print("the characteristic induction stripe.")


if __name__ == "__main__":
    main()
