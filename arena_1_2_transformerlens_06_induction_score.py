"""
ARENA 1.2 — TransformerLens Exercise 06
Canonical induction score

Goal
----
Measure how strongly each GPT-2 attention head attends to the position that
should contain the continuation of a repeated sequence.

We already established the behavioral experiment:
    [BOS, A, B, C, ..., X, A, B, C, ..., X]

For the second occurrence of a token, an induction head should attend to the
position immediately AFTER the token's first occurrence.

For a repeated-half length S, that creates an attention stripe with offset:

    key_position - query_position = -(S - 1)

Example with S=5:
    [BOS, A B C D E, A B C D E]

The second C is at index 8. The position immediately after the first C is
index 4, so the relevant attention connection is:

    query 8 -> key 4
    4 - 8 = -4 = -(5 - 1)

Current TODO
------------
Implement `induction_score`.

For every layer/head, return the mean attention probability along the
induction stripe for the second half, excluding the final repeated token
because there is no repeated continuation after it.

Input pattern shape:
    (batch, heads, query_position, key_position)

This script uses `remove_batch_dim=True`, so the real cache pattern has:
    (heads, query_position, key_position)

Run
---
    python arena_1_2_transformerlens_06_induction_score.py
"""

from __future__ import annotations

import torch as t
from transformer_lens import HookedTransformer


MODEL_NAME = "gpt2-small"
SEQ_LEN = 50
SEED = 0


def generate_repeated_tokens(model: HookedTransformer, seq_len: int) -> t.Tensor:
    """Create [BOS, random_sequence, same_random_sequence]."""
    t.manual_seed(SEED)

    bos = int(model.tokenizer.bos_token_id)
    random_tokens = t.randint(
        low=0,
        high=model.cfg.d_vocab,
        size=(seq_len,),
        dtype=t.long,
    )

    tokens = t.cat([
        t.tensor([bos], dtype=t.long),
        random_tokens,
        random_tokens.clone(),
    ])

    return tokens.unsqueeze(0)


def induction_score(
    pattern: t.Tensor,
    seq_len: int,
) -> t.Tensor:
    """
    Compute induction scores for every attention head.

    Parameters
    ----------
    pattern:
        Attention pattern of shape (heads, total_seq, total_seq).
    seq_len:
        Length of each repeated random half, excluding BOS.

    Returns
    -------
    scores:
        Tensor of shape (heads,), where each entry is the mean attention
        probability on the canonical induction stripe.

    TODO:
        Select the relevant query positions in the second half and the
        corresponding key positions immediately after the first occurrence.
        Return the mean across those positions for each head.
    """

    # TODO: implement
    x = pattern.diagonal(1-seq_len,dim1=-2,dim2=-1)
    x = x[...,2:-1].mean(dim=-1)
    return x


def synthetic_pattern(seq_len: int = 5) -> t.Tensor:
    """
    Make a tiny attention pattern with two heads.

    Head 0 is deliberately an induction head.
    Head 1 deliberately is not.

    Shape:
        (heads, total_seq, total_seq)
    """
    total = 1 + 2 * seq_len
    pattern = t.zeros(2, total, total)

    # Put mass uniformly on the correct induction stripe for head 0.
    # For seq_len=5 the offset is -4.
    query_positions = t.arange(seq_len + 1, 2 * seq_len)
    key_positions = query_positions - (seq_len - 1)

    for q, k in zip(query_positions.tolist(), key_positions.tolist()):
        pattern[0, q, k] = 1.0

    # Head 1 gets a previous-token pattern instead.
    for q in range(1, total):
        pattern[1, q, q - 1] = 1.0

    return pattern


def run_synthetic_tests() -> None:
    """Verify the indexing logic on a hand-constructed example."""
    pattern = synthetic_pattern(seq_len=5)
    scores = induction_score(pattern, seq_len=5)

    assert scores.shape == (2,), f"Expected (2,), got {tuple(scores.shape)}"
    assert t.allclose(scores[0], t.tensor(1.0)), (
        f"Synthetic induction head should score 1.0, got {scores[0].item()}"
    )
    assert scores[1].item() < 0.2, (
        f"Previous-token head should have a low induction score, got {scores[1].item()}"
    )

    print("SYNTHETIC TESTS PASSED")


def main() -> None:
    print("=" * 72)
    print("ARENA 1.2 — TransformerLens Exercise 06")
    print("Canonical induction score")
    print("=" * 72)
    print()
    print(f"Repeated-half length: {SEQ_LEN}")
    print(f"Induction stripe offset: {- (SEQ_LEN - 1)}")
    print()

    run_synthetic_tests()
    print()

    print(f"Loading {MODEL_NAME}...")
    model = HookedTransformer.from_pretrained(MODEL_NAME)
    model.eval()
    print("Model loaded.")
    print()

    tokens = generate_repeated_tokens(model, SEQ_LEN)
    _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
    pattern = t.stack([cache["pattern", layer] for layer in range(model.cfg.n_layers)])

    print(f"tokens.shape  = {tuple(tokens.shape)}")
    print(f"pattern.shape = {tuple(pattern.shape)}")
    print()

    scores = induction_score(pattern, SEQ_LEN)

    # Arrange into a layer x head matrix for easy reading.
    score_matrix = scores.reshape(model.cfg.n_layers, model.cfg.n_heads)

    flat_scores = scores.flatten()
    top_values, top_indices = flat_scores.topk(10)

    print("Top 10 heads by induction score:")
    for rank, (score, flat_idx) in enumerate(
        zip(top_values.tolist(), top_indices.tolist()), start=1
    ):
        layer = flat_idx // model.cfg.n_heads
        head = flat_idx % model.cfg.n_heads
        print(f"  {rank:2d}. L{layer}H{head}: {score:.4f}")

    best_flat_idx = int(top_indices[0])
    best_layer = best_flat_idx // model.cfg.n_heads
    best_head = best_flat_idx % model.cfg.n_heads
    best_pattern = pattern[best_layer, best_head]

    # The displayed row is the second-half query corresponding to the first
    # induction-scored position. For SEQ_LEN=50 this is query index 51.
    example_query = SEQ_LEN + 1
    example_weights = best_pattern[example_query]
    top_key_value, top_key_index = example_weights.topk(5)

    print()
    print(f"Best head: L{best_layer}H{best_head}")
    print(f"Example query position: {example_query}")
    print("Top attended key positions for that query:")
    for value, key in zip(top_key_value.tolist(), top_key_index.tolist()):
        print(f"  key index {key:3d}: attention={value:.4f}")

    print()
    print("LOCAL TESTS:"
          " synthetic indexing passed; inspect the real scores above.")
    print()
    print("EXERCISE 06 COMPLETE")
    print("Next: inspect the best head's full attention pattern and connect")
    print("the stripe to the repeated-token behavior visually and positionally.")


if __name__ == "__main__":
    main()
