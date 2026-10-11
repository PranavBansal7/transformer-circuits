"""
ARENA 1.2 — TransformerLens Exercise 07
Inspect an induction head's attention pattern

Goal
----
Take the best induction head from the repeated-token experiment, inspect its
attention on individual query positions, and verify the characteristic
induction stripe directly from the activation pattern.

This is the transition from:
    "the head has a high score"

to:
    "I can point to the exact positions the head is using and explain why."

Current TODOs
-------------
1. Given an attention pattern and query position, find the key position with
   maximum attention.
2. For several second-half query positions, determine the expected induction
   key position from the repeated-sequence geometry.
3. Count how many inspected queries put their maximum attention on the
   expected induction key.

You should reason about the indexing before writing code.

Run
---
    python arena_1_2_transformerlens_07_inspect_induction_head.py
"""

from __future__ import annotations

import torch as t
from transformer_lens import HookedTransformer


MODEL_NAME = "gpt2-small"
SEQ_LEN = 50
SEED = 0

# GPT-2 Small attention heads are addressed as L[layer]H[head].
# Exercise 06 found L7H10 as the top head for this repeated sequence.
BEST_LAYER = 7
BEST_HEAD = 10

# A handful of positions in the repeated half. We deliberately inspect more
# than one query rather than trusting a single attention link.
QUERY_OFFSETS = [0, 1, 5, 10, 20, 30, 40, 49]


def generate_repeated_tokens(model: HookedTransformer, seq_len: int) -> t.Tensor:
    """Generate [BOS, A...X, A...X] deterministically."""
    t.manual_seed(SEED)

    bos_id = model.tokenizer.bos_token_id
    random_tokens = t.randint(
        low=0,
        high=model.cfg.d_vocab,
        size=(seq_len,),
        dtype=t.long,
    )

    tokens = t.cat(
        [
            t.tensor([bos_id], dtype=t.long),
            random_tokens,
            random_tokens.clone(),
        ]
    ).unsqueeze(0)

    return tokens


def get_attention_pattern(
    model: HookedTransformer,
    tokens: t.Tensor,
    layer: int,
) -> t.Tensor:
    """
    Run the model and return one layer's attention pattern.

    Output shape:
        (heads, query_position, key_position)
    """
    _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
    return cache["pattern", layer]


def max_attended_key(pattern: t.Tensor, head: int, query_position: int) -> int:
    """
    Return the key position receiving the most attention.

    TODO 1:
        Select this head + query position, then find the key-position index
        containing the largest attention value.
    """
    # TODO 1
    return pattern[head][query_position].argmax()


def expected_induction_key(query_position: int, seq_len: int) -> int:
    """
    Return the key position predicted by induction geometry.

    For this experiment:
        sequence = [BOS, first_half, repeated_half]

    A query in the repeated half at position q corresponds to the token at
    position q - seq_len in the first half. The induction head should attend
    to the token immediately AFTER that earlier occurrence.

    TODO 2:
        Derive the expected key position from query_position and seq_len.
    """
    # TODO 2
    return query_position-seq_len+1


def count_correct_induction_links(
    pattern: t.Tensor,
    head: int,
    seq_len: int,
    query_offsets: list[int],
) -> int:
    """
    Count how many inspected second-half queries point to the expected key.

    TODO 3:
        For each query offset:
            - construct the query position in the repeated half
            - find the max-attended key
            - compare it with expected_induction_key(...)
    """
    # TODO 3
    ans = 0
    for query in query_offsets:
        if pattern[head][query].argmax() == expected_induction_key(query,seq_len):
            ans = ans + 1
    return ans


def run_tests() -> None:
    """Deterministic tests for the pure indexing helper functions."""
    # A tiny fake pattern: one head, 10 positions.
    fake = t.zeros(1, 10, 10)

    # For query position 6 with seq_len=5, expected key is 2.
    fake[0, 6, 2] = 0.9
    fake[0, 6, 6] = 0.1

    assert max_attended_key(fake, head=0, query_position=6) == 2
    assert expected_induction_key(query_position=6, seq_len=5) == 2

    # For q=6,7,8,9 and seq_len=5, expected keys are 2,3,4,5.
    assert [expected_induction_key(q, 5) for q in [6, 7, 8, 9]] == [2, 3, 4, 5]

    print("SYNTHETIC TESTS PASSED")


def main() -> None:
    print("=" * 72)
    print("ARENA 1.2 — TransformerLens Exercise 07")
    print("Inspect an induction head")
    print("=" * 72)
    print()
    print("Target head: L7H10")
    print("Repeated-half length: 50")
    print()
    print("We will inspect several query positions in the repeated half and ask:")
    print("  Where does this head actually attend?")
    print("  Where SHOULD an induction head attend?")
    print()

    run_tests()
    print()

    print(f"Loading {MODEL_NAME}...")
    model = HookedTransformer.from_pretrained(MODEL_NAME)
    model.eval()
    print("Model loaded.")
    print()

    tokens = generate_repeated_tokens(model, SEQ_LEN)
    pattern = get_attention_pattern(model, tokens, BEST_LAYER)

    print(f"pattern.shape = {tuple(pattern.shape)}")
    print()

    correct = 0
    for offset in QUERY_OFFSETS:
        # BOS is index 0 and the first half starts at index 1.
        query_position = 1 + SEQ_LEN + offset
        expected_key = expected_induction_key(query_position, SEQ_LEN)
        actual_key = max_attended_key(pattern, BEST_HEAD, query_position)
        actual_attention = pattern[BEST_HEAD, query_position, actual_key].item()
        expected_attention = pattern[BEST_HEAD, query_position, expected_key].item()

        if actual_key == expected_key:
            correct += 1

        print(
            f"query {query_position:3d} -> "
            f"expected key {expected_key:3d}, "
            f"actual top key {actual_key:3d}, "
            f"expected attention {expected_attention:8.4f}, "
            f"top attention {actual_attention:8.4f}"
        )

    print()
    print(
        f"Correct max-attention links: {correct}/{len(QUERY_OFFSETS)}"
    )

    print()
    print("Interpretation:")
    print("  For the second copy, the head should attend to the token")
    print("  immediately after the matching token in the first copy.")
    print("  Repeated-half query positions therefore produce a stripe at")
    print(f"  key_position - query_position = -(SEQ_LEN - 1) = {- (SEQ_LEN - 1)}")
    print()

    print("If your TODOs are correct, the synthetic tests will pass and the")
    print("real-model inspection should show the predicted stripe repeatedly.")


if __name__ == "__main__":
    main()
