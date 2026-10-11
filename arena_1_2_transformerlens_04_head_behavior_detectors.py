"""
ARENA 1.2 — TransformerLens Exercise 04
Head-behavior detectors

Goal
----
Turn qualitative observations about attention patterns into quantitative
scores which can identify heads that mostly attend to:
    - the current token
    - the previous token
    - the first token

This is directly based on the current ARENA 1.2 exercise. The important
lesson is methodological:

    visual / qualitative observation
        -> quantitative statistic
        -> detector
        -> test against the actual activation pattern

The ARENA source explicitly notes that there is no uniquely correct detector;
the score should be plausible and validated against the observed patterns.

Current TODOs
-------------
Implement these three functions:

    current_attn_detector(cache)
    prev_attn_detector(cache)
    first_attn_detector(cache)

Each should return a list of strings like "5.7", meaning layer 5, head 7.

Suggested threshold
-------------------
Use a threshold around 0.4 for this exercise. The exact threshold is not the
point, and your output may differ slightly from ARENA's example.

Run
---
    python arena_1_2_transformerlens_04_head_behavior_detectors.py
"""

from __future__ import annotations

import torch as t
from transformer_lens import ActivationCache, HookedTransformer


MODEL_NAME = "gpt2-small"
TEXT = (
    "Natural language processing tasks, such as question answering, "
    "machine translation, reading comprehension, and summarization, "
    "are typically approached with supervised learning on task-specific datasets."
)
THRESHOLD = 0.4


def current_attn_detector(cache: ActivationCache, n_layers: int, n_heads: int) -> list[str]:
    """
    Return layer.head labels for heads which strongly attend to the current token.

    TODO:
        For each layer and head, compute the mean attention probability on the
        main diagonal of the head's attention pattern.

        If the score is above THRESHOLD, append f"{layer}.{head}".
    """
    # TODO: implement
    curr = []
    for i in range(n_layers):
        pattern = cache["pattern",i]
        for j in range(n_heads):
            if pattern[j].diagonal().mean()>THRESHOLD:
                curr.append(f"{i}.{j}")
    return curr


def prev_attn_detector(cache: ActivationCache, n_layers: int, n_heads: int) -> list[str]:
    """
    Return layer.head labels for heads which strongly attend to the previous token.

    TODO:
        For each layer and head, compute the mean attention probability on the
        diagonal immediately below the main diagonal.
    """
    # TODO: implement
    prev = []
    for i in range(n_layers):
        pattern = cache["pattern",i]
        for j in range(n_heads):
            if pattern[j].diagonal(-1).mean() > THRESHOLD:
                prev.append(f"{i}.{j}")
    return prev


def first_attn_detector(cache: ActivationCache, n_layers: int, n_heads: int) -> list[str]:
    """
    Return layer.head labels for heads which strongly attend to the first token.

    TODO:
        For each layer and head, compute the mean attention probability in
        column 0 of the attention pattern.
    """
    # TODO: implement
    first = []
    for i in range(n_layers):
        pattern = cache["pattern",i]
        for j in range(n_heads):
            if pattern[j][:,0].mean() > THRESHOLD:
                first.append(f"{i}.{j}")
    return first


def make_synthetic_cache() -> dict[tuple[str, int], t.Tensor]:
    """
    Build a tiny fake cache for deterministic unit tests.

    Shape for every layer's pattern:
        (heads, query_position, key_position)

    We deliberately create three known heads:
        layer 0, head 0 -> current-token head
        layer 0, head 1 -> previous-token head
        layer 0, head 2 -> first-token head
    """
    H, T = 3, 5
    pattern = t.full((H, T, T), 0.02)

    # Current-token head: diagonal is dominant.
    for i in range(T):
        pattern[0, i, i] = 0.92

    # Previous-token head: sub-diagonal is dominant where defined.
    for i in range(1, T):
        pattern[1, i, i - 1] = 0.92

    # First-token head: column 0 is dominant.
    for i in range(T):
        pattern[2, i, 0] = 0.92

    # Renormalize rows so they look like attention probabilities.
    pattern = pattern / pattern.sum(dim=-1, keepdim=True)

    return {("pattern", 0): pattern}


def run_synthetic_tests() -> None:
    """Test the detector logic on patterns with known answers."""
    cache = make_synthetic_cache()
    n_layers = 1
    n_heads = 3

    current = current_attn_detector(cache, n_layers, n_heads)
    previous = prev_attn_detector(cache, n_layers, n_heads)
    first = first_attn_detector(cache, n_layers, n_heads)

    assert current == ["0.0"], f"Current-token detector failed: {current}"
    assert previous == ["0.1"], f"Previous-token detector failed: {previous}"
    assert first == ["0.2"], f"First-token detector failed: {first}"

    print("SYNTHETIC TESTS PASSED")


def run_model_demo(model: HookedTransformer) -> None:
    """
    Run the real GPT-2 model and print the heads identified by each detector.
    """
    tokens = model.to_tokens(TEXT)
    _, cache = model.run_with_cache(tokens, remove_batch_dim=True)

    print("\nREAL MODEL RESULTS")
    print("=" * 72)
    print(f"Text: {TEXT}")
    print(f"Tokens: {len(tokens)}")
    print(f"Threshold: {THRESHOLD}")
    print()

    current = current_attn_detector(cache, model.cfg.n_layers, model.cfg.n_heads)
    previous = prev_attn_detector(cache, model.cfg.n_layers, model.cfg.n_heads)
    first = first_attn_detector(cache, model.cfg.n_layers, model.cfg.n_heads)

    print("Heads attending to current token :", ", ".join(current) or "none")
    print("Heads attending to previous token:", ", ".join(previous) or "none")
    print("Heads attending to first token   :", ", ".join(first) or "none")

    print()
    print("Important: these are behavioral classifications, not proof of a")
    print("mechanistic role. We will validate hypotheses using the patterns")
    print("and later with causal interventions.")


def main() -> None:
    print("=" * 72)
    print("ARENA 1.2 — TransformerLens Exercise 04")
    print("Head-behavior detectors")
    print("=" * 72)
    print()
    print("You are turning attention-pattern intuition into quantitative tests.")
    print()
    print("Definitions:")
    print("  current-token: average weight on the main diagonal")
    print("  previous-token: average weight one diagonal below the main diagonal")
    print("  first-token: average weight in key-position column 0")
    print()
    print("Threshold:", THRESHOLD)
    print()

    # Deterministic logic tests should work even without loading GPT-2.
    run_synthetic_tests()
    print()

    print(f"Loading {MODEL_NAME}...")
    model = HookedTransformer.from_pretrained(MODEL_NAME)
    model.eval()
    print("Model loaded.")

    run_model_demo(model)

    print()
    print("EXERCISE 04 COMPLETE")
    print("Next: repeated-token sequences and induction-head behavior.")


if __name__ == "__main__":
    main()
