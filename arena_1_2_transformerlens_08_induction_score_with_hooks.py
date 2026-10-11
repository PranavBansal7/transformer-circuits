"""
ARENA 1.2 — TransformerLens Exercise 08
Induction score with hooks

Goal
----
Use `run_with_hooks` to compute induction scores while the model is running,
rather than retrieving the attention pattern with `run_with_cache` and
processing it afterward.

This is the first real hook exercise. We are initially using hooks
observationally: the hook reads the activation and does NOT modify it.

Current TODO
------------
Implement `induction_score_hook`.

The hook receives an attention pattern with shape:
    (batch, heads, query_position, key_position)

For repetition length SEQ_LEN=50, the induction stripe has offset:
    -(SEQ_LEN - 1) = -49

For every head, write its mean induction-stripe attention into `score_store`.

The provided tests verify that the hook agrees with a direct cache-based
calculation on the same model and input.

Run
---
    python arena_1_2_transformerlens_08_induction_score_with_hooks.py
"""

from __future__ import annotations

import torch as t
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint


MODEL_NAME = "gpt2-small"
SEQ_LEN = 50
BATCH_SIZE = 10


def generate_repeated_tokens(
    model: HookedTransformer,
    seq_len: int,
    batch_size: int,
) -> t.Tensor:
    """Generate [BOS, random block, same block] for each batch item."""
    t.manual_seed(0)

    bos = t.full(
        (batch_size, 1),
        int(model.tokenizer.bos_token_id),
        dtype=t.long,
        device=model.cfg.device,
    )
    first = t.randint(
        0,
        model.cfg.d_vocab,
        (batch_size, seq_len),
        dtype=t.long,
        device=model.cfg.device,
    )
    repeated = t.cat([first, first], dim=1)
    return t.cat([bos, repeated], dim=1)


def induction_score_hook(
    attn_pattern: t.Tensor,
    hook: HookPoint,
    score_store: t.Tensor,
    seq_len: int,
) -> None:
    """
    Compute induction scores for every head at this layer.

    Args:
        attn_pattern:
            Shape (batch, heads, query_position, key_position).
        hook:
            TransformerLens HookPoint. `hook.layer()` tells us the layer.
        score_store:
            Tensor of shape (n_layers, n_heads) where this hook should write
            the scores for its layer.
        seq_len:
            Length of one repeated half.

    TODO:
        1. Select the induction diagonal with offset -(seq_len - 1).
        2. Average over batch and diagonal/query positions.
        3. Write one score per head into score_store[hook.layer(), :].

    Do not loop over layers or heads here. The model will call this hook once
    per matching layer, and the tensor operation should handle all heads at
    once.
    """

    # TODO: implement
    pattern = attn_pattern.diagonal(1-seq_len,dim1=-2,dim2=-1)
    pattern = pattern.mean(dim=-1)
    pattern = pattern.mean(dim=0)
    score_store[hook.layer(),:] = pattern


def direct_induction_scores(
    cache,
    seq_len: int,
    n_layers: int,
    n_heads: int,
) -> t.Tensor:
    """Reference calculation from the cached attention patterns."""
    scores = t.zeros((n_layers, n_heads), device=cache["pattern", 0].device)
    offset = -(seq_len - 1)

    for layer in range(n_layers):
        pattern = cache["pattern", layer]  # (B, H, T, T)
        for head in range(n_heads):
            scores[layer, head] = pattern[:, head].diagonal(offset=offset,dim1=-2,dim2=-1).mean()

    return scores


def run_tests(
    model: HookedTransformer,
    tokens: t.Tensor,
    hook_scores: t.Tensor,
    cache_scores: t.Tensor,
) -> None:
    """Compare hook-based results against the direct cache calculation."""

    assert hook_scores.shape == (
        model.cfg.n_layers,
        model.cfg.n_heads,
    )
    assert t.isfinite(hook_scores).all()

    # The hook must reproduce the direct calculation.
    max_error = (hook_scores - cache_scores).abs().max().item()
    assert max_error < 1e-6, (
        f"Hook scores do not match cache scores. Max error: {max_error:.3e}"
    )

    # We expect the strongest induction heads to have large scores.
    best_value = hook_scores.max().item()
    assert best_value > 0.7, (
        f"Unexpectedly weak induction signal: best score = {best_value:.3f}"
    )

    print("LOCAL TESTS PASSED")
    print(f"Maximum hook-vs-cache error: {max_error:.3e}")


def main() -> None:
    print("=" * 72)
    print("ARENA 1.2 — TransformerLens Exercise 08")
    print("Induction score with hooks")
    print("=" * 72)
    print()
    print("The hook reads attention patterns during the forward pass.")
    print("It does not modify the model or its activations yet.")
    print()

    model = HookedTransformer.from_pretrained(MODEL_NAME)
    model.eval()

    tokens = generate_repeated_tokens(model, SEQ_LEN, BATCH_SIZE)

    score_store = t.zeros(
        (model.cfg.n_layers, model.cfg.n_heads),
        device=model.cfg.device,
    )

    # The hook below is attached to every attention-pattern activation.
    # You only need to implement the hook function itself.
    hook_fn = lambda attn, hook: induction_score_hook(
        attn,
        hook,
        score_store,
        SEQ_LEN,
    )

    pattern_hook_names_filter = lambda name: name.endswith("hook_pattern")

    model.run_with_hooks(
        tokens,
        return_type=None,
        fwd_hooks=[(pattern_hook_names_filter, hook_fn)],
    )

    # Independent reference calculation.
    _, cache = model.run_with_cache(tokens, remove_batch_dim=False)
    cache_scores = direct_induction_scores(
        cache,
        SEQ_LEN,
        model.cfg.n_layers,
        model.cfg.n_heads,
    )

    print(f"tokens.shape = {tuple(tokens.shape)}")
    print(f"score_store.shape = {tuple(score_store.shape)}")
    print()

    flat_scores = score_store.flatten()
    top_values, top_indices = flat_scores.topk(10)

    print("Top 10 heads by hook-computed induction score:")
    for rank, (value, flat_idx) in enumerate(
        zip(top_values.tolist(), top_indices.tolist()), start=1
    ):
        layer = flat_idx // model.cfg.n_heads
        head = flat_idx % model.cfg.n_heads
        print(f"  {rank:2}. L{layer}H{head}: {value:.4f}")

    print()
    run_tests(model, tokens, score_store, cache_scores)

    print()
    print("EXERCISE 08 PASSED")
    print("Next: use hooks to intervene on a head (ablation).")


if __name__ == "__main__":
    main()
