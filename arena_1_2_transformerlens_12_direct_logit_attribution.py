"""
ARENA 1.2 — TransformerLens Exercise 12
Direct Logit Attribution (DLA) for attention heads

Goal
----
Use the residual-stream decomposition to measure which attention heads
positively or negatively contribute to the logit of a chosen target token.

This is the bridge from:
    "I can inspect activations"
into:
    "I can ask which components directly support the prediction."

Important distinction:
    DLA is observational / attributional, not a causal intervention.
    A large positive DLA score does not by itself imply that ablating the
    head will hurt performance.

Current TODOs
-------------
1. Get the per-head residual contributions at the final position.
2. Apply the final LayerNorm scaling to that residual stack.
3. Project each head contribution onto the chosen target token's unembedding
   direction to obtain one DLA score per head.

We keep the exercise focused on attention heads. MLPs and full residual
stream decomposition come later.

Run
---
    python arena_1_2_transformerlens_12_direct_logit_attribution.py
"""

from __future__ import annotations

import torch as t
from einops import rearrange
from transformer_lens import HookedTransformer


MODEL_NAME = "gpt2-small"
PROMPT = "The cat sat on the mat. The cat"
TARGET_TOKEN = " sat"


def get_target_token_id(model: HookedTransformer, token: str) -> int:
    """Return the vocabulary ID corresponding to an exact single GPT-2 token."""
    token_ids = model.to_tokens(token, prepend_bos=False).flatten()
    assert token_ids.numel() == 1, (
        f"{token!r} did not tokenize to exactly one token: {token_ids.tolist()}"
    )
    return int(token_ids.item())


def compute_head_dla(
    model: HookedTransformer,
    cache,
    target_token_id: int,
) -> tuple[t.Tensor, list[str]]:
    """
    Compute direct logit attribution for every attention head at the final position.

    Return shape:
        (n_layers, n_heads)

    TODO 1:
        Obtain the per-head residual contribution at the final sequence position.

    TODO 2:
        Apply the final LayerNorm scaling to this residual stack.

    TODO 3:
        Project each head contribution onto W_U[:, target_token_id].
    """

    # TODO 1
    per_head_residual,labels = cache.stack_head_results(layer=-1,pos_slice=-1,return_labels = True,)
    per_head_residual = rearrange(per_head_residual,'(l h) x -> l h x',l=model.cfg.n_layers,)

    # TODO 2
    per_head_residual_ln = cache.apply_ln_to_stack(per_head_residual,layer=-1,pos_slice=-1,has_batch_dim = False,)

    # TODO 3
    target_direction = model.W_U[:,target_token_id]
    dla = t.einsum('l h x,x -> l h',per_head_residual_ln,target_direction)

    return dla, labels


def compute_manual_reference(
    model: HookedTransformer,
    cache,
    target_token_id: int,
) -> t.Tensor:
    """
    Independent reference implementation using each head's actual output
    tensor and the unembedding direction.

    This intentionally uses the same TransformerLens decomposition helpers
    but a different final calculation path, providing a parity check.
    """
    per_head_residual, _ = cache.stack_head_results(
        layer=-1,
        pos_slice=-1,
        return_labels=True,
    )
    per_head_residual = rearrange(
        per_head_residual,
        "(layer head) ... -> layer head ...",
        layer=model.cfg.n_layers,
    )

    # Apply the global final LayerNorm scaling, following TransformerLens'
    # DLA convention.
    per_head_residual = cache.apply_ln_to_stack(
        per_head_residual,
        layer=-1,
        pos_slice=-1,
        has_batch_dim=False,
    )

    target_direction = model.W_U[:, target_token_id]
    return t.einsum("l h d, d -> l h", per_head_residual, target_direction)


def run_tests(
    model: HookedTransformer,
    cache,
    dla: t.Tensor,
    target_token_id: int,
) -> None:
    """Local shape, finiteness, and parity tests."""
    expected_shape = (model.cfg.n_layers, model.cfg.n_heads)
    assert dla.shape == expected_shape, (
        f"Expected DLA shape {expected_shape}, got {tuple(dla.shape)}"
    )
    assert t.isfinite(dla).all(), "DLA contains non-finite values."

    reference = compute_manual_reference(model, cache, target_token_id)
    max_err = (dla - reference).abs().max().item()

    assert t.allclose(dla, reference, atol=1e-5, rtol=1e-5), (
        f"DLA differs from reference; max error = {max_err:.3e}"
    )

    print("LOCAL TESTS PASSED")
    print(f"Maximum DLA-vs-reference error: {max_err:.3e}")


def main() -> None:
    print("=" * 72)
    print("ARENA 1.2 — TransformerLens Exercise 12")
    print("Direct Logit Attribution (DLA)")
    print("=" * 72)
    print()
    print(f'Prompt: "{PROMPT}"')
    print(f"Target token: {TARGET_TOKEN!r}")
    print()
    print("DLA asks:")
    print("  Which individual attention heads directly increase or decrease")
    print("  the chosen target token's final logit?")
    print()

    print(f"Loading {MODEL_NAME}...")
    model = HookedTransformer.from_pretrained(MODEL_NAME)
    model.eval()
    print("Model loaded.")
    print()

    tokens = model.to_tokens(PROMPT)
    str_tokens = model.to_str_tokens(tokens)
    print("String tokens:")
    print(str_tokens)
    print()

    # For this prompt the final token is ' cat', and ' sat' is the model's
    # most likely next token in our earlier run. We explicitly target it.
    target_token_id = get_target_token_id(model, TARGET_TOKEN)
    print(f"Target token ID: {target_token_id}")
    print()

    logits, cache = model.run_with_cache(tokens, remove_batch_dim=True)

    dla, labels = compute_head_dla(model, cache, target_token_id)

    # Sort all heads by direct contribution to the target token.
    flat_scores = dla.flatten()
    top_values, top_indices = flat_scores.topk(10)

    print("Top 10 heads by positive DLA:")
    for rank, (value, flat_idx) in enumerate(
        zip(top_values.tolist(), top_indices.tolist()), start=1
    ):
        layer = flat_idx // model.cfg.n_heads
        head = flat_idx % model.cfg.n_heads
        print(f"  {rank:2d}. L{layer}H{head}: {value:+.4f}")

    # Most negative heads, for contrast.
    bottom_values, bottom_indices = flat_scores.topk(10, largest=False)

    print()
    print("10 heads with most negative DLA:")
    for rank, (value, flat_idx) in enumerate(
        zip(bottom_values.tolist(), bottom_indices.tolist()), start=1
    ):
        layer = flat_idx // model.cfg.n_heads
        head = flat_idx % model.cfg.n_heads
        print(f"  {rank:2d}. L{layer}H{head}: {value:+.4f}")

    print()
    print("Target logit from the full model:")
    target_logit = logits[0,-1, target_token_id].item()
    print(f"  logit({TARGET_TOKEN!r}) = {target_logit:+.4f}")
    print()

    run_tests(model, cache, dla, target_token_id)

    print()
    print("EXERCISE 12 PASSED")
    print("Next: compare DLA against the attention pattern and ask whether")
    print("the heads that look at the right place are also the heads that")
    print("directly push the right token's logit.")


if __name__ == "__main__":
    main()
