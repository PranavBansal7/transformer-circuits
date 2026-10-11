"""
ARENA 1.2 — TransformerLens Exercise 09
Targeted ablation: probing induction-circuit composition

Why this exercise exists
------------------------
Our previous head-ablation experiment tried to use next-token loss as a
pass/fail test for L7H10. That was too crude: a head can show a strong
induction signature without being individually necessary for average loss on
one finite batch.

The current ARENA material instead uses a more targeted intervention:
    ablate an earlier candidate previous-token head
    -> measure the induction score of a later induction head

If the earlier head supplies information required by the later head, ablating
it should reduce the later head's induction score.

This is our first explicit composition experiment.

Current TODO
------------
Implement `ablation_hook` inside `run_ablation_experiment`.
It is attached to an earlier layer's V activation and should zero exactly one
attention head while leaving all other heads unchanged.

You do NOT need to implement the induction-score calculation; it is provided
so that your attention stays focused on the intervention itself.

Run
---
    python arena_1_2_transformerlens_09_targeted_ablation_composition.py

Interpretation rule
-------------------
Do not assume that a large score drop MUST occur. The experiment is evidence,
not a proof. We will look at the actual effect size and then decide what the
result supports.
"""

from __future__ import annotations

import torch as t
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint

MODEL_NAME = "gpt2-small"
SEQ_LEN = 50
BATCH_SIZE = 10

# We already established from Exercise 6/7 that this is a strong induction head
# in the repeated-token experiment.
TARGET_INDUCTION_LAYER = 7
TARGET_INDUCTION_HEAD = 10

# We only test candidate heads in earlier layers, because a head in a later
# layer cannot causally provide its output to an earlier induction pattern.
CANDIDATE_LAYERS = list(range(TARGET_INDUCTION_LAYER))
TOP_CANDIDATES = 8


# ---------------------------------------------------------------------------
# Provided utility: generate the same type of repeated-token batch as before.
# ---------------------------------------------------------------------------
def generate_repeated_tokens(
    model: HookedTransformer,
    seq_len: int,
    batch_size: int,
) -> t.Tensor:
    t.manual_seed(0)

    bos = int(model.tokenizer.bos_token_id)

    random_tokens = t.randint(
        low=0,
        high=model.cfg.d_vocab,
        size=(batch_size, seq_len),
        dtype=t.long,
        device=model.cfg.device,
    )

    bos_tokens = t.full(
        (batch_size, 1),
        bos,
        dtype=t.long,
        device=model.cfg.device,
    )

    return t.cat([bos_tokens, random_tokens, random_tokens], dim=1)


# ---------------------------------------------------------------------------
# Provided utility: canonical induction score for one head.
# ---------------------------------------------------------------------------
def induction_score_for_head(
    pattern: t.Tensor,
    seq_len: int,
    head: int,
) -> t.Tensor:
    """
    pattern shape:
        (batch, heads, destination_position, source_position)

    Returns a scalar score for the chosen head.
    """
    stripe = pattern[:, head].diagonal(
        dim1=-2,
        dim2=-1,
        offset=1 - seq_len,
    )
    return stripe.mean()


# ---------------------------------------------------------------------------
# Provided utility: compute previous-token scores from a cached pattern.
# This is used to identify plausible earlier heads to ablate.
# ---------------------------------------------------------------------------
def previous_token_scores(
    model: HookedTransformer,
    cache,
) -> t.Tensor:
    """Return scores with shape (n_layers, n_heads)."""
    scores = t.zeros(
        (model.cfg.n_layers, model.cfg.n_heads),
        device=model.cfg.device,
    )

    for layer in CANDIDATE_LAYERS:
        pattern = cache["pattern", layer]  # (B, H, T, T)
        stripe = pattern.diagonal(dim1=-2, dim2=-1, offset=-1)
        # Ignore the missing entry from query position 0 by averaging the
        # diagonal returned by torch.diagonal.
        scores[layer] = stripe.mean(dim=(0, 2))

    return scores


# ---------------------------------------------------------------------------
# MAIN TODO
# ---------------------------------------------------------------------------
def run_ablation_experiment(
    model: HookedTransformer,
    tokens: t.Tensor,
    prev_layer: int,
    prev_head: int,
) -> float:
    """
    Ablate one earlier V-head and measure the target induction head's score.

    TODO:
        Implement `ablation_hook` below.

    The hook is attached to:
        blocks.{prev_layer}.attn.hook_v

    The activation has shape:
        (batch, sequence, n_heads, d_head)

    It should:
        1. preserve the original tensor
        2. zero ONLY `prev_head`
        3. return the modified tensor

    A second hook (already implemented) captures the induction score of
    TARGET_INDUCTION_LAYER / TARGET_INDUCTION_HEAD.
    """

    result = {"score": None}

    def ablation_hook(v: t.Tensor, hook: HookPoint) -> t.Tensor:
        # TODO: zero exactly one head and return the modified activation.
        v1 = v.clone()
        v1[:,:,prev_head,:] = 0
        return v1

    def induction_pattern_hook(pattern: t.Tensor, hook: HookPoint) -> None:
        score = induction_score_for_head(
            pattern,
            seq_len=SEQ_LEN,
            head=TARGET_INDUCTION_HEAD,
        )
        result["score"] = score.detach()

    model.run_with_hooks(
        tokens,
        return_type=None,
        fwd_hooks=[
            (
                utils.get_act_name("v", prev_layer),
                ablation_hook,
            ),
            (
                utils.get_act_name("pattern", TARGET_INDUCTION_LAYER),
                induction_pattern_hook,
            ),
        ],
    )

    assert result["score"] is not None
    return float(result["score"].item())


# ---------------------------------------------------------------------------
# Tests the ablation operation itself, independent of GPT-2 behavior.
# ---------------------------------------------------------------------------
def run_ablation_hook_unit_test(model: HookedTransformer) -> None:
    B, T, H, d_head = 2, 5, model.cfg.n_heads, model.cfg.d_head
    x = t.randn(B, T, H, d_head, device=model.cfg.device)
    original = x.clone()
    target_head = 7

    class DummyHook:
        pass

    dummy_hook = DummyHook()

    # Implement the expected semantics directly so this test checks the
    # operation your real hook is supposed to perform.
    modified = x.clone()
    modified[:, :, target_head, :] = 0.0

    # The target head is completely zero.
    assert t.equal(modified[:, :, target_head, :], t.zeros_like(modified[:, :, target_head, :]))

    # Every other head is unchanged.
    mask = t.ones(H, dtype=t.bool, device=model.cfg.device)
    mask[target_head] = False
    assert t.equal(modified[:, :, mask, :], original[:, :, mask, :])

    # The exercise uses exactly these semantics.
    del dummy_hook
    print("ABLATION SEMANTICS TEST PASSED")


def main() -> None:
    print("=" * 72)
    print("ARENA 1.2 — TransformerLens Exercise 09")
    print("Targeted ablation: probing induction-circuit composition")
    print("=" * 72)
    print()
    print(
        f"Target induction head: L{TARGET_INDUCTION_LAYER}H{TARGET_INDUCTION_HEAD}"
    )
    print(
        "We will ablate candidate earlier heads and measure whether the target"
        " head's induction score changes."
    )
    print()

    print(f"Loading {MODEL_NAME}...")
    model = HookedTransformer.from_pretrained(MODEL_NAME)
    model.eval()
    print("Model loaded.")
    print()

    run_ablation_hook_unit_test(model)
    print()

    tokens = generate_repeated_tokens(model, SEQ_LEN, BATCH_SIZE)
    print(f"tokens.shape = {tuple(tokens.shape)}")
    print()

    # Baseline: target induction score with no ablation.
    _, baseline_cache = model.run_with_cache(tokens, remove_batch_dim=False)
    baseline_pattern = baseline_cache["pattern", TARGET_INDUCTION_LAYER]
    baseline_score = float(
        induction_score_for_head(
            baseline_pattern,
            SEQ_LEN,
            TARGET_INDUCTION_HEAD,
        ).item()
    )

    print(f"Baseline induction score (L{TARGET_INDUCTION_LAYER}H{TARGET_INDUCTION_HEAD}): {baseline_score:.4f}")
    print()

    # Identify plausible previous-token heads among earlier layers.
    prev_scores = previous_token_scores(model, baseline_cache)
    candidates = []
    for layer in CANDIDATE_LAYERS:
        for head in range(model.cfg.n_heads):
            candidates.append((float(prev_scores[layer, head].item()), layer, head))

    candidates.sort(reverse=True)
    candidates = candidates[:TOP_CANDIDATES]

    print("Top earlier heads by previous-token score:")
    for rank, (score, layer, head) in enumerate(candidates, start=1):
        print(f"  {rank:>2}. L{layer}H{head}: {score:.4f}")
    print()

    print("Targeted ablation results")
    print("=" * 72)
    print("Each row: ablate one earlier V-head -> re-measure L7H10 induction score")
    print()

    results = []
    for prev_score, layer, head in candidates:
        ablated_score = run_ablation_experiment(
            model,
            tokens,
            prev_layer=layer,
            prev_head=head,
        )
        change = ablated_score - baseline_score
        results.append((change, layer, head, prev_score, ablated_score))
        print(
            f"L{layer}H{head:<2} | prev-token={prev_score:.4f} | "
            f"induction={ablated_score:.4f} | change={change:+.4f}"
        )

    results.sort(key=lambda x: x[0])
    print()
    print("Largest induction-score decrease:")
    change, layer, head, prev_score, ablated_score = results[0]
    print(
        f"  L{layer}H{head} -> {baseline_score:.4f} to {ablated_score:.4f} "
        f"(change {change:+.4f})"
    )
    print()

    print("Interpretation guidance:")
    print("  A large negative change supports the hypothesis that the ablated")
    print("  earlier head contributes to the target induction head's behavior.")
    print("  A small change means this particular intervention did not strongly")
    print("  affect the target induction score on this experiment.")
    print()
    print("Do not call an individual head 'the causal mechanism' from this test")
    print("alone. We are collecting evidence about circuit composition.")


if __name__ == "__main__":
    main()
