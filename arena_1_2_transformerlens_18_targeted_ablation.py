"""
ARENA 1.2 — Exercise 18
Targeted ablation: does L0H7 enable L1H4's induction pattern?

Canonical model:
    callummcdougall/attn_only_2L_half

Canonical circuit hypothesis:
    L0H7   previous-token head
    L1H4   induction head

Experiment:
    1. Build [BOS, random tokens, the same random tokens].
    2. Measure L1H4's induction-stripe score with no intervention.
    3. Zero the V activation of each layer-0 head, one at a time.
    4. Measure L1H4's induction-stripe score again.

This is targeted ablation: we measure the later head's attention pattern,
not just the final language-model loss.

Run from PowerShell:
    python arena_1_2_transformerlens_18_targeted_ablation.py

The checkpoint is loaded with ARENA's explicit config + state dict because it
is a custom checkpoint, not a TransformerLens official model-name alias.
"""

from __future__ import annotations

import functools

import torch as t
from huggingface_hub import hf_hub_download
from transformer_lens import HookedTransformer, HookedTransformerConfig


REPO_ID = "callummcdougall/attn_only_2L_half"
FILENAME = "attn_only_2L_half.pth"
SEQ_LEN = 50
INDUCTION_LAYER = 1
INDUCTION_HEAD = 4
PREVIOUS_TOKEN_LAYER = 0
EXPECTED_PREVIOUS_HEAD = 7
DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")


def load_canonical_model() -> HookedTransformer:
    """Load the exact ARENA toy-model checkpoint."""
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
    weights_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    state_dict = t.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def generate_repeated_tokens(model: HookedTransformer, seq_len: int) -> t.Tensor:
    """Generate shape (1, 1 + 2*seq_len): [BOS, random, same random]."""
    # Keep sampling reproducible and on CPU so the token sequence does not
    # depend on whether the model itself runs on CPU or GPU.
    t.manual_seed(0)
    random_tokens = t.randint(
        low=0,
        high=model.cfg.d_vocab,
        size=(1, seq_len),
        dtype=t.long,
        device="cpu",
    )
    bos_id = int(model.tokenizer.bos_token_id)
    bos = t.full((1, 1), bos_id, dtype=t.long)
    tokens = t.cat([bos, random_tokens, random_tokens.clone()], dim=1)
    return tokens.to(DEVICE)


def value_ablation_hook(
    v: t.Tensor,
    hook,
    head_index_to_ablate: int | None,
) -> t.Tensor:
    """Zero one layer-0 head's V activations, or leave V unchanged if None.

    Input shape: (batch, sequence, n_heads, d_head)

    TODO 1:
        If head_index_to_ablate is not None, zero that head at every batch
        item and sequence position. Return the resulting tensor.
    """
    # TODO 1
    if head_index_to_ablate is not None:
        v[:,:,head_index_to_ablate,:] = 0
    return v


def induction_pattern_score_hook(
    attn_pattern: t.Tensor,
    hook,
    induction_head_index: int,
    seq_len: int,
    score_store: dict[str, t.Tensor],
) -> None:
    """Save L1 induction head's mean attention on the induction stripe.

    Input shape: (batch, n_heads, query_position, key_position)

    For a repeated-half length S, the canonical induction stripe is the
    diagonal with offset -(S - 1). This hook stores one scalar score.

    TODO 2:
        Select batch item 0 and induction_head_index, extract that diagonal,
        take its mean, detach it, and store it in score_store["score"].
    """
    # TODO 2
    score_store["score"] = attn_pattern[0,induction_head_index].diagonal(1-seq_len)[2:].mean().detach()


def ablation_induction_score(
    model: HookedTransformer,
    tokens: t.Tensor,
    prev_head_index: int | None,
    induction_head_index: int = INDUCTION_HEAD,
    seq_len: int = SEQ_LEN,
) -> float:
    """Run one forward pass and return the induction score after ablation.

    `prev_head_index=None` measures the baseline without ablating a head.
    Otherwise, that head's layer-0 V activation is zeroed.

    TODO 3:
        - Create a dictionary to receive the induction score.
        - Attach value_ablation_hook at layer 0's hook_v, using functools.partial
          to bind prev_head_index.
        - Attach induction_pattern_score_hook at layer 1's hook_pattern.
        - Call model.run_with_hooks with both hooks in one forward pass.
        - Return the stored scalar score as a Python float.

    Hint: full hook names are:
        "blocks.0.attn.hook_v"
        "blocks.1.attn.hook_pattern"
    """
    # TODO 3
    score_store = {}
    temp_hook_fn = functools.partial(value_ablation_hook,head_index_to_ablate=prev_head_index)
    temp_hook_fn2 = functools.partial(induction_pattern_score_hook,induction_head_index=induction_head_index,seq_len=seq_len,score_store=score_store)
    model.run_with_hooks(tokens,return_type="logits",fwd_hooks=[("blocks.0.attn.hook_v",temp_hook_fn),("blocks.1.attn.hook_pattern",temp_hook_fn2)])
    return float(score_store["score"].item())


def run_synthetic_tests() -> None:
    """Test that a selected V head is zeroed and all others are untouched."""
    t.manual_seed(123)
    v = t.randn(2, 7, 12, 64)
    original = v.clone()
    ablated = value_ablation_hook(v.clone(), None, head_index_to_ablate=7)

    assert ablated.shape == original.shape
    assert t.count_nonzero(ablated[:, :, 7, :]) == 0
    keep_mask = t.ones(12, dtype=t.bool)
    keep_mask[7] = False
    assert t.equal(ablated[:, :, keep_mask, :], original[:, :, keep_mask, :])

    # Verify the stripe-score helper's offset on a known synthetic pattern.
    S = 5
    T = 1 + 2 * S
    fake = t.zeros(1, 12, T, T)
    fake[0, 4].diagonal(offset=-(S - 1)).fill_(1.0)
    score_store: dict[str, t.Tensor] = {}
    induction_pattern_score_hook(fake, None, 4, S, score_store)
    assert t.allclose(score_store["score"], t.tensor(1.0)), (
        "The score hook must read the expected induction diagonal."
    )

    print("SYNTHETIC TARGETED-ABLATION TESTS PASSED")


def main() -> None:
    print("=" * 76)
    print("ARENA 1.2 — Exercise 18")
    print("Targeted ablation: L0 candidate head -> L1H4 induction score")
    print("=" * 76)
    print()
    print(f"Exact model: {REPO_ID}")
    print(f"Device: {DEVICE}")
    print(f"Repeated-half length: {SEQ_LEN}")
    print(f"Target induction head: L{INDUCTION_LAYER}H{INDUCTION_HEAD}")
    print()
    print("Question:")
    print("  Does ablating a layer-0 head reduce L1H4's induction-stripe score?")
    print("  ARENA's canonical hypothesis is that L0H7 should reduce it sharply.")
    print()

    run_synthetic_tests()
    print()

    print("Loading canonical ARENA checkpoint...")
    model = load_canonical_model()
    print("Model loaded.")
    print(f"Config: {model.cfg.n_layers} layers, {model.cfg.n_heads} heads, "
          f"d_model={model.cfg.d_model}, d_head={model.cfg.d_head}")
    print()

    tokens = generate_repeated_tokens(model, SEQ_LEN)
    expected_shape = (1, 1 + 2 * SEQ_LEN)
    assert tuple(tokens.shape) == expected_shape, (
        f"Expected token shape {expected_shape}, got {tuple(tokens.shape)}"
    )
    first_copy = tokens[:, 1 : SEQ_LEN + 1]
    second_copy = tokens[:, SEQ_LEN + 1 : 2 * SEQ_LEN + 1]
    assert t.equal(first_copy, second_copy), "Repeated token halves do not match."
    print(f"tokens.shape = {tuple(tokens.shape)}")
    print()

    with t.inference_mode():
        baseline = ablation_induction_score(
            model, tokens, prev_head_index=None, induction_head_index=INDUCTION_HEAD
        )
        print(f"Baseline L1H4 induction score: {baseline:.5f}")
        print()

        results: list[tuple[float, int, float]] = []
        print("Ablating each layer-0 head's V activation:")
        print("-" * 76)
        for head in range(model.cfg.n_heads):
            score = ablation_induction_score(
                model,
                tokens,
                prev_head_index=head,
                induction_head_index=INDUCTION_HEAD,
            )
            change = score - baseline
            results.append((change, head, score))
            print(
                f"L0H{head:02d}: score={score:.5f}  "
                f"change={change:+.5f}"
            )

    print()
    print("RANKED BY DROP IN L1H4 INDUCTION SCORE")
    print("-" * 76)
    for rank, (change, head, score) in enumerate(sorted(results), start=1):
        print(f"{rank:2d}. L0H{head:02d}: {baseline:.5f} -> {score:.5f} "
              f"(change {change:+.5f})")

    target = next(item for item in results if item[1] == EXPECTED_PREVIOUS_HEAD)
    print()
    print("CANONICAL CHECK")
    print("-" * 76)
    print(f"Ablating L0H7: {baseline:.5f} -> {target[2]:.5f} "
          f"(change {target[0]:+.5f})")
    print()
    print("Interpretation:")
    print("  If L0H7 sharply reduces L1H4's induction score while most other")
    print("  heads have much smaller effects, that supports the hypothesis that")
    print("  L0H7 supplies information L1H4 uses for induction on this task.")
    print("  This is targeted causal evidence, not proof that no alternative")
    print("  pathways or redundant mechanisms exist.")
    print()
    print("EXERCISE 18 COMPLETE")


if __name__ == "__main__":
    main()
