"""
ARENA 1.2 — Exercise 17
Canonical induction-circuit mean ablation

This is a standalone version of the current ARENA mean-ablation exercise.
It reuses the canonical toy checkpoint and the completed scoring infrastructure
from Exercise 16; the only implementation TODO is the mean-ablation hook.

Experiment:
    1. Generate 10 examples of [BOS, random 50 tokens, same 50 tokens].
    2. Compute clean next-token NLL on the induction-relevant final 49 targets.
    3. For every layer/head, replace that head's z activation with its mean
       across the batch (separately at every sequence position and feature).
    4. Measure the loss increase.

Why mean ablation?
    Zero ablation substitutes an all-zero vector. Mean ablation instead removes
    example-specific variation while preserving an average contribution drawn
    from activations on this batch. It is an alternative intervention, not a
    universally better or perfectly distribution-preserving intervention.

Run:
    python arena_1_2_transformerlens_17_canonical_mean_ablation.py
"""

from __future__ import annotations

import functools
from typing import Callable

import torch as t
from huggingface_hub import hf_hub_download
from transformer_lens import HookedTransformer, HookedTransformerConfig, utils
from transformer_lens.hook_points import HookPoint


# Canonical ARENA toy model: callummcdougall/attn_only_2L_half
D_MODEL = 768
D_HEAD = 64
N_HEADS = 12
N_LAYERS = 2
N_CTX = 2048
D_VOCAB = 50278
TOKENIZER_NAME = "EleutherAI/gpt-neox-20b"
MODEL_SEED = 398
REPO_ID = "callummcdougall/attn_only_2L_half"
FILENAME = "attn_only_2L_half.pth"

DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")
SEQ_LEN = 50
BATCH_SIZE = 10


def load_canonical_model() -> HookedTransformer:
    """Load the canonical ARENA checkpoint using its explicit config + state dict."""
    cfg = HookedTransformerConfig(
        d_model=D_MODEL,
        d_head=D_HEAD,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        n_ctx=N_CTX,
        d_vocab=D_VOCAB,
        attention_dir="causal",
        attn_only=True,
        tokenizer_name=TOKENIZER_NAME,
        seed=MODEL_SEED,
        use_attn_result=True,
        normalization_type=None,
        positional_embedding_type="shortformer",
    )

    print("Loading canonical ARENA checkpoint...")
    model = HookedTransformer(cfg)
    weights_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    state_dict = t.load(weights_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def generate_repeated_tokens(
    model: HookedTransformer,
    seq_len: int,
    batch_size: int = 1,
) -> t.Tensor:
    """Generate [BOS, random sequence, same random sequence] for each batch row."""
    t.manual_seed(0)
    bos = t.full(
        (batch_size, 1),
        int(model.tokenizer.bos_token_id),
        dtype=t.long,
    )
    first_half = t.randint(
        low=0,
        high=model.cfg.d_vocab,
        size=(batch_size, seq_len),
        dtype=t.long,
    )
    return t.cat([bos, first_half, first_half], dim=1).to(DEVICE)


def get_log_probs(logits: t.Tensor, tokens: t.Tensor) -> t.Tensor:
    """Return log p(tokens[:, 1:] | tokens[:, :-1]), shape (batch, seq-1)."""
    logprobs = logits.log_softmax(dim=-1)
    correct_next_token_ids = tokens[:, 1:]
    return logprobs[:, :-1, :].gather(
        dim=-1,
        index=correct_next_token_ids.unsqueeze(-1),
    ).squeeze(-1)


def head_zero_ablation_hook(
    z: t.Tensor,
    hook: HookPoint,
    head_index_to_ablate: int,
) -> None:
    """Completed from Exercise 16: zero one head's z activation in-place."""
    z[:, :, head_index_to_ablate, :] = 0.0


def head_mean_ablation_hook(
    z: t.Tensor,
    hook: HookPoint,
    head_index_to_ablate: int,
) -> None:
    """Replace the selected head's z values with the mean across examples.

    z shape: (batch, sequence, n_heads, d_head)

    TODO:
        Replace z[:, :, head_index_to_ablate, :] with its mean across the
        batch dimension. Preserve sequence and d_head dimensions so PyTorch
        broadcasts the same per-position mean back to every example.

        Mutate `z` in-place and return None, matching ARENA's hook convention.
    """
    # TODO: implement mean ablation across the batch dimension only.
    z[:,:,head_index_to_ablate,:] = z[:,:,head_index_to_ablate,:].mean(dim=0)


def get_ablation_scores(
    model: HookedTransformer,
    tokens: t.Tensor,
    ablation_function: Callable = head_mean_ablation_hook,
) -> t.Tensor:
    """Return (n_layers, n_heads) increases in induction-half NLL."""
    scores = t.zeros((model.cfg.n_layers, model.cfg.n_heads), device=DEVICE)
    seq_len = (tokens.shape[1] - 1) // 2

    model.reset_hooks()
    with t.inference_mode():
        clean_logits = model(tokens, return_type="logits")
        clean_log_probs = get_log_probs(clean_logits, tokens)
        # Match ARENA: only the final seq_len - 1 target predictions.
        clean_nll = -clean_log_probs[:, -(seq_len - 1):].mean()

        for layer in range(model.cfg.n_layers):
            for head in range(model.cfg.n_heads):
                hook_fn = functools.partial(
                    ablation_function,
                    head_index_to_ablate=head,
                )
                ablated_logits = model.run_with_hooks(
                    tokens,
                    fwd_hooks=[(utils.get_act_name("z", layer), hook_fn)],
                    return_type="logits",
                )
                ablated_log_probs = get_log_probs(ablated_logits, tokens)
                ablated_nll = -ablated_log_probs[:, -(seq_len - 1):].mean()
                scores[layer, head] = ablated_nll - clean_nll
                model.reset_hooks()

    return scores


def run_synthetic_hook_test() -> None:
    """Ensure only the chosen head is changed and batch mean is broadcast."""
    batch, seq, n_heads, d_head = 4, 3, N_HEADS, D_HEAD
    z = t.arange(
        batch * seq * n_heads * d_head,
        dtype=t.float32,
        device=DEVICE,
    ).reshape(batch, seq, n_heads, d_head)
    original = z.clone()
    target_head = 7

    head_mean_ablation_hook(z, None, target_head)

    expected_mean = original[:, :, target_head, :].mean(dim=0, keepdim=True)
    expected = expected_mean.expand(batch, -1, -1)
    assert t.allclose(z[:, :, target_head, :], expected)

    # Every non-target head must remain unchanged.
    for head in range(n_heads):
        if head != target_head:
            assert t.equal(z[:, :, head, :], original[:, :, head, :])

    print("SYNTHETIC MEAN-ABLATION TEST PASSED")


def run_tests(model: HookedTransformer, tokens: t.Tensor, scores: t.Tensor) -> None:
    assert tokens.shape == (BATCH_SIZE, 1 + 2 * SEQ_LEN)
    assert t.equal(tokens[:, 1:1 + SEQ_LEN], tokens[:, 1 + SEQ_LEN:])
    assert scores.shape == (N_LAYERS, N_HEADS)
    assert t.isfinite(scores).all()

    print("LOCAL TESTS PASSED")
    print("Mean-ablation shape, repetition, and finite-score checks passed.")


def main() -> None:
    print("=" * 72)
    print("ARENA 1.2 — Exercise 17")
    print("Canonical induction-circuit mean ablation")
    print("=" * 72)
    print()
    print("Exact model: callummcdougall/attn_only_2L_half")
    print("Device:", DEVICE)
    print(f"Sequence: [BOS, {SEQ_LEN} random tokens, same {SEQ_LEN} tokens]")
    print("Batch size:", BATCH_SIZE)
    print()
    print("Intervention:")
    print("  For each layer/head, replace z for that head with its batch mean.")
    print("  Sequence-position and d_head dimensions are preserved.")
    print("  Measure NLL increase only on the induction-relevant final 49 targets.")
    print()

    run_synthetic_hook_test()
    print()

    model = load_canonical_model()
    tokens = generate_repeated_tokens(model, seq_len=SEQ_LEN, batch_size=BATCH_SIZE)

    print("MODEL CONFIG")
    print("-" * 72)
    print(f"n_layers = {model.cfg.n_layers}")
    print(f"n_heads  = {model.cfg.n_heads}")
    print(f"tokens.shape = {tuple(tokens.shape)}")
    print()

    mean_scores = get_ablation_scores(model, tokens, ablation_function=head_mean_ablation_hook)
    ranked = sorted(
        [
            (float(mean_scores[layer, head].item()), layer, head)
            for layer in range(model.cfg.n_layers)
            for head in range(model.cfg.n_heads)
        ],
        reverse=True,
    )

    print("MEAN-ABLATION SCORES")
    print("-" * 72)
    for score, layer, head in ranked:
        print(f"L{layer}H{head:<2}  loss increase = {score:+.6f}")

    print()
    print("TOP HEADS")
    print("-" * 72)
    for rank, (score, layer, head) in enumerate(ranked[:8], start=1):
        print(f"{rank:2d}. L{layer}H{head:<2}  {score:+.6f}")

    print()
    print("Canonical circuit heads:")
    for layer, head, label in [
        (0, 7, "previous-token head"),
        (1, 4, "induction head"),
        (1, 10, "induction head"),
    ]:
        print(f"  L{layer}H{head:<2} ({label}): {mean_scores[layer, head].item():+.6f}")

    print()
    print("Interpretation reminder:")
    print("  Mean ablation removes example-specific variation, not all output.")
    print("  It is still an intervention with limitations; don't infer a circuit")
    print("  from one number without checking the pattern across heads.")
    print()

    run_tests(model, tokens, mean_scores)
    print("\nEXERCISE 17 COMPLETE")
    print("Next: compare zero vs mean ablation and move to targeted ablation.")


if __name__ == "__main__":
    main()
