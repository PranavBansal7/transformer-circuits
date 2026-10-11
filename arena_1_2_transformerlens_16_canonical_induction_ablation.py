"""
ARENA 1.2 — Exercise 16
Canonical induction-head ablation

This is the exact current ARENA Section 3 ablation task, translated into a
standalone script.

Canonical model:
    callummcdougall/attn_only_2L_half

Canonical heads we expect to matter:
    L0H7   previous-token head
    L1H4   induction head
    L1H10  induction head

Experiment:
    1. Build [BOS, random_50_tokens, same_random_50_tokens].
    2. Measure clean NLL only on the induction-relevant part of the second
       copy: the final 49 next-token predictions.
    3. Zero-ablate one attention head at a time at hook_z.
    4. Measure the increase in NLL.

Why only the final 49 predictions?
    The first token of the repeated half is predicted from the last token of
    the first half, so there is no previous occurrence of the current token
    for the induction mechanism to match. The remaining 49 predictions can
    exploit the repeated context.

TODOs:
    1. Implement head_zero_ablation_hook.
    2. Implement the missing inner part of get_ablation_scores.

Run:
    python arena_1_2_transformerlens_17_canonical_induction_ablation.py
"""

from __future__ import annotations

import functools
from pathlib import Path
import sys
from typing import Callable

import torch as t
from huggingface_hub import hf_hub_download
from transformer_lens import HookedTransformer, HookedTransformerConfig, utils
from transformer_lens.hook_points import HookPoint


# -----------------------------------------------------------------------------
# Canonical ARENA model configuration (copied from the current notebook)
# -----------------------------------------------------------------------------
D_MODEL = 768
D_HEAD = 64
N_HEADS = 12
N_LAYERS = 2
N_CTX = 2048
D_VOCAB = 50278
TOKENIZER_NAME = "EleutherAI/gpt-neox-20b"
SEED = 398

REPO_ID = "callummcdougall/attn_only_2L_half"
FILENAME = "attn_only_2L_half.pth"

DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")

SEQ_LEN = 50
BATCH_SIZE = 1


# -----------------------------------------------------------------------------
# Model setup
# -----------------------------------------------------------------------------
def load_canonical_model() -> HookedTransformer:
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
        seed=SEED,
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


# -----------------------------------------------------------------------------
# Repeated-token experiment
# -----------------------------------------------------------------------------
def generate_repeated_tokens(
    model: HookedTransformer,
    seq_len: int,
    batch_size: int = 1,
) -> t.Tensor:
    """Generate [BOS, random sequence, same random sequence]."""
    t.manual_seed(0)
    prefix = t.full(
        (batch_size, 1),
        int(model.tokenizer.bos_token_id),
        dtype=t.long,
    )
    repeated_half = t.randint(
        low=0,
        high=model.cfg.d_vocab,
        size=(batch_size, seq_len),
        dtype=t.long,
    )
    return t.cat([prefix, repeated_half, repeated_half], dim=-1).to(DEVICE)


def get_log_probs(logits: t.Tensor, tokens: t.Tensor) -> t.Tensor:
    """
    Return log p(tokens[:, 1:] | tokens[:, :-1]).

    Shape:
        logits: (batch, seq, vocab)
        tokens: (batch, seq)
        result: (batch, seq-1)
    """
    logprobs = logits.log_softmax(dim=-1)
    correct_next_token_ids = tokens[:, 1:]
    return logprobs[:, :-1, :].gather(
        dim=-1,
        index=correct_next_token_ids.unsqueeze(-1),
    ).squeeze(-1)


# -----------------------------------------------------------------------------
# TODO 1
# -----------------------------------------------------------------------------
def head_zero_ablation_hook(
    z: t.Tensor,
    hook: HookPoint,
    head_index_to_ablate: int,
) -> None:
    """
    Zero-ablate exactly one attention head's output z.

    z shape:
        (batch, sequence, n_heads, d_head)

    TODO:
        Zero the entire target-head slice in-place.

    The current ARENA solution mutates z and returns None.
    """
    # TODO 1
    z[:,:,head_index_to_ablate,:] = 0


# -----------------------------------------------------------------------------
# TODO 2
# -----------------------------------------------------------------------------
def get_ablation_scores(
    model: HookedTransformer,
    tokens: t.Tensor,
    ablation_function: Callable = head_zero_ablation_hook,
) -> t.Tensor:
    """
    Return a (n_layers, n_heads) tensor.

    Entry [layer, head] = increase in NLL on the induction-relevant part
    of the repeated sequence after ablating that head.

    TODO 2:
        Implement the nested layer/head loop.

    Constraints:
        - reset hooks before the clean forward pass
        - calculate clean NLL on only the final seq_len-1 log-prob values
        - use functools.partial to fix the head index
        - attach the hook at utils.get_act_name("z", layer)
        - calculate the ablated NLL the same way
        - store loss_ablated - loss_clean
    """
    ablation_scores = t.zeros(
        (model.cfg.n_layers, model.cfg.n_heads),
        device=DEVICE,
    )

    model.reset_hooks()
    seq_len = (tokens.shape[1] - 1) // 2

    clean_logits = model(tokens, return_type="logits")
    clean_log_probs = get_log_probs(clean_logits, tokens)

    # Exact ARENA alignment: final seq_len-1 predictions only.
    clean_nll = -clean_log_probs[:, -(seq_len - 1):].mean()

    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            # TODO 2a: create a temporary hook with this head fixed.
            temp_hook_fn = functools.partial(ablation_function,head_index_to_ablate=head)

            # TODO 2b: run the model with that hook at this layer's z activation.
            ablated_logits = model.run_with_hooks(tokens,return_type="logits",fwd_hooks=[(utils.get_act_name("z",layer),temp_hook_fn)])

            # TODO 2c: compute NLL on the same final seq_len-1 positions.
            ablated_log_probs = get_log_probs(ablated_logits,tokens)
            ablated_nll = -ablated_log_probs[:,-(1-seq_len):].mean()
            # TODO 2d: store the increase in loss.
            ablation_scores[layer, head] = ablated_nll - clean_nll

            model.reset_hooks()

    return ablation_scores


# -----------------------------------------------------------------------------
# Tests / reporting
# -----------------------------------------------------------------------------
def run_tests(
    model: HookedTransformer,
    tokens: t.Tensor,
    ablation_scores: t.Tensor,
) -> None:
    assert tokens.shape == (BATCH_SIZE, 1 + 2 * SEQ_LEN)
    assert ablation_scores.shape == (N_LAYERS, N_HEADS)
    assert t.isfinite(ablation_scores).all()

    # A pure no-op test of the ablation hook's shape semantics.
    z = t.ones((2, 7, N_HEADS, D_HEAD), device=DEVICE)
    head_zero_ablation_hook(z, None, 3)
    assert t.equal(z[:, :, 3, :], t.zeros_like(z[:, :, 3, :]))
    assert t.equal(z[:, :, 0, :], t.ones_like(z[:, :, 0, :]))
    assert t.equal(z[:, :, 11, :], t.ones_like(z[:, :, 11, :]))

    print("LOCAL TESTS PASSED")


def main() -> None:
    print("=" * 72)
    print("ARENA 1.2 — Exercise 16")
    print("Canonical induction-head ablation")
    print("=" * 72)
    print()
    print("Exact model: callummcdougall/attn_only_2L_half")
    print("Device:", DEVICE)
    print("Sequence:", "[BOS, 50 random tokens, same 50 tokens]")
    print()

    model = load_canonical_model()
    tokens = generate_repeated_tokens(model, SEQ_LEN, BATCH_SIZE)

    print("MODEL CONFIG")
    print("-" * 72)
    print("n_layers =", model.cfg.n_layers)
    print("n_heads  =", model.cfg.n_heads)
    print("tokens.shape =", tuple(tokens.shape))
    print()

    print("Running 24 causal ablation experiments...")
    print("Each experiment zeroes one head at hook_z and recomputes the")
    print("induction-half NLL.")
    print()

    ablation_scores = get_ablation_scores(model, tokens)

    flat = [
        (float(ablation_scores[layer, head]), layer, head)
        for layer in range(model.cfg.n_layers)
        for head in range(model.cfg.n_heads)
    ]
    flat.sort(reverse=True)

    print("ABLATION SCORES")
    print("-" * 72)
    for score, layer, head in flat:
        print(f"L{layer}H{head:<2}  loss increase = {score:+.6f}")

    print()
    print("TOP HEADS")
    print("-" * 72)
    for rank, (score, layer, head) in enumerate(flat[:6], start=1):
        print(f"{rank:2d}. L{layer}H{head:<2}  {score:+.6f}")

    print()
    print("Canonical heads to inspect:")
    for layer, head, label in [
        (0, 7, "previous-token head"),
        (1, 4, "induction head"),
        (1, 10, "induction head"),
    ]:
        print(
            f"  L{layer}H{head:<2} ({label}): "
            f"{ablation_scores[layer, head].item():+.6f}"
        )

    print()
    run_tests(model, tokens, ablation_scores)
    print()
    print("EXERCISE 16 COMPLETE")
    print("Interpret the ranking before moving on to mean ablation.")


if __name__ == "__main__":
    main()
