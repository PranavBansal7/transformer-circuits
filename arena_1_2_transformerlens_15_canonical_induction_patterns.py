"""
ARENA 1.2 — Exercise 15
Canonical induction-head attention patterns

We are intentionally filling the gap in our curriculum here.

Goal:
    Run the exact ARENA attn_only_2L_half model on a repeated random sequence
    and inspect the attention pattern of the two-layer model before writing an
    induction-head detector.

Canonical ARENA setup:
    Model checkpoint: callummcdougall/attn_only_2L_half
    L0H7: previous-token head
    L1H4, L1H10: strongly induction-y heads
    L1H6: weakly induction-y head

Input structure:
    [BOS, A, B, C, ..., X, A, B, C, ..., X]

For repeated-half length N, an induction head should attend from a destination
in the second copy to the token immediately after the matching token in the
first copy. With BOS included, this corresponds to diagonal offset:
    -(N - 1)

This exercise is deliberately observational. We are NOT modifying activations.

Current TODOs:
    1. Build the repeated-token input using the exact ARENA indexing convention.
    2. Compute the induction-stripe score for one head.
    3. Scan all heads in the model and report their scores.

Run:
    python arena_1_2_transformerlens_15_canonical_induction_patterns.py
"""

from __future__ import annotations

import os
from pathlib import Path

import torch as t
from huggingface_hub import hf_hub_download
from transformer_lens import ActivationCache, HookedTransformer, HookedTransformerConfig


REPO_ID = "callummcdougall/attn_only_2L_half"
FILENAME = "attn_only_2L_half.pth"

SEQ_LEN = 20
EXPECTED_LAYER_1_HEADS = {4, 10}
WEAK_HEAD = 6


def load_canonical_model() -> HookedTransformer:
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

    model_dir = os.environ.get("ARENA_MODEL_DIR")
    if model_dir:
        weights_path = Path(model_dir) / FILENAME
    else:
        weights_path = Path(
            hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
        )

    state_dict = t.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def generate_repeated_tokens(
    model: HookedTransformer,
    seq_len: int,
) -> t.Tensor:
    """
    Generate [BOS, rand_tokens, rand_tokens].

    TODO 1:
        Use torch.manual_seed(0), generate seq_len random vocabulary IDs,
        and concatenate BOS + first copy + second copy.

    Expected shape:
        (1, 1 + 2 * seq_len)
    """
    # TODO 1
    t.manual_seed(0)
    bos_id = model.tokenizer.bos_token_id
    bos_tensor = t.full(size=(1,1),fill_value=bos_id,dtype=t.long)
    token_tensor = t.randint(low=0,high=model.cfg.d_vocab,size=(1,seq_len),dtype=t.long)
    rep_tok = t.cat([bos_tensor,token_tensor,token_tensor],dim=1)
    return rep_tok


def induction_stripe_score(
    pattern: t.Tensor,
    seq_len: int,
) -> t.Tensor:
    """
    Compute the mean attention paid to the canonical induction stripe.

    Input:
        pattern: (T, T) for one head

    Output:
        scalar tensor

    The second-copy destination positions are:
        seq_len + 1, ..., 2 * seq_len

    The corresponding induction source positions are:
        2, ..., seq_len + 1

    TODO 2:
        Use the correct offset-diagonal indexing, rather than averaging the
        whole matrix or using the ordinary diagonal.
    """
    # TODO 2
    return pattern.diagonal(1-seq_len)[2:].mean()


def all_head_induction_scores(
    cache: ActivationCache,
    seq_len: int,
) -> t.Tensor:
    """
    Return a score matrix with shape (n_layers, n_heads).

    TODO 3:
        Loop over layers and heads, retrieve cache["pattern", layer][head],
        and apply induction_stripe_score.
    """
    # TODO 3
    nl = cache.model.cfg.n_layers
    nh = cache.model.cfg.n_heads
    score = t.empty((nl,nh))
    for i in range(nl):
        for j in range(nh):
            sc = cache["pattern",i][j]
            score[i,j] = induction_stripe_score(sc,seq_len)
    return score


def run_tests(model: HookedTransformer, tokens: t.Tensor, scores: t.Tensor) -> None:
    """Tests the indexing and expected canonical behavior."""
    expected_tokens_shape = (1, 1 + 2 * SEQ_LEN)
    assert tokens.shape == expected_tokens_shape, (
        f"Expected tokens shape {expected_tokens_shape}, got {tuple(tokens.shape)}"
    )

    first_copy = tokens[:, 1 : SEQ_LEN + 1]
    second_copy = tokens[:, SEQ_LEN + 1 : 2 * SEQ_LEN + 1]
    assert t.equal(first_copy, second_copy), "The two random copies differ."
    assert int(tokens[0, 0]) == int(model.tokenizer.bos_token_id)

    assert scores.shape == (model.cfg.n_layers, model.cfg.n_heads)
    assert t.isfinite(scores).all()
    assert t.all((scores >= 0) & (scores <= 1))

    # Canonical model behavior from the current ARENA exercise:
    # L1H4 and L1H10 are strongly induction-y.
    assert scores[1, 4] > 0.4, f"L1H4 score unexpectedly low: {scores[1, 4].item():.4f}"
    assert scores[1, 10] > 0.8, f"L1H10 score unexpectedly low: {scores[1, 10].item():.4f}"

    # L1H6 is described by ARENA as weakly induction-y.
    assert scores[1, WEAK_HEAD] < 0.5, (
        f"L1H6 score unexpectedly high: {scores[1, WEAK_HEAD].item():.4f}"
    )

    print("LOCAL TESTS PASSED")


def main() -> None:
    print("=" * 72)
    print("ARENA 1.2 — Exercise 15")
    print("Canonical induction-head attention patterns")
    print("=" * 72)
    print()
    print("Exact model: attn_only_2L_half")
    print("The goal is to OBSERVE the induction stripe before detecting it.")
    print()

    print("Loading canonical ARENA checkpoint...")
    model = load_canonical_model()
    print("Model loaded.")
    print()

    tokens = generate_repeated_tokens(model, SEQ_LEN)
    _, cache = model.run_with_cache(tokens, remove_batch_dim=True)

    print(f"tokens.shape = {tuple(tokens.shape)}")
    print(f"pattern shape per head = {tuple(cache['pattern', 0][0].shape)}")
    print()

    print("String tokens (first 15):")
    print(model.to_str_tokens(tokens[0])[:15])
    print()

    scores = all_head_induction_scores(cache, SEQ_LEN)

    print("INDUCTION STRIPE SCORES")
    print("-" * 72)
    for layer in range(model.cfg.n_layers):
        row = "  ".join(
            f"H{head}: {scores[layer, head].item():.4f}"
            for head in range(model.cfg.n_heads)
        )
        print(f"L{layer}: {row}")
    print()

    flat = [
        (scores[layer, head].item(), layer, head)
        for layer in range(model.cfg.n_layers)
        for head in range(model.cfg.n_heads)
    ]
    flat.sort(reverse=True)

    print("TOP HEADS")
    print("-" * 72)
    for rank, (score, layer, head) in enumerate(flat[:8], start=1):
        print(f"{rank:2d}. L{layer}H{head}: {score:.4f}")
    print()

    # Give the learner concrete positions to inspect.
    pattern = cache["pattern", 1]
    query_positions = [SEQ_LEN + 1, SEQ_LEN + 5, 2 * SEQ_LEN]
    source_positions = [q - (SEQ_LEN - 1) for q in query_positions]

    print("CANONICAL INDUCTION LINKS FOR L1H4")
    print("-" * 72)
    for q, k in zip(query_positions, source_positions):
        value = pattern[4, q, k].item()
        top_k = int(pattern[4, q].argmax().item())
        print(
            f"query index {q:2d} -> expected key {k:2d} | "
            f"actual top key {top_k:2d} | attention={value:.4f}"
        )
    print()

    run_tests(model, tokens, scores)

    print()
    print("EXERCISE 15 PASSED")
    print("Next: implement the canonical induction-head detector.")


if __name__ == "__main__":
    main()
