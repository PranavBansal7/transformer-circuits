"""
ARENA 1.2 — Exercise 19
Decomposing L1H4's pre-softmax QK scores by component pairs

Curriculum position
-------------------
Canonical model: callummcdougall/attn_only_2L_half
Canonical circuit: L0H7 -> L1H4 / L1H10

We have already observed the L1H4 induction stripe and shown that ablating
L0H7 nearly destroys that stripe. Now we ask a more mechanistic question:
which residual-stream components contribute to L1H4's query/key dot products?

For each component y_i, its projected query/key contribution is:
    q_i = y_i @ W_Q
    k_i = y_i @ W_K

The scaled, pre-softmax score is bilinear:
    score[q, k] = q[q] dot k[k] / sqrt(d_head)

Since q = sum_i q_i and k = sum_j k_j:
    score[q, k] = sum_i sum_j (q_i[q] dot k_j[k]) / sqrt(d_head)

We will verify that summing the component-pair score matrices reconstructs the
original unmasked pre-softmax score matrix. We do NOT decompose post-softmax
attention probabilities, because softmax is nonlinear.

Current TODOs
-------------
1. pairwise_qk_score_contributions(...)
   Return a score matrix for every query-component/key-component pair.
2. sum_pairwise_scores(...)
   Sum over the two component axes to recover (batch, query, key) scores.

Run from PowerShell:
    python arena_1_2_transformerlens_19_qk_component_decomposition.py
"""

from __future__ import annotations

import math
from typing import Sequence

import torch as t
from huggingface_hub import hf_hub_download
from transformer_lens import HookedTransformer, HookedTransformerConfig


REPO_ID = "callummcdougall/attn_only_2L_half"
FILENAME = "attn_only_2L_half.pth"
SEQ_LEN = 50
TARGET_LAYER = 1
TARGET_HEAD = 4
DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")


def load_canonical_model() -> HookedTransformer:
    """Load the exact custom ARENA checkpoint using config + state dict."""
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
    """Return [BOS, random tokens, same random tokens], reproducibly."""
    t.manual_seed(0)
    first_copy = t.randint(
        low=0, high=model.cfg.d_vocab, size=(1, seq_len),
        dtype=t.long, device="cpu",
    )
    bos = t.full((1, 1), int(model.tokenizer.bos_token_id), dtype=t.long)
    tokens = t.cat([bos, first_copy, first_copy.clone()], dim=1)
    return tokens.to(DEVICE)


def build_qk_components(
    model: HookedTransformer,
    cache,
    layer: int,
    head: int,
) -> tuple[t.Tensor, t.Tensor, list[str], list[str]]:
    """Project residual/positional components into target head's Q and K.

    Returns q_components and k_components with shapes:
        (n_q_components, batch, sequence, d_head)
        (n_k_components, batch, sequence, d_head)

    For this Shortformer model, position embeddings are added to Q/K inputs,
    not to the residual stream itself. The components are token embedding,
    positional embedding, each layer-0 head's residual result, a residual
    remainder (the layer-0 output bias), and the target Q/K projection biases.
    """
    embed = cache["embed"]                     # (B, T, D)
    pos_embed = cache["pos_embed"]             # (B, T, D)
    result0 = cache["result", 0]               # (B, T, H, D)
    resid_pre_target = cache["resid_pre", layer]  # (B, T, D)

    batch, seq, d_model = embed.shape
    # This remainder equals layer 0's output bias in the canonical attention-only model.
    residual_remainder = resid_pre_target - embed - result0.sum(dim=2)

    input_components = [embed, pos_embed]
    input_names = ["token_embed", "pos_embed (Shortformer Q/K input)"]
    for h in range(model.cfg.n_heads):
        input_components.append(result0[:, :, h, :])
        input_names.append(f"L0H{h}")
    input_components.append(residual_remainder)
    input_names.append("L0 output-bias / residual remainder")
    x_components = t.stack(input_components, dim=0)  # (C, B, T, D)

    attn = model.blocks[layer].attn
    W_Q = attn.W_Q[head]  # (D, d_head)
    W_K = attn.W_K[head]  # (D, d_head)
    b_Q = attn.b_Q[head]  # (d_head,)
    b_K = attn.b_K[head]  # (d_head,)

    # Project each input component independently. Then make Q/K biases explicit
    # components so summing all components reconstructs the real q and k exactly.
    q_projected = t.einsum("cbtd,dh->cbth", x_components, W_Q)
    k_projected = t.einsum("cbtd,dh->cbth", x_components, W_K)
    q_bias_component = b_Q.view(1, 1, 1, -1).expand(1, batch, seq, -1)
    k_bias_component = b_K.view(1, 1, 1, -1).expand(1, batch, seq, -1)

    q_components = t.cat([q_projected, q_bias_component], dim=0)
    k_components = t.cat([k_projected, k_bias_component], dim=0)
    q_names = input_names + [f"L{layer}H{head} query bias"]
    k_names = input_names + [f"L{layer}H{head} key bias"]
    return q_components, k_components, q_names, k_names


def pairwise_qk_score_contributions(
    q_components: t.Tensor,
    k_components: t.Tensor,
    d_head: int,
) -> t.Tensor:
    """Return every query-component/key-component scaled score matrix.

    Args:
        q_components: (Cq, B, Q, d_head)
        k_components: (Ck, B, K, d_head)

    Returns:
        (Cq, Ck, B, Q, K)

    TODO 1:
        Use one torch.einsum to calculate all component-pair dot products,
        and divide by sqrt(d_head). Do not apply causal masking or softmax.
    """
    # TODO 1: implement the pairwise bilinear expansion.
    pair_sc = t.einsum("i b q d,j b k d -> i j b q k",q_components,k_components)/math.sqrt(d_head)
    return pair_sc


def sum_pairwise_scores(pairwise_scores: t.Tensor) -> t.Tensor:
    """Sum component-pair scores to reconstruct (batch, query, key).

    TODO 2:
        Sum over the first two axes: query-component and key-component.
    """
    # TODO 2: reduce the two component axes, preserving batch/query/key.
    return pairwise_scores.sum(dim=(0,1))


def run_synthetic_tests() -> None:
    """Prove the tensor contraction reconstructs scores on small tensors."""
    t.manual_seed(123)
    q_components = t.randn(2, 1, 5, 4)
    k_components = t.randn(3, 1, 5, 4)

    pairs = pairwise_qk_score_contributions(q_components, k_components, d_head=4)
    assert pairs.shape == (2, 3, 1, 5, 5), (
        f"Expected pairwise shape (2, 3, 1, 5, 5), got {tuple(pairs.shape)}"
    )
    total = sum_pairwise_scores(pairs)
    q_full = q_components.sum(dim=0)
    k_full = k_components.sum(dim=0)
    expected = t.einsum("bqd,bkd->bqk", q_full, k_full) / math.sqrt(4)
    assert total.shape == (1, 5, 5)
    assert t.allclose(total, expected, atol=1e-6, rtol=1e-6), (
        "Summed component-pair scores do not match the full QK score matrix."
    )
    print("SYNTHETIC QK-DECOMPOSITION TESTS PASSED")


def main() -> None:
    print("=" * 78)
    print("ARENA 1.2 — Exercise 19")
    print("L1H4 pre-softmax QK score decomposition by component pairs")
    print("=" * 78)
    print()
    print(f"Model: {REPO_ID}")
    print(f"Target head: L{TARGET_LAYER}H{TARGET_HEAD}")
    print(f"Repeated-half length: {SEQ_LEN}")
    print()
    print("We decompose the scaled, UNMASKED, pre-softmax scores.")
    print("We do not attempt to additively decompose post-softmax probabilities.")
    print()

    run_synthetic_tests()
    print("\nLoading canonical ARENA checkpoint...")
    model = load_canonical_model()
    print("Model loaded.")
    print(f"Config: {model.cfg.n_layers} layers, {model.cfg.n_heads} heads, "
          f"d_model={model.cfg.d_model}, d_head={model.cfg.d_head}")

    tokens = generate_repeated_tokens(model, SEQ_LEN)
    print(f"tokens.shape = {tuple(tokens.shape)}")

    with t.inference_mode():
        logits, cache = model.run_with_cache(tokens)
        q_components, k_components, q_names, k_names = build_qk_components(
            model, cache, TARGET_LAYER, TARGET_HEAD
        )

        q_actual = cache["q", TARGET_LAYER][:, :, TARGET_HEAD, :]
        k_actual = cache["k", TARGET_LAYER][:, :, TARGET_HEAD, :]
        q_rebuilt = q_components.sum(dim=0)
        k_rebuilt = k_components.sum(dim=0)
        q_error = (q_rebuilt - q_actual).abs().max().item()
        k_error = (k_rebuilt - k_actual).abs().max().item()
        print("\nQUERY / KEY RECONSTRUCTION")
        print("-" * 78)
        print(f"q_components.shape = {tuple(q_components.shape)}")
        print(f"k_components.shape = {tuple(k_components.shape)}")
        print(f"Max query reconstruction error = {q_error:.6e}")
        print(f"Max key reconstruction error   = {k_error:.6e}")

        pairwise_scores = pairwise_qk_score_contributions(
            q_components, k_components, model.cfg.d_head
        )
        reconstructed_scores = sum_pairwise_scores(pairwise_scores)
        original_scores = (
            t.einsum("bqd,bkd->bqk", q_actual, k_actual)
            / math.sqrt(model.cfg.d_head)
        )
        score_error = (reconstructed_scores - original_scores).abs().max().item()

        print("\nPAIRWISE PRE-SOFTMAX SCORE DECOMPOSITION")
        print("-" * 78)
        print(f"pairwise_scores.shape = {tuple(pairwise_scores.shape)}")
        print(f"reconstructed shape   = {tuple(reconstructed_scores.shape)}")
        print(f"original score shape  = {tuple(original_scores.shape)}")
        print(f"Max score reconstruction error = {score_error:.6e}")

        # Canonical induction stripe: query positions in the second copy, and
        # the token immediately after the matching token in the first copy.
        q_start, q_end = SEQ_LEN + 1, 2 * SEQ_LEN + 1
        k_start, k_end = 2, SEQ_LEN + 2
        stripe = pairwise_scores[:, :, 0, q_start:q_end, k_start:k_end]
        stripe = stripe.diagonal(dim1=-2, dim2=-1)
        mean_pair_contribution = stripe.mean(dim=-1)  # (Cq, Ck)
        total_stripe_mean = original_scores[0, q_start:q_end, k_start:k_end].diagonal().mean()
        pair_stripe_sum = mean_pair_contribution.sum()

        print("\nLARGEST COMPONENT-PAIR CONTRIBUTIONS ON THE INDUCTION STRIPE")
        print("-" * 78)
        print("Each value is the mean pre-softmax score contribution along the")
        print(f"offset -(SEQ_LEN-1) = -{SEQ_LEN - 1} stripe.")
        flat = mean_pair_contribution.flatten()
        order = t.argsort(flat.abs(), descending=True)[:12]
        n_k_components = mean_pair_contribution.shape[1]
        for rank, flat_index in enumerate(order.tolist(), start=1):
            qi = flat_index // n_k_components
            ki = flat_index % n_k_components
            value = mean_pair_contribution[qi, ki].item()
            print(f"{rank:2d}. Q[{q_names[qi]:<31}] × K[{k_names[ki]:<31}] {value:+.6f}")

        print("\nSTRIPE RECONSTRUCTION CHECK")
        print("-" * 78)
        print(f"Mean original score on induction stripe = {total_stripe_mean.item():+.6f}")
        print(f"Sum of component-pair means             = {pair_stripe_sum.item():+.6f}")
        stripe_error = (pair_stripe_sum - total_stripe_mean).abs().item()
        print(f"Absolute stripe-sum discrepancy         = {stripe_error:.6e}")

        assert q_error < 2e-3, f"Query reconstruction error too large: {q_error}"
        assert k_error < 2e-3, f"Key reconstruction error too large: {k_error}"
        assert score_error < 2e-3, f"Score reconstruction error too large: {score_error}"
        assert stripe_error < 2e-3, f"Stripe attribution does not add up: {stripe_error}"

    print("\nLOCAL TESTS PASSED")
    print("The component-pair contributions reconstruct L1H4's pre-softmax scores.")
    print("EXERCISE 19 COMPLETE")
    print("Next: interpret the Q/K component pairs and connect them to K-composition.")


if __name__ == "__main__":
    main()
