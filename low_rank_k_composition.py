"""
ARENA 1.2 — Exercise 20
Canonical induction-circuit K-composition scores

Model:
    callummcdougall/attn_only_2L_half

Circuit hypothesis:
    L0H7 (previous-token head) -> L1H4 / L1H10 (induction heads)

For an earlier head A and later head B, define
    W_OV^A = W_V^A @ W_O^A
    W_QK^B = W_Q^B @ (W_K^B).T

K-composition compares W_OV^A with (W_QK^B).T:
    comp(A, B) = || W_OV^A @ (W_QK^B).T ||_F
                 / (||W_OV^A||_F * ||W_QK^B||_F)

To make the all-head scan efficient on CPU, this script keeps the factors
low-rank and evaluates Frobenius norms using small d_head x d_head Gram
matrices. It does not materialize full composed 768 x 768 matrices.

Current TODO:
    Implement `k_composition_score` using the explicitly documented shapes.

Run:
    python arena_1_2_transformerlens_20_canonical_k_composition.py
"""
from __future__ import annotations

import math
import torch as t
from huggingface_hub import hf_hub_download
from transformer_lens import HookedTransformer, HookedTransformerConfig

REPO_ID = "callummcdougall/attn_only_2L_half"
FILENAME = "attn_only_2L_half.pth"
DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")
RANDOM_BASELINE = 1 / math.sqrt(768)


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
    path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    state_dict = t.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def k_composition_score(
    W_V_A: t.Tensor,
    W_O_A: t.Tensor,
    W_Q_B: t.Tensor,
    W_K_B: t.Tensor,
) -> t.Tensor:
    """Compute normalized K-composition score without full d_model matrices.

    Inputs use TransformerLens' row-vector convention:
        W_V_A: (d_model, d_head)  earlier head's value projection
        W_O_A: (d_head, d_model)  earlier head's output projection
        W_Q_B: (d_model, d_head)  later head's query projection
        W_K_B: (d_model, d_head)  later head's key projection

    The target score is
        || (W_V_A W_O_A) (W_Q_B W_K_B.T).T ||_F
        ------------------------------------------------
        ||W_V_A W_O_A||_F * ||W_Q_B W_K_B.T||_F

    TODO:
        Compute this score efficiently using only small d_head x d_head
        Gram matrices. Useful identities:

        ||V O||_F^2 = tr((V.T @ V) @ (O @ O.T))
        ||Q K.T||_F^2 = tr((Q.T @ Q) @ (K.T @ K))

        For the numerator, (V O) (Q K.T).T = V (O K) Q.T.
        Let L = V @ (O @ K), so ||L Q.T||_F^2 can be computed from
        L.T @ L and Q.T @ Q.

        Return a scalar tensor. Clamp squared norms at zero before sqrt to
        guard against tiny negative roundoff from trace arithmetic.
    """
    # TODO: implement the low-rank K-composition score described above.
    q_gram = W_Q_B.T @ W_Q_B
    kt_gram = W_K_B.T @ W_K_B
    v_gram = W_V_A.T @ W_V_A
    o_gram = W_O_A @ W_O_A.T
    V_O = t.sqrt(t.clamp(t.trace(v_gram @ o_gram),min=0.0))
    Q_K = t.sqrt(t.clamp(t.trace(q_gram @ kt_gram),min=0.0))
    den = V_O * Q_K
    L = W_V_A @ (W_O_A @ W_K_B)
    l_gram = L.T @ L
    num = t.sqrt(t.clamp(t.trace(l_gram @ q_gram),min=0.0))
    return num / den


def direct_k_composition_score(
    W_V_A: t.Tensor,
    W_O_A: t.Tensor,
    W_Q_B: t.Tensor,
    W_K_B: t.Tensor,
) -> t.Tensor:
    """Slow, transparent reference implementation for small synthetic tests."""
    W_OV_A = W_V_A @ W_O_A
    W_QK_B = W_Q_B @ W_K_B.T
    numerator = t.linalg.matrix_norm(W_OV_A @ W_QK_B.T, ord="fro")
    denominator = t.linalg.matrix_norm(W_OV_A, ord="fro") * t.linalg.matrix_norm(W_QK_B, ord="fro")
    return numerator / denominator


def run_synthetic_tests() -> None:
    t.manual_seed(2026)
    d_model, d_head = 13, 4
    V = t.randn(d_model, d_head, dtype=t.float64)
    O = t.randn(d_head, d_model, dtype=t.float64)
    Q = t.randn(d_model, d_head, dtype=t.float64)
    K = t.randn(d_model, d_head, dtype=t.float64)

    score_fast = k_composition_score(V, O, Q, K)
    score_direct = direct_k_composition_score(V, O, Q, K)
    assert score_fast.ndim == 0, f"Expected scalar score, got {tuple(score_fast.shape)}"
    assert t.isfinite(score_fast)
    assert t.allclose(score_fast, score_direct, atol=1e-9, rtol=1e-8), (
        f"Low-rank score disagrees with direct reference: {score_fast.item():.10f} "
        f"vs {score_direct.item():.10f}"
    )
    print("SYNTHETIC K-COMPOSITION TEST PASSED")


def extract_head_weights(model: HookedTransformer, layer: int, head: int):
    attn = model.blocks[layer].attn
    return attn.W_V[head], attn.W_O[head], attn.W_Q[head], attn.W_K[head]


def main() -> None:
    print("=" * 76)
    print("ARENA 1.2 — Exercise 20")
    print("Canonical induction-circuit K-composition scores")
    print("=" * 76)
    print(f"Model: {REPO_ID}")
    print(f"Device: {DEVICE}")
    print("Earlier heads: layer 0; later heads: layer 1")
    print("Expected circuit candidates: L0H7 -> L1H4 / L1H10")
    print(f"Approx. random baseline: 1/sqrt(d_model) = {RANDOM_BASELINE:.4f}")
    print()

    run_synthetic_tests()
    print("\nLoading canonical ARENA checkpoint...")
    model = load_canonical_model()
    print("Model loaded.")
    print(f"Config: {model.cfg.n_layers} layers, {model.cfg.n_heads} heads, "
          f"d_model={model.cfg.d_model}, d_head={model.cfg.d_head}")

    scores = t.empty((model.cfg.n_heads, model.cfg.n_heads), dtype=t.float64)
    with t.inference_mode():
        layer0 = [extract_head_weights(model, 0, h) for h in range(model.cfg.n_heads)]
        layer1 = [extract_head_weights(model, 1, h) for h in range(model.cfg.n_heads)]
        for earlier_head, (V, O, _, _) in enumerate(layer0):
            for later_head, (_, _, Q, K) in enumerate(layer1):
                scores[earlier_head, later_head] = k_composition_score(V, O, Q, K).detach().cpu().double()

    pairs = [
        (float(scores[a, b]), a, b)
        for a in range(model.cfg.n_heads)
        for b in range(model.cfg.n_heads)
    ]
    pairs.sort(reverse=True)

    print("\nTOP LAYER-0 -> LAYER-1 K-COMPOSITION PAIRS")
    print("-" * 76)
    for rank, (score, a, b) in enumerate(pairs[:12], start=1):
        marker = "  <== canonical pair" if a == 7 and b in (4, 10) else ""
        print(f"{rank:2d}. L0H{a:<2} -> L1H{b:<2}: {score:.5f}{marker}")

    print("\nCANONICAL PAIR SCORES")
    print("-" * 76)
    for b in (4, 10):
        print(f"L0H7 -> L1H{b}: {scores[7, b].item():.5f}")

    print("\nFULL SCORE MATRIX (rows=L0 heads, columns=L1 heads)")
    print(scores.float().numpy().round(3))
    print("\nInterpretation:")
    print("  A high K-composition score is weight-level evidence that an earlier")
    print("  head's OV map aligns with a later head's key-side QK computation.")
    print("  Compare the canonical pairs against other pairs and the random baseline;")
    print("  no arbitrary pass/fail threshold is imposed.")

    assert scores.shape == (12, 12)
    assert t.isfinite(scores).all()
    print("\nLOCAL SHAPE / FINITENESS TESTS PASSED")
    print("EXERCISE 20 COMPLETE")


if __name__ == "__main__":
    main()
