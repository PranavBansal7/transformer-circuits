"""
ARENA 1.2 — TransformerLens Exercise 01
Model configuration inspection

Goal:
    Learn how TransformerLens exposes a model's architecture through `model.cfg`.

You have already reasoned through the access path:
    model -> cfg -> architecture field

Current TODOs:
    1. Get the number of transformer layers.
    2. Get the number of attention heads per layer.
    3. Get the maximum context window.

Run:
    python arena_1_2_transformerlens_01_model_inspection.py

This exercise intentionally does NOT cover tokenization, caches, hooks,
or attention inspection yet. Those come next.
"""

from transformer_lens import HookedTransformer


MODEL_NAME = "gpt2-small"


def inspect_model_config(model: HookedTransformer) -> dict[str, int]:
    """
    Return the three configuration quantities we care about.

    TODO 1:
        Fill in "n_layers".

    TODO 2:
        Fill in "n_heads".

    TODO 3:
        Fill in "n_ctx".
    """

    config = {
        # TODO 1
        "n_layers": model.cfg.n_layers,

        # TODO 2
        "n_heads": model.cfg.n_heads,

        # TODO 3
        "n_ctx": model.cfg.n_ctx,
    }

    return config


def run_tests(model: HookedTransformer, config: dict[str, int]) -> None:
    """Small local tests for this exercise."""

    required_keys = {"n_layers", "n_heads", "n_ctx"}
    assert set(config.keys()) == required_keys, (
        f"Expected exactly {required_keys}, got {set(config.keys())}"
    )

    for key, value in config.items():
        assert isinstance(value, int), (
            f"{key} should be an int, got {type(value).__name__}"
        )
        assert value > 0, f"{key} should be positive, got {value}"

    # Parity check: your extracted values must match TransformerLens directly.
    assert config["n_layers"] == model.cfg.n_layers
    assert config["n_heads"] == model.cfg.n_heads
    assert config["n_ctx"] == model.cfg.n_ctx

    print("LOCAL TESTS PASSED")
    print("TransformerLens config inspection is correct.")


def main() -> None:
    print("=" * 72)
    print("ARENA 1.2 — TransformerLens Exercise 01")
    print("=" * 72)
    print()
    print("Task:")
    print("  Inspect `model.cfg` and extract:")
    print("    - number of transformer layers")
    print("    - number of attention heads per layer")
    print("    - maximum context window")
    print()
    print("Use the exact attribute names we just discussed.")
    print("Do not modify the tests.")
    print()

    print(f"Loading {MODEL_NAME}...")
    model = HookedTransformer.from_pretrained(MODEL_NAME)
    print("Model loaded.")
    print()

    config = inspect_model_config(model)

    print("Your extracted configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    run_tests(model, config)

    print()
    print("Exercise 01 PASSED.")
    print("Next: tokenization and running HookedTransformer on real text.")


if __name__ == "__main__":
    main()
