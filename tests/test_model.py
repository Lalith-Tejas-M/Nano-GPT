"""Sanity, architecture, and correctness tests for model.py.
Run:  pytest -q
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
import torch

from model import GPT, GPTConfig


def make(ln="pre", **kw):
    cfg = GPTConfig(
        block_size=16,
        vocab_size=65,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
        ln_placement=ln,
        **kw,
    )
    return GPT(cfg)


def test_shapes():
    """Verify forward output shapes for both Pre-LN and Post-LN architectures."""
    for ln in ("pre", "post"):
        m = make(ln)
        x = torch.randint(0, 65, (3, 16))
        logits, loss = m(x, x)
        assert logits.shape == (3, 16, 65)
        assert loss.ndim == 0


def test_causality():
    """Causal attention invariant: changing the LAST token must not alter logits at earlier positions."""
    torch.manual_seed(0)
    m = make().eval()
    x = torch.randint(0, 65, (2, 16))
    x2 = x.clone()
    x2[:, -1] = (x2[:, -1] + 1) % 65
    l1, _ = m(x)
    l2, _ = m(x2)
    assert torch.allclose(l1[:, :-1], l2[:, :-1], atol=1e-5)
    assert not torch.allclose(l1[:, -1], l2[:, -1], atol=1e-5)


def test_initial_loss_is_near_uniform():
    """Verify unlearned model produces cross-entropy loss close to uniform random guessing ln(V).
    
    CRITICAL: Targets 'y' must be sampled independently of inputs 'x'. If targets are identical to
    inputs (m(x, x)), weight tying (wte == head) induces a positive dot-product bias on the target
    logit, which artificially depresses initial cross entropy below uniform ln(V).
    """
    torch.manual_seed(0)
    m = make().eval()
    x = torch.randint(0, 65, (8, 16))
    # Independent random targets:
    y = torch.randint(0, 65, (8, 16))
    _, loss = m(x, y)
    expected_uniform_loss = math.log(65)  # ~4.1744
    assert abs(loss.item() - expected_uniform_loss) < 0.3, (
        f"Initial loss {loss.item():.4f} diverged from uniform ln(65)={expected_uniform_loss:.4f}"
    )


def test_weight_tying_downward_bias_regression():
    """Regression test documenting why m(x, x) is an invalid uniform-loss test fixture.
    
    Because token embeddings (wte) and output projection (head) are tied, passing identical input
    and target tokens m(x, x) creates a positive correlation between x and the target logit,
    depressing loss below ln(65).
    """
    torch.manual_seed(0)
    m = make().eval()
    x = torch.randint(0, 65, (8, 16))
    _, loss_same = m(x, x)
    expected_uniform_loss = math.log(65)
    # The loss for identical input/target is biased downwards (> 0.3 below ln(65))
    assert expected_uniform_loss - loss_same.item() > 0.3, (
        f"Expected downward bias from tied weights, got loss={loss_same.item()}"
    )


def test_can_overfit_one_batch():
    """Ensure gradients flow properly and a small batch can be overfit."""
    torch.manual_seed(0)
    m = make()
    x = torch.randint(0, 65, (4, 16))
    y = torch.roll(x, -1, dims=1)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    _, first = m(x, y)
    for _ in range(150):
        _, loss = m(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.5 * first.item(), (first.item(), loss.item())


def test_generate_length():
    """Verify autoregressive generation yields the requested sequence length."""
    m = make().eval()
    out = m.generate(torch.zeros((1, 1), dtype=torch.long), 20, top_k=10)
    assert out.shape == (1, 21)


def test_parameter_counts_and_ln_difference():
    """Verify exact parameter counts for XS, S, M, L against documented formulas.
    
    Verifies that weight tying is respected and Post-LN has exactly 2*n_embd fewer parameters
    due to omitting the final LayerNorm.
    """
    sizes = {
        "xs": (2, 2, 64, 112448),
        "s": (4, 4, 128, 818048),
        "m": (6, 6, 192, 2706624),
        "l": (6, 6, 384, 10721664),
    }
    for name, (n_layer, n_head, n_embd, expected_params) in sizes.items():
        cfg_pre = GPTConfig(
            block_size=128, vocab_size=65, n_layer=n_layer, n_head=n_head, n_embd=n_embd, ln_placement="pre"
        )
        m_pre = GPT(cfg_pre)
        assert m_pre.num_params() == expected_params, (
            f"{name} Pre-LN params {m_pre.num_params()} != {expected_params}"
        )

        cfg_post = GPTConfig(
            block_size=128, vocab_size=65, n_layer=n_layer, n_head=n_head, n_embd=n_embd, ln_placement="post"
        )
        m_post = GPT(cfg_post)
        # Post-LN omits final LayerNorm (weight + bias = 2*n_embd params)
        expected_post_params = expected_params - 2 * n_embd
        assert m_post.num_params() == expected_post_params, (
            f"{name} Post-LN params {m_post.num_params()} != {expected_post_params}"
        )


def test_config_validation():
    """Verify invalid configurations raise descriptive ValueError."""
    with pytest.raises(ValueError, match="block_size must be positive"):
        GPTConfig(block_size=0)
    with pytest.raises(ValueError, match="vocab_size must be positive"):
        GPTConfig(vocab_size=-1)
    with pytest.raises(ValueError, match="divisible by n_head"):
        GPTConfig(n_embd=65, n_head=4)
    with pytest.raises(ValueError, match="dropout must be in"):
        GPTConfig(dropout=1.5)
    with pytest.raises(ValueError, match="ln_placement must be 'pre' or 'post'"):
        GPTConfig(ln_placement="middle")


def test_forward_and_generate_error_handling():
    """Verify sequence length checks and sampling boundary validations."""
    m = make().eval()
    # Sequence length exceeding block_size
    too_long = torch.randint(0, 65, (1, 17))
    with pytest.raises(ValueError, match="exceeds model block_size"):
        m(too_long)

    # Targets shape mismatch
    x = torch.randint(0, 65, (2, 16))
    bad_targets = torch.randint(0, 65, (2, 10))
    with pytest.raises(ValueError, match="targets shape"):
        m(x, bad_targets)

    # Invalid generate parameters
    with pytest.raises(ValueError, match="temperature must be positive"):
        m.generate(x, 10, temperature=0.0)
    with pytest.raises(ValueError, match="max_new_tokens must be non-negative"):
        m.generate(x, -5)
    with pytest.raises(ValueError, match="top_k must be positive"):
        m.generate(x, 5, top_k=0)


if __name__ == "__main__":
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    sys.exit(pytest.main([__file__, "-v"]))
