"""Sanity tests. Run:  python tests/test_model.py   (or: pytest -q)"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch

from model import GPT, GPTConfig


def make(ln="pre", **kw):
    cfg = GPTConfig(block_size=16, vocab_size=65, n_layer=2, n_head=2, n_embd=32, dropout=0.0, ln_placement=ln, **kw)
    return GPT(cfg)


def test_shapes():
    for ln in ("pre", "post"):
        m = make(ln)
        x = torch.randint(0, 65, (3, 16))
        logits, loss = m(x, x)
        assert logits.shape == (3, 16, 65)
        assert loss.ndim == 0


def test_causality():
    """Changing the LAST token must not change logits at earlier positions."""
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
    torch.manual_seed(0)
    m = make().eval()
    x = torch.randint(0, 65, (8, 16))
    _, loss = m(x, x)
    assert abs(loss.item() - math.log(65)) < 0.3, loss.item()


def test_can_overfit_one_batch():
    torch.manual_seed(0)
    m = make()
    x = torch.randint(0, 65, (4, 16))
    y = torch.roll(x, -1, dims=1)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    _, first = m(x, y)
    for _ in range(150):
        _, loss = m(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 0.5 * first.item(), (first.item(), loss.item())


def test_generate_length():
    m = make().eval()
    out = m.generate(torch.zeros((1, 1), dtype=torch.long), 20, top_k=10)
    assert out.shape == (1, 21)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
