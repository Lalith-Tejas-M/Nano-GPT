"""Unit tests for service.py (ModelRegistry and ModelService)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
import torch

from model import GPT, GPTConfig
from service import ModelRegistry, ModelService


@pytest.fixture
def mock_checkpoint(tmp_path):
    """Creates a temporary valid checkpoint on disk."""
    chars = sorted(list("abcdefghijklmnopqrstuvwxyz \n"))
    cfg = GPTConfig(block_size=16, vocab_size=len(chars), n_layer=2, n_head=2, n_embd=32)
    m = GPT(cfg)
    ckpt_dir = tmp_path / "models" / "s_pre_seed0"
    ckpt_dir.mkdir(parents=True)
    ckpt_path = str(ckpt_dir / "ckpt.pt")

    ckpt_payload = {
        "format_version": "1.0",
        "model_config": cfg.to_dict(),
        "model": m.state_dict(),
        "chars": chars,
    }
    torch.save(ckpt_payload, ckpt_path)
    return str(tmp_path / "models"), ckpt_path, chars


def test_registry_discovery_and_manual(mock_checkpoint):
    search_dir, ckpt_path, _ = mock_checkpoint
    registry = ModelRegistry([search_dir])

    # Should discover s_pre_seed0
    meta = registry.get("s_pre_seed0")
    assert meta["model_id"] == "s_pre_seed0"
    assert os.path.samefile(meta["checkpoint_path"], ckpt_path)

    # Unknown model raises KeyError
    with pytest.raises(KeyError, match="not found in registry"):
        registry.get("non_existent_model")


def test_service_model_caching(mock_checkpoint):
    search_dir, _, _ = mock_checkpoint
    service = ModelService([search_dir], default_device="cpu")

    entry1 = service.load_model("s_pre_seed0")
    assert entry1["model"].training is False  # In eval mode

    # Second load should return cached object
    entry2 = service.load_model("s_pre_seed0")
    assert entry1["model"] is entry2["model"]


def test_service_generation_and_validation(mock_checkpoint):
    search_dir, _, chars = mock_checkpoint
    service = ModelService([search_dir], default_device="cpu")

    # Valid generation
    res = service.generate_text(
        "s_pre_seed0",
        prompt="hello ",
        max_new_tokens=10,
        temperature=0.8,
        top_k=5,
        seed=42,
    )
    assert res["prompt"] == "hello "
    assert res["tokens_generated"] == 10
    assert len(res["completion"]) == 10
    assert res["full_text"] == "hello " + res["completion"]

    # Empty or None prompt defaults to newline
    res_empty = service.generate_text("s_pre_seed0", prompt="", max_new_tokens=5)
    assert res_empty["prompt"] == "\n"

    res_none = service.generate_text("s_pre_seed0", prompt=None, max_new_tokens=5)
    assert res_none["prompt"] == "\n"

    # Type coercion from strings (e.g. from web query parameters)
    res_coerced = service.generate_text("s_pre_seed0", prompt="hi", max_new_tokens="5", temperature="0.7", top_k="10")
    assert res_coerced["tokens_generated"] == 5

    # top_k <= 0 disables top_k filtering without crashing
    res_no_topk = service.generate_text("s_pre_seed0", prompt="hi", max_new_tokens=5, top_k=0)
    assert res_no_topk["tokens_generated"] == 5

    # Unsupported character in prompt raises ValueError
    with pytest.raises(ValueError, match="Unsupported character.*'@'"):
        service.generate_text("s_pre_seed0", prompt="hello @world")

    # Invalid temperature
    with pytest.raises(ValueError, match="temperature must be in range"):
        service.generate_text("s_pre_seed0", prompt="hi", temperature=0.0)

    # Invalid max_new_tokens
    with pytest.raises(ValueError, match="max_new_tokens must be in range"):
        service.generate_text("s_pre_seed0", prompt="hi", max_new_tokens=0)


if __name__ == "__main__":
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    sys.exit(pytest.main([__file__, "-v"]))
