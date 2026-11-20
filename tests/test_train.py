"""Unit and integration tests for train.py."""
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
import torch

from model import GPT, GPTConfig
from train import get_lr, train


def test_learning_rate_schedule():
    """Verify linear warmup and cosine decay to min_lr."""
    lr, min_lr, warmup, max_iters = 1e-3, 1e-4, 100, 1000

    # Start of warmup
    assert get_lr(0, max_iters, lr, min_lr, warmup) == pytest.approx(1e-5)
    # End of warmup
    assert get_lr(99, max_iters, lr, min_lr, warmup) == pytest.approx(lr)
    # Midpoint of cosine decay (halfway between 100 and 1000 is 550)
    # ratio = (550 - 100) / 900 = 0.5; cos(pi * 0.5) = 0.0 -> min_lr + 0.5 * (lr - min_lr)
    expected_mid = min_lr + 0.5 * (lr - min_lr)
    assert get_lr(550, max_iters, lr, min_lr, warmup) == pytest.approx(expected_mid, rel=1e-3)
    # At or past max_iters
    assert get_lr(1000, max_iters, lr, min_lr, warmup) == min_lr
    assert get_lr(1050, max_iters, lr, min_lr, warmup) == min_lr


def test_short_cpu_train_and_checkpoint_schema(tmp_path):
    """Run a tiny CPU training session and verify saved results and checkpoint format."""
    corpus = "The quick brown fox jumps over the lazy dog! 1234567890\n" * 20
    out_dir = str(tmp_path / "test_run")

    result = train(
        data_text=corpus,
        out_dir=out_dir,
        block_size=16,
        batch_size=4,
        n_layer=2,
        n_head=2,
        n_embd=32,
        max_iters=10,
        eval_interval=5,
        eval_iters=2,
        warmup=2,
        device="cpu",
        save_ckpt=True,
    )

    assert result["schema_version"] == "1.0"
    assert math.isfinite(result["final_train"])
    assert math.isfinite(result["final_val"])
    assert math.isfinite(result["best_val"])
    assert len(result["history"]) == 3  # iter 0, 5, 10

    # Verify result.json on disk
    json_path = os.path.join(out_dir, "result.json")
    assert os.path.isfile(json_path)
    with open(json_path, encoding="utf-8") as f:
        disk_result = json.load(f)
    assert disk_result["schema_version"] == "1.0"
    assert disk_result["fingerprint"] == result["fingerprint"]

    # Verify ckpt.pt on disk
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    assert os.path.isfile(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    assert ckpt["format_version"] == "1.0"
    assert "model" in ckpt
    assert "model_config" in ckpt
    assert "chars" in ckpt

    # Verify model reloads and runs forward pass
    loaded_model = GPT(GPTConfig(**ckpt["model_config"]))
    loaded_model.load_state_dict(ckpt["model"])
    loaded_model.eval()
    test_x = torch.zeros((1, 8), dtype=torch.long)
    logits, _ = loaded_model(test_x)
    assert logits.shape == (1, 8, ckpt["model_config"]["vocab_size"])


def test_step_level_resumption(tmp_path):
    """Verify that training can pause at step 4 and resume to step 8 bit-for-bit."""
    corpus = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 \n" * 15
    out_dir = str(tmp_path / "resume_run")
    baseline_dir = str(tmp_path / "baseline_run")

    # Run baseline: uninterrupted 0 -> 4
    res_base = train(
        data_text=corpus,
        out_dir=baseline_dir,
        block_size=16,
        batch_size=4,
        n_layer=2,
        n_head=2,
        n_embd=32,
        max_iters=4,
        eval_interval=2,
        eval_iters=2,
        device="cpu",
        seed=42,
        save_resume=True,
    )

    # Run part 1: 0 -> 4 with resume_state.pt saved at iter 2
    res1 = train(
        data_text=corpus,
        out_dir=out_dir,
        block_size=16,
        batch_size=4,
        n_layer=2,
        n_head=2,
        n_embd=32,
        max_iters=4,
        eval_interval=2,
        eval_iters=2,
        device="cpu",
        seed=42,
        save_resume=True,
    )

    resume_ckpt = os.path.join(out_dir, "resume_state.pt")
    assert os.path.isfile(resume_ckpt)

    saved_state = torch.load(resume_ckpt, map_location="cpu")
    assert "rng_state_generator" in saved_state
    assert saved_state["iteration"] == 2

    # Run part 2: resume from resume_state.pt (iter 2) to iter 4
    res2 = train(
        data_text=corpus,
        out_dir=out_dir,
        block_size=16,
        batch_size=4,
        n_layer=2,
        n_head=2,
        n_embd=32,
        max_iters=4,
        eval_interval=2,
        eval_iters=2,
        device="cpu",
        seed=42,
        resume_from=resume_ckpt,
        save_resume=True,
    )

    # Losses must be bit-exact with uninterrupted run!
    base_vals = [h["val"] for h in res_base["history"]]
    resumed_vals = [h["val"] for h in res2["history"]]
    assert base_vals == resumed_vals, f"Resumption loss mismatch: {base_vals} vs {resumed_vals}"


if __name__ == "__main__":
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    sys.exit(pytest.main([__file__, "-v"]))
