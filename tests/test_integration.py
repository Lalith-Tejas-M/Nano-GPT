"""End-to-end integration and system tests for nanoGPT.

Verifies complete pipelines:
1. Training loop -> Checkpoint creation -> ModelService inference.
2. Sweep execution -> Results validation -> Plotting and metric export.
3. CLI error boundaries and user-facing error reporting.
"""
import json
import os
import subprocess
import sys

import pytest
import torch

from data import CharDataset
from model import GPT, GPTConfig
from plot import aggregate_and_plot
from service import ModelRegistry, ModelService
from sweep import run_sweep
from train import train


@pytest.fixture
def mini_corpus():
    return (
        "ROMEO:\nGive me a torch: I am not for this ambling;\nBeing but heavy, I will bear the light.\n"
        "MERCUTIO:\nNay, gentle Romeo, we must have you dance.\n"
    ) * 10


def test_end_to_end_train_and_inference(tmp_path, mini_corpus):
    """Verifies that a model trained end-to-end produces a valid checkpoint usable by ModelService."""
    out_dir = str(tmp_path / "run_e2e")

    train_res = train(
        data_text=mini_corpus,
        out_dir=out_dir,
        block_size=16,
        batch_size=4,
        n_layer=2,
        n_head=2,
        n_embd=32,
        max_iters=6,
        eval_interval=3,
        eval_iters=2,
        device="cpu",
        save_ckpt=True,
        save_resume=True,
    )

    ckpt_file = os.path.join(out_dir, "ckpt.pt")
    result_file = os.path.join(out_dir, "result.json")
    assert os.path.isfile(ckpt_file)
    assert os.path.isfile(result_file)

    # Use ModelService to query and generate text
    service = ModelService(search_dirs=[str(tmp_path)], default_device="cpu")
    service.registry.register_manual("e2e_model", ckpt_file)

    gen_res = service.generate_text(
        model_id="e2e_model",
        prompt="ROMEO:",
        max_new_tokens=30,
        temperature=0.8,
        top_k=20,
    )

    assert "generated_text" in gen_res
    assert gen_res["generated_text"].startswith("ROMEO:")
    assert gen_res["tokens_generated"] == 30
    assert gen_res["generation_time_ms"] > 0
    assert gen_res["config"]["n_layer"] == 2


def test_end_to_end_sweep_and_plotting(tmp_path, mini_corpus):
    """Verifies that sweep execution produces valid results that plot.py can aggregate into figures."""
    results_dir = str(tmp_path / "results")
    figures_dir = str(tmp_path / "figures")

    # Run tiny 2-iteration sweep
    completed = run_sweep(
        out=results_dir,
        sizes=["xs"],
        seeds=[0, 1],
        max_iters=4,
        device="cpu",
        data_text=mini_corpus,
        block_size=16,
        batch_size=4,
        eval_interval=2,
        eval_iters=2,
    )
    assert len(completed) == 2

    # Aggregate and plot
    plot_res = aggregate_and_plot(results_dir=results_dir, out_dir=figures_dir, metric="best_val")

    assert os.path.isfile(plot_res["summary_md"])
    assert os.path.isfile(plot_res["summary_json"])
    assert os.path.isfile(os.path.join(figures_dir, "scaling.png"))
    assert os.path.isfile(os.path.join(figures_dir, "curves.png"))

    with open(plot_res["summary_json"], encoding="utf-8") as f:
        summary_data = json.load(f)
    assert "pre_xs" in summary_data["stats"]
    assert summary_data["stats"]["pre_xs"]["n"] == 2


def test_cli_clean_error_handling():
    """Verifies CLI utilities catch invalid arguments and output clean errors instead of raw unhandled tracebacks."""
    py_exec = sys.executable

    # 1. Non-existent checkpoint to sample.py
    cmd_sample = [py_exec, "sample.py", "--ckpt", "nonexistent_ckpt_123.pt"]
    p1 = subprocess.run(cmd_sample, capture_output=True, text=True)
    assert p1.returncode == 1
    assert "Error: Checkpoint not found" in p1.stderr
    assert "Traceback (most recent call last)" not in p1.stderr

    # 2. Non-existent results dir to plot.py
    cmd_plot = [py_exec, "plot.py", "--results", "nonexistent_dir_123"]
    p2 = subprocess.run(cmd_plot, capture_output=True, text=True)
    assert p2.returncode == 1
    assert "Error: No valid result.json files found" in p2.stderr
    assert "Traceback (most recent call last)" not in p2.stderr


if __name__ == "__main__":
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    sys.exit(pytest.main([__file__, "-v"]))
