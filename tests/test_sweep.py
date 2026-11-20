"""Unit tests for experiment orchestration in sweep.py."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest

from sweep import is_run_valid, run_sweep


def test_is_run_valid_validation_checks(tmp_path):
    result_file = str(tmp_path / "result.json")

    # Non-existent file
    valid, msg = is_run_valid(result_file, "xs", "pre", 0, 100)
    assert not valid
    assert "does not exist" in msg

    # Empty file
    with open(result_file, "w") as f:
        pass
    valid, msg = is_run_valid(result_file, "xs", "pre", 0, 100)
    assert not valid
    assert "0 bytes" in msg

    # Incomplete dictionary
    with open(result_file, "w") as f:
        json.dump({"best_val": 2.1}, f)
    valid, msg = is_run_valid(result_file, "xs", "pre", 0, 100)
    assert not valid
    assert "Missing required key" in msg

    # Valid dictionary
    valid_data = {
        "best_val": 2.1,
        "final_val": 2.2,
        "n_params": 1000,
        "config": {"ln_placement": "pre", "seed": 0},
        "history": [{"iter": 0}, {"iter": 100}],
    }
    with open(result_file, "w") as f:
        json.dump(valid_data, f)

    valid, msg = is_run_valid(result_file, "xs", "pre", 0, 100)
    assert valid
    assert msg == "valid"

    # Mismatched iterations
    valid, msg = is_run_valid(result_file, "xs", "pre", 0, 200)
    assert not valid
    assert "Incomplete run" in msg


def test_sweep_execution_and_skip_logic(tmp_path):
    corpus = "abcdefghijklmnopqrstuvwxyz1234567890\n" * 20
    out_dir = str(tmp_path / "sweep_out")

    # Run 1: tiny smoke run (1 model size, 1 seed, 4 iterations)
    completed_1 = run_sweep(
        out=out_dir,
        sizes=["xs"],
        seeds=[0],
        max_iters=4,
        device="cpu",
        data_text=corpus,
        block_size=16,
        batch_size=4,
        eval_interval=2,
        eval_iters=2,
    )
    assert len(completed_1) == 1
    run_file = os.path.join(out_dir, "xs_pre_seed0", "result.json")
    assert os.path.isfile(run_file)

    # Run 2: execute again - should validate and skip without errors
    completed_2 = run_sweep(
        out=out_dir,
        sizes=["xs"],
        seeds=[0],
        max_iters=4,
        device="cpu",
        data_text=corpus,
        block_size=16,
        batch_size=4,
        eval_interval=2,
        eval_iters=2,
    )
    assert len(completed_2) == 1
    assert completed_2[0]["best_val"] == completed_1[0]["best_val"]

    # Plant corrupted file: should quarantine and rerun
    with open(run_file, "w") as f:
        f.write("{invalid json")

    completed_3 = run_sweep(
        out=out_dir,
        sizes=["xs"],
        seeds=[0],
        max_iters=4,
        device="cpu",
        data_text=corpus,
        block_size=16,
        batch_size=4,
        eval_interval=2,
        eval_iters=2,
    )
    assert len(completed_3) == 1
    # Check that corrupt file was preserved with timestamp
    parent = os.path.join(out_dir, "xs_pre_seed0")
    files = os.listdir(parent)
    assert any("corrupt" in f for f in files)
    assert os.path.isfile(run_file)  # Fresh valid file replaced it


if __name__ == "__main__":
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    sys.exit(pytest.main([__file__, "-v"]))
