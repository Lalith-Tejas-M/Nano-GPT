"""Unit tests for statistical aggregation and plotting in plot.py."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pytest

from plot import aggregate_and_plot, compute_group_stats, t_crit


def test_t_crit_degrees_of_freedom():
    """Verify Student-t critical values for two-sided 95% confidence intervals."""
    assert t_crit(0) == 0.0
    assert t_crit(1) == 12.706  # df=1 (n=2)
    assert t_crit(2) == 4.303   # df=2 (n=3)
    assert t_crit(9) == 2.262
    assert t_crit(35) == 1.96   # Large sample asymptotic normal limit


def test_hand_computed_statistics():
    """Verify exact mean, sample std (ddof=1), and 95% CI half-width against known numbers."""
    vals = [2.0, 2.1, 2.2]
    stats = compute_group_stats(vals, n_params=1000, n_params_ne=800)

    assert stats["n"] == 3
    assert stats["mean"] == pytest.approx(2.1)
    # Variance = [(2.0-2.1)^2 + (2.1-2.1)^2 + (2.2-2.1)^2] / (3-1) = 0.02 / 2 = 0.01 -> std = 0.1
    assert stats["std"] == pytest.approx(0.1)
    # CI = 4.303 * 0.1 / sqrt(3) ~= 0.248434
    expected_ci = 4.303 * 0.1 / np.sqrt(3)
    assert stats["ci95"] == pytest.approx(expected_ci, rel=1e-4)


def test_empty_statistics():
    """Verify that empty value list returns safe zeroed metrics without warnings or exceptions."""
    stats = compute_group_stats([], n_params=500, n_params_ne=400)
    assert stats["n"] == 0
    assert stats["mean"] == 0.0
    assert stats["std"] == 0.0
    assert stats["ci95"] == 0.0
    assert stats["raw_values"] == []


def test_aggregate_and_plot_with_synthetic_runs(tmp_path):
    """Test full pipeline loading synthetic result files and creating figures/summaries."""
    results_dir = str(tmp_path / "results")
    figures_dir = str(tmp_path / "figures")

    # Create synthetic runs: xs and s, 2 seeds each
    configurations = [
        ("xs", "pre", 0, 112448, 2.50),
        ("xs", "pre", 1, 112448, 2.52),
        ("s", "pre", 0, 818048, 2.10),
        ("s", "pre", 1, 818048, 2.12),
    ]

    for size, ln, seed, params, val_loss in configurations:
        run_name = f"{size}_{ln}_seed{seed}"
        run_dir = os.path.join(results_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        result_payload = {
            "schema_version": "1.0",
            "best_val": val_loss,
            "final_val": val_loss + 0.02,
            "n_params": params,
            "n_params_non_embedding": params - 5000,
            "config": {"ln_placement": ln, "seed": seed, "n_layer": 2, "n_embd": 64},
            "history": [
                {"iter": 0, "val": 4.18},
                {"iter": 50, "val": val_loss},
            ],
        }
        with open(os.path.join(run_dir, "result.json"), "w", encoding="utf-8") as f:
            json.dump(result_payload, f)

    res = aggregate_and_plot(results_dir, figures_dir, metric="best_val")

    # Check files created
    assert os.path.isfile(os.path.join(figures_dir, "scaling.png"))
    assert os.path.isfile(os.path.join(figures_dir, "curves.png"))
    assert os.path.isfile(os.path.join(figures_dir, "summary.md"))
    assert os.path.isfile(os.path.join(figures_dir, "summary.json"))

    # Check stats content
    stats = res["stats"]
    assert "pre_xs" in stats
    assert "pre_s" in stats
    assert stats["pre_xs"]["n"] == 2
    assert stats["pre_xs"]["mean"] == pytest.approx(2.51)


if __name__ == "__main__":
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    sys.exit(pytest.main([__file__, "-v"]))
