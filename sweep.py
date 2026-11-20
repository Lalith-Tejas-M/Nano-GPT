"""Scaling study: train several model sizes x seeds (plus an optional LayerNorm-placement ablation).

Preserves standard model sizes:
  xs: (2 layers, 2 heads, 64 embd)   ~0.11M parameters
  s:  (4 layers, 4 heads, 128 embd)  ~0.82M parameters
  m:  (6 layers, 6 heads, 192 embd)  ~2.71M parameters
  l:  (6 layers, 6 heads, 384 embd)  ~10.72M parameters

Validation and Resumption:
  Validates result.json integrity, schema, and hyperparameter matching before skipping a run.
  Quarantines corrupted or incomplete result files so runs are safely re-executed.
"""
import argparse
import json
import math
import os
import shutil
import time

from train import train

# Standard model configurations: (n_layer, n_head, n_embd)
SIZES = {
    "xs": (2, 2, 64),
    "s": (4, 4, 128),
    "m": (6, 6, 192),
    "l": (6, 6, 384),
}


def is_run_valid(
    result_path: str,
    expected_size: str,
    expected_ln: str,
    expected_seed: int,
    expected_max_iters: int,
) -> tuple[bool, str]:
    """Validates whether an existing result.json is complete, uncorrupted, and matches configuration."""
    if not os.path.isfile(result_path):
        return False, "file does not exist"
    if os.path.getsize(result_path) == 0:
        return False, "file is empty (0 bytes)"

    try:
        with open(result_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return False, f"JSON parse error: {e}"

    # Required top-level keys
    for req in ("config", "best_val", "final_val", "history", "n_params"):
        if req not in data:
            return False, f"Missing required key: {req!r}"

    # Verify finite loss values
    for metric in ("best_val", "final_val"):
        val = data[metric]
        if not isinstance(val, (int, float)) or not math.isfinite(val):
            return False, f"Non-finite or invalid metric: {metric}={val}"

    # Verify history completeness
    history = data.get("history", [])
    if not history or not isinstance(history, list):
        return False, "Empty or invalid history list"
    if history[-1].get("iter") != expected_max_iters:
        return False, f"Incomplete run: history ended at iter {history[-1].get('iter')}, expected {expected_max_iters}"

    # Verify configuration matching
    cfg = data.get("config", {})
    if cfg.get("ln_placement") != expected_ln:
        return False, f"ln_placement mismatch: found {cfg.get('ln_placement')}, expected {expected_ln}"
    if cfg.get("seed") != expected_seed:
        return False, f"seed mismatch: found {cfg.get('seed')}, expected {expected_seed}"

    return True, "valid"


def run_sweep(
    out: str = "results",
    sizes: list[str] | None = None,
    seeds: list[int] | None = None,
    max_iters: int = 3000,
    ablate: bool = False,
    quick: bool = False,
    force: bool = False,
    save_ckpt: bool = True,
    device: str | None = None,
    data_text: str | None = None,
    **train_overrides,
) -> list[dict]:
    """Orchestrates multi-size, multi-seed training sweep."""
    if sizes is None:
        sizes = list(SIZES.keys())
    if seeds is None:
        seeds = [0, 1, 2]

    for s in sizes:
        if s not in SIZES:
            raise ValueError(f"Unknown size: {s!r}. Supported sizes: {list(SIZES.keys())}")

    extra = {}
    if quick:
        sizes = [s for s in sizes if s in ("xs", "s")]
        seeds = [s for s in seeds if s in (0, 1)]
        max_iters = min(max_iters, 60)
        extra = dict(batch_size=16, block_size=32, eval_interval=30, eval_iters=5)

    placements = ["pre", "post"] if ablate else ["pre"]
    completed_runs = []

    for ln in placements:
        for size in sizes:
            n_layer, n_head, n_embd = SIZES[size]
            for seed in seeds:
                name = f"{size}_{ln}_seed{seed}"
                out_dir = os.path.join(out, name)
                result_path = os.path.join(out_dir, "result.json")

                if not force and os.path.exists(result_path):
                    valid, reason = is_run_valid(result_path, size, ln, seed, max_iters)
                    if valid:
                        print(f"skip {name} (already completed and verified)")
                        with open(result_path, encoding="utf-8") as f:
                            completed_runs.append(json.load(f))
                        continue
                    else:
                        corrupt_dest = result_path + f".corrupt_{int(time.time())}"
                        print(f"Quarantining corrupted result for {name} ({reason}) -> {corrupt_dest}")
                        shutil.move(result_path, corrupt_dest)

                print(f"\n=== Running {name} (size={size}, ln={ln}, seed={seed}) ===", flush=True)
                run_res = train(
                    n_layer=n_layer,
                    n_head=n_head,
                    n_embd=n_embd,
                    ln_placement=ln,
                    seed=seed,
                    max_iters=max_iters,
                    out_dir=out_dir,
                    save_ckpt=save_ckpt,
                    device=device,
                    data_text=data_text,
                    **{**extra, **train_overrides},
                )
                completed_runs.append(run_res)

    return completed_runs


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="nanoGPT Scaling Study Sweep")
    p.add_argument("--out", default="results", help="Directory to store run outputs")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Random seeds to evaluate")
    p.add_argument("--sizes", nargs="+", default=list(SIZES.keys()), help="Model sizes to run")
    p.add_argument("--max_iters", type=int, default=3000, help="Training iterations per run")
    p.add_argument("--ablate", action="store_true", help="Also train Post-LN variant for each configuration")
    p.add_argument("--quick", action="store_true", help="Tiny CPU smoke test")
    p.add_argument("--force", action="store_true", help="Force rerun even if valid result exists")
    p.add_argument("--device", default=None, help="Device override ('cpu', 'cuda')")
    a = p.parse_args()

    try:
        run_sweep(
            out=a.out,
            sizes=a.sizes,
            seeds=a.seeds,
            max_iters=a.max_iters,
            ablate=a.ablate,
            quick=a.quick,
            force=a.force,
            device=a.device,
        )
    except Exception as e:
        import sys
        sys.stderr.write(f"Error during sweep: {e}\n")
        sys.exit(1)
