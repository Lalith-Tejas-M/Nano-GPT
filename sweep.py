"""Scaling study: train several model sizes x seeds (plus an optional LayerNorm-placement ablation).

Resumable: finished runs (result.json exists) are skipped, so you can re-run after a Colab disconnect.
  python sweep.py --out results                # full sweep (about 12 runs, plus ablation with --ablate)
  python sweep.py --out results --quick        # tiny CPU smoke test
"""
import argparse
import os

from train import train

# name: (n_layer, n_head, n_embd)
SIZES = {"xs": (2, 2, 64), "s": (4, 4, 128), "m": (6, 6, 192), "l": (6, 6, 384)}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--sizes", nargs="+", default=list(SIZES))
    p.add_argument("--max_iters", type=int, default=3000)
    p.add_argument("--ablate", action="store_true", help="also run post-LN variants for each size")
    p.add_argument("--quick", action="store_true", help="tiny smoke test for CPU")
    a = p.parse_args()

    extra = {}
    if a.quick:
        a.sizes, a.seeds, a.max_iters = ["xs", "s"], [0, 1], 60
        extra = dict(batch_size=16, block_size=32, eval_interval=30, eval_iters=5)

    placements = ["pre", "post"] if a.ablate else ["pre"]
    for ln in placements:
        for size in a.sizes:
            n_layer, n_head, n_embd = SIZES[size]
            for seed in a.seeds:
                name = f"{size}_{ln}_seed{seed}"
                out_dir = os.path.join(a.out, name)
                if os.path.exists(os.path.join(out_dir, "result.json")):
                    print(f"skip {name} (done)")
                    continue
                print(f"\n=== {name} ===", flush=True)
                train(
                    n_layer=n_layer, n_head=n_head, n_embd=n_embd, ln_placement=ln, seed=seed,
                    max_iters=a.max_iters, out_dir=out_dir, save_ckpt=False, **extra,
                )
