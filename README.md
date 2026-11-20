# nanoGPT Scaling Study (from scratch, PyTorch)

A small decoder-only transformer (GPT) written from scratch and used for a mini scaling-law study on Tiny Shakespeare (character level).

**Research question:** How does validation loss scale with model size, and does LayerNorm placement (pre-LN vs post-LN) change training outcomes?

> Status: code is ready, **results are not**. Run the experiments yourself, then fill in the results section below with your real numbers. Never publish numbers you did not produce.

## Files
| File | What it does |
|---|---|
| `model.py` | GPT from scratch: causal self-attention, MLP, pre/post-LN blocks, weight tying, `generate()` |
| `data.py` | Downloads Tiny Shakespeare, character vocabulary, random-batch sampler |
| `train.py` | Training loop: AdamW, warmup + cosine LR, grad clipping, mixed precision on GPU, fixed-batch evaluation, checkpoint |
| `sweep.py` | Trains 4 model sizes x 3 seeds (optional post-LN ablation). Resumable after Colab disconnects |
| `plot.py` | Aggregates results into `scaling.png`, `curves.png`, and `summary.md` (mean, std, Student-t 95% CI) |
| `sample.py` | Generates text from a checkpoint |
| `tests/test_model.py` | Shape, **causality**, initial-loss, and overfit-one-batch tests |
| `colab_nanogpt.ipynb` | Everything above in one Colab notebook |

## Quick start (Google Colab)
1. Upload `colab_nanogpt.ipynb` to Colab, set Runtime > Change runtime type > GPU (T4).
2. Run cells top to bottom. The tests run first: **if a test fails, stop and debug before training.**
3. Run the sweep, then the plot cell. Save the `results/` and `figures/` folders (the notebook can copy them to Google Drive).

## Quick start (laptop, CPU only: tests and tiny runs)
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy matplotlib pytest
python tests/test_model.py
python sweep.py --out results_quick --quick            # tiny smoke test
python plot.py --results results_quick --out figures_quick
```
Full training belongs on a GPU (Colab/Kaggle).

## Experiments
1. **Scaling:** sizes xs (0.11M), s (0.82M), m (2.7M), l (10.7M parameters), 3 seeds each, 3000 steps, identical hyperparameters.
2. **Ablation:** `python sweep.py --ablate` adds post-LN variants of every size.
3. **Optional extensions (your own ideas):** learned vs sinusoidal positions, different learning rates per size, longer context, dropout sweep, a BPE tokenizer.

## Things to be careful about (good interview material)
- Same number of steps for every size is a simple choice, not compute-optimal. Say so in the write-up.
- The validation set here is the last 10% of the text, which is one contiguous region. Say so too.
- With 3 seeds, confidence intervals are rough. Report them anyway and don't over-claim.
- Hyperparameters (LR 1e-3) are not re-tuned per size. Larger models may prefer a smaller LR.

## Your tasks (learning goals)
1. Explain in your own words why the causality test passes. What would happen if `is_causal` were removed?
2. Why is the initial loss close to `ln(65)`? What does that tell you about your initialization?
3. Predict which size will overfit first, then check your training curves.
4. Explain why `weight tying` reduces parameters, and by how many.
5. What does the log-log slope in `summary.md` mean, and what is it NOT evidence for?

## Results (fill in after running)
| size | params | val loss (best_val, mean +/- std, 3 seeds) |
|---|---|---|
| xs | | |
| s | | |
| m | | |
| l | | |

Figures: `figures/scaling.png`, `figures/curves.png`

## What I got wrong / limitations
(write this honestly after your runs)

## Reproduce
```bash
python sweep.py --out results [--ablate]
python plot.py --results results --out figures
```

## Resume bullet (use only after you have real numbers)
"Implemented a GPT-style transformer from scratch in PyTorch and ran a 12-run scaling study (4 sizes x 3 seeds, 95% CIs) on character-level text, measuring a log-log validation-loss slope of [X] and the effect of LayerNorm placement ([Y])."
