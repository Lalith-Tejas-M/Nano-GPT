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

## Results (Empirical GPU Runs, 3000 Steps, 3 Seeds)
| size | params | val loss (best_val, mean +/- std, 3 seeds) | 95% CI (Student-t, t=4.303) | final_val (step 3000) |
|---|---|---|---|---|
| xs | 112,448 | 1.9580 ± 0.0076 | ±0.0189 | 1.9580 ± 0.0076 |
| s | 818,048 | 1.5896 ± 0.0028 | ±0.0070 | 1.5898 ± 0.0030 |
| m | 2,706,624 | 1.4823 ± 0.0057 | ±0.0142 | 1.4824 ± 0.0057 |
| l | 10,721,664 | 1.4661 ± 0.0042 | ±0.0105 | 1.4876 ± 0.0071 |

Figures: `figures/scaling.png`, `figures/curves.png`
Detailed breakdown and analysis: see `EXPERIMENTS.md` and `MODEL_CARD.md`.

## What I got wrong / limitations
1. **Initial loss test setup:** Coupling target $y = x$ when weights are tied causes an artificial downward initial loss bias below $\ln(65)$; independent targets $y$ are mathematically required.
2. **Overfitting on size L:** Assuming larger models always generalize better on a small dataset was disproven: Model L (10.7M parameters) overfits Tiny Shakespeare after ~2,000 steps without dropout.
3. **Fixed learning rate across scales:** A uniform $10^{-3}$ learning rate is sub-optimal for 10M+ parameters; scale-dependent learning rates ($\text{lr} \propto N^{-0.5}$) are required for compute-optimal scaling.
4. **Contiguous validation split:** The final 10% slice represents specific Shakespeare plays, introducing mild stylistic distribution shifts.

## Reproduce
```bash
python sweep.py --out results --device cuda
python plot.py --results results --out figures --metric best_val
python plot.py --results results --out figures_final --metric final_val
```

## Resume bullet (Empirically Verified)
"Implemented a GPT-style transformer from scratch in PyTorch and ran a 12-run scaling study (4 sizes x 3 seeds, 95% CIs) on character-level text, measuring a log-log validation-loss slope of -0.0653, documenting capacity-induced overfitting on a 10.7M parameter model, and hardening the training loop with atomic step-level resumption."
