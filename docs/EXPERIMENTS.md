# EXPERIMENTS: nanoGPT Character-Level Scaling Study

## 1. Executive Summary & Research Questions

This study investigates neural scaling laws and architectural dynamics for autoregressive character-level Transformers trained on the Tiny Shakespeare corpus (1,115,394 characters). The research is structured around four primary research questions (RQs):

- **RQ1 (Scaling Behavior):** Does validation cross-entropy loss scale predictably as model size increases from XS (0.11M) to L (10.72M parameters)?
  - *Finding:* Validation loss exhibits power-law scaling across XS, S, and M. Across the four scales, the empirical log-log scaling slope is **-0.0653** for `best_val` and **-0.0625** for `final_val`.
- **RQ2 (LayerNorm Placement):** How does Pre-LayerNorm topology behave across parameter orders of magnitude?
  - *Finding:* Pre-LN provides stable gradient propagation and smooth optimization across all 12 baseline runs with zero divergent runs or NaNs under identical hyperparameter settings.
- **RQ3 (Seed Dispersion & Statistical Uncertainty):** How much dispersion occurs across random initializations and batch orderings (seeds 0, 1, 2)?
  - *Finding:* Across 3 random seeds, sample standard deviations ($\text{ddof}=1$) are remarkably tight: $\sigma \in [0.0028, 0.0076]$ nats. Student-t 95% confidence intervals ($t=4.303$ for $n=3$, $\text{df}=2$) are within $\pm 0.0070$ to $\pm 0.0189$ nats.
- **RQ4 (Capacity Limits & Overfitting Onset):** Does increasing model parameter capacity on a fixed corpus (1.1 MB) encounter capacity limits or overfitting?
  - *Finding:* **Yes.** While XS, S, and M generalize without validation degradation, Model L (10.72M parameters) enters the overfitting regime around step 2,000. For `l_pre_seed0`, train loss decreases to $0.9804$ at step 3,000, while validation loss bottoms at $1.4704$ at step 2,250 and rises to $1.4887$ (+0.0183 nats) at step 3,000.

---

## 2. Experimental Controls & Protocol

All 12 baseline runs were executed under strictly controlled, identical conditions:

| Parameter | Value | Rationale |
|---|---|---|
| **Dataset** | Tiny Shakespeare (`data/input.txt`) | Standard Karpathy nanoGPT corpus; SHA-256: `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed` |
| **Data Split** | 90% train (1,003,854 chars), 10% val (111,540 chars) | Strict chronological split; offline validated |
| **Tokenization** | Character-level ($V=65$ tokens) | Vocabulary extracted directly from corpus; strict OOV rejection |
| **Context Length (`block_size`)** | 128 characters | Uniform sequence length across all configurations |
| **Batch Size** | 64 sequences (8,192 chars/step) | Uniform batch throughput across all configurations |
| **Training Steps** | 3,000 iterations | Total compute budget: $2.4576 \times 10^7$ tokens seen per run |
| **Evaluation Frequency** | Every 250 iterations (200 val batches) | High-resolution evaluation trajectory tracking |
| **Optimizer** | AdamW ($\beta_1=0.9, \beta_2=0.99, \epsilon=10^{-8}$) | Standard decoupled weight decay |
| **Weight Decay** | 0.1 on $\ge 2\text{D}$ weights; 0.0 on 1D biases/norms | Prevents penalizing normalization scale/biases |
| **Learning Rate Schedule** | Warmup (100 steps to $10^{-3}$), Cosine decay to $10^{-4}$ | Standard cosine decay schedule |
| **Gradient Clipping** | Maximum norm 1.0 | Prevents exploding gradient spikes |
| **Hardware** | NVIDIA GeForce RTX 4050 Laptop GPU (6.44 GB VRAM) | BFloat16 automatic mixed precision (`torch.autocast`) |
| **Environment** | Python 3.11.9, PyTorch 2.5.1+cu121, CUDA 12.1 | Seeded across Python, NumPy, PyTorch CPU, and PyTorch CUDA |

---

## 3. Parameter Counts & Analytical Invariants

Embedding weights (`wte`) and output projection weights (`lm_head`) are explicitly tied (`wte.weight == lm_head.weight`).

$$\text{Params}_{\text{Pre-LN}} = (V \cdot C + B \cdot C) + L \cdot \left[ (4 C^2 + 4 C) + (8 C^2 + 5 C) + 4 C \right] + 2 C$$

| Size | Layers ($L$) | Heads ($H$) | Dim ($C$) | Head Dim ($d_k$) | Total Params (Pre-LN) | Non-Embedding Params | Post-LN Params ($2C$ less) |
|---|:---:|:---:|:---:|:---:|---:|---:|---:|
| **XS** | 2 | 2 | 64 | 32 | **112,448** | 100,096 | 112,320 |
| **S** | 4 | 4 | 128 | 32 | **818,048** | 793,344 | 817,792 |
| **M** | 6 | 6 | 192 | 32 | **2,706,624** | 2,669,568 | 2,706,240 |
| **L** | 6 | 6 | 384 | 64 | **10,721,664** | 10,647,552 | 10,720,896 |

---

## 4. Complete Baseline Run Matrix (Actual Empirical Results)

All 12 runs completed successfully on the local RTX 4050 GPU. Raw metrics were verified and loaded from `results/<run_id>/result.json`.

| Run ID | Size | Seed | Step 0 Val Loss | Final Train Loss | Best Val Loss (Step) | Final Val Loss (Step 3000) | Runtime (s) | Best Val PPL |
|---|:---:|:---:|---:|---:|---:|---:|---:|---:|
| `xs_pre_seed0` | xs | 0 | 4.1751 | 1.8319 | 1.9505 (3000) | 1.9505 | 44s | 7.032 |
| `xs_pre_seed1` | xs | 1 | 4.1749 | 1.8354 | 1.9577 (3000) | 1.9577 | 44s | 7.083 |
| `xs_pre_seed2` | xs | 2 | 4.1741 | 1.8415 | 1.9658 (3000) | 1.9658 | 49s | 7.141 |
| `s_pre_seed0` | s | 0 | 4.1747 | 1.3934 | 1.5886 (3000) | 1.5886 | 58s | 4.897 |
| `s_pre_seed1` | s | 1 | 4.1742 | 1.3912 | 1.5875 (2750) | 1.5875 | 58s | 4.891 |
| `s_pre_seed2` | s | 2 | 4.1740 | 1.3956 | 1.5928 (2750) | 1.5932 | 58s | 4.918 |
| `m_pre_seed0` | m | 0 | 4.1801 | 1.1895 | 1.4778 (3000) | 1.4778 | 85s | 4.383 |
| `m_pre_seed1` | m | 1 | 4.1748 | 1.1834 | 1.4888 (3000) | 1.4888 | 84s | 4.432 |
| `m_pre_seed2` | m | 2 | 4.1749 | 1.1843 | 1.4804 (2750) | 1.4807 | 85s | 4.395 |
| `l_pre_seed0` | l | 0 | 4.3392 | 0.9804 | 1.4704 (2250) | 1.4887 | 207s | 4.351 |
| `l_pre_seed1` | l | 1 | 4.1838 | 0.9827 | 1.4619 (1750) | 1.4799 | 206s | 4.314 |
| `l_pre_seed2` | l | 2 | 4.2566 | 0.9785 | 1.4658 (2000) | 1.4941 | 206s | 4.331 |

---

## 5. Statistical Summary & Scaling Laws

### 5.1 Aggregate Statistics across Seeds ($N=3$)

Uncertainty is evaluated using sample standard deviation ($\text{ddof}=1$) and two-sided Student-t 95% confidence interval half-width:
$$\text{CI}_{95\%} = t_{\alpha/2, \text{df}} \cdot \frac{s}{\sqrt{n}} = 4.303 \cdot \frac{s}{\sqrt{3}} \approx 2.4843 \cdot s$$

#### Metric: `best_val` (Minimum Validation Loss across Steps)
| Size | Parameters | Mean Val Loss | Sample Std ($\text{ddof}=1$) | 95% CI Half-Width | Mean Perplexity |
|:---:|---:|---:|---:|---:|---:|
| **XS** | 112,448 | 1.9580 | 0.0076 | $\pm 0.0189$ | 7.085 |
| **S** | 818,048 | 1.5896 | 0.0028 | $\pm 0.0070$ | 4.902 |
| **M** | 2,706,624 | 1.4823 | 0.0057 | $\pm 0.0142$ | 4.403 |
| **L** | 10,721,664 | 1.4661 | 0.0042 | $\pm 0.0105$ | 4.332 |

- **Empirical Scaling Fit:** Pre-LN log-log scaling slope $\alpha = \mathbf{-0.0653}$.

#### Metric: `final_val` (Step 3,000 Validation Loss)
| Size | Parameters | Mean Val Loss | Sample Std ($\text{ddof}=1$) | 95% CI Half-Width | Overfitting Delta ($\Delta_{\text{final} - \text{best}}$) |
|:---:|---:|---:|---:|---:|---:|
| **XS** | 112,448 | 1.9580 | 0.0076 | $\pm 0.0189$ | $+0.0000$ |
| **S** | 818,048 | 1.5898 | 0.0030 | $\pm 0.0075$ | $+0.0002$ |
| **M** | 2,706,624 | 1.4824 | 0.0057 | $\pm 0.0141$ | $+0.0001$ |
| **L** | 10,721,664 | 1.4876 | 0.0071 | $\pm 0.0177$ | **$+0.0215$** |

- **Empirical Scaling Fit:** Pre-LN log-log scaling slope $\alpha = \mathbf{-0.0625}$.

---

## 6. Overfitting Analysis on Model Size L (Addressing RQ4)

The empirical data reveals a critical qualitative divergence between small/medium models and Model L:
- For **XS, S, and M**, validation loss continuously improves throughout the 3,000 steps. The step of best validation loss is at step 2,750 or 3,000.
- For **Model L (10.72M parameters)**:
  - Training loss drops rapidly to $0.98$ nats (far below any other model).
  - Validation loss reaches its minimum between steps 1,750 and 2,250 ($1.4619 - 1.4704$).
  - Over the final 1,000 steps, validation loss consistently degrades across all three seeds, ending at $1.4876$ mean ($+0.0215$ nats higher than minimum).
  - Crucially, at step 3,000, Model L's final validation loss ($1.4876$) is **worse** than Model M's final validation loss ($1.4824$).

**Root Cause:**
Tiny Shakespeare contains only $1.00 \times 10^6$ training characters. At 10.72M parameters, Model L has ~10 parameters per training token. Without dropout (`dropout=0.0`) and under a fixed learning rate ($10^{-3}$), the model easily memorizes training n-grams, resulting in empirical overfitting.

---

## 7. Artifacts Generated

1. `figures/scaling.png` — Log-log scaling plot with sample standard deviation error bars ($\text{ddof}=1$) and fitted scaling power-law line for `best_val`.
2. `figures/curves.png` — Iteration-level training and validation trajectories across all 12 runs, illustrating convergence and Model L overfitting.
3. `figures/summary.md` & `figures/summary.json` — Aggregate statistical summaries for `best_val`.
4. `figures_final/scaling.png` & `figures_final/summary.md` — Corresponding figures and metrics for `final_val`.

---

## 8. Reproducibility & CLI Execution

### Reproduce Baseline Sweep (12 runs)
```bash
# Execute the full 12-run matrix on GPU
python sweep.py --out results --device cuda --max_iters 3000

# Generate plots and summary tables
python plot.py --results results --out figures --metric best_val
python plot.py --results results --out figures_final --metric final_val
```

### Reproduce Optional Post-LN Ablation (12 runs)
```bash
python sweep.py --out results_ablate --ablate --device cuda --max_iters 3000
python plot.py --results results_ablate --out figures_ablate
```

### Inference Generation
```bash
python sample.py --ckpt results/m_pre_seed0/ckpt.pt --prompt "ROMEO:" --tokens 200 --temperature 0.8 --top_k 40
```

---

## 9. Research Limitations

1. **Corpus Scope:** Results are specific to Tiny Shakespeare (character-level, 1.1 MB). They do not generalize to subword tokenized BPE models or web-scale pre-training datasets.
2. **Fixed Learning Rate:** All models were trained with peak learning rate $10^{-3}$. In formal scaling literature (Chinchilla / Llama), larger models require learning rate scaling ($\text{lr} \propto N^{-0.5}$).
3. **Contiguous Validation Split:** The validation set is the final 10% slice of the corpus, representing specific Shakespeare plays (e.g., Antony and Cleopatra), introducing mild play-specific vocabulary shift.
4. **Sample Size ($N=3$):** While Student-t intervals account for small sample size ($t=4.303$), 3 seeds provide preliminary dispersion rather than full posterior variance estimates.
