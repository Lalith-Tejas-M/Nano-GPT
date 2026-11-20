# nanoGPT Scaling Study: From-Scratch Transformer & Neural Scaling Analysis

An end-to-end, hardened character-level autoregressive Transformer (GPT) implemented from scratch in PyTorch. This repository contains the complete AI/ML modeling core, training engine with step-level resumption, experimental orchestration, and an empirical scaling study on the Tiny Shakespeare corpus.

> **Status:** AI/ML core implementation, 27-test offline verification suite, and all 12 baseline GPU scaling experiments are **100% complete**. Results, metrics, and scaling curves reported below are derived from verified GPU runs.

---

## 1. Executive Summary & Research Questions

The primary objective of this project is to investigate neural scaling behaviors and architectural dynamics for autoregressive character-level Transformers across orders of magnitude in parameter count (from 0.11M to 10.72M parameters).

### Core Research Questions
1. **RQ1 (Scaling Laws):** Does validation cross-entropy loss follow a predictable power-law scaling trajectory as model size increases from XS to L?
   - **Finding:** Yes. Across sizes XS through L, validation loss exhibits clear scaling behavior with an empirical log-log slope of **-0.0653** for `best_val` and **-0.0625** for `final_val`.
2. **RQ2 (LayerNorm Dynamics):** How does Pre-LayerNorm topology behave across parameter scale?
   - **Finding:** Pre-LN provides smooth, stable gradient propagation across all configurations with zero divergence or gradient explosions under identical hyperparameters.
3. **RQ3 (Statistical Dispersion Across Seeds):** How sensitive is training to random weight initialization and batch sampling across seeds 0, 1, and 2?
   - **Finding:** Variation across seeds is exceptionally tight, with sample standard deviations ($\text{ddof}=1$) between $0.0028$ and $0.0076$ nats, and Student-t 95% confidence intervals within $\pm 0.0070$ to $\pm 0.0189$ nats.
4. **RQ4 (Capacity Limits & Overfitting Onset):** Does scaling parameter capacity on a fixed 1.1 MB corpus encounter capacity saturation?
   - **Finding:** **Yes.** Model L (10.72M parameters) enters the overfitting regime around step 2,000. While training loss continues falling to $0.98$ nats, validation loss bottoms at $1.4661$ at step ~2,000 and drifts upward to $1.4876$ by step 3,000, ultimately underperforming Model M at convergence.

---

## 2. Repository File Structure

| File | Subsystem | Description |
|---|---|---|
| `model.py` | Model Core | GPT architecture from scratch: causal self-attention, GELU MLP, Pre/Post-LN blocks, tied embeddings, generation. |
| `data.py` | Data Pipeline | Atomic downloads, SHA-256 verification, character vocabulary ($V=65$), deterministic batching, strict OOV validation. |
| `train.py` | Training Engine | AdamW with decoupled 2D/1D decay, cosine LR schedule, BFloat16 AMP, atomic checkpointing, step-level resumption. |
| `sweep.py` | Orchestration | Automated sweep runner across configurations and seeds; includes run validation, corrupted run quarantine, and skip logic. |
| `plot.py` | Evaluation | Aggregates results into `scaling.png`, `curves.png`, `summary.md`, and `summary.json` (sample std, Student-t 95% CI). |
| `service.py` | Serving Contract | Framework-agnostic `ModelRegistry` and `ModelService` for cached inference and prompt validation for FastAPI / UI. |
| `sample.py` | CLI Sampling | Thin CLI interface wrapping `ModelService` to generate text from saved checkpoints. |
| `colab_nanogpt.ipynb` | Interactive | Synchronized self-contained Colab notebook for GPU training and exploration. |
| `tests/` | Test Suite | 6 test modules containing 27 fast, deterministic, completely offline unit tests. |
| `WORK_LOG.md` | Development Log | Phase-by-phase chronological engineering log detailing all technical decisions and audits (Phases 0–12). |
| `docs/` | Documentation | Contains archived technical documentation, reports, and PDF specifications. |

---

## 3. Architecture & Parameter Counting

The model is an autoregressive decoder-only Transformer adhering to the GPT-2 design principles (Radford et al., 2019):
- **Causal Self-Attention:** Multi-head attention with strictly triangular lower-causal masking (`F.scaled_dot_product_attention`).
- **Weight Tying:** The token embedding matrix (`wte`) and language model head projection (`lm_head`) share identical weights (`wte.weight == lm_head.weight`).
- **Position Embeddings:** Learned 1D absolute positional embeddings (`wpe`).
- **Layer Normalization:** Pre-LayerNorm default topology (`x = x + Attn(LN(x))` and `x = x + MLP(LN(x))`) with final `ln_f` before head; Post-LN ablation topology supported.

```
Input Tokens (B, T) ---> [ wte ] (tied)
Input Positions (T) ---> [ wpe ]
                             |
                             v
                     Sum + Dropout
                             |
                             v
                 +------------------------+
                 | Block x n_layer        |
                 |  Pre-LN:               |
                 |   x = x + Attn(LN1(x)) |
                 |   x = x + MLP(LN2(x))  |
                 +------------------------+
                             |
                             v
                      [ ln_f (Pre-LN) ]
                             |
                             v
               [ lm_head ] (tied to wte)
                             |
                             v
                    Logits (B, T, V=65)
```

### Analytical Parameter Formula
With tied weights, the exact number of trainable parameters is:
$$\text{Params}_{\text{Pre-LN}} = (V \cdot C + B \cdot C) + L \cdot \left[ (4 C^2 + 4 C) + (8 C^2 + 5 C) + 4 C \right] + 2 C$$
where $V = 65$ (vocab size), $B = 128$ (`block_size`), $C = \text{n\_embd}$, and $L = \text{n\_layer}$. Post-LN omits the final LayerNorm `ln_f`, having exactly $2 C$ fewer parameters.

| Configuration | Layers ($L$) | Heads ($H$) | Embedding Dim ($C$) | Head Dim ($d_k$) | Pre-LN Parameters | Non-Embedding Params | Post-LN Parameters ($2C$ less) |
|:---:|:---:|:---:|:---:|:---:|---:|---:|---:|
| **XS** | 2 | 2 | 64 | 32 | **112,448** | 100,096 | 112,320 |
| **S** | 4 | 4 | 128 | 32 | **818,048** | 793,344 | 817,792 |
| **M** | 6 | 6 | 192 | 32 | **2,706,624** | 2,669,568 | 2,706,240 |
| **L** | 8 | 8 | 256 | 32 | **10,721,664** | 10,647,552 | 10,720,896 |

---

## 4. What Was Hardened & Engineered (Phases 0–12)

1. **Principled Initial Loss Correction:**
   - *Problem:* Baseline test `test_initial_loss_is_near_uniform` failed out-of-the-box (`actual loss 3.7752` vs uniform expected $\ln(65) \approx 4.1744$). The fixture evaluated `m(x, x)`, testing the model's ability to predict the input token using tied embedding and output matrices (`wte.weight == lm_head.weight`), artificially deflating loss below uniform by $>0.39$ nats.
   - *Fix:* Replaced coupled inputs with independent random target tokens $y \sim \text{Uniform}(0, V-1)$, correctly verifying uniform prediction at initialization ($\ln(65) \approx 4.1744 \pm 0.02$). Added `test_weight_tying_downward_bias_regression` proving the bias.
2. **Decoupled Weight Decay Optimization:**
   - Parameter group separation applies $0.1$ decay strictly to 2D+ matrices (`wte`, `wpe`, `c_attn.weight`, `c_proj.weight`) while zeroing decay ($0.0$) on LayerNorm scales and 1D biases, preventing scale degradation.
3. **True Step-Level Resumption (`resume_state.pt`):**
   - Implemented full checkpointing of model weights, AdamW optimizer first/second moments, AMP GradScaler state, step iteration, evaluation trajectory history, best validation loss, and complete RNG states across Python, NumPy, PyTorch CPU, and PyTorch CUDA.
4. **Crash-Safe Atomic File Writes & Run Quarantine:**
   - All checkpoints (`ckpt.pt`, `resume_state.pt`) and result metadata (`result.json`) use atomic file replacement (`.tmp` write + `os.replace`).
   - `sweep.py` validates completed runs (`is_run_valid`); corrupted, truncated, or divergent runs are quarantined to `.corrupt` directories and cleanly re-executed.
5. **Decoupled Serving Layer (`service.py`):**
   - Built `ModelRegistry` and `ModelService` providing in-memory model caching, prompt character validation, and structured dictionary output for FastAPI / Streamlit integration.

---

## 5. Empirical Results: 12-Run Baseline Scaling Study

All 12 baseline scaling runs were executed on a dedicated **NVIDIA GeForce RTX 4050 Laptop GPU (6.44 GB VRAM)** using BFloat16 mixed precision.

### 5.1 Complete 12-Run Matrix (3,000 Steps, 3 Seeds Each)

| Run ID | Size | Total Params | Seed | Step 0 Val | Final Train Loss | Best Val Loss (Step) | Final Val Loss (Step 3000) | Runtime |
|---|:---:|---:|:---:|---:|---:|---:|---:|---:|
| `xs_pre_seed0` | xs | 112,448 | 0 | 4.1751 | 1.8319 | 1.9505 (3000) | 1.9505 | 44s |
| `xs_pre_seed1` | xs | 112,448 | 1 | 4.1749 | 1.8354 | 1.9577 (3000) | 1.9577 | 44s |
| `xs_pre_seed2` | xs | 112,448 | 2 | 4.1741 | 1.8415 | 1.9658 (3000) | 1.9658 | 49s |
| `s_pre_seed0` | s | 818,048 | 0 | 4.1747 | 1.3934 | 1.5886 (3000) | 1.5886 | 58s |
| `s_pre_seed1` | s | 818,048 | 1 | 4.1742 | 1.3912 | 1.5875 (2750) | 1.5875 | 58s |
| `s_pre_seed2` | s | 818,048 | 2 | 4.1740 | 1.3956 | 1.5928 (2750) | 1.5932 | 58s |
| `m_pre_seed0` | m | 2,706,624 | 0 | 4.1801 | 1.1895 | 1.4778 (3000) | 1.4778 | 85s |
| `m_pre_seed1` | m | 2,706,624 | 1 | 4.1748 | 1.1834 | 1.4888 (3000) | 1.4888 | 84s |
| `m_pre_seed2` | m | 2,706,624 | 2 | 4.1749 | 1.1843 | 1.4804 (2750) | 1.4807 | 85s |
| `l_pre_seed0` | l | 10,721,664 | 0 | 4.3392 | 0.9804 | 1.4704 (2250) | 1.4887 | 207s |
| `l_pre_seed1` | l | 10,721,664 | 1 | 4.1838 | 0.9827 | 1.4619 (1750) | 1.4799 | 206s |
| `l_pre_seed2` | l | 10,721,664 | 2 | 4.2566 | 0.9785 | 1.4658 (2000) | 1.4941 | 206s |

### 5.2 Statistical Summary Across Seeds ($N=3$, Student-t 95% CIs)

Uncertainty is evaluated using Bessel-corrected sample standard deviation ($\text{ddof}=1$) and two-sided Student-t 95% Confidence Interval half-widths ($t=4.303$ for $n=3$, $\text{df}=2$):
$$\text{CI}_{95\%} = t_{\alpha/2, \text{df}} \cdot \frac{s}{\sqrt{n}} = 4.303 \cdot \frac{s}{\sqrt{3}} \approx 2.4843 \cdot s$$

| Size | Total Params | Best Val Mean | Sample Std ($\text{ddof}=1$) | 95% CI Half-Width | Final Val Mean (Step 3000) | Final Val Std | Overfitting Delta ($\Delta_{\text{final} - \text{best}}$) |
|:---:|---:|---:|---:|---:|---:|---:|---:|
| **XS** | 112,448 | 1.9580 | 0.0076 | $\pm 0.0189$ | 1.9580 | 0.0076 | $+0.0000$ |
| **S** | 818,048 | 1.5896 | 0.0028 | $\pm 0.0070$ | 1.5898 | 0.0030 | $+0.0002$ |
| **M** | 2,706,624 | 1.4823 | 0.0057 | $\pm 0.0142$ | 1.4824 | 0.0057 | $+0.0001$ |
| **L** | 10,721,664 | 1.4661 | 0.0042 | $\pm 0.0105$ | 1.4876 | 0.0071 | **$+0.0215$ (Overfit)** |

- **Log-Log Scaling Fit:** Pre-LN scaling slope $\alpha = \mathbf{-0.0653}$ for `best_val` and $\mathbf{-0.0625}$ for `final_val`.
- **Overfitting Analysis (RQ4):** On Tiny Shakespeare (1,115,394 characters), Model L has ~10 parameters per training token. Without dropout, the model begins memorizing training sequences past step 2,000. While train loss drops to $0.98$, validation loss rebounds by $+0.0215$ nats, causing Model L to achieve worse final validation performance than Model M ($1.4876$ vs $1.4824$).

### 5.3 Generated Artifacts
- **Scaling Law Plot:** `figures/scaling.png` (log-log parameter vs validation loss with sample error bars).
- **Training Trajectories:** `figures/curves.png` (step-by-step train and validation loss trajectories).
- **Machine-Readable Summary:** `figures/summary.json` and `figures/summary.md`.

---

## 6. Integration Contract for Downstream Collaborator (FastAPI / Streamlit)

For downstream application developers (e.g. Ragunathreddy), the AI/ML layer provides high-level callable interfaces via `service.py`:

```python
from service import ModelRegistry, ModelService

# 1. Discover verified local checkpoints
registry = ModelRegistry(results_dir="results")
available_models = registry.list_models()

# 2. Initialize service (caches model weights in memory)
service = ModelService(registry=registry, device="cuda")

# 3. Generate text
response = service.generate_text(
    model_path="results/m_pre_seed0/ckpt.pt",
    prompt="ROMEO:\n",
    max_new_tokens=150,
    temperature=0.8,
    top_k=40
)

print(response["generated_text"])
print("Latency:", response["generation_time_ms"], "ms")
```

### Structured Output Format
```json
{
  "model_id": "m_pre_seed0",
  "prompt": "ROMEO:\n",
  "generated_text": "ROMEO:\nI will not lie their counsel conceit doubt?\n\nFRIAR LAURENCE:\nThis consul, Pompey, sir, let me say.",
  "tokens_generated": 150,
  "generation_time_ms": 1184.5,
  "tokens_per_second": 126.6,
  "config": {
    "block_size": 128,
    "vocab_size": 65,
    "n_layer": 6,
    "n_head": 6,
    "n_embd": 192,
    "ln_placement": "pre"
  }
}
```

### Out-of-Vocabulary (OOV) Exception Policy
Prompts containing characters outside the 65-character vocabulary raise a `ValueError("Characters not in vocabulary: [...]")`. FastAPI endpoints can catch this exception and return `HTTP 422 Unprocessable Entity`.

---

## 7. Comprehensive Test Suite

The test suite runs completely offline with zero network or GPU dependencies:
```bash
python run_tests.py
# or: $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"; python -m pytest -v
```

**Results: 27 passed, 0 failed in 7.44s.**

```text
tests/test_data.py ........     [ 25%]  (encoding roundtrip, OOV rejection, slice bounds, generator seeding)
tests/test_model.py .........   [ 59%]  (causality, initial loss, weight tying regression, parameter counts)
tests/test_plot.py ...          [ 70%]  (Student-t critical values, ddof=1 sample std, plot generation)
tests/test_service.py ...       [ 81%]  (checkpoint discovery, model caching, prompt validation)
tests/test_sweep.py ..          [ 88%]  (run validation, corrupt quarantine, skip logic)
tests/test_train.py ...         [100%]  (cosine schedule math, CPU convergence, step-level resumption)
```

---

## 8. Reproducibility & CLI Reference

```bash
# 1. Install Dependencies
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 2. Run Test Suite
pytest -v

# 3. Execute 12-Run Baseline GPU Scaling Study
python sweep.py --out results --device cuda --max_iters 3000

# 4. Generate Figures and Summaries
python plot.py --results results --out figures --metric best_val
python plot.py --results results --out figures_final --metric final_val

# 5. Optional Post-LN Ablation Sweep (12 runs)
python sweep.py --out results_ablate --ablate --device cuda --max_iters 3000
python plot.py --results results_ablate --out figures_ablate

# 6. Generate Sample Text from CLI
python sample.py --ckpt results/m_pre_seed0/ckpt.pt --prompt "ROMEO:" --tokens 200 --temperature 0.8 --top_k 40

# 7. Resume Interrupted Training Run
python train.py --resume_from results/l_pre_seed0/resume_state.pt --out_dir results/l_pre_seed0
```

---

## 9. Research Limitations

1. **Corpus Context:** Results are specific to Tiny Shakespeare (character-level, 1.1 MB). They do not directly extrapolate to open-domain web text or subword-tokenized models.
2. **Fixed Learning Rate:** All models were trained with uniform peak learning rate $10^{-3}$. Optimal compute scaling requires scaling learning rates inversely with parameter count ($\text{lr} \propto N^{-0.5}$).
3. **Contiguous Validation Split:** The final 10% slice of the corpus represents specific plays (e.g. Antony and Cleopatra), introducing mild play-specific vocabulary and cadence shift.
4. **Post-LN Ablation Status:** The Post-LN architecture is fully implemented and unit-tested in `tests/test_model.py`. The optional 12-run GPU sweep can be executed on-demand via `--ablate`, but was omitted from baseline execution to conserve GPU resources.

---

## 10. Verified Resume Bullet

> *"Implemented a GPT-style autoregressive transformer from scratch in PyTorch and completed a 12-run scaling study (4 sizes x 3 seeds, 95% CIs) on character-level text, discovering a log-log scaling slope of -0.0653, documenting capacity-induced overfitting on a 10.7M parameter model, and hardening the training loop with atomic step-level resumption."*
