# PROJECT AUDIT: nanoGPT Scaling Study

**Auditor:** Lalith's AI/ML Engineering Workstream  
**Date:** 2026-11-20  
**Repository Working Directory:** `c:\Users\tejas\OneDrive\Desktop\NanoGPT\nanogpt-scaling`  
**Git Remote:** `https://github.com/Lalith-Tejas-M/Nano-GPT.git` (`main` branch)  
**Host Environment:** Windows (PowerShell), Python 3.10.11, PyTorch 2.6.0+cu124, CUDA 12.4 (1 GPU detected)

---

## 1. Existing Repository Structure

```text
nanogpt-scaling/
├── .gitignore                      # Configured in Phase 0 (protects *.pt, data/, results/, cache)
├── pytest.ini                      # Pytest configuration
├── README.md                       # Project overview, research questions, instructions
├── requirements.txt                # Dependencies (torch>=2.3, numpy, matplotlib, pytest)
├── model.py                        # From-scratch GPT architecture (decoder-only)
├── data.py                         # Tiny Shakespeare dataset downloader & CharDataset
├── train.py                        # Single-run training loop with AdamW, AMP, and eval
├── sweep.py                        # Model size & seed experiment orchestration
├── plot.py                         # Statistics aggregation, Student-t CI, and matplotlib plots
├── sample.py                       # CLI inference script from saved checkpoint
├── colab_nanogpt.ipynb             # All-in-one Google Colab interactive notebook
├── tests/
│   └── test_model.py               # 5 sanity tests (shapes, causality, init loss, overfit, generate)
└── docs/
    ├── nanoGPT_Project_Documentation.docx
    └── nanoGPT_Project_Documentation.pdf
```

---

## 2. Module-by-Module Responsibilities

| Module | Core Responsibilities | Key Classes / Functions | Observed Deficiencies / Gaps |
|---|---|---|---|
| `model.py` | Transformer architecture definition | `GPTConfig`, `CausalSelfAttention`, `MLP`, `Block`, `GPT` | Raw `assert` statements used for external input validation; missing parameter validation on `GPTConfig`; missing explicit OOV/token clamp safeguards. |
| `data.py` | Character tokenization & batch generation | `load_text()`, `CharDataset` (`encode`, `decode`, `get_batch`) | Non-atomic download (`urllib.request.urlretrieve` directly to target file); missing dataset checksum validation; silent truncation or indexing crash if unknown characters encountered; global RNG leak if generator not passed. |
| `train.py` | Training loop, LR schedule, optimization, checkpointing | `get_lr()`, `estimate_loss()`, `train()` | Only final weights are saved (`save_ckpt=True`); optimizer state, scaler, iteration counter, and RNG state are NOT saved, preventing true step-level resumption; no NaN/Inf loss detection. |
| `sweep.py` | Multi-size, multi-seed experiment execution | `SIZES` dict, CLI loop over sizes/seeds | Skips a run merely if `result.json` exists on disk without checking file integrity, schema completeness, or configuration fingerprint; no detection of corrupt/partial runs. |
| `plot.py` | Statistical analysis & visualization | `t_crit()`, `load()`, `main()` | Hardcoded Student-t table lookup; crashes if directory structure contains unexpected folders; error bars can be ambiguous between standard deviation and 95% CI; lacks machine-readable JSON/CSV summary export. |
| `sample.py` | Text generation CLI | Standalone CLI script | Directly accesses checkpoint dictionary internals; duplicates decoding logic; lacks a reusable Python service class for downstream applications. |
| `tests/test_model.py` | Unit tests | 5 model tests | 1 failing test (`test_initial_loss_is_near_uniform`) due to flawed input-as-target construction `m(x, x)`; no coverage for data, train, sweep, or plot modules. |
| `colab_nanogpt.ipynb` | Colab notebook workflow | 28 cells (18 code, 10 markdown) | Uses `%%writefile` to overwrite source files, creating severe risk of code drift between repository modules and embedded notebook strings. |

---

## 3. Actual Architecture and Data Flow

### Architecture Details
- **Type:** Decoder-only autoregressive character-level Transformer.
- **Embeddings:** Learned token embeddings (`wte`, shape `[vocab_size, n_embd]`) + learned absolute positional embeddings (`wpe`, shape `[block_size, n_embd]`).
- **Attention:** Multi-head causal self-attention (`CausalSelfAttention`) utilizing PyTorch's native `F.scaled_dot_product_attention(..., is_causal=True)`. Fused QKV linear projection ($3 \cdot C$).
- **Feed-Forward (MLP):** Linear $C \to 4C$, GELU non-linearity, projection $4C \to C$, dropout.
- **Normalizations:** 
  - `ln_placement == "pre"`: Pre-LayerNorm on residual branches with a final `ln_f` LayerNorm before the language model head.
  - `ln_placement == "post"`: Post-LayerNorm after residual additions with `nn.Identity()` as `ln_f`.
- **Weight Tying:** Output projection (`head.weight`) is tied to the token embedding matrix (`wte.weight`).
- **Initialization:** Gaussian initialization ($\mu=0, \sigma=0.02$); residual projection weights (`proj.weight`) scaled down by $1/\sqrt{2 \cdot n_{\text{layer}}}$.

### Data Flow
1. Text corpus loaded $\to$ character set extracted and sorted $\to$ `stoi` and `itos` mappings constructed.
2. Contiguous 90% split for training, contiguous final 10% split for validation.
3. Batch sampling randomly selects chunks of length `block_size` for inputs $x$, and target sequence $y$ is the input shifted by $+1$ token.
4. Forward pass: tokens embedded $\to$ positional embeddings added $\to$ dropout $\to$ transformer blocks $\to$ final LayerNorm $\to$ language model head logits $\to$ cross-entropy loss against targets $y$.

---

## 4. Existing Implemented Features vs. Proposed / Missing Features

| Feature | Documented Status | Actual Implementation Status | Notes |
|---|---|---|---|
| Decoder-only GPT core | Implemented | Implemented & verified | Works as designed, causal masking verified. |
| Weight tying | Implemented | Implemented & verified | Tied weights counted once in `num_params()`. |
| Pre-LN & Post-LN | Implemented | Implemented & verified | Both code paths present in `Block` and `GPT`. |
| Character Dataset | Implemented | Implemented | Works, but lacks atomic caching & checksums. |
| Warmup + Cosine LR | Implemented | Implemented | Schedule formula verified in `get_lr()`. |
| Mixed Precision (AMP) | Implemented | Implemented | `GradScaler` used on CUDA with float16; bfloat16 used when supported. |
| Step-level Resumption | Documented in docx | **MISSING** | `train.py` cannot resume mid-training; sweep only skips runs if completed `result.json` exists. |
| Resilient Run Validation | Documented in docx | **MISSING** | `sweep.py` does not validate schema or config hash of existing `result.json`. |
| Framework-free Model Service | Proposed for API | **MISSING** | Only CLI `sample.py` exists; no `ModelService` for Ragunathreddy's FastAPI layer. |
| Regression test for initial loss | Needed | **MISSING** | Current test fails due to biased target fixture. |
| Full unit & integration test suite | Needed | **MISSING** | Only 5 tests in `test_model.py`; no data/training/orchestration tests. |

---

## 5. Dependency and Python/PyTorch Compatibility

- **Python Version:** 3.10.11 (Fully compatible with PyTorch 2.3+).
- **PyTorch Version:** 2.6.0+cu124 (Supports `scaled_dot_product_attention`, `torch.amp.autocast`, `torch.amp.GradScaler`).
- **Hardware:** 1 NVIDIA CUDA GPU detected locally.
- **Pytest Plugin Conflict:** The host environment had an extraneous global package (`langsmith`) whose dependency `_xxhash` was blocked by Windows Application Control. This is resolved by passing `-p no:langsmith` or configuring environment variable `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.

---

## 6. Current Baseline Test Suite Results & Failure Analysis

### Exact Pytest Output
Command: `$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"; python -m pytest -q`
```text
..F..                                                                    [100%]
================================== FAILURES ===================================
______________________ test_initial_loss_is_near_uniform ______________________

    def test_initial_loss_is_near_uniform():
        torch.manual_seed(0)
        m = make().eval()
        x = torch.randint(0, 65, (8, 16))
        _, loss = m(x, x)
>       assert abs(loss.item() - math.log(65)) < 0.3, loss.item()
E       AssertionError: 3.775192975997925
E       assert 0.399194293897712 < 0.3
E        +  where 0.399194293897712 = abs((3.775192975997925 - 4.174387269895637))
E        +    where 3.775192975997925 = <built-in method item of Tensor object at 0x000001B2DCD06480>()
E        +      where <built-in method item of Tensor object at 0x000001B2DCD06480> = tensor(3.7752, grad_fn=<NllLossBackward0>).item
E        +    and   4.174387269895637 = <built-in function log>(65)
E        +      where <built-in function log> = math.log

tests\test_model.py:44: AssertionError
=========================== short test summary info ===========================
FAILED tests/test_model.py::test_initial_loss_is_near_uniform - AssertionError: 3.775192975997925
1 failed, 4 passed in 7.76s
```

### Mathematical & Engineering Analysis of the Failure
1. **Expected Value:** For a uniform random distribution over $V=65$ tokens, the cross-entropy loss is $\ln(65) \approx 4.1744\text{ nats}$.
2. **Flaw in Fixture:** The test passes `m(x, x)`—evaluating the model's loss when predicting the exact same token that was fed as input ($y_t = x_t$).
3. **Weight Tying Bias:** In `GPT`, `self.head.weight = self.wte.weight`. When input token $k$ passes through the embedding layer, its representation has a positive dot product with row $k$ of the tied output matrix. Even with small random weights and LayerNorm, this introduces a systematic upward bias in logit $z_k$ corresponding to the input token $k$.
4. **Consequence:** Target token $y_t = x_t$ is predicted with higher probability than $1/V$, artificially lowering the loss to $\sim 3.7752$ (difference of $0.3992$, exceeding tolerance $0.3$).
5. **Validation with Independent Targets:** Sampling independent random targets $y \sim \text{Uniform}(0, 64)$ produces an initial loss of $\sim 4.1933$ ($\text{diff} \approx 0.0189 \ll 0.3$).
6. **Remediation:** The test must be updated to use independent target tokens $y$ to verify that unlearned initial predictions are near uniform, and a regression test should be added to demonstrate the weight-tying correlation when $y=x$.

---

## 7. Risks: Correctness, Reproducibility, and Integrity

1. **Reproducibility Risk (Global RNG Corruption):** `data.py` uses `torch.randint` without passing a generator in some paths, which can alter the global PyTorch RNG state and cause training runs to diverge across environments.
2. **Integrity Risk (Unvalidated Skip Logic):** `sweep.py` checks `if os.path.exists(os.path.join(out_dir, "result.json")): continue`. If a run was interrupted mid-write, or if an empty/corrupted file was created, the sweep silently skips it and treats it as complete.
3. **Overfitting / Protocol Drift:** The Tiny Shakespeare validation set is the last 10% contiguous block. If character distributions vary across the play, validation loss might reflect character shift rather than general scaling. This must be clearly documented in the research limitations.
4. **Notebook Drift:** Any bug fix applied to `model.py` or `train.py` will not take effect in Colab if the student runs the embedded `%%writefile` cells from `colab_nanogpt.ipynb`.

---

## 8. Prioritized Remediation Table

| Priority | Component | Issue | Remediation Plan |
|---|---|---|---|
| **CRITICAL** | `tests/test_model.py` | Biased `m(x, x)` initial loss test fails | Fix test fixture with independent targets; add explicit regression test explaining weight-tying bias. |
| **CRITICAL** | `train.py` | Incomplete checkpointing (weights only, no optimizer/resumption) | Implement versioned checkpoint schema including optimizer state, scaler, iteration, best val, and RNG states. Add atomic file writes. |
| **HIGH** | `sweep.py` | Naive `result.json` existence check | Implement config fingerprinting and JSON schema validation before skipping runs. Quarantine or rerun corrupt runs. |
| **HIGH** | `data.py` | Non-atomic download, missing checksum, unhandled OOV | Download to temp file with SHA-256 validation; add OOV exception handling; pass explicit generator. |
| **HIGH** | `model.py` | Raw `assert` config validation | Replace assertions with structured `ValueError` validation for all config fields. |
| **MEDIUM** | `service.py` | Missing Python inference service for FastAPI integration | Implement `ModelService` and `ModelRegistry` with checkpoint caching, OOV validation, and metadata generation. |
| **MEDIUM** | `plot.py` | Hardcoded Student-t values, no machine-readable output | Improve CI calculation robustness, label error bars accurately (std vs 95% CI), export `summary.json`. |
| **MEDIUM** | `colab_nanogpt.ipynb`| Embedded source file drift | Synchronize all notebook `%%writefile` cells with updated repository code. |
| **LOW** | `sample.py` | Redundant sampling code | Refactor CLI to wrap `ModelService`. |

---

## 9. Implementation Plan for Lalith's Workstream

* **Phase 2:** Freeze model interface and checkpoint contract (`MODEL_INTERFACE.md`).
* **Phase 3:** Harden `model.py`, resolve test failure, verify parameter counts and LayerNorm variants.
* **Phase 4:** Harden `data.py` with atomic download, SHA-256 validation, and offline synthetic tests.
* **Phase 5:** Implement resilient training loop with true resumption and atomic checkpoints in `train.py`.
* **Phase 6:** Implement verified orchestration and fingerprinting in `sweep.py`.
* **Phase 7:** Harden `plot.py` with Student-t 95% CI and JSON export.
* **Phase 8:** Create reusable `service.py` and align `sample.py`.
* **Phase 9:** Build comprehensive unit/integration test suite covering all modules.
* **Phase 10:** Align `colab_nanogpt.ipynb` with repository source.
* **Phase 11:** Execute verification sweeps and research runs on the available CUDA GPU.
* **Phase 12:** Complete AI-side documentation (`EXPERIMENTS.md`, `MODEL_CARD.md`, `WORK_LOG.md`).
