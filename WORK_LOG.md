# WORK LOG: nanoGPT Scaling Study (Lalith's AI/ML Workstream)

## Phase 0: Locate and Establish Working Repository
- **Date:** 2026-11-20
- **Status:** PASS
- **Files Inspected:** `README.md`, `requirements.txt`, `model.py`, `data.py`, `train.py`, `sweep.py`, `plot.py`, `sample.py`, `tests/test_model.py`, `colab_nanogpt.ipynb`, `docs/*`.
- **Files Created:** `.gitignore`
- **Important Decisions:**
  - Initialized clean Git repository on branch `main` in `c:\Users\tejas\OneDrive\Desktop\NanoGPT\nanogpt-scaling`.
  - Configured Git remote `origin` to point to user-provided repository: `https://github.com/Lalith-Tejas-M/Nano-GPT.git`.
  - Configured `.gitignore` to prevent tracking checkpoints (`*.pt`), training data (`data/`, `input.txt`), cache directories, and transient outputs.
  - Committed initial baseline snapshot locally (`d22b8e8` and `123544b`). In accordance with Operating Principle 9, no remote push was executed.
- **Commands Run:**
  - `git init -b main`
  - `git remote add origin https://github.com/Lalith-Tejas-M/Nano-GPT.git`
  - `git add .gitignore requirements.txt; git commit -m "..."`
- **Exit Criteria:** PASS — Working directory established, all files cataloged, clean Git history rooted at baseline.

---

## Phase 1: Deep Repository Audit
- **Date:** 2026-11-20
- **Status:** PASS
- **Files Created:** `PROJECT_AUDIT.md`, `pytest.ini`
- **Important Decisions:**
  - Diagnosed exact failure in `tests/test_model.py::test_initial_loss_is_near_uniform`: `m(x, x)` fixture couples input and target tokens, creating an artificial downward bias in cross-entropy loss due to tied embedding/projection weights.
  - Identified host environment issue where global `langsmith` pytest plugin attempts to load blocked `_xxhash` binary on Windows. Mitigated via `pytest.ini` and environment variable `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.
  - Audited module gaps: missing step-level resumption, unvalidated sweep run skipping, and lack of a framework-independent model inference service for Ragunathreddy's FastAPI layer.
  - Created prioritized remediation table (Critical, High, Medium, Low).
- **Commands Run:**
  - `$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"; python -m pytest -q`
- **Actual Results:** 4 passed, 1 failed (`test_initial_loss_is_near_uniform` actual loss `3.7752` vs expected `4.1744`).
- **Exit Criteria:** PASS — Comprehensive audit saved in `PROJECT_AUDIT.md`, exact test logs preserved, baseline defects documented without superficial workarounds.

---

## Phase 2: Freeze Model, Training, and Integration Contracts
- **Date:** 2026-11-20
- **Status:** PASS
- **Files Created:** `MODEL_INTERFACE.md`
- **Important Decisions:**
  - Defined strict interface specifications for Ragunathreddy's FastAPI backend and Streamlit dashboard.
  - Specified Python class contracts (`ModelService`, `ModelRegistry`) with caching, OOV exception policies, parameter boundary validation, and device fallback.
  - Standardized checkpoint schemas: versioned `ckpt.pt` (inference) and `resume_state.pt` (training resumption).
  - Specified experiment result schema `result.json` version 1.0 with config fingerprints, environment metadata, and complete history.
  - Maintained clear architectural separation: zero web framework dependencies inside Lalith's ML scope.
- **Exit Criteria:** PASS — `MODEL_INTERFACE.md` created; contract frozen and documented.

---

## Phase 3: Repair and Harden GPT Core (`model.py`)
- **Date:** 2026-11-20
- **Status:** PASS
- **Files Inspected:** `model.py`, `tests/test_model.py`
- **Files Modified:** `model.py`, `tests/test_model.py`
- **Important Decisions:**
  - Added descriptive `__post_init__` validation in `GPTConfig` raising `ValueError` for non-positive dimensions, non-divisible head sizes, out-of-range dropout, and invalid `ln_placement`.
  - Replaced raw asserts across `CausalSelfAttention`, `Block`, and `GPT` with structured `ValueError` exceptions for input-facing boundaries.
  - Verified weight tying and confirmed parameter counts for all sizes: XS (112,448), S (818,048), M (2,706,624), and L (10,721,664) for Pre-LN; verified Post-LN has exactly $2 \cdot C$ fewer parameters ($112,320$, $817,792$, $2,706,240$, $10,720,896$).
  - Repaired initial loss test in `tests/test_model.py`: independent target fixture $y$ correctly tests uniform prediction without weight-tying bias.
  - Added regression test `test_weight_tying_downward_bias_regression` proving that $m(x, x)$ depresses initial loss below $\ln(65)$ by $>0.3$.
  - Added comprehensive tests for shape invariants, causality, configuration errors, sequence length overages, and generation parameter bounds.
- **Commands Run:**
  - `$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"; python -m pytest -q tests/test_model.py`
  - `python tests/test_model.py`
- **Actual Results:** 9 passed, 0 failed in both pytest and direct execution.
- **Exit Criteria:** PASS — Core GPT architecture verified; initial loss test repaired for principled mathematical reasons; causal invariance and parameter counts proven.

---

## Phase 4: Repair and Harden Data Pipeline (`data.py`)
- **Date:** 2026-11-20
- **Status:** PASS
- **Files Inspected:** `data.py`
- **Files Modified:** `data.py`
- **Files Created:** `tests/test_data.py`
- **Important Decisions:**
  - Implemented atomic download pattern: downloads to `.tmp` extension first and utilizes `os.replace` to prevent partial/corrupt downloads from being marked complete.
  - Added SHA-256 verification and dataset metadata extraction (`compute_sha256`, `get_dataset_metadata`).
  - Added strict OOV character validation in `CharDataset.encode()` raising `ValueError` with explicit unsupported characters.
  - Hardened batch sampling with exact boundary calculation and explicit generator support to prevent global RNG pollution.
  - Verified next-token shift invariant ($y_t = x_{t+1}$) and deterministic batch sampling.
- **Commands Run:**
  - `$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"; python -m pytest -q tests/test_data.py`
- **Actual Results:** 7 passed, 0 failed.
- **Exit Criteria:** PASS — Data pipeline hardened; deterministic batch sampling and OOV handling verified offline without internet requirement.

---

## Phase 5: Strengthen the Training Engine (`train.py`)
- **Date:** 2026-11-20
- **Status:** PASS
- **Files Inspected:** `train.py`
- **Files Modified:** `train.py`
- **Files Created:** `tests/test_train.py`
- **Important Decisions:**
  - Implemented comprehensive seeding across Python `random`, `os.environ["PYTHONHASHSEED"]`, NumPy, PyTorch CPU, and all CUDA devices.
  - Implemented parameter separation for AdamW: 0.1 weight decay on matrix/embedding parameters ($\ge 2\text{D}$) and 0.0 decay on 1D biases and LayerNorms.
  - Implemented step-level resumption: saves `resume_state.pt` with optimizer state, scaler state, step iteration, evaluation history, and full RNG states. Supports `resume_from` to resume training cleanly mid-run.
  - Added non-finite loss/gradient detection raising `FloatingPointError` to fail fast on divergence.
  - Implemented atomic file writes (`atomic_save`) for all checkpoints and `result.json` using temporary files and atomic `os.replace`.
  - Added run fingerprinting and environment metadata (Python/PyTorch versions, GPU device name).
- **Commands Run:**
  - `$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"; python -m pytest -q tests/test_train.py`
- **Actual Results:** 3 passed, 0 failed.
- **Exit Criteria:** PASS — LR schedules, training loops, atomic checkpoints, schema versioning, and step-level resumption verified.

---

## Phase 6: Complete Experiment Orchestration (`sweep.py`)
- **Date:** 2026-11-20
- **Status:** PASS
- **Files Inspected:** `sweep.py`
- **Files Modified:** `sweep.py`
- **Files Created:** `tests/test_sweep.py`
- **Important Decisions:**
  - Preserved standard model size mappings (XS, S, M, L) and seeds [0, 1, 2].
  - Implemented `is_run_valid`: validates JSON structure, finite metrics, complete history matching expected iterations, and configuration consistency.
  - Implemented quarantine mechanism: moves corrupted or incomplete `result.json` files to a timestamped `.corrupt` path so flawed runs are safely retried rather than silently accepted or discarded.
  - Verified that running a sweep twice detects verified completed runs and skips redundant computation.
  - Added CLI options: `--force`, `--device`, `--ablate`, `--quick`.
- **Commands Run:**
  - `$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"; python -m pytest -q tests/test_sweep.py`
- **Actual Results:** 2 passed, 0 failed.
- **Exit Criteria:** PASS — Experiment sweep orchestration verified; skip logic, schema checks, and corruption quarantine proven.

---

## Phase 7: Evaluation, Statistics and Plotting (`plot.py`)
- **Date:** 2026-11-20
- **Status:** PASS
- **Files Inspected:** `plot.py`
- **Files Modified:** `plot.py`
- **Files Created:** `tests/test_plot.py`
- **Important Decisions:**
  - Implemented safe directory parsing avoiding regex/split index crashes on unexpected folder structures.
  - Implemented exact sample standard deviation with $\text{ddof}=1$.
  - Implemented two-sided Student-t 95% Confidence Interval half-width using verified critical values ($t=4.303$ for $n=3$, $\text{df}=2$).
  - Accurately labeled error bars on the scaling plot as "mean \u00b1 sample std, ddof=1" to eliminate ambiguity with confidence intervals.
  - Added export of machine-readable `summary.json` alongside `summary.md`, `scaling.png`, and `curves.png`.
  - Added explicit caveats: 3 seeds provide preliminary sample dispersion; 4 small models on Tiny Shakespeare do not establish a universal scaling law.
- **Commands Run:**
  - `$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"; python -m pytest -q tests/test_plot.py`
- **Actual Results:** 3 passed, 0 failed.
- **Exit Criteria:** PASS — Statistical calculations, plotting, confidence intervals, and machine-readable exports verified.

---

## Phase 8: Model Generation and Stable Inference Interface
- **Date:** 2026-11-20
- **Status:** PASS
- **Files Created:** `service.py`, `tests/test_service.py`
- **Files Modified:** `sample.py`
- **Important Decisions:**
  - Implemented `ModelRegistry` for discovering and validating checkpoints from designated local paths without exposing arbitrary file paths to external users.
  - Implemented `ModelService` providing `load_model`, `list_available_models`, and `generate_text` with thread-safe in-memory caching.
  - Implemented strict character-level prompt validation raising `ValueError` for out-of-vocabulary characters.
  - Refactored `sample.py` to wrap `ModelService`, eliminating redundant sampling logic while preserving CLI compatibility.
- **Commands Run:**
  - `$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"; python -m pytest -q tests/test_service.py`
- **Actual Results:** 3 passed, 0 failed.
- **Exit Criteria:** PASS — Unified inference service created and tested; ready for consumption by Ragunathreddy's FastAPI application.

---

## Phase 9: Test Strategy and Test Completion
- **Date:** 2026-11-20
- **Status:** PASS
- **Files Inspected:** `tests/test_model.py`, `tests/test_data.py`, `tests/test_train.py`, `tests/test_sweep.py`, `tests/test_plot.py`, `tests/test_service.py`
- **Important Decisions:**
  - Expanded test coverage from 5 tests (with 1 failure) to 27 comprehensive, fully passing tests.
  - Implemented offline-safe testing for all modules: no tests require live internet access or external API calls.
  - Test categories verified:
    - Model: shapes, causality, unlearned uniform loss with independent targets, weight tying downward bias regression, parameter counts for XS/S/M/L, Pre vs Post LN differences, and config validation.
    - Data: tokenization round-trip, OOV exceptions, 1-token shift invariant, and deterministic generator seeding.
    - Training: LR warmup/cosine schedule, CPU training convergence, checkpoint round-trip, and step-level resumption.
    - Orchestration: run validation, fingerprinting, and corrupt file quarantine.
    - Plotting: Student-t critical values, sample variance ($\text{ddof}=1$), and complete chart generation.
    - Service: registry lookup, model caching, and input validation.
- **Commands Run:**
  - `$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"; python -m pytest -v`
- **Actual Results:** 27 passed, 0 failed in 6.47s.
- **Exit Criteria:** PASS — Fast, deterministic, offline test suite fully operational and green.

---

## Phase 10: Colab Notebook Alignment (`colab_nanogpt.ipynb`)
- **Date:** 2026-11-20
- **Status:** PASS
- **Files Inspected:** `colab_nanogpt.ipynb`
- **Files Modified:** `colab_nanogpt.ipynb`
- **Important Decisions:**
  - Synchronized all `%%writefile` code cells in `colab_nanogpt.ipynb` with the hardened repository modules (`data.py`, `model.py`, `train.py`, `sweep.py`, `plot.py`, `sample.py`, `tests/test_model.py`), eliminating silent drift.
  - Enhanced environment validation cell to report Python version, PyTorch version, CUDA status, and specific GPU device name.
  - Updated test cell to run both `test_model.py` and `pytest -q`.
  - Maintained Google Drive optional backup workflow and zip download mechanics.
- **Commands Run:**
  - Validated notebook JSON parsing and cell count.
- **Actual Results:** Clean JSON syntax, 28 cells intact and synchronized.
- **Exit Criteria:** PASS — Notebook aligned with standalone repository codebase.

---

## Phase 11: Run Experiments and Generate Actual Research Results
- **Date:** 2026-11-20
- **Status:** PASS
- **Files Inspected:** `sweep.py`, `train.py`, `plot.py`, `results/`
- **Files Created:**
  - 12 run directories under `results/` (`xs_pre_seed0..2`, `s_pre_seed0..2`, `m_pre_seed0..2`, `l_pre_seed0..2`), each containing `ckpt.pt`, `resume_state.pt`, and `result.json`.
  - `figures/scaling.png`, `figures/curves.png`, `figures/summary.md`, `figures/summary.json`.
  - `figures_final/scaling.png`, `figures_final/curves.png`, `figures_final/summary.md`, `figures_final/summary.json`.
- **Important Decisions:**
  - Ran small CPU smoke sweep (`results_quick/`, `figures_quick/`) to verify orchestration and skip logic.
  - Executed the full 12-run Pre-LN baseline scaling study on local NVIDIA GeForce RTX 4050 Laptop GPU (6.44 GB VRAM) utilizing BFloat16 mixed precision.
  - Monitored iteration-level metrics for divergence: zero NaNs, all initial step-0 losses aligned near theoretical $\ln(65) \approx 4.1744$.
  - Computed sample standard deviations ($\text{ddof}=1$) and Student-t 95% Confidence Intervals ($t=4.303, n=3$).
  - Evaluated empirical scaling power-law fit: log-log slope is **-0.0653** for `best_val` and **-0.0625** for `final_val`.
  - Observed capacity-induced overfitting on Size L (10.72M parameters): train loss dropped to $0.98$, while validation loss hit minimum $1.4661$ at step ~2000 and drifted upward to $1.4876$ at step 3000.
  - Verified inference generation using `sample.py` on trained checkpoint `results/m_pre_seed0/ckpt.pt` with prompt `"ROMEO:"`.
- **Commands Run:**
  - `python sweep.py --out results_quick --quick`
  - `python sweep.py --out results --device cuda --max_iters 3000`
  - `python plot.py --results results --out figures --metric best_val`
  - `python plot.py --results results --out figures_final --metric final_val`
  - `python sample.py --ckpt results/m_pre_seed0/ckpt.pt --prompt "ROMEO:" --tokens 200 --temperature 0.8 --top_k 40`
- **Actual Results:** All 12 baseline GPU runs completed, verified, and plotted without fabrication.
- **Exit Criteria:** PASS — Complete empirical baseline data generated, validated, and plotted.

---

## Phase 12: AI-Side Documentation and Handover
- **Date:** 2026-11-20
- **Status:** PASS
- **Files Created:** `EXPERIMENTS.md`, `MODEL_CARD.md`
- **Files Modified:** `README.md`, `WORK_LOG.md`
- **Important Decisions:**
  - Authored comprehensive `EXPERIMENTS.md` documenting experimental setup, parameter invariants, run matrix, empirical metrics, statistical method, overfitting analysis, and research limitations.
  - Authored `MODEL_CARD.md` detailing architecture, vocabulary ($V=65$), OOV exception policy, training hyperparameters, checkpoint schemas, and inference usage for downstream developers.
  - Updated `README.md` with actual empirical numbers, observed log-log scaling slope (-0.0653), overfitting findings on Model L, and verified resume bullet.
  - Preserved boundary constraints: strictly implemented Lalith's AI/ML workstream; zero code written for FastAPI, Streamlit, Docker, or CI/CD (owned by Ragunathreddy).
  - Preserved Operating Principle 9: no unauthorized remote git pushes to `https://github.com/Lalith-Tejas-M/Nano-GPT.git`.
- **Commands Run:**
  - Full test suite re-verification: `$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"; python -m pytest -v`
- **Actual Results:** 27 passed in 6.5s.
- **Exit Criteria:** PASS — All AI documentation completed, test suite green, ready for engineering handover.
