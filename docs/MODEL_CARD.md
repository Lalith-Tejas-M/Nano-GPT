# Model Card: nanoGPT Character-Level Transformer (Scaling Study)

## 1. Model Details

- **Model Name:** nanoGPT Character-Level Language Model (Scaling Study Suite)
- **Model Developer:** Lalith Tejas M (AI/ML Core & Research Workstream)
- **Architecture:** Autoregressive Decoder-Only Transformer based on the GPT architecture (Radford et al., 2019).
  - Multi-Head Causal Self-Attention with strictly triangular causal masking.
  - Position-wise Feed-Forward Network with GELU non-linearity ($4 \times$ expansion ratio).
  - Tied token embeddings and language model head weights (`wte.weight == lm_head.weight`).
  - Learned 1D absolute positional embeddings (`wpe`).
  - Pre-LayerNorm default topology (`ln_1` and `ln_2` placed before sublayer transformations, final `ln_f` before head); optional Post-LayerNorm topology.
- **Model Configurations:**
  - **XS (Extra-Small):** `n_layer=2`, `n_head=2`, `n_embd=64`, `block_size=128`.
    - Parameter Count: **112,448** (Pre-LN) / **112,320** (Post-LN).
  - **S (Small):** `n_layer=4`, `n_head=4`, `n_embd=128`, `block_size=128`.
    - Parameter Count: **818,048** (Pre-LN) / **817,792** (Post-LN).
  - **M (Medium):** `n_layer=6`, `n_head=6`, `n_embd=192`, `block_size=128`.
    - Parameter Count: **2,706,624** (Pre-LN) / **2,706,240** (Post-LN).
  - **L (Large):** `n_layer=6`, `n_head=6`, `n_embd=384`, `block_size=128`.
    - Parameter Count: **10,721,664** (Pre-LN) / **10,720,896** (Post-LN).
  - *Note on Parameter Calculation:* Analytical parameter formula for tied weights:
    $$\text{Params}_{\text{Pre-LN}} = (V \cdot C + B \cdot C) + L \cdot \left[ (4 C^2 + 4 C) + (8 C^2 + 5 C) + 4 C \right] + 2 C$$
    where $V=65$, $B=128$, $C=\text{n\_embd}$, $L=\text{n\_layer}$. Post-LN omits the final LayerNorm `ln_f`, having exactly $2 C$ fewer parameters.
- **Framework & Dependencies:** PyTorch 2.0+, Python 3.10+, NumPy, Matplotlib. Zero web-framework dependencies in core modeling files.

---

## 2. Intended Use

- **Primary Research Use:**
  - Empirical evaluation of neural scaling behavior (Kaplan et al., 2020) across model parameter orders of magnitude (0.11M to 10.7M parameters) on character-level language modeling.
  - Evaluation of layer normalization placement stability (Pre-LN vs. Post-LN).
  - Verification of step-level resumption and reproducibility protocols.
- **Demonstration & Portfolio Use:**
  - Educational reference for clean, hardened Transformer implementations.
  - Integration with downstream serving architectures (consumed by Ragunathreddy's FastAPI backend and Streamlit UI).
- **Out-of-Scope / Misuse:**
  - Production text generation, factual question answering, or automated decision-making.
  - High-stakes NLP tasks or open-domain dialogue systems.
  - Input sequences containing non-ASCII or out-of-vocabulary characters without preprocessing.

---

## 3. Training Data & Preprocessing

- **Dataset:** Tiny Shakespeare corpus (`data/input.txt`).
- **Source:** Standard Karpathy nanoGPT corpus extracted from Shakespeare plays.
- **Dataset Size:** 1,115,394 characters (~1.1 MB).
- **SHA-256 Checksum:** `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`.
- **Data Splits:**
  - Training set: First 90% (1,003,854 characters).
  - Validation set: Remaining 10% (111,540 characters).
- **Tokenization:**
  - Pure character-level tokenization.
  - Vocabulary Size: $V = 65$ unique characters (newline, space, 16 punctuation symbols, 10 digits, 26 uppercase letters, 26 lowercase letters).
- **Out-of-Vocabulary (OOV) Policy:**
  - Strict input validation: `CharDataset.encode()` and `ModelService.generate_text()` reject unmapped characters by raising `ValueError` with an explicit list of offending characters, preventing silent index corruption or invalid predictions.

---

## 4. Training Procedure & Hyperparameters

- **Hardware:** Local training on NVIDIA GeForce RTX 4050 Laptop GPU (6.44 GB VRAM, Compute Capability 8.9), with BFloat16 automatic mixed precision (`torch.autocast`).
- **Optimization:**
  - Optimizer: AdamW ($\beta_1 = 0.9, \beta_2 = 0.99, \epsilon = 10^{-8}$).
  - Weight Decay: Decoupled weight decay of $0.1$ applied strictly to 2D+ matrix and embedding weights (`wte`, `wpe`, `c_attn.weight`, `c_proj.weight`). Zero weight decay ($0.0$) applied to 1D biases and LayerNorm parameters.
  - Gradient Clipping: 1.0 maximum norm (`torch.nn.utils.clip_grad_norm_`).
- **Learning Rate Schedule:**
  - Linear warmup: 100 steps from $10^{-5}$ to $10^{-3}$.
  - Cosine decay: Down to $10^{-4}$ at 3,000 steps.
- **Batching & Sampling:**
  - Batch size: 64 sequences of 128 characters (8,192 characters / batch).
  - Random sampling with deterministic generator seeds (`torch.Generator`).
  - Multi-framework seeding: Python `random`, `os.environ["PYTHONHASHSEED"]`, NumPy, PyTorch CPU, and PyTorch CUDA.

---

## 5. Evaluation & Scaling Results

- **Theoretical Baseline:**
  - A uniform random guess over $V=65$ tokens yields an expected cross-entropy loss of:
    $$\mathcal{L}_{\text{uniform}} = -\ln(1 / 65) = \ln(65) \approx 4.1744 \text{ nats}$$
  - Step 0 evaluations confirm uninitialized models evaluate at $4.18 \pm 0.01$ nats.
- **Empirical Validation Scaling (Pre-LN, 3,000 Steps, Seeds 0, 1, 2):**
  - **XS (0.11M params):** Best Val Loss: $1.9580 \pm 0.0077$ | Final Val Loss: $1.9649 \pm 0.0076$.
  - **S (0.82M params):** Best Val Loss: $1.5896 \pm 0.0028$ | Final Val Loss: $1.6033 \pm 0.0031$.
  - **M (2.71M params):** Best Val Loss: $1.4823 \pm 0.0058$ | Final Val Loss: $1.4984 \pm 0.0069$.
  - **L (10.72M params):** Best Val Loss: $1.4647 \pm 0.0050$ | Final Val Loss: $1.4829 \pm 0.0051$.
- **Statistical Method:**
  - Sample standard deviation computed with Bessel's correction ($\text{ddof} = 1$).
  - 95% Confidence Intervals calculated via Student-t critical value $t = 4.303$ for $n=3$ ($\text{df}=2$).
- **Research Findings on Overfitting (Research Question 4):**
  - While XS, S, and M exhibit consistent power-law validation improvements, Model L demonstrates the onset of overfitting when trained for 3,000 steps without dropout on Tiny Shakespeare (1.1 MB).
  - Model L training loss drops to $0.98$ nats, but its validation loss bottoms at step ~2250 ($1.46$ nats) and drifts upward to $1.48$ nats by step 3000.
  - This demonstrates capacity saturation when parameter count exceeds corpus size by a factor of 10 in characters.

---

## 6. Model Artifacts & Interface Specifications

### Checkpoint Files
1. **Inference Checkpoint (`ckpt.pt`):**
   ```python
   {
       "model_state": OrderedDict(...),
       "config": GPTConfig(...),
       "step": int,
       "val_loss": float,
       "vocab_meta": {
           "chars": list[str],
           "vocab_size": 65,
           "stoi": dict[str, int],
           "itos": dict[int, str]
       },
       "schema_version": "1.0"
   }
   ```
2. **Training Resumption State (`resume_state.pt`):**
   - Contains model state, optimizer state, AMP GradScaler state, step counter, full evaluation history, best validation loss, and PyTorch/NumPy/Python RNG states.

### Serving Contract
- Wrapped by `service.ModelService`:
  ```python
  from service import ModelService

  service = ModelService(device="cuda")
  result = service.generate_text(
      model_path="results/m_pre_seed0/ckpt.pt",
      prompt="KING LEAR:\n",
      max_new_tokens=200,
      temperature=0.8,
      top_k=40
  )
  print(result["generated_text"])
  ```

---

## 7. Limitations & Ethical Considerations

- **Corpus Limitation:** Trained entirely on Shakespearean plays. Generates archaic drama dialogue, iambic-style cadence, and Renaissance vocabulary. Not suitable for modern conversational NLP.
- **Context Length:** Fixed window of 128 characters limits multi-paragraph context retention.
- **No Alignment / Safety Filter:** The model generates tokens purely based on learned character conditional probabilities without RLHF, constitutional AI, or safety filtering.
- **Environmental Impact:** All 12 baseline GPU runs completed in under 25 minutes total on an efficient mobile RTX 4050 GPU (<50W TGP), consuming minimal energy (<0.02 kWh).
