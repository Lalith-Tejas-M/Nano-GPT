# MODEL INTERFACE SPECIFICATION & INTEGRATION CONTRACT

**Document Version:** 1.0.0  
**Owner:** Lalith's AI/ML Engineering Workstream  
**Consumer:** Ragunathreddy's Application & MLOps Workstream (FastAPI backend, Streamlit dashboard)  
**Status:** Frozen Contract Draft

---

## 1. Overview and Scope

This document specifies the stable Python-level API, checkpoint format, model registry, inference contracts, and experiment result schemas provided by Lalith's AI/ML workstream. 

**Architectural Boundary:**
- Lalith provides framework-agnostic Python classes (`ModelService`, `ModelRegistry`) and pure Python data structures.
- Ragunathreddy imports and invokes these Python interfaces inside FastAPI routes and Streamlit views.
- No FastAPI, Starlette, or Streamlit dependencies are required by the core model code.

---

## 2. Python Service Interface (`service.py`)

A unified service module `service.py` is provided to encapsulate checkpoint loading, model caching, prompt validation, and autoregressive token generation.

### 2.1 Core Signatures

```python
class ModelService:
    def __init__(self, registry_dir: str = "out", device: str | None = None):
        """Initializes the service with a directory containing model checkpoints.
        
        Args:
            registry_dir: Path to directory containing trained model subdirectories or checkpoints.
            device: Optional explicit compute device ("cuda", "cpu"). Defaults to auto-detection.
        """
        ...

    def list_available_models(self) -> list[dict]:
        """Lists metadata for all registered models discovered in the registry.
        
        Returns:
            List of dictionaries containing model metadata:
            [
                {
                    "model_id": "s_pre",
                    "size_name": "s",
                    "ln_placement": "pre",
                    "n_layer": 4,
                    "n_head": 4,
                    "n_embd": 128,
                    "n_params": 818048,
                    "checkpoint_path": "out/s_pre_seed0/ckpt.pt",
                    "vocab_size": 65,
                    "is_loaded": True
                },
                ...
            ]
        """
        ...

    def load_model(self, model_id: str, device: str | None = None) -> None:
        """Loads and caches model weights into memory in eval mode.
        
        Args:
            model_id: Identifier of model (e.g., 'xs_pre', 's_pre', 'm_pre', 'l_pre').
            device: Optional device override for this specific model.
            
        Raises:
            KeyError: If model_id is not found in the registry.
            FileNotFoundError: If the model's checkpoint file is missing.
            ValueError: If the checkpoint is corrupted or incompatible.
        """
        ...

    def generate_text(
        self,
        model_id: str,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int | None = 40,
        seed: int | None = None,
    ) -> dict:
        """Generates text from a prompt using autoregressive sampling.
        
        Args:
            model_id: Registered model identifier.
            prompt: Seed text string. Must only contain characters present in model's vocabulary.
            max_new_tokens: Number of characters to generate (1 <= max_new_tokens <= 2000).
            temperature: Softmax sampling temperature (0.01 <= temperature <= 5.0).
            top_k: Top-k filtering limit (None or 1 <= top_k <= vocab_size).
            seed: Optional integer seed for deterministic generation.
            
        Returns:
            Dictionary containing generation results and metadata:
            {
                "model_id": "s_pre",
                "prompt": "ROMEO:",
                "completion": " What light through yonder window breaks?",
                "full_text": "ROMEO: What light through yonder window breaks?",
                "tokens_generated": 42,
                "temperature": 0.8,
                "top_k": 40,
                "seed": 42,
                "device": "cuda:0"
            }
            
        Raises:
            KeyError: If model_id is not registered.
            ValueError: If prompt contains unsupported characters, or if parameters violate bounds.
        """
        ...

    def load_experiment_metrics(self, results_dir: str = "results") -> dict:
        """Loads and returns aggregated experiment metrics, scaling data, and curves.
        
        Returns:
            Structured dictionary matching the schema defined in Section 4.
        """
        ...
```

---

## 3. Checkpoint Structure and Schema

Checkpoints are saved as PyTorch `.pt` files containing versioned dictionaries. Arbitrary pickle objects are prohibited; checkpoints only contain standard Python dictionaries, lists, strings, and PyTorch `state_dict` tensors.

### 3.1 Inference Checkpoint (`ckpt.pt`)
Stored in each run directory: `<out_dir>/ckpt.pt`.

```python
{
    "format_version": "1.0",
    "timestamp": 1732087200.0,
    "model_config": {
        "block_size": 128,
        "vocab_size": 65,
        "n_layer": 4,
        "n_head": 4,
        "n_embd": 128,
        "dropout": 0.1,
        "ln_placement": "pre"
    },
    "model": {
        # OrderedDict of PyTorch parameter tensors
        "wte.weight": ...,
        "wpe.weight": ...,
        "blocks.0.ln1.weight": ...,
        ...
    },
    "chars": ["\n", " ", "!", "$", "&", "'", ...],  # Ordered 65-character vocabulary
    "chars_sha256": "4b6807...",                     # Hash of sorted character string
    "train_config": {
        "max_iters": 3000,
        "batch_size": 64,
        "lr": 0.001,
        "seed": 0
    },
    "metrics": {
        "best_val": 1.6241,
        "final_val": 1.6310,
        "final_train": 1.4820
    }
}
```

### 3.2 Full Resumption Checkpoint (`resume_state.pt`)
Used only when resuming interrupted training.
Contains everything from `ckpt.pt` plus:
```python
{
    "optimizer_state_dict": ...,    # AdamW state (momentum buffers)
    "scaler_state_dict": ...,       # torch.amp.GradScaler state
    "iteration": 1250,              # Current training step
    "history": [...],               # Training evaluation logs up to step 1250
    "rng_state_cpu": ...,           # torch.get_rng_state()
    "rng_state_cuda": ...,          # torch.cuda.get_rng_state_all() if CUDA used
    "python_rng_state": ...         # random.getstate()
}
```

---

## 4. Experiment Result JSON Schema (`result.json`)

Saved at `<results_dir>/<size>_<ln>_seed<seed>/result.json`.

```json
{
  "schema_version": "1.0",
  "run_id": "s_pre_seed0",
  "fingerprint": "a9f82d1c6b3e4018",
  "model_size": "s",
  "ln_placement": "pre",
  "seed": 0,
  "config": {
    "block_size": 128,
    "batch_size": 64,
    "n_layer": 4,
    "n_head": 4,
    "n_embd": 128,
    "dropout": 0.1,
    "ln_placement": "pre",
    "max_iters": 3000,
    "eval_interval": 250,
    "eval_iters": 50,
    "lr": 0.001,
    "min_lr": 0.0001,
    "warmup": 100,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "seed": 0
  },
  "n_params": 818048,
  "n_params_non_embedding": 809728,
  "dataset": {
    "name": "tinyshakespeare",
    "vocab_size": 65,
    "sha256": "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
  },
  "environment": {
    "python_version": "3.10.11",
    "torch_version": "2.6.0+cu124",
    "device_name": "NVIDIA GeForce RTX ...",
    "cuda_available": true
  },
  "runtime_seconds": 184.2,
  "final_train": 1.4820,
  "final_val": 1.6310,
  "best_val": 1.6241,
  "history": [
    {
      "iter": 0,
      "train": 4.1850,
      "val": 4.1892,
      "lr": 0.00001,
      "seconds": 1.2
    },
    {
      "iter": 250,
      "train": 2.4510,
      "val": 2.4780,
      "lr": 0.00098,
      "seconds": 15.6
    }
  ]
}
```

---

## 5. Input Validation Rules and Policies

1. **Character Validation Policy:**
   - Prompts are strictly checked against the model's vocabulary (`chars`).
   - If any character in the prompt is not in `chars`, `ModelService.generate_text()` raises a `ValueError` indicating the exact offending characters:
     `ValueError: Unsupported character(s) in prompt: ['@', '#']. Vocabulary only supports valid Tiny Shakespeare characters.`
   - Downstream FastAPI applications should catch `ValueError` and return an HTTP `422 Unprocessable Entity` or `400 Bad Request` with the detail message.

2. **Parameter Bounds:**
   - `max_new_tokens`: Integer in range `[1, 2000]`.
   - `temperature`: Float in range `[0.01, 5.0]`. Values $\le 0$ are rejected.
   - `top_k`: Integer in range `[1, vocab_size]` or `None`.
   - `prompt`: String. Empty string `""` is automatically converted to default newline `"\n"`.

3. **Concurrency and Caching:**
   - Models are loaded into memory once and reused across requests.
   - Inference runs inside `torch.no_grad()` blocks and with `model.eval()`.
   - In PyTorch, model evaluation (`forward`) on a shared model instance is thread-safe for reading.
