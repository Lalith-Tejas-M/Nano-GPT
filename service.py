"""Framework-independent Python Model Service and Registry for nanoGPT.

Provides a clean, thread-safe inference interface for downstream applications
such as Ragunathreddy's FastAPI backend and Streamlit dashboard.
"""
import glob
import json
import os
import random

import torch

from data import compute_sha256
from model import GPT, GPTConfig


class ModelRegistry:
    """Discovers and resolves trusted model checkpoints within designated repository directories."""

    def __init__(self, search_dirs: list[str] | None = None):
        if search_dirs is None:
            search_dirs = ["out", "results"]
        self.search_dirs = search_dirs
        self._registry: dict[str, dict] = {}
        self.refresh()

    def refresh(self):
        """Scans search directories for valid checkpoints and registers them."""
        self._registry.clear()
        for base_dir in self.search_dirs:
            if not os.path.isdir(base_dir):
                continue
            for ckpt_path in glob.glob(os.path.join(base_dir, "**", "ckpt.pt"), recursive=True):
                dir_name = os.path.basename(os.path.dirname(ckpt_path))
                # Check if there is an accompanying result.json
                result_path = os.path.join(os.path.dirname(ckpt_path), "result.json")
                result_meta = {}
                if os.path.isfile(result_path):
                    try:
                        with open(result_path, encoding="utf-8") as f:
                            result_meta = json.load(f)
                    except Exception:
                        pass

                # Derive model_id from directory name (e.g. s_pre_seed0 or s_pre)
                model_id = dir_name
                self._registry[model_id] = {
                    "model_id": model_id,
                    "checkpoint_path": os.path.abspath(ckpt_path),
                    "dir_name": dir_name,
                    "metadata": result_meta,
                }

    def register_manual(self, model_id: str, checkpoint_path: str):
        """Manually registers a trusted checkpoint path."""
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        self._registry[model_id] = {
            "model_id": model_id,
            "checkpoint_path": os.path.abspath(checkpoint_path),
            "dir_name": os.path.basename(os.path.dirname(checkpoint_path)),
            "metadata": {},
        }

    def get(self, model_id: str) -> dict:
        """Retrieves checkpoint metadata for model_id. Raises KeyError if unknown."""
        if model_id not in self._registry:
            self.refresh()
        if model_id not in self._registry:
            available = list(self._registry.keys())
            raise KeyError(f"Model ID '{model_id}' not found in registry. Available models: {available}")
        return self._registry[model_id]

    def list_models(self) -> list[dict]:
        """Returns metadata list of all currently registered models."""
        self.refresh()
        return list(self._registry.values())


class ModelService:
    """Manages model loading, in-memory caching, character validation, and text generation."""

    def __init__(self, search_dirs: list[str] | None = None, default_device: str | None = None):
        self.registry = ModelRegistry(search_dirs)
        self.default_device = default_device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._cache: dict[str, dict] = {}

    def list_available_models(self) -> list[dict]:
        """Lists metadata for all registered models."""
        models = self.registry.list_models()
        res = []
        for m in models:
            is_cached = m["model_id"] in self._cache
            item = {
                "model_id": m["model_id"],
                "checkpoint_path": m["checkpoint_path"],
                "is_loaded": is_cached,
            }
            if is_cached:
                cached = self._cache[m["model_id"]]
                cfg = cached["config"]
                item.update(
                    {
                        "n_layer": cfg.n_layer,
                        "n_head": cfg.n_head,
                        "n_embd": cfg.n_embd,
                        "vocab_size": cfg.vocab_size,
                        "ln_placement": cfg.ln_placement,
                        "n_params": cached["model"].num_params(),
                    }
                )
            res.append(item)
        return res

    def load_model(self, model_id: str, device: str | None = None) -> dict:
        """Loads and caches a model in memory. Reuses cached instance if already loaded."""
        target_device = device or self.default_device
        cache_key = f"{model_id}:{target_device}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        meta = self.registry.get(model_id)
        ckpt_path = meta["checkpoint_path"]

        try:
            ck = torch.load(ckpt_path, map_location=target_device)
        except Exception as e:
            raise ValueError(f"Failed to load checkpoint at '{ckpt_path}': {e}") from e

        if not isinstance(ck, dict) or "model" not in ck or "model_config" not in ck or "chars" not in ck:
            raise ValueError(f"Checkpoint at '{ckpt_path}' is corrupted or missing required schema keys")

        cfg = GPTConfig(**ck["model_config"])
        model = GPT(cfg).to(target_device)
        model.load_state_dict(ck["model"])
        model.eval()

        chars = ck["chars"]
        stoi = {c: i for i, c in enumerate(chars)}
        itos = {i: c for i, c in enumerate(chars)}

        cached_entry = {
            "model_id": model_id,
            "model": model,
            "config": cfg,
            "chars": chars,
            "stoi": stoi,
            "itos": itos,
            "device": target_device,
        }
        self._cache[cache_key] = cached_entry
        self._cache[model_id] = cached_entry  # also index by model_id
        return cached_entry

    def generate_text(
        self,
        model_id: str,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int | None = 40,
        seed: int | None = None,
        device: str | None = None,
    ) -> dict:
        """Autoregressively generates text from a prompt with validation."""
        # Validate generation parameters
        if not (1 <= max_new_tokens <= 2000):
            raise ValueError(f"max_new_tokens must be in range [1, 2000], got {max_new_tokens}")
        if not (0.01 <= temperature <= 5.0):
            raise ValueError(f"temperature must be in range [0.01, 5.0], got {temperature}")
        if top_k is not None and top_k <= 0:
            raise ValueError(f"top_k must be a positive integer or None, got {top_k}")

        loaded = self.load_model(model_id, device=device)
        model = loaded["model"]
        stoi = loaded["stoi"]
        itos = loaded["itos"]
        model_device = loaded["device"]

        # Default prompt handling
        if prompt == "":
            prompt = "\n"

        # Validate character vocabulary
        unsupported = [c for c in prompt if c not in stoi]
        if unsupported:
            unique_unsupported = sorted(set(unsupported))
            raise ValueError(
                f"Unsupported character(s) in prompt: {unique_unsupported!r}. "
                f"Model vocabulary only supports: {loaded['chars']!r}"
            )

        # Seed handling
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        # Tokenize prompt
        idx = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long, device=model_device)

        # Generate tokens
        out = model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k)
        full_ids = out[0].tolist()
        completion_ids = full_ids[len(idx[0]) :]

        completion = "".join(itos[i] for i in completion_ids)
        full_text = "".join(itos[i] for i in full_ids)

        return {
            "model_id": model_id,
            "prompt": prompt,
            "completion": completion,
            "full_text": full_text,
            "tokens_generated": len(completion_ids),
            "temperature": temperature,
            "top_k": top_k,
            "seed": seed,
            "device": str(model_device),
        }
