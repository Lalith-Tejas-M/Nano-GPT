"""Training loop for nanoGPT with AdamW, cosine LR schedule, mixed precision, and step-level resumption.

Supports execution as:
- A Python library: train(**overrides)
- A command-line script: python train.py --n_layer 4 --lr 1e-3
"""
import argparse
import hashlib
import json
import math
import os
import platform
import random
import time
from contextlib import nullcontext

import torch

from data import CharDataset, compute_sha256, load_text
from model import GPT, GPTConfig

DEFAULTS = dict(
    block_size=128,
    batch_size=64,
    n_layer=4,
    n_head=4,
    n_embd=128,
    dropout=0.1,
    ln_placement="pre",
    max_iters=3000,
    eval_interval=250,
    eval_iters=50,
    lr=1e-3,
    min_lr=1e-4,
    warmup=100,
    weight_decay=0.1,
    grad_clip=1.0,
    seed=0,
    out_dir="out/run",
    save_ckpt=True,
    save_resume=True,
    resume_from=None,
    dataset_path="data/input.txt",
    data_text=None,
    device=None,
)


def seed_everything(seed: int):
    """Seed Python, random, NumPy (if present), PyTorch CPU and all CUDA devices."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def get_lr(it: int, max_iters: int, lr: float, min_lr: float, warmup: int) -> float:
    """Linear warmup followed by cosine decay down to min_lr."""
    if it < warmup:
        return lr * (it + 1) / max(1, warmup)
    if it >= max_iters:
        return min_lr
    ratio = (it - warmup) / max(1, max_iters - warmup)
    return min_lr + 0.5 * (1 + math.cos(math.pi * ratio)) * (lr - min_lr)


def atomic_save(obj, filepath: str):
    """Atomically save an object or dictionary to disk using a temporary file."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    tmp_path = filepath + f".tmp_{os.getpid()}_{time.time_ns()}"
    if filepath.endswith(".json"):
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
    else:
        torch.save(obj, tmp_path)
    os.replace(tmp_path, filepath)


@torch.no_grad()
def estimate_loss(model: GPT, ds: CharDataset, cfg: dict, device: str, amp_ctx) -> dict:
    """Computes average loss over fixed evaluation batches.
    
    Using a fixed generator seed (1234) ensures loss numbers are directly comparable
    across evaluation checkpoints.
    """
    model.eval()
    out = {}
    for split in ("train", "val"):
        g = torch.Generator().manual_seed(1234)
        losses = []
        for _ in range(cfg["eval_iters"]):
            x, y = ds.get_batch(split, cfg["batch_size"], cfg["block_size"], device, g)
            with amp_ctx:
                _, loss = model(x, y)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


def train(**overrides) -> dict:
    """Executes a single model training run with checkpointing and metric logging."""
    cfg = {**DEFAULTS, **overrides}

    # Device selection
    if cfg["device"] is not None:
        device = cfg["device"]
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    seed_everything(cfg["seed"])

    # Load corpus
    if cfg["data_text"] is not None:
        raw_text = cfg["data_text"]
    else:
        raw_text = load_text(cfg["dataset_path"])

    ds = CharDataset(raw_text)
    text_sha256 = compute_sha256(raw_text)

    # Initialize model
    mc = GPTConfig(
        block_size=cfg["block_size"],
        vocab_size=ds.vocab_size,
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        n_embd=cfg["n_embd"],
        dropout=cfg["dropout"],
        ln_placement=cfg["ln_placement"],
    )
    model = GPT(mc).to(device)
    n_params = model.num_params()
    n_params_ne = model.num_params(non_embedding=True)

    # Setup mixed precision
    use_cuda = device.startswith("cuda")
    if use_cuda and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
        use_scaler = False
    elif use_cuda:
        amp_dtype = torch.float16
        use_scaler = True
    else:
        amp_dtype = torch.float32
        use_scaler = False

    amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_cuda else nullcontext()
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    # Optimizer: weight decay on 2D parameter tensors, no decay on 1D biases/LayerNorms
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg["weight_decay"]},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg["lr"],
        betas=(0.9, 0.95),
    )

    start_iter = 0
    history = []
    best_val = float("inf")
    t0 = time.time()
    g_train = torch.Generator().manual_seed(cfg["seed"])

    # Handle step-level resumption if requested
    if cfg["resume_from"] is not None:
        if not os.path.isfile(cfg["resume_from"]):
            raise FileNotFoundError(f"Resume checkpoint not found: {cfg['resume_from']}")
        checkpoint = torch.load(cfg["resume_from"], map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if use_scaler and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_iter = checkpoint["iteration"] + 1
        history = checkpoint.get("history", [])
        best_val = checkpoint.get("best_val", float("inf"))
        if "rng_state_cpu" in checkpoint:
            torch.set_rng_state(checkpoint["rng_state_cpu"].cpu())
        if use_cuda and "rng_state_cuda" in checkpoint and checkpoint["rng_state_cuda"] is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint["rng_state_cuda"]])
        if "python_rng_state" in checkpoint:
            random.setstate(checkpoint["python_rng_state"])
        print(f"Resumed training from iteration {start_iter} (best_val: {best_val:.4f})")

    # Training loop
    for it in range(start_iter, cfg["max_iters"] + 1):
        lr = get_lr(it, cfg["max_iters"], cfg["lr"], cfg["min_lr"], cfg["warmup"])
        for group in optimizer.param_groups:
            group["lr"] = lr

        # Periodic evaluation
        if it % cfg["eval_interval"] == 0 or it == cfg["max_iters"]:
            l = estimate_loss(model, ds, cfg, device, amp_ctx)
            if l["val"] < best_val:
                best_val = l["val"]
            elapsed = time.time() - t0
            history.append(
                dict(iter=it, train=l["train"], val=l["val"], lr=lr, seconds=elapsed)
            )
            print(
                f"iter {it:5d} | train {l['train']:.4f} | val {l['val']:.4f} | lr {lr:.2e} | {elapsed:.0f}s",
                flush=True,
            )

            # Save intermediate resumable state
            if cfg["save_resume"] and it < cfg["max_iters"]:
                resume_dict = {
                    "format_version": "1.0",
                    "iteration": it,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict() if use_scaler else None,
                    "history": history,
                    "best_val": best_val,
                    "model_config": mc.to_dict(),
                    "chars": ds.chars,
                    "cfg": cfg,
                    "rng_state_cpu": torch.get_rng_state(),
                    "rng_state_cuda": torch.cuda.get_rng_state_all() if use_cuda else None,
                    "python_rng_state": random.getstate(),
                }
                atomic_save(resume_dict, os.path.join(cfg["out_dir"], "resume_state.pt"))

        if it == cfg["max_iters"]:
            break

        # Sample batch & forward pass
        x, y = ds.get_batch("train", cfg["batch_size"], cfg["block_size"], device, g_train)
        with amp_ctx:
            _, loss = model(x, y)

        if not math.isfinite(loss.item()):
            raise FloatingPointError(f"Loss became non-finite ({loss.item()}) at iteration {it}")

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

    total_time = time.time() - t0
    final_train = history[-1]["train"] if history else float("nan")
    final_val = history[-1]["val"] if history else float("nan")
    best_val_calc = min((h["val"] for h in history), default=float("nan"))

    # Construct unique run fingerprint
    fingerprint_raw = f"{cfg['n_layer']}_{cfg['n_head']}_{cfg['n_embd']}_{cfg['ln_placement']}_{cfg['seed']}_{cfg['max_iters']}_{cfg['lr']}"
    fingerprint = hashlib.sha256(fingerprint_raw.encode("utf-8")).hexdigest()[:16]

    result = dict(
        schema_version="1.0",
        fingerprint=fingerprint,
        config=cfg,
        n_params=n_params,
        n_params_non_embedding=n_params_ne,
        dataset=dict(vocab_size=ds.vocab_size, sha256=text_sha256),
        environment=dict(
            python_version=platform.python_version(),
            torch_version=torch.__version__,
            device=device,
            cuda_available=torch.cuda.is_available(),
            device_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        ),
        final_train=final_train,
        final_val=final_val,
        best_val=best_val_calc,
        history=history,
        seconds=total_time,
    )

    # Save final results atomically
    os.makedirs(cfg["out_dir"], exist_ok=True)
    atomic_save(result, os.path.join(cfg["out_dir"], "result.json"))

    if cfg["save_ckpt"]:
        ckpt_dict = dict(
            format_version="1.0",
            model_config=mc.to_dict(),
            model=model.state_dict(),
            chars=ds.chars,
            chars_sha256=compute_sha256("".join(ds.chars)),
            train_config=dict(
                max_iters=cfg["max_iters"],
                batch_size=cfg["batch_size"],
                lr=cfg["lr"],
                seed=cfg["seed"],
            ),
            metrics=dict(
                best_val=best_val_calc,
                final_val=final_val,
                final_train=final_train,
            ),
        )
        atomic_save(ckpt_dict, os.path.join(cfg["out_dir"], "ckpt.pt"))

    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="nanoGPT training CLI")
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            p.add_argument(f"--{k}", type=lambda s: s.lower() in ("1", "true", "yes"), default=v)
        elif v is None:
            p.add_argument(f"--{k}", type=str, default=None)
        else:
            p.add_argument(f"--{k}", type=type(v), default=v)
    args = vars(p.parse_args())
    try:
        train(**args)
    except Exception as e:
        import sys
        sys.stderr.write(f"Error during training: {e}\n")
        sys.exit(1)
