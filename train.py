"""Training loop for the small GPT. Use as a library (train(**overrides)) or from the CLI."""
import argparse
import json
import math
import os
import time
from contextlib import nullcontext

import torch

from data import CharDataset, load_text
from model import GPT, GPTConfig

DEFAULTS = dict(
    block_size=128, batch_size=64,
    n_layer=4, n_head=4, n_embd=128, dropout=0.1, ln_placement="pre",
    max_iters=3000, eval_interval=250, eval_iters=50,
    lr=1e-3, min_lr=1e-4, warmup=100, weight_decay=0.1, grad_clip=1.0,
    seed=0, out_dir="out/run", save_ckpt=True,
)


def get_lr(it, max_iters, lr, min_lr, warmup):
    """Linear warmup, then cosine decay to min_lr."""
    if it < warmup:
        return lr * (it + 1) / warmup
    if it >= max_iters:
        return min_lr
    ratio = (it - warmup) / max(1, max_iters - warmup)
    return min_lr + 0.5 * (1 + math.cos(math.pi * ratio)) * (lr - min_lr)


@torch.no_grad()
def estimate_loss(model, ds, cfg, device, amp_ctx):
    """Average loss over fixed batches (same batches every call, so numbers are comparable)."""
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


def train(**overrides):
    cfg = {**DEFAULTS, **overrides}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg["seed"])

    ds = CharDataset(load_text())
    mc = GPTConfig(
        block_size=cfg["block_size"], vocab_size=ds.vocab_size, n_layer=cfg["n_layer"],
        n_head=cfg["n_head"], n_embd=cfg["n_embd"], dropout=cfg["dropout"],
        ln_placement=cfg["ln_placement"],
    )
    model = GPT(mc).to(device)
    n_params = model.num_params()
    n_params_ne = model.num_params(non_embedding=True)

    use_cuda = device == "cuda"
    amp_dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else torch.float16
    amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_cuda else nullcontext()
    scaler = torch.amp.GradScaler("cuda", enabled=(use_cuda and amp_dtype == torch.float16))

    # no weight decay on biases and LayerNorm params (1-D tensors)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg["lr"], betas=(0.9, 0.95),
    )

    g = torch.Generator().manual_seed(cfg["seed"])
    history, t0 = [], time.time()
    for it in range(cfg["max_iters"] + 1):
        lr = get_lr(it, cfg["max_iters"], cfg["lr"], cfg["min_lr"], cfg["warmup"])
        for group in optimizer.param_groups:
            group["lr"] = lr

        if it % cfg["eval_interval"] == 0 or it == cfg["max_iters"]:
            l = estimate_loss(model, ds, cfg, device, amp_ctx)
            history.append(dict(iter=it, train=l["train"], val=l["val"], lr=lr, seconds=time.time() - t0))
            print(f"iter {it:5d} | train {l['train']:.4f} | val {l['val']:.4f} | lr {lr:.2e} | {time.time()-t0:.0f}s", flush=True)
        if it == cfg["max_iters"]:
            break

        x, y = ds.get_batch("train", cfg["batch_size"], cfg["block_size"], device, g)
        with amp_ctx:
            _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

    result = dict(
        config=cfg, n_params=n_params, n_params_non_embedding=n_params_ne,
        final_train=history[-1]["train"], final_val=history[-1]["val"],
        best_val=min(h["val"] for h in history), history=history,
        seconds=time.time() - t0, device=device,
    )
    os.makedirs(cfg["out_dir"], exist_ok=True)
    with open(os.path.join(cfg["out_dir"], "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    if cfg["save_ckpt"]:
        torch.save(
            dict(model=model.state_dict(), model_config=mc.to_dict(), chars=ds.chars),
            os.path.join(cfg["out_dir"], "ckpt.pt"),
        )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            p.add_argument(f"--{k}", type=lambda s: s.lower() in ("1", "true", "yes"), default=v)
        else:
            p.add_argument(f"--{k}", type=type(v), default=v)
    args = vars(p.parse_args())
    train(**args)
