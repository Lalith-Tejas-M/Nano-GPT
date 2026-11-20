"""Generate text from a saved checkpoint: python sample.py --ckpt out/run/ckpt.pt --prompt "ROMEO:" """
import argparse

import torch

from model import GPT, GPTConfig

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="out/run/ckpt.pt")
    p.add_argument("--prompt", default="\n")
    p.add_argument("--tokens", type=int, default=400)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    ck = torch.load(a.ckpt, map_location=device)
    model = GPT(GPTConfig(**ck["model_config"])).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    stoi = {c: i for i, c in enumerate(ck["chars"])}
    itos = {i: c for c, i in stoi.items()}
    idx = torch.tensor([[stoi[c] for c in a.prompt if c in stoi]], dtype=torch.long, device=device)
    out = model.generate(idx, a.tokens, temperature=a.temperature, top_k=a.top_k)
    print("".join(itos[int(i)] for i in out[0]))
