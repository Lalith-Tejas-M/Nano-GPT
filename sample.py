"""Generate text from a saved checkpoint via CLI.

Usage:
  python sample.py --ckpt out/run/ckpt.pt --prompt "ROMEO:" --tokens 200 --temperature 0.8 --top_k 40
"""
import argparse
import os

from service import ModelService


def main():
    p = argparse.ArgumentParser(description="nanoGPT text generation CLI")
    p.add_argument("--ckpt", default="out/run/ckpt.pt", help="Path to checkpoint (.pt) file")
    p.add_argument("--prompt", default="\n", help="Initial prompt text")
    p.add_argument("--tokens", type=int, default=200, help="Number of new tokens/characters to generate")
    p.add_argument("--temperature", type=float, default=0.8, help="Softmax sampling temperature")
    p.add_argument("--top_k", type=int, default=40, help="Top-k filtering limit (set <= 0 to disable)")
    p.add_argument("--seed", type=int, default=0, help="Random seed for generation")
    p.add_argument("--device", default=None, help="Device ('cpu', 'cuda')")
    a = p.parse_args()

    if not os.path.isfile(a.ckpt):
        import sys
        sys.stderr.write(f"Error: Checkpoint not found at '{a.ckpt}'. Please train a model first.\n")
        sys.exit(1)

    try:
        service = ModelService(default_device=a.device)
        service.registry.register_manual("cli_model", a.ckpt)

        top_k_val = None if a.top_k <= 0 else a.top_k
        res = service.generate_text(
            model_id="cli_model",
            prompt=a.prompt,
            max_new_tokens=a.tokens,
            temperature=a.temperature,
            top_k=top_k_val,
            seed=a.seed,
        )
        print(res["full_text"])
    except Exception as e:
        import sys
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
