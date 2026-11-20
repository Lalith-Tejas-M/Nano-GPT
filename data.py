"""Character-level dataset (Tiny Shakespeare by default)."""
import os
import urllib.request

import torch

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def load_text(path="data/input.txt"):
    """Load the corpus, downloading Tiny Shakespeare on first use."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not os.path.exists(path):
        try:
            urllib.request.urlretrieve(URL, path)
        except Exception as e:  # network problems
            raise RuntimeError(
                f"Could not download the dataset ({e}). Download it manually from\n{URL}\n"
                f"and save it as {path}."
            )
    with open(path, encoding="utf-8") as f:
        return f.read()


class CharDataset:
    def __init__(self, text, val_frac=0.1):
        chars = sorted(set(text))
        self.chars = chars
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars)
        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        n = int(len(data) * (1 - val_frac))
        self.train, self.val = data[:n], data[n:]

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)

    def get_batch(self, split, batch_size, block_size, device, generator=None):
        d = self.train if split == "train" else self.val
        ix = torch.randint(len(d) - block_size - 1, (batch_size,), generator=generator)
        x = torch.stack([d[i : i + block_size] for i in ix])
        y = torch.stack([d[i + 1 : i + 1 + block_size] for i in ix])
        return x.to(device), y.to(device)
