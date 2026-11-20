"""Character-level dataset and preprocessing for nanoGPT (Tiny Shakespeare by default).

Responsibilities:
- Reliable, atomic dataset download with SHA-256 verification.
- Deterministic character-level tokenization (stoi, itos).
- Sequential train / validation split.
- Deterministic batch generation using explicit torch.Generator.
- Strict error handling for out-of-vocabulary characters.
"""
import hashlib
import os
import urllib.request

import torch

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
TINY_SHAKESPEARE_SHA256 = "4b6807d90d796e9526e033d5ba4b54e38c92a543f076b1580cae43b185e597c5"


def compute_sha256(text: str) -> str:
    """Returns SHA-256 hexadecimal digest of UTF-8 encoded text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_dataset_metadata(text: str) -> dict:
    """Computes basic structural metadata for a text corpus."""
    chars = sorted(set(text))
    return {
        "character_count": len(text),
        "byte_count": len(text.encode("utf-8")),
        "vocab_size": len(chars),
        "sha256": compute_sha256(text),
    }


def load_text(
    path: str = "data/input.txt",
    url: str = TINY_SHAKESPEARE_URL,
    verify_sha256: bool = False,
    expected_sha256: str | None = None,
) -> str:
    """Load the text corpus, downloading atomically to a temporary file on first use."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not os.path.exists(path):
        tmp_path = path + ".tmp"
        try:
            urllib.request.urlretrieve(url, tmp_path)
            with open(tmp_path, encoding="utf-8") as f:
                content = f.read()
            if verify_sha256 and expected_sha256:
                actual_hash = compute_sha256(content)
                if actual_hash != expected_sha256:
                    raise ValueError(
                        f"Dataset SHA-256 mismatch!\nExpected: {expected_sha256}\nActual:   {actual_hash}"
                    )
            os.replace(tmp_path, path)
            return content
        except Exception as e:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise RuntimeError(
                f"Could not load or download dataset ({e}).\n"
                f"Please manually download from:\n{url}\nand save as: {path}"
            ) from e

    with open(path, encoding="utf-8") as f:
        content = f.read()

    if verify_sha256 and expected_sha256:
        actual_hash = compute_sha256(content)
        if actual_hash != expected_sha256:
            raise ValueError(
                f"Dataset file at {path} has corrupted SHA-256!\nExpected: {expected_sha256}\nActual:   {actual_hash}"
            )
    return content


class CharDataset:
    """Character-level tokenizer and training dataset."""

    def __init__(self, text: str, val_frac: float = 0.1):
        if not isinstance(text, str) or len(text) == 0:
            raise ValueError("text must be a non-empty string")
        if not (0.0 < val_frac < 1.0):
            raise ValueError(f"val_frac must be in the open interval (0.0, 1.0), got {val_frac}")

        chars = sorted(set(text))
        self.chars = chars
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars)

        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        n = int(len(data) * (1 - val_frac))
        self.train, self.val = data[:n], data[n:]

        if len(self.train) == 0 or len(self.val) == 0:
            raise ValueError(
                f"Dataset split resulted in empty slice: train length {len(self.train)}, val length {len(self.val)}"
            )

    def encode(self, s: str) -> list[int]:
        """Encodes string to integer token IDs. Raises ValueError on unknown characters."""
        unknown = [c for c in s if c not in self.stoi]
        if unknown:
            unique_unk = sorted(set(unknown))
            raise ValueError(
                f"Unsupported character(s) in prompt: {unique_unk!r}. "
                f"Vocabulary only contains {self.vocab_size} characters: {self.chars!r}"
            )
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int] | torch.Tensor) -> str:
        """Decodes integer token IDs back to a string."""
        return "".join(self.itos[int(i)] for i in ids)

    def get_batch(
        self,
        split: str,
        batch_size: int,
        block_size: int,
        device: str | torch.device,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Samples a batch of inputs x and targets y with y shifted by +1."""
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")

        d = self.train if split == "train" else self.val
        if len(d) <= block_size:
            raise ValueError(
                f"Dataset slice '{split}' has length {len(d)}, which is <= block_size {block_size}"
            )

        high = len(d) - block_size
        ix = torch.randint(high, (batch_size,), generator=generator)
        x = torch.stack([d[i : i + block_size] for i in ix])
        y = torch.stack([d[i + 1 : i + 1 + block_size] for i in ix])
        return x.to(device), y.to(device)
