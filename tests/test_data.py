"""Offline unit tests for data.py using synthetic in-memory text."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
import torch

from data import CharDataset, compute_sha256, get_dataset_metadata


@pytest.fixture
def sample_text():
    return "The quick brown fox jumps over the lazy dog!\n1234567890\n"


def test_dataset_init_and_vocab(sample_text):
    ds = CharDataset(sample_text, val_frac=0.2)
    assert ds.vocab_size == len(sorted(set(sample_text)))
    assert ds.chars == sorted(set(sample_text))
    assert len(ds.train) + len(ds.val) == len(sample_text)


def test_encode_decode_roundtrip(sample_text):
    ds = CharDataset(sample_text)
    phrase = "quick fox 123"
    encoded = ds.encode(phrase)
    decoded = ds.decode(encoded)
    assert decoded == phrase
    assert all(isinstance(i, int) for i in encoded)


def test_oov_character_raises_value_error(sample_text):
    ds = CharDataset(sample_text)
    with pytest.raises(ValueError, match="Unsupported character.*'@'"):
        ds.encode("Hello @world")


def test_get_batch_shift_and_shapes(sample_text):
    ds = CharDataset(sample_text * 10, val_frac=0.1)
    g = torch.Generator().manual_seed(42)
    batch_size = 4
    block_size = 8
    x, y = ds.get_batch("train", batch_size, block_size, "cpu", generator=g)

    assert x.shape == (batch_size, block_size)
    assert y.shape == (batch_size, block_size)

    # Next-token prediction invariant: y is x shifted by exactly 1 token!
    # For each row, y[:, :-1] must equal x[:, 1:]
    assert torch.equal(x[:, 1:], y[:, :-1]), "Target y must be input x shifted by +1 token"


def test_deterministic_batch_sampling(sample_text):
    ds = CharDataset(sample_text * 10, val_frac=0.1)
    g1 = torch.Generator().manual_seed(1234)
    x1, y1 = ds.get_batch("train", 2, 4, "cpu", generator=g1)

    g2 = torch.Generator().manual_seed(1234)
    x2, y2 = ds.get_batch("train", 2, 4, "cpu", generator=g2)

    assert torch.equal(x1, x2)
    assert torch.equal(y1, y2)


def test_validation_errors():
    with pytest.raises(ValueError, match="non-empty string"):
        CharDataset("")
    with pytest.raises(ValueError, match="val_frac must be in the open interval"):
        CharDataset("abc", val_frac=1.5)

    ds = CharDataset("short", val_frac=0.2)
    with pytest.raises(ValueError, match="is <= block_size"):
        ds.get_batch("train", 1, 10, "cpu")


def test_sha256_metadata():
    text = "hello world"
    meta = get_dataset_metadata(text)
    assert meta["character_count"] == 11
    assert meta["vocab_size"] == 8
    assert len(meta["sha256"]) == 64
    assert meta["sha256"] == compute_sha256(text)
