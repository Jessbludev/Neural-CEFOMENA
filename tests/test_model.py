import torch

from cefomena.config import PRESETS, CefomenaConfig
from cefomena.model.model import CefomenaLM


def test_forward_loss(model):
    idx = torch.randint(0, model.config.vocab_size, (2, 16))
    logits, loss = model(idx, idx)
    assert logits.shape == (2, 16, model.config.vocab_size)
    assert loss is not None and torch.isfinite(loss)


def test_causality(model):
    torch.manual_seed(1)
    a = torch.randint(0, model.config.vocab_size, (1, 20))
    b = a.clone()
    b[:, 12:] = torch.randint(0, model.config.vocab_size, (1, 8))
    la, _ = model(a)
    lb, _ = model(b)
    assert torch.allclose(la[:, :12], lb[:, :12], atol=1e-5)


def test_early_exit_and_memory(model):
    idx = torch.randint(0, model.config.vocab_size, (1, 10))
    mem = model.initial_memory()
    mem = model.write_memory(mem, idx)
    logits, _ = model(idx, memory=mem, exit_after=1)
    assert logits.shape[-1] == model.config.vocab_size


def test_generate_stops(model):
    idx = torch.zeros(1, 3, dtype=torch.long)
    out = model.generate(idx, 5, temperature=0.0)
    assert out.shape[1] == 8


def test_tiny_preset_budget():
    n = CefomenaLM(PRESETS["tiny"]).num_parameters()
    assert 15_000_000 < n < 40_000_000


def test_config_roundtrip(tmp_path):
    cfg = PRESETS["test"]
    cfg.save(tmp_path / "c.json")
    assert CefomenaConfig.load(tmp_path / "c.json") == cfg
