import pytest

from cefomena.config import PRESETS, CefomenaConfig
from cefomena.model.model import CefomenaLM
from cefomena.train.data import TokenDataset, iter_conversations
from cefomena.train.metabolic_fsm import MetabolicFSM, MetabolicState, ReplayBuffer
from cefomena.train.trainer import TrainConfig, adapt_peripheral, train


def test_fsm_cycle():
    fsm = MetabolicFSM(active_steps=2, rest_steps=1, sleep_steps=2, wake_steps=1)
    seen = [fsm.state]
    for _ in range(6):
        seen.append(fsm.step())
    assert seen[:7] == [
        MetabolicState.ACTIVE,
        MetabolicState.ACTIVE,
        MetabolicState.REST,
        MetabolicState.SLEEP,
        MetabolicState.SLEEP,
        MetabolicState.WAKE,
        MetabolicState.ACTIVE,
    ]
    assert MetabolicFSM.inference_budget(MetabolicState.SLEEP, 9) < 9


def test_replay_keeps_highest_losses():
    import torch

    buf = ReplayBuffer(capacity=2)
    for loss in (1.0, 5.0, 3.0):
        buf.add(loss, (torch.tensor([loss]),))
    assert sorted(float(buf._heap[i][2][0]) for i in range(2)) == [3.0, 5.0]


@pytest.fixture
def dataset(tokenizer, synth_path):
    return TokenDataset.from_conversations(tokenizer, iter_conversations([synth_path]))


def test_smoke_train_reduces_loss(tokenizer, dataset):
    cfg = CefomenaConfig.from_dict(
        {**PRESETS["test"].to_dict(), "vocab_size": tokenizer.vocab_size}
    )
    train_ds, val_ds = dataset.split(0.1)
    tcfg = TrainConfig(
        steps=60,
        batch_size=8,
        context=32,
        lr=3e-3,
        warmup_steps=5,
        eval_batches=2,
        log_every=10,
        device="cpu",
        metabolic=MetabolicFSM(active_steps=30, rest_steps=1, sleep_steps=5, wake_steps=3),
    )
    result = train(CefomenaLM(cfg), train_ds, val_ds, tcfg, log=None)
    losses = [h["loss"] for h in result.history if "loss" in h]
    assert losses[-1] < losses[0]
    assert any(h["state"] == "rest" for h in result.history)


def test_adapt_peripheral_only_touches_cache(tokenizer, dataset):
    cfg = CefomenaConfig.from_dict(
        {**PRESETS["test"].to_dict(), "vocab_size": tokenizer.vocab_size}
    )
    model = CefomenaLM(cfg)
    before = model.embed.weight.detach().clone()
    adapt_peripheral(model, dataset, steps=3, batch_size=2, context=16)
    assert (model.embed.weight == before).all()
