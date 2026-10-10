import torch

from cefomena.model.dual_cache import DualCache
from cefomena.model.fungal_memory import FungalMemory
from cefomena.model.nervenet import NerveNetMixer
from cefomena.model.oscillator_router import OscillatorMoE
from cefomena.model.sparse_gate import SparseGateMLP, k_winners


def test_k_winners_keeps_k():
    h = torch.randn(2, 5, 40).abs() + 0.1
    out = k_winners(h, 10)
    assert int((out != 0).sum(-1).max()) == 10


def test_sparse_gate_shape_and_reward_params():
    mlp = SparseGateMLP(16, 64, 0.25)
    assert mlp(torch.randn(2, 7, 16)).shape == (2, 7, 16)
    assert mlp.reward_parameters()[0] is mlp.gate.weight


def test_nervenet_is_causal():
    torch.manual_seed(0)
    mixer = NerveNetMixer(8, kernel=3, dilation=2)
    x = torch.randn(1, 12, 8)
    y = x.clone()
    y[:, 8:] = torch.randn(1, 4, 8)
    assert torch.allclose(mixer(x)[:, :8], mixer(y)[:, :8], atol=1e-6)


def test_oscillator_router_weights_sum_to_one():
    moe = OscillatorMoE(16, n_experts=3, hidden=32, sparse_fraction=0.5)
    x = torch.randn(2, 9, 16)
    w = moe.routing(x)
    assert torch.allclose(w.sum(-1), torch.ones(2, 9), atol=1e-5)
    assert moe(x).shape == x.shape
    assert torch.isclose(moe.last_load.sum(), torch.tensor(1.0), atol=1e-5)


def test_fungal_memory_local_rules():
    mem = FungalMemory(8, n_slots=4, novelty_threshold=0.9)
    state = mem.initial_state()
    trace = torch.ones(1, 8)
    new = mem.write(state, trace)
    weakest_replaced = torch.isclose(
        torch.nn.functional.cosine_similarity(new.slots, trace, dim=-1), torch.tensor(1.0), atol=0.2
    ).any()
    assert weakest_replaced
    reinforced = mem.write(new, trace)
    assert float(reinforced.conductance.max()) > float(new.conductance.max())
    assert mem(torch.randn(1, 3, 8), reinforced).shape == (1, 3, 8)


def test_dual_cache_starts_as_identity():
    dc = DualCache(50, 8, 2)
    idx = torch.randint(0, 50, (2, 5))
    x = torch.randn(2, 5, 8)
    assert torch.allclose(dc(idx, x), x)
