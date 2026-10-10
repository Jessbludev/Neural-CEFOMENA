"""CefomenaLM: decoder causal híbrido bioinspirado."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from cefomena.config import CefomenaConfig
from cefomena.model.attention import CausalSelfAttention
from cefomena.model.common import RMSNorm
from cefomena.model.dual_cache import DualCache
from cefomena.model.fungal_memory import FungalMemory, FungalMemoryState
from cefomena.model.nervenet import NerveNetMixer
from cefomena.model.oscillator_router import OscillatorMoE
from cefomena.model.sparse_gate import SparseGateMLP


class NerveNetBlock(nn.Module):
    """Bloque 'N': mezcla local (medusa) + MLP escaso (delfín)."""

    def __init__(self, cfg: CefomenaConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.mixer = NerveNetMixer(cfg.d_model, cfg.conv_kernel, cfg.conv_dilation)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = SparseGateMLP(cfg.d_model, int(cfg.d_model * cfg.mlp_ratio), cfg.sparse_fraction)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, memory: FungalMemoryState | None = None) -> torch.Tensor:
        x = x + self.drop(self.mixer(self.norm1(x)))
        return x + self.drop(self.mlp(self.norm2(x)))


class AttentionBlock(nn.Module):
    """Bloque 'A': atención causal + lectura de memoria fúngica + MoE de osciladores (coral)."""

    def __init__(self, cfg: CefomenaConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_heads, cfg.dropout, cfg.rope_base)
        self.norm_mem = RMSNorm(cfg.d_model)
        self.memory = FungalMemory(cfg.d_model, cfg.memory_slots)
        self.norm2 = RMSNorm(cfg.d_model)
        self.moe = OscillatorMoE(
            cfg.d_model,
            cfg.n_experts,
            int(cfg.d_model * cfg.expert_ratio),
            cfg.sparse_fraction,
            cfg.oscillator_coupling,
            cfg.oscillator_sync_steps,
        )
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, memory: FungalMemoryState | None = None) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.memory(self.norm_mem(x), memory))
        return x + self.drop(self.moe(self.norm2(x)))


class CefomenaLM(nn.Module):
    def __init__(self, cfg: CefomenaConfig) -> None:
        super().__init__()
        self.config = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.dual_cache = DualCache(cfg.vocab_size, cfg.d_model, cfg.dual_cache_rank)
        self.blocks = nn.ModuleList(
            NerveNetBlock(cfg) if kind == "N" else AttentionBlock(cfg) for kind in cfg.layer_pattern
        )
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith(("out_proj.weight", "proj.weight", "down.weight", "out.weight")):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layers))
        if cfg.tie_embeddings:
            self.head.weight = self.embed.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def memory_modules(self) -> list[FungalMemory]:
        return [b.memory for b in self.blocks if isinstance(b, AttentionBlock)]

    def initial_memory(self) -> list[FungalMemoryState]:
        return [m.initial_state() for m in self.memory_modules()]

    def peripheral_parameters(self) -> list[nn.Parameter]:
        """Parámetros del cache periférico (mariposa) para adaptación rápida."""
        return list(self.dual_cache.parameters())

    def reward_parameters(self) -> list[nn.Parameter]:
        """Compuertas escasas (delfín) que ajusta el canal de recompensa."""
        params: list[nn.Parameter] = []
        for m in self.modules():
            if isinstance(m, SparseGateMLP):
                params.extend(m.reward_parameters())
        return params

    def hidden_states(
        self,
        idx: torch.Tensor,
        memory: list[FungalMemoryState] | None = None,
        exit_after: int | None = None,
    ) -> torch.Tensor:
        x = self.dual_cache(idx, self.embed(idx))
        n_blocks = (
            len(self.blocks) if exit_after is None else max(1, min(exit_after, len(self.blocks)))
        )
        mem_iter = iter(memory) if memory is not None else None
        for block in list(self.blocks)[:n_blocks]:
            state = None
            if isinstance(block, AttentionBlock) and mem_iter is not None:
                state = next(mem_iter, None)
            x = block(x, state)
        return self.norm(x)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        memory: list[FungalMemoryState] | None = None,
        exit_after: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if idx.shape[1] > self.config.context_length:
            raise ValueError("secuencia más larga que context_length")
        logits = self.head(self.hidden_states(idx, memory, exit_after))
        loss = None
        if targets is not None:
            per_tok = F.cross_entropy(
                logits.float().view(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
            )
            if loss_mask is not None:
                m = loss_mask.reshape(-1).float()
                loss = (per_tok * m).sum() / m.sum().clamp_min(1.0)
            else:
                loss = per_tok.mean()
        return logits, loss

    @torch.no_grad()
    def write_memory(
        self, memory: list[FungalMemoryState], idx: torch.Tensor
    ) -> list[FungalMemoryState]:
        """Resume un turno (idx: (1, T)) y lo escribe en cada memoria fúngica."""
        x = self.dual_cache(idx, self.embed(idx))
        new_states: list[FungalMemoryState] = []
        mem_iter = iter(memory)
        for block in self.blocks:
            if isinstance(block, AttentionBlock):
                state = next(mem_iter)
                trace = block.norm_mem(x).mean(dim=1)
                new_states.append(block.memory.write(state, trace))
                x = block(x, state)
            else:
                x = block(x)
        return new_states

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int | None = 50,
        top_p: float | None = 0.95,
        stop_ids: set[int] | None = None,
        memory: list[FungalMemoryState] | None = None,
        exit_after: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.config.context_length :]
            logits, _ = self(ctx, memory=memory, exit_after=exit_after)
            next_id = sample_next(logits[:, -1, :], temperature, top_k, top_p, generator)
            idx = torch.cat([idx, next_id], dim=1)
            if stop_ids and int(next_id[0, 0]) in stop_ids:
                break
        return idx


def sample_next(
    logits: torch.Tensor,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits.float() / temperature
    if top_k is not None and top_k < logits.shape[-1]:
        kth = logits.topk(top_k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
        cum = sorted_logits.softmax(-1).cumsum(-1)
        remove = cum - sorted_logits.softmax(-1) > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    probs = logits.softmax(-1)
    return torch.multinomial(probs, 1, generator=generator)
