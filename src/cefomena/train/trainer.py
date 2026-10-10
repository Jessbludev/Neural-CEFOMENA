"""Bucle de entrenamiento controlado por la máquina de estados metabólicos."""

from __future__ import annotations

import copy
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from cefomena.model.model import CefomenaLM
from cefomena.train.data import TokenDataset
from cefomena.train.metabolic_fsm import MetabolicFSM, MetabolicState, ReplayBuffer


@dataclass
class TrainConfig:
    steps: int = 5000
    batch_size: int = 32
    context: int = 512
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    eval_batches: int = 20
    log_every: int = 50
    use_loss_mask: bool = True
    seed: int = 1337
    device: str = "auto"
    metabolic: MetabolicFSM = field(default_factory=MetabolicFSM)


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cosine_lr(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    cos = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return cfg.lr * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * cos)


def make_optimizer(
    model: CefomenaLM, cfg: TrainConfig, params: list[torch.nn.Parameter] | None = None
) -> torch.optim.Optimizer:
    params = params if params is not None else [p for p in model.parameters() if p.requires_grad]
    decay = [p for p in params if p.dim() >= 2]
    no_decay = [p for p in params if p.dim() < 2]
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        betas=(0.9, 0.95),
    )


@torch.no_grad()
def evaluate(
    model: CefomenaLM,
    data: TokenDataset,
    cfg: TrainConfig,
    device: torch.device,
    generator: torch.Generator,
) -> float:
    model.eval()
    losses = []
    for _ in range(cfg.eval_batches):
        x, y, m = data.get_batch(cfg.batch_size, cfg.context, generator, device)
        _, loss = model(x, y, m if cfg.use_loss_mask else None)
        if loss is None:
            raise RuntimeError("el modelo no devolvió pérdida")
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


@dataclass
class TrainResult:
    model: CefomenaLM
    ema_model: CefomenaLM
    history: list[dict[str, float | str]]


def train(
    model: CefomenaLM,
    train_data: TokenDataset,
    val_data: TokenDataset | None,
    cfg: TrainConfig,
    log: Callable[[dict[str, float | str]], None] | None = print,
) -> TrainResult:
    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg.device)
    model.to(device).train()
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = make_optimizer(model, cfg)
    gen = torch.Generator().manual_seed(cfg.seed)
    fsm = copy.deepcopy(cfg.metabolic)
    replay = ReplayBuffer()
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    history: list[dict[str, float | str]] = []
    t0 = time.time()

    for step in range(cfg.steps):
        state = fsm.state
        if state is MetabolicState.REST:
            if val_data is not None and fsm._t == 0:
                val = evaluate(model, val_data, cfg, device, gen)
                rec: dict[str, float | str] = {"step": step, "state": state.value, "val_loss": val}
                history.append(rec)
                if log:
                    log(rec)
            fsm.step()
            continue

        if fsm.use_replay() and len(replay) > 0:
            x, y, m = (t.to(device) for t in replay.sample(gen))
        else:
            x, y, m = train_data.get_batch(cfg.batch_size, cfg.context, gen, device)

        lr = cosine_lr(step, cfg) * fsm.lr_multiplier()
        for group in opt.param_groups:
            group["lr"] = lr
        with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            _, loss = model(x, y, m if cfg.use_loss_mask else None)
        if loss is None:
            raise RuntimeError("el modelo no devolvió pérdida")
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(opt)
        scaler.update()

        loss_val = float(loss.detach())
        if state is MetabolicState.ACTIVE:
            replay.add(loss_val, (x, y, m))
        decay = cfg.ema_decay if state is not MetabolicState.SLEEP else cfg.ema_decay**0.5
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters(), strict=True):
                pe.lerp_(pm, 1 - decay)

        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            rec = {
                "step": step,
                "state": state.value,
                "loss": loss_val,
                "lr": lr,
                "elapsed_s": round(time.time() - t0, 1),
            }
            history.append(rec)
            if log:
                log(rec)
        fsm.step()

    return TrainResult(model, ema, history)


def adapt_peripheral(
    model: CefomenaLM,
    data: TokenDataset,
    steps: int = 100,
    lr: float = 1e-3,
    batch_size: int = 8,
    context: int = 256,
) -> list[float]:
    """Adaptación rápida (mariposa): entrena solo el cache periférico."""
    device = next(model.parameters()).device
    peripheral = {id(p) for p in model.peripheral_parameters()}
    for p in model.parameters():
        p.requires_grad_(id(p) in peripheral)
    opt = torch.optim.AdamW(model.peripheral_parameters(), lr=lr, weight_decay=0.0)
    gen = torch.Generator().manual_seed(0)
    losses = []
    model.train()
    for _ in range(steps):
        x, y, m = data.get_batch(batch_size, context, gen, device)
        _, loss = model(x, y, m)
        if loss is None:
            raise RuntimeError("el modelo no devolvió pérdida")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    for p in model.parameters():
        p.requires_grad_(True)
    return losses
