"""CLI de CEFOMENA: ``uv run cefomena <comando>``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from cefomena.config import PRESETS, CefomenaConfig
from cefomena.model.model import CefomenaLM
from cefomena.tokenizer.bpe import BPETokenizer


def _info(args: argparse.Namespace) -> None:
    cfg = PRESETS[args.preset]
    model = CefomenaLM(cfg)
    print(
        json.dumps(
            {"preset": args.preset, "params": model.num_parameters(), **cfg.to_dict()}, indent=2
        )
    )


def _synth(args: argparse.Namespace) -> None:
    from cefomena.train.sources import write_jsonl
    from cefomena.train.synth import synthetic_conversations

    n = write_jsonl(synthetic_conversations(args.n, args.seed), args.out)
    print(f"{n} conversaciones sintéticas → {args.out}")


def _oasst(args: argparse.Namespace) -> None:
    from cefomena.train.sources import download_oasst2, oasst2_spanish, write_jsonl

    src = Path(args.source) if args.source else download_oasst2()
    n = write_jsonl(oasst2_spanish(src), args.out)
    print(f"{n} conversaciones en español (oasst2) → {args.out}")


def _train_tokenizer(args: argparse.Namespace) -> None:
    from cefomena.train.data import iter_texts

    tok = BPETokenizer.train(iter_texts(args.data), args.vocab_size)
    tok.save(args.out)
    print(f"tokenizador ({tok.vocab_size} tokens) → {args.out}")


def _prepare(args: argparse.Namespace) -> None:
    from cefomena.train.data import TokenDataset, iter_conversations

    tok = BPETokenizer.load(args.tokenizer)
    ds = TokenDataset.from_conversations(tok, iter_conversations(args.data))
    ds.save(args.out)
    print(f"{len(ds)} tokens → {args.out}")


def _train(args: argparse.Namespace) -> None:
    from cefomena.export.safetensors_io import save_pretrained
    from cefomena.train.data import TokenDataset
    from cefomena.train.metabolic_fsm import MetabolicFSM
    from cefomena.train.trainer import TrainConfig, train

    tok = BPETokenizer.load(args.tokenizer)
    cfg = CefomenaConfig.from_dict({**PRESETS[args.preset].to_dict(), "vocab_size": tok.vocab_size})
    ds = TokenDataset.load(args.data)
    train_ds, val_ds = ds.split(0.05)
    tcfg = TrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        context=min(args.context, cfg.context_length),
        lr=args.lr,
        device=args.device,
        metabolic=MetabolicFSM(active_steps=args.active_steps),
    )
    result = train(CefomenaLM(cfg), train_ds, val_ds, tcfg)
    save_pretrained(result.ema_model.cpu(), tok, args.out)
    print(f"modelo exportado → {args.out}")


def _chat(args: argparse.Namespace) -> None:
    from cefomena.chat.generate import ChatSession
    from cefomena.export.safetensors_io import load_pretrained

    model, tok = load_pretrained(args.model)
    session = ChatSession(model, tok)
    print("CEFOMENA listo. Escribe 'salir' para terminar.")
    while True:
        try:
            user = input("tú> ").strip()
        except EOFError:
            break
        if user.lower() in {"salir", "exit"}:
            break
        reply = session.reply(user)
        print("cefomena>", reply.content or f"[herramienta] {reply.tool_call}")


def _agent(args: argparse.Namespace) -> None:
    from cefomena.agent import (
        Orchestrator,
        SafetyPolicy,
        ToolRegistry,
        computer_tools,
        default_tools,
        make_backend,
        model_policy,
    )
    from cefomena.chat.generate import ChatSession
    from cefomena.export.safetensors_io import load_pretrained

    model, tok = load_pretrained(args.model)
    registry = ToolRegistry(default_tools() + computer_tools(make_backend(args.backend)))

    def confirm(call: object) -> bool:
        return input(f"¿Permitir {call}? [s/N] ").strip().lower() == "s"

    safety = SafetyPolicy(confirm=confirm, audit_path=Path(args.audit) if args.audit else None)
    orch = Orchestrator(model_policy(ChatSession(model, tok, use_memory=False)), registry, safety)
    ep = orch.run(args.task)
    print(json.dumps(ep.to_dict(), ensure_ascii=False, indent=2))


def _push(args: argparse.Namespace) -> None:
    from cefomena.export.hub import push_to_hub

    print(push_to_hub(args.model, args.repo_id, private=args.private))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cefomena", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("info", help="muestra parámetros de un preset")
    s.add_argument("--preset", default="tiny", choices=sorted(PRESETS))
    s.set_defaults(func=_info)

    s = sub.add_parser("synth", help="genera diálogos sintéticos en español")
    s.add_argument("--n", type=int, default=5000)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True)
    s.set_defaults(func=_synth)

    s = sub.add_parser("oasst-es", help="extrae conversaciones en español de oasst2")
    s.add_argument("--source", help="ruta local a oasst2 trees.jsonl.gz (si no, se descarga)")
    s.add_argument("--out", required=True)
    s.set_defaults(func=_oasst)

    s = sub.add_parser("train-tokenizer")
    s.add_argument("--data", nargs="+", required=True)
    s.add_argument("--vocab-size", type=int, default=8192)
    s.add_argument("--out", required=True)
    s.set_defaults(func=_train_tokenizer)

    s = sub.add_parser("prepare", help="tokeniza JSONL a .npz")
    s.add_argument("--data", nargs="+", required=True)
    s.add_argument("--tokenizer", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(func=_prepare)

    s = sub.add_parser("train")
    s.add_argument("--data", required=True)
    s.add_argument("--tokenizer", required=True)
    s.add_argument("--preset", default="tiny", choices=sorted(PRESETS))
    s.add_argument("--steps", type=int, default=5000)
    s.add_argument("--batch-size", type=int, default=32)
    s.add_argument("--context", type=int, default=512)
    s.add_argument("--lr", type=float, default=3e-4)
    s.add_argument("--active-steps", type=int, default=800)
    s.add_argument("--device", default="auto")
    s.add_argument("--out", required=True)
    s.set_defaults(func=_train)

    s = sub.add_parser("chat")
    s.add_argument("--model", required=True)
    s.set_defaults(func=_chat)

    s = sub.add_parser("agent")
    s.add_argument("--model", required=True)
    s.add_argument("--task", required=True)
    s.add_argument("--backend", default="dryrun", choices=["dryrun", "xdotool", "pyautogui"])
    s.add_argument("--audit")
    s.set_defaults(func=_agent)

    s = sub.add_parser("push", help="sube el modelo a Hugging Face Hub")
    s.add_argument("--model", required=True)
    s.add_argument("--repo-id", required=True)
    s.add_argument("--private", action="store_true")
    s.set_defaults(func=_push)
    return p


def main(argv: list[str] | None = None) -> int:
    torch.set_num_threads(max(1, torch.get_num_threads()))
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
