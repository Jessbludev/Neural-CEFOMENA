"""Sesión de chat sobre CefomenaLM con memoria fúngica y presupuesto metabólico."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from cefomena.chat.template import DEFAULT_SYSTEM_PROMPT, Message, parse_assistant, render
from cefomena.model.fungal_memory import FungalMemoryState
from cefomena.model.model import CefomenaLM
from cefomena.tokenizer.bpe import BPETokenizer
from cefomena.train.metabolic_fsm import MetabolicFSM, MetabolicState


@dataclass
class GenerationConfig:
    max_new_tokens: int = 200
    temperature: float = 0.7
    top_k: int | None = 40
    top_p: float | None = 0.9
    seed: int | None = None


@dataclass
class ChatSession:
    model: CefomenaLM
    tokenizer: BPETokenizer
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    metabolic_state: MetabolicState = MetabolicState.ACTIVE
    use_memory: bool = True
    history: list[Message] = field(default_factory=list)
    memory: list[FungalMemoryState] | None = None

    def __post_init__(self) -> None:
        if self.use_memory and self.memory is None:
            self.memory = self.model.initial_memory()

    def messages(self) -> list[Message]:
        return [Message("system", self.system_prompt), *self.history]

    def _device(self) -> torch.device:
        return next(self.model.parameters()).device

    def complete(self, messages: list[Message] | None = None) -> Message:
        msgs = messages if messages is not None else self.messages()
        prompt = render(msgs, add_generation_prompt=True)
        ids = self.tokenizer.encode(prompt)[-self.model.config.context_length + 1 :]
        x = torch.tensor([ids], dtype=torch.long, device=self._device())
        gen = None
        if self.generation.seed is not None:
            gen = torch.Generator(device=x.device).manual_seed(self.generation.seed)
        budget = MetabolicFSM.inference_budget(self.metabolic_state, self.model.config.n_layers)
        self.model.eval()
        out = self.model.generate(
            x,
            self.generation.max_new_tokens,
            temperature=self.generation.temperature,
            top_k=self.generation.top_k,
            top_p=self.generation.top_p,
            stop_ids={self.tokenizer.end_id, self.tokenizer.eos_id},
            memory=self.memory,
            exit_after=budget,
            generator=gen,
        )
        text = self.tokenizer.decode(out[0, x.shape[1] :].tolist())
        return parse_assistant(text)

    def add(self, message: Message) -> None:
        self.history.append(message)
        if self.memory is not None and message.content:
            ids = self.tokenizer.encode(message.content)[: self.model.config.context_length]
            if ids:
                x = torch.tensor([ids], dtype=torch.long, device=self._device())
                self.memory = self.model.write_memory(self.memory, x)
        max_turns = 24
        if len(self.history) > max_turns:
            self.history = self.history[-max_turns:]

    def reply(self, user_text: str) -> Message:
        self.add(Message("user", user_text))
        answer = self.complete()
        self.add(answer)
        return answer
