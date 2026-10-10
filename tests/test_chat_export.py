import pytest

from cefomena.chat.generate import ChatSession, GenerationConfig
from cefomena.chat.template import Message, ToolCall, parse_assistant, render, render_ids
from cefomena.config import PRESETS, CefomenaConfig
from cefomena.export.safetensors_io import load_pretrained, save_pretrained, verify_checksums
from cefomena.model.model import CefomenaLM


def test_render_and_mask(tokenizer):
    msgs = [Message("user", "hola"), Message("assistant", "¡Hola!")]
    assert render(msgs, add_generation_prompt=True).endswith("<|assistant|>")
    ids, mask = render_ids(tokenizer, msgs)
    assert len(ids) == len(mask)
    learned = tokenizer.decode([i for i, m in zip(ids, mask, strict=True) if m])
    assert learned == "¡Hola!<|end|>"


def test_parse_tool_call_with_noise():
    text = (
        '<|tool_call|> basura {"name": "calculadora", "arguments": {"expresion": "2+2"}} x<|end|>'
    )
    msg = parse_assistant(text)
    assert msg.tool_call == ToolCall("calculadora", {"expresion": "2+2"})
    assert parse_assistant("Hola<|end|>resto").content == "Hola"


def test_export_roundtrip_and_integrity(tokenizer, tmp_path):
    cfg = CefomenaConfig.from_dict(
        {**PRESETS["test"].to_dict(), "vocab_size": tokenizer.vocab_size}
    )
    model = CefomenaLM(cfg).eval()
    out = save_pretrained(model, tokenizer, tmp_path / "m")
    loaded, tok = load_pretrained(out)
    assert loaded.head.weight.data_ptr() == loaded.embed.weight.data_ptr()
    for a, b in zip(model.state_dict().values(), loaded.state_dict().values(), strict=True):
        assert (a == b).all()
    session = ChatSession(loaded, tok, generation=GenerationConfig(max_new_tokens=5, seed=0))
    reply = session.reply("hola")
    assert reply.role == "assistant"
    (out / "config.json").write_text((out / "config.json").read_text() + " ")
    with pytest.raises(ValueError, match="checksum"):
        verify_checksums(out)
