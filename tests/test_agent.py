from cefomena.agent import (
    DryRunComputer,
    Orchestrator,
    SafetyPolicy,
    ToolRegistry,
    computer_tools,
    default_tools,
)
from cefomena.agent.orchestrator import AgentState, episodes_to_conversations
from cefomena.agent.tools import safe_calc
from cefomena.chat.template import Message, ToolCall


def scripted(replies):
    it = iter(replies)
    return lambda messages: next(it)


def registry(backend=None):
    return ToolRegistry(default_tools() + computer_tools(backend or DryRunComputer()))


def test_happy_path_computer_use(tmp_path):
    backend = DryRunComputer()
    policy = scripted(
        [
            Message("assistant", tool_call=ToolCall("observar", {})),
            Message("assistant", tool_call=ToolCall("escribir", {"texto": "hola"})),
            Message("assistant", "Listo."),
        ]
    )
    log = tmp_path / "episodes.jsonl"
    orch = Orchestrator(
        policy, registry(backend), SafetyPolicy(audit_path=tmp_path / "a.jsonl"), episode_log=log
    )
    ep = orch.run("escribe hola")
    assert ep.success and ep.final_answer == "Listo."
    assert backend.actions == [{"action": "type", "text": "hola"}]
    assert orch.state is AgentState.DONE
    assert log.read_text().count("\n") == 1
    assert episodes_to_conversations([ep]) == [ep.messages]


def test_dangerous_text_requires_confirmation():
    backend = DryRunComputer()
    policy = scripted(
        [
            Message("assistant", tool_call=ToolCall("escribir", {"texto": "sudo rm -rf /"})),
            Message("assistant", "No puedo hacerlo."),
        ]
    )
    ep = Orchestrator(policy, registry(backend), SafetyPolicy(confirm=lambda c: False)).run("x")
    assert backend.actions == []
    assert any(m.role == "tool" and m.content.startswith("denegado") for m in ep.messages)


def test_dangerous_key_blocked_without_confirm():
    backend = DryRunComputer()
    policy = scripted(
        [
            Message("assistant", tool_call=ToolCall("tecla", {"combinacion": "alt+F4"})),
            Message("assistant", "ok"),
        ]
    )
    Orchestrator(policy, registry(backend)).run("cierra")
    assert backend.actions == []


def test_recover_then_fail_on_empty_outputs():
    policy = scripted([Message("assistant", "")] * 5)
    orch = Orchestrator(policy, registry(), max_retries=2)
    ep = orch.run("x")
    assert not ep.success and orch.state is AgentState.FAILED


def test_step_limit():
    policy = scripted([Message("assistant", tool_call=ToolCall("observar", {}))] * 50)
    orch = Orchestrator(policy, registry(), SafetyPolicy(max_steps=3))
    ep = orch.run("x")
    assert not ep.success
    assert sum(1 for s in ep.steps if s.state == "act") == 3


def test_unknown_args_are_errors():
    policy = scripted(
        [
            Message("assistant", tool_call=ToolCall("calculadora", {"x": "1"})),
            Message("assistant", "fin"),
        ]
    )
    ep = Orchestrator(policy, registry()).run("x")
    assert any(m.role == "tool" and m.content.startswith("error") for m in ep.messages)


def test_safe_calc():
    assert safe_calc("2 + 3 * 4") == "14"
    for bad in ("__import__('os')", "2 ** 1000"):
        try:
            safe_calc(bad)
        except ValueError:
            continue
        raise AssertionError(bad)
