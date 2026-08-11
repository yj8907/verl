"""Unit tests for LLMTool (examples/oracle_trainer/llm_tool.py)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("anthropic")
pytest.importorskip("openai")

from verl.tools.schemas import OpenAIFunctionToolSchema  # noqa: E402

# llm_tool.py lives under examples/, not the verl package, so load it by path.
_MODULE_PATH = Path(__file__).parent / "llm_tool.py"
_spec = importlib.util.spec_from_file_location("oracle_trainer_llm_tool", _MODULE_PATH)
_llm_tool_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _llm_tool_mod
_spec.loader.exec_module(_llm_tool_mod)
LLMTool = _llm_tool_mod.LLMTool

TOOL_SCHEMA = OpenAIFunctionToolSchema.model_validate(
    {
        "type": "function",
        "function": {
            "name": "ask_llm",
            "description": "Ask a frontier LLM a question.",
            "parameters": {
                "type": "object",
                "properties": {"content": {"type": "string", "description": "The message to send."}},
                "required": ["content"],
            },
        },
    }
)


def _base_config(**overrides) -> dict:
    config = {"model": "claude-3-5-sonnet-20241022", "max_tokens": 256}
    config.update(overrides)
    return config


@pytest.fixture(autouse=True)
def _api_keys(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")


def _anthropic_response(text: str):
    block = type("Block", (), {"text": text})()
    return type("Message", (), {"content": [block]})()


def _openai_response(text: str):
    message = type("Message", (), {"content": text})()
    choice = type("Choice", (), {"message": message})()
    return type("ChatCompletion", (), {"choices": [choice]})()


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


def test_init_infers_anthropic_provider_from_model_name():
    tool = LLMTool(_base_config(model="claude-3-5-sonnet-20241022"), TOOL_SCHEMA)
    assert tool.provider == "anthropic"
    assert tool.model == "claude-3-5-sonnet-20241022"


def test_init_infers_openai_provider_from_model_name():
    tool = LLMTool(_base_config(model="gpt-4o"), TOOL_SCHEMA)
    assert tool.provider == "openai"


def test_init_respects_explicit_provider_override():
    tool = LLMTool(_base_config(model="some-custom-model", provider="openai"), TOOL_SCHEMA)
    assert tool.provider == "openai"


def test_init_raises_for_unsupported_provider():
    with pytest.raises(ValueError, match="Unsupported provider"):
        LLMTool(_base_config(model="some-custom-model", provider="cohere"), TOOL_SCHEMA)


def test_init_reads_default_system_prompt():
    tool = LLMTool(_base_config(system_prompt="You are terse."), TOOL_SCHEMA)
    assert tool.default_system_prompt == "You are terse."


def test_init_defaults_system_prompt_to_empty_string():
    tool = LLMTool(_base_config(), TOOL_SCHEMA)
    assert tool.default_system_prompt == ""


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_generates_instance_id_and_seeds_conversation():
    tool = LLMTool(_base_config(system_prompt="default prompt"), TOOL_SCHEMA)
    instance_id, response = await tool.create()

    assert instance_id in tool._conversations
    assert tool._conversations[instance_id] == {"system_prompt": "default prompt", "messages": []}
    assert response.text is None


@pytest.mark.asyncio
async def test_create_uses_provided_instance_id():
    tool = LLMTool(_base_config(), TOOL_SCHEMA)
    instance_id, _ = await tool.create(instance_id="my-id")
    assert instance_id == "my-id"
    assert "my-id" in tool._conversations


@pytest.mark.asyncio
async def test_create_overrides_default_system_prompt_per_instance():
    tool = LLMTool(_base_config(system_prompt="default prompt"), TOOL_SCHEMA)
    instance_id, _ = await tool.create(system_prompt="per-trajectory prompt")
    assert tool._conversations[instance_id]["system_prompt"] == "per-trajectory prompt"


# ---------------------------------------------------------------------------
# execute
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_anthropic_sends_system_prompt_and_returns_text():
    tool = LLMTool(_base_config(model="claude-3-5-sonnet-20241022", system_prompt="Be concise."), TOOL_SCHEMA)
    instance_id, _ = await tool.create()
    tool.client.messages.create = AsyncMock(return_value=_anthropic_response("42"))

    response, reward, metrics = await tool.execute(instance_id, {"content": "What is 6*7?"})

    tool.client.messages.create.assert_awaited_once_with(
        model="claude-3-5-sonnet-20241022",
        max_tokens=256,
        system="Be concise.",
        messages=[{"role": "user", "content": "What is 6*7?"}],
    )
    assert response.text == "42"
    assert reward == 0.0
    assert metrics == {}


@pytest.mark.asyncio
async def test_execute_openai_prepends_system_message_when_present():
    tool = LLMTool(_base_config(model="gpt-4o", system_prompt="Be concise."), TOOL_SCHEMA)
    instance_id, _ = await tool.create()
    tool.client.chat.completions.create = AsyncMock(return_value=_openai_response("42"))

    response, _, _ = await tool.execute(instance_id, {"content": "What is 6*7?"})

    tool.client.chat.completions.create.assert_awaited_once_with(
        model="gpt-4o",
        max_tokens=256,
        messages=[
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "What is 6*7?"},
        ],
    )
    assert response.text == "42"


@pytest.mark.asyncio
async def test_execute_openai_omits_system_message_when_absent():
    tool = LLMTool(_base_config(model="gpt-4o"), TOOL_SCHEMA)
    instance_id, _ = await tool.create()
    tool.client.chat.completions.create = AsyncMock(return_value=_openai_response("42"))

    await tool.execute(instance_id, {"content": "What is 6*7?"})

    called_messages = tool.client.chat.completions.create.await_args.kwargs["messages"]
    assert called_messages == [{"role": "user", "content": "What is 6*7?"}]


@pytest.mark.asyncio
async def test_execute_accumulates_multi_turn_history():
    tool = LLMTool(_base_config(model="claude-3-5-sonnet-20241022"), TOOL_SCHEMA)
    instance_id, _ = await tool.create()
    tool.client.messages.create = AsyncMock(
        side_effect=[_anthropic_response("first answer"), _anthropic_response("second answer")]
    )

    await tool.execute(instance_id, {"content": "first question"})
    await tool.execute(instance_id, {"content": "second question"})

    second_call_messages = tool.client.messages.create.await_args_list[1].kwargs["messages"]
    assert second_call_messages == [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
    ]


# ---------------------------------------------------------------------------
# release
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_release_drops_conversation_state():
    tool = LLMTool(_base_config(), TOOL_SCHEMA)
    instance_id, _ = await tool.create()
    assert instance_id in tool._conversations

    await tool.release(instance_id)

    assert instance_id not in tool._conversations


@pytest.mark.asyncio
async def test_release_is_a_noop_for_unknown_instance_id():
    tool = LLMTool(_base_config(), TOOL_SCHEMA)
    await tool.release("never-created")  # should not raise
