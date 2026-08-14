# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for ``ContinualAgentLoop`` (no GPU required).

The state-machine handlers and the per-``(class, tag)`` instance cache are
exercised without going through ``AgentLoopBase.__init__`` (which needs a live
tokenizer / rollout config / LLM server) -- ``ContinualAgentLoop.__new__`` is
called directly to obtain a bare, un-``__init__``-ed instance, and the exact
attributes each handler touches are set by hand, mirroring the approach used
in ``test_call_tool_on_cpu.py`` for ``ToolAgentLoop._call_tool``.

``env_manager``, however, is the real ``LLMFeedbackEnvironmentManager`` --
the exact class ``ContinualAgentLoop`` wires up in production -- rather than a
``MagicMock`` stand-in, so this exercises the actual scoring/feedback
lifecycle end-to-end. This requires ``ANTHROPIC_API_KEY`` to be set (no
skip-if-missing fallback: tests raise ``KeyError`` immediately without it),
and tests where the sample answer is scored incorrect make a real, billed LLM
call for feedback (``max_tokens`` is kept small to bound cost). Everything
``LLMFeedbackEnvironmentManager`` doesn't touch -- the tokenizer, the event
loop's executor, ``apply_chat_template``, and ``ToolAgentLoop``'s own
generation/tool-calling internals -- stays mocked: making those real would
need a live HF tokenizer, rollout config, and inference server, which is out
of scope for a CPU-only test file.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from verl.experimental.agent_loop.continual_agent_loop import AgentState, ContinualAgentLoop
from verl.experimental.agent_loop.environment_manager import LLMFeedbackEnvironmentManager
from verl.experimental.agent_loop.tool_agent_loop import AgentData
from verl.experimental.agent_loop.tool_agent_loop import AgentState as ToolAgentState
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop


def _close_real_client_sync(env_manager):
    """Explicitly close a real ``LLMFeedbackEnvironmentManager``'s Anthropic
    client from a *synchronous* test. Without this, the client's own
    ``__del__`` finalizer races the test framework's event-loop teardown: it
    schedules its cleanup as a task on whatever loop is running when the
    client is garbage-collected, but that loop can be closed before the task
    runs, producing a spurious 'Task exception was never retrieved:
    RuntimeError(Event loop is closed)' warning at process exit. Async tests
    should instead just ``await env_manager.client.close()`` directly, while
    their own loop is still running."""
    client = getattr(env_manager, "client", None)
    if client is not None:
        asyncio.run(client.close())


def _make_bare_loop(**attrs):
    """A ``ContinualAgentLoop`` instance created via ``__new__`` only, so
    ``__init__`` (env manager / tool setup) never runs. The cache entry is
    popped immediately so these throwaway instances don't leak into the
    process-wide ``_instances`` dict shared by other tests."""
    tag = f"bare-{uuid4().hex}"
    instance = ContinualAgentLoop.__new__(ContinualAgentLoop, name=tag)
    ContinualAgentLoop._instances.pop((ContinualAgentLoop, tag), None)
    for key, value in attrs.items():
        setattr(instance, key, value)
    return instance


def _make_real_env_manager(tag_prefix="continual-env"):
    """Builds a real, fully ``__init__``-ed ``LLMFeedbackEnvironmentManager``
    backed by a real Anthropic client -- requires ``ANTHROPIC_API_KEY``.
    Returns ``(instance, tag)``; callers must pop
    ``LLMFeedbackEnvironmentManager._instances[(LLMFeedbackEnvironmentManager, tag)]``
    in a ``finally`` block so the throwaway instance doesn't leak into other
    tests."""
    tag = f"{tag_prefix}-{uuid4().hex}"
    config = {"name": tag, "max_tokens": 64}
    instance = LLMFeedbackEnvironmentManager.__new__(LLMFeedbackEnvironmentManager, config=config)
    LLMFeedbackEnvironmentManager._instances.pop((LLMFeedbackEnvironmentManager, tag), None)
    LLMFeedbackEnvironmentManager.__init__(instance, config=config)
    return instance, tag


def _make_agent_data(**overrides):
    data = AgentData(
        messages=[{"role": "user", "content": "What is 2+2?"}],
        image_data=None,
        video_data=None,
        audio_data=None,
        mm_processor_kwargs=None,
        metrics={},
        request_id="req-1",
        tools_kwargs={},
    )
    data.prompt_ids = [1, 2, 3]
    data.response_ids = [1, 2, 3]
    data.response_mask = [1, 1, 1]
    data._env_instance_id = "env-1"
    data._ground_truth = "4"
    for key, value in overrides.items():
        setattr(data, key, value)
    return data


def _run_in_executor(_executor, func, *args):
    """Fake ``asyncio`` executor that just calls ``func`` inline."""
    return func(*args)


class TestHandleObservingState(unittest.IsolatedAsyncioTestCase):
    async def test_correct_answer_terminates(self):
        env_manager, env_tag = _make_real_env_manager()
        try:
            env_instance_id = await env_manager.create()
            agent_data = _make_agent_data(_env_instance_id=env_instance_id, _ground_truth="4")
            loop = _make_bare_loop(response_length=1000, turn_separator=[])
            loop.loop = MagicMock(run_in_executor=AsyncMock(side_effect=_run_in_executor))
            loop.tokenizer = MagicMock(decode=MagicMock(return_value="4"))
            loop.env_manager = env_manager

            state = await ContinualAgentLoop._handle_observing_state(loop, agent_data)

            assert state == AgentState.TERMINATED
            assert agent_data.turn_scores == [1.0]
            # No feedback turn should be appended once the episode is done.
            assert agent_data.messages[-1] == {"role": "user", "content": "What is 2+2?"}
            # Correct answers short-circuit before any LLM call, so the real
            # environment manager never touched its conversation state.
            assert env_manager._conversations[env_instance_id]["messages"] == []
        finally:
            await env_manager.client.close()
            LLMFeedbackEnvironmentManager._instances.pop((LLMFeedbackEnvironmentManager, env_tag), None)

    async def test_incorrect_answer_appends_feedback_and_continues(self):
        env_manager, env_tag = _make_real_env_manager()
        try:
            env_instance_id = await env_manager.create()
            agent_data = _make_agent_data(_env_instance_id=env_instance_id, _ground_truth="4")
            loop = _make_bare_loop(response_length=1000, turn_separator=[9])
            loop.loop = MagicMock(run_in_executor=AsyncMock(side_effect=_run_in_executor))
            loop.tokenizer = MagicMock(decode=MagicMock(return_value="5"))
            loop.env_manager = env_manager
            loop.apply_chat_template = AsyncMock(return_value=[10, 11])

            state = await ContinualAgentLoop._handle_observing_state(loop, agent_data)

            assert state == AgentState.GENERATING
            assert agent_data.turn_scores == [0.0]
            # The feedback text is real, non-deterministic LLM output -- only its
            # shape is checked; the exact wording isn't pinned down.
            feedback_message = agent_data.messages[-1]
            assert feedback_message["role"] == "user"
            assert isinstance(feedback_message["content"], str) and feedback_message["content"].strip()
            assert env_manager._conversations[env_instance_id]["messages"][-1] == {
                "role": "assistant",
                "content": feedback_message["content"],
            }
            # prompt_ids/response_mask grow by turn_separator + templated feedback tokens
            # (apply_chat_template is stubbed to a fixed [10, 11] regardless of content).
            assert agent_data.prompt_ids == [1, 2, 3, 9, 10, 11]
            assert agent_data.response_mask == [1, 1, 1, 0, 0, 0]
            assert agent_data.user_turns == 1
            loop.apply_chat_template.assert_awaited_once_with([feedback_message], remove_system_prompt=True)
        finally:
            await env_manager.client.close()
            LLMFeedbackEnvironmentManager._instances.pop((LLMFeedbackEnvironmentManager, env_tag), None)

    async def test_feedback_exceeding_response_length_terminates_without_extending_state(self):
        env_manager, env_tag = _make_real_env_manager()
        try:
            env_instance_id = await env_manager.create()
            agent_data = _make_agent_data(
                _env_instance_id=env_instance_id,
                _ground_truth="4",
                response_mask=[1] * 10,
                prompt_ids=list(range(10)),
            )
            loop = _make_bare_loop(response_length=12, turn_separator=[])
            loop.loop = MagicMock(run_in_executor=AsyncMock(side_effect=_run_in_executor))
            loop.tokenizer = MagicMock(decode=MagicMock(return_value="5"))
            loop.env_manager = env_manager
            # len(response_mask)=10 + len(response_ids)=3 >= response_length=12 -> hard stop.
            loop.apply_chat_template = AsyncMock(return_value=[1, 2, 3])

            state = await ContinualAgentLoop._handle_observing_state(loop, agent_data)

            assert state == AgentState.TERMINATED
            # The feedback message is appended to conversation history unconditionally
            # (before the length check), even though the trajectory terminates here.
            feedback_message = agent_data.messages[-1]
            assert feedback_message["role"] == "user"
            assert isinstance(feedback_message["content"], str) and feedback_message["content"].strip()
            # But token-level state is left untouched since we bail before extending it.
            assert agent_data.response_mask == [1] * 10
            assert agent_data.prompt_ids == list(range(10))
            assert agent_data.user_turns == 0
        finally:
            await env_manager.client.close()
            LLMFeedbackEnvironmentManager._instances.pop((LLMFeedbackEnvironmentManager, env_tag), None)


class TestHandleGeneratingState(unittest.IsolatedAsyncioTestCase):
    @patch.object(ToolAgentLoop, "_handle_generating_state", new_callable=AsyncMock)
    async def test_non_terminal_result_passes_through_unchanged(self, mock_super_generate):
        mock_super_generate.return_value = ToolAgentState.PROCESSING_TOOLS
        loop = _make_bare_loop(response_length=1000, max_assistant_turns=0, max_user_turns=0)
        agent_data = _make_agent_data()

        state = await ContinualAgentLoop._handle_generating_state(loop, agent_data, {})

        assert state == ToolAgentState.PROCESSING_TOOLS

    @patch.object(ToolAgentLoop, "_handle_generating_state", new_callable=AsyncMock)
    async def test_no_tool_call_reclassified_as_observing_when_under_limits(self, mock_super_generate):
        mock_super_generate.return_value = ToolAgentState.TERMINATED
        loop = _make_bare_loop(response_length=1000, max_assistant_turns=0, max_user_turns=0)
        agent_data = _make_agent_data(response_mask=[1] * 5, assistant_turns=1, user_turns=0)

        state = await ContinualAgentLoop._handle_generating_state(loop, agent_data, {})

        assert state == AgentState.OBSERVING

    @patch.object(ToolAgentLoop, "_handle_generating_state", new_callable=AsyncMock)
    async def test_response_length_hard_limit_stays_terminated(self, mock_super_generate):
        mock_super_generate.return_value = ToolAgentState.TERMINATED
        loop = _make_bare_loop(response_length=5, max_assistant_turns=0, max_user_turns=0)
        agent_data = _make_agent_data(response_mask=[1] * 5)

        state = await ContinualAgentLoop._handle_generating_state(loop, agent_data, {})

        assert state == AgentState.TERMINATED

    @patch.object(ToolAgentLoop, "_handle_generating_state", new_callable=AsyncMock)
    async def test_max_assistant_turns_hard_limit_stays_terminated(self, mock_super_generate):
        mock_super_generate.return_value = ToolAgentState.TERMINATED
        loop = _make_bare_loop(response_length=1000, max_assistant_turns=2, max_user_turns=0)
        agent_data = _make_agent_data(response_mask=[1], assistant_turns=2)

        state = await ContinualAgentLoop._handle_generating_state(loop, agent_data, {})

        assert state == AgentState.TERMINATED

    @patch.object(ToolAgentLoop, "_handle_generating_state", new_callable=AsyncMock)
    async def test_max_user_turns_hard_limit_stays_terminated(self, mock_super_generate):
        mock_super_generate.return_value = ToolAgentState.TERMINATED
        loop = _make_bare_loop(response_length=1000, max_assistant_turns=0, max_user_turns=3)
        agent_data = _make_agent_data(response_mask=[1], user_turns=3)

        state = await ContinualAgentLoop._handle_generating_state(loop, agent_data, {})

        assert state == AgentState.TERMINATED

    @patch.object(ToolAgentLoop, "_handle_generating_state", new_callable=AsyncMock)
    async def test_ignore_termination_skips_response_length_check(self, mock_super_generate):
        mock_super_generate.return_value = ToolAgentState.TERMINATED
        loop = _make_bare_loop(response_length=5, max_assistant_turns=0, max_user_turns=0)
        agent_data = _make_agent_data(response_mask=[1] * 10)

        state = await ContinualAgentLoop._handle_generating_state(loop, agent_data, {}, ignore_termination=True)

        assert state == AgentState.OBSERVING


class TestInit(unittest.TestCase):
    """``ContinualAgentLoop.__init__`` on a not-yet-initialized instance must
    delegate to ``ToolAgentLoop.__init__`` and build a real ``env_manager``
    from the ``env_config`` sub-dict of ``self.config`` (set by the parent
    init). ``ToolAgentLoop.__init__`` itself is still stubbed -- driving the
    real one needs a live tokenizer/rollout config/dataset, which is out of
    scope here; only the ``env_manager`` it hands off to is real."""

    def test_init_builds_env_manager_from_config(self):
        key = (ContinualAgentLoop, "init-test-tag")
        ContinualAgentLoop._instances.pop(key, None)
        env_tag = "init-test-tag-env"
        try:
            instance = ContinualAgentLoop.__new__(ContinualAgentLoop, name="init-test-tag")
            assert instance._initialized is False

            def fake_super_init(self, *args, tools=None, **kwargs):
                self.config = {"env_config": {"name": env_tag, "max_tokens": 64}}

            with patch.object(ToolAgentLoop, "__init__", new=fake_super_init):
                ContinualAgentLoop.__init__(instance, name="init-test-tag", tools=None)

            assert instance._initialized is True
            assert isinstance(instance.env_manager, LLMFeedbackEnvironmentManager)
            assert instance.env_manager.config == {"name": env_tag, "max_tokens": 64}
            assert instance.env_manager.max_tokens == 64
        finally:
            _close_real_client_sync(getattr(instance, "env_manager", None))
            ContinualAgentLoop._instances.pop(key, None)
            LLMFeedbackEnvironmentManager._instances.pop((LLMFeedbackEnvironmentManager, env_tag), None)

    def test_init_defaults_env_config_to_empty_dict_when_absent(self):
        key = (ContinualAgentLoop, "init-test-tag-2")
        ContinualAgentLoop._instances.pop(key, None)
        default_tag = LLMFeedbackEnvironmentManager.__name__
        try:
            instance = ContinualAgentLoop.__new__(ContinualAgentLoop, name="init-test-tag-2")

            def fake_super_init(self, *args, tools=None, **kwargs):
                self.config = {}

            with patch.object(ToolAgentLoop, "__init__", new=fake_super_init):
                ContinualAgentLoop.__init__(instance, name="init-test-tag-2")

            assert isinstance(instance.env_manager, LLMFeedbackEnvironmentManager)
            assert instance.env_manager.config == {}
        finally:
            _close_real_client_sync(getattr(instance, "env_manager", None))
            ContinualAgentLoop._instances.pop(key, None)
            LLMFeedbackEnvironmentManager._instances.pop((LLMFeedbackEnvironmentManager, default_tag), None)


class TestRun(unittest.IsolatedAsyncioTestCase):
    """``run`` wires ``process_multi_modal_info``, per-sample tool selection, the
    environment manager's create/release lifecycle, the state machine, and the
    final ``AgentLoopOutput`` construction (including the ``turn_scores`` /
    ``tool_rewards`` extra fields ``ToolAgentLoop.run`` doesn't have)."""

    async def test_run_drives_state_machine_and_builds_output(self):
        env_manager, env_tag = _make_real_env_manager()
        try:
            loop = _make_bare_loop(response_length=100, tools={}, tool_schemas=[])
            loop.process_multi_modal_info = AsyncMock(return_value={})
            loop._get_mm_processor_kwargs = MagicMock(return_value={})
            loop.env_manager = env_manager

            captured = {}

            async def fake_pending(agent_data, sampling_params):
                # ``run`` must call the real ``env_manager.create`` before entering
                # the state machine -- the id it hands to ``agent_data`` should
                # already be a live conversation.
                captured["env_instance_id"] = agent_data._env_instance_id
                captured["created_before_run"] = agent_data._env_instance_id in env_manager._conversations
                agent_data.prompt_ids = [1, 2, 3]
                return AgentState.GENERATING

            async def fake_generating(agent_data, sampling_params):
                agent_data.prompt_ids += [4, 5]
                agent_data.response_ids = [4, 5]
                agent_data.response_mask = [1, 1]
                agent_data.assistant_turns = 1
                return AgentState.OBSERVING

            async def fake_observing(agent_data):
                agent_data.turn_scores.append(1.0)
                return AgentState.TERMINATED

            loop._handle_pending_state = AsyncMock(side_effect=fake_pending)
            loop._handle_generating_state = AsyncMock(side_effect=fake_generating)
            loop._handle_processing_tools_state = AsyncMock()
            loop._handle_observing_state = AsyncMock(side_effect=fake_observing)

            output = await ContinualAgentLoop.run(
                loop,
                sampling_params={},
                raw_prompt=[{"role": "user", "content": "What is 2+2?"}],
                reward_model={"ground_truth": "4"},
                tools_kwargs={},
            )

            assert captured["created_before_run"] is True
            # ``run`` must call the real ``env_manager.release`` after the state
            # machine terminates -- the conversation should be gone by now.
            assert captured["env_instance_id"] not in env_manager._conversations
            loop._handle_pending_state.assert_awaited_once()
            loop._handle_generating_state.assert_awaited_once()
            loop._handle_observing_state.assert_awaited_once()
            loop._handle_processing_tools_state.assert_not_awaited()

            assert output.prompt_ids == [1, 2, 3]
            assert output.response_ids == [4, 5]
            assert output.response_mask == [1, 1]
            assert output.num_turns == 2  # assistant_turns(1) + user_turns(0) + 1
            assert output.extra_fields["turn_scores"] == [1.0]
            assert output.extra_fields["tool_rewards"] == []
        finally:
            await env_manager.client.close()
            LLMFeedbackEnvironmentManager._instances.pop((LLMFeedbackEnvironmentManager, env_tag), None)

    async def test_run_filters_tools_by_extra_info_tool_selection(self):
        env_manager, env_tag = _make_real_env_manager()
        try:
            tool_a = MagicMock()
            tool_a.tool_schema.model_dump.return_value = {"name": "tool_a"}
            tool_b = MagicMock()
            tool_b.tool_schema.model_dump.return_value = {"name": "tool_b"}

            loop = _make_bare_loop(
                response_length=100,
                tools={"tool_a": tool_a, "tool_b": tool_b},
                tool_schemas=[{"name": "tool_a"}, {"name": "tool_b"}],
            )
            loop.process_multi_modal_info = AsyncMock(return_value={})
            loop._get_mm_processor_kwargs = MagicMock(return_value={})
            loop.env_manager = env_manager

            captured = {}

            async def fake_pending(agent_data, sampling_params):
                captured["active_tools"] = agent_data._active_tools
                captured["active_tool_schemas"] = agent_data._active_tool_schemas
                agent_data.prompt_ids = []
                agent_data.response_mask = []
                return AgentState.TERMINATED

            loop._handle_pending_state = AsyncMock(side_effect=fake_pending)

            await ContinualAgentLoop.run(
                loop,
                sampling_params={},
                raw_prompt=[{"role": "user", "content": "hi"}],
                reward_model={"ground_truth": "4"},
                extra_info={"tool_selection": ["tool_b"]},
            )

            assert list(captured["active_tools"].keys()) == ["tool_b"]
            assert captured["active_tool_schemas"] == [{"name": "tool_b"}]
        finally:
            await env_manager.client.close()
            LLMFeedbackEnvironmentManager._instances.pop((LLMFeedbackEnvironmentManager, env_tag), None)

    async def test_run_defaults_to_all_tools_without_tool_selection(self):
        env_manager, env_tag = _make_real_env_manager()
        try:
            tool_a = MagicMock()
            tool_a.tool_schema.model_dump.return_value = {"name": "tool_a"}

            loop = _make_bare_loop(response_length=100, tools={"tool_a": tool_a}, tool_schemas=[{"name": "tool_a"}])
            loop.process_multi_modal_info = AsyncMock(return_value={})
            loop._get_mm_processor_kwargs = MagicMock(return_value={})
            loop.env_manager = env_manager

            captured = {}

            async def fake_pending(agent_data, sampling_params):
                captured["active_tools"] = agent_data._active_tools
                captured["ground_truth"] = agent_data._ground_truth
                captured["env_instance_id"] = agent_data._env_instance_id
                agent_data.prompt_ids = []
                agent_data.response_mask = []
                return AgentState.TERMINATED

            loop._handle_pending_state = AsyncMock(side_effect=fake_pending)

            await ContinualAgentLoop.run(
                loop,
                sampling_params={},
                raw_prompt=[{"role": "user", "content": "hi"}],
                reward_model={"ground_truth": "4"},
            )

            assert captured["active_tools"] is loop.tools
            assert captured["ground_truth"] == "4"
            assert captured["env_instance_id"] is not None
            # ``run`` releases the conversation after the state machine terminates.
            assert captured["env_instance_id"] not in env_manager._conversations
        finally:
            await env_manager.client.close()
            LLMFeedbackEnvironmentManager._instances.pop((LLMFeedbackEnvironmentManager, env_tag), None)


class TestInstanceCaching(unittest.TestCase):
    """``ContinualAgentLoop`` caches instances per ``(class, tag)`` so that
    ``hydra.utils.instantiate``, called fresh on every trajectory, doesn't
    rebuild the loop (and its environment manager / LLM clients) each time."""

    def test_same_tag_returns_cached_instance(self):
        key = (ContinualAgentLoop, "cache-test-tag")
        ContinualAgentLoop._instances.pop(key, None)
        try:
            first = ContinualAgentLoop.__new__(ContinualAgentLoop, name="cache-test-tag")
            assert first._initialized is False
            second = ContinualAgentLoop.__new__(ContinualAgentLoop, name="cache-test-tag")
            assert first is second
        finally:
            ContinualAgentLoop._instances.pop(key, None)

    def test_different_tags_get_distinct_instances(self):
        key_a = (ContinualAgentLoop, "cache-tag-a")
        key_b = (ContinualAgentLoop, "cache-tag-b")
        ContinualAgentLoop._instances.pop(key_a, None)
        ContinualAgentLoop._instances.pop(key_b, None)
        try:
            a = ContinualAgentLoop.__new__(ContinualAgentLoop, name="cache-tag-a")
            b = ContinualAgentLoop.__new__(ContinualAgentLoop, name="cache-tag-b")
            assert a is not b
        finally:
            ContinualAgentLoop._instances.pop(key_a, None)
            ContinualAgentLoop._instances.pop(key_b, None)

    def test_tag_defaults_to_class_name(self):
        key = (ContinualAgentLoop, "ContinualAgentLoop")
        already_cached = key in ContinualAgentLoop._instances
        try:
            instance = ContinualAgentLoop.__new__(ContinualAgentLoop)
            assert ContinualAgentLoop._instances[key] is instance
        finally:
            if not already_cached:
                ContinualAgentLoop._instances.pop(key, None)

    def test_init_is_a_noop_once_initialized(self):
        """``__init__`` must short-circuit on a cached instance instead of
        rebuilding ``env_manager`` (and its LLM clients) on every trajectory."""
        key = (ContinualAgentLoop, "skip-init-tag")
        ContinualAgentLoop._instances.pop(key, None)
        try:
            instance = ContinualAgentLoop.__new__(ContinualAgentLoop, name="skip-init-tag")
            instance._initialized = True
            sentinel = object()
            instance.env_manager = sentinel

            ContinualAgentLoop.__init__(instance, name="skip-init-tag")

            assert instance.env_manager is sentinel
        finally:
            ContinualAgentLoop._instances.pop(key, None)


if __name__ == "__main__":
    unittest.main()
