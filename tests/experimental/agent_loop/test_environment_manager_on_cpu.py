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

"""Unit tests for ``BaseEnvironmentManager`` / ``LLMFeedbackEnvironmentManager``.

No ``unittest.mock`` anywhere: ``verify_math`` runs the real ``math_verify``-backed
``compute_score`` (a local library call, no network), and every
``LLMFeedbackEnvironmentManager`` test that needs a client builds a real
``AsyncAnthropic``/``AsyncOpenAI`` instance and calls the real API. That means:

- ``ANTHROPIC_API_KEY`` must be set for the (default-provider) init tests, the
  ``step``/``_generate_feedback`` tests, and the Anthropic-path tests.
- ``OPENAI_API_KEY`` must additionally be set for the OpenAI-path tests and the
  explicit-provider-override test.
- Tests that call ``_generate_feedback``/``step`` on an incorrect answer make a
  real, billed LLM request. ``max_tokens`` is kept small to limit cost.
- These tests hit the network and are non-deterministic (real LLM output), and
  will raise ``KeyError`` immediately if the relevant API key isn't set --
  there is no skip-if-missing fallback.
"""

import os
import unittest
from uuid import uuid4

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from verl.experimental.agent_loop.environment_manager import (
    BaseEnvironmentManager,
    EnvStepResult,
    LLMFeedbackEnvironmentManager,
)


def _make_bare_base(**attrs):
    """A ``BaseEnvironmentManager`` created via ``__new__`` only, so ``__init__``
    never runs. The cache entry is popped immediately so it doesn't leak into
    the process-wide ``_instances`` dict shared by other tests."""
    tag = f"bare-{uuid4().hex}"
    instance = BaseEnvironmentManager.__new__(BaseEnvironmentManager, config={"name": tag})
    BaseEnvironmentManager._instances.pop((BaseEnvironmentManager, tag), None)
    for key, value in attrs.items():
        setattr(instance, key, value)
    return instance


def _make_real_llm(config_overrides=None, tag_prefix="real"):
    """Builds a fully ``__init__``-ed ``LLMFeedbackEnvironmentManager`` backed by
    a real Anthropic/OpenAI client -- requires the matching API key to be set in
    the environment (``ANTHROPIC_API_KEY`` by default, or ``OPENAI_API_KEY`` if
    ``config_overrides`` selects an OpenAI model/provider).

    Returns ``(instance, tag)``; callers must pop ``LLMFeedbackEnvironmentManager
    ._instances[(LLMFeedbackEnvironmentManager, tag)]`` in a ``finally`` block so
    the throwaway instance doesn't leak into other tests.
    """
    tag = f"{tag_prefix}-{uuid4().hex}"
    config = {"name": tag, "max_tokens": 64}
    if config_overrides:
        config.update(config_overrides)
    instance = LLMFeedbackEnvironmentManager.__new__(LLMFeedbackEnvironmentManager, config=config)
    LLMFeedbackEnvironmentManager._instances.pop((LLMFeedbackEnvironmentManager, tag), None)
    LLMFeedbackEnvironmentManager.__init__(instance, config=config)
    return instance, tag


class TestBaseEnvironmentManagerCaching(unittest.TestCase):
    """Instances are cached per ``(class, tag)`` so building an agent loop for
    every trajectory doesn't rebuild the environment (and its LLM clients)."""

    def test_same_tag_returns_cached_instance(self):
        key = (BaseEnvironmentManager, "cache-tag")
        BaseEnvironmentManager._instances.pop(key, None)
        try:
            first = BaseEnvironmentManager.__new__(BaseEnvironmentManager, config={"name": "cache-tag"})
            assert first._initialized is False
            second = BaseEnvironmentManager.__new__(BaseEnvironmentManager, config={"name": "cache-tag"})
            assert first is second
        finally:
            BaseEnvironmentManager._instances.pop(key, None)

    def test_different_tags_get_distinct_instances(self):
        key_a = (BaseEnvironmentManager, "tag-a")
        key_b = (BaseEnvironmentManager, "tag-b")
        BaseEnvironmentManager._instances.pop(key_a, None)
        BaseEnvironmentManager._instances.pop(key_b, None)
        try:
            a = BaseEnvironmentManager.__new__(BaseEnvironmentManager, config={"name": "tag-a"})
            b = BaseEnvironmentManager.__new__(BaseEnvironmentManager, config={"name": "tag-b"})
            assert a is not b
        finally:
            BaseEnvironmentManager._instances.pop(key_a, None)
            BaseEnvironmentManager._instances.pop(key_b, None)

    def test_tag_defaults_to_class_name_when_config_has_no_name(self):
        key = (BaseEnvironmentManager, "BaseEnvironmentManager")
        already_cached = key in BaseEnvironmentManager._instances
        try:
            instance = BaseEnvironmentManager.__new__(BaseEnvironmentManager)
            assert BaseEnvironmentManager._instances[key] is instance
        finally:
            if not already_cached:
                BaseEnvironmentManager._instances.pop(key, None)

    def test_init_stores_config_and_sets_initialized(self):
        key = (BaseEnvironmentManager, "init-tag")
        BaseEnvironmentManager._instances.pop(key, None)
        try:
            instance = BaseEnvironmentManager.__new__(BaseEnvironmentManager, config={"name": "init-tag"})
            BaseEnvironmentManager.__init__(instance, config={"name": "init-tag", "foo": "bar"})
            assert instance.config == {"name": "init-tag", "foo": "bar"}
            assert instance._initialized is True
        finally:
            BaseEnvironmentManager._instances.pop(key, None)

    def test_init_defaults_config_to_empty_dict_when_none(self):
        key = (BaseEnvironmentManager, "BaseEnvironmentManager")
        already_cached = key in BaseEnvironmentManager._instances
        try:
            instance = BaseEnvironmentManager.__new__(BaseEnvironmentManager)
            BaseEnvironmentManager.__init__(instance)
            assert instance.config == {}
        finally:
            if not already_cached:
                BaseEnvironmentManager._instances.pop(key, None)

    def test_init_is_noop_once_initialized(self):
        key = (BaseEnvironmentManager, "skip-init-tag")
        BaseEnvironmentManager._instances.pop(key, None)
        try:
            instance = BaseEnvironmentManager.__new__(BaseEnvironmentManager, config={"name": "skip-init-tag"})
            instance._initialized = True
            sentinel = {"already": "set"}
            instance.config = sentinel

            BaseEnvironmentManager.__init__(instance, config={"different": "config"})

            assert instance.config is sentinel
        finally:
            BaseEnvironmentManager._instances.pop(key, None)


class TestBaseEnvironmentManagerLifecycle(unittest.IsolatedAsyncioTestCase):
    async def test_create_returns_given_instance_id(self):
        instance = _make_bare_base()
        result = await BaseEnvironmentManager.create(instance, instance_id="explicit-id")
        assert result == "explicit-id"

    async def test_create_generates_a_fresh_id_when_missing(self):
        instance = _make_bare_base()
        first = await BaseEnvironmentManager.create(instance)
        second = await BaseEnvironmentManager.create(instance)
        assert isinstance(first, str) and first
        assert first != second

    def test_verify_math_scores_matching_answer_as_correct(self):
        # Real math_verify call -- no network, but exercises the actual grading logic.
        instance = _make_bare_base()
        assert instance.verify_math("42", "42") == 1.0

    def test_verify_math_scores_mismatched_answer_as_incorrect(self):
        instance = _make_bare_base()
        assert instance.verify_math("41", "42") == 0.0

    async def test_step_raises_not_implemented(self):
        instance = _make_bare_base()
        with self.assertRaises(NotImplementedError):
            await BaseEnvironmentManager.step(instance, "env-1", "response")

    async def test_release_is_a_noop(self):
        instance = _make_bare_base()
        result = await BaseEnvironmentManager.release(instance, "env-1")
        assert result is None


class TestLLMFeedbackInit(unittest.TestCase):
    """``LLMFeedbackEnvironmentManager.__init__`` picks defaults/overrides from
    config, derives the provider from the model name (or an explicit override),
    and builds a real client from the corresponding API-key env var."""

    def _bare(self, tag):
        instance = LLMFeedbackEnvironmentManager.__new__(LLMFeedbackEnvironmentManager, config={"name": tag})
        LLMFeedbackEnvironmentManager._instances.pop((LLMFeedbackEnvironmentManager, tag), None)
        return instance

    # def test_defaults_use_claude_model_and_anthropic_provider(self):
    #     tag = f"init-{uuid4().hex}"
    #     instance = self._bare(tag)
    #     try:
    #         LLMFeedbackEnvironmentManager.__init__(instance, config={"name": tag})

    #         assert instance.model == "claude-3-5-sonnet-20241022"
    #         assert instance.provider == "anthropic"
    #         assert instance.max_tokens == 256
    #         assert instance.correct_feedback == "Your answer is correct."
    #         assert instance._conversations == {}
    #         assert isinstance(instance.client, AsyncAnthropic)
    #     finally:
    #         LLMFeedbackEnvironmentManager._instances.pop((LLMFeedbackEnvironmentManager, tag), None)

    def test_non_claude_model_defaults_to_openai_provider(self):
        tag = f"init-{uuid4().hex}"
        instance = self._bare(tag)
        try:
            LLMFeedbackEnvironmentManager.__init__(instance, config={"name": tag, "model": "gpt-4o-mini"})

            assert instance.provider == "openai"
            assert isinstance(instance.client, AsyncOpenAI)
        finally:
            LLMFeedbackEnvironmentManager._instances.pop((LLMFeedbackEnvironmentManager, tag), None)

    def test_explicit_provider_overrides_model_based_default(self):
        tag = f"init-{uuid4().hex}"
        instance = self._bare(tag)
        try:
            LLMFeedbackEnvironmentManager.__init__(
                instance,
                config={"name": tag, "model": "gpt-5.6-luna", "provider": "openai"},
            )

            assert instance.provider == "openai"
            assert isinstance(instance.client, AsyncOpenAI)
        finally:
            LLMFeedbackEnvironmentManager._instances.pop((LLMFeedbackEnvironmentManager, tag), None)

    def test_base_url_is_forwarded_to_client(self):
        tag = f"init-{uuid4().hex}"
        instance = self._bare(tag)
        try:
            LLMFeedbackEnvironmentManager.__init__(
                instance, config={"name": tag, "base_url": "https://proxy.internal/v1"}
            )

            assert str(instance.client.base_url).rstrip("/") == "https://proxy.internal/v1"
        finally:
            LLMFeedbackEnvironmentManager._instances.pop((LLMFeedbackEnvironmentManager, tag), None)

    @unittest.skip("")
    def test_missing_anthropic_api_key_raises(self):
        tag = f"init-{uuid4().hex}"
        instance = self._bare(tag)
        original = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            with self.assertRaises(KeyError):
                LLMFeedbackEnvironmentManager.__init__(instance, config={"name": tag})
        finally:
            if original is not None:
                os.environ["ANTHROPIC_API_KEY"] = original
            LLMFeedbackEnvironmentManager._instances.pop((LLMFeedbackEnvironmentManager, tag), None)

    def test_unsupported_provider_raises_value_error(self):
        tag = f"init-{uuid4().hex}"
        instance = self._bare(tag)
        try:
            with self.assertRaises(ValueError):
                LLMFeedbackEnvironmentManager.__init__(instance, config={"name": tag, "provider": "mistral"})
        finally:
            LLMFeedbackEnvironmentManager._instances.pop((LLMFeedbackEnvironmentManager, tag), None)

    def test_init_is_noop_once_initialized(self):
        tag = f"init-{uuid4().hex}"
        instance = self._bare(tag)
        try:
            instance._initialized = True
            sentinel = object()
            instance.client = sentinel

            LLMFeedbackEnvironmentManager.__init__(instance, config={"name": tag, "model": "should-be-ignored"})

            assert instance.client is sentinel
        finally:
            LLMFeedbackEnvironmentManager._instances.pop((LLMFeedbackEnvironmentManager, tag), None)


class TestLLMFeedbackCreate(unittest.IsolatedAsyncioTestCase):
    """No LLM client is touched during ``create`` -- these can stay on a bare
    (un-``__init__``-ed) instance."""

    def _bare(self, **attrs):
        tag = f"bare-{uuid4().hex}"
        instance = LLMFeedbackEnvironmentManager.__new__(LLMFeedbackEnvironmentManager, config={"name": tag})
        LLMFeedbackEnvironmentManager._instances.pop((LLMFeedbackEnvironmentManager, tag), None)
        for key, value in attrs.items():
            setattr(instance, key, value)
        return instance

    async def test_create_initializes_conversation_with_default_system_prompt(self):
        instance = self._bare(default_system_prompt="Be a good tutor.", _conversations={})
        result = await LLMFeedbackEnvironmentManager.create(instance)
        assert result in instance._conversations
        assert instance._conversations[result] == {"system_prompt": "Be a good tutor.", "messages": []}

    async def test_create_overrides_system_prompt(self):
        instance = self._bare(default_system_prompt="Default prompt", _conversations={})
        result = await LLMFeedbackEnvironmentManager.create(instance, system_prompt="Custom prompt")
        assert instance._conversations[result]["system_prompt"] == "Custom prompt"

    async def test_create_uses_given_instance_id(self):
        instance = self._bare(default_system_prompt="Default", _conversations={})
        result = await LLMFeedbackEnvironmentManager.create(instance, instance_id="fixed-id")
        assert result == "fixed-id"
        assert "fixed-id" in instance._conversations


class TestLLMFeedbackStep(unittest.IsolatedAsyncioTestCase):
    """End-to-end against a real, fully ``__init__``-ed instance: ``verify_math``
    runs real ``math_verify``, and an incorrect answer triggers a real LLM call
    for feedback. Requires ``ANTHROPIC_API_KEY``."""

    async def test_correct_answer_terminates_without_calling_the_llm(self):
        instance, tag = _make_real_llm()
        try:
            instance_id = await instance.create()

            result = await instance.step(instance_id, "42", ground_truth="42")

            assert result == EnvStepResult(
                feedback=instance.correct_feedback, score=1.0, done=True, metrics={"env_score": 1.0}
            )
            # Correct answers short-circuit before any LLM call, so no turn is recorded.
            assert instance._conversations[instance_id]["messages"] == []
        finally:
            LLMFeedbackEnvironmentManager._instances.pop((LLMFeedbackEnvironmentManager, tag), None)

    async def test_incorrect_answer_calls_the_real_llm_for_feedback(self):
        instance, tag = _make_real_llm()
        try:
            instance_id = await instance.create()

            result = await instance.step(instance_id, "41", ground_truth="42", question="What is 19 + 23?")

            assert result.done is False
            assert result.score == 0.0
            assert result.metrics == {"env_score": 0.0}
            assert isinstance(result.feedback, str) and result.feedback.strip()
            messages = instance._conversations[instance_id]["messages"]
            assert len(messages) == 2
            assert messages[0]["role"] == "user"
            assert messages[1] == {"role": "assistant", "content": result.feedback}
        finally:
            LLMFeedbackEnvironmentManager._instances.pop((LLMFeedbackEnvironmentManager, tag), None)


class TestLLMFeedbackGenerateFeedback(unittest.IsolatedAsyncioTestCase):
    """Real API calls for both providers. The user-message text is our own code's
    output (built before the request is sent) so it's asserted exactly; the
    assistant reply is real LLM output, so only checked for being non-empty."""

    @unittest.skip("")
    async def test_anthropic_path_calls_the_real_api_and_records_history(self):
        instance, tag = _make_real_llm({"provider": "anthropic"})
        try:
            instance_id = await instance.create()

            text = await instance._generate_feedback(instance_id, "What is 19+23?", "41", "42")

            assert isinstance(text, str) and text.strip()
            conversation = instance._conversations[instance_id]["messages"]
            expected_content = "\n".join(
                [
                    "Question: What is 19+23?",
                    "Student's latest answer:\n41",
                    "Correct answer: 42",
                    "In one or two sentences, point out the likely mistake without revealing the final answer.",
                ]
            )
            assert conversation[0] == {"role": "user", "content": expected_content}
            assert conversation[1] == {"role": "assistant", "content": text}
        finally:
            LLMFeedbackEnvironmentManager._instances.pop((LLMFeedbackEnvironmentManager, tag), None)

    async def test_question_is_omitted_on_subsequent_attempts(self):
        instance, tag = _make_real_llm({"provider": "openai"})
        try:
            instance_id = await instance.create()
            await instance._generate_feedback(instance_id, "What is 19+23?", "41", "42")

            await instance._generate_feedback(instance_id, "What is 19+23?", "40", "42")

            conversation = instance._conversations[instance_id]["messages"]
            # [user#1, assistant#1, user#2, assistant#2] -- the second attempt's
            # user turn (index 2) must not repeat the question.
            assert len(conversation) == 4
            assert conversation[2]["role"] == "user"
            assert not conversation[2]["content"].startswith("Question:")
        finally:
            LLMFeedbackEnvironmentManager._instances.pop((LLMFeedbackEnvironmentManager, tag), None)

    async def test_openai_path_calls_the_real_api_and_records_history(self):
        instance, tag = _make_real_llm({"provider": "openai", "model": "gpt-4o-mini"})
        try:
            instance_id = await instance.create()

            text = await instance._generate_feedback(instance_id, "What is 19+23?", "41", "42")

            assert isinstance(text, str) and text.strip()
            conversation = instance._conversations[instance_id]["messages"]
            assert conversation[0]["role"] == "user"
            assert conversation[1] == {"role": "assistant", "content": text}
        finally:
            LLMFeedbackEnvironmentManager._instances.pop((LLMFeedbackEnvironmentManager, tag), None)

    async def test_openai_path_works_without_a_system_prompt(self):
        instance, tag = _make_real_llm({"provider": "openai", "model": "gpt-4o-mini"})
        try:
            instance_id = await instance.create(system_prompt="")

            text = await instance._generate_feedback(instance_id, "What is 19+23?", "41", "42")

            assert isinstance(text, str) and text.strip()
        finally:
            LLMFeedbackEnvironmentManager._instances.pop((LLMFeedbackEnvironmentManager, tag), None)


class TestLLMFeedbackRelease(unittest.IsolatedAsyncioTestCase):
    def _bare(self, **attrs):
        tag = f"bare-{uuid4().hex}"
        instance = LLMFeedbackEnvironmentManager.__new__(LLMFeedbackEnvironmentManager, config={"name": tag})
        LLMFeedbackEnvironmentManager._instances.pop((LLMFeedbackEnvironmentManager, tag), None)
        for key, value in attrs.items():
            setattr(instance, key, value)
        return instance

    async def test_release_pops_conversation(self):
        instance = self._bare(_conversations={"env-1": {"system_prompt": "", "messages": []}})
        await LLMFeedbackEnvironmentManager.release(instance, "env-1")
        assert "env-1" not in instance._conversations

    async def test_release_missing_instance_is_a_noop(self):
        instance = self._bare(_conversations={})
        await LLMFeedbackEnvironmentManager.release(instance, "missing")
        assert instance._conversations == {}


if __name__ == "__main__":
    unittest.main()
