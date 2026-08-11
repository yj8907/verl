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
"""Environment manager abstraction for agent loops.

Unlike a ``verl.tools.base_tool.BaseTool``, an environment manager is not invoked
through a model-issued function call parsed by the ``ToolParser``: the agent loop
calls it directly after an assistant turn to score the response and decide
whether the agent should keep iterating. This is the mechanism
``ContinualAgentLoop`` uses to let the model retry after incorrect answers.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from verl.utils.reward_score.math_verify import compute_score


@dataclass
class EnvStepResult:
    """Result of one environment step."""

    feedback: str
    score: float
    done: bool
    metrics: dict[str, Any] = field(default_factory=dict)


class BaseEnvironmentManager:
    """Base class for environment managers.

    Lifecycle mirrors ``BaseTool``: ``create`` once per trajectory, ``step`` after
    each assistant turn, ``release`` when the trajectory ends. ``verify_math`` is a
    shared capability available to any subclass that needs exact/symbolic answer
    checking -- it is not an environment on its own.

    Instances are cached per ``(class, tag)`` -- ``tag`` is ``config["name"]``,
    defaulting to the class name -- so building an agent loop for every trajectory
    doesn't rebuild the environment (and its LLM clients) each time. Multiple
    distinctly-configured environments of the same class stay separate by giving
    each a distinct ``name`` in its config.
    """

    _instances: dict[tuple[type, str], "BaseEnvironmentManager"] = {}

    def __new__(cls, config: Optional[dict] = None):
        tag = (config or {}).get("name", cls.__name__)
        key = (cls, tag)
        if key not in cls._instances:
            instance = super().__new__(cls)
            instance._initialized = False
            cls._instances[key] = instance
        return cls._instances[key]

    def __init__(self, config: Optional[dict] = None):
        if self._initialized:
            return
        self.config = config or {}
        self._initialized = True

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        return instance_id or str(uuid4())

    def verify_math(self, response_text: str, ground_truth: str) -> float:
        """Score ``response_text`` against ``ground_truth`` using ``math_verify``."""
        return compute_score(response_text, ground_truth)

    async def step(self, instance_id: str, response_text: str, **kwargs) -> EnvStepResult:
        """Score ``response_text`` and return feedback plus whether the episode is done."""
        raise NotImplementedError

    async def release(self, instance_id: str) -> None:
        pass


class LLMFeedbackEnvironmentManager(BaseEnvironmentManager):
    """Environment that checks a response with ``verify_math`` and, when it's
    wrong, asks a frontier LLM (OpenAI/Anthropic) to generate a hint explaining
    the mistake -- instead of returning a fixed "incorrect, try again" string."""

    def __init__(self, config: Optional[dict] = None):
        if self._initialized:
            return
        super().__init__(config)
        self.model = self.config.get("model", "claude-3-5-sonnet-20241022")
        self.max_tokens = self.config.get("max_tokens", 256)
        self.correct_feedback = self.config.get("correct_feedback", "Your answer is correct.")
        self.default_system_prompt = self.config.get(
            "system_prompt",
            "You are a tutor helping a student solve a problem. Given the student's latest "
            "attempt and the correct answer, give a short hint about their mistake without "
            "revealing the final answer. You will see their past attempts and your past hints "
            "in this conversation -- avoid repeating a hint you already gave.",
        )
        self.provider = self.config.get("provider") or ("anthropic" if self.model.startswith("claude") else "openai")

        if self.provider == "anthropic":
            api_key = os.environ["ANTHROPIC_API_KEY"]
            self.client = AsyncAnthropic(api_key=api_key, base_url=self.config.get("base_url"))
        elif self.provider == "openai":
            api_key = os.environ["OPENAI_API_KEY"]
            self.client = AsyncOpenAI(api_key=api_key, base_url=self.config.get("base_url"))
        else:
            raise ValueError(f"Unsupported provider '{self.provider}' for model '{self.model}'")

        # Per-instance conversation state: instance_id -> {"system_prompt": str, "messages": list[dict]}
        self._conversations: dict[str, dict[str, Any]] = {}

    async def create(
        self, instance_id: Optional[str] = None, system_prompt: Optional[str] = None, **kwargs
    ) -> str:
        instance_id = await super().create(instance_id)
        self._conversations[instance_id] = {
            "system_prompt": system_prompt if system_prompt is not None else self.default_system_prompt,
            "messages": [],
        }
        return instance_id

    async def step(
        self, instance_id: str, response_text: str, ground_truth: str = "", question: str = "", **kwargs
    ) -> EnvStepResult:
        score = self.verify_math(response_text, ground_truth)
        if score == 1.0:
            return EnvStepResult(feedback=self.correct_feedback, score=score, done=True, metrics={"env_score": score})

        feedback = await self._generate_feedback(instance_id, question, response_text, ground_truth)
        return EnvStepResult(feedback=feedback, score=score, done=False, metrics={"env_score": score})

    async def _generate_feedback(self, instance_id: str, question: str, response_text: str, ground_truth: str) -> str:
        conversation = self._conversations[instance_id]

        prompt_parts = []
        if question and not conversation["messages"]:
            # Only needed on the first attempt -- subsequent turns already have it in history.
            prompt_parts.append(f"Question: {question}")
        prompt_parts.append(f"Student's latest answer:\n{response_text}")
        prompt_parts.append(f"Correct answer: {ground_truth}")
        prompt_parts.append("In one or two sentences, point out the likely mistake without revealing the final answer.")
        conversation["messages"].append({"role": "user", "content": "\n".join(prompt_parts)})

        if self.provider == "anthropic":
            message = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=conversation["system_prompt"],
                messages=conversation["messages"],
            )
            text = message.content[0].text
        else:
            openai_messages = conversation["messages"]
            if conversation["system_prompt"]:
                openai_messages = [{"role": "system", "content": conversation["system_prompt"]}] + openai_messages
            completion = await self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=openai_messages,
            )
            text = completion.choices[0].message.content

        conversation["messages"].append({"role": "assistant", "content": text})
        return text

    async def release(self, instance_id: str) -> None:
        self._conversations.pop(instance_id, None)
