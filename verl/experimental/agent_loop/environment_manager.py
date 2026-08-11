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

from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4

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
    each assistant turn, ``release`` when the trajectory ends.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        return instance_id or str(uuid4())

    async def step(self, instance_id: str, response_text: str, **kwargs) -> EnvStepResult:
        """Score ``response_text`` and return feedback plus whether the episode is done."""
        raise NotImplementedError

    async def release(self, instance_id: str) -> None:
        pass


class MathVerifyEnvironmentManager(BaseEnvironmentManager):
    """Scores a response against a ground-truth answer with ``math_verify`` and
    returns retry feedback for the agent to act on next turn."""

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self.correct_feedback = self.config.get("correct_feedback", "Your answer is correct.")
        self.incorrect_feedback = self.config.get(
            "incorrect_feedback", "Your answer is incorrect. Please reconsider and try again."
        )

    async def step(self, instance_id: str, response_text: str, ground_truth: str = "", **kwargs) -> EnvStepResult:
        score = compute_score(response_text, ground_truth)
        done = score == 1.0
        feedback = self.correct_feedback if done else self.incorrect_feedback
        return EnvStepResult(feedback=feedback, score=score, done=done, metrics={"env_score": score})
