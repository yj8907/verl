# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Uniform async interface for one agent's turn generation, with three backends:

- ``TrainableVerlAgentBackend``: a verl-managed rollout server whose weights PPO updates.
- ``FrozenVerlAgentBackend``: a verl-managed, inference-only rollout server.
- ``ExternalAPIAgentBackend``: an external OpenAI/Anthropic-compatible endpoint.

Both verl-managed backends return token ids (needed to build a trainable ``AgentLoopOutput``
for ``trainable=True`` agents, and to let observing agents re-tokenize what a frozen verl agent
said). The external backend returns text only: it is never trained, and other agents re-tokenize
its text into their own context using their own tokenizer.
"""

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4

from transformers import PreTrainedTokenizerBase

from verl.experimental.multiagent.config.agent_config import ExternalApiConfig
from verl.utils.tokenizer.chat_template import apply_chat_template
from verl.workers.rollout.llm_server import LLMServerClient

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass
class AgentTurnResult:
    """Result of one agent's turn."""

    text: str
    """Decoded text of the turn, always populated (used to render the turn into other agents' context)."""
    response_token_ids: Optional[list[int]] = None
    """Token ids generated for this turn, in this agent's own tokenizer. None for external_api agents."""
    response_logprobs: Optional[list[float]] = None
    """Per-token log probs aligned with ``response_token_ids``, if the backend returns them."""
    trainable: bool = False
    """Whether this turn came from a trainable agent (only trainable turns get a TQ row)."""
    metrics: dict[str, Any] = field(default_factory=dict)
    """Backend/turn metadata, e.g. {"done": True} used by termination checks."""


class AgentBackend(ABC):
    """Uniform interface an agent's model is accessed through, regardless of how it's served."""

    trainable: bool = False

    @abstractmethod
    async def respond(
        self,
        messages: list[dict],
        system_prompt: str,
        sampling_params: dict[str, Any],
    ) -> AgentTurnResult:
        """Generate this agent's next turn given the conversation so far.

        Args:
            messages: Chat-format message history visible to this agent (may be a subset/rendering
                of the full episode, depending on the caller).
            system_prompt: This agent's own system prompt.
            sampling_params: Sampling parameters for generation.
        """
        raise NotImplementedError


class _VerlAgentBackendBase(AgentBackend):
    """Shared ``LLMServerClient``-based generation logic for verl-managed agents."""

    def __init__(self, agent_id: str, client: LLMServerClient, tokenizer: PreTrainedTokenizerBase):
        self.agent_id = agent_id
        self.client = client
        self.tokenizer = tokenizer

    async def respond(
        self,
        messages: list[dict],
        system_prompt: str,
        sampling_params: dict[str, Any],
    ) -> AgentTurnResult:
        full_messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + list(messages)
        prompt_ids = apply_chat_template(
            self.tokenizer,
            full_messages,
            add_generation_prompt=True,
            tokenize=True,
        )
        output = await self.client.generate(
            request_id=uuid4().hex,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
        )
        response_token_ids = list(output.token_ids)
        text = self.tokenizer.decode(response_token_ids, skip_special_tokens=True)
        response_logprobs = list(output.log_probs) if getattr(output, "log_probs", None) is not None else None
        return AgentTurnResult(
            text=text,
            response_token_ids=response_token_ids,
            response_logprobs=response_logprobs,
            trainable=self.trainable,
            metrics={},
        )


class TrainableVerlAgentBackend(_VerlAgentBackendBase):
    """A verl-managed rollout server whose weights are updated by PPO."""

    trainable = True


class FrozenVerlAgentBackend(_VerlAgentBackendBase):
    """A verl-managed, inference-only rollout server (never trained)."""

    trainable = False


class ExternalAPIAgentBackend(AgentBackend):
    """An external OpenAI/Anthropic-compatible endpoint, never trained.

    Bounds concurrency and applies a per-call timeout + retry/backoff so a slow or
    rate-limited provider can't stall the rest of the rollout pipeline: turns for
    other samples keep running on GPU-managed backends while this one waits.
    """

    trainable = False

    def __init__(self, agent_id: str, config: ExternalApiConfig):
        self.agent_id = agent_id
        self.config = config
        self.provider = config.provider or ("anthropic" if config.model.startswith("claude") else "openai")
        self._semaphore = asyncio.Semaphore(config.max_concurrency)

        if self.provider == "anthropic":
            from anthropic import AsyncAnthropic

            self.client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"], base_url=config.base_url)
        elif self.provider == "openai":
            from openai import AsyncOpenAI

            self.client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=config.base_url)
        else:
            raise ValueError(f"Unsupported provider {self.provider!r} for model {config.model!r}.")

    async def respond(
        self,
        messages: list[dict],
        system_prompt: str,
        sampling_params: dict[str, Any],
    ) -> AgentTurnResult:
        temperature = sampling_params.get("temperature")
        async with self._semaphore:
            text = await self._call_with_retry(messages, system_prompt, temperature)
        return AgentTurnResult(text=text, trainable=False, metrics={})

    async def _call_with_retry(
        self, messages: list[dict], system_prompt: str, temperature: Optional[float]
    ) -> str:
        last_error: Optional[BaseException] = None
        for attempt in range(self.config.max_retries + 1):
            try:
                return await asyncio.wait_for(
                    self._call_once(messages, system_prompt, temperature),
                    timeout=self.config.timeout_s,
                )
            except Exception as e:  # noqa: BLE001 - broad: network/timeout/provider errors all get retried
                last_error = e
                if attempt < self.config.max_retries:
                    backoff_s = 2**attempt
                    logger.warning(
                        f"External agent {self.agent_id!r} call failed (attempt {attempt + 1}/"
                        f"{self.config.max_retries + 1}), retrying in {backoff_s}s: {e}"
                    )
                    await asyncio.sleep(backoff_s)
        raise RuntimeError(
            f"External agent {self.agent_id!r} failed after {self.config.max_retries + 1} attempts"
        ) from last_error

    async def _call_once(self, messages: list[dict], system_prompt: str, temperature: Optional[float]) -> str:
        if self.provider == "anthropic":
            kwargs = {"temperature": temperature} if temperature is not None else {}
            response = await self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                system=system_prompt,
                messages=messages,
                **kwargs,
            )
            return response.content[0].text
        else:
            kwargs = {"temperature": temperature} if temperature is not None else {}
            openai_messages = messages
            if system_prompt:
                openai_messages = [{"role": "system", "content": system_prompt}] + openai_messages
            completion = await self.client.chat.completions.create(
                model=self.config.model,
                max_completion_tokens=self.config.max_tokens,
                messages=openai_messages,
                **kwargs,
            )
            return completion.choices[0].message.content
