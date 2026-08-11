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
import logging
import os
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopOutput,
    ToolListWrap,
    register,
)
from verl.experimental.agent_loop.environment_manager import LLMFeedbackEnvironmentManager
from verl.experimental.agent_loop.tool_agent_loop import AgentData, ToolAgentLoop
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    OBSERVING = "observing"
    TERMINATED = "terminated"


@register("continual_agent")
class ContinualAgentLoop(ToolAgentLoop):
    """Tool-calling agent loop that also consults an ``EnvironmentManager`` after
    every non-tool-call response: the environment scores the answer and returns
    feedback, letting the agent retry (subject to ``multi_turn.max_user_turns``
    and ``response_length``) instead of terminating on the first attempt.

    Instances are cached per ``(class, tag)`` -- ``tag`` is the ``name`` kwarg (the
    ``name:`` field of an ``agent_loop_config_path`` entry flows straight through),
    defaulting to the class name -- so ``hydra.utils.instantiate``, which is called
    fresh on every trajectory by ``AgentLoopWorker._run_agent_loop`` and has no
    caching of its own, doesn't rebuild the loop (tools, tool parser, environment
    manager, LLM clients) every single call.
    """

    _instances: dict[tuple[type, str], "ContinualAgentLoop"] = {}

    def __new__(cls, *args, name: Optional[str] = None, **kwargs):
        tag = name or cls.__name__
        key = (cls, tag)
        if key not in cls._instances:
            instance = super().__new__(cls)
            instance._initialized = False
            cls._instances[key] = instance
        return cls._instances[key]

    def __init__(
        self,
        *args,
        name: Optional[str] = None,
        tools: Optional[ToolListWrap] = None,
        env_config: Optional[dict] = None,
        **kwargs,
    ):
        """Args:
        tools: Tools to use for the tool agent loop.
        env_config: Config dict forwarded to ``LLMFeedbackEnvironmentManager``
            (model, max_tokens, provider, system_prompt, ...). Settable per agent loop
            via ``rollout.agent.agent_loop_config_path``, e.g.::

                - name: continual_agent
                  _target_: verl.experimental.agent_loop.continual_agent_loop.ContinualAgentLoop
                  env_config:
                    model: gpt-4o
                    max_tokens: 300
        name: Cache tag for this loop; also the registry entry's ``name:`` when
            loaded via ``agent_loop_config_path``. Not otherwise used.
        """
        if self._initialized:
            return
        super().__init__(*args, tools=tools, **kwargs)
        self.env_manager = LLMFeedbackEnvironmentManager(config=self.config.get("env_config", {}))
        self._initialized = True

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        # extract multimodal inputs from messages
        multi_modal_data = await self.process_multi_modal_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        audios = multi_modal_data.get("audios")
        mm_processor_kwargs = self._get_mm_processor_kwargs(audios)

        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})

        agent_data = AgentData(
            messages=messages,
            image_data=images,
            video_data=videos,
            audio_data=audios,
            mm_processor_kwargs=mm_processor_kwargs,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
        )
        agent_data._ground_truth = kwargs.get("reward_model", {}).get("ground_truth", "")
        agent_data._env_instance_id = await self.env_manager.create()

        # Per-sample tool selection: filter global tools by extra_info.tool_selection
        extra_info = kwargs.get("extra_info", {}) or {}
        tool_selection = extra_info.get("tool_selection")
        if tool_selection and self.tools:
            selected = {name: self.tools[name] for name in tool_selection if name in self.tools}
            agent_data._active_tools = selected
            agent_data._active_tool_schemas = [
                t.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for t in selected.values()
            ]
        else:
            agent_data._active_tools = self.tools
            agent_data._active_tool_schemas = self.tool_schemas

        # State machine loop. Handlers inherited unchanged from ToolAgentLoop return
        # its own AgentState enum, so states are compared by `.value` rather than
        # identity -- the two enums can't share members (Enum forbids extending an
        # enum that already has members), but their value strings line up.
        state = AgentState.PENDING
        while state.value != AgentState.TERMINATED.value:
            if state.value == AgentState.PENDING.value:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state.value == AgentState.GENERATING.value:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state.value == AgentState.PROCESSING_TOOLS.value:
                state = await self._handle_processing_tools_state(agent_data)
            elif state.value == AgentState.OBSERVING.value:
                state = await self._handle_observing_state(agent_data)
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED

        await self.env_manager.release(agent_data._env_instance_id)

        # Finalize output
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        multi_modal_data = {}
        if agent_data.image_data is not None:
            multi_modal_data["images"] = agent_data.image_data
        if agent_data.video_data is not None:
            multi_modal_data["videos"] = agent_data.video_data
        if agent_data.audio_data is not None:
            multi_modal_data["audios"] = agent_data.audio_data

        output: AgentLoopOutput = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            mm_processor_kwargs=agent_data.mm_processor_kwargs,
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs
            else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            routed_experts=(
                agent_data.routed_experts[: len(prompt_ids) + self.response_length]
                if agent_data.routed_experts is not None
                else None
            ),
            extra_fields=agent_data.extra_fields,
        )
        output.extra_fields.update({"turn_scores": agent_data.turn_scores, "tool_rewards": agent_data.tool_rewards})
        return output

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        """Delegates to ``ToolAgentLoop._handle_generating_state`` for the actual
        generation and bookkeeping (continuous-token merging, spec-decode metrics,
        etc.), then reclassifies its "no tool call" termination as ``OBSERVING``
        (environment scoring) instead of ending the trajectory outright. The hard
        turn/length-limit terminations are recomputed below from the exact fields
        and thresholds the parent already checked against, so they stay in
        lockstep with it and are never overridden."""
        result = await super()._handle_generating_state(agent_data, sampling_params, ignore_termination)
        if result.value != AgentState.TERMINATED.value:
            return result

        hit_hard_limit = (
            (not ignore_termination and len(agent_data.response_mask) >= self.response_length)
            or (self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns)
            or (self.max_user_turns and agent_data.user_turns >= self.max_user_turns)
        )
        return AgentState.TERMINATED if hit_hard_limit else AgentState.OBSERVING

    async def _handle_observing_state(self, agent_data: AgentData) -> AgentState:
        """Score the latest response with the environment manager. Terminates on a
        correct answer; otherwise feeds the environment's feedback back to the
        model as the next user turn, subject to the same ``max_user_turns``/
        response-length limits that gate tool responses."""
        text = await self.loop.run_in_executor(None, self.tokenizer.decode, agent_data.response_ids)
        result = await self.env_manager.step(agent_data._env_instance_id, text, ground_truth=agent_data._ground_truth)
        agent_data.turn_scores.append(result.score)

        if result.done:
            return AgentState.TERMINATED

        feedback_message = [{"role": "user", "content": result.feedback}]
        agent_data.messages.extend(feedback_message)
        response_ids = await self.apply_chat_template(feedback_message, remove_system_prompt=True)
        response_ids = self.turn_separator + response_ids

        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            return AgentState.TERMINATED

        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1
        return AgentState.GENERATING
