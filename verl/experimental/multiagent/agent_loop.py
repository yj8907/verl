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
"""``MultiAgentLoop``: drives one fleet episode (rigid communication policy + per-agent backends)
and returns one ``AgentLoopOutput`` per trainable agent, so the existing TransferQueue-writing and
reward-broadcast pipeline (``AgentLoopWorkerTQ._agent_loop_postprocess``) needs no changes -- it
already accepts ``list[AgentLoopOutput]`` per session and already broadcasts the last output's
score to every earlier one.

v1 scope: text-only (no multimodal), every agent uses the same rollout sampling params.
"""

import logging
import os
from typing import Any, Optional
from uuid import uuid4

import hydra
import ray

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
    DictConfigWrap,
    register,
)
from verl.experimental.multiagent.agent_backend import (
    AgentBackend,
    AgentTurnResult,
    ExternalAPIAgentBackend,
    FrozenVerlAgentBackend,
    TrainableVerlAgentBackend,
)
from verl.experimental.multiagent.config.agent_config import MultiAgentFleetConfig
from verl.experimental.multiagent.comm_policy import EpisodeState, build_policy
from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopManagerTQ as _AgentLoopManagerTQBase
from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopWorkerTQ
from verl.utils import hf_tokenizer
from verl.utils.rollout_trace import rollout_trace_attr
from verl.utils.tokenizer import normalize_token_ids
from verl.utils.tokenizer.chat_template import apply_chat_template, initialize_system_prompt, initialize_turn_separator
from verl.workers.rollout.llm_server import LLMServerClient

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _derive_generation_marker(tokenizer) -> list[int]:
    """The tokens a chat template appends when ``add_generation_prompt=True`` vs ``False`` -- the
    trailing "assistant turn opener" that must precede a fresh generation call. Derived the same
    way ``initialize_turn_separator``/``initialize_system_prompt`` derive their fixed constants:
    diff two renderings that differ only in the flag of interest.
    """
    probe = [{"role": "user", "content": ""}]
    without = normalize_token_ids(tokenizer.apply_chat_template(probe, add_generation_prompt=False, tokenize=True))
    with_marker = normalize_token_ids(tokenizer.apply_chat_template(probe, add_generation_prompt=True, tokenize=True))
    return list(with_marker[len(without) :])


class AgentTurnRenderer:
    """Builds one trainable agent's own token buffer across a multi-agent episode.

    Uses the same incremental turn-separator technique as ``ToolAgentLoop``: every extension is
    rendered with ``add_generation_prompt=False`` and reproduces exactly what re-templating the
    full conversation from scratch would produce, so a fresh ``add_generation_prompt=True`` call
    (what ``AgentBackend.respond`` does internally to build its own generation prompt) is
    mathematically guaranteed to equal ``buffer + generation_marker`` -- keeping the sequence
    stored for training identical to what generation was actually conditioned on.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.system_prompt = initialize_system_prompt(tokenizer)
        self.turn_separator = initialize_turn_separator(tokenizer)
        self.generation_marker = _derive_generation_marker(tokenizer)

    def build_initial_prompt(self, messages: list[dict]) -> list[int]:
        ids = apply_chat_template(self.tokenizer, messages, add_generation_prompt=False, tokenize=True)
        return normalize_token_ids(ids)

    def render_observation(self, message: dict) -> list[int]:
        ids = apply_chat_template(self.tokenizer, [message], add_generation_prompt=False, tokenize=True)
        ids = normalize_token_ids(ids)
        return self.turn_separator + ids[len(self.system_prompt) :]


class AgentBackendsWrap:
    """Wraps the agent-backend dict so ``hydra.utils.instantiate`` doesn't recursively resolve it."""

    def __init__(self, agent_backends: dict[str, AgentBackend]):
        self.agent_backends = agent_backends


def _observer_message(agent_id: str, text: str) -> dict:
    return {"role": "user", "content": f"[{agent_id}] {text}"}


@register("multi_agent")
class MultiAgentLoop(AgentLoopBase):
    """Drives one episode across a rigid fleet of agents and returns one ``AgentLoopOutput`` per
    trainable agent (last entry = the one the reward function scores)."""

    def __init__(self, *args, agent_backends: Optional[AgentBackendsWrap] = None, **kwargs):
        super().__init__(*args, **kwargs)
        if agent_backends is None:
            raise ValueError("MultiAgentLoop requires agent_backends (see MultiAgentLoopWorkerTQ).")
        self.agent_backends: dict[str, AgentBackend] = agent_backends.agent_backends
        self.fleet_config: MultiAgentFleetConfig = MultiAgentFleetConfig.from_omegaconf(self.config.multiagent)
        self.comm_policy = build_policy(self.fleet_config.policy)

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> list[AgentLoopOutput]:
        raw_prompt = list(kwargs["raw_prompt"])
        fleet = self.fleet_config
        trainable_agent_ids = fleet.trainable_agent_ids()

        # Every agent's own running view of the conversation (system prompt excluded here --
        # AgentBackend.respond re-adds it, since it varies per agent).
        agent_messages: dict[str, list[dict]] = {agent_id: list(raw_prompt) for agent_id in fleet.agents}

        renderers: dict[str, AgentTurnRenderer] = {}
        buffers: dict[str, dict[str, list]] = {}
        initial_lens: dict[str, int] = {}
        for agent_id in trainable_agent_ids:
            tokenizer = self.agent_backends[agent_id].tokenizer
            renderer = AgentTurnRenderer(tokenizer)
            system_message = {"role": "system", "content": fleet.agents[agent_id].system_prompt}
            initial_ids = renderer.build_initial_prompt([system_message] + raw_prompt)
            renderers[agent_id] = renderer
            initial_lens[agent_id] = len(initial_ids)
            buffers[agent_id] = {"prompt_ids": list(initial_ids), "response_mask": [], "response_logprobs": []}

        state = EpisodeState(turn_index=0, messages_per_agent=agent_messages)
        num_turns = 0
        while True:
            next_agent_id = self.comm_policy.next_agent(state)
            if next_agent_id is None:
                break
            num_turns += 1

            agent_cfg = fleet.agents[next_agent_id]
            backend = self.agent_backends[next_agent_id]
            result: AgentTurnResult = await backend.respond(
                messages=agent_messages[next_agent_id],
                system_prompt=agent_cfg.system_prompt,
                sampling_params=sampling_params,
            )

            own_message = {"role": "assistant", "content": result.text}
            speaker_message = _observer_message(next_agent_id, result.text)
            for agent_id in fleet.agents:
                agent_messages[agent_id].append(own_message if agent_id == next_agent_id else speaker_message)

            for agent_id in trainable_agent_ids:
                renderer, buf = renderers[agent_id], buffers[agent_id]
                if agent_id == next_agent_id:
                    marker = renderer.generation_marker
                    response_ids = result.response_token_ids or []
                    buf["prompt_ids"] += marker + response_ids
                    buf["response_mask"] += [0] * len(marker) + [1] * len(response_ids)
                    logprobs = result.response_logprobs or [0.0] * len(response_ids)
                    buf["response_logprobs"] += [0.0] * len(marker) + list(logprobs)
                else:
                    ids = renderer.render_observation(speaker_message)
                    buf["prompt_ids"] += ids
                    buf["response_mask"] += [0] * len(ids)
                    buf["response_logprobs"] += [0.0] * len(ids)

            state = EpisodeState(
                turn_index=state.turn_index + 1,
                messages_per_agent=agent_messages,
                last_agent_id=next_agent_id,
                last_result=result,
            )

        return self._build_outputs(buffers, initial_lens, state, num_turns)

    def _build_outputs(
        self,
        buffers: dict[str, dict[str, list]],
        initial_lens: dict[str, int],
        state: EpisodeState,
        num_turns: int,
    ) -> list[AgentLoopOutput]:
        # The agent scored by the reward function is whichever trainable agent took the episode's
        # last turn (naturally "who finished the episode"); if the episode ended on a non-trainable
        # agent's turn, fall back to the fleet's main agent. It must be ordered last: the existing
        # TQ postprocess broadcasts outputs[-1]'s reward_score to every earlier output in the list.
        trainable_agent_ids = list(buffers.keys())
        scored_agent_id = state.last_agent_id if state.last_agent_id in buffers else self.fleet_config.main_agent_id
        ordered_agent_ids = [a for a in trainable_agent_ids if a != scored_agent_id] + [scored_agent_id]

        outputs = []
        for agent_id in ordered_agent_ids:
            buf = buffers[agent_id]
            prompt_len = initial_lens[agent_id]
            response_mask = buf["response_mask"]
            response_ids = buf["prompt_ids"][prompt_len:]
            prompt_ids = buf["prompt_ids"][:prompt_len]
            response_logprobs = buf["response_logprobs"]
            outputs.append(
                AgentLoopOutput(
                    prompt_ids=prompt_ids,
                    response_ids=response_ids,
                    response_mask=response_mask,
                    response_logprobs=response_logprobs if any(response_logprobs) else None,
                    num_turns=num_turns,
                    metrics=AgentLoopMetrics(),
                    extra_fields={
                        "agent_id": agent_id,
                        "model_ref": self.fleet_config.agents[agent_id].model_ref,
                        "trainable": True,
                    },
                )
            )
        return outputs


def _unwrap_actor_class(actor_cls):
    """Return the plain (pre-``@ray.remote``) class backing a Ray actor class.

    ``AgentLoopWorkerTQ`` is decorated with ``@ray.remote`` directly on the class (unlike
    ``AgentLoopWorker``, which stays a plain class and is wrapped later via
    ``ray.remote(AgentLoopWorker)`` inside ``AgentLoopManager``), so subclassing it directly raises
    Ray's ``ActorClassInheritanceException``. Ray's actor decorator wraps the original class in a
    thin subclass that adds actor-specific methods; unwrapping one level of ``__bases__`` recovers
    the original plain class, which normal Python subclassing works on -- re-wrapped afterward via
    ``ray.remote(...)``, the same way ``AgentLoopManager`` builds ``agent_loop_workers_class``.
    """
    metadata = getattr(actor_cls, "__ray_metadata__", None)
    if metadata is None or not metadata.modified_class.__bases__:
        raise RuntimeError(
            f"{actor_cls!r} does not look like a @ray.remote-decorated class; can't recover its "
            "pre-decoration base to subclass. This depends on Ray's internal ActorClass layout."
        )
    return metadata.modified_class.__bases__[0]


_AgentLoopWorkerTQBase = _unwrap_actor_class(AgentLoopWorkerTQ)


class MultiAgentLoopWorkerTQ(_AgentLoopWorkerTQBase):
    """``AgentLoopWorkerTQ`` extended to build and inject per-agent backends.

    ``AgentLoopWorker._run_agent_loop``'s ``hydra.utils.instantiate`` call doesn't pass
    ``teacher_client``-like dicts through to the agent loop instance, so this subclass overrides
    it (copy-adapted from ``agent_loop.py:_run_agent_loop``) to add ``agent_backends``. Not
    ``@ray.remote``-decorated itself -- see ``_unwrap_actor_class`` -- ``MultiAgentLoopManagerTQ``
    wraps it with ``ray.remote(...)`` before spawning.
    """

    def __init__(self, config, llm_client, teacher_client, reward_loop_worker_handles, agent_llm_clients):
        super().__init__(config, llm_client, teacher_client, reward_loop_worker_handles)
        self.fleet_config: MultiAgentFleetConfig = MultiAgentFleetConfig.from_omegaconf(config.multiagent)
        self.agent_backends: dict[str, AgentBackend] = self._build_agent_backends(agent_llm_clients)

    def _build_agent_backends(self, agent_llm_clients: dict[str, LLMServerClient]) -> dict[str, AgentBackend]:
        backends: dict[str, AgentBackend] = {}
        for agent_id, agent in self.fleet_config.agents.items():
            if agent.backend == "external_api":
                backends[agent_id] = ExternalAPIAgentBackend(agent_id, agent.external_api)
            elif agent.is_main:
                backends[agent_id] = TrainableVerlAgentBackend(agent_id, self.llm_client, self.tokenizer)
            else:
                tokenizer = hf_tokenizer(agent.actor_rollout_ref.model.path)
                client = agent_llm_clients[agent_id]
                backend_cls = TrainableVerlAgentBackend if agent.trainable else FrozenVerlAgentBackend
                backends[agent_id] = backend_cls(agent_id, client, tokenizer)
        return backends

    async def _run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        trace: bool = True,
        **kwargs,
    ) -> None:
        from verl.experimental.agent_loop.agent_loop import _agent_loop_registry

        with rollout_trace_attr(
            step=trajectory["step"],
            sample_index=trajectory["sample_index"],
            rollout_n=trajectory["rollout_n"],
            validate=trajectory["validate"],
            name="agent_loop",
            trace=trace,
        ):
            assert agent_name in _agent_loop_registry, (
                f"Agent loop {agent_name} not registered, registered agent loops: {_agent_loop_registry.keys()}"
            )
            agent_loop_config = _agent_loop_registry[agent_name]
            agent_loop = hydra.utils.instantiate(
                config=agent_loop_config,
                trainer_config=DictConfigWrap(config=self.config),
                server_manager=self.llm_client,
                tokenizer=self.tokenizer,
                processor=self.processor,
                dataset_cls=self.dataset_cls,
                data_config=DictConfigWrap(self.config.data),
                agent_backends=AgentBackendsWrap(self.agent_backends),
                **kwargs,
            )
            output = await agent_loop.run(sampling_params, **kwargs)
            return await self._agent_loop_postprocess(output, trajectory["validate"], **kwargs)


class MultiAgentLoopManagerTQ(_AgentLoopManagerTQBase):
    """``AgentLoopManagerTQ`` extended to spawn ``MultiAgentLoopWorkerTQ`` workers with the
    fleet's non-main agent ``LLMServerClient``\\s (frozen or trainable-non-main)."""

    def __init__(self, *args, agent_llm_clients: Optional[dict[str, LLMServerClient]] = None, **kwargs):
        self.agent_llm_clients = agent_llm_clients or {}
        super().__init__(*args, **kwargs)
        # AgentLoopManagerTQ.__init__ unconditionally sets this to AgentLoopWorkerTQ; override after.
        self.agent_loop_workers_class = ray.remote(MultiAgentLoopWorkerTQ)

    async def _init_agent_loop_workers(self):
        self.agent_loop_workers = []
        num_workers = self.rollout_config.agent.num_workers
        node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]
        for i in range(num_workers):
            node_id = node_ids[i % len(node_ids)]
            self.agent_loop_workers.append(
                self.agent_loop_workers_class.options(
                    name=f"agent_loop_worker_{i}_{uuid4().hex[:8]}",
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=True
                    ),
                ).remote(
                    self.config,
                    self.llm_client,
                    self.teacher_client,
                    self.reward_loop_worker_handles,
                    self.agent_llm_clients,
                )
            )


__all__ = [
    "AgentBackendsWrap",
    "AgentTurnRenderer",
    "MultiAgentLoop",
    "MultiAgentLoopManagerTQ",
    "MultiAgentLoopWorkerTQ",
]
