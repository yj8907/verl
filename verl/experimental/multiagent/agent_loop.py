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
"""``MultiAgentLoop``: drives one fleet episode (rigid communication policy + per-actor backends)
and returns one ``AgentLoopOutput`` per trainable actor, so the existing TransferQueue-writing and
reward-broadcast pipeline (``AgentLoopWorkerTQ._agent_loop_postprocess``) needs no changes -- it
already accepts ``list[AgentLoopOutput]`` per session and already broadcasts the last output's
score to every earlier one.

v1 scope: text-only (no multimodal), every actor uses the same rollout sampling params.
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
from verl.experimental.multiagent.actor_backend import (
    ActorBackend,
    ActorTurnResult,
    ExternalAPIActorBackend,
    FrozenVerlActorBackend,
    TrainableVerlActorBackend,
)
from verl.experimental.multiagent.config.actor_config import MultiAgentFleetConfig
from verl.experimental.multiagent.policy import EpisodeState, build_policy
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


class ActorTurnRenderer:
    """Builds one trainable actor's own token buffer across a multi-agent episode.

    Uses the same incremental turn-separator technique as ``ToolAgentLoop``: every extension is
    rendered with ``add_generation_prompt=False`` and reproduces exactly what re-templating the
    full conversation from scratch would produce, so a fresh ``add_generation_prompt=True`` call
    (what ``ActorBackend.respond`` does internally to build its own generation prompt) is
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


class ActorBackendsWrap:
    """Wraps the actor-backend dict so ``hydra.utils.instantiate`` doesn't recursively resolve it."""

    def __init__(self, actor_backends: dict[str, ActorBackend]):
        self.actor_backends = actor_backends


def _observer_message(actor_id: str, text: str) -> dict:
    return {"role": "user", "content": f"[{actor_id}] {text}"}


@register("multi_agent")
class MultiAgentLoop(AgentLoopBase):
    """Drives one episode across a rigid fleet of actors and returns one ``AgentLoopOutput`` per
    trainable actor (last entry = the one the reward function scores)."""

    def __init__(self, *args, actor_backends: Optional[ActorBackendsWrap] = None, **kwargs):
        super().__init__(*args, **kwargs)
        if actor_backends is None:
            raise ValueError("MultiAgentLoop requires actor_backends (see MultiAgentLoopWorkerTQ).")
        self.actor_backends: dict[str, ActorBackend] = actor_backends.actor_backends
        self.fleet_config: MultiAgentFleetConfig = MultiAgentFleetConfig.from_omegaconf(self.config.multiagent)
        self.policy = build_policy(self.fleet_config.policy)

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> list[AgentLoopOutput]:
        raw_prompt = list(kwargs["raw_prompt"])
        fleet = self.fleet_config
        trainable_actor_ids = fleet.trainable_actor_ids()

        # Every actor's own running view of the conversation (system prompt excluded here --
        # ActorBackend.respond re-adds it, since it varies per actor).
        actor_messages: dict[str, list[dict]] = {actor_id: list(raw_prompt) for actor_id in fleet.actors}

        renderers: dict[str, ActorTurnRenderer] = {}
        buffers: dict[str, dict[str, list]] = {}
        initial_lens: dict[str, int] = {}
        for actor_id in trainable_actor_ids:
            tokenizer = self.actor_backends[actor_id].tokenizer
            renderer = ActorTurnRenderer(tokenizer)
            system_message = {"role": "system", "content": fleet.actors[actor_id].system_prompt}
            initial_ids = renderer.build_initial_prompt([system_message] + raw_prompt)
            renderers[actor_id] = renderer
            initial_lens[actor_id] = len(initial_ids)
            buffers[actor_id] = {"prompt_ids": list(initial_ids), "response_mask": [], "response_logprobs": []}

        state = EpisodeState(turn_index=0, messages_per_actor=actor_messages)
        num_turns = 0
        while True:
            next_actor_id = self.policy.next_actor(state)
            if next_actor_id is None:
                break
            num_turns += 1

            actor_cfg = fleet.actors[next_actor_id]
            backend = self.actor_backends[next_actor_id]
            result: ActorTurnResult = await backend.respond(
                messages=actor_messages[next_actor_id],
                system_prompt=actor_cfg.system_prompt,
                sampling_params=sampling_params,
            )

            own_message = {"role": "assistant", "content": result.text}
            speaker_message = _observer_message(next_actor_id, result.text)
            for actor_id in fleet.actors:
                actor_messages[actor_id].append(own_message if actor_id == next_actor_id else speaker_message)

            for actor_id in trainable_actor_ids:
                renderer, buf = renderers[actor_id], buffers[actor_id]
                if actor_id == next_actor_id:
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
                messages_per_actor=actor_messages,
                last_actor_id=next_actor_id,
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
        # The actor scored by the reward function is whichever trainable actor took the episode's
        # last turn (naturally "who finished the episode"); if the episode ended on a non-trainable
        # actor's turn, fall back to the fleet's main actor. It must be ordered last: the existing
        # TQ postprocess broadcasts outputs[-1]'s reward_score to every earlier output in the list.
        trainable_actor_ids = list(buffers.keys())
        scored_actor_id = state.last_actor_id if state.last_actor_id in buffers else self.fleet_config.main_actor_id
        ordered_actor_ids = [a for a in trainable_actor_ids if a != scored_actor_id] + [scored_actor_id]

        outputs = []
        for actor_id in ordered_actor_ids:
            buf = buffers[actor_id]
            prompt_len = initial_lens[actor_id]
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
                        "actor_id": actor_id,
                        "model_ref": self.fleet_config.actors[actor_id].model_ref,
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
    """``AgentLoopWorkerTQ`` extended to build and inject per-actor backends.

    ``AgentLoopWorker._run_agent_loop``'s ``hydra.utils.instantiate`` call doesn't pass
    ``teacher_client``-like dicts through to the agent loop instance, so this subclass overrides
    it (copy-adapted from ``agent_loop.py:_run_agent_loop``) to add ``actor_backends``. Not
    ``@ray.remote``-decorated itself -- see ``_unwrap_actor_class`` -- ``MultiAgentLoopManagerTQ``
    wraps it with ``ray.remote(...)`` before spawning.
    """

    def __init__(self, config, llm_client, teacher_client, reward_loop_worker_handles, actor_llm_clients):
        super().__init__(config, llm_client, teacher_client, reward_loop_worker_handles)
        self.fleet_config: MultiAgentFleetConfig = MultiAgentFleetConfig.from_omegaconf(config.multiagent)
        self.actor_backends: dict[str, ActorBackend] = self._build_actor_backends(actor_llm_clients)

    def _build_actor_backends(self, actor_llm_clients: dict[str, LLMServerClient]) -> dict[str, ActorBackend]:
        backends: dict[str, ActorBackend] = {}
        for actor_id, actor in self.fleet_config.actors.items():
            if actor.backend == "external_api":
                backends[actor_id] = ExternalAPIActorBackend(actor_id, actor.external_api)
            elif actor.is_main:
                backends[actor_id] = TrainableVerlActorBackend(actor_id, self.llm_client, self.tokenizer)
            else:
                tokenizer = hf_tokenizer(actor.actor_rollout_ref.model.path)
                client = actor_llm_clients[actor_id]
                backend_cls = TrainableVerlActorBackend if actor.trainable else FrozenVerlActorBackend
                backends[actor_id] = backend_cls(actor_id, client, tokenizer)
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
                actor_backends=ActorBackendsWrap(self.actor_backends),
                **kwargs,
            )
            output = await agent_loop.run(sampling_params, **kwargs)
            return await self._agent_loop_postprocess(output, trajectory["validate"], **kwargs)


class MultiAgentLoopManagerTQ(_AgentLoopManagerTQBase):
    """``AgentLoopManagerTQ`` extended to spawn ``MultiAgentLoopWorkerTQ`` workers with the
    fleet's non-main actor ``LLMServerClient``\\s (frozen or trainable-non-main)."""

    def __init__(self, *args, actor_llm_clients: Optional[dict[str, LLMServerClient]] = None, **kwargs):
        self.actor_llm_clients = actor_llm_clients or {}
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
                    self.actor_llm_clients,
                )
            )


__all__ = [
    "ActorBackendsWrap",
    "ActorTurnRenderer",
    "MultiAgentLoop",
    "MultiAgentLoopManagerTQ",
    "MultiAgentLoopWorkerTQ",
]
