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

import logging
import os
from dataclasses import dataclass, field
from typing import Literal, Optional

from omegaconf import DictConfig

from verl.base_config import BaseConfig
from verl.utils.config import omega_conf_to_dataclass

__all__ = [
    "AgentConfig",
    "CommunicationPolicyConfig",
    "ExternalApiConfig",
    "MultiAgentFleetConfig",
]

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

AgentBackendKind = Literal["trainable_verl", "frozen_verl", "external_api"]


@dataclass
class ExternalApiConfig(BaseConfig):
    """Configuration for an agent served by an external OpenAI/Anthropic-compatible API.

    provider (str):
        "openai" or "anthropic". Defaults to inferring from ``model`` (``claude*`` -> anthropic).
    model (str):
        Model name to request from the provider.
    max_tokens (int):
        Max tokens to generate per turn.
    base_url (str, optional):
        Override API base URL, e.g. for a proxy.
    max_concurrency (int):
        Upper bound on in-flight requests to this backend per ``AgentLoopWorker`` process.
        Bounds how much a slow/rate-limited external provider can back up rollout throughput.
    timeout_s (float):
        Per-request timeout. A turn that times out is treated as a retryable failure.
    max_retries (int):
        Number of retries (with exponential backoff) after a timeout or transient error.
    """

    provider: Optional[str] = None
    model: str = "gpt-4o-mini"
    max_tokens: int = 256
    base_url: Optional[str] = None
    max_concurrency: int = 8
    timeout_s: float = 30.0
    max_retries: int = 2


@dataclass
class AgentConfig(BaseConfig):
    """Configuration for one agent role in the fleet.

    agent_id (str):
        Unique identifier for this agent within the fleet. Must match a key in
        ``MultiAgentFleetConfig.agents``.
    system_prompt (str):
        System prompt that defines this agent's role.
    backend (str):
        "trainable_verl": a verl-managed rollout server whose weights are updated by PPO.
        "frozen_verl": a verl-managed, inference-only rollout server (never trained).
        "external_api": an external OpenAI/Anthropic-compatible endpoint (never trained).
    engine_ref (str):
        "main" reuses the trainer's primary ``actor_rollout_ref`` config and worker group.
        Any other value must have a matching ``actor_rollout_ref`` sub-config on this agent
        (for "trainable_verl"/"frozen_verl") and names this agent's own resource pool.
    actor_rollout_ref (DictConfig, optional):
        Full ``actor_rollout_ref``-shaped sub-config for this agent's own model. Required for
        "trainable_verl"/"frozen_verl" agents whose ``engine_ref`` is not "main". Deliberately kept
        as a raw ``DictConfig`` (built by ``from_omegaconf``, not by generic dataclass conversion):
        ``omega_conf_to_dataclass``'s ``OmegaConf.to_object`` step would flatten it into a plain
        dict, breaking the attribute access (``agent.actor_rollout_ref.model.path``) every
        consumer of this field relies on, matching how ``config.actor_rollout_ref`` is used
        everywhere else in the trainer.
    n_gpus_per_node (int):
        GPUs per node for this agent's dedicated resource pool. Only used when ``engine_ref``
        is not "main" and ``backend`` is verl-managed.
    nnodes (int):
        Nodes for this agent's dedicated resource pool. Only used when ``engine_ref`` is not
        "main" and ``backend`` is verl-managed.
    external_api (ExternalApiConfig, optional):
        Required when ``backend == "external_api"``.
    """


    _mutable_fields = BaseConfig._mutable_fields | {"actor_rollout_ref"}

    agent_id: str = ""
    engine_ref: str = "main"

    system_prompt: str = ""
    backend: AgentBackendKind = "trainable_verl"
    actor_rollout_ref: Optional[DictConfig] = None
    n_gpus_per_node: int = 0
    nnodes: int = 0
    external_api: Optional[ExternalApiConfig] = None

    @property
    def trainable(self) -> bool:
        return self.backend == "trainable_verl"

    @property
    def is_main(self) -> bool:
        return self.engine_ref == "main"

    @classmethod
    def from_omegaconf(cls, agent_id: str, cfg: DictConfig) -> "AgentConfig":
        external_api = None
        if cfg.get("external_api") is not None:
            external_api = omega_conf_to_dataclass(cfg.external_api, dataclass_type=ExternalApiConfig)
        return cls(
            agent_id=cfg.get("agent_id") or agent_id,
            system_prompt=cfg.get("system_prompt", ""),
            backend=cfg.get("backend", "trainable_verl"),
            engine_ref=cfg.get("engine_ref", "main"),
            actor_rollout_ref=cfg.get("actor_rollout_ref"),
            n_gpus_per_node=cfg.get("n_gpus_per_node", 0),
            nnodes=cfg.get("nnodes", 0),
            external_api=external_api,
        )

    def check_configured(self) -> None:
        if not self.agent_id:
            raise ValueError("AgentConfig.agent_id must be set.")
        if self.backend == "external_api":
            if self.external_api is None:
                raise ValueError(
                    f"Agent {self.agent_id!r} has backend='external_api' but no external_api config."
                )
            if self.is_main:
                raise ValueError(
                    f"Agent {self.agent_id!r} is 'main' but backend='external_api'; main must be verl-hosted."
                )
        elif not self.is_main:
            if self.actor_rollout_ref is None:
                raise ValueError(
                    f"Agent {self.agent_id!r} (backend={self.backend!r}, engine_ref={self.engine_ref!r}) "
                    "needs its own actor_rollout_ref config."
                )
            if self.n_gpus_per_node <= 0 or self.nnodes <= 0:
                raise ValueError(
                    f"Agent {self.agent_id!r} needs n_gpus_per_node > 0 and nnodes > 0 for its own resource pool."
                )


@dataclass
class CommunicationPolicyConfig(BaseConfig):
    """Configuration for the (pluggable) communication policy that decides turn order.

    kind (str):
        Which ``CommunicationPolicy`` implementation to use. v1 only ships "rigid_sequence".
    turn_order (list[str]):
        Agent ids, cycled in order by ``RigidSequencePolicy``.
    max_turns (int):
        Hard cap on the number of turns in one episode, regardless of termination.
    termination_agent_id (str, optional):
        If set, the episode may end early when this agent's turn result reports done.
    termination_metric (str):
        Key read from an ``AgentTurnResult.metrics`` dict to detect early termination.
    """

    # str, not Literal: OmegaConf.structured (used by omega_conf_to_dataclass) doesn't support
    # Literal type annotations on structured-config fields.
    kind: str = "rigid_sequence"
    turn_order: list[str] = field(default_factory=list)
    max_turns: int = 4
    termination_agent_id: Optional[str] = None
    termination_metric: str = "done"

    def check_configured(self) -> None:
        if not self.turn_order:
            raise ValueError("CommunicationPolicyConfig.turn_order must be non-empty.")
        if self.max_turns <= 0:
            raise ValueError("CommunicationPolicyConfig.max_turns must be > 0.")


@dataclass
class MultiAgentFleetConfig(BaseConfig):
    """Top-level configuration for a multi-agent fleet.

    agents (dict[str, AgentConfig]):
        Fleet members, keyed by agent_id (the key and ``AgentConfig.agent_id`` must match).
        Exactly one agent must have ``engine_ref == "main"``.
    policy (CommunicationPolicyConfig):
        Turn-taking policy shared by every episode.
    """

    agents: dict[str, AgentConfig] = field(default_factory=dict)
    policy: CommunicationPolicyConfig = field(default_factory=CommunicationPolicyConfig)

    @classmethod
    def from_omegaconf(cls, cfg: DictConfig) -> "MultiAgentFleetConfig":
        """Build from the raw ``config.multiagent`` DictConfig.

        Doesn't use ``omega_conf_to_dataclass(cfg, dataclass_type=MultiAgentFleetConfig)`` directly:
        that would route every nested field (including each agent's ``actor_rollout_ref``) through
        ``OmegaConf.structured``/``to_object``, flattening ``actor_rollout_ref`` into a plain dict.
        See ``AgentConfig.actor_rollout_ref`` docstring.
        """
        agents = {
            agent_id: AgentConfig.from_omegaconf(agent_id, agent_cfg)
            for agent_id, agent_cfg in cfg.get("agents", {}).items()
        }
        policy_cfg = cfg.get("policy")
        policy = (
            omega_conf_to_dataclass(policy_cfg, dataclass_type=CommunicationPolicyConfig)
            if policy_cfg is not None
            else CommunicationPolicyConfig()
        )
        return cls(agents=agents, policy=policy)

    def __post_init__(self):
        if not self.agents:
            return
        main_agents = [agent_id for agent_id, agent in self.agents.items() if agent.is_main]
        if len(main_agents) != 1:
            raise ValueError(
                f"MultiAgentFleetConfig.agents must contain exactly one engine_ref='main' agent, got {main_agents}."
            )
        if not self.agents[main_agents[0]].trainable:
            # Not just a style preference: MultiAgentLoop only builds an AgentLoopOutput (and
            # so a trainable TQ row) for trainable agents. A non-trainable main with no other
            # trainable agent in the fleet would silently produce zero trainable rows per episode.
            raise ValueError(f"The main agent ({main_agents[0]!r}) must have backend='trainable_verl'.")
        for agent_id, agent in self.agents.items():
            if agent.agent_id != agent_id:
                raise ValueError(f"agents key {agent_id!r} does not match AgentConfig.agent_id {agent.agent_id!r}.")
            agent.check_configured()
        for agent_id in self.policy.turn_order:
            if agent_id not in self.agents:
                raise ValueError(f"policy.turn_order references unknown agent_id {agent_id!r}.")
        self.policy.check_configured()
        self._check_shared_model_groups()

    def _check_shared_model_groups(self) -> None:
        """Trainable agents that share a ``engine_ref`` (other than the solo "main") collapse onto
        one resource pool, worker group, and ``LLMServerManager`` in the trainer -- see
        ``MultiAgentPPOTrainer._setup_agent_groups`` -- so they must agree on model config.
        """
        groups: dict[str, list[AgentConfig]] = {}
        for agent in self.agents.values():
            if agent.is_main or agent.backend != "trainable_verl":
                continue
            groups.setdefault(agent.engine_ref, []).append(agent)
        for engine_ref, group in groups.items():
            reference = group[0]
            for other in group[1:]:
                if other.actor_rollout_ref != reference.actor_rollout_ref:
                    raise ValueError(
                        f"Trainable agents sharing engine_ref={engine_ref!r} must have identical "
                        f"actor_rollout_ref configs ({reference.agent_id!r} vs {other.agent_id!r})."
                    )
                if (other.n_gpus_per_node, other.nnodes) != (reference.n_gpus_per_node, reference.nnodes):
                    raise ValueError(
                        f"Trainable agents sharing engine_ref={engine_ref!r} must have identical "
                        f"n_gpus_per_node/nnodes ({reference.agent_id!r} vs {other.agent_id!r})."
                    )

    @property
    def main_agent_id(self) -> str:
        return next(agent_id for agent_id, agent in self.agents.items() if agent.is_main)

    def trainable_agent_ids(self) -> list[str]:
        return [agent_id for agent_id, agent in self.agents.items() if agent.trainable]
