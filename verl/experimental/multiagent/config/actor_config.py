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
    "ActorConfig",
    "CommunicationPolicyConfig",
    "ExternalApiConfig",
    "MultiAgentFleetConfig",
]

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

ActorBackendKind = Literal["trainable_verl", "frozen_verl", "external_api"]


@dataclass
class ExternalApiConfig(BaseConfig):
    """Configuration for an actor served by an external OpenAI/Anthropic-compatible API.

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
class ActorConfig(BaseConfig):
    """Configuration for one actor role in the fleet.

    actor_id (str):
        Unique identifier for this actor within the fleet. Must match a key in
        ``MultiAgentFleetConfig.actors``.
    system_prompt (str):
        System prompt that defines this actor's role.
    backend (str):
        "trainable_verl": a verl-managed rollout server whose weights are updated by PPO.
        "frozen_verl": a verl-managed, inference-only rollout server (never trained).
        "external_api": an external OpenAI/Anthropic-compatible endpoint (never trained).
    model_ref (str):
        "main" reuses the trainer's primary ``actor_rollout_ref`` config and worker group.
        Any other value must have a matching ``actor_rollout_ref`` sub-config on this actor
        (for "trainable_verl"/"frozen_verl") and names this actor's own resource pool.
    actor_rollout_ref (DictConfig, optional):
        Full ``actor_rollout_ref``-shaped sub-config for this actor's own model. Required for
        "trainable_verl"/"frozen_verl" actors whose ``model_ref`` is not "main". Deliberately kept
        as a raw ``DictConfig`` (built by ``from_omegaconf``, not by generic dataclass conversion):
        ``omega_conf_to_dataclass``'s ``OmegaConf.to_object`` step would flatten it into a plain
        dict, breaking the attribute access (``actor.actor_rollout_ref.model.path``) every
        consumer of this field relies on, matching how ``config.actor_rollout_ref`` is used
        everywhere else in the trainer.
    n_gpus_per_node (int):
        GPUs per node for this actor's dedicated resource pool. Only used when ``model_ref``
        is not "main" and ``backend`` is verl-managed.
    nnodes (int):
        Nodes for this actor's dedicated resource pool. Only used when ``model_ref`` is not
        "main" and ``backend`` is verl-managed.
    external_api (ExternalApiConfig, optional):
        Required when ``backend == "external_api"``.
    """

    _mutable_fields = BaseConfig._mutable_fields | {"actor_rollout_ref"}

    actor_id: str = ""
    system_prompt: str = ""
    backend: ActorBackendKind = "trainable_verl"
    model_ref: str = "main"
    actor_rollout_ref: Optional[DictConfig] = None
    n_gpus_per_node: int = 0
    nnodes: int = 0
    external_api: Optional[ExternalApiConfig] = None

    @property
    def trainable(self) -> bool:
        return self.backend == "trainable_verl"

    @property
    def is_main(self) -> bool:
        return self.model_ref == "main"

    @classmethod
    def from_omegaconf(cls, actor_id: str, cfg: DictConfig) -> "ActorConfig":
        external_api = None
        if cfg.get("external_api") is not None:
            external_api = omega_conf_to_dataclass(cfg.external_api, dataclass_type=ExternalApiConfig)
        return cls(
            actor_id=cfg.get("actor_id") or actor_id,
            system_prompt=cfg.get("system_prompt", ""),
            backend=cfg.get("backend", "trainable_verl"),
            model_ref=cfg.get("model_ref", "main"),
            actor_rollout_ref=cfg.get("actor_rollout_ref"),
            n_gpus_per_node=cfg.get("n_gpus_per_node", 0),
            nnodes=cfg.get("nnodes", 0),
            external_api=external_api,
        )

    def check_configured(self) -> None:
        if not self.actor_id:
            raise ValueError("ActorConfig.actor_id must be set.")
        if self.backend == "external_api":
            if self.external_api is None:
                raise ValueError(
                    f"Actor {self.actor_id!r} has backend='external_api' but no external_api config."
                )
            if self.is_main:
                raise ValueError(
                    f"Actor {self.actor_id!r} is 'main' but backend='external_api'; main must be verl-hosted."
                )
        elif not self.is_main:
            if self.actor_rollout_ref is None:
                raise ValueError(
                    f"Actor {self.actor_id!r} (backend={self.backend!r}, model_ref={self.model_ref!r}) "
                    "needs its own actor_rollout_ref config."
                )
            if self.n_gpus_per_node <= 0 or self.nnodes <= 0:
                raise ValueError(
                    f"Actor {self.actor_id!r} needs n_gpus_per_node > 0 and nnodes > 0 for its own resource pool."
                )


@dataclass
class CommunicationPolicyConfig(BaseConfig):
    """Configuration for the (pluggable) communication policy that decides turn order.

    kind (str):
        Which ``CommunicationPolicy`` implementation to use. v1 only ships "rigid_sequence".
    turn_order (list[str]):
        Actor ids, cycled in order by ``RigidSequencePolicy``.
    max_turns (int):
        Hard cap on the number of turns in one episode, regardless of termination.
    termination_actor_id (str, optional):
        If set, the episode may end early when this actor's turn result reports done.
    termination_metric (str):
        Key read from an ``ActorTurnResult.metrics`` dict to detect early termination.
    """

    # str, not Literal: OmegaConf.structured (used by omega_conf_to_dataclass) doesn't support
    # Literal type annotations on structured-config fields.
    kind: str = "rigid_sequence"
    turn_order: list[str] = field(default_factory=list)
    max_turns: int = 4
    termination_actor_id: Optional[str] = None
    termination_metric: str = "done"

    def check_configured(self) -> None:
        if not self.turn_order:
            raise ValueError("CommunicationPolicyConfig.turn_order must be non-empty.")
        if self.max_turns <= 0:
            raise ValueError("CommunicationPolicyConfig.max_turns must be > 0.")


@dataclass
class MultiAgentFleetConfig(BaseConfig):
    """Top-level configuration for a multi-agent actor fleet.

    actors (dict[str, ActorConfig]):
        Fleet members, keyed by actor_id (the key and ``ActorConfig.actor_id`` must match).
        Exactly one actor must have ``model_ref == "main"``.
    policy (CommunicationPolicyConfig):
        Turn-taking policy shared by every episode.
    """

    actors: dict[str, ActorConfig] = field(default_factory=dict)
    policy: CommunicationPolicyConfig = field(default_factory=CommunicationPolicyConfig)

    @classmethod
    def from_omegaconf(cls, cfg: DictConfig) -> "MultiAgentFleetConfig":
        """Build from the raw ``config.multiagent`` DictConfig.

        Doesn't use ``omega_conf_to_dataclass(cfg, dataclass_type=MultiAgentFleetConfig)`` directly:
        that would route every nested field (including each actor's ``actor_rollout_ref``) through
        ``OmegaConf.structured``/``to_object``, flattening ``actor_rollout_ref`` into a plain dict.
        See ``ActorConfig.actor_rollout_ref`` docstring.
        """
        actors = {
            actor_id: ActorConfig.from_omegaconf(actor_id, actor_cfg)
            for actor_id, actor_cfg in cfg.get("actors", {}).items()
        }
        policy_cfg = cfg.get("policy")
        policy = (
            omega_conf_to_dataclass(policy_cfg, dataclass_type=CommunicationPolicyConfig)
            if policy_cfg is not None
            else CommunicationPolicyConfig()
        )
        return cls(actors=actors, policy=policy)

    def __post_init__(self):
        if not self.actors:
            return
        main_actors = [actor_id for actor_id, actor in self.actors.items() if actor.is_main]
        if len(main_actors) != 1:
            raise ValueError(
                f"MultiAgentFleetConfig.actors must contain exactly one model_ref='main' actor, got {main_actors}."
            )
        if not self.actors[main_actors[0]].trainable:
            # Not just a style preference: MultiAgentLoop only builds an AgentLoopOutput (and
            # so a trainable TQ row) for trainable actors. A non-trainable main with no other
            # trainable actor in the fleet would silently produce zero trainable rows per episode.
            raise ValueError(f"The main actor ({main_actors[0]!r}) must have backend='trainable_verl'.")
        for actor_id, actor in self.actors.items():
            if actor.actor_id != actor_id:
                raise ValueError(f"actors key {actor_id!r} does not match ActorConfig.actor_id {actor.actor_id!r}.")
            actor.check_configured()
        for actor_id in self.policy.turn_order:
            if actor_id not in self.actors:
                raise ValueError(f"policy.turn_order references unknown actor_id {actor_id!r}.")
        self.policy.check_configured()

    @property
    def main_actor_id(self) -> str:
        return next(actor_id for actor_id, actor in self.actors.items() if actor.is_main)

    def trainable_actor_ids(self) -> list[str]:
        return [actor_id for actor_id, actor in self.actors.items() if actor.trainable]
