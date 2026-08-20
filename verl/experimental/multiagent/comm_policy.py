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
"""Pluggable communication policy: decides which actor speaks next in an episode.

v1 ships only ``RigidSequencePolicy`` (a fixed, config-defined turn order). ``next_actor`` is
synchronous and pure so a future learned/consensus policy (e.g. one that calls a model to pick
the next speaker) is a drop-in subclass with no interface change.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from verl.experimental.multiagent.actor_backend import ActorTurnResult
from verl.experimental.multiagent.config.actor_config import CommunicationPolicyConfig

__all__ = ["CommunicationPolicy", "EpisodeState", "RigidSequencePolicy", "build_policy"]


@dataclass
class EpisodeState:
    """Everything a communication policy may need to decide the next speaker."""

    turn_index: int = 0
    """Number of turns already taken in this episode (0-indexed next-turn counter)."""
    messages_per_actor: dict[str, list[dict]] = field(default_factory=dict)
    """Each actor's own view of the conversation so far, keyed by actor_id."""
    last_actor_id: Optional[str] = None
    last_result: Optional[ActorTurnResult] = None


class CommunicationPolicy(ABC):
    """Decides who speaks next (or that the episode is over)."""

    def __init__(self, config: CommunicationPolicyConfig):
        self.config = config

    @abstractmethod
    def next_actor(self, state: EpisodeState) -> Optional[str]:
        """Return the actor_id that should take the next turn, or None to end the episode."""
        raise NotImplementedError


class RigidSequencePolicy(CommunicationPolicy):
    """Cycles a fixed, config-defined ``turn_order``.

    Stops at ``max_turns``, or earlier if ``termination_actor_id``'s last turn reports
    ``metrics[termination_metric]`` truthy.
    """

    def next_actor(self, state: EpisodeState) -> Optional[str]:
        if state.turn_index >= self.config.max_turns:
            return None
        if (
            self.config.termination_actor_id is not None
            and state.last_actor_id == self.config.termination_actor_id
            and state.last_result is not None
            and state.last_result.metrics.get(self.config.termination_metric)
        ):
            return None
        return self.config.turn_order[state.turn_index % len(self.config.turn_order)]


def build_policy(config: CommunicationPolicyConfig) -> CommunicationPolicy:
    if config.kind == "rigid_sequence":
        return RigidSequencePolicy(config)
    raise ValueError(f"Unknown communication policy kind: {config.kind!r}")
