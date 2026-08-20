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

import unittest

from verl.experimental.multiagent.actor_backend import ActorTurnResult
from verl.experimental.multiagent.config.actor_config import CommunicationPolicyConfig
from verl.experimental.multiagent.policy import EpisodeState, RigidSequencePolicy, build_policy


def _drive(policy, max_iters=10, done_after_actor=None, done_after_n_turns=None):
    """Run a policy to completion, marking done=True once `done_after_actor` has spoken
    `done_after_n_turns` times, and return the sequence of actor ids that spoke."""
    state = EpisodeState()
    turns = []
    counts = {}
    for _ in range(max_iters):
        next_actor = policy.next_actor(state)
        if next_actor is None:
            break
        turns.append(next_actor)
        counts[next_actor] = counts.get(next_actor, 0) + 1
        done = next_actor == done_after_actor and counts[next_actor] >= (done_after_n_turns or 0)
        state = EpisodeState(
            turn_index=state.turn_index + 1,
            last_actor_id=next_actor,
            last_result=ActorTurnResult(text="x", trainable=True, metrics={"done": done}),
        )
    return turns


class TestRigidSequencePolicy(unittest.TestCase):
    def test_cycles_turn_order_until_max_turns(self):
        policy = RigidSequencePolicy(CommunicationPolicyConfig(turn_order=["a", "b"], max_turns=5))
        assert _drive(policy) == ["a", "b", "a", "b", "a"]

    def test_single_actor_turn_order_repeats(self):
        policy = RigidSequencePolicy(CommunicationPolicyConfig(turn_order=["main"], max_turns=3))
        assert _drive(policy) == ["main", "main", "main"]

    def test_terminates_early_on_configured_actor_done_signal(self):
        policy = RigidSequencePolicy(
            CommunicationPolicyConfig(
                turn_order=["main", "oracle"], max_turns=10, termination_actor_id="main", termination_metric="done"
            )
        )
        turns = _drive(policy, done_after_actor="main", done_after_n_turns=2)
        assert turns == ["main", "oracle", "main"]

    def test_termination_only_checked_on_the_configured_actor(self):
        # oracle reports done, but termination_actor_id is "main" -- must not stop early.
        policy = RigidSequencePolicy(
            CommunicationPolicyConfig(
                turn_order=["main", "oracle"], max_turns=4, termination_actor_id="main", termination_metric="done"
            )
        )
        turns = _drive(policy, done_after_actor="oracle", done_after_n_turns=1)
        assert turns == ["main", "oracle", "main", "oracle"]

    def test_no_termination_actor_configured_runs_to_max_turns(self):
        policy = RigidSequencePolicy(CommunicationPolicyConfig(turn_order=["a"], max_turns=2))
        assert _drive(policy, done_after_actor="a", done_after_n_turns=1) == ["a", "a"]


class TestBuildPolicy(unittest.TestCase):
    def test_rigid_sequence_kind_builds_rigid_sequence_policy(self):
        policy = build_policy(CommunicationPolicyConfig(kind="rigid_sequence", turn_order=["a"], max_turns=1))
        assert isinstance(policy, RigidSequencePolicy)

    def test_unknown_kind_raises(self):
        with self.assertRaises(ValueError):
            build_policy(CommunicationPolicyConfig(kind="learned_consensus", turn_order=["a"], max_turns=1))


if __name__ == "__main__":
    unittest.main()
