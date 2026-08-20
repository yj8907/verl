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

from omegaconf import OmegaConf

from verl.experimental.multiagent.config.actor_config import MultiAgentFleetConfig

MAIN_ORACLE_CFG = {
    "actors": {
        "main": {
            "actor_id": "main",
            "system_prompt": "solve the problem",
            "backend": "trainable_verl",
            "model_ref": "main",
        },
        "oracle": {
            "actor_id": "oracle",
            "system_prompt": "give a one-line hint",
            "backend": "external_api",
            "model_ref": "oracle",
            "external_api": {"provider": "openai", "model": "gpt-4o-mini"},
        },
    },
    "policy": {"kind": "rigid_sequence", "turn_order": ["main", "oracle", "main"], "max_turns": 3},
}


class TestMultiAgentFleetConfigParsing(unittest.TestCase):
    def test_parses_main_and_external_api_oracle(self):
        fleet = MultiAgentFleetConfig.from_omegaconf(OmegaConf.create(MAIN_ORACLE_CFG))
        assert fleet.main_actor_id == "main"
        assert fleet.trainable_actor_ids() == ["main"]
        assert fleet.actors["oracle"].trainable is False
        assert fleet.actors["oracle"].external_api.model == "gpt-4o-mini"
        assert fleet.policy.turn_order == ["main", "oracle", "main"]

    def test_actor_rollout_ref_stays_a_dictconfig_not_a_plain_dict(self):
        cfg = OmegaConf.create(MAIN_ORACLE_CFG)
        cfg.actors.main.actor_rollout_ref = {"model": {"path": "Qwen/Qwen2.5-0.5B-Instruct"}}
        fleet = MultiAgentFleetConfig.from_omegaconf(cfg)
        # Must support attribute access (actor.actor_rollout_ref.model.path), matching how
        # config.actor_rollout_ref is used everywhere else in the trainer -- a plain dict would break it.
        assert fleet.actors["main"].actor_rollout_ref.model.path == "Qwen/Qwen2.5-0.5B-Instruct"

    def test_empty_actors_is_a_noop(self):
        fleet = MultiAgentFleetConfig.from_omegaconf(OmegaConf.create({}))
        assert fleet.actors == {}

    def test_requires_exactly_one_main_actor(self):
        cfg = OmegaConf.create(MAIN_ORACLE_CFG)
        cfg.actors.oracle.model_ref = "main"
        with self.assertRaises(ValueError):
            MultiAgentFleetConfig.from_omegaconf(cfg)

    def test_main_actor_must_be_trainable(self):
        cfg = OmegaConf.create(MAIN_ORACLE_CFG)
        cfg.actors.main.backend = "frozen_verl"
        cfg.actors.main.actor_rollout_ref = {"model": {"path": "Qwen/Qwen2.5-0.5B-Instruct"}}
        with self.assertRaises(ValueError):
            MultiAgentFleetConfig.from_omegaconf(cfg)

    def test_no_main_actor_raises(self):
        cfg = OmegaConf.create(MAIN_ORACLE_CFG)
        cfg.actors.main.model_ref = "not_main"
        cfg.actors.main.actor_rollout_ref = {"model": {"path": "x"}}
        cfg.actors.main.n_gpus_per_node = 1
        cfg.actors.main.nnodes = 1
        with self.assertRaises(ValueError):
            MultiAgentFleetConfig.from_omegaconf(cfg)

    def test_external_api_actor_missing_config_raises(self):
        cfg = OmegaConf.create(MAIN_ORACLE_CFG)
        cfg.actors.oracle.external_api = None
        with self.assertRaises(ValueError):
            MultiAgentFleetConfig.from_omegaconf(cfg)

    def test_main_actor_cannot_be_external_api(self):
        cfg = OmegaConf.create(MAIN_ORACLE_CFG)
        cfg.actors.main.backend = "external_api"
        cfg.actors.main.external_api = {"provider": "openai", "model": "gpt-4o-mini"}
        with self.assertRaises(ValueError):
            MultiAgentFleetConfig.from_omegaconf(cfg)

    def test_non_main_verl_actor_requires_own_actor_rollout_ref(self):
        cfg = OmegaConf.create(MAIN_ORACLE_CFG)
        cfg.actors.oracle.backend = "frozen_verl"
        cfg.actors.oracle.n_gpus_per_node = 1
        cfg.actors.oracle.nnodes = 1
        # actor_rollout_ref intentionally left unset.
        with self.assertRaises(ValueError):
            MultiAgentFleetConfig.from_omegaconf(cfg)

    def test_non_main_verl_actor_requires_positive_resource_pool(self):
        cfg = OmegaConf.create(MAIN_ORACLE_CFG)
        cfg.actors.oracle.backend = "frozen_verl"
        cfg.actors.oracle.actor_rollout_ref = {"model": {"path": "Qwen/Qwen2.5-0.5B-Instruct"}}
        # n_gpus_per_node/nnodes left at their zero defaults.
        with self.assertRaises(ValueError):
            MultiAgentFleetConfig.from_omegaconf(cfg)

    def test_policy_turn_order_referencing_unknown_actor_raises(self):
        cfg = OmegaConf.create(MAIN_ORACLE_CFG)
        cfg.policy.turn_order = ["main", "ghost"]
        with self.assertRaises(ValueError):
            MultiAgentFleetConfig.from_omegaconf(cfg)

    def test_actors_key_must_match_actor_id_field(self):
        cfg = OmegaConf.create(MAIN_ORACLE_CFG)
        cfg.actors.main.actor_id = "someone_else"
        with self.assertRaises(ValueError):
            MultiAgentFleetConfig.from_omegaconf(cfg)


if __name__ == "__main__":
    unittest.main()
