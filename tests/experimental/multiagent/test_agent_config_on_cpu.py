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

from verl.experimental.multiagent.config.agent_config import MultiAgentFleetConfig

MAIN_ORACLE_CFG = {
    "agents": {
        "main": {
            "agent_id": "main",
            "system_prompt": "solve the problem",
            "backend": "trainable_verl",
            "model_ref": "main",
        },
        "oracle": {
            "agent_id": "oracle",
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
        assert fleet.main_agent_id == "main"
        assert fleet.trainable_agent_ids() == ["main"]
        assert fleet.agents["oracle"].trainable is False
        assert fleet.agents["oracle"].external_api.model == "gpt-4o-mini"
        assert fleet.policy.turn_order == ["main", "oracle", "main"]

    def test_actor_rollout_ref_stays_a_dictconfig_not_a_plain_dict(self):
        cfg = OmegaConf.create(MAIN_ORACLE_CFG)
        cfg.agents.main.actor_rollout_ref = {"model": {"path": "Qwen/Qwen2.5-0.5B-Instruct"}}
        fleet = MultiAgentFleetConfig.from_omegaconf(cfg)
        # Must support attribute access (agent.actor_rollout_ref.model.path), matching how
        # config.actor_rollout_ref is used everywhere else in the trainer -- a plain dict would break it.
        assert fleet.agents["main"].actor_rollout_ref.model.path == "Qwen/Qwen2.5-0.5B-Instruct"

    def test_empty_agents_is_a_noop(self):
        fleet = MultiAgentFleetConfig.from_omegaconf(OmegaConf.create({}))
        assert fleet.agents == {}

    def test_requires_exactly_one_main_agent(self):
        cfg = OmegaConf.create(MAIN_ORACLE_CFG)
        cfg.agents.oracle.model_ref = "main"
        with self.assertRaises(ValueError):
            MultiAgentFleetConfig.from_omegaconf(cfg)

    def test_main_agent_must_be_trainable(self):
        cfg = OmegaConf.create(MAIN_ORACLE_CFG)
        cfg.agents.main.backend = "frozen_verl"
        cfg.agents.main.actor_rollout_ref = {"model": {"path": "Qwen/Qwen2.5-0.5B-Instruct"}}
        with self.assertRaises(ValueError):
            MultiAgentFleetConfig.from_omegaconf(cfg)

    def test_no_main_agent_raises(self):
        cfg = OmegaConf.create(MAIN_ORACLE_CFG)
        cfg.agents.main.model_ref = "not_main"
        cfg.agents.main.actor_rollout_ref = {"model": {"path": "x"}}
        cfg.agents.main.n_gpus_per_node = 1
        cfg.agents.main.nnodes = 1
        with self.assertRaises(ValueError):
            MultiAgentFleetConfig.from_omegaconf(cfg)

    def test_external_api_agent_missing_config_raises(self):
        cfg = OmegaConf.create(MAIN_ORACLE_CFG)
        cfg.agents.oracle.external_api = None
        with self.assertRaises(ValueError):
            MultiAgentFleetConfig.from_omegaconf(cfg)

    def test_main_agent_cannot_be_external_api(self):
        cfg = OmegaConf.create(MAIN_ORACLE_CFG)
        cfg.agents.main.backend = "external_api"
        cfg.agents.main.external_api = {"provider": "openai", "model": "gpt-4o-mini"}
        with self.assertRaises(ValueError):
            MultiAgentFleetConfig.from_omegaconf(cfg)

    def test_non_main_verl_agent_requires_own_actor_rollout_ref(self):
        cfg = OmegaConf.create(MAIN_ORACLE_CFG)
        cfg.agents.oracle.backend = "frozen_verl"
        cfg.agents.oracle.n_gpus_per_node = 1
        cfg.agents.oracle.nnodes = 1
        # actor_rollout_ref intentionally left unset.
        with self.assertRaises(ValueError):
            MultiAgentFleetConfig.from_omegaconf(cfg)

    def test_non_main_verl_agent_requires_positive_resource_pool(self):
        cfg = OmegaConf.create(MAIN_ORACLE_CFG)
        cfg.agents.oracle.backend = "frozen_verl"
        cfg.agents.oracle.actor_rollout_ref = {"model": {"path": "Qwen/Qwen2.5-0.5B-Instruct"}}
        # n_gpus_per_node/nnodes left at their zero defaults.
        with self.assertRaises(ValueError):
            MultiAgentFleetConfig.from_omegaconf(cfg)

    def test_policy_turn_order_referencing_unknown_agent_raises(self):
        cfg = OmegaConf.create(MAIN_ORACLE_CFG)
        cfg.policy.turn_order = ["main", "ghost"]
        with self.assertRaises(ValueError):
            MultiAgentFleetConfig.from_omegaconf(cfg)

    def test_agents_key_must_match_agent_id_field(self):
        cfg = OmegaConf.create(MAIN_ORACLE_CFG)
        cfg.agents.main.agent_id = "someone_else"
        with self.assertRaises(ValueError):
            MultiAgentFleetConfig.from_omegaconf(cfg)


if __name__ == "__main__":
    unittest.main()
