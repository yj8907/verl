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
"""Hydra entrypoint for multi-agent fleet PPO training.

Mirrors ``verl.trainer.main_ppo.TaskRunnerV1`` (config -> init -> init_agent_loop_manager -> fit),
with ``MultiAgentTaskRunner`` swapping in ``MultiAgentPPOTrainer``/``MultiAgentLoopManagerTQ`` and
wiring the fleet's non-main agent LLM clients through. Uses the unmodified
``verl.trainer.main_ppo.run_ppo(config, task_runner_class=...)`` extension point (the same one
``verl/experimental/one_step_off_policy/main_ppo.py`` already uses), so no shared file is touched.
"""

import logging
import os
from pprint import pprint

import hydra
import ray
from omegaconf import OmegaConf

from verl.experimental.multiagent.agent_loop import MultiAgentLoopManagerTQ
from verl.experimental.multiagent.trainer import MultiAgentPPOTrainer
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device
from verl.utils.logging_utils import configure_verl_logging

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


@ray.remote
class MultiAgentTaskRunner:
    """TaskRunner for multi-agent fleet PPO training."""

    def __init__(self):
        self.config = None
        self.trainer = None
        self.agent_loop_manager = None

    def init_agent_loop_manager(self):
        self.agent_loop_manager = MultiAgentLoopManagerTQ.create(
            config=self.config,
            llm_client=self.trainer.get_llm_client(),
            teacher_client=self.trainer.get_teacher_client(),
            reward_loop_worker_handles=self.trainer.get_reward_handles(),
            agent_llm_clients=self.trainer.get_agent_llm_clients(),
        )

    def run(self, config):
        """Run multi-agent fleet PPO training."""
        configure_verl_logging()

        import transfer_queue as tq

        config.transfer_queue.enable = True
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)
        self.config = config

        tq.init(config.transfer_queue)
        succeeded = False
        try:
            self.trainer = MultiAgentPPOTrainer(config=config)
            self.trainer.init()
            self.init_agent_loop_manager()
            self.trainer.fit(self.agent_loop_manager)
            succeeded = True
        finally:
            try:
                tracking = getattr(self.trainer, "logger", None)
                if tracking is not None:
                    tracking.finish(exit_code=0 if succeeded else 1)
            finally:
                tq.close()


@hydra.main(config_path="config", config_name="multiagent_ppo_trainer", version_base=None)
def main(config):
    """Main entry point for multi-agent fleet PPO training."""
    from verl.trainer.main_ppo import run_ppo

    auto_set_device(config)

    validate_config(
        config=config,
        use_reference_policy=need_reference_policy(config),
        use_critic=need_critic(config),
    )

    run_ppo(config, task_runner_class=MultiAgentTaskRunner)


if __name__ == "__main__":
    main()
