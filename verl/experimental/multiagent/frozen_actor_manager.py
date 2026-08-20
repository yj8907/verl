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
"""One inference-only, verl-managed rollout server pool per "frozen_verl" fleet actor.

Thin re-parameterization of ``verl.experimental.teacher_loop.teacher_model.TeacherModelManager``
(reused unmodified) off ``ActorConfig`` instead of ``DistillationTeacherModelConfig``. We don't go
through ``MultiTeacherModelManager``/``DistillationConfig.__post_init__`` because those apply
on-policy-distillation-specific rewriting (a frozen teacher does a single-token logprob forward
pass, so ``validate_and_prepare_for_distillation`` collapses its response_length to 1) that does
not apply to a frozen fleet actor, which must generate full responses like any other actor.
"""

import logging
import os

from omegaconf import DictConfig

from verl.experimental.multiagent.config.actor_config import MultiAgentFleetConfig
from verl.experimental.teacher_loop.teacher_model import TeacherModelManager
from verl.single_controller.ray import ResourcePoolManager
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import DistillationConfig, DistillationTeacherModelConfig, RolloutConfig
from verl.workers.rollout.llm_server import LLMServerClient

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class FrozenActorManager:
    """Builds and owns one ``TeacherModelManager`` per "frozen_verl" actor in the fleet."""

    def __init__(
        self,
        config: DictConfig,
        fleet_config: MultiAgentFleetConfig,
        resource_pool_manager: ResourcePoolManager,
    ):
        self.config = config
        self._managers: dict[str, TeacherModelManager] = {}

        for actor_id, actor in fleet_config.actors.items():
            if actor.backend != "frozen_verl":
                continue

            inference_config: RolloutConfig = omega_conf_to_dataclass(
                actor.actor_rollout_ref.rollout, dataclass_type=RolloutConfig
            )
            per_replica_world_size = (
                inference_config.tensor_model_parallel_size
                * inference_config.data_parallel_size
                * inference_config.pipeline_model_parallel_size
            )
            pool_size = actor.n_gpus_per_node * actor.nnodes
            if pool_size % per_replica_world_size != 0:
                raise ValueError(
                    f"Frozen actor {actor_id!r}: per_replica_world_size ({per_replica_world_size}) must "
                    f"divide its resource pool size ({actor.n_gpus_per_node=} * {actor.nnodes=} = {pool_size})."
                )

            teacher_model_config = DistillationTeacherModelConfig(
                key=actor_id,
                model_path=actor.actor_rollout_ref.model.path,
                inference=inference_config,
                num_replicas=pool_size // per_replica_world_size,
            )
            # enabled=False short-circuits DistillationConfig.__post_init__'s teacher-specific
            # validation/resolution; only n_gpus_per_node is read by TeacherModelManager itself.
            distillation_config = DistillationConfig(
                enabled=False, n_gpus_per_node=actor.n_gpus_per_node, nnodes=actor.nnodes
            )
            resource_pool = resource_pool_manager.get_resource_pool(actor_id)
            self._managers[actor_id] = TeacherModelManager(
                distillation_config=distillation_config,
                teacher_model_config=teacher_model_config,
                resource_pool=resource_pool,
            )
            logger.info(f"FrozenActorManager: initialized rollout servers for actor {actor_id!r}")

    def actor_ids(self) -> list[str]:
        return list(self._managers)

    def get_client(self, actor_id: str) -> LLMServerClient:
        manager = self._managers[actor_id]
        return LLMServerClient(config=self.config, load_balancer_handle=manager.load_balancer_handle)
