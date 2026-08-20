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
"""``MultiAgentPPOTrainer``: elevates the v1 sync PPO trainer from one actor to a fleet.

Design (see verl/experimental/multiagent/CLAUDE.md and the accompanying plan):
- One dedicated Ray resource pool per non-"main" verl-managed actor (trainable or frozen),
  generalizing the existing ``teacher_pool`` precedent in ``trainer_base.py``.
- Six training-pipeline methods (``_balance_batch``, ``_compute_old_log_prob``,
  ``_compute_ref_log_prob``, ``_compute_advantage``, ``_update_actor``) are reused UNMODIFIED
  from ``PPOTrainer``/``PPOTrainerSync`` per trainable group: filter the step's batch to that
  group's rows, temporarily rebind ``self.actor_rollout_wg``/``self.config``/``self.tokenizer``
  to the group, call the inherited method, restore. This avoids touching the shared base class.
- v1 scope: critic/GAE and reference-policy KL stay main-only (the target recipes use GRPO with
  a shared episodic reward, so a per-group critic isn't needed); on-policy distillation (the
  ``distillation``/teacher-logprob KD subsystem) is a separate, unrelated mechanism and is not
  wired to fleet actors -- cross-actor teaching happens entirely through the conversation itself.
"""

import copy
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass

import ray
from omegaconf import DictConfig
from transfer_queue import KVBatchMeta
from transformers import PreTrainedTokenizerBase

from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.multiagent.config.actor_config import ActorConfig, MultiAgentFleetConfig
from verl.experimental.multiagent.frozen_actor_manager import FrozenActorManager
from verl.single_controller.ray import (
    RayClassWithInitArgs,
    RayWorkerGroup,
    ResourcePoolManager,
    create_colocated_worker_cls,
)
from verl.trainer.distillation import is_distillation_enabled
from verl.trainer.ppo.utils import Role, need_critic, need_reference_policy
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
from verl.utils import hf_tokenizer
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.workers.engine_workers import ActorRolloutRefWorker
from verl.workers.rollout.llm_server import LLMServerClient, LLMServerManager

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


@dataclass
class ActorGroupHandle:
    """Everything needed to train and serve one non-"main" trainable fleet actor."""

    actor_id: str
    config: DictConfig
    """Shallow clone of the trainer config with ``actor_rollout_ref`` swapped to this actor's own."""
    actor_wg: RayWorkerGroup
    llm_server_manager: LLMServerManager
    checkpoint_manager: CheckpointEngineManager
    tokenizer: PreTrainedTokenizerBase


class MultiAgentPPOTrainer(PPOTrainerSync):
    """Synchronous multi-agent PPO trainer: colocated fleet, one dedicated pool per non-main
    verl-managed actor."""

    def _init_resource_pool_mgr(self):
        config = self.config
        self.role_worker_mapping = {}
        self.mapping = {}

        # --- replicate PPOTrainer._init_resource_pool_mgr's main/critic/reward/teacher setup ---
        # (can't call super() here: it also constructs self.resource_pool_manager as its last
        # step, before this method gets a chance to add the fleet's per-actor pools.)
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        main_role = Role.ActorRolloutRef if need_reference_policy(config) and not ref_in_actor else Role.ActorRollout
        self.role_worker_mapping[main_role] = ray.remote(ActorRolloutRefWorker)
        self.mapping[main_role] = "global_pool"

        if need_critic(config):
            from verl.workers.engine_workers import TrainingWorker

            self.role_worker_mapping[Role.Critic] = ray.remote(TrainingWorker)
            self.mapping[Role.Critic] = "global_pool"

        resource_pool_spec = {
            "global_pool": [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }

        if config.reward.reward_model.enable_resource_pool:
            if config.reward.reward_model.n_gpus_per_node <= 0 or config.reward.reward_model.nnodes <= 0:
                raise ValueError("config.reward.reward_model.n_gpus_per_node/nnodes must be > 0")
            resource_pool_spec["reward_pool"] = [config.reward.reward_model.n_gpus_per_node] * (
                config.reward.reward_model.nnodes
            )
            self.mapping[Role.RewardModel] = "reward_pool"
        else:
            config.reward.reward_model.nnodes = config.trainer.nnodes
            config.reward.reward_model.n_gpus_per_node = config.trainer.n_gpus_per_node
            self.mapping[Role.RewardModel] = "global_pool"

        distillation_config = config.get("distillation")
        if is_distillation_enabled(distillation_config):
            if distillation_config.n_gpus_per_node <= 0 or distillation_config.nnodes <= 0:
                raise ValueError("config.distillation.n_gpus_per_node/nnodes must be > 0")
            resource_pool_spec["teacher_pool"] = [distillation_config.n_gpus_per_node] * distillation_config.nnodes
            self.mapping[Role.TeacherModel] = "teacher_pool"

        # --- fleet: one dedicated pool per non-"main" verl-managed actor ---
        self.fleet_config: MultiAgentFleetConfig = MultiAgentFleetConfig.from_omegaconf(config.multiagent)
        for actor_id, actor in self.fleet_config.actors.items():
            if actor.is_main or actor.backend == "external_api":
                continue
            pool_name = f"{actor_id}_pool"
            resource_pool_spec[pool_name] = [actor.n_gpus_per_node] * actor.nnodes
            self.mapping[actor_id] = pool_name
            if actor.backend == "trainable_verl":
                self.role_worker_mapping[actor_id] = ray.remote(ActorRolloutRefWorker)
            # frozen_verl actors are built directly against their pool in _setup_actor_groups via
            # FrozenActorManager, bypassing the RayWorkerGroup/training-engine machinery entirely.

        self.resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)

    def _setup(self):
        super()._setup()
        self._setup_actor_groups()
        # PPOTrainer._setup calls self._load_checkpoint() internally, near its own end -- before
        # _setup_actor_groups (above) has run, so self.actor_groups doesn't exist at that point
        # and fleet checkpoints can't be restored there. Load them explicitly, now that it does.
        self._load_fleet_checkpoint()

    def _setup_actor_groups(self):
        self.actor_groups: dict[str, ActorGroupHandle] = {}
        wg_kwargs = {"device_name": self.config.trainer.device}

        for actor_id, actor in self.fleet_config.actors.items():
            if actor.is_main or actor.backend != "trainable_verl":
                continue

            actor_config = self._actor_specific_config(actor)
            resource_pool = self.resource_pool_manager.get_resource_pool(actor_id)
            ray_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_id],
                config=actor_config.actor_rollout_ref,
                role="actor_rollout",
            )
            worker_dict_cls = create_colocated_worker_cls(class_dict={actor_id: ray_cls})
            wg_dict = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, **wg_kwargs)
            actor_wg = wg_dict.spawn(prefix_set={actor_id})[actor_id]
            actor_wg.init_model()

            llm_server_manager: LLMServerManager = LLMServerManager.create(
                config=actor_config, worker_group=actor_wg, rollout_resource_pool=resource_pool
            )
            checkpoint_engine_config = omega_conf_to_dataclass(actor_config.actor_rollout_ref.rollout.checkpoint_engine)
            checkpoint_engine_config.backend = "naive"
            checkpoint_manager = CheckpointEngineManager(
                config=checkpoint_engine_config, actor_wg=actor_wg, replicas=llm_server_manager.get_replicas()
            )
            checkpoint_manager.sleep_replicas()

            self.actor_groups[actor_id] = ActorGroupHandle(
                actor_id=actor_id,
                config=actor_config,
                actor_wg=actor_wg,
                llm_server_manager=llm_server_manager,
                checkpoint_manager=checkpoint_manager,
                tokenizer=hf_tokenizer(actor.actor_rollout_ref.model.path),
            )
            logger.info(f"MultiAgentPPOTrainer: initialized trainable actor group {actor_id!r}")

        self.frozen_actor_manager: FrozenActorManager | None = None
        if any(a.backend == "frozen_verl" for a in self.fleet_config.actors.values()):
            self.frozen_actor_manager = FrozenActorManager(
                config=self.config, fleet_config=self.fleet_config, resource_pool_manager=self.resource_pool_manager
            )

    def _actor_specific_config(self, actor: ActorConfig) -> DictConfig:
        """A clone of the trainer config with ``actor_rollout_ref`` swapped to this actor's own.

        ``LLMServerManager``/``CheckpointEngineManager`` and the six per-group training methods
        all read ``config.actor_rollout_ref.*`` off whatever config object they're given, so each
        actor group needs its own config object, not the shared ``self.config``.
        """
        cfg = copy.deepcopy(self.config)
        cfg.actor_rollout_ref = actor.actor_rollout_ref
        # rollout.n drives GRPO group size (compute_advantage_for_multi_trajectories groups by
        # uid across `n` sessions); every fleet actor must agree with main's, since every trainable
        # actor gets exactly one row per session.
        cfg.actor_rollout_ref.rollout.n = self.config.actor_rollout_ref.rollout.n
        return cfg

    def get_actor_llm_clients(self) -> dict[str, LLMServerClient]:
        """LLM server clients for every non-"main" fleet actor (trainable_verl and frozen_verl).

        External-API actors need no client here: ``MultiAgentLoopWorkerTQ`` builds their backend
        directly from ``config.multiagent`` inside the worker process.
        """
        clients = {actor_id: group.llm_server_manager.get_client() for actor_id, group in self.actor_groups.items()}
        if self.frozen_actor_manager is not None:
            for actor_id in self.frozen_actor_manager.actor_ids():
                clients[actor_id] = self.frozen_actor_manager.get_client(actor_id)
        return clients

    # ------------------------------ weight sync across the fleet ------------------------------

    def on_init_end(self):
        super().on_init_end()
        for group in self.actor_groups.values():
            group.checkpoint_manager.update_weights(self.global_steps)

    def on_step_end(self):
        super().on_step_end()
        with marked_timer("update_weights_fleet", self.timing_raw, color="red"):
            for group in self.actor_groups.values():
                group.checkpoint_manager.update_weights(self.global_steps)

    def on_sample_end(self):
        super().on_sample_end()
        for group in self.actor_groups.values():
            group.checkpoint_manager.sleep_replicas()

    # ------------------------------ per-group training pipeline ------------------------------

    @contextmanager
    def _bind_actor_group(self, actor_id: str):
        """Temporarily rebind the singular training attrs the inherited pipeline methods read.

        Must run sequentially, never concurrently across actor ids: the six pipeline methods
        dispatch Ray remote calls synchronously (reading ``self.config``/``self.actor_rollout_wg``
        at dispatch time, before returning), so as long as one group's block fully dispatches
        before the next group's rebind happens, there's no race -- but parallelizing this loop
        (e.g. via asyncio.gather) would break that invariant.
        """
        if actor_id == self.fleet_config.main_actor_id:
            yield
            return
        group = self.actor_groups[actor_id]
        prev_wg, prev_config, prev_tokenizer = self.actor_rollout_wg, self.config, self.tokenizer
        self.actor_rollout_wg, self.config, self.tokenizer = group.actor_wg, group.config, group.tokenizer
        try:
            yield
        finally:
            self.actor_rollout_wg, self.config, self.tokenizer = prev_wg, prev_config, prev_tokenizer

    def _filter_batch_by_actor(self, batch: KVBatchMeta, actor_id: str) -> KVBatchMeta:
        """Keep only the rows one trainable actor wrote (tagged via ``AgentLoopOutput.extra_fields``
        in ``MultiAgentLoop._build_outputs``), mirroring the extra_fields-filtering technique
        ``ReplayBuffer._dapo_filtered_keys`` already uses for DAPO metric filtering."""
        import transfer_queue as tq

        if not batch.keys:
            return KVBatchMeta(partition_id=batch.partition_id, keys=[], tags=[])
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=["extra_fields"])
        extra_fields_list = list(data["extra_fields"])
        keys, tags = [], []
        for key, tag, extra_fields in zip(batch.keys, batch.tags, extra_fields_list, strict=True):
            extra_fields = getattr(extra_fields, "data", extra_fields)
            if isinstance(extra_fields, dict) and extra_fields.get("model_ref") == actor_id:
                keys.append(key)
                tags.append(tag)
        return KVBatchMeta(partition_id=batch.partition_id, keys=keys, tags=tags)

    def _step_once(self, metrics: dict, timing_raw: dict, sample_batch_size: int) -> KVBatchMeta:
        with marked_timer("gen", timing_raw, color="red"):
            self.on_sample_begin()
            batch, off_policy_metrics = self.replay_buffer.sample(
                global_steps=self.global_steps, partition_id="train", batch_size=sample_batch_size
            )
            metrics.update(off_policy_metrics)
            batch.extra_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
            self.on_sample_end()

        if self.reward_loop_manager.reward_loop_worker_handles is None:
            with marked_timer("reward", timing_raw, color="yellow"):
                batch = self._compute_reward_colocate(batch, metrics=metrics)

        for actor_id in self.fleet_config.trainable_actor_ids():
            with self._bind_actor_group(actor_id):
                group_batch = self._filter_batch_by_actor(batch, actor_id)
                if not group_batch.keys:
                    logger.warning(f"No trajectories for trainable actor {actor_id!r} in this step; skipping.")
                    continue

                group_batch = self._balance_batch(group_batch, metrics=metrics)

                with marked_timer(f"old_log_prob/{actor_id}", timing_raw, color="blue"):
                    group_batch = self._compute_old_log_prob(group_batch, metrics=metrics)

                if self.use_reference_policy and actor_id == self.fleet_config.main_actor_id:
                    with marked_timer(f"ref/{actor_id}", timing_raw, color="olive"):
                        group_batch = self._compute_ref_log_prob(group_batch, metrics=metrics)

                with marked_timer(f"adv/{actor_id}", timing_raw, color="brown"):
                    group_batch = self._compute_advantage(group_batch, metrics=metrics)

                if self.config.trainer.critic_warmup <= self.global_steps:
                    with marked_timer(f"update_actor/{actor_id}", timing_raw, color="red"):
                        self._update_actor(group_batch, metrics=metrics)

        return batch

    # ------------------------------ checkpoints ------------------------------

    def _save_checkpoint(self):
        super()._save_checkpoint()
        step_dir = f"global_step_{self.global_steps}"
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, step_dir)
        max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None)
        for actor_id, group in self.actor_groups.items():
            group.actor_wg.save_checkpoint(
                os.path.join(local_global_step_folder, actor_id),
                None,
                self.global_steps,
                max_ckpt_to_keep=max_actor_ckpt_to_keep,
            )

    def _load_fleet_checkpoint(self):
        if self.global_steps == 0 or not self.actor_groups:
            return
        # NOTE: assumes the default checkpoint layout (global_step_N under default_local_dir).
        # trainer.resume_from_path pointing at a non-default location needs the same per-actor
        # subfolders under that path for fleet checkpoints to resume correctly.
        checkpoint_folder = self.config.trainer.default_local_dir
        if not os.path.isabs(checkpoint_folder):
            checkpoint_folder = os.path.join(os.getcwd(), checkpoint_folder)
        global_step_folder = os.path.join(checkpoint_folder, f"global_step_{self.global_steps}")
        for actor_id, group in self.actor_groups.items():
            group.actor_wg.load_checkpoint(
                local_path=os.path.join(global_step_folder, actor_id),
                del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
            )


__all__ = ["ActorGroupHandle", "MultiAgentPPOTrainer"]
