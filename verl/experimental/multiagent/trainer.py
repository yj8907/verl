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
- One dedicated Ray resource pool per unique shared-model group among non-"main" verl-managed
  trainable actors (actors with the same ``engine_ref`` are the same model and share one pool,
  worker group, and ``LLMServerManager``; distinct engine_refs each get their own), generalizing
  the existing ``teacher_pool`` precedent in ``trainer_base.py``. Frozen actors still get one
  dedicated pool per actor_id via ``FrozenActorManager``.
- Six training-pipeline methods (``_balance_batch``, ``_compute_old_log_prob``,
  ``_compute_ref_log_prob``, ``_compute_advantage``, ``_update_actor``) are reused UNMODIFIED
  from ``PPOTrainer``/``PPOTrainerSync`` per model group: filter the step's batch to that
  group's rows (combining every actor_id sharing the engine_ref, since they're on-policy for the
  same weights), temporarily rebind ``self.actor_rollout_wg``/``self.config``/``self.tokenizer``
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
    """Everything needed to train and serve one shared-model group of non-"main" trainable fleet
    actors: actors that share a ``engine_ref`` are the same model and share one resource pool,
    worker group, ``LLMServerManager``, and ``CheckpointEngineManager`` rather than each getting
    a dedicated copy."""

    engine_ref: str
    actor_ids: list[str]
    """All actor_ids in the fleet that share this model (``ActorConfig.engine_ref``)."""
    config: DictConfig
    """Shallow clone of the trainer config with ``actor_rollout_ref`` swapped to this group's own."""
    actor_wg: RayWorkerGroup
    llm_server_manager: LLMServerManager
    checkpoint_manager: CheckpointEngineManager
    tokenizer: PreTrainedTokenizerBase


class MultiAgentPPOTrainer(PPOTrainerSync):
    """Synchronous multi-agent PPO trainer: colocated fleet, one dedicated pool per unique
    shared-model group among non-main verl-managed actors."""

    #: Sentinel ``engine_ref`` reused from ``ActorConfig.is_main`` -- the trainer's own primary
    #: actor, never a key in ``self.engine_groups``.
    _MAIN_engine_ref = "main"

    def _get_pool_name(self, engine_ref):

        pool_name = f"{engine_ref}_pool"

        return pool_name
    
    def _init_resource_pool_mgr(self):
        config = self.config
        self.role_worker_mapping = {}
        # role resource pool mapping
        self.mapping = {}

        # --- replicate PPOTrainer._init_resource_pool_mgr's main/critic/reward/teacher setup ---
        # (can't call super() here: it also constructs self.resource_pool_manager as its last
        # step, before this method gets a chance to add the fleet's per-actor pools. So we re-create source pool manager)
        super()._init_resource_pool_mgr()

        # --- fleet: one dedicated pool per unique shared-model group among non-"main" trainable
        # actors (actors with the same engine_ref collapse onto one pool/worker group), and one
        # pool per frozen_verl actor. ---
        self.fleet_config: MultiAgentFleetConfig = MultiAgentFleetConfig.from_omegaconf(config.multiagent)
        seen_engine_refs: set[str] = set()
        for actor_id, actor in self.fleet_config.actors.items():
            if actor.is_main or actor.backend == "external_api":
                continue
            if actor.backend == "trainable_verl":
                # each engine group own a separate resource pool (GPU)
                pool_name = self._get_pool_name(actor.engine_ref)
                self.mapping[actor_id] = pool_name
                # skip if already seen
                if actor.engine_ref in seen_engine_refs:
                    continue  # pool/worker role already registered for this shared model
                seen_engine_refs.add(actor.engine_ref)

                self.mapping[actor.engine_ref] = pool_name
                self.resource_pool_manager.resource_pool_spec[pool_name] = [actor.n_gpus_per_node] * actor.nnodes
                self.role_worker_mapping[actor.engine_ref] = ray.remote(ActorRolloutRefWorker)
            else:
                # frozen_verl actors are built directly against their own pool in
                # _setup_actor_groups via FrozenActorManager, bypassing the
                # RayWorkerGroup/training-engine machinery entirely.
                pool_name = self._get_pool_name(actor_id) 
                self.resource_pool_manager.resource_pool_spec[pool_name] = [actor.n_gpus_per_node] * actor.nnodes
                self.mapping[actor_id] = pool_name

        # reuse resource_pool_manager initialized from parent class.
        self.resource_pool_manager = ResourcePoolManager(resource_pool_spec=self.resource_pool_manager, mapping=self.mapping)

    def _setup(self):
        super()._setup()
        self._setup_actor_groups()
        # PPOTrainer._setup calls self._load_checkpoint() internally, near its own end -- before
        # _setup_actor_groups (above) has run, so self.engine_groups doesn't exist at that point
        # and fleet checkpoints can't be restored there. Load them explicitly, now that it does.
        self._load_fleet_checkpoint()

    def _setup_actor_groups(self):
        """Build one resource pool / worker group / ``LLMServerManager`` per unique ``engine_ref``
        among the fleet's non-main trainable actors. Actors that share a engine_ref are the same
        model (``MultiAgentFleetConfig._check_shared_engine_groups`` enforces they agree on model
        config) and must share serving/training resources rather than each getting a dedicated
        copy -- see ``self.mapping``/``self.role_worker_mapping`` built per engine_ref in
        ``_init_resource_pool_mgr``.
        """
        self.engine_groups: dict[str, ActorGroupHandle] = {}
        self.actor_groups: dict[str, ActorGroupHandle] = {}
        wg_kwargs = {"device_name": self.config.trainer.device}

        # extract unique engine_ref from fleet_config.actors. per engine_ref, create a list of actors
        engine_actors: dict[str, list[ActorConfig]] = {}
        for actor in self.fleet_config.actors.values():
            if actor.is_main or actor.backend != "trainable_verl":
                continue
            engine_actors.setdefault(actor.engine_ref, []).append(actor)

        for engine_ref, actors in engine_actors.items():
            representative = actors[0]
            actor_config = self._actor_specific_config(representative)

            # assign 
            resource_pool = self.resource_pool_manager.get_resource_pool(engine_ref)

            ray_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[engine_ref],
                config=actor_config.actor_rollout_ref,
                role="actor_rollout",
            )
            worker_dict_cls = create_colocated_worker_cls(class_dict={engine_ref: ray_cls})
            wg_dict = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, **wg_kwargs)
            actor_wg = wg_dict.spawn(prefix_set={engine_ref})[engine_ref]
            actor_wg.init_model()

            # only one LLM server per engine_ref as each actor only differs by system prompt and conversation history
            llm_server_manager: LLMServerManager = LLMServerManager.create(
                config=actor_config, worker_group=actor_wg, rollout_resource_pool=resource_pool
            )
            checkpoint_engine_config = omega_conf_to_dataclass(actor_config.actor_rollout_ref.rollout.checkpoint_engine)
            checkpoint_engine_config.backend = "naive"
            checkpoint_manager = CheckpointEngineManager(
                config=checkpoint_engine_config, actor_wg=actor_wg, replicas=llm_server_manager.get_replicas()
            )
            checkpoint_manager.sleep_replicas()

            group = ActorGroupHandle(
                engine_ref=engine_ref,
                actor_ids=[actor.actor_id for actor in actors],
                config=actor_config,
                actor_wg=actor_wg,
                llm_server_manager=llm_server_manager,
                checkpoint_manager=checkpoint_manager,
                tokenizer=hf_tokenizer(representative.actor_rollout_ref.model.path),
            )
            self.engine_groups[engine_ref] = group
            for actor in actors:
                self.actor_groups[actor.actor_id] = group
            logger.info(
                f"MultiAgentPPOTrainer: initialized shared model group {engine_ref!r} for actors {group.actor_ids!r}"
            )

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
        for group in self.engine_groups.values():
            group.checkpoint_manager.update_weights(self.global_steps)

    def on_step_end(self):
        super().on_step_end()
        with marked_timer("update_weights_fleet", self.timing_raw, color="red"):
            for group in self.engine_groups.values():
                group.checkpoint_manager.update_weights(self.global_steps)

    def on_sample_end(self):
        super().on_sample_end()
        for group in self.engine_groups.values():
            group.checkpoint_manager.sleep_replicas()

    # ------------------------------ per-group training pipeline ------------------------------

    @contextmanager
    def _bind_actor_group(self, engine_ref: str):
        """Temporarily rebind the singular training attrs the inherited pipeline methods read.

        Must run sequentially, never concurrently across model groups: the six pipeline methods
        dispatch Ray remote calls synchronously (reading ``self.config``/``self.actor_rollout_wg``
        at dispatch time, before returning), so as long as one group's block fully dispatches
        before the next group's rebind happens, there's no race -- but parallelizing this loop
        (e.g. via asyncio.gather) would break that invariant.
        """
        if engine_ref == self._MAIN_engine_ref:
            yield
            return
        group = self.engine_groups[engine_ref]
        prev_wg, prev_config, prev_tokenizer = self.actor_rollout_wg, self.config, self.tokenizer
        self.actor_rollout_wg, self.config, self.tokenizer = group.actor_wg, group.config, group.tokenizer
        try:
            yield
        finally:
            self.actor_rollout_wg, self.config, self.tokenizer = prev_wg, prev_config, prev_tokenizer

    def _filter_batch_by_engine_ref(self, batch: KVBatchMeta, engine_ref: str) -> KVBatchMeta:
        """Keep only the rows written by trainable actors sharing this model (tagged via
        ``AgentLoopOutput.extra_fields["engine_ref"]`` in ``MultiAgentLoop._build_outputs``),
        mirroring the extra_fields-filtering technique ``ReplayBuffer._dapo_filtered_keys``
        already uses for DAPO metric filtering. Actors that share a engine_ref train together off
        their combined trajectories, since they're on-policy for the same weights."""
        import transfer_queue as tq

        if not batch.keys:
            return KVBatchMeta(partition_id=batch.partition_id, keys=[], tags=[])
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=["extra_fields"])
        extra_fields_list = list(data["extra_fields"])
        keys, tags = [], []
        for key, tag, extra_fields in zip(batch.keys, batch.tags, extra_fields_list, strict=True):
            extra_fields = getattr(extra_fields, "data", extra_fields)
            if isinstance(extra_fields, dict) and extra_fields.get("engine_ref") == engine_ref:
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

        # One training step per unique model: the trainer's own main model, plus every shared
        # fleet engine_ref group (each combining trajectories from every actor_id that shares it).
        for engine_ref in [self._MAIN_engine_ref, *self.engine_groups]:
            with self._bind_actor_group(engine_ref):
                group_batch = self._filter_batch_by_engine_ref(batch, engine_ref)
                if not group_batch.keys:
                    logger.warning(f"No trajectories for model {engine_ref!r} in this step; skipping.")
                    continue

                group_batch = self._balance_batch(group_batch, metrics=metrics)

                with marked_timer(f"old_log_prob/{engine_ref}", timing_raw, color="blue"):
                    group_batch = self._compute_old_log_prob(group_batch, metrics=metrics)

                if self.use_reference_policy and engine_ref == self._MAIN_engine_ref:
                    with marked_timer(f"ref/{engine_ref}", timing_raw, color="olive"):
                        group_batch = self._compute_ref_log_prob(group_batch, metrics=metrics)

                with marked_timer(f"adv/{engine_ref}", timing_raw, color="brown"):
                    group_batch = self._compute_advantage(group_batch, metrics=metrics)

                if self.config.trainer.critic_warmup <= self.global_steps:
                    with marked_timer(f"update_actor/{engine_ref}", timing_raw, color="red"):
                        self._update_actor(group_batch, metrics=metrics)

        return batch

    # ------------------------------ checkpoints ------------------------------

    def _save_checkpoint(self):
        super()._save_checkpoint()
        step_dir = f"global_step_{self.global_steps}"
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, step_dir)
        max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None)
        # Keyed by engine_ref, not actor_id: actors sharing a model share one actor_wg, so saving
        # once per engine_ref (rather than once per actor_id) avoids redundant duplicate writes.
        for engine_ref, group in self.engine_groups.items():
            group.actor_wg.save_checkpoint(
                os.path.join(local_global_step_folder, engine_ref),
                None,
                self.global_steps,
                max_ckpt_to_keep=max_actor_ckpt_to_keep,
            )

    def _load_fleet_checkpoint(self):
        if self.global_steps == 0 or not self.engine_groups:
            return
        # NOTE: assumes the default checkpoint layout (global_step_N under default_local_dir).
        # trainer.resume_from_path pointing at a non-default location needs the same per-model
        # subfolders under that path for fleet checkpoints to resume correctly.
        checkpoint_folder = self.config.trainer.default_local_dir
        if not os.path.isabs(checkpoint_folder):
            checkpoint_folder = os.path.join(os.getcwd(), checkpoint_folder)
        global_step_folder = os.path.join(checkpoint_folder, f"global_step_{self.global_steps}")
        for engine_ref, group in self.engine_groups.items():
            group.actor_wg.load_checkpoint(
                local_path=os.path.join(global_step_folder, engine_ref),
                del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
            )


__all__ = ["ActorGroupHandle", "MultiAgentPPOTrainer"]
