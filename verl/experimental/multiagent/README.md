# Multi-agent fleet PPO training

Elevates verl's v1 synchronous PPO trainer from a single actor to a **fleet of actors with rigid,
config-defined roles** (e.g. a trainable "main" solver plus a frozen "oracle"), interacting through
a shared conversation and trained with PPO/GRPO. See `CLAUDE.md` in this directory for the original
goal statement.

## Concepts

- **Actor**: one role in the fleet (`verl/experimental/multiagent/config/actor_config.py:ActorConfig`),
  identified by `actor_id`, with its own `system_prompt` and a `backend`:
  - `trainable_verl`: a verl-managed rollout server whose weights PPO updates. Exactly one actor
    must have `model_ref: main` (reuses the trainer's primary `actor_rollout_ref` config and
    worker group); every other trainable actor is configured via its own `actor_rollout_ref`
    sub-config plus `n_gpus_per_node`/`nnodes`, and actors that set the *same* `model_ref` are the
    same model -- they share one dedicated resource pool, worker group, and `LLMServerManager`
    (config must match exactly across the group) instead of each getting a dedicated copy.
  - `frozen_verl`: a verl-managed, inference-only rollout server (never trained) -- for a stronger
    local model you can host but don't want/need to fine-tune.
  - `external_api`: an external OpenAI/Anthropic-compatible endpoint (never trained) -- for a
    model you can't host at all. Bounded concurrency + timeout + retry
    (`ActorConfig.external_api`) keep a slow/rate-limited provider from stalling the rest of
    rollout (see `actor_backend.py:ExternalAPIActorBackend`).
- **Communication policy** (`policy.py`): decides who speaks next each turn. v1 ships
  `RigidSequencePolicy`: a fixed, config-defined `turn_order`, cycled up to `max_turns`, with
  optional early termination when a configured actor's turn reports
  `metrics["done"]`. `CommunicationPolicy` is an interface specifically so a future
  learned/consensus-driven policy is a drop-in subclass -- see "Deferred" below.
- **Episode**: one multi-turn conversation, driven by `agent_loop.py:MultiAgentLoop`
  (`AgentLoopBase` subclass, registered as agent loop `"multi_agent"`). Returns one
  `AgentLoopOutput` per **trainable** actor -- non-trainable (frozen/external) actors' text still
  shapes what trainable actors are conditioned on, but they get no trainable row of their own.
- **Reward**: v1 default is a single shared episodic reward, broadcast to every trainable actor's
  output in the episode. This is not new code -- it's the existing
  `AgentLoopWorkerTQ._agent_loop_postprocess` behavior (broadcasts the last output's score to
  every earlier output in a `list[AgentLoopOutput]`); `MultiAgentLoop` just orders its outputs so
  the actor who took the episode's last turn is last in the list.

## Configuring a new fleet

Add a file under `config/actor_fleet/<name>.yaml` (see `math_teacher_student.yaml` for a worked
example), then point `config/multiagent_ppo_trainer.yaml`'s
`defaults: - actor_fleet@multiagent: <name>` at it (or override on the CLI:
`+actor_fleet@multiagent=<name>`). Each actor needs:

```yaml
actors:
  main:                        # actor_id
    actor_id: main
    system_prompt: "..."
    backend: trainable_verl
    model_ref: main             # exactly one actor must be model_ref: main

  weak_student:                 # a second trainable actor on its own model/pool
    actor_id: weak_student
    system_prompt: "..."
    backend: trainable_verl
    model_ref: weak_student
    actor_rollout_ref: {...}    # full actor_rollout_ref-shaped sub-config for this actor's model
    n_gpus_per_node: 2
    nnodes: 1

policy:
  kind: rigid_sequence
  turn_order: [main, weak_student]
  max_turns: 4
```

Two actor_ids can share one model by giving them the same `model_ref` (e.g. two "critic" roles
both played by `main`'s own weights, or two personas on one auxiliary model) -- they must then set
identical `actor_rollout_ref`/`n_gpus_per_node`/`nnodes` (enforced by
`MultiAgentFleetConfig._check_shared_model_groups`), and the trainer collapses them onto one
resource pool, worker group, and `LLMServerManager`:

```yaml
  reviewer_a:
    actor_id: reviewer_a
    backend: trainable_verl
    model_ref: reviewer          # same model_ref as reviewer_b -> shared pool/worker/server
    actor_rollout_ref: {...}
    n_gpus_per_node: 2
    nnodes: 1

  reviewer_b:
    actor_id: reviewer_b
    backend: trainable_verl
    model_ref: reviewer          # must match reviewer_a's actor_rollout_ref/n_gpus_per_node/nnodes
    actor_rollout_ref: {...}
    n_gpus_per_node: 2
    nnodes: 1
```
```

## Architecture notes (for anyone extending this)

- **No shared-file edits.** Everything here subclasses/extends `verl/trainer/ppo/v1/trainer_base.py`,
  `verl/experimental/agent_loop/agent_loop.py`, and `verl/trainer/ppo/v1/agent_loop_tq.py` rather
  than modifying them, and reaches Ray via the existing `run_ppo(config, task_runner_class=...)`
  extension point (`main_multiagent.py`, the same pattern
  `verl/experimental/one_step_off_policy/main_ppo.py` uses).
- **Resource pools**: `trainer.py:MultiAgentPPOTrainer._init_resource_pool_mgr` gives each
  *unique* `model_ref` among non-main trainable actors its own dedicated Ray resource pool
  (multiple actor_ids sharing a `model_ref` collapse onto one pool/worker group/`LLMServerManager`
  in `_setup_actor_groups`), generalizing the existing `teacher_pool` precedent for on-policy
  distillation. Frozen actors still get one dedicated pool per actor_id via `FrozenActorManager`.
  Actor ids/model_refs are used directly as string keys into `ResourcePoolManager`'s mapping --
  `Role` (`verl/trainer/ppo/utils.py`) can't be subclassed (Python disallows extending an `Enum`
  that already has members).
- **Per-group training**: the six single-actor pipeline methods (`_balance_batch`,
  `_compute_old_log_prob`, `_compute_ref_log_prob`, `_compute_advantage`, `_update_actor`) are
  reused *unmodified* per model group: filter the step's TransferQueue rows to that group (by the
  `model_ref` field `MultiAgentLoop` stamps into `AgentLoopOutput.extra_fields`, which combines
  every actor_id sharing that model_ref into one on-policy training batch), temporarily rebind
  `self.actor_rollout_wg`/`self.config`/`self.tokenizer` to the group
  (`trainer.py:_bind_actor_group`), call the inherited method, restore. This must stay a
  sequential loop across groups -- see the docstring on `_bind_actor_group` for why.
- **Token-level correctness**: `agent_loop.py:ActorTurnRenderer` incrementally builds each
  trainable actor's own view of the episode (own turns masked in for training, everyone else's
  masked out), using the same turn-separator technique `ToolAgentLoop` uses for tool
  observations. This is the highest-risk piece of the design: the sequence stored for training
  must be byte-identical to what generation was actually conditioned on, or recomputed
  `old_log_probs` stop corresponding to what was actually sampled and silently corrupt the PPO
  importance ratio. Verified against a real tokenizer/chat-template in
  `tests/experimental/multiagent/test_actor_turn_renderer_on_cpu.py`.
- **Frozen verl-hosted actors** reuse `verl/experimental/teacher_loop/teacher_model.py`'s
  `TeacherModelManager` (inference-only rollout replicas, no training engine) rather than the
  on-policy-distillation `MultiTeacherModelManager`/`DistillationConfig` path: a frozen fleet
  actor must generate full responses, while the distillation path's
  `validate_and_prepare_for_distillation` collapses response length to 1 for a single-token
  logprob forward pass -- semantics that don't apply here (see `frozen_actor_manager.py`).

## Try it

```bash
python3 examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k
OPENAI_API_KEY=... bash verl/experimental/multiagent/shell/run_math_teacher_student.sh
```

## Deferred (explicitly out of scope for v1)

- **Learned/consensus communication policy** -- only the `CommunicationPolicy` interface exists;
  `RigidSequencePolicy` is the only implementation.
- **Per-actor reward** -- v1 is shared-episodic-reward-only, by construction of
  `_agent_loop_postprocess`'s broadcast. The `actor_id`/`model_ref` tagging already in place on
  every trainable turn is what a future per-actor reward function would key off.
- **Prefix/KV-cache sharing** across actors that share a model.
- **Per-group critic** -- critic/GAE stays main-only; the target recipes use GRPO with a shared
  episodic reward, so this wasn't needed to validate the architecture.
- **Mid-episode correctness checks** -- `RigidSequencePolicy`'s early-termination hook
  (`termination_actor_id`/`termination_metric`) exists, but no shipped `ActorBackend` sets
  `ActorTurnResult.metrics["done"]`; only the post-episode reward function scores correctness.
