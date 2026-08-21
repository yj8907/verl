#!/usr/bin/env bash
set -xeuo pipefail

# Multi-agent fleet PPO: a trainable "main" solver plus a frozen "oracle" (external OpenAI-
# compatible API) that hints when main's attempt looks wrong. See
# verl/experimental/multiagent/config/agent_fleet/math_teacher_student.yaml for the fleet
# definition and verl/experimental/multiagent/README.md for the full writeup.
#
# Prepare data first, e.g.:
#   python3 examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k
#
# Requires OPENAI_API_KEY (or ANTHROPIC_API_KEY, if you switch the oracle's provider in the
# agent_fleet config) in the environment for the oracle agent.

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY for the oracle agent}"

project_name='multiagent'
exp_name='math-teacher-student-qwen2.5-0.5b'

adv_estimator=grpo

train_prompt_bsz=64
n_resp_per_prompt=4
train_prompt_mini_bsz=16

max_prompt_length=1024
max_response_length=1024

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}

RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-0.5B-Instruct"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${HOME}/data/gsm8k/train.parquet"}
TEST_FILE=${TEST_FILE:-"${HOME}/data/gsm8k/test.parquet"}

python3 -m verl.experimental.multiagent.main_multiagent \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=False \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size=${train_prompt_bsz} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.prompt_length=${max_prompt_length} \
    actor_rollout_ref.rollout.response_length=${max_response_length} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    critic.enable=False \
    trainer.logger=['console'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.val_before_train=True \
    trainer.test_freq=5 \
    trainer.save_freq=-1 \
    trainer.total_epochs=5 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}"
