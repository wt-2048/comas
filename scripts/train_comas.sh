#!/bin/bash
set -x
ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265 --include-dashboard=true

ROOT_DIR=$(pwd)
DATE=$(date +%m%d)
TIME=$(date +%H%M%S)
EXPERIMENT="${DATE}-${TIME}-train-comas"

mkdir -p "${ROOT_DIR}/saves/${EXPERIMENT}"
PROMPT_DATA="json@${ROOT_DIR}/datasets/blended"
SAVE_PATH="${ROOT_DIR}/saves/${EXPERIMENT}/save"
CKPT_PATH="${ROOT_DIR}/saves/${EXPERIMENT}/ckpt"
LOG_PATH="${ROOT_DIR}/saves/${EXPERIMENT}/run.log"
PRETRAIN="/root/siton-data-WWDisk/wt/Qwen2.5-7B-Instruct"

WANDB_MODE="offline"
WANDB_KEY="input_your_wandb_key"
WANDB_PATH="${ROOT_DIR}/saves/${EXPERIMENT}"

PROMPT_MAX_LEN=28672
GENERATE_MAX_LEN=4096

export PYTORCH_NVML_BASED_CUDA_CHECK=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=WARN

ENV_JSON=$(cat <<EOF
{
  "working_dir": "${ROOT_DIR}",
  "excludes": [".git/", "datasets/", "saves/"],
  "env_vars": {"no_proxy": "localhost,127.0.0.1"}
}
EOF
)

ray job submit --address="http://localhost:8265" \
    --runtime-env-json="${ENV_JSON}" \
    -- python -m marti.cli.commands.train --config-name "comas" \
    workflow_version=old \
    parallel_loading=True \
    default_agent.is_reasoning_model=False \
    default_agent.ref_num_nodes=1 \
    default_agent.ref_num_gpus_per_node=2 \
    default_agent.critic_num_nodes=1 \
    default_agent.critic_num_gpus_per_node=2 \
    default_agent.actor_num_nodes=1 \
    default_agent.actor_num_gpus_per_node=2 \
    default_agent.vllm_num_engines=2 \
    default_agent.vllm_tensor_parallel_size=1 \
    default_agent.vllm_sync_backend="nccl" \
    default_agent.colocate_all_models=True \
    default_agent.vllm_enable_sleep=True \
    default_agent.deepspeed_enable_sleep=True \
    default_agent.vllm_gpu_memory_utilization=0.6 \
    default_agent.pretrain="${PRETRAIN}" \
    default_agent.save_path="${SAVE_PATH}" \
    default_agent.micro_train_batch_size=2 \
    default_agent.train_batch_size=64 \
    default_agent.num_episodes=1 \
    default_agent.save_steps=10 \
    default_agent.eval_steps=5 \
    default_agent.logging_steps=1 \
    default_agent.max_samples=2000 \
    default_agent.micro_rollout_batch_size=2 \
    default_agent.rollout_batch_size=64 \
    default_agent.training_mode="rl" \
    default_agent.n_samples_per_prompt=4 \
    default_agent.max_epochs=1 \
    default_agent.prompt_max_len=${PROMPT_MAX_LEN} \
    default_agent.generate_max_len=${GENERATE_MAX_LEN} \
    default_agent.advantage_estimator="reinforce" \
    default_agent.temperature=1.0 \
    default_agent.eval_temperature=0.7 \
    default_agent.lambd=1.0 \
    default_agent.gamma=1.0 \
    default_agent.zero_stage=3 \
    default_agent.bf16=True \
    default_agent.actor_learning_rate=1e-6 \
    default_agent.critic_learning_rate=9e-6 \
    default_agent.init_kl_coef=0.00 \
    default_agent.use_kl_loss=True \
    default_agent.max_ckpt_num=1 \
    default_agent.normalize_reward=False \
    default_agent.adam_offload=True \
    default_agent.gradient_checkpointing=True \
    default_agent.ckpt_path="${CKPT_PATH}" \
    workflow_args.num_rounds=2 \
    workflow_args.num_references=2 \
    agents.0.agent-0.pretrain="${PRETRAIN}" \
    agents.1.agent-1.pretrain="${PRETRAIN}" \
    agents.2.agent-2.pretrain="${PRETRAIN}" \
    agents.3.agent-3.pretrain="${PRETRAIN}" \
    mask_truncated_completions=True \
    shared_agents=False \
    packing_samples=True \
    prompt_data="${PROMPT_DATA}" \
    input_key="prompt" \
    label_key="answer" \
    add_prompt_suffix=null \
    extra_eval_dir="${ROOT_DIR}/datasets/separated" \
    extra_eval_tasks=["math","coding","science"] \
    use_wandb=False \
    wandb_mode="${WANDB_MODE}" \
    wandb_key= \
    wandb_path="${WANDB_PATH}" \
    wandb_project="comas" \
    wandb_run_name="${EXPERIMENT}" 2>&1 | tee "${LOG_PATH}"

echo "Model Training Finished. Shutting down..."
