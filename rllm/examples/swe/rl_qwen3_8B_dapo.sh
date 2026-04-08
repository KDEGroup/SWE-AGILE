set -x
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RLLM_DIR_DEFAULT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28


max_prompt_length=22000
max_response_length=$((1024 * 3))
# max_response_length=$((1024 * 8))

loss_agg_mode="token-mean"

train_prompt_bsz=16
n_resp_per_prompt=8
train_prompt_mini_bsz=8

sp_size="${SP_SIZE:-8}"
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length)))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length)))
offload=True

compression_reward_weight=0.2
max_compression_rate=0.55
formaterr_penalty=-0.5

MODEL_PATH=/mnt/69_store/lsq/SWE-bench-agent/storage/rl_outputs/sft_step_fillv4_r2e_0_850-filtered-8B-btsz64-4ep-DAPO-cprw0.2-mcr0.55-fmp-0.15-step56
export RAY_DISABLE_DASHBOARD=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export STORAGE_DIR=/mnt/69_store/lsq/SWE-bench-agent/storage
# WANDB_MODE=offline
export n_gpus_per_node=4
if [ "${sp_size}" -gt "${n_gpus_per_node}" ]; then
    sp_size="${n_gpus_per_node}"
fi
RLLM_DIR="$(${PYTHON_BIN} -c "import rllm, os; p=getattr(rllm, '__file__', None); print(os.path.dirname(os.path.dirname(p)) if p else os.path.abspath(list(rllm.__path__)[0]))" 2>/dev/null || echo "${RLLM_DIR_DEFAULT}")"
export PYTHONPATH="${RLLM_DIR}:${PYTHONPATH}"
export LOG_LEVEL=WARNING

# export EXPERIMENT_NAME=sft_step_fillv4_r2e_0_850-filtered-8B-btsz64-4ep-DAPO-cprw${compression_reward_weight}-mcr${max_compression_rate}-fmp${formaterr_penalty}
export EXPERIMENT_NAME=sft_step_fillv4_r2e_0_850-filtered-8B-btsz64-4ep-DAPO-cprw${compression_reward_weight}-mcr${max_compression_rate}-fmp${formaterr_penalty}-step56-continued

WANDB_RESUME=allow WANDB_RUN_ID=k3z9w6rj CUDA_VISIBLE_DEVICES=4,5,6,7 time "${PYTHON_BIN}" -m rllm.trainer.verl.train_agent_ppo \
    data.train_files=${RLLM_DIR}/rllm/data/datasets/R2E_Gym_Subset/train_verl.parquet \
    data.val_files=${RLLM_DIR}/rllm/data/datasets/R2E_Gym_Subset/train_verl.parquet \
    data.train_batch_size=${train_prompt_bsz} \
    data.val_batch_size=16 \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=False \
    data.filter_overlong_prompts_workers=8 \
    data.shuffle=False \
    data.reverse_order=True \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=4 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.temperature=1.0 \
    +sampling_params.repetition_penalty=1.15 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    +critic.enable=False \
    rllm.mask_truncated_samples=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='R2EGym-RL' \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=${n_gpus_per_node} \
    trainer.nnodes=1 \
    trainer.save_freq=4 \
    trainer.test_freq=9999 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=1 \
    trainer.default_local_dir=${STORAGE_DIR}/rl_outputs/${EXPERIMENT_NAME} \
    trainer.resume_mode=auto \
    rllm.env.name=swe \
    rllm.filter_token_mismatch=False \
    rllm.stepwise_advantage.enable=True \
    +rllm.compression_reward_weight=${compression_reward_weight} \
    +rllm.max_compression_rate=${max_compression_rate} \
    +rllm.formaterr_penalty=${formaterr_penalty} \
    rllm.rejection_sample.enable=True \
    +rllm.env.env_args.scaffold=continuous_reasoning_window \
    +rllm.agent.agent_args.scaffold=continuous_reasoning_window \
    +rllm.agent.agent_args.use_tool_calling=False \
    +rllm.agent.engine_args.max_workers=16 \
    +rllm.agent.engine_args.n_parallel_agents=64 \
    rllm.agent.name=sweagent \
    rllm.agent.max_steps=65 \
    rllm.agent.overlong_filter=False \
    rllm.agent.trajectory_timeout=4200 \
    +rllm.agent.env_timeout=540 \




ray stop

docker rm -f $(docker ps -aq)
