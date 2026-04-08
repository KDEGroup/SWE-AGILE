# swe dev data + r2egym scaffold

unset last_n_reasoning
export WANDB_MODE=offline
export EXPERIMENT_NAME=my-swedev-8B
HYDRA_FULL_ERROR=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=8 \
    train_swe_sft.py \
    model.partial_pretrain=/mnt/82_store/huggingface_cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 \
    model.trust_remote_code=true \
    model.enable_gradient_checkpointing=true \
    trainer.total_epochs=4 \
    use_remove_padding=true \
    ulysses_sequence_parallel_size=8 \
    data.train_batch_size=32 \
    data.micro_batch_size_per_gpu=8 \
    data.max_length=28000 \
    data.truncation=right \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.train_files=/mnt/69_store/tmp/SWE-bench-agent/storage/zai-org/SWE-Dev-train/SWE-Dev-2276-trajectories_normal_2027_trajectory_none.parquet \
    data.val_files=default \
    data.rllm.tokenize_and_mask_method=cumulative \
    +data.rllm.reasoning_mode=none \
    trainer.default_local_dir=/mnt/69_store/tmp/SWE-bench-agent/storage/sft_outputs/${EXPERIMENT_NAME} \
    trainer.save_freq=70 \
    trainer.test_freq=32 \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name=SWE-SFT \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    optim.lr=1e-5

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /mnt/69_store/tmp/SWE-bench-agent/storage/sft_outputs/my-swedev-8B/global_step_252 \
    --target_dir /mnt/69_store/tmp/SWE-bench-agent/storage/sft_outputs/my-swedev-8B/global_step_252/huggingface

    