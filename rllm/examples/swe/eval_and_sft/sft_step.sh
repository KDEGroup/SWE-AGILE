# if self.config.model.get("lora_rank", 0) > 0: self.model = get_peft_model(self.model, LoraConfig(**lora_config))

### stepwise  continuous_reasoning_window


## 8B

# Data have been processed in 'last random [1, 4] reasoning' format 
export WANDB_MODE=offline
export EXPERIMENT_NAME=sft_step_fillv4_r2e_0_850-filtered-8B-btsz128-4ep
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
    data.train_batch_size=128 \
    data.micro_batch_size_per_gpu=8 \
    data.max_length=27000 \
    data.truncation=right \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.train_files=/mnt/69_store/tmp/SWE-bench-agent/storage/sft_step_fillv4_1987ex_r2e_0_850_226ex_filtered.parquet \
    data.val_files=default \
    data.rllm.tokenize_and_mask_method=stepwise \
    +data.rllm.reasoning_mode=all \
    trainer.default_local_dir=/mnt/69_store/tmp/SWE-bench-agent/storage/sft_outputs/${EXPERIMENT_NAME} \
    trainer.save_freq=382 \
    trainer.test_freq=300 \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name=SWE-SFT \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    optim.lr=1e-5

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /mnt/69_store/lsq/SWE-bench-agent/storage/sft_outputs/sft_step_fillv4_r2e_0_850-filtered-14B-btsz128-lr1e-5-4ep/global_step_760 \
    --target_dir /mnt/69_store/lsq/SWE-bench-agent/storage/sft_outputs/sft_step_fillv4_r2e_0_850-filtered-14B-btsz128-lr1e-5-4ep/global_step_760/huggingface


# try use_dynamic_bsz + sequence_parallel in verl sft_trainer.py
# TODO: 速度比原来还慢。。。或许修改verl的FSDPSFTTrainer比适配verl的SFTTrainer更好
RLLM_DIR=$(python3 -c "import rllm; import os; print(os.path.dirname(os.path.dirname(rllm.__file__)))")
export WANDB_MODE=offline
export btsz=256
export epoch=4
export lr=1e-5
export EXPERIMENT_NAME=sft_step_fillv4_r2e_0_850-filtered-8B-btsz${btsz}-lr${lr}-${epoch}ep
HYDRA_FULL_ERROR=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    train_swe_sft_engine.py \
    model.path=/mnt/82_store/huggingface_cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 \
    model.trust_remote_code=true \
    model.enable_gradient_checkpointing=true \
    model.use_remove_padding=true \
    engine.ulysses_sequence_parallel_size=2 \
    data.train_batch_size=${epoch} \
    data.micro_batch_size_per_gpu=256 \
    data.max_token_len_per_gpu=27000 \
    data.max_length=27000 \
    data.truncation=right \
    data.use_dynamic_bsz=true \
    data.train_files=/mnt/69_store/lsq/SWE-bench-agent/storage/sft_step_fillv4_1987ex_r2e_0_850_226ex_filtered.parquet \
    data.val_files=null \
    data.rllm.tokenize_and_mask_method=stepwise \
    data.rllm.reasoning_mode=all \
    data.custom_cls.path=${RLLM_DIR}/rllm/trainer/verl/sft_dataset.py \
    trainer.default_local_dir=/mnt/69_store/lsq/SWE-bench-agent/storage/sft_outputs/${EXPERIMENT_NAME} \
    trainer.total_epochs=${epoch} \
    trainer.save_freq=382 \
    trainer.test_freq=300 \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name=SWE-SFT \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    optim.lr=${lr}


# Data is full traj. Expand_trajectory_to_steps uses fixed last_n_reasoning in RLLMSFTDataset

export WANDB_MODE=offline
export last_n_reasoning=3
export project_name=sft_step_last_${last_n_reasoning}_r_fillv4_1987ex_r2e_107ex-8B-2ep
HYDRA_FULL_ERROR=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=8 \
    train_swe_sft.py \
    model.partial_pretrain=/mnt/82_store/huggingface_cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 \
    model.trust_remote_code=true \
    model.enable_gradient_checkpointing=true \
    trainer.total_epochs=2 \
    use_remove_padding=true \
    ulysses_sequence_parallel_size=8 \
    data.train_batch_size=32 \
    data.micro_batch_size_per_gpu=8 \
    data.max_length=27000 \
    data.truncation=right \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.val_files=default \
    data.rllm.tokenize_and_mask_method=stepwise \
    +data.rllm.reasoning_mode=last_n \
    +data.rllm.last_n_reasoning=${last_n_reasoning} \
    +data.rllm.expand_trajectory_to_steps=true \
    trainer.default_local_dir=/mnt/69_store/tmp/SWE-bench-agent/storage/sft_outputs/${project_name} \
    trainer.save_freq=800 \
    trainer.test_freq=200 \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name=${project_name} \
    optim.lr=1e-5

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /mnt/69_store/tmp/SWE-bench-agent/storage/sft_outputs/sft_last_3_r-randn14-8B-btsz16-lr1e-5-rllmv021/global_step_2874 \
    --target_dir /mnt/69_store/tmp/SWE-bench-agent/storage/sft_outputs/sft_last_3_r-randn14-8B-btsz16-lr1e-5-rllmv021/global_step_2874/huggingface




# 14B
RLLM_DIR=$(python3 -c "import rllm; import os; print(os.path.dirname(os.path.dirname(rllm.__file__)))")
export WANDB_MODE=offline
export btsz=128
export epoch=4
export lr=1e-5
export EXPERIMENT_NAME=sft_step_fillv4_r2e_0_850-filtered-14B-btsz${btsz}-lr${lr}-${epoch}ep
export WANDB_MODE=offline
HYDRA_FULL_ERROR=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=8 \
    train_swe_sft.py \
    model.partial_pretrain=/mnt/82_store/huggingface_cache/hub/models--Qwen--Qwen3-14B/snapshots/40c069824f4251a91eefaf281ebe4c544efd3e18 \
    model.trust_remote_code=true \
    model.enable_gradient_checkpointing=true \
    trainer.total_epochs=${epoch} \
    use_remove_padding=true \
    ulysses_sequence_parallel_size=8 \
    data.train_batch_size=${btsz} \
    data.micro_batch_size_per_gpu=8 \
    data.max_length=26000 \
    data.truncation=right \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.train_files=/mnt/69_store/lsq/SWE-bench-agent/storage/sft_step_fillv4_1987ex_r2e_0_850_226ex_filtered.parquet \
    data.val_files=default \
    data.rllm.tokenize_and_mask_method=stepwise \
    +data.rllm.reasoning_mode=all \
    trainer.default_local_dir=/mnt/69_store/lsq/SWE-bench-agent/storage/sft_outputs/${EXPERIMENT_NAME} \
    trainer.save_freq=191 \
    trainer.test_freq=191 \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name=SWE-SFT \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    optim.lr=${lr}


# 8B 4k data
RLLM_DIR=$(python3 -c "import rllm; import os; print(os.path.dirname(os.path.dirname(rllm.__file__)))")
export WANDB_MODE=offline
export btsz=64
export epoch=3
export lr=1e-5
export EXPERIMENT_NAME=sft_step_4k-8B-btsz${btsz}-lr${lr}-${epoch}ep
export WANDB_MODE=offline
HYDRA_FULL_ERROR=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=8 \
    train_swe_sft.py \
    model.partial_pretrain=/mnt/82_store/huggingface_cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 \
    model.trust_remote_code=true \
    model.enable_gradient_checkpointing=true \
    trainer.total_epochs=${epoch} \
    use_remove_padding=true \
    ulysses_sequence_parallel_size=8 \
    data.train_batch_size=${btsz} \
    data.micro_batch_size_per_gpu=8 \
    data.max_length=27000 \
    data.truncation=right \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.train_files=/mnt/69_store/lsq/SWE-bench-agent/storage/about1987swedev-1ksmith-1000rft-ex.parquet \
    data.val_files=default \
    data.rllm.tokenize_and_mask_method=stepwise \
    +data.rllm.reasoning_mode=all \
    trainer.default_local_dir=/mnt/69_store/lsq/SWE-bench-agent/storage/sft_outputs/${EXPERIMENT_NAME} \
    trainer.save_freq=226 \
    trainer.test_freq=226 \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name=SWE-SFT \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    optim.lr=${lr}


python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /data/lsq/SWE-bench-agent/storage/rl_outputs/sft_step_fillv4_r2e_0_850-filtered-8B-btsz64-4ep-DAPO-cprw0.2-mcr0.55-fmp-0.5-step56-continued/global_step_18/actor \
    --target_dir /data/lsq/SWE-bench-agent/storage/rl_outputs/sft_step_fillv4_r2e_0_850-filtered-8B-btsz64-4ep-DAPO-cprw0.2-mcr0.55-fmp-0.5-step56-continued/global_step_18/actor/huggingface

    