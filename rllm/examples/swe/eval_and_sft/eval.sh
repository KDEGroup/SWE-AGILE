# /mnt/82_store/huggingface_cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
# /mnt/82_store/huggingface_cache/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137
# /mnt/82_store/huggingface_cache/hub/models--Qwen--Qwen3-Coder-30B-A3B-Instruct/snapshots/573fa3901e5799703b1e60825b0ec024a4c0f1d3
# /mnt/82_store/huggingface_cache/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554
# /mnt/69_store/huggingface_cache/Qwen/Qwen3-14B
# /mnt/69_store/huggingface_cache/Qwen/Qwen3-235B-A22B-Instruct-2507
# /mnt/69_store/huggingface_cache/Qwen/Qwen3-30B-A3B-Thinking-2507
# /mnt/69_store/huggingface_cache/Qwen/Qwen3-Coder-30B-A3B-Instruct

# /mnt/69_store/tmp/SWE-bench-agent/storage/sft_outputs/SWE-Dev-fillback_processed_normal_digest_only_2267_trajs-4B/global_step_423/huggingface
# /mnt/69_store/tmp/SWE-bench-agent/storage/sft_outputs/SWE-Dev-fillback_processed_normal_2267_trajs_all_think-8B/global_step_140/huggingface
# /mnt/69_store/tmp/SWE-bench-agent/storage/sft_outputs/SWE-Dev-fillback_processed_normal_digest_only_2267_trajs-8B/global_step_70/huggingface

# SWE_Bench_Verified
# multi_swe_bench_flash
# multi_swe_bench_flash_subset_32
# SWE_Bench_Verified_subset_100
# R2E_Gym_Subset_0_2000_SFT

#   +rollout_engine_args.base_url=https://openrouter.ai/api/v1 \
#   +rollout_engine_args.api_key=xxx \
#   +rollout_engine_args.model=google/gemini-2.5-flash-preview-09-2025 \
#   +for_close_source_api.tokenizer=/mnt/82_store/huggingface_cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218


### collect r2egym traj

export MAX_CONTEXT_LEN=32768
export TENSOR_PARALLEL_SIZE=8
export MODEL=/mnt/69_store/huggingface_cache/Qwen/Qwen3-235B-A22B-Thinking-2507
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve ${MODEL} \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --max-model-len $MAX_CONTEXT_LEN \
    --hf-overrides '{"max_position_embeddings": '$MAX_CONTEXT_LEN'}' \
    --enable_prefix_caching \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --gpu-memory-utilization 0.95 \
    --enforce-eager

export MAX_CONTEXT_LEN=32768
export TENSOR_PARALLEL_SIZE=8
export MODEL=/mnt/69_store/huggingface_cache/Qwen/Qwen3-235B-A22B-Instruct-2507
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve ${MODEL} \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --max-model-len $MAX_CONTEXT_LEN \
    --hf-overrides '{"max_position_embeddings": '$MAX_CONTEXT_LEN'}' \
    --enable_prefix_caching \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --gpu-memory-utilization 0.95 \
    --enforce-eager


export LOG_LEVEL=WARNING
export STORAGE_DIR=/mnt/69_store/tmp/SWE-bench-agent/storage
time python run_deepswe.py \
  +rllm.env.env_args.scaffold=continuous_reasoning_window \
  +rllm.agent.agent_args.scaffold=continuous_reasoning_window \
  +rllm.agent.agent_args.use_tool_calling=False \
  +rllm.agent.engine_args.max_workers=16 \
  +rllm.agent.engine_args.n_parallel_agents=10 \
  +rllm.run.target=sft \
  +rllm.run.sft_mode=step \
  rllm.agent.max_steps=50 \
  +registry_dataset_name=R2E_Gym_Subset \
  +registry_dataset_split=train \
  data.max_prompt_length=32768 \
  data.max_response_length=5120 \
  +sampling_params.temperature=0.1 \
  +sampling_params.repetition_penalty=1.0 \
  +rollout_engine_args.base_url=http://localhost:8000/v1 \
  +rollout_engine_args.api_key=sk-xxx \
  +rollout_engine_args.model=${MODEL} \
  +for_close_source_api.tokenizer=${MODEL}



# open router
# qwen/qwen3-235b-a22b-2507
# qwen/qwen3-235b-a22b
# deepseek/deepseek-v3.2
# z-ai/glm-4.7
# minimax/minimax-m2
export MODEL=minimax/minimax-m2
export LOG_LEVEL=DEBUG
export STORAGE_DIR=/mnt/69_store/tmp/SWE-bench-agent/storage
time python run_deepswe.py \
  +rllm.env.env_args.scaffold=continuous_reasoning_window \
  +rllm.agent.agent_args.scaffold=continuous_reasoning_window \
  +rllm.agent.agent_args.use_tool_calling=False \
  +rllm.agent.engine_args.max_workers=16 \
  +rllm.agent.engine_args.n_parallel_agents=1 \
  +rllm.run.target=sft \
  +rllm.run.sft_mode=step \
  rllm.agent.max_steps=50 \
  +registry_dataset_name=R2E_Gym_Subset \
  +registry_dataset_split=train \
  data.max_prompt_length=32768 \
  data.max_response_length=6144 \
  +sampling_params.temperature=0.1 \
  +sampling_params.repetition_penalty=1.0 \
  +rollout_engine_args.base_url=https://openrouter.ai/api/v1 \
  +rollout_engine_args.api_key=sk-xxx \
  +rollout_engine_args.model=${MODEL} \
  +for_close_source_api.tokenizer=/mnt/69_store/huggingface_cache/Qwen/Qwen3-235B-A22B-Instruct-2507





### eval base model
export MAX_CONTEXT_LEN=65536
export TENSOR_PARALLEL_SIZE=4
export MODEL=/mnt/82_store/huggingface_cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
CUDA_VISIBLE_DEVICES=4,5,6,7 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve ${MODEL} \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --max-model-len $MAX_CONTEXT_LEN \
    --hf-overrides '{"max_position_embeddings": '$MAX_CONTEXT_LEN'}' \
    --enable_prefix_caching \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --port 8083
    
export LOG_LEVEL=WARNING
export STORAGE_DIR=/data/tmp/SWE-bench-agent/storage
time python run_deepswe.py \
  +rllm.env.env_args.scaffold=normal \
  +rllm.agent.agent_args.scaffold=normal \
  +rllm.agent.agent_args.use_tool_calling=False \
  +rllm.agent.engine_args.max_workers=16 \
  +rllm.agent.engine_args.n_parallel_agents=32 \
  +rllm.run.target=eval \
  +rllm.check_format=False \
  +rllm.check_repeating=False \
  +rllm.last_n_reasoning=-999 \
  rllm.agent.max_steps=80 \
  +registry_dataset_name=SWE_Bench_Verified \
  +registry_dataset_split=test \
  data.max_prompt_length=9220 \
  data.max_response_length=56316 \
  +sampling_params.temperature=0.0 \
  +sampling_params.repetition_penalty=1.0 \
  +rollout_engine_args.base_url=http://localhost:8083/v1 \
  +rollout_engine_args.api_key=sk-xxx \
  +rollout_engine_args.model=${MODEL} \
  +for_close_source_api.tokenizer=${MODEL}



### eval normal-all-CoT

export MAX_CONTEXT_LEN=65536
export TENSOR_PARALLEL_SIZE=4
export MODEL=/mnt/69_store/tmp/SWE-bench-agent/storage/sft_outputs/sft_all_CoT-fillv4_1987ex_r2e_0_850_filtered-8B-4ep/global_step_272/huggingface
CUDA_VISIBLE_DEVICES=4,5,6,7 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve ${MODEL} \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --max-model-len $MAX_CONTEXT_LEN \
    --hf-overrides '{"max_position_embeddings": '$MAX_CONTEXT_LEN'}' \
    --enable_prefix_caching \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --port 8082
    
export LOG_LEVEL=WARNING
export STORAGE_DIR=/data/tmp/SWE-bench-agent/storage
time python run_deepswe.py \
  +rllm.env.env_args.scaffold=normal \
  +rllm.agent.agent_args.scaffold=normal \
  +rllm.agent.agent_args.use_tool_calling=False \
  +rllm.agent.engine_args.max_workers=16 \
  +rllm.agent.engine_args.n_parallel_agents=16 \
  +rllm.run.target=eval \
  +rllm.check_format=False \
  +rllm.check_repeating=True \
  rllm.agent.max_steps=50 \
  +registry_dataset_name=SWE_Bench_Verified \
  +registry_dataset_split=test \
  data.max_prompt_length=8192 \
  data.max_response_length=32768 \
  +sampling_params.temperature=0.0 \
  +sampling_params.repetition_penalty=1.15 \
  +rollout_engine_args.base_url=http://localhost:8082/v1 \
  +rollout_engine_args.api_key=sk-xxx \
  +rollout_engine_args.model=${MODEL} \
  +for_close_source_api.tokenizer=${MODEL}


### eval normal-digest_only
export MAX_CONTEXT_LEN=65536
export TENSOR_PARALLEL_SIZE=4
export MODEL=/mnt/69_store/tmp/SWE-bench-agent/storage/sft_outputs/my-swedev-Qwen2.5-Coder-7B/global_step_252/huggingface
CUDA_VISIBLE_DEVICES=4,5,6,7 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve ${MODEL} \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --max-model-len $MAX_CONTEXT_LEN \
    --hf-overrides '{"max_position_embeddings": '$MAX_CONTEXT_LEN'}' \
    --enable_prefix_caching \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --port 8081
    
export LOG_LEVEL=WARNING
export STORAGE_DIR=/data/tmp/SWE-bench-agent/storage
time python run_deepswe.py \
  +rllm.env.env_args.scaffold=normal \
  +rllm.agent.agent_args.scaffold=normal \
  +rllm.agent.agent_args.use_tool_calling=False \
  +rllm.agent.engine_args.max_workers=16 \
  +rllm.agent.engine_args.n_parallel_agents=32 \
  +rllm.run.target=eval \
  +rllm.check_format=False \
  +rllm.check_repeating=True \
  +rllm.last_n_reasoning=-999 \
  rllm.agent.max_steps=50 \
  +registry_dataset_name=SWE_Bench_Verified \
  +registry_dataset_split=test \
  data.max_prompt_length=8192 \
  data.max_response_length=32768 \
  +sampling_params.temperature=0.0 \
  +sampling_params.repetition_penalty=1.15 \
  +rollout_engine_args.base_url=http://localhost:8081/v1 \
  +rollout_engine_args.api_key=sk-xxx \
  +rollout_engine_args.model=${MODEL} \
  +for_close_source_api.tokenizer=${MODEL}


### eval last_n_reasoning

export MAX_CONTEXT_LEN=65536
export TENSOR_PARALLEL_SIZE=2
export MODEL=/data/lsq/SWE-bench-agent/storage/sft_outputs/sft_step_fillv4_r2e_0_850-filtered-14B-btsz128-lr1e-5-4ep/global_step_760/huggingface
CUDA_VISIBLE_DEVICES=6,7 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve ${MODEL} \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --max-model-len $MAX_CONTEXT_LEN \
    --hf-overrides '{"max_position_embeddings": '$MAX_CONTEXT_LEN'}' \
    --enable_prefix_caching \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --gpu_memory_utilization 0.8 \
    --port 8084

export LOG_LEVEL=WARNING
export STORAGE_DIR=/data/lsq/SWE-bench-agent/storage
time python run_deepswe.py \
  +rllm.env.env_args.scaffold=continuous_reasoning_window \
  +rllm.agent.agent_args.scaffold=continuous_reasoning_window \
  +rllm.agent.agent_args.use_tool_calling=False \
  +rllm.agent.engine_args.max_workers=16 \
  +rllm.agent.engine_args.n_parallel_agents=16 \
  +rllm.agent.env_timeout=450 \
  +rllm.agent.trajectory_timeout=12600 \
  +rllm.run.target=eval \
  +rllm.check_format=False \
  +rllm.check_repeating=False \
  +rllm.last_n_reasoning=1 \
  rllm.agent.max_steps=100 \
  +registry_dataset_name=SWE_Bench_Verified \
  +registry_dataset_split=test \
  data.max_prompt_length=9220 \
  data.max_response_length=56316 \
  +sampling_params.temperature=0.0 \
  +sampling_params.repetition_penalty=1.0 \
  +rollout_engine_args.base_url=http://localhost:8084/v1 \
  +rollout_engine_args.api_key=sk-xxx \
  +rollout_engine_args.model=${MODEL} \
  +for_close_source_api.tokenizer=${MODEL}

docker rm -f $(docker ps -aq)


  +registry_dataset_name=SWE_Bench_Verified_subset_100 \
  +registry_dataset_split=train \

  +registry_dataset_name=R2E_Gym_Subset \
  +registry_dataset_split=train \



  +registry_dataset_name=SWE_Bench_Verified \
  +registry_dataset_split=test \