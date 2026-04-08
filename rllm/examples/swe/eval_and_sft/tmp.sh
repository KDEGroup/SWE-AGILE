# 1. 获取当前 Python 环境下 nvidia 包的安装路径
ENV_NVIDIA_DIR=$(python3 -c 'import os, nvidia; print(os.path.dirname(nvidia.__file__))' 2>/dev/null)

if [ -n "$ENV_NVIDIA_DIR" ]; then
    echo "Found local NVIDIA packages at: $ENV_NVIDIA_DIR"

    # 2. 找到该目录下所有的 'lib' 文件夹，并用冒号拼接
    # 通常包括: nvidia/cuda_runtime/lib, nvidia/cublas/lib, nvidia/cudnn/lib 等
    LOCAL_CUDA_LIBS=$(find "$ENV_NVIDIA_DIR" -name "lib" -type d | paste -sd ":" -)

    # 3. 将这些路径添加到 LD_LIBRARY_PATH 的最前面
    export LD_LIBRARY_PATH="$LOCAL_CUDA_LIBS:$LD_LIBRARY_PATH"
    
    # 4. 这是一个额外的保险措施，防止某些库查找 CUDA_HOME
    # 注意：pip 安装的 cuda 通常不在标准结构中，但有时设置这个能抑制某些 warning
    export CUDA_HOME="$ENV_NVIDIA_DIR/cuda_runtime"
    
    echo "Updated LD_LIBRARY_PATH to prioritize venv CUDA libraries."
else
    echo "Warning: Could not find 'nvidia' package in python environment. using system CUDA."
fi

export MAX_CONTEXT_LEN=65536
export TENSOR_PARALLEL_SIZE=4
export MODEL=/mnt/69_store/lsq/SWE-bench-agent/storage/sft_outputs/sft_step_fillv4_r2e_0_850-filtered-14B-btsz128-lr1e-5-4ep/global_step_760/huggingface
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_DEBUG=INFO CUDA_VISIBLE_DEVICES=0,1,2,6 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve ${MODEL} \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --max-model-len $MAX_CONTEXT_LEN \
    --hf-overrides '{"max_position_embeddings": '$MAX_CONTEXT_LEN'}' \
    --enable_prefix_caching \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --gpu_memory_utilization 0.85 \
    --port 8083