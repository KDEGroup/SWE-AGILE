#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# ================= Core Configuration =================
# 1. Cache and Repository Paths
PREFERRED_CACHE="/mnt/69_store/lsq/uv/uv_cache_global"
WHEEL_STORE="/mnt/69_store/lsq/uv/wheels_store"

# 2. Project Paths
MAIN_REPO_PATH="rllm"
R2E_PATH="R2E-Gym"

TARGET_TORCH_VERSION="2.8.0"
# ======================================================

# --- 0. Environment Initialization ---
mkdir -p "$WHEEL_STORE"
SYSTEM_PYTHON=$(which python)
UV_CMD="$SYSTEM_PYTHON -m uv"
if [ -n "$1" ]; then ENV_NAME="$1"; else ENV_NAME=".venv_sgl_$(hostname)"; fi

# Set cache directory
if [ -d "$PREFERRED_CACHE" ] && [ -w "$PREFERRED_CACHE" ]; then
    export UV_CACHE_DIR="$PREFERRED_CACHE"
    echo "✅ [Cache] Shared cache enabled: $PREFERRED_CACHE"
fi

# Ensure uv exists
if ! $UV_CMD --version &> /dev/null; then
    echo "⬇️  Installing uv..."
    $SYSTEM_PYTHON -m pip install uv
fi


# ==========================================================
# 🆕 2. Create Environment
# ==========================================================
if [ ! -d "$ENV_NAME" ]; then
    echo "🆕 Creating virtual environment: $ENV_NAME ..."
    $UV_CMD venv "$ENV_NAME" --python 3.11
fi
VENV_PYTHON="$(pwd)/$ENV_NAME/bin/python"

$VENV_PYTHON -m ensurepip --default-pip

# ==========================================================
# 🧠 3. torch and Flash-Attention
# ==========================================================

# 1.1 Install specific PyTorch version first
echo "⬇️  [Step 1] Installing PyTorch ${TARGET_TORCH_VERSION}..."
$UV_CMD pip install --python "$VENV_PYTHON" "torch==${TARGET_TORCH_VERSION}" setuptools wheel packaging


echo "🔍 [Step 2] Searching for matching Flash-Attention Wheel..."

# A. Detect current Python version tag (e.g., cp310, cp311)
PY_TAG=$($VENV_PYTHON -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
echo "   Current Python environment tag: $PY_TAG"

# B. Construct search keywords
# Extract Torch short version (e.g., 2.4.0 -> 2.4) to match 'torch2.4' in filenames
TORCH_SHORT_VER=$(echo $TARGET_TORCH_VERSION | cut -d. -f1,2)
SEARCH_PATTERN="flash_attn*torch${TORCH_SHORT_VER}*${PY_TAG}*.whl"

echo "   Searching for matching file: $SEARCH_PATTERN"

# C. Search in the repository
FOUND_WHEEL=$(find "$WHEEL_STORE" -name "$SEARCH_PATTERN" | head -n 1)

if [ -n "$FOUND_WHEEL" ]; then
    echo "✅ [Hit] Found matching pre-compiled wheel!"
    echo "   File: $FOUND_WHEEL"
    
    # D. Install Wheel directly
    $UV_CMD pip install --python "$VENV_PYTHON" "$FOUND_WHEEL"
else
    echo "❌ [Failed] No matching Wheel found in $WHEEL_STORE."
    echo "   Please ensure you have downloaded a file meeting the following criteria:"
    echo "   1. Contains '$PY_TAG' (Python version)"
    echo "   2. Contains 'torch$TORCH_SHORT_VER' (Torch version)"
    echo "   Example filename: flash_attn-2.6.3+cu12torch${TORCH_SHORT_VER}cxx11abiFALSE-${PY_TAG}-${PY_TAG}-linux_x86_64.whl"
    exit 1
fi


# 🛠️ [Fix] Pre-install build dependencies
# Since --no-build-isolation is used later, build backend tools must be installed manually in the environment
echo "⬇️  Pre-installing build dependencies (poetry)..."
$UV_CMD pip install --python "$VENV_PYTHON" poetry


# ==========================================================
# 📦 4. Install Business Code
# ==========================================================

# rllm
if [ -d "$MAIN_REPO_PATH" ]; then
    echo "🔄 Installing rllm ($MAIN_REPO_PATH)..."
    $UV_CMD pip install --python "$VENV_PYTHON" \
        --no-build-isolation \
        -e "${MAIN_REPO_PATH}[verl]"
else
    echo "❌ rllm directory not found"; exit 1
fi

# Refresh constraints
$UV_CMD pip freeze --python "$VENV_PYTHON" | grep -v "file:///" > constraints.txt

# R2E-Gym
if [ -d "$R2E_PATH" ]; then
    echo "🔄 Installing R2E-Gym..."
    $UV_CMD pip install --python "$VENV_PYTHON" -e "$R2E_PATH" -c constraints.txt
fi

$UV_CMD pip install --python "$VENV_PYTHON" json5 wandb loguru jsonlines pandas

echo "========================================"
echo "🎉 Environment deployment completed!"
echo "Current environment: $ENV_NAME"
echo "Activation command: source $ENV_NAME/bin/activate"
echo "========================================"