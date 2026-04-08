#!/bin/bash
# Convert .pt file to parquet and run trajectory statistics
# Usage: ./run_pt2parquet_and_stats.sh <pt_file> [tokenizer_name]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <pt_file> [tokenizer_name]"
    echo "Example: $0 /path/to/trajectories.pt"
    echo "Example: $0 /path/to/trajectories.pt /path/to/tokenizer"
    exit 1
fi

PT_FILE="$1"
TOKENIZER_NAME="${2:-/mnt/82_store/huggingface_cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}"

if [ ! -f "$PT_FILE" ]; then
    echo "Error: File $PT_FILE does not exist"
    exit 1
fi

# Generate parquet path (same name with .parquet extension)
PARQUET_FILE="${PT_FILE%.pt}.parquet"

echo "=========================================="
echo "Step 1: Converting .pt to .parquet"
echo "=========================================="
python "$SCRIPT_DIR/pt2parquet.py" "$PT_FILE" --output "$PARQUET_FILE"

echo ""
echo "=========================================="
echo "Step 2: Running trajectory statistics"
echo "=========================================="
python "$SCRIPT_DIR/traj_statics.py" --parquet_path "$PARQUET_FILE" --tokenizer_name "$TOKENIZER_NAME"

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
