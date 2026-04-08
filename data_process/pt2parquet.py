#!/usr/bin/env python3
"""
Convert PyTorch .pt trajectory files to Parquet format.
Output format is compatible with traj_statics.py.
"""

import torch
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd


def convert_numpy_types(obj):
    """Recursively convert torch tensors and numpy types to Python native types."""
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return list(convert_numpy_types(item) for item in obj)
    else:
        return obj


def load_and_convert_to_parquet(pt_file_path, output_parquet_path=None):
    """
    Load a PyTorch .pt file containing trajectories and convert to Parquet.
    Output format is compatible with traj_statics.py (requires 'messages' column).

    Args:
        pt_file_path: Path to the .pt file
        output_parquet_path: Optional output path for Parquet file. If None, uses same name with .parquet extension
    """
    print(f"Loading trajectories from: {pt_file_path}")

    # Load the trajectories
    trajectories = torch.load(pt_file_path, map_location='cpu', weights_only=False)

    print(f"Loaded {len(trajectories)} trajectories")

    # Convert tensors to native Python types
    print("Converting tensors to native types...")
    converted_trajectories = convert_numpy_types(trajectories)

    # Build records for DataFrame
    # traj_statics.py expects a 'messages' column
    records = []
    for traj in converted_trajectories:
        if isinstance(traj, dict):
            record = dict(traj)  # Copy all fields

            # Map 'chat_completions' to 'messages' if needed (for traj_statics.py compatibility)
            if 'messages' not in record and 'chat_completions' in record:
                record['messages'] = record['chat_completions']

            records.append(record)
        else:
            # Handle non-dict trajectories
            records.append({'data': traj})

    # Create DataFrame
    df = pd.DataFrame(records)

    # Generate output path if not provided
    if output_parquet_path is None:
        pt_path = Path(pt_file_path)
        output_parquet_path = pt_path.with_suffix('.parquet')

    # Save as Parquet
    print(f"Saving to Parquet: {output_parquet_path}")
    df.to_parquet(output_parquet_path, index=False)

    print(f"✓ Conversion complete! Parquet saved to: {output_parquet_path}")

    # Print some basic statistics
    print(f"\nBasic Statistics:")
    print(f"  Total records: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    if 'messages' in df.columns:
        # Count messages per trajectory
        msg_counts = df['messages'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        print(f"  Messages per trajectory: min={msg_counts.min()}, max={msg_counts.max()}, avg={msg_counts.mean():.1f}")

    if 'trajectory_reward' in df.columns:
        successful = (df['trajectory_reward'] > 0).sum()
        print(f"  Successful trajectories: {successful}")
        print(f"  Success rate: {successful/len(df)*100:.1f}%")

    return output_parquet_path


def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch trajectory .pt files to Parquet')
    parser.add_argument('pt_file', help='Path to the .pt trajectory file')
    parser.add_argument('--output', '-o', help='Output Parquet file path (optional)')

    args = parser.parse_args()

    if not os.path.exists(args.pt_file):
        print(f"Error: File {args.pt_file} does not exist")
        return

    load_and_convert_to_parquet(args.pt_file, args.output)


if __name__ == "__main__":
    main()
