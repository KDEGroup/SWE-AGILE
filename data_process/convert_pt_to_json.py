#!/usr/bin/env python3
"""
Convert PyTorch .pt trajectory files to JSON format for inspection.
"""

import torch
import json
import argparse
import os
from pathlib import Path
import numpy as np


def convert_tensor_to_list(obj):
    """Recursively convert torch tensors and numpy types to JSON-serializable format."""
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
        return {key: convert_tensor_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_tensor_to_list(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_tensor_to_list(item) for item in obj)
    else:
        return obj


def load_and_convert_to_json(pt_file_path, output_json_path=None):
    """
    Load a PyTorch .pt file containing trajectories and convert to JSON.

    Args:
        pt_file_path: Path to the .pt file
        output_json_path: Optional output path for JSON file. If None, uses same name with .json extension
    """
    print(f"Loading trajectories from: {pt_file_path}")

    # Load the trajectories
    trajectories = torch.load(pt_file_path, map_location='cpu', weights_only=False)

    print(f"Loaded {len(trajectories)} trajectories")

    # Convert tensors to serializable format
    print("Converting tensors to serializable format...")
    json_trajectories = convert_tensor_to_list(trajectories)

    # Generate output path if not provided
    if output_json_path is None:
        pt_path = Path(pt_file_path)
        output_json_path = pt_path.with_suffix('.json')

    # Save as JSON
    print(f"Saving to JSON: {output_json_path}")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(json_trajectories, f, indent=2, ensure_ascii=False)

    print(f"✓ Conversion complete! JSON saved to: {output_json_path}")

    # Print some basic statistics
    if json_trajectories:
        print(f"\nBasic Statistics:")
        print(f"  Total trajectories: {len(json_trajectories)}")

        if isinstance(json_trajectories[0], dict):
            sample_keys = list(json_trajectories[0].keys())
            print(f"  Sample trajectory keys: {sample_keys}")

            # Count successful trajectories
            successful = sum(1 for traj in json_trajectories if traj.get('trajectory_reward', 0) > 0)
            print(f"  Successful trajectories: {successful}")
            print(f"  Success rate: {successful/len(json_trajectories)*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch trajectory .pt files to JSON')
    parser.add_argument('pt_file', help='Path to the .pt trajectory file')
    parser.add_argument('--output', '-o', help='Output JSON file path (optional)')

    args = parser.parse_args()

    if not os.path.exists(args.pt_file):
        print(f"Error: File {args.pt_file} does not exist")
        return

    load_and_convert_to_json(args.pt_file, args.output)


if __name__ == "__main__":
    main()
