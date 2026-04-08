import torch

from rllm.utils import overall_metrics

from argparse import ArgumentParser


if __name__ == "__main__":
    import os
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, default="/data/tmp/SWE-bench-agent/storage/test_outputs/trajectories_multi_swe_bench_flash_models--Qwen--Qwen3-8B-snapshots-b968826d9c46dd6066d109eabc6255188de91218_mswemyagent.pt")

    args = parser.parse_args()

    results = torch.load(args.path, weights_only=False)
    
    overall_metrics(results)
