import asyncio
import json
import os
from collections import defaultdict

import pandas as pd
from transformers import AutoTokenizer

import hydra
from rllm.agents.swe_agent import SWEAgent
from rllm.data.dataset import DatasetRegistry
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.environments.swe.swe import SWEEnv
from rllm.trainer.agent_sft_trainer import AgentSFTTrainer
from rllm.utils.compute_pass_at_k import overall_metrics, save_trajectories


from loguru import logger


def analyze_passn_statistics(results, trajectories_per_problem: int, reward_threshold: float = 1.0):
    """
    Analyze pass@n statistics for each problem.

    Args:
        results: List of Trajectory objects from execute_tasks
        trajectories_per_problem: Number of trajectories per problem (n in pass@n)
        reward_threshold: Threshold for considering a trajectory successful

    Returns:
        dict: {
            "statistics": {instance_id: {"success_count": int, "total": int, "success_indices": list}},
            "successful_trajectories": list of successful Trajectory objects,
            "summary": {"total_problems": int, "problems_with_k_success": {k: count}}
        }
    """
    from rllm.agents.agent import Trajectory

    # Group results by base instance_id (without _traj_N suffix)
    problem_results = defaultdict(list)

    for idx, traj in enumerate(results):
        # Handle both Trajectory objects and dicts
        assert isinstance(traj, Trajectory)
        task = traj.task or {}
        reward = traj.reward
        uid = task.get("uid", None)

        # instance_id is the unique identifier for each problem (set in load_swe_data)
        # {base_id}_traj_{i}
        if uid and "_traj_" in str(uid):
            base_id = str(uid).rsplit("_traj_", 1)[0]
        else:
            num_unique_problems = len(results) // trajectories_per_problem
            base_id = task.get("instance_id", f"problem_{idx % num_unique_problems}") if isinstance(task, dict) else f"problem_{idx % num_unique_problems}"

        problem_results[base_id].append({
            "result": traj,
            "index": idx,
            "uid": uid,
            "reward": reward,
            "is_success": reward >= reward_threshold
        })

    # Build statistics
    statistics = {}
    successful_trajectories = []
    problems_with_k_success = defaultdict(int)  # k -> count of problems with exactly k successes

    for instance_id, traj_list in problem_results.items():
        success_count = sum(1 for t in traj_list if t["is_success"])
        success_indices = [t["index"] for t in traj_list if t["is_success"]]
        success_uids = [t["uid"] for t in traj_list if t["is_success"]]

        statistics[instance_id] = {
            "success_count": success_count,
            "total": len(traj_list),
            "success_indices": success_indices,
            "success_uids": success_uids,
            "rewards": [float(t["reward"]) for t in traj_list]  # Convert to native float for JSON
        }

        # Count problems by success count
        problems_with_k_success[success_count] += 1

        # Collect successful trajectories
        for t in traj_list:
            if t["is_success"]:
                successful_trajectories.append(t["result"])

    # Build summary
    total_problems = len(problem_results)
    summary = {
        "total_problems": total_problems,
        "trajectories_per_problem": trajectories_per_problem,
        "total_trajectories": len(results),
        "total_successful": len(successful_trajectories),
        "problems_with_k_success": dict(problems_with_k_success),  # k -> count of problems with exactly k successes
        # pass_at_n = (total_problems - problems_with_0_success) / total_problems
        "pass_at_n": (total_problems - problems_with_k_success.get(0, 0)) / total_problems if total_problems else 0,
    }

    return {
        "statistics": statistics,
        "successful_trajectories": successful_trajectories,
        "summary": summary
    }


def load_swe_data(registry_dataset_name, registry_dataset_split, num_samples=None, trajectories_per_problem=1, sample_begin=None, sample_end=None):
    """
    Loads SWE data, with optional sampling and problem replication.

    Args:
        sample_begin: Start index for slicing original problems (before replication)
        sample_end: End index for slicing original problems (before replication)
    """
    if DatasetRegistry.dataset_exists(registry_dataset_name, registry_dataset_split):
        test_dataset = DatasetRegistry.load_dataset(registry_dataset_name, registry_dataset_split)
        data = test_dataset.get_data()  # This is a list of dicts

        # 1. Apply sample_begin/sample_end slicing (before any replication)
        if sample_begin is not None or sample_end is not None:
            original_len = len(data)
            sample_begin = sample_begin or 0
            sample_end = sample_end or len(data)
            data = data[sample_begin:sample_end]
            print(f"Sliced problems [{sample_begin}:{sample_end}] from {original_len}, got {len(data)} problems")

        # 2. Apply num_samples (random sampling)
        if num_samples is not None and num_samples > 0 and num_samples < len(data):
            print(f"Sampling {num_samples} problems from {len(data)}...")
            df = pd.DataFrame(data)
            df = df.sample(n=num_samples, random_state=42)  # Use a fixed random state for reproducibility
            data = df.to_dict("records")

        # 2. Apply trajectories_per_problem
        if trajectories_per_problem > 1:
            print(f"Replicating {len(data)} problems {trajectories_per_problem} times...")
            tasks = []
            for i in range(trajectories_per_problem):
                for j, row in enumerate(data):
                    # Create a *copy* of the row and add a unique ID for this trajectory
                    new_task = row.copy()
                    # Assume original task (row) has a unique 'instance_id'
                    base_id = row.get('instance_id', f"problem_{j}")  # Fallback to index
                    new_task['uid'] = f"{base_id}_traj_{i}"  # Create a new unique ID for the run
                    tasks.append(new_task)
            print(f"Total tasks generated: {len(tasks)}")
            return tasks
        else:
            # If trajectories_per_problem is 1, just return the (potentially sampled) data
            print(f"Total tasks loaded: {len(data)}")
            return data

    raise ValueError(
        f"{registry_dataset_name} {registry_dataset_split} dataset not found. "
        f"Please run `python prepare_swe_data.py` to create the dataset."
    )


def run_deepswe(config):
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # Target modes:
    # - 'eval': Run evaluation and save trajectories
    # - 'sft': Generate SFT data (trajectory-level or step-level)
    target = config.rllm.run.get("target", "eval")

    # SFT mode: 'traj' (trajectory-level) or 'step' (step-level)
    # Only used when target='sft'
    sft_mode = config.rllm.run.get("sft_mode", "step")

    # Data params
    num_samples = config.data.get("num_samples", None)
    trajectories_per_problem = config.data.get("trajectories_per_problem", 1)
    sample_begin = config.data.get("sample_begin", None)
    sample_end = config.data.get("sample_end", None)

    # Handle training mode directly
    if target == "train_sft_step":
        print("=" * 80)
        print("Training Stepwise SFT from existing data files...")
        print("=" * 80)

        trainer = AgentSFTTrainer(config=config)
        trainer.train()
        return

    # For other targets, we need the execution engine
    tokenizer = AutoTokenizer.from_pretrained(config.for_close_source_api.tokenizer)
    engine = AgentExecutionEngine(
        config=config,
        engine_name="openai",
        max_steps=config.rllm.agent.max_steps,
        max_response_length=config.data.max_response_length,
        max_prompt_length=config.data.max_prompt_length,
        agent_class=SWEAgent,
        agent_args=config.rllm.agent.get("agent_args"),
        env_class=SWEEnv,
        env_args=config.rllm.env.get("env_args"),
        enforce_max_prompt_length=config.rllm.get("enforce_max_prompt_length", False),
        trajectory_timeout=config.rllm.agent.trajectory_timeout,
        overlong_filter=config.rllm.agent.get("overlong_filter", False),
        disable_thinking=config.rllm.disable_thinking,
        rollout_engine_args=config.rollout_engine_args,
        tokenizer=tokenizer,
        **config.rllm.agent.get("engine_args", {}),
    )

    # Load tasks
    tasks = load_swe_data(
        config.registry_dataset_name,
        config.registry_dataset_split,
        num_samples=num_samples,
        trajectories_per_problem=trajectories_per_problem,
        sample_begin=sample_begin,
        sample_end=sample_end,
    )

    if not tasks:
        print("No tasks loaded. Exiting.")
        return

    # Generate filename components
    model_name = config.rllm.get("eval", {}).get("model_name", None)
    if not model_name:
        model_name = "-".join(config.rollout_engine_args.model.split("/")[-3:])
    scaffold = config.rllm.agent.agent_args.scaffold
    accumulate_reasoning = config.rllm.agent.agent_args.accumulate_reasoning
    use_tool_calling = config.rllm.agent.agent_args.use_tool_calling

    reasoning_tag = config.rllm.get('last_n_reasoning', f'randn14')
    tool_calling_tag = "json_tool" if use_tool_calling else "xml_tool"

    # ========================================================================
    # SFT Data Generation
    # ========================================================================
    if target == "sft":
        sft_config = config.get("sft", {})
        reward_threshold = sft_config.get("reward_threshold", 1.0)

        # Generate trajectories in "Text" mode
        results = asyncio.run(engine.execute_tasks(tasks, mode="Text"))

        # Setup storage directory
        storage_dir = os.path.join(os.getenv("STORAGE_DIR", "."), "sft_outputs")
        os.makedirs(storage_dir, exist_ok=True)
        n_suffix = f"_n{trajectories_per_problem}" if trajectories_per_problem > 1 else ""
        # Range tag for filenames (only add if explicitly specified)
        range_tag = f"_{sample_begin}_{sample_end}" if sample_begin is not None or sample_end is not None else ""

        # ====================================================================
        # Pass@N Statistics Analysis (when trajectories_per_problem > 1)
        # ====================================================================
        if trajectories_per_problem > 1:
            print("\n" + "=" * 80)
            print(f"Pass@{trajectories_per_problem} Statistics Analysis")
            print("=" * 80)

            passn_analysis = analyze_passn_statistics(
                results,
                trajectories_per_problem=trajectories_per_problem,
                reward_threshold=reward_threshold
            )

            # Print summary
            summary = passn_analysis["summary"]
            print(f"\nTotal problems: {summary['total_problems']}")
            print(f"Trajectories per problem: {summary['trajectories_per_problem']}")
            print(f"Total trajectories: {summary['total_trajectories']}")
            print(f"Total successful: {summary['total_successful']}")
            print(f"Pass@{trajectories_per_problem}: {summary['pass_at_n']:.4f}")

            print(f"\nDistribution of success counts per problem:")
            for k in sorted(summary["problems_with_k_success"].keys()):
                count = summary["problems_with_k_success"][k]
                print(f"  {k} success(es): {count} problems")

            # Save statistics to JSON
            stats_filename = f"passn_stats_{config.registry_dataset_name}{range_tag}_{model_name}_{scaffold}{n_suffix}.json"
            stats_file = os.path.join(storage_dir, stats_filename)
            with open(stats_file, "w") as f:
                json.dump({
                    "summary": summary,
                    "statistics": passn_analysis["statistics"]
                }, f, indent=2)
            print(f"\n✓ Saved pass@n statistics to: {stats_file}")

            # Save all raw trajectories (for debugging/analysis)
            import torch
            raw_traj_filename = f"raw_trajectories_{config.registry_dataset_name}{range_tag}_{model_name}_{scaffold}{n_suffix}.pt"
            torch.save(results, os.path.join(storage_dir, raw_traj_filename))
            print(f"✓ Saved all raw trajectories to: {os.path.join(storage_dir, raw_traj_filename)}")

            # Use only successful trajectories for SFT data generation
            results_for_sft = passn_analysis["successful_trajectories"]
            print(f"\nUsing {len(results_for_sft)} successful trajectories for SFT data generation")
        else:
            results_for_sft = results

        # sft_mode == "step": also needs to save sft-traj data due to last_n_reasoning
        # but needs to apply python to_training_data_verl.py --scaffold normal --add_digest_tag false --remove_digest_tag true --sft_mode trajectory --reasoning_filter none --input_file
        try:
            sft_data = AgentSFTTrainer.process_trajectories(
                results_for_sft,
                reward_threshold=reward_threshold,
            )
            # Save to parquet
            filename = f"sft_traj_{config.registry_dataset_name}{range_tag}_{model_name}_{scaffold}_{reasoning_tag}_{tool_calling_tag}{n_suffix}.parquet"
            output_file = os.path.join(storage_dir, filename)

            if sft_data:
                df = pd.DataFrame(sft_data)
                df.to_parquet(output_file, index=False)
                print(f"✓ Saved {len(sft_data)} SFT traj examples to: {output_file}")
            else:
                print(f"\n⚠ No valid SFT data generated (all trajectories below reward threshold {reward_threshold})")
        except Exception as e:
            print(f"traj level saving failed: {e}")

        if sft_mode == "step":
            sft_data = AgentSFTTrainer.process_trajectories_every_steps(
                results_for_sft,
                reward_threshold=reward_threshold,
            )
            # Save to parquet
            filename = f"sft_step_{config.registry_dataset_name}{range_tag}_{model_name}_{scaffold}_{reasoning_tag}_{tool_calling_tag}{n_suffix}.parquet"
            output_file = os.path.join(storage_dir, filename)

            if sft_data:
                df = pd.DataFrame(sft_data)
                df.to_parquet(output_file, index=False)
                print(f"✓ Saved {len(sft_data)} SFT step examples to: {output_file}")
            else:
                print(f"\n⚠ No valid SFT data generated (all trajectories below reward threshold {reward_threshold})")


    elif target == "eval":
        # tasks = tasks[:64]
        results = asyncio.run(engine.execute_tasks(tasks, mode="Token"))

        # Calculate metrics
        overall_metrics(results)

        # Save trajectories
        filename = f"trajectories_{config.registry_dataset_name}_{model_name}_{scaffold}_{reasoning_tag}_{tool_calling_tag}.pt"
        storage_dir = os.path.join(os.getenv("STORAGE_DIR", "."), "test_outputs")
        os.makedirs(storage_dir, exist_ok=True)
        save_path = os.path.join(storage_dir, filename)

        save_trajectories(results, storage_dir, filename)
        print(f"✓ Saved full trajectories to: {save_path}")


    else:
        raise ValueError(
            f"Unknown target: '{target}'. "
            f"Valid options: 'eval', 'sft', 'train_sft_step'"
        )


@hydra.main(
    config_path="../../../rllm/trainer/config",
    config_name="agent_ppo_trainer",
    version_base=None
)
def main(config):
    run_deepswe(config)


if __name__ == "__main__":
    main()