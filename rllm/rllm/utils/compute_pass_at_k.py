import os
from collections import defaultdict

import torch


# def compute_pass_at_k(results):
#     import hashlib
#     import json

#     # Create a map to store correct answers per problem
#     problem_correct_map: defaultdict[str, int] = defaultdict(int)
#     problem_total_map: defaultdict[str, int] = defaultdict(int)

#     # Count correct answers for each problem
#     for trajectory in results:
#         task = trajectory.task

#         # Generate hash of problem dict/string
#         if isinstance(task, dict):
#             problem_str = json.dumps(task, sort_keys=True)
#         else:
#             problem_str = str(task)
#         problem_hash = hashlib.md5(problem_str.encode()).hexdigest()

#         is_correct = 1 if trajectory.reward > 0 else 0

#         problem_correct_map[problem_hash] += is_correct
#         problem_total_map[problem_hash] += 1

#     # Calculate pass@1 and pass@16
#     total_problems = len(problem_correct_map)
#     pass_at_1 = sum(problem_correct_map.values()) / sum(problem_total_map.values())
#     pass_at_k = sum(1 for problem, correct in problem_correct_map.items() if correct > 0) / total_problems

#     print("Total unique problems:", total_problems)
#     print("Average Pass@1 Accuracy:", pass_at_1)
#     print("Average Pass@k Accuracy:", pass_at_k)


# use "Token" mode res plus excute_tasks() return results
# classify by language
def compute_pass_at_k(results):
    import hashlib
    import json

    # Create a map to store correct answers per problem
    problem_correct_map: defaultdict[str, int] = defaultdict(int)
    problem_total_map: defaultdict[str, int] = defaultdict(int)
    
    # Create metrics collection
    all_metrics = {"steps": [], "reward_time": [], "env_time": [], "llm_time": [], "total_time": [], "full_trajectory_token_length": [], "actual_trajectory_token_length": []}
    
    # Count correct answers for each problem
    for res in results:
        task = res["task"]

        # Generate hash of problem dict/string
        if isinstance(task, dict):
            problem_str = json.dumps(task, sort_keys=True)
        else:
            problem_str = str(task)
        problem_hash = hashlib.md5(problem_str.encode()).hexdigest()

        is_correct = 1 if res["trajectory_reward"] > 0 else 0

        problem_correct_map[problem_hash] += is_correct
        problem_total_map[problem_hash] += 1
        
        # Collect metrics
        if "metrics" in res:
            metrics = res["metrics"]
            for key in all_metrics.keys():
                if key in metrics and metrics[key] is not None:
                    all_metrics[key].append(metrics[key])

    # Calculate pass@1 and pass@16
    total_problems = len(problem_correct_map)
    pass_at_1 = sum(problem_correct_map.values()) / sum(problem_total_map.values()) if sum(problem_total_map.values()) > 0 else 0
    pass_at_k = sum(1 for problem, correct in problem_correct_map.items() if correct > 0) / total_problems if total_problems > 0 else 0

    print("Total unique problems:", total_problems)
    print("Average Pass@1 Accuracy:", pass_at_1)
    print("Average Pass@k Accuracy:", pass_at_k)
    
    # Print metrics
    print("\nMetrics:")
    for key in all_metrics.keys():
        if all_metrics[key]:
            avg_value = sum(all_metrics[key]) / len(all_metrics[key])
            print(f"  Average {key}: {avg_value:.4f}")


def overall_metrics(results):
    # Overall statistics
    print("\n" + "="*80)
    print("Overall Statistics:")
    print("="*80)
    compute_pass_at_k(results)
    
    # Group results by language
    from collections import defaultdict
    results_by_language = defaultdict(list)
    for res in results:
        language = res.get("language", "unknown")
        results_by_language[language].append(res)
    
    # Compute pass@k for each language
    print("\n" + "="*80)
    print("Language-Specific Statistics:")
    print("="*80)
    for language in sorted(results_by_language.keys()):
        print(f"\nLanguage: {language}")
        print("-" * 80)
        compute_pass_at_k(results_by_language[language])
    print("="*80 + "\n")


def save_trajectories(results, save_dir="./trajectories", filename="trajectories.pt"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    torch.save(results, save_path)
    print(f"Trajectories saved to {save_path}")
    return save_path