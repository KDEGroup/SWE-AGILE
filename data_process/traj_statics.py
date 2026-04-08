import os
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from rllm.parser.chat_template_parser import ChatTemplateParser

# Set matplotlib backend to Agg to prevent errors on servers without a display
plt.switch_backend('Agg')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_path", type=str, required=True)
    parser.add_argument("--tokenizer_name", type=str, default="/mnt/82_store/huggingface_cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save JSON results")
    parser.add_argument("--plot_dir", type=str, default=None, help="Directory to save histogram images. If not specified, defaults to the same directory as the parquet file.")
    return parser.parse_args()

def build_user_key(messages):
    """
    Default: messages[1] is the user message containing the task description.
    """
    if messages is None:
        return None

    if hasattr(messages, "tolist"):
        messages = messages.tolist()

    if len(messages) < 2:
        return None

    m = messages[1]
    if m.get("role") != "user":
        return None
    return str(m.get("content", "")).strip()

def count_tokens_for_messages(messages, tokenizer, parser):
    """
    Calculate total tokens for the entire trajectory (all messages) after applying Chat Template.
    """
    if messages is None:
        return 0

    if hasattr(messages, "tolist"):
        messages = messages.tolist()

    for m in messages:
        tc = m.get("tool_calls")
        if isinstance(tc, np.ndarray):
            m["tool_calls"] = tc.tolist()
            
    # Set environment variable for inference-related parser logic
    os.environ["last_n_reasoning"] = '999'
    
    try:
        text = parser.parse(messages, is_first_msg=True, add_generation_prompt=False, last_n_reasoning=999)
        ids = tokenizer.encode(text, add_special_tokens=False)
        return len(ids)
    except Exception as e:
        print(f"Chat Template Parse Error: {e}")
        return 0

def count_field_tokens(text, tokenizer):
    """
    Calculate token count for a text segment only, without special tokens.
    """
    if not text:
        return 0
    # Ensure it is a string
    text = str(text)
    ids = tokenizer.encode(text, add_special_tokens=False)
    return len(ids)


def remove_function_call(text):
    """
    Remove the function call part at the end of content (starting with "<function").
    """
    if not text:
        return ""
    text = str(text)
    # Find the position of "<function" and remove content from that position to the end
    func_idx = text.find("<function")
    if func_idx != -1:
        return text[:func_idx].rstrip()
    return text

def save_histogram(data_series, title, save_path, xlabel):
    """
    Plot and save histogram.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(data_series, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.5)
    
    # Annotate some basic statistical information on the plot
    mean_val = data_series.mean()
    max_val = data_series.max()
    min_val = data_series.min()
    text_str = f'Mean: {mean_val:.2f}\nMax: {max_val:.2f}\nMin: {min_val:.2f}'
    # Place in the top right corner
    plt.text(0.95, 0.95, text_str, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.savefig(save_path)
    plt.close()

def main():
    args = parse_args()

    print(f"Loading parquet: {args.parquet_path}")
    df = pd.read_parquet(args.parquet_path)

    # 1) Construct trajectory ID
    df["traj_key"] = df["messages"].apply(build_user_key)
    df = df[~df["traj_key"].isna()].copy()

    # 2) Prepare tokenizer & parser
    print(f"Loading tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    parser = ChatTemplateParser.get_parser(tokenizer)

    records = []
    # Globally collect tokens for all steps (regardless of trajectory)
    all_step_reasoning_tokens = []
    all_step_content_tokens = []
    all_step_content_no_func_tokens = []  # tokens for content after removing function call

    grouped = df.groupby("traj_key", sort=False)

    print(f"Processing {len(grouped)} trajectories...")

    for traj_key, group in grouped:
        # 2.2 Select the sample with the longest messages
        idx_max = group["messages"].apply(len).idxmax()
        full_messages = df.loc[idx_max, "messages"]

        # Compatible with numpy array
        if hasattr(full_messages, "tolist"):
            full_messages = full_messages.tolist()

        # 2.3 Calculate total tokens for the entire trajectory (including Template)
        traj_token_count = count_tokens_for_messages(full_messages, tokenizer, parser)

        # 2.4 Calculate Token counts for Reasoning and Content in Assistant messages
        total_reasoning_tokens = 0
        total_content_tokens = 0
        total_content_no_func_tokens = 0  # tokens for content after removing function call
        num_steps = 0  # Count number of assistant messages as step count

        for msg in full_messages:
            if msg.get("role") == "assistant":
                num_steps += 1
                # Calculate reasoning
                reasoning_content = msg.get("reasoning", "")
                step_reasoning = count_field_tokens(reasoning_content, tokenizer)
                total_reasoning_tokens += step_reasoning
                all_step_reasoning_tokens.append(step_reasoning)

                # Calculate content
                content_text = msg.get("content", "")
                step_content = count_field_tokens(content_text, tokenizer)
                total_content_tokens += step_content
                all_step_content_tokens.append(step_content)

                # Calculate tokens for content after removing function call
                content_no_func = remove_function_call(content_text)
                step_content_no_func = count_field_tokens(content_no_func, tokenizer)
                total_content_no_func_tokens += step_content_no_func
                all_step_content_no_func_tokens.append(step_content_no_func)

        records.append(
            {
                "num_steps": num_steps,
                "traj_token_count": traj_token_count,
                "traj_reasoning_tokens": total_reasoning_tokens,
                "traj_content_tokens": total_content_tokens,
                "traj_content_no_func_tokens": total_content_no_func_tokens,
            }
        )

    if not records:
        print("No valid records found.")
        return

    # 3) Statistical Aggregation (Avg, Max, Min)
    df_records = pd.DataFrame(records)

    # Define metric columns to be aggregated
    metrics = ["num_steps", "traj_token_count", "traj_reasoning_tokens", "traj_content_tokens", "traj_content_no_func_tokens"]

    # Use IQR method to detect outliers in traj_content_tokens
    Q1 = df_records["traj_content_tokens"].quantile(0.15)
    Q3 = df_records["traj_content_tokens"].quantile(0.85)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR

    # Mark outliers (traj_content_tokens exceeds upper bound)
    outlier_mask = df_records["traj_content_tokens"] > upper_bound
    num_outliers = outlier_mask.sum()
    df_records_clean = df_records[~outlier_mask].copy()

    print(f"\n--- Outlier Detection (traj_content_tokens) ---")
    print(f"IQR method: Q1={Q1:.0f}, Q3={Q3:.0f}, IQR={IQR:.0f}, Upper bound={upper_bound:.0f}")
    print(f"Outliers detected: {num_outliers} / {len(df_records)} ({num_outliers/len(df_records)*100:.1f}%)")

    stats_summary = {
        "traj_key": "TOTAL_STATS",
        "num_records": len(records),
        "num_outliers": int(num_outliers),
        "outlier_upper_bound": float(upper_bound),
    }

    # Original statistics
    print("\n--- Statistics Summary (ALL data) ---")
    for metric in metrics:
        if metric in df_records.columns:
            stats_summary[f"{metric}_avg"] = float(df_records[metric].mean())
            stats_summary[f"{metric}_max"] = float(df_records[metric].max())
            stats_summary[f"{metric}_min"] = float(df_records[metric].min())
            print(f"{metric}: Avg={stats_summary[f'{metric}_avg']:.2f}, Max={stats_summary[f'{metric}_max']}, Min={stats_summary[f'{metric}_min']}")

    # Statistics excluding outliers (only output per step global average)
    if num_outliers > 0 and len(df_records_clean) > 0:
        print(f"\n--- Statistics Summary (excluding {num_outliers} outliers, per step avg only) ---")
        for metric in metrics:
            if metric in df_records_clean.columns:
                # Calculate per step average = total tokens / total steps
                total_metric = df_records_clean[metric].sum()
                total_steps_clean = df_records_clean["num_steps"].sum()
                if total_steps_clean > 0:
                    per_step_avg = total_metric / total_steps_clean
                    stats_summary[f"{metric}_clean_per_step_avg"] = float(per_step_avg)
                    print(f"{metric}: per_step_avg={per_step_avg:.2f}")

    # Global step-level statistics (regardless of trajectory, excluding steps with empty reasoning)
    print("\n--- Global Step-Level Statistics (excluding empty reasoning) ---")
    # Filter out steps where reasoning=0
    valid_indices = [i for i, r in enumerate(all_step_reasoning_tokens) if r > 0]
    filtered_reasoning = [all_step_reasoning_tokens[i] for i in valid_indices]
    filtered_content = [all_step_content_tokens[i] for i in valid_indices]
    filtered_content_no_func = [all_step_content_no_func_tokens[i] for i in valid_indices]

    total_steps = len(all_step_reasoning_tokens)
    valid_steps = len(filtered_reasoning)
    print(f"Total steps: {total_steps}, Valid steps (reasoning > 0): {valid_steps}")

    if filtered_reasoning:
        global_reasoning_avg = float(np.mean(filtered_reasoning))
        global_reasoning_max = max(filtered_reasoning)
        global_reasoning_min = min(filtered_reasoning)
        global_content_avg = float(np.mean(filtered_content))
        global_content_max = max(filtered_content)
        global_content_min = min(filtered_content)
        global_content_no_func_avg = float(np.mean(filtered_content_no_func))
        global_content_no_func_max = max(filtered_content_no_func)
        global_content_no_func_min = min(filtered_content_no_func)

        stats_summary["global_total_steps"] = total_steps
        stats_summary["global_valid_steps"] = valid_steps
        stats_summary["global_step_reasoning_avg"] = global_reasoning_avg
        stats_summary["global_step_reasoning_max"] = global_reasoning_max
        stats_summary["global_step_reasoning_min"] = global_reasoning_min
        stats_summary["global_step_content_avg"] = global_content_avg
        stats_summary["global_step_content_max"] = global_content_max
        stats_summary["global_step_content_min"] = global_content_min
        stats_summary["global_step_content_no_func_avg"] = global_content_no_func_avg
        stats_summary["global_step_content_no_func_max"] = global_content_no_func_max
        stats_summary["global_step_content_no_func_min"] = global_content_no_func_min

        print(f"global_step_reasoning: Avg={global_reasoning_avg:.2f}, Max={global_reasoning_max}, Min={global_reasoning_min}")
        print(f"global_step_content: Avg={global_content_avg:.2f}, Max={global_content_max}, Min={global_content_min}")
        print(f"global_step_content_no_func: Avg={global_content_no_func_avg:.2f}, Max={global_content_no_func_max}, Min={global_content_no_func_min}")

    # Append statistical results to the end of the records list (maintain original output structure)
    records.append(stats_summary)

    # 4) Determine output directory (use parquet filename prefix as folder name)
    parquet_dir = os.path.dirname(os.path.abspath(args.parquet_path))
    parquet_basename = os.path.splitext(os.path.basename(args.parquet_path))[0]
    output_dir = os.path.join(parquet_dir, parquet_basename)
    os.makedirs(output_dir, exist_ok=True)

    # Save JSON
    out_path = args.output_path or os.path.join(output_dir, "stats.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"\nSaving statistics JSON to: {out_path}")

    # 5) Plot and save histograms
    plot_dir = args.plot_dir or output_dir
    os.makedirs(plot_dir, exist_ok=True)

    # Define plotting configuration (keep only key charts)
    plot_configs = [
        ("num_steps", "Distribution of Steps per Trajectory", "Steps"),
        ("traj_token_count", "Distribution of Total Trajectory Tokens", "Tokens"),
    ]

    print("Generating histograms...")
    for col, title, xlabel in plot_configs:
        if col in df_records.columns:
            save_name = os.path.join(plot_dir, f"{col}_hist.png")
            save_histogram(df_records[col], title, save_name, xlabel)
            print(f"  -> Saved: {save_name}")

    # Plot global step-level distribution charts (excluding steps with empty reasoning)
    if filtered_reasoning:
        save_name = os.path.join(plot_dir, "global_step_reasoning_hist.png")
        save_histogram(pd.Series(filtered_reasoning), "Distribution of Per-Step Reasoning Tokens (Global)", save_name, "Tokens")
        print(f"  -> Saved: {save_name}")

        save_name = os.path.join(plot_dir, "global_step_content_hist.png")
        save_histogram(pd.Series(filtered_content), "Distribution of Per-Step Content Tokens (Global)", save_name, "Tokens")
        print(f"  -> Saved: {save_name}")

        save_name = os.path.join(plot_dir, "global_step_content_no_func_hist.png")
        save_histogram(pd.Series(filtered_content_no_func), "Distribution of Per-Step Content (No FuncCall) Tokens (Global)", save_name, "Tokens")
        print(f"  -> Saved: {save_name}")

if __name__ == "__main__":
    main()