# convert parquet to jsonl
from datasets import load_dataset
import json
import jsonlines
import os
import sys
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from r2egym.agenthub.utils.log import get_logger, LOG_LEVEL_MAP
logger = get_logger(__name__, LOG_LEVEL_MAP[os.getenv("LOG_LEVEL", "INFO")])
import traceback
import re
from tqdm import tqdm
import argparse

from openai import OpenAI
api_key = None
base_url = None
model = None
client = None

write_lock = threading.Lock()

# fill_back generate <reasoning> and <summary>.  But <think>...</think> is saved


# ==============================================================================
# PROMPT  v4 - some steps need only short reasoning)
# ==============================================================================
PROMPT_TEMPLATE = """
You are an expert AI Agent. Your task is to "backfill" the missing `<reasoning>` process and a high-quality `<summary>` for an existing AI Agent execution trajectory.

Your response **must** be in the following format, with no other text:
<reasoning>
... Your detailed thought process ...
</reasoning>
<summary>
... Your new, high-quality, user-facing summary of current thought. This should conclude important information....
</summary>

---
## Constraints

1.  **Roleplay:** You must fully embody the role of a code agent solving a task.

2.  **Adaptive Reasoning Depth :**
    * **Complex Reasoning:** Use deep, detailed reasoning when the step involves uncertainty, diagnosis, or design.
        * *Examples:* Analyzing a confusing error message, designing a new function structure, figuring out why a bug occurred, or deciding a complex test strategy.
    * **Routine Execution:** Use concise reasoning or reuse the [ORIGINAL HINT] instead of over-analyze  when the step is **deterministic, mechanical, or part of an already-made plan**.
        * *Examples:* Executing a script you just decided to run.

3.  **Context & Continuity:**
    * **Bridge the Gap:** Start your reasoning by explicitly connecting to the previous reasoning and the execution result of the last step, building a logical bridge to the [TARGET ACTION].
    * **Handle Errors:** If the [CONVERSATION HISTORY] shows an error or failure, you may explain it in your reasoning.
    * **No Redundancy:** Do not re-state the overall project goal or background info if not necessary. Do not re-analyze content that was already covered in previous reasoning steps unless an error or unexpected result necessitates re-reanalyzing.

4.  **`summary` Block:** * This is the compressed content of reasoning for the future.
    * If the [ORIGINAL HINT] was good and the step is simple, you may reuse the hint.
    * If the step was complex, write a concise summary of the reasoning.
---



## Task Inputs

### 1. Conversation History
[CONVERSATION HISTORY]
{history_json}
[END CONVERSATION HISTORY]

### 2. Original Summary (Hint)
This is original content agent provided. Use it as a *hint* for its intent.
* If this hint is high-quality, your new summary can be similar.
* If this hint is low-quality (e.g., empty or "Done"), you *must* generate a new, high-quality summary that conclude your reasoning.

[ORIGINAL HINT]
{original_hint}
[END ORIGINAL HINT]

### 3. Target Action
You *must* generate a `<reasoning>` process and `<summary>` that logically result in executing this exact action:

[TARGET ACTION]
{target_action}
[END TARGET ACTION]

---

Please now generate the `<reasoning>` and `<summary>` blocks.
"""




def _split_content_to_summary_and_action(target_content):
    """
    Helper function: Splits 'content' into 'summary' and 'action'.
    """
    if "<function=" not in target_content:
        return target_content, None # No action

    parts = target_content.split('<function=', 1)
    if len(parts) == 2:
        summary = parts[0].strip()
        action = '<function=' + parts[1]
        
        # Handle case where content starts with an action
        if summary == "" and action.startswith(target_content):
             return "", action # Empty summary, valid action
        
        return summary, action
    else:
        logger.error(f"_split_content_to_summary_and_action failed.\ntarget_content:{target_content}\nparts:{parts}")
        # Should not happen
        return target_content, None

def _build_prompt(history, original_hint, target_action):
    """
    Helper function: Builds the full prompt to be sent to the LLM.
    """
    # Format history as a JSON string for clear presentation in the prompt
    history_json = json.dumps(history, indent=2, ensure_ascii=False)
    
    return PROMPT_TEMPLATE.format(
        history_json=history_json,
        original_hint=original_hint if original_hint else "(No original summary was provided)",
        target_action=target_action
    )

def generate_think_and_summary(history, original_hint, target_action, turn_index):
    """
    *** 这是你需要用真实 LLM API 替换的模拟函数 ***
    
    Receives history, the *original* summary as a hint, and the target action.
    Returns a single string containing <reasoning>...</reasoning> and <summary>...</summary>.
    """
    # 1. Build the full prompt
    full_prompt = _build_prompt(history, original_hint, target_action)
    
    # logger.debug(f"\n--- 🤖 Generating <reasoning>/<summary> for Assistant Turn {turn_index} ---\n{full_prompt}")
    
    messages = [{"role": "user", "content": full_prompt}]



    estimated_input_tokens = len(full_prompt) / 4.6
    logger.info(f"Estimated Input Tokens: {int(estimated_input_tokens)}")
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
        max_tokens=4096,
    )
    
    llm_output = response.choices[0].message.content
    return llm_output




def extract_last_block(text, tag_name):
    """
    Helper function: Extracts the content of the *last* valid pair of tags.
    Robust against:
    - Multiple blocks (takes the last one)
    - Unclosed tags before the final block (e.g., <think> ... <think> content </think>)
    """
    start_tag = f"<{tag_name}>"
    end_tag = f"</{tag_name}>"
    
    # 1. find the last end tag
    end_idx = text.rfind(end_tag)
    if end_idx == -1:
        return None 
        
    start_idx = text.rfind(start_tag, 0, end_idx)
    if start_idx == -1:
        return None
        
    content_start = start_idx + len(start_tag)
    return text[content_start:end_idx].strip()

def parse_llm_output(llm_output):
    """
    Parses the content in <reasoning> and <summary> using robust string extraction.
    """
    try:
        think_content = extract_last_block(llm_output, "reasoning")
        if think_content is None:
            logger.error("Parse Error: <reasoning>...</reasoning> not found")
            logger.error(f"LLM Output: {llm_output}")
            raise ValueError("Parse Error: <reasoning>...</reasoning> not found")

        summary_content = extract_last_block(llm_output, "summary")
        if summary_content is None:
            logger.error("Parse Error: <summary>...</summary> not found")
            logger.error(f"LLM Output: {llm_output}")
            raise ValueError("Parse Error: <summary>...</summary> not found")
            
        if not think_content:
            logger.warning("empty <reasoning>")
        if not summary_content:
            logger.warning("empty <summary>")

        return think_content, summary_content

    except Exception as e:
        logger.error(f"{e}")
        raise e


def process_trajectory(trajectory):
    """
    Main processing function.
    Iterates through a trajectory, backfills <reasoning> tags.
    Includes full <think> content for the preceding n (1-4) steps in the history context.
    """
    
    last_n_reasoning = random.randint(1, 4)
    
    new_trajectory = []
    processed_history_buffer = [] 
    
    assistant_turn_counter = 0

    for turn in trajectory:
        current_role = turn["role"]
        
        if current_role == "assistant":
            assistant_turn_counter += 1
            
            # 1. Get the *original* summary (as hint) and action
            original_content = turn["content"]
            original_hint, target_action = _split_content_to_summary_and_action(original_content)
            
            if target_action is None:
                target_action = original_content # Was pure text

            # ==============================================================================
            # history_for_llm is re-generated。
            # ==============================================================================
            history_for_llm = []
            
            # find all indexed of assistant in buffer
            assistant_indices = [i for i, item in enumerate(processed_history_buffer) if item["role"] == "assistant"]
            
            # determine which thought should be remained
            indices_with_full_thought = set(assistant_indices[-last_n_reasoning:]) if assistant_indices else set()
            
            for i, item in enumerate(processed_history_buffer):
                if item["role"] == "assistant":
                    if i in indices_with_full_thought:
                        history_for_llm.append({"role": "assistant", "content": item["content_full"]})
                    else:
                        history_for_llm.append({"role": "assistant", "content": item["content_summary"]})
                else:
                    history_for_llm.append({"role": item["role"], "content": item["content_full"]})
            # ==============================================================================

            # 2. Use history, hint, and action to generate new <reasoning> and <summary>
            llm_output = generate_think_and_summary(
                history_for_llm, 
                original_hint,
                target_action,
                assistant_turn_counter
            )
            logger.debug(f"llm_output: {llm_output}")
            
            # 3. Parse the LLM output
            think_content, summary_content = parse_llm_output(llm_output)

            # 4. Construct the new content
            new_content_full = f"<reasoning>{think_content}</reasoning>\n{summary_content}\n{target_action}"
            new_content_summary = f"{summary_content}\n{target_action}"

            # Validate counts
            def count_words(content):
                think_count = content.count("<think>") + content.count("</think>")
                reasoning_count = content.count("<reasoning>") + content.count("</reasoning>")
                summary_count = content.count("<summary>") + content.count("</summary>")
                return think_count, reasoning_count, summary_count
                
            tc, rc, sc = count_words(new_content_full)
            if tc + rc > 2 or sc > 1:
                logger.warning(f"Weird content generated: {llm_output}\n\n\ncontent: {new_content_full}")
                return None

            new_turn = {
                "role": "assistant",
                "content": new_content_full
            }
            new_trajectory.append(new_turn)
            
            # 5. Update the processed_history_buffer
            processed_history_buffer.append({
                "role": "assistant",
                "content_full": new_content_full,
                "content_summary": new_content_summary
            })
            
        else:
            # System and User messages
            new_trajectory.append(turn)
            processed_history_buffer.append({
                "role": turn["role"],
                "content_full": turn["content"]
            })
            
    return new_trajectory, last_n_reasoning





# ==============================================================================
# 4. execute
# ==============================================================================

def get_processed_traj_ids(output_file):
    processed_ids = set()
    if os.path.exists(output_file):
        try:
            with jsonlines.open(output_file, mode='r') as reader:
                for obj in reader:
                    if 'traj_id' in obj:
                        processed_ids.add(obj['traj_id'])
        except Exception as e:
            logger.error(f"reading processed records error: {traceback.format_exc()}")
    return processed_ids

def process_single_data_safe(data, output_file):
    try:
        traj_id = data.get('traj_id', 'unknown')
        messages = data.get('messages', [])
        
        if not messages:
            return False
        
        final_augmented_trajectory, last_n_reasoning = process_trajectory(messages)
        
        if not final_augmented_trajectory:
            return False
        
        output_data = {
            'traj_id': traj_id,
            'model': model,
            'messages': final_augmented_trajectory,
            'last_n_reasoning': last_n_reasoning
        }
        
        with write_lock:
            with jsonlines.open(output_file, mode='a') as writer:
                writer.write(output_data)
                writer._fp.flush()
                os.fsync(writer._fp.fileno())
        
        return True
        
    except Exception as e:
        logger.error(f"processing traj_id {data.get('traj_id', 'unknown')}: {e}")
        output_data = {
            'traj_id': traj_id,
            'model': model,
            'error': str(e)
        }
        with write_lock:
            with jsonlines.open(output_file, mode='a') as writer:
                writer.write(output_data)
                writer._fp.flush()
                os.fsync(writer._fp.fileno())
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='backfilling')
    parser.add_argument('--input_file', type=str, 
                       default='/mnt/82_store/tmp/SWE-bench-agent/storage/zai-org/SWE-Dev-train/SWE-Dev-train-trajectories_rft_2276.jsonl')
    parser.add_argument('--output_file', type=str,
                       default='/mnt/69_store/tmp/SWE-bench-agent/storage/zai-org/SWE-Dev-train/SWE-Dev-fillback-v4-fixed.jsonl')
    parser.add_argument('--api_key', type=str, default='sk-xxx')
    parser.add_argument('--base_url', type=str, default='http://localhost:8000/v1')
    parser.add_argument('--model', type=str, default='/mnt/69_store/huggingface_cache/Qwen/Qwen3-235B-A22B-Instruct-2507')
    parser.add_argument('--max_workers', type=int, default=60, help='max async workers')
# --base_url https://openrouter.ai/api/v1
# --model qwen/qwen3-235b-a22b-2507
# --max_workers 16
# --base_url https://dashscope.aliyuncs.com/compatible-mode/v1
# --model qwen3-235b-a22b-instruct-2507
    args = parser.parse_args()
    
    api_key = args.api_key
    base_url = args.base_url
    model = args.model
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    input_file = args.input_file
    output_file = args.output_file
    max_workers = args.max_workers
    
    logger.info(f"Input: {input_file}")
    logger.info(f"Output: {output_file}")
    logger.info(f"Max Workers: {max_workers}")
    
    processed_ids = get_processed_traj_ids(output_file)
    logger.info(f"已完成: {len(processed_ids)} 条")
    
    pending_data = []
    total_input = 0
    with jsonlines.open(input_file, mode='r') as reader:
        for data in reader:
            total_input += 1
            traj_id = data.get('traj_id', f'unknown_{total_input}')
            if traj_id not in processed_ids:
                pending_data.append(data)
    logger.info(f"总数据: {total_input} 条")
    logger.info(f"待处理: {len(pending_data)} 条")
    pending_data.reverse()
    processed_count = 0
    error_count = 0
    
    if len(pending_data) > 0:
        logger.info("🚀 processing")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_traj = {
                executor.submit(process_single_data_safe, data, output_file): data.get('traj_id') 
                for data in pending_data
            }
            
            for future in tqdm(as_completed(future_to_traj), total=len(pending_data), desc="processing"):
                traj_id = future_to_traj[future]
                try:
                    success = future.result()
                    if success:
                        processed_count += 1
                    else:
                        error_count += 1
                except Exception as e:
                    logger.error(f" traj_id={traj_id}: {e}")
                    error_count += 1

    logger.info("\n" + "="*80)
    logger.info(f"processed_count: {processed_count}")
    logger.info(f"error_count: {error_count}")