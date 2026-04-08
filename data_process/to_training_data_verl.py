import jsonlines
import json
import os
import sys
import re
import pandas as pd
import argparse
import copy

from rllm.agents.prompts.mswe_myagent_prompts import *
from r2egym.agenthub.utils.log import get_logger, LOG_LEVEL_MAP
logger = get_logger(__name__, LOG_LEVEL_MAP[os.getenv("LOG_LEVEL", "DEBUG")])



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def extract_last_block(text, tag_name):
    """
    Extracts the content of the *last* valid pair of tags
    and returns (content, start_idx, end_idx)
    """
    start_tag = f"<{tag_name}>"
    end_tag = f"</{tag_name}>"

    end_idx = text.rfind(end_tag)
    if end_idx == -1:
        return None

    start_idx = text.rfind(start_tag, 0, end_idx)
    if start_idx == -1:
        return None

    content_start = start_idx + len(start_tag)
    content = text[content_start:end_idx].strip()

    block_start = start_idx
    block_end = end_idx + len(end_tag)

    return content, block_start, block_end


def filter_reasoning_in_messages(messages: list[dict], last_n_reasoning: int) -> list[dict]:
    """
    Filter reasoning content in messages based on last_n_reasoning.
    This follows the same logic as _tokenize_and_mask_stepwise in sft_dataset.py:
    - The last assistant message is the training target
    - Keep reasoning for the last_n_reasoning assistant messages before the target
    - Remove reasoning from earlier assistant messages

    The cutoff formula: i >= len(messages) - (last_n_reasoning + 1) * 2
    """
    if not messages:
        return messages

    # Calculate cutoff point
    cutoff = len(messages) - (last_n_reasoning + 1) * 2

    result = []
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            result.append(msg)
            continue

        if msg.get("role") == "assistant":
            if i >= cutoff:
                # Keep reasoning for messages within the window
                result.append(msg)
            else:
                # Remove reasoning for messages outside the window
                new_msg = {k: v for k, v in msg.items() if k != "reasoning"}
                result.append(new_msg)
        else:
            result.append(msg)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=r'''Process trajectory data to match Verl training data format. 
    This extracts <reasoning> into message["reasoning"] and constructs f"<reasoning_digest>{normal_text}</reasoning_digest>\n{function_text}".
    Whether to include "reasoning" is handled by the parser in agent_sft_trainer.py. Defaults to requiring transform_prompt.''')
    parser.add_argument('--input_file', type=str,
                       default='/mnt/69_store/tmp/SWE-bench-agent/storage/zai-org/SWE-Dev-train/SWE-Dev-fillback-v4-fixed.jsonl',
                       help='Path to input jsonl file')

    parser.add_argument('--transform_prompt', type=str2bool, default=True)
    parser.add_argument('--add_digest_tag', type=str2bool, default=True)
    parser.add_argument('--remove_digest_tag', type=str2bool, default=False)
    parser.add_argument('--scaffold', type=str, default="continuous_reasoning_window")
    parser.add_argument('--sft_mode', type=str, default='step', choices=['step', 'trajectory'],
                       help='Data construction mode: "step" (each step as a sample) or "trajectory" (entire trajectory as a sample)')
    parser.add_argument('--reasoning_filter', type=str, default='preprocess',
                       choices=['none', 'preprocess'],
                       help='''Reasoning filter mode:
                       - "none": Keep all reasoning, let sft_dataset.py handle it based on config during training
                       - "preprocess": Filter during preprocessing based on last_n_reasoning for each data item (suitable for fillback data)''')
    
    
    args = parser.parse_args()
    assert args.scaffold in ["continuous_reasoning_window", "dynamic_reasoning_window", "normal"]
    assert not (args.add_digest_tag and args.remove_digest_tag)

    input_file = args.input_file


    sft_data = []
    logger.info(f"Starting processing for {input_file}...")
    
    # Determine input format based on file extension
    file_ext = os.path.splitext(input_file)[1].lower()
    if file_ext == '.parquet':
        # Read parquet format
        df = pd.read_parquet(input_file)
        data_iter = df.to_dict('records')
    else:
        # Read jsonl format (original logic)
        with jsonlines.open(input_file, mode='r') as reader:
            data_iter = list(reader)
    
    for data in data_iter:
        skip_sample = False
        if data.get("messages") is None:
            continue
        messages = data['messages']
        
        if args.transform_prompt:
            # --- Start transformation logic ---
            # 1. Need to change prompt
            # 2. Need to replace keywords, e.g., workspace and str_replace_editor
            # 3. Need to extract content to message["reasoning"]
            transformed_messages = []
            problem_statement = ""  # Used to store the extracted problem description

            for i, msg in enumerate(messages):
                new_msg = copy.deepcopy(msg)
                
                if new_msg['role'] == 'system':
                    # 1. Replace System Prompt
                    if args.scaffold == "continuous_reasoning_window":
                        new_msg['content'] = MSWE_MYAGENT_SYSTEM_PROMPT_XML_CONTINUOUS_REASONING_WINDOW
                    elif args.scaffold == "dynamic_reasoning_window":
                        new_msg['content'] = MSWE_MYAGENT_SYSTEM_PROMPT_XML_DYNAMIC_REASONING_WINDOW
                    if args.scaffold == "normal":
                        new_msg['content'] = MSWE_MYAGENT_SYSTEM_PROMPT_XML_NORMAL                                                
                elif new_msg['role'] == 'user' and i==1:

                    # 2. Replace the first User Prompt (assuming i==1 is the main task)
                    content = new_msg['content']
                    
                    # 2a. Extract <pr_description> from SWEDEV_USER_PROMPT format
                    match = re.search(r"<pr_description>(.*?)</pr_description>", content, re.DOTALL)
                    
                    if match:
                        problem_statement = match.group(1).strip()
                    else:
                        skip_sample = True
                        logger.error(f"Could not find <pr_description> in traj_id {data.get('traj_id')}")
                        break
                    # 2b. Replace with MSWE User Prompt
                    test_desc_placeholder = PYTHON_TEST_DESCRIPTION
                    
                    # First replace {problem_statement}
                    temp_content = MSWE_USER_PROMPT_XML.format(
                        problem_statement=problem_statement
                    )
                    # Then replace {{test_description}} placeholder
                    new_msg['content'] = temp_content.replace(
                        "{test_description}", 
                        test_desc_placeholder
                    )
                    # Skip some examples to avoid incorrect keyword replacement
                    if 'workspace' in new_msg['content'] or 'str_replace_editor' in new_msg['content']:
                        skip_sample = True
                        break   # Break existing message loop, discard the whole sample
                else:
                    content = new_msg['content']
                    
                    # Replace tool names
                    content = content.replace("str_replace_editor", "file_editor")
                    content = content.replace("str_replace editor", "file_editor")
                    content = content.replace("workspace", "testbed")
                    
                    if new_msg['role'] == 'assistant':
                        # --- New logic starts: Extract reasoning and restructure content ---
                        # A. Extract
                        result = extract_last_block(content, "reasoning")
                        if result:
                            reasoning_content, start, end = result
                            new_msg["reasoning"] = reasoning_content

                            # Remove the entire <reasoning>...</reasoning>
                            content = (content[:start] + content[end:]).strip()
                        else:
                            logger.warning(f"no reasoning block in\n{content}")

                        func_start_index = content.find("<function=")
                        if func_start_index == -1:
                            logger.warning(f'''no '<function=' found in \n{content}''')
                            skip_sample = True
                            break 
                        
                        if args.add_digest_tag:
                            # B. Separate Normal Content and Function Call
                            # Find the position of <function=
                            # Split: first part is digest, second part is function
                            normal_text = content[:func_start_index].strip()
                            function_text = content[func_start_index:]
                            
                            if normal_text:
                                content = f"<reasoning_digest>{normal_text}</reasoning_digest>\n{function_text}"
                            else:
                                # If there is no text in between, keep the function part directly
                                content = function_text
     
                        elif args.remove_digest_tag:
                            # Remove <reasoning_digest>...</reasoning_digest> tags, keep content
                            # function call is outside </reasoning_digest>, need to keep it
                            digest_match = re.search(r"<reasoning_digest>(.*?)</reasoning_digest>", content, re.DOTALL)
                            if digest_match:
                                # Extract content inside tags
                                digest_content = digest_match.group(1).strip()
                                # Get content after tags (including function call)
                                after_digest = content[digest_match.end():].strip()
                                # Combine: digest content + subsequent content (function call, etc.)
                                if after_digest:
                                    content = f"{digest_content}\n{after_digest}"
                                else:
                                    content = digest_content
                                # Clean up excess newlines
                                content = re.sub(r'\n+', '\n', content).strip()
                            # If digest tag is not found, keep as is
                            
                    new_msg['content'] = content

                transformed_messages.append(new_msg)
            if skip_sample:
                continue
            # --- End transformation logic ---
        else:
            transformed_messages = messages

        # Decide whether to save Step or Trajectory based on arguments
        # Get last_n_reasoning (randomly generated or default value from fillback.py)
        last_n_reasoning = data.get('last_n_reasoning', 0)

        # Whether to apply reasoning filter during preprocessing
        apply_reasoning_filter = (args.reasoning_filter == 'preprocess')

        if args.sft_mode == 'step':
            # Generate one sample per step
            messages_step = []
            for msg in transformed_messages:
                messages_step.append(msg)
                if msg['role'] == 'assistant':
                    if apply_reasoning_filter:
                        # Apply last_n_reasoning filter to currently accumulated messages
                        filtered_messages = filter_reasoning_in_messages(
                            copy.deepcopy(messages_step), last_n_reasoning
                        )
                    else:
                        # Keep all reasoning, let training process handle it
                        filtered_messages = copy.deepcopy(messages_step)
                    sft_data.append({"messages": filtered_messages})

        elif args.sft_mode == 'trajectory':
            # The entire trajectory as one sample
            if apply_reasoning_filter:
                # filtered_messages = filter_reasoning_in_messages(
                #     copy.deepcopy(transformed_messages), last_n_reasoning
                # )
                raise ValueError("args.sft_mode == 'trajectory' and apply_reasoning_filter is not valid!")
            else:
                # Keep all reasoning, let training process handle it
                filtered_messages = copy.deepcopy(transformed_messages)
            sft_data.append({"messages": filtered_messages})
    # for i in range(3):
    #     logger.debug(sft_data[i])
    # Save results
    if sft_data:
        logger.info(f"Processed {len(sft_data)} SFT examples.")
        # Generate output filename from input file path: remove extension, add new naming format
        input_dir = os.path.dirname(input_file)
        input_basename = os.path.basename(input_file)
        input_name_without_ext = os.path.splitext(input_basename)[0]
        
        output_filename = f"{input_name_without_ext}_{args.scaffold}_{len(sft_data)}_{args.sft_mode}_{args.reasoning_filter}.parquet"
        output_path = os.path.join(input_dir, output_filename) if input_dir else output_filename
        pd.DataFrame(sft_data).to_parquet(output_path, index=False)
        print(f"Saved to: {output_path}")
        # Calculate statistics (consistent with original script logic)
        lengths = [len(" ".join([m["content"] for m in ex["messages"] if m["role"] == "assistant"])) for ex in sft_data]
        if lengths:
            logger.info(f"Saved {len(sft_data)} examples.")
        else:
            logger.info(f"Saved {len(sft_data)} examples.")
    else:
        logger.error("No valid SFT data was generated!")