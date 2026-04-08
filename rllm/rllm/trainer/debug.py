import os
import logging
import numpy as np
import pandas as pd
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.data import Subset
from transformers import AutoTokenizer
from rllm.agents.agent import Trajectory
from rllm.parser.chat_template_parser import ChatTemplateParser

os.environ['last_n_reasoning'] = "0"
class DebugDataSet:
    def __init__(self, tokenizer, config=None):

        self.tokenizer = tokenizer
        self.parser = ChatTemplateParser.get_parser(tokenizer)

    def _tokenize_and_mask(self, messages):
        # Now parser.parse don't need args `add_generation_prompt`, but depending on env args `last_n_reasoning` to parse thinking content (<think> or <reasoning>)
        for m in messages:
            tc = m.get("tool_calls")
            if isinstance(tc, np.ndarray):
                m["tool_calls"] = tc.tolist()
                
        if self.tokenize_and_mask_method == "cumulative":
            return self._tokenize_and_mask_cumulative(messages)
        elif self.tokenize_and_mask_method == "stepwise":
            return self._tokenize_and_mask_stepwise(messages)
        else:
            raise ValueError(f"Unknown tokenize_and_mask_method {self.tokenize_and_mask_method}")

    def _tokenize_and_mask_cumulative(self, messages):
        tokens = []
        loss_mask = []

        for i in range(len(messages)):
            parsed_msg = self.parser.parse([messages[i]], is_first_msg=(i == 0), add_generation_prompt=False)
            ids = self.tokenizer.encode(parsed_msg, add_special_tokens=False)
            if messages[i]["role"] == "assistant":
                loss_mask.extend([1] * len(ids))
            else:
                loss_mask.extend([0] * len(ids))
            tokens.extend(ids)

        return tokens, loss_mask

    def _tokenize_and_mask_stepwise(self, messages):
        tokens = []
        loss_mask = []
        texts = []
        last_n_reasoning = int(os.environ['last_n_reasoning'])
        # Find the index of the last assistant message
        last_assistant_idx = -1
        for i in range(len(messages)):
            if messages[i]["role"] == "assistant":
                last_assistant_idx = i
        assert last_assistant_idx != -1, "No assistant message found in chat_completions"

        for i in range(len(messages)):
            # Note that here only parse one step! So can't directly use export last_n_reasoning=2 in parse() to do last_n_reasoning training!
            if messages[i]["role"] == "assistant":
                # Note that here last assistant answer is training target, so last_n_reasoning should +1
                should_accumulate = i >= len(messages) - (last_n_reasoning + 1) * 2

                if i == last_assistant_idx:
                    # For last assistant message:
                    # 1. Add generation_prompt to prompt (loss_mask=0)
                    # 2. Add only content (without assistant header) to target (loss_mask=1)
                    # This ensures inference with add_generation_prompt=True works correctly

                    gen_prompt = self.parser.generation_prompt  # "<|im_start|>assistant\n"
                    gen_ids = self.tokenizer.encode(gen_prompt, add_special_tokens=False)
                    tokens.extend(gen_ids)
                    loss_mask.extend([0] * len(gen_ids))

                    # Build target WITHOUT assistant header
                    content = (messages[i].get("content", "") or "").strip()
                    reasoning = (messages[i].get("reasoning", "") or "").strip()

                    if should_accumulate and reasoning:
                        parsed_msg = f"{self.parser.think_token_begin}\n{reasoning}\n{self.parser.think_token_end}\n\n{content}{self.parser.eot_token}"
                    else:
                        parsed_msg = f"{content}{self.parser.eot_token}"
                    target_ids = self.tokenizer.encode(parsed_msg, add_special_tokens=False)
                    tokens.extend(target_ids)
                    loss_mask.extend([1] * len(target_ids))
                else:
                    # Non-target assistant: include full message with header
                    if should_accumulate:
                        parsed_msg = self.parser.parse_assistant(messages[i], accumulate_reasoning=True)
                    else:
                        parsed_msg = self.parser.parse_assistant(messages[i], accumulate_reasoning=False)
                    ids = self.tokenizer.encode(parsed_msg, add_special_tokens=False)
                    tokens.extend(ids)
                    loss_mask.extend([0] * len(ids))
            else:
                parsed_msg = self.parser.parse([messages[i]], is_first_msg=(i == 0), add_generation_prompt=False)
                ids = self.tokenizer.encode(parsed_msg, add_special_tokens=False)
                tokens.extend(ids)
                loss_mask.extend([0] * len(ids))
            messages[i]['content'] = parsed_msg
            messages[i].pop("reasoning")
        return tokens, loss_mask, messages


if __name__ == '__main__':
    
    model_path = "/mnt/82_store/huggingface_cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    train_files = "/mnt/69_store/tmp/SWE-bench-agent/storage/zai-org/SWE-Dev-train/SWE-Dev-fillbacked_23003_step.parquet"
    train_dataset = DebugDataSet(tokenizer)






    df = pd.read_parquet(train_files)

    chat_col = None
    possible_cols = ['messages', 'conversations', 'chat', 'chat_completions']
    for col in possible_cols:
        if col in df.columns:
            chat_col = col
            break


    for index, row in df.iterrows():
        
        # Extract messages. 
        # Parquet usually stores lists as numpy arrays or python lists.
        raw_messages = row[chat_col]
        
        # Ensure it is a pure python list of dicts
        if isinstance(raw_messages, np.ndarray):
            messages = raw_messages.tolist()
        else:
            messages = list(raw_messages)

        print(f"--- Processing Row {index} ---")
        
        for m in messages:
            tc = m.get("tool_calls")
            if isinstance(tc, np.ndarray):
                m["tool_calls"] = tc.tolist()

        if index == 6:
            tokens, loss_mask, messages = train_dataset._tokenize_and_mask_stepwise(messages)
            
            print(f"messages: {messages}")
            break


