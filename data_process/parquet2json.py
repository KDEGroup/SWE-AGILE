import json
import os
import sys
import argparse
import pandas as pd
import numpy as np
from transformers import AutoTokenizer

def get_max_char_length(messages, key):
    """
    not using tokenizer, for fast sort.
    """
    if messages is None:
        return 0, ""
    
    if hasattr(messages, "tolist"):
        messages = messages.tolist()
        
    if not isinstance(messages, list):
        return 0, ""

    max_len = 0
    preview = ""
    
    for m in messages:
        if not isinstance(m, dict):
            continue
        if not m['role']=='assistant':
            continue
        val = m.get(key, "")
        if val:
            val_str = str(val)
            current_len = len(val_str)
            
            if current_len > max_len:
                max_len = current_len
                preview = val_str[:50].replace('\n', ' ') + "..."
                
    return max_len, preview

def get_exact_token_length(messages, key, tokenizer):
    if messages is None:
        return 0
    
    if hasattr(messages, "tolist"):
        messages = messages.tolist()
        
    if not isinstance(messages, list):
        return 0

    max_token_len = 0
    
    for m in messages:
        if not isinstance(m, dict):
            continue
        if not m['role']=='assistant':
            continue
        val = m.get(key, "")
        if val:
            val_str = str(val)
            ids = tokenizer.encode(val_str, add_special_tokens=False)
            current_len = len(ids)
            
            if current_len > max_token_len:
                max_token_len = current_len
                
    return max_token_len

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, 
                       default='/mnt/69_store/tmp/SWE-bench-agent/storage/collected_3007_r2e_steps_by_Qwen3Th_last2r.parquet')
    parser.add_argument('--output_file', type=str, default=None)
    parser.add_argument('--limit', type=int, default=20)
    
    parser.add_argument('--sort_field', type=str, choices=['reasoning', 'content'], default=None)
    parser.add_argument('--sort_mode', type=str, choices=['longest', 'shortest'], default='longest',
                       help="longest=Descending order, shortest=Ascending order")
    parser.add_argument('--tokenizer_path', type=str, 
                       default='/mnt/69_store/huggingface_cache/Qwen/Qwen3-14B')

    args = parser.parse_args()
    
    input_file = args.input_file
    output_file = args.output_file
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)

    print(f"Reading parquet: {input_file}...")
    try:
        df1 = pd.read_parquet(input_file, engine='fastparquet')
    except Exception as e:
        print(f"Error reading parquet: {e}")
        sys.exit(1)
    print(f"  - Shape: {df1.shape}")

    if args.sort_field:
        print(f"Processing sort: Field='{args.sort_field}', Mode='{args.sort_mode}'")
        
        print("  - Step 1: Sorting by character length (Heuristic)...")
        temp_char_data = df1['messages'].apply(lambda x: get_max_char_length(x, args.sort_field))
        df1['_char_len'] = temp_char_data.apply(lambda x: x[0])
        df1['_preview'] = temp_char_data.apply(lambda x: x[1])
        
        is_ascending = (args.sort_mode == 'shortest')
        df1 = df1.sort_values(by='_char_len', ascending=is_ascending)
        

        limit = args.limit
            
        print(f"  - Step 2: Selecting top {limit} candidates based on chars...")
        candidates = df1.head(limit).copy()
        
        print(f"  - Step 3: Loading Tokenizer and calculating tokens for {limit} rows...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            sys.exit(1)
            
        candidates['_token_len'] = candidates['messages'].apply(
            lambda x: get_exact_token_length(x, args.sort_field, tokenizer)
        )
        
        candidates = candidates.sort_values(by='_token_len', ascending=is_ascending)
        
        final_df = candidates.head(limit)
        
        print("\n" + "=" * 90)
        print(f"TOP {len(final_df)} Results (Filtered by chars -> Sorted by Token Count)")
        print(f"{'Index':<8} | {'Tokens':<8} | {'Chars':<8} | {'Preview'}")
        print("-" * 90)
        
        for idx, row in final_df.iterrows():
            t_len = row['_token_len']
            c_len = row['_char_len']
            prev = row['_preview']
            print(f"{idx:<8} | {t_len:<8} | {c_len:<8} | {prev}")
            
        print("=" * 90 + "\n")
        
        df1 = final_df.drop(columns=['_char_len', '_token_len', '_preview'])

    else:
        if args.limit:
            df1 = df1.head(args.limit)
            print(f"  - Sliced to first {args.limit} rows.")

    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        if args.sort_field:
            suffix = f"_{args.sort_mode}_{args.sort_field}_tokens"
        else:
            suffix = ""
        output_file = f"{base_name}{suffix}.json"

    print(f"  - Converting to JSON...")
    df1.to_json(
        output_file, 
        orient='records', 
        lines=False, 
        indent=4, 
        force_ascii=False
    )
    print(f"  - Saved to: {output_file}")