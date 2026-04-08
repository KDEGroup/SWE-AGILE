# /data/lsq/tmp/SWE-AGILE/patch_pandas_parquet.py
import pandas as pd
import polars as pl

def read_parquet_with_polars(path, *args, **kwargs):
    # 只处理你关心的 parquet，或者干脆对所有 parquet 都用 polars
    if isinstance(path, str):
        df_pl = pl.read_parquet(path)
        # 注意：这里不指定 pyarrow engine，而是直接让 Polars 做转换
        return df_pl.to_pandas()
    else:
        # 兜底调用原始实现
        return _orig_read_parquet(path, *args, **kwargs)

_orig_read_parquet = pd.read_parquet
pd.read_parquet = read_parquet_with_polars
import jsonlines
import json
import os
import sys

import pandas as pd
import argparse
import copy

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='merge 2 verl datasets')
    parser.add_argument('--input_file1', type=str, 
                       default='/data/tmp/SWE-bench-agent/storage/SWE-bench/SWE-smith-trajectories/SWE-smith-trajectories_xml_6445_fillback.parquet')
    parser.add_argument('--input_file2', type=str, 
                       default='/data/tmp/SWE-bench-agent/storage/zai-org/SWE-Dev-train/SWE-Dev-train-trajectories_rft_2276_fillback_tmp.parquet')
    parser.add_argument('--output_file', type=str,
                       default='/data/tmp/SWE-bench-agent/storage/SWE-bench/SWE-smith-trajectories/smith_swedev_fillback.parquet',
                       help='output parquet path')
    parser.add_argument('--shuffle', type=str2bool, default=False, help='shuffle the dataset')
    
    args = parser.parse_args()
    
    input_file1 = args.input_file1
    input_file2 = args.input_file2
    output_file = args.output_file


    try:
        df1 = pd.read_parquet(input_file1)
        print(f"df1 shape: {df1.shape}")
        
        df2 = pd.read_parquet(input_file2)
        print(f"df2 shape: {df2.shape}")

        combined_df = pd.concat([df1, df2], ignore_index=True)
        print(f"new dataset shape: {combined_df.shape}")

        # random_seed = 42
        # shuffled_df = combined_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"  - 已创建输出目录: {output_dir}")

        combined_df.to_parquet(output_file, index=False)
        
        print(f"saved at {output_file}")
    except Exception as e:
        print(f"\n合并过程中发生错误: {e}")
