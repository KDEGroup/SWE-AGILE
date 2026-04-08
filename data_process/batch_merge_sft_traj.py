#!/usr/bin/env python3
"""
批量合并 sft_traj_*.parquet 文件。

使用 merge_verl_datasets.py 来两两合并多个 parquet 文件。
"""

import os
import sys
import glob
import subprocess
import argparse
from pathlib import Path


def find_sft_traj_files(input_dir):
    """
    查找所有 sft_traj_*.parquet 文件。
    
    Args:
        input_dir: 输入目录路径
        
    Returns:
        排序后的文件列表
    """
    pattern = os.path.join(input_dir, 'sft_step_*.parquet')
    files = glob.glob(pattern)
    files.sort()
    return files


def merge_two_files(merge_script, file1, file2, output_file):
    """
    使用 merge_verl_datasets.py 合并两个文件。
    
    Args:
        merge_script: merge_verl_datasets.py 的路径
        file1: 第一个输入文件
        file2: 第二个输入文件
        output_file: 输出文件路径
        
    Returns:
        True 如果成功，False 否则
    """
    cmd = [
        sys.executable,
        merge_script,
        '--input_file1', file1,
        '--input_file2', file2,
        '--output_file', output_file
    ]
    
    print(f"\n合并文件:")
    print(f"  文件1: {os.path.basename(file1)}")
    print(f"  文件2: {os.path.basename(file2)}")
    print(f"  输出: {os.path.basename(output_file)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("警告:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"错误: 合并失败")
        print(f"  命令: {' '.join(cmd)}")
        print(f"  返回码: {e.returncode}")
        if e.stdout:
            print(f"  标准输出: {e.stdout}")
        if e.stderr:
            print(f"  标准错误: {e.stderr}")
        return False


def batch_merge_files(merge_script, input_files, output_file, temp_dir=None, keep_temp=False):
    """
    批量合并多个文件。
    
    策略：逐步合并，每次将一个新文件合并到累积结果中。
    
    Args:
        merge_script: merge_verl_datasets.py 的路径
        input_files: 输入文件列表
        output_file: 最终输出文件路径
        temp_dir: 临时文件目录（如果为None，使用输出文件所在目录）
        keep_temp: 是否保留临时文件
        
    Returns:
        True 如果成功，False 否则
    """
    if len(input_files) == 0:
        print("错误: 没有找到需要合并的文件")
        return False
    
    if len(input_files) == 1:
        print(f"只有一个文件，直接复制到输出位置")
        import shutil
        shutil.copy2(input_files[0], output_file)
        return True
    
    # 设置临时文件目录
    if temp_dir is None:
        temp_dir = os.path.dirname(output_file)
    os.makedirs(temp_dir, exist_ok=True)
    
    # 第一步：合并前两个文件
    current_result = os.path.join(temp_dir, 'temp_merged_0.parquet')
    if not merge_two_files(merge_script, input_files[0], input_files[1], current_result):
        return False
    
    # 后续步骤：将剩余文件逐个合并到累积结果中
    for i, next_file in enumerate(input_files[2:], start=1):
        next_result = os.path.join(temp_dir, f'temp_merged_{i}.parquet')
        if not merge_two_files(merge_script, current_result, next_file, next_result):
            # 清理临时文件
            if not keep_temp:
                if os.path.exists(current_result):
                    os.remove(current_result)
            return False
        
        # 删除旧的临时文件（如果不是第一个）
        if not keep_temp and i > 1:
            prev_temp = os.path.join(temp_dir, f'temp_merged_{i-1}.parquet')
            if os.path.exists(prev_temp):
                os.remove(prev_temp)
        
        current_result = next_result
    
    # 将最终结果移动到输出位置
    if current_result != output_file:
        import shutil
        shutil.move(current_result, output_file)
        print(f"\n最终结果已保存到: {output_file}")
    
    # 清理最后一个临时文件（如果还在）
    if not keep_temp:
        if os.path.exists(current_result) and current_result != output_file:
            os.remove(current_result)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='批量合并 sft_traj_*.parquet 文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 合并指定目录下的所有 sft_traj_*.parquet 文件
  python batch_merge_sft_traj.py \\
      --input_dir /mnt/69_store/lsq/SWE-bench-agent/storage/sft_outputs \\
      --output_file /mnt/69_store/lsq/SWE-bench-agent/storage/sft_outputs/merged_sft_traj.parquet
  
  # 指定要合并的文件列表
  python batch_merge_sft_traj.py \\
      --input_files file1.parquet file2.parquet file3.parquet \\
      --output_file merged.parquet
        """
    )
    
    parser.add_argument(
        '--input_dir',
        type=str,
        help='包含 sft_traj_*.parquet 文件的目录'
    )
    parser.add_argument(
        '--input_files',
        type=str,
        nargs='+',
        help='要合并的文件列表（如果指定，则忽略 --input_dir）'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help='输出文件路径'
    )
    parser.add_argument(
        '--merge_script',
        type=str,
        default='/mnt/69_store/lsq/tmp/SWE-AGILE/data_process/merge_verl_datasets.py',
        help='merge_verl_datasets.py 脚本的路径'
    )
    parser.add_argument(
        '--temp_dir',
        type=str,
        default=None,
        help='临时文件目录（默认使用输出文件所在目录）'
    )
    parser.add_argument(
        '--keep_temp',
        action='store_true',
        help='保留临时文件（用于调试）'
    )
    
    args = parser.parse_args()
    
    # 获取输入文件列表
    if args.input_files:
        input_files = args.input_files
        # 验证文件是否存在
        for f in input_files:
            if not os.path.exists(f):
                print(f"错误: 文件不存在: {f}")
                sys.exit(1)
    elif args.input_dir:
        if not os.path.isdir(args.input_dir):
            print(f"错误: 目录不存在: {args.input_dir}")
            sys.exit(1)
        input_files = find_sft_traj_files(args.input_dir)
        if len(input_files) == 0:
            print(f"错误: 在目录 {args.input_dir} 中没有找到 sft_traj_*.parquet 文件")
            sys.exit(1)
    else:
        print("错误: 必须指定 --input_dir 或 --input_files")
        parser.print_help()
        sys.exit(1)
    
    # 验证 merge_script 是否存在
    if not os.path.exists(args.merge_script):
        print(f"错误: merge_verl_datasets.py 不存在: {args.merge_script}")
        sys.exit(1)
    
    print("=" * 60)
    print("批量合并 sft_traj 文件")
    print("=" * 60)
    print(f"\n找到 {len(input_files)} 个文件:")
    for i, f in enumerate(input_files, 1):
        file_size = os.path.getsize(f) / (1024 * 1024)  # MB
        print(f"  {i}. {os.path.basename(f)} ({file_size:.2f} MB)")
    print(f"\n输出文件: {args.output_file}")
    print(f"使用合并脚本: {args.merge_script}")
    
    # 确认
    response = input("\n是否继续? (y/n): ")
    if response.lower() != 'y':
        print("已取消")
        sys.exit(0)
    
    # 执行合并
    success = batch_merge_files(
        merge_script=args.merge_script,
        input_files=input_files,
        output_file=args.output_file,
        temp_dir=args.temp_dir,
        keep_temp=args.keep_temp
    )
    
    if success:
        print("\n" + "=" * 60)
        print("合并完成!")
        print("=" * 60)
        if os.path.exists(args.output_file):
            file_size = os.path.getsize(args.output_file) / (1024 * 1024)  # MB
            print(f"输出文件: {args.output_file}")
            print(f"文件大小: {file_size:.2f} MB")
    else:
        print("\n" + "=" * 60)
        print("合并失败!")
        print("=" * 60)
        sys.exit(1)


if __name__ == '__main__':
    main()
