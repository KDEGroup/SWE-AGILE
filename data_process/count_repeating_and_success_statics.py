#!/usr/bin/env python3
"""
统计轨迹数据中的repeating和成功失败情况。

统计内容：
1. 总共有多少个轨迹是repeating的
2. 总共有多少个成功和失败
3. repeating里有多少成功失败
"""

import json
import re
from typing import List, Dict, Any


def extract_action_text(content: str) -> str:
    """
    从assistant消息的content中提取action文本。
    
    Args:
        content: assistant消息的content字段
        
    Returns:
        提取的action文本（<function=...></function>部分），如果没有则返回空字符串
    """
    if not content:
        return ""
    
    # 提取<function=...></function>块
    pattern = r'<function=.*?</function>'
    match = re.search(pattern, content, re.DOTALL)
    if match:
        return match.group(0).strip()
    return ""


def is_repeating_trajectory(trajectory: Dict[str, Any]) -> bool:
    """
    判断一个轨迹是否是repeating的。
    
    repeating定义：连续3个步骤执行了相同的action（通过action_text比较）。
    或者termination_reason为"REPEATING"。
    
    Args:
        trajectory: 轨迹数据字典
        
    Returns:
        True如果是repeating，False否则
    """
    # 首先检查termination_reason
    termination_reason = trajectory.get('termination_reason', '')
    if termination_reason == 'REPEATING':
        return True
    
    # 从chat_completions中提取action序列
    chat_completions = trajectory.get('chat_completions', [])
    actions = []
    
    for msg in chat_completions:
        if msg.get('role') == 'assistant':
            content = msg.get('content', '')
            action_text = extract_action_text(content)
            if action_text:  # 只记录非空的action
                actions.append(action_text)
    
    # 检查是否有连续3个相同的action
    if len(actions) >= 6:
        for i in range(len(actions) - 2):
            if actions[i] == actions[i+1] == actions[i+2]:
                return True
    
    return False


def is_successful_trajectory(trajectory: Dict[str, Any]) -> bool:
    """
    判断一个轨迹是否成功。
    
    Args:
        trajectory: 轨迹数据字典
        
    Returns:
        True如果成功（trajectory_reward > 0），False否则
    """
    reward = trajectory.get('trajectory_reward', 0.0)
    return reward > 0


def count_statistics(trajectories: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    统计轨迹数据。
    
    Args:
        trajectories: 轨迹数据列表
        
    Returns:
        统计结果字典
    """
    total_trajectories = len(trajectories)
    repeating_count = 0
    success_count = 0
    failure_count = 0
    repeating_success_count = 0
    repeating_failure_count = 0
    
    for traj in trajectories:
        is_repeating = is_repeating_trajectory(traj)
        is_success = is_successful_trajectory(traj)
        
        if is_repeating:
            repeating_count += 1
            if is_success:
                repeating_success_count += 1
            else:
                repeating_failure_count += 1
        
        if is_success:
            success_count += 1
        else:
            failure_count += 1
    
    return {
        'total_trajectories': total_trajectories,
        'repeating_count': repeating_count,
        'non_repeating_count': total_trajectories - repeating_count,
        'success_count': success_count,
        'failure_count': failure_count,
        'repeating_success_count': repeating_success_count,
        'repeating_failure_count': repeating_failure_count,
        'non_repeating_success_count': success_count - repeating_success_count,
        'non_repeating_failure_count': failure_count - repeating_failure_count,
    }


def print_statistics(stats: Dict[str, Any]):
    """
    打印统计结果。
    
    Args:
        stats: 统计结果字典
    """
    print("=" * 60)
    print("轨迹统计结果")
    print("=" * 60)
    print(f"\n总轨迹数: {stats['total_trajectories']}")
    print(f"\n--- Repeating统计 ---")
    print(f"Repeating轨迹数: {stats['repeating_count']}")
    print(f"  其中成功: {stats['repeating_success_count']}")
    print(f"  其中失败: {stats['repeating_failure_count']}")
    print(f"\n非Repeating轨迹数: {stats['non_repeating_count']}")
    print(f"  其中成功: {stats['non_repeating_success_count']}")
    print(f"  其中失败: {stats['non_repeating_failure_count']}")
    print(f"\n--- 总体成功失败统计 ---")
    print(f"成功轨迹数: {stats['success_count']}")
    print(f"失败轨迹数: {stats['failure_count']}")
    print(f"\n--- 比例统计 ---")
    if stats['total_trajectories'] > 0:
        print(f"Repeating比例: {stats['repeating_count'] / stats['total_trajectories'] * 100:.2f}%")
        print(f"成功率: {stats['success_count'] / stats['total_trajectories'] * 100:.2f}%")
        if stats['repeating_count'] > 0:
            print(f"Repeating中成功率: {stats['repeating_success_count'] / stats['repeating_count'] * 100:.2f}%")
        if stats['non_repeating_count'] > 0:
            print(f"非Repeating中成功率: {stats['non_repeating_success_count'] / stats['non_repeating_count'] * 100:.2f}%")
    print("=" * 60)


def main():
    """主函数"""
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python count_repeating_and_success_statics.py <trajectory_json_file>")
        print("\n示例:")
        print("  python count_repeating_and_success_statics.py /path/to/trajectories.json")
        sys.exit(1)
    
    json_file = sys.argv[1]
    
    print(f"正在读取轨迹文件: {json_file}")
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            trajectories = json.load(f)
    except Exception as e:
        print(f"错误: 无法读取文件 {json_file}: {e}")
        sys.exit(1)
    
    print(f"成功加载 {len(trajectories)} 个轨迹")
    print("正在统计...")
    
    stats = count_statistics(trajectories)
    print_statistics(stats)
    
    # 可选：保存统计结果到JSON文件
    output_file = json_file.replace('.json', '_statistics.json')
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"\n统计结果已保存到: {output_file}")
    except Exception as e:
        print(f"\n警告: 无法保存统计结果到文件: {e}")


if __name__ == '__main__':
    main()
