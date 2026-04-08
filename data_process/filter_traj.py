import argparse
import json
from pathlib import Path

import pandas as pd


def count_assistant_messages(messages) -> int:
    """统计 messages 列表中 assistant 的条数，兼容 None / 非列表情况。"""
    if messages is None:
        return 0
    # pandas 读 parquet 时，可能是 numpy 数组 / Series
    if hasattr(messages, "tolist"):
        messages = messages.tolist()
    if not isinstance(messages, list):
        return 0
    return sum(
        1 for m in messages
        if isinstance(m, dict) and m.get("role") == "assistant"
    )


def filter_jsonl(path: Path, min_ass: int, max_ass: int) -> None:
    out_path = path.with_name(f"{path.stem}_assistant{min_ass}_{max_ass}{path.suffix}")

    kept = skipped_low = skipped_high = 0

    with path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            messages = obj.get("messages") or []
            n_assistant = count_assistant_messages(messages)
            if n_assistant < min_ass:
                skipped_low += 1
                continue
            if n_assistant > max_ass:
                skipped_high += 1
                continue
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1

    print("input :", path)
    print("output:", out_path)
    print("kept examples       :", kept)
    print(f"skipped (<{min_ass})       :", skipped_low)
    print(f"skipped (>{max_ass})       :", skipped_high)


def filter_parquet_snapshot(path: Path, min_ass: int, max_ass: int) -> None:
    """过滤 parquet 快照轨迹数据，假定有 messages 列。"""
    df = pd.read_parquet(path)
    if "messages" not in df.columns:
        raise ValueError(f"{path} 中未找到 'messages' 列，无法按 assistant 数过滤")
    counts = df["messages"].apply(count_assistant_messages)
    # 分别统计两端被过滤的数量
    mask_low = counts < min_ass
    mask_high = counts > max_ass
    mask_keep = ~(mask_low | mask_high)

    kept = int(mask_keep.sum())
    skipped_low = int(mask_low.sum())
    skipped_high = int(mask_high.sum())

    out_path = path.with_name(f"{path.stem}_assistant{min_ass}_{max_ass}{path.suffix}")
    df_filtered = df.loc[mask_keep]
    df_filtered.to_parquet(out_path, index=False)

    print("input parquet :", path)
    print("output parquet:", out_path)
    print("total examples      :", len(df))
    print("kept examples       :", kept)
    print(f"skipped (<{min_ass})       :", skipped_low)
    print(f"skipped (>{max_ass})       :", skipped_high)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="按 assistant 回答数过滤数据，支持 JSONL 和 parquet 快照轨迹。"
    )
    parser.add_argument("--input_file", type=str, help="输入文件路径（.jsonl 或 .parquet）")
    parser.add_argument("--min_assistant", type=int, required=True, help="保留的最少 assistant 条数（含）")
    parser.add_argument("--max_assistant", type=int, required=True, help="保留的最多 assistant 条数（含）")
    parser.add_argument(
        "--parquet_snapshot",
        action="store_true",
        help="输入是否为 parquet 格式的快照轨迹数据（有 messages 列）",
    )

    args = parser.parse_args()
    in_path = Path(args.input_file)

    if args.parquet_snapshot or in_path.suffix.lower() == ".parquet":
        filter_parquet_snapshot(in_path, args.min_assistant, args.max_assistant)
    else:
        # 默认按 JSONL 逐行处理
        filter_jsonl(in_path, args.min_assistant, args.max_assistant)