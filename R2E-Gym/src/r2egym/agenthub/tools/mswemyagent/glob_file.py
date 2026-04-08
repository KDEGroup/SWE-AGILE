#!/tools_uv_env/.venv/bin/python
"""
Description: Find files recursively using a glob pattern.
             Prioritizes 'rg' (ripgrep) for performance if available, 
             otherwise falls back to Python's built-in glob module.
             Results are sorted by modification time (newest first)
             and truncated to the first 100 files.

Parameters:
  --pattern (string, required): The glob pattern to match (e.g., "**/*.py", "src/**/*.{js,ts}").
  --path (string, optional):    The directory to search in. If not provided, 
                                defaults to the current working directory or the 
                                base path extracted from an absolute pattern.
"""

import argparse
import glob as glob_module
import os
import subprocess
import sys
import shutil
from pathlib import Path


def _check_ripgrep_available() -> bool:
    """Check if 'rg' (ripgrep) is available in the system path."""
    return shutil.which("rg") is not None


def _log_ripgrep_fallback_warning(tool_name: str, fallback: str):
    """Log a warning to stderr if ripgrep is not available."""
    print(
        f"Warning: '{tool_name}' is falling back to {fallback} because 'rg' (ripgrep) "
        "is not installed or not in PATH. For better performance, "
        "please install ripgrep.",
        file=sys.stderr,
    )


def _execute_with_ripgrep(
    pattern: str, search_path: Path
) -> tuple[list[str], bool]:
    """Execute glob pattern matching using ripgrep."""
    # rg --files {path} -g {pattern} --sortr=modified
    cmd = [
        "rg",
        "--files",
        str(search_path),
        "-g",
        pattern,
        "--sortr=modified",
    ]

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=30, check=False
    )

    file_paths = []
    if result.stdout:
        for line in result.stdout.strip().split("\n"):
            if line:
                file_paths.append(line)
                if len(file_paths) >= 100:
                    break

    truncated = len(file_paths) >= 100
    return file_paths, truncated


def expand_brace_patterns(patterns_list: list) -> list:
    """
    Expands shell-style brace patterns.
    E.g., ["*.{py,sh}", "*.txt"] becomes ["*.py", "*.sh", "*.txt"]
    
    This is a simple parser; it does not handle nested braces.
    """
    if not patterns_list:
        return []

    expanded_list = []
    for pattern in patterns_list:
        brace_open_idx = pattern.find('{')
        brace_close_idx = pattern.find('}')

        # 检查非嵌套的、有效的大括号模式
        if 0 <= brace_open_idx < brace_close_idx and pattern.find('{', brace_open_idx + 1) == -1:
            prefix = pattern[:brace_open_idx]
            suffix = pattern[brace_close_idx + 1:]
            
            middle_content = pattern[brace_open_idx + 1 : brace_close_idx]
            items = middle_content.split(',')
            
            if not items:
                expanded_list.append(pattern)
                continue
            
            for item in items:
                expanded_list.append(f"{prefix}{item}{suffix}")
        
        else:
            expanded_list.append(pattern)
            
    return expanded_list


def _execute_with_glob(
    pattern: str, search_path: Path
) -> tuple[list[str], bool]:
    """Execute glob pattern matching using Python's glob module, with brace expansion."""
    original_cwd = os.getcwd()
    try:
        os.chdir(search_path)

        # 扩展大括号模式，例如 **/*.{json,yaml}
        patterns_to_search = expand_brace_patterns([pattern])
        
        # 使用 set 来防止多个模式匹配到同一个文件时产生重复
        all_matches = set()

        for p in patterns_to_search:
            # Ripgrep 的 -g 标志总是递归的，所以我们需要使模式
            # 递归，如果它还不包含 **
            if "**" not in p:
                p_recursive = f"**/{p}"
            else:
                p_recursive = p

            # 使用 glob 查找匹配的文件
            try:
                matches = glob_module.glob(p_recursive, recursive=True)
                all_matches.update(matches)
            except Exception as e:
                # 捕获无效 glob 模式可能引发的错误
                print(f"Warning: Invalid glob pattern '{p_recursive}', skipping. Error: {e}", file=sys.stderr)

        # 转换为绝对路径（不解析符号链接）
        # 并获取修改时间
        file_paths = []
        for match in all_matches:
            abs_path = str((search_path / match).absolute())
            if os.path.isfile(abs_path):
                try:
                    file_paths.append((abs_path, os.path.getmtime(abs_path)))
                except OSError:
                    # 处理损坏的符号链接等情况
                    continue

        # 按修改时间排序（最新优先）并提取路径
        file_paths.sort(key=lambda x: x[1], reverse=True)
        sorted_files = [path for path, _ in file_paths[:100]]

        truncated = len(file_paths) > 100

        return sorted_files, truncated
    finally:
        os.chdir(original_cwd)


def _extract_search_path_from_pattern(pattern: str) -> tuple[Path | None, str]:
    """Extract search path and relative pattern from an absolute path pattern."""
    if not pattern:
        return None, "**/*"

    pattern = os.path.expanduser(pattern)

    if not pattern.startswith("/"):
        return None, pattern

    path_obj = Path(pattern)
    parts = path_obj.parts

    search_parts = []
    for part in parts:
        if glob_module.has_magic(part):
            break
        search_parts.append(part)

    if not search_parts:
        search_path = Path("/")
        adjusted_pattern = pattern.lstrip("/")
    else:
        search_path = Path(*search_parts)
        remaining = parts[len(search_parts) :]
        adjusted_pattern = str(Path(*remaining)) if remaining else "**/*"

    return search_path.resolve(), adjusted_pattern


def main():
    parser = argparse.ArgumentParser(
        description="Find files recursively using a glob pattern."
    )
    parser.add_argument(
        "--pattern",
        required=True,
        help='The glob pattern to match files (e.g., "**/*.js", "src/**/*.{js,ts}")',
    )
    parser.add_argument(
        "--path",
        default=None,
        help=(
            "The directory (absolute path) to search in. "
            "Defaults to the current working directory."
        ),
    )
    args = parser.parse_args()

    working_dir: Path = Path(os.getcwd()).resolve()
    ripgrep_available: bool = _check_ripgrep_available()
    # if not ripgrep_available:
    #     _log_ripgrep_fallback_warning("glob", "Python glob module (with brace expansion)")

    try:
        original_pattern = args.pattern

        if args.path:
            search_path = Path(args.path).resolve()
            pattern = args.pattern
        else:
            extracted_path, pattern = _extract_search_path_from_pattern(args.pattern)
            search_path = (
                extracted_path if extracted_path is not None else working_dir
            )

        if not search_path.is_dir():
            print("[STDOUT]\n", file=sys.stdout)
            print(f"Error: Search path '{search_path}' is not a valid directory", file=sys.stderr)
            sys.exit(1)

        if ripgrep_available:
            files, truncated = _execute_with_ripgrep(pattern, search_path)
        else:
            files, truncated = _execute_with_glob(pattern, search_path)

        # Format output similar to GlobObservation
        if not files:
            content = (
                f"No files found matching pattern '{original_pattern}' "
                f"in directory '{search_path}'"
            )
        else:
            file_list = "\n".join(files)
            content = (
                f"Found {len(files)} file(s) matching pattern "
                f"'{original_pattern}' in '{search_path}':\n{file_list}"
            )
            if truncated:
                content += (
                    "\n\n[Results truncated to first 100 files. "
                    "Consider using a more specific pattern.]"
                )
        
        print("[STDOUT]\n")
        print(content.strip())
        print("\n[STDERR]\n") # 成功时为空的 stderr

    except Exception as e:
        try:
            if args.path:
                error_search_path = str(Path(args.path).resolve())
            else:
                error_search_path = str(working_dir)
        except Exception:
            error_search_path = "unknown"

        print("[STDOUT]\n", file=sys.stdout) # 错误时为空的 stdout
        print("\n[STDERR]\n")
        print(f"Error executing glob with pattern '{args.pattern}' in path '{error_search_path}': {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()