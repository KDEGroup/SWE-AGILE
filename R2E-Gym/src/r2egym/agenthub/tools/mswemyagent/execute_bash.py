#!/tools_uv_env/.venv/bin/python
"""
Description: Execute a bash command in the terminal.

Parameters:
  (1) command (string): The bash command to execute.
      This can be provided either as a positional argument OR using the --command flag.

Examples:
  execute_bash.py "python my_script.py"
  execute_bash.py --command "python my_script.py"
"""

import argparse
import subprocess
import sys

BLOCKED_BASH_COMMANDS = ["git", "ipython", "jupyter", "nohup"]


def run_command(cmd):
    try:
        # Try to use the new parameters (Python 3.7+)
        return subprocess.run(cmd, shell=True, capture_output=True, text=True)
    except TypeError:
        # Fallback for Python 3.5 and 3.6:
        return subprocess.run(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Execute a bash command.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  execute_bash.py "ls -l"
  execute_bash.py --command "ls -l"
"""
    )
    
    # nargs='?' means it's optional
    parser.add_argument(
        "command_pos",
        nargs='?',
        default=None,
        help="The command to execute (positional method)."
    )
    
    parser.add_argument(
        "--command",
        dest="command_flag",
        default=None,
        help="The command to execute (flag method)."
    )
    
    args = parser.parse_args()

    command_to_run = None

    if args.command_pos and args.command_flag:
        parser.error("You cannot use both a positional command and the --command flag at the same time.")
    elif args.command_pos:
        command_to_run = args.command_pos
    elif args.command_flag:
        command_to_run = args.command_flag
    else:
        parser.error("You must provide a command, either positionally or with --command.")

    first_token = command_to_run.strip().split()[0]
    if first_token in BLOCKED_BASH_COMMANDS:
        print(
            f"Bash command '{first_token}' is not allowed. "
            "Please use a different command or tool."
        )
        sys.exit(1)

    result = run_command(command_to_run)

    if result.returncode != 0:
        print(f"Error executing command:\n")
        print("[STDOUT]\n")
        print(result.stdout.strip(), "\n")
        print("[STDERR]\n")
        print(result.stderr.strip())
        sys.exit(result.returncode)

    print("[STDOUT]\n")
    print(result.stdout.strip(), "\n")
    print("[STDERR]\n")
    print(result.stderr.strip())


if __name__ == "__main__":
    main()

