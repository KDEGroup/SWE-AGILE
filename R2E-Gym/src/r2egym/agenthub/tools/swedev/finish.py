#!/tools_uv_env/.venv/bin/python
"""
Description:
Finish the interaction when the task is complete OR if the assistant cannot proceed further with the task.
If no `result` is provided, it defaults to an empty string.

**Parameters:**
  1. **result** (`string`, optional): The result text to submit. Defaults to an empty string.
"""

import argparse
import sys


def submit(result: str = ""):
    """
    Submits a final result, printing a message that includes the result.
    """
    print("<<<Finished>>>")
    if result:
        print(f"Final result submitted: {result}")
    # You can add more logic here as needed


def main():
    parser = argparse.ArgumentParser(
        description="submit tool: run the `submit` command with an optional `--result` argument."
    )
    parser.add_argument(
        "--result", help="The result text to submit (optional).", default=""
    )

    args = parser.parse_args()

    submit(args.result)



if __name__ == "__main__":
    main()
