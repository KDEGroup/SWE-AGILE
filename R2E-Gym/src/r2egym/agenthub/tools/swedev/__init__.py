##############################################################################
# tool definitions
##############################################################################

# Import allowed commands from the editor module

_STR_REPLACE_EDITOR_DESCRIPTION = """Custom editing tool for viewing, creating and editing files
* State is persistent across command calls and discussions with the user
* If `path` is a file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep
* The `create` command cannot be used if the specified `path` already exists as a file
* If a `command` generates a long output, it will be truncated and marked with `<response clipped>`
* The `undo_edit` command will revert the last edit made to the file at `path`
* If you want to delete some content, you can use `range_modify` or `str_replace` with empty `new_str`.

Notes for using the `str_replace` command:
* The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!
* If the `old_str` parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in `old_str` to make it unique
* The `new_str` parameter should contain the edited lines that should replace the `old_str`

VERY IMPORTANT: str_replace_editor old_str and new_str must be w/o the line numbers. line numbers are only shown in the view for clarity.
After a command like str_replace, insert, range_modify, undo_edit, line number of the file may change. So it's a good idea to view the file near the edited location before trying to use line number based commands.
"""

str_replace_editor = {
    "type": "function",
    "function": {
        "name": "str_replace_editor",
        "description": _STR_REPLACE_EDITOR_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "description": "The command to run. Allowed options are: `view`, `create`, `str_replace`, `insert`, `range_modify`, `undo_edit`.",
                    "enum": ["view", "create", "str_replace", "insert", "range_modify", "undo_edit"],
                    "type": "string",
                },
                "path": {
                    "description": "Absolute path to file or directory, e.g. `/testbed/file.py` or `/testbed`.",
                    "type": "string",
                },
                "file_text": {
                    "description": "Required for the `create` command, contains the content of the file to be created.",
                    "type": "string",
                },
                "old_str": {
                    "description": "Required for the `str_replace` command, specifies the string in `path` to replace.",
                    "type": "string",
                },
                "new_str": {
                    "description": "Optional for `str_replace` and `range_modify` commands. Required for `insert` to specify the new string.",
                    "type": "string",
                },
                "insert_line": {
                    "description": "Required for the `insert` command. The `new_str` will be inserted AFTER the line specified.",
                    "type": "integer",
                },
                "view_range": {
                    "description": "Optional for `view` command. Specifies the line range to view. E.g., [11, 12] shows lines 11 and 12. Indexing starts at 1. Use [start_line, -1] to show all lines from `start_line` to the end.",
                    "type": "array",
                    "items": {"type": "integer"},
                },
                "old_content_range": {
                    "description": "Required for `range_modify` command. The 1-based line range to replace [start_line, end_line]. E.g., [10, 12] replaces lines 10, 11, and 12. Use [start_line, -1] to replace to the end of the file.",
                    "type": "array",
                    "items": {"type": "integer"},
                },
                "concise": {
                    "description": "Optional for the `view` command. If `True`, displays a concise skeletal view of the file. Very useful for localization tasks. Highly recommended for large files.",
                    "type": "boolean",
                },
            },
            "required": ["command", "path"],
        },
    },
}





_BASH_DESCRIPTION = """
Description: Execute a bash command in the terminal.

Parameters:
  (1) command (string): The bash command to execute.

Examples: execute_bash --command "python my_script.py"
"""

execute_bash_tool = {
    "type": "function",
    "function": {
        "name": "execute_bash",
        "description": _BASH_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute.",
                }
            },
            "required": ["command"],
        },
    },
}

_FINISH_DESCRIPTION = """
Finish the interaction when the task is complete OR if the assistant cannot proceed further with the task.
* If no `--result` is provided, it defaults to an empty string.
**Parameters:**
  1. **result** (`string`, optional): The result text to submit. Defaults to an empty string.
"""
finish_tool = {
    "type": "function",
    "function": {
        "name": "finish",
        "description": _FINISH_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "result": {
                    "description": "Optional. The result text to submit. Defaults to an empty string if not provided.",
                    "type": "string",
                },
            },
            "required": [],
        },
        # "cache_control": {"type": "ephemeral"},
    },
}