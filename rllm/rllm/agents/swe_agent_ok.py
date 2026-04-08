import sys
import os
import json
import os
from r2egym.agenthub.utils.log import get_logger, LOG_LEVEL_MAP
logger = get_logger(__name__, LOG_LEVEL_MAP[os.getenv("LOG_LEVEL", "INFO")])

import re
from typing import Tuple, Union
from openai import ChatCompletion


from r2egym.agenthub.action import Action as SWEAction


from rllm.agents.agent import Action, BaseAgent, Step, Trajectory
from rllm.engine.rollout.rollout_engine import ModelOutput
from rllm.agents.system_prompts import SWE_SYSTEM_PROMPT, SWE_SYSTEM_PROMPT_FN_CALL, SWE_USER_PROMPT, SWE_USER_PROMPT_FN_CALL, SWEAGENT_SYSTEM_PROMPT, SWEAGENT_USER_PROMPT
from rllm.agents.mswe_myagent_prompts import *
from rllm.parser.chat_template_parser import ChatTemplateParser

MSWE_USER_PROMPT_FN_CALL_MAP = {
    "python": MSWE_PYTHON_USER_PROMPT_FN_CALL,
    "java": MSWE_JAVA_USER_PROMPT_FN_CALL,
    "javascript": MSWE_JAVASCRIPT_USER_PROMPT_FN_CALL,
    "typescript": MSWE_TYPESCRIPT_USER_PROMPT_FN_CALL,
    "rust": MSWE_RUST_USER_PROMPT_FN_CALL,
    "go": MSWE_GO_USER_PROMPT_FN_CALL,
    "c": MSWE_C_USER_PROMPT_FN_CALL,
    "cpp": MSWE_CPP_USER_PROMPT_FN_CALL,
}

TOKEN_WARNING_THRESHOLD = 28000

import r2egym
R2EGYM_PATH = os.path.dirname(r2egym.__file__)
# Mapping of scaffold types to their tool schema definitions
# These are imported directly from R2E-Gym

def get_tools_for_scaffold(scaffold: str = "mswemyagent"):
    """
    Get the OpenAI function calling tools schema for a given scaffold.

    Args:
        scaffold: The scaffold type ("r2egym", "sweagent", or "mswemyagent")

    Returns:
        List of tool schemas in OpenAI function calling format
    """
    if scaffold == "mswemyagent":
        from r2egym.agenthub.tools.mswemyagent import (
            file_editor,
            search_tool,
            glob_file_tool,
            execute_bash_tool,
            finish_tool,
        )
        return [file_editor, search_tool, glob_file_tool, execute_bash_tool, finish_tool]

    from r2egym.agenthub.tools import (
        file_editor,
        search_tool,
        r2egym_bash_execute_tool,
        finish_tool,
        str_replace_editor_tool,
        execute_bash_tool,
        submit_tool,
    )

    if scaffold == "r2egym":
        return [
            file_editor,
            search_tool,
            r2egym_bash_execute_tool,
            finish_tool,
        ]
    elif scaffold == "sweagent":
        return [
            str_replace_editor_tool,
            execute_bash_tool,
            submit_tool,
        ]
    raise ValueError(f"Invalid scaffold: {scaffold}")


def parse_oai_response(response: ModelOutput) -> tuple[str, SWEAction]:
    if isinstance(response, ModelOutput):
        content = response.content
        if len(response.tool_calls) == 0:
            logger.warning(f"No tool calls found in the ModelOutput. Last 500 chars of the response: ...{response.text[-500:]} Returning empty action.")
            return content, SWEAction(function_name="", parameters={})
        if not isinstance(response.tool_calls[0].arguments, dict):
            logger.warning(f"Arguments is not a dict, got {type(response.tool_calls[0].arguments)}: {response.tool_calls[0].arguments}")
            response.tool_calls[0].arguments = {}
        action = SWEAction(function_name=response.tool_calls[0].name, parameters=response.tool_calls[0].arguments)
        return content, action
    else:
        raise ValueError(f"Invalid response type: {type(response)}. Expected ChatCompletion or ModelOutput object.")

def parse_xml_response(response_text: str) -> tuple[str, SWEAction]:
    """
    Extracts:
    - thought: everything before the first <function=...> block
    - action: the entire first <function=...></function> block
    Returns (thought, action).
    """
    # Regex to match (non-greedily) from `<function=` up to the first `</function>`
    pattern = re.compile(r"(?s)(<function=.*?</function>)")
    match = pattern.search(response_text)

    if match:
        action = match.group(1)  # The entire <function=...></function> block
        content = response_text[: match.start()]  # Everything before the block
    else:
        # If no match, treat entire text as "thought"
        content = response_text
        action = ""

    # Strip leading/trailing whitespace
    content = content.strip()
    action = action.strip()

    # convert action to Action object
    action = SWEAction.from_string(action)

    return content, action




class SWEAgent(BaseAgent):
    def __init__(self, use_tool_calling: bool = True, scaffold: str = "mswemyagent", chat_template_parser: ChatTemplateParser = None, accumulate_reasoning: bool = False, **kwargs):
        self.use_tool_calling = use_tool_calling
        self.scaffold = scaffold
        self.accumulate_reasoning = accumulate_reasoning
        assert scaffold in ["r2egym", "sweagent", "mswemyagent"], f"Invalid scaffold: {scaffold}, must be one of ['r2egym', 'sweagent', 'mswemyagent']"
        if scaffold == "sweagent":
            self.system_prompt = SWEAGENT_SYSTEM_PROMPT
            self.user_prompt_template = SWEAGENT_USER_PROMPT
        elif scaffold == "r2egym":
            self.system_prompt = SWE_SYSTEM_PROMPT_FN_CALL if use_tool_calling else SWE_SYSTEM_PROMPT
            self.user_prompt_template = SWE_USER_PROMPT_FN_CALL if use_tool_calling else SWE_USER_PROMPT
        elif scaffold == "mswemyagent":
            language = kwargs.get("language", "python")
            logger.debug(f"mswemyagent: language of the Agent is {language}")
            self.system_prompt = MSWE_MYAGENT_SYSTEM_PROMPT_FN_CALL
            self.user_prompt_template = MSWE_USER_PROMPT_FN_CALL_MAP[language]

        # Store tools for function calling
        self._tools_schema = None
        self.chat_template_parser = chat_template_parser
        self._trajectory = Trajectory()
        self.reset()



    def update_from_env(self, observation, reward, done, info):
        # Convert commands to tools schema on first step if using function calling
        if not self._trajectory.steps and self.use_tool_calling:
            self._tools_schema = get_tools_for_scaffold(self.scaffold)

        # If the first step in environment, we need to update the state from the environment
        if self._trajectory.steps:
            observation = str(observation)
        else:
            observation = str(observation)
            observation = self.user_prompt_template.format(problem_statement=observation)
            if self.accumulate_reasoning:
                observation = observation + "\n\nPrevious responses shows normal content and function call, not thinking content, so this time please think deeply and thoroughly. And you should include necessary information in the normal content for future reference."

        max_steps = info.get("max_steps", None)
        if max_steps:
            remaining_steps = max_steps - self.step - 1
            if remaining_steps > 0:
                observation += f"\nSteps Remaining: {remaining_steps}"
            else:
                observation += "\nYou have reached the maximum number of steps. Please submit your answer NOW."

        cur_tokens = info.get("cur_tokens", None)
        if cur_tokens is not None and cur_tokens >= TOKEN_WARNING_THRESHOLD:
            observation += "\nYou are running out of tokens. Please submit your answer NOW."

        if self._trajectory.steps:
            prior_step = self._trajectory.steps[-1]
            prior_step.next_observation = observation
            prior_step.reward = reward
            prior_step.done = done
            prior_step.info = info

        self.messages.append({"role": "user", "content": observation})
        self.cur_step = Step(observation=observation)

    def update_from_model(self, model_output: ModelOutput, **kwargs)->Action:
        """
        Updates the agent's internal state after an environment step.

        This function is called during environment interaction to incorporate the latest action's
        outcome into the agent's learning process.

        Args:
            model_output ModelOutput: The response from the model.
        Returns:
            Action: The action to take.
        """
        response = model_output.text
        self._trajectory.steps.append(self.cur_step)

        if self.use_tool_calling:
            content, action = parse_oai_response(model_output)
        else:
            content, action = parse_xml_response(response)
        if len(model_output.tool_calls) > 0:
            action_str = self.chat_template_parser.tool_parser.tool_call_to_str(model_output.tool_calls[0])
        else:
            action_str = ""
        logger.debug(f"update_from_model: action_str: {action_str}")
        assert self._trajectory.steps, "Trajectory should not be empty when update_from_model is called."


        # Update Trajectory
        cur_step = self._trajectory.steps[-1]
        cur_step.reasoning = model_output.reasoning
        cur_step.content = model_output.content
        cur_step.text = model_output.text
        cur_step.action = action
        cur_step.model_response = response  # TODO: add model response

        # Update Chat Completions

        self.messages.append({"role": "assistant", "content": response})
        self.step += 1
        return Action(action=cur_step.action)

    def get_current_state(self) -> Step:
        assert self._trajectory.steps, "Trajectory should not be empty when get_current_state is called."
        return self._trajectory.steps[-1]

    def get_tools(self):
        """Get OpenAI function calling tools schema if available."""
        if not self.use_tool_calling:
            raise ValueError("get_tools is only supported for function calling agents.")
        return self._tools_schema

    def reset(self):
        self._trajectory = Trajectory()
        self.messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            }
        ]
        self.step = 0

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    @property
    def chat_completions(self):
        return self.messages
