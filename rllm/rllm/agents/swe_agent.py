import os
import json
import os
import copy
from r2egym.agenthub.utils.log import get_logger, LOG_LEVEL_MAP
logger = get_logger(__name__, LOG_LEVEL_MAP[os.getenv("LOG_LEVEL", "INFO")])

import re
from typing import Tuple, Union
from openai import ChatCompletion


from r2egym.agenthub.action import Action as SWEAction


from rllm.agents.agent import Action, BaseAgent, Step, Trajectory
from rllm.engine.rollout.rollout_engine import ModelOutput
from rllm.agents.prompts.mswe_myagent_prompts import *
from rllm.agents.prompts.swedev import * 
from rllm.agents.prompts.r2egym import * 
from rllm.parser.chat_template_parser import ChatTemplateParser

MSWE_LANGUAGE_DESP_MAP = {
    "python": PYTHON_TEST_DESCRIPTION,
    "java": JAVA_TEST_DESCRIPTION,
    "javascript": JAVASCRIPT_TEST_DESCRIPTION,
    "typescript": TYPESCRIPT_TEST_DESCRIPTION,
    "rust": RUST_TEST_DESCRIPTION,
    "go": GO_TEST_DESCRIPTION,
    "c": C_TEST_DESCRIPTION,
    "cpp": CPP_TEST_DESCRIPTION,
}

TOKEN_WARNING_THRESHOLD = 28000

import r2egym
R2EGYM_PATH = os.path.dirname(r2egym.__file__)
# Mapping of scaffold types to their tool schema definitions
# These are imported directly from R2E-Gym

def get_tools_for_scaffold(scaffold: str = "continuous_reasoning_window"):
    """
    Get the OpenAI function calling tools schema for a given scaffold.
    Returns:
        List of tool schemas in OpenAI function calling format
    """
    if scaffold == "continuous_reasoning_window":
        from r2egym.agenthub.tools.mswemyagent import (
            file_editor,
            search_tool,
            glob_file_tool,
            execute_bash_tool,
            finish_tool,
        )
        return [file_editor, search_tool, glob_file_tool, execute_bash_tool, finish_tool]

    if scaffold == "swedev":
        from r2egym.agenthub.tools.swedev import (
            str_replace_editor,
            execute_bash_tool,
            finish_tool,
        )
        return [str_replace_editor, execute_bash_tool, finish_tool]

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


def parse_oai_response(response: ModelOutput) -> tuple[str, SWEAction]:
    if isinstance(response, ModelOutput):
        content = response.content
        if len(response.tool_calls) == 0:
            logger.warning(f"No tool calls found in the ModelOutput. Last 500 chars of the response: ...{response.text[-500:]} Returning empty action.")
            return content, SWEAction(function_name="", parameters={})
        if not isinstance(response.tool_calls[0].arguments, dict):
            logger.warning(f"Arguments is not a dict, got {type(response.tool_calls[0].arguments)}: {response.tool_calls[0].arguments}")
            response.tool_calls[0].arguments = {}
        
        function_name = response.tool_calls[0].name
        action = SWEAction(function_name=function_name, parameters=response.tool_calls[0].arguments)
        return content, action
    else:
        raise ValueError(f"Invalid response type: {type(response)}. Expected ChatCompletion or ModelOutput object.")

def parse_xml_response(response_text: str) -> tuple[str, SWEAction, str]:
    """
    Extracts:
    - thought: everything before the first <function=...> block
    - action: the entire first <function=...></function> block
    Returns (thought, action, action_text).
    """
    # Regex to match the last `<function=` `</function>`
    # fallback to <function=...>
    pattern = re.compile(r"(?s)(<function=.*?</function>)")

    matches = list(pattern.finditer(response_text))
    if matches:
        last = matches[-1]
        action_text = last.group(1)
        content = response_text[: last.start()]
    else:
        pattern = re.compile(r"(?s)(<function=.*?>)")
        match = pattern.search(response_text)

        if match:
            action_text = match.group(1)  # The entire <function=...></function> block
            content = response_text[: match.start()]  # Everything before the block
        else:
            # If no match, treat entire text as "thought"
            content = response_text
            action_text = ""

    # Strip leading/trailing whitespace
    content = content.strip()
    action_text = action_text.strip()

    # convert action to Action object
    action = SWEAction.from_string(action_text)
    return content, action, action_text




class SWEAgent(BaseAgent):
    def __init__(self, use_tool_calling: bool = True, scaffold: str = "continuous_reasoning_window", chat_template_parser: ChatTemplateParser = None, accumulate_reasoning: bool = False, **kwargs):
        self.use_tool_calling = use_tool_calling
        self.scaffold = scaffold
        self.accumulate_reasoning = accumulate_reasoning
        assert scaffold in ["r2egym", "sweagent", "normal", "continuous_reasoning_window", "dynamic_reasoning_window", "swedev"], f"Invalid scaffold: {scaffold}"
        if scaffold == "sweagent":
            self.system_prompt = SWEAGENT_SYSTEM_PROMPT
            self.user_prompt_template = SWEAGENT_USER_PROMPT
        elif scaffold == "r2egym":
            self.system_prompt = R2EGYM_SYSTEM_PROMPT_JSON if use_tool_calling else R2EGYM_SYSTEM_PROMPT_XML
            self.user_prompt_template = R2EGYM_USER_PROMPT_JSON if use_tool_calling else R2EGYM_USER_PROMPT_XML
        elif scaffold == "normal":
            language = kwargs.get("language", "python")
            logger.debug(f"Language of the Agent is {language}")
            if use_tool_calling:
                self.system_prompt = MSWE_MYAGENT_SYSTEM_PROMPT_JSON
                self.user_prompt_template = MSWE_USER_PROMPT_JSON.replace("{{test_description}}", MSWE_LANGUAGE_DESP_MAP[language]) 
            else:
                self.system_prompt = MSWE_MYAGENT_SYSTEM_PROMPT_XML_NORMAL
                self.user_prompt_template = MSWE_USER_PROMPT_XML.replace("{{test_description}}", MSWE_LANGUAGE_DESP_MAP[language])        
        elif scaffold == "continuous_reasoning_window":
            language = kwargs.get("language", "python")
            logger.debug(f"Language of the Agent is {language}")
            self.system_prompt = MSWE_MYAGENT_SYSTEM_PROMPT_XML_CONTINUOUS_REASONING_WINDOW
            self.user_prompt_template = MSWE_USER_PROMPT_XML.replace("{{test_description}}", MSWE_LANGUAGE_DESP_MAP[language])
        elif scaffold == "dynamic_reasoning_window":
            language = kwargs.get("language", "python")
            logger.debug(f"Language of the Agent is {language}")
            self.system_prompt = MSWE_MYAGENT_SYSTEM_PROMPT_XML_DYNAMIC_REASONING_WINDOW
            self.user_prompt_template = MSWE_USER_PROMPT_XML.replace("{{test_description}}", MSWE_LANGUAGE_DESP_MAP[language])
        elif scaffold == "swedev":
            self.system_prompt = SWEDEV_SYSTEM_PROMPT
            self.user_prompt_template = SWEDEV_USER_PROMPT
        self.chat_template_parser = chat_template_parser
        if self.use_tool_calling:
            tools_schema = json.dumps(get_tools_for_scaffold(scaffold))
            self.tools_prompt = self.chat_template_parser.tool_parser.get_tool_prompt(tools_schema)
        
        self._trajectory = Trajectory()
        self.reset()



    def update_from_env(self, observation, reward, done, info):
        # If the first step in environment, we need to update the state from the environment
        if self._trajectory.steps:
            observation = str(observation)
        else:
            observation = str(observation)
            observation = self.user_prompt_template.format(problem_statement=observation)
            # if self.accumulate_reasoning:
            #     observation = observation + "\n\nPrevious responses shows normal content and function call, not reasoning content, so this time please think deeply and thoroughly. And you should include necessary information in the normal content for future reference."

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
            prior_step = self.get_current_state()
            # somewhere need next_observation?
            prior_step.next_observation = observation
            prior_step.reward = reward
            prior_step.done = done
            prior_step.info = info

        # [STATE: Step 5] is a mark for recall_thought
        # self.messages.append({"role": "user", "content": f"[STATE: Step {self.step}]\n{observation}"})
        self.messages.append({"role": "user", "content": f"{observation}"})
        # last step will not be appended to agent.trajectory but appended to agent.chat_completions (messages)
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
        # logger.debug(f"reasoning in update_from_model: {model_output.reasoning}")
        response = model_output.text
        self._trajectory.steps.append(self.cur_step)

        if self.use_tool_calling:
            content, action = parse_oai_response(model_output)
            action_text = ""  # tool calling mode doesn't have raw action text
        else:
            content, action, action_text = parse_xml_response(response)
        if len(model_output.tool_calls) > 0:
            action_str = self.chat_template_parser.tool_parser.tool_call_to_str(model_output.tool_calls[0])
        else:
            action_str = ""
        # logger.debug(f"update_from_model: action_str: {action_str}")
        assert self._trajectory.steps, "Trajectory should not be empty when update_from_model is called."

        # Update self.messages
        self.messages.append({"role": "assistant", "content": model_output.content, "reasoning": model_output.reasoning, "tool_calls": model_output.tool_calls})

        # Update cur_step
        cur_step = self.get_current_state()
        cur_step.reasoning = model_output.reasoning
        cur_step.content = model_output.content
        cur_step.text = model_output.text
        cur_step.action = action
        cur_step.action_text = action_text
        cur_step.model_response = response
        cur_step.chat_completions = copy.deepcopy(self.chat_completions)

        self.step += 1
        return Action(action=cur_step.action)

    def get_current_state(self) -> Step:
        assert self._trajectory.steps, "Trajectory should not be empty when get_current_state is called."
        return self._trajectory.steps[-1]


    def reset(self):
        self._trajectory = Trajectory()
        if self.use_tool_calling:
            self.messages = [{"role": "system", "content": self.system_prompt + self.tools_prompt + JSON_EXAMPLE}]
        else:
            self.messages = [{"role": "system", "content": self.system_prompt}]
        self.step = 0

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    @property
    def chat_completions(self):
        return self.messages

    def check_format(self):
        """
        检查每个 assistant 消息是否都包含 'reasoning' 字段且不为空。
        返回 True 如果所有 assistant 消息都符合格式要求，否则返回 False。
        """
        for message in self.chat_completions:
            if message.get("role") == "assistant":
                reasoning = message.get("reasoning")
                if reasoning is None or reasoning == "":
                    return False
        return True

    @staticmethod
    def same_action(step_a: Step, step_b: Step) -> bool:
        """Compare actions by function_name, parameters and raw action_text to detect repeats.

        This compares both the parsed action objects AND the raw action text to avoid
        false positives when different malformed action texts are parsed into identical
        (possibly empty) action objects.

        Args:
            step_a: First step to compare
            step_b: Second step to compare

        Returns:
            True if the actions are considered the same, False otherwise
        """
        if step_a is None or step_b is None:
            return False
        # Compare raw action text
        action_text_a = getattr(step_a, "action_text", "")
        action_text_b = getattr(step_b, "action_text", "")
        return (action_text_a == action_text_b)