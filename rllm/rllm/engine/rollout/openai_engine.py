import asyncio
import logging
import os
import traceback
import openai

from rllm.engine.rollout.rollout_engine import ModelOutput, RolloutEngine, TaskStuck
from rllm.globals import THOUGHT_DELIMITER_END, THOUGHT_DELIMITER_START
from rllm.parser import ChatTemplateParser
from rllm.tools.tool_base import Tool
from rllm.workflows import TerminationEvent, TerminationReason
from rllm.parser.tool_parser import ToolParseError

import traceback
from r2egym.agenthub.utils.log import get_logger, LOG_LEVEL_MAP
logger = get_logger(__name__, LOG_LEVEL_MAP[os.getenv("LOG_LEVEL", "INFO")])


class OpenAIEngine(RolloutEngine):
    def __init__(self, model: str = "", tokenizer=None, max_prompt_length: int = 4096, max_response_length: int = 4096, max_model_length: int | None = None, api_retries: int = 3, base_url: str = "https://api.openai.com/v1", api_key: str = os.getenv("OPENAI_API_KEY"), sampling_params: dict | None = None, tools: list[Tool | dict] = None, accumulate_reasoning: bool = False, **kwargs):
        self.model = model
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.max_model_length = max_model_length - 1 if max_model_length is not None else max_prompt_length + max_response_length - 1
        self.api_retries = api_retries
        self.sampling_params = sampling_params or {}
        self.tools = tools or []
        self.accumulate_reasoning = accumulate_reasoning

        self.tokenizer = tokenizer
        if self.tokenizer is not None:
            self.chat_parser = ChatTemplateParser.get_parser(self.tokenizer, disable_thinking=kwargs.get("disable_thinking", False))
            self._use_chat_completions = False
        else:
            # In this case, we cannot enforce max prompt length or dynamically adjust max_tokens <= max_response_length if needed
            print("No tokenizer provided to OpenAIEngine, will use the chat completions endpoint.")
            self._use_chat_completions = True

        self.client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key)
        logging.getLogger("httpx").setLevel(logging.WARNING)

    def _prepare_max_tokens_param(self, sampling_params: dict, prompt_length: int = None) -> dict:
        """Prepare max tokens parameter for API call (supports O3's max_completion_tokens)."""
        if "max_completion_tokens" in sampling_params:
            return {"max_completion_tokens": sampling_params.pop("max_completion_tokens")}

        max_tokens = sampling_params.pop("max_tokens", sampling_params.pop("max_new_tokens", self.max_response_length))

        # Adjust for prompt length if provided (completion method needs this)
        if prompt_length and self.max_model_length:
            remaining = self.max_model_length - prompt_length
            if remaining <= max_tokens:
                max_tokens = remaining if remaining > 0 else 1
                logger.debug(f"OpenAIEngine: Decreasing max_tokens to {max_tokens} to stay within max_model_length")

        return {"max_tokens": max_tokens}

    async def chat_completion(self, messages: list[dict], **kwargs) -> ModelOutput:
        kwargs.pop("application_id", None)
        kwargs.pop("validate", None)
        kwargs.pop("model", None)
        kwargs.pop("enforce_max_prompt_length", None)
        kwargs.pop("meta_info", None)  # Remove meta_info as it's not a valid OpenAI API parameter

        sampling_params = self.sampling_params.copy()
        sampling_params.update(kwargs)

        create_params = self._prepare_max_tokens_param(sampling_params)

        retries = self.api_retries
        while retries > 0:
            try:
                response = await self.client.chat.completions.create(model=self.model, messages=messages, timeout=3600, **create_params, **sampling_params)

                content = response.choices[0].message.content
                reasoning = response.choices[0].message.reasoning if hasattr(response.choices[0].message, "reasoning") and isinstance(response.choices[0].message.reasoning, str) else ""
                tool_calls = response.choices[0].message.tool_calls if hasattr(response.choices[0].message, "tool_calls") and isinstance(response.choices[0].message.tool_calls, list) else []

                # Build text with reasoning if available, otherwise use content
                if reasoning:
                    text = f"{THOUGHT_DELIMITER_START}\n{reasoning}\n{THOUGHT_DELIMITER_END}\n\n{content}"
                else:
                    text = content

                prompt_length = response.usage.prompt_tokens
                completion_length = response.usage.completion_tokens
                finish_reason = response.choices[0].finish_reason

                return ModelOutput(
                    text=text,
                    content=content,
                    reasoning=reasoning,
                    tool_calls=tool_calls,
                    prompt_ids=[],
                    completion_ids=[],
                    logprobs=[],
                    prompt_length=prompt_length,
                    completion_length=completion_length,
                    finish_reason=finish_reason,
                )

            except openai.RateLimitError:
                retries -= 1
                if retries == 0:
                    raise Exception("Rate limit reached and retries exhausted.") from None
                print("Sleep for 5 seconds for API limit.")
                await asyncio.sleep(5)

    async def completion(self, prompt: str, **kwargs) -> ModelOutput:
        # logger.debug(f"prompt in OpenAIEngine.completion(): {prompt}")
        kwargs.pop("application_id", None)
        kwargs.pop("validate", None)
        kwargs.pop("model", None)
        kwargs.pop("meta_info", None)  # Remove meta_info as it's not a valid OpenAI API parameter
        enforce_max_prompt_length = kwargs.pop("enforce_max_prompt_length", True)

        sampling_params = self.sampling_params.copy()
        sampling_params.update(kwargs)

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        prompt_length = len(prompt_ids)

        if enforce_max_prompt_length and (prompt_length > self.max_prompt_length or prompt_length > self.max_model_length):
            raise TerminationEvent(TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED)

        create_params = self._prepare_max_tokens_param(sampling_params, prompt_length)

        retries = self.api_retries
        while retries > 0:
            try:
                extra_body = sampling_params.setdefault("extra_body", {})
                if sampling_params.get('repetition_penalty'):
                    extra_body['repetition_penalty'] = sampling_params.pop('repetition_penalty')
                response = await self.client.completions.create(model=self.model, prompt=prompt, timeout=3600, **create_params, **sampling_params)

                text = response.choices[0].text

                prompt_length = response.usage.prompt_tokens
                completion_length = response.usage.completion_tokens
                finish_reason = response.choices[0].finish_reason

                # 添加诊断日志
                # if "</think>" not in text and self.config.rllm.get('check_format', True):
                #     logger.warning(f"=== completion text length: {completion_length}, finish_reason: {finish_reason} ===")
                #     logger.warning(f"=== NO </think> tag found! First 500 chars: {text[:500]} ===")
                completion_ids = self.tokenizer.encode(text, add_special_tokens=False)
                parsed_output = self.chat_parser.parse_completion(completion_ids)

                return ModelOutput(
                    text=text,
                    content=parsed_output["content"],
                    reasoning=parsed_output["reasoning"],
                    tool_calls=parsed_output["tool_calls"],
                    prompt_ids=prompt_ids,
                    completion_ids=completion_ids,
                    logprobs=[],
                    prompt_length=prompt_length,
                    completion_length=completion_length,
                    finish_reason=finish_reason,
                )
            except ToolParseError as e:
                retries -= 1
                if retries == 0:
                    raise TaskStuck(traceback.format_exc)
                logger.warning(f"Tool parse error: {e.message}, retrying...")
                # start = text.find(self.chat_parser.tool_parser.tool_call_begin) + len(self.chat_parser.tool_parser.tool_call_begin)
                # end = text.find(self.chat_parser.tool_parser.tool_call_begin)
                # if end == -1:
                #     json_content = text[start:].strip()
                # else:
                #     json_content = text[start:end].strip()
                # prompt += f"\n\nVERY IMPORTANT: Toll call must be a valid JSON string. Arguments of the function call must be a valid dictionary. {json_content} is an invalid JSON string!"
            except openai.RateLimitError:
                retries -= 1
                if retries == 0:
                    raise TaskStuck(traceback.format_exc)
                logger.warning("Sleep for 5 seconds for API limit.")
                await asyncio.sleep(5)

    async def get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
        if self._use_chat_completions:
            accumulate_reasoning = kwargs.pop("accumulate_reasoning", self.accumulate_reasoning)
            if accumulate_reasoning:
                raise ValueError("Accumulate reasoning is not supported for chat completions endpoint.")
            return await self.chat_completion(messages, **kwargs)
        else:
            tools = kwargs.pop("tools", self.tools)
            accumulate_reasoning = kwargs.pop("accumulate_reasoning", self.accumulate_reasoning)
            prompt = self.chat_parser.parse(messages, add_generation_prompt=True, is_first_msg=True, tools=tools, accumulate_reasoning=accumulate_reasoning, **kwargs)
            kwargs.pop('last_n_reasoning')
            return await self.completion(prompt, **kwargs)
