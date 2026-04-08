import os
import logging
import copy

import torch
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from rllm.parser.chat_template_parser import ChatTemplateParser

logger = logging.getLogger(__name__)


class RLLMSFTDataset(MultiTurnSFTDataset):
    """
    SFT Dataset supporting multiple training modes:

    Configuration:
        tokenize_and_mask_method: "cumulative" | "stepwise"
        reasoning_mode: "all" | "none" | "last_n"
        last_n_reasoning: int (used when reasoning_mode="last_n")
        expand_trajectory_to_steps: bool (expand full trajectory to step snapshots)

    Mode combinations:
        1. cumulative + all    : Full SFT, keep all reasoning, all assistants contribute to loss
        2. cumulative + none   : Full SFT, remove all reasoning, all assistants contribute to loss
        3. stepwise + last_n   : Stepwise SFT, keep last N reasoning, only last assistant contributes to loss
        4. stepwise + expand   : Auto-expand full trajectory to snapshots, then apply stepwise
        5. stepwise + all      : Stepwise SFT, data is already expanded to step snapshots, apply stepwise
    """

    def __init__(self, parquet_files: str | list[str], tokenizer, config=None, max_samples=None):
        # Don't call super().__init__ yet, we need to handle data expansion first
        self.tokenizer = tokenizer
        self.config = config

        # Parse configuration
        self.tokenize_and_mask_method = config.rllm.tokenize_and_mask_method
        self.reasoning_mode = getattr(config.rllm, 'reasoning_mode', 'all')
        self.last_n_reasoning = getattr(config.rllm, 'last_n_reasoning', 0)
        self.expand_trajectory_to_steps = getattr(config.rllm, 'expand_trajectory_to_steps', False)

        logger.info(f"RLLMSFTDataset config:")
        logger.info(f"  tokenize_and_mask_method: {self.tokenize_and_mask_method}")
        logger.info(f"  reasoning_mode: {self.reasoning_mode}")
        logger.info(f"  last_n_reasoning: {self.last_n_reasoning}")
        logger.info(f"  expand_trajectory_to_steps: {self.expand_trajectory_to_steps}")

        # Initialize parser
        self.parser = ChatTemplateParser.get_parser(tokenizer)

        # Call parent init to load data
        super().__init__(parquet_files, tokenizer, config)

        # Expand trajectories to steps if needed
        if self.expand_trajectory_to_steps and self.tokenize_and_mask_method == "stepwise":
            self._expand_trajectories_to_steps()

    def _expand_trajectories_to_steps(self):
        """
        Expand full trajectory data into step snapshots.
        Each snapshot contains messages from start up to (and including) each assistant message.
        This allows stepwise training on trajectory-level data.
        """
        logger.info(f"Expanding {len(self.messages)} trajectories to step snapshots...")
        expanded_messages = []

        for messages in self.messages:
            # Generate snapshots for each assistant turn
            current_snapshot = []
            for msg in messages:
                current_snapshot.append(msg)
                if msg.get("role") == "assistant":
                    # Create a snapshot at this assistant turn
                    expanded_messages.append(copy.deepcopy(current_snapshot))

        logger.info(f"Expanded to {len(expanded_messages)} step snapshots")
        self.messages = expanded_messages

    def _tokenize_and_mask(self, messages):
        if self.tokenize_and_mask_method == "cumulative":
            return self._tokenize_and_mask_cumulative(messages)
        elif self.tokenize_and_mask_method == "stepwise":
            return self._tokenize_and_mask_stepwise(messages)
        else:
            raise ValueError(f"Unknown tokenize_and_mask_method {self.tokenize_and_mask_method}")

    def _should_include_reasoning(self, msg_index: int, total_messages: int) -> bool:
        """
        Determine whether to include reasoning for a message at given index.

        Args:
            msg_index: Index of the message in the conversation
            total_messages: Total number of messages

        Returns:
            bool: Whether to include reasoning content
        """
        if self.reasoning_mode == "all":
            return True
        elif self.reasoning_mode == "none":
            return False
        elif self.reasoning_mode == "last_n":
            # Use the same cutoff formula as before
            # Keep reasoning for messages at index >= cutoff
            cutoff = total_messages - (self.last_n_reasoning + 1) * 2
            return msg_index >= cutoff
        else:
            raise ValueError(f"Unknown reasoning_mode: {self.reasoning_mode}")

    def _tokenize_and_mask_cumulative(self, messages):
        """
        Cumulative mode: All assistant messages contribute to loss.
        Reasoning inclusion is controlled by reasoning_mode.
        """
        tokens = []
        loss_mask = []
        total_messages = len(messages)

        for i, msg in enumerate(messages):
            if msg["role"] == "assistant":
                # Determine whether to include reasoning
                include_reasoning = self._should_include_reasoning(i, total_messages)
                parsed_msg = self.parser.parse_assistant(msg, accumulate_reasoning=include_reasoning)
            else:
                parsed_msg = self.parser.parse([msg], is_first_msg=(i == 0), add_generation_prompt=False)

            ids = self.tokenizer.encode(parsed_msg, add_special_tokens=False)

            if msg["role"] == "assistant":
                loss_mask.extend([1] * len(ids))
            else:
                loss_mask.extend([0] * len(ids))
            tokens.extend(ids)

        return tokens, loss_mask

    def _tokenize_and_mask_stepwise(self, messages):
        """
        Stepwise mode: Only the last assistant message contributes to loss.
        Reasoning inclusion is controlled by reasoning_mode and last_n_reasoning.

        This is designed for step snapshot data where each sample ends with
        an assistant message that is the training target.
        """
        tokens = []
        loss_mask = []
        total_messages = len(messages)

        # Find the index of the last assistant message (training target)
        last_assistant_idx = -1
        for i in range(len(messages)):
            if messages[i]["role"] == "assistant":
                last_assistant_idx = i

        if last_assistant_idx == -1:
            raise ValueError("No assistant message found in chat_completions")

        for i, msg in enumerate(messages):
            if msg["role"] == "assistant":
                # Determine whether to include reasoning based on position
                include_reasoning = self._should_include_reasoning(i, total_messages)
                parsed_msg = self.parser.parse_assistant(msg, accumulate_reasoning=include_reasoning)
            else:
                parsed_msg = self.parser.parse([msg], is_first_msg=(i == 0), add_generation_prompt=False)

            ids = self.tokenizer.encode(parsed_msg, add_special_tokens=False)

            # Only the last assistant message contributes to loss
            if i == last_assistant_idx:
                loss_mask.extend([1] * len(ids))
            else:
                loss_mask.extend([0] * len(ids))
            tokens.extend(ids)

        return tokens, loss_mask

    def __getitem__(self, item):
        messages = self.messages[item]

        tokens, loss_mask = self._tokenize_and_mask(messages)

        input_ids = torch.tensor(tokens, dtype=torch.long)
        loss_mask = torch.tensor(loss_mask, dtype=torch.long)
        attention_mask = torch.tensor([1] * len(tokens), dtype=torch.long)

        # Handle sequence length
        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            # Pad sequences
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            padded_input_ids = torch.full((self.max_length - sequence_length,), pad_token_id, dtype=input_ids.dtype)
            padded_attention_mask = torch.zeros((self.max_length - sequence_length,), dtype=attention_mask.dtype)
            padded_loss_mask = torch.zeros((self.max_length - sequence_length,), dtype=loss_mask.dtype)

            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
            loss_mask = torch.cat((loss_mask, padded_loss_mask))

        elif sequence_length > self.max_length:
            if self.truncation == "left":
                input_ids = input_ids[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
                loss_mask = loss_mask[-self.max_length :]
            elif self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
                loss_mask = loss_mask[: self.max_length]
            elif self.truncation == "error":
                raise ValueError(f"{sequence_length=} is larger than {self.max_length=}")
            else:
                raise ValueError(f"Unknown truncation method {self.truncation}")

        # Create position IDs
        position_ids = torch.arange(len(input_ids), dtype=torch.long)
        # Zero out position IDs for padding
        position_ids = position_ids * attention_mask

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }

    def __len__(self):
        return len(self.messages)
