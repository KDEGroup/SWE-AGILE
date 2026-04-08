import os
import logging
import numpy as np

from torch.distributed.device_mesh import init_device_mesh
from torch.utils.data import Subset
from rllm.agents.agent import Trajectory

logger = logging.getLogger(__name__)


class AgentSFTTrainer:
    def __init__(self, config, train_dataset=None, val_dataset=None, backend="verl"):
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.backend = backend

        assert self.backend in ["verl", "verl_engine", "tinker"], f"Unsupported backend: {self.backend}, must be one of ['verl', 'verl_engine', 'tinker']"

    def train(self):
        """Start training with the selected backend."""
        if self.backend == "verl":
            self._train_verl()
        elif self.backend == "verl_engine":
            self._train_verl_sft_trainer()
        elif self.backend == "tinker":
            self._train_tinker()

    def _train_verl(self):
        from verl.trainer.fsdp_sft_trainer import FSDPSFTTrainer
        from verl.utils import hf_tokenizer
        from verl.utils.device import get_device_name
        from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group
        from verl.utils.fs import copy_to_local

        from rllm.trainer.verl.sft_dataset import RLLMSFTDataset

        config = self.config

        output_dir = config.trainer.default_local_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            print(f"日志和模型输出目录已确保存在: {output_dir}")
        else:
            print("警告: config.trainer.default_local_dir 未设置，可能无法保存输出。")        


        device_name = get_device_name()
        local_rank, rank, world_size = initialize_global_process_group()

        device_mesh = init_device_mesh(device_type=device_name, mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
        dp_size = world_size // config.ulysses_sequence_parallel_size
        ulysses_device_mesh = init_device_mesh(
            device_type=device_name,
            mesh_shape=(dp_size, config.ulysses_sequence_parallel_size),
            mesh_dim_names=("dp", "sp"),
        )
        # build tokenizer and datasets first
        local_model_path = copy_to_local(src=config.model.partial_pretrain, verbose=True)
        tokenizer = hf_tokenizer(local_model_path, trust_remote_code=config.model.trust_remote_code)

        train_dataset = RLLMSFTDataset(config.data.train_files, tokenizer, config.data)

        if config.data.val_files == "default":
            logger.info("val_files=default, sampling 5% from train dataset (train not reduced)")
            val_ratio = getattr(config.data, "val_ratio", 0.05)
            total_len = len(train_dataset)
            val_len = int(total_len * val_ratio)
            rng = np.random.default_rng(seed=config.trainer.seed)
            val_indices = rng.choice(total_len, size=val_len, replace=False)
            val_dataset = Subset(train_dataset, val_indices)
        else:
            val_dataset = RLLMSFTDataset(config.data.val_files, tokenizer, config.data)

        trainer = FSDPSFTTrainer(
            config=config,
            device_mesh=device_mesh,
            ulysses_device_mesh=ulysses_device_mesh,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )

        trainer.fit()

        destroy_global_process_group()

    def _train_verl_sft_trainer(self):
        """
        Train using verl's SFTTrainer with engine backend.
        Supports use_dynamic_bsz and ulysses_sequence_parallel_size.
        """
        from verl.trainer.sft_trainer import SFTTrainer, create_sft_dataset
        from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group

        config = self.config

        output_dir = config.trainer.default_local_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            print(f"日志和模型输出目录已确保存在: {output_dir}")
        else:
            print("警告: config.trainer.default_local_dir 未设置，可能无法保存输出。")

        initialize_global_process_group()

        trainer = SFTTrainer(config=config)
        trainer.fit()

        destroy_global_process_group()

    def _train_tinker(self):
        """Train using Tinker backend."""
        from rllm.trainer.tinker.tinker_sft_trainer import TinkerSFTTrainer

        trainer = TinkerSFTTrainer(
            config=self.config,
            train_dataset=self.train_dataset,
            val_dataset=self.val_dataset,
        )
        trainer.fit_sft()

    @staticmethod
    def process_trajectories(trajectories: list[Trajectory], reward_threshold: float):
        """Process trajectories into SFT format."""
        sft_data = []

        for traj in trajectories:
            if not traj:
                continue

            reward = traj.reward

            if reward < reward_threshold:
                continue

            # Get chat_completions from the last step of the trajectory
            messages = None
            if traj.steps and hasattr(traj.steps[-1], "chat_completions"):
                messages = traj.steps[-1].chat_completions

            if not messages:
                continue
            # This step will filter "reasoning" and "tool_calls" in message
            # clean_messages = [{"role": msg["role"], "content": str(msg["content"]).strip()} for msg in messages if isinstance(msg, dict) and msg.get("role") and str(msg.get("content", "")).strip()]

            if len(messages) >= 2:
                sft_data.append({"messages": messages})

        print(f"Processed {len(trajectories)} trajectories -> {len(sft_data)} valid examples")
        return sft_data


    @staticmethod
    def _filter_reasoning_in_messages(messages: list[dict], last_n_reasoning: int) -> list[dict]:
        """
        Filter reasoning content in messages based on last_n_reasoning.
        Only keep reasoning for the last N assistant messages before the training target.

        This follows the same logic as _tokenize_and_mask_stepwise in sft_dataset.py:
        - The last assistant message is the training target
        - Keep reasoning for the last_n_reasoning assistant messages before the target
        - Remove reasoning from earlier assistant messages

        The cutoff formula: i >= len(messages) - (last_n_reasoning + 1) * 2
        - The +1 accounts for the training target (last assistant message)
        - The *2 accounts for assistant-user message pairs
        """
        if not messages:
            return messages

        # Calculate cutoff point using the same logic as _tokenize_and_mask_stepwise
        # if i >= len(messages) - (last_n_reasoning + 1) * 2, keep reasoning
        cutoff = len(messages) - (last_n_reasoning + 1) * 2

        result = []
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                result.append(msg)
                continue

            if msg.get("role") == "assistant":
                # Keep reasoning if within the window (same logic as _tokenize_and_mask_stepwise)
                # Note: When last_n_reasoning=0, cutoff = len(messages) - 2, so the last assistant
                # (training target) still keeps its reasoning, which is consistent with sft_dataset.py
                if i >= cutoff:
                    # Keep reasoning for messages within the window
                    result.append(msg)
                else:
                    # Remove reasoning for messages outside the window
                    new_msg = {k: v for k, v in msg.items() if k != "reasoning"}
                    result.append(new_msg)
            else:
                result.append(msg)

        return result

    @staticmethod
    def process_trajectories_every_steps(trajectories: list[Trajectory], reward_threshold: float):
        """Process trajectories into SFT format. Each step is considered as a traj.

        Note: Trajectory.steps contains snapshots at each step time, where each step's
        chat_completions is cumulative (contains all previous messages as prefix).

        This function applies last_n_reasoning filtering to retain only the reasoning
        content from the last N steps for each snapshot.
        """
        sft_data = []
        valid_traj_num = sum(
            1 for traj in trajectories
            if traj and getattr(traj, "reward", None) is not None and traj.reward >= reward_threshold
        )

        for traj in trajectories:
            if not traj:
                continue

            reward = getattr(traj, "reward", None)

            if reward is None or reward < reward_threshold:
                continue

            # Get last_n_reasoning from trajectory (set by agent_execution_engine.py)
            last_n_reasoning = getattr(traj, "last_n_reasoning", None)

            # Get chat_completions from each step of the trajectory
            for step in traj.steps:
                messages = None
                if hasattr(step, "chat_completions"):
                    messages = step.chat_completions
                if not messages:
                    continue
                if len(messages) >= 2:
                    # Apply last_n_reasoning filtering to remove reasoning from earlier steps
                    filtered_messages = AgentSFTTrainer._filter_reasoning_in_messages(
                        messages, last_n_reasoning
                    )
                    sft_data.append({"messages": filtered_messages})

        print(f"Processed {len(trajectories)} trajectories -> {valid_traj_num} valid trajs, {len(sft_data)} SFT steps")
        return sft_data

