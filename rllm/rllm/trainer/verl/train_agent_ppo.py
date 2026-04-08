"""
Copyright under Agentica Project.

Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import os
import socket


def _enable_ray_startup_patches():
    # Ray launches dashboard_agent and rollout workers in separate Python
    # processes. They inherit this env var, and repo-level sitecustomize.py
    # applies startup monkey patches before the slow GPU probe / HTTP server
    # bootstrap paths run.
    os.environ.setdefault("RLLM_ENABLE_RAY_GPU_PROBE_PATCH", "1")


# Monkey patch verl's parquet loading to fix PyArrow nested data conversion bug
def _patch_verl_dataset():
    import datasets
    import numpy as np
    import polars as pl
    from verl.utils.dataset import rl_dataset

    def patched_read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # Use polars to avoid PyArrow chunked array bug with nested data
            pl_df = pl.read_parquet(parquet_file)
            dataframe = datasets.Dataset.from_pandas(pl_df.to_pandas())
            dataframes.append(dataframe)
        self.dataframe = datasets.concatenate_datasets(dataframes)
        total = len(self.dataframe)

        if self.config.get("reverse_order", False) and total > 0:
            reverse_indices = np.arange(total - 1, -1, -1, dtype=np.int64)
            self.dataframe = self.dataframe.select(reverse_indices.tolist())

        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rngs_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rngs_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
                self.dataframe = self.dataframe.select(indices.tolist())
                print(f"selected {self.max_samples} random samples out of {total}")
            else:
                self.dataframe = self.dataframe.select(np.arange(self.max_samples).tolist())
                order_name = "reversed" if self.config.get("reverse_order", False) else "sequential"
                print(f"selected first {self.max_samples} samples from {order_name} order out of {total}")

        print(f"dataset len: {len(self.dataframe)}")
        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)

    rl_dataset.RLHFDataset._read_files_and_tokenize = patched_read_files_and_tokenize




def patch_flashinfer_build():
    from vllm.v1.attention.backends.flashinfer import FlashInferMetadataBuilder
    import torch
    old_build = FlashInferMetadataBuilder.build
    def new_build(*args, **kwargs):
        self = args[0]
        max_num_pages_per_req = self.block_table_arange.numel()
        self.block_table_arange.copy_(
            torch.arange(
                max_num_pages_per_req,
                device=self.block_table_arange.device,
                dtype=self.block_table_arange.dtype,
            )
        )
        return old_build(*args, **kwargs)

    FlashInferMetadataBuilder.build = new_build



import hydra
import ray
from omegaconf import OmegaConf
from verl.trainer.ppo.reward import load_reward_manager
from verl.utils.device import is_cuda_available

from rllm.trainer.env_agent_mappings import AGENT_CLASS_MAPPING, ENV_CLASS_MAPPING
from rllm.trainer.verl.agent_ppo_trainer import AgentPPOTrainer

# Local application imports
from rllm.trainer.verl.agent_workflow_trainer import AgentWorkflowPPOTrainer
from rllm.trainer.verl.ray_runtime_env import get_ppo_ray_runtime_env


@hydra.main(config_path="../config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    run_ppo_agent(config)


def run_ppo_agent(config):
    # Check if Ray is not initialized
    if not ray.is_initialized():
        _enable_ray_startup_patches()
        # read off all the `ray_init` settings from the config
        if config is not None and hasattr(config, "ray_init"):
            ray_init_settings = {k: v for k, v in config.ray_init.items() if v is not None}
        else:
            ray_init_settings = {}
        ray.init(runtime_env=get_ppo_ray_runtime_env(), include_dashboard=False, **ray_init_settings)

    # Create a remote instance of the TaskRunner class, and
    # Execute the `run` method of the TaskRunner instance remotely and wait for it to complete
    if is_cuda_available and config.trainer.get("profile_steps") is not None and len(config.trainer.get("profile_steps", [])) > 0:
        nsight_options = OmegaConf.to_container(config.trainer.controller_nsight_options)
        runner = TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))

    # [Optional] get the path of the timeline trace file from the configuration, default to None
    # This file is used for performance analysis
    timeline_json_file = config.ray_init.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    """Ray remote class for executing distributed PPO training tasks.

    This class encapsulates the main training logic and runs as a Ray remote actor
    to enable distributed execution across multiple nodes and GPUs.
    """

    def run(self, config, workflow_class=None, workflow_args=None, agent_class=None, env_class=None, agent_args=None, env_args=None, agent_run_func=None):
        """Execute the main PPO training workflow.

        This method sets up the distributed training environment, initializes
        workers, datasets, and reward functions, then starts the training process.

        Args:
            config: Training configuration object containing all parameters needed
                   for setting up and running the PPO training process.
        """

        # Apply patch in Ray worker process to fix PyArrow nested data bug
        _patch_verl_dataset()
        # patch_flashinfer_build()

        # Print the initial configuration. `resolve=True` will evaluate symbolic values.
        from pprint import pprint

        from omegaconf import OmegaConf
        from verl.single_controller.ray import RayWorkerGroup
        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        OmegaConf.register_new_resolver("mul", lambda x, y: int(x) * int(y))
        OmegaConf.resolve(config)
        pprint(OmegaConf.to_container(config))

        # Download the checkpoint from HDFS to the local machine.
        # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
        local_path = copy_to_local(config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False))

        # Instantiate the tokenizer and processor.
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # Used for multimodal LLM, could be None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # Starting from verl 0.6.1, this cls has been standardized for both fsdp and megatron backends.
        ray_worker_group_cls = RayWorkerGroup
        # Define worker classes based on the actor strategy.
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            assert config.critic.strategy in {"fsdp", "fsdp2"}
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

            use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
            if use_legacy_worker_impl in ["auto", "enable"]:
                # import warnings
                # warnings.warn(f"Legacy worker impl is going to be deprecated, will be removed in the future. \
                #   Please set trainer.use_legacy_worker_impl = false to switch to the new worker implementation.")
                from verl.workers.fsdp_workers import CriticWorker
            elif use_legacy_worker_impl == "disable":
                from verl.workers.roles import CriticWorker

                print("Using new worker implementation")
            else:
                raise ValueError(f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}")

            actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        # Map roles to their corresponding remote worker classes.
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        # Define the resource pool specification.
        # Map roles to the resource pool.
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # Add a reference policy worker if KL loss or KL reward is used.
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        # Load the reward manager for training and validation.
        reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
        val_reward_fn = load_reward_manager(config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {}))
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        # if config.rllm.workflow.use_workflow:
        if agent_run_func is not None:
            print("IMPORTANT: Using AgentSdkTrainer")
            from rllm.trainer.verl.agent_sdk_trainer import AgentSdkTrainer

            trainer = AgentSdkTrainer(
                config=config,
                tokenizer=tokenizer,
                role_worker_mapping=role_worker_mapping,
                resource_pool_manager=resource_pool_manager,
                ray_worker_group_cls=ray_worker_group_cls,
                agent_run_func=agent_run_func,
            )
        elif workflow_class is not None:
            workflow_args = workflow_args or {}
            if config.rllm.workflow.get("workflow_args") is not None:
                for key, value in config.rllm.workflow.get("workflow_args").items():
                    if value is not None:
                        if key in workflow_args and isinstance(workflow_args[key], dict):
                            workflow_args[key].update(value)
                        else:
                            workflow_args[key] = value

            trainer = AgentWorkflowPPOTrainer(
                config=config,
                tokenizer=tokenizer,
                processor=processor,
                role_worker_mapping=role_worker_mapping,
                resource_pool_manager=resource_pool_manager,
                ray_worker_group_cls=ray_worker_group_cls,
                reward_fn=reward_fn,
                val_reward_fn=val_reward_fn,
                workflow_class=workflow_class,
                workflow_args=workflow_args,
            )

        else:
            if env_class is None:
                env_class = ENV_CLASS_MAPPING[config.rllm.env.name]
            if agent_class is None:
                agent_class = AGENT_CLASS_MAPPING[config.rllm.agent.name]

            env_args = env_args or {}
            agent_args = agent_args or {}
            if config.rllm.env.get("env_args") is not None:
                env_args.update(config.rllm.env.get("env_args"))
            if config.rllm.agent.get("agent_args") is not None:
                agent_args.update(config.rllm.agent.get("agent_args"))

            # Select trainer based on engine_name config
            engine_name = config.rllm.agent.get("engine_name", "verl")
            if engine_name == "openai":
                from rllm.trainer.verl.agent_ppo_trainer_openai import AgentPPOTrainerOpenAI

                trainer_cls = AgentPPOTrainerOpenAI
                print(f"Using AgentPPOTrainerOpenAI with OpenAI engine for rollout")
            else:
                trainer_cls = AgentPPOTrainer
                print(f"Using AgentPPOTrainer with veRL engine for rollout")

            trainer = trainer_cls(
                config=config,
                tokenizer=tokenizer,
                role_worker_mapping=role_worker_mapping,
                resource_pool_manager=resource_pool_manager,
                ray_worker_group_cls=ray_worker_group_cls,
                reward_fn=reward_fn,
                val_reward_fn=val_reward_fn,
                env_class=env_class,
                agent_class=agent_class,
                env_args=env_args,
                agent_args=agent_args,
            )

        trainer.init_workers()
        try:
            trainer.fit_agent()
        finally:
            trainer.shutdown()


if __name__ == "__main__":
    main()
