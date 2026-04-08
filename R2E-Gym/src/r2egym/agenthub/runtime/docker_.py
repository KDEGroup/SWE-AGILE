"""
Refactored DockerRuntime with separated concerns using inheritance and mixins.

This module provides a cleaner architecture by:
1. Separating backend logic (Docker vs Kubernetes) into mixins
2. Creating specialized subclasses for different benchmark environments
3. Eliminating conditional logic scattered throughout the codebase
"""

import os
import sys
import json
from time import sleep
import time
import uuid
import tempfile
import datetime
import hashlib
import shutil
import re
import logging
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict

import docker
from docker.models.containers import Container
import kubernetes
from kubernetes import client, config, watch
from kubernetes.stream import stream

from r2egym.repo_analysis.execution_log_parser import parse_log_fn, decolor_dict_keys
from r2egym.agenthub.runtime.base import ExecutionEnvironment
from r2egym.agenthub.utils.log import get_logger, LOG_LEVEL_MAP
from r2egym.agenthub.utils.utils import match_dockerimage_to_repo
from r2egym.agenthub import SUPPORTED_REPOS, SKIP_FILES, SKIP_FILES_NEW, CMD_TIMEOUT
from r2egym.agenthub.trajectory.swebench_utils import swebench_parse, TestSpec
from r2egym.commit_models.diff_classes import ParsedCommit
from r2egym.swesmith.utils import get_test_command

import concurrent.futures
import tarfile
import io
from loguru import logger as loguru_logger

# Constants
DEFAULT_NAMESPACE = "default"
DEFAULT_PATH_VARIABLE_IN_DOCKER = "/root/.venv/bin:/root/.local/bin:/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/go/bin:/usr/local/go/bin:/usr/local/cargo/bin:/usr/local/cargo/bin/cargo"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

# SWEBench imports
from swebench.harness.constants import (
    APPLY_PATCH_FAIL,
    END_TEST_OUTPUT,
    FAIL_TO_FAIL,
    FAIL_TO_PASS,
    KEY_INSTANCE_ID,
    KEY_PREDICTION,
    MAP_REPO_VERSION_TO_SPECS,
    PASS_TO_FAIL,
    PASS_TO_PASS,
    RESET_FAILED,
    START_TEST_OUTPUT,
    TESTS_ERROR,
    TESTS_TIMEOUT,
    EvalType,
    ResolvedStatus,
    TestStatus,
)
from swebench.harness.test_spec.test_spec import TestSpec
from swebench.harness.log_parsers import MAP_REPO_TO_PARSER, get_eval_type
from swebench.harness.grading import get_eval_tests_report, get_resolution_status


##############################################################################
# Backend Mixins
##############################################################################

class DockerBackendMixin:
    """Mixin for Docker-specific backend operations."""

    def _init_backend_client(self):
        """Initialize Docker client."""
        self.client = docker.from_env(timeout=300)

    def _start_backend_container(self, docker_image: str, command: str, ctr_name: str, **docker_kwargs):
        """
        Start or reuse a Docker container.
        For multi test, we ban reuse.
        """
        containers = self.client.containers.list(all=True, filters={"name": ctr_name})
        # if containers:
        #     self.container = containers[0]
        #     if self.container.status != "running":
        #         self.container.start()
        # else:
        self.container = self.client.containers.run(
            docker_image,
            command,
            name=ctr_name,
            detach=True,
            tty=True,
            stdin_open=True,
            **docker_kwargs,
        )

    def _stop_backend_container(self):
        """Stop and remove Docker container."""
        if self.container:
            self.container.stop()
            self.container.remove()

    def _backend_run(self, code: str, timeout: int, args: str, workdir: str) -> Tuple[str, str]:
        """Execute command in Docker container."""
        command = f"timeout {timeout} {code} {args}"
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    self.container.exec_run,
                    cmd=["/bin/sh", "-c", command],
                    workdir=workdir,
                    stdout=True,
                    stderr=True,
                    environment={"PATH": self.PATH_VARIABLE_IN_DOCKER},
                )
                exec_result = future.result(timeout=timeout + 5)

            output = exec_result.output.decode("utf-8", errors="replace")
            error_code = exec_result.exit_code

            if error_code == 124:
                self.logger.error(f"Internal Timeout: {timeout}s \nCommand: {command} {args}")
                return f"The command took too long to execute (>{timeout}s)", "-1"

            if error_code != 0:
                self.logger.info(f"Error: Exit code {error_code} \nCommand: {command} {args} \nOutput: {output}")
                return output, f"Error: Exit code {error_code}"

            output = re.sub(r"\x1b\[[0-9;]*m|\r", "", output)
            return output, str(error_code)

        except concurrent.futures.TimeoutError:
            self.logger.error(f"Timeout: {timeout}s \nCommand: {command} {args}")
            return f"The command took too long to execute (>{timeout}s)", "-1"

        except Exception as e:
            return f"Error: {repr(e)} \nCommand: {command} {args}", "-1"

    def _backend_copy_to_container(self, src_path: str, dest_path: str):
        """Copy file to Docker container."""
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            tar.add(src_path, arcname=os.path.basename(dest_path))
        tar_stream.seek(0)
        self.container.put_archive(os.path.dirname(dest_path), tar_stream.read())

    def _close_backend_client(self):
        """Close Docker client."""
        self.client.close()


class KubernetesBackendMixin:
    """Mixin for Kubernetes-specific backend operations."""

    def _init_backend_client(self):
        """Initialize Kubernetes client."""
        try:
            config.load_incluster_config()
        except Exception:
            config.load_kube_config()
        self.client = client.CoreV1Api()

    def _start_backend_container(self, docker_image: str, command: str, pod_name: str, **docker_kwargs):
        """Start or connect to a Kubernetes pod."""
        not_found_error = None
        try:
            self.container = self.client.read_namespaced_pod(
                name=pod_name, namespace=DEFAULT_NAMESPACE, _request_timeout=60,
            )
            self.logger.info(f"Found existing Kubernetes pod: {pod_name}")
            return
        except client.ApiException as e:
            not_found_error = e

        if not_found_error.status != 404:
            self.logger.error(
                f"Error checking Kubernetes pod '{pod_name}' status: {not_found_error}. Check Kubernetes configuration and permissions."
            )
            raise not_found_error

        env_vars = {"PATH": DEFAULT_PATH_VARIABLE_IN_DOCKER, **docker_kwargs.get("environment", {})}
        env_spec = [{"name": k, "value": str(v)} for k, v in env_vars.items()]
        pod_body = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {"name": pod_name},
            "spec": {
                "restartPolicy": "Never",
                "containers": [
                    {
                        "name": pod_name,
                        "image": docker_image,
                        "command": ["/bin/sh", "-c"],
                        "args": [command] if isinstance(command, str) else command,
                        "stdin": True,
                        "tty": True,
                        "env": env_spec,
                        "resources": {
                            "requests": {"cpu": "1", "memory": "1Gi"},
                        },
                    }
                ],
                "imagePullSecrets": [{"name": "dockerhub-pro"}],
                "nodeSelector": {"karpenter.sh/nodepool": "bigcpu-standby"},
                "tolerations": [
                    {
                        "key": "node.kubernetes.io/disk-pressure",
                        "operator": "Exists",
                        "effect": "NoExecute",
                        "tolerationSeconds": 10800
                    }
                ],
            },
        }

        max_retries = 5
        backoff = 5
        pod = None
        for attempt in range(1, max_retries + 1):
            try:
                pod = self.client.create_namespaced_pod(
                    namespace=DEFAULT_NAMESPACE, body=pod_body, _request_timeout=120,
                )
                break
            except client.ApiException as e:
                if e.status in (409, 429, 500, 503):
                    self.logger.warning(
                        f"Transient Kubernetes error {e.status} while creating pod "
                        f"'{pod_name}' (attempt {attempt}/{max_retries}); "
                        f"retrying in {backoff}s"
                    )
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 60)
                    continue
                self.logger.error(f"Failed to create Kubernetes pod '{pod_name}': {e}")
                raise
        else:
            raise RuntimeError(
                f"Exceeded retry limit ({max_retries}) while creating pod '{pod_name}'."
            )

        try:
            rv = pod.metadata.resource_version
            w = watch.Watch()
            stream_obj = w.stream(
                self.client.list_namespaced_pod,
                namespace=DEFAULT_NAMESPACE,
                field_selector=f"metadata.name={pod_name}",
                resource_version=rv,
                timeout_seconds=1200,
            )
            start_time = time.time()
            for event in stream_obj:
                obj = event["object"]
                phase = obj.status.phase
                if time.time() - start_time > 1200:
                    w.stop()
                    raise RuntimeError(f"Kubernetes pod '{pod_name}' timed out after 1200 seconds.")
                if phase == "Running":
                    self.logger.info(f"Kubernetes pod '{pod_name}' is Running.")
                    w.stop()
                    break
                if phase in ["Failed", "Succeeded", "Unknown"]:
                    w.stop()
                    raise RuntimeError(
                        f"Kubernetes pod '{pod_name}' entered terminal phase '{phase}'."
                    )
            self.container = pod
        except Exception as e:
            self.logger.error(f"Error waiting for pod to start: {e}")
            try:
                pod_status = self.client.read_namespaced_pod(
                    name=pod_name, namespace=DEFAULT_NAMESPACE, _request_timeout=60,
                )
                if pod_status.status.phase == "Running":
                    self.logger.info(f"Pod '{pod_name}' is running (verified after watch error)")
                    self.container = pod_status
                else:
                    self.logger.warning(f"Pod '{pod_name}' is in state {pod_status.status.phase}")
                    raise RuntimeError(f"Pod '{pod_name}' failed to reach Running state: {pod_status.status.phase}")
            except Exception as status_error:
                self.logger.error(f"Failed to check pod status after watch error: {status_error}")
                raise RuntimeError(f"Failed to verify pod status: {status_error}")

    def _stop_backend_container(self):
        """Stop and delete Kubernetes pod."""
        try:
            self.client.delete_namespaced_pod(
                name=self.container_name,
                namespace=DEFAULT_NAMESPACE,
                body=kubernetes.client.V1DeleteOptions(grace_period_seconds=0),
                _request_timeout=60,
            )

            w = watch.Watch()
            stream_obj = w.stream(
                self.client.list_namespaced_pod,
                namespace=DEFAULT_NAMESPACE,
                field_selector=f"metadata.name={self.container_name}",
                timeout_seconds=60,
            )

            deletion_confirmed = False
            for event in stream_obj:
                if event["type"] == "DELETED":
                    self.logger.info(f"Kubernetes pod {self.container_name} deleted.")
                    deletion_confirmed = True
                    w.stop()
                    break

            if not deletion_confirmed:
                try:
                    self.client.read_namespaced_pod(
                        name=self.container_name, namespace=DEFAULT_NAMESPACE
                    )
                    self.logger.warning(
                        f"Watch timed out but pod {self.container_name} still exists. Forcing deletion."
                    )
                    self.client.delete_namespaced_pod(
                        name=self.container_name,
                        namespace=DEFAULT_NAMESPACE,
                        body=kubernetes.client.V1DeleteOptions(
                            grace_period_seconds=0,
                            force=True
                        ),
                    )
                except kubernetes.client.rest.ApiException as e:
                    if e.status == 404:
                        self.logger.info(f"Confirmed pod {self.container_name} is deleted.")
                    else:
                        self.logger.error(f"Error checking pod status after timeout: {e}")
        except kubernetes.client.rest.ApiException as e:
            if e.status == 404:
                self.logger.info(
                    f"Kubernetes pod '{self.container_name}' not found, likely already deleted."
                )
            else:
                self.logger.error(
                    f"Error deleting Kubernetes pod '{self.container_name}': {e}"
                )
                raise e

    def _backend_run(self, code: str, timeout: int, args: str, workdir: str) -> Tuple[str, str]:
        """Execute command in Kubernetes pod."""
        command = ""
        if workdir:
            command += f"cd {workdir} && "
        command += f"timeout {timeout} {code} {args}"
        full_command = ["/bin/sh", "-c", command]

        try:
            def execute_command():
                resp = stream(
                    self.client.connect_get_namespaced_pod_exec,
                    self.container_name,
                    DEFAULT_NAMESPACE,
                    command=full_command,
                    stderr=True,
                    stdin=False,
                    stdout=True,
                    tty=False,
                    _preload_content=False,
                )
                combined_chunks = []
                while resp.is_open():
                    resp.update(timeout=1)
                    if resp.peek_stdout():
                        chunk = resp.read_stdout()
                        combined_chunks.append(chunk)
                    if resp.peek_stderr():
                        chunk = resp.read_stderr()
                        combined_chunks.append(chunk)
                resp.close()
                exit_code = resp.returncode
                combined_output = "".join(combined_chunks)
                return combined_output, exit_code

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(execute_command)
                combined_output, exit_code = future.result(timeout=timeout + 5)

            output = combined_output

            if exit_code is None:
                self.logger.error("Kubernetes exec: Exit code not found.")
                return output, "-1"

            if exit_code == 124:
                self.logger.error(f"Internal Timeout via 'timeout' command: {timeout}s")
                return f"The command took too long to execute (>{timeout}s)", "-1"

            if exit_code != 0:
                self.logger.error(
                    f"Kubernetes exec Error: Exit code {exit_code}\nError Message: {output}"
                )
                return output, f"Error: Exit code {exit_code}"

            output = re.sub(r"\x1b\[[0-9;]*m|\r", "", output)
            return output, str(exit_code)
        except concurrent.futures.TimeoutError:
            self.logger.error(f"Kubernetes exec Overall Timeout: {timeout + 5}s")
            return f"The command took too long to execute (>{timeout}s)", "-1"
        except client.ApiException as e:
            self.logger.error(f"Kubernetes API Error during exec: {e}")
            return f"Error executing command in pod: {repr(e)}", "-1"
        except Exception as e:
            self.logger.error(f"Unexpected error during Kubernetes exec: {repr(e)}")
            return f"Error: {repr(e)}", "-1"

    def _backend_copy_to_container(self, src_path: str, dest_path: str):
        """Copy file to Kubernetes pod."""
        dest_dir = os.path.dirname(dest_path)
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            tar.add(src_path, arcname=os.path.basename(dest_path))
        tar_stream.seek(0)

        max_retries = 5
        retry_delay = 5
        for attempt in range(max_retries):
            try:
                exec_command = ["tar", "xmf", "-", "-C", dest_dir]
                resp = stream(
                    self.client.connect_get_namespaced_pod_exec,
                    self.container_name,
                    DEFAULT_NAMESPACE,
                    command=exec_command,
                    stderr=True,
                    stdin=True,
                    stdout=True,
                    tty=False,
                    _preload_content=False,
                )
                resp.write_stdin(tar_stream.read())
                resp.close()
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"Copy to container failed (attempt {attempt+1}/{max_retries}): {str(e)}")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    retry_delay = min(retry_delay, 60)
                    tar_stream.seek(0)
                else:
                    self.logger.error(f"Copy to container failed after {max_retries} attempts: {str(e)}")
                    raise

    def _close_backend_client(self):
        """Close Kubernetes client (no-op)."""
        pass


##############################################################################
# Base Runtime Class
##############################################################################

class DockerRuntimeBase(ExecutionEnvironment, ABC):
    """
    Base class for all Docker/Kubernetes runtime implementations.

    Provides common functionality for:
    - Container lifecycle management
    - File operations
    - Git operations
    - Command execution

    Subclasses must implement:
    - setup_env(): Environment-specific setup
    - _calculate_reward(): Benchmark-specific reward calculation
    - parse_logs(): Benchmark-specific log parsing
    """

    def __init__(
        self,
        ds,
        repo_path: str = "/testbed",
        alt_path: str = "/root",
        docker_image: str = None,
        command: str = "/bin/bash",
        logger=None,
        backend="docker",
        **docker_kwargs,
    ):
        assert ds, f"Dataset not provided for docker image: {docker_image}"
        assert backend in ["docker", "kubernetes"], f"Invalid backend: {backend}"

        self.ds = ds
        self.backend = backend
        self.docker_kwargs = docker_kwargs
        self.repo_path = repo_path
        self.alt_path = alt_path
        self.command = command
        self.PATH_VARIABLE_IN_DOCKER = DEFAULT_PATH_VARIABLE_IN_DOCKER


        # Get docker image from dataset
        ds_image = None
        if "docker_image" in self.ds:
            ds_image = self.ds["docker_image"]
        elif "image_name" in self.ds:
            ds_image = self.ds["image_name"]
        else:
            raise ValueError(f"No docker image found in ds: {self.ds}")
        self.docker_image = ds_image if not docker_image else docker_image

        # Initialize logger
        if logger is None:
            logger_name = "KubernetesRuntime" if backend == "kubernetes" else "DockerRuntime"
            self.logger = get_logger(logger_name, level=LOG_LEVEL_MAP[os.getenv("LOG_LEVEL", "INFO")])
            self.loguru_logger = loguru_logger
        else:
            self.logger = logger
            self.loguru_logger = loguru_logger

        # Initialize backend client
        self._init_backend_client()

        # Start container
        self.container = None
        self.container_name = self._get_container_name(self.docker_image)
        if self.backend == "kubernetes":
            self.container_name = str(uuid.uuid4())

        self.start_container(
            self.docker_image, command, self.container_name, **docker_kwargs
        )

        # Initialize environment (subclass-specific)
        self.setup_env()

        self.logger.info(f"Runtime initialized with Docker image: {self.docker_image}")

    # Abstract methods that subclasses must implement
    @abstractmethod
    def setup_env(self):
        """Setup environment-specific configuration."""
        pass

    @abstractmethod
    def _calculate_reward(self, get_test_output=False, timeout: int = 480) -> float:
        """Calculate benchmark-specific reward."""
        pass

    @abstractmethod
    def parse_logs(self, log_output: str) -> dict:
        """Parse logs using benchmark-specific parser."""
        pass

    # Common utility methods
    @staticmethod
    def _get_container_name(image_name: str) -> str:
        """Return name of container."""
        process_id = str(os.getpid())
        current_time = str(datetime.datetime.now())
        unique_string = current_time + process_id
        hash_object = hashlib.sha256(unique_string.encode())
        image_name_sanitized = image_name.replace("/", "-").replace(":", "-")
        return f"{image_name_sanitized}-{hash_object.hexdigest()[:10]}"

    def start_container(
        self, docker_image: str, command: str, ctr_name: str, **docker_kwargs
    ):
        """Start or reuse a container."""
        try:
            self._start_backend_container(docker_image, command, ctr_name, **docker_kwargs)
        except Exception as e:
            import traceback
            tb_str = traceback.format_exc()
            error_msg = f"Container start error: {repr(e)}, docker_image: {docker_image}, ctr_name: {ctr_name}, traceback: {tb_str}"
            self.logger.error(error_msg)
            with open("container_start_error.log", "w") as f:
                f.write(error_msg + "\n")
            self.stop_container()
            raise RuntimeError(error_msg) from e

    def stop_container(self):
        """Stop and remove container."""
        try:
            if self.container:
                self._stop_backend_container()
        except Exception as e:
            print("Container stop/delete error:", repr(e))

    def run(
        self,
        code: str,
        timeout: int = CMD_TIMEOUT,
        args: str = "",
        workdir=None,
        type: str = None,
    ) -> Tuple[str, str]:
        """Execute code or commands in the container with a timeout."""
        exec_workdir = self.repo_path if workdir is None else workdir
        return self._backend_run(code, timeout, args, exec_workdir)

    def copy_to_container(self, src_path: str, dest_path: str):
        """Copy a file or directory from the host into the container."""
        parent_dir = os.path.dirname(dest_path)
        if parent_dir and parent_dir != '/':
            self.run(f"mkdir -p {parent_dir}", timeout=30)
        self._backend_copy_to_container(src_path, dest_path)
        # print(f"Copied {src_path} to container at {dest_path}")

    def write2container(self, content: str, suffix: str, file_path: str):
        """Write content to a file in the container."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=suffix) as temp_file:
            temp_file.write(content)
            temp_file.flush()
            temp_file_path = temp_file.name
        self.copy_to_container(temp_file_path, file_path)
        os.unlink(temp_file_path)

    def setup_tools_uv_env(self):
        """Setup tools UV environment (common across all benchmarks)."""
        self.copy_to_container(
            os.path.join(PROJECT_ROOT, "tools_uv_env/cpython-3.10.14+20240415-x86_64-unknown-linux-gnu-install_only.tar.gz"),
            "/tools_uv_env/cpython-3.10.14+20240415-x86_64-unknown-linux-gnu-install_only.tar.gz"
        )
        self.run("tar -xzf /tools_uv_env/cpython-3.10.14+20240415-x86_64-unknown-linux-gnu-install_only.tar.gz -C /tools_uv_env/")
        self.run("rm -f /tools_uv_env/cpython-3.10.14+20240415-x86_64-unknown-linux-gnu-install_only.tar.gz")

        self.copy_to_container(os.path.join(PROJECT_ROOT, "tools_uv_env/uv"), "/usr/local/bin/uv")
        self.run("chmod +x /usr/local/bin/uv")

        self.copy_to_container(os.path.join(PROJECT_ROOT, "tools_uv_env/"), "/tools_uv_env/")
        self.run("uv venv -p /tools_uv_env/python/bin/python3 /tools_uv_env/.venv/  --clear", timeout=1200)
        self.run("uv pip install -p /tools_uv_env/.venv/bin/python  --no-index  --find-links /tools_uv_env/wheels  -r /tools_uv_env/requirements.txt")

#         self.run(f"mkdir -p /home/root/.pip")
#         pip_mirror_conf = """
# [global]
# index-url = https://mirrors.aliyun.com/pypi/simple/
# """
#         self.write2container(pip_mirror_conf, ".conf", "/home/root/.pip/pip.conf")
        self.run("pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple")

        # python_path, _ = self.run("which python")

    # Git operations
    def checkout(self, commit_hash: str) -> Tuple[str, str]:
        """Checkout a specific commit."""
        output, error_code = self.run(f"git checkout {commit_hash}")
        return output, error_code

    def get_patch(self) -> str:
        """Get the diff of the current state of the repository."""
        output, _ = self.run("git add -A && git diff --cached")
        return output

    def apply_patch(self, patch: str) -> Tuple[str, str]:
        """Apply a patch to the repository."""
        uuid_ = uuid.uuid4()
        patch_path = f"{self.container_name}_{uuid_}.patch"
        patch_path = os.path.join("/tmp", patch_path)
        with open(patch_path, "w") as f:
            f.write(patch)
        self.copy_to_container(patch_path, f"/{patch_path}")
        output, error_code = self.run(f"git apply --whitespace=fix /{patch_path}")
        return output, error_code

    def reverse_patch(self, patch: str) -> Tuple[str, str]:
        """Reverse a patch."""
        patch_path = f"{self.container_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.patch"
        patch_path = os.path.join("/tmp", patch_path)
        with open(patch_path, "w") as f:
            f.write(patch)
        self.copy_to_container(patch_path, f"/{patch_path}")
        output, error_code = self.run(f"git apply -R /{patch_path}")
        return output, error_code

    def start_new_branch(self, branch_name: str = "exp") -> Tuple[str, str]:
        """Start a new git branch and configure git."""
        output, error_code = self.run("git config --global user.email 'you@example.com'")
        output, error_code = self.run("git config --global user.name 'Your Name'")
        output, error_code = self.run("git rev-parse HEAD")
        self.current_commit = output.strip()
        return output, error_code

    def commit_after_step(self, step_idx: int) -> Tuple[str, str]:
        """Commit changes after a step."""
        output, error_code = self.run("git add .")
        output, error_code = self.run(f"git commit -m '{step_idx}'")
        return output, error_code

    def undo_last_commit(self) -> Tuple[str, str]:
        """Undo the last commit."""
        output, error_code = self.run("git reset --hard HEAD~1")
        return output, error_code

    def get_current_commit_hash(self) -> str:
        """Get current commit hash."""
        output, _ = self.run("git rev-parse HEAD")
        return output.strip()

    def soft_git_reset(self) -> Tuple[str, str]:
        """Soft reset to saved commit."""
        output, error_code = self.run(f"git reset --soft {self.current_commit}")
        return output, error_code

    # Test operations
    def run_tests(self, timeout: int = 300) -> Tuple[str, str]:
        """Run tests using the test script."""
        output, error_code = self.run(f"bash {self.alt_path}/run_tests.sh", timeout=timeout)
        output = re.sub(r"\x1b\[[0-9;]*m|\r", "", output)
        return output, error_code

    def get_task_instruction(self) -> str:
        """Get task instruction with repo path info."""
        try:
            content = self.ds["problem_statement"]
            base_instruction = re.search(r"\[ISSUE\](.*)\[/ISSUE\]", content, re.DOTALL).group(1)
        except Exception:
            base_instruction = self.ds["problem_statement"]

        repo_info = f"\n\n[IMPORTANT] The repository is at '/testbed' and the current working directory is already '/testbed'\n"
        return base_instruction + repo_info


    # Lifecycle methods
    def reset(self):
        """Reset the container."""
        self.stop_container()
        self.start_container(
            self.docker_image, self.command, self.container_name, **self.docker_kwargs
        )

    def close(self):
        """Close the runtime and cleanup resources."""
        self.stop_container()
        self._close_backend_client()

    @DeprecationWarning
    def read_file(self, rel_file_path: str) -> str:
        """Read a file from the container (deprecated)."""
        output, _ = self.run(f"cat /{self.alt_path}/{rel_file_path}")
        return output


##############################################################################
# Specialized Runtime Implementations
##############################################################################

# hack for swebench verified, skip make_repo_script_list, etc
from swebench.harness.constants import (
    DEFAULT_DOCKER_SPECS,
    KEY_INSTANCE_ID,
    LATEST,
    MAP_REPO_TO_EXT,
    MAP_REPO_VERSION_TO_SPECS,
    USE_X86,
)
from swebench.harness.constants.constants import SWEbenchInstance
from swebench.harness.test_spec.create_scripts import (
make_env_script_list,
make_eval_script_list,
)
import platform

from typing import Any, Union, cast

def make_test_spec(
        instance: SWEbenchInstance,
        namespace: str=None,
        base_image_tag: str=LATEST,
        env_image_tag: str=LATEST,
        instance_image_tag: str=LATEST,
    ) -> TestSpec:
    if isinstance(instance, TestSpec):
        return instance
    assert base_image_tag is not None, "base_image_tag cannot be None"
    assert env_image_tag is not None, "env_image_tag cannot be None"
    assert instance_image_tag is not None, "instance_image_tag cannot be None"
    instance_id = instance[KEY_INSTANCE_ID]
    repo = instance["repo"]
    version = instance.get("version")
    base_commit = instance["base_commit"]
    problem_statement = instance.get("problem_statement")
    hints_text = instance.get("hints_text")  # Unused
    test_patch = instance["test_patch"]

    def _from_json_or_obj(key: str) -> Any:
        """If key points to string, load with json"""
        if key not in instance:
            # If P2P, F2P keys not found, it's a validation instance
            return []
        if isinstance(instance[key], str):
            return json.loads(instance[key])
        return instance[key]

    pass_to_pass = _from_json_or_obj("PASS_TO_PASS")
    fail_to_pass = _from_json_or_obj("FAIL_TO_PASS")

    env_name = "testbed"
    repo_directory = f"/{env_name}"
    specs = MAP_REPO_VERSION_TO_SPECS[repo][version]
    docker_specs = specs.get("docker_specs", {})

    repo_script_list = []
    env_script_list = []
    eval_script_list = []
    if platform.machine() in {"aarch64", "arm64"}:
        # use arm64 unless explicitly specified
        arch = "arm64" if instance_id not in USE_X86 else "x86_64"
    else:
        arch = "x86_64"

    return TestSpec(
        instance_id=instance_id,
        repo=repo,
        env_script_list=env_script_list,
        repo_script_list=repo_script_list,
        eval_script_list=eval_script_list,
        version=version,
        arch=arch,
        FAIL_TO_PASS=fail_to_pass,
        PASS_TO_PASS=pass_to_pass,
        language=MAP_REPO_TO_EXT[repo],
        docker_specs=docker_specs,
        namespace=namespace,
        base_image_tag=base_image_tag,
        env_image_tag=env_image_tag,
        instance_image_tag=instance_image_tag,
    )
class SWEBenchRuntime(DockerRuntimeBase):
    """Runtime for SWE-bench verified benchmark."""



    def __init__(self, ds, backend="docker", **kwargs):
        # Detect if this is SWE-bench verified
        ds_image = ds.get("docker_image") or ds.get("image_name")
        if not ("swebench" in ds_image and "mswebench" not in ds_image):
            raise ValueError(f"Not a SWE-bench verified image: {ds_image}")

        super().__init__(ds, backend=backend, **kwargs)

        # Create test spec for swebench verified
        self.test_spec = make_test_spec(self.ds)

        # Parse commit
        self.commit_json = self.ds["parsed_commit"]
        self.commit = ParsedCommit(**json.loads(self.commit_json))

        self.repo_name = self.ds["repo"]

    def setup_env(self):
        """Setup SWE-bench verified environment."""
        self.PATH_VARIABLE_IN_DOCKER = "/opt/miniconda3/envs/testbed/bin:" + DEFAULT_PATH_VARIABLE_IN_DOCKER
        self.setup_tools_uv_env()

        try:
            self.run("chmod +x /run_tests.sh")
            self.alt_path = "/"
            # no need since already set_up_uv_env
            # self.run(f"ln -s /opt/miniconda3/envs/testbed /root/.venv")
            # self.run("python -m pip install chardet")
            # self.run("python -m pip install packaging")
        except Exception as e:
            self.logger.error(f"Error setting up environment: {repr(e)} @ {self.docker_image}")

    def parse_logs(self, log_output: str) -> dict:
        """Parse logs using SWE-bench parser."""
        parsed_output, patch_apply_success = self.get_logs_eval(self.test_spec, log_output)
        return parsed_output

    def get_logs_eval(self, test_spec: TestSpec, content: str) -> Tuple[Dict[str, str], bool]:
        """Retrieve evaluation results from log content."""
        repo = test_spec.repo
        version = test_spec.version
        log_parser = MAP_REPO_TO_PARSER[repo]
        test_cmd = MAP_REPO_VERSION_TO_SPECS[repo][version]["test_cmd"]
        if isinstance(test_cmd, list):
            test_cmd = test_cmd[-1]

        bad_codes = list(
            filter(
                lambda x: x in content,
                [APPLY_PATCH_FAIL, RESET_FAILED, TESTS_ERROR, TESTS_TIMEOUT],
            )
        )
        if bad_codes:
            self.logger.error(f"Bad code found in log: {bad_codes}")
            return {}, False

        content = content.split(test_cmd)[-1]
        return log_parser(content, test_spec), True

    def _calculate_reward(self, get_test_output=False, timeout: int = 300) -> float:
        """Calculate reward for SWE-bench."""
        out, _ = self.run("/run_tests.sh", timeout=timeout)
        eval_status_map, found = self.get_logs_eval(self.test_spec, out)
        eval_ref = {
            KEY_INSTANCE_ID: self.test_spec.instance_id,
            FAIL_TO_PASS: self.test_spec.FAIL_TO_PASS,
            PASS_TO_PASS: self.test_spec.PASS_TO_PASS,
        }
        report = get_eval_tests_report(
            eval_status_map, eval_ref, eval_type=get_eval_type(self.test_spec)
        )
        success = get_resolution_status(report) == ResolvedStatus.FULL.value
        if get_test_output:
            return success, out
        return int(success)


    def run_swebv_regression(self, run_tests_regression: str | None = None, timeout: int = 300) -> dict[str, str]:
        """Run regression tests for SWE-bench verified."""
        if run_tests_regression is None:
            run_tests_regression = self.ds["run_tests_regression"]

        with tempfile.NamedTemporaryFile("w") as f:
            f.write(run_tests_regression)
            f.flush()
            self.copy_to_container(f.name, "/run_tests_regression.sh")

        self.run("chmod +x /run_tests_regression.sh")
        output, error_code = self.run("/run_tests_regression.sh", timeout=timeout)
        return output


class SweSmithRuntime(DockerRuntimeBase):
    """Runtime for SweSmith benchmark."""

    def __init__(self, ds, backend="docker", **kwargs):
        # Detect if this is SweSmith
        ds_image = ds.get("docker_image") or ds.get("image_name")
        if "swesmith" not in ds_image:
            raise ValueError(f"Not a SweSmith image: {ds_image}")

        # Adjust docker image name for SweSmith
        image_name = ds['image_name'].replace('__', '_1776_')
        ds = ds.copy()
        ds['docker_image'] = f'jyangballin/{image_name}:latest'

        super().__init__(ds, backend=backend, **kwargs)

        self.repo_name = self.ds["repo"]

    def setup_env(self):
        """Setup SweSmith environment."""
        self.PATH_VARIABLE_IN_DOCKER = "/opt/miniconda3/envs/testbed/bin:" + DEFAULT_PATH_VARIABLE_IN_DOCKER
        self.setup_tools_uv_env()

        try:
            commit_id = self.ds['base_commit']
            self.run("git fetch")
            self.run(f"git checkout {commit_id}")

            test_command, _ = get_test_command(self.ds)
            eval_script_content = "\n".join([
                "#!/bin/bash",
                "set -uxo pipefail",
                "source /opt/miniconda3/bin/activate",
                f"conda activate testbed",
                f"cd testbed/",
                f": '>>>>> Start Test Output'",
                test_command,
                f": '>>>>> End Test Output'",
            ]) + "\n"

            self.write2container(eval_script_content, ".sh", "/run_tests.sh")
            self.run("chmod +x /run_tests.sh")
        except Exception as e:
            self.logger.error(f"Error setting up environment: {repr(e)}")

    def reset_swesmith_tests(self):
        """Reset test files to base commit."""
        f2p_files = list(set([x.split("::", 1)[0] for x in self.ds[FAIL_TO_PASS]]))
        p2p_files = list(set([x.split("::", 1)[0] for x in self.ds[PASS_TO_PASS]]))
        all_files = list(set(f2p_files + p2p_files))
        all_files = [f for f in all_files if
             os.path.basename(f).startswith('test_') and os.path.basename(f).endswith('.py') or
             os.path.basename(f).endswith('_test.py')]
        commit_id = self.ds['base_commit']
        reset_command = (
            f'printf "%s\\n" {" ".join(all_files)} | '
            f'xargs -n1 -I{{}} git checkout {commit_id} -- "{{}}" 2>/dev/null'
        )
        self.run(reset_command)

    def parse_logs(self, log_output: str) -> dict:
        """Parse logs using repository-specific parser."""
        return parse_log_fn(f"{self.repo_name}")(log_output)

    def _calculate_reward(self, get_test_output=False, timeout: int = 300) -> float:
        """Calculate reward for SweSmith."""
        self.reset_swesmith_tests()
        output, error_msg = self.run("/run_tests.sh", timeout=timeout)
        parse = self.parse_logs(output)

        fail2pass = [".".join(line.split("::")[1:]) for line in self.ds['FAIL_TO_PASS']]
        pass2pass = [".".join(line.split("::")[1:]) for line in self.ds['PASS_TO_PASS']]

        if not parse:
            return 0.0

        # Check fail2pass
        for test_name in fail2pass:
            if test_name not in parse:
                matching_key = next((k for k in parse.keys() if test_name in k), None)
                if matching_key is None:
                    return 0.0
                if parse[matching_key] != 'PASSED':
                    return 0.0
                test_name = matching_key
            if parse[test_name] != 'PASSED':
                return 0.0

        # Check pass2pass
        for test_name in pass2pass:
            if test_name not in parse:
                matching_key = next((k for k in parse.keys() if test_name in k), None)
                if matching_key is None:
                    return 0.0
                test_name = matching_key
            if parse[test_name] != 'PASSED':
                return 0.0

        return 1.0


class MSWEBenchRuntime(DockerRuntimeBase):
    """Runtime for Multi-SWE-bench (mswebench)."""

    def __init__(self, ds, backend="docker", **kwargs):
        # Detect if this is MSWEBench
        ds_image = ds.get("docker_image") or ds.get("image_name")
        if "mswebench" not in ds_image:
            raise ValueError(f"Not a MSWEBench image: {ds_image}")

        self.repo_name = ds["repo"]

        # MSWebench uses /home/{repo_name} instead of /testbed
        repo_short_name = self.repo_name.split('/')[-1] if '/' in self.repo_name else self.repo_name
        repo_path = f"/home/{repo_short_name}"

        super().__init__(ds, repo_path=repo_path, backend=backend, **kwargs)

        # Parse commit
        from r2egym.commit_models.parse_diff import CommitParser
        commit_parser = CommitParser()
        try:
            self.commit = commit_parser.parse_commit(
                self.ds["base_commit"],
                "new_commit_hash",
                self.ds["patch"],
                "commit_message",
                datetime.datetime.now(),
                None,
            )
        except Exception as e:
            self.logger.error(f"Error parsing commit: {e}")
            self.commit = None

    def setup_env(self):
        """Setup MSWEBench environment."""
        self.setup_tools_uv_env()

        pre_setup_bashrc = f"""
#!/bin/bash
echo 'export PIP_CACHE_DIR=~/.cache/pip' >> ~/.bashrc
echo 'export REPO_NAME={self.repo_name}' >> ~/.bashrc
source ~/.bashrc
"""
        self.write2container(pre_setup_bashrc, ".sh", "/pre_setup.sh")
        self.run("chmod +x /pre_setup.sh")
        self.run("bash /pre_setup.sh")

        self.run("chmod +x /home/*.sh 2>/dev/null || true")
        self.run("bash /home/prepare.sh", timeout=1200)

        repo_name = self.ds['repo'].split('/')[-1].replace('-', '_')
        org_name = self.ds['repo'].split('/')[-2].replace('-', '_')
        repo_host_dir = os.path.join(PROJECT_ROOT, f"multi-swe-bench/multi_swe_bench/harness/repos/{self.ds['language']}/{org_name}")
        repo_host_path = os.path.join(repo_host_dir, f"{repo_name}_{self.ds['number']}.log")

        # Load offline test results
        if os.path.exists(repo_host_path):
            with open(repo_host_path, "r", encoding="utf-8") as f:
                self.mswebench_test_run_result = f.read()
        else:
            self.logger.error(f"Report not found: {repo_host_path}")
            raise FileNotFoundError(f"Report not found: {repo_host_path}")

    def parse_logs(self, log_output: str) -> dict:
        """Parse logs for MSWEBench."""
        return parse_log_fn(f"{self.repo_name}")(log_output)

    def _calculate_reward(self, get_test_output=False, timeout: int = 300) -> float:
        """Calculate reward for MSWEBench."""
        from multi_swe_bench.harness.image import Config
        from multi_swe_bench.harness.instance import Instance
        from multi_swe_bench.harness.pull_request import Base, PullRequest
        from multi_swe_bench.harness.report import generate_report
        import multi_swe_bench.harness.repos  # side-effect import to register instances
        # Ensure all repo instances are registered via package import side-effects
        import multi_swe_bench.harness  # noqa: F401

        def instance(org, repo, number) -> Instance:
            pr = PullRequest(
                org=org,
                repo=repo,
                number=number,
                state="",
                title="",
                body="",
                base=Base(label="", ref="", sha=""),
                resolved_issues=[],
                fix_patch="",
                test_patch="",
            )
            config = Config(
                need_clone=False,
                global_env=None,
                clear_env=False,
            )
            return Instance.create(pr, config)

        try:
            fix_run_result, run_exit_code = self.run("/home/test-run.sh", timeout=timeout)

            org = self.ds['repo'].split('/')[0]
            repo = self.ds['repo'].split('/')[1]
            inst = instance(org, repo, self.ds['number'])

            report = generate_report(inst, "", test_patch_result=self.mswebench_test_run_result, fix_patch_result=fix_run_result)

            is_valid, error_msg = report.check()

            if is_valid:
                self.logger.debug(f"MSWebench: Fix is VALID - {report.fix_patch_result.passed_count - report.test_patch_result.passed_count} tests fixed")
                reward = 1.0
            else:
                self.logger.debug(f"MSWebench: Fix is INVALID - {error_msg}")
                reward = 0.0

            if get_test_output:
                return reward, error_msg
            return reward

        except Exception as e:
            self.logger.error(f"Error in _calculate_reward_mswebench: {repr(e)}")
            if get_test_output:
                return 0.0, str(e)
            return 0.0

    def get_task_instruction(self) -> str:
        """Get task instruction with repo path info."""
        try:
            content = self.ds["problem_statement"]
            base_instruction = re.search(r"\[ISSUE\](.*)\[/ISSUE\]", content, re.DOTALL).group(1)
        except Exception:
            base_instruction = self.ds["problem_statement"]

        repo_info = f"\n\n[IMPORTANT] The repository code is located at: '{self.repo_path}' and the current working directory is '/home'\n"
        return base_instruction + repo_info


class R2EGymRuntime(DockerRuntimeBase):
    """Runtime for R2E-Gym benchmark."""

    def __init__(self, ds, backend="docker", **kwargs):
        # Detect if this is R2E-Gym
        ds_image = ds.get("docker_image") or ds.get("image_name")
        if "namanjain12" not in ds_image:
            raise ValueError(f"Not an R2E-Gym image: {ds_image}")

        super().__init__(ds, backend=backend, **kwargs)

        self.repo_name = self.ds["repo_name"]

        # Parse commit
        self.commit_json = self.ds["parsed_commit_content"]
        self.commit = ParsedCommit(**json.loads(self.commit_json))

    def setup_env(self):
        """Setup R2E-Gym environment."""
        self.setup_tools_uv_env()

        # Setup venv symlinks
        self.run(f"ln -s {self.repo_path}/.venv {self.alt_path}/.venv")
        self.run(f"ln -s {self.repo_path}/.venv/bin/python {self.alt_path}/.local/bin/python")
        self.run(f"ln -s {self.repo_path}/.venv/bin/python {self.alt_path}/.local/bin/python3")
        self.run(f"find {self.repo_path}/.venv/bin -type f -executable -exec ln -sf {{}} {self.alt_path}/.local/bin/ \\;")

        # Install required packages
        # self.run("uv pip install chardet")

        # Clean pyc files
        self.run("find . -name '*.pyc' -delete")
        self.run("find . -name '__pycache__' -exec rm -rf {} +")
        self.run("find /r2e_tests -name '*.pyc' -delete")
        self.run("find /r2e_tests -name '__pycache__' -exec rm -rf {} +")

        # Move skip files
        for skip_file in SKIP_FILES_NEW:
            self.run(f"mv {self.repo_path}/{skip_file} {self.alt_path}/{skip_file}")

        # Move r2e_tests to alt_path
        self.run(f"mv /r2e_tests {self.alt_path}/r2e_tests")
        self.run(f"ln -s {self.alt_path}/r2e_tests {self.repo_path}/r2e_tests")

    def parse_logs(self, log_output: str) -> dict:
        """Parse logs using R2E parser."""
        return parse_log_fn(f"{self.repo_name}")(log_output)

    def _calculate_reward(self, get_test_output=False, timeout: int = 300) -> float:
        """Calculate reward for R2E-Gym."""
        output, error_code = self.run_tests(timeout=timeout)
        parse = self.parse_logs(output)
        parse = decolor_dict_keys(parse)

        try:
            expected_json = self.ds["expected_output_json"]
        except Exception:
            expected_json = self.read_file("expected_test_output.json")

        expected: dict = json.loads(expected_json)
        expected = decolor_dict_keys(expected)
        parse = {k.split(" - ")[0]: parse[k] for k in sorted(parse.keys())}
        expected = {k.split(" - ")[0]: expected[k] for k in sorted(expected.keys())}

        if len(parse) != len(expected):
            reward = 0.0
        else:
            match = True
            for k in parse.keys():
                if not k:
                    continue
                if k not in expected:
                    match = False
                    break
                if parse[k] != expected[k]:
                    match = False
                    break
            reward = 1.0 if match else 0.0

        if get_test_output:
            return reward, output
        return reward


##############################################################################
# Concrete Runtime Classes (with backend mixins)
##############################################################################

class DockerSWEBenchRuntime(DockerBackendMixin, SWEBenchRuntime):
    """SWE-bench runtime with Docker backend."""
    pass


class KubernetesSWEBenchRuntime(KubernetesBackendMixin, SWEBenchRuntime):
    """SWE-bench runtime with Kubernetes backend."""
    pass


class DockerSweSmithRuntime(DockerBackendMixin, SweSmithRuntime):
    """SweSmith runtime with Docker backend."""
    pass


class KubernetesSweSmithRuntime(KubernetesBackendMixin, SweSmithRuntime):
    """SweSmith runtime with Kubernetes backend."""
    pass


class DockerMSWEBenchRuntime(DockerBackendMixin, MSWEBenchRuntime):
    """MSWEBench runtime with Docker backend."""
    pass


class KubernetesMSWEBenchRuntime(KubernetesBackendMixin, MSWEBenchRuntime):
    """MSWEBench runtime with Kubernetes backend."""
    pass


class DockerR2EGymRuntime(DockerBackendMixin, R2EGymRuntime):
    """R2E-Gym runtime with Docker backend."""
    pass


class KubernetesR2EGymRuntime(KubernetesBackendMixin, R2EGymRuntime):
    """R2E-Gym runtime with Kubernetes backend."""
    pass


##############################################################################
# Factory Function
##############################################################################

def create_runtime(ds, backend="docker", **kwargs) -> DockerRuntimeBase:
    """
    Factory function to create the appropriate runtime based on dataset configuration.

    Args:
        ds: Dataset entry containing docker_image or image_name
        backend: "docker" or "kubernetes"
        **kwargs: Additional arguments passed to runtime constructor

    Returns:
        Appropriate DockerRuntimeBase subclass instance

    Example:
        >>> runtime = create_runtime(ds, backend="docker")
        >>> runtime.setup_env()
        >>> patch = runtime.get_patch()
    """
    # Get docker image from dataset
    ds_image = ds.get("docker_image") or ds.get("image_name")
    if not ds_image:
        raise ValueError(f"No docker image found in ds: {ds}")

    # Determine runtime type from image name
    is_swebench = "swebench" in ds_image and "mswebench" not in ds_image
    is_swesmith = "swesmith" in ds_image
    is_mswebench = "mswebench" in ds_image
    is_r2egym = "namanjain12" in ds_image

    # Select appropriate runtime class
    if is_swebench:
        if backend == "docker":
            return DockerSWEBenchRuntime(ds, backend=backend, **kwargs)
        else:
            return KubernetesSWEBenchRuntime(ds, backend=backend, **kwargs)
    elif is_swesmith:
        if backend == "docker":
            return DockerSweSmithRuntime(ds, backend=backend, **kwargs)
        else:
            return KubernetesSweSmithRuntime(ds, backend=backend, **kwargs)
    elif is_mswebench:
        if backend == "docker":
            return DockerMSWEBenchRuntime(ds, backend=backend, **kwargs)
        else:
            return KubernetesMSWEBenchRuntime(ds, backend=backend, **kwargs)
    elif is_r2egym:
        if backend == "docker":
            return DockerR2EGymRuntime(ds, backend=backend, **kwargs)
        else:
            return KubernetesR2EGymRuntime(ds, backend=backend, **kwargs)
    else:
        raise ValueError(f"Unknown runtime type for docker image: {ds_image}")


# Backward compatibility alias
def DockerRuntime(ds, backend="docker", **kwargs):
    """
    Backward compatibility wrapper for existing code.

    This function maintains the same interface as the original DockerRuntime class
    but uses the new factory pattern internally.
    """
    return create_runtime(ds, backend=backend, **kwargs)
