import functools
import os
import subprocess
from contextlib import suppress

_PATCH_ENV = "RLLM_ENABLE_RAY_GPU_PROBE_PATCH"
_TIMEOUT_ENV = "RLLM_RAY_GPU_PROBE_TIMEOUT_S"
_FALSEY = {"", "0", "false", "no", "off"}


def _is_enabled() -> bool:
    value = str(os.environ.get(_PATCH_ENV, "")).strip().lower()
    return value not in _FALSEY


def _get_timeout_s() -> float:
    try:
        return float(os.environ.get(_TIMEOUT_ENV, "3"))
    except ValueError:
        return 3.0


def _apply_gpu_probe_patch(module) -> None:
    cls = getattr(module, "GpuProfilingManager", None)
    if cls is None:
        return

    descriptor = cls.__dict__.get("node_has_gpus")
    original = getattr(descriptor, "__func__", descriptor)
    if getattr(original, "_rllm_fast_probe_patched", False):
        return

    timeout_s = _get_timeout_s()

    @functools.cache
    def _fast_node_has_gpus(_cls) -> bool:
        try:
            subprocess.run(
                ["nvidia-smi", "-L"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout_s,
                check=True,
            )
            return True
        except Exception:
            return False

    _fast_node_has_gpus._rllm_fast_probe_patched = True
    cls.node_has_gpus = classmethod(_fast_node_has_gpus)


def _apply_verl_uvicorn_patch(module) -> None:
    current = getattr(module, "run_unvicorn", None)
    if getattr(current, "_rllm_uvicorn_patched", False):
        return

    async def _patched_run_unvicorn(app, server_args, server_address, max_retries=5):
        import asyncio

        import uvicorn

        server_port = None
        server_task = None

        for i in range(max_retries):
            sock = None
            try:
                server_port, sock = module.get_free_port(server_address)
                app.server_args = server_args
                config = uvicorn.Config(app, host=server_address, port=server_port, log_level="warning")
                server = uvicorn.Server(config)
                server_task = asyncio.create_task(server.serve(sockets=[sock]))

                while not server.started and not server_task.done():
                    await asyncio.sleep(0.1)

                if server_task.done():
                    exc = server_task.exception()
                    if exc is not None:
                        raise exc
                    raise RuntimeError("HTTP server exited before startup completed")

                break
            except (OSError, RuntimeError, SystemExit) as e:
                module.logger.error(f"Failed to start HTTP server on port {server_port} at try {i}, error: {e}")
                if server_task is not None:
                    server_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await server_task
                    server_task = None
                if sock is not None:
                    with suppress(OSError):
                        sock.close()
        else:
            module.logger.error(f"Failed to start HTTP server after {max_retries} retries, exiting...")
            os._exit(-1)

        module.logger.info(f"HTTP server started on port {server_port}")
        return server_port, server_task

    _patched_run_unvicorn._rllm_uvicorn_patched = True
    module.run_unvicorn = _patched_run_unvicorn


if _is_enabled():
    try:
        import wrapt
    except Exception:
        try:
            from ray.dashboard.modules.reporter import gpu_profile_manager as _gpu_profile_manager
        except Exception:
            pass
        else:
            _apply_gpu_probe_patch(_gpu_profile_manager)
        try:
            from verl.workers.rollout import utils as _verl_rollout_utils
        except Exception:
            pass
        else:
            _apply_verl_uvicorn_patch(_verl_rollout_utils)
    else:
        @wrapt.when_imported("ray.dashboard.modules.reporter.gpu_profile_manager")
        def _patch_gpu_profile_manager(module):
            _apply_gpu_probe_patch(module)

        @wrapt.when_imported("verl.workers.rollout.utils")
        def _patch_verl_rollout_utils(module):
            _apply_verl_uvicorn_patch(module)
