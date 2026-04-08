# DockerRuntime Refactoring Guide

## Overview

The original `DockerRuntime` class has been refactored into a cleaner architecture with the following improvements:

1. **Separated Backend Logic**: Docker and Kubernetes operations are isolated into mixins
2. **Environment-Specific Subclasses**: Each benchmark (SWE-bench, SweSmith, MSWEBench, R2E-Gym) has its own class
3. **Eliminated Conditional Logic**: Removed scattered `if` statements for different environments
4. **Better Maintainability**: Each class has a single responsibility
5. **Easier Testing**: Each component can be tested independently

## Architecture

```
DockerRuntimeBase (Abstract Base)
├── Common operations: run, copy_to_container, apply_patch, etc.
├── Abstract methods: setup_env, _calculate_reward, parse_logs
│
├── Mixins:
│   ├── DockerBackendMixin (Docker SDK implementation)
│   └── KubernetesBackendMixin (K8s API implementation)
│
└── Specialized Runtimes:
    ├── SWEBenchRuntime (SWE-bench verified)
    ├── SweSmithRuntime (SweSmith benchmark)
    ├── MSWEBenchRuntime (Multi-SWE-bench)
    └── R2EGymRuntime (R2E-Gym benchmark)
```

## Usage

### Basic Usage (Recommended)

The simplest way to use the refactored code is through the factory function:

```python
from r2egym.agenthub.runtime.docker_ import create_runtime

# Automatically detects the correct runtime from dataset
runtime = create_runtime(ds, backend="docker")

# Use the runtime
runtime.setup_env()
patch = runtime.get_patch()
reward = runtime._calculate_reward()
runtime.close()
```

### Backward Compatibility

The refactored code maintains backward compatibility with the original interface:

```python
from r2egym.agenthub.runtime.docker_ import DockerRuntime

# This works exactly like the original DockerRuntime
runtime = DockerRuntime(ds, backend="docker")
```

### Direct Class Usage

If you know the specific benchmark type, you can instantiate classes directly:

```python
from r2egym.agenthub.runtime.docker_ import (
    DockerSWEBenchRuntime,
    KubernetesSWEBenchRuntime,
    DockerSweSmithRuntime,
    # ... etc
)

# For SWE-bench with Docker
runtime = DockerSWEBenchRuntime(ds)

# For SWE-bench with Kubernetes
runtime = KubernetesSWEBenchRuntime(ds)
```

## Migration Guide

### For Existing Code

**Option 1: Minimal Change (Recommended for quick migration)**

Simply change your import:

```python
# Before
from r2egym.agenthub.runtime.docker import DockerRuntime

# After
from r2egym.agenthub.runtime.docker_ import DockerRuntime
```

Everything else remains the same!

**Option 2: Use Factory Function**

```python
# Before
from r2egym.agenthub.runtime.docker import DockerRuntime
runtime = DockerRuntime(ds, backend="docker")

# After
from r2egym.agenthub.runtime.docker_ import create_runtime
runtime = create_runtime(ds, backend="docker")
```

### For New Code

Use the factory function for automatic runtime detection:

```python
from r2egym.agenthub.runtime.docker_ import create_runtime

runtime = create_runtime(
    ds=dataset_entry,
    backend="docker",  # or "kubernetes"
    repo_path="/testbed",
    alt_path="/root",
)

try:
    # Your code here
    runtime.run("git status")
    patch = runtime.get_patch()
finally:
    runtime.close()
```

## Class Hierarchy Details

### DockerRuntimeBase

The abstract base class providing common functionality:

**Common Methods:**
- `run(code, timeout, args, workdir)` - Execute commands
- `copy_to_container(src, dest)` - Copy files
- `apply_patch(patch)` - Apply git patches
- `get_patch()` - Get current diff
- `checkout(commit_hash)` - Checkout commits
- `run_tests(timeout)` - Run test scripts
- `start_container()` / `stop_container()` - Lifecycle management

**Abstract Methods (implemented by subclasses):**
- `setup_env()` - Environment-specific setup
- `_calculate_reward()` - Benchmark-specific reward calculation
- `parse_logs()` - Benchmark-specific log parsing

### Backend Mixins

**DockerBackendMixin:**
- Uses Docker SDK (`docker.from_env()`)
- Implements: `_init_backend_client()`, `_start_backend_container()`, `_backend_run()`, etc.

**KubernetesBackendMixin:**
- Uses Kubernetes API (`client.CoreV1Api()`)
- Implements: Pod creation, exec via streams, file copying via tar

### Specialized Runtimes

#### SWEBenchRuntime
- Parses `parsed_commit` from dataset
- Creates `TestSpec` for evaluation
- Uses SWE-bench log parsers
- Supports regression tests

#### SweSmithRuntime
- Adjusts image name format
- Manages test file resets
- Evaluates FAIL_TO_PASS and PASS_TO_PASS tests

#### MSWEBenchRuntime
- Uses `/home/{repo_name}` as repo path
- Loads offline test results
- Uses Multi-SWE-bench report generation

#### R2EGymRuntime
- Sets up virtual environment symlinks
- Manages skip files
- Compares against expected test outputs

## Examples

### Example 1: Running Tests

```python
from r2egym.agenthub.runtime.docker_ import create_runtime

runtime = create_runtime(ds, backend="docker")

try:
    # Run tests
    output, exit_code = runtime.run_tests(timeout=300)

    # Parse logs
    parsed = runtime.parse_logs(output)
    print(f"Test results: {parsed}")

    # Calculate reward
    reward = runtime._calculate_reward()
    print(f"Reward: {reward}")
finally:
    runtime.close()
```

### Example 2: Applying Patches

```python
from r2egym.agenthub.runtime.docker_ import create_runtime

runtime = create_runtime(ds, backend="kubernetes")

try:
    # Apply a patch
    patch = """
    diff --git a/file.py b/file.py
    ...
    """
    output, error = runtime.apply_patch(patch)

    # Run tests to verify
    test_output, _ = runtime.run_tests()

    # Get current changes
    current_patch = runtime.get_patch()
    print(f"Current changes:\n{current_patch}")
finally:
    runtime.close()
```

### Example 3: Environment-Specific Operations

```python
from r2egym.agenthub.runtime.docker_ import create_runtime

runtime = create_runtime(ds, backend="docker")

try:
    # For SWE-bench verified, run regression tests
    if isinstance(runtime, SWEBenchRuntime):
        regression_output = runtime.run_swebv_regression()
        print(f"Regression: {regression_output}")

    # For SweSmith, reset test files
    if isinstance(runtime, SweSmithRuntime):
        runtime.reset_swesmith_tests()
finally:
    runtime.close()
```

## Benefits of the Refactoring

### 1. Single Responsibility Principle
Each class has one job:
- `DockerBackendMixin` → Docker operations
- `KubernetesBackendMixin` → Kubernetes operations
- `SWEBenchRuntime` → SWE-bench specific logic
- etc.

### 2. Open/Closed Principle
Easy to add new environments without modifying existing code:

```python
class NewBenchmarkRuntime(DockerRuntimeBase):
    def setup_env(self):
        # New environment setup
        pass

    def _calculate_reward(self, ...):
        # New reward calculation
        pass

    def parse_logs(self, log_output):
        # New log parsing
        pass
```

### 3. Reduced Complexity
**Before:** 1400+ line file with nested conditionals
**After:** Modular classes averaging 100-200 lines each

### 4. Better Testing
Each component can be tested in isolation:

```python
def test_swebench_setup():
    runtime = DockerSWEBenchRuntime(mock_ds)
    runtime.setup_env()
    # Assert environment is set up correctly

def test_docker_backend_run():
    runtime = DockerSWEBenchRuntime(mock_ds)
    output, code = runtime.run("echo test")
    assert output == "test\n"
```

### 5. Type Safety
Clear interfaces make it easier to use type hints:

```python
def process_runtime(runtime: DockerRuntimeBase) -> float:
    """Process any runtime type."""
    runtime.run_tests()
    return runtime._calculate_reward()
```

## Troubleshooting

### Issue: "Not a {benchmark} image" error

**Cause:** Using the wrong runtime class directly

**Solution:** Use the factory function:
```python
# Wrong
runtime = SWEBenchRuntime(ds)  # If ds is not SWE-bench

# Right
runtime = create_runtime(ds)  # Auto-detects
```

### Issue: Missing methods from original DockerRuntime

**Cause:** Some rarely-used methods might not be ported yet

**Solution:**
1. Check if the method is in `DockerRuntimeBase`
2. If not, file an issue or add it to the appropriate subclass

### Issue: Import errors

**Cause:** Missing dependencies

**Solution:** Ensure all imports in `docker_.py` are available in your environment

## Future Enhancements

Potential improvements to the architecture:

1. **Strategy Pattern for Reward Calculation**
   ```python
   class RewardCalculator(ABC):
       @abstractmethod
       def calculate(self, runtime, timeout): pass
   ```

2. **Plugin System for New Benchmarks**
   - Auto-discover runtime classes
   - Register new environments dynamically

3. **Configuration Files**
   - Move environment-specific settings to YAML/JSON
   - Reduce hardcoded values

4. **Async Support**
   - Async command execution
   - Parallel container operations

## Contributing

When adding new features:

1. **Add common functionality** → `DockerRuntimeBase`
2. **Add backend-specific code** → Appropriate mixin
3. **Add benchmark-specific code** → Appropriate runtime subclass
4. **Add tests** → Test each component separately
5. **Update this guide** → Document new features

## Questions?

If you have questions about the refactoring:
- Check this guide first
- Look at the inline documentation in `docker_.py`
- Review the examples above
- Check the original `docker.py` for comparison
