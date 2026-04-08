#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MSWebench 评估器 - 独立版本
整合了 multi-swe-bench 的核心测试解析和评估逻辑

用途：
1. 解析多语言测试输出
2. 计算 MSWebench 任务的成功率
3. 提供统一的语言基类，无需为每个仓库创建子类

使用方法：
    evaluator = MSWebenchEvaluator(language='golang')  # 或 'java', 'rust', 'python' 等
    reward = evaluator.calculate_reward(test_output, language='golang')
"""

import re
from typing import Dict, Tuple, Optional
from enum import Enum
from loguru import logger as loguru_logger


class Language(str, Enum):
    """支持的编程语言"""
    GOLANG = 'golang'
    JAVA = 'java'
    RUST = 'rust'
    PYTHON = 'python'
    JAVASCRIPT = 'javascript'
    TYPESCRIPT = 'typescript'
    C = 'c'
    CPP = 'cpp'
    PHP = 'php'
    RUBY = 'ruby'
    KOTLIN = 'kotlin'
    SCALA = 'scala'
    SWIFT = 'swift'
    CSHARP = 'csharp'


class TestResult:
    """测试结果数据类"""
    def __init__(self):
        self.passed: set[str] = set()
        self.failed: set[str] = set()
        self.skipped: set[str] = set()
        self.error: set[str] = set()
    
    @property
    def all_count(self) -> int:
        return len(self.passed) + len(self.failed) + len(self.skipped) + len(self.error)
    
    @property
    def pass_count(self) -> int:
        return len(self.passed)
    
    @property
    def fail_count(self) -> int:
        return len(self.failed) + len(self.error)


class MSWebenchEvaluator:
    """
    MSWebench 统一评估器
    
    特性：
    1. 支持多种编程语言
    2. 自动检测测试框架
    3. 统一的测试结果解析接口
    4. 无需为每个仓库创建子类
    """
    
    def __init__(self, language: Optional[str] = None, logger=None):
        """
        Args:
            language: 编程语言类型（可选，会自动检测）
            logger: 日志记录器（可选）
        """
        self.language = language
        self.logger = logger
    
    def _log_info(self, msg: str):
        if self.logger:
            self.logger.info(msg)
    
    def _log_warning(self, msg: str):
        if self.logger:
            self.logger.warning(msg)
    
    def _log_error(self, msg: str):
        if self.logger:
            self.logger.error(msg)
    
    # ========================================================================
    # Go 语言测试解析
    # ========================================================================
    
    def _parse_golang_output(self, test_log: str) -> TestResult:
        """
        解析 Go test 输出
        
        格式示例：
            --- PASS: TestAdd (0.00s)
            --- FAIL: TestSubtract (0.01s)
            --- SKIP: TestMultiply (0.00s)
        """
        result = TestResult()
        
        # 匹配 go test 输出
        pass_pattern = r'--- PASS:\s+(\S+)'
        fail_pattern = r'--- FAIL:\s+(\S+)'
        skip_pattern = r'--- SKIP:\s+(\S+)'
        
        for match in re.finditer(pass_pattern, test_log):
            result.passed.add(match.group(1))
        
        for match in re.finditer(fail_pattern, test_log):
            result.failed.add(match.group(1))
        
        for match in re.finditer(skip_pattern, test_log):
            result.skipped.add(match.group(1))
        
        return result
    
    # ========================================================================
    # Rust 语言测试解析
    # ========================================================================
    
    def _parse_rust_output(self, test_log: str) -> TestResult:
        """
        解析 Cargo test 输出
        
        格式示例：
            test tests::test_add ... ok
            test tests::test_subtract ... FAILED
            test tests::test_multiply ... ignored
        """
        result = TestResult()
        
        # 匹配 cargo test 输出
        pass_pattern = r'test\s+(\S+)\s+\.\.\.\s+ok'
        fail_pattern = r'test\s+(\S+)\s+\.\.\.\s+FAILED'
        skip_pattern = r'test\s+(\S+)\s+\.\.\.\s+ignored'
        
        for match in re.finditer(pass_pattern, test_log):
            result.passed.add(match.group(1))
        
        for match in re.finditer(fail_pattern, test_log):
            result.failed.add(match.group(1))
        
        for match in re.finditer(skip_pattern, test_log):
            result.skipped.add(match.group(1))
        
        return result
    
    # ========================================================================
    # Java 语言测试解析
    # ========================================================================
    
    def _parse_java_output(self, test_log: str) -> TestResult:
        """
        解析 Maven/Gradle 测试输出
        
        支持格式：
        - Maven Surefire: [INFO] Tests run: 5, Failures: 1, Errors: 0, Skipped: 1
        - JUnit: ✓ testAdd, ✗ testSubtract
        """
        result = TestResult()
        
        # Maven Surefire 格式
        surefire_pattern = r'Tests run:\s*(\d+),\s*Failures:\s*(\d+),\s*Errors:\s*(\d+),\s*Skipped:\s*(\d+)'
        match = re.search(surefire_pattern, test_log)
        if match:
            total = int(match.group(1))
            failures = int(match.group(2))
            errors = int(match.group(3))
            skipped = int(match.group(4))
            passed = total - failures - errors - skipped
            
            # 创建虚拟测试名称（因为 Surefire 不总是提供详细名称）
            for i in range(passed):
                result.passed.add(f"test_{i}")
            for i in range(failures):
                result.failed.add(f"failed_test_{i}")
            for i in range(errors):
                result.error.add(f"error_test_{i}")
            for i in range(skipped):
                result.skipped.add(f"skipped_test_{i}")
            
            return result
        
        # JUnit 详细格式
        pass_pattern = r'\[INFO\]\s+(\S+)\s+Time elapsed.*SUCCESS'
        fail_pattern = r'\[INFO\]\s+(\S+)\s+Time elapsed.*FAILURE'
        error_pattern = r'\[ERROR\]\s+(\S+)'
        skip_pattern = r'\[INFO\]\s+(\S+).*SKIPPED'
        
        for match in re.finditer(pass_pattern, test_log):
            result.passed.add(match.group(1))
        
        for match in re.finditer(fail_pattern, test_log):
            result.failed.add(match.group(1))
        
        for match in re.finditer(error_pattern, test_log):
            result.error.add(match.group(1))
        
        for match in re.finditer(skip_pattern, test_log):
            result.skipped.add(match.group(1))
        
        return result
    
    # ========================================================================
    # Python 语言测试解析
    # ========================================================================
    
    def _parse_python_output(self, test_log: str) -> TestResult:
        """
        解析 pytest/unittest 输出
        
        pytest 格式：
            test_module.py::test_function PASSED
            test_module.py::test_function FAILED
        
        unittest 格式:
            test_add (test_module.TestClass) ... ok
            test_subtract (test_module.TestClass) ... FAIL
        """
        result = TestResult()
        
        # pytest 格式
        pytest_pass = r'(\S+::\S+)\s+PASSED'
        pytest_fail = r'(\S+::\S+)\s+FAILED'
        pytest_skip = r'(\S+::\S+)\s+SKIPPED'
        pytest_error = r'(\S+::\S+)\s+ERROR'
        
        for match in re.finditer(pytest_pass, test_log):
            result.passed.add(match.group(1))
        
        for match in re.finditer(pytest_fail, test_log):
            result.failed.add(match.group(1))
        
        for match in re.finditer(pytest_skip, test_log):
            result.skipped.add(match.group(1))
        
        for match in re.finditer(pytest_error, test_log):
            result.error.add(match.group(1))
        
        # unittest 格式
        unittest_ok = r'(\w+)\s+\([^)]+\)\s+\.\.\.\s+ok'
        unittest_fail = r'(\w+)\s+\([^)]+\)\s+\.\.\.\s+FAIL'
        unittest_skip = r'(\w+)\s+\([^)]+\)\s+\.\.\.\s+skipped'
        unittest_error = r'(\w+)\s+\([^)]+\)\s+\.\.\.\s+ERROR'
        
        for match in re.finditer(unittest_ok, test_log):
            result.passed.add(match.group(1))
        
        for match in re.finditer(unittest_fail, test_log):
            result.failed.add(match.group(1))
        
        for match in re.finditer(unittest_skip, test_log):
            result.skipped.add(match.group(1))
        
        for match in re.finditer(unittest_error, test_log):
            result.error.add(match.group(1))
        
        return result
    
    # ========================================================================
    # JavaScript/TypeScript 测试解析
    # ========================================================================
    
    def _parse_javascript_output(self, test_log: str) -> TestResult:
        """
        解析 Jest/Mocha 输出
        
        Jest 格式：
            ✓ should add two numbers (5ms)
            ✗ should subtract two numbers
        
        Mocha 格式：
            √ should add two numbers
            × should subtract two numbers
        """
        result = TestResult()
        
        # Jest/Mocha 格式 (多种符号)
        pass_patterns = [
            r'✓\s+(.+?)(?:\s+\(\d+ms\))?$',
            r'√\s+(.+?)(?:\s+\(\d+ms\))?$',
            r'PASS\s+(.+)$',
        ]
        
        fail_patterns = [
            r'✗\s+(.+)$',
            r'×\s+(.+)$',
            r'FAIL\s+(.+)$',
        ]
        
        skip_patterns = [
            r'○\s+(.+)$',
            r'SKIP\s+(.+)$',
        ]
        
        for pattern in pass_patterns:
            for match in re.finditer(pattern, test_log, re.MULTILINE):
                result.passed.add(match.group(1).strip())
        
        for pattern in fail_patterns:
            for match in re.finditer(pattern, test_log, re.MULTILINE):
                result.failed.add(match.group(1).strip())
        
        for pattern in skip_patterns:
            for match in re.finditer(pattern, test_log, re.MULTILINE):
                result.skipped.add(match.group(1).strip())
        
        return result
    
    # ========================================================================
    # C/C++ 测试解析
    # ========================================================================
    
    def _parse_c_cpp_output(self, test_log: str) -> TestResult:
        """
        解析 Make/CTest/Google Test 输出
        
        Google Test 格式：
            [       OK ] TestSuite.TestName (0 ms)
            [  FAILED  ] TestSuite.TestName (5 ms)
        
        CTest 格式：
            Test #1: test_add ...................   Passed
            Test #2: test_subtract ...............   Failed
        """
        result = TestResult()
        
        # Google Test 格式
        gtest_pass = r'\[\s+OK\s+\]\s+(\S+)'
        gtest_fail = r'\[\s+FAILED\s+\]\s+(\S+)'
        gtest_skip = r'\[\s+SKIPPED\s+\]\s+(\S+)'
        
        for match in re.finditer(gtest_pass, test_log):
            result.passed.add(match.group(1))
        
        for match in re.finditer(gtest_fail, test_log):
            result.failed.add(match.group(1))
        
        for match in re.finditer(gtest_skip, test_log):
            result.skipped.add(match.group(1))
        
        # CTest 格式
        ctest_pass = r'Test\s+#\d+:\s+(\S+)\s+\.+\s+Passed'
        ctest_fail = r'Test\s+#\d+:\s+(\S+)\s+\.+\s+Failed'
        
        for match in re.finditer(ctest_pass, test_log):
            result.passed.add(match.group(1))
        
        for match in re.finditer(ctest_fail, test_log):
            result.failed.add(match.group(1))
        
        # Make check 格式 (pass/fail count)
        make_summary = r'# TOTAL:\s*(\d+)\s+# PASS:\s*(\d+)\s+# SKIP:\s*(\d+)\s+# XFAIL:\s*(\d+)\s+# FAIL:\s*(\d+)'
        match = re.search(make_summary, test_log)
        if match:
            total = int(match.group(1))
            passed = int(match.group(2))
            skipped = int(match.group(3))
            failed = int(match.group(5))
            
            for i in range(passed):
                result.passed.add(f"test_{i}")
            for i in range(failed):
                result.failed.add(f"failed_test_{i}")
            for i in range(skipped):
                result.skipped.add(f"skipped_test_{i}")
        
        return result
    
    # ========================================================================
    # PHP 测试解析
    # ========================================================================
    
    def _parse_php_output(self, test_log: str) -> TestResult:
        """
        解析 PHPUnit 输出
        
        格式：
            OK (10 tests, 25 assertions)
            FAILURES!
            Tests: 10, Assertions: 25, Failures: 2
        """
        result = TestResult()
        
        # PHPUnit 摘要格式
        success_pattern = r'OK\s+\((\d+)\s+tests?,\s+\d+\s+assertions?\)'
        match = re.search(success_pattern, test_log)
        if match:
            test_count = int(match.group(1))
            for i in range(test_count):
                result.passed.add(f"test_{i}")
            return result
        
        # 失败格式
        failure_pattern = r'Tests:\s*(\d+),\s*Assertions:\s*\d+,\s*Failures:\s*(\d+)(?:,\s*Errors:\s*(\d+))?(?:,\s*Skipped:\s*(\d+))?'
        match = re.search(failure_pattern, test_log)
        if match:
            total = int(match.group(1))
            failures = int(match.group(2))
            errors = int(match.group(3)) if match.group(3) else 0
            skipped = int(match.group(4)) if match.group(4) else 0
            passed = total - failures - errors - skipped
            
            for i in range(passed):
                result.passed.add(f"test_{i}")
            for i in range(failures):
                result.failed.add(f"failed_test_{i}")
            for i in range(errors):
                result.error.add(f"error_test_{i}")
            for i in range(skipped):
                result.skipped.add(f"skipped_test_{i}")
        
        return result
    
    # ========================================================================
    # 语言自动检测
    # ========================================================================
    
    def _detect_language(self, test_log: str) -> Optional[str]:
        """
        根据测试输出自动检测编程语言
        
        Returns:
            检测到的语言名称，如果无法检测则返回 None
        """
        log_lower = test_log.lower()
        
        # Go 特征
        if '--- pass:' in log_lower or '--- fail:' in log_lower or 'go test' in log_lower:
            return Language.GOLANG
        
        # Rust 特征
        if 'cargo test' in log_lower or 'test result: ok' in log_lower or 'test result: failed' in log_lower:
            return Language.RUST
        
        # Java 特征
        if 'tests run:' in log_lower and ('failures:' in log_lower or 'errors:' in log_lower):
            return Language.JAVA
        
        # Python 特征
        if 'pytest' in log_lower or '::test_' in log_lower or 'unittest' in log_lower:
            return Language.PYTHON
        
        # JavaScript/TypeScript 特征
        if 'jest' in log_lower or 'mocha' in log_lower or '✓' in test_log or '✗' in test_log:
            return Language.JAVASCRIPT
        
        # C/C++ 特征
        if '[       ok ]' in log_lower or '[  failed  ]' in log_lower or 'ctest' in log_lower:
            return Language.CPP
        
        # PHP 特征
        if 'phpunit' in log_lower:
            return Language.PHP
        
        self._log_warning(f"无法自动检测语言类型")
        return None
    
    # ========================================================================
    # 统一解析接口
    # ========================================================================
    
    def parse_test_output(self, test_log: str, language: Optional[str] = None) -> TestResult:
        """
        统一的测试输出解析接口
        
        Args:
            test_log: 测试输出日志
            language: 编程语言（可选，会自动检测）
        
        Returns:
            TestResult 对象
        """
        # 使用指定语言或自动检测
        lang = language or self.language or self._detect_language(test_log)
        
        if not lang:
            self._log_error("无法确定语言类型，返回空结果")
            return TestResult()
        
        self._log_info(f"使用语言解析器: {lang}")
        
        # 调用对应语言的解析器
        parsers = {
            Language.GOLANG: self._parse_golang_output,
            Language.RUST: self._parse_rust_output,
            Language.JAVA: self._parse_java_output,
            Language.PYTHON: self._parse_python_output,
            Language.JAVASCRIPT: self._parse_javascript_output,
            Language.TYPESCRIPT: self._parse_javascript_output,  # 使用相同解析器
            Language.C: self._parse_c_cpp_output,
            Language.CPP: self._parse_c_cpp_output,
            Language.PHP: self._parse_php_output,
        }
        
        parser = parsers.get(lang)
        if not parser:
            self._log_warning(f"语言 {lang} 暂无专用解析器，尝试通用解析")
            return self._parse_generic_output(test_log)
        
        return parser(test_log)
    
    def _parse_generic_output(self, test_log: str) -> TestResult:
        """
        通用解析器（作为后备方案）
        尝试识别常见的 pass/fail 模式
        """
        result = TestResult()
        
        # 通用的成功模式
        pass_keywords = ['pass', 'ok', 'success', 'passed', '✓', '√']
        # 通用的失败模式
        fail_keywords = ['fail', 'error', 'failed', '✗', '×']
        
        lines = test_log.split('\n')
        for line in lines:
            line_lower = line.lower()
            
            # 跳过空行和纯符号行
            if not line.strip() or len(line.strip()) < 3:
                continue
            
            # 检查是否包含测试名称的模式
            if 'test' in line_lower:
                # 判断是通过还是失败
                if any(keyword in line_lower for keyword in pass_keywords):
                    # 尝试提取测试名称
                    test_name = line.strip()[:100]  # 限制长度
                    result.passed.add(test_name)
                elif any(keyword in line_lower for keyword in fail_keywords):
                    test_name = line.strip()[:100]
                    result.failed.add(test_name)
        
        return result
    
    # ========================================================================
    # 奖励计算（用于 RL 训练）
    # ========================================================================
    
    def calculate_reward(
        self, 
        test_output: str, 
        exit_code: str = "0",
        language: Optional[str] = None,
        strict_mode: bool = False
    ) -> Tuple[float, Dict[str, any]]:
        """
        计算测试奖励（用于强化学习训练）
        
        Args:
            test_output: 测试输出日志
            exit_code: 退出码
            language: 编程语言（可选）
            strict_mode: 严格模式（True=必须有明确的通过信号，False=允许退出码判断）
        
        Returns:
            (reward, details) 元组
            - reward: 0.0 到 1.0 之间的奖励值
            - details: 包含详细信息的字典
        """
        # 解析测试输出
        test_result = self.parse_test_output(test_output, language)
        
        # 详细信息
        details = {
            'total_tests': test_result.all_count,
            'passed': test_result.pass_count,
            'failed': test_result.fail_count,
            'skipped': len(test_result.skipped),
            'exit_code': exit_code,
        }
        
        # 如果没有解析到任何测试，使用后备策略
        if test_result.all_count == 0:
            self._log_warning("没有解析到测试结果，使用后备策略")
            
            if strict_mode:
                # 严格模式：必须有明确的测试输出
                details['reward_reason'] = '没有测试输出 (严格模式)'
                return 0.0, details
            
            # 非严格模式：使用退出码和输出模式判断
            reward = self._calculate_fallback_reward(test_output, exit_code)
            details['reward_reason'] = '使用后备策略判断'
            return reward, details
        
        # 如果有失败的测试，奖励为 0
        if test_result.fail_count > 0:
            details['reward_reason'] = f'{test_result.fail_count} 个测试失败'
            return 0.0, details
        
        # 所有测试通过，奖励为 1.0
        if test_result.pass_count > 0:
            details['reward_reason'] = f'所有 {test_result.pass_count} 个测试通过'
            return 1.0, details
        
        # 只有跳过的测试（视为部分成功）
        if test_result.all_count == len(test_result.skipped):
            details['reward_reason'] = '所有测试被跳过'
            return 0.5, details
        
        # 默认情况
        details['reward_reason'] = '未知状态'
        return 0.0, details
    
    def _calculate_fallback_reward(self, output: str, exit_code: str) -> float:
        """
        后备奖励计算策略（当无法解析测试输出时使用）
        
        策略：
        1. 检查明确的失败模式
        2. 检查明确的成功模式
        3. 使用退出码判断
        """
        out_lower = output.lower()
        
        # === 明确的失败模式 ===
        failure_patterns = [
            'build failure',
            'build failed',
            'compilation error',
            'test result: failed',
            'tests failed',
            'failure!',
            'error:',
            'fatal:',
        ]
        
        for pattern in failure_patterns:
            if pattern in out_lower:
                self._log_info(f"检测到失败模式: {pattern}")
                return 0.0
        
        # === 明确的成功模式 ===
        success_patterns = [
            'build success',
            'test result: ok',
            'all tests passed',
            'tests passed',
            'ok (',  # 某些框架的格式
        ]
        
        for pattern in success_patterns:
            if pattern in out_lower:
                self._log_info(f"检测到成功模式: {pattern}")
                return 1.0
        
        # === 使用退出码 ===
        if exit_code in ['0', 0]:
            # 退出码为 0，但没有明确的测试输出
            # 检查是否有任何测试相关的输出
            if 'test' in out_lower:
                self._log_info("退出码为 0 且有测试输出")
                return 1.0
            else:
                self._log_warning("退出码为 0 但没有测试输出，可能不是测试命令")
                return 0.5  # 部分成功
        
        # 非零退出码
        self._log_info(f"退出码为 {exit_code}，判定为失败")
        return 0.0


# ============================================================================
# 便捷函数
# ============================================================================

def evaluate_mswebench_test(
    test_output: str,
    exit_code: str = "0",
    language: Optional[str] = None,
    strict_mode: bool = False,
    logger=None
) -> Tuple[float, Dict[str, any]]:
    """
    便捷函数：评估 MSWebench 测试结果
    
    Args:
        test_output: 测试输出日志
        exit_code: 退出码（字符串）
        language: 编程语言（可选，会自动检测）
        strict_mode: 严格模式（默认 False）
        logger: 日志记录器（可选）
    
    Returns:
        (reward, details) 元组
    
    示例:
        >>> output = "--- PASS: TestAdd (0.00s)\\n--- PASS: TestSub (0.00s)"
        >>> reward, details = evaluate_mswebench_test(output, exit_code="0")
        >>> print(f"奖励: {reward}, 通过: {details['passed']}")
    """
    evaluator = MSWebenchEvaluator(language=language, logger=logger)
    return evaluator.calculate_reward(test_output, exit_code, language, strict_mode)


# ============================================================================
# 用于 DockerRuntime 集成的方法
# ============================================================================

def calculate_reward_mswebench(
    docker_runtime,
    get_test_output: bool = False,
    timeout: int = 300,
    language: Optional[str] = None,
    strict_mode: bool = False,
) -> float:
    """
    集成到 DockerRuntime 的奖励计算方法
    Args:
        docker_runtime: DockerRuntime 实例（需要有 run, logger 等方法）
        get_test_output: 是否返回测试输出
        timeout: 超时时间（秒）
        language: 编程语言（可选）
        strict_mode: 严格模式
    
    Returns:
        如果 get_test_output=True: 返回 (reward, test_output)
        否则: 返回 reward
    """
    logger = getattr(docker_runtime, 'logger', loguru_logger)

    
    # 创建评估器
    evaluator = MSWebenchEvaluator(language=language, logger=logger)
    
    logger.info("运行 MSWebench fix-run.sh")
    
    # agent修复之后, apply test.patch + 测试
    fix_output, fix_exit_code = docker_runtime.run("/home/fix-run.sh", timeout=timeout)
    
    logger.debug(f"MSWebench fix测试输出长度: {len(fix_output)} 字符")
    logger.debug(f"MSWebench fix退出码: {fix_exit_code}")
    
    # 计算奖励
    try:
        fix_result = evaluator.parse_test_output(fix_output, language)
        logger.info(f"Fix测试结果: 总计={fix_result.all_count}, "
                    f"通过={fix_result.pass_count}, "
                    f"失败={fix_result.fail_count}")
        

        reward, details = evaluator.calculate_reward(
            fix_output, 
            fix_exit_code, 
            language=language,
            strict_mode=strict_mode
        )
        
        logger.info(f"奖励: {reward:.2f}")
        logger.info(f"原因: {details['reward_reason']}")
        logger.info(f"测试统计: 总计={details['total_tests']}, "
                    f"通过={details['passed']}, "
                    f"失败={details['failed']}, "
                    f"跳过={details['skipped']}")
        
        if get_test_output:
            return reward, fix_output
        return reward
        
    except Exception as e:
        if logger:
            logger.error(f"解析 MSWebench 测试输出时出错: {repr(e)}")
        
        if get_test_output:
            return 0.0, fix_output if 'fix_output' in locals() else ""
        return 0.0


# ============================================================================
# 主程序（用于测试）
# ============================================================================

if __name__ == '__main__':
    # 测试示例
    print("=" * 70)
    print("MSWebench 评估器测试")
    print("=" * 70)
    
    # Go 语言测试
    print("\n【Go 语言测试】")
    go_output = """
=== RUN   TestAdd
--- PASS: TestAdd (0.00s)
=== RUN   TestSubtract
--- PASS: TestSubtract (0.01s)
=== RUN   TestMultiply
--- SKIP: TestMultiply (0.00s)
PASS
ok      example.com/calc    0.015s
"""
    reward, details = evaluate_mswebench_test(go_output, "0", language='golang')
    print(f"Go 测试奖励: {reward}")
    print(f"详情: {details}")
    
    # Rust 语言测试
    print("\n【Rust 语言测试】")
    rust_output = """
running 3 tests
test tests::test_add ... ok
test tests::test_subtract ... ok
test tests::test_multiply ... ignored

test result: ok. 2 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out
"""
    reward, details = evaluate_mswebench_test(rust_output, "0", language='rust')
    print(f"Rust 测试奖励: {reward}")
    print(f"详情: {details}")
    
    # Java 语言测试
    print("\n【Java 语言测试】")
    java_output = """
[INFO] Tests run: 5, Failures: 0, Errors: 0, Skipped: 1
[INFO] BUILD SUCCESS
"""
    reward, details = evaluate_mswebench_test(java_output, "0", language='java')
    print(f"Java 测试奖励: {reward}")
    print(f"详情: {details}")
    
    # 失败案例
    print("\n【失败案例】")
    fail_output = """
[INFO] Tests run: 5, Failures: 2, Errors: 1, Skipped: 0
[INFO] BUILD FAILURE
"""
    reward, details = evaluate_mswebench_test(fail_output, "1", language='java')
    print(f"失败测试奖励: {reward}")
    print(f"详情: {details}")
    
    # 自动检测语言
    print("\n【自动检测语言】")
    reward, details = evaluate_mswebench_test(go_output, "0")  # 不指定语言
    print(f"自动检测奖励: {reward}")
    print(f"详情: {details}")
    
    print("\n" + "=" * 70)
    print("测试完成！")
