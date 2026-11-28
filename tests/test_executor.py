"""Unit tests for code execution."""
import platform
import pytest

from odin.executor import execute_code
from odin.state import FileContent


def test_successful_execution():
    """Test executing code that succeeds."""
    code = "print('Hello, world!')"
    files = [FileContent(path="main.py", content=code)]
    
    result = execute_code(files, timeout=5, test_input="")
    
    assert result.success is True
    assert result.exit_code == 0
    assert "Hello, world!" in result.stdout
    assert result.stderr == ""


def test_failed_execution():
    """Test executing code that fails."""
    code = "x = 1 / 0"
    files = [FileContent(path="main.py", content=code)]
    
    result = execute_code(files, timeout=5, test_input="")
    
    assert result.success is False
    assert result.exit_code != 0
    assert "ZeroDivisionError" in result.stderr


def test_timeout_handling():
    """Test that timeout is enforced."""
    code = "import time\ntime.sleep(10)"
    files = [FileContent(path="main.py", content=code)]
    
    result = execute_code(files, timeout=1, test_input="")
    
    assert result.success is False
    assert "Timeout" in result.stderr or "Timeout" in result.traceback


def test_stdin_input():
    """Test passing input via stdin."""
    code = "x = input()\nprint(f'You entered: {x}')"
    files = [FileContent(path="main.py", content=code)]
    
    result = execute_code(files, timeout=5, test_input="hello")
    
    assert result.success is True
    assert "You entered: hello" in result.stdout


def test_syntax_error():
    """Test executing code with syntax errors."""
    code = "def foo(\n  pass"
    files = [FileContent(path="main.py", content=code)]
    
    result = execute_code(files, timeout=5, test_input="")
    
    assert result.success is False
    assert "SyntaxError" in result.stderr


def test_execution_time_measurement():
    """Test that execution time is measured."""
    code = "import time\ntime.sleep(0.1)\nprint('done')"
    files = [FileContent(path="main.py", content=code)]
    
    result = execute_code(files, timeout=5, test_input="")
    
    assert result.success is True
    assert result.execution_time_ms >= 100  # At least 100ms


@pytest.mark.skipif(platform.system() == "Windows", reason="Resource limits not supported on Windows")
def test_resource_limits_memory():
    """Test that memory limits are enforced in both single and multi-file modes."""
    # Attempt to allocate 300MB
    code = "x = ' ' * (300 * 1024 * 1024)" 
    files_single = [FileContent(path="main.py", content=code)]
    files_multi = [FileContent(path="main.py", content="import foo"), FileContent(path="foo.py", content=code)]

    from odin.config import config
    original_limit = config.sandbox_memory_limit_mb
    config.sandbox_memory_limit_mb = 100
    try:
        res_single = execute_code(files_single, timeout=5)
        assert res_single.success is False
        assert "MemoryError" in res_single.stderr or res_single.exit_code != 0

        res_multi = execute_code(files_multi, timeout=5)
        assert res_multi.success is False
        assert "MemoryError" in res_multi.stderr or res_multi.exit_code != 0
    finally:
        config.sandbox_memory_limit_mb = original_limit


@pytest.mark.skipif(platform.system() == "Windows", reason="Resource limits not supported on Windows")
def test_resource_limits_file_size():
    """Test that file size limits are enforced in both single and multi-file modes."""
    # Attempt to write 15MB file
    code = "with open('out.txt', 'wb') as f: f.write(b'0' * (15 * 1024 * 1024))" 
    files_single = [FileContent(path="main.py", content=code)]
    files_multi = [FileContent(path="main.py", content="import foo"), FileContent(path="foo.py", content=code)]

    from odin.config import config
    original_limit = config.sandbox_file_size_limit_mb
    config.sandbox_file_size_limit_mb = 10
    try:
        res_single = execute_code(files_single, timeout=5)
        assert res_single.success is False
        assert "File too large" in res_single.stderr or "OSError" in res_single.stderr or res_single.exit_code != 0

        res_multi = execute_code(files_multi, timeout=5)
        assert res_multi.success is False
        assert "File too large" in res_multi.stderr or "OSError" in res_multi.stderr or res_multi.exit_code != 0
    finally:
        config.sandbox_file_size_limit_mb = original_limit


@pytest.mark.skipif(platform.system() == "Windows", reason="Resource limits not supported on Windows")
def test_resource_limits_cpu():
    """Test that CPU limits are enforced in both single and multi-file modes."""
    code = "while True: pass"
    files_single = [FileContent(path="main.py", content=code)]
    files_multi = [FileContent(path="main.py", content="import foo"), FileContent(path="foo.py", content=code)]

    res_single = execute_code(files_single, timeout=1)
    assert res_single.success is False

    res_multi = execute_code(files_multi, timeout=1)
    assert res_multi.success is False


def test_single_file_isolated_temp_dir(tmp_path, monkeypatch):
    """Single-file runs execute inside an isolated temp dir, not the caller cwd."""
    monkeypatch.chdir(tmp_path)
    code = "with open('stray_file.txt', 'w') as f: f.write('hello')\nprint('ok')"
    result = execute_code([FileContent(path="main.py", content=code)], timeout=5)

    assert result.success is True
    assert not (tmp_path / "stray_file.txt").exists()


def test_sandbox_strips_secret_env_vars(monkeypatch):
    """Secret env vars like GEMINI_API_KEY must not leak into executed code."""
    monkeypatch.setenv("GEMINI_API_KEY", "super_secret_test_key")
    code = "import os\nprint(os.getenv('GEMINI_API_KEY', 'REDACTED'))"
    result = execute_code([FileContent(path="main.py", content=code)], timeout=5)

    assert result.success is True
    assert result.stdout.strip() == "REDACTED"


def test_test_cases_verifier_all_pass():
    code = "def add(a, b):\n    return a + b\n"
    tests = "assert add(1, 2) == 3\nassert add(-1, 1) == 0\nassert add(0, 0) == 0"
    result = execute_code([FileContent(path="main.py", content=code)], timeout=5, test_cases=tests)

    assert result.success is True
    assert result.test_passed is True
    assert result.tests_passed == 3
    assert result.tests_total == 3


def test_test_cases_verifier_partial_failure():
    code = "def add(a, b):\n    return a - b  # bug\n"
    tests = "assert add(0, 0) == 0\nassert add(2, 3) == 5"
    result = execute_code([FileContent(path="main.py", content=code)], timeout=5, test_cases=tests)

    assert result.success is False
    assert result.test_passed is False
    assert result.tests_passed == 1
    assert result.tests_total == 2
    assert "AssertionError" in result.stderr


def test_expected_output_mismatch_fails_as_logical():
    code = "print(41)"
    result = execute_code([FileContent(path="main.py", content=code)], timeout=5, expected_output="42")

    assert result.success is False
    assert result.test_passed is False
    assert result.tests_passed == 0
    assert result.tests_total == 1
    assert "AssertionError" in result.stderr


