"""Code execution with sandboxing, resource limits, and test-case verification.

Supports:
- Isolated temporary working directory for BOTH single-file and multi-file runs
- Secret-stripped child environment (GEMINI_API_KEY etc. never leaked to user code)
- Hard wall-clock timeout enforcement
- OS resource limits:
  * Unix: resource.setrlimit (CPU, memory, file size)
  * Windows: Win32 Job Object memory/process limits + kill-on-close (with safe fallback)
  * Optional Docker sandbox mode (ODIN_SANDBOX_MODE=docker) with --network none
- Functional verification via optional `test_cases` (Python asserts) and/or
  `expected_output` (stdout comparison), recording per-test pass counts for
  Test Pass Rate and Partial Repair Rate metrics.
"""
from __future__ import annotations

import ast
import json
import logging
import os
import platform
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Optional

from odin.config import config
from odin.state import ExecutionRecord, FileContent

logger = logging.getLogger(__name__)

_SECRET_ENV_PATTERNS = ("API_KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


def execute_code(
    files: list[FileContent],
    timeout: int = 5,
    test_input: str = "",
    entry_point: Optional[str] = None,
    test_cases: str = "",
    expected_output: str = "",
) -> ExecutionRecord:
    """Execute Python code in an isolated sandbox and optionally verify tests.

    Args:
        files:           List of files comprising the program/project.
        timeout:         Maximum execution time in seconds.
        test_input:      Text piped to stdin during execution.
        entry_point:     Path to the main file to execute (defaults to main.py/first file).
        test_cases:      Optional Python assertion code executed against the program namespace.
        expected_output: Optional expected stdout string (trimmed comparison).

    Returns:
        ExecutionRecord with stdout, stderr, exit_code, traceback, timing, and test metrics.
    """
    start_time = time.perf_counter()

    if not files:
        return ExecutionRecord(
            iteration=0,
            stdout="",
            stderr="No source files provided for execution.",
            exit_code=-1,
            traceback="No source files provided for execution.",
            success=False,
            execution_time_ms=0.0,
        )

    try:
        result = _execute_in_sandbox(
            files=files,
            timeout=timeout,
            test_input=test_input,
            entry_point=entry_point,
            test_cases=test_cases,
        )

        stdout = result["stdout"]
        stderr = result["stderr"]
        exit_code = result["exit_code"]
        traceback_text = result["traceback"]
        tests_passed = result.get("tests_passed", 0)
        tests_total = result.get("tests_total", 0)
        test_passed: bool | None = result.get("test_passed")
        success = exit_code == 0 and (test_passed is not False)

        # Verify expected stdout when provided and process didn't crash
        if expected_output and expected_output.strip() and success:
            actual_norm = stdout.replace("\r\n", "\n").strip()
            expected_norm = expected_output.replace("\r\n", "\n").strip()
            tests_total += 1
            if actual_norm == expected_norm:
                tests_passed += 1
                test_passed = True if test_passed is None else (test_passed and True)
            else:
                success = False
                exit_code = 1
                test_passed = False
                mismatch = (
                    "AssertionError: Output mismatch\n"
                    f"Expected output:\n{expected_norm}\n\n"
                    f"Actual output:\n{actual_norm}"
                )
                stderr = f"{stderr}\n{mismatch}".strip() if stderr else mismatch
                traceback_text = mismatch

        execution_time_ms = (time.perf_counter() - start_time) * 1000
        return ExecutionRecord(
            iteration=0,  # set by caller
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            traceback=traceback_text,
            success=success,
            execution_time_ms=execution_time_ms,
            tests_passed=tests_passed,
            tests_total=tests_total,
            test_passed=test_passed,
        )

    except Exception as exc:
        execution_time_ms = (time.perf_counter() - start_time) * 1000
        logger.exception("Execution failed with infrastructure exception")
        return ExecutionRecord(
            iteration=0,
            stdout="",
            stderr=f"Execution infrastructure error: {exc}",
            exit_code=-1,
            traceback=str(exc),
            success=False,
            execution_time_ms=execution_time_ms,
        )


def _sanitized_env() -> dict[str, str]:
    """Return a copy of os.environ with secret keys removed."""
    clean: dict[str, str] = {}
    for k, v in os.environ.items():
        upper = k.upper()
        if any(pat in upper for pat in _SECRET_ENV_PATTERNS):
            continue
        clean[k] = v
    clean["PYTHONDONTWRITEBYTECODE"] = "1"
    clean["PYTHONIOENCODING"] = "utf-8"
    return clean


def _get_resource_preamble(timeout: int) -> str:
    """Return Python code to set Unix resource limits."""
    if platform.system() == "Windows":
        return ""

    mem_bytes = config.sandbox_memory_limit_mb * 1024 * 1024
    file_bytes = config.sandbox_file_size_limit_mb * 1024 * 1024

    return textwrap.dedent(f"""\
        import resource
        try:
            resource.setrlimit(resource.RLIMIT_CPU, ({timeout}, {timeout}))
            resource.setrlimit(resource.RLIMIT_AS, ({mem_bytes}, {mem_bytes}))
            resource.setrlimit(resource.RLIMIT_FSIZE, ({file_bytes}, {file_bytes}))
        except Exception:
            pass
    """)


def _build_test_harness_code(rel_main: str, test_cases: str, result_filename: str) -> str:
    """Build a runner that executes `rel_main` and evaluates `test_cases` per assertion."""
    # Pre-split top-level statements if possible so we can score partial pass rates
    individual_asserts: list[str] = []
    can_split_asserts = False
    try:
        tree = ast.parse(test_cases)
        if tree.body and all(isinstance(node, ast.Assert) for node in tree.body):
            can_split_asserts = True
            for node in tree.body:
                seg = ast.get_source_segment(test_cases, node)
                if seg:
                    individual_asserts.append(seg)
    except SyntaxError:
        can_split_asserts = False

    return textwrap.dedent(f"""\
        import json
        import runpy
        import sys
        import traceback

        sys.argv = [{rel_main!r}]
        ns = runpy.run_path({rel_main!r}, run_name="__odin_tested__")

        result_path = {result_filename!r}
        can_split = {can_split_asserts!r}
        assert_stmts = {individual_asserts!r}
        raw_tests = {test_cases!r}

        passed = 0
        total = 0
        failures = []

        if can_split and assert_stmts:
            total = len(assert_stmts)
            for idx, stmt in enumerate(assert_stmts, 1):
                try:
                    exec(compile(stmt, f"<test_case_{{idx}}>", "exec"), ns)
                    passed += 1
                except BaseException as exc:
                    tb = traceback.format_exc()
                    failures.append(f"Test {{idx}} failed ({{stmt}}):\\n{{tb}}")
        else:
            total = 1
            try:
                exec(compile(raw_tests, "<test_cases>", "exec"), ns)
                passed = 1
            except BaseException as exc:
                failures.append(traceback.format_exc())

        with open(result_path, "w", encoding="utf-8") as fh:
            json.dump({{"passed": passed, "total": total, "failures": failures}}, fh)

        if failures:
            sys.stderr.write("\\n".join(failures) + "\\n")
            sys.exit(1)
    """)


def _select_entry_point(files: list[FileContent], tmp_path: Path, entry_point: Optional[str]) -> Optional[Path]:
    if entry_point:
        candidate = tmp_path / entry_point
        return candidate if candidate.exists() else None
    for name in ("main.py", "app.py", files[0].path):
        candidate = tmp_path / name
        if candidate.exists():
            return candidate
    return None


def _execute_in_sandbox(
    files: list[FileContent],
    timeout: int,
    test_input: str,
    entry_point: Optional[str],
    test_cases: str = "",
) -> dict:
    """Write files to an isolated temporary directory and execute in subprocess or Docker."""
    with tempfile.TemporaryDirectory(prefix="odin_sandbox_") as tmpdir:
        tmp_path = Path(tmpdir)

        for file in files:
            file_path = tmp_path / file.path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(file.content, encoding="utf-8")

        main_file = _select_entry_point(files, tmp_path, entry_point)
        if main_file is None:
            return {
                "stdout": "",
                "stderr": "No entry point specified and no main.py/app.py found",
                "exit_code": -1,
                "traceback": "No entry point found",
            }

        rel_main = main_file.relative_to(tmp_path).as_posix()
        result_file = tmp_path / "__odin_test_result.json"
        has_tests = bool(test_cases and test_cases.strip())
        preamble = _get_resource_preamble(timeout)

        if has_tests:
            runner_code = preamble + _build_test_harness_code(
                rel_main=rel_main,
                test_cases=test_cases,
                result_filename=result_file.name,
            )
            runner_file = tmp_path / "__odin_runner__.py"
            runner_file.write_text(runner_code, encoding="utf-8")
            target_script = runner_file.name
        elif preamble:
            bootstrap_code = preamble + textwrap.dedent(f"""\
                import runpy
                import sys
                sys.argv = [{rel_main!r}]
                runpy.run_path({rel_main!r}, run_name="__main__")
            """)
            bootstrap_file = tmp_path / "__odin_bootstrap__.py"
            bootstrap_file.write_text(bootstrap_code, encoding="utf-8")
            target_script = bootstrap_file.name
        else:
            target_script = rel_main

        if config.sandbox_mode == "docker":
            exec_res = _run_in_docker(tmp_path, target_script, timeout, test_input)
        else:
            exec_res = _run_in_subprocess(tmp_path, target_script, timeout, test_input)

        if has_tests:
            if result_file.exists():
                try:
                    summary = json.loads(result_file.read_text(encoding="utf-8"))
                    exec_res["tests_passed"] = int(summary.get("passed", 0))
                    exec_res["tests_total"] = int(summary.get("total", 0))
                    exec_res["test_passed"] = (
                        exec_res["exit_code"] == 0
                        and exec_res["tests_total"] > 0
                        and exec_res["tests_passed"] == exec_res["tests_total"]
                    )
                except Exception:
                    exec_res["tests_passed"] = 0
                    exec_res["tests_total"] = 1
                    exec_res["test_passed"] = False
            else:
                # Program crashed (e.g. SyntaxError / runtime error) before tests ran
                exec_res["tests_passed"] = 0
                exec_res["tests_total"] = _count_test_cases(test_cases)
                exec_res["test_passed"] = False

        return exec_res


def _count_test_cases(test_cases: str) -> int:
    try:
        tree = ast.parse(test_cases)
        asserts = [n for n in tree.body if isinstance(n, ast.Assert)]
        return len(asserts) if asserts else 1
    except SyntaxError:
        return 1


def _run_in_docker(tmp_path: Path, target_script: str, timeout: int, test_input: str) -> dict:
    """Execute code inside a network-isolated, memory-capped Docker container."""
    cmd = [
        "docker", "run", "--rm", "-i",
        "--network", "none",
        "--memory", f"{config.sandbox_memory_limit_mb}m",
        "--cpus", "1",
        "--pids-limit", "64",
        "-v", f"{tmp_path.resolve()}:/workspace",
        "-w", "/workspace",
        config.sandbox_docker_image,
        "python", target_script,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            input=test_input,
            timeout=timeout + 5,  # allow container startup overhead
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode,
            "traceback": _extract_traceback(result.stderr),
        }
    except subprocess.TimeoutExpired as exc:
        stdout = _decode_output(exc.stdout)
        stderr = _decode_output(exc.stderr)
        return {
            "stdout": stdout,
            "stderr": stderr + f"\n\nTimeoutError: Execution exceeded {timeout} seconds",
            "exit_code": -1,
            "traceback": f"TimeoutError: Execution exceeded {timeout} seconds",
        }


def _run_in_subprocess(tmp_path: Path, target_script: str, timeout: int, test_input: str) -> dict:
    """Execute code in a local isolated subprocess with OS resource limits."""
    cmd = [sys.executable, target_script]
    env = _sanitized_env()

    if platform.system() == "Windows":
        return _run_windows_sandboxed_subprocess(cmd, tmp_path, env, timeout, test_input)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            input=test_input,
            timeout=timeout,
            cwd=str(tmp_path),
            env=env,
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode,
            "traceback": _extract_traceback(result.stderr),
        }
    except subprocess.TimeoutExpired as exc:
        stdout = _decode_output(exc.stdout)
        stderr = _decode_output(exc.stderr)
        return {
            "stdout": stdout,
            "stderr": stderr + f"\n\nTimeoutError: Execution exceeded {timeout} seconds",
            "exit_code": -1,
            "traceback": f"TimeoutError: Execution exceeded {timeout} seconds",
        }


def _run_windows_sandboxed_subprocess(
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    timeout: int,
    test_input: str,
) -> dict:
    """Run subprocess on Windows and attach a Win32 Job Object for memory & process cleanup."""
    job_handle = _create_windows_job_object(config.sandbox_memory_limit_mb)
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(cwd),
        env=env,
    )
    if job_handle:
        _assign_process_to_job(job_handle, proc._handle)  # type: ignore[attr-defined]

    try:
        stdout, stderr = proc.communicate(input=test_input, timeout=timeout)
        return {
            "stdout": stdout or "",
            "stderr": stderr or "",
            "exit_code": proc.returncode,
            "traceback": _extract_traceback(stderr or ""),
        }
    except subprocess.TimeoutExpired as exc:
        proc.kill()
        out_after, err_after = proc.communicate()
        stdout = _decode_output(exc.stdout) + _decode_output(out_after)
        stderr = _decode_output(exc.stderr) + _decode_output(err_after)
        return {
            "stdout": stdout,
            "stderr": stderr + f"\n\nTimeoutError: Execution exceeded {timeout} seconds",
            "exit_code": -1,
            "traceback": f"TimeoutError: Execution exceeded {timeout} seconds",
        }
    finally:
        if job_handle:
            _close_windows_handle(job_handle)


def _create_windows_job_object(memory_limit_mb: int) -> int | None:
    """Create a Win32 Job Object enforcing kill-on-close and per-process memory cap."""
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return None

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_uint64),
                ("WriteOperationCount", ctypes.c_uint64),
                ("OtherOperationCount", ctypes.c_uint64),
                ("ReadTransferCount", ctypes.c_uint64),
                ("WriteTransferCount", ctypes.c_uint64),
                ("OtherTransferCount", ctypes.c_uint64),
            ]

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
        JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
        JobObjectExtendedLimitInformation = 9

        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = (
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            | JOB_OBJECT_LIMIT_PROCESS_MEMORY
            | JOB_OBJECT_LIMIT_ACTIVE_PROCESS
        )
        info.BasicLimitInformation.ActiveProcessLimit = 16
        info.ProcessMemoryLimit = max(memory_limit_mb, 512) * 1024 * 1024

        ok = kernel32.SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if not ok:
            kernel32.CloseHandle(job)
            return None
        return int(job)
    except Exception:
        return None


def _assign_process_to_job(job_handle: int, proc_handle: int) -> None:
    try:
        import ctypes
        ctypes.windll.kernel32.AssignProcessToJobObject(
            ctypes.c_void_p(job_handle),
            ctypes.c_void_p(proc_handle),
        )
    except Exception:
        pass


def _close_windows_handle(handle: int) -> None:
    try:
        import ctypes
        ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(handle))
    except Exception:
        pass


def _decode_output(data: bytes | str | None) -> str:
    """Safely convert subprocess stdout/stderr (bytes, str, or None) to str."""
    if data is None:
        return ""
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return data


def _extract_traceback(stderr: str) -> str:
    """Extract the Python traceback from stderr."""
    if "Traceback (most recent call last):" in stderr:
        return stderr
    return stderr
