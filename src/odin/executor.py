import subprocess
import sys
from dataclasses import dataclass


@dataclass
class ExecutionResult:
    success: bool
    stdout: str
    stderr: str


def run_code(code: str, timeout: int = 5) -> ExecutionResult:
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = (exc.stderr or "") + "Execution timed out."
        return ExecutionResult(False, stdout, stderr)

    return ExecutionResult(result.returncode == 0, result.stdout, result.stderr)