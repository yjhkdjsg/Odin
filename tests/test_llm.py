import pytest
from odin.llm import _select_relevant_files_for_repair
from odin.state import FileContent
from odin.config import config

def test_select_relevant_files_single_file():
    files = [FileContent(path="main.py", content="x = 1/0")]
    result = _select_relevant_files_for_repair(files, "Traceback main.py line 1", "ZeroDivisionError")
    assert len(result) == 1

def test_select_relevant_files_multi_file_isolated():
    files = [
        FileContent(path="main.py", content="import helper; helper.do()"),
        FileContent(path="helper.py", content="def do(): return 1/0"),
        FileContent(path="other.py", content="pass")
    ]
    traceback = 'File "helper.py", line 1, in do\\nZeroDivisionError'
    result = _select_relevant_files_for_repair(files, traceback, "ZeroDivisionError")
    
    assert len(result) == 2
    paths = [f.path for f in result]
    assert "main.py" in paths  # always included
    assert "helper.py" in paths
    assert "other.py" not in paths

def test_select_relevant_files_import_error_fallback():
    files = [
        FileContent(path="main.py", content="import missing"),
        FileContent(path="helper.py", content="pass"),
        FileContent(path="other.py", content="pass")
    ]
    traceback = 'File "main.py", line 1, in <module>\\nImportError: missing'
    
    original_limit = config.repair_token_limit
    config.repair_token_limit = 1000
    try:
        result = _select_relevant_files_for_repair(files, traceback, "ImportError")
        assert len(result) == 3  # fallback includes all
    finally:
        config.repair_token_limit = original_limit

def test_select_relevant_files_token_truncation():
    files = [
        FileContent(path="main.py", content="m" * 10), # 10 chars = ~2 tokens
        FileContent(path="helper.py", content="h" * 100), # 100 chars = ~25 tokens
        FileContent(path="other.py", content="o" * 10), # 10 chars = ~2 tokens
    ]
    traceback = 'File "main.py", line 1, in <module>\\nImportError: missing'
    
    original_limit = config.repair_token_limit
    config.repair_token_limit = 10  # 10 tokens limit
    try:
        result = _select_relevant_files_for_repair(files, traceback, "ImportError")
        paths = [f.path for f in result]
        assert "main.py" in paths
        assert "helper.py" not in paths # truncated
        assert "other.py" not in paths # truncated
    finally:
        config.repair_token_limit = original_limit


def test_clean_single_file_strips_file_header_and_fences():
    from odin.llm import _parse_code_response

    raw = "```python\nFILE: main.py\nimport sys\nprint('hi')\n```"
    files = _parse_code_response(raw, is_multi_file=False)
    assert len(files) == 1
    assert files[0].path == "main.py"
    assert not files[0].content.startswith("FILE:")
    assert files[0].content.startswith("import sys")


def test_single_file_feedback_prompt_omits_file_header():
    from odin.llm import _build_feedback_prompt

    prompt = _build_feedback_prompt(
        requirement="palindrome checker",
        current_files=[FileContent(path="main.py", content="print('ok')")],
        user_feedback="read input from stdin",
        is_multi=False,
    )
    assert "FILE: main.py" not in prompt
    assert "Do NOT include any FILE: header lines" in prompt


def test_format_previous_attempts_includes_history():
    from odin.llm import _format_previous_attempts

    txt = _format_previous_attempts([
        {"iteration": "1", "category": "runtime", "error": "ZeroDivisionError"},
    ])
    assert "Iteration 1 [runtime]" in txt
    assert "ZeroDivisionError" in txt

