"""Unit tests for error classification."""
import pytest

from odin.error_classifier import classify_error
from odin.state import ExecutionRecord


def test_syntax_error_classification():
    """Test that syntax errors are correctly classified."""
    execution = ExecutionRecord(
        iteration=1,
        stdout="",
        stderr="  File \"<string>\", line 1\n    def foo(\n          ^\nSyntaxError: unexpected EOF while parsing",
        exit_code=1,
        traceback="SyntaxError: unexpected EOF while parsing",
        success=False,
        execution_time_ms=10.0,
    )
    
    category, reasoning = classify_error(execution, test_passed=False)
    
    assert category == "syntax"
    assert "SyntaxError" in reasoning


def test_runtime_error_classification():
    """Test that simple runtime errors are correctly classified."""
    execution = ExecutionRecord(
        iteration=1,
        stdout="",
        stderr="Traceback (most recent call last):\n  File \"<string>\", line 2, in <module>\n    x = 1 / 0\nZeroDivisionError: division by zero",
        exit_code=1,
        traceback="ZeroDivisionError: division by zero",
        success=False,
        execution_time_ms=10.0,
    )
    
    category, reasoning = classify_error(execution, test_passed=False)
    
    assert category == "runtime"
    assert "ZeroDivisionError" in reasoning


def test_logical_error_classification():
    """Test that assertion failures are classified as logical errors."""
    execution = ExecutionRecord(
        iteration=1,
        stdout="",
        stderr="Traceback (most recent call last):\n  File \"<string>\", line 3, in <module>\n    assert result == 10\nAssertionError",
        exit_code=1,
        traceback="AssertionError",
        success=False,
        execution_time_ms=10.0,
    )
    
    category, reasoning = classify_error(execution, test_passed=False)
    
    assert category == "logical"
    assert "assertion" in reasoning.lower()


def test_library_api_error_classification():
    """Test that library API errors are correctly classified."""
    execution = ExecutionRecord(
        iteration=1,
        stdout="",
        stderr="Traceback (most recent call last):\n  File \"<string>\", line 5, in <module>\n    df.append(new_row)\nAttributeError: 'DataFrame' object has no attribute 'append'",
        exit_code=1,
        traceback="Traceback with pandas in it\nAttributeError: 'DataFrame' object has no attribute 'append'",
        success=False,
        execution_time_ms=10.0,
    )
    
    category, reasoning = classify_error(execution, test_passed=False)
    
    assert category == "library_api"
    assert "AttributeError" in reasoning


def test_version_specific_error_classification():
    """Test that version-specific errors are correctly classified."""
    execution = ExecutionRecord(
        iteration=1,
        stdout="",
        stderr="Traceback (most recent call last):\n  File \"<string>\", line 3\nAttributeError: DataFrame.append is deprecated and removed in pandas 2.0",
        exit_code=1,
        traceback="AttributeError: DataFrame.append is deprecated and removed in pandas 2.0",
        success=False,
        execution_time_ms=10.0,
    )
    
    category, reasoning = classify_error(execution, test_passed=False)
    
    assert category == "version_specific"
    assert "deprecated" in reasoning.lower() or "version" in reasoning.lower()


def test_uncommon_pattern_error_classification():
    """Test that uncommon patterns are detected."""
    execution = ExecutionRecord(
        iteration=1,
        stdout="",
        stderr="Traceback (most recent call last):\n  File \"<string>\", line 5, in <module>\n    result = itertools.groupby(data)\nAttributeError: something wrong with itertools usage",
        exit_code=1,
        traceback="Traceback with itertools\nAttributeError: something wrong with itertools usage",
        success=False,
        execution_time_ms=10.0,
    )
    
    category, reasoning = classify_error(execution, test_passed=False)
    
    assert category == "uncommon_pattern"
    assert "itertools" in reasoning.lower() or "uncommon" in reasoning.lower()


def test_successful_execution():
    """Test that successful executions don't classify errors."""
    execution = ExecutionRecord(
        iteration=1,
        stdout="Hello, world!",
        stderr="",
        exit_code=0,
        traceback="",
        success=True,
        execution_time_ms=10.0,
    )
    
    category, reasoning = classify_error(execution, test_passed=True)
    
    # Should still return a category (likely "logical")
    assert category == "logical"


def test_c_builtin_functools_reduce_classified_as_uncommon_pattern():
    execution = ExecutionRecord(
        iteration=1,
        stdout="",
        stderr="Traceback (most recent call last):\n  File \"main.py\", line 3\nTypeError: reduce() of empty iterable with no initial value",
        exit_code=1,
        traceback="TypeError: reduce() of empty iterable with no initial value",
        success=False,
        execution_time_ms=5.0,
    )
    category, reasoning = classify_error(execution, test_passed=False)
    assert category == "uncommon_pattern"
    assert "reduce" in reasoning.lower()


def test_removed_api_patterns_classified_as_version_specific():
    for msg in [
        "TypeError: NDFrame.fillna() got an unexpected keyword argument 'method'",
        "AttributeError: module 'numpy' has no attribute 'trapz'",
        "TypeError: OneHotEncoder.__init__() got an unexpected keyword argument 'sparse'",
    ]:
        execution = ExecutionRecord(
            iteration=1, stdout="", stderr=msg, exit_code=1, traceback=msg,
            success=False, execution_time_ms=5.0,
        )
        category, _ = classify_error(execution, test_passed=False)
        assert category == "version_specific", f"Failed for: {msg}"

