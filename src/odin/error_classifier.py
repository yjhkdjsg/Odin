"""Error classification logic for the ODIN framework.

This module analyzes execution failures and assigns them to one of six
error categories, each with different retrieval implications:

1. syntax - Code fails to parse (SyntaxError, IndentationError)
2. runtime - Invalid operations at runtime (TypeError, ZeroDivisionError, etc.)
3. logical - Code runs but produces wrong output (AssertionError, test failures)
4. library_api - Incorrect library/API usage (wrong method, wrong arguments)
5. version_specific - API changed across library versions
6. uncommon_pattern - Valid but rare usage patterns

The classification uses heuristics based on exception type, traceback content,
and error messages. Where signals are ambiguous, documented defaults are applied.
"""
import logging
import re
from typing import Optional

from odin.state import ErrorCategory, ExecutionRecord

logger = logging.getLogger(__name__)


# Exception types that map directly to categories
_SYNTAX_EXCEPTIONS = {"SyntaxError", "IndentationError", "TabError"}
_RUNTIME_EXCEPTIONS = {
    "ZeroDivisionError",
    "IndexError",
    "KeyError",
    "NameError",
    "UnboundLocalError",
    "RecursionError",
    "MemoryError",
    "OverflowError",
    "FloatingPointError",
}
_LOGICAL_EXCEPTIONS = {"AssertionError"}

# Patterns in error messages that suggest library API issues
_LIBRARY_API_PATTERNS = [
    r"got an unexpected keyword argument",
    r"takes \d+ positional argument",
    r"missing \d+ required positional argument",
    r"object has no attribute '[^']+'\s*$",  # Simple attribute access, likely API misuse
    r"module '[^']+' has no attribute",
    r"'[^']+' object is not callable",
]

# Patterns suggesting version-specific issues
_VERSION_SPECIFIC_PATTERNS = [
    r"deprecated",
    r"removed in version",
    r"no longer supported",
    r"has been renamed to",
    r"use .+ instead",
    r"'DataFrame' object has no attribute '(iteritems|applymap|swapaxes)'",
    r"fillna\(\) got an unexpected keyword argument 'method'",
    r"module 'numpy' has no attribute '(int|float|bool|complex|object|str|trapz|in1d|row_stack|product|cumproduct)'",
    r"OneHotEncoder\.__init__\(\) got an unexpected keyword argument 'sparse'",
    r"got an unexpected keyword argument 'squared'",
    r"has no attribute '(parse_obj|parse_raw)'",
]

# Library names that, when appearing in traceback, suggest API issues
_KNOWN_LIBRARIES = [
    "pandas",
    "numpy",
    "requests",
    "sklearn",
    "torch",
    "tensorflow",
    "matplotlib",
    "scipy",
    "flask",
    "django",
    "fastapi",
]


def classify_error(
    execution: ExecutionRecord,
    test_passed: bool = False,
) -> tuple[ErrorCategory, str]:
    """Classify an execution error into one of six categories.
    
    Args:
        execution: The execution record containing error information
        test_passed: Whether a test case passed (for logical error detection)
    
    Returns:
        A tuple of (category, reasoning) explaining the classification
    
    Classification logic:
    - syntax: Direct exception type match
    - runtime: Direct exception type match (simple runtime errors)
    - logical: AssertionError or test failure without crash
    - library_api: TypeError/AttributeError with library context or API-related message
    - version_specific: Error message contains version/deprecation keywords
    - uncommon_pattern: AttributeError/TypeError in stdlib with obscure patterns
    
    Ambiguity handling:
    - AttributeError defaults to runtime, but upgrades to library_api if a known
      library is in the traceback or message matches API patterns
    - TypeError is similar: runtime by default, library_api if context suggests it
    - If execution succeeds but test fails, it's a logical error
    """
    if execution.success:
        if not test_passed:
            return (
                "logical",
                "Code executed without crashing but failed test assertion",
            )
        # Should not reach here if calling code is correct
        return "logical", "Execution succeeded with test pass"
    
    stderr = execution.stderr
    traceback = execution.traceback or stderr
    
    # Extract the exception type
    exception_type = _extract_exception_type(traceback)
    
    # Check for syntax errors first (highest priority)
    if exception_type in _SYNTAX_EXCEPTIONS:
        return "syntax", f"Parser failure: {exception_type}"
    
    # Check for version-specific patterns early (before library_api)
    # These often manifest as AttributeError or similar but are version issues
    if _matches_any_pattern(_VERSION_SPECIFIC_PATTERNS, stderr):
        return (
            "version_specific",
            f"{exception_type} with deprecation/version keywords in message",
        )
    
    # Check for logical errors (explicit assertion failures)
    if exception_type in _LOGICAL_EXCEPTIONS:
        return "logical", "Explicit assertion failure"
    
    # Check for simple runtime errors
    if exception_type in _RUNTIME_EXCEPTIONS:
        return "runtime", f"Standard runtime exception: {exception_type}"
    
    # Now handle the ambiguous cases: TypeError and AttributeError
    # These can be runtime, library_api, or uncommon_pattern depending on context
    if exception_type in {"TypeError", "AttributeError"}:
        # Check library context
        library_context = _detect_library_context(traceback)
        api_pattern_match = _matches_any_pattern(_LIBRARY_API_PATTERNS, stderr)
        
        if library_context:
            return (
                "library_api",
                f"{exception_type} in {library_context} context",
            )
        
        if api_pattern_match:
            return (
                "library_api",
                f"{exception_type} with API-related message pattern",
            )
        
        # Check if it's a truly obscure pattern
        uncommon_signal = _detect_uncommon_pattern(traceback, stderr)
        if uncommon_signal:
            return (
                "uncommon_pattern",
                f"{exception_type} in uncommon pattern ({uncommon_signal})",
            )
        
        # Default to runtime for TypeError/AttributeError without special context
        return "runtime", f"Standard {exception_type} without library context"
    
    # ImportError/ModuleNotFoundError are often library_api (wrong import)
    if exception_type in {"ImportError", "ModuleNotFoundError"}:
        return "library_api", f"Import failure: {exception_type}"
    
    # FileNotFoundError, PermissionError, etc. - runtime issues
    if exception_type in {
        "FileNotFoundError",
        "PermissionError",
        "IsADirectoryError",
        "NotADirectoryError",
        "IOError",
        "OSError",
    }:
        return "runtime", f"File system or I/O error: {exception_type}"
    
    # Default: if we can't confidently classify, call it runtime
    # This is a safe default that won't trigger retrieval unless there's
    # a strong signal otherwise
    return "runtime", f"Unclassified {exception_type or 'error'}, defaulting to runtime"


def _extract_exception_type(traceback: str) -> Optional[str]:
    """Extract the exception type from a traceback.
    
    Looks for patterns like "TypeError:" or bare "AssertionError" at the start of a line.
    """
    matches = re.findall(
        r"^([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception))(?::|\s*$)",
        traceback,
        re.MULTILINE,
    )
    return matches[-1] if matches else None


def _matches_any_pattern(patterns: list[str], text: str) -> bool:
    """Check if text matches any of the given regex patterns."""
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def _detect_library_context(traceback: str) -> Optional[str]:
    """Detect if a known library appears in the traceback.
    
    Returns the library name if found, else None.
    """
    for lib in _KNOWN_LIBRARIES:
        # Look for library name in file paths or module references
        if re.search(rf"\b{lib}\b", traceback, re.IGNORECASE):
            return lib
    return None


def _detect_uncommon_pattern(traceback: str, stderr: str) -> Optional[str]:
    """Heuristic for detecting uncommon patterns.
    
    This looks for:
    - References to itertools, functools, collections.abc, asyncio, heapq
    - Built-in/C-extension functions whose module name doesn't appear in tracebacks
      (e.g. functools.reduce(), itertools.groupby(), MutableMapping, event loop)
    - Complex method chaining or descriptor protocol issues
    
    Returns the matched signal name if detected, else None.
    """
    uncommon_modules = ["itertools", "functools", "collections.abc", "asyncio", "heapq"]
    combined = f"{traceback}\n{stderr}"
    
    for module in uncommon_modules:
        if module in combined:
            return module

    uncommon_patterns = [
        (r"\breduce\(", "functools.reduce"),
        (r"\bgroupby\b", "itertools.groupby"),
        (r"\bMutableMapping\b|\bMutableSequence\b", "collections.abc"),
        (r"\blru_cache\b", "functools.lru_cache"),
        (r"event loop|coroutine", "asyncio"),
    ]
    for pat, label in uncommon_patterns:
        if re.search(pat, combined, re.IGNORECASE):
            return label
    
    # Async-related errors can be uncommon patterns
    if "async" in stderr.lower() or "await" in stderr.lower():
        return "async/await"
    
    # Descriptor protocol issues (less common)
    if "__get__" in stderr or "__set__" in stderr or "__delete__" in stderr:
        return "descriptor protocol"
    
    return None


def _is_uncommon_pattern(traceback: str, stderr: str) -> bool:
    """Boolean wrapper retained for backwards compatibility."""
    return _detect_uncommon_pattern(traceback, stderr) is not None
