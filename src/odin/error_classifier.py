import re


_CATEGORY_BY_EXCEPTION = {
    "SyntaxError": "syntax",
    "IndentationError": "syntax",
    "TabError": "syntax",
    "ImportError": "import",
    "ModuleNotFoundError": "import",
    "NameError": "runtime",
    "TypeError": "runtime",
    "ValueError": "runtime",
    "ZeroDivisionError": "runtime",
    "IndexError": "runtime",
    "KeyError": "runtime",
    "AttributeError": "runtime",
    "AssertionError": "logical",
}


def classify_error(traceback_text: str) -> str:
    matches = re.findall(r"(?m)^([A-Za-z_][A-Za-z0-9_]*Error)(?::|$)", traceback_text)
    if not matches:
        return "unknown"
    return _CATEGORY_BY_EXCEPTION.get(matches[-1], "unknown")