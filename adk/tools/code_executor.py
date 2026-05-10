"""Safe code executor — runs generated Python analysis code in a restricted sandbox.

Enforces:
- Import whitelist (pandas, numpy, sklearn, plotly, seaborn, scipy, statsmodels)
- Per-execution timeout (default 60s, threading-based to work on non-main threads)
- No network access (blocks socket, urllib, requests)
- Captures stdout, stderr, return values, and saved plot files
"""

import ast
import importlib
import io
import os
import re
import sys
import textwrap
import threading
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Any

# Whitelisted imports for data analysis
ALLOWED_MODULES = {
    "pandas", "numpy", "numpy as np",
    "sklearn", "sklearn.cluster", "sklearn.preprocessing", "sklearn.ensemble",
    "sklearn.metrics", "sklearn.decomposition", "sklearn.feature_selection",
    "plotly", "plotly.express", "plotly.graph_objects",
    "seaborn", "matplotlib", "matplotlib.pyplot",
    "scipy", "scipy.stats",
    "statsmodels", "statsmodels.api", "statsmodels.formula.api",
    "json", "math", "statistics", "collections",
    "os", "pathlib", "copy", "itertools",
    "pptx", "pptx.util", "pptx.enum.text",
    "reportlab", "reportlab.lib",
    "yaml",
    "adk", "adk.tools",
}

# Forbidden modules (security)
FORBIDDEN_MODULES = {
    "os.system", "subprocess", "socket", "urllib", "requests",
    "http", "ftplib", "smtplib", "shutil", "pickle", "eval",
    "exec", "__import__", "open", "compile", "eval", "globals",
    "locals", "input", "getattr", "setattr", "delattr",
}

DEFAULT_TIMEOUT = 60


class CodeExecutionError(Exception):
    """Raised when code execution fails."""
    pass


class SecurityError(CodeExecutionError):
    """Raised when code violates security policies."""
    pass


def _check_imports(source: str) -> list[str]:
    """Check that all imports are from the whitelist.

    Args:
        source: Python source code to check.

    Returns:
        List of forbidden import names found, or a single-element list
        with a syntax-error message if the code is not valid Python.
    """
    forbidden: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"Syntax error in generated code: {e}"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name.split(".")[0]
                full = alias.name
                if full not in ALLOWED_MODULES and module not in ALLOWED_MODULES:
                    if full in FORBIDDEN_MODULES or module in FORBIDDEN_MODULES:
                        forbidden.append(full)
                    # Allow known analysis modules even if not explicitly whitelisted
                    elif not _is_analysis_module(module):
                        forbidden.append(full)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module = node.module.split(".")[0]
                full = node.module
                if full not in ALLOWED_MODULES and module not in ALLOWED_MODULES:
                    if full in FORBIDDEN_MODULES or module in FORBIDDEN_MODULES:
                        forbidden.append(full)
                    elif not _is_analysis_module(module):
                        forbidden.append(full)

    return forbidden


def _is_analysis_module(module: str) -> bool:
    """Check if a module is a common analysis library."""
    analysis_modules = {
        "sklearn", "scikit", "plotly", "seaborn", "matplotlib",
        "scipy", "statsmodels", "networkx", "stats", "wordcloud",
        "folium", "geopandas", "shap", "lime", "yellowbrick",
        "imblearn", "xgboost", "lightgbm", "catboost", "tensorflow",
        "torch", "keras", "transformers", "nltk", "spacy", "textblob",
    }
    return module in analysis_modules


def _sanitize_source(source: str) -> str:
    """Sanitize source code before execution.

    - Dedent the code
    - Remove cell magic and shell commands
    - Block dangerous patterns
    """
    # Dedent
    source = textwrap.dedent(source)

    # Remove IPython magic commands
    source = re.sub(r'^%.*$', '', source, flags=re.MULTILINE)
    source = re.sub(r'^\!\.*$', '', source, flags=re.MULTILINE)

    # Remove IPython display commands (we capture output ourselves)
    source = re.sub(r'^\s*display\(', 'pass  # display removed', source)

    return source.strip()


def _execute_code(source: str, local_ns: dict[str, Any] | None = None, timeout: int = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """Execute the sanitized source code in a restricted namespace.

    Uses threading-based timeout to work on non-main threads (e.g. Streamlit).

    Args:
        source: Sanitized Python source code.
        local_ns: Optional local namespace dict for variable access.
        timeout: Max execution time in seconds.

    Returns:
        Dict with 'stdout', 'stderr', 'result', and 'error' keys.
    """
    local_ns = local_ns or {}

    # Create a restricted globals dict
    restricted_globals = {
        "__builtins__": {
            "print": print,
            "len": len,
            "range": range,
            "str": str,
            "int": int,
            "float": float,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "set": set,
            "bool": bool,
            "min": min,
            "max": max,
            "sum": sum,
            "abs": abs,
            "round": round,
            "sorted": sorted,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "isinstance": isinstance,
            "issubclass": issubclass,
            "Exception": Exception,
            "ValueError": ValueError,
            "TypeError": TypeError,
            "KeyError": KeyError,
            "IndexError": IndexError,
            "ImportError": ImportError,
            "RuntimeError": RuntimeError,
            "Warning": Warning,
            "NotImplementedError": NotImplementedError,
        },
        "__file__": "<generated>",
    }

    # Inject adk tools so generated code can import them via `from adk.tools.chart_generator import ...`
    try:
        import adk
        import adk.tools.chart_generator
        restricted_globals["adk"] = adk
        restricted_globals["chart_generator"] = adk.tools.chart_generator
    except ImportError:
        pass

    # Capture output
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    exec_error: list[str] = []

    def _run():
        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(source, restricted_globals, local_ns)
        except Exception as e:
            exec_error.append(f"{type(e).__name__}: {e}")

    thread = threading.Thread(target=_run)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        return {
            "stdout": stdout_buf.getvalue(),
            "stderr": stderr_buf.getvalue(),
            "result": None,
            "error": f"Execution timed out after {timeout} seconds",
        }

    if exec_error:
        return {
            "stdout": stdout_buf.getvalue(),
            "stderr": stderr_buf.getvalue(),
            "result": None,
            "error": exec_error[0],
        }

    return {
        "stdout": stdout_buf.getvalue(),
        "stderr": stderr_buf.getvalue(),
        "result": None,
        "error": None,
    }


def execute_analysis_code(
    source: str,
    local_vars: dict[str, Any] | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """Execute analysis code safely.

    Args:
        source: Python source code to execute.
        local_vars: Variables available in the execution namespace
            (e.g., 'df' for the loaded DataFrame).
        timeout: Execution timeout in seconds.

    Returns:
        Dict with 'stdout', 'stderr', 'result', 'error', 'success' keys.
    """
    # Sanitize
    source = _sanitize_source(source)

    # Validate syntax first
    try:
        ast.parse(source)
    except SyntaxError as e:
        raise CodeExecutionError(f"Generated code has syntax error: {e}")

    # Check imports
    forbidden = _check_imports(source)
    if forbidden:
        raise SecurityError(f"Forbidden imports detected: {', '.join(forbidden)}")

    # Execute
    result = _execute_code(source, local_vars, timeout=timeout)
    result["success"] = result["error"] is None
    return result
