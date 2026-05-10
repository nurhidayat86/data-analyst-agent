"""Analyst agent — generates Python analysis code, executes it, and interprets results.

This is the core agent that handles data analysis requests. It:
1. Searches the tool registry for similar past analyses
2. Generates Python code using pandas, sklearn, plotly, etc.
3. Executes the code in a safe sandbox
4. Interprets the results and produces a narrative summary
"""

from typing import Any

import pandas as pd

from adk.models.router import ModelRouter
from adk.tools.code_executor import execute_analysis_code, SecurityError
from adk.tools.chart_generator import get_registry as get_chart_registry
from adk.tools.tool_registry import ToolRegistry
from adk.tools.schema_discovery import schema_to_prompt


# System prompt for the analyst agent
ANALYST_SYSTEM_PROMPT = """You are a data analysis expert. Write Python code that analyzes `df` (pandas DataFrame) and produces a `results` dict.

Available chart functions from `chart_generator`: scatter, histogram, bar, boxplot, heatmap, roc_curve, feature_importance, violin_plot, pie_chart
All chart functions accept df and return a Path. Extra kwargs like color_palette are ignored.

Rules:
- Always end with: print(results)
- Store results in a dict with keys: summary, findings, charts, details, stats
- Do NOT use try/except that swallows errors
- df is always available — do not check if it exists
- Keep code concise. Avoid verbose strings.

Example:
  from adk.tools.chart_generator import bar, boxplot
  chart = bar(df, x="col", title="Title")
  results = {"summary": "...", "findings": [...], "charts": [chart], "details": "", "stats": {}}
  print(results)"""


class AnalystAgent:
    """Agent that generates and executes analysis code.

    Usage:
        router = ModelRouter()
        registry = ToolRegistry()
        analyst = AnalystAgent(router, registry)
        result = analyst.analyze("cluster customers", df=my_dataframe)
    """

    def __init__(self, router: ModelRouter, tool_registry: ToolRegistry | None = None):
        self.router = router
        self.tool_registry = tool_registry or ToolRegistry()
        self.chart_registry = get_chart_registry()
        self.max_retries = 3

    def analyze(
        self,
        request: str,
        df: pd.DataFrame,
        schema: dict | None = None,
        model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, Any]:
        """Execute an analysis request.

        Args:
            request: User's analysis request in natural language.
            df: pandas DataFrame with the data.
            schema: Schema dict from schema_discovery (optional).
            model: Model name (optional).
            api_key: Runtime API key override.
            api_base: Runtime base URL override (for local models).

        Returns:
            Dict with 'success', 'results', 'error', 'model', 'code'.
        """
        # Build context
        schema_prompt = ""
        if schema:
            schema_prompt = schema_to_prompt(schema)

        # Search for similar tools
        similar_tools = self.tool_registry.search(request, top_k=2)
        tool_context = ""
        if similar_tools:
            tool_context = "\n\nRelevant past analyses:\n"
            for tool in similar_tools:
                tool_context += f"- {tool['name']}: {tool['description']}\n"
                tool_context += f"  Use case: {tool.get('use_case', 'N/A')}\n"

        # Build the code generation prompt
        code_prompt = self._build_code_prompt(request, schema_prompt, tool_context)

        # Generate and execute code (with retries)
        for attempt in range(self.max_retries):
            try:
                # Get code from LLM
                code_response = self.router.chat(
                    system_prompt=ANALYST_SYSTEM_PROMPT,
                    user_message=code_prompt,
                    model=model,
                    temperature=0.2,
                    max_tokens=8192,
                    api_key=api_key,
                    api_base=api_base,
                )

                import sys
                print(f"[analyst] Attempt {attempt + 1}: LLM response:\n{code_response}", file=sys.stderr)

                # Extract Python code from response
                code = self._extract_code(code_response)

                if not code:
                    code_prompt = (
                        f"You didn't provide Python code. The response was:\n{code_response}\n\n"
                        f"Please provide ONLY Python code. Start with 'from adk.tools.chart_generator import...'"
                    )
                    continue

                # Execute the code
                result = execute_analysis_code(
                    code,
                    local_vars={"df": df},
                )

                if result["success"]:
                    # Parse results from the execution
                    parsed_results = self._parse_results(result)
                    return {
                        "success": True,
                        "results": parsed_results,
                        "error": None,
                        "model": model or self.router.default_model,
                        "code": code,
                        "attempt": attempt + 1,
                    }
                else:
                    # Retry with error info
                    error_msg = result.get("error", "Unknown error")
                    stdout = result.get("stdout", "")
                    stderr = result.get("stderr", "")
                    code_prompt = (
                        f"Previous code failed with error:\n{error_msg}\n\n"
                        f"Stdout: {stdout[:500] if stdout else '(empty)'}\n"
                        f"Stderr: {stderr[:500] if stderr else '(empty)'}\n\n"
                        f"Here's what you tried:\n{code}\n\n"
                        f"CRITICAL FIXES NEEDED: 1) Make sure to use print(results) at the end. "
                        f"2) Do NOT use try/except that swallows errors. 3) df is always available — "
                        f"do not check if it exists."
                    )

            except SecurityError as e:
                error_msg = str(e)
                retry_code = code if 'code' in dir() else "<not yet generated>"
                code_prompt = (
                    f"Previous code failed: {error_msg}\n\n"
                    f"Here's what you tried:\n{retry_code}\n\n"
                    f"Provide corrected, valid Python code only."
                )
                # Treat syntax errors as retryable; only fail on actual security violations
                if "Syntax error" in error_msg or "Forbidden imports" in error_msg:
                    continue
                return {
                    "success": False,
                    "results": None,
                    "error": f"Security error: {e}",
                    "model": model or self.router.default_model,
                    "code": None,
                    "attempt": attempt + 1,
                }

        return {
            "success": False,
            "results": None,
            "error": f"Failed after {self.max_retries} attempts",
            "model": model or self.router.default_model,
            "code": None,
            "attempt": self.max_retries,
        }

    def _build_code_prompt(
        self,
        request: str,
        schema_prompt: str,
        tool_context: str,
    ) -> str:
        """Build the prompt for code generation.

        Args:
            request: User's analysis request.
            schema_prompt: Schema description string.
            tool_context: Tool registry context string.

        Returns:
            Formatted prompt string.
        """
        return f"""Analyze this data. Request: "{request}"

{schema_prompt}
{tool_context}

Write Python code using `df`. End with: print(results).
"""

    def _extract_code(self, response: str) -> str:
        """Extract Python code from the LLM response.

        Handles fenced code blocks (```python ... ```) and raw code.
        Handles truncated responses (no closing ```).

        Args:
            response: LLM response text.

        Returns:
            Extracted Python code string, or empty string if none found.
        """
        import re

        # Try fenced code blocks first
        fence_match = re.search(r"```(?:python|py)?\s*\n", response)
        if fence_match:
            # Extract everything after the opening fence
            start = fence_match.end()
            remaining = response[start:]

            # Check if there's a closing fence
            close_match = re.search(r"\s*```", remaining)
            if close_match:
                code = remaining[:close_match.start()].strip()
            else:
                # Truncated — take everything and strip artifacts
                code = remaining.strip()

            return self._sanitize_truncated(code)

        # If no fences, return the whole response as code
        if response.strip() and any(
            line.startswith(("import ", "from ", "df.", "results ="))
            for line in response.strip().split("\n")
        ):
            return self._sanitize_truncated(response.strip())

        return ""

    def _sanitize_truncated(self, code: str) -> str:
        """Remove trailing truncation artifacts from code.

        Handles cases where the LLM response was cut off mid-line.
        """
        import re

        lines = code.split("\n")
        clean_lines = []

        for line in lines:
            stripped = line.rstrip()
            # Skip lines that are clearly truncation artifacts
            if re.search(r"```", stripped):
                continue
            clean_lines.append(line)

        # Strip trailing lines that are truncated
        while clean_lines:
            last = clean_lines[-1].rstrip()
            if not last:
                clean_lines.pop()
            elif re.search(r"[a-zA-Z0-9_]\s*[^a-zA-Z0-9_\]\),;}\.]*$", last):
                # Ends with identifier or partial expression — likely truncated
                clean_lines.pop()
            else:
                break

        return "\n".join(l for l in clean_lines if l.strip()).strip()

    def _parse_results(self, execution_result: dict[str, Any]) -> dict[str, Any]:
        """Parse the results from code execution.

        Args:
            execution_result: Dict from code_executor with 'stdout', 'stderr', 'result'.

        Returns:
            Standardized results dict.
        """
        import re

        stdout = execution_result.get("stdout", "")
        stderr = execution_result.get("stderr", "")

        if not stdout:
            return {
                "summary": "Analysis completed but produced no output.",
                "findings": [],
                "charts": [],
                "details": stderr if stderr else "",
            }

        # Try to find a results dict in stdout
        # Pattern 1: `results = {...}` (assignment)
        match = re.search(r'results\s*=\s*(\{.*\})', stdout, re.DOTALL)
        if match:
            try:
                import ast
                return ast.literal_eval(match.group(1))
            except (ValueError, SyntaxError):
                pass

        # Pattern 2: Bare dict repr from print(results) — find outermost {...}
        # This handles output like "{'summary': '...', 'findings': [...]}"
        match = re.search(r'\{[^{}]*"summary"[^{}]*\}', stdout, re.DOTALL)
        if match:
            try:
                import ast
                return ast.literal_eval(match.group(0))
            except (ValueError, SyntaxError):
                pass

        # Pattern 3: Try to find any dict with 'summary' key
        # Handle nested dicts by finding the outermost balanced braces
        start = stdout.find('{')
        end = stdout.rfind('}')
        if start != -1 and end != -1 and end > start:
            try:
                import ast
                dict_str = stdout[start:end+1]
                result = ast.literal_eval(dict_str)
                if isinstance(result, dict) and "summary" in result:
                    return result
            except (ValueError, SyntaxError):
                pass

        # Fallback: return stdout as text
        return {
            "summary": stdout[:500] or "Analysis completed. See details below.",
            "findings": [],
            "charts": [],
            "details": stdout + ("\n\nErrors:\n" + stderr if stderr else ""),
        }
