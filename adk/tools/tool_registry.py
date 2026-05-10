"""Tool registry — scans, indexes, and searches for saved analysis .py tools.

Each tool file has YAML frontmatter with a description, enabling semantic
matching when the agent encounters similar analysis requests.
"""

import re
from pathlib import Path
from typing import Any

# Default tool directories to scan
DEFAULT_TOOL_DIRS = [
    Path.home() / ".adk-tools",
    Path(__file__).parent.parent.parent / "examples",
]


class ToolRegistry:
    """Registry of analysis tools with semantic search.

    Scans directories for .py files with YAML frontmatter and indexes
    their descriptions for matching against user requests.
    """

    def __init__(self, tool_dirs: list[Path] | None = None):
        self.tool_dirs = tool_dirs or DEFAULT_TOOL_DIRS
        self.tools: list[dict[str, Any]] = []
        self._scan()

    def _parse_frontmatter(self, source: str) -> tuple[dict[str, Any], str]:
        """Parse YAML frontmatter from Python source.

        Args:
            source: Python source code with optional YAML frontmatter.

        Returns:
            Tuple of (metadata_dict, code_without_frontmatter).
        """
        match = re.match(r"^---\n(.*?)\n---\n(.*)", source, re.DOTALL)
        if not match:
            return {}, source

        import yaml
        metadata = yaml.safe_load(match.group(1)) or {}
        code = match.group(2)
        return metadata, code

    def _scan(self):
        """Scan all tool directories for .py files."""
        self.tools = []
        for tool_dir in self.tool_dirs:
            if not tool_dir.exists():
                continue
            for py_file in tool_dir.glob("*.py"):
                try:
                    source = py_file.read_text()
                    metadata, code = self._parse_frontmatter(source)
                    if metadata:
                        self.tools.append({
                            "name": py_file.stem,
                            "path": str(py_file),
                            "description": metadata.get("description", ""),
                            "use_case": metadata.get("use_case", ""),
                            "parameters": metadata.get("parameters", []),
                            "code": code,
                            "tags": metadata.get("tags", []),
                        })
                except Exception:
                    # Skip files that can't be parsed
                    continue

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Search tools by keyword matching against descriptions and tags.

        Args:
            query: User's analysis request or keywords.
            top_k: Maximum number of results to return.

        Returns:
            List of matching tool dicts, sorted by relevance.
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())

        scored = []
        for tool in self.tools:
            score = 0
            # Description match
            desc_lower = tool["description"].lower()
            for word in query_words:
                if word in desc_lower:
                    score += 2
                # Partial match for longer words
                if len(word) > 4:
                    for desc_word in desc_lower.split():
                        if word in desc_word or desc_word in word:
                            score += 1

            # Tag match
            for tag in tool.get("tags", []):
                if tag.lower() in query_lower:
                    score += 3

            # Use case match
            use_case = tool.get("use_case", "").lower()
            for word in query_words:
                if word in use_case:
                    score += 1

            if score > 0:
                scored.append((score, tool))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        return [tool for _, tool in scored[:top_k]]

    def get_all(self) -> list[dict[str, Any]]:
        """Return all registered tools."""
        return self.tools

    def get_tool_by_name(self, name: str) -> dict[str, Any] | None:
        """Get a tool by its name (filename stem)."""
        for tool in self.tools:
            if tool["name"] == name:
                return tool
        return None

    def list_tools(self) -> str:
        """Return a formatted list of available tools for prompt injection."""
        if not self.tools:
            return "No analysis tools found."

        lines = ["## Available Analysis Tools"]
        for tool in self.tools:
            lines.append(f"- **{tool['name']}**: {tool['description']}")
            if tool.get("parameters"):
                lines.append(f"  - Parameters: {', '.join(tool['parameters'])}")
            if tool.get("use_case"):
                lines.append(f"  - Use case: {tool['use_case']}")
            lines.append("")

        return "\n".join(lines)
