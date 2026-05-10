"""Orchestrator agent — root agent that manages session state and routes requests.

Receives user requests, manages conversation history, and delegates to the
Analyst Agent for analysis tasks. Coordinates report and slide generation.
"""

from typing import Any

from adk.models.router import ModelRouter


# System prompt for the orchestrator
ORCHESTRATOR_SYSTEM_PROMPT = """You are a Data Analyst Agent orchestrator. You manage a conversation with a user about their data analysis needs.

Your role:
1. Understand the user's request
2. Determine if it requires data analysis
3. Delegate to the Analyst Agent for analysis
4. Coordinate generation of reports and slides

When the user asks for analysis:
- If data hasn't been loaded yet, ask them to upload a dataset first
- Pass the user's request to the Analyst Agent
- Present the Analyst Agent's results back to the user
- Offer to generate a report (.md) or slides (.pptx) after analysis

When the user asks for a report or slides:
- Check if analysis results are available in the session
- If yes, call the appropriate tool (report_writer or slide_writer)
- If no, remind the user to run an analysis first

Be concise, professional, and helpful. Focus on delivering actionable insights."""


class Orchestrator:
    """Root agent that manages session state and routes requests.

    Usage:
        router = ModelRouter()
        orchestrator = Orchestrator(router)
        result = orchestrator.process("Cluster customers by payment behavior")
    """

    def __init__(self, router: ModelRouter):
        self.router = router
        self.session_id: str = ""
        self.conversation_history: list[dict[str, str]] = []
        self.analysis_results: dict[str, Any] = {}
        self.data_loaded = False
        self.schema = None

    def reset(self):
        """Reset the session state."""
        self.conversation_history = []
        self.analysis_results = {}
        self.data_loaded = False
        self.schema = None

    def add_message(self, role: str, content: str):
        """Add a message to the conversation history.

        Args:
            role: 'user' or 'assistant'.
            content: Message content.
        """
        self.conversation_history.append({"role": role, "content": content})

    def process(self, user_message: str, model: str | None = None) -> dict[str, Any]:
        """Process a user message and return the response.

        Args:
            user_message: The user's request.
            model: Model name (optional, uses default if None).

        Returns:
            Dict with 'response', 'analysis_results', 'model'.
        """
        # Add user message to history
        self.add_message("user", user_message)

        # Build the prompt
        prompt = self._build_prompt(user_message)

        # Call the LLM
        response = self.router.complete(
            messages=[
                {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
                *self.conversation_history[-10:],  # Keep last 10 messages
            ],
            model=model,
            temperature=0.3,
            max_tokens=2048,
        )

        # Add assistant response to history
        assistant_response = response["content"]
        self.add_message("assistant", assistant_response)

        return {
            "response": assistant_response,
            "model": response.get("model", "unknown"),
            "usage": response.get("usage", {}),
        }

    def _build_prompt(self, user_message: str) -> str:
        """Build the prompt with context for the orchestrator.

        Args:
            user_message: The user's latest message.

        Returns:
            Formatted prompt string.
        """
        parts = [f"User: {user_message}"]

        if self.data_loaded and self.schema:
            from adk.tools.schema_discovery import schema_to_prompt
            parts.append("\n---\nCurrent dataset schema:\n" + schema_to_prompt(self.schema))

        if self.analysis_results:
            parts.append("\n---\nPrevious analysis results are available.")

        return "\n".join(parts)
