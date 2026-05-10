"""Model router — wraps LiteLLM for multi-model support.

Supports Claude, Gemini, OpenAI, DeepSeek, Kimi, and local models
(llama.cpp, vLLM) via OpenAI-compatible API.

Configuration:
  - YAML config: config/models.yaml
  - API keys: environment variables (ANTHROPIC_API_KEY, GEMINI_API_KEY, etc.)
  - Local models: LOCAL_API_URL env var overrides base_url in config
"""

import os
from pathlib import Path
from typing import Any, Optional

import yaml
import litellm


# Map of provider names to their env var names
PROVIDER_ENV_VARS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "moonshot": "KIMI_API_KEY",
}

# Default local model URL
DEFAULT_LOCAL_URL = "http://localhost:8080/v1"


class ModelRouter:
    """Routes LLM calls to the configured model via LiteLLM.

    Usage:
        router = ModelRouter()
        response = router.complete(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "Hello"}]
        )
    """

    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = str(Path(__file__).parent.parent.parent / "config" / "models.yaml")
        self.config_path = config_path
        self.config = self._load_config()
        self.default_model = self.config.get("default_model", "claude-sonnet-4-20250514")
        self.model_specs = self.config.get("models", {})

    def _load_config(self) -> dict:
        """Load model configuration from YAML file."""
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def get_model_spec(self, model_name: str) -> dict:
        """Get the model specification for a given model name."""
        if model_name not in self.model_specs:
            available = list(self.model_specs.keys())
            raise ValueError(
                f"Unknown model '{model_name}'. Available: {available}"
            )
        return self.model_specs[model_name]

    def list_models(self) -> list[str]:
        """Return list of available model names."""
        return list(self.model_specs.keys())

    def get_provider(self, model_name: str) -> str:
        """Return the provider name for a model."""
        return self.model_specs[model_name]["provider"]

    def is_local_model(self, model_name: str) -> bool:
        """Check if a model uses a local server (has base_url configured)."""
        return "base_url" in self.model_specs.get(model_name, {})

    def complete(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        """Make a completion call via LiteLLM.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            model: Model name from config. Defaults to default_model.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
            api_key: Runtime API key override.
            api_base: Runtime base URL override (for local models).
            **kwargs: Additional kwargs passed to litellm.completion.

        Returns:
            Dict with 'content', 'model', and 'usage' keys.
        """
        model_name = model or self.default_model
        spec = self.get_model_spec(model_name)

        # Build the model string for LiteLLM
        provider = spec["provider"]
        model_id = spec["model_id"]

        # Override base_url for local models if LOCAL_API_URL is set
        if provider == "openai" and "base_url" in spec:
            local_url = os.environ.get("LOCAL_API_URL", spec.get("base_url", DEFAULT_LOCAL_URL))
            # Check if this is configured as a local model
            if local_url in model_id or "local" in model_name:
                spec = {**spec, "base_url": local_url}

        # Build the LiteLLM model string
        if provider == "openai":
            litellm_model = f"openai/{model_id}"
        elif provider == "anthropic":
            litellm_model = f"anthropic/{model_id}"
        elif provider == "google":
            litellm_model = f"google/{model_id}"
        elif provider == "deepseek":
            litellm_model = f"deepseek/{model_id}"
        elif provider == "moonshot":
            litellm_model = f"moonshot/{model_id}"
        else:
            litellm_model = f"{provider}/{model_id}"

        # LiteLLM auto-discovers API keys from env vars based on provider
        # No need to manually set them — LiteLLM handles this

        response = litellm.completion(
            model=litellm_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=api_base or spec.get("base_url"),
            **kwargs,
        )

        # Normalize response format
        choice = response.choices[0]
        return {
            "content": choice.message.content,
            "model": model_name,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        }

    def chat(
        self,
        system_prompt: str,
        user_message: str,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Convenience method for simple chat.

        Args:
            system_prompt: System prompt.
            user_message: User message.
            model: Model name.
            temperature: Sampling temperature.
            max_tokens: Max response tokens.
            api_key: Runtime API key override.
            api_base: Runtime base URL override.
            **kwargs: Additional kwargs.

        Returns:
            Response content string.
        """
        result = self.complete(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            api_base=api_base,
            **kwargs,
        )
        return result["content"]
