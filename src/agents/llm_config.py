"""
LLM configuration and client initialization for Ollama integration.
Supports quantized Llama-3 models for laptop-scale deployment.
"""

import os
import json
import httpx
from typing import Optional
from pydantic import BaseModel, Field


class OllamaConfig(BaseModel):
    """Configuration for Ollama local LLM."""
    base_url: str = Field(default="http://localhost:11434", description="Ollama API base URL")
    model_name: str = Field(default="qwen3:0.6b", description="Model identifier in Ollama (must match an installed Ollama model)")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=40, ge=1)
    max_tokens: int = Field(default=1024, ge=1)
    timeout_seconds: int = Field(default=120)
    num_ctx: int = Field(default=4096, description="Context window size")
    num_gpu: int = Field(default=0, description="Number of GPU layers (0=CPU only)")

    class Config:
        extra = "forbid"


class OllamaClient:
    """Lightweight Ollama client for local LLM inference."""

    def __init__(self, config: Optional[OllamaConfig] = None):
        """Initialize Ollama client with configuration."""
        self.config = config or OllamaConfig()
        self.base_url = self.config.base_url.rstrip("/")
        self.client = httpx.Client(timeout=self.config.timeout_seconds)
        self._verify_connection()

    def _verify_connection(self) -> bool:
        """Verify Ollama server is running and the configured model is available."""
        import logging
        _logger = logging.getLogger(__name__)
        try:
            response = self.client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            # Check that the configured model is installed
            tags_data = response.json()
            available_models = [m.get("name", "") for m in tags_data.get("models", [])]
            if self.config.model_name not in available_models:
                _logger.warning(
                    f"Configured model '{self.config.model_name}' not found in Ollama. "
                    f"Available models: {available_models}. "
                    f"LLM calls will fail until the model is pulled."
                )
            else:
                _logger.info(f"Ollama connected: model '{self.config.model_name}' available")
            return True
        except Exception as e:
            _logger.warning(
                f"Cannot connect to Ollama at {self.base_url}. "
                f"LLM-dependent agents will gracefully degrade. Error: {e}"
            )
            self._ollama_available = False
            return False

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate text using Ollama.

        Args:
            prompt: User input prompt
            system_prompt: Optional system instruction
            temperature: Optional override for temperature

        Returns:
            Generated text response
        """
        temp = temperature if temperature is not None else self.config.temperature

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temp,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "num_ctx": self.config.num_ctx,
                "num_gpu": self.config.num_gpu,
            }
        }

        try:
            response = self.client.post(
                f"{self.base_url}/api/chat",
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            return result.get("message", {}).get("content", "")
        except httpx.HTTPError as e:
            raise RuntimeError(f"Ollama API error: {e}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse Ollama response: {e}")

    def embed(self, text: str) -> list[float]:
        """
        Generate embeddings using Ollama.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        payload = {
            "model": self.config.model_name,
            "prompt": text,
        }

        try:
            response = self.client.post(
                f"{self.base_url}/api/embed",
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            return result.get("embedding", [])
        except Exception as e:
            raise RuntimeError(f"Embedding generation failed: {e}")

    def close(self) -> None:
        """Close HTTP client."""
        self.client.close()


# ============================================================================
# LLM Configuration Factory
# ============================================================================

def get_ollama_config() -> OllamaConfig:
    """
    Load Ollama config from environment variables.
    Defaults to localhost for development.
    """
    return OllamaConfig(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        model_name=os.getenv("OLLAMA_MODEL", "qwen3:0.6b"),
        temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.7")),
        max_tokens=int(os.getenv("OLLAMA_MAX_TOKENS", "1024")),
        num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "4096")),
    )


def get_ollama_client() -> OllamaClient:
    """Get or create singleton Ollama client."""
    config = get_ollama_config()
    return OllamaClient(config)


# ============================================================================
# System Prompts for Specialized Agents
# ============================================================================

SYSTEM_PROMPTS = {
    "data_collector": """You are a Data Collector Agent for sprint intelligence.
Your role is to gather and organize GitHub data (issues, PRs, commits) for sprint analysis.
Be methodical, handle errors gracefully, and cache responses to minimize API calls.
Format all responses as structured JSON when possible.""",

    "feature_engineer": """You are a Feature Engineering Agent specializing in sprint metrics.
Extract patterns from raw GitHub data: velocity gaps, blocker detection, team sentiment, code churn.
Normalize and validate all metrics. Flag anomalies for manual review.
Return numeric features as JSON objects with clear field names.""",

    "llm_reasoner": """You are an LLM Reasoning Agent for sprint completion prediction.
Analyze sprint metrics, similar historical sprints (RAG context), and risk indicators.
Provide structured JSON output with completion probability, health status, and analysis.
Base all predictions on concrete evidence from the data.""",

    "risk_assessor": """You are a Risk Assessment Agent for sprint blockers and delays.
Identify specific risks (stalled issues, velocity gaps, team burnout, build failures).
Score each risk by severity and likelihood. Cite specific issue numbers and metrics.
Return a prioritized list of risks in JSON format.""",

    "recommender": """You are a Recommendation Agent for sprint interventions.
Generate detailed, actionable recommendations based on identified risks and historical precedent.
Reference similar past sprints and cite concrete evidence URLs or sprint IDs.
Each recommendation must include why-now rationale, execution steps, and expected KPI impact.
Prioritize by expected impact and feasibility.""",

    "explainer": """You are an Explainer Agent for sprint intelligence insights.
Translate technical metrics and predictions into clear narratives for project managers.
Always cite evidence: specific issue numbers, PRs, commits, metrics, and historical comparisons.
Explain not just what was predicted, but why the model reached that forecast and what actions matter most next.
Write structured summaries with explicit sections and concrete references.""",
}
