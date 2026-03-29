"""
Deep LLM Agent Architecture for Sprint Intelligence.
Multi-agent system using LangChain's DeepAgents harness and LangGraph orchestration.
"""

from src.agents.orchestrator import MasterOrchestrator, create_orchestrator
from src.agents.state import OrchestratorState
from src.agents.llm_config import OllamaClient, get_ollama_client, get_ollama_config
from src.agents.tools import ToolRegistry
from src.agents.agents import (
    DataCollectorAgent,
    FeatureEngineerAgent,
    EmbeddingAgent,
    LLMReasonerAgent,
    RiskAssessorAgent,
    RecommenderAgent,
    ExplainerAgent,
)
from src.agents.dependency_graph_agent import DependencyGraphAgent
from src.agents.synthetic_data_generator_agent import SyntheticDataGeneratorAgent
from src.agents.lora_training_orchestrator_agent import LoRATrainingOrchestratorAgent

__version__ = "0.1.0"
__all__ = [
    "MasterOrchestrator",
    "create_orchestrator",
    "OrchestratorState",
    "OllamaClient",
    "get_ollama_client",
    "get_ollama_config",
    "ToolRegistry",
    "DataCollectorAgent",
    "FeatureEngineerAgent",
    "EmbeddingAgent",
    "LLMReasonerAgent",
    "RiskAssessorAgent",
    "RecommenderAgent",
    "ExplainerAgent",
    "DependencyGraphAgent",
    "SyntheticDataGeneratorAgent",
    "LoRATrainingOrchestratorAgent",
]
