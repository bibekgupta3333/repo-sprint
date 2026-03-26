"""
Master Orchestrator Agent using LangGraph.
Coordinates workflow execution across all specialized agents.
Implements LangChain DeepAgents harness patterns.
"""

from typing import Optional
from langgraph.graph import StateGraph, END
from src.agents.state import OrchestratorState
from src.agents.llm_config import OllamaClient, SYSTEM_PROMPTS, get_ollama_client
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
from src.agents.metrics_logger import (
    deduplicate_state_collections,
    build_run_metrics,
    persist_run_metrics,
)
import json
import logging
from datetime import datetime
import time

logger = logging.getLogger(__name__)


# ============================================================================
# Master Orchestrator Configuration
# ============================================================================

class OrchestratorConfig:
    """Configuration for the master orchestrator."""

    # Workflow stages
    STAGES = [
        "data_collection",
        "dependency_analysis",
        "feature_engineering",
        "embedding_and_rag",
        "llm_reasoning",
        "sprint_analysis",
        "risk_assessment",
        "recommendations",
        "explanations",
    ]

    # Agent execution order (DAG-based)
    AGENT_SEQUENCE = {
        "data_collector": "→ dependency_graph",
        "dependency_graph": "→ feature_engineer",
        "feature_engineer": "→ synthetic_data_generator",
        "synthetic_data_generator": "→ embedding_agent",
        "embedding_agent": "→ llm_reasoner",
        "llm_reasoner": "→ sprint_analyzer",
        "sprint_analyzer": "→ lora_training_orchestrator",
        "lora_training_orchestrator": "→ risk_assessor",
        "risk_assessor": "→ recommender",
        "recommender": "→ explainer",
        "explainer": "→ END",
    }


# ============================================================================
# Orchestrator Implementation
# ============================================================================

class MasterOrchestrator:
    """
    Main orchestration engine for multi-agent sprint intelligence workflow.
    Uses LangGraph StateGraph with DeepAgents patterns.
    """

    def __init__(
        self,
        ollama_client: Optional[OllamaClient] = None,
        tool_registry: Optional[ToolRegistry] = None,
    ):
        """Initialize orchestrator with clients and tools."""
        self.ollama_client = ollama_client or get_ollama_client()
        self.tool_registry = (
            tool_registry or ToolRegistry(self.ollama_client)
        )
        self.config = OrchestratorConfig()

        # Initialize agent implementations
        self.data_collector = DataCollectorAgent(self.tool_registry)
        self.dependency_graph = DependencyGraphAgent(self.tool_registry)
        self.feature_engineer = FeatureEngineerAgent(self.tool_registry)
        self.synthetic_data_generator = SyntheticDataGeneratorAgent(self.tool_registry)
        self.embedding_agent = EmbeddingAgent(self.tool_registry)  # No Ollama needed for embeddings
        self.llm_reasoner = LLMReasonerAgent(self.tool_registry, self.ollama_client)
        self.lora_training_orchestrator = LoRATrainingOrchestratorAgent(self.tool_registry)
        self.risk_assessor = RiskAssessorAgent(self.tool_registry, self.ollama_client)
        self.recommender = RecommenderAgent(self.tool_registry, self.ollama_client)
        self.explainer = ExplainerAgent()

        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """
        Construct LangGraph StateGraph with all agents as nodes.
        Graph topology corresponds to AGENT_SEQUENCE DAG.
        """
        graph = StateGraph(OrchestratorState)

        # Add agent nodes
        graph.add_node("data_collector", self.data_collector_node)
        graph.add_node("dependency_graph", self.dependency_graph_node)
        graph.add_node("feature_engineer", self.feature_engineer_node)
        graph.add_node("synthetic_data_generator", self.synthetic_data_generator_node)
        graph.add_node("embedding_agent", self.embedding_agent_node)
        graph.add_node("llm_reasoner", self.llm_reasoner_node)
        graph.add_node("sprint_analyzer", self.sprint_analyzer_node)
        graph.add_node("lora_training_orchestrator", self.lora_training_orchestrator_node)
        graph.add_node("risk_assessor", self.risk_assessor_node)
        graph.add_node("recommender", self.recommender_node)
        graph.add_node("explainer", self.explainer_node)

        # Add router nodes for conditional logic
        graph.add_node("router", self.router_node)

        # Set entry point and edges
        graph.set_entry_point("router")

        # Router logic: branch based on input
        graph.add_edge("router", "data_collector")

        # Linear pipeline (for now—can add parallel branches later)
        graph.add_edge("data_collector", "dependency_graph")
        graph.add_edge("dependency_graph", "feature_engineer")
        graph.add_edge("feature_engineer", "synthetic_data_generator")
        graph.add_edge("synthetic_data_generator", "embedding_agent")
        graph.add_edge("embedding_agent", "llm_reasoner")
        graph.add_edge("llm_reasoner", "sprint_analyzer")
        graph.add_edge("sprint_analyzer", "lora_training_orchestrator")
        graph.add_edge("lora_training_orchestrator", "risk_assessor")
        graph.add_edge("risk_assessor", "recommender")
        graph.add_edge("recommender", "explainer")
        graph.add_edge("explainer", END)

        return graph.compile()

    # ========================================================================
    # Router Node
    # ========================================================================

    def router_node(self, state: OrchestratorState) -> OrchestratorState:
        """Entry point router. Validates input and prepares state."""
        logger.info("=== Master Orchestrator Started ===")
        logger.info(f"Input repositories: {state.repositories}")

        # Validation
        if not state.repositories:
            raise ValueError("No repositories provided in input state")

        state.current_agent = "router"
        state.execution_logs.append(
            f"[router] Initialized orchestration at {datetime.now().isoformat()}"
        )
        state.execution_logs.append(f"[router] Processing {len(state.repositories)} repositories")

        return state

    # ========================================================================
    # Agent Nodes (DeepAgents Pattern)
    # ========================================================================

    def data_collector_node(self, state: OrchestratorState) -> OrchestratorState:
        """Data Collector Agent node."""
        logger.info("=== Data Collector Agent Starting ===")
        state.current_agent = "data_collector"
        return self.data_collector.execute(state)

    def dependency_graph_node(self, state: OrchestratorState) -> OrchestratorState:
        """Dependency Graph Agent node (cross-repo analysis)."""
        logger.info("=== Dependency Graph Agent Starting ===")
        state.current_agent = "dependency_graph"
        return self.dependency_graph.execute(state)

    def feature_engineer_node(self, state: OrchestratorState) -> OrchestratorState:
        """Feature Engineering Agent node."""
        logger.info("=== Feature Engineering Agent Starting ===")
        state.current_agent = "feature_engineer"
        return self.feature_engineer.execute(state)

    def synthetic_data_generator_node(self, state: OrchestratorState) -> OrchestratorState:
        """Synthetic Data Generator Agent node (cold-start scenarios)."""
        logger.info("=== Synthetic Data Generator Agent Starting ===")
        state.current_agent = "synthetic_data_generator"
        return self.synthetic_data_generator.execute(state)

    def embedding_agent_node(self, state: OrchestratorState) -> OrchestratorState:
        """Embedding & RAG Agent node."""
        logger.info("=== Embedding & RAG Agent Starting ===")
        state.current_agent = "embedding_agent"
        return self.embedding_agent.execute(state)

    def llm_reasoner_node(self, state: OrchestratorState) -> OrchestratorState:
        """LLM Reasoning Agent node."""
        logger.info("=== LLM Reasoner Agent Starting ===")
        state.current_agent = "llm_reasoner"
        return self.llm_reasoner.execute(state)

    def sprint_analyzer_node(self, state: OrchestratorState) -> OrchestratorState:
        """Sprint Analyzer Agent node (metrics aggregation)."""
        logger.info("=== Sprint Analyzer Agent Starting ===")
        state.current_agent = "sprint_analyzer"

        try:
            # Compute sprint health score from features and LLM analysis
            if state.sprint_analysis:
                completion = state.sprint_analysis.get("completion_probability", 50) / 100
                confidence = state.sprint_analysis.get("confidence_score", 0.5)
                health_score = (completion + confidence) / 2 * 100
                logger.info(f"  ✓ Sprint health score: {health_score:.1f}")

            state.execution_logs.append(f"[sprint_analyzer] Health metrics computed")

        except Exception as e:
            state.errors.append(f"Sprint analysis failed: {str(e)}")
            logger.error(f"Sprint analyzer error: {e}")

        return state

    def lora_training_orchestrator_node(self, state: OrchestratorState) -> OrchestratorState:
        """LoRA Training Orchestrator Agent node (continuous learning)."""
        logger.info("=== LoRA Training Orchestrator Agent Starting ===")
        state.current_agent = "lora_training_orchestrator"
        return self.lora_training_orchestrator.execute(state)

    def risk_assessor_node(self, state: OrchestratorState) -> OrchestratorState:
        """Risk Assessment Agent node."""
        logger.info("=== Risk Assessor Agent Starting ===")
        state.current_agent = "risk_assessor"
        return self.risk_assessor.execute(state)

    def recommender_node(self, state: OrchestratorState) -> OrchestratorState:
        """Recommender Agent node."""
        logger.info("=== Recommender Agent Starting ===")
        state.current_agent = "recommender"
        return self.recommender.execute(state)

    def explainer_node(self, state: OrchestratorState) -> OrchestratorState:
        """Explainer Agent node."""
        logger.info("=== Explainer Agent Starting ===")
        state.current_agent = "explainer"
        state = self.explainer.execute(state)
        state.workflow_complete = True
        return state

    # ========================================================================
    # Workflow Execution
    # ========================================================================

    def invoke(self, input_state: OrchestratorState) -> OrchestratorState:
        """
        Execute the full agent workflow.

        Args:
            input_state: Initial OrchestratorState with repositories

        Returns:
            Final OrchestratorState with all agent outputs
        """
        logger.info("Starting orchestrator invocation...")
        started_at = time.perf_counter()

        final_state = self.graph.invoke(
            input_state.dict(),
            config={"recursion_limit": 50},
        )

        # Convert back to OrchestratorState
        state_obj = OrchestratorState(**final_state)
        finished_at = time.perf_counter()

        # Normalize reducer-accumulated fields to avoid inflated run metrics.
        state_obj = deduplicate_state_collections(state_obj)

        run_metrics = build_run_metrics(state_obj, started_at, finished_at)
        artifact_path = persist_run_metrics(run_metrics)
        state_obj.run_metrics = run_metrics
        state_obj.run_metrics_artifact = artifact_path

        return state_obj

    def stream(self, input_state: OrchestratorState):
        """
        Stream results from the agent workflow.
        Useful for monitoring and logging.
        """
        logger.info("Starting orchestrator stream...")

        for output in self.graph.stream(
            input_state.dict(),
            config={"recursion_limit": 50},
        ):
            yield output


# ============================================================================
# Factory Functions
# ============================================================================

def create_orchestrator(
    ollama_client: Optional[OllamaClient] = None,
) -> MasterOrchestrator:
    """Create and initialize master orchestrator."""
    return MasterOrchestrator(ollama_client=ollama_client)
