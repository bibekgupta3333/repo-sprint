"""
Agent state schema for the deep LLM agent architecture.
Defines OrchestratorState using Pydantic V2 for type-safe state management across LangGraph.
"""

from typing import Annotated, Any, Optional, Literal
from datetime import datetime
import operator
from pydantic import BaseModel, Field


# ============================================================================
# Shared Data Models
# ============================================================================

class GitHubIssue(BaseModel):
    """Minimal GitHub issue representation for state."""
    number: int
    title: str
    body: Optional[str] = None
    state: str  # 'open' | 'closed'
    labels: list[str] = Field(default_factory=list)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class SprintMetrics(BaseModel):
    """Sprint-level aggregated metrics."""
    sprint_id: str
    completion_percentage: float = 0.0
    velocity_gap: float = 0.0  # Required rate - actual rate
    daily_closure_rate: float = 0.0
    required_daily_rate: float = 0.0
    pr_merge_rate: float = 0.0
    build_success_rate: float = 0.0
    team_sentiment: float = 0.0
    stalled_issues_count: int = 0
    code_churn: float = 0.0
    regression_risk: float = 0.0


class RiskItem(BaseModel):
    """Identified risk with severity and rationale."""
    risk_type: str  # 'velocity_gap', 'stalled_blockers', 'team_burnout', etc.
    severity: float  # 0-1
    description: str
    affected_issues: list[int] = Field(default_factory=list)
    evidence: Optional[str] = None


class Recommendation(BaseModel):
    """Actionable recommendation with supporting context."""
    title: str
    description: str
    priority: str  # 'high' | 'medium' | 'low'
    expected_impact: str  # What will improve (e.g., "velocity", "team_morale")
    action: str
    evidence_source: Optional[str] = None  # e.g., "Similar sprint #42 reduced blocker time by 2 days"


class RAGContext(BaseModel):
    """Retrieved context from vector store for RAG."""
    similar_sprints: list[dict] = Field(default_factory=list)  # [{sprint_id, similarity_score, metrics}]
    intervention_history: list[dict] = Field(default_factory=list)  # [{recommendation, outcome}]
    evidence_citations: list[str] = Field(default_factory=list)  # GitHub links, issue numbers


class SprintAnalysisResult(BaseModel):
    """Complete sprint analysis from LLM Reasoner."""
    completion_probability: float  # 0-100
    predicted_completion_date: Optional[str] = None
    health_status: str  # 'on_track' | 'at_risk' | 'critical'
    confidence_score: float  # 0-1
    reasoning: Optional[str] = None


# ============================================================================
# Main Orchestrator State (for LangGraph)
# ============================================================================

class OrchestratorState(BaseModel):
    """
    Unified state for the multi-agent orchestration.
    Annotations support reducer functions for streaming/accumulation.
    """

    # ========== Input & Context ==========
    repository_url: str = ""
    repositories: list[str] = Field(default_factory=list)  # GitHub repo URLs
    sprint_id: Optional[str] = None
    milestone_data: dict = Field(default_factory=dict)  # Raw milestone data from GitHub
    eval_mode: Literal["strict", "resilient"] = "resilient"

    # ========== Data Collection (from Data Collector Agent) ==========
    github_issues: list[GitHubIssue] = Field(default_factory=list)
    github_prs: list[dict] = Field(default_factory=list)
    commits: list[dict] = Field(default_factory=list)
    comments: list[dict] = Field(default_factory=list)

    # ========== Features (from Feature Engineering Agent) ==========
    features: dict[str, Any] = Field(default_factory=dict)  # {temporal, code, sentiment, graph, cicd}
    sprint_metrics: Optional[SprintMetrics] = None
    feature_vector: list[float] = Field(default_factory=list)  # Flattened feature vector

    # ========== RAG Context (from Embedding Agent) ==========
    rag_context: Optional[RAGContext] = None
    similar_sprint_ids: list[str] = Field(default_factory=list)
    vector_embeddings: dict[str, list[float]] = Field(default_factory=dict)

    # ========== Dependency Graph (from Dependency Graph Agent) ==========
    dependency_graph: dict[str, Any] = Field(default_factory=dict)  # nodes, edges, chains, propagation

    # ========== Synthetic Data (from Synthetic Data Generator Agent) ==========
    processed_data: list[dict] = Field(default_factory=list)  # Real sprint data for validation
    synthetic_sprints: list[dict] = Field(default_factory=list)  # Generated synthetic scenarios
    synthetic_validation: dict[str, Any] = Field(default_factory=dict)  # KS test results + realism score
    synthetic_embedded_count: int = 0  # Docs embedded in vector store

    # ========== LoRA Training (from LoRA Training Orchestrator Agent) ==========
    lora_performance_tracker: dict[str, Any] = Field(default_factory=dict)  # project_id → performance metrics
    lora_adapters_active: list[dict] = Field(default_factory=list)  # Active LoRA adapters with configs
    lora_training_triggered: bool = False  # Did training execute in this run
    lora_metrics: dict[str, Any] = Field(default_factory=dict)  # Training results (F1, loss, time)

    # ========== LLM Analysis (from LLM Reasoner Agent) ==========
    sprint_analysis: Optional[dict] = None  # Parsed LLM result dict
    analysis_source: Optional[str] = None  # llm | fallback | rule | none | error
    llm_reasoning_explanation: Optional[str] = None

    # ========== Risk Assessment (from Risk Assessor Agent) ==========
    identified_risks: Annotated[list[RiskItem], operator.add] = Field(default_factory=list)
    risk_source: Optional[str] = None  # llm | fallback | rule | none | error
    risk_summary: Optional[str] = None

    # ========== Recommendations (from Recommender Agent) ==========
    recommendations: Annotated[list[Recommendation], operator.add] = Field(default_factory=list)
    recommendation_source: Optional[str] = None  # llm | fallback | rule | none | error

    # ========== Explanations (from Explainer Agent) ==========
    narrative_explanation: Optional[str] = None
    evidence_citations: list[str] = Field(default_factory=list)

    # ========== Meta / Execution ==========
    current_agent: str = "master_orchestrator"
    execution_timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    execution_logs: Annotated[list[str], operator.add] = Field(default_factory=list)
    errors: Annotated[list[str], operator.add] = Field(default_factory=list)
    run_metrics: dict[str, Any] = Field(default_factory=dict)
    run_metrics_artifact: Optional[str] = None
    cache_hits: int = 0
    cache_misses: int = 0

    # ========== Workflow Control ==========
    workflow_complete: bool = False
    should_continue: bool = True
    next_agent: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


# ============================================================================
# Agent-Specific Sub-States (Optional, for specialized agents)
# ============================================================================

class DataCollectorState(BaseModel):
    """State for Data Collector sub-workflows."""
    api_rate_limit_remaining: int = 5000
    cache_key: str = ""
    batch_size: int = 100
    pagination_token: Optional[str] = None


class FeatureEngineerState(BaseModel):
    """State for Feature Engineering sub-workflows."""
    raw_features: dict[str, Any] = Field(default_factory=dict)
    normalized_features: dict[str, Any] = Field(default_factory=dict)
    feature_importance_scores: dict[str, float] = Field(default_factory=dict)
    anomaly_flags: list[str] = Field(default_factory=list)


class LLMReasonerState(BaseModel):
    """State for LLM Reasoner sub-workflows."""
    model_name: str = "ollama:qwen3:0.6b"
    temperature: float = 0.7
    max_tokens: int = 1000
    prompt_template: str = ""
    llm_response_raw: Optional[str] = None
    parsing_errors: list[str] = Field(default_factory=list)
