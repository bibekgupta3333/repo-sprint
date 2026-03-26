"""
Detailed agent implementations with full integration to existing pipeline.
Each agent is a production-ready node in the LangGraph orchestration.
"""

import logging
import json
from typing import Optional, Any
from src.agents.state import OrchestratorState, RiskItem, Recommendation, SprintMetrics
from src.agents.tools import ToolRegistry
from src.agents.llm_config import OllamaClient, SYSTEM_PROMPTS

logger = logging.getLogger(__name__)


# ============================================================================
# Agent Implementations
# ============================================================================

class DataCollectorAgent:
    """
    Fetches GitHub data with caching and error recovery.
    Integrates with scripts/_core/ LocalScraper.
    """

    def __init__(self, tool_registry: ToolRegistry):
        self.github_tool = tool_registry.github_tool
        self.logger = logging.getLogger(self.__class__.__name__)

    def execute(self, state: OrchestratorState) -> OrchestratorState:
        """Execute data collection for all repositories."""
        self.logger.info(f"Collecting data for {len(state.repositories)} repositories...")

        for repo_url in state.repositories:
            try:
                # Fetch issues
                issues_result = self.github_tool.fetch_issues(repo_url)
                if issues_result["status"] == "success":
                    state.github_issues.extend(issues_result.get("issues", []))
                    self.logger.info(f"  ✓ Issues: {issues_result['total_count']}")
                else:
                    self.logger.warning(f"  ✗ Issues fetch failed: {issues_result.get('error')}")

                # Fetch PRs
                prs_result = self.github_tool.fetch_pull_requests(repo_url)
                if prs_result["status"] == "success":
                    state.github_prs.extend(prs_result.get("pull_requests", []))
                    self.logger.info(f"  ✓ PRs: {prs_result['total_count']}")
                else:
                    self.logger.warning(f"  ✗ PRs fetch failed: {prs_result.get('error')}")

                # Fetch commits
                commits_result = self.github_tool.fetch_commits(repo_url)
                if commits_result["status"] == "success":
                    state.commits.extend(commits_result.get("commits", []))
                    self.logger.info(f"  ✓ Commits: {commits_result['total_count']}")
                else:
                    self.logger.warning(f"  ✗ Commits fetch failed: {commits_result.get('error')}")

                # Track cache hits
                if issues_result.get("cache_hit"):
                    state.cache_hits += 1
                else:
                    state.cache_misses += 1

            except Exception as e:
                state.errors.append(f"Data collection for {repo_url} failed: {str(e)}")
                self.logger.error(f"Error collecting {repo_url}: {e}")

        # Summary
        state.execution_logs.append(
            f"[data_collector] Collected {len(state.github_issues)} issues, "
            f"{len(state.github_prs)} PRs, {len(state.commits)} commits"
        )

        return state


class FeatureEngineerAgent:
    """
    Computes sprint metrics from raw GitHub data.
    Integrates with src/data/FeatureExtractor.
    """

    def __init__(self, tool_registry: ToolRegistry):
        self.feature_tool = tool_registry.feature_tool
        self.logger = logging.getLogger(self.__class__.__name__)

    def execute(self, state: OrchestratorState) -> OrchestratorState:
        """Execute feature engineering."""
        try:
            self.logger.info("Engineering features from raw data...")

            # Aggregate GitHub data
            github_data = {
                "issues": [
                    i.dict() if hasattr(i, "dict") else i
                    for i in state.github_issues
                ],
                "prs": state.github_prs,
                "commits": state.commits,
                "sprint_start": state.milestone_data.get("created_at", ""),
                "sprint_end": state.milestone_data.get("due_on", ""),
            }

            # Extract features using actual pipeline
            features = self.feature_tool.extract_all_features(github_data)
            state.features = features

            # Flatten features for ML models
            state.feature_vector = self._flatten_features(features)

            self.logger.info(f"  ✓ Computed {len(state.feature_vector)} features")
            state.execution_logs.append(
                f"[feature_engineer] Extracted {len(state.feature_vector)} features"
            )

        except Exception as e:
            state.errors.append(f"Feature engineering failed: {str(e)}")
            self.logger.error(f"Feature engineering error: {e}")

        return state

    def _flatten_features(self, features: dict) -> list[float]:
        """Flatten nested feature dict into vector."""
        vector = []
        for category_name in sorted(features.keys()):
            category = features[category_name]
            if isinstance(category, dict):
                for value in sorted(category.values()):
                    if isinstance(value, (int, float)):
                        vector.append(float(value))
        return vector


class EmbeddingAgent:
    """
    Converts features to embeddings and retrieves similar historical sprints.
    Uses ChromaDB for RAG with local embeddings (no Ollama required).
    """

    def __init__(self, tool_registry: ToolRegistry, ollama_client: Optional[OllamaClient] = None):
        self.vector_tool = tool_registry.vector_tool
        self.logger = logging.getLogger(self.__class__.__name__)

    def execute(self, state: OrchestratorState) -> OrchestratorState:
        """Execute embedding and RAG retrieval using local ChromaDB embeddings."""
        try:
            self.logger.info("Generating embeddings and retrieving similar sprints...")

            # Build context text for embedding
            context_text = self._build_context_text(state)

            # Retrieve similar sprints from ChromaDB (uses local embeddings, no Ollama)
            # ChromaDB will automatically embed using its default local embedding model
            similar_result = self.vector_tool.find_similar_sprints_by_text(context_text, k=5)
            if similar_result["status"] == "success":
                state.similar_sprint_ids = [
                    s["sprint_id"] for s in similar_result["similar_sprints"]
                ]
                self.logger.info(f"  ✓ Retrieved {len(state.similar_sprint_ids)} similar sprints (local embeddings)")

            # Store embedding representation (simulated for compatibility)
            state.vector_embeddings["sprint_context"] = self._simple_embedding(context_text)

            state.execution_logs.append(
                f"[embedding_agent] Generated local embeddings, found {len(state.similar_sprint_ids)} similar sprints"
            )

        except Exception as e:
            state.errors.append(f"Embedding/RAG failed: {str(e)}")
            self.logger.error(f"Embedding error: {e}")

        return state

    def _simple_embedding(self, text: str) -> list:
        """Create a simple embedding representation without Ollama."""
        # Hash-based pseudo-embedding for compatibility
        hash_val = hash(text) % 256
        return [float(hash_val / 256.0)] * 10  # 10-dim vector

    def _build_context_text(self, state: OrchestratorState) -> str:
        """Build text for embedding."""
        parts = [
            f"Sprint {state.sprint_id or 'N/A'}",
            f"Issues: {len(state.github_issues)}",
            f"PRs: {len(state.github_prs)}",
            f"Commits: {len(state.commits)}",
        ]

        # Add feature summary
        if state.features:
            for category, metrics in state.features.items():
                if isinstance(metrics, dict):
                    parts.append(f"{category}: {', '.join(f'{k}={v:.2f}' for k, v in list(metrics.items())[:3])}")

        return "\n".join(parts)


class LLMReasonerAgent:
    """
    Performs deep contextual analysis using Llama with RAG context.
    Predicts sprint completion and health.
    """

    def __init__(self, tool_registry: ToolRegistry, ollama_client: OllamaClient):
        self.llm_tool = tool_registry.llm_tool
        self.ollama_client = ollama_client
        self.logger = logging.getLogger(self.__class__.__name__)

    def execute(self, state: OrchestratorState) -> OrchestratorState:
        """Execute LLM reasoning."""
        try:
            self.logger.info("Running LLM analysis...")
            strict_mode = state.eval_mode == "strict"

            # Build RAG context
            rag_context = {
                "similar_sprints": state.similar_sprint_ids,
                "feature_vector_length": len(state.feature_vector),
            }

            # Predict completion
            analysis = self.llm_tool.predict_completion_probability(
                features=state.features.get("temporal", {}),
                rag_context=rag_context,
                system_prompt=SYSTEM_PROMPTS["llm_reasoner"],
            )

            if analysis:
                parse_fallback = str(analysis.get("reasoning", "")).startswith("Could not parse LLM response")
                if strict_mode and parse_fallback:
                    state.sprint_analysis = None
                    state.analysis_source = "error"
                    state.errors.append("LLM analysis parse error in strict mode")
                    self.logger.warning("Strict mode: rejected malformed LLM analysis output")
                else:
                    state.sprint_analysis = analysis
                    state.analysis_source = "fallback" if parse_fallback else "llm"
                    self.logger.info(
                        f"  ✓ Predicted: {analysis.get('completion_probability', 0):.0f}% "
                        f"({analysis.get('health_status', 'unknown')})"
                    )
            else:
                state.analysis_source = "none"

            state.execution_logs.append(
                f"[llm_reasoner] Analysis complete: "
                f"{state.sprint_analysis.get('completion_probability', 0) if state.sprint_analysis else 0:.0f}% completion"
            )

        except Exception as e:
            state.analysis_source = "error"
            state.errors.append(f"LLM reasoning failed: {str(e)}")
            self.logger.error(f"LLM reasoner error: {e}")

        return state


class RiskAssessorAgent:
    """
    Identifies and quantifies risks using LLM analysis.
    Detects blockers, delays, team issues.
    """

    def __init__(self, tool_registry: ToolRegistry, ollama_client: OllamaClient):
        self.llm_tool = tool_registry.llm_tool
        self.logger = logging.getLogger(self.__class__.__name__)

    def execute(self, state: OrchestratorState) -> OrchestratorState:
        """Identify risks."""
        try:
            self.logger.info("Assessing risks...")
            strict_mode = state.eval_mode == "strict"

            # Analyze risks using LLM
            risk_result = self.llm_tool.analyze_risks(
                features=state.features,
                rag_context={"similar_sprints": state.similar_sprint_ids},
            )

            if risk_result.get("error"):
                self.logger.warning(f"Risk analysis parsing error: {risk_result['error']}")
                state.errors.append(f"Risk model error: {risk_result['error']}")

            # Convert to RiskItem objects
            risks_data = risk_result.get("risks", [])
            if not risks_data:
                if strict_mode:
                    state.risk_source = "none"
                    state.errors.append("Strict mode: no valid LLM risks produced")
                else:
                    risks_data = self._build_fallback_risks(state)
                    state.risk_source = "fallback"
            else:
                state.risk_source = "llm"

            for risk_data in risks_data:
                if isinstance(risk_data, dict):
                    try:
                        risk = RiskItem(**risk_data)
                        state.identified_risks.append(risk)
                    except Exception as e:
                        self.logger.warning(f"Could not create RiskItem: {e}")
                        if strict_mode:
                            state.errors.append(f"Strict mode risk schema rejection: {str(e)}")

            if strict_mode and not state.identified_risks:
                state.risk_source = "error"

            self.logger.info(f"  ✓ Identified {len(state.identified_risks)} risks")
            state.execution_logs.append(
                f"[risk_assessor] Identified {len(state.identified_risks)} risks"
            )

        except Exception as e:
            state.risk_source = "error"
            state.errors.append(f"Risk assessment failed: {str(e)}")
            self.logger.error(f"Risk assessor error: {e}")

        return state

    def _build_fallback_risks(self, state: OrchestratorState) -> list[dict[str, Any]]:
        """Create deterministic risks when LLM yields no valid output."""
        temporal = state.features.get("temporal", {}) if isinstance(state.features, dict) else {}
        risks: list[dict[str, Any]] = []

        completion = state.sprint_analysis.get("completion_probability", 50) if state.sprint_analysis else 50
        velocity_gap = float(temporal.get("velocity_gap", 0.0) or 0.0)
        pr_merge_rate = float(temporal.get("pr_merge_rate", 0.0) or 0.0)

        if completion < 50:
            risks.append({
                "risk_type": "completion_risk",
                "severity": 0.8,
                "description": "Low predicted completion probability indicates sprint delivery risk.",
                "affected_issues": [],
            })

        if velocity_gap > 0:
            risks.append({
                "risk_type": "velocity_gap",
                "severity": min(1.0, 0.5 + velocity_gap),
                "description": "Current closure pace is below required sprint velocity.",
                "affected_issues": [],
            })

        if pr_merge_rate < 0.5:
            risks.append({
                "risk_type": "review_bottleneck",
                "severity": 0.6,
                "description": "Pull request merge rate is low, indicating possible review bottlenecks.",
                "affected_issues": [],
            })

        # Guarantee at least one risk for downstream recommendation logic.
        if not risks:
            risks.append({
                "risk_type": "execution_uncertainty",
                "severity": 0.4,
                "description": "Insufficient stable signals; monitoring risk conservatively.",
                "affected_issues": [],
            })

        return risks


class RecommenderAgent:
    """
    Generates actionable interventions based on risks and precedent.
    Uses RAG to find similar past sprints and their solutions.
    """

    def __init__(self, tool_registry: ToolRegistry, ollama_client: OllamaClient):
        self.llm_tool = tool_registry.llm_tool
        self.logger = logging.getLogger(self.__class__.__name__)

    def execute(self, state: OrchestratorState) -> OrchestratorState:
        """Generate recommendations."""
        try:
            self.logger.info("Generating recommendations...")
            strict_mode = state.eval_mode == "strict"

            # Convert risks to dict for LLM
            risk_dicts = [
                r.dict() if hasattr(r, "dict") else r
                for r in state.identified_risks
            ]

            # Generate recommendations
            recommendation_result = self.llm_tool.generate_recommendations(
                risks=risk_dicts,
                rag_context={
                    "similar_sprints": state.similar_sprint_ids,
                },
            )
            recommendations = recommendation_result.get("recommendations", [])
            if recommendation_result.get("error"):
                state.errors.append(f"Recommendation model error: {recommendation_result['error']}")

            if recommendations:
                state.recommendation_source = "llm"
            elif risk_dicts and not strict_mode:
                # Fallback to deterministic recommendations when LLM output is empty/unparseable.
                recommendations = self._build_fallback_recommendations(risk_dicts)
                state.recommendation_source = "fallback"
            else:
                state.recommendation_source = "none"
                if strict_mode:
                    state.errors.append("Strict mode: no valid LLM recommendations produced")

            # Convert to Recommendation objects
            for rec_data in recommendations:
                if isinstance(rec_data, dict):
                    try:
                        # Fill in missing required fields with defaults
                        if "description" not in rec_data:
                            rec_data["description"] = rec_data.get(
                                "expected_impact",
                                rec_data.get("title", "No description")
                            )
                        if "action" not in rec_data:
                            rec_data["action"] = rec_data.get(
                                "title", "Take action"
                            )
                        rec = Recommendation(**rec_data)
                        state.recommendations.append(rec)
                    except Exception as e:
                        self.logger.warning(
                            f"Could not create Recommendation: {e}"
                        )
                        if strict_mode:
                            state.errors.append(f"Strict mode recommendation schema rejection: {str(e)}")

            if strict_mode and not state.recommendations:
                state.recommendation_source = "error"

            self.logger.info(f"  ✓ Generated {len(state.recommendations)} recommendations")
            state.execution_logs.append(
                f"[recommender] Generated {len(state.recommendations)} recommendations"
            )

        except Exception as e:
            state.recommendation_source = "error"
            state.errors.append(f"Recommendation generation failed: {str(e)}")
            self.logger.error(f"Recommender error: {e}")

        return state

    def _build_fallback_recommendations(self, risks: list[dict]) -> list[dict[str, str]]:
        """Build simple deterministic recommendations from risks."""
        output: list[dict[str, str]] = []
        for risk in risks[:5]:
            risk_type = str(risk.get("risk_type", "delivery_risk"))
            severity = float(risk.get("severity", 0.5))
            priority = "high" if severity >= 0.7 else "medium"

            output.append({
                "title": f"Mitigate {risk_type.replace('_', ' ')}",
                "description": f"Address {risk_type.replace('_', ' ')} based on current sprint signals.",
                "priority": priority,
                "expected_impact": "Improved sprint predictability and reduced blocker risk",
                "action": "Assign owner, define mitigation tasks, and review progress daily",
                "evidence_source": "fallback_policy",
            })
        return output


class ExplainerAgent:
    """
    Generates natural language narratives with evidence attribution.
    Synthesizes all agent outputs into readable summary.
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def execute(self, state: OrchestratorState) -> OrchestratorState:
        """Generate explanation."""
        try:
            self.logger.info("Generating narrative explanation...")

            narrative = self._build_narrative(state)
            state.narrative_explanation = narrative

            self.logger.info("  ✓ Narrative generated")
            state.execution_logs.append("[explainer] Narrative explanation generated")

            # Mark workflow complete
            state.workflow_complete = True

        except Exception as e:
            state.errors.append(f"Explanation generation failed: {str(e)}")
            self.logger.error(f"Explainer error: {e}")

        return state

    def _build_narrative(self, state: OrchestratorState) -> str:
        """Build natural language narrative."""
        lines = ["# Sprint Intelligence Report\n"]

        # Overview
        analysis = state.sprint_analysis or {}
        if analysis:
            lines.append("## Executive Summary")
            completion = analysis.get("completion_probability", 0)
            health = analysis.get("health_status", "Unknown")
            lines.append(f"- **Completion Probability**: {completion:.0f}%")
            lines.append(f"- **Sprint Health**: {health.title()}")
            lines.append("")

        # Risks
        if state.identified_risks:
            lines.append("## Key Risks")
            for risk in state.identified_risks[:5]:
                risk_dict = risk.dict() if hasattr(risk, "dict") else risk
                severity = risk_dict.get("severity", 0)
                risk_type = risk_dict.get("risk_type", "Unknown")
                lines.append(f"- **{risk_type.title()}** (severity: {severity:.1f}/1.0)")
                if desc := risk_dict.get("description"):
                    lines.append(f"  {desc}")
            lines.append("")

        # Recommendations
        if state.recommendations:
            lines.append("## Recommended Actions")
            for rec in state.recommendations[:5]:
                rec_dict = rec.dict() if hasattr(rec, "dict") else rec
                title = rec_dict.get("title", "Action")
                priority = rec_dict.get("priority", "medium")
                lines.append(f"- **{title}** [{priority.upper()}]")
                if desc := rec_dict.get("description"):
                    lines.append(f"  {desc}")
            lines.append("")

        # Evidence
        if state.similar_sprint_ids:
            lines.append("## Evidence Base")
            lines.append(f"Analysis based on {len(state.similar_sprint_ids)} similar historical sprints")
            lines.append("")

        return "\n".join(lines)
