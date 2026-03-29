"""
Detailed agent implementations with full integration to existing pipeline.
Each agent is a production-ready node in the LangGraph orchestration.
"""

import logging
import json
from typing import Optional, Any
from src.agents.state import OrchestratorState, RiskItem, Recommendation, SprintMetrics, RAGContext
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
    Retrieves similar historical sprints from ChromaDB for RAG context.
    Uses SprintChromaDB with local embeddings (all-MiniLM-L6-v2, no Ollama).
    Populates state.rag_context with full documents and citation URLs.
    """

    def __init__(self, tool_registry: ToolRegistry, ollama_client: Optional[OllamaClient] = None):
        self.vector_tool = tool_registry.vector_tool
        self.logger = logging.getLogger(self.__class__.__name__)
        self._chroma: Optional[Any] = None

    def _get_chroma(self):
        """Lazy-init SprintChromaDB."""
        if self._chroma is None:
            from src.chromadb import SprintChromaDB
            self._chroma = SprintChromaDB()
        return self._chroma

    def execute(self, state: OrchestratorState) -> OrchestratorState:
        """Execute RAG retrieval against ChromaDB sprint collection."""
        try:
            self.logger.info("Retrieving similar sprints from ChromaDB...")
            chroma = self._get_chroma()

            # Determine owner/repo from first repository in state
            owner, repo = self._parse_owner_repo(state)

            # Query ChromaDB with current sprint features
            rag_result = chroma.query_similar_sprints(
                owner=owner,
                repo=repo,
                sprint_id=state.sprint_id,
                features=state.features if isinstance(state.features, dict) else None,
                k=5,
            )

            # Populate state with RAG context
            similar = rag_result.get("similar_sprints", [])
            state.similar_sprint_ids = [s["sprint_id"] for s in similar]
            state.evidence_citations = rag_result.get("evidence_citations", [])

            state.rag_context = RAGContext(
                similar_sprints=similar,
                evidence_citations=rag_result.get("evidence_citations", []),
            )

            # Store formatted context text for downstream LLM prompt injection
            state.vector_embeddings["rag_context_text"] = [
                float(len(rag_result.get("context_text", "")))
            ]
            # attach the full text as a state field agents can read
            if not hasattr(state, "_rag_context_text"):
                object.__setattr__(state, "_rag_context_text", rag_result.get("context_text", ""))

            self.logger.info(
                f"  ✓ Retrieved {len(similar)} similar sprints, "
                f"{len(state.evidence_citations)} citations"
            )
            state.execution_logs.append(
                f"[embedding_agent] RAG retrieval: {len(similar)} similar sprints, "
                f"{len(state.evidence_citations)} evidence citations"
            )

        except Exception as e:
            state.errors.append(f"Embedding/RAG failed: {str(e)}")
            self.logger.error(f"Embedding error: {e}")

        return state

    @staticmethod
    def _parse_owner_repo(state: OrchestratorState) -> tuple[str, str]:
        """Extract owner/repo from the first repository URL or slug."""
        for r in state.repositories:
            # Handle full URLs or owner/repo slugs
            slug = r.replace("https://github.com/", "").strip("/")
            parts = slug.split("/")
            if len(parts) >= 2:
                return parts[0], parts[1]
        return "", ""


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

            # Build rich RAG context from embedded sprint data
            rag_context = self._build_rag_context(state)

            # Provide richer, multi-modal context so sprint analysis is not reduced
            # to a narrow temporal/risk snapshot.
            llm_features = {
                "temporal": state.features.get("temporal", {}),
                "activity": state.features.get("activity", {}),
                "code": state.features.get("code", {}),
                "risk": state.features.get("risk", {}),
                "team": state.features.get("team", {}),
            }

            # Predict completion
            analysis = self.llm_tool.predict_completion_probability(
                features=llm_features,
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

    def _build_rag_context(self, state: OrchestratorState) -> dict:
        """Build rich RAG context dict from embedded sprint data for LLM prompts."""
        rag = state.rag_context
        context: dict[str, Any] = {
            "similar_sprints": state.similar_sprint_ids,
            "feature_vector_length": len(state.feature_vector),
            "dependency_risk_propagation": (
                state.dependency_graph.get("risk_propagation", {})
                if isinstance(state.dependency_graph, dict)
                else {}
            ),
        }

        # Inject full historical sprint context if available from EmbeddingAgent
        if rag:
            context["similar_sprint_details"] = []
            for s in (rag.similar_sprints or [])[:5]:
                context["similar_sprint_details"].append({
                    "sprint_id": s.get("sprint_id", ""),
                    "repo": s.get("repo", ""),
                    "similarity": s.get("similarity", 0),
                    "risk_score": s.get("risk_score", 0),
                    "is_at_risk": s.get("is_at_risk", False),
                    "content": (s.get("content") or "")[:600],
                })
            context["evidence_citations"] = rag.evidence_citations or []

        return context


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

            # Analyze risks using LLM with rich RAG context
            rag_ctx: dict[str, Any] = {"similar_sprints": state.similar_sprint_ids}
            if state.rag_context and state.rag_context.evidence_citations:
                rag_ctx["evidence_citations"] = state.rag_context.evidence_citations
            risk_result = self.llm_tool.analyze_risks(
                features=state.features,
                rag_context=rag_ctx,
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

            # Generate recommendations with evidence citations
            rag_ctx: dict[str, Any] = {"similar_sprints": state.similar_sprint_ids}
            if state.rag_context:
                if state.rag_context.evidence_citations:
                    rag_ctx["evidence_citations"] = state.rag_context.evidence_citations
                # Include similar sprint detail summaries for precedent matching
                if state.rag_context.similar_sprints:
                    rag_ctx["similar_sprint_details"] = [
                        {"sprint_id": s.get("sprint_id"), "risk_score": s.get("risk_score", 0),
                         "is_at_risk": s.get("is_at_risk", False)}
                        for s in state.rag_context.similar_sprints[:3]
                    ]
            recommendation_result = self.llm_tool.generate_recommendations(
                risks=risk_dicts,
                rag_context=rag_ctx,
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
    Synthesizes all agent outputs into readable summary with GitHub citations.
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def execute(self, state: OrchestratorState) -> OrchestratorState:
        """Generate explanation with evidence citations."""
        try:
            self.logger.info("Generating narrative explanation...")

            # Gather evidence from ChromaDB for the current sprint
            self._enrich_evidence(state)

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

    def _enrich_evidence(self, state: OrchestratorState) -> None:
        """Fetch sprint evidence (issues/PRs/commits) from ChromaDB for citations."""
        try:
            from src.chromadb import SprintChromaDB
            chroma = SprintChromaDB()

            # Parse owner/repo
            owner, repo = "", ""
            for r in state.repositories:
                slug = r.replace("https://github.com/", "").strip("/")
                parts = slug.split("/")
                if len(parts) >= 2:
                    owner, repo = parts[0], parts[1]
                    break

            if not owner:
                return

            evidence = chroma.get_sprint_evidence(
                owner=owner, repo=repo,
                sprint_id=state.sprint_id or "",
            )

            # Collect citation URLs from evidence
            citations: list[str] = list(state.evidence_citations or [])
            for issue in evidence.get("issues", [])[:5]:
                url = issue.get("url", "")
                if url and url not in citations:
                    citations.append(url)
            for pr in evidence.get("prs", [])[:5]:
                url = pr.get("url", "")
                if url and url not in citations:
                    citations.append(url)
            for commit in evidence.get("commits", [])[:3]:
                url = commit.get("url", "")
                if url and url not in citations:
                    citations.append(url)

            state.evidence_citations = citations

        except Exception as e:
            self.logger.warning(f"Evidence enrichment skipped: {e}")

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
            if "health_score" in analysis:
                lines.append(f"- **Composite Health Score**: {analysis.get('health_score', 0):.1f}/100")
            if "dependency_risk_score" in analysis:
                lines.append(f"- **Cross-Repo Dependency Risk**: {analysis.get('dependency_risk_score', 0):.1f}/100")
            if analysis.get("key_signals"):
                lines.append("- **Key Signals**:")
                for signal in list(analysis.get("key_signals", []))[:4]:
                    lines.append(f"  - {signal}")
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

        # Evidence — cite similar sprints and GitHub URLs
        rag = state.rag_context
        if rag and rag.similar_sprints:
            lines.append("## Evidence Base")
            lines.append(f"Analysis grounded in {len(rag.similar_sprints)} similar historical sprints:\n")
            for s in rag.similar_sprints[:5]:
                sid = s.get("sprint_id", "unknown")
                repo = s.get("repo", "")
                sim_score = s.get("similarity", 0)
                risk = s.get("risk_score", 0)
                lines.append(f"- **{sid}** ({repo}) — similarity: {sim_score:.2f}, risk: {risk:.2f}")
            lines.append("")
        elif state.similar_sprint_ids:
            lines.append("## Evidence Base")
            lines.append(f"Analysis based on {len(state.similar_sprint_ids)} similar historical sprints")
            lines.append("")

        # Citations
        if state.evidence_citations:
            lines.append("## Citations")
            for url in state.evidence_citations[:15]:
                lines.append(f"- {url}")
            lines.append("")

        return "\n".join(lines)
