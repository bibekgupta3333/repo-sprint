"""
Tool definitions for agent workflows.
Tools represent distinct capabilities that agents can invoke.
Compatible with LangChain's tool_choice and tool_use patterns.
"""

import json
import logging
from typing import Any, Optional
from datetime import datetime, timedelta
import hashlib

logger = logging.getLogger(__name__)


# ============================================================================
# GitHub Data Tools
# ============================================================================

class GitHubDataTool:
    """Fetch and cache GitHub data (issues, PRs, commits)."""

    def __init__(self, cache_dir: str = ".cache/github"):
        self.cache_dir = cache_dir
        self._initialize_cache()

    def _initialize_cache(self) -> None:
        """Initialize cache directory."""
        import os
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_key(self, repo_url: str, data_type: str) -> str:
        """Generate cache key from repo URL and data type."""
        key = f"{repo_url}:{data_type}"
        return hashlib.md5(key.encode()).hexdigest()

    def fetch_issues(
        self,
        repo_url: str,
        state: str = "all",
        cache_ttl_hours: int = 6,
    ) -> dict[str, Any]:
        """
        Fetch GitHub issues for a repository.
        Implements caching and rate limit handling.
        Integrates with LocalScraper from scripts/_core/.
        """
        try:
            from scripts._core.local_scraper import LocalScraper

            scraper = LocalScraper(repo_url)
            issues = scraper.scrape_issues(state=state)

            return {
                "status": "success",
                "issues": issues,
                "total_count": len(issues),
                "cache_hit": False,
                "last_updated": datetime.now().isoformat(),
            }
        except ImportError:
            logger.warning("LocalScraper not available, returning empty")
            return {
                "status": "success",
                "issues": [],
                "total_count": 0,
                "cache_hit": False,
                "last_updated": datetime.now().isoformat(),
            }
        except Exception as e:
            error_msg = str(e)
            # Local scraper can require a checked-out repository path.
            # For slug-only repos, degrade gracefully to empty data.
            if "Not a git repository" in error_msg:
                logger.warning(f"Issue fetch skipped for non-local repo '{repo_url}'")
                return {
                    "status": "success",
                    "issues": [],
                    "total_count": 0,
                    "cache_hit": False,
                    "last_updated": datetime.now().isoformat(),
                }

            logger.error(f"Issue fetch failed: {e}")
            return {
                "status": "error",
                "issues": [],
                "error": error_msg,
            }

    def fetch_pull_requests(
        self,
        repo_url: str,
        state: str = "all",
    ) -> dict[str, Any]:
        """Fetch GitHub pull requests."""
        try:
            from scripts._core.local_scraper import LocalScraper

            scraper = LocalScraper(repo_url)
            prs = scraper.scrape_pull_requests(state=state)

            return {
                "status": "success",
                "pull_requests": prs,
                "total_count": len(prs),
            }
        except Exception as e:
            error_msg = str(e)
            if "Not a git repository" in error_msg:
                logger.warning(f"PR fetch skipped for non-local repo '{repo_url}'")
                return {
                    "status": "success",
                    "pull_requests": [],
                    "total_count": 0,
                }

            logger.error(f"PR fetch failed: {e}")
            return {
                "status": "error",
                "pull_requests": [],
                "error": error_msg,
            }

    def fetch_commits(
        self,
        repo_url: str,
        since: Optional[str] = None,
    ) -> dict[str, Any]:
        """Fetch commit history with metadata."""
        try:
            from scripts._core.local_scraper import LocalScraper

            scraper = LocalScraper(repo_url)
            commits = scraper.scrape_commits(since=since) if since else scraper.scrape_commits()

            return {
                "status": "success",
                "commits": commits,
                "total_count": len(commits),
            }
        except Exception as e:
            error_msg = str(e)
            if "Not a git repository" in error_msg:
                logger.warning(f"Commit fetch skipped for non-local repo '{repo_url}'")
                return {
                    "status": "success",
                    "commits": [],
                    "total_count": 0,
                }

            logger.error(f"Commit fetch failed: {e}")
            return {
                "status": "error",
                "commits": [],
                "error": error_msg,
            }


# ============================================================================
# Feature Extraction Tools
# ============================================================================

class FeatureExtractionTool:
    """Extract and compute sprint metrics from raw GitHub data."""

    def compute_temporal_features(
        self,
        issues: list[dict],
        pull_requests: list[dict],
        sprint_start: str,
        sprint_end: str,
    ) -> dict[str, float]:
        """
        Compute time-based sprint metrics.

        Returns:
            velocity_gap, daily_closure_rate, required_daily_rate, completion_percentage
        """
        # TODO: Integrate with src/data/FeatureExtractor
        return {
            "completion_percentage": 0.0,
            "velocity_gap": 0.0,
            "daily_closure_rate": 0.0,
            "required_daily_rate": 0.0,
            "pr_merge_rate": 0.0,
        }

    def compute_code_complexity_features(
        self,
        commits: list[dict],
    ) -> dict[str, float]:
        """Analyze code changes for churn, complexity, risk."""
        return {
            "code_churn": 0.0,
            "cyclomatic_complexity": 0.0,
            "dependency_additions": 0,
            "breaking_changes": 0,
        }

    def compute_team_sentiment(
        self,
        comments: list[dict],
    ) -> dict[str, float]:
        """Analyze comment sentiment for team morale."""
        return {
            "avg_sentiment": 0.0,
            "negative_comment_ratio": 0.0,
            "engagement_score": 0.0,
        }

    def extract_all_features(
        self,
        github_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Master feature extraction coordinating all sub-feature computations.
        Integrates with src/data/FeatureExtractor.
        """
        try:
            # FeatureExtractor in src.data.features expects repo_data + sprint payload.
            from src.data.features import FeatureExtractor

            sprint_data = {
                "issues": github_data.get("issues", []),
                "prs": github_data.get("prs", []),
                "commits": github_data.get("commits", []),
                "commit_diffs": github_data.get("commit_diffs", []),
            }

            extractor = FeatureExtractor(repo_data={"name": "runtime_repo"}, sprint=sprint_data)
            metrics = extractor.extract_metrics()

            # Normalize into modality buckets expected by downstream agents.
            return {
                "temporal": {
                    "days_span": float(metrics.get("days_span", 0)),
                    "issue_age_avg": float(metrics.get("issue_age_avg", 0)),
                    "pr_age_avg": float(metrics.get("pr_age_avg", 0)),
                },
                "activity": {
                    "total_issues": float(metrics.get("total_issues", 0)),
                    "total_prs": float(metrics.get("total_prs", 0)),
                    "total_commits": float(metrics.get("total_commits", 0)),
                    "closed_issues": float(metrics.get("closed_issues", 0)),
                    "merged_prs": float(metrics.get("merged_prs", 0)),
                    "issue_resolution_rate": float(metrics.get("issue_resolution_rate", 0)),
                    "pr_merge_rate": float(metrics.get("pr_merge_rate", 0)),
                    "commit_frequency": float(metrics.get("commit_frequency", 0)),
                    "code_changes": float(metrics.get("code_changes", 0)),
                },
                "code": {
                    "total_code_changes": float(metrics.get("total_code_changes", 0)),
                    "avg_pr_size": float(metrics.get("avg_pr_size", 0)),
                    "code_concentration": float(metrics.get("code_concentration", 0)),
                    "total_additions": float(metrics.get("total_additions", 0)),
                    "total_deletions": float(metrics.get("total_deletions", 0)),
                    "files_changed": float(metrics.get("files_changed", 0)),
                },
                "risk": {
                    "stalled_issues": float(metrics.get("stalled_issues", 0)),
                    "unreviewed_prs": float(metrics.get("unreviewed_prs", 0)),
                    "abandoned_prs": float(metrics.get("abandoned_prs", 0)),
                    "long_open_issues": float(metrics.get("long_open_issues", 0)),
                },
                "team": {
                    "unique_authors": float(metrics.get("unique_authors", 0)),
                    "author_participation": float(metrics.get("author_participation", 0)),
                },
                "language": metrics.get("language_breakdown", {}),
            }
        except ImportError:
            logger.warning("FeatureExtractor not available, computing basic features")
            return self._compute_fallback_features(github_data)
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return self._compute_fallback_features(github_data)

    def _compute_fallback_features(self, github_data: dict[str, Any]) -> dict[str, Any]:
        """Fallback feature computation if main extractor unavailable."""
        issues = github_data.get("issues", [])
        prs = github_data.get("prs", [])
        commits = github_data.get("commits", [])

        return {
            "temporal": self.compute_temporal_features(
                issues, prs,
                github_data.get("sprint_start", ""),
                github_data.get("sprint_end", ""),
            ),
            "code": self.compute_code_complexity_features(commits),
            "sentiment": self.compute_team_sentiment([]),
        }


# ============================================================================
# Vector Store Tools (ChromaDB)
# ============================================================================

class VectorStoreTool:
    """Store and retrieve sprint embeddings for RAG."""

    def __init__(self, db_path: str = "./chroma_db"):
        self.db_path = db_path
        self._initialize_db()

    def _initialize_db(self):
        """Initialize or connect to ChromaDB."""
        try:
            import chromadb
            # Handle stale ONNX model cache (Errno 17: File exists)
            self._fix_onnx_cache()
            self.client = chromadb.PersistentClient(path=self.db_path)
            self.collection = self.client.get_or_create_collection(
                name="sprints",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"ChromaDB initialized at {self.db_path}")
        except ImportError:
            logger.warning("ChromaDB not available")
            self.client = None
            self.collection = None
        except Exception as e:
            logger.warning(f"ChromaDB init failed: {e}")
            self.client = None
            self.collection = None

    @staticmethod
    def _fix_onnx_cache():
        """Fix stale ONNX model cache that causes [Errno 17]."""
        import os
        import shutil
        cache_path = os.path.expanduser(
            "~/.cache/chroma/onnx_models/all-MiniLM-L6-v2"
        )
        if os.path.exists(cache_path) and not os.path.isdir(cache_path):
            # It's a file but should be a directory — remove it
            os.remove(cache_path)
            logger.info("Removed stale ONNX cache file")
        elif os.path.isdir(cache_path):
            # Check if directory is corrupt (empty or missing model)
            contents = os.listdir(cache_path)
            if not contents:
                shutil.rmtree(cache_path)
                logger.info("Removed empty ONNX cache directory")

    def upsert_sprint(
        self,
        sprint_id: str,
        embedding: list[float],
        metadata: dict[str, Any],
        text_content: str,
    ) -> dict[str, Any]:
        """Store sprint embedding with metadata."""
        if not self.collection:
            return {"status": "error", "error": "ChromaDB not initialized"}

        try:
            self.collection.upsert(
                ids=[sprint_id],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[text_content],
            )
            return {
                "status": "success",
                "sprint_id": sprint_id,
                "stored": True,
            }
        except Exception as e:
            logger.error(f"Upsert failed: {e}")
            return {"status": "error", "error": str(e)}

    def find_similar_sprints(
        self,
        query_embedding: list[float],
        k: int = 5,
    ) -> dict[str, Any]:
        """Retrieve top-k similar sprints by embedding similarity."""
        if not self.collection:
            return {
                "status": "success",
                "similar_sprints": [],
                "scores": [],
            }

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
            )

            similar_sprints = []
            scores = []

            if results and results.get("ids") and len(results["ids"]) > 0:
                for idx, (sprint_id, distance, metadata) in enumerate(
                    zip(
                        results["ids"][0],
                        results.get("distances", [[]])[0],
                        results.get("metadatas", [[]])[0] if results.get("metadatas") else [{}] * k,
                    )
                ):
                    # Convert distance to similarity (cosine: 0=identical, 1=opposite)
                    similarity = 1 - distance if distance else 0
                    similar_sprints.append({
                        "sprint_id": sprint_id,
                        "similarity_score": similarity,
                        **metadata
                    })
                    scores.append(similarity)

            return {
                "status": "success",
                "similar_sprints": similar_sprints,
                "scores": scores,
            }
        except Exception as e:
            logger.error(f"Similarity query failed: {e}")
            return {
                "status": "error",
                "similar_sprints": [],
                "error": str(e),
            }

    def find_similar_sprints_by_text(
        self,
        query_text: str,
        k: int = 5,
    ) -> dict[str, Any]:
        """Retrieve top-k similar sprints by text query using ChromaDB's local embeddings."""
        if not self.collection:
            return {
                "status": "success",
                "similar_sprints": [],
                "scores": [],
            }

        try:
            # ChromaDB automatically embeds the query text using its local embedding model
            # No Ollama needed - uses sentence-transformer under the hood
            results = self.collection.query(
                query_texts=[query_text],
                n_results=k,
            )

            similar_sprints = []
            scores = []

            if results and results.get("ids") and len(results["ids"]) > 0:
                for idx, (sprint_id, distance, metadata) in enumerate(
                    zip(
                        results["ids"][0],
                        results.get("distances", [[]])[0],
                        results.get("metadatas", [[]])[0] if results.get("metadatas") else [{}] * k,
                    )
                ):
                    # Convert distance to similarity
                    similarity = 1 - distance if distance else 0
                    similar_sprints.append({
                        "sprint_id": sprint_id,
                        "similarity_score": similarity,
                        **metadata
                    })
                    scores.append(similarity)

            return {
                "status": "success",
                "similar_sprints": similar_sprints,
                "scores": scores,
            }
        except Exception as e:
            logger.error(f"Text-based similarity query failed: {e}")
            return {
                "status": "error",
                "similar_sprints": [],
                "error": str(e),
            }

    def retrieve_intervention_history(
        self,
        risk_type: str,
        k: int = 3,
    ) -> list[dict[str, Any]]:
        """Retrieve successful interventions for a risk type."""
        if not self.collection:
            return []

        try:
            # Query for sprints that had this risk type and were resolved
            results = self.collection.get(
                where={"risk_type": risk_type, "resolved": True},
                limit=k,
            )

            interventions = []
            if results and results.get("ids"):
                for metadata in results.get("metadatas", []):
                    if metadata:
                        interventions.append({
                            "sprint_id": metadata.get("sprint_id"),
                            "risk_type": metadata.get("risk_type"),
                            "intervention": metadata.get("intervention"),
                            "outcome": metadata.get("outcome"),
                            "impact": metadata.get("impact"),
                        })

            return interventions
        except Exception as e:
            logger.warning(f"Intervention history retrieval failed: {e}")
            return []


# ============================================================================
# LLM Inference Tools
# ============================================================================

class LLMInferenceTool:
    """Run LLM inference with structured prompting."""

    def __init__(self, ollama_client):
        self.ollama_client = ollama_client

    def predict_completion_probability(
        self,
        features: dict[str, float],
        rag_context: Optional[dict] = None,
        system_prompt: Optional[str] = None,
    ) -> dict[str, Any]:
        """Predict sprint completion using LLM reasoning."""
        # Structured prompt assembly
        prompt = self._build_completion_prompt(features, rag_context)

        response = self.ollama_client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
        )

        return self._parse_completion_response(response)

    def analyze_risks(
        self,
        features: dict[str, float],
        rag_context: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Identify and rank risks using LLM."""
        prompt = self._build_risk_analysis_prompt(features, rag_context)
        response = self.ollama_client.generate(prompt=prompt)
        return self._parse_risk_response(response)

    def generate_recommendations(
        self,
        risks: list[dict],
        rag_context: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Generate interventions based on risks and precedent."""
        prompt = self._build_recommendation_prompt(risks, rag_context)
        response = self.ollama_client.generate(prompt=prompt)
        return self._parse_recommendation_response(response)

    def _build_completion_prompt(
        self,
        features: dict[str, float],
        rag_context: Optional[dict],
    ) -> str:
        """Build structured prompt for completion prediction with RAG context."""
        metrics_str = json.dumps(features, indent=2)

        # Format RAG context — include full historical sprint details when available
        rag_parts = []
        if rag_context:
            for detail in rag_context.get("similar_sprint_details", [])[:8]:
                rag_parts.append(
                    f"Sprint {detail.get('sprint_id', '?')} ({detail.get('repo', '')}) — "
                    f"risk: {detail.get('risk_score', 0):.2f}, at-risk: {detail.get('is_at_risk', False)}\n"
                    f"{detail.get('content', '')}"
                )
            evidence = rag_context.get("evidence_citations", [])
            if evidence:
                rag_parts.append("Evidence URLs: " + ", ".join(evidence[:10]))
            context_text = str(rag_context.get("context_text", "") or "").strip()
            if context_text:
                rag_parts.append("Retrieved context excerpt:\n" + context_text[:2000])

        rag_str = "\n---\n".join(rag_parts) if rag_parts else "No historical data available."

        return f"""Analyze this sprint and predict completion probability.

**Current Metrics:**
{metrics_str}

**Similar Historical Sprints (RAG Context):**
{rag_str}

Cite specific GitHub issues, PRs, or commits (by URL or number) when explaining your reasoning.
Your reasoning must include:
1) What historical patterns match the current sprint,
2) Which signals drive the probability up/down,
3) What would most improve the forecast.

Provide JSON response with:
{{
  "completion_probability": <0-100>,
  "health_status": "on_track|at_risk|critical",
  "confidence_score": <0-1>,
    "reasoning": "<detailed explanation citing evidence from similar sprints and metrics>"
}}
"""

    def _build_risk_analysis_prompt(
        self,
        features: dict[str, float],
        rag_context: Optional[dict],
    ) -> str:
        """Build prompt for risk identification."""
        metrics_str = json.dumps(features, indent=2)

        return f"""Identify risks in this sprint based on metrics:

**Metrics:**
{metrics_str}

Return JSON array of risks:
[
  {{
    "risk_type": "<type>",
    "severity": <0-1>,
    "description": "<details>",
    "affected_issues": [<issue_numbers>]
  }}
]

Rules:
- "affected_issues" must be an array of integers only.
- If no issue numbers are known, return an empty array [].
- Return only valid JSON. No markdown code blocks.
"""

    def _build_recommendation_prompt(
        self,
        risks: list[dict],
        rag_context: Optional[dict],
    ) -> str:
        """Build prompt for recommendation generation with evidence citations."""
        risks_str = json.dumps(risks, indent=2)

        # Build precedent block from RAG context
        precedent_parts = []
        if rag_context:
            for detail in rag_context.get("similar_sprint_details", [])[:8]:
                precedent_parts.append(
                    f"Sprint {detail.get('sprint_id', '?')} — "
                    f"risk: {detail.get('risk_score', 0):.2f}, at-risk: {detail.get('is_at_risk', False)}, "
                    f"similarity: {detail.get('similarity', 0):.2f}\n"
                    f"Context: {(detail.get('content', '') or '')[:260]}"
                )
            evidence = rag_context.get("evidence_citations", [])
            if evidence:
                precedent_parts.append("Evidence URLs: " + ", ".join(evidence[:12]))
            current_analysis = rag_context.get("current_analysis", {})
            if current_analysis:
                precedent_parts.append("Current analysis snapshot: " + json.dumps(current_analysis, ensure_ascii=True)[:700])

        precedent_str = "\n".join(precedent_parts) if precedent_parts else "No precedent data available."

        return f"""Generate interventions for these sprint risks.

**Identified Risks:**
{risks_str}

**Successful Precedents:**
{precedent_str}

IMPORTANT: Each recommendation MUST include an "evidence_source" field that cites
a specific GitHub issue, PR, commit URL, or historical sprint ID as evidence for why
this intervention is recommended. If no specific URL is available, cite the similar
sprint ID (e.g., "Based on sprint_003 which had similar risk patterns").

Generate 4-7 recommendations and prioritize by expected impact.
Each recommendation description should explain why the intervention is needed now,
what risk it addresses, and what success signal to watch after execution.

Return JSON array of recommendations:
[
  {{
    "title": "<action>",
        "description": "<detailed rationale and why-now context>",
    "priority": "high|medium|low",
        "expected_impact": "<what improves and expected KPI shift>",
        "action": "<specific next steps with ownership/timeline hints>",
    "evidence_source": "<GitHub URL or sprint reference>"
  }}
]

Return only valid JSON. No markdown code blocks.
"""

    def _parse_completion_response(self, response: str) -> dict[str, Any]:
        """Parse LLM completion prediction response."""
        try:
            # Extract JSON from response
            json_str = response[response.find("{"):response.rfind("}")+1]
            return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            return {
                "completion_probability": 50,
                "health_status": "at_risk",
                "confidence_score": 0.3,
                "reasoning": f"Could not parse LLM response: {response[:100]}",
            }

    def _parse_risk_response(self, response: str) -> dict[str, Any]:
        """Parse LLM risk analysis response."""
        try:
            cleaned = response.strip()
            if "```" in cleaned:
                cleaned = cleaned.replace("```json", "").replace("```", "").strip()

            risks: list[dict[str, Any]] = []

            # Case 1: direct array
            if "[" in cleaned and "]" in cleaned:
                json_str = cleaned[cleaned.find("["):cleaned.rfind("]") + 1]
                parsed = json.loads(json_str)
                if isinstance(parsed, list):
                    risks = [r for r in parsed if isinstance(r, dict)]

            # Case 2: wrapped object, e.g. {"risks": [...]}
            if not risks and "{" in cleaned and "}" in cleaned:
                json_obj = cleaned[cleaned.find("{"):cleaned.rfind("}") + 1]
                parsed_obj = json.loads(json_obj)
                if isinstance(parsed_obj, dict) and isinstance(parsed_obj.get("risks"), list):
                    risks = [r for r in parsed_obj.get("risks", []) if isinstance(r, dict)]

            normalized: list[dict[str, Any]] = []
            for risk in risks:
                affected_raw = risk.get("affected_issues", [])
                affected_issues: list[int] = []

                if isinstance(affected_raw, list):
                    for item in affected_raw:
                        if isinstance(item, int):
                            affected_issues.append(item)
                        elif isinstance(item, str):
                            digits = "".join(ch for ch in item if ch.isdigit())
                            if digits:
                                affected_issues.append(int(digits))

                severity_raw = risk.get("severity", 0.5)
                try:
                    severity = float(severity_raw)
                except (TypeError, ValueError):
                    severity = 0.5
                severity = max(0.0, min(1.0, severity))

                normalized.append({
                    "risk_type": str(risk.get("risk_type", "delivery_risk")),
                    "severity": severity,
                    "description": str(risk.get("description", "Risk identified from sprint signals")),
                    "affected_issues": affected_issues,
                })

            return {"risks": normalized, "error": None}
        except (json.JSONDecodeError, ValueError):
            return {"risks": [], "error": f"Parse failed: {response[:100]}"}

    def _parse_recommendation_response(self, response: str) -> dict[str, Any]:
        """Parse LLM recommendation response."""
        try:
            cleaned = response.strip()

            # Handle markdown code fences
            if "```" in cleaned:
                cleaned = cleaned.replace("```json", "").replace("```", "").strip()

            # Case 1: direct array
            if "[" in cleaned and "]" in cleaned:
                json_str = cleaned[cleaned.find("["):cleaned.rfind("]") + 1]
                parsed = json.loads(json_str)
                if isinstance(parsed, list):
                    return {
                        "recommendations": [p for p in parsed if isinstance(p, dict)],
                        "error": None,
                    }

            # Case 2: wrapped object, e.g. {"recommendations": [...]}
            if "{" in cleaned and "}" in cleaned:
                json_obj = cleaned[cleaned.find("{"):cleaned.rfind("}") + 1]
                parsed_obj = json.loads(json_obj)
                if isinstance(parsed_obj, dict):
                    recommendations = parsed_obj.get("recommendations", [])
                    if isinstance(recommendations, list):
                        return {
                            "recommendations": [p for p in recommendations if isinstance(p, dict)],
                            "error": None,
                        }

            return {"recommendations": [], "error": "No valid JSON recommendations found"}
        except (json.JSONDecodeError, ValueError):
            return {"recommendations": [], "error": f"Parse failed: {response[:100]}"}


# ============================================================================
# Tool Registry
# ============================================================================

class ToolRegistry:
    """Central registry of all available tools."""

    def __init__(self, ollama_client):
        self.github_tool = GitHubDataTool()
        self.feature_tool = FeatureExtractionTool()
        self.vector_tool = VectorStoreTool()
        self.llm_tool = LLMInferenceTool(ollama_client)

    def get_tool(self, tool_name: str) -> Any:
        """Get tool by name."""
        tools = {
            "github": self.github_tool,
            "features": self.feature_tool,
            "vectors": self.vector_tool,
            "llm": self.llm_tool,
        }
        return tools.get(tool_name)
