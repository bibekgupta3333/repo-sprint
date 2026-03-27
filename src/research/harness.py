"""Objective-level research harness for sprint intelligence."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from src.agents.state import OrchestratorState, GitHubIssue
from src.agents.tools import ToolRegistry
from src.agents.llm_config import get_ollama_client
from src.agents.dependency_graph_agent import DependencyGraphAgent
from src.agents.synthetic_data_generator_agent import SyntheticDataGeneratorAgent
from src.agents.lora_training_orchestrator_agent import LoRATrainingOrchestratorAgent
from src.agents.agents import ExplainerAgent
from src.agents.orchestrator import MasterOrchestrator


@dataclass
class ObjectiveResult:
    objective: str
    passed: bool
    claim_eligible: bool
    details: dict[str, Any]


def _discover_local_repo_candidates(limit: int = 2) -> list[str]:
    """Best-effort discovery of local repositories for objective realism checks."""
    roots = [Path("repos"), Path("Mintplex-Labs")]
    found: list[str] = []

    for root in roots:
        if not root.exists() or not root.is_dir():
            continue

        for path in root.rglob(".git"):
            repo_path = path.parent
            candidate = str(repo_path)
            if candidate not in found:
                found.append(candidate)
            if len(found) >= limit:
                return found

    return found


def _compute_wilson_ci(successes: int, total: int, z: float = 1.96) -> dict[str, float] | None:
    """Compute Wilson score confidence interval for a Bernoulli proportion."""
    if total <= 0:
        return None

    p = successes / total
    denominator = 1 + (z * z) / total
    center = (p + (z * z) / (2 * total)) / denominator
    margin = (z / denominator) * math.sqrt((p * (1 - p) / total) + ((z * z) / (4 * total * total)))
    return {
        "lower": max(0.0, center - margin),
        "upper": min(1.0, center + margin),
        "point_estimate": p,
    }


def _binomial_test_against_threshold(successes: int, total: int, threshold: float = 0.8) -> float | None:
    """One-sided binomial test p-value for H1: true_pass_rate < threshold."""
    if total <= 0:
        return None

    try:
        from scipy.stats import binomtest

        result = binomtest(k=successes, n=total, p=threshold, alternative="less")
        return float(result.pvalue)
    except Exception:
        return None


def _score_citations(citations: list[str], narrative: str | None) -> dict[str, Any]:
    """Simple citation validity and relevance scoring protocol."""
    valid_count = 0
    relevant_count = 0
    cleaned = [c.strip() for c in citations if isinstance(c, str) and c.strip()]
    text = (narrative or "").lower()

    for citation in cleaned:
        c_lower = citation.lower()
        valid_format = (
            "#" in citation
            or citation.startswith("http://")
            or citation.startswith("https://")
            or citation.startswith("issue")
            or citation.startswith("pr")
        )
        if valid_format:
            valid_count += 1

        tokens = [t for t in c_lower.replace("#", " ").replace("/", " ").split() if len(t) >= 2]
        if any(token in text for token in tokens):
            relevant_count += 1

    total = len(cleaned)
    validity_score = valid_count / total if total else 0.0
    relevance_score = relevant_count / total if total else 0.0

    return {
        "total": total,
        "valid_count": valid_count,
        "relevant_count": relevant_count,
        "validity_score": round(validity_score, 4),
        "relevance_score": round(relevance_score, 4),
        "composite_score": round((validity_score + relevance_score) / 2, 4),
    }


def _objective_cross_repo_detection(tool_registry: ToolRegistry) -> ObjectiveResult:
    local_repos = _discover_local_repo_candidates(limit=2)
    uses_local_repos = len(local_repos) >= 2

    if uses_local_repos:
        repos = local_repos[:2]
        # Extract org/repo format from local path for consistent issue reference
        # If path is 'repos/owner/name', format as 'owner/name'; if 'name-only', use default org
        repo_path = repos[1].split(os.sep)[-1]
        if '/' in repos[1]:
            repo_ref = repos[1].split(os.sep)[-2] + '/' + repos[1].split(os.sep)[-1]
        else:
            repo_ref = f"local/{repo_path}"
        issue_body = f"Blocked by {repo_ref}#42"
    else:
        repos = ["org/repo-a", "org/repo-b"]
        issue_body = "Waiting on org/repo-b#42"

    agent = DependencyGraphAgent(tool_registry)
    state = OrchestratorState(
        repositories=repos,
        github_issues=[
            GitHubIssue(
                number=10,
                title="Repo A blocked",
                body=issue_body,
                state="open",
                labels=["blocker"],
            )
        ],
    )
    state = agent.execute(state)
    edges = state.dependency_graph.get("edges", [])
    issue_edges = [e for e in edges if e.get("type") == "issue"]
    passed = len(issue_edges) > 0

    return ObjectiveResult(
        objective="objective_1_cross_repo_detection",
        passed=passed,
        claim_eligible=uses_local_repos,
        details={
            "edge_count": len(issue_edges),
            "uses_local_repos": uses_local_repos,
            "dataset_size": len(repos),
            "eval_mode": "strict",
            "is_simulated": False,
            "uses_stub": False,
        },
    )


def _objective_blocker_detection(tool_registry: ToolRegistry) -> ObjectiveResult:
    from src.agents.agents import RiskAssessorAgent

    class _StubLLMTool:
        def analyze_risks(
            self,
            features: dict[str, float],
            rag_context: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            return {
                "risks": [
                    {
                        "risk_type": "stalled_blockers",
                        "severity": 0.8,
                        "description": "Multiple blocked issues detected",
                        "affected_issues": [101, 102],
                    }
                ],
                "error": None,
            }

    class _StubToolRegistry:
        def __init__(self, real_registry: ToolRegistry):
            self.github_tool = real_registry.github_tool
            self.feature_tool = real_registry.feature_tool
            self.vector_tool = real_registry.vector_tool
            self.llm_tool = _StubLLMTool()

    stub_registry = _StubToolRegistry(tool_registry)
    agent = RiskAssessorAgent(stub_registry, get_ollama_client())

    state = OrchestratorState(
        repositories=["org/repo-a"],
        eval_mode="strict",
        features={"temporal": {"issue_resolution_rate": 0.2, "pr_merge_rate": 0.3}},
    )
    state = agent.execute(state)

    has_blocker = any(getattr(r, "risk_type", "") == "stalled_blockers" for r in state.identified_risks)
    return ObjectiveResult(
        objective="objective_2_blocker_detection",
        passed=has_blocker,
        claim_eligible=False,
        details={
            "risk_count": len(state.identified_risks),
            "risk_source": state.risk_source,
            "dataset_size": len(state.identified_risks),
            "eval_mode": "strict",
            "is_simulated": True,
            "uses_stub": True,
        },
    )


def _objective_cold_start_synthetic(tool_registry: ToolRegistry) -> ObjectiveResult:
    agent = SyntheticDataGeneratorAgent(tool_registry)
    state = OrchestratorState(
        repositories=["org/repo-a"],
        processed_data=[
            {
                "metrics": {
                    "issues_opened": 12,
                    "prs_opened": 8,
                    "commits_count": 35,
                    "issue_resolution_rate": 0.66,
                    "pr_merge_rate": 0.75,
                    "code_changes": 1400,
                    "avg_pr_size": 120,
                    "code_concentration": 0.3,
                    "abandoned_prs": 1,
                    "long_open_issues": 2,
                    "unique_authors": 5,
                }
            }
        ],
    )
    state = agent.execute(state)

    validation_results = state.synthetic_validation.get("validation_results", [])
    validation_count = len(validation_results)
    passed = len(state.synthetic_sprints) > 0 and validation_count > 0

    return ObjectiveResult(
        objective="objective_3_cold_start_synthetic",
        passed=passed,
        claim_eligible=validation_count > 0,
        details={
            "synthetic_count": len(state.synthetic_sprints),
            "validation_count": validation_count,
            "realism_score": state.synthetic_validation.get("realism_score", 0.0),
            "dataset_size": len(state.processed_data),
            "eval_mode": "strict",
            "is_simulated": False,
            "uses_stub": False,
        },
    )


def _objective_lora_drift_adaptation(tool_registry: ToolRegistry) -> ObjectiveResult:
    agent = LoRATrainingOrchestratorAgent(tool_registry)
    repo = "org/repo-a"
    state = OrchestratorState(
        repositories=[repo],
        sprint_analysis={"confidence_score": 0.5},
        lora_performance_tracker={
            repo: {
                "project_id": repo,
                "baseline_f1": 0.8,
                "current_f1": 0.79,
                "drift_detected": False,
                "drift_magnitude": 0.01,
                "last_updated": datetime.now().isoformat(),
                "num_observations": 3,
            }
        },
        synthetic_sprints=[
            {
                "sprint_id": f"syn_{i}",
                "metrics": {"issue_resolution_rate": 0.3, "pr_merge_rate": 0.4},
                "risk_label": {"is_at_risk": True},
            }
            for i in range(10)
        ],
    )
    state = agent.execute(state)
    passed = bool(state.lora_training_triggered)

    return ObjectiveResult(
        objective="objective_4_lora_drift_adaptation",
        passed=passed,
        claim_eligible=False,
        details={
            "training_triggered": state.lora_training_triggered,
            "lora_metrics": state.lora_metrics,
            "dataset_size": len(state.synthetic_sprints),
            "eval_mode": "strict",
            "is_simulated": True,
            "uses_stub": False,
        },
    )


def _objective_explainability_quality() -> ObjectiveResult:
    agent = ExplainerAgent()
    state = OrchestratorState(
        repositories=["org/repo-a"],
        sprint_analysis={"completion_probability": 74, "health_status": "at_risk"},
        evidence_citations=["issue#1", "pr#5", ""],
    )
    state = agent.execute(state)

    narrative_ok = bool(state.narrative_explanation and len(state.narrative_explanation) > 0)
    citation_scores = _score_citations(state.evidence_citations, state.narrative_explanation)
    passed = narrative_ok and citation_scores["composite_score"] >= 0.5

    return ObjectiveResult(
        objective="objective_5_explainability_quality",
        passed=passed,
        claim_eligible=False,
        details={
            "narrative_generated": narrative_ok,
            "citation_scores": citation_scores,
            "dataset_size": citation_scores["total"],
            "eval_mode": "strict",
            "is_simulated": True,
            "uses_stub": False,
        },
    )


def _run_strict_pipeline_benchmark() -> dict[str, Any]:
    """Run a strict orchestrator pass to support fallback-rate acceptance gate."""
    now = datetime.now()
    issues = [
        GitHubIssue(
            number=1,
            title="Auth bug",
            body="OAuth flow broken",
            state="open",
            labels=["bug", "blocker"],
            created_at=(now - timedelta(days=2)).isoformat(),
        )
    ]

    state = OrchestratorState(
        repositories=["Mintplex-Labs/anything-llm"],
        eval_mode="strict",
        github_issues=issues,
        milestone_data={
            "created_at": (now - timedelta(days=7)).isoformat(),
            "due_on": now.isoformat(),
        },
    )

    orchestrator = MasterOrchestrator()
    result = orchestrator.invoke(state)
    sources = {
        "analysis": result.analysis_source,
        "risk": result.risk_source,
        "recommendation": result.recommendation_source,
    }
    valid_sources = [v for v in sources.values() if v is not None]
    fallback_rate = (
        sum(1 for v in valid_sources if v == "fallback") / len(valid_sources)
        if valid_sources
        else 1.0
    )

    return {
        "eval_mode": result.eval_mode,
        "source_breakdown": sources,
        "fallback_rate": round(float(fallback_rate), 4),
        "errors": list(result.errors or []),
        "run_metrics_artifact": result.run_metrics_artifact,
    }


def _evaluate_acceptance_gates(
    results: list[ObjectiveResult],
    strict_pipeline: dict[str, Any],
    include_statistics: bool,
) -> list[dict[str, Any]]:
    """Evaluate acceptance gates before enabling research claims."""
    by_name = {r.objective: r for r in results}

    non_empty_objective_metrics = all((r.details.get("dataset_size", 0) or 0) > 0 for r in results)
    cross_repo_real_local = by_name["objective_1_cross_repo_detection"].details.get("uses_local_repos", False)
    synthetic_non_empty_validation = by_name["objective_3_cold_start_synthetic"].details.get("validation_count", 0) > 0
    fallback_rate_under_10 = strict_pipeline.get("fallback_rate", 1.0) < 0.10

    gates = [
        {
            "name": "strict_pipeline_fallback_rate_under_10_percent",
            "passed": fallback_rate_under_10,
            "value": strict_pipeline.get("fallback_rate"),
            "threshold": 0.10,
        },
        {
            "name": "non_empty_objective_datasets_all_5",
            "passed": non_empty_objective_metrics,
        },
        {
            "name": "cross_repo_benchmark_uses_local_repos",
            "passed": cross_repo_real_local,
        },
        {
            "name": "synthetic_validation_has_non_zero_metrics",
            "passed": synthetic_non_empty_validation,
            "value": by_name["objective_3_cold_start_synthetic"].details.get("validation_count", 0),
        },
        {
            "name": "statistical_summary_present",
            "passed": include_statistics,
        },
    ]

    return gates


def _build_claim_report(
    results: list[ObjectiveResult],
    acceptance_gates: list[dict[str, Any]],
    stats: dict[str, Any],
) -> dict[str, Any]:
    """Build strict-only claim report from claim-eligible objectives."""
    eligible = [r for r in results if r.claim_eligible]
    excluded = [
        {
            "objective": r.objective,
            "reason": "simulated_or_stubbed_or_non_local",
        }
        for r in results
        if not r.claim_eligible
    ]

    passed_eligible = sum(1 for r in eligible if r.passed)
    claim_ready = bool(eligible) and all(g["passed"] for g in acceptance_gates) and passed_eligible == len(eligible)

    return {
        "mode": "strict",
        "claim_ready": claim_ready,
        "eligible_objectives": [asdict(r) for r in eligible],
        "excluded_objectives": excluded,
        "acceptance_gates": acceptance_gates,
        "statistics": stats,
    }


def run_research_harness(
    output_path: str = "artifacts/research/research_harness.json",
    claim_output_path: str = "artifacts/research/research_claim_report.json",
) -> dict[str, Any]:
    """Run objective-level harness and persist JSON artifacts."""
    tool_registry = ToolRegistry(get_ollama_client())

    results = [
        _objective_cross_repo_detection(tool_registry),
        _objective_blocker_detection(tool_registry),
        _objective_cold_start_synthetic(tool_registry),
        _objective_lora_drift_adaptation(tool_registry),
        _objective_explainability_quality(),
    ]

    strict_pipeline = _run_strict_pipeline_benchmark()
    passed_count = sum(1 for r in results if r.passed)
    total = len(results)

    ci = _compute_wilson_ci(passed_count, total)
    pvalue = _binomial_test_against_threshold(passed_count, total, threshold=0.8)
    stats = {
        "objective_pass_rate": passed_count / total if total else 0.0,
        "objective_pass_rate_wilson_ci_95": ci,
        "binomial_test_pvalue_vs_0_8": pvalue,
    }

    acceptance_gates = _evaluate_acceptance_gates(results, strict_pipeline, include_statistics=ci is not None)
    claim_report = _build_claim_report(results, acceptance_gates, stats)

    payload = {
        "timestamp": datetime.now().isoformat(),
        "mode": "strict",
        "status": "pass" if passed_count == total else "partial",
        "passed_objectives": passed_count,
        "total_objectives": total,
        "results": [asdict(r) for r in results],
        "strict_pipeline_benchmark": strict_pipeline,
        "statistics": stats,
        "acceptance_gates": acceptance_gates,
        "claim_ready": claim_report["claim_ready"],
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    claim_file = Path(claim_output_path)
    claim_file.parent.mkdir(parents=True, exist_ok=True)
    with claim_file.open("w", encoding="utf-8") as handle:
        json.dump(claim_report, handle, indent=2)

    payload["artifact_path"] = str(out_file)
    payload["claim_artifact_path"] = str(claim_file)
    return payload
