"""Run-level metrics logging and state normalization utilities."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _stable_serialize(obj: Any) -> str:
    """Serialize objects deterministically for deduplication keys."""
    try:
        if hasattr(obj, "dict"):
            obj = obj.dict()
        return json.dumps(obj, sort_keys=True, default=str)
    except Exception:
        return str(obj)


def deduplicate_state_collections(state: Any) -> Any:
    """Remove duplicate reducer-accumulated entries to avoid inflated metrics."""
    for attr_name in ["execution_logs", "errors"]:
        values = list(getattr(state, attr_name, []) or [])
        seen: set[str] = set()
        unique: list[str] = []
        for value in values:
            key = str(value)
            if key in seen:
                continue
            seen.add(key)
            unique.append(value)
        setattr(state, attr_name, unique)

    for attr_name in ["identified_risks", "recommendations"]:
        values = list(getattr(state, attr_name, []) or [])
        seen_obj: set[str] = set()
        unique_obj: list[Any] = []
        for value in values:
            key = _stable_serialize(value)
            if key in seen_obj:
                continue
            seen_obj.add(key)
            unique_obj.append(value)
        setattr(state, attr_name, unique_obj)

    return state


def build_run_metrics(state: Any, started_at: float, finished_at: float) -> dict[str, Any]:
    """Build standardized per-run research metrics."""
    stage_sources = {
        "analysis": getattr(state, "analysis_source", None),
        "risk": getattr(state, "risk_source", None),
        "recommendation": getattr(state, "recommendation_source", None),
    }

    valid_sources = [s for s in stage_sources.values() if s is not None]
    parse_success_count = sum(1 for s in valid_sources if s in {"llm", "fallback", "rule"})
    parse_success_rate = (
        parse_success_count / len(valid_sources) if valid_sources else None
    )

    fallback_count = sum(1 for s in valid_sources if s == "fallback")
    fallback_rate = fallback_count / len(valid_sources) if valid_sources else None

    citations = list(getattr(state, "evidence_citations", []) or [])
    non_empty_citations = [c for c in citations if isinstance(c, str) and c.strip()]
    citation_quality_score = (
        len(non_empty_citations) / max(1, len(citations)) if citations else 0.0
    )

    repo_id = (
        state.repositories[0]
        if getattr(state, "repositories", None)
        else "unknown"
    )
    tracker = getattr(state, "lora_performance_tracker", {}) or {}
    tracker_f1 = None
    if repo_id in tracker:
        tracker_f1 = tracker[repo_id].get("current_f1")
    elif getattr(state, "lora_metrics", None):
        tracker_f1 = state.lora_metrics.get("f1_after")

    # Per-sprint F1 is only meaningful when the POSTed payload carries real
    # delivery signals (issues, PRs, commits).  If those are missing we
    # return None + an explicit warning rather than a synthetic number, so
    # the UI can flag that the metric is not applicable instead of showing
    # a misleading score.
    f1_score: float | None = None
    f1_method: str = "unavailable"
    f1_warnings: list[str] = []
    try:
        feats = getattr(state, "features", {}) or {}
        activity = feats.get("activity", {}) or {}
        precision = float(activity.get("pr_merge_rate", 0.0) or 0.0)
        recall    = float(activity.get("issue_resolution_rate", 0.0) or 0.0)

        missing = []
        if "pr_merge_rate" not in activity:
            missing.append("pull_requests")
        if "issue_resolution_rate" not in activity:
            missing.append("issues")
        if "commit_frequency" not in activity:
            missing.append("commits")

        if (precision + recall) > 0:
            sprint_f1 = 2 * precision * recall / (precision + recall)
            f1_score = round(sprint_f1, 4)
            f1_method = "per_sprint_delivery_f1"
        elif not activity:
            f1_warnings.append(
                "F1 is not computable: the request did not include "
                "issues / pull_requests / commits. Use JSON Input mode "
                "with a sprint payload to get a real per-sprint F1."
            )
        else:
            f1_warnings.append(
                "F1 is 0 because this sprint has no closed issues and "
                "no merged pull requests yet. It will update once "
                "delivery activity is recorded."
            )
            f1_score = 0.0
            f1_method = "per_sprint_delivery_f1"

        if missing and activity:
            f1_warnings.append(
                "Missing signals in payload: " + ", ".join(missing)
                + ". F1 reflects only the fields that were provided."
            )
    except Exception as exc:
        f1_warnings.append(f"F1 computation failed: {exc!s}")
        f1_score = None
        f1_method = "error"

    return {
        "timestamp": datetime.now().isoformat(),
        "latency_seconds": round(float(finished_at - started_at), 4),
        "f1_score": f1_score,
        "f1_method": f1_method,
        "f1_warnings": f1_warnings,
        "parse_success_rate": parse_success_rate,
        "fallback_rate": fallback_rate,
        "citation_quality": {
            "total_citations": len(citations),
            "non_empty_citations": len(non_empty_citations),
            "score": round(float(citation_quality_score), 4),
        },
        "source_breakdown": stage_sources,
        "counts": {
            "risks": len(getattr(state, "identified_risks", []) or []),
            "recommendations": len(getattr(state, "recommendations", []) or []),
            "errors": len(getattr(state, "errors", []) or []),
            "execution_logs": len(getattr(state, "execution_logs", []) or []),
        },
    }


def persist_run_metrics(metrics: dict[str, Any], output_dir: str = "artifacts/runs") -> str:
    """Persist run metrics to a JSON artifact and return file path."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"run_metrics_{stamp}.json"

    with out_file.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    return str(out_file)
