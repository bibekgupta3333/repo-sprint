"""
Feature extraction from sprint data.
Computes 18 metrics for risk classification: temporal, activity, code, risk, team.
"""

from typing import TypedDict


class SprintMetrics(TypedDict):
    """18 computed metrics per sprint."""
    days_span: int
    issue_age_avg: float
    pr_age_avg: float
    total_issues: int
    total_prs: int
    total_commits: int
    issue_resolution_rate: float
    pr_merge_rate: float
    commit_frequency: float
    total_code_changes: int
    avg_pr_size: int
    code_concentration: float
    stalled_issues: int
    unreviewed_prs: int
    abandoned_prs: int
    long_open_issues: int
    unique_authors: int
    author_participation: float


class FeatureExtractor:
    """Extract 18 metrics from sprint data."""

    def __init__(self, repo_data: dict, sprint: dict):
        self.repo = repo_data.get("name", "unknown")
        self.issues = sprint.get("issues", [])
        self.prs = sprint.get("prs", [])
        self.commits = sprint.get("commits", [])

    def extract_metrics(self) -> SprintMetrics:
        """Compute all 18 metrics."""
        return {
            **self._temporal_metrics(),
            **self._activity_metrics(),
            **self._code_metrics(),
            **self._risk_indicators(),
            **self._team_metrics(),
        }

    def _temporal_metrics(self) -> dict:
        """Days span, issue age, PR age."""
        return {
            "days_span": 13,  # Always 2-week sprint
            "issue_age_avg": self._avg_age(self.issues) if self.issues else 0,
            "pr_age_avg": self._avg_age(self.prs) if self.prs else 0,
        }

    def _activity_metrics(self) -> dict:
        """Issue/PR/commit counts and resolution rates."""
        total_issues = len(self.issues)
        total_prs = len(self.prs)
        closed_issues = sum(1 for i in self.issues if i.get("state") == "closed")
        merged_prs = sum(1 for p in self.prs if p.get("state") == "merged")

        return {
            "total_issues": total_issues,
            "total_prs": total_prs,
            "total_commits": len(self.commits),
            "issue_resolution_rate": closed_issues / total_issues if total_issues > 0 else 0,
            "pr_merge_rate": merged_prs / total_prs if total_prs > 0 else 0,
            "commit_frequency": len(self.commits) / 13.0,
        }

    def _code_metrics(self) -> dict:
        """Code changes, PR size, concentration."""
        total_changes = sum(
            p.get("additions", 0) + p.get("deletions", 0)
            for p in self.prs
        )
        avg_size = total_changes / len(self.prs) if self.prs else 0

        # Code concentration: % of changes in top 2 PRs
        if self.prs:
            changes_per_pr = [
                p.get("additions", 0) + p.get("deletions", 0)
                for p in self.prs
            ]
            top_2_changes = sum(sorted(changes_per_pr, reverse=True)[:2])
            concentration = top_2_changes / total_changes if total_changes > 0 else 0
        else:
            concentration = 0

        return {
            "total_code_changes": total_changes,
            "avg_pr_size": int(avg_size),
            "code_concentration": min(concentration, 1.0),
        }

    def _risk_indicators(self) -> dict:
        """Stalled issues, unreviewed PRs, abandoned PRs, long-open issues."""
        stalled = sum(1 for i in self.issues if i.get("state") == "open")
        unreviewed = sum(1 for p in self.prs if p.get("state") == "open")
        abandoned = sum(1 for p in self.prs if p.get("state") == "closed" and not p.get("merged"))
        long_open = sum(1 for i in self.issues if i.get("state") == "open")

        return {
            "stalled_issues": min(stalled, len(self.issues)) if self.issues else 0,
            "unreviewed_prs": min(unreviewed, len(self.prs)) if self.prs else 0,
            "abandoned_prs": abandoned,
            "long_open_issues": min(long_open, len(self.issues)) if self.issues else 0,
        }

    def _team_metrics(self) -> dict:
        """Unique authors, participation rate."""
        authors = set(
            i.get("author", "unknown") for i in self.issues
            if i.get("author")
        )
        authors.update(
            p.get("author", "unknown") for p in self.prs
            if p.get("author")
        )
        authors.update(
            c.get("author", "unknown") for c in self.commits
            if c.get("author")
        )

        total_items = len(self.issues) + len(self.prs) + len(self.commits)
        unique_authors = len(authors)
        participation = unique_authors / total_items if total_items > 0 else 0

        return {
            "unique_authors": unique_authors,
            "author_participation": min(participation, 1.0),
        }

    @staticmethod
    def _avg_age(items: list) -> float:
        """Average days since creation."""
        if not items:
            return 0
        ages = [item.get("age", 0) for item in items if "age" in item]
        return sum(ages) / len(ages) if ages else 0


class RiskLabeler:
    """Label sprints as at-risk (binary classification)."""

    @staticmethod
    def label_sprint(metrics: SprintMetrics) -> dict:
        """Compute risk_score (0-1) and is_at_risk flag."""
        risk_score = 0.0
        risk_factors = []

        # Stalled issues (0.3 weight)
        if metrics["stalled_issues"] >= 3:
            risk_score += 0.3
            risk_factors.append("stalled_issues")

        # Low resolution rate (0.2 weight - only if issues exist)
        if metrics["total_issues"] > 0 and metrics["issue_resolution_rate"] < 0.5:
            risk_score += 0.2
            risk_factors.append("low_issue_resolution")

        # Low merge rate (0.2 weight - only if PRs exist)
        if metrics["total_prs"] > 0 and metrics["pr_merge_rate"] < 0.5:
            risk_score += 0.2
            risk_factors.append("low_pr_merge_rate")

        # Long-open issues (0.15 weight)
        if metrics["long_open_issues"] >= 2:
            risk_score += 0.15
            risk_factors.append("long_open_issues")

        # No activity (0.1 weight - only if truly empty)
        if (metrics["total_issues"] == 0 and
            metrics["total_prs"] == 0 and
            metrics["total_commits"] == 0):
            risk_score += 0.1
            risk_factors.append("no_activity")

        risk_score = min(risk_score, 1.0)

        return {
            "risk_score": risk_score,
            "is_at_risk": risk_score >= 0.4,
            "risk_factors": risk_factors,
            "blocker_indicators": [
                f"stalled: {metrics['stalled_issues']}"
                if metrics["stalled_issues"] > 0 else None,
                f"unreviewed: {metrics['unreviewed_prs']}"
                if metrics["unreviewed_prs"] > 0 else None,
            ],
        }
