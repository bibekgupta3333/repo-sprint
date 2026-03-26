"""
Feature extraction from sprint data.

Computes 25 metrics per sprint for risk classification.
Grouped into: temporal (3), activity (9), code (6), risk (4), team (2),
plus a ``language_breakdown`` dict.

The ``SprintMetrics`` TypedDict is the canonical schema shared by:
  - ``SprintPreprocessor``  (real sprints from git data)
  - ``SyntheticSprintGenerator``  (synthetic training data)
  - ``FeatureExtractor``  (standalone metric computation)
  - ``RiskLabeler``  (binary risk classification)
"""

from typing import TypedDict


class SprintMetrics(TypedDict):
    """25 metrics per sprint (canonical schema).

    Temporal (3):
        days_span, issue_age_avg, pr_age_avg
    Activity (9):
        total_issues, total_prs, total_commits,
        closed_issues, merged_prs,
        issue_resolution_rate, pr_merge_rate,
        commit_frequency, code_changes
    Code (6):
        total_code_changes, avg_pr_size, code_concentration,
        total_additions, total_deletions, files_changed
    Risk (4):
        stalled_issues, unreviewed_prs, abandoned_prs,
        long_open_issues
    Team (2):
        unique_authors, author_participation
    Other:
        language_breakdown (dict)
    """
    # Temporal
    days_span: int
    issue_age_avg: float
    pr_age_avg: float
    # Activity
    total_issues: int
    total_prs: int
    total_commits: int
    closed_issues: int
    merged_prs: int
    issue_resolution_rate: float
    pr_merge_rate: float
    commit_frequency: float
    code_changes: int
    # Code
    total_code_changes: int
    avg_pr_size: int
    code_concentration: float
    total_additions: int
    total_deletions: int
    files_changed: int
    # Risk
    stalled_issues: int
    unreviewed_prs: int
    abandoned_prs: int
    long_open_issues: int
    # Team
    unique_authors: int
    author_participation: float
    # Other
    language_breakdown: dict


def _author_login(author: object) -> str:
    """Normalize GitHub user field (dict from API or string)."""
    if author is None:
        return "unknown"
    if isinstance(author, dict):
        return author.get("login") or "unknown"
    return str(author)


class FeatureExtractor:
    """Extract 25 metrics from sprint data.

    Can be used standalone (without ``SprintPreprocessor``) to
    compute the full ``SprintMetrics`` schema from raw sprint items.
    """

    def __init__(self, repo_data: dict, sprint: dict):
        self.repo = repo_data.get("name", "unknown")
        self.issues = sprint.get("issues", [])
        self.prs = sprint.get("prs", [])
        self.commits = sprint.get("commits", [])
        self.commit_diffs = sprint.get("commit_diffs", [])

    def extract_metrics(self) -> dict:
        """Compute all 25 metrics.

        Returns a dict conforming to the ``SprintMetrics`` schema.
        """
        result: dict = {}
        result.update(self._temporal_metrics())
        result.update(self._activity_metrics())
        result.update(self._code_metrics())
        result.update(self._risk_indicators())
        result.update(self._team_metrics())
        return result

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
        closed_issues = sum(
            1 for i in self.issues if i.get("state") == "closed"
        )
        merged_prs = sum(
            1 for p in self.prs
            if p.get("state") == "merged"
            or p.get("merged_at") is not None
        )
        code_changes = sum(
            p.get("additions", 0) + p.get("deletions", 0)
            for p in self.prs
        )

        return {
            "total_issues": total_issues,
            "total_prs": total_prs,
            "total_commits": len(self.commits),
            "closed_issues": closed_issues,
            "merged_prs": merged_prs,
            "issue_resolution_rate": (
                closed_issues / total_issues if total_issues > 0 else 0
            ),
            "pr_merge_rate": (
                merged_prs / total_prs if total_prs > 0 else 0
            ),
            "commit_frequency": len(self.commits) / 13.0,
            "code_changes": code_changes,
        }

    def _code_metrics(self) -> dict:
        """Code churn, PR size, concentration, language breakdown."""
        # PR-level aggregate
        total_changes = sum(
            p.get("additions", 0) + p.get("deletions", 0)
            for p in self.prs
        )
        avg_size = total_changes / len(self.prs) if self.prs else 0

        # Code concentration: % of changes in top 2 PRs
        if self.prs:
            changes_per_pr: list[int] = [
                int(p.get("additions") or 0) + int(p.get("deletions") or 0)
                for p in self.prs
            ]
            changes_per_pr.sort(reverse=True)
            top_2 = 0
            if len(changes_per_pr) > 0:
                top_2 += changes_per_pr[0]
            if len(changes_per_pr) > 1:
                top_2 += changes_per_pr[1]
                
            concentration = (
                top_2 / total_changes if total_changes > 0 else 0
            )
        else:
            concentration = 0

        # Commit-diff level churn
        total_additions = sum(
            d.get("total_additions", 0) for d in self.commit_diffs
        )
        total_deletions = sum(
            d.get("total_deletions", 0) for d in self.commit_diffs
        )
        files_changed = sum(
            d.get("files_changed", 0) for d in self.commit_diffs
        )

        # Language breakdown
        lang_breakdown: dict[str, int] = {}
        for diff in self.commit_diffs:
            for lang, count in diff.get(
                "language_breakdown", {},
            ).items():
                lang_breakdown[lang] = (
                    lang_breakdown.get(lang, 0) + count
                )

        return {
            "total_code_changes": total_changes,
            "avg_pr_size": int(avg_size),
            "code_concentration": min(concentration, 1.0),
            "total_additions": total_additions,
            "total_deletions": total_deletions,
            "files_changed": files_changed,
            "language_breakdown": lang_breakdown,
        }

    def _risk_indicators(self) -> dict:
        """Stalled, unreviewed, abandoned, long-open."""
        stalled = sum(
            1 for i in self.issues
            if i.get("state") == "open"
        )
        unreviewed = sum(
            1 for p in self.prs
            if p.get("state") == "open"
        )
        abandoned = sum(
            1 for p in self.prs
            if p.get("state") == "closed"
            and not p.get("merged")
        )
        long_open = sum(
            1 for i in self.issues
            if i.get("state") == "open"
        )

        n_issues = len(self.issues)
        n_prs = len(self.prs)
        return {
            "stalled_issues": (
                min(stalled, n_issues) if self.issues else 0
            ),
            "unreviewed_prs": (
                min(unreviewed, n_prs) if self.prs else 0
            ),
            "abandoned_prs": abandoned,
            "long_open_issues": (
                min(long_open, n_issues) if self.issues else 0
            ),
        }

    def _team_metrics(self) -> dict:
        """Unique authors, participation rate."""
        authors = set(
            _author_login(i.get("author"))
            for i in self.issues if i.get("author")
        )
        authors.update(
            _author_login(p.get("author"))
            for p in self.prs if p.get("author")
        )
        authors.update(
            _author_login(c.get("author"))
            for c in self.commits if c.get("author")
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

        # Low resolution rate (0.2 weight)
        if (
            metrics["total_issues"] > 0
            and metrics["issue_resolution_rate"] < 0.5
        ):
            risk_score += 0.2
            risk_factors.append("low_issue_resolution")

        # Low merge rate (0.2 weight)
        if (
            metrics["total_prs"] > 0
            and metrics["pr_merge_rate"] < 0.5
        ):
            risk_score += 0.2
            risk_factors.append("low_pr_merge_rate")

        # Long-open issues (0.15 weight)
        if metrics["long_open_issues"] >= 2:
            risk_score += 0.15
            risk_factors.append("long_open_issues")

        # No activity (0.1 weight)
        if (
            metrics["total_issues"] == 0
            and metrics["total_prs"] == 0
            and metrics["total_commits"] == 0
        ):
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
