"""Preprocess GitHub data into 2-week sprints."""
from datetime import datetime, timedelta
from typing import TypedDict
from collections import defaultdict
from .features import FeatureExtractor, RiskLabeler


class SprintData(TypedDict):
    """Sprint data structure."""
    sprint_id: str
    start_date: str
    end_date: str
    repo: str
    issues: list
    prs: list
    commits: list
    metrics: dict


class SprintPreprocessor:
    """Preprocess repository data into 2-week sprints."""
    def __init__(self, repo_data: dict):
        self.repo = f"{repo_data['owner']}/{repo_data['name']}"
        self.issues = repo_data["issues"]
        self.prs = repo_data["prs"]
        self.commits = repo_data["commits"]
        self.commit_diffs = repo_data.get("commit_diffs", [])

    def _get_date(self, item: dict, date_field: str = "created_at") -> datetime:
        """Extract date from item."""
        date_str = item.get(date_field, item.get("created_at", ""))
        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))

    def _group_by_sprint(self, items: list, sprint_mapping: dict) -> dict:
        """Group items into sprints."""
        sprints = defaultdict(list)
        for item in items:
            date = self._get_date(item)
            sprint_id = sprint_mapping.get(date.date())
            if sprint_id:
                sprints[sprint_id].append(item)
        return sprints

    def _group_commit_diffs_by_sprint(self, commit_diffs: list, sprint_mapping: dict) -> dict:
        """Group commit diffs into sprints by created_at date."""
        sprints = defaultdict(list)
        for diff in commit_diffs:
            date = self._get_date(diff)
            sprint_id = sprint_mapping.get(date.date())
            if sprint_id:
                sprints[sprint_id].append(diff)
        return sprints

    def create_sprints(self) -> list[SprintData]:
        """Create synthetic 2-week sprints."""
        if not self.issues and not self.prs and not self.commits:
            return []

        all_dates = set()
        for item in self.issues + self.prs + self.commits:
            all_dates.add(self._get_date(item).date())

        if not all_dates:
            return []

        min_date = min(all_dates)
        max_date = max(all_dates)

        sprint_mapping = {}
        current_sprint_date = min_date
        sprint_num = 0

        while current_sprint_date <= max_date:
            sprint_end = current_sprint_date + timedelta(days=13)
            sprint_id = f"sprint_{sprint_num:03d}"

            for date in all_dates:
                if current_sprint_date <= date <= sprint_end:
                    sprint_mapping[date] = sprint_id

            current_sprint_date = sprint_end + timedelta(days=1)
            sprint_num += 1

        issue_sprints = self._group_by_sprint(self.issues, sprint_mapping)
        pr_sprints = self._group_by_sprint(self.prs, sprint_mapping)
        commit_sprints = self._group_by_sprint(self.commits, sprint_mapping)
        commit_diff_sprints = self._group_commit_diffs_by_sprint(self.commit_diffs, sprint_mapping)

        # Map short SHA -> diff for commits that only appear on PRs
        sha_to_diff = {}
        for diff in self.commit_diffs:
            sha = diff.get("sha", "") or ""
            sha_to_diff[sha] = diff
            sha_to_diff[sha[:7]] = diff

        sprints = []
        all_sprint_ids = set(issue_sprints.keys()) | set(pr_sprints.keys())
        all_sprint_ids |= set(commit_sprints.keys()) | set(commit_diff_sprints.keys())
        for sprint_id in sorted(all_sprint_ids):
            issues = issue_sprints.get(sprint_id, [])
            prs = pr_sprints.get(sprint_id, [])
            commits = commit_sprints.get(sprint_id, [])
            commit_diffs = commit_diff_sprints.get(sprint_id, [])

            # Enrich branch commits with diff (prefer value from ingest; else lookup)
            for commit in commits:
                if commit.get("diff") is None:
                    key = (commit.get("sha") or "")[:7]
                    commit["diff"] = sha_to_diff.get(key) or sha_to_diff.get(
                        commit.get("sha", "")
                    )

            # Same for commits attached to PRs in this sprint
            for pr in prs:
                for pc in pr.get("commits") or []:
                    if pc.get("diff") is None:
                        pk = (pc.get("sha") or "")[:7]
                        pc["diff"] = sha_to_diff.get(pk) or sha_to_diff.get(
                            pc.get("sha", "")
                        )

            if issues:
                start_date = self._get_date(issues[0])
            elif prs:
                start_date = self._get_date(prs[0])
            elif commits:
                start_date = self._get_date(commits[0])
            else:
                start_date = self._get_date(commit_diffs[0])
            end_date = start_date + timedelta(days=13)

            # Calculate code diff metrics
            total_additions = sum(d.get("total_additions", 0) for d in commit_diffs)
            total_deletions = sum(d.get("total_deletions", 0) for d in commit_diffs)
            total_files_changed = sum(d.get("files_changed", 0) for d in commit_diffs)

            # Aggregate language breakdown
            lang_breakdown = {}
            for diff in commit_diffs:
                for lang, count in diff.get("language_breakdown", {}).items():
                    lang_breakdown[lang] = lang_breakdown.get(lang, 0) + count

            metrics = {
                "total_issues": len(issues),
                "total_prs": len(prs),
                "total_commits": len(commits),
                "closed_issues": len([i for i in issues if i["state"] == "closed"]),
                "merged_prs": len([
                    p for p in prs
                    if p.get("state") == "merged"
                    or p.get("merged_at") is not None
                ]),
                "code_changes": sum(p.get("additions", 0) + p.get("deletions", 0) for p in prs),
                "total_additions": total_additions,
                "total_deletions": total_deletions,
                "files_changed": total_files_changed,
                "language_breakdown": lang_breakdown,
            }

            # Create sprint dict for feature extraction
            sprint_dict = {
                "sprint_id": sprint_id,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "repo": self.repo,
                "issues": issues,
                "prs": prs,
                "commits": commits,
                "commit_diffs": commit_diffs,
                "metrics": metrics,
            }

            # Extract 18 metrics and compute risk label
            try:
                repo_data = {
                    "owner": self.repo.split("/")[0],
                    "name": self.repo.split("/")[1],
                }
                extractor = FeatureExtractor(repo_data, sprint_dict)
                computed_metrics = extractor.extract_metrics()
                sprint_dict["metrics"].update(computed_metrics)
                risk_label = RiskLabeler.label_sprint(computed_metrics)
                sprint_dict["risk_label"] = risk_label
            except Exception:
                sprint_dict["risk_label"] = {
                    "risk_score": 0,
                    "is_at_risk": False,
                    "risk_factors": [],
                    "blocker_indicators": [],
                }

            sprints.append(sprint_dict)

        return sprints
