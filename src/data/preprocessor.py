"""Preprocess GitHub data into 2-week sprints."""
from datetime import datetime, timedelta
from typing import TypedDict
from collections import defaultdict


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

        sprints = []
        all_sprint_ids = set(issue_sprints.keys()) | set(pr_sprints.keys())
        all_sprint_ids |= set(commit_sprints.keys())
        for sprint_id in sorted(all_sprint_ids):
            issues = issue_sprints.get(sprint_id, [])
            prs = pr_sprints.get(sprint_id, [])
            commits = commit_sprints.get(sprint_id, [])

            if issues:
                start_date = self._get_date(issues[0])
            elif prs:
                start_date = self._get_date(prs[0])
            else:
                start_date = self._get_date(commits[0])
            end_date = start_date + timedelta(days=13)

            metrics = {
                "total_issues": len(issues),
                "total_prs": len(prs),
                "total_commits": len(commits),
                "closed_issues": len([i for i in issues if i["state"] == "closed"]),
                "merged_prs": len([p for p in prs if p["state"] == "merged"]),
                "code_changes": sum(p.get("additions", 0) + p.get("deletions", 0) for p in prs),
            }

            sprints.append({
                "sprint_id": sprint_id,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "repo": self.repo,
                "issues": issues,
                "prs": prs,
                "commits": commits,
                "metrics": metrics,
            })

        return sprints
