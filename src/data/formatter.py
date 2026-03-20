"""Format sprint data for Chroma ingestion."""
from typing import TypedDict


class ChromaDocument(TypedDict):
    """Chroma vector store document."""
    id: str
    content: str
    metadata: dict


class ChromaFormatter:
    """Format sprint data for Chroma ingestion."""
    def __init__(self, sprint_data: dict):
        self.sprint = sprint_data

    def format_documents(self) -> list[ChromaDocument]:
        """Format sprint data as Chroma documents."""
        docs = []

        sprint_id = self.sprint["sprint_id"]
        repo = self.sprint["repo"]
        metrics = self.sprint["metrics"]

        sprint_context = f"""
Sprint: {sprint_id}
Repository: {repo}
Period: {self.sprint['start_date']} to {self.sprint['end_date']}
Issues: {metrics['total_issues']} (closed: {metrics['closed_issues']})
PRs: {metrics['total_prs']} (merged: {metrics['merged_prs']})
Commits: {metrics['total_commits']}
Code Changes: {metrics['code_changes']} (additions + deletions)
"""

        docs.append({
            "id": f"{sprint_id}_summary",
            "content": sprint_context.strip(),
            "metadata": {
                "sprint_id": sprint_id,
                "repo": repo,
                "type": "sprint_summary",
                "date": self.sprint["start_date"],
                "risk_score": self.sprint.get("risk_label", {}).get("risk_score", 0),
                "is_at_risk": self.sprint.get("risk_label", {}).get("is_at_risk", False),
                **metrics,
            },
        })

        for issue in self.sprint["issues"]:
            docs.append({
                "id": f"{sprint_id}_issue_{issue['number']}",
                "content": f"Issue #{issue['number']}: {issue['title']}\n\nState: {issue['state']}\nLabels: {', '.join(issue['labels']) or 'none'}\n\n{issue['body']}",
                "metadata": {
                    "sprint_id": sprint_id,
                    "repo": repo,
                    "type": "issue",
                    "issue_number": issue["number"],
                    "state": issue["state"],
                    "labels": issue["labels"],
                    "date": issue["created_at"],
                },
            })

        for pr in self.sprint["prs"]:
            docs.append({
                "id": f"{sprint_id}_pr_{pr['number']}",
                "content": f"PR #{pr['number']}: {pr['title']}\n\nState: {pr['state']}\nChanges: +{pr['additions']} -{pr['deletions']}\n\n{pr['body']}",
                "metadata": {
                    "sprint_id": sprint_id,
                    "repo": repo,
                    "type": "pr",
                    "pr_number": pr["number"],
                    "state": pr["state"],
                    "additions": pr["additions"],
                    "deletions": pr["deletions"],
                    "date": pr["created_at"],
                },
            })

        for commit in self.sprint["commits"]:
            docs.append({
                "id": f"{sprint_id}_commit_{commit['sha']}",
                "content": (
                    f"Commit {commit['sha']} by {commit['author']}\n\n"
                    f"{commit['message']}"
                ),
                "metadata": {
                    "sprint_id": sprint_id,
                    "repo": repo,
                    "type": "commit",
                    "sha": commit["sha"],
                    "author": commit["author"],
                    "date": commit["created_at"],
                },
            })

        return docs
