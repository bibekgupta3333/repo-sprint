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

    def _build_flat_documents(self) -> list[ChromaDocument]:
        """Build flat, chunk-style documents used by vector stores."""
        docs = []

        sprint_id = self.sprint["sprint_id"]
        repo = self.sprint["repo"]
        metrics = self.sprint["metrics"]

        total_additions = metrics.get("total_additions", 0)
        total_deletions = metrics.get("total_deletions", 0)
        files_changed = metrics.get("files_changed", 0)

        sprint_context = (
            f"Sprint: {sprint_id}\n"
            f"Repository: {repo}\n"
            f"Period: {self.sprint['start_date']} to {self.sprint['end_date']}\n"
            f"Issues: {metrics['total_issues']} (closed: {metrics['closed_issues']})\n"
            f"PRs: {metrics['total_prs']} (merged: {metrics['merged_prs']})\n"
            f"Commits: {metrics['total_commits']}\n"
            f"Code Changes: +{total_additions} -{total_deletions} across {files_changed} files"
        )

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
            raw_author = commit.get("author")
            if isinstance(raw_author, dict):
                author_label = raw_author.get("login") or raw_author.get("url") or "unknown"
            else:
                author_label = raw_author or "unknown"

            diff = commit.get("diff") or {}
            has_diff = bool(diff)
            diff_additions = diff.get("total_additions", 0)
            diff_deletions = diff.get("total_deletions", 0)
            diff_files_changed = diff.get("files_changed", 0)

            file_summaries = []
            for file_diff in (diff.get("file_diffs") or [])[:10]:
                file_summaries.append(
                    f"- {file_diff.get('filename', '')} [{file_diff.get('status', 'modified')}] "
                    f"(+{file_diff.get('additions', 0)} -{file_diff.get('deletions', 0)})"
                )

            file_summary_text = "\n".join(file_summaries) if file_summaries else "- none"

            commit_content = (
                f"Commit {commit['sha']} by {author_label}\n\n"
                f"{commit['message']}\n\n"
                f"Code Changes: +{diff_additions} -{diff_deletions} across {diff_files_changed} files\n"
                f"Files:\n{file_summary_text}"
            )

            docs.append({
                "id": f"{sprint_id}_commit_{commit['sha']}",
                "content": commit_content,
                "metadata": {
                    "sprint_id": sprint_id,
                    "repo": repo,
                    "type": "commit",
                    "sha": commit["sha"],
                    "author": author_label,
                    "date": commit["created_at"],
                    "has_diff": has_diff,
                    "total_additions": diff_additions,
                    "total_deletions": diff_deletions,
                    "files_changed": diff_files_changed,
                    "file_diffs": diff.get("file_diffs", []),
                },
            })

        return docs

    def format_documents(self) -> list[ChromaDocument]:
        """Format sprint data as Chroma flat documents."""
        return self._build_flat_documents()

    def format_documents_by_sprint(self) -> dict:
        """Format sprint data in a sprint-aligned document structure."""
        flat_docs = self._build_flat_documents()

        summary = next((d for d in flat_docs if d["metadata"].get("type") == "sprint_summary"), None)
        issues = [d for d in flat_docs if d["metadata"].get("type") == "issue"]
        prs = [d for d in flat_docs if d["metadata"].get("type") == "pr"]
        commits = [d for d in flat_docs if d["metadata"].get("type") == "commit"]

        return {
            "sprint_id": self.sprint["sprint_id"],
            "repo": self.sprint["repo"],
            "start_date": self.sprint["start_date"],
            "end_date": self.sprint["end_date"],
            "summary": summary,
            "issues": issues,
            "prs": prs,
            "commits": commits,
        }
