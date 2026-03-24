"""Simple GitHub scraper for single repository."""
import os
import re
from datetime import datetime
from typing import TypedDict, Optional
from collections import defaultdict

import requests


_ISSUE_LINK_PATTERN = re.compile(
    r"(?:close|closes|closed|fix|fixes|fixed|resolve|resolves|resolved)\s+#(\d+)",
    re.IGNORECASE,
)


def _link_issues_to_pull_requests(issues: list, prs: list) -> None:
    """Populate issue['related_prs'] from PR closing-keyword references."""
    for issue in issues:
        inum = issue["number"]
        related: list[int] = []
        for pr in prs:
            text = f"{pr.get('title', '')}\n{pr.get('body') or ''}"
            for match in _ISSUE_LINK_PATTERN.finditer(text):
                if int(match.group(1)) == inum:
                    related.append(pr["number"])
                    break
        issue["related_prs"] = sorted(set(related))


class IssueData(TypedDict):
    """GitHub issue data."""
    number: int
    title: str
    body: str
    state: str
    created_at: str
    updated_at: str
    labels: list


class PRData(TypedDict):
    """GitHub pull request data."""
    number: int
    title: str
    body: str
    state: str
    created_at: str
    updated_at: str
    additions: int
    deletions: int
    labels: list


class CommitData(TypedDict):
    """GitHub commit data."""
    sha: str
    message: str
    author: str
    created_at: str


class FileDiffData(TypedDict):
    """Per-file diff data."""
    filename: str
    status: str
    additions: int
    deletions: int
    changes: int
    patch: str


class CommitDiffData(TypedDict):
    """Detailed commit diff data."""
    sha: str
    message: str
    author: str
    created_at: str
    total_additions: int
    total_deletions: int
    files_changed: int
    file_diffs: list[FileDiffData]
    language_breakdown: dict


class RepoData(TypedDict):
    """GitHub repository data."""
    owner: str
    name: str
    url: str
    stars: int
    forks: int
    language: str
    description: str
    issues: list[IssueData]
    prs: list[PRData]
    commits: list[CommitData]
    commit_diffs: list[CommitDiffData]


class GitHubScraper:
    BASE_URL = "https://api.github.com"

    def __init__(self, per_page: int = 50):
        self.per_page = per_page
        self.headers = {"Accept": "application/vnd.github.v3+json"}
        token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        if token:
            self.headers["Authorization"] = f"token {token}"

    def _get(self, endpoint: str, params: dict = None) -> list:
        """Fetch paginated data from GitHub API."""
        url = f"{self.BASE_URL}{endpoint}"
        all_data = []
        page = 1

        while page <= 5:
            parms = params or {}
            parms["page"] = page
            parms["per_page"] = self.per_page

            resp = requests.get(url, headers=self.headers, params=parms, timeout=10)

            if resp.status_code == 422:
                break
            if resp.status_code != 200:
                resp.raise_for_status()

            data = resp.json()
            if not data:
                break

            all_data.extend(data)
            page += 1

        return all_data

    def _parse_issue(self, issue: dict) -> IssueData:
        """Parse GitHub issue."""
        return {
            "number": issue["number"],
            "title": issue["title"],
            "body": issue["body"] or "",
            "state": issue["state"],
            "created_at": issue["created_at"],
            "updated_at": issue["updated_at"],
            "labels": [l["name"] for l in issue.get("labels", [])],
            "related_prs": [],
        }

    def _parse_pr(self, pr: dict) -> PRData:
        """Parse GitHub pull request."""
        return {
            "number": pr["number"],
            "title": pr["title"],
            "body": pr["body"] or "",
            "state": pr["state"],
            "created_at": pr["created_at"],
            "updated_at": pr["updated_at"],
            "additions": pr.get("additions", 0),
            "deletions": pr.get("deletions", 0),
            "labels": [l["name"] for l in pr.get("labels", [])],
            "url": pr.get("html_url", ""),
            "commits": [],
            "file_diffs": [],
        }

    def _parse_commit(self, commit: dict) -> CommitData:
        """Parse GitHub commit."""
        full_sha = commit.get("sha", "") or ""
        return {
            "sha": full_sha[:7] if full_sha else "",
            "sha_full": full_sha,
            "message": commit["commit"]["message"].split("\n")[0],
            "author": commit["commit"]["author"]["name"],
            "created_at": commit["commit"]["author"]["date"],
            "url": commit.get("html_url", ""),
        }

    def _extract_language_from_filename(self, filename: str) -> str:
        """Extract language from filename extension."""
        ext_to_lang = {
            '.py': 'Python', '.js': 'JavaScript', '.ts': 'TypeScript',
            '.jsx': 'JSX', '.tsx': 'TSX', '.go': 'Go', '.java': 'Java',
            '.cpp': 'C++', '.c': 'C', '.h': 'C Header', '.rs': 'Rust',
            '.rb': 'Ruby', '.php': 'PHP', '.cs': 'C#', '.swift': 'Swift',
            '.kt': 'Kotlin', '.scala': 'Scala', '.r': 'R', '.sql': 'SQL',
            '.sh': 'Shell', '.json': 'JSON', '.yaml': 'YAML', '.yml': 'YAML',
            '.xml': 'XML', '.html': 'HTML', '.css': 'CSS', '.scss': 'SCSS',
            '.md': 'Markdown', '.txt': 'Text', '.vim': 'Vim'
        }
        file_ext = '.' + filename.split('.')[-1].lower() if '.' in filename else ''
        return ext_to_lang.get(file_ext, 'Other')

    def get_commit_diff(
        self, owner: str, repo: str, commit_sha: str
    ) -> Optional[CommitDiffData]:
        """Fetch detailed diff for a single commit."""
        if not (commit_sha or "").strip():
            return None
        try:
            url = f"{self.BASE_URL}/repos/{owner}/{repo}/commits/{commit_sha}"
            resp = requests.get(url, headers=self.headers, timeout=30)
            resp.raise_for_status()
            commit = resp.json()

            file_diffs = []
            language_breakdown = defaultdict(int)
            total_additions = 0
            total_deletions = 0

            for file_info in commit.get("files", []):
                lang = self._extract_language_from_filename(file_info["filename"])
                additions = file_info.get("additions", 0)
                deletions = file_info.get("deletions", 0)

                language_breakdown[lang] += additions + deletions
                total_additions += additions
                total_deletions += deletions

                file_diffs.append({
                    "filename": file_info["filename"],
                    "status": file_info.get("status", "modified"),
                    "additions": additions,
                    "deletions": deletions,
                    "changes": file_info.get("changes", additions + deletions),
                    "patch": file_info.get("patch", "")
                })

            return {
                "sha": commit["sha"][:7],
                "message": commit["commit"]["message"].split("\n")[0],
                "author": commit["commit"]["author"]["name"],
                "created_at": commit["commit"]["author"]["date"],
                "total_additions": total_additions,
                "total_deletions": total_deletions,
                "files_changed": len(file_diffs),
                "file_diffs": file_diffs,
                "language_breakdown": dict(language_breakdown)
            }
        except Exception as e:
            print(f"  Warning: Could not fetch diff for {commit_sha}: {e}")
            return None

    def scrape(
        self,
        owner: str,
        repo: str,
        fetch_diffs: bool = False,
        pr_diff_limit: int = 20,
    ) -> RepoData:
        """Scrape single repository."""
        print(f"Scraping {owner}/{repo}...")

        try:
            repo_info = requests.get(
                f"{self.BASE_URL}/repos/{owner}/{repo}",
                headers=self.headers,
                timeout=10,
            ).json()
        except Exception as e:
            print(f"Error fetching repo info: {e}")
            return None

        print("  Fetching issues...")
        try:
            raw_issues = self._get(
                f"/repos/{owner}/{repo}/issues",
                {"state": "all", "pullRequest": False}
            )
            issues = [self._parse_issue(i) for i in raw_issues if "pull_request" not in i]
        except Exception as e:
            print(f"  Warning: Could not fetch issues: {e}")
            issues = []

        print("  Fetching PRs...")
        try:
            raw_prs = self._get(
                f"/repos/{owner}/{repo}/pulls",
                {"state": "all"}
            )
            prs = [self._parse_pr(p) for p in raw_prs]
        except Exception as e:
            print(f"  Warning: Could not fetch PRs: {e}")
            prs = []

        _link_issues_to_pull_requests(issues, prs)

        print("  Fetching commits...")
        try:
            raw_commits = self._get(f"/repos/{owner}/{repo}/commits")
            commits = [self._parse_commit(c) for c in raw_commits]
        except Exception as e:
            print(f"  Warning: Could not fetch commits: {e}")
            commits = []

        commit_diffs: list = []
        if fetch_diffs:
            if pr_diff_limit <= 0:
                for pr in prs:
                    pr.setdefault("commits", [])
                    pr.setdefault("file_diffs", [])
            else:
                print(
                    f"  Fetching PR commits and file lists (up to {pr_diff_limit} PRs)..."
                )
                head_prs = prs[:pr_diff_limit]
                tail_prs = prs[pr_diff_limit:]
                for pr in head_prs:
                    raw_pc = self._get(
                        f"/repos/{owner}/{repo}/pulls/{pr['number']}/commits", {}
                    )
                    pr["commits"] = [self._parse_commit(c) for c in raw_pc]
                    raw_files = self._get(
                        f"/repos/{owner}/{repo}/pulls/{pr['number']}/files", {}
                    )
                    pr["file_diffs"] = [
                        {
                            "filename": item.get("filename", ""),
                            "status": item.get("status", "modified"),
                            "additions": item.get("additions", 0),
                            "deletions": item.get("deletions", 0),
                            "changes": item.get(
                                "changes",
                                item.get("additions", 0) + item.get("deletions", 0),
                            ),
                            "patch": item.get("patch") or "",
                        }
                        for item in raw_files
                    ]
                for pr in tail_prs:
                    pr.setdefault("commits", [])
                    pr.setdefault("file_diffs", [])

            ordered_shas: list[str] = []
            seen_short: set[str] = set()

            def remember_sha(c: dict) -> None:
                full = (c.get("sha_full") or c.get("sha") or "").strip()
                if not full:
                    return
                short = full[:7]
                if short not in seen_short:
                    seen_short.add(short)
                    ordered_shas.append(full)

            for c in commits:
                remember_sha(c)
            for pr in prs:
                for c in pr.get("commits") or []:
                    remember_sha(c)

            print(
                f"  Fetching commit diffs for {len(ordered_shas)} unique commit(s)..."
            )
            diff_by_short: dict[str, dict] = {}
            for i, sha in enumerate(ordered_shas, 1):
                diff = self.get_commit_diff(owner, repo, sha)
                if diff:
                    short = diff["sha"][:7]
                    if short not in diff_by_short:
                        diff_by_short[short] = diff
                        commit_diffs.append(diff)
                if i % 10 == 0 or i == len(ordered_shas):
                    print(f"    Diffs fetched: {i}/{len(ordered_shas)}")

            for c in commits:
                c["diff"] = diff_by_short.get((c.get("sha") or "")[:7])
            for pr in prs:
                for c in pr.get("commits") or []:
                    c["diff"] = diff_by_short.get((c.get("sha") or "")[:7])
        else:
            for c in commits:
                c["diff"] = None
            for pr in prs:
                for c in pr.get("commits") or []:
                    c["diff"] = None

        return {
            "owner": owner,
            "name": repo,
            "url": repo_info.get("html_url", ""),
            "stars": repo_info.get("stargazers_count", 0),
            "forks": repo_info.get("forks_count", 0),
            "language": repo_info.get("language", ""),
            "description": repo_info.get("description", ""),
            "issues": issues,
            "prs": prs,
            "commits": commits,
            "commit_diffs": commit_diffs,
        }
