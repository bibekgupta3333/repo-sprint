"""Simple GitHub scraper for single repository."""
from datetime import datetime
from typing import TypedDict

import requests


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


class GitHubScraper:
    BASE_URL = "https://api.github.com"

    def __init__(self, per_page: int = 50):
        self.per_page = per_page
        self.headers = {"Accept": "application/vnd.github.v3+json"}

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
        }

    def _parse_commit(self, commit: dict) -> CommitData:
        """Parse GitHub commit."""
        return {
            "sha": commit["sha"][:7],
            "message": commit["commit"]["message"].split("\n")[0],
            "author": commit["commit"]["author"]["name"],
            "created_at": commit["commit"]["author"]["date"],
        }

    def scrape(self, owner: str, repo: str) -> RepoData:
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

        print("  Fetching commits...")
        try:
            raw_commits = self._get(f"/repos/{owner}/{repo}/commits")
            commits = [self._parse_commit(c) for c in raw_commits]
        except Exception as e:
            print(f"  Warning: Could not fetch commits: {e}")
            commits = []

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
        }
