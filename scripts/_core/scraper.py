"""Core scraper module."""
from datetime import datetime
from pathlib import Path
from typing import Optional, List
import json
import requests


def parse_author(user_data: dict) -> dict:
    """Extract author information."""
    if not user_data:
        return {'login': 'unknown', 'url': ''}
    return {
        'login': user_data.get('login', 'unknown'),
        'url': user_data.get('html_url', '')
    }


def parse_issue(issue_data: dict) -> dict:
    """Parse issue from GitHub API."""
    created = issue_data['created_at'].replace('Z', '+00:00')
    updated = issue_data['updated_at'].replace('Z', '+00:00')
    closed = issue_data.get('closed_at')
    closed_at = None
    if closed:
        closed_at = datetime.fromisoformat(closed.replace('Z', '+00:00'))

    body = issue_data.get('body')
    body_text = body[:500] if body else None

    return {
        'number': issue_data['number'],
        'title': issue_data['title'],
        'state': issue_data['state'],
        'created_at': issue_data['created_at'],
        'updated_at': issue_data['updated_at'],
        'closed_at': closed_at.isoformat() if closed_at else None,
        'author': parse_author(issue_data.get('user')),
        'url': issue_data['html_url'],
        'labels': [label['name'] for label in issue_data.get('labels', [])],
        'body': body_text
    }


def parse_pr(pr_data: dict) -> dict:
    """Parse pull request from GitHub API."""
    body = pr_data.get('body')
    body_text = body[:500] if body else None

    return {
        'number': pr_data['number'],
        'title': pr_data['title'],
        'state': pr_data['state'],
        'created_at': pr_data['created_at'],
        'updated_at': pr_data['updated_at'],
        'merged_at': pr_data.get('merged_at'),
        'closed_at': pr_data.get('closed_at'),
        'author': parse_author(pr_data.get('user')),
        'url': pr_data['html_url'],
        'labels': [label['name'] for label in pr_data.get('labels', [])],
        'body': body_text,
        'additions': pr_data.get('additions', 0),
        'deletions': pr_data.get('deletions', 0)
    }


def parse_commit(commit_data: dict) -> dict:
    """Parse commit from GitHub API."""
    commit_info = commit_data.get('commit', {})
    author_info = commit_info.get('author', {})

    return {
        'sha': commit_data.get('sha', '')[:7],
        'message': commit_info.get('message', '').split('\n')[0][:200],
        'author': parse_author(commit_data.get('author')),
        'created_at': author_info.get('date', datetime.now().isoformat()),
        'url': commit_data.get('html_url', '')
    }


class Scraper:
    """GitHub API scraper."""

    BASE_URL = "https://api.github.com"

    def __init__(self):
        """Initialize scraper."""
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'ResearchBot/1.0'
        })

    def get(self, url: str, params: dict = None):
        """Make GET request."""
        try:
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code in [404, 500]:
                return None
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"  Request failed: {e}")
            return None

    def get_issues(self, owner: str, repo: str, limit: int = 50) -> List[dict]:
        """Fetch issues."""
        issues = []
        page = 1

        while len(issues) < limit:
            url = f"{self.BASE_URL}/repos/{owner}/{repo}/issues"
            data = self.get(url, {
                'state': 'all',
                'per_page': 100,
                'page': page,
                'direction': 'desc'
            })

            if not data:
                break

            for item in data:
                if len(issues) >= limit:
                    break
                if 'pull_request' not in item:
                    try:
                        issues.append(parse_issue(item))
                    except Exception as e:
                        print(f"  Warning: Failed to parse issue: {e}")
                        continue

            if len(data) < 100:
                break
            page += 1

        return issues

    def get_prs(self, owner: str, repo: str, limit: int = 50) -> List[dict]:
        """Fetch pull requests."""
        prs = []
        page = 1

        while len(prs) < limit:
            url = f"{self.BASE_URL}/repos/{owner}/{repo}/pulls"
            data = self.get(url, {
                'state': 'all',
                'per_page': 100,
                'page': page
            })

            if not data:
                break

            for item in data:
                if len(prs) >= limit:
                    break
                try:
                    prs.append(parse_pr(item))
                except Exception as e:
                    print(f"  Warning: Failed to parse PR: {e}")
                    continue

            if len(data) < 100:
                break
            page += 1

        return prs

    def get_commits(self, owner: str, repo: str, limit: int = 50) -> List[dict]:
        """Fetch commits."""
        commits = []
        page = 1

        while len(commits) < limit:
            url = f"{self.BASE_URL}/repos/{owner}/{repo}/commits"
            data = self.get(url, {'per_page': 100, 'page': page})

            if not data:
                break

            for item in data:
                if len(commits) >= limit:
                    break
                try:
                    commits.append(parse_commit(item))
                except Exception as e:
                    print(f"  Warning: Failed to parse commit: {e}")
                    continue

            if len(data) < 100:
                break
            page += 1

        return commits

    def ingest(self, owner: str, repo: str,
               issues_limit: int = 50,
               prs_limit: int = 50,
               commits_limit: int = 50) -> Optional[dict]:
        """Download all data for a repository."""
        url = f"{self.BASE_URL}/repos/{owner}/{repo}"
        repo_data = self.get(url)

        if not repo_data:
            return None

        print(f"\nFetching {owner}/{repo}...")
        issues = self.get_issues(owner, repo, limit=issues_limit)
        print(f"  Issues: {len(issues)}")

        prs = self.get_prs(owner, repo, limit=prs_limit)
        print(f"  PRs: {len(prs)}")

        commits = self.get_commits(owner, repo, limit=commits_limit)
        print(f"  Commits: {len(commits)}")

        return {
            'owner': owner,
            'name': repo,
            'url': repo_data['html_url'],
            'description': repo_data.get('description'),
            'stars': repo_data['stargazers_count'],
            'forks': repo_data['forks_count'],
            'language': repo_data.get('language'),
            'issues': issues,
            'pull_requests': prs,
            'commits': commits
        }

    def save(self, repo_data: dict, output_dir: str = 'data/raw') -> str:
        """Save to JSON file."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        filename = f"{output_dir}/{repo_data['owner']}_{repo_data['name']}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(repo_data, f, indent=2)
        print(f"Saved to {filename}\n")
        return filename
