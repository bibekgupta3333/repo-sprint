"""Core analyzer module."""
import json
from pathlib import Path
from datetime import datetime
from typing import Dict


class Analyzer:
    """Analyze raw repository data."""

    def __init__(self, repo_file: str):
        """Load repository data."""
        with open(repo_file, encoding='utf-8') as f:
            self.data = json.load(f)

    def repo_info(self) -> Dict:
        """Get repository metadata."""
        return {
            'name': f"{self.data['owner']}/{self.data['name']}",
            'url': self.data['url'],
            'language': self.data.get('language'),
            'stars': self.data['stars'],
            'forks': self.data['forks'],
            'description': (self.data.get('description') or 'N/A')[:100]
        }

    def issue_stats(self) -> Dict:
        """Get issue statistics."""
        issues = self.data['issues']
        open_count = sum(1 for i in issues if i['state'] == 'open')
        closed_count = sum(1 for i in issues if i['state'] == 'closed')

        return {
            'total': len(issues),
            'open': open_count,
            'closed': closed_count
        }

    def pr_stats(self) -> Dict:
        """Get PR statistics."""
        prs = self.data.get('prs', self.data.get('pull_requests', []))
        merged_count = sum(1 for p in prs if p['state'] == 'merged')
        open_count = sum(1 for p in prs if p['state'] == 'open')
        closed_count = sum(1 for p in prs if p['state'] == 'closed')

        total_additions = sum(p.get('additions', 0) for p in prs)
        total_deletions = sum(p.get('deletions', 0) for p in prs)

        return {
            'total': len(prs),
            'merged': merged_count,
            'open': open_count,
            'closed': closed_count,
            'total_additions': total_additions,
            'total_deletions': total_deletions
        }

    def commit_stats(self) -> Dict:
        """Get commit statistics."""
        commits = self.data['commits']
        authors = set(c['author']['login'] for c in commits)
        return {
            'total': len(commits),
            'unique_authors': len(authors)
        }

    def analyze(self):
        """Print full analysis."""
        print(f"\n{'='*60}")
        print(f"Repository: {self.data['owner']}/{self.data['name']}")
        print(f"{'='*60}")

        repo = self.repo_info()
        for key, val in repo.items():
            print(f"{key:15}: {val}")

        print(f"\nIssues:")
        issues = self.issue_stats()
        for key, val in issues.items():
            print(f"  {key:15}: {val}")

        print(f"\nPull Requests:")
        prs = self.pr_stats()
        for key, val in prs.items():
            print(f"  {key:15}: {val}")

        print(f"\nCommits:")
        commits = self.commit_stats()
        for key, val in commits.items():
            print(f"  {key:15}: {val}")

        print(f"\n{'='*60}\n")
