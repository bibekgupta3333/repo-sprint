"""Core processor module."""
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List


class Processor:
    """Process raw GitHub data."""

    def __init__(self, repo_file: str):
        """Load raw repository data."""
        with open(repo_file, encoding='utf-8') as f:
            self.data = json.load(f)
        self.repo_name = f"{self.data['owner']}/{self.data['name']}"

    def extract_issues(self) -> List[Dict]:
        """Extract issue features."""
        features = []
        for issue in self.data['issues']:
            created = datetime.fromisoformat(
                issue['created_at'].replace('Z', '+00:00')
            )
            closed_at = None
            days_open = (datetime.now(created.tzinfo) - created).days

            if issue['closed_at']:
                closed_at = datetime.fromisoformat(
                    issue['closed_at'].replace('Z', '+00:00')
                )
                days_open = (closed_at - created).days

            features.append({
                'repo': self.repo_name,
                'type': 'issue',
                'number': issue['number'],
                'title': issue['title'],
                'state': issue['state'],
                'author': issue['author']['login'],
                'labels': issue['labels'],
                'created_at': issue['created_at'],
                'days_open': days_open,
                'has_body': bool(issue.get('body')),
                'url': issue['url']
            })

        return features

    def extract_prs(self) -> List[Dict]:
        """Extract PR features."""
        features = []
        for pr in self.data['pull_requests']:
            created = datetime.fromisoformat(
                pr['created_at'].replace('Z', '+00:00')
            )
            days_open = (datetime.now(created.tzinfo) - created).days

            if pr['merged_at']:
                merged_at = datetime.fromisoformat(
                    pr['merged_at'].replace('Z', '+00:00')
                )
                days_open = (merged_at - created).days

            features.append({
                'repo': self.repo_name,
                'type': 'pr',
                'number': pr['number'],
                'title': pr['title'],
                'state': pr['state'],
                'author': pr['author']['login'],
                'labels': pr['labels'],
                'created_at': pr['created_at'],
                'days_open': days_open,
                'additions': pr['additions'],
                'deletions': pr['deletions'],
                'has_body': bool(pr.get('body')),
                'url': pr['url']
            })

        return features

    def extract_timeline(self) -> Dict:
        """Extract activity timeline."""
        timeline = defaultdict(lambda: {'issues': 0, 'prs': 0, 'commits': 0})

        for issue in self.data['issues']:
            date = issue['created_at'][:10]
            timeline[date]['issues'] += 1

        for pr in self.data['pull_requests']:
            date = pr['created_at'][:10]
            timeline[date]['prs'] += 1

        for commit in self.data['commits']:
            date = commit['created_at'][:10]
            timeline[date]['commits'] += 1

        return dict(sorted(timeline.items()))

    def extract_contributors(self) -> Dict:
        """Extract contributor stats."""
        contributors = defaultdict(lambda: {'issues': 0, 'prs': 0, 'commits': 0})

        for issue in self.data['issues']:
            author = issue['author']['login']
            contributors[author]['issues'] += 1

        for pr in self.data['pull_requests']:
            author = pr['author']['login']
            contributors[author]['prs'] += 1

        for commit in self.data['commits']:
            author = commit['author']['login']
            contributors[author]['commits'] += 1

        return dict(contributors)

    def process(self) -> Dict:
        """Process all features."""
        return {
            'repo': self.repo_name,
            'metadata': {
                'stars': self.data['stars'],
                'forks': self.data['forks'],
                'language': self.data.get('language'),
                'description': self.data.get('description')
            },
            'issues': self.extract_issues(),
            'prs': self.extract_prs(),
            'timeline': self.extract_timeline(),
            'contributors': self.extract_contributors()
        }

    @staticmethod
    def save(processed_data: Dict, output_dir: str = 'data/processed') -> str:
        """Save processed data."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        repo = processed_data['repo'].replace('/', '_')
        filename = f"{output_dir}/{repo}_processed.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2)
        return filename
