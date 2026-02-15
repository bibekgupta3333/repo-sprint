#!/usr/bin/env python3
"""
Collect GitHub repository data using GitHub API.

This script collects detailed information about repositories, milestones,
issues, pull requests, and commits for dataset creation.

Usage:
    python collect_github_data.py --repos-file repos.txt --output-dir ../data/raw
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import time

from github import Github, GithubException
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()


class GitHubDataCollector:
    """Collect detailed GitHub data for sprint analysis."""

    def __init__(self, token: Optional[str] = None, output_dir: str = "../data/raw"):
        self.token = token or os.getenv('GITHUB_TOKEN')
        if not self.token:
            raise ValueError("GitHub token required. Set GITHUB_TOKEN env variable.")

        self.gh = Github(self.token)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Check rate limit
        rate_limit = self.gh.get_rate_limit()
        print(f"🔑 GitHub API Rate Limit: {rate_limit.core.remaining}/{rate_limit.core.limit}")
        print(f"   Resets at: {rate_limit.core.reset}")
        print()

    def collect_repository_data(self, repo_full_name: str) -> Dict:
        """Collect comprehensive data for a single repository."""
        try:
            repo = self.gh.get_repo(repo_full_name)

            print(f"📦 Collecting: {repo_full_name}")

            # Repository metadata
            repo_data = {
                'full_name': repo.full_name,
                'description': repo.description,
                'stars': repo.stargazers_count,
                'forks': repo.forks_count,
                'language': repo.language,
                'created_at': repo.created_at.isoformat() if repo.created_at else None,
                'updated_at': repo.updated_at.isoformat() if repo.updated_at else None,
                'milestones': [],
                'collected_at': datetime.now().isoformat(),
            }

            # Collect milestones
            milestones = repo.get_milestones(state='all')
            for milestone in tqdm(milestones, desc="  Milestones", leave=False):
                milestone_data = self._collect_milestone_data(repo, milestone)
                if milestone_data:
                    repo_data['milestones'].append(milestone_data)

            # Save repository data
            output_file = self.output_dir / 'repositories' / f"{repo.owner.login}_{repo.name}.json"
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w') as f:
                json.dump(repo_data, f, indent=2)

            print(f"   ✅ Saved {len(repo_data['milestones'])} milestones")
            return repo_data

        except GithubException as e:
            print(f"   ❌ Error: {e}")
            return None

    def _collect_milestone_data(self, repo, milestone) -> Optional[Dict]:
        """Collect data for a single milestone."""
        try:
            milestone_data = {
                'id': milestone.id,
                'number': milestone.number,
                'title': milestone.title,
                'description': milestone.description,
                'state': milestone.state,
                'created_at': milestone.created_at.isoformat() if milestone.created_at else None,
                'due_on': milestone.due_on.isoformat() if milestone.due_on else None,
                'closed_at': milestone.closed_at.isoformat() if milestone.closed_at else None,
                'open_issues': milestone.open_issues,
                'closed_issues': milestone.closed_issues,
                'issues': [],
                'pull_requests': [],
            }

            # Get issues for this milestone
            issues = repo.get_issues(milestone=milestone, state='all')
            for issue in issues:
                if issue.pull_request:
                    # It's a PR
                    pr_data = {
                        'number': issue.number,
                        'title': issue.title,
                        'state': issue.state,
                        'created_at': issue.created_at.isoformat() if issue.created_at else None,
                        'closed_at': issue.closed_at.isoformat() if issue.closed_at else None,
                        'labels': [label.name for label in issue.labels],
                        'assignees': [assignee.login for assignee in issue.assignees],
                        'comments': issue.comments,
                    }
                    milestone_data['pull_requests'].append(pr_data)
                else:
                    # Regular issue
                    issue_data = {
                        'number': issue.number,
                        'title': issue.title,
                        'body': issue.body[:500] if issue.body else None,  # Truncate
                        'state': issue.state,
                        'created_at': issue.created_at.isoformat() if issue.created_at else None,
                        'closed_at': issue.closed_at.isoformat() if issue.closed_at else None,
                        'labels': [label.name for label in issue.labels],
                        'assignees': [assignee.login for assignee in issue.assignees],
                        'comments': issue.comments,
                    }
                    milestone_data['issues'].append(issue_data)

            return milestone_data

        except GithubException as e:
            print(f"      ❌ Error collecting milestone {milestone.number}: {e}")
            return None

    def collect_from_file(self, repos_file: str, max_repos: Optional[int] = None):
        """Collect data for repositories listed in a file."""
        with open(repos_file, 'r') as f:
            repos = [line.strip() for line in f if line.strip() and not line.startswith('#')]

        if max_repos:
            repos = repos[:max_repos]

        print(f"📋 Found {len(repos)} repositories to collect")
        print()

        successful = 0
        total_milestones = 0

        for repo_name in repos:
            repo_data = self.collect_repository_data(repo_name)
            if repo_data:
                successful += 1
                total_milestones += len(repo_data.get('milestones', []))

            # Check rate limit
            rate_limit = self.gh.get_rate_limit()
            if rate_limit.core.remaining < 100:
                wait_time = (rate_limit.core.reset - datetime.now()).total_seconds()
                print(f"⏸️  Rate limit low. Waiting {wait_time:.0f}s until reset...")
                time.sleep(wait_time + 10)

            time.sleep(1)  # Be nice to GitHub API

        print()
        print(f"✅ Collection complete!")
        print(f"   Repositories: {successful}/{len(repos)}")
        print(f"   Total milestones: {total_milestones}")
        print(f"   Output directory: {self.output_dir.absolute()}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect GitHub repository data for sprint analysis"
    )
    parser.add_argument(
        '--repos-file',
        type=str,
        required=True,
        help='File containing repository names (owner/repo format, one per line)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='../data/raw',
        help='Output directory for collected data'
    )
    parser.add_argument(
        '--token',
        type=str,
        help='GitHub personal access token (or set GITHUB_TOKEN env var)'
    )
    parser.add_argument(
        '--max-repos',
        type=int,
        help='Maximum number of repositories to collect (for testing)'
    )

    args = parser.parse_args()

    collector = GitHubDataCollector(token=args.token, output_dir=args.output_dir)
    collector.collect_from_file(args.repos_file, max_repos=args.max_repos)


if __name__ == '__main__':
    main()
