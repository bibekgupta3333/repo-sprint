"""Core local scraper module — mirrors the Scraper interface for local git repos.

Combines LocalGitScraper (commits/diffs) with the existing API Scraper
(issues/PRs) for hybrid ingestion.
"""
import json
from pathlib import Path
from typing import Optional, List

from .scraper import Scraper, link_issues_to_pull_requests

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.scrapper.local_git import LocalGitScraper


class LocalScraper:
    """Hybrid scraper: local git for commits/diffs, GitHub API for issues/PRs.

    Implements the same ``ingest()`` → ``save()`` interface as
    ``scripts/_core.Scraper`` so the downstream pipeline (Analyzer, Processor,
    SprintPreprocessor, ChromaFormatter) works without changes.
    """

    def __init__(self, repo_path: str):
        """
        Args:
            repo_path: Absolute or relative path to the local git clone.
        """
        self.git = LocalGitScraper(repo_path)
        self.api = Scraper()  # used for issues/PRs only

    # ------------------------------------------------------------------ #
    #  Public interface                                                   #
    # ------------------------------------------------------------------ #

    def ingest(
        self,
        owner: str,
        repo: str,
        issues_limit: int = 50,
        prs_limit: int = 50,
        commits_limit: Optional[int] = None,
        commit_since: Optional[str] = None,
        commit_until: Optional[str] = None,
        fetch_diffs: bool = False,
        offline: bool = False,
        verbose: bool = True,
        skip_local_prs: bool = False,
        skip_local_issues: bool = False,
    ) -> Optional[dict]:
        """Ingest repository data using hybrid local + API approach.

        - **Commits & diffs** come from the local git clone (no rate limit).
        - **Issues & PRs** come from the GitHub API (low call volume).

        When ``offline=True``, the API is skipped entirely and issues/PRs
        are set to empty lists.

        Returns the same ``RepoData`` dict schema as ``Scraper.ingest()``.
        """
        if verbose:
            mode = "offline" if offline else "hybrid (local commits + API issues/PRs)"
            print(f"\nIngesting {owner}/{repo} in {mode} mode ...")

        # ----- Local: commits & diffs --------------------------------- #
        if verbose:
            print("\n--- Local Git: Commits & Diffs ---")

        local_data = self.git.scrape(
            owner=owner,
            repo=repo,
            fetch_diffs=fetch_diffs,
            since=commit_since,
            until=commit_until,
            limit=commits_limit,
            verbose=verbose,
            skip_local_prs=skip_local_prs,
            skip_local_issues=skip_local_issues,
        )

        if not local_data:
            print("ERROR: Failed to extract local git data")
            return None

        # ----- API: Issues & PRs -------------------------------------- #
        # local_data already contains PRs/issues from commit trailers
        local_issues = local_data.get("issues", [])
        local_prs = local_data.get("prs", [])
        issues: List[dict] = local_issues
        prs: List[dict] = local_prs

        if not offline:
            if verbose:
                print("\n--- GitHub API: Issues & PRs ---")

            try:
                api_issues = self.api.get_issues(owner, repo, limit=issues_limit)
                if verbose:
                    print(f"  API issues fetched: {len(api_issues)}")
                # API issues have richer data (title, body, labels)
                # Merge: API issues take priority, local fill gaps
                api_issue_nums = {i["number"] for i in api_issues}
                # Keep API issues + local-only issues not in API
                issues = api_issues + [
                    i for i in local_issues
                    if i["number"] not in api_issue_nums
                ]
            except Exception as e:
                print(f"  Warning: Could not fetch issues via API: {e}")
                print(f"  Using {len(local_issues)} issues from commit trailers")

            try:
                api_prs = self.api.get_prs(owner, repo, limit=prs_limit)
                if verbose:
                    print(f"  API PRs fetched: {len(api_prs)}")
                # API PRs have richer data; local PRs fill in older/merged CLs
                api_pr_nums = {p["number"] for p in api_prs}
                prs = api_prs + [
                    p for p in local_prs
                    if p["number"] not in api_pr_nums
                ]
            except Exception as e:
                print(f"  Warning: Could not fetch PRs via API: {e}")
                print(f"  Using {len(local_prs)} PRs/CLs from commit trailers")

            # Link issues to PRs using closing keywords
            link_issues_to_pull_requests(issues, prs)

            # Ensure PR sub-fields expected by downstream
            for pr in prs:
                pr.setdefault("commits", [])
                pr.setdefault("file_diffs", [])
        else:
            if verbose:
                print(f"\n--- Offline mode: {len(local_prs)} PRs/CLs, "
                      f"{len(local_issues)} issues from commit trailers ---")

        # ----- Merge into unified RepoData schema --------------------- #
        result = {
            "owner": owner,
            "name": repo,
            "url": local_data.get("url", f"https://github.com/{owner}/{repo}"),
            "description": local_data.get("description", ""),
            "stars": local_data.get("stars", 0),
            "forks": local_data.get("forks", 0),
            "language": local_data.get("language", ""),
            "issues": issues,
            "prs": prs,
            "commits": local_data["commits"],
            "commit_diffs": local_data.get("commit_diffs", []),
        }

        if verbose:
            print(f"\n  Total: {len(result['commits'])} commits, "
                  f"{len(result['issues'])} issues, {len(result['prs'])} PRs, "
                  f"{len(result['commit_diffs'])} diffs")

        return result

    def save(self, repo_data: dict, output_dir: str = "data/raw") -> str:
        """Save ingested data to JSON — same as ``Scraper.save()``."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        filename = (
            f"{output_dir}/{repo_data['owner']}_{repo_data['name']}.json"
        )
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(repo_data, f, indent=2)
        print(f"  Saved to {filename}")
        return filename
