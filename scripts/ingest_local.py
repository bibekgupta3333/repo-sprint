"""Hybrid ingestion: local git for commits/diffs + GitHub API for issues/PRs.

This script replaces ``scripts/ingest.py`` for local repository clones,
avoiding GitHub API rate limits for the expensive commit-diff operations
while still fetching issue and PR metadata from the API.

Usage:
    python scripts/ingest_local.py repos/go --owner golang --repo go
    python scripts/ingest_local.py repos/go --owner golang --repo go --diffs
    python scripts/ingest_local.py repos/go --owner golang --repo go --diffs --days 30
    python scripts/ingest_local.py repos/go --owner golang --repo go --offline --diffs
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env file (for GITHUB_TOKEN, etc.)
try:
    from dotenv import load_dotenv
    _proj_root = Path(__file__).parent.parent
    for _env_name in ('.env', '.env.local'):
        _env_path = _proj_root / _env_name
        try:
            if _env_path.exists():
                load_dotenv(_env_path)
        except (PermissionError, OSError):
            pass
except ImportError:
    pass

from scripts._core import Analyzer, Processor
from scripts._core.local_scraper import LocalScraper
from src.data.preprocessor import SprintPreprocessor
from src.data.formatter import ChromaFormatter


# --------------------------------------------------------------------------- #
#  Time window helpers (same logic as scripts/ingest.py)                      #
# --------------------------------------------------------------------------- #

def _parse_iso8601(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _to_github_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def resolve_time_window(
    since: Optional[str],
    until: Optional[str],
    days: Optional[int],
) -> tuple[Optional[str], Optional[str]]:
    """Resolve and validate commit time filters."""
    since_dt = _parse_iso8601(since) if since else None
    until_dt = _parse_iso8601(until) if until else None

    if days is not None:
        if days <= 0:
            raise ValueError("--days must be a positive integer")
        if since_dt or until_dt:
            raise ValueError(
                "--days cannot be used together with --since or --until"
            )
        since_dt = datetime.now(timezone.utc) - timedelta(days=days)

    if since_dt and until_dt and since_dt > until_dt:
        raise ValueError("--since must be earlier than or equal to --until")

    return (
        _to_github_iso(since_dt) if since_dt else None,
        _to_github_iso(until_dt) if until_dt else None,
    )


# --------------------------------------------------------------------------- #
#  Main ingestion                                                             #
# --------------------------------------------------------------------------- #

def ingest_local(
    repo_path: str,
    owner: str,
    repo: str,
    output_dir: str = "data",
    fetch_diffs: bool = False,
    offline: bool = False,
    verbose: bool = True,
    issues_limit: int = 50,
    prs_limit: int = 50,
    commits_limit: Optional[int] = None,
    commit_since: Optional[str] = None,
    commit_until: Optional[str] = None,
    pr_diff_limit: int = 20,
    skip_local_prs: bool = False,
    skip_local_issues: bool = False,
) -> dict:
    """Run the hybrid ingestion pipeline.

    Steps:
    1. Extract commits/diffs from local git clone (no rate limit)
    2. Fetch issues/PRs from GitHub API (low volume, unless --offline)
    3. Analyze raw data
    4. Process and extract features
    5. Create sprint-based output with code-change metrics

    Returns:
        Dictionary with paths to saved files.
    """
    result: dict = {}
    Path(output_dir).mkdir(exist_ok=True)

    # ---- Step 1: Hybrid ingest ----------------------------------------- #
    if verbose:
        print(f"\n{'='*60}")
        print(f"STEP 1: Ingesting {owner}/{repo}")
        mode = "OFFLINE (local only)" if offline else "HYBRID (local + API)"
        print(f"  Mode: {mode}")
        print(f"  Local clone: {repo_path}")
        print(f"{'='*60}")

    scraper = LocalScraper(repo_path)
    repo_data = scraper.ingest(
        owner=owner,
        repo=repo,
        issues_limit=issues_limit,
        prs_limit=prs_limit,
        commits_limit=commits_limit,
        commit_since=commit_since,
        commit_until=commit_until,
        fetch_diffs=fetch_diffs,
        offline=offline,
        verbose=verbose,
        skip_local_prs=skip_local_prs,
        skip_local_issues=skip_local_issues,
    )

    if not repo_data:
        print(f"ERROR: Failed to ingest {owner}/{repo}")
        return result

    # If hybrid mode with diffs, also enrich PRs with commits + file lists
    if fetch_diffs and not offline and repo_data.get("prs"):
        _enrich_prs_from_api(
            owner, repo, repo_data, scraper.api,
            pr_diff_limit=pr_diff_limit, verbose=verbose,
        )

    # Save raw
    raw_file = scraper.save(repo_data)
    result["raw_file"] = raw_file

    # ---- Step 2: Analyze ----------------------------------------------- #
    if verbose:
        print(f"\n{'='*60}")
        print(f"STEP 2: Analyzing {owner}/{repo}")
        print(f"{'='*60}")

    analyzer = Analyzer(raw_file)
    analyzer.analyze()

    # ---- Step 3: Process ----------------------------------------------- #
    if verbose:
        print(f"{'='*60}")
        print(f"STEP 3: Processing {owner}/{repo}")
        print(f"{'='*60}")

    processor = Processor(raw_file)
    processed_data = processor.process()
    processed_file = Processor.save(processed_data)
    result["processed_file"] = processed_file

    # ---- Step 4: Sprint output ----------------------------------------- #
    if verbose:
        print(f"{'='*60}")
        print(f"STEP 4: Creating sprint-based output")
        print(f"{'='*60}")

    try:
        preprocessor = SprintPreprocessor(repo_data)
        sprints = preprocessor.create_sprints()

        # Save sprint data
        sprint_file = Path(output_dir) / f"{owner}_{repo}_sprints.json"
        with open(sprint_file, "w", encoding="utf-8") as f:
            json.dump(sprints, f, indent=2)
        result["sprints_file"] = str(sprint_file)
        print(f"  Saved {len(sprints)} sprints to {sprint_file}")

        # Create both sprint-aligned and flat Chroma documents
        all_docs: list = []
        sprint_aligned_docs: list = []
        for sprint in sprints:
            formatter = ChromaFormatter(sprint)
            sprint_aligned_docs.append(formatter.format_documents_by_sprint())
            all_docs.extend(formatter.format_documents())

        docs_file = Path(output_dir) / f"{owner}_{repo}_documents.json"
        with open(docs_file, "w", encoding="utf-8") as f:
            json.dump(sprint_aligned_docs, f, indent=2)
        result["documents_file"] = str(docs_file)

        chroma_file = Path(output_dir) / f"{owner}_{repo}_chroma_documents.json"
        with open(chroma_file, "w", encoding="utf-8") as f:
            json.dump(all_docs, f, indent=2)
        result["chroma_documents_file"] = str(chroma_file)

        print(f"  Saved {len(sprint_aligned_docs)} sprint-aligned documents")
        print(f"  Saved {len(all_docs)} flat Chroma documents")

        # Print code-change summary
        if fetch_diffs and repo_data.get("commit_diffs"):
            _print_code_summary(repo_data)

    except Exception as e:
        print(f"  Warning: Could not create sprint output: {e}")
        import traceback
        traceback.print_exc()

    # ---- Done ---------------------------------------------------------- #
    print(f"\n✓ Ingestion complete for {owner}/{repo}")
    print(f"  Raw:       {raw_file}")
    print(f"  Processed: {processed_file}")
    if "sprints_file" in result:
        print(f"  Sprints:   {result['sprints_file']}")
    if "documents_file" in result:
        print(f"  Documents: {result['documents_file']}")
    if "chroma_documents_file" in result:
        print(f"  Chroma:    {result['chroma_documents_file']}")

    return result


# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #

def _build_pr_commit_map(
    repo_data: dict,
) -> dict:
    """Map PR numbers to local commits and diffs.

    Parses PR numbers from commit messages using common
    GitHub patterns:
      - ``(#123)`` — squash-merge default
      - ``Merge pull request #123 from ...``

    Returns ``{pr_number: {'commits': [...], 'diffs': [...]}}``
    """
    import re
    # Patterns that reference a PR number in commit messages
    _PR_REF = re.compile(
        r'\(#(\d+)\)'
        r'|Merge pull request #(\d+)',
    )

    # Build SHA → diff lookup
    diff_by_sha: dict = {}
    for d in repo_data.get("commit_diffs", []):
        short = (d.get("sha") or "")[:7]
        if short:
            diff_by_sha[short] = d

    pr_map: dict = {}  # pr_num → {commits, diffs}
    for commit in repo_data.get("commits", []):
        msg = commit.get("message", "")
        for m in _PR_REF.finditer(msg):
            pr_num = int(m.group(1) or m.group(2))
            if pr_num not in pr_map:
                pr_map[pr_num] = {
                    "commits": [],
                    "diffs": [],
                }
            pr_map[pr_num]["commits"].append({
                "sha": commit.get("sha", ""),
                "message": msg[:200],
            })
            sha_short = (commit.get("sha") or "")[:7]
            diff = diff_by_sha.get(sha_short)
            if diff:
                pr_map[pr_num]["diffs"].append(diff)

    return pr_map


def _enrich_prs_from_api(
    owner: str,
    repo: str,
    repo_data: dict,
    api_scraper,
    pr_diff_limit: int = 20,
    verbose: bool = True,
):
    """Enrich PR entries with commits/file_diffs.

    **Fully local** strategy — zero API calls:
      1. Parse PR numbers from commit messages
         (e.g. ``(#123)``, ``Merge pull request #123``).
      2. Match PRs to local commits and diffs.
      3. Unmatched PRs get empty commits/file_diffs
         (their metadata from the API is still preserved).
    """
    prs = repo_data.get("prs", [])
    if not prs:
        return

    # --- Build local PR → commits/diffs mapping ---
    pr_map = _build_pr_commit_map(repo_data)
    local_matched = 0
    unmatched = 0

    for pr in prs:
        num = pr["number"]
        local = pr_map.get(num)
        if local and local["commits"]:
            pr["commits"] = local["commits"]
            # Merge file_diffs from all matching commits
            all_files: list = []
            total_add = 0
            total_del = 0
            for d in local["diffs"]:
                all_files.extend(d.get("file_diffs", []))
                total_add += d.get("total_additions", 0)
                total_del += d.get("total_deletions", 0)
            pr["file_diffs"] = all_files
            # Update additions/deletions if they were 0
            if pr.get("additions", 0) == 0:
                pr["additions"] = total_add
            if pr.get("deletions", 0) == 0:
                pr["deletions"] = total_del
            local_matched += 1
        else:
            pr.setdefault("commits", [])
            pr.setdefault("file_diffs", [])
            unmatched += 1

    if verbose:
        print(f"\n  PR enrichment (local only): "
              f"{local_matched} matched, "
              f"{unmatched} unmatched")


def _print_code_summary(repo_data: dict):
    """Print code-change summary."""
    diffs = repo_data.get("commit_diffs", [])
    if not diffs:
        return

    total_add = sum(d.get("total_additions", 0) for d in diffs)
    total_del = sum(d.get("total_deletions", 0) for d in diffs)
    total_files = sum(d.get("files_changed", 0) for d in diffs)

    print(f"\n{'='*60}")
    print("Code-Change Summary")
    print(f"{'='*60}")
    print(f"  Commits with diffs: {len(diffs)}")
    print(f"  Total additions:    +{total_add:,}")
    print(f"  Total deletions:    -{total_del:,}")
    print(f"  Total files changed: {total_files:,}")

    lang_freq: dict[str, int] = {}
    for diff in diffs:
        for lang, count in diff.get("language_breakdown", {}).items():
            lang_freq[lang] = lang_freq.get(lang, 0) + count
    if lang_freq:
        sorted_langs = sorted(lang_freq.items(), key=lambda x: -x[1])[:10]
        print(f"  Top languages: {dict(sorted_langs)}")


# --------------------------------------------------------------------------- #
#  CLI                                                                        #
# --------------------------------------------------------------------------- #

USAGE = """
Usage: python scripts/ingest_local.py <repo_path> --owner <owner> --repo <repo> [options]

Description:
  Hybrid ingestion — local git for commits/diffs, GitHub API for issues/PRs.
  Avoids GitHub API rate limits for the expensive commit-diff operations.

Arguments:
  <repo_path>       Path to the local git clone (e.g., repos/go)

Required options:
  --owner OWNER     Repository owner (e.g., golang)
  --repo REPO       Repository name (e.g., go)

Options:
  --help              Show this help message and exit
  --diffs             Extract commit diffs locally (via git show)
  --offline           Skip GitHub API entirely (PRs/issues from commit trailers)
  --skip-local-prs    Skip extraction of PRs/CLs from commit trailers
  --skip-local-issues Skip extraction of issue refs from commit trailers
  --quiet             Suppress progress output
  --output DIR        Output directory (default: data)
  --issues-limit N    Max issues to fetch from API (default: 50, 0 = no limit)
  --prs-limit N       Max PRs to fetch from API (default: 50, 0 = no limit)
  --commits-limit N   Max commits to extract locally (default: all)
  --pr-diff-limit N   Max PRs to enrich with commits/files (default: 20, 0 = no limit)
  --since ISO8601     Include commits on/after this timestamp
  --until ISO8601     Include commits on/before this timestamp
  --days N            Include commits from last N days

Examples:
  # Basic: local commits + API issues/PRs
  python scripts/ingest_local.py repos/go --owner golang --repo go

  # With diffs: extract full code changes locally
  python scripts/ingest_local.py repos/go --owner golang --repo go --diffs

  # Last 30 days with diffs
  python scripts/ingest_local.py repos/go --owner golang --repo go --diffs --days 30

  # Date range with diffs
  python scripts/ingest_local.py repos/go --owner golang --repo go --diffs \\
    --since 2026-03-01T00:00:00Z --until 2026-03-24T23:59:59Z

  # Fully offline (no API calls at all)
  python scripts/ingest_local.py repos/go --owner golang --repo go --diffs --offline

  # High-volume: all commits, all diffs, many issues/PRs
  python scripts/ingest_local.py repos/go --owner golang --repo go --diffs \\
    --issues-limit 500 --prs-limit 500
"""


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("repo_path", nargs="?", default=None)
    parser.add_argument("--owner", default=None)
    parser.add_argument("--repo", default=None)
    parser.add_argument("--help", action="store_true")
    parser.add_argument("--diffs", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--skip-local-prs", action="store_true")
    parser.add_argument("--skip-local-issues", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output", default="data")
    parser.add_argument("--issues-limit", type=int, default=50)
    parser.add_argument("--prs-limit", type=int, default=50)
    parser.add_argument("--commits-limit", type=int, default=None)
    parser.add_argument("--pr-diff-limit", type=int, default=20)
    parser.add_argument("--since", default=None)
    parser.add_argument("--until", default=None)
    parser.add_argument("--days", type=int, default=None)

    args = parser.parse_args()

    if args.help or not args.repo_path or not args.owner or not args.repo:
        print(USAGE)
        sys.exit(0)

    try:
        commit_since, commit_until = resolve_time_window(
            args.since, args.until, args.days,
        )
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # 0 means "no limit" — use a large sentinel so pagination loops
    # continue until the API is exhausted.
    _NO_LIMIT = 999_999
    issues_limit = args.issues_limit if args.issues_limit else _NO_LIMIT
    prs_limit = args.prs_limit if args.prs_limit else _NO_LIMIT
    pr_diff_limit = args.pr_diff_limit if args.pr_diff_limit else _NO_LIMIT

    ingest_local(
        repo_path=args.repo_path,
        owner=args.owner,
        repo=args.repo,
        output_dir=args.output,
        fetch_diffs=args.diffs,
        offline=args.offline,
        verbose=not args.quiet,
        issues_limit=issues_limit,
        prs_limit=prs_limit,
        commits_limit=args.commits_limit,
        commit_since=commit_since,
        commit_until=commit_until,
        pr_diff_limit=pr_diff_limit,
        skip_local_prs=args.skip_local_prs,
        skip_local_issues=args.skip_local_issues,
    )


if __name__ == "__main__":
    main()
