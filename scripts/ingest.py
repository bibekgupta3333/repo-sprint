"""Ingest repository data with complete code-change intelligence."""
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts._core import Scraper, Analyzer, Processor
from src.data.preprocessor import SprintPreprocessor
from src.data.formatter import ChromaFormatter


def _parse_iso8601(value: str) -> datetime:
    """Parse ISO 8601 string and return timezone-aware datetime."""
    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _to_github_iso(dt: datetime) -> str:
    """Format datetime in GitHub API expected UTC timestamp format."""
    return dt.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def resolve_commit_time_window(
    since: Optional[str],
    until: Optional[str],
    days: Optional[int],
) -> tuple[Optional[str], Optional[str]]:
    """Resolve and validate commit time filters for ingestion."""
    since_dt = _parse_iso8601(since) if since else None
    until_dt = _parse_iso8601(until) if until else None

    if days is not None:
        if days <= 0:
            raise ValueError('--days must be a positive integer')
        if since_dt or until_dt:
            raise ValueError('--days cannot be used together with --since or --until')
        since_dt = datetime.now(timezone.utc) - timedelta(days=days)

    if since_dt and until_dt and since_dt > until_dt:
        raise ValueError('--since must be earlier than or equal to --until')

    return (
        _to_github_iso(since_dt) if since_dt else None,
        _to_github_iso(until_dt) if until_dt else None,
    )


def _commit_in_time_window(
    commit: dict,
    since_dt: Optional[datetime],
    until_dt: Optional[datetime],
) -> bool:
    """Check whether commit.created_at is inside the configured time window."""
    if not since_dt and not until_dt:
        return True

    created = commit.get('created_at')
    if not created:
        return True

    try:
        created_dt = _parse_iso8601(created)
    except Exception:
        return True

    if since_dt and created_dt < since_dt:
        return False
    if until_dt and created_dt > until_dt:
        return False
    return True


def extract_language_from_filename(filename: str) -> str:
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


def _commit_sha_for_api(commit: dict) -> str:
    """Resolve full SHA for GET /commits/{sha} (short SHA from UI is valid)."""
    return (commit.get('sha_full') or commit.get('sha') or '').strip()


def get_commit_diff(
    owner: str,
    repo: str,
    commit_sha: str,
    session: requests.Session,
) -> Optional[dict]:
    """Fetch detailed diff for a single commit."""
    if not commit_sha:
        return None
    try:
        url = f"https://api.github.com/repos/{owner}/{repo}/commits/{commit_sha}"
        response = session.get(url, timeout=30)
        if response.status_code != 200:
            return None

        commit = response.json()
        file_diffs = []
        language_breakdown = defaultdict(int)
        total_additions = 0
        total_deletions = 0

        for file_info in commit.get("files", []):
            lang = extract_language_from_filename(file_info["filename"])
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


def _enrich_commit_diffs(
    owner: str,
    repo: str,
    repo_data: dict,
    scraper: Scraper,
    verbose: bool = True,
    pr_diff_limit: int = 20,
    commit_since: Optional[str] = None,
    commit_until: Optional[str] = None,
) -> None:
    """Fetch PR commits/files, then per-commit diffs for all unique SHAs (deduped).

    Each PR needs two API calls (commits + files). ``pr_diff_limit`` caps how many
    PRs are enriched to stay within GitHub rate limits without a token; raise it or
    set GITHUB_TOKEN when you need full coverage.
    """
    session = scraper.session
    prs = repo_data.get('prs') or []
    since_dt = _parse_iso8601(commit_since) if commit_since else None
    until_dt = _parse_iso8601(commit_until) if commit_until else None

    if pr_diff_limit <= 0:
        for pr in prs:
            pr.setdefault('commits', [])
            pr.setdefault('file_diffs', [])
    else:
        if verbose:
            print(
                f"  Fetching commits and file lists for up to {pr_diff_limit} PR(s)..."
            )
        head = prs[:pr_diff_limit]
        tail = prs[pr_diff_limit:]
        for pr in head:
            num = pr['number']
            pr['commits'] = scraper.get_pull_commits(owner, repo, num)
            pr['file_diffs'] = scraper.get_pull_files(owner, repo, num)
        for pr in tail:
            pr.setdefault('commits', [])
            pr.setdefault('file_diffs', [])

    ordered_shas: list[str] = []
    seen: set[str] = set()

    def remember(commit: dict) -> None:
        if not _commit_in_time_window(commit, since_dt, until_dt):
            return
        sha = _commit_sha_for_api(commit)
        if not sha:
            return
        key = sha[:7]
        if key not in seen:
            seen.add(key)
            ordered_shas.append(sha)

    for c in repo_data.get('commits') or []:
        remember(c)
    for pr in prs:
        for c in pr.get('commits') or []:
            remember(c)

    if verbose:
        print(f"  Fetching commit diffs for {len(ordered_shas)} unique commit(s)...")

    diff_by_short: dict[str, dict] = {}
    commit_diffs_ordered: list[dict] = []

    for i, sha in enumerate(ordered_shas, 1):
        diff = get_commit_diff(owner, repo, sha, session)
        if diff:
            short = diff['sha'][:7]
            if short not in diff_by_short:
                diff_by_short[short] = diff
                commit_diffs_ordered.append(diff)
        if verbose and (i % 10 == 0 or i == len(ordered_shas)):
            print(f"    Diffs fetched: {i}/{len(ordered_shas)}")

    repo_data['commit_diffs'] = commit_diffs_ordered

    for c in repo_data.get('commits') or []:
        key = (c.get('sha') or '')[:7]
        c['diff'] = diff_by_short.get(key)

    for pr in prs:
        for c in pr.get('commits') or []:
            key = (c.get('sha') or '')[:7]
            c['diff'] = diff_by_short.get(key)


def ingest_repo(
    owner: str,
    repo: str,
    output_dir: str = "data",
    fetch_diffs: bool = False,
    verbose: bool = True,
    pr_diff_limit: int = 20,
    commit_since: Optional[str] = None,
    commit_until: Optional[str] = None,
) -> dict:
    """
    Ingest a single repository with complete code-change intelligence.

    Steps:
    1. Download from GitHub API (with optional detailed diffs)
    2. Analyze raw data
    3. Process and extract features
    4. Create sprint-based output with code-change metrics

    Args:
        owner: Repository owner (e.g., 'golang')
        repo: Repository name (e.g., 'go')
        output_dir: Output directory for JSON files
        fetch_diffs: Fetch detailed code diffs for commits
        verbose: Print progress messages
        pr_diff_limit: Max PRs to fetch commits/files for when fetch_diffs is True
            (0 = skip PR enrichment; use a high value with GITHUB_TOKEN for full PRs)
        commit_since: Optional lower bound for commit timestamps (ISO 8601)
        commit_until: Optional upper bound for commit timestamps (ISO 8601)

    Returns:
        Dictionary with paths to saved files
    """
    result = {}
    Path(output_dir).mkdir(exist_ok=True)

    # Step 1: Download
    if verbose:
        print(f"\n{'='*60}")
        print(f"STEP 1: Downloading {owner}/{repo}" + (" with code diffs" if fetch_diffs else ""))
        print(f"{'='*60}")

    scraper = Scraper()
    repo_data = scraper.ingest(
        owner,
        repo,
        commit_since=commit_since,
        commit_until=commit_until,
    )

    if not repo_data:
        print(f"ERROR: Failed to download {owner}/{repo}")
        return result

    # Fetch PR file lists, PR commits, and per-commit diffs (BEFORE saving raw file)
    if fetch_diffs:
        _enrich_commit_diffs(
            owner,
            repo,
            repo_data,
            scraper,
            verbose=verbose,
            pr_diff_limit=pr_diff_limit,
            commit_since=commit_since,
            commit_until=commit_until,
        )
    else:
        for c in repo_data.get('commits') or []:
            c['diff'] = None
        for pr in repo_data.get('prs') or []:
            pr.setdefault('commits', [])
            pr.setdefault('file_diffs', [])

    # Save raw file AFTER diffs are fetched
    raw_file = scraper.save(repo_data)
    result['raw_file'] = raw_file

    # Step 2: Analyze
    if verbose:
        print(f"{'='*60}")
        print(f"STEP 2: Analyzing {owner}/{repo}")
        print(f"{'='*60}")

    analyzer = Analyzer(raw_file)
    analyzer.analyze()

    # Step 3: Process
    if verbose:
        print(f"{'='*60}")
        print(f"STEP 3: Processing {owner}/{repo}")
        print(f"{'='*60}")

    processor = Processor(raw_file)
    processed_data = processor.process()
    processed_file = Processor.save(processed_data)
    result['processed_file'] = processed_file

    # Step 4: Create sprint-based output with code-change intelligence
    if verbose:
        print(f"{'='*60}")
        print(f"STEP 4: Creating sprint-based output with code metrics")
        print(f"{'='*60}")

    try:
        preprocessor = SprintPreprocessor(repo_data)
        sprints = preprocessor.create_sprints()

        # Save sprint data
        sprint_output_file = Path(output_dir) / f"{owner}_{repo}_sprints.json"
        with open(sprint_output_file, "w", encoding="utf-8") as f:
            json.dump(sprints, f, indent=2)
        result['sprints_file'] = str(sprint_output_file)
        print(f"  Saved {len(sprints)} sprints to {sprint_output_file}")

        # Create both sprint-aligned and flat Chroma documents
        all_docs = []
        sprint_aligned_docs = []
        for sprint in sprints:
            formatter = ChromaFormatter(sprint)
            sprint_aligned_docs.append(formatter.format_documents_by_sprint())
            all_docs.extend(formatter.format_documents())

        documents_file = Path(output_dir) / f"{owner}_{repo}_documents.json"
        with open(documents_file, "w", encoding="utf-8") as f:
            json.dump(sprint_aligned_docs, f, indent=2)
        result['documents_file'] = str(documents_file)

        chroma_documents_file = Path(output_dir) / f"{owner}_{repo}_chroma_documents.json"
        with open(chroma_documents_file, "w", encoding="utf-8") as f:
            json.dump(all_docs, f, indent=2)
        result['chroma_documents_file'] = str(chroma_documents_file)

        print(f"  Saved {len(sprint_aligned_docs)} sprint-aligned documents to {documents_file}")
        print(f"  Saved {len(all_docs)} flat Chroma documents to {chroma_documents_file}")

        # Print code-change summary
        if fetch_diffs and repo_data.get('commit_diffs'):
            print(f"\n{'='*60}")
            print("Code-Change Summary")
            print(f"{'='*60}")
            total_additions = sum(d.get("total_additions", 0) for d in repo_data["commit_diffs"])
            total_deletions = sum(d.get("total_deletions", 0) for d in repo_data["commit_diffs"])
            total_files = sum(d.get("files_changed", 0) for d in repo_data["commit_diffs"])

            print(f"Commits analyzed: {len(repo_data.get('commit_diffs', []))}")
            print(f"Total additions: {total_additions}")
            print(f"Total deletions: {total_deletions}")
            print(f"Total files changed: {total_files}")

            if repo_data["commit_diffs"]:
                lang_freq = {}
                for diff in repo_data["commit_diffs"]:
                    for lang, count in diff.get("language_breakdown", {}).items():
                        lang_freq[lang] = lang_freq.get(lang, 0) + count
                print(f"Language breakdown: {lang_freq}")

    except Exception as e:
        print(f"Warning: Could not create sprint output: {e}")

    print(f"\n✓ Ingestion complete for {owner}/{repo}")
    print(f"  Raw: {raw_file}")
    print(f"  Processed: {processed_file}")
    if 'sprints_file' in result:
        print(f"  Sprints: {result['sprints_file']}")
    if 'documents_file' in result:
        print(f"  Documents: {result['documents_file']}\n")
    if 'chroma_documents_file' in result:
        print(f"  Chroma Documents: {result['chroma_documents_file']}\n")

    return result


def show_usage():
    """Show usage information."""
    print("""
Usage: python scripts/ingest.py <owner> <repo> [options]

Description:
  Ingest a GitHub repository: download, analyze, and process code changes

Arguments:
  <owner>         Repository owner (e.g., golang)
  <repo>          Repository name (e.g., go)

Options:
  --help          Show this help message and exit
  --quiet         Suppress progress output
  --diffs         Fetch detailed code diffs with per-file patches (slower)
  --pr-diff-limit N  Max PRs to enrich with commits+files (default: 1000; 0=none)
    --since ISO8601  Include commits on/after this timestamp
    --until ISO8601  Include commits on/before this timestamp
    --days N         Include commits from last N days (exclusive with --since/--until)
  --output DIR    Output directory for results (default: data)

Outputs:
  sprints.json    Sprint data with code metrics and commits (includes PR commits)
  documents.json  ChromaDB documents with code-change intelligence

Examples:
  python scripts/ingest.py golang go
  python scripts/ingest.py rust-lang rust --diffs
  python scripts/ingest.py torvalds linux --diffs --output results --quiet
  python scripts/ingest.py torvalds linux --diffs --pr-diff-limit 500
    python scripts/ingest.py golang go --diffs --since 2026-03-01T00:00:00Z --until 2026-03-24T23:59:59Z
    python scripts/ingest.py golang go --days 14
    """)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('owner', nargs='?', default=None)
    parser.add_argument('repo', nargs='?', default=None)
    parser.add_argument('--help', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--diffs', action='store_true', help='Fetch detailed code diffs with per-file patches')
    parser.add_argument(
        '--pr-diff-limit',
        type=int,
        default=1000,
        help='Max PRs to fetch commits+file patches for with --diffs (0=skip PR enrichment, default=1000)',
    )
    parser.add_argument('--since', default=None, help='Include commits on/after this ISO 8601 timestamp')
    parser.add_argument('--until', default=None, help='Include commits on/before this ISO 8601 timestamp')
    parser.add_argument('--days', type=int, default=None, help='Include commits from the last N days')
    parser.add_argument('--output', default='data', help='Output directory for results')

    args = parser.parse_args()

    if args.help or not args.owner or not args.repo:
        show_usage()
        sys.exit(0)

    try:
        commit_since, commit_until = resolve_commit_time_window(
            args.since,
            args.until,
            args.days,
        )
    except ValueError as e:
        print(f'ERROR: {e}')
        sys.exit(1)

    ingest_repo(
        args.owner,
        args.repo,
        verbose=not args.quiet,
        fetch_diffs=args.diffs,
        output_dir=args.output,
        pr_diff_limit=args.pr_diff_limit,
        commit_since=commit_since,
        commit_until=commit_until,
    )


if __name__ == '__main__':
    main()
