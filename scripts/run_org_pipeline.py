"""End-to-end org ingestion pipeline.

Workflow:
1. Parse GitHub organization URL/name
2. Clone or update 3 repositories under repos/<org>/
3. Run local hybrid ingestion for each repo
4. Run synthetic step (calibrate or generate)
5. Run Chroma ingestion for each repo

Example:
    python scripts/run_org_pipeline.py https://github.com/Mintplex-Labs --repos anything-llm vector-admin docs
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clone/update 3 org repositories, ingest local sprint data, "
            "run synthetic step, and ingest into ChromaDB."
        )
    )
    parser.add_argument(
        "org_link",
        help="GitHub org URL or org name (e.g. https://github.com/Mintplex-Labs)",
    )
    parser.add_argument(
        "repos_positional",
        nargs="*",
        help=(
            "Optional repo names after org_link. Useful with npm run where "
            "flags like --repos may be swallowed."
        ),
    )
    parser.add_argument(
        "--repos",
        nargs="+",
        default=None,
        help=(
            "Repository names to prioritize. If fewer than --repo-count are provided, "
            "remaining repos are auto-selected from most recently updated org repos."
        ),
    )
    parser.add_argument(
        "--repo-count",
        type=int,
        default=1,
        help="Target number of unique repos to process (default: 1).",
    )
    parser.add_argument(
        "--include-forks",
        action="store_true",
        help="Include fork repositories when auto-selecting repos.",
    )
    parser.add_argument(
        "--synthetic-step",
        choices=["calibrate", "generate", "skip"],
        default="calibrate",
        help="Synthetic stage to run after local ingest (default: calibrate).",
    )
    parser.add_argument(
        "--synthetic-count",
        type=int,
        default=100,
        help="Number of synthetic sprints when --synthetic-step generate (default: 100).",
    )
    parser.add_argument(
        "--synthetic-personas",
        choices=["auto", "startup", "large_oss", "all"],
        default="auto",
        help="Persona set for synthetic generation (default: auto).",
    )
    parser.add_argument(
        "--no-query-test",
        action="store_true",
        help="Disable --query-test during Chroma ingestion.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use for child commands (default: current interpreter).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser.parse_args()


def parse_org_name(org_link: str) -> str:
    value = org_link.strip()
    parsed = urlparse(value)

    if parsed.scheme and parsed.netloc:
        host = parsed.netloc.lower()
        if host not in {"github.com", "www.github.com"}:
            raise ValueError(f"Unsupported host '{parsed.netloc}'. Please use github.com.")

        parts = [p for p in parsed.path.split("/") if p]
        if not parts:
            raise ValueError("Organization URL path is empty.")

        if parts[0] == "orgs":
            if len(parts) < 2:
                raise ValueError("Expected organization after '/orgs/'.")
            org = parts[1]
        else:
            org = parts[0]
    else:
        org = value.strip("/")

    if org.endswith(".git"):
        org = org[:-4]

    if not org:
        raise ValueError("Could not parse organization name.")

    return org


def _github_api_get(url: str, token: str | None) -> Any:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "repo-sprint-org-pipeline",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = Request(url, headers=headers)
    with urlopen(request, timeout=30) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload)


def fetch_top_repos(
    org: str,
    repo_count: int,
    include_forks: bool,
    token: str | None,
) -> list[str]:
    repos: list[str] = []
    seen: set[str] = set()
    page = 1

    while len(repos) < repo_count:
        url = (
            f"https://api.github.com/orgs/{quote(org)}/repos"
            f"?type=all&sort=updated&per_page=100&page={page}"
        )
        data = _github_api_get(url, token)

        if isinstance(data, dict) and data.get("message"):
            raise ValueError(f"GitHub API error: {data.get('message')}")
        if not isinstance(data, list):
            raise ValueError("Unexpected GitHub API response while listing repositories.")
        if not data:
            break

        for item in data:
            name = item.get("name")
            if not name:
                continue
            if not include_forks and item.get("fork"):
                continue
            normalized = name.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            repos.append(name)
            if len(repos) >= repo_count:
                break

        page += 1

    if len(repos) < repo_count:
        raise ValueError(
            f"Found only {len(repos)} repos for org '{org}', but {repo_count} were requested."
        )

    return repos[:repo_count]


def dedupe_repo_names(repo_names: list[str]) -> tuple[list[str], list[str]]:
    """Return ordered unique repo names and duplicates (case-insensitive)."""
    unique_names: list[str] = []
    duplicates: list[str] = []
    seen: set[str] = set()

    for raw_name in repo_names:
        name = raw_name.strip()
        if not name:
            continue
        normalized = name.lower()
        if normalized in seen:
            duplicates.append(raw_name)
            continue
        seen.add(normalized)
        unique_names.append(name)

    return unique_names, duplicates


def run_command(
    cmd: list[str],
    *,
    cwd: Path,
    dry_run: bool,
    env: dict[str, str] | None = None,
) -> None:
    printable = " ".join(shlex.quote(part) for part in cmd)
    print(f"\n$ {printable}")

    if dry_run:
        return

    completed = subprocess.run(cmd, cwd=str(cwd), env=env, check=False)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, cmd)


def clone_or_update_repo(
    root_dir: Path,
    org: str,
    repo: str,
    *,
    dry_run: bool,
) -> Path:
    target = root_dir / "repos" / org / repo
    target.parent.mkdir(parents=True, exist_ok=True)
    clone_url = f"https://github.com/{org}/{repo}.git"

    if (target / ".git").exists():
        run_command(["git", "-C", str(target), "pull", "--ff-only"], cwd=root_dir, dry_run=dry_run)
    elif target.exists():
        raise ValueError(f"Target path exists but is not a git repo: {target}")
    else:
        run_command(["git", "clone", clone_url, str(target)], cwd=root_dir, dry_run=dry_run)

    return target


def run_local_hybrid_ingest(
    root_dir: Path,
    python_cmd: str,
    org: str,
    repo: str,
    repo_path: Path,
    *,
    dry_run: bool,
) -> None:
    run_command(
        [
            python_cmd,
            "scripts/ingest_local.py",
            str(repo_path),
            "--owner",
            org,
            "--repo",
            repo,
            "--diffs",
            "--issues-limit",
            "0",
            "--prs-limit",
            "0",
            "--pr-diff-limit",
            "0",
        ],
        cwd=root_dir,
        dry_run=dry_run,
    )


def run_synthetic_step(
    root_dir: Path,
    python_cmd: str,
    step: str,
    count: int,
    personas: str,
    *,
    dry_run: bool,
) -> None:
    if step == "skip":
        print("\nSkipping synthetic stage (--synthetic-step skip)")
        return

    if step == "calibrate":
        run_command(
            [python_cmd, "src/data/synthetic_generator.py", "--calibrate", "all"],
            cwd=root_dir,
            dry_run=dry_run,
        )
        return

    run_command(
        [
            python_cmd,
            "src/data/synthetic_generator.py",
            "--count",
            str(count),
            "--personas",
            personas,
        ],
        cwd=root_dir,
        dry_run=dry_run,
    )


def run_chroma_ingest(
    root_dir: Path,
    python_cmd: str,
    org: str,
    repo: str,
    *,
    query_test: bool,
    dry_run: bool,
) -> None:
    cmd = [python_cmd, "-m", "src.chromadb", "--org", org, "--repo", repo]
    if query_test:
        cmd.append("--query-test")

    run_command(cmd, cwd=root_dir, dry_run=dry_run)


def main() -> int:
    args = parse_args()
    root_dir = Path(__file__).resolve().parent.parent
    org = parse_org_name(args.org_link)

    manual_repo_input: list[str] = []
    if args.repos:
        manual_repo_input.extend(args.repos)
    if args.repos_positional:
        manual_repo_input.extend(args.repos_positional)

    selected_repos, duplicates = dedupe_repo_names(manual_repo_input)
    if duplicates:
        joined = ", ".join(duplicates)
        raise ValueError(f"Duplicate repos detected in input: {joined}")

    if len(selected_repos) > args.repo_count:
        raise ValueError(
            f"Received {len(selected_repos)} repos but --repo-count is {args.repo_count}. "
            "Increase --repo-count or pass fewer repos."
        )

    if len(selected_repos) < args.repo_count:
        token = os.getenv("GITHUB_TOKEN")
        auto_repos = fetch_top_repos(
            org=org,
            repo_count=args.repo_count,
            include_forks=args.include_forks,
            token=token,
        )

        seen = {name.lower() for name in selected_repos}
        for repo_name in auto_repos:
            normalized = repo_name.lower()
            if normalized in seen:
                continue
            selected_repos.append(repo_name)
            seen.add(normalized)
            if len(selected_repos) >= args.repo_count:
                break

    if len(selected_repos) < args.repo_count:
        raise ValueError(
            f"Could not resolve {args.repo_count} unique repos for org '{org}'. "
            f"Resolved only {len(selected_repos)}."
        )

    print("=" * 72)
    print("ORG PIPELINE START")
    print("=" * 72)
    print(f"Organization: {org}")
    print(f"Repositories ({len(selected_repos)}): {', '.join(selected_repos)}")
    print(f"Synthetic step: {args.synthetic_step}")
    print(f"Query test: {not args.no_query_test}")
    print(f"Dry run: {args.dry_run}")

    repo_paths: dict[str, Path] = {}

    print("\n[1/4] Clone or update repositories")
    for repo in selected_repos:
        repo_paths[repo] = clone_or_update_repo(
            root_dir=root_dir,
            org=org,
            repo=repo,
            dry_run=args.dry_run,
        )

    print("\n[2/4] Run local hybrid ingest")
    for repo in selected_repos:
        run_local_hybrid_ingest(
            root_dir=root_dir,
            python_cmd=args.python,
            org=org,
            repo=repo,
            repo_path=repo_paths[repo],
            dry_run=args.dry_run,
        )

    print("\n[3/4] Run synthetic step")
    run_synthetic_step(
        root_dir=root_dir,
        python_cmd=args.python,
        step=args.synthetic_step,
        count=args.synthetic_count,
        personas=args.synthetic_personas,
        dry_run=args.dry_run,
    )

    print("\n[4/4] Ingest documents into ChromaDB")
    for repo in selected_repos:
        run_chroma_ingest(
            root_dir=root_dir,
            python_cmd=args.python,
            org=org,
            repo=repo,
            query_test=not args.no_query_test,
            dry_run=args.dry_run,
        )

    print("\n" + "=" * 72)
    print("ORG PIPELINE COMPLETE")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nERROR: {exc}")
        raise SystemExit(1)
