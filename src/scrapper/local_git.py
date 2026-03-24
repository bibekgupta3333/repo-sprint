"""Local git repository scraper — extracts commits and diffs from a cloned repo.

Produces the same data structures as GitHubScraper so that downstream
SprintPreprocessor / ChromaFormatter / FeatureExtractor work unchanged.

Usage:
    scraper = LocalGitScraper("/path/to/repos/go")
    commits = scraper.get_commits(since="2026-03-01", until="2026-03-24")
    diff    = scraper.get_commit_diff("abc1234")
"""

import os
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List


# --------------------------------------------------------------------------- #
#  Language detection (mirrors GitHubScraper._extract_language_from_filename)  #
# --------------------------------------------------------------------------- #

_EXT_TO_LANG = {
    ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript",
    ".jsx": "JSX", ".tsx": "TSX", ".go": "Go", ".java": "Java",
    ".cpp": "C++", ".c": "C", ".h": "C Header", ".rs": "Rust",
    ".rb": "Ruby", ".php": "PHP", ".cs": "C#", ".swift": "Swift",
    ".kt": "Kotlin", ".scala": "Scala", ".r": "R", ".sql": "SQL",
    ".sh": "Shell", ".json": "JSON", ".yaml": "YAML", ".yml": "YAML",
    ".xml": "XML", ".html": "HTML", ".css": "CSS", ".scss": "SCSS",
    ".md": "Markdown", ".txt": "Text", ".vim": "Vim", ".s": "Assembly",
}


def _lang_from_filename(filename: str) -> str:
    """Map file extension to language label."""
    ext = os.path.splitext(filename)[1].lower()
    return _EXT_TO_LANG.get(ext, "Other")


# --------------------------------------------------------------------------- #
#  Git CLI helpers                                                            #
# --------------------------------------------------------------------------- #

def _run_git(repo_path: str, args: list[str], timeout: int = 120) -> str:
    """Run a git command and return stdout.

    Reads output as bytes and decodes with ``errors='replace'`` so that
    binary content in patches (images, test fixtures) doesn't crash.

    Raises ``subprocess.CalledProcessError`` on non-zero exit.
    """
    cmd = ["git", "-C", repo_path] + args
    result = subprocess.run(
        cmd,
        capture_output=True,
        timeout=timeout,
    )
    stdout = result.stdout.decode("utf-8", errors="replace")
    stderr = result.stderr.decode("utf-8", errors="replace")
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd, stdout, stderr,
        )
    return stdout


# --------------------------------------------------------------------------- #
#  Commit log separator — we use a unique delimiter so that multi-line        #
#  commit messages don't break the parser.                                    #
# --------------------------------------------------------------------------- #

_FIELD_SEP = "†"          # separates fields within a record
_RECORD_SEP = "‡‡‡END‡‡‡"  # separates records
_BODY_SEP = "‡‡‡BODY‡‡‡"   # separates commit body from header fields

_LOG_FORMAT = _FIELD_SEP.join([
    "%H",     # full SHA
    "%h",     # short SHA
    "%s",     # subject (first line of message)
    "%an",    # author name
    "%ae",    # author email
    "%aI",    # author date ISO 8601
]) + _BODY_SEP + "%b" + _RECORD_SEP

# Separator for batch diff extraction via `git log --numstat`
_BATCH_COMMIT_SEP = "‡‡‡COMMIT‡‡‡"


# --------------------------------------------------------------------------- #
#  LocalGitScraper                                                            #
# --------------------------------------------------------------------------- #

class LocalGitScraper:
    """Extract commits and code diffs from a local git clone.

    Designed as a drop-in replacement for the *commit* and *diff* portions of
    ``GitHubScraper`` / ``Scraper``.  Issues and PRs must still come from the
    GitHub API (see hybrid ingestion in ``scripts/ingest_local.py``).
    """

    def __init__(self, repo_path: str):
        self.repo_path = str(Path(repo_path).resolve())
        if not Path(self.repo_path, ".git").exists():
            raise FileNotFoundError(
                f"Not a git repository: {self.repo_path}"
            )

    # ------------------------------------------------------------------ #
    #  Repository metadata                                                #
    # ------------------------------------------------------------------ #

    def get_repo_info(self, owner: str, repo: str) -> dict:
        """Return basic repository metadata derived from the local clone.

        Fields that only exist in the GitHub API (stars, forks, description)
        are set to sensible defaults so that the ``RepoData`` schema is valid.
        """
        # Detect primary language from tracked files
        language = self._detect_primary_language()

        remote_url = ""
        try:
            remote_url = _run_git(
                self.repo_path, ["remote", "get-url", "origin"]
            ).strip()
        except Exception:
            pass

        html_url = remote_url or f"https://github.com/{owner}/{repo}"
        # Normalise SSH style URLs to HTTPS
        if html_url.startswith("git@github.com:"):
            html_url = html_url.replace(
                "git@github.com:", "https://github.com/"
            )
        if html_url.endswith(".git"):
            html_url = html_url[:-4]

        return {
            "owner": owner,
            "name": repo,
            "url": html_url,
            "stars": 0,
            "forks": 0,
            "language": language,
            "description": "",
        }

    # ------------------------------------------------------------------ #
    #  Commits                                                            #
    # ------------------------------------------------------------------ #

    def get_commits(
        self,
        since: Optional[str] = None,
        until: Optional[str] = None,
        limit: Optional[int] = None,
        branch: str = "HEAD",
    ) -> List[dict]:
        """Extract commits via ``git log``.

        Returns a list of dicts matching the ``CommitData`` schema used by
        ``scripts/_core/scraper.parse_commit``.
        """
        args = [
            "log",
            branch,
            f"--format={_LOG_FORMAT}",
        ]
        if since:
            args.append(f"--since={since}")
        if until:
            args.append(f"--until={until}")
        if limit:
            args.append(f"-n{limit}")

        raw = _run_git(self.repo_path, args, timeout=300)
        commits = self._parse_log_output(raw)
        return commits

    def get_commit_count(
        self,
        since: Optional[str] = None,
        until: Optional[str] = None,
        branch: str = "HEAD",
    ) -> int:
        """Fast commit count without parsing full log."""
        args = ["rev-list", "--count", branch]
        if since:
            args.extend([f"--since={since}"])
        if until:
            args.extend([f"--until={until}"])
        return int(_run_git(self.repo_path, args).strip())

    # ------------------------------------------------------------------ #
    #  Commit diffs                                                       #
    # ------------------------------------------------------------------ #

    def get_commit_diff(self, sha: str) -> Optional[dict]:
        """Extract detailed diff for a single commit via ``git show``.

        Returns a dict matching the ``CommitDiffData`` schema, or ``None`` if
        the commit cannot be found.
        """
        if not sha:
            return None

        try:
            # Get commit metadata
            meta_raw = _run_git(self.repo_path, [
                "show", sha, "--format=%H†%s†%an†%aI", "--no-patch",
            ])
            parts = meta_raw.strip().split("†")
            if len(parts) < 4:
                return None

            full_sha, message, author, date_str = (
                parts[0], parts[1], parts[2], parts[3],
            )

            # Get numstat (additions  deletions  filename)
            numstat_raw = _run_git(self.repo_path, [
                "show", sha, "--numstat", "--format=",
            ])

            # Get name-status (status  filename)
            status_raw = _run_git(self.repo_path, [
                "diff-tree", "--no-commit-id", "-r", "--name-status", sha,
            ])

            # Build status lookup
            status_map: dict[str, str] = {}
            for line in status_raw.strip().splitlines():
                if not line.strip():
                    continue
                status_parts = line.split("\t", 1)
                if len(status_parts) == 2:
                    status_code, fname = status_parts
                    status_map[fname] = self._git_status_to_api(status_code)

            # Parse numstat lines
            file_diffs, language_breakdown, total_additions, total_deletions = (
                self._parse_numstat_lines(numstat_raw, status_map)
            )

            return {
                "sha": full_sha[:7],
                "message": message[:200],
                "author": author,
                "created_at": date_str,
                "total_additions": total_additions,
                "total_deletions": total_deletions,
                "files_changed": len(file_diffs),
                "file_diffs": file_diffs,
                "language_breakdown": dict(language_breakdown),
            }

        except Exception as e:
            print(f"  Warning: Could not get diff for {sha}: {e}")
            return None

    def get_commit_diffs_batch(
        self,
        since: Optional[str] = None,
        until: Optional[str] = None,
        limit: Optional[int] = None,
        branch: str = "HEAD",
        include_patches: bool = True,
        verbose: bool = True,
    ) -> list[dict]:
        """Extract diffs for many commits in a SINGLE git process.

        Uses ``git log --numstat -p`` which outputs commit metadata, per-file
        stats, AND full patch text together.  This replaces N individual
        ``git show`` calls with one command — **100x+ faster** for large repos.

        Args:
            include_patches: If True, include the ``patch`` text per file
                (uses ``-p``).  Set to False for faster runs when patches
                are not needed.

        Returns a list of ``CommitDiffData``-compatible dicts.
        """
        fmt = f"{_BATCH_COMMIT_SEP}%n%H{_FIELD_SEP}%s{_FIELD_SEP}%an{_FIELD_SEP}%aI"
        args = [
            "log", branch,
            f"--format={fmt}",
            "--numstat",
        ]
        if include_patches:
            args.append("-p")
        if since:
            args.append(f"--since={since}")
        if until:
            args.append(f"--until={until}")
        if limit:
            args.append(f"-n{limit}")

        label = "git log --numstat -p" if include_patches else "git log --numstat"
        if verbose:
            print(f"  Extracting diffs in batch via {label} ...")

        raw = _run_git(self.repo_path, args, timeout=1800)

        # Split output into per-commit blocks
        blocks = raw.split(_BATCH_COMMIT_SEP)
        diffs: list[dict] = []
        seen: set[str] = set()

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            lines = block.split("\n")
            # First line is the metadata
            meta_line = lines[0].strip()
            meta_parts = meta_line.split(_FIELD_SEP)
            if len(meta_parts) < 4:
                continue

            full_sha = meta_parts[0].strip()
            message = meta_parts[1].strip()
            author = meta_parts[2].strip()
            date_str = meta_parts[3].strip()

            short_sha = full_sha[:7]
            if short_sha in seen:
                continue
            seen.add(short_sha)

            # Separate numstat lines from patch lines.
            # numstat lines match: <digits>\t<digits>\t<filename>
            # patch sections start with "diff --git a/... b/..."
            remaining = lines[1:]
            numstat_lines: list[str] = []
            patch_lines: list[str] = []
            in_patch = False

            for line in remaining:
                if not in_patch:
                    stripped = line.strip()
                    if stripped.startswith("diff --git "):
                        in_patch = True
                        patch_lines.append(line)
                    elif stripped:
                        numstat_lines.append(line)
                else:
                    patch_lines.append(line)

            numstat_text = "\n".join(numstat_lines)
            file_diffs, language_breakdown, total_add, total_del = (
                self._parse_numstat_lines(numstat_text)
            )

            # Parse patches and attach to file_diffs
            if include_patches and patch_lines:
                file_patches = self._split_patches(patch_lines)
                for fd in file_diffs:
                    fd["patch"] = file_patches.get(fd["filename"], "")

            diffs.append({
                "sha": short_sha,
                "sha_full": full_sha,
                "message": message[:200],
                "author": author,
                "created_at": date_str,
                "total_additions": total_add,
                "total_deletions": total_del,
                "files_changed": len(file_diffs),
                "file_diffs": file_diffs,
                "language_breakdown": dict(language_breakdown),
            })

        if verbose:
            print(f"  Batch extracted {len(diffs)} commit diffs"
                  f"{' (with patches)' if include_patches else ''}")

        return diffs

    @staticmethod
    def _split_patches(patch_lines: list[str]) -> dict[str, str]:
        """Split combined patch output into per-file patches.

        Returns a dict mapping filename → patch text.
        """
        file_patches: dict[str, str] = {}
        current_file: Optional[str] = None
        current_chunk: list[str] = []

        for line in patch_lines:
            m = re.match(r"^diff --git a/(.+) b/(.+)$", line)
            if m:
                if current_file is not None:
                    file_patches[current_file] = "\n".join(current_chunk)
                current_file = m.group(2)
                current_chunk = [line]
            else:
                current_chunk.append(line)

        if current_file is not None:
            file_patches[current_file] = "\n".join(current_chunk)

        return file_patches

    def get_commit_diff_with_patch(self, sha: str) -> Optional[dict]:
        """Like ``get_commit_diff`` but also includes the ``patch`` text.

        This is slower because it runs ``git show -p`` which can produce
        large output for big commits.
        """
        diff = self.get_commit_diff(sha)
        if diff is None:
            return None

        try:
            patch_raw = _run_git(self.repo_path, [
                "show", sha, "--format=", "-p",
            ], timeout=30)

            # Split patch by file headers ("diff --git a/... b/...")
            file_patches: dict[str, str] = {}
            current_file = None
            current_lines: list[str] = []

            for line in patch_raw.splitlines():
                m = re.match(r"^diff --git a/(.+) b/(.+)$", line)
                if m:
                    if current_file is not None:
                        file_patches[current_file] = "\n".join(current_lines)
                    current_file = m.group(2)
                    current_lines = [line]
                else:
                    current_lines.append(line)
            if current_file is not None:
                file_patches[current_file] = "\n".join(current_lines)

            # Attach patches to file_diffs
            for fd in diff["file_diffs"]:
                fd["patch"] = file_patches.get(fd["filename"], "")

        except Exception:
            pass  # file_diffs keep empty patch strings

        return diff

    # ------------------------------------------------------------------ #
    #  Contributor statistics                                             #
    # ------------------------------------------------------------------ #

    def get_contributors(
        self,
        since: Optional[str] = None,
        until: Optional[str] = None,
    ) -> List[dict]:
        """Return contributor commit counts via ``git shortlog``."""
        args = ["shortlog", "-sne", "--all"]
        if since:
            args.append(f"--since={since}")
        if until:
            args.append(f"--until={until}")

        raw = _run_git(self.repo_path, args, timeout=120)
        contributors = []
        for line in raw.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            # format: "  <count>\t<name> <email>"
            m = re.match(r"(\d+)\t(.+?)\s+<(.+?)>", line)
            if m:
                contributors.append({
                    "commits": int(m.group(1)),
                    "name": m.group(2),
                    "email": m.group(3),
                })
        return contributors

    # ------------------------------------------------------------------ #
    #  PRs & Issues from commit trailers (fully offline)                  #
    # ------------------------------------------------------------------ #

    def get_prs_from_commits(self, commits: List[dict]) -> List[dict]:
        """Extract PR/CL-like objects from commit body trailers.

        Parses structured metadata that exists in every merged commit:
        - ``Change-Id`` → unique CL identifier
        - ``Reviewed-on`` → code review URL (acts as PR link)
        - ``Reviewed-by`` → reviewers
        - ``Auto-Submit`` → auto-merge flag
        - ``Fixes/Updates #N`` → linked issues

        Returns a list of dicts matching the PR schema expected by
        downstream components (Processor, Analyzer, etc.).
        """
        prs: list[dict] = []
        seen_change_ids: set[str] = set()

        for commit in commits:
            body = commit.get("body", "")
            if not body:
                continue

            # Extract Change-Id (unique per CL/PR)
            change_id_match = re.search(r"Change-Id:\s+(\S+)", body)
            if not change_id_match:
                continue  # Not a CL-style commit

            change_id = change_id_match.group(1)
            if change_id in seen_change_ids:
                continue  # Deduplicate
            seen_change_ids.add(change_id)

            # Extract review URL
            review_match = re.search(
                r"Reviewed-on:\s+(https?://\S+)", body,
            )
            review_url = review_match.group(1) if review_match else ""

            # Extract CL number from review URL
            cl_number = 0
            cl_match = re.search(r"/\+/(\d+)", review_url)
            if cl_match:
                cl_number = int(cl_match.group(1))

            # Extract reviewers
            reviewers = re.findall(
                r"Reviewed-by:\s+(.+?)\s*<(.+?)>", body,
            )

            # Extract labels from commit subject prefix (e.g., "net/http:")
            subject = commit.get("message", "")
            labels: list[str] = []
            prefix_match = re.match(r"^([\w/]+):", subject)
            if prefix_match:
                labels.append(prefix_match.group(1))

            # Detect auto-submit
            auto_submit = "Auto-Submit:" in body

            # Extract linked issue numbers
            issue_refs = re.findall(
                r"(?:Fixes|Updates|Closes|Resolves)\s+#(\d+)", body,
            )

            # LUCI CI result
            ci_passed = "LUCI-TryBot-Result:" in body

            pr_data = {
                "number": cl_number,
                "title": subject[:200],
                "state": "merged",  # On main branch = merged
                "created_at": commit.get("created_at", ""),
                "updated_at": commit.get("created_at", ""),
                "merged_at": commit.get("created_at", ""),
                "closed_at": commit.get("created_at", ""),
                "author": commit.get("author", {"login": "", "url": ""}),
                "url": review_url,
                "labels": labels,
                "body": body.split("Change-Id:")[0].strip(),
                "additions": 0,
                "deletions": 0,
                "commits": [{
                    "sha": commit.get("sha", ""),
                    "message": subject[:200],
                }],
                "file_diffs": [],
                "reviewers": [
                    {"login": name.strip(), "email": email}
                    for name, email in reviewers
                ],
                "change_id": change_id,
                "auto_submit": auto_submit,
                "ci_passed": ci_passed,
                "related_issues": [int(n) for n in issue_refs],
            }

            # Enrich with diff stats if available
            diff = commit.get("diff")
            if diff:
                pr_data["additions"] = diff.get("total_additions", 0)
                pr_data["deletions"] = diff.get("total_deletions", 0)
                pr_data["file_diffs"] = diff.get("file_diffs", [])

            prs.append(pr_data)

        return prs

    def get_issues_from_commits(self, commits: List[dict]) -> List[dict]:
        """Extract issue references from commit bodies.

        Parses ``Fixes #N`` / ``Updates #N`` / ``Closes #N`` patterns to
        build issue-like objects.  Limited to what git provides — no title
        or body from GitHub, but we get the issue number, state, and which
        commits reference it.
        """
        issue_map: dict[int, dict] = {}

        for commit in commits:
            body = commit.get("body", "")
            if not body:
                continue

            # Parse "Fixes #N" -> issue is closed by this commit
            for m in re.finditer(r"Fixes\s+#(\d+)", body):
                num = int(m.group(1))
                if num not in issue_map:
                    issue_map[num] = {
                        "number": num,
                        "title": f"Issue #{num}",
                        "state": "closed",
                        "created_at": commit.get("created_at", ""),
                        "updated_at": commit.get("created_at", ""),
                        "closed_at": commit.get("created_at", ""),
                        "author": commit.get("author", {"login": "", "url": ""}),
                        "url": f"https://github.com/golang/go/issues/{num}",
                        "labels": [],
                        "body": "",
                        "related_prs": [],
                    }
                issue_map[num]["related_prs"].append(commit.get("sha", ""))

            # Parse "Updates #N" -> issue is still open
            for m in re.finditer(r"(?:Updates|Closes|Resolves)\s+#(\d+)", body):
                num = int(m.group(1))
                if num not in issue_map:
                    issue_map[num] = {
                        "number": num,
                        "title": f"Issue #{num}",
                        "state": "open",
                        "created_at": commit.get("created_at", ""),
                        "updated_at": commit.get("created_at", ""),
                        "closed_at": None,
                        "author": commit.get("author", {"login": "", "url": ""}),
                        "url": f"https://github.com/golang/go/issues/{num}",
                        "labels": [],
                        "body": "",
                        "related_prs": [],
                    }
                issue_map[num]["related_prs"].append(commit.get("sha", ""))

        return list(issue_map.values())

    # ------------------------------------------------------------------ #
    #  Full scrape (commits + diffs)                                      #
    # ------------------------------------------------------------------ #

    def scrape(
        self,
        owner: str,
        repo: str,
        fetch_diffs: bool = False,
        since: Optional[str] = None,
        until: Optional[str] = None,
        limit: Optional[int] = None,
        verbose: bool = True,
        skip_local_prs: bool = False,
        skip_local_issues: bool = False,
    ) -> dict:
        """Extract commits and optionally diffs from the local clone.

        Returns the same ``RepoData`` schema as ``GitHubScraper.scrape()``,
        with ``issues`` and ``prs`` populated from commit body trailers
        (Change-Id, Reviewed-on, Reviewed-by, Fixes/Updates #NNN).

        Diffs are extracted in batch using a single ``git log --numstat``
        command instead of spawning a separate process per commit.
        """
        if verbose:
            total = self.get_commit_count(since=since, until=until)
            effective = min(total, limit) if limit else total
            print(f"  Local repo: {self.repo_path}")
            print(f"  Commits in range: {total}" + (
                f" (limited to {effective})" if limit and limit < total else ""
            ))

        repo_info = self.get_repo_info(owner, repo)

        if verbose:
            print("  Extracting commits via git log ...")
        commits = self.get_commits(
            since=since, until=until, limit=limit,
        )
        if verbose:
            print(f"  Extracted {len(commits)} commits")

        # Build diffs — batch mode (single git process)
        commit_diffs: list[dict] = []
        if fetch_diffs:
            commit_diffs = self.get_commit_diffs_batch(
                since=since, until=until, limit=limit, verbose=verbose,
            )

            # Build lookup and attach diffs to commits
            diff_by_short: dict[str, dict] = {
                d["sha"][:7]: d for d in commit_diffs
            }
            for commit in commits:
                short = (commit.get("sha") or "")[:7]
                commit["diff"] = diff_by_short.get(short)
        else:
            for c in commits:
                c["diff"] = None

        # Extract PRs and issues from commit trailers (fully local)
        prs: list[dict] = []
        issues: list[dict] = []

        if not skip_local_prs or not skip_local_issues:
            if verbose:
                print("  Extracting PRs/CLs from commit trailers ...")
            if not skip_local_prs:
                prs = self.get_prs_from_commits(commits)
            if not skip_local_issues:
                issues = self.get_issues_from_commits(commits)
            if verbose:
                print(f"  Found {len(prs)} PRs/CLs, "
                      f"{len(issues)} issue refs")
        elif verbose:
            print("  Skipping local PR/issue extraction (flags set)")

        result = {
            **repo_info,
            "issues": issues,
            "prs": prs,
            "commits": commits,
            "commit_diffs": commit_diffs,
        }
        return result

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _parse_log_output(self, raw: str) -> list[dict]:
        """Parse ``git log`` output produced with ``_LOG_FORMAT``."""
        commits = []
        for record in raw.split(_RECORD_SEP):
            record = record.strip()
            if not record:
                continue

            # Split header fields from body
            if _BODY_SEP in record:
                header_part, body = record.split(_BODY_SEP, 1)
            else:
                header_part = record
                body = ""

            parts = header_part.split(_FIELD_SEP)
            if len(parts) < 6:
                continue

            full_sha = parts[0].strip()
            short_sha = parts[1].strip()
            message = parts[2].strip()
            author_name = parts[3].strip()
            author_email = parts[4].strip()
            date_str = parts[5].strip()

            commits.append({
                "sha": short_sha or full_sha[:7],
                "sha_full": full_sha,
                "message": message[:200],
                "author": {
                    "login": author_name,
                    "url": "",
                },
                "created_at": date_str,
                "url": "",
                "body": body.strip(),
                "diff": None,
            })
        return commits

    @staticmethod
    def _parse_numstat_lines(
        numstat_text: str,
        status_map: Optional[dict] = None,
    ) -> tuple[list[dict], dict[str, int], int, int]:
        """Parse ``--numstat`` output into file_diffs + language breakdown.

        Returns (file_diffs, language_breakdown, total_additions, total_deletions).
        """
        file_diffs: list[dict] = []
        language_breakdown: dict[str, int] = defaultdict(int)
        total_additions = 0
        total_deletions = 0

        for line in numstat_text.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            # numstat format: additions\tdeletions\tfilename
            num_parts = line.split("\t")
            if len(num_parts) < 3:
                continue

            add_str, del_str, filename = (
                num_parts[0], num_parts[1], num_parts[2],
            )

            # Binary files show "-" for additions/deletions
            additions = int(add_str) if add_str != "-" else 0
            deletions = int(del_str) if del_str != "-" else 0

            lang = _lang_from_filename(filename)
            language_breakdown[lang] += additions + deletions
            total_additions += additions
            total_deletions += deletions

            file_status = "modified"
            if status_map:
                file_status = status_map.get(filename, "modified")

            file_diffs.append({
                "filename": filename,
                "status": file_status,
                "additions": additions,
                "deletions": deletions,
                "changes": additions + deletions,
                "patch": "",
            })

        return file_diffs, dict(language_breakdown), total_additions, total_deletions

    def _detect_primary_language(self) -> str:
        """Detect the primary language by sampling tracked file extensions."""
        try:
            raw = _run_git(self.repo_path, [
                "ls-files", "--cached",
            ], timeout=30)
            counts: dict[str, int] = defaultdict(int)
            for line in raw.splitlines():
                lang = _lang_from_filename(line.strip())
                if lang != "Other":
                    counts[lang] += 1
            if counts:
                return max(counts, key=counts.get)
        except Exception:
            pass
        return ""

    @staticmethod
    def _git_status_to_api(code: str) -> str:
        """Map git diff-tree status code to GitHub API status string."""
        mapping = {
            "A": "added",
            "D": "removed",
            "M": "modified",
            "R": "renamed",
            "C": "copied",
            "T": "changed",
        }
        return mapping.get(code[0].upper(), "modified")
