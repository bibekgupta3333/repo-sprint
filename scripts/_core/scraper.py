"""Core scraper module."""
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple
import json
import os
import re
import time
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
        'body': body_text,
        'related_prs': [],
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
        'deletions': pr_data.get('deletions', 0),
        'commits': [],
        'file_diffs': [],
    }


def parse_pr_from_issue(item: dict) -> dict:
    """Parse PR data from the ``/issues`` endpoint response.

    The ``/issues`` endpoint includes a ``pull_request`` sub-object with
    ``merged_at``.  ``additions``/``deletions`` are not available and
    default to 0.
    """
    body = item.get('body')
    body_text = body[:500] if body else None
    pr_sub = item.get('pull_request', {})

    return {
        'number': item['number'],
        'title': item['title'],
        'state': item['state'],
        'created_at': item['created_at'],
        'updated_at': item['updated_at'],
        'merged_at': pr_sub.get('merged_at'),
        'closed_at': item.get('closed_at'),
        'author': parse_author(item.get('user')),
        'url': item['html_url'],
        'labels': [
            label['name'] for label in item.get('labels', [])
        ],
        'body': body_text,
        'additions': 0,
        'deletions': 0,
        'commits': [],
        'file_diffs': [],
    }


def parse_commit(commit_data: dict) -> dict:
    """Parse commit from GitHub API."""
    commit_info = commit_data.get('commit', {})
    author_info = commit_info.get('author', {})

    full_sha = commit_data.get('sha', '') or ''
    return {
        'sha': full_sha[:7] if full_sha else '',
        'sha_full': full_sha,
        'message': commit_info.get('message', '').split('\n')[0][:200],
        'author': parse_author(commit_data.get('author')),
        'created_at': author_info.get('date', datetime.now().isoformat()),
        'url': commit_data.get('html_url', ''),
    }


# PR body/title patterns that link a PR to an issue (GitHub closing keywords).
_ISSUE_LINK_PATTERN = re.compile(
    r'(?:close|closes|closed|fix|fixes|fixed|resolve|resolves|resolved)\s+#(\d+)',
    re.IGNORECASE,
)


def link_issues_to_pull_requests(
    issues: List[dict], prs: List[dict],
) -> None:
    """Populate ``issue['related_prs']`` via closing keywords.

    Uses a two-pass O(issues + PRs) approach:
      1. Scan every PR's title+body once → build a dict
         mapping referenced issue numbers to PR numbers.
      2. For each issue, look up the dict.

    This avoids the previous O(issues × PRs) nested loop
    that was ~3.5 billion iterations for golang/go.
    """
    # Pass 1: PR text → extract referenced issue numbers
    issue_to_prs: dict = {}  # issue_num → set of PR nums
    for pr in prs:
        text = f"{pr.get('title', '')}\n{pr.get('body') or ''}"
        pr_num = pr['number']
        for match in _ISSUE_LINK_PATTERN.finditer(text):
            ref = int(match.group(1))
            if ref not in issue_to_prs:
                issue_to_prs[ref] = set()
            issue_to_prs[ref].add(pr_num)

    # Pass 2: assign to each issue
    for issue in issues:
        linked = issue_to_prs.get(issue['number'])
        issue['related_prs'] = sorted(linked) if linked else []


# GitHub REST API limits page-based pagination to 1000 results
# (10 pages × 100 per_page).  This constant caps the inner loop.
_MAX_PAGES_PER_WINDOW = 10


class Scraper:
    """GitHub API scraper with rate-limit resilience."""

    BASE_URL = "https://api.github.com"

    def __init__(
        self,
        throttle_delay: float = 0.1,
        max_retries: int = 3,
    ):
        """Initialize scraper.

        Args:
            throttle_delay: Seconds to wait between successive
                API calls (prevents bursting into rate limits).
            max_retries: Number of times to retry on 429/403
                rate-limit errors before giving up.
        """
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'ResearchBot/1.0',
        })
        self.throttle_delay = throttle_delay
        self.max_retries = max_retries
        self._last_request_time: float = 0.0

        token = (
            os.environ.get('GITHUB_TOKEN')
            or os.environ.get('GH_TOKEN')
        )
        if token:
            self.session.headers[
                'Authorization'] = f'token {token}'
            print("  Using GITHUB_TOKEN "
                  "(5,000 requests/hr)")
        else:
            print("  Warning: No GITHUB_TOKEN set — "
                  "rate limit is 60 requests/hr. "
                  "Set GITHUB_TOKEN or GH_TOKEN "
                  "for 5,000/hr.")

    # ---------------------------------------------------------- #
    #  Throttled + retrying HTTP GET                              #
    # ---------------------------------------------------------- #

    def _throttle(self):
        """Enforce minimum delay between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.throttle_delay:
            time.sleep(self.throttle_delay - elapsed)

    @staticmethod
    def _rate_limit_wait(response) -> float:
        """Compute how long to wait from rate-limit headers.

        GitHub sends either ``Retry-After`` (seconds) or
        ``x-ratelimit-reset`` (UTC epoch) on 429/403.
        """
        retry_after = response.headers.get('Retry-After')
        if retry_after:
            try:
                return max(float(retry_after), 1.0)
            except ValueError:
                pass

        reset_epoch = response.headers.get(
            'x-ratelimit-reset')
        if reset_epoch:
            try:
                wait = float(reset_epoch) - time.time()
                return max(wait, 1.0)
            except ValueError:
                pass

        return 0.0  # no header found

    def get(self, url: str, params: dict = None):
        """Make GET request with rate-limit retry.

        On 429 or 403 (rate limit), the method reads
        GitHub's ``Retry-After`` / ``x-ratelimit-reset``
        headers and sleeps accordingly.  If no header is
        present it uses exponential backoff (2 / 4 / 8 s).
        """
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                response = self.session.get(
                    url, params=params, timeout=30)
                self._last_request_time = time.time()

                if response.status_code in [404, 500]:
                    return None

                # ---- Rate-limit handling ----
                if response.status_code in [429, 403]:
                    remaining = response.headers.get(
                        'x-ratelimit-remaining', '')
                    is_rate_limit = (
                        response.status_code == 429
                        or remaining == '0'
                    )
                    if is_rate_limit:
                        wait = self._rate_limit_wait(
                            response)
                        if wait <= 0:
                            # Exponential backoff fallback
                            wait = 2 ** (attempt + 1)
                        if attempt < self.max_retries:
                            print(
                                f"  Rate limited — "
                                f"waiting {wait:.0f}s "
                                f"(retry "
                                f"{attempt + 1}/"
                                f"{self.max_retries})")
                            time.sleep(wait)
                            continue
                        # Final attempt exhausted
                        print(
                            "  Rate limit: retries "
                            "exhausted, skipping")
                        return None

                # ---- 422: pagination limit ----
                if response.status_code == 422:
                    return None

                response.raise_for_status()
                return response.json()

            except requests.exceptions.ConnectionError:
                if attempt < self.max_retries:
                    wait = 2 ** (attempt + 1)
                    print(f"  Connection error — "
                          f"retrying in {wait}s")
                    time.sleep(wait)
                    continue
                print("  Connection error: "
                      "retries exhausted")
                return None
            except Exception as e:
                print(f"  Request failed: {e}")
                return None
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

    def get_issues_and_prs(
        self,
        owner: str,
        repo: str,
        issues_limit: int = 50,
        prs_limit: int = 50,
    ) -> Tuple[List[dict], List[dict]]:
        """Fetch issues and PRs via the ``/issues`` endpoint.

        GitHub limits page-based pagination to 1000 results
        (10 pages × 100 per_page).  To retrieve more, this method
        uses **date-based windowing**: it sorts by ``updated``
        ascending and advances a ``since`` cursor each time a
        10-page window is exhausted, effectively removing the
        1000-item ceiling.

        Phase 1 — ``/issues`` (returns both issues and PRs):
            Collects parsed issues and raw PR items.

        Phase 2 — ``/pulls`` (up to 1000 most-recent PRs):
            Enriches PR items with full metadata
            (``additions``, ``deletions``).  Any PRs beyond the
            ``/pulls`` window are parsed from the ``/issues`` data
            (``merged_at`` is available; ``additions``/``deletions``
            default to 0).

        Returns:
            ``(issues, prs)`` tuple.
        """
        issues: List[dict] = []
        pr_raw: dict = {}          # number → raw /issues item
        seen_numbers: set = set()  # dedup across windows
        since_cursor: Optional[str] = None

        # ---- Phase 1: /issues with date-window pagination ----
        while (len(issues) < issues_limit
               or len(pr_raw) < prs_limit):
            page = 1
            last_updated: Optional[str] = None
            new_in_window = 0

            while page <= _MAX_PAGES_PER_WINDOW:
                both_full = (len(issues) >= issues_limit
                             and len(pr_raw) >= prs_limit)
                if both_full:
                    break

                url = (f"{self.BASE_URL}/repos/"
                       f"{owner}/{repo}/issues")
                params: dict = {
                    'state': 'all',
                    'per_page': 100,
                    'page': page,
                    'sort': 'updated',
                    'direction': 'asc',
                }
                if since_cursor:
                    params['since'] = since_cursor

                data = self.get(url, params)
                if not data:
                    break

                for item in data:
                    num = item['number']
                    if num in seen_numbers:
                        continue
                    seen_numbers.add(num)
                    new_in_window += 1
                    last_updated = item.get('updated_at')

                    if 'pull_request' in item:
                        if len(pr_raw) < prs_limit:
                            pr_raw[num] = item
                    else:
                        if len(issues) < issues_limit:
                            try:
                                issues.append(
                                    parse_issue(item))
                            except Exception as e:
                                print("  Warning: Failed "
                                      f"to parse issue: {e}")

                if len(data) < 100:
                    # Last page — no more items in this window.
                    last_updated = None
                    break
                page += 1

            # Decide whether to open a new window.
            if last_updated is None:
                break   # API exhausted or last page was partial
            if new_in_window == 0:
                break   # No new items → stuck, avoid loop
            since_cursor = last_updated
            print(
                f"    ... {len(issues)} issues, "
                f"{len(pr_raw)} PRs so far "
                f"(windowing past {since_cursor})"
            )

        # ---- Phase 2: enrich PRs from /pulls -----------------
        prs: List[dict] = []
        enriched: set = set()

        if pr_raw:
            needed = set(pr_raw.keys())
            prs_page = 1
            while (len(enriched) < len(needed)
                   and prs_page <= _MAX_PAGES_PER_WINDOW):
                url = (f"{self.BASE_URL}/repos/"
                       f"{owner}/{repo}/pulls")
                data = self.get(url, {
                    'state': 'all',
                    'per_page': 100,
                    'page': prs_page,
                })
                if not data:
                    break

                for item in data:
                    num = item['number']
                    if num in needed and num not in enriched:
                        try:
                            prs.append(parse_pr(item))
                            enriched.add(num)
                        except Exception as e:
                            print("  Warning: Failed "
                                  f"to parse PR: {e}")

                if len(data) < 100:
                    break
                prs_page += 1

            # Fallback: parse remaining PRs from /issues data.
            missing = needed - enriched
            if missing:
                print(
                    f"  Note: {len(missing)} PR(s) enriched "
                    "from /issues data "
                    "(missing additions/deletions)"
                )
                for num in missing:
                    raw = pr_raw[num]
                    try:
                        prs.append(
                            parse_pr_from_issue(raw))
                    except Exception as e:
                        print("  Warning: Failed to parse "
                              f"PR #{num}: {e}")

        return issues, prs

    def get_commits(
        self,
        owner: str,
        repo: str,
        limit: int = 50,
        since: Optional[str] = None,
        until: Optional[str] = None,
    ) -> List[dict]:
        """Fetch commits."""
        commits = []
        page = 1

        while len(commits) < limit:
            url = f"{self.BASE_URL}/repos/{owner}/{repo}/commits"
            params = {'per_page': 100, 'page': page}
            if since:
                params['since'] = since
            if until:
                params['until'] = until
            data = self.get(url, params)

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

    def get_pull_commits(self, owner: str, repo: str, pr_number: int) -> List[dict]:
        """Fetch commits for a pull request (paginated)."""
        commits: List[dict] = []
        page = 1
        while True:
            url = f"{self.BASE_URL}/repos/{owner}/{repo}/pulls/{pr_number}/commits"
            data = self.get(url, {'per_page': 100, 'page': page})
            if not data:
                break
            for item in data:
                try:
                    commits.append(parse_commit(item))
                except Exception as e:
                    print(f"  Warning: Failed to parse PR #{pr_number} commit: {e}")
            if len(data) < 100:
                break
            page += 1
        return commits

    def get_pull_files(self, owner: str, repo: str, pr_number: int) -> List[dict]:
        """Fetch per-file patches for a pull request (paginated)."""
        files_out: List[dict] = []
        page = 1
        while True:
            url = f"{self.BASE_URL}/repos/{owner}/{repo}/pulls/{pr_number}/files"
            data = self.get(url, {'per_page': 100, 'page': page})
            if not data:
                break
            for item in data:
                files_out.append({
                    'filename': item.get('filename', ''),
                    'status': item.get('status', 'modified'),
                    'additions': item.get('additions', 0),
                    'deletions': item.get('deletions', 0),
                    'changes': item.get(
                        'changes',
                        item.get('additions', 0) + item.get('deletions', 0),
                    ),
                    'patch': item.get('patch') or '',
                })
            if len(data) < 100:
                break
            page += 1
        return files_out

    def ingest(self, owner: str, repo: str,
               issues_limit: int = 50,
               prs_limit: int = 50,
               commits_limit: int = 50,
               commit_since: Optional[str] = None,
               commit_until: Optional[str] = None) -> Optional[dict]:
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

        commits = self.get_commits(
            owner,
            repo,
            limit=commits_limit,
            since=commit_since,
            until=commit_until,
        )
        print(f"  Commits: {len(commits)}")

        link_issues_to_pull_requests(issues, prs)

        return {
            'owner': owner,
            'name': repo,
            'url': repo_data['html_url'],
            'description': repo_data.get('description'),
            'stars': repo_data['stargazers_count'],
            'forks': repo_data['forks_count'],
            'language': repo_data.get('language'),
            'issues': issues,
            'prs': prs,
            'commits': commits,
            'commit_diffs': []  # Will be populated by ingest.py if --diffs is used
        }

    def save(self, repo_data: dict, output_dir: str = 'data/raw') -> str:
        """Save to JSON file."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        filename = f"{output_dir}/{repo_data['owner']}_{repo_data['name']}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(repo_data, f, indent=2)
        print(f"Saved to {filename}\n")
        return filename
