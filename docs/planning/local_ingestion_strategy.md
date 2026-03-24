# Local Git Repository Ingestion Strategy

**Document**: Local Ingestion as a GitHub API Replacement  
**Date**: March 24, 2026  
**Related WBS Tasks**: 2.2 (GitHub Data Collection), 2.3 (Synthetic Data Generation), 2.4 (Feature Engineering)  
**Status**: 📋 Investigation Complete — Ready for Implementation

---

## 1. Problem Statement

### Current Limitation: GitHub API Rate Limits

The existing ingestion pipeline (`scripts/ingest.py`, `src/scrapper/github.py`) relies entirely on the GitHub REST API to collect:

| Data Type | API Endpoint | Calls per Entity |
|-----------|-------------|-----------------|
| Repository info | `GET /repos/{owner}/{repo}` | 1 |
| Issues | `GET /repos/{owner}/{repo}/issues` | 1 per page (50/page) |
| Pull Requests | `GET /repos/{owner}/{repo}/pulls` | 1 per page (50/page) |
| Commits | `GET /repos/{owner}/{repo}/commits` | 1 per page (50/page) |
| Commit diffs | `GET /repos/{owner}/{repo}/commits/{sha}` | **1 per commit** |
| PR commits | `GET /repos/{owner}/{repo}/pulls/{n}/commits` | 1 per PR |
| PR files | `GET /repos/{owner}/{repo}/pulls/{n}/files` | 1 per PR |

**Rate Limits** (documented in WBS `risks` section):

- **Without token**: 60 requests/hour → ~60 commits max per hour
- **With token**: 5,000 requests/hour → ~250 commits with diffs per hour
- For the **Go repo** (70,029 commits): ~**280 hours** at 5K rate to get all commit diffs

> [!WARNING]
> For our local Go clone (`repos/golang/go/`) with 70,029 commits, full GitHub API ingestion of diffs alone would consume **~14,000 API calls** (at 5 pages of 50) — requiring **2.8 hours** even WITH a token, and we can only get 250 paginated pages (12,500 commits) before throttling.

### What We Have Locally

The Go repository is cloned at `repos/golang/go/` with **full git history**:

- **70,029 commits** across all branches
- **220 commits** in the last 24 days (March 1-24, 2026)
- **Multiple branches** including `master`, `dev.*`, release tags (`go1.0` through `go1.24`)
- **Complete contributor data**: 7,861+ commits from top contributor (Russ Cox)
- **All file diffs** accessible locally via `git diff` / `git show`

---

## 2. Proposed Solution: `LocalGitScraper`

### 2.1 Architecture Overview

Create a `LocalGitScraper` class that mirrors the exact same `RepoData` output schema as `GitHubScraper` / `Scraper`, but reads everything from the local `.git` directory instead of making HTTP requests.

```
┌──────────────────────────────────────────────────────────┐
│                    Unified Interface                      │
│              ingest_repo(owner, repo, ...)                │
│                                                          │
│   ┌─────────────────┐        ┌─────────────────────┐    │
│   │  GitHubScraper   │   OR   │   LocalGitScraper    │    │
│   │  (API-based)     │        │   (git CLI-based)    │    │
│   │                  │        │                      │    │
│   │ • requests.get() │        │ • git log            │    │
│   │ • Rate limited   │        │ • git show           │    │
│   │ • 5K/hr ceiling  │        │ • git diff           │    │
│   │ • No patch depth │        │ • git shortlog       │    │
│   └────────┬─────────┘        └──────────┬───────────┘    │
│            │                             │               │
│            └──────────┬──────────────────┘               │
│                       ▼                                  │
│              Same RepoData schema                        │
│                       │                                  │
│                       ▼                                  │
│         SprintPreprocessor → ChromaFormatter              │
│                       │                                  │
│                       ▼                                  │
│         Training / Validation / Synthetic                 │
└──────────────────────────────────────────────────────────┘
```

### 2.2 Data Extraction via Git CLI

Every data point currently fetched from the API has a **local git equivalent**:

| GitHub API Data | Git CLI Equivalent | No Rate Limit |
|----------------|-------------------|:-------------:|
| `GET /commits` (list) | `git log --format=json` | ✅ |
| `GET /commits/{sha}` (diff) | `git show {sha} --stat --numstat --diff-filter=ACDMRT` | ✅ |
| `GET /pulls` (list) | Parse merge commits + refs/pull/* | ⚠️ Partial |
| `GET /issues` (list) | Not in git — needs GitHub API or local cache | ❌ |
| PR file diffs | `git diff {merge_base}..{head}` | ✅ |
| Contributor stats | `git shortlog -sne` | ✅ |
| Language breakdown | File extension analysis on changed files | ✅ |
| Code churn | `git log --numstat` | ✅ |

> [!IMPORTANT]
> **Issues and PR metadata** (title, body, labels, state) are NOT stored in git. Two strategies address this:
> 1. **Hybrid approach**: Use GitHub API only for issues/PRs (low volume, ~2-4 API pages), extract commits/diffs locally
> 2. **Full offline**: Cache issue/PR data from one API call, then reuse for all subsequent ingestion runs

### 2.3 Git Commands Mapping

#### Commits (replaces `GET /repos/{owner}/{repo}/commits`)

```bash
# Full commit log with all metadata (JSON-friendly)
git log --format='{"sha":"%H","sha_short":"%h","message":"%s","author":"%an","email":"%ae","date":"%aI"}' \
  --since="2026-03-01" --until="2026-03-24" \
  -n 500
```

#### Commit Diffs (replaces `GET /repos/{owner}/{repo}/commits/{sha}`)

```bash
# Per-commit numstat (additions/deletions per file)
git show {sha} --numstat --format=""

# Per-commit stat summary
git show {sha} --stat --format=""

# Full diff with patches (equivalent to API's "patch" field)
git show {sha} --format="" -p
```

#### File Changes per Commit (replaces `file_diffs` in `CommitDiffData`)

```bash
# Status + additions + deletions per file
git diff-tree --no-commit-id -r --numstat --diff-filter=ACDMRT {sha}

# File status (added/modified/deleted/renamed)
git diff-tree --no-commit-id -r --name-status {sha}
```

#### Contributor Analysis (replaces contributor API)

```bash
# All authors with commit counts
git shortlog -sne --all

# Authors in a date range
git shortlog -sne --since="2026-03-01" --until="2026-03-24"
```

#### PRs via Merge Commits (partial replacement)

```bash
# Find merge commits (often correspond to PR merges)
git log --merges --format='{"sha":"%H","message":"%s","date":"%aI","author":"%an"}' \
  --since="2026-03-01"

# PRs stored in reflog (if fetched)
git ls-remote origin 'refs/pull/*/head'
```

---

## 3. Implementation Plan

### 3.1 Module: `src/scrapper/local_git.py`

New file that implements the same interface as `GitHubScraper`:

```python
class LocalGitScraper:
    """Extract repository data from local git clone."""
    
    def __init__(self, repo_path: str, per_page: int = 50):
        self.repo_path = Path(repo_path)
        self.per_page = per_page
    
    def scrape(
        self,
        owner: str,
        repo: str,
        fetch_diffs: bool = False,
        pr_diff_limit: int = 20,
        since: str = None,
        until: str = None,
    ) -> RepoData:
        """Extract data from local git repo - same output as GitHubScraper.scrape()."""
        ...
    
    def _get_commits(self, since=None, until=None, limit=None) -> list[CommitData]:
        """Extract commits via git log."""
        ...
    
    def _get_commit_diff(self, sha: str) -> CommitDiffData:
        """Extract diff via git show (no API call!)."""
        ...
    
    def _get_merge_commits_as_prs(self) -> list[PRData]:
        """Parse merge commits as pseudo-PRs."""
        ...
    
    def _get_contributor_stats(self) -> dict:
        """Extract full contributor statistics."""
        ...
```

### 3.2 Module: `scripts/_core/local_scraper.py`

Core scraper that replaces `scripts/_core/scraper.py` for local repositories:

```python
class LocalScraper:
    """Local git-based scraper matching the Scraper interface."""
    
    def ingest(self, repo_path: str, owner: str, repo: str,
               commits_limit=None, commit_since=None, commit_until=None) -> dict:
        """Ingest from local git repo — same return schema as Scraper.ingest()."""
        ...
```

### 3.3 Updated Entry Point: `scripts/ingest_local.py`

```bash
# Usage examples:
python scripts/ingest_local.py repos/golang/go --owner golang --repo go
python scripts/ingest_local.py repos/golang/go --owner golang --repo go --diffs --days 30
python scripts/ingest_local.py repos/golang/go --owner golang --repo go --diffs --since 2026-01-01 --until 2026-03-24
```

### 3.4 Hybrid Mode: `scripts/ingest.py` Enhancement

Add `--local <path>` flag to existing ingest script:

```bash
# Hybrid: commits/diffs from local git, issues/PRs from API
python scripts/ingest.py golang go --local repos/golang/go --diffs

# Full local: everything from git (no issues/PR metadata)
python scripts/ingest.py golang go --local repos/golang/go --offline
```

---

## 4. Alignment with Research Objectives

### 4.1 Objective 1: Instant Setup Multi-Repository Sprint Intelligence

| Requirement | GitHub API | Local Git | Improvement |
|------------|-----------|----------|------------|
| Setup time | Needs token, rate-limited | Just `git clone` | **>10x faster** |
| 2-3 repos | ~6-8 hours with diffs | **Minutes** | **>50x faster** |
| Zero historical data | Limited to API pagination | Full history available | **Complete coverage** |

### 4.2 Objective 2: Ultra-Fast Blocker Detection with RAG

| Requirement | GitHub API | Local Git | Improvement |
|------------|-----------|----------|------------|
| Event-to-recommendation latency | 15-30s per API call chain | **<1s** per diff | **>30x faster** |
| Real-time capability | Limited by rate limits | Instant via `git pull` | **True real-time** |

### 4.3 Objective 3: Synthetic Data for Zero-History Startups

Local ingestion directly enables the synthetic data pipeline:

```
┌──────────────────┐     ┌────────────────────┐     ┌──────────────────┐
│  Local Git Clone  │────▶│  Real Sprint Data   │────▶│  Pattern Library  │
│  (repos/golang/go)       │     │  (70K+ commits)     │     │  (statistical)   │
└──────────────────┘     └────────────────────┘     └────────┬─────────┘
                                                              │
                                                              ▼
┌──────────────────┐     ┌────────────────────┐     ┌──────────────────┐
│  Synthetic 5K+   │◀────│  LLM Augmentation   │◀────│  SprintPersona   │
│  Sprint Dataset  │     │  (Ollama Local)     │     │  Templates       │
└──────────────────┘     └────────────────────┘     └──────────────────┘
```

- **Statistical patterns** from real Go repo commits feed into `SyntheticSprintGenerator` persona templates
- **Language breakdown**, **commit frequency**, **contributor distribution** from `git shortlog` calibrate synthetic ranges
- **Validation**: Compare synthetic distributions against real local data (no API needed)

### 4.4 Objective 4: Laptop-Scale LLM Architecture

| Resource | GitHub API Pipeline | Local Git Pipeline |
|----------|-------------------|--------------------|
| Network I/O | High (thousands of HTTP requests) | Near-zero (local disk) |
| RAM usage | Holds large JSON responses | Streaming git output | 
| Storage | Redundant: API cache + processed data | Git repo + processed data |
| CPU | Idle (waiting on network) | Efficient (local processing) |

### 4.5 Objective 5: Small Team Explainability

Local access provides **richer evidence** for explainability:

- **Full commit messages** (not truncated)
- **Complete file diffs** with patches (API limits patch size)
- **Branch history** and merge patterns
- **Tag/release correlation** with sprint timelines

---

## 5. Online vs. Offline Processes

### 5.1 Process Classification

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DATA PIPELINE MODES                            │
├─────────────────────────────────────┬───────────────────────────────────┤
│          OFFLINE PROCESSES          │        ONLINE PROCESSES           │
│       (Local Git + Cached Data)     │   (Requires Network / Live Data) │
├─────────────────────────────────────┼───────────────────────────────────┤
│                                     │                                   │
│  ✅ Commit extraction (git log)     │  🌐 Issues/PR metadata fetch     │
│  ✅ Diff computation (git show)     │  🌐 GitHub webhook listener      │
│  ✅ Contributor analysis            │  🌐 CI/CD metrics (GitHub Actions)│
│  ✅ Sprint grouping (preprocessor)  │  🌐 git pull (incremental sync)  │
│  ✅ Feature extraction (18 metrics) │  🌐 Ollama model download (once) │
│  ✅ Risk labeling                   │  🌐 PR state (open/merged/closed)│
│  ✅ ChromaDB embedding              │                                   │
│  ✅ Synthetic data generation       │                                   │
│  ✅ Training & validation           │                                   │
│  ✅ Ablation studies                │                                   │
│  ✅ Language breakdown              │                                   │
│  ✅ Code churn analysis             │                                   │
│                                     │                                   │
└─────────────────────────────────────┴───────────────────────────────────┘
```

### 5.2 Offline Pipeline (No Network Required)

```bash
# 1. INGEST — Extract from local clone (no API)
python scripts/ingest_local.py repos/golang/go --owner golang --repo go --diffs --offline

# 2. TRAIN — Generate synthetic data + train risk model
python src/data/synthetic_generator.py  # Calibrated from local stats
python scripts/prepare_training_data.py

# 3. VALIDATE — Cross-validation, ablation studies
python scripts/evaluate.py --cross-validate --ablation

# 4. EMBED — Index into ChromaDB for RAG
python scripts/prepare_embeddings.py
```

**What this produces:**
- `data/golang_go_sprints.json` — Sprint-aligned data with full metrics
- `data/golang_go_chroma_documents.json` — Vector-ready documents
- `data/synthetic_sprints.json` — 5K+ synthetic sprints calibrated to real patterns
- `data/processed/chromadb/` — Indexed vector store

### 5.3 Online Pipeline (Incremental Updates)

```bash
# 1. SYNC — Pull latest commits
cd repos/golang/go && git pull origin master

# 2. INCREMENTAL INGEST — Only new commits since last run
python scripts/ingest_local.py repos/golang/go --owner golang --repo go --diffs \
  --since "2026-03-20T00:00:00Z"

# 3. WEBHOOK MODE — Listen for real-time events (issues/PRs)
python scripts/webhook_listener.py --port 8000

# 4. HYBRID — Local diffs + API for issues/PRs
python scripts/ingest.py golang go --local repos/golang/go --diffs --days 7
```

### 5.4 Training & Validation Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TRAINING & VALIDATION FLOW                          │
│                                                                            │
│  ┌──────────────┐    ┌──────────────────┐    ┌────────────────────────┐    │
│  │  Local Git    │───▶│  Real Sprint Data │───▶│  Feature Extraction    │    │
│  │  repos/golang/go     │    │  (from git log)   │    │  (18 metrics/sprint)   │    │
│  └──────────────┘    └──────────────────┘    └──────────┬─────────────┘    │
│                                                          │                 │
│  ┌──────────────┐    ┌──────────────────┐               │                 │
│  │  Synthetic    │───▶│  5K+ Synthetic   │───▶ Combined  │                 │
│  │  Generator    │    │  Sprints         │    Dataset ◀──┘                 │
│  └──────────────┘    └──────────────────┘        │                        │
│                                                   │                        │
│                                        ┌──────────▼──────────┐            │
│                                        │  70% Train          │            │
│                                        │  15% Validation     │            │
│                                        │  15% Test           │            │
│                                        └──────────┬──────────┘            │
│                                                   │                        │
│                      ┌────────────────────────────┼────────────────┐       │
│                      ▼                            ▼                ▼       │
│              ┌──────────────┐           ┌──────────────┐  ┌────────────┐  │
│              │  Risk Model   │           │  Ablation    │  │  Cross-    │  │
│              │  Training     │           │  Studies     │  │  Validation│  │
│              │  (Ollama LLM) │           │  (per-agent) │  │  (5-fold)  │  │
│              └──────────────┘           └──────────────┘  └────────────┘  │
│                                                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Synthetic Data Generation from Local Repository

### 6.1 Calibrating `SyntheticSprintGenerator` with Real Data

The current `SyntheticSprintGenerator` uses hardcoded persona ranges. With local git data, we can **calibrate** these from real statistics:

```python
# Extract real statistics from local Go repo
git_stats = {
    "total_commits": 70029,
    "avg_commits_per_2_weeks": 220 / 1.7,  # ~130 commits per 2-week sprint
    "unique_authors": 20+,  # from git shortlog in date range
    "top_languages": {"Go": 85, "Assembly": 8, "C": 4, "Other": 3},
    "avg_files_per_commit": 3.2,  # from git log --numstat
}

# Calibrated personas (replace hardcoded values)
CALIBRATED_PERSONAS = [
    SprintPersona("go_typical",    (5, 20),  (10, 40), (80, 160),  (5, 15), weight=3.0),
    SprintPersona("go_release",    (2, 10),  (30, 80), (100, 300), (10, 25), weight=1.0),
    SprintPersona("go_quiet",      (0, 5),   (2, 10),  (10, 40),   (2, 5),  weight=1.0),
    SprintPersona("go_hotfix",     (1, 3),   (5, 15),  (5, 20),    (1, 3),  weight=1.0),
]
```

### 6.2 Enhanced Synthetic Generation Pipeline

```
Local Go Repo (repos/golang/go)
    │
    ├──▶ git log --numstat → Real commit size distribution
    ├──▶ git log --format → Real author/date distribution  
    ├──▶ git shortlog -sne → Real contributor patterns
    └──▶ git diff-tree + extension analysis → Language breakdown
         │
         ▼
    StatisticalProfiler
    (fits distributions to real data)
         │
         ├──▶ Commit count distribution (per 2-week window)
         ├──▶ File churn distribution (adds/deletes per commit)
         ├──▶ Author diversity distribution
         ├──▶ Merge frequency distribution
         └──▶ Language mix distribution
              │
              ▼
    CalibratedSyntheticGenerator
    (replaces hardcoded SyntheticSprintGenerator)
         │
         ├──▶ 5,000+ sprints matching real Go repo patterns
         ├──▶ Temporal realism (weekday/weekend patterns)
         ├──▶ Author collaboration patterns
         └──▶ Risk scenario injection (blocked, stalled, abandoned)
```

### 6.3 Validation: Synthetic vs. Real Data

```python
# Offline validation (no API needed)
from scipy.stats import ks_2samp, wasserstein_distance

def validate_synthetic_realism(real_sprints, synthetic_sprints):
    """Statistical comparison — all computed locally."""
    metrics = [
        "total_commits", "total_prs", "commit_frequency",
        "unique_authors", "issue_resolution_rate"
    ]
    results = {}
    for metric in metrics:
        real_vals = [s["metrics"][metric] for s in real_sprints]
        syn_vals = [s["metrics"][metric] for s in synthetic_sprints]
        
        ks_stat, p_value = ks_2samp(real_vals, syn_vals)
        w_dist = wasserstein_distance(real_vals, syn_vals)
        
        results[metric] = {
            "ks_statistic": ks_stat,
            "p_value": p_value,
            "wasserstein_distance": w_dist,
            "pass": p_value > 0.05  # Can't reject null hypothesis
        }
    return results
```

---

## 7. Data Mapping: GitHub API → Local Git

### 7.1 Complete Field Mapping

| `RepoData` Field | Source: GitHub API | Source: Local Git | Notes |
|---|---|---|---|
| `owner` | API response | CLI argument | Same |
| `name` | API response | CLI argument / folder name | Same |
| `url` | `html_url` | Constructed from owner/name | Same |
| `stars` | `stargazers_count` | N/A (not in git) | Set to 0 or cache from API |
| `forks` | `forks_count` | N/A (not in git) | Set to 0 or cache from API |
| `language` | API `language` field | Inferred from file extensions | Local is MORE accurate |
| `description` | API `description` | N/A | Optional, cache from API |
| `issues[]` | `GET /issues` endpoint | **Not in git** | Hybrid or cache |
| `prs[]` | `GET /pulls` endpoint | Merge commits + `refs/pull/*` | Partial (no metadata) |
| `commits[]` | `GET /commits` endpoint | `git log` | **Complete — unlimited** |
| `commit_diffs[]` | `GET /commits/{sha}` per commit | `git show {sha}` | **Complete — no rate limit** |

### 7.2 Handling Missing Data (Issues & PRs)

**Strategy A: Hybrid Mode (Recommended for initial implementation)**
```
Issues/PRs: GitHub API (low volume, ~2 pages of API calls)
Commits/Diffs: Local git (high volume, no limits)
```

**Strategy B: Full Offline with Cached Metadata**
```bash
# One-time cache of issues/PRs (run once with API)
python scripts/cache_github_metadata.py golang go --output data/cache/

# All subsequent runs use cache + local git
python scripts/ingest_local.py repos/golang/go --cache data/cache/golang_go.json --offline
```

**Strategy C: Commit-Only Mode (No Issues/PRs)**
```
For research purposes, focus exclusively on commit-based features:
- Code churn metrics ✅
- Contributor patterns ✅
- Language breakdown ✅
- Temporal patterns ✅
- File coupling ✅
```

> [!TIP]
> **Strategy A (Hybrid)** is recommended as the first implementation. Issues and PRs require very few API calls (~2-4 per entity type) and contain critical metadata (labels, state) that can't be derived from git alone. The expensive operation — commit diffs — is what gets moved to local.

---

## 8. Implementation Phases

### Phase 1: `LocalGitScraper` Core (Week 1)

**Files to create:**
- `src/scrapper/local_git.py` — Main local git scraper class
- `scripts/_core/local_scraper.py` — Core module matching `Scraper` interface
- `scripts/ingest_local.py` — CLI entry point for local ingestion

**Output**: Same `RepoData` dict from local clone, compatible with `SprintPreprocessor`

**Scope**: Commits + diffs only (skip issues/PRs initially)

### Phase 2: Hybrid Mode (Week 1-2)

**Files to modify:**
- `scripts/ingest.py` — Add `--local <path>` flag
- `scripts/_core/scraper.py` — Add optional local fallback

**Output**: Unified ingestion: API for issues/PRs, local git for commits/diffs

### Phase 3: Calibrated Synthetic Data (Week 2)

**Files to modify:**
- `src/data/synthetic_generator.py` — Add `CalibratedSyntheticGenerator`

**Files to create:**
- `src/data/statistical_profiler.py` — Profile real repos for synthetic calibration
- `scripts/calibrate_synthetic.py` — CLI for calibration pipeline

**Output**: Statistically validated synthetic sprints matching real Go repo patterns

### Phase 4: Full Offline Pipeline (Week 2-3)

**Files to create:**
- `scripts/cache_github_metadata.py` — One-time issue/PR metadata cache
- `scripts/validate_synthetic.py` — Statistical validation script

**Output**: Complete offline ingestion → training → validation pipeline

---

## 9. Performance Comparison

### Estimated Time for Go Repository (70K commits)

| Operation | GitHub API | Local Git | Speedup |
|-----------|-----------|----------|---------|
| List 500 commits | ~10 API calls, 5-10s | `git log` <1s | **10x** |
| 500 commit diffs | ~500 API calls, 100-600s | `git show` <5s | **100x** |
| All 70K commits | ~1,400 pages, impossible without token | `git log` <10s | **∞** |
| All 70K diffs | ~14K+ API calls, **hours** | `git show` loop, **minutes** | **>100x** |
| Contributor stats | Multiple API calls | `git shortlog` <1s | **50x** |
| Full ingest + diffs | **2.8+ hours** (with token) | **~10-15 minutes** | **>10x** |

---

## 10. File Structure

```
repo-sprint/
├── repos/
│   └── go/                          # Cloned Go repository (already exists)
│       └── .git/                    # Full git history (70K+ commits)
├── src/
│   └── scrapper/
│       ├── github.py                # Existing GitHub API scraper
│       └── local_git.py             # NEW: Local git scraper (same interface)
├── scripts/
│   ├── _core/
│   │   ├── scraper.py               # Existing API scraper core
│   │   └── local_scraper.py         # NEW: Local git scraper core
│   ├── ingest.py                    # Modified: add --local flag
│   ├── ingest_local.py              # NEW: Dedicated local ingestion CLI
│   ├── cache_github_metadata.py     # NEW: One-time issue/PR cache
│   ├── calibrate_synthetic.py       # NEW: Calibrate synthetic from local
│   └── validate_synthetic.py        # NEW: Statistical validation
├── src/data/
│   ├── preprocessor.py              # Unchanged: works with both sources
│   ├── formatter.py                 # Unchanged: works with both sources
│   ├── features.py                  # Unchanged: works with both sources
│   ├── synthetic_generator.py       # Modified: add calibrated generator
│   └── statistical_profiler.py      # NEW: Profile real repos
└── data/
    ├── cache/                       # NEW: Cached issue/PR metadata
    ├── raw/                         # Raw ingested data (both sources)
    └── processed/                   # Processed data (same as before)
```

---

## 11. Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Git CLI parsing errors | Use `--format` with JSON-safe delimiters; comprehensive unit tests |
| Large repo performance (70K commits) | Streaming output, `--since`/`--until` filters, batch processing |
| Missing issue/PR data | Hybrid mode (Strategy A); cache metadata for offline use |
| Inconsistent data schemas | Same `RepoData` TypedDict ensures downstream compatibility |
| Merge commit parsing | Not a true PR replacement — document limitations clearly |
| Cross-platform git differences | Standardize on `git` binary; test on macOS (M4 Pro) |

---

## 12. Success Criteria

- [ ] `LocalGitScraper.scrape()` returns identical `RepoData` schema as `GitHubScraper.scrape()`
- [ ] `SprintPreprocessor` produces same sprint structure from both data sources
- [ ] Full Go repo (70K commits) ingests in **<15 minutes** with diffs (vs. hours with API)
- [ ] Synthetic data calibrated from local stats passes KS-test (p > 0.05) against real data
- [ ] Offline pipeline runs end-to-end without network access
- [ ] Feature extraction (18 metrics) produces comparable results from both data sources

---

## 13. Next Steps

1. **Implement `src/scrapper/local_git.py`** — Core local git scraper with `git log` / `git show` parsing
2. **Create `scripts/ingest_local.py`** — CLI for local-only ingestion
3. **Add `--local` flag to `scripts/ingest.py`** — Hybrid mode support
4. **Build `src/data/statistical_profiler.py`** — Profile Go repo for synthetic calibration
5. **Update `src/data/synthetic_generator.py`** — Add calibrated generator class
6. **Create validation script** — Compare local vs. API ingestion output
7. **Update WBS** — Mark Task 2.2.1 as having a local alternative path

---

> [!NOTE]
> This strategy preserves **100% backward compatibility** with the existing GitHub API pipeline. The local git approach is an **additive capability** — both modes produce the same `RepoData` schema, so all downstream components (preprocessor, formatter, feature extractor, synthetic generator, ChromaDB indexer) work without modification.
