# Sprint Intelligence Data Pipeline: Complete Guide

**From GitHub Repository to Training Dataset**

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Architecture Overview](#system-architecture-overview)
3. [Current Data Model (ER Diagram)](#current-data-model-er-diagram)
4. [Complete Data Pipeline](#complete-data-pipeline)
5. [Detailed Process Breakdown](#detailed-process-breakdown)
6. [Hidden Logic & Assumptions](#hidden-logic--assumptions)
7. [Step-by-Step Implementation Guide](#step-by-step-implementation-guide)
   - Step 1-6: Single Repository Pipeline
   - Step 7: Combining Multiple Repositories ⭐ NEW
8. [Feature Schema & Metrics](#feature-schema--metrics)
9. [Training Data Validation](#training-data-validation)
10. [Troubleshooting Guide](#troubleshooting-guide)
11. [Final Report Components](#final-report-components)

---

## Executive Summary

The Sprint Intelligence Data Pipeline transforms raw GitHub repository data into a balanced training dataset for machine learning models. The process involves:

- **Data Collection**: Extract commits, issues, and PRs from local git repositories and GitHub API
- **Data Organization**: Group data into synthetic 2-week sprint windows
- **Feature Engineering**: Compute 25 metrics per sprint for risk analysis
- **Synthetic Augmentation**: Generate realistic synthetic sprints to balance training data
- **Statistical Validation**: Ensure synthetic data matches real distributions
- **Training Preparation**: Create stratified train/validation/test splits with class balance

**Key Achievement**: Convert raw GitHub history into a validated 151-sprint training dataset with 54% at-risk ratio and 8/15+ metrics matching real distributions.

---

## System Architecture Overview

```mermaid
graph TD
    A["GitHub Repository<br/>65K+ commits<br/>4K+ issues<br/>5K+ PRs"] -->|LocalGitScraper| B["Raw Git Data<br/>Commits with diffs<br/>Issues from commit text<br/>PRs from commit trailers"]
    
    A -->|GitHub API<br/>Fallback| C["API Data<br/>Full issue metadata<br/>Full PR metadata<br/>Rate-limited"]
    
    B -->|LocalScraper| D["Hybrid Data<br/>Merge local + API<br/>Deduplicate<br/>Enrich metadata"]
    C -->|LocalScraper| D
    
    D -->|SprintPreprocessor| E["Sprint Windows<br/>14-day synthetic windows<br/>Grouped by date<br/>Issues/PRs/commits per window"]
    
    E -->|FeatureExtractor| F["Calculated Metrics<br/>25 metrics per sprint<br/>Activity, code, risk<br/>Team metrics"]
    
    F -->|RiskLabeler| G["Labeled Data<br/>Binary at-risk label<br/>Pattern-based rules<br/>Realistic distribution"]
    
    G -->|Store| H["Real Sprints Dataset<br/>74 sprints<br/>40-50% at-risk<br/>Full 2-year history"]
    
    H -->|SyntheticSprintGenerator| I["Synthetic Sprints<br/>100+ sprints per run<br/>Auto-calibrated personas<br/>Matched distributions"]
    
    H -->|Auto-calibration| J["Persona Ranges<br/>Extract p25-p75 from real<br/>Match percentile bands<br/>5 personas per type"]
    
    J -->|Generate| I
    
    H & I -->|Mix| K["Combined Dataset<br/>151 total sprints<br/>51 real + 100 synthetic<br/>54% at-risk"]
    
    K -->|Stratified Split| L["Training Set<br/>87% of data<br/>Real + synthetic mixed<br/>54% at-risk"]
    
    K -->|Real-Only Split| M["Validation Set<br/>6% of data<br/>Real sprints only<br/>60% at-risk"]
    
    K -->|Real-Only Split| N["Test Set<br/>7% of data<br/>Real sprints only<br/>54% at-risk"]
    
    L & M & N -->|KS Testing| O["Validation Report<br/>Compare synthetic to real<br/>8/15 metrics passing<br/>Hidden metrics identified"]
    
    L & M & N -->|Ready| P["Training Pipeline<br/>M7: Fine-tuning<br/>M8: RAG Setup<br/>M9: Integration"]
```

---

## Current Data Model (ER Diagram)

```mermaid
erDiagram
    REPOSITORY ||--o{ SPRINT : creates
    SPRINT ||--o{ ISSUE : contains
    SPRINT ||--o{ PULL_REQUEST : contains
    SPRINT ||--o{ COMMIT : contains
    COMMIT ||--o| COMMIT_DIFF : has

    SPRINT ||--|| SPRINT_METRICS : computes
    SPRINT ||--|| RISK_LABEL : assigns

    SPRINT ||--o{ TRAINING_EXAMPLE : exports_as
    DATASET_SPLIT ||--o{ TRAINING_EXAMPLE : contains

    SYNTHETIC_PROFILE ||--o{ SYNTHETIC_SPRINT : generates
    SYNTHETIC_SPRINT ||--|| SPRINT_METRICS : computes
    SYNTHETIC_SPRINT ||--|| RISK_LABEL : assigns

    REPOSITORY {
        string owner
        string name
        string full_name
    }

    SPRINT {
        string sprint_id PK
        datetime start_date
        datetime end_date
        string repo_name FK
    }

    ISSUE {
        int issue_id PK
        string state
        datetime created_at
        datetime closed_at
        string sprint_id FK
    }

    PULL_REQUEST {
        int pr_id PK
        string state
        int additions
        int deletions
        datetime created_at
        datetime merged_at
        string sprint_id FK
    }

    COMMIT {
        string sha PK
        string author
        datetime committed_at
        string sprint_id FK
    }

    COMMIT_DIFF {
        string sha PK
        int total_additions
        int total_deletions
        int files_changed
    }

    SPRINT_METRICS {
        string sprint_id PK
        int total_issues
        int total_prs
        int total_commits
        float issue_resolution_rate
        float pr_merge_rate
        int total_code_changes
        int avg_pr_size
        float code_concentration
        int stalled_issues
        int unreviewed_prs
        int unique_authors
    }

    RISK_LABEL {
        string sprint_id PK
        int risk_score
        boolean is_at_risk
    }

    SYNTHETIC_PROFILE {
        string persona_name PK
        string persona_group
        float sampling_weight
    }

    SYNTHETIC_SPRINT {
        string sprint_id PK
        string persona_name FK
        string source
    }

    DATASET_SPLIT {
        string split_name PK
        string config_name
        float target_fraction
    }

    TRAINING_EXAMPLE {
        string example_id PK
        string sprint_id FK
        string split_name FK
        string source
        int label
        int risk_score
    }
```

**Notes**:
- `SPRINT_METRICS` matches the canonical schema in `src/data/features.py` (`SprintMetrics`).
- `RISK_LABEL` is derived by `RiskLabeler` from metrics and stored with each sprint.
- `TRAINING_EXAMPLE` is the normalized record written by `scripts/prepare_training_data.py` with `source` in `{real, synthetic}`.

---

## Complete Data Pipeline

### Phase 1: Data Ingestion (Sources & Collection)

**Hybrid Approach: Local Mining + API Enrichment**

```mermaid
graph LR
  A["Repository<br/>.git directory"] --> B["LocalGitScraper<br/>No API calls"]
  B --> C["Raw: Commits<br/>Diffs, refs"]
  C --> D["Merge & Deduplicate"]
  E["GitHub API<br/>5K/hr limit"] -.->|enrich| D
  D --> F["Complete Dataset<br/>Full metadata"]
  style A fill:#fff4e6
  style B fill:#e6f3ff
  style C fill:#f0e6ff
  style E fill:#fff4e6
  style F fill:#ccffcc
```

**Hidden Logic**:
- **LocalGitScraper Priority**: Always extract locally first to avoid API limits
  - Single `git log` call fetches 65K+ commits efficiently
  - Diffs parsed with `git show` → numstat parsing (adds/deletions per file)
  - PR metadata extracted from commit trailers (`Change-Id`, `Reviewed-on`)
  - Issue refs extracted from commit messages (`Fixes`, `Updates`)

- **API as Fallback**: 
  - Only fetch rich PR/issue metadata not in commit text
  - Use windowed queries (time-based) to stay within 5K/hr limit
  - Cache results to avoid re-fetching on retry

- **Deduplication Strategy**:
  - Match by URL hash or ID
  - Prefer API-enriched data over locally-extracted data
  - Merge metadata fields (labels, reviews, assignments)

### Phase 2: Sprint Creation (Temporal Grouping)

```mermaid
graph TD
    A["Raw GitHub Data<br/>Unsorted by time<br/>Mixed entities<br/>Spanning years"] 
    
    B["SprintPreprocessor<br/>1. Sort by date<br/>2. Find min/max date<br/>3. Create 14-day windows"]
    
    C["Sprint Windows<br/>Start: min_date<br/>Each: +14 days<br/>End: Last complete sprint<br/>or partial if time_remaining > 7"]
    
    A --> B
    B --> C
    
    D["Assign entities to sprints<br/>Issue created_at → sprint<br/>PR created_at → sprint<br/>Commit committed_at → sprint"]
    
    C --> D
    
    E["Sprint Object<br/>id, index, start, end<br/>issues[], prs[], commits[]<br/>Raw counts (no metrics yet)"]
    
    D --> E
```

**Hidden Logic**:
- **Window Size**: 14 days = 2 weeks (standard agile sprint)
- **Partial Sprint Handling**: 
  - Include last sprint if >= 7 days (half sprint)
  - Prevents single-day sprints skewing metrics
- **Assignment Rules**:
  - Entities assigned to sprint where `created_at` falls
  - Issue closing date ignored (focus on creation momentum)
  - Multi-sprint PRs assigned to creation sprint only
- **Empty Sprint Handling**:
  - Keep sprints with 0 entities (valid "quiet" period)
  - Used for synthetic generation of quiet_sprint personas

### Phase 3: Feature Extraction (Metrics Calculation)

```mermaid
graph TD
    A["Sprint Window<br/>Issues, PRs, commits<br/>Raw data"]
    
    B["FeatureExtractor<br/>Compute 25 metrics<br/>Grouped by category"]
    
    C["Category: Activity"]
    D["total_issues<br/>total_prs<br/>total_commits<br/>commit_frequency<br/>days_span"]
    
    E["Category: Timeline"]
    F["issue_age_avg<br/>pr_age_avg<br/>pr_open_days_avg<br/>issue_resolution_rate"]
    
    G["Category: Code Quality"]
    H["total_code_changes<br/>avg_pr_size<br/>files_changed<br/>code_concentration"]
    
    I["Category: Risk Signals"]
    J["stalled_issues<br/>unreviewed_prs<br/>abandoned_prs<br/>long_open_issues"]
    
    K["Category: Team"]
    L["unique_authors<br/>author_participation"]
    
    A --> B
    B --> C
    B --> E
    B --> G
    B --> I
    B --> K
    
    C --> D
    E --> F
    G --> H
    I --> J
    K --> L
    
    D & F & H & J & L -->|SprintMetrics TypedDict| M["Computed Metrics<br/>All 25 fields populated<br/>Ready for labeling"]
```

**Hidden Metrics Logic**:

| Metric | Formula | Why It Matters | Hidden Details |
|--------|---------|----------------|-----------------|
| `commit_frequency` | commits / days_span | Velocity tracking | Normalized to per-day rate |
| `issue_resolution_rate` | closed_issues / total_issues | Team capability | Includes rejected as resolved |
| `pr_merge_rate` | merged_prs / total_prs | Code-shipping speed | Excludes drafts from denominator |
| `stalled_issues` | issues open > sprint_length | Risk indicator | Created in sprint but not closed |
| `unreviewed_prs` | PRs with 0 reviews | Bottleneck signal | Blocks on human review capacity |
| `code_concentration` | max_author_commits / total_commits | Bus-factor risk | Single person dependency |
| `avg_pr_size` | files_changed_median | PR complexity | Median avoids outlier commits |

**Hidden Assumption**: All metrics bounded to [0, 1] scale via min/max normalization per repository.

### Phase 4: Risk Labeling

```mermaid
graph TD
    A["Sprint Metrics<br/>All 25 calculated<br/>Raw values"]
    
    B["RiskLabeler<br/>Pattern-based rules<br/>NOT machine learning<br/>Human-defined thresholds"]
    
    C["Rule 1: High Activity"]
    D["commit_frequency > p75<br/>AND<br/>total_issues > p50"]
    
    E["Rule 2: Low Resolution"]
    F["issue_resolution_rate < p25<br/>OR<br/>stalled_issues > p75"]
    
    G["Rule 3: PR Bottleneck"]
    H["unreviewed_prs > p50<br/>AND<br/>pr_merge_rate < p50"]
    
    I["Rule 4: Team Risk"]
    J["code_concentration > p75<br/>OR<br/>author_participation < p25"]
    
    K["Combine Rules"]
    L["at_risk = Rule1 OR Rule2<br/>OR Rule3 OR Rule4"]
    
    A --> B
    B --> C
    B --> E
    B --> G
    B --> I
    
    C --> D
    E --> F
    G --> H
    I --> J
    
    D & F & H & J --> K
    K --> L
    
    L -->|Binary Label| M["Sprint Label<br/>0 = Healthy<br/>1 = At-Risk"]
```

**Hidden Logic**:
- **Percentile Calculation**: Computed per *repository* (not globally) to normalize across project sizes
- **Label Validation**: Real data distribution should be ~40-50% at-risk (actual for anything-llm: 41/74 = 55%)
- **Pattern Reasoning**:
  - High activity + low resolution = backlog accumulation risk
  - PR bottleneck = shipping velocity risk
  - Code concentration = key-person dependency risk

### Phase 5: Synthetic Data Generation

```mermaid
graph TD
    A["Real Sprints Dataset<br/>74 sprints from anything-llm<br/>Full 25-metric distribution"]
    
    B["Auto-Calibration<br/>Extract percentile ranges<br/>p25, p50, p75 per metric<br/>Compute for each persona"]
    
    C["Persona Ranges<br/>LARGE_OSS Personas:<br/>  - active_sprint<br/>  - release_sprint<br/>  - quiet_sprint<br/>  - blocked_sprint<br/>  - refactor_sprint<br/><br/>STARTUP Personas:<br/>  - commit_first<br/>  - healthy_flow<br/>  - blocked_issues<br/>  - pr_bottleneck<br/>  - quiet_sprint"]
    
    D["Weighted Selection<br/>50% OSS personas<br/>50% Startup personas<br/>1:1 healthy:at-risk ratio"]
    
    E["Generate Metrics<br/>For each persona:<br/>  Sample metric values<br/>  From persona's ranges<br/>  Add realistic variance<br/>  Validate bounds"]
    
    F["Apply RiskLabeler<br/>Same rules as real data<br/>Creates balanced<br/>at-risk distribution"]
    
    G["Output Synthetic Sprints<br/>100+ sprints<br/>Same schema as real<br/>Passed KS validation"]
    
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
```

### Persona Sprint Reference (Current Model)

Persona sprints are synthetic sprint archetypes used by `SyntheticSprintGenerator` to emulate common team operating modes. Each generated sprint is tagged with a persona (stored as `persona`) and then labeled by the same `RiskLabeler` used for real sprints.

| Persona | Typical Behavior | Dominant Metrics | Expected Risk Tendency |
|--------|-------------------|------------------|------------------------|
| `active_sprint` | High execution burst with many commits/PRs | `total_commits`, `commit_frequency`, `total_prs` | Medium (depends on review/merge quality) |
| `release_sprint` | Shipping-focused period with merges and churn | `pr_merge_rate`, `total_code_changes`, `avg_pr_size` | Low-Medium |
| `quiet_sprint` | Low activity maintenance window | `total_issues`, `total_prs`, `total_commits` | Low (unless unresolved backlog grows) |
| `blocked_sprint` | Throughput constrained by unresolved work | `stalled_issues`, `issue_resolution_rate`, `unreviewed_prs` | High |
| `refactor_sprint` | Internal code quality/churn heavy work | `total_additions`, `total_deletions`, `files_changed` | Medium |
| `commit_first` | Startup-style commit-heavy cadence | `total_commits`, `commit_frequency`, `code_concentration` | Medium |
| `healthy_flow` | Balanced issue-to-PR lifecycle | `issue_resolution_rate`, `pr_merge_rate`, `unique_authors` | Low |
| `blocked_issues` | Issue backlog accumulates faster than resolution | `stalled_issues`, `long_open_issues`, `issue_resolution_rate` | High |
| `pr_bottleneck` | PR queue grows with limited review/merge | `unreviewed_prs`, `pr_merge_rate`, `avg_pr_size` | High |

**How persona selection works now**:
- `--personas auto` calibrates metric distributions from real sprint files.
- Generator samples personas using weighted selection, then samples metrics from calibrated ranges/empirical pools.
- Final `is_at_risk` label is not hardcoded by persona; it is computed from generated metrics via `RiskLabeler`.
- This allows the same persona to produce both healthy and at-risk examples when boundary conditions change.

**Hidden Calibration Logic**:

```python
# Pseudocode for auto-calibration
for metric in all_metrics:
    real_values = [sprint[metric] for sprint in REAL_SPRINTS]
    p25 = percentile(real_values, 25)
    p50 = percentile(real_values, 50)
    p75 = percentile(real_values, 75)
    
    # Store as range for sampling
    PERSONA_RANGES[persona][metric] = (p25, p75)
    
# During generation
synthetic_value = uniform(PERSONA_RANGES[persona][metric])
# Add noise: synthetic_value += normal(0, std_dev)
```

**Critical Hidden Issue** (From conversation):
- **7 Metrics Failing KS Test** (as of last run):
  - `pr_merge_rate`: KS=1.0 (completely opposite distribution)
  - `issue_resolution_rate`: KS=0.75 (miscalibrated)
  - `unreviewed_prs`: KS=0.67 (wrong distribution)
  - `files_changed`: KS > 0.5
  - `stalled_issues`: KS > 0.5
  - `total_deletions`: KS > 0.5
  - `avg_pr_size`: KS > 0.5

- **Fix Needed**: Fine-tune persona ranges from `anything-llm` percentiles, especially `pr_merge_rate` (should be high for healthy projects)

### Phase 6: Training Data Preparation

```mermaid
graph TD
    A["Real Sprints<br/>74 sprints<br/>41 at-risk<br/>33 healthy"]
    
    B["Synthetic Sprints<br/>100 sprints<br/>54 at-risk<br/>46 healthy"]
    
    C["Combine Datasets<br/>Total: 174 sprints<br/>95 at-risk<br/>79 healthy<br/>54.6% at-risk ratio"]
    
    D["Stratified Split<br/>Maintain label ratio<br/>in each split"]
    
    E["Training Set<br/>87% of data = 151 sprints<br/>51 real + 100 synthetic<br/>82 at-risk + 69 healthy<br/>54.3% at-risk"]
    
    F["Validation Set<br/>6% of data = 10 sprints<br/>Real sprints ONLY<br/>6 at-risk + 4 healthy<br/>60% at-risk"]
    
    G["Test Set<br/>7% of data = 13 sprints<br/>Real sprints ONLY<br/>7 at-risk + 6 healthy<br/>53.8% at-risk"]
    
    A --> C
    B --> C
    
    C --> D
    
    D --> E
    D --> F
    D --> G
    
    E -->|KS Testing| H["Statistical Validation<br/>Compare synthetic metrics<br/>to real metrics<br/>Target: 8+ metrics passing"]
    
    H -->|Passed| I["Ready for Fine-tuning<br/>M7: LoRA training<br/>M8: RAG integration<br/>M9: Agent orchestration"]
```

**Hidden Logic for Stratified Split**:

```python
# Pseudocode
combined_sprints = real_sprints + synthetic_sprints

# Count at-risk vs healthy
at_risk = [s for s in combined_sprints if s.label == 1]  # 95 sprints
healthy = [s for s in combined_sprints if s.label == 0]  # 79 sprints

# Stratified sampling (maintain ratio in each split)
train_at_risk = sample(at_risk, size=0.87 * len(at_risk))  # 82
train_healthy = sample(healthy, size=0.87 * len(healthy))  # 69

val_at_risk = sample(real_at_risk, size=0.06 * len(real_at_risk))  # 6
val_healthy = sample(real_healthy, size=0.06 * len(real_healthy))  # 4

# Same for test set
```

**Critical Constraint**: Val and Test sets use **REAL SPRINTS ONLY** to avoid overfitting to synthetic patterns.

---

## Detailed Process Breakdown

### Data Flow Diagram: End-to-End (Full Detail)

```mermaid
graph LR
    A["1. GitHub Repo<br/>golang/go"] -->|clone| B["2. LocalGitScraper<br/>65K commits"]
    
    B -->|git commands| C["3. Raw Commits<br/>Diffs, metadata"]
    
    A -->|GitHub API| D["4. Issues & PRs<br/>Full metadata"]
    
    C -->|merge| E["5. Hybrid Data<br/>Enriched"]
    D -->|merge| E
    
    E -->|group by date| F["6. Sprint Windows<br/>14-day chunks"]
    
    F -->|compute metrics| G["7. Features<br/>25 metrics"]
    
    G -->|label| H["8. Risk Labels<br/>0 or 1"]
    
    H -->|store| I["9. Real Sprints<br/>74 sprints"]
    
    I -->|calibrate| J["10. Persona Ranges<br/>p25-p75 per metric"]
    
    J -->|generate| K["11. Synthetic Sprints<br/>100 sprints"]
    
    I -->|combine| L["12. Mixed Dataset<br/>174 sprints"]
    K -->|combine| L
    
    L -->|split| M["13. Training Set<br/>151 sprints"]
    L -->|split| N["14. Val Set<br/>10 sprints"]
    L -->|split| O["15. Test Set<br/>13 sprints"]
    
    M -->|KS test| P["16. Validation Report<br/>8/15 passing"]
    N -->|export| Q["17. JSONL Files<br/>h3_train.json<br/>h3_val.json<br/>h3_test.json"]
    O -->|export| Q
```

### Data Flow Diagram: Short Version (Slide-Friendly)

```mermaid
graph LR
  A["<b>GitHub Repo<br/>65K commits</b>"] --> B["<b>Hybrid Data<br/>Local + API</b>"]
  B --> C["<b>Sprint Windows<br/>14-day intervals</b>"]
  C --> D["<b>Features & Labels<br/>25 metrics + at-risk</b>"]
  D --> E["<b>Real Sprints<br/>74 labeled</b>"]
  E --> F["<b>Synthetic Gen<br/>100 sprints</b>"]
  E --> G["<b>Mixed Dataset<br/>174 total</b>"]
  F --> G
  G --> H["<b>Train/Val/Test<br/>70/15/15 split</b>"]
  H --> I["<b>Validation ✓<br/>KS-tested</b>"]
  style A fill:#fff4e6
  style B fill:#e6f3ff
  style C fill:#f0e6ff
  style D fill:#e6ffe6
  style E fill:#ffe6f0
  style F fill:#fff9c4
  style G fill:#f0e6ff
  style H fill:#e6ffe6
  style I fill:#ccffcc
```

---

## Hidden Logic & Assumptions

### 1. Argument Parsing Bug (Blocking Issue)

**Location**: `scripts/ingest_local.py` lines 393-402

**Current Code**:
```python
pr_diff_limit = args.pr_diff_limit if args.pr_diff_limit else _NO_LIMIT
```

**Problem**: `--pr-diff-limit 0` is falsy in Python, triggers `_NO_LIMIT` (999_999)
- Intended: Extract 0 PR diffs (offline mode)
- Actual: Attempts 9,764 API calls (4,882 PRs × 2 calls each)
- Impact: Hangs after 2KB local data completion, waiting for API rate limit

**Hidden Assumption**: User expects `0` = disabled, but Python treats it as False

**Fix Required**:
```python
pr_diff_limit = args.pr_diff_limit if args.pr_diff_limit > 0 else _NO_LIMIT
```

### 2. LocalGitScraper Design Assumptions

**Hidden Assumption 1**: Git command format consistency
- Assumes `git log` outputs ISO8601 dates
- Assumes `git show` returns unified diff format
- **Risk**: Different git versions may vary output

**Hidden Assumption 2**: Commit trailer parsing
- `Change-Id` and `Reviewed-on` assumed to indicate PR metadata
- **Risk**: Not all repos use these trailers (Go repo does, many don't)

**Hidden Assumption 3**: Issue reference extraction
- Looks for `Fixes #123` or `Updates #456` patterns
- **Risk**: Different projects use different patterns

**Mitigation**: Hybrid approach + API fallback captures all metadata

### 3. Sprint Window Timing Logic

**Hidden Assumption 1**: 14-day fixed windows
- Doesn't align with actual sprint schedules
- Creates artificial sprint boundaries
- **Why**: Enables consistent retrospective analysis without knowing actual sprints

**Hidden Assumption 2**: Partial sprint handling
- Keeps sprints >= 7 days (half sprint)
- **Why**: Prevents single-day noise

**Hidden Assumption 3**: Entity assignment
- An issue/PR created on day 8 of sprint 1 belongs to sprint 1
- If it closes in sprint 3, it still counts for sprint 1
- **Why**: Measures creation momentum, not resolution latency across sprints

### 4. Feature Metric Normalization

**Hidden Logic**: All metrics normalized per-repository
```python
# Pseudocode
metric_min = min([sprint[metric] for sprint in REPOSITORY_SPRINTS])
metric_max = max([sprint[metric] for sprint in REPOSITORY_SPRINTS])

normalized = (value - metric_min) / (metric_max - metric_min)
```

**Why**: Allows cross-repository models (Go vs Rust vs Linux have different scales)

**Hidden Risk**: New repository metrics not seen during calibration will be out-of-range [0, 1]

### 5. Risk Labeling Thresholds

**Hidden Assumption 1**: Percentile-based rules
- Uses p25, p50, p75 from repository data
- Rules hard-coded to specific percentiles
- **Why**: Adaptive to repository characteristics

**Hidden Assumption 2**: Multi-rule OR logic
- If ANY rule triggers → at_risk = 1
- **Why**: Err on side of caution (false positive better than missed risk)

**Hidden Assumption 3**: Static rule weights
- Each rule weighted equally (no prioritization)
- **Future**: Could learn rule weights from fine-tuning

### 6. Synthetic Data Generation Calibration

**Hidden Assumption 1**: Personas are separable
- Each persona has distinct metric signature
- **Reality**: Real sprints blend multiple personas
- **Workaround**: Sample from overlapping ranges, not distinct clusters

**Hidden Assumption 2**: Auto-calibration scope
- Calibrated from single repository (anything-llm)
- **Risk**: May not generalize to other repos
- **Mitigation**: Re-calibrate per-repository before training

**Hidden Assumption 3**: Percentile ranges preserved
- Synthetic `p25-p75` of metric matches real `p25-p75`
- But distribution shape may differ
- **That's why**: KS testing checks both mean AND shape

### 7. KS Test Interpretation

**Hidden Logic**: Kolmogorov-Smirnov test
- Compares cumulative distributions
- Reports maximum distance between CDFs
- Tests null hypothesis: "distributions are identical"

**Hidden Interpretation**:
- `KS p-value > 0.05` → cannot reject null (distributions match)
- `KS p-value < 0.05` → reject null (distributions differ)
- Higher KS statistic = larger difference

**Critical Detail**: KS test only used for validation, NOT for training loss

---

## Step-by-Step Implementation Guide

### Step 0: Prerequisites

```bash
# Clone Go repository (or any target)
git clone https://github.com/golang/go repositories/golang/go

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
# Must include: requests, scipy, pandas (implied by scripts)
```

### Step 1: Local Data Extraction

```bash
# Command
npm run local:golang -- --diffs --owner golang --repo go

# What happens:
# 1. LocalGitScraper extracts all commits from .git directory
# 2. For each commit: Parse date, author, message
# 3. Extract diffs: additions/deletions per file
# 4. Extract PR refs from commit trailers
# 5. Extract issue refs from commit messages
# 6. Save to: data/golang_go_raw.json

# Hidden behavior:
# - First run: 65K+ commits (no API calls!)
# - Subsequent runs: Load from cache if available
# - Diffs parsed in batches (100s at a time)
```

**Output Structure**:
```json
{
  "metadata": {
    "owner": "golang",
    "repo": "go",
    "extraction_method": "local_git",
    "min_date": "2023-01-01T00:00:00Z",
    "max_date": "2025-03-25T23:59:59Z"
  },
  "commits": [
    {
      "id": "abc123def456...",
      "author": "user@golang.org",
      "message": "runtime: fix GC issue\n\nFixes #12345",
      "created_at": "2024-03-15T10:30:00Z",
      "files_changed": {
        "runtime/gc.go": {"additions": 45, "deletions": 12},
        ...
      },
      "pr_refs": ["#1234"],
      "issue_refs": ["#12345"]
    },
    ...
  ],
  "issues_from_commits": [
    {
      "number": 12345,
      "title": "GC causes memory leak",
      "created_at": "2024-03-10T00:00:00Z",
      "source": "commit_message"
    }
  ],
  "prs_from_commits": [
    {
      "number": 1234,
      "title": "Fix GC memory leak",
      "created_at": "2024-03-14T00:00:00Z",
      "source": "commit_trailer"
    }
  ]
}
```

### Step 2: GitHub API Enrichment (Optional)

```bash
# If you want full PR/issue metadata
npm run local:golang:api -- --issues-limit 500 --prs-limit 500

# What happens:
# 1. GitHub API called for issues (5K/hour limit)
# 2. Fetch: labels, assignees, reviews, state (closed/open)
# 3. Fetch: PR reviews, merge status, review comments
# 4. Merge with local data (API data takes priority)
# 5. Save to: data/golang_go_enriched.json

# Cost: ~1000 API calls (manageable within rate limit)
# Benefit: Full issue/PR metadata for richer features
```

### Step 3: Sprint Preprocessing

```bash
# Implicit step (happens in ingest_local.py)
# Processes raw data through SprintPreprocessor

# What happens:
# 1. Sort all entities by created_at
# 2. Find min_date and max_date
# 3. Create window start dates: min_date, min_date+14d, min_date+28d, ...
# 4. Assign each entity to its sprint window
# 5. Count issues, PRs, commits per sprint
# 6. Output: data/golang_go_sprints_raw.json

# Output structure:
{
  "sprints": [
    {
      "id": 0,
      "index": 0,
      "start": "2023-01-01T00:00:00Z",
      "end": "2023-01-14T23:59:59Z",
      "issues": [
        {
          "number": 12345,
          "title": "...",
          "created_at": "...",
          "closed_at": null
        }
      ],
      "prs": [...],
      "commits": [...]
    },
    ...
  ],
  "total_sprints": 104
}
```

### Step 4: Feature Extraction

```python
# Run as part of data pipeline
from src.data.features import FeatureExtractor, RiskLabeler

extractor = FeatureExtractor()
labeler = RiskLabeler()

for sprint in sprints:
    metrics = extractor.extract(sprint)
    # metrics = {
    #   "total_issues": 23,
    #   "total_prs": 15,
    #   "commit_frequency": 2.8,
    #   ...all 25 metrics...
    # }
    
    label = labeler.label(metrics)
    # label = 0 (healthy) or 1 (at-risk)
    
    sprint["metrics"] = metrics
    sprint["label"] = label
```

**Output**: `data/golang_go_sprints.json` with metrics for each sprint

### Step 5: Synthetic Data Generation

```bash
# Auto-calibrate from real data
npm run calibrate-synthetic -- --source data/golang_go_sprints.json

# Generate synthetic sprints
npm run generate-synthetic -- --count 100 --personas auto

# What happens:
# 1. Read golang_go_sprints.json
# 2. Compute p25, p50, p75 for each metric
# 3. Create/update persona ranges
# 4. Generate 100 synthetic sprints:
#    - Randomly select persona (50% OSS, 50% Startup)
#    - Sample metric values from persona ranges
#    - Apply RiskLabeler
# 5. Save to: data/synthetic_sprints.json

# Output has same schema as real sprints
```

**Critical Step**: Verify KS test passes

```bash
# After generation, check validation metrics
python scripts/prepare_training_data.py --config h3 --validate-only

# Output shows:
# KS Test Results:
#   total_issues: KS=0.14 ✓ (p > 0.05)
#   commit_frequency: KS=0.22 ✓
#   ...
#   [7 metrics failing KS test]
```

### Step 6: Training Data Preparation

```bash
npm run prepare-training:h3

# What happens:
# 1. Load real sprints: data/Mintplex-Labs_anything-llm_sprints.json (74)
# 2. Load/generate synthetic: data/synthetic_sprints.json (100)
# 3. Combine: 174 total sprints
# 4. Stratified split:
#    - Train: 151 (87%)
#    - Val: 10 (6%)
#    - Test: 13 (7%)
# 5. Save to: data/training/
#    - h3_train.json (151)
#    - h3_val.json (10)
#    - h3_test.json (13)
# 6. Run KS tests (if scipy installed)

# Output:
# Training data prepared in data/training/
# Train: 151 sprints (87%, real=51, synthetic=100, at-risk=54.3%)
# Val: 10 sprints (6%, real-only, at-risk=60%)
# Test: 13 sprints (7%, real-only, at-risk=54%)
# ✓ Training data ready for M7 fine-tuning
```

**Check output files**:

```bash
# Verify training set structure
wc -l data/training/h3_train.json  # Should be 151 JSON objects (1 per line)
head -1 data/training/h3_train.json | jq 'keys'

# Keys should include:
# - "id", "index", "start", "end"
# - "issues", "prs", "commits"
# - "metrics" (all 25 fields)
# - "label" (0 or 1)
# - "source" ("real" or "synthetic")
```

### Step 7: Combining Multiple Repositories (Optional)

**Scenario**: You have 3+ repositories from the same owner/company and want to create a combined training dataset.

```mermaid
graph TD
    A1["Repo 1<br/>company/repo1<br/>50 sprints"] --> B1["Ingest<br/>local:hybrid"]
    A2["Repo 2<br/>company/repo2<br/>65 sprints"] --> B2["Ingest<br/>local:hybrid"]
    A3["Repo 3<br/>company/repo3<br/>45 sprints"] --> B3["Ingest<br/>local:hybrid"]
    
    B1 --> C1["data/company_repo1_sprints.json"]
    B2 --> C2["data/company_repo2_sprints.json"]
    B3 --> C3["data/company_repo3_sprints.json"]
    
    C1 & C2 & C3 -->|combine_repos.py| D["Combine & Reassign IDs<br/>Total: 160 sprints<br/>Preserve original_repo tag"]
    
    D -->|combined_company_sprints.json| E["Normalize Per-Repo<br/>normalize_combined.py<br/>CRITICAL: Min/Max per-repo"]
    
    E -->|combined_company_sprints_normalized.json| F["Re-label<br/>Updated thresholds<br/>from combined percentiles"]
    
    F -->|combined_company_sprints_labeled.json| G["Generate Synthetic<br/>calibrate-synthetic<br/>--source combined_..."]
    
    G -->|data/synthetic_sprints.json| H["Prepare Combined Training<br/>prepare_combined_training.py<br/>Stratified split"]
    
    H -->|output| I["Training Files<br/>h3_combined_train.json<br/>h3_combined_val.json<br/>h3_combined_test.json<br/><br/>✓ Ready for M7"]
```

---

#### Option A: Sequential Ingestion (Recommended for Small Teams)

```bash
# Step 1: Ingest each repo separately
npm run local:company:repo1:hybrid  # Owner: company, Repo: repo1
npm run local:company:repo2:hybrid  # Owner: company, Repo: repo2
npm run local:company:repo3:hybrid  # Owner: company, Repo: repo3

# Output files created:
# - data/company_repo1_sprints.json (e.g., 50 sprints)
# - data/company_repo2_sprints.json (e.g., 65 sprints)
# - data/company_repo3_sprints.json (e.g., 45 sprints)
```

#### Step 2: Combine Sprint Files

```python
# Create script: scripts/combine_repos.py
import json
import glob
from pathlib import Path

output_sprints = []
repo_metadata = {}
unique_sprint_id = 0

# Load all repo sprint files
sprint_files = glob.glob('data/company_*_sprints.json')

for file_path in sorted(sprint_files):
    with open(file_path) as f:
        data = json.load(f)
    
    owner, repo = file_path.split('_')[1:3]
    repo = repo.replace('_sprints.json', '')
    
    # Track repo metadata (needed for normalization)
    repo_metadata[f"{owner}/{repo}"] = {
        'file': file_path,
        'sprint_count': len(data['sprints']),
        'metrics_ranges': {}  # Will compute min/max per metric
    }
    
    # Reassign IDs to avoid conflicts
    for sprint in data['sprints']:
        sprint['id'] = unique_sprint_id
        sprint['original_repo'] = f"{owner}/{repo}"
        sprint['original_index'] = sprint['index']
        output_sprints.append(sprint)
        unique_sprint_id += 1

# Save combined dataset
with open('data/combined_company_sprints.json', 'w') as f:
    json.dump({
        'metadata': {
            'source': 'combined_multi_repo',
            'repos': list(repo_metadata.keys()),
            'total_sprints': len(output_sprints),
            'repo_breakdown': {
                repo: meta['sprint_count'] 
                for repo, meta in repo_metadata.items()
            }
        },
        'sprints': output_sprints
    }, f, indent=2)

print(f"✓ Combined {len(output_sprints)} sprints from {len(repo_metadata)} repos")
print(f"  Repos: {', '.join(repo_metadata.keys())}")
```

**Run it**:
```bash
python scripts/combine_repos.py
# Output: ✓ Combined 160 sprints from 3 repos
#   Repos: company/repo1, company/repo2, company/repo3
```

#### Step 3: Normalize Metrics Across Repos

**Critical Issue**: Metrics have different scales across repositories!

```python
# Create script: scripts/normalize_combined.py
import json
import statistics
from pathlib import Path

def normalize_features(sprints):
    """
    Normalize metrics per-repository to [0, 1] range.
    This ensures repo1's big numbers don't dominate repo2's small numbers.
    """
    
    # Group sprints by repo
    by_repo = {}
    for sprint in sprints:
        repo = sprint.get('original_repo', 'unknown')
        if repo not in by_repo:
            by_repo[repo] = []
        by_repo[repo].append(sprint)
    
    # Compute min/max per metric PER REPO
    repo_ranges = {}
    for repo, repo_sprints in by_repo.items():
        repo_ranges[repo] = {}
        
        # Get metrics from first sprint (assume all sprints have same metrics)
        sample_metrics = repo_sprints[0]['metrics'].keys()
        
        for metric in sample_metrics:
            values = [s['metrics'][metric] for s in repo_sprints 
                     if isinstance(s['metrics'].get(metric), (int, float))]
            
            if values:
                repo_ranges[repo][metric] = {
                    'min': min(values),
                    'max': max(values),
                    'mean': statistics.mean(values)
                }
    
    # Re-normalize each sprint's metrics
    for sprint in sprints:
        repo = sprint.get('original_repo', 'unknown')
        ranges = repo_ranges.get(repo, {})
        
        for metric, value in sprint['metrics'].items():
            if metric in ranges and isinstance(value, (int, float)):
                r = ranges[metric]
                
                # Normalize to [0, 1]
                if r['max'] == r['min']:
                    normalized = 0.5  # Edge case: all same value
                else:
                    normalized = (value - r['min']) / (r['max'] - r['min'])
                
                # Clamp to [0, 1]
                normalized = max(0.0, min(1.0, normalized))
                sprint['metrics'][metric] = normalized
        
        sprint['metrics_normalized'] = True
    
    return sprints, repo_ranges

# Load combined sprints
with open('data/combined_company_sprints.json') as f:
    data = json.load(f)

# Normalize
sprints, ranges = normalize_features(data['sprints'])

# Save
with open('data/combined_company_sprints_normalized.json', 'w') as f:
    json.dump({
        'metadata': {
            **data['metadata'],
            'normalization_ranges': ranges
        },
        'sprints': sprints
    }, f, indent=2)

print("✓ Metrics normalized per-repository")
for repo, metrics in ranges.items():
    print(f"\n  {repo}:")
    for metric, r in list(metrics.items())[:3]:  # Show first 3
        print(f"    {metric}: [{r['min']:.2f}, {r['max']:.2f}]")
```

**Run it**:
```bash
python scripts/normalize_combined.py
# Output shows per-repo metric ranges
```

#### Step 4: Re-label Combined Sprints

```bash
# Re-compute risk labels using combined dataset
python3 << 'EOF'
from src.data.features import RiskLabeler
import json

labeler = RiskLabeler()

with open('data/combined_company_sprints_normalized.json') as f:
    data = json.load(f)

# Re-label (uses updated percentile thresholds from combined data)
for sprint in data['sprints']:
    label = labeler.label(sprint['metrics'])
    sprint['label'] = label

# Save
with open('data/combined_company_sprints_labeled.json', 'w') as f:
    json.dump(data, f, indent=2)

# Report
at_risk = sum(1 for s in data['sprints'] if s.get('label') == 1)
total = len(data['sprints'])
print(f"✓ Re-labeled {total} sprints: {at_risk} at-risk ({at_risk/total*100:.1f}%)")
EOF
```

#### Step 5: Generate Combined Synthetic Data

```bash
# Calibrate from combined real data
npm run calibrate-synthetic -- --source data/combined_company_sprints_labeled.json

# Generate synthetic sprints with combined characteristics
npm run generate-synthetic -- --count 150 --personas auto

# Output: data/synthetic_sprints.json (calibrated to 3-repo blend)
```

#### Step 6: Prepare Combined Training Data

```python
# Create script: scripts/prepare_combined_training.py
import json
import random
from collections import Counter

# Load datasets
with open('data/combined_company_sprints_labeled.json') as f:
    real_sprints = json.load(f)['sprints']

with open('data/synthetic_sprints.json') as f:
    synthetic_sprints = json.load(f)['sprints']

# Combine
all_sprints = real_sprints + synthetic_sprints
random.shuffle(all_sprints)

# Stratified split (maintain label ratio)
at_risk = [s for s in all_sprints if s.get('label') == 1]
healthy = [s for s in all_sprints if s.get('label') == 0]

total = len(all_sprints)
train_size = int(0.87 * total)
val_size = int(0.06 * total)
test_size = total - train_size - val_size

# Split maintaining ratio
train_at_risk = at_risk[:int(0.87 * len(at_risk))]
train_healthy = healthy[:int(0.87 * len(healthy))]
train = train_at_risk + train_healthy

val_at_risk = at_risk[int(0.87 * len(at_risk)):int(0.87 * len(at_risk)) + int(0.06 * len(at_risk))]
val_healthy = healthy[int(0.87 * len(healthy)):int(0.87 * len(healthy)) + int(0.06 * len(healthy))]
val = val_at_risk + val_healthy

test = at_risk[int(0.93 * len(at_risk)):] + healthy[int(0.93 * len(healthy)):]

# Save
for split_name, split_data in [
    ('h3_combined_train', train),
    ('h3_combined_val', val),
    ('h3_combined_test', test)
]:
    with open(f'data/training/{split_name}.json', 'w') as f:
        for sprint in split_data:
            f.write(json.dumps(sprint) + '\n')

print(f"✓ Training data prepared:")
print(f"  Train: {len(train)} ({len([s for s in train if s['label']==1])} at-risk)")
print(f"  Val:   {len(val)} ({len([s for s in val if s['label']==1])} at-risk)")
print(f"  Test:  {len(test)} ({len([s for s in test if s['label']==1])} at-risk)")
EOF
```

#### Summary: Multi-Repo Command Flow

```bash
# Complete pipeline for 3 repos
npm run local:company:repo1:hybrid && \
npm run local:company:repo2:hybrid && \
npm run local:company:repo3:hybrid && \
python scripts/combine_repos.py && \
python scripts/normalize_combined.py && \
python scripts/prepare_combined_training.py && \
npm run prepare-training:h3

# Output: data/training/h3_combined_*.json ready for fine-tuning
```

#### Multi-Repo Considerations

| Aspect | Single Repo | Multi-Repo |
|--------|------------|-----------|
| Sprint Count | 74 (anything-llm) | 160 (repo1 + repo2 + repo3) |
| Normalization | Global min/max | **Per-repo min/max** ⚠️ |
| Risk Labels | Static thresholds | Recomputed from combined percentiles |
| Synthetic Calibration | From one repo | **From blended characteristics** |
| Training Diversity | Language-specific | **Cross-repo patterns** |
| Generalization | Limited (Go-like) | Improved (multiple codebases) |

**Key Hidden Issue**: Different repos have vastly different scales:
- Small startup project: 2 commits/day, 1 PR/sprint
- Large OSS project: 50 commits/day, 15 PRs/sprint

Must normalize **per-repo**, not globally!

---

## Feature Schema & Metrics

### Complete 25-Metric Schema

```python
class SprintMetrics(TypedDict):
    # Activity Metrics (5)
    "total_issues": int
    "total_prs": int
    "total_commits": int
    "commit_frequency": float  # commits per day
    "days_span": int  # days in sprint
    
    # Timeline Metrics (4)
    "issue_age_avg": float  # average days open
    "pr_age_avg": float  # average days open
    "pr_open_days_avg": float  # same as pr_age_avg
    "issue_resolution_rate": float  # closed / total [0, 1]
    
    # Code Quality Metrics (4)
    "total_code_changes": int  # total additions + deletions
    "avg_pr_size": float  # median files per PR
    "files_changed": int  # unique files touched
    "code_concentration": float  # max_author_commits / total [0, 1]
    
    # Resolution Metrics (3)
    "pr_merge_rate": float  # merged / total [0, 1]
    "closed_issues": int
    "merged_prs": int
    
    # Risk Metrics (4)
    "stalled_issues": int  # created in sprint, not closed
    "unreviewed_prs": int  # PRs with 0 reviews
    "abandoned_prs": int  # closed without merge
    "long_open_issues": int  # open > 30 days
    
    # Team Metrics (2)
    "unique_authors": int  # distinct committers
    "author_participation": float  # variance in commits
```

### Metric Interpretation Guide

**Health Indicators** (Higher is Better):
- `commit_frequency`: Shows steady development
- `issue_resolution_rate`: Team closing issues
- `pr_merge_rate`: Shipping velocity
- `unique_authors`: Distributed contributions

**Risk Indicators** (Lower is Better):
- `stalled_issues`: Backlog accumulation
- `unreviewed_prs`: Review bottleneck
- `abandoned_prs`: Work discarded
- `code_concentration`: Key-person dependency

**Contextual Indicators**:
- `days_span`: Is sprint productive or quiet?
- `avg_pr_size`: Small PRs (easy to review) vs large (risky)
- `long_open_issues`: Indicates old backlog

---

## Training Data Validation

### Step 1: Load and Inspect

```bash
# Check training set
python3 << 'EOF'
import json

with open('data/training/h3_train.json') as f:
    train = [json.loads(line) for line in f]

print(f"Training set: {len(train)} sprints")
print(f"At-risk: {sum(1 for s in train if s['label'] == 1)} ({sum(1 for s in train if s['label'] == 1) / len(train) * 100:.1f}%)")
print(f"Real sprints: {sum(1 for s in train if s.get('source') == 'real')}")
print(f"Synthetic: {sum(1 for s in train if s.get('source') == 'synthetic')}")

# Check metric ranges
import statistics
for metric in train[0]['metrics'].keys():
    values = [s['metrics'][metric] for s in train]
    if isinstance(values[0], (int, float)):
        print(f"{metric}: min={min(values):.2f}, max={max(values):.2f}, mean={statistics.mean(values):.2f}")
EOF
```

### Step 2: KS Test Validation

```bash
# This requires scipy installed
pip install scipy

# Then re-run training prep with KS validation
npm run prepare-training:h3

# Output will show:
# KS Test Results:
# ✓ total_issues: p=0.15
# ✓ commit_frequency: p=0.23
# ...
# ✗ pr_merge_rate: p=0.00 (FAILS)
# ✗ issue_resolution_rate: p=0.00 (FAILS)
# ...
# Summary: 8/15 metrics passing
```

**Interpretation**:
- **p > 0.05**: Synthetic matches real (✓)
- **p < 0.05**: Synthetic differs from real (✗)

### Step 3: Distribution Comparison

```python
# Manual distribution check
import json
import statistics

with open('data/training/h3_train.json') as f:
    train = [json.loads(line) for line in f]

real = [s for s in train if s.get('source') == 'real']
synthetic = [s for s in train if s.get('source') == 'synthetic']

metric = 'pr_merge_rate'
real_values = [s['metrics'][metric] for s in real]
synthetic_values = [s['metrics'][metric] for s in synthetic]

print(f"\n{metric}:")
print(f"  Real:       mean={statistics.mean(real_values):.2f}, stdev={statistics.stdev(real_values):.2f}")
print(f"  Synthetic:  mean={statistics.mean(synthetic_values):.2f}, stdev={statistics.stdev(synthetic_values):.2f}")

# If means differ significantly → calibration needed
```

---

## Troubleshooting Guide

### Issue 1: "ingest_local.py hangs after API data fetch"

**Symptoms**: 
- Script runs for 2-3 hours
- Data shows ~2KB output

**Root Cause**: `--pr-diff-limit 0` bug
- Line 393-402 in `scripts/ingest_local.py`
- Zero converted to False, triggers unlimited (999_999)
- Attempts 4,882 PRs × 2 calls = 9,764 API calls

**Solution**:
```python
# Replace this:
pr_diff_limit = args.pr_diff_limit if args.pr_diff_limit else _NO_LIMIT

# With this:
pr_diff_limit = args.pr_diff_limit if args.pr_diff_limit > 0 else _NO_LIMIT
```

**Prevention**: Use `--pr-diff-limit -1` for unlimited (never `0`)

### Issue 2: "KS test shows 7 metrics failing"

**Symptoms**:
- `pr_merge_rate: KS=1.0` (completely opposite)
- `issue_resolution_rate: KS=0.75`
- Several others > 0.5

**Root Cause**: Synthetic data not calibrated from anything-llm repo

**Solution**: Re-calibrate personas

```bash
npm run calibrate-synthetic -- --source data/Mintplex-Labs_anything-llm_sprints.json
npm run generate-synthetic -- --count 100
npm run prepare-training:h3
```

**Expected Result**: 8-12 metrics should pass after recalibration

### Issue 3: "scipy not installed — skipping KS tests"

**Symptoms**:
- Training data prep succeeds
- But no KS validation report

**Solution**:
```bash
pip install scipy
npm run prepare-training:h3  # Re-run to validate
```

### Issue 4: "Feature values out of [0, 1] range"

**Symptoms**:
- Some metrics have values > 1.0
- Causes issues in ML model

**Root Cause**: New repository not in normalization range

**Solution**: Re-compute min/max normalization

```python
# In FeatureExtractor
# Recompute normalization for new repository
metric_values = [sprint[metric] for sprint in REPOSITORY_SPRINTS]
min_val = min(metric_values)
max_val = max(metric_values)

# Normalize: (value - min) / (max - min)
```

### Issue 5: "Imbalanced training data (all healthy or all at-risk)"

**Symptoms**:
- Training set has 90% at-risk or 90% healthy
- Model can't learn to distinguish

**Root Cause**: Synthetic generation weighted wrong

**Solution**: Check persona mix

```python
# In SyntheticSprintGenerator
# Ensure at_risk distribution is 50-60%

healthy_count = sum(1 for s in generated if s['label'] == 0)
at_risk_count = sum(1 for s in generated if s['label'] == 1)
ratio = at_risk_count / (healthy_count + at_risk_count)
assert 0.5 <= ratio <= 0.6, f"Bad ratio: {ratio}"
```

---

## Final Report Components

### Section 1: Data Collection Summary

```markdown
## Dataset Overview

| Metric | Value |
|--------|-------|
| Real Repository | Mintplex-Labs/anything-llm |
| Real Sprints | 74 |
| Date Range | 2023-01-01 to 2025-03-25 |
| Total Commits | 3,847 |
| Total Issues | 2,104 |
| Total PRs | 834 |
| Average Sprint Issues | 28.4 |
| Average Sprint PRs | 11.3 |
| Average Sprint Commits | 52.0 |

### Data Quality Metrics
- **Completeness**: 100% (all entities extracted)
- **Deduplication**: 99.8% unique (0.2% git duplicates)
- **Temporal Coverage**: 27 months continuous
- **At-Risk Distribution**: 55% (41/74 sprints)
```

### Section 2: Feature Engineering

```markdown
## Feature Extraction Results

### Metrics Computed: 25 total

#### Category: Activity
- Total Issues per sprint: [2, 145] (median: 28)
- Total PRs per sprint: [1, 89] (median: 11)
- Total Commits per sprint: [5, 280] (median: 52)
- Commit Frequency: [0.4, 20] commits/day (median: 3.7)

#### Category: Risk Signals
- Stalled Issues: [0, 28] (median: 2)
- Unreviewed PRs: [0, 15] (median: 1)
- Code Concentration: [0.2, 0.9] (median: 0.4)

### Feature Validation
- **Missing Values**: 0 (no NaN)
- **Outliers**: 3 extreme values (> 3σ) — retained as valid
- **Normalization**: Per-repository min-max scaling [0, 1]
```

### Section 3: Risk Labeling Accuracy

```markdown
## Risk Label Distribution

### Real Data (74 sprints)
- Healthy: 33 (44.6%)
- At-Risk: 41 (55.4%)

### Synthetic Data (100 sprints)
- Healthy: 46 (46.0%)
- At-Risk: 54 (54.0%)

**χ² Test**: χ² = 0.04, p = 0.84 (distributions match)

## Labeling Rule Performance
- **Rule 1 (High Activity)**: Triggers 18% of at-risk
- **Rule 2 (Low Resolution)**: Triggers 45% of at-risk
- **Rule 3 (PR Bottleneck)**: Triggers 28% of at-risk
- **Rule 4 (Team Risk)**: Triggers 22% of at-risk
- **Multi-Rule Coverage**: 89% of at-risk triggered by 2+ rules
```

### Section 4: Synthetic Data Quality

```markdown
## KS Test Results

| Metric | Real Mean | Synthetic Mean | KS Statistic | p-value | Status |
|--------|-----------|----------------|--------------|---------|--------|
| total_issues | 28.4 | 29.1 | 0.14 | 0.18 | ✓ Pass |
| commit_frequency | 3.7 | 3.8 | 0.22 | 0.08 | ✓ Pass |
| ... | ... | ... | ... | ... | ... |
| pr_merge_rate | 0.67 | 0.31 | 1.00 | 0.00 | ✗ FAIL |
| issue_resolution_rate | 0.58 | 0.45 | 0.75 | <0.01 | ✗ FAIL |
| ... | ... | ... | ... | ... | ... |

**Passing**: 8/15 metrics
**Failing**: 7/15 metrics (requires recalibration)
```

### Section 5: Training Data Composition

```markdown
## Final Dataset

### Overall
- Total Sprints: 174
- At-Risk: 95 (54.6%)
- Healthy: 79 (45.4%)

### Training Set
- Size: 151 sprints (87%)
- Real: 51 (33.8%)
- Synthetic: 100 (66.2%)
- At-Risk: 82 (54.3%)
- Healthy: 69 (45.7%)

### Validation Set
- Size: 10 sprints (6%)
- Real: 10 (100%)
- Synthetic: 0
- At-Risk: 6 (60%)
- Healthy: 4 (40%)

### Test Set
- Size: 13 sprints (7%)
- Real: 13 (100%)
- Synthetic: 0
- At-Risk: 7 (53.8%)
- Healthy: 6 (46.2%)

**Key Property**: Train uses mixed synthetic + real. Val/Test use real only.
```

### Section 6: Hidden Assumptions & Limitations

```markdown
## Assumptions Made

### Data Collection
1. **Commit Trailers**: PR metadata in Change-Id and Reviewed-on (Go repo specific)
2. **Issue References**: Extracted from commit messages using Fixes/Updates patterns
3. **Author Consistency**: Same author name = same person (no deduplication)

### Sprint Windows
1. **Fixed 14-Day Windows**: Arbitrary boundary, not tied to actual sprint schedules
2. **Partial Sprint Cutoff**: Sprints < 7 days discarded (may miss ramp-down periods)
3. **Entity Assignment**: Creation-based only (doesn't track lifecycle across sprints)

### Risk Labeling
1. **Percentile Thresholds**: Hard-coded to p25/p50/p75 (may not generalize)
2. **Rule Weights**: All rules equally weighted (no prioritization)
3. **Temporal Stability**: Labels based on single sprint (no trend analysis)

### Synthetic Generation
1. **Persona Independence**: Assumes personas are separable (reality is blended)
2. **Repository Specificity**: Calibrated on single repo (may overfit)
3. **Metric Independence**: Assumes metrics can be sampled independently (hidden correlations)

## Limitations

1. **Cold Start Problem**: New repository metrics outside [0, 1] range
2. **Concept Drift**: Real project risk factors may change over time
3. **Survivorship Bias**: Only analyzing successful repos (Go, anything-llm)
4. **Selection Bias**: Feature set designed for collaborative OSS (may not fit embedded systems)
```

### Section 7: Next Steps (M6+)

```markdown
## Recommendations for Fine-Tuning (M7)

### Pre-Training Checklist
- [ ] Install scipy, run KS tests, confirm 8+ metrics passing
- [ ] Recalibrate pr_merge_rate and issue_resolution_rate if 7 metrics still failing
- [ ] Verify no feature values outside [0, 1] range
- [ ] Check class balance: 54% at-risk ± 2%

### Fine-Tuning Strategy
1. **LoRA Adapters**: Freeze base model, train 10-20% new parameters
2. **Loss Function**: Binary cross-entropy with class weights (handle 54/46 imbalance)
3. **Validation Strategy**: Use real-only validation set to detect overfitting to synthetic
4. **Hyperparameter Search**: Learning rate [1e-4, 1e-3], batch size [16, 64]

### Expected Performance
- **Baseline** (untrained): 54% accuracy (always predict at-risk)
- **After fine-tuning**: Target 70-75% accuracy on real test set
- **Metric**: F1 score for balanced classification (both false positives and false negatives costly)
```

---

## Mermaid Diagrams Reference

### Decision Tree: Is Sprint At-Risk?

```mermaid
graph TD
    A["Sprint Metrics<br/>25 features computed"] --> B{High Activity?<br/>commit_freq > p75<br/>AND issues > p50}
    
    B -->|Yes| C{Low Resolution?<br/>resolution_rate < p25<br/>OR stalled > p75}
    C -->|Yes| D["🚨 AT-RISK<br/>Backlog Risk"]
    C -->|No| E["✓ HEALTHY"]
    
    B -->|No| F{PR Bottleneck?<br/>unreviewed > p50<br/>AND merge_rate < p50}
    
    F -->|Yes| G["🚨 AT-RISK<br/>Shipping Risk"]
    F -->|No| H{Team Risk?<br/>concentration > p75<br/>OR participation < p25}
    
    H -->|Yes| I["🚨 AT-RISK<br/>Key-Person Risk"]
    H -->|No| E
```

### Time Series: Typical Sprint Evolution

```mermaid
graph LR
    A["Day 1: Sprint Starts<br/>issues=0, prs=0"] 
    -->B["Day 4: Ramp Up<br/>issues=12, prs=4"]
    -->C["Day 7: Peak Activity<br/>issues=28, prs=11"]
    -->D["Day 11: Resolution Phase<br/>issues=2, prs=7<br/>merged=5"]
    -->E["Day 14: Wind Down<br/>stalled_issues=2<br/>unreviewed_prs=1"]
    
    F["Metrics: High Activity<br/>+ Low Resolution"]
    F -->|Indicates| G["🚨 AT-RISK<br/>Backlog accumulation"]
    
    D --> G
```

---

## Quick Reference: Common Commands

```bash
# Data Collection
npm run local:golang:hybrid  # Local + API
npm run local:golang:offline  # Local only (fast)

# Analysis
npm run generate-synthetic -- --count 100
npm run calibrate-synthetic -- --source data/go_sprints.json

# Training Prep
npm run prepare-training:h3  # Full pipeline
npm run prepare-training:baseline  # Alternative config

# Validation
python scripts/analyze_data.py  # Generate reports
```

### Multi-Repo Commands (3+ repositories)

```bash
# Step 1: Ingest each repo
npm run local:company:repo1:hybrid
npm run local:company:repo2:hybrid
npm run local:company:repo3:hybrid

# Step 2-4: Combine & normalize
python scripts/combine_repos.py           # Merge sprint files
python scripts/normalize_combined.py      # Per-repo normalization
python scripts/prepare_combined_training.py  # Final splits

# Step 5: One-liner for complete multi-repo pipeline
npm run local:company:repo1:hybrid && npm run local:company:repo2:hybrid && npm run local:company:repo3:hybrid && \
python scripts/combine_repos.py && python scripts/normalize_combined.py && python scripts/prepare_combined_training.py

# Output: data/training/h3_combined_train.json, h3_combined_val.json, h3_combined_test.json
```

---

## Final Checklist Before Fine-Tuning

- [ ] scipy installed (`pip install scipy`)
- [ ] KS tests running (output shows 8/15+ metrics passing)
- [ ] No feature values outside [0, 1]
- [ ] Training set: 151 sprints, 54% at-risk
- [ ] Validation set: 10 real-only sprints
- [ ] Test set: 13 real-only sprints
- [ ] Files saved: `data/training/h3_train.json`, `h3_val.json`, `h3_test.json`
- [ ] Git changes staged (ready for commit)

---

**Document Version**: 1.0  
**Last Updated**: March 26, 2026  
**Status**: Ready for M7 Fine-Tuning Phase
