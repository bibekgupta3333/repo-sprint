# Synthetic Data & Augmented Training Strategy

**Date**: March 24, 2026  
**Status**: Implementation Plan  
**Objective**: Validate Hypothesis H3 — Cold-Start Deployment via Synthetic Data

---

## 1. Research Context

### Core Hypothesis (H3)

> "A model trained on **70% synthetic + 30% real** data will perform within **5% F1** of a fully real-data trained model for sprint risk classification."

### Why This Matters

| Problem | Impact | Solution |
| --- | --- | --- |
| New startups have 0 historical sprints | Can't train risk model | Synthetic generator produces 5K+ scenarios |
| GitHub API rate limits block data collection | Slow onboarding (days) | Local git ingestion extracts 65K commits in 40s |
| Small teams (3-10 devs) have sparse signals | High variance in metrics | Persona-based generation covers edge cases |

---

## 2. Current State Audit

### ✅ What's Built

| Component | File | Output | Status |
| --- | --- | --- | --- |
| Local Git Scraper | `src/scrapper/local_git.py` | 65,730 commits, 44,411 PRs, 17,936 issues | ✅ Complete |
| Sprint Preprocessor | `src/data/preprocessor.py` | 475 2-week sprints from golang/go | ✅ Complete |
| Feature Extractor | `src/data/features.py` | 18 metrics per sprint + risk labels | ✅ Complete |
| Chroma Formatter | `src/data/formatter.py` | 128,552 vector-ready documents | ✅ Complete |
| Synthetic Generator | `src/data/synthetic_generator.py` | 5K random sprints (uncalibrated) | ⚠️ Needs calibration |

### Real Data Distribution (golang/go, 475 sprints)

```
Metric                 min     median    mean      max       p90
─────────────────────────────────────────────────────────────────
total_commits           1      131       138       421       220
total_prs               0      100        94       319       208
total_issues            0       36        38       154        69
unique_authors          1       37        35        77        54
total_additions         1    13514     19669    302364     38760
total_deletions         0     6103     11984    222951     23994
files_changed           1      471       539      3134       965
issue_resolution_rate   0       0.9       0.8       1.0       1.0
pr_merge_rate           0       1.0       0.6       1.0       1.0
commit_frequency (d)    0.1    10.1      10.6      32.4      16.9
```

**Risk distribution**: 263/475 (55.4%) at-risk — indicates the `RiskLabeler` thresholds may need tuning for large repos.

### Schema Gap: Synthetic vs Real

| In Real but Missing from Synthetic | Purpose |
| --- | --- |
| `closed_issues` | Needed for resolution rate validation |
| `merged_prs` | Needed for merge rate validation |
| `total_additions` / `total_deletions` | Code churn signal (strongest risk predictor) |
| `files_changed` | Scope signal |
| `language_breakdown` | Multi-language risk |
| `code_changes` | Aggregate PR-level changes |

**Action needed**: Update `SyntheticSprintGenerator` to emit all 25 metrics matching the real schema.

---

## 3. Synthetic Sprint Generator — Calibration Plan

### 3.1 Architecture (No LLM needed for v1)

The current `SyntheticSprintGenerator` uses **persona-based statistical generation** which is correct for v1. LLM-based narrative generation is deferred to v2 (when we need explainability training data).

```
┌────────────────────┐    ┌──────────────────┐    ┌───────────────┐
│  Real Sprint Stats │───▶│ Distribution Fit │───▶│ 5 Personas    │
│  (475 golang/go)   │    │ (per metric)     │    │ (calibrated)  │
└────────────────────┘    └──────────────────┘    └───────┬───────┘
                                                          │
                          ┌──────────────────┐            ▼
                          │ RiskLabeler      │◀── 5K Synthetic Sprints
                          │ (same as real)   │    (schema-compatible)
                          └──────────────────┘
```

### 3.2 Persona Recalibration (from real data)

Current personas are for small startups (3-10 devs). For calibration we need **two persona sets**:

**Set A: golang/go scale (large OSS project)**

| Persona | Commits | PRs | Issues | Authors | Weight | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `active_sprint` | 100-250 | 80-200 | 30-80 | 25-55 | 3.0 | Normal 2-week cycle |
| `release_sprint` | 200-420 | 150-320 | 50-150 | 40-77 | 1.0 | Pre-release push |
| `quiet_sprint` | 1-50 | 0-30 | 0-10 | 1-15 | 1.0 | Holiday/freeze |
| `blocked_sprint` | 30-100 | 20-80 | 60-150 | 15-40 | 1.0 | Many open issues |
| `refactor_sprint` | 50-150 | 30-80 | 5-20 | 10-30 | 1.0 | High churn, few issues |

**Set B: Small startup (2-3 repos, target use case)**

| Persona | Commits | PRs | Issues | Authors | Weight | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `commit_first` | 5-30 | 0-10 | 0-2 | 1-2 | 3.0 | Solo dev, commits direct |
| `healthy_flow` | 10-40 | 5-20 | 3-15 | 2-4 | 2.0 | Normal sprint |
| `blocked_issues` | 5-20 | 2-10 | 10-50 | 1-3 | 1.0 | Support burden |
| `pr_bottleneck` | 15-50 | 20-60 | 0-3 | 3-6 | 1.0 | Review backlog |
| `quiet_sprint` | 0-10 | 0-3 | 0-1 | 0-2 | 1.0 | Vacation / early stage |

### 3.3 Missing Metrics to Add

Update `_generate_metrics()` to also emit:

```python
# Code churn (derived from total_code_changes)
"total_additions": int(total_changes * 0.65),  # ~65% adds typical
"total_deletions": int(total_changes * 0.35),  # ~35% deletes
"files_changed": random.randint(10, total_changes // 50),

# Activity details
"closed_issues": int(total_issues * issue_resolution),
"merged_prs": int(total_prs * pr_merge),
"code_changes": total_changes,

# Language (simplified)
"language_breakdown": {"Go": total_changes},
```

---

## 4. Augmented Training Dataset — Build Plan

### 4.1 Data Sources

| Source | Sprints | Type | Schema | Available |
| --- | --- | --- | --- | --- |
| golang/go (local) | 475 | Real | 25 metrics + risk | ✅ Now |
| Synthetic Set A (large OSS) | 2,000 | Synthetic | 25 metrics + risk | 🔜 Next |
| Synthetic Set B (small startup) | 3,000 | Synthetic | 25 metrics + risk | 🔜 Next |
| **Total** | **5,475** | **Mixed** | **Unified** | — |

### 4.2 Split Strategy (H3 validation)

```
┌──────────────────────────────────────────────────────┐
│              Augmented Dataset (5,475 sprints)        │
├──────────────────────────────────────────────────────┤
│                                                      │
│  Training (70%)  │  Validation (15%)  │  Test (15%)  │
│  3,832 sprints   │  821 sprints       │  822 sprints │
│                  │                    │              │
│  ┌──────────┐    │  ┌──────────────┐  │  Real only   │
│  │Syn: 3,500│    │  │ Mixed        │  │  (72 real    │
│  │Real: 332 │    │  │              │  │   sprints)   │
│  └──────────┘    │  └──────────────┘  │              │
└──────────────────────────────────────────────────────┘
```

**Key design decision**: Test set is **real-only** (72 golang/go sprints) to ensure we measure performance on actual data, not synthetic.

### 4.3 Experimental Configurations

| Config | Training Data | Test Data | Purpose |
| --- | --- | --- | --- |
| **Baseline** | 332 real sprints (70%) | 72 real sprints | Real-only performance |
| **H3-Test** | 3,500 syn + 332 real (70/30) | 72 real sprints | Hypothesis validation |
| **Syn-Only** | 3,832 synthetic | 72 real sprints | Pure synthetic quality |
| **Ablation A** | Set A only + real | 72 real sprints | Large OSS persona value |
| **Ablation B** | Set B only + real | 72 real sprints | Small startup persona value |

**Success criterion (H3)**: `F1(H3-Test) ≥ F1(Baseline) - 0.05`

---

## 5. Validation Pipeline

### 5.1 Metrics

| Metric | Target | Measures |
| --- | --- | --- |
| F1 Score (binary risk) | > 0.85 | Risk classification accuracy |
| Precision (at-risk) | > 0.80 | False alarm rate |
| Recall (at-risk) | > 0.85 | Missed risk rate |
| KL Divergence | < 0.1 | Distribution similarity (syn vs real) |
| Statistical Tests | p < 0.05 | Kolmogorov-Smirnov per metric |

### 5.2 Realism Validation (synthetic quality)

Before training, validate synthetic data realism:

1. **Per-metric KS test**: Compare each of 18 metrics between synthetic and real distributions
2. **Risk label distribution**: Ensure at-risk ratio is within ±10% of real (55.4%)
3. **Cross-correlation**: Verify inter-metric correlations match (e.g., high commits → high files_changed)
4. **Temporal plausibility**: Sprint progressions should show realistic patterns

---

## 6. Implementation Sequence

### Phase 1: Calibrate Generator (Est. 2-3 hours)

> No new files needed — update existing `src/data/synthetic_generator.py`

1. Add the 7 missing metrics to `_generate_metrics()`
2. Create `GoPersona` set calibrated from real 475-sprint stats
3. Create `StartupPersona` set for target use case
4. Add `--calibrate` flag to read real sprint stats and auto-fit ranges
5. Validate output schema matches `SprintPreprocessor` output exactly

### Phase 2: Build Training Script (Est. 2-3 hours)

> New file: `scripts/prepare_training_data.py`

1. Load real sprints (`data/golang_go_sprints.json`)
2. Generate synthetic sprints (5K)
3. Merge with configurable ratios (70/30 default)
4. Apply stratified split (train/val/test)
5. Save splits to `data/training/`
6. Run KS tests for realism validation
7. Output summary report

### Phase 3: Add npm Commands (Est. 5 min)

```json
"generate-synthetic": "python src/data/synthetic_generator.py --count 5000 --personas all",
"prepare-training": "python scripts/prepare_training_data.py --real data/golang_go_sprints.json --synthetic 5000 --ratio 0.7",
"validate-synthetic": "python scripts/validate_synthetic.py"
```

---

## 7. Connection to Downstream Components

```
Local Git Clone (golang/go)
        │
        ▼
LocalGitScraper ──────────── 65K commits, 44K PRs, 18K issues
        │
        ▼
SprintPreprocessor ────────── 475 real sprints (25 metrics each)
        │                          │
        ▼                          ▼
 ChromaFormatter              SyntheticSprintGenerator
 128K vector docs              5K synthetic sprints
        │                          │
        ▼                          ▼
   RAG Pipeline          Augmented Training Dataset
   (Phase 3)              (70% syn + 30% real)
        │                          │
        ▼                          ▼
   LLM Reasoning          Risk Classifier (Phase 3)
   Agent (Phase 4)         F1 > 0.85 target
```

---

## 8. Risk Labeler Calibration Note

Current real data shows **55.4% at-risk** which is unusually high. This is because:

- golang/go always has `stalled_issues ≥ 3` (large project = many open issues)
- The `RiskLabeler` thresholds were designed for small startups, not large OSS

**Action**: Create repo-size-aware thresholds using percentile-based scoring:

```python
# Instead of absolute thresholds:
if metrics["stalled_issues"] >= 3:  # too aggressive for large repos

# Use percentile-relative thresholds:
if metrics["stalled_issues"] > p75_stalled:  # adapts to repo scale
```

This will be addressed when the risk classifier model is built (Phase 3), as the model will learn appropriate thresholds from data rather than hard-coded rules.
