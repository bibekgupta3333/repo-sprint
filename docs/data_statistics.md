# Data Statistics Report

## M2: Dataset Collection Summary

### Real GitHub Data Collected

| Repository | Documents | Sprints | Size |
|---|---|---|---|
| golang/go | 753 | 22 | 923 KB |
| kubernetes/kubernetes | 568 | 16 | 906 KB |
| torvalds/linux | 257 | 8 | 96 KB |
| **Total** | **1,578** | **46** | **1.9 MB** |

### Data Pipeline Statistics

**Ingestion:**
- GitHub API calls: 3 repositories
- Issues processed: ~1,200+
- PRs processed: ~800+
- Commits processed: ~2,000+

**Preprocessing:**
- Sprint windows: 14-day periods
- Sprints created: 46 real sprints
- Field completeness: 100% (no null critical fields)

**Feature Extraction:**
- Features per sprint: 18 metrics
- Metric categories: Temporal (3), Activity (6), Code (3), Risk (4), Team (2)
- Features calculated: 100% success rate

**Labeling:**
- Risk labels created: 46 real sprints labeled
- Label distribution: Random (baseline for real data)
- Risk score range: 0.0-0.8 (realistic)

### Labeled Dataset

| Dataset | Count | Percentage |
|---|---|---|
| Training | 4,032 | 80% |
| Validation | 504 | 10% |
| Test | 504 | 10% |
| **Total** | **5,040** | **100%** |

**Data Sources:**
- Synthetic: 5,000 sprints (99.2%)
- Real (GitHub API): 40 sprints (0.8%)

**Class Distribution (Training Set):**
- Not at-risk (label=0): 3,222 (79.8%)
- At-risk (label=1): 810 (20.2%)

### Data Quality

**Metrics:**
- Zero null values in features
- No duplicate sprints
- All features normalized to valid ranges
- Risk scores: 0.0-1.0 range

**Data Format:**
```json
{
  "sprint_id": "string",
  "repo": "string",
  "features": {
    "days_span": 14,
    "total_issues": 8,
    "total_prs": 12,
    ...18 features total
  },
  "label": 0 or 1,
  "risk_score": 0.15,
  "source": "synthetic" or "real"
}
```

### Ready for Training

- ✓ 5,040 examples with complete features
- ✓ 18 metrics per sprint
- ✓ Binary classification labels
- ✓ Risk scores for regression (optional)
- ✓ Train/val/test splits ready
