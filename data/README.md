# Dataset Directory

This directory contains the downloaded and processed dataset for the Sprint Intelligence system.

## Directory Structure

```
data/
├── raw/                    # Raw data from GitHub Archive
│   ├── events/            # GitHub events (JSON)
│   ├── milestones/        # Milestone data
│   └── repositories/      # Repository metadata
├── processed/             # Processed feature vectors
│   ├── features/          # Parquet files with 524-dim features
│   └── embeddings/        # Cached embeddings
├── synthetic/             # LLM-generated synthetic data
└── splits/                # Train/validation/test splits
    ├── train.json
    ├── val.json
    └── test.json
```

## Dataset Specifications

- **Source**: GitHub Archive + GHTorrent + GitHub API
- **Timespan**: March 2020 - February 2026 (6 years)
- **Organizations**: 500 active organizations
- **Repositories**: 15,000 repositories with active milestones
- **Total Samples**: 38,000 sprint/milestone instances
  - Training: 25,000
  - Validation: 8,000
  - Testing: 5,000
- **Events**: ~3.8M GitHub events
- **Synthetic**: 5,000 LLM-generated scenarios

## Download Instructions

See `../scripts/README.md` for data collection instructions.

## Data Format

### Raw Events (JSON Lines)
```json
{"type": "IssuesEvent", "repo": "microsoft/vscode", "created_at": "2026-02-14T12:00:00Z", ...}
```

### Processed Features (Parquet)
- 524-dimensional feature vectors
- Columns: sprint_id, org_id, repo_id, features (array)
- Compressed with Snappy

### Dataset Splits (JSON)
```json
{
  "sprint_id": "sprint_12345",
  "org_id": "microsoft",
  "repo_id": "vscode",
  "outcome": "success",
  "completion_rate": 0.92
}
```

## Storage Requirements

- Raw data: ~450GB
- Processed data: ~50GB
- Embeddings: ~2GB
- Total: ~500GB
