# Dataset Collection Scripts

Scripts for downloading and processing the Sprint Intelligence dataset.

## Prerequisites

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up GitHub token:**
   ```bash
   # Copy .env.example to .env
   cp ../.env.example ../.env
   
   # Edit .env and add your GitHub personal access token
   # Get token from: https://github.com/settings/tokens
   # Required scopes: repo, read:org
   ```

## Quick Start (Sample Data)

Download sample data for testing:

```bash
# Option 1: GitHub Archive (last 7 days)
python download_github_archive.py --sample

# Option 2: GitHub API (sample repos)
python collect_github_data.py --repos-file sample_repos.txt --max-repos 5
```

## Full Dataset Collection

### Option 1: GitHub Archive (Recommended for large-scale)

Download historical GitHub events:

```bash
# Download full date range (WARNING: ~450GB, takes hours/days)
python download_github_archive.py \
    --start-date 2020-03-01 \
    --end-date 2026-02-14 \
    --output-dir ../data/raw/events

# Download specific month (for testing)
python download_github_archive.py \
    --start-date 2026-01-01 \
    --end-date 2026-01-31 \
    --max-files 100
```

**Output:** JSONL files in `../data/raw/events/` (one per hour)

### Option 2: GitHub API (Recommended for targeted collection)

Collect specific repositories with milestone data:

```bash
# Prepare repository list
# Create a file 'repos.txt' with format: owner/repo (one per line)

# Collect data
python collect_github_data.py \
    --repos-file repos.txt \
    --output-dir ../data/raw

# Collect sample repos (for testing)
python collect_github_data.py \
    --repos-file sample_repos.txt \
    --max-repos 10
```

**Output:** JSON files in `../data/raw/repositories/` (one per repo)

### Option 3: GHTorrent (Alternative)

Download pre-collected data from GHTorrent:

```bash
# Visit: https://ghtorrent.org/downloads.html
# Download MySQL dumps or MongoDB dumps
# Import into local database
```

## Data Processing Pipeline

After downloading raw data:

```bash
# 1. Extract features (TODO: implement feature_extraction.py)
python feature_extraction.py \
    --input-dir ../data/raw \
    --output-dir ../data/processed/features

# 2. Generate embeddings (TODO: implement generate_embeddings.py)
python generate_embeddings.py \
    --input-dir ../data/processed/features \
    --output-dir ../data/processed/embeddings

# 3. Create train/val/test splits (TODO: implement create_splits.py)
python create_splits.py \
    --input-dir ../data/processed \
    --output-dir ../data/splits \
    --train-ratio 0.65 \
    --val-ratio 0.21 \
    --test-ratio 0.14
```

## Scripts Overview

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `download_github_archive.py` | Download GitHub Archive events | Date range | JSONL files |
| `collect_github_data.py` | Collect via GitHub API | Repo list | JSON files |
| `feature_extraction.py` | Extract 524-dim features | Raw data | Parquet files |
| `generate_embeddings.py` | Generate text embeddings | Processed data | Vector files |
| `create_splits.py` | Create dataset splits | All data | Train/val/test JSON |

## Repository Selection Criteria

For GitHub API collection, select repositories with:
- ✅ Active milestone usage (10+ milestones)
- ✅ Consistent activity (weekly commits)
- ✅ Team collaboration (3+ contributors)
- ✅ Mixed outcomes (both successful and delayed sprints)
- ✅ Public visibility

## Dataset Statistics Targets

Target statistics for full dataset:

- **Timespan:** March 2020 - February 2026 (6 years)
- **Organizations:** 500
- **Repositories:** 15,000
- **Milestones/Sprints:** 38,000
  - Train: 25,000 (65.8%)
  - Validation: 8,000 (21.1%)
  - Test: 5,000 (13.2%)
- **Events:** 3.8M GitHub events
- **Synthetic samples:** 5,000 (13.2% of total)

## Storage Requirements

```
data/
├── raw/                   ~450GB
│   ├── events/           ~400GB (GitHub Archive)
│   └── repositories/     ~50GB  (API data)
├── processed/            ~50GB
│   ├── features/         ~40GB  (Parquet)
│   └── embeddings/       ~10GB  (ChromaDB)
└── splits/               ~10MB  (Metadata)
```

## Troubleshooting

### Rate Limit Issues (GitHub API)

```python
# Check your rate limit
curl -H "Authorization: token YOUR_TOKEN" https://api.github.com/rate_limit

# Solutions:
# 1. Wait for rate limit reset
# 2. Use multiple tokens (not recommended)
# 3. Use GitHub Archive instead (no rate limits)
```

### Download Failures (GitHub Archive)

```bash
# Resume download from specific date
python download_github_archive.py \
    --start-date 2025-06-01 \
    --end-date 2026-02-14

# Script automatically skips existing files
```

### Storage Issues

```bash
# Check disk space
df -h

# Compress old files
gzip ../data/raw/events/*.jsonl

# Archive to external storage
tar -czf data-backup-$(date +%Y%m%d).tar.gz ../data/
```

## Next Steps

After collecting raw data:
1. ✅ Verify data quality and completeness
2. 📊 Run data analysis and statistics
3. 🔧 Implement feature extraction pipeline
4. 🧠 Generate embeddings using sentence-transformers
5. 📦 Create dataset splits
6. 🚀 Begin model training

## References

- [GitHub Archive](https://www.gharchive.org/)
- [GHTorrent Project](https://ghtorrent.org/)
- [GitHub API Documentation](https://docs.github.com/en/rest)
- [PyGithub Documentation](https://pygithub.readthedocs.io/)
