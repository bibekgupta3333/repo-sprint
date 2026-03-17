# Data Pipeline

Simple GitHub data collection and preprocessing for sprint intelligence research.

## Overview

Three-step pipeline:
1. **Download** - Scrape GitHub public API (no token needed)
2. **Preprocess** - Group data into synthetic 2-week sprints
3. **Format** - Prepare documents for Chroma vector ingestion

## Usage

```bash
python ingest.py <owner> <repo> [--output data]
```

Examples:
```bash
python ingest.py golang go
python ingest.py kubernetes kubernetes
python ingest.py torvalds linux
```

## Output

Creates JSON files in `data/` directory with documents formatted for Chroma:
- Sprint summary metadata
- Individual issues with state, labels
- Pull request changes (additions/deletions)
- Commit history

## Modules

**src/scrapper/github.py**
- `GitHubScraper` - Fetch issues, PRs, commits from GitHub API
- Handles pagination and errors gracefully
- Returns structured data with date fields

**src/data/preprocessor.py**
- `SprintPreprocessor` - Group commits/issues/PRs into 2-week sprints
- Creates synthetic sprint timeline (sprint_000, sprint_001, ...)
- Calculates sprint metrics (total items, closed count, code changes)

**src/data/formatter.py**
- `ChromaFormatter` - Convert sprint data to vector-ready documents
- Creates summary document per sprint
- Individual documents for issues, PRs, commits with metadata

## Design

- **Simple**: No unnecessary abstractions, focused on core functionality
- **Flexible**: Handles repos with/without PRs, issues, commits
- **Date-controlled**: Groups by 2-week windows for reproducible sprints
- **Chroma-ready**: Output format directly usable for embedding ingestion
