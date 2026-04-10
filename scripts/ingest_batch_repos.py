#!/usr/bin/env python3
"""
Simple batch ingestion script for multiple repos.
Clones/updates repos and then indexes them in ChromaDB.
"""
import subprocess
import csv
import os
from pathlib import Path

# Configuration
REPOS_CSV = "data/repos_for_ingestion_100.csv"
REPOS_DIR = "repos"
MAX_REPOS = 100
DIFFS = True

def run_command(cmd, description):
    """Run a shell command and report status."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    try:
        result = subprocess.run(cmd, shell=True, check=True)
        print(f"✓ Success: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed: {description}")
        print(f"  Error: {e}")
        return False

def clone_or_update_repo(owner, repo, git_url):
    """Clone repo if it doesn't exist, or update it if it does."""
    repo_path = f"{REPOS_DIR}/{owner}/{repo}"

    if os.path.exists(repo_path):
        print(f"  → Repo exists, updating...")
        update_cmd = f"cd {repo_path} && git pull origin main 2>/dev/null || git pull origin master"
        run_command(update_cmd, f"Updating {owner}/{repo}")
    else:
        print(f"  → Cloning repo...")
        # Create owner directory if needed
        os.makedirs(f"{REPOS_DIR}/{owner}", exist_ok=True)
        clone_cmd = f"git clone {git_url} {repo_path}"
        if not run_command(clone_cmd, f"Cloning {owner}/{repo}"):
            return False

    return True

def ingest_repo(owner, repo):
    """Ingest a single repo and its ChromaDB data."""
    repo_path = f"{REPOS_DIR}/{owner}/{repo}"

    # Check if repo exists locally after clone
    if not os.path.exists(repo_path):
        print(f"⊘ Skipping {owner}/{repo} - repo directory not found")
        return False

    print(f"\n[{owner}/{repo}]")

    # Step 1: Ingest local repo with diffs, issues, and PRs
    ingest_cmd = (
        f"python scripts/ingest_local.py {repo_path} "
        f"--owner {owner} --repo {repo} --diffs "
        f"--issues-limit 0 --prs-limit 0 --pr-diff-limit 0"
    )
    if not run_command(ingest_cmd, f"Ingesting {owner}/{repo} (commits + issues + PRs)"):
        return False

    # Step 2: Index in ChromaDB for RAG
    chroma_cmd = f"python3 -m src.chromadb --org {owner} --repo {repo}"
    if not run_command(chroma_cmd, f"Indexing {owner}/{repo} in ChromaDB"):
        return False

    return True

def main():
    """Main batch ingestion process."""
    # Read repos from CSV
    repos = []
    with open(REPOS_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= MAX_REPOS:
                break
            full_name = row['full_name']
            owner, repo = full_name.split('/')
            git_url = row['html_url'] + '.git'
            repos.append((owner, repo, git_url))

    print(f"Found {len(repos)} repos to ingest")
    print(f"Max repos to process: {MAX_REPOS}")

    # Process repos one by one
    successful = 0
    failed = 0
    skipped = 0

    for i, (owner, repo, git_url) in enumerate(repos, 1):
        print(f"\n[{i}/{len(repos)}] Processing {owner}/{repo}...")

        # Clone/update repo first
        if not clone_or_update_repo(owner, repo, git_url):
            skipped += 1
            continue

        # Then ingest it
        if ingest_repo(owner, repo):
            successful += 1
        else:
            skipped += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"  BATCH INGESTION COMPLETE")
    print(f"{'='*60}")
    print(f"✓ Successful: {successful}")
    print(f"⊘ Skipped: {skipped}")
    print(f"\nTotal repos processed: {successful + skipped}/{len(repos)}")

if __name__ == "__main__":
    main()
