"""Ingest repository data: download, analyze, and process."""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts._core import Scraper, Analyzer, Processor


def ingest_repo(owner: str, repo: str, verbose: bool = True) -> dict:
    """
    Ingest a single repository.

    Steps:
    1. Download from GitHub API
    2. Analyze raw data
    3. Process and extract features

    Args:
        owner: Repository owner (e.g., 'golang')
        repo: Repository name (e.g., 'go')
        verbose: Print progress messages

    Returns:
        Dictionary with paths to saved files
    """
    result = {}

    # Step 1: Download
    if verbose:
        print(f"\n{'='*60}")
        print(f"STEP 1: Downloading {owner}/{repo}")
        print(f"{'='*60}")

    scraper = Scraper()
    repo_data = scraper.ingest(owner, repo)

    if not repo_data:
        print(f"ERROR: Failed to download {owner}/{repo}")
        return result

    raw_file = scraper.save(repo_data)
    result['raw_file'] = raw_file

    # Step 2: Analyze
    if verbose:
        print(f"{'='*60}")
        print(f"STEP 2: Analyzing {owner}/{repo}")
        print(f"{'='*60}")

    analyzer = Analyzer(raw_file)
    analyzer.analyze()

    # Step 3: Process
    if verbose:
        print(f"{'='*60}")
        print(f"STEP 3: Processing {owner}/{repo}")
        print(f"{'='*60}")

    processor = Processor(raw_file)
    processed_data = processor.process()
    processed_file = Processor.save(processed_data)
    result['processed_file'] = processed_file

    print(f"✓ Ingestion complete for {owner}/{repo}")
    print(f"  Raw: {raw_file}")
    print(f"  Processed: {processed_file}\n")

    return result


def show_usage():
    """Show usage information."""
    print("""
Usage: python scripts/ingest.py <owner> <repo> [options]

Description:
  Ingest a GitHub repository: download, analyze, and process

Arguments:
  <owner>         Repository owner (e.g., golang)
  <repo>          Repository name (e.g., go)

Options:
  --help          Show this help message and exit
  --quiet         Suppress progress output

Examples:
  python scripts/ingest.py golang go
  python scripts/ingest.py rust-lang rust
  python scripts/ingest.py torvalds linux --quiet
    """)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('owner', nargs='?', default=None)
    parser.add_argument('repo', nargs='?', default=None)
    parser.add_argument('--help', action='store_true')
    parser.add_argument('--quiet', action='store_true')

    args = parser.parse_args()

    if args.help or not args.owner or not args.repo:
        show_usage()
        sys.exit(0)

    ingest_repo(args.owner, args.repo, verbose=not args.quiet)


if __name__ == '__main__':
    main()
