"""Simple data ingestion pipeline."""
import json
import argparse
from pathlib import Path
from src.scrapper.github import GitHubScraper
from src.data.preprocessor import SprintPreprocessor
from src.data.formatter import ChromaFormatter


def ingest_repo(owner: str, repo: str, output_dir: str = "data"):
    """Download, preprocess, and format repository data."""
    Path(output_dir).mkdir(exist_ok=True)

    scraper = GitHubScraper()
    print("\n=== STEP 1: Download ===\n")
    repo_data = scraper.scrape(owner, repo)

    print("\n=== STEP 2: Preprocess ===\n")
    preprocessor = SprintPreprocessor(repo_data)
    sprints = preprocessor.create_sprints()
    print(f"Created {len(sprints)} sprints")

    print("\n=== STEP 3: Format ===\n")
    all_docs = []
    for sprint in sprints:
        formatter = ChromaFormatter(sprint)
        docs = formatter.format_documents()
        all_docs.extend(docs)

    output_file = Path(output_dir) / f"{owner}_{repo}_documents.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_docs, f, indent=2)

    print(f"Saved {len(all_docs)} documents to {output_file}")
    print(f"\nIngestion complete: {owner}/{repo}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest GitHub repository data")
    parser.add_argument("owner", help="Repository owner")
    parser.add_argument("repo", help="Repository name")
    parser.add_argument("--output", default="data", help="Output directory")

    args = parser.parse_args()
    ingest_repo(args.owner, args.repo, args.output)
