"""Simple data ingestion pipeline with complete code-change intelligence."""
import json
import argparse
from pathlib import Path
from src.scrapper.github import GitHubScraper
from src.data.preprocessor import SprintPreprocessor
from src.data.formatter import ChromaFormatter


def ingest_repo(
    owner: str,
    repo: str,
    output_dir: str = "data",
    fetch_diffs: bool = False,
    pr_diff_limit: int = 20,
):
    """Download, preprocess, and format repository data with detailed code-change intelligence."""
    Path(output_dir).mkdir(exist_ok=True)

    scraper = GitHubScraper()
    print("\n=== STEP 1: Download Code-Change Intelligence ===\n")
    repo_data = scraper.scrape(
        owner, repo, fetch_diffs=fetch_diffs, pr_diff_limit=pr_diff_limit
    )

    print("\n=== STEP 2: Preprocess into Sprints ===\n")
    preprocessor = SprintPreprocessor(repo_data)
    sprints = preprocessor.create_sprints()
    print(f"Created {len(sprints)} sprints")

    print("\n=== STEP 3: Format for ChromaDB ===\n")
    all_docs = []
    sprint_aligned_docs = []
    for sprint in sprints:
        formatter = ChromaFormatter(sprint)
        sprint_aligned_docs.append(formatter.format_documents_by_sprint())
        all_docs.extend(formatter.format_documents())

    # Save sprint-aligned documents
    output_file = Path(output_dir) / f"{owner}_{repo}_documents.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sprint_aligned_docs, f, indent=2)
    print(f"Saved {len(sprint_aligned_docs)} sprint-aligned documents to {output_file}")

    # Save flat Chroma documents
    chroma_output_file = Path(output_dir) / f"{owner}_{repo}_chroma_documents.json"
    with open(chroma_output_file, "w", encoding="utf-8") as f:
        json.dump(all_docs, f, indent=2)
    print(f"Saved {len(all_docs)} flat Chroma documents to {chroma_output_file}")

    # Save agent-ready sprint data with complete code-change intelligence
    sprint_output_file = Path(output_dir) / f"{owner}_{repo}_sprints.json"
    with open(sprint_output_file, "w", encoding="utf-8") as f:
        json.dump(sprints, f, indent=2)
    print(f"Saved {len(sprints)} sprints with code-change intelligence to {sprint_output_file}")

    # Print code-change summary
    if fetch_diffs and repo_data.get("commit_diffs"):
        print("\n=== Code-Change Summary ===\n")
        total_additions = sum(d.get("total_additions", 0) for d in repo_data["commit_diffs"])
        total_deletions = sum(d.get("total_deletions", 0) for d in repo_data["commit_diffs"])
        total_files = sum(d.get("files_changed", 0) for d in repo_data["commit_diffs"])

        print(f"Commits analyzed: {len(repo_data.get('commit_diffs', []))}")
        print(f"Total additions: {total_additions}")
        print(f"Total deletions: {total_deletions}")
        print(f"Total files changed: {total_files}")

        if repo_data["commit_diffs"]:
            lang_freq = {}
            for diff in repo_data["commit_diffs"]:
                for lang, count in diff.get("language_breakdown", {}).items():
                    lang_freq[lang] = lang_freq.get(lang, 0) + count
            print(f"Language breakdown: {lang_freq}")

    print(f"\n✅ Ingestion complete: {owner}/{repo}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest GitHub repository data with complete code-change intelligence")
    parser.add_argument("owner", help="Repository owner")
    parser.add_argument("repo", help="Repository name")
    parser.add_argument("--output", default="data", help="Output directory")
    parser.add_argument("--diffs", action="store_true", help="Fetch detailed code diffs (slower but more detailed)")
    parser.add_argument(
        "--pr-diff-limit",
        type=int,
        default=20,
        help="Max PRs to enrich with commits+files when using --diffs (0=skip)",
    )

    args = parser.parse_args()
    ingest_repo(
        args.owner,
        args.repo,
        args.output,
        fetch_diffs=args.diffs,
        pr_diff_limit=args.pr_diff_limit,
    )
