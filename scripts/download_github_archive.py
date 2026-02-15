#!/usr/bin/env python3
"""
Download GitHub Archive data for sprint/milestone analysis.

GitHub Archive: https://www.gharchive.org/
Downloads hourly event files and filters for relevant events.

Usage:
    python download_github_archive.py --start-date 2020-03-01 --end-date 2026-02-14
"""

import os
import gzip
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict
import requests
from tqdm import tqdm
import time

# Event types relevant to sprint/milestone tracking
RELEVANT_EVENT_TYPES = {
    'IssuesEvent',
    'PullRequestEvent',
    'PushEvent',
    'IssueCommentEvent',
    'PullRequestReviewEvent',
    'PullRequestReviewCommentEvent',
    'MilestoneEvent',
}


class GitHubArchiveDownloader:
    """Download and filter GitHub Archive data."""

    def __init__(self, output_dir: str = "data/raw/events"):
        # If running from scripts/ directory, adjust path
        if not Path(output_dir).is_absolute() and not output_dir.startswith('.'):
            # Assume we're in project root or scripts directory
            if Path('scripts').exists():
                # We're in project root
                self.output_dir = Path(output_dir)
            else:
                # We're in scripts directory
                self.output_dir = Path('..') / output_dir
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = "https://data.gharchive.org"

    def generate_urls(self, start_date: datetime, end_date: datetime) -> List[str]:
        """Generate list of GitHub Archive URLs for date range."""
        urls = []
        current = start_date

        while current <= end_date:
            for hour in range(24):
                timestamp = current.replace(hour=hour)
                url = f"{self.base_url}/{timestamp.strftime('%Y-%m-%d')}-{hour}.json.gz"
                urls.append((timestamp, url))
            current += timedelta(days=1)

        return urls

    def download_and_filter(self, timestamp: datetime, url: str) -> int:
        """
        Download a single archive file and filter events.

        Returns:
            Number of relevant events extracted.
        """
        output_file = self.output_dir / f"{timestamp.strftime('%Y-%m-%d-%H')}.jsonl"

        # Skip if already processed
        if output_file.exists():
            print(f"⏭️  Skipping {timestamp.strftime('%Y-%m-%d %H:00')} (already exists)")
            return 0

        try:
            # Download compressed file
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # Decompress and filter
            data = gzip.decompress(response.content).decode('utf-8')
            lines = data.strip().split('\n')

            relevant_events = []
            for line in lines:
                try:
                    event = json.loads(line)

                    # Filter by event type
                    if event.get('type') in RELEVANT_EVENT_TYPES:
                        # Extract key fields to reduce storage
                        relevant_events.append({
                            'id': event.get('id'),
                            'type': event.get('type'),
                            'actor': event.get('actor'),
                            'repo': event.get('repo'),
                            'payload': event.get('payload'),
                            'created_at': event.get('created_at'),
                            'org': event.get('org'),
                        })
                except json.JSONDecodeError:
                    continue

            # Write filtered events
            if relevant_events:
                with open(output_file, 'w') as f:
                    for event in relevant_events:
                        f.write(json.dumps(event) + '\n')

            return len(relevant_events)

        except requests.RequestException as e:
            print(f"❌ Error downloading {url}: {e}")
            return 0
        except Exception as e:
            print(f"❌ Error processing {url}: {e}")
            return 0

    def download_range(self, start_date: datetime, end_date: datetime,
                       max_files: int = None):
        """Download and process a date range."""
        urls = self.generate_urls(start_date, end_date)

        if max_files:
            urls = urls[:max_files]

        print(f"📥 Downloading {len(urls)} archive files...")
        print(f"📅 Date range: {start_date.date()} to {end_date.date()}")
        print(f"📁 Output directory: {self.output_dir.absolute()}")
        print()

        total_events = 0
        successful = 0

        for timestamp, url in tqdm(urls, desc="Downloading"):
            count = self.download_and_filter(timestamp, url)
            if count > 0:
                total_events += count
                successful += 1

            # Be nice to GitHub Archive
            time.sleep(0.5)

        print()
        print(f"✅ Download complete!")
        print(f"   Files processed: {successful}/{len(urls)}")
        print(f"   Total events extracted: {total_events:,}")
        print(f"   Average events per file: {total_events // max(successful, 1):,}")


def main():
    parser = argparse.ArgumentParser(
        description="Download GitHub Archive data for sprint analysis"
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default='2020-03-01',
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default='2026-02-14',
        help='End date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/raw/events',
        help='Output directory for downloaded data'
    )
    parser.add_argument(
        '--max-files',
        type=int,
        help='Maximum number of files to download (for testing)'
    )
    parser.add_argument(
        '--sample',
        action='store_true',
        help='Download sample data (last 7 days only)'
    )

    args = parser.parse_args()

    # Parse dates
    if args.sample:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        print("🧪 Sample mode: Downloading last 7 days")
    else:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')

    # Create downloader and run
    downloader = GitHubArchiveDownloader(args.output_dir)
    downloader.download_range(start_date, end_date, args.max_files)


if __name__ == '__main__':
    main()
