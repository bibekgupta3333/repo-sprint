"""Process downloaded data and extract features for agents."""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts._core import Processor

def process_all_repos(data_dir: str = 'data/raw',
                      output_dir: str = 'data/processed') -> None:
    """Process all repositories and extract features."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    data_path = Path(data_dir)
    json_files = sorted(data_path.glob('*.json'))

    if not json_files:
        print(f"No JSON files found in {data_dir}")
        return

    print(f"\nProcessing {len(json_files)} file(s)...\n")
    for json_file in json_files:
        try:
            processor = Processor(str(json_file))
            processed = processor.process()
            output_file = Processor.save(processed, output_dir)
            print(f"  ✓ {processor.repo_name}")
        except Exception as e:
            print(f"  ✗ Error: {json_file.name}: {e}")

    print()


def show_usage():
    """Show usage information."""
    print("""
Usage: python scripts/process_data.py [options]

Description:
  Extract features from raw repository data for agent analysis

Options:
  --help           Show this help message and exit
  --input DIR      Input directory with raw data (default: data/raw)
  --output DIR     Output directory for processed data (default: data/processed)

Output:
  JSON files with extracted features:
    - Issue features (state, days open, labels)
    - PR features (additions, deletions, state)
    - Activity timeline
    - Contributor statistics

Example:
  python scripts/process_data.py
  python scripts/process_data.py --input data/raw --output data/processed
    """)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--help', action='store_true', help='Show help message')
    parser.add_argument('--input', type=str, default='data/raw', help='Input directory')
    parser.add_argument('--output', type=str, default='data/processed', help='Output directory')
    args = parser.parse_args()

    if args.help or '--help' in sys.argv or '-h' in sys.argv:
        show_usage()
    else:
        process_all_repos(args.input, args.output)
