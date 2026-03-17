"""Analyze downloaded GitHub repository data."""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts._core import Analyzer

def analyze_all_repos(data_dir: str = 'data/raw') -> None:
    """Analyze all repositories in data directory."""
    data_path = Path(data_dir)
    json_files = sorted(data_path.glob('*.json'))

    if not json_files:
        print(f"No JSON files found in {data_dir}")
        return

    for json_file in json_files:
        try:
            analyzer = Analyzer(str(json_file))
            analyzer.analyze()
        except Exception as e:
            print(f"Error analyzing {json_file}: {e}")

def show_usage():
    """Show usage information."""
    print("""
Usage: python scripts/analyze_data.py [options]

Description:
  Analyze downloaded GitHub repository data and show statistics

Options:
  --help          Show this help message and exit
  --dir DIR       Data directory (default: data/raw)

Output:
  Console output with statistics for each repository

Example:
  python scripts/analyze_data.py
  python scripts/analyze_data.py --dir data/raw
    """)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--help', action='store_true', help='Show help message')
    parser.add_argument('--dir', type=str, default='data/raw', help='Data directory')
    args = parser.parse_args()

    if args.help or '--help' in sys.argv or '-h' in sys.argv:
        show_usage()
    else:
        analyze_all_repos(args.dir)
