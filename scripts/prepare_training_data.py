"""
Prepare augmented training dataset from real + synthetic sprints.

Implements H3 validation: test set is always **real-only** so we
measure model performance on actual data, not synthetic.

Supports multiple experimental configurations for ablation:
  - baseline:  real-only train/test
  - h3:        70% synthetic + 30% real train, real-only test
  - syn_only:  100% synthetic train, real-only test
  - custom:    user-specified ratio

Usage:
    python scripts/prepare_training_data.py
    python scripts/prepare_training_data.py --real data/golang_go_sprints.json --synthetic 5000
    python scripts/prepare_training_data.py --config h3
    python scripts/prepare_training_data.py --config baseline
"""

import json
import glob
import random
import argparse
import os
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.synthetic_generator import SyntheticSprintGenerator


# ------------------------------------------------------------------ #
#  Core pipeline                                                      #
# ------------------------------------------------------------------ #

def load_real_sprints(path: str) -> list[dict]:
    """Load real sprints and normalize to training format.

    ``path`` can be:
      - ``"all"`` to auto-discover every
        ``*_sprints.json`` in ``data/``
      - A specific file path
    """
    if path == "all":
        data_dir = os.path.join(
            os.path.dirname(__file__), "..", "data")
        files = sorted(glob.glob(
            os.path.join(data_dir, "*_sprints.json")))
        # Exclude synthetic sprint files
        files = [
            f for f in files
            if "synthetic" not in os.path.basename(f)
        ]
        if not files:
            print("  No *_sprints.json found in data/")
            return []
        print(f"  Found {len(files)} sprint file(s): "
              + ", ".join(os.path.basename(f) for f in files))
    else:
        files = [path]

    examples = []
    for fpath in files:
        with open(fpath, encoding="utf-8") as f:
            sprints = json.load(f)
        repo_name = os.path.basename(fpath).replace(
            "_sprints.json", "")
        for s in sprints:
            m = s.get("metrics", {})
            examples.append({
                "sprint_id": s["sprint_id"],
                "repo": s.get("repo", repo_name),
                "features": m,
                "label": (
                    1 if s.get("risk_label", {}).get(
                        "is_at_risk") else 0),
                "risk_score": s.get(
                    "risk_label", {}).get(
                    "risk_score", 0),
                "source": "real",
            })
    return examples


def load_or_generate_synthetic(
    synthetic_path: str | None,
    count: int,
    personas: str,
    seed: int,
) -> list[dict]:
    """Load existing or generate fresh synthetic sprints."""
    if synthetic_path and os.path.exists(synthetic_path):
        print(f"  Loading existing synthetic from {synthetic_path}")
        with open(synthetic_path, encoding="utf-8") as f:
            sprints = json.load(f)
    else:
        print(f"  Generating {count} synthetic sprints "
              f"(personas={personas}, seed={seed})")
        gen = SyntheticSprintGenerator(personas=personas, seed=seed)
        sprints = gen.generate(count=count)

    examples = []
    for s in sprints:
        m = s.get("metrics", {})
        examples.append({
            "sprint_id": s["sprint_id"],
            "repo": s.get("repo", ""),
            "features": m,
            "label": 1 if s.get("risk_label", {}).get("is_at_risk") else 0,
            "risk_score": s.get("risk_label", {}).get("risk_score", 0),
            "source": "synthetic",
            "persona": s.get("persona", ""),
        })
    return examples


def stratified_split(
    examples: list[dict],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[list, list, list]:
    """Split with stratification on label."""
    pos = [e for e in examples if e["label"] == 1]
    neg = [e for e in examples if e["label"] == 0]
    random.shuffle(pos)
    random.shuffle(neg)

    def _split(items, tf, vf):
        n = len(items)
        t = int(n * tf)
        v = int(n * vf)
        return items[:t], items[t:t + v], items[t + v:]

    pos_train, pos_val, pos_test = _split(pos, train_frac, val_frac)
    neg_train, neg_val, neg_test = _split(neg, train_frac, val_frac)

    train = pos_train + neg_train
    val = pos_val + neg_val
    test = pos_test + neg_test
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    return train, val, test


def ks_test_realism(real: list, synthetic: list) -> dict:
    """Kolmogorov-Smirnov test per metric between real and synthetic."""
    try:
        from scipy.stats import ks_2samp
    except ImportError:
        print("  (scipy not installed — skipping KS tests)")
        return {}

    numeric_keys = [
        "total_commits", "total_prs", "total_issues", "unique_authors",
        "total_additions", "total_deletions", "files_changed",
        "issue_resolution_rate", "pr_merge_rate", "commit_frequency",
        "total_code_changes", "avg_pr_size", "code_concentration",
        "stalled_issues", "unreviewed_prs",
    ]

    results: dict = {}
    for key in numeric_keys:
        real_vals = [
            e["features"].get(key, 0) for e in real
            if isinstance(e["features"].get(key, 0), (int, float))
        ]
        syn_vals = [
            e["features"].get(key, 0) for e in synthetic
            if isinstance(e["features"].get(key, 0), (int, float))
        ]
        if not real_vals or not syn_vals:
            continue
        stat, pval = ks_2samp(real_vals, syn_vals)
        results[key] = {
            "ks_statistic": round(stat, 4),
            "p_value": round(pval, 6),
            "similar": pval > 0.05,
        }
    return results


def print_split_report(
    label: str,
    train: list,
    val: list,
    test: list,
    ks_results: dict | None = None,
):
    """Print summary of a dataset split."""
    total = len(train) + len(val) + len(test)
    print(f"\n{'=' * 60}")
    print(f"  Config: {label}")
    print(f"{'=' * 60}")
    print(f"  Total:       {total:,}")
    print(f"  Train:       {len(train):,}  "
          f"({100 * len(train) / total:.0f}%)")
    print(f"  Validation:  {len(val):,}  "
          f"({100 * len(val) / total:.0f}%)")
    print(f"  Test:        {len(test):,}  "
          f"({100 * len(test) / total:.0f}%)")

    def _dist(data, name):
        src = Counter(e["source"] for e in data)
        lbl = Counter(e["label"] for e in data)
        parts = []
        for s, c in sorted(src.items()):
            parts.append(f"{s}={c}")
        risk_pct = (
            100 * lbl.get(1, 0) / len(data)
        ) if data else 0
        print(f"    {name}: {', '.join(parts)} | "
              f"at-risk={risk_pct:.1f}%")

    _dist(train, "Train")
    _dist(val, "Val  ")
    _dist(test, "Test ")

    if ks_results:
        passing = sum(1 for v in ks_results.values() if v["similar"])
        total_tests = len(ks_results)
        print(f"\n  KS Realism Test: {passing}/{total_tests} metrics "
              f"pass (p > 0.05)")
        for key, res in sorted(ks_results.items()):
            flag = "✅" if res["similar"] else "❌"
            print(f"    {flag} {key:<28s} KS={res['ks_statistic']:.3f}  "
                  f"p={res['p_value']:.4f}")


def save_split(output_dir: str, name: str, train, val, test):
    """Save train/val/test to JSON files."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for split_name, data in [
        ("train", train), ("val", val), ("test", test),
    ]:
        fpath = os.path.join(output_dir, f"{name}_{split_name}.json")
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    # Also save combined
    combined = train + val + test
    fpath = os.path.join(output_dir, f"{name}_all.json")
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    print(f"  Saved to {output_dir}/{name}_*.json")


# ------------------------------------------------------------------ #
#  Experimental configurations                                        #
# ------------------------------------------------------------------ #

def build_baseline(real: list[dict]) -> tuple[list, list, list]:
    """Config: Real-only (baseline for H3 comparison)."""
    return stratified_split(real, 0.70, 0.15)


def build_h3(
    real: list[dict],
    synthetic: list[dict],
    syn_ratio: float = 0.70,
) -> tuple[list, list, list]:
    """Config: H3 test — syn_ratio synthetic + (1-syn_ratio) real train.

    Test set is always real-only.
    """
    # Split real: 70% for train pool, 15% val, 15% test
    real_train_pool, real_val, real_test = stratified_split(
        real, 0.70, 0.15,
    )

    # Calculate how many synthetic to add for the desired ratio
    n_real_train = len(real_train_pool)
    n_syn_needed = int(n_real_train * syn_ratio / (1 - syn_ratio))
    n_syn_needed = min(n_syn_needed, len(synthetic))

    syn_sample = random.sample(synthetic, n_syn_needed)
    train = real_train_pool + syn_sample
    random.shuffle(train)

    return train, real_val, real_test


def build_syn_only(
    real: list[dict],
    synthetic: list[dict],
) -> tuple[list, list, list]:
    """Config: 100% synthetic train, real-only test."""
    _, real_val, real_test = stratified_split(real, 0.70, 0.15)

    syn_train, syn_val_extra, _ = stratified_split(
        synthetic, 0.85, 0.15,
    )

    return syn_train, real_val, real_test


# ------------------------------------------------------------------ #
#  CLI                                                                 #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Prepare augmented training dataset",
    )
    parser.add_argument(
        "--real", default="all",
        help=(
            "Path to real sprints JSON, or 'all' to "
            "auto-discover *_sprints.json in data/"
        ),
    )
    parser.add_argument(
        "--synthetic-path", default="data/synthetic_sprints.json",
        help="Path to existing synthetic sprints (or generate if absent)",
    )
    parser.add_argument(
        "--synthetic-count", type=int, default=5000,
        help="Number of synthetic sprints to generate if no file exists",
    )
    parser.add_argument(
        "--personas", default="auto",
        choices=["large_oss", "startup", "all", "auto"],
        help="Persona set for generation (auto = calibrate from real data)",
    )
    parser.add_argument(
        "--config", default="all",
        choices=["baseline", "h3", "syn_only", "all"],
        help="Experimental config to build",
    )
    parser.add_argument(
        "--syn-ratio", type=float, default=0.70,
        help="Synthetic ratio for H3 config (default: 0.70)",
    )
    parser.add_argument(
        "--output", default="data/training",
        help="Output directory",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # Load data
    print("Loading data ...")
    real = load_real_sprints(args.real)
    print(f"  Real: {len(real)} sprints "
          f"(at-risk: {sum(1 for e in real if e['label'] == 1)})")

    synthetic = load_or_generate_synthetic(
        args.synthetic_path,
        args.synthetic_count,
        args.personas,
        args.seed,
    )
    print(f"  Synthetic: {len(synthetic)} sprints "
          f"(at-risk: {sum(1 for e in synthetic if e['label'] == 1)})")

    # KS realism test
    ks = ks_test_realism(real, synthetic)

    # Build configs
    configs_to_build = (
        ["baseline", "h3", "syn_only"]
        if args.config == "all"
        else [args.config]
    )

    for config_name in configs_to_build:
        if config_name == "baseline":
            train, val, test = build_baseline(real)
        elif config_name == "h3":
            train, val, test = build_h3(
                real, synthetic, syn_ratio=args.syn_ratio,
            )
        elif config_name == "syn_only":
            train, val, test = build_syn_only(real, synthetic)
        else:
            continue

        print_split_report(config_name, train, val, test, ks if config_name == "h3" else None)
        save_split(args.output, config_name, train, val, test)

    print(f"\n✅ Training data prepared in {args.output}/")


if __name__ == "__main__":
    main()
