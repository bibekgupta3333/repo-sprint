"""
Prepare training dataset from real + synthetic sprints.
Creates train/val/test splits for baseline model training (M5).
"""

import json
from pathlib import Path
import random


def prepare_training_data():
    """Combine real + synthetic sprints, create train/val/test splits."""

    all_examples = []

    # Load synthetic sprints
    print("Loading synthetic sprints...")
    with open("data/synthetic_sprints.json") as f:
        synthetic = json.load(f)

    for s in synthetic:
        all_examples.append({
            "sprint_id": s["sprint_id"],
            "repo": s["repo"],
            "features": s["metrics"],
            "label": 1 if s["risk_label"]["is_at_risk"] else 0,
            "risk_score": s["risk_label"]["risk_score"],
            "source": "synthetic",
        })

    print(f"  ✓ {len(synthetic)} synthetic sprints")

    # Load real sprints from documents
    print("Loading real sprints...")
    real_count = 0
    for filepath in Path("data").glob("*_documents.json"):
        if "synthetic" not in str(filepath):
            with open(filepath) as f:
                docs = json.load(f)

            for doc in docs:
                if doc["metadata"].get("type") == "sprint_summary":
                    m = doc["metadata"]
                    # Extract metrics (exclude sprints-specific fields)
                    features = {
                        k: v for k, v in m.items()
                        if k not in ["sprint_id", "repo", "type", "date", "risk_score", "is_at_risk"]
                    }

                    all_examples.append({
                        "sprint_id": m.get("sprint_id"),
                        "repo": m.get("repo"),
                        "features": features,
                        "label": 1 if m.get("is_at_risk", False) else 0,
                        "risk_score": m.get("risk_score", 0),
                        "source": "real",
                    })
                    real_count += 1

    print(f"  ✓ {real_count} real sprints")

    # Shuffle
    random.shuffle(all_examples)

    # Split: 80% train, 10% val, 10% test
    train_size = int(0.8 * len(all_examples))
    val_size = int(0.1 * len(all_examples))

    train = all_examples[:train_size]
    val = all_examples[train_size:train_size + val_size]
    test = all_examples[train_size + val_size:]

    # Save splits
    datasets = [
        ("data/training_data.json", all_examples),
        ("data/train_data.json", train),
        ("data/val_data.json", val),
        ("data/test_data.json", test),
    ]

    for filepath, data in datasets:
        with open(filepath, "w") as f:
            json.dump(data, f)

    # Report
    print(f"\n✓ Dataset prepared:")
    print(f"  Total:      {len(all_examples):,}")
    print(f"  Training:   {len(train):,} ({100*len(train)/len(all_examples):.0f}%)")
    print(f"  Validation: {len(val):,} ({100*len(val)/len(all_examples):.0f}%)")
    print(f"  Test:       {len(test):,} ({100*len(test)/len(all_examples):.0f}%)")

    # Class distribution
    at_risk = sum(1 for ex in all_examples if ex["label"] == 1)
    not_at_risk = len(all_examples) - at_risk
    print(f"\n✓ Class distribution:")
    print(f"  Not at risk: {not_at_risk:,} ({100*not_at_risk/len(all_examples):.0f}%)")
    print(f"  At risk:     {at_risk:,} ({100*at_risk/len(all_examples):.0f}%)")


if __name__ == "__main__":
    prepare_training_data()
