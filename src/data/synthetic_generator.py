"""
Synthetic sprint generator for zero-history startups (M3).

Generates 5K+ realistic sprints calibrated from real golang/go data.
Supports two persona sets:
  - **Large OSS** (golang/go scale): calibrated from 475 real sprints
  - **Small Startup** (target use case): 3-10 devs, 2-3 repos

Produces 25 metrics per sprint (full schema match with
``SprintPreprocessor`` output) plus risk labels from ``RiskLabeler``.

Usage:
    python src/data/synthetic_generator.py
    python src/data/synthetic_generator.py --count 10000 --personas startup
    python src/data/synthetic_generator.py --calibrate data/golang_go_sprints.json
"""

import json
import random
import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.data.features import RiskLabeler


# ------------------------------------------------------------------ #
#  Persona definition                                                 #
# ------------------------------------------------------------------ #

@dataclass
class SprintPersona:
    """Template defining a realistic development pattern.

    Each range is (min, max) used with ``random.randint`` or
    ``random.uniform`` to produce per-sprint variation.
    """
    name: str
    commit_range: tuple = (1, 50)
    pr_range: tuple = (0, 30)
    issue_range: tuple = (0, 5)
    authors_range: tuple = (1, 5)
    additions_range: tuple = (100, 5000)
    deletion_ratio: tuple = (0.25, 0.45)   # deletions as fraction of additions
    files_per_1k_changes: tuple = (20, 60)  # files changed per 1000 LOC
    weight: float = 1.0


# ------------------------------------------------------------------ #
#  Persona sets                                                       #
# ------------------------------------------------------------------ #

# Calibrated from 475 golang/go real sprints (2-week windows)
LARGE_OSS_PERSONAS = [
    SprintPersona(
        "active_sprint",
        commit_range=(100, 250), pr_range=(80, 200),
        issue_range=(30, 80), authors_range=(25, 55),
        additions_range=(8000, 25000), deletion_ratio=(0.30, 0.50),
        files_per_1k_changes=(25, 45),
        weight=3.0,
    ),
    SprintPersona(
        "release_sprint",
        commit_range=(200, 420), pr_range=(150, 320),
        issue_range=(50, 155), authors_range=(40, 77),
        additions_range=(20000, 80000), deletion_ratio=(0.25, 0.45),
        files_per_1k_changes=(20, 40),
        weight=1.0,
    ),
    SprintPersona(
        "quiet_sprint",
        commit_range=(1, 50), pr_range=(0, 30),
        issue_range=(0, 10), authors_range=(1, 15),
        additions_range=(100, 3000), deletion_ratio=(0.20, 0.60),
        files_per_1k_changes=(30, 70),
        weight=1.0,
    ),
    SprintPersona(
        "blocked_sprint",
        commit_range=(30, 100), pr_range=(20, 80),
        issue_range=(60, 155), authors_range=(15, 40),
        additions_range=(3000, 15000), deletion_ratio=(0.30, 0.50),
        files_per_1k_changes=(25, 50),
        weight=1.0,
    ),
    SprintPersona(
        "refactor_sprint",
        commit_range=(50, 150), pr_range=(30, 80),
        issue_range=(5, 20), authors_range=(10, 30),
        additions_range=(10000, 50000), deletion_ratio=(0.50, 0.80),
        files_per_1k_changes=(15, 35),
        weight=1.0,
    ),
]

# Target use case: small startups (3-10 devs, 2-3 repos)
STARTUP_PERSONAS = [
    SprintPersona(
        "commit_first",
        commit_range=(5, 30), pr_range=(0, 10),
        issue_range=(0, 2), authors_range=(1, 2),
        additions_range=(50, 2000), deletion_ratio=(0.15, 0.40),
        files_per_1k_changes=(30, 80),
        weight=3.0,
    ),
    SprintPersona(
        "healthy_flow",
        commit_range=(10, 40), pr_range=(5, 20),
        issue_range=(3, 15), authors_range=(2, 4),
        additions_range=(500, 5000), deletion_ratio=(0.25, 0.45),
        files_per_1k_changes=(25, 60),
        weight=2.0,
    ),
    SprintPersona(
        "blocked_issues",
        commit_range=(5, 20), pr_range=(2, 10),
        issue_range=(10, 50), authors_range=(1, 3),
        additions_range=(200, 3000), deletion_ratio=(0.20, 0.50),
        files_per_1k_changes=(30, 70),
        weight=1.0,
    ),
    SprintPersona(
        "pr_bottleneck",
        commit_range=(15, 50), pr_range=(20, 60),
        issue_range=(0, 3), authors_range=(3, 6),
        additions_range=(1000, 8000), deletion_ratio=(0.25, 0.45),
        files_per_1k_changes=(20, 50),
        weight=1.0,
    ),
    SprintPersona(
        "quiet_sprint",
        commit_range=(0, 10), pr_range=(0, 3),
        issue_range=(0, 1), authors_range=(0, 2),
        additions_range=(0, 500), deletion_ratio=(0.10, 0.60),
        files_per_1k_changes=(30, 100),
        weight=1.0,
    ),
]

PERSONA_SETS = {
    "large_oss": LARGE_OSS_PERSONAS,
    "startup": STARTUP_PERSONAS,
    "all": LARGE_OSS_PERSONAS + STARTUP_PERSONAS,
}


# ------------------------------------------------------------------ #
#  Generator                                                          #
# ------------------------------------------------------------------ #

class SyntheticSprintGenerator:
    """Generate realistic synthetic sprints for model training.

    Outputs the full 25-metric schema matching ``SprintPreprocessor``
    plus ``RiskLabeler`` risk labels.
    """

    def __init__(
        self,
        personas: str = "all",
        seed: int = 42,
    ):
        random.seed(seed)
        if isinstance(personas, str):
            self.personas = PERSONA_SETS.get(personas, PERSONA_SETS["all"])
        else:
            self.personas = personas

    def generate(
        self,
        count: int = 5000,
        repo_name: str = "synthetic/repo",
    ) -> list[dict]:
        """Generate ``count`` synthetic sprints.

        Returns list of dicts with the same schema as
        ``SprintPreprocessor.create_sprints()`` output.
        """
        sprints: list[dict] = []
        total_weight = sum(p.weight for p in self.personas)

        for i in range(count):
            persona = random.choices(
                self.personas,
                weights=[p.weight / total_weight for p in self.personas],
                k=1,
            )[0]

            metrics = self._generate_metrics(persona)
            risk_label = RiskLabeler.label_sprint(metrics)

            sprints.append({
                "sprint_id": f"synthetic_{i:05d}",
                "repo": repo_name,
                "persona": persona.name,
                "metrics": metrics,
                "risk_label": risk_label,
            })

        return sprints

    # -------------------------------------------------------------- #
    #  Metrics generation (full 25-field schema)                       #
    # -------------------------------------------------------------- #

    def _generate_metrics(self, p: SprintPersona) -> dict:
        """Generate all 25 metrics matching real sprint schema."""

        # --- Core counts ---
        total_commits = random.randint(*p.commit_range)
        total_prs = random.randint(*p.pr_range)
        total_issues = random.randint(*p.issue_range)
        authors = random.randint(*p.authors_range)

        # --- Resolution / merge rates ---
        issue_resolution = random.uniform(0.3, 1.0) if total_issues > 0 else 0
        pr_merge = random.uniform(0.3, 1.0) if total_prs > 0 else 0

        closed_issues = int(total_issues * issue_resolution)
        merged_prs = int(total_prs * pr_merge)

        # --- Code churn ---
        total_additions = random.randint(*p.additions_range)
        del_ratio = random.uniform(*p.deletion_ratio)
        total_deletions = int(total_additions * del_ratio)
        total_code_changes = total_additions + total_deletions

        # Files changed (correlated with code volume)
        files_per_k = random.randint(*p.files_per_1k_changes)
        files_changed = max(1, int(total_code_changes * files_per_k / 1000))

        # PR-level code changes (distribute total across PRs)
        code_changes = total_code_changes  # aggregate PR-level
        avg_pr_size = code_changes // max(1, total_prs)

        # Code concentration: fraction in top 2 PRs
        if total_prs >= 2:
            concentration = random.uniform(0.3, 0.9)
        elif total_prs == 1:
            concentration = 1.0
        else:
            concentration = 0.0

        # --- Risk indicators ---
        open_issues = total_issues - closed_issues
        stalled = min(random.randint(0, max(1, open_issues)), open_issues)
        unreviewed = random.randint(0, max(1, total_prs // 3))
        abandoned = random.randint(0, max(1, (total_prs - merged_prs) // 2))
        long_open = min(
            random.randint(0, max(1, open_issues // 2)),
            open_issues,
        )

        # --- Language breakdown (simplified) ---
        primary_lang = "Go"
        primary_pct = random.uniform(0.70, 0.95)
        lang_breakdown = {
            primary_lang: int(total_code_changes * primary_pct),
        }
        remainder = total_code_changes - lang_breakdown[primary_lang]
        if remainder > 0:
            for lang in ["Other", "C", "Assembly", "Shell"]:
                chunk = random.randint(0, remainder)
                if chunk > 0:
                    lang_breakdown[lang] = chunk
                    remainder -= chunk
                if remainder <= 0:
                    break

        return {
            # Temporal (3)
            "days_span": 13,
            "issue_age_avg": random.uniform(2, 10) if total_issues else 0,
            "pr_age_avg": random.uniform(1, 7) if total_prs else 0,
            # Activity (6)
            "total_issues": total_issues,
            "total_prs": total_prs,
            "total_commits": total_commits,
            "issue_resolution_rate": round(issue_resolution, 4),
            "pr_merge_rate": round(pr_merge, 4),
            "commit_frequency": round(total_commits / 13.0, 4),
            # Code (3)
            "total_code_changes": total_code_changes,
            "avg_pr_size": avg_pr_size,
            "code_concentration": round(concentration, 4),
            # Risk (4)
            "stalled_issues": stalled,
            "unreviewed_prs": unreviewed,
            "abandoned_prs": abandoned,
            "long_open_issues": long_open,
            # Team (2)
            "unique_authors": authors,
            "author_participation": round(
                random.uniform(0.5, 1.0) if authors else 0, 4,
            ),
            # === 7 NEW fields (match real schema) ===
            "closed_issues": closed_issues,
            "merged_prs": merged_prs,
            "code_changes": code_changes,
            "total_additions": total_additions,
            "total_deletions": total_deletions,
            "files_changed": files_changed,
            "language_breakdown": lang_breakdown,
        }

    # -------------------------------------------------------------- #
    #  Calibration helper                                              #
    # -------------------------------------------------------------- #

    @staticmethod
    def calibrate_from_real(
        sprints_path: str,
    ) -> dict:
        """Read real sprint data and print distribution stats.

        Useful for tuning persona ranges.
        """
        with open(sprints_path, encoding="utf-8") as f:
            sprints = json.load(f)

        numeric_keys = [
            "total_commits", "total_prs", "total_issues",
            "unique_authors", "total_additions", "total_deletions",
            "files_changed", "issue_resolution_rate", "pr_merge_rate",
            "commit_frequency", "total_code_changes", "avg_pr_size",
            "code_concentration", "stalled_issues", "unreviewed_prs",
        ]

        stats: dict = {}
        for key in numeric_keys:
            vals = sorted(
                s["metrics"].get(key, 0) for s in sprints
                if isinstance(s["metrics"].get(key, 0), (int, float))
            )
            if not vals:
                continue
            n = len(vals)
            stats[key] = {
                "min": vals[0],
                "p10": vals[int(n * 0.10)],
                "p25": vals[int(n * 0.25)],
                "median": vals[n // 2],
                "p75": vals[int(n * 0.75)],
                "p90": vals[int(n * 0.90)],
                "max": vals[-1],
                "mean": sum(vals) / n,
            }

        at_risk = sum(
            1 for s in sprints
            if s.get("risk_label", {}).get("is_at_risk", False)
        )

        return {
            "total_sprints": len(sprints),
            "at_risk_count": at_risk,
            "at_risk_pct": round(100 * at_risk / len(sprints), 1),
            "distributions": stats,
        }


# ------------------------------------------------------------------ #
#  CLI                                                                 #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic sprint data for training",
    )
    parser.add_argument(
        "--count", type=int, default=5000,
        help="Number of synthetic sprints to generate (default: 5000)",
    )
    parser.add_argument(
        "--personas", default="all",
        choices=["large_oss", "startup", "all"],
        help="Persona set to use (default: all)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output path (default: data/synthetic_sprints.json)",
    )
    parser.add_argument(
        "--calibrate", default=None,
        help="Path to real sprints JSON for calibration stats",
    )
    args = parser.parse_args()

    # Calibration mode
    if args.calibrate:
        print(f"Calibrating from {args.calibrate} ...\n")
        stats = SyntheticSprintGenerator.calibrate_from_real(args.calibrate)
        print(f"Total real sprints: {stats['total_sprints']}")
        print(f"At-risk: {stats['at_risk_count']} "
              f"({stats['at_risk_pct']}%)\n")
        print(f"{'Metric':<30s}  {'min':>8s}  {'p25':>8s}  "
              f"{'median':>8s}  {'p75':>8s}  {'p90':>8s}  {'max':>8s}")
        print("-" * 96)
        for key, d in stats["distributions"].items():
            print(f"{key:<30s}  {d['min']:8.1f}  {d['p25']:8.1f}  "
                  f"{d['median']:8.1f}  {d['p75']:8.1f}  "
                  f"{d['p90']:8.1f}  {d['max']:8.1f}")
        return

    # Generation mode
    gen = SyntheticSprintGenerator(personas=args.personas, seed=args.seed)
    sprints = gen.generate(count=args.count)

    output_path = args.output or os.path.join(
        os.path.dirname(__file__), "..", "..", "data",
        "synthetic_sprints.json",
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sprints, f, indent=2)

    # Summary
    at_risk = sum(
        1 for s in sprints if s["risk_label"]["is_at_risk"]
    )
    persona_dist: dict[str, int] = {}
    for s in sprints:
        p = s.get("persona", "unknown")
        persona_dist[p] = persona_dist.get(p, 0) + 1

    print(f"Generated {len(sprints)} synthetic sprints")
    print(f"  Personas: {args.personas}")
    print(f"  At-risk: {at_risk} ({100 * at_risk / len(sprints):.1f}%)")
    print(f"  Persona distribution:")
    for name, cnt in sorted(persona_dist.items(), key=lambda x: -x[1]):
        print(f"    {name}: {cnt} ({100 * cnt / len(sprints):.1f}%)")
    print(f"  Saved to {output_path}")


if __name__ == "__main__":
    main()
