"""
Synthetic sprint generator for zero-history startups (M3).
Generates 5K+ realistic sprints based on small team patterns.
"""

import json
import random
from dataclasses import dataclass
from features import SprintMetrics, RiskLabeler


@dataclass
class SprintPersona:
    """Template defining realistic development pattern."""
    name: str
    issue_range: tuple = (0, 5)
    pr_range: tuple = (0, 30)
    commit_range: tuple = (1, 50)
    authors_range: tuple = (1, 5)
    weight: float = 1.0  # Sampling probability


class SyntheticSprintGenerator:
    """Generate realistic synthetic sprints for model training."""

    # Development patterns for small startups
    PERSONAS = [
        SprintPersona("commit_first", (0, 2), (0, 10), (5, 30), (1, 2), weight=3.0),
        SprintPersona("healthy_flow", (3, 15), (5, 20), (10, 40), (2, 4), weight=2.0),
        SprintPersona("blocked_issues", (10, 50), (2, 10), (5, 20), (1, 3), weight=1.0),
        SprintPersona("pr_bottleneck", (0, 3), (20, 60), (15, 50), (3, 6), weight=1.0),
        SprintPersona("quiet_sprint", (0, 1), (0, 3), (0, 10), (0, 2), weight=1.0),
    ]

    def __init__(self, seed: int = 42):
        random.seed(seed)

    def generate(self, count: int = 5000) -> list:
        """Generate count synthetic sprints."""
        sprints = []
        total_weight = sum(p.weight for p in self.PERSONAS)

        for i in range(count):
            # Sample persona based on weights
            persona = random.choices(
                self.PERSONAS,
                weights=[p.weight / total_weight for p in self.PERSONAS],
                k=1
            )[0]

            # Generate metrics from persona
            metrics = self._generate_metrics(persona)

            # Compute risk label
            risk_label = RiskLabeler.label_sprint(metrics)

            sprints.append({
                "sprint_id": f"synthetic_{i:05d}",
                "repo": "synthetic/repo",
                "metrics": metrics,
                "risk_label": risk_label,
            })

        return sprints

    def _generate_metrics(self, persona: SprintPersona) -> SprintMetrics:
        """Generate metrics matching persona pattern."""
        total_issues = random.randint(*persona.issue_range)
        total_prs = random.randint(*persona.pr_range)
        total_commits = random.randint(*persona.commit_range)

        # Resolution rates (0-1)
        issue_resolution = random.uniform(0.3, 1.0) if total_issues > 0 else 0
        pr_merge = random.uniform(0.3, 1.0) if total_prs > 0 else 0

        # Code changes
        total_changes = random.randint(100, 5000)
        avg_pr_size = total_changes // max(1, total_prs)

        # Risk indicators
        stalled = random.randint(0, max(1, total_issues // 2))
        unreviewed = random.randint(0, max(1, total_prs // 3))

        # Team
        authors = random.randint(*persona.authors_range)

        return {
            "days_span": 13,
            "issue_age_avg": random.uniform(2, 10) if total_issues else 0,
            "pr_age_avg": random.uniform(1, 7) if total_prs else 0,
            "total_issues": total_issues,
            "total_prs": total_prs,
            "total_commits": total_commits,
            "issue_resolution_rate": issue_resolution,
            "pr_merge_rate": pr_merge,
            "commit_frequency": total_commits / 13.0,
            "total_code_changes": total_changes,
            "avg_pr_size": avg_pr_size,
            "code_concentration": random.uniform(0.3, 1.0),
            "stalled_issues": stalled,
            "unreviewed_prs": unreviewed,
            "abandoned_prs": random.randint(0, max(1, total_prs // 5)),
            "long_open_issues": random.randint(0, max(1, total_issues // 3)),
            "unique_authors": authors,
            "author_participation": random.uniform(0.5, 1.0) if authors else 0,
        }


if __name__ == "__main__":
    import os
    gen = SyntheticSprintGenerator(seed=42)
    sprints = gen.generate(5000)

    # Save to data/ directory (two levels up from src/data/)
    output_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "synthetic_sprints.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(sprints, f)

    # Stats
    at_risk = sum(1 for s in sprints if s["risk_label"]["is_at_risk"])
    print(f"Generated {len(sprints)} synthetic sprints")
    print(f"At-risk sprints: {at_risk} ({100*at_risk/len(sprints):.1f}%)")
    print(f"Saved to {output_path}")
