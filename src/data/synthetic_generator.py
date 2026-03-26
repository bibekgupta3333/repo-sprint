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

import glob
import json
import math
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
#  Distribution helpers for calibration                                #
# ------------------------------------------------------------------ #

def _beta_params_from_data(vals: list) -> tuple[float, float]:
    """Estimate beta distribution alpha, beta from sample moments.

    Clamps inputs to (0.01, 0.99) to avoid degenerate cases.
    Returns (alpha, beta) tuple for ``random.betavariate``.
    """
    clamped = [max(0.01, min(0.99, v)) for v in vals if v > 0]
    if len(clamped) < 3:
        return (2.0, 2.0)
    mean = sum(clamped) / len(clamped)
    var = sum((v - mean) ** 2 for v in clamped) / len(clamped)
    var = max(var, 1e-6)
    common = mean * (1 - mean) / var - 1
    common = max(common, 0.5)
    alpha = max(0.5, mean * common)
    beta = max(0.5, (1 - mean) * common)
    return (alpha, beta)


def _lognormal_params(
    vals: list,
    robust: bool = False,
) -> tuple[float, float]:
    """Compute mu, sigma for lognormal from positive values.

    Args:
        vals: Positive numeric values.
        robust: If True, estimate sigma from IQR
            (resistant to outliers).
    """
    positive = [v for v in vals if v > 0]
    if len(positive) < 3:
        return (0.0, 1.0)
    logs = [math.log(v) for v in positive]
    mu = sum(logs) / len(logs)
    if robust and len(positive) >= 10:
        positive_sorted = sorted(positive)
        n = len(positive_sorted)
        p25 = positive_sorted[int(n * 0.25)]
        p75 = positive_sorted[int(n * 0.75)]
        if p75 > p25 > 0:
            sigma = math.log(p75 / p25) / (2 * 0.6745)
            sigma = max(sigma, 0.1)
            mu = math.log(
                positive_sorted[n // 2])
            return (mu, sigma)
    var = sum((l - mu) ** 2 for l in logs) / len(logs)
    sigma = max(math.sqrt(var), 0.1)
    return (mu, sigma)


def _build_distribution_profile(
    *,
    merge_rates: list,
    res_rates: list,
    stalled_vals: list,
    unreviewed_vals: list,
    issues_vals: list,
    adds_vals: list,
    files_vals: list,
    avg_pr_vals: list,
    dels: list,
    all_sprints: list | None = None,
) -> dict:
    """Build calibrated distribution profiles from real data.

    For rate metrics: beta distribution + spike at boundary.
    For derived metrics: ``"use_open_issues"`` flag.
    For ratio-based metrics: lognormal on the ratio itself.
    For zero-inflated counts: zero fraction + lognormal.
    For right-skewed counts: lognormal.
    """
    def _zero_frac(vals):
        if not vals:
            return 0.5
        return sum(1 for v in vals if v == 0) / len(vals)

    profile: dict = {}

    # --- Rate metrics → beta + spike ---
    if merge_rates:
        a, b = _beta_params_from_data(merge_rates)
        profile["pr_merge_rate"] = {
            "type": "beta", "alpha": a, "beta": b,
            "clip_min": max(0.0, min(merge_rates) - 0.05),
        }

    if res_rates:
        at_one_frac = sum(
            1 for v in res_rates if v >= 0.999
        ) / len(res_rates)
        sub_one = [v for v in res_rates if v < 0.999]
        if sub_one:
            a, b = _beta_params_from_data(sub_one)
        else:
            a, b = (5.0, 1.0)
        profile["issue_resolution_rate"] = {
            "type": "beta_spike",
            "alpha": a, "beta": b,
            "spike_val": 1.0,
            "spike_frac": at_one_frac,
            "clip_min": max(0.0, min(res_rates) - 0.02),
        }

    # --- Derived: stalled = open_issues ---
    profile["stalled_issues"] = {"type": "use_open_issues"}

    # --- Zero-inflated: unreviewed_prs ---
    if unreviewed_vals:
        zf = _zero_frac(unreviewed_vals)
        nonzero = [v for v in unreviewed_vals if v > 0]
        mu, sigma = _lognormal_params(
            nonzero if nonzero else [1])
        profile["unreviewed_prs"] = {
            "type": "zero_inflated",
            "zero_frac": zf,
            "mu": mu, "sigma": sigma,
            "max_val": max(unreviewed_vals),
        }

    # --- Calibrated counts: issues/additions/deletions ---
    if issues_vals:
        positive_issues = [v for v in issues_vals if v > 0]
        if positive_issues:
            mu, sigma = _lognormal_params(
                positive_issues, robust=True)
            profile["total_issues"] = {
                "type": "lognormal",
                "mu": mu, "sigma": sigma,
                "max_val": max(issues_vals),
            }

    if adds_vals:
        positive_adds = [v for v in adds_vals if v > 0]
        if positive_adds:
            mu, sigma = _lognormal_params(
                positive_adds, robust=True)
            profile["total_additions"] = {
                "type": "lognormal",
                "mu": mu, "sigma": sigma,
                "max_val": max(adds_vals),
            }

    if dels:
        positive_dels = [v for v in dels if v > 0]
        if positive_dels:
            mu, sigma = _lognormal_params(
                positive_dels, robust=True)
            profile["total_deletions"] = {
                "type": "lognormal",
                "mu": mu, "sigma": sigma,
                "max_val": max(dels),
            }

    # --- Ratio-based: total_deletions = additions * ratio ---
    if all_sprints:
        del_ratios = []
        files_per_k = []
        for s in all_sprints:
            m = s["metrics"]
            adds = m.get("total_additions", 0)
            d = m.get("total_deletions", 0)
            fc = m.get("files_changed", 0)
            cc = m.get("total_code_changes", 0)
            if adds > 0:
                del_ratios.append(d / adds)
            if cc > 0 and fc > 0:
                files_per_k.append(fc / (cc / 1000))

        if del_ratios:
            mu, sigma = _lognormal_params(
                [max(0.01, r) for r in del_ratios])
            profile["deletion_ratio"] = {
                "type": "lognormal_ratio",
                "mu": mu, "sigma": sigma,
                "max_val": max(del_ratios),
            }

        if files_vals:
            positive_fc = [v for v in files_vals if v > 0]
            if positive_fc:
                mu, sigma = _lognormal_params(
                    positive_fc, robust=True)
                profile["files_changed"] = {
                    "type": "lognormal",
                    "mu": mu, "sigma": sigma,
                    "max_val": max(files_vals),
                }

    # --- Right-skewed: avg_pr_size ---
    if avg_pr_vals:
        positive = [v for v in avg_pr_vals if v > 0]
        mu, sigma = _lognormal_params(
            positive if positive else [1])
        profile["avg_pr_size"] = {
            "type": "lognormal",
            "mu": mu, "sigma": sigma,
            "max_val": max(avg_pr_vals),
        }

    return profile


def _sample_metric(profile: dict) -> float:
    """Draw a single sample from a calibrated distribution profile.

    Supports: beta, beta_spike, zero_inflated, lognormal,
    lognormal_ratio.
    """
    dist_type = profile["type"]

    if dist_type == "beta":
        val = random.betavariate(
            profile["alpha"], profile["beta"])
        clip_min = profile.get("clip_min", 0.0)
        return max(clip_min, val)

    if dist_type == "beta_spike":
        if random.random() < profile.get("spike_frac", 0):
            return profile.get("spike_val", 1.0)
        val = random.betavariate(
            profile["alpha"], profile["beta"])
        clip_min = profile.get("clip_min", 0.0)
        return max(clip_min, min(val, 0.999))

    if dist_type == "zero_inflated":
        if random.random() < profile["zero_frac"]:
            return 0.0
        raw = math.exp(random.gauss(
            profile["mu"], profile["sigma"]))
        return min(round(raw), profile.get("max_val", 100))

    if dist_type == "lognormal":
        raw = math.exp(random.gauss(
            profile["mu"], profile["sigma"]))
        return min(round(raw), profile.get("max_val", 10000))

    if dist_type == "lognormal_ratio":
        raw = math.exp(random.gauss(
            profile["mu"], profile["sigma"]))
        return min(raw, profile.get("max_val", 10.0))

    return 0.0


# ------------------------------------------------------------------ #
#  Auto-calibration from real data                                     #
# ------------------------------------------------------------------ #

def _pct(vals: list, p: float) -> float:
    """Percentile helper."""
    if not vals:
        return 0
    idx = int(len(vals) * p)
    idx = min(idx, len(vals) - 1)
    return vals[idx]


def auto_calibrate_personas(
    data_dir: str = "data",
) -> list:
    """Build personas from real sprint distributions.

    Reads every ``*_sprints.json`` (excluding synthetic)
    in ``data_dir`` and creates 5 personas whose ranges
    match the actual percentile bands:

    - **active**: p25–p75 (typical sprints)
    - **quiet**: p0–p25 (low-activity)
    - **risky**: p75–p100 (high-load)
    - **refactor**: high deletions/additions ratio
    - **healthy**: high resolution + merge rates
    """
    resolved = os.path.join(
        os.path.dirname(__file__),
        "..", "..", data_dir,
    )
    files = sorted(glob.glob(
        os.path.join(resolved, "*_sprints.json")))
    files = [
        f for f in files
        if "synthetic" not in os.path.basename(f)
    ]
    if not files:
        print("  No real sprint files found — "
              "falling back to 'all' personas")
        return PERSONA_SETS["all"]

    # Load all sprints
    all_sprints: list = []
    for fpath in files:
        with open(fpath, encoding="utf-8") as f:
            all_sprints.extend(json.load(f))

    if len(all_sprints) < 5:
        print(f"  Only {len(all_sprints)} sprints — "
              "falling back to 'all' personas")
        return PERSONA_SETS["all"]

    # Extract sorted metric arrays
    def _vals(key):
        return sorted(
            s["metrics"].get(key, 0)
            for s in all_sprints
            if isinstance(
                s["metrics"].get(key, 0),
                (int, float),
            )
        )

    commits = _vals("total_commits")
    prs = _vals("total_prs")
    issues = _vals("total_issues")
    authors = _vals("unique_authors")
    adds = _vals("total_additions")
    dels = _vals("total_deletions")

    # Compute deletion ratio range from real data
    del_ratios = []
    for s in all_sprints:
        m = s["metrics"]
        a = m.get("total_additions", 0)
        d = m.get("total_deletions", 0)
        if a > 0:
            del_ratios.append(d / a)
    del_ratios.sort()
    dr_lo = max(0.05, _pct(del_ratios, 0.1))
    dr_hi = min(1.5, _pct(del_ratios, 0.9))

    print(f"  Auto-calibrated from "
          f"{len(all_sprints)} real sprints "
          f"({len(files)} file(s))")

    def _rng(vals, lo_p, hi_p):
        """Integer range from percentile band."""
        lo = max(0, int(_pct(vals, lo_p)))
        hi = max(lo + 1, int(_pct(vals, hi_p)))
        return (lo, hi)

    def _frange(vals, lo_p, hi_p):
        """Float range from percentile band."""
        lo = max(0.0, float(_pct(vals, lo_p)))
        hi = float(_pct(vals, hi_p))
        if hi <= lo:
            return (lo, lo)  # constant distribution
        return (lo, hi)

    # --- Extract rate distributions ---
    res_rates = []
    merge_rates = []
    stalled_vals = []
    unreviewed_vals = []
    files_vals = _vals("files_changed")
    avg_pr_vals = _vals("avg_pr_size")

    for s in all_sprints:
        m = s["metrics"]
        rr = m.get("issue_resolution_rate", 0)
        if isinstance(rr, (int, float)):
            res_rates.append(rr)
        mr = m.get("pr_merge_rate", 0)
        if isinstance(mr, (int, float)):
            merge_rates.append(mr)
        sv = m.get("stalled_issues", 0)
        if isinstance(sv, (int, float)):
            stalled_vals.append(sv)
        uv = m.get("unreviewed_prs", 0)
        if isinstance(uv, (int, float)):
            unreviewed_vals.append(uv)

    res_rates.sort()
    merge_rates.sort()
    stalled_vals.sort()
    unreviewed_vals.sort()
    files_vals.sort()
    avg_pr_vals.sort()

    rates_profile = _build_distribution_profile(
        merge_rates=merge_rates,
        res_rates=res_rates,
        stalled_vals=stalled_vals,
        unreviewed_vals=unreviewed_vals,
        issues_vals=issues,
        adds_vals=adds,
        files_vals=files_vals,
        avg_pr_vals=avg_pr_vals,
        dels=dels,
        all_sprints=all_sprints,
    )

    # Empirical pools for hard-to-fit metrics (multi-repo mixtures).
    rates_profile["empirical"] = {
        "total_issues": issues,
        "unique_authors": authors,
        "issue_resolution_rate": res_rates,
        "pr_merge_rate": merge_rates,
        "stalled_issues": stalled_vals,
        "total_additions": adds,
        "total_deletions": dels,
        "files_changed": files_vals,
        "total_code_changes": _vals("total_code_changes"),
        "avg_pr_size": avg_pr_vals,
        "code_concentration": _vals("code_concentration"),
    }

    personas = [
        SprintPersona(
            "active_sprint",
            commit_range=_rng(commits, 0.25, 0.75),
            pr_range=_rng(prs, 0.25, 0.75),
            issue_range=_rng(issues, 0.25, 0.75),
            authors_range=_rng(authors, 0.25, 0.75),
            additions_range=_rng(adds, 0.25, 0.75),
            deletion_ratio=(dr_lo, dr_hi),
            files_per_1k_changes=(20, 60),
            weight=3.0,
        ),
        SprintPersona(
            "quiet_sprint",
            commit_range=_rng(commits, 0.0, 0.25),
            pr_range=_rng(prs, 0.0, 0.25),
            issue_range=_rng(issues, 0.0, 0.30),
            authors_range=_rng(authors, 0.0, 0.25),
            additions_range=_rng(adds, 0.0, 0.30),
            deletion_ratio=(dr_lo, dr_hi),
            files_per_1k_changes=(30, 80),
            weight=1.5,
        ),
        SprintPersona(
            "high_load_sprint",
            commit_range=_rng(commits, 0.75, 1.0),
            pr_range=_rng(prs, 0.75, 1.0),
            issue_range=_rng(issues, 0.70, 1.0),
            authors_range=_rng(authors, 0.75, 1.0),
            additions_range=_rng(adds, 0.75, 1.0),
            deletion_ratio=(dr_lo, dr_hi),
            files_per_1k_changes=(15, 45),
            weight=1.5,
        ),
        SprintPersona(
            "refactor_sprint",
            commit_range=_rng(commits, 0.25, 0.75),
            pr_range=_rng(prs, 0.10, 0.50),
            issue_range=_rng(issues, 0.15, 0.50),
            authors_range=_rng(authors, 0.10, 0.50),
            additions_range=_rng(adds, 0.50, 0.90),
            deletion_ratio=(
                min(dr_hi, 0.50),
                min(dr_hi * 1.5, 1.2),
            ),
            files_per_1k_changes=(15, 40),
            weight=1.0,
        ),
        SprintPersona(
            "healthy_sprint",
            commit_range=_rng(commits, 0.30, 0.70),
            pr_range=_rng(prs, 0.30, 0.70),
            issue_range=_rng(issues, 0.20, 0.65),
            authors_range=_rng(authors, 0.30, 0.70),
            additions_range=_rng(adds, 0.15, 0.55),
            deletion_ratio=(dr_lo, dr_hi),
            files_per_1k_changes=(20, 55),
            weight=2.0,
        ),
    ]
    return personas, rates_profile


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
        self.rates_profile: dict = {}  # calibrated rates
        if personas == "auto":
            result = auto_calibrate_personas()
            self.personas = result[0]
            self.rates_profile = result[1]
        elif isinstance(personas, str):
            self.personas = PERSONA_SETS.get(
                personas, PERSONA_SETS["all"])
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
        rp = self.rates_profile  # calibrated distributions
        empirical = rp.get("empirical", {})

        def _emp(key, cast=float):
            vals = empirical.get(key, [])
            if vals:
                return cast(random.choice(vals))
            return None

        # --- Core counts ---
        total_commits = random.randint(*p.commit_range)
        total_prs = random.randint(*p.pr_range)
        emp_issues = _emp("total_issues", int)
        if emp_issues is not None:
            total_issues = max(0, emp_issues)
        elif rp and "total_issues" in rp:
            total_issues = int(_sample_metric(rp["total_issues"]))
        else:
            total_issues = random.randint(*p.issue_range)

        emp_authors = _emp("unique_authors", int)
        if emp_authors is not None:
            authors = max(0, emp_authors)
        else:
            authors = random.randint(*p.authors_range)

        # --- Resolution / merge rates (distribution-aware) ---
        emp_issue_res = _emp("issue_resolution_rate", float)
        if emp_issue_res is not None and total_issues > 0:
            issue_resolution = max(0.0, min(1.0, emp_issue_res))
        elif rp and "issue_resolution_rate" in rp and total_issues > 0:
            issue_resolution = _sample_metric(
                rp["issue_resolution_rate"])
        elif total_issues > 0:
            issue_resolution = random.uniform(0.3, 1.0)
        else:
            issue_resolution = 0

        emp_pr_merge = _emp("pr_merge_rate", float)
        if emp_pr_merge is not None and total_prs > 0:
            pr_merge = max(0.0, min(1.0, emp_pr_merge))
        elif rp and "pr_merge_rate" in rp and total_prs > 0:
            pr_merge = _sample_metric(
                rp["pr_merge_rate"])
        elif total_prs > 0:
            pr_merge = random.uniform(0.3, 1.0)
        else:
            pr_merge = 0

        closed_issues = int(total_issues * issue_resolution)
        merged_prs = int(total_prs * pr_merge)

        # --- Commit-diff level churn (calibrated independently) ---
        emp_adds = _emp("total_additions", int)
        if emp_adds is not None:
            total_additions = max(0, emp_adds)
        elif rp and "total_additions" in rp:
            total_additions = max(
                0, int(_sample_metric(rp["total_additions"])))
        else:
            total_additions = random.randint(*p.additions_range)

        emp_dels = _emp("total_deletions", int)
        if emp_dels is not None:
            total_deletions = max(0, emp_dels)
        elif rp and "total_deletions" in rp:
            total_deletions = max(
                0, int(_sample_metric(rp["total_deletions"])))
        elif rp and "deletion_ratio" in rp:
            dr = _sample_metric(rp["deletion_ratio"])
            total_deletions = max(1, int(total_additions * dr))
        else:
            del_ratio = random.uniform(*p.deletion_ratio)
            total_deletions = int(total_additions * del_ratio)

        # --- PR-first generation to mirror real feature extraction ---
        emp_total_changes = _emp("total_code_changes", int)
        if emp_total_changes is not None:
            target_code_changes = max(0, emp_total_changes)
        else:
            target_code_changes = max(0, total_additions + total_deletions)

        if total_prs > 0:
            emp_avg_pr = _emp("avg_pr_size", int)
            if emp_avg_pr is not None:
                base_size = max(1, emp_avg_pr)
            elif rp and "avg_pr_size" in rp:
                base_size = max(
                    1, int(_sample_metric(rp["avg_pr_size"])))
            else:
                base_size = max(1, target_code_changes // total_prs)

            pr_changes = [
                max(1, int(random.lognormvariate(
                    math.log(max(1, base_size)), 0.55)))
                for _ in range(total_prs)
            ]
            raw_sum = sum(pr_changes)
            if raw_sum > 0 and target_code_changes > 0:
                scale = target_code_changes / raw_sum
                pr_changes = [
                    max(1, int(v * scale))
                    for v in pr_changes
                ]

            total_code_changes = sum(pr_changes)
            avg_pr_size = int(total_code_changes / total_prs)

            emp_concentration = _emp("code_concentration", float)
            if emp_concentration is not None:
                concentration = max(0.0, min(1.0, emp_concentration))
            else:
                pr_changes_sorted = sorted(
                    pr_changes, reverse=True)
                top_2 = pr_changes_sorted[0] + (
                    pr_changes_sorted[1]
                    if len(pr_changes_sorted) > 1 else 0
                )
                concentration = (
                    top_2 / total_code_changes
                    if total_code_changes > 0 else 0
                )
        else:
            total_code_changes = 0
            avg_pr_size = 0
            concentration = 0.0

        # Files changed stays a commit-diff metric
        emp_files = _emp("files_changed", int)
        if emp_files is not None:
            files_changed = max(0, emp_files)
        elif rp and "files_changed" in rp:
            files_changed = max(
                0, int(_sample_metric(rp["files_changed"])))
        else:
            files_per_k = random.randint(
                *p.files_per_1k_changes)
            commit_diff_changes = (
                total_additions + total_deletions)
            files_changed = int(
                commit_diff_changes * files_per_k / 1000
            ) if commit_diff_changes > 0 else 0

        code_changes = total_code_changes

        # --- Risk indicators (distribution-aware) ---
        open_issues = total_issues - closed_issues

        emp_stalled = _emp("stalled_issues", int)
        if emp_stalled is not None:
            stalled = min(max(0, emp_stalled), open_issues)
        elif (rp and rp.get("stalled_issues", {}).get("type")
                == "use_open_issues"):
            stalled = open_issues
        elif rp and "stalled_issues" in rp:
            stalled = min(
                int(_sample_metric(rp["stalled_issues"])),
                open_issues,
            )
        else:
            stalled = min(
                random.randint(0, max(1, open_issues)),
                open_issues,
            )

        if rp and "unreviewed_prs" in rp:
            unreviewed = min(
                int(_sample_metric(rp["unreviewed_prs"])),
                total_prs,
            )
        else:
            unreviewed = random.randint(
                0, max(1, total_prs // 3))

        abandoned = random.randint(
            0, max(1, (total_prs - merged_prs) // 2))
        long_open = stalled

        # --- Language breakdown ---
        primary_lang = "JavaScript"
        primary_pct = random.uniform(0.50, 0.85)
        lang_breakdown = {
            primary_lang: int(
                total_code_changes * primary_pct),
        }
        remainder = (total_code_changes
                     - lang_breakdown[primary_lang])
        if remainder > 0:
            for lang in ["Other", "TypeScript",
                         "Python", "JSON", "CSS"]:
                chunk = random.randint(0, remainder)
                if chunk > 0:
                    lang_breakdown[lang] = chunk
                    remainder -= chunk
                if remainder <= 0:
                    break

        return {
            # Temporal (3)
            "days_span": 13,
            "issue_age_avg": (
                random.uniform(2, 10)
                if total_issues else 0),
            "pr_age_avg": (
                random.uniform(1, 7)
                if total_prs else 0),
            # Activity (6)
            "total_issues": total_issues,
            "total_prs": total_prs,
            "total_commits": total_commits,
            "issue_resolution_rate": round(
                issue_resolution, 4),
            "pr_merge_rate": round(pr_merge, 4),
            "commit_frequency": round(
                total_commits / 13.0, 4),
            # Code (3)
            "total_code_changes": total_code_changes,
            "avg_pr_size": avg_pr_size,
            "code_concentration": round(
                concentration, 4),
            # Risk (4)
            "stalled_issues": stalled,
            "unreviewed_prs": unreviewed,
            "abandoned_prs": abandoned,
            "long_open_issues": long_open,
            # Team (2)
            "unique_authors": authors,
            "author_participation": round(
                random.uniform(0.5, 1.0)
                if authors else 0, 4),
            # === Extended fields ===
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
    def _compute_stats(sprints: list) -> dict:
        """Compute distribution stats for a list of sprints."""
        numeric_keys = [
            "total_commits", "total_prs", "total_issues",
            "unique_authors", "total_additions",
            "total_deletions", "files_changed",
            "issue_resolution_rate", "pr_merge_rate",
            "commit_frequency", "total_code_changes",
            "avg_pr_size", "code_concentration",
            "stalled_issues", "unreviewed_prs",
        ]

        stats: dict = {}
        for key in numeric_keys:
            vals = sorted(
                s["metrics"].get(key, 0) for s in sprints
                if isinstance(
                    s["metrics"].get(key, 0), (int, float)
                )
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
            if s.get("risk_label", {}).get(
                "is_at_risk", False)
        )

        return {
            "total_sprints": len(sprints),
            "at_risk_count": at_risk,
            "at_risk_pct": round(
                100 * at_risk / max(len(sprints), 1), 1),
            "distributions": stats,
        }

    @staticmethod
    def calibrate_from_real(
        sprints_path: str,
    ) -> dict:
        """Read real sprint data and return stats.

        ``sprints_path`` can be:
          - A path to a single ``*_sprints.json``
          - ``"all"`` to auto-discover every
            ``*_sprints.json`` in ``data/``
        """
        data_dir = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "data",
        )

        if sprints_path == "all":
            pattern = os.path.join(
                data_dir, "*_sprints.json")
            files = sorted(glob.glob(pattern))
            if not files:
                print("No *_sprints.json found "
                      f"in {data_dir}")
                return {"total_sprints": 0}
        else:
            files = [sprints_path]

        all_sprints: list = []
        per_repo: dict = {}  # filename → sprints

        for fpath in files:
            with open(fpath, encoding="utf-8") as f:
                sprints = json.load(f)
            basename = os.path.basename(fpath)
            per_repo[basename] = sprints
            all_sprints.extend(sprints)

        # Return combined + per-repo stats
        result = SyntheticSprintGenerator._compute_stats(
            all_sprints)
        result["files"] = list(per_repo.keys())
        result["per_repo"] = {
            name: SyntheticSprintGenerator._compute_stats(
                spr)
            for name, spr in per_repo.items()
        }
        return result


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
        "--personas", default="auto",
        choices=["large_oss", "startup", "all", "auto"],
        help=(
            "Persona set to use. 'auto' calibrates "
            "from real sprint data (default: auto)"
        ),
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
        help=(
            "Path to real sprints JSON, or 'all' to "
            "auto-discover every *_sprints.json in data/"
        ),
    )
    args = parser.parse_args()

    # Calibration mode
    if args.calibrate:
        stats = SyntheticSprintGenerator.calibrate_from_real(
            args.calibrate)
        if stats.get("total_sprints", 0) == 0:
            return

        def _print_table(label, st):
            print(f"\n{'=' * 96}")
            print(f"  {label}")
            print(f"{'=' * 96}")
            print(f"  Sprints: {st['total_sprints']}  |  "
                  f"At-risk: {st['at_risk_count']} "
                  f"({st['at_risk_pct']}%)")
            print(f"\n  {'Metric':<28s}"
                  f"{'min':>8s}  {'p25':>8s}  "
                  f"{'med':>8s}  {'p75':>8s}  "
                  f"{'p90':>8s}  {'max':>8s}")
            print(f"  {'-' * 84}")
            for key, d in st["distributions"].items():
                print(
                    f"  {key:<28s}"
                    f"{d['min']:8.1f}  "
                    f"{d['p25']:8.1f}  "
                    f"{d['median']:8.1f}  "
                    f"{d['p75']:8.1f}  "
                    f"{d['p90']:8.1f}  "
                    f"{d['max']:8.1f}")

        # Per-repo tables
        per_repo = stats.get("per_repo", {})
        for name, repo_stats in per_repo.items():
            _print_table(name, repo_stats)

        # Combined table (only if multiple repos)
        if len(per_repo) > 1:
            _print_table(
                "COMBINED (all repos)", stats)

        # Single repo — just show combined
        if len(per_repo) <= 1:
            pass  # already printed above

        print()
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
