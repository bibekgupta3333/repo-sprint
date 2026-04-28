#!/usr/bin/env python
# coding: utf-8

# # Final ML Project — Experiment Runner
# 
# Reproduces the results table used in `docs/PRESENTATION_SLIDES.md`.
# 
# **Task:** binary classification of `is_at_risk` per 14-day sprint.
# **Data:** real sprints from 3 ingested repos + 200 persona-calibrated synthetic sprints.
# **Models:** rule-based oracle · XGBoost (real-only) · XGBoost (real+synth, "H3") · zero-shot LLM (optional) · agentic + RAG (aggregated from logged runs).
# 
# All outputs are written to `artifacts/final_experiment/` so they survive kernel restarts.

# In[1]:


import json, glob, sys, warnings, time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.0)
np.random.seed(42)

ROOT = Path("..").resolve()
OUT  = ROOT / "artifacts" / "final_experiment"
OUT.mkdir(parents=True, exist_ok=True)
print("Project root:", ROOT)
print("Artifacts   :", OUT)


# ## 1. Load real + synthetic sprints into a unified DataFrame

# In[2]:


NUMERIC_FEATURES = [
    "days_span", "issue_age_avg", "pr_age_avg",
    "total_issues", "total_prs", "total_commits",
    "closed_issues", "merged_prs",
    "issue_resolution_rate", "pr_merge_rate", "commit_frequency",
    "total_code_changes", "avg_pr_size", "code_concentration",
    "stalled_issues", "unreviewed_prs", "abandoned_prs", "long_open_issues",
    "unique_authors", "author_participation",
    "total_additions", "total_deletions", "files_changed",
]

def sprint_row(sp: dict, source: str) -> dict:
    m = sp.get("metrics", {})
    row = {k: float(m.get(k, 0) or 0) for k in NUMERIC_FEATURES}
    row["label"]     = int(bool(sp["risk_label"]["is_at_risk"]))
    row["sprint_id"] = sp.get("sprint_id", "")
    row["source"]    = source
    return row

real_files = [
    "data/mintplex-labs_anything-llm_sprints.json",
    "data/MemPalace_mempalace_sprints.json",
    "data/bibekgupta3333_repo-sprint_sprints.json",
]
real_rows = []
for f in real_files:
    real_rows.extend(sprint_row(s, "real") for s in json.load(open(ROOT / f)))

syn_rows = [sprint_row(s, "synthetic")
            for s in json.load(open(ROOT / "data/synthetic_sprints.json"))]

df = pd.DataFrame(real_rows + syn_rows)
print(f"Real sprints      : {len(real_rows)}  (positive={sum(r['label'] for r in real_rows)})")
print(f"Synthetic sprints : {len(syn_rows)}  (positive={sum(r['label'] for r in syn_rows)})")
print(f"Total             : {len(df)}")
df.head()


# ## 2. Build train / val / test splits (real-only val & test)

# In[3]:


real_df = df[df.source == "real"].reset_index(drop=True)
syn_df  = df[df.source == "synthetic"].reset_index(drop=True)

# Real-only val/test: stratify on label
real_train, real_holdout = train_test_split(
    real_df, test_size=0.40, random_state=42,
    stratify=real_df["label"] if real_df["label"].nunique() > 1 else None,
)
real_val, real_test = train_test_split(
    real_holdout, test_size=0.50, random_state=42,
    stratify=real_holdout["label"] if real_holdout["label"].nunique() > 1 else None,
)

# Baseline (real-only) training
train_bl = real_train.copy()
# H3 (real + synthetic) training
train_h3 = pd.concat([real_train, syn_df], ignore_index=True)

print(f"train_bl (real-only)     : {len(train_bl):>3d}   pos rate = {train_bl.label.mean():.2f}")
print(f"train_h3 (real+synthetic): {len(train_h3):>3d}   pos rate = {train_h3.label.mean():.2f}")
print(f"val  (real-only)         : {len(real_val):>3d}   pos rate = {real_val.label.mean():.2f}")
print(f"test (real-only, FROZEN) : {len(real_test):>3d}   pos rate = {real_test.label.mean():.2f}")

X_train_bl, y_train_bl = train_bl[NUMERIC_FEATURES].values, train_bl["label"].values
X_train_h3, y_train_h3 = train_h3[NUMERIC_FEATURES].values, train_h3["label"].values
X_val,      y_val      = real_val[NUMERIC_FEATURES].values,  real_val["label"].values
X_test,     y_test     = real_test[NUMERIC_FEATURES].values, real_test["label"].values


# ## 3. B1 — Rule-based oracle (label consistency check)

# In[4]:


def rule_predict(X: np.ndarray, feat=NUMERIC_FEATURES) -> np.ndarray:
    idx = {f: i for i, f in enumerate(feat)}
    stalled = X[:, idx["stalled_issues"]]
    pmr     = X[:, idx["pr_merge_rate"]]
    irr     = X[:, idx["issue_resolution_rate"]]
    longop  = X[:, idx["long_open_issues"]]
    cf      = X[:, idx["commit_frequency"]]
    score = (
        0.30 * (stalled >= 3).astype(float)
        + 0.20 * (pmr < 0.5).astype(float)
        + 0.15 * (irr < 0.4).astype(float)
        + 0.15 * (longop >= 2).astype(float)
        + 0.20 * (cf < 1.0).astype(float)
    )
    return (score >= 0.40).astype(int)

y_pred_rule = rule_predict(X_test)
print("B1 · Rule-based")
print(classification_report(y_test, y_pred_rule, target_names=["healthy", "at-risk"], zero_division=0))


# ## 4. B2 & H3 — XGBoost with grid-search + early stopping

# In[5]:


param_grid = {
    "max_depth":     [3, 5, 7],
    "learning_rate": [0.05, 0.1, 0.2],
    "n_estimators":  [100, 200],
    "subsample":     [0.8, 1.0],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def tune_and_fit(X_tr, y_tr, X_val, y_val, name=""):
    base = XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1, verbosity=0)
    grid = GridSearchCV(base, param_grid, cv=cv, scoring="f1", n_jobs=-1)
    grid.fit(X_tr, y_tr)
    best = {k: v for k, v in grid.best_params_.items() if k != "n_estimators"}
    final = XGBClassifier(**best, n_estimators=500, early_stopping_rounds=20,
                          eval_metric="logloss", random_state=42, verbosity=0)
    final.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    print(f"{name}: CV F1 = {grid.best_score_:.3f} ± {grid.cv_results_['std_test_score'][grid.best_index_]:.3f}"
          f"   best params = {grid.best_params_}   early-stop = {final.best_iteration}")
    return final, float(grid.best_score_), float(grid.cv_results_["std_test_score"][grid.best_index_])

xgb_bl, cv_bl_mean, cv_bl_std = tune_and_fit(X_train_bl, y_train_bl, X_val, y_val, "B2 · XGB-Baseline")
xgb_h3, cv_h3_mean, cv_h3_std = tune_and_fit(X_train_h3, y_train_h3, X_val, y_val, "H3 · XGB+synthetic")


# In[6]:


y_pred_bl = xgb_bl.predict(X_test);  y_proba_bl = xgb_bl.predict_proba(X_test)[:, 1]
y_pred_h3 = xgb_h3.predict(X_test);  y_proba_h3 = xgb_h3.predict_proba(X_test)[:, 1]

print("B2 · XGB-Baseline (real-only train) on real-only TEST:")
print(classification_report(y_test, y_pred_bl, target_names=["healthy","at-risk"], zero_division=0))
print("\nH3 · XGB (real + synthetic) on real-only TEST:")
print(classification_report(y_test, y_pred_h3, target_names=["healthy","at-risk"], zero_division=0))


# ## 5. B4 — Single-LLM zero-shot (optional; skipped if Ollama is unavailable)

# In[7]:


import re, requests

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"
def ollama_ok() -> bool:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        if r.status_code == 200:
            names = [m["name"] for m in r.json().get("models", [])]
            return any(OLLAMA_MODEL in n for n in names)
    except Exception:
        return False
    return False

LLM_ON = ollama_ok()
print("Ollama available:", LLM_ON)


# In[8]:


PROMPT = """You are a sprint health analyst. Given these sprint metrics, classify as AT-RISK or HEALTHY.

Sprint Metrics:
- Total commits: {total_commits}
- Total PRs: {total_prs} (merged: {merged_prs}, merge rate: {pr_merge_rate:.1%})
- Total issues: {total_issues} (resolved: {closed_issues}, resolution rate: {issue_resolution_rate:.1%})
- Stalled issues: {stalled_issues}
- Unreviewed PRs: {unreviewed_prs}
- Unique authors: {unique_authors}
- Commit frequency: {commit_frequency:.1f}/day

AT-RISK if it has many stalled issues, low merge/resolution rates, or stagnant activity.
Respond with ONLY one word: AT-RISK or HEALTHY"""

def llm_predict_row(row: pd.Series) -> int:
    prompt = PROMPT.format(**{k: row[k] for k in NUMERIC_FEATURES if k in row.index})
    try:
        r = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
            "options": {"temperature": 0.0},
        }, timeout=60)
        t = r.json().get("response", "").strip().upper()
        if re.search(r"AT[_\s-]?RISK", t): return 1
        if "HEALTHY" in t:                 return 0
        return -1
    except Exception:
        return -1

y_pred_llm = None
if LLM_ON:
    t0 = time.time(); preds = []
    for i, (_, row) in enumerate(real_test.iterrows()):
        preds.append(llm_predict_row(row))
        if (i+1) % 5 == 0: print(f"  ... {i+1}/{len(real_test)}")
    y_pred_llm = np.array(preds)
    elapsed = time.time() - t0
    valid = y_pred_llm >= 0
    print(f"B4 · Single-LLM zero-shot — {elapsed:.1f}s "
          f"({elapsed/len(real_test):.1f}s/sprint)  valid={valid.sum()}/{len(preds)}")
    if valid.sum() > 0:
        print(classification_report(y_test[valid], y_pred_llm[valid],
                                    target_names=["healthy","at-risk"], zero_division=0))
else:
    print("Skipping B4 — Ollama not available. Start with `ollama serve && ollama pull llama3`.")


# ## 6. AG — Agentic + RAG (aggregated from 28 logged production runs)

# In[9]:


run_files = sorted(glob.glob(str(ROOT / "artifacts/runs/run_metrics_*.json")))
runs = [json.load(open(p)) for p in run_files]
agentic = pd.DataFrame([{
    "f1":       r["f1_score"] if r["f1_score"] <= 1 else r["f1_score"]/100,
    "latency":  r["latency_seconds"],
    "parse":    r["parse_success_rate"],
    "fallback": r["fallback_rate"],
    "citation": r["citation_quality"]["score"],
    "risks":    r["counts"]["risks"],
} for r in runs])

print(f"n runs: {len(agentic)}")
summary = pd.DataFrame({
    "all (n=%d)" % len(agentic): agentic.median(),
    "last 10":                   agentic.tail(10).median(),
})
print("\nMedian operational metrics:")
print(summary.round(3).to_string())


# ## 7. Consolidated results table (all models on real-only test set)

# In[10]:


def metrics(y_true, y_pred, y_proba=None) -> dict:
    out = {
        "F1 (at-risk)": f1_score(y_true, y_pred, zero_division=0),
        "F1 (macro)":   f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Accuracy":     accuracy_score(y_true, y_pred),
        "Precision":    precision_score(y_true, y_pred, zero_division=0),
        "Recall":       recall_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None and len(np.unique(y_true)) == 2:
        out["AUC-ROC"] = roc_auc_score(y_true, y_proba)
    return out

results = {
    "B1 · Rule-based (oracle)":  metrics(y_test, y_pred_rule),
    "B2 · XGB-Baseline":         metrics(y_test, y_pred_bl, y_proba_bl),
    "H3 · XGB + synthetic":      metrics(y_test, y_pred_h3, y_proba_h3),
}
if y_pred_llm is not None and (y_pred_llm >= 0).sum() > 0:
    v = y_pred_llm >= 0
    results["B4 · Single-LLM zero-shot"] = metrics(y_test[v], y_pred_llm[v])

# Agentic: per-run F1 is single-sprint; report median as its best summary
results["AG · Agentic + RAG (median per-run)"] = {
    "F1 (at-risk)": float(agentic["f1"].median()),
    "F1 (macro)":   np.nan,
    "Accuracy":     np.nan,
    "Precision":    np.nan,
    "Recall":       np.nan,
}

results_df = pd.DataFrame(results).T.round(3)
print("=" * 78)
print(f"  FINAL RESULTS  (test set = real-only, n={len(y_test)})")
print("=" * 78)
print(results_df.to_string())
print("=" * 78)

results_df.to_csv(OUT / "results_table.csv")


# In[11]:


fig, ax = plt.subplots(figsize=(11, 5))
plot_df = results_df[["F1 (at-risk)", "F1 (macro)", "Accuracy"]].dropna(how="all")
plot_df.plot.bar(ax=ax, width=0.75)
ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
ax.set_title(f"Model comparison on real-only test set (n={len(y_test)})")
for c in ax.containers: ax.bar_label(c, fmt="%.2f", fontsize=8, padding=2)
plt.xticks(rotation=20, ha="right"); plt.tight_layout()
plt.savefig(OUT / "results_barplot.png", dpi=120)
plt.show()


# ## 8. Confusion matrices

# In[12]:


panes = [("B1 · Rule",  y_pred_rule),
         ("B2 · XGB-BL", y_pred_bl),
         ("H3 · XGB+syn", y_pred_h3)]
if y_pred_llm is not None and (y_pred_llm >= 0).sum() > 0:
    v = y_pred_llm >= 0
    panes.append(("B4 · LLM", y_pred_llm[v]))

fig, axes = plt.subplots(1, len(panes), figsize=(4 * len(panes), 3.5))
if len(panes) == 1: axes = [axes]
for ax, (name, yp) in zip(axes, panes):
    yt = y_test if name != "B4 · LLM" else y_test[y_pred_llm >= 0]
    cm = confusion_matrix(yt, yp)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["healthy","at-risk"], yticklabels=["healthy","at-risk"], ax=ax)
    ax.set_title(f"{name}\nF1={f1_score(yt, yp, zero_division=0):.2f}")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout(); plt.savefig(OUT / "confusion_matrices.png", dpi=120); plt.show()


# ## 9. Feature importance (XGB-H3)

# In[13]:


imp = xgb_h3.get_booster().get_score(importance_type="gain")
imp_s = pd.Series(imp).reindex(NUMERIC_FEATURES, fill_value=0).sort_values()
fig, ax = plt.subplots(figsize=(8, 6))
imp_s.plot.barh(ax=ax, color=sns.color_palette("viridis", len(imp_s)))
ax.set_xlabel("Gain"); ax.set_title("XGB-H3 · feature importance (gain)")
plt.tight_layout(); plt.savefig(OUT / "feature_importance.png", dpi=120); plt.show()

print("Top 5 features (gain):")
for f, v in imp_s.tail(5)[::-1].items():
    print(f"  {f:<28s} {v:.3f}")


# ## 10. Persist final benchmark JSON (for slides)

# In[14]:


benchmark = {
    "experiment":   "Final ML project results",
    "n_real_train": int(len(train_bl)),
    "n_real_val":   int(len(real_val)),
    "n_real_test":  int(len(real_test)),
    "n_h3_train":   int(len(train_h3)),
    "cv_bl":        {"mean": cv_bl_mean, "std": cv_bl_std},
    "cv_h3":        {"mean": cv_h3_mean, "std": cv_h3_std},
    "results":      {k: {kk: (None if pd.isna(vv) else float(vv)) for kk, vv in v.items()}
                     for k, v in results.items()},
    "agentic_medians_all":     {c: float(agentic[c].median())        for c in agentic.columns},
    "agentic_medians_last10":  {c: float(agentic.tail(10)[c].median()) for c in agentic.columns},
}
(OUT / "final_benchmark.json").write_text(json.dumps(benchmark, indent=2))
print(f"Wrote {OUT / 'final_benchmark.json'}")
print("\nHeadline numbers for the slide deck:")
print(f"  XGB-Baseline  CV F1 = {cv_bl_mean:.3f} ± {cv_bl_std:.3f}")
print(f"  XGB-H3        CV F1 = {cv_h3_mean:.3f} ± {cv_h3_std:.3f}")
print(f"  XGB-Baseline  test F1 = {results['B2 · XGB-Baseline']['F1 (at-risk)']:.3f}")
print(f"  XGB-H3        test F1 = {results['H3 · XGB + synthetic']['F1 (at-risk)']:.3f}")
if y_pred_llm is not None and (y_pred_llm >= 0).sum() > 0:
    llm_m = results["B4 · Single-LLM zero-shot"]
    print(f"  LLM zero-shot test F1={llm_m['F1 (at-risk)']:.3f}  macroF1={llm_m['F1 (macro)']:.3f}  acc={llm_m['Accuracy']:.3f}")
print(f"  Agentic median per-run F1 (all)    = {agentic['f1'].median():.2f}")
print(f"  Agentic median per-run F1 (last 10)= {agentic.tail(10)['f1'].median():.2f}")
print(f"  Agentic median latency             = {agentic['latency'].median():.1f} s")
print(f"  Agentic median citation quality (all / last 10) = "
      f"{agentic['citation'].median():.2f} / {agentic.tail(10)['citation'].median():.2f}")

