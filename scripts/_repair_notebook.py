"""Repair cells 15 and 24 of final_experiment.ipynb."""
import json, pathlib

NB = pathlib.Path("notebooks/final_experiment.ipynb")
nb = json.loads(NB.read_text())

CELL15 = '''from src.agents.state import OrchestratorState, GitHubIssue
from src.agents.orchestrator import MasterOrchestrator

AG_SAMPLE_SIZE = int(os.environ.get("AG_SAMPLE_SIZE", "8"))  # each run ~15-30s

test_with_idx = real_test.reset_index(drop=True)
if test_with_idx["label"].nunique() > 1:
    ag_sample = (test_with_idx.groupby("label", group_keys=False)
                 .apply(lambda g: g.sample(min(len(g), max(1, AG_SAMPLE_SIZE // 2)), random_state=42)))
else:
    ag_sample = test_with_idx.sample(min(len(test_with_idx), AG_SAMPLE_SIZE), random_state=42)
ag_sample = ag_sample.reset_index(drop=True)
print(f"Running agentic pipeline on {len(ag_sample)} sprints "
      f"(pos={int(ag_sample.label.sum())}, neg={int((1 - ag_sample.label).sum())})")


def _pull_sprint_issues(sprint_id, repo_full, limit=30):
    try:
        r = col.get(
            where={"$and": [{"type": "issue"}, {"sprint_id": sprint_id}, {"repo_full": repo_full}]},
            limit=limit, include=["metadatas", "documents"],
        )
    except Exception:
        return []
    out = []
    for i, meta in enumerate(r.get("metadatas", [])):
        try:
            title = ""
            if r.get("documents") and i < len(r["documents"]):
                title = r["documents"][i].splitlines()[0][:120]
            out.append(GitHubIssue(
                number=int(meta.get("number", i)),
                title=title,
                body="",
                state=meta.get("state", "open") or "open",
                labels=[],
                created_at=meta.get("date", "") or None,
            ))
        except Exception:
            continue
    return out


def _health_to_label(analysis):
    if not analysis:
        return -1
    hs = (analysis.get("health_status") or "").lower()
    if hs in ("at_risk", "critical"):
        return 1
    if hs in ("on_track", "healthy"):
        return 0
    cp = analysis.get("completion_probability")
    if isinstance(cp, (int, float)):
        return 0 if cp >= 50 else 1
    return -1


orch = MasterOrchestrator()
ag_records = []
t_start_all = time.time()
for i, (_, row) in enumerate(ag_sample.iterrows()):
    sid, repo_full = row["sprint_id"], row["repo_full"]
    issues = _pull_sprint_issues(sid, repo_full)
    state = OrchestratorState(
        repositories=[repo_full],
        repository_url=f"https://github.com/{repo_full}",
        sprint_id=sid,
        eval_mode="resilient",
        github_issues=issues,
    )
    t0 = time.time()
    try:
        result = orch.invoke(state)
        pred = _health_to_label(result.sprint_analysis)
        ag_records.append({
            "sprint_id": sid, "repo": repo_full,
            "y_true": int(row["label"]), "y_pred": pred,
            "health_status": (result.sprint_analysis or {}).get("health_status"),
            "analysis_source": result.analysis_source,
            "risk_source": result.risk_source,
            "n_risks": len(result.identified_risks),
            "n_recs": len(result.recommendations),
            "n_citations": len(result.evidence_citations),
            "latency_s": time.time() - t0,
            "errors": len(result.errors),
        })
    except Exception as e:
        ag_records.append({
            "sprint_id": sid, "repo": repo_full,
            "y_true": int(row["label"]), "y_pred": -1,
            "health_status": None, "analysis_source": None, "risk_source": None,
            "n_risks": 0, "n_recs": 0, "n_citations": 0,
            "latency_s": time.time() - t0, "errors": 1, "error_msg": str(e)[:200],
        })
    print(f"  [{i+1}/{len(ag_sample)}] {repo_full} {sid}  "
          f"true={row['label']} pred={ag_records[-1]['y_pred']}  "
          f"{ag_records[-1]['latency_s']:.1f}s")
print(f"\\nTotal agentic wall-time: {time.time() - t_start_all:.1f}s")

agentic = pd.DataFrame(ag_records)
agentic.to_csv(OUT / "agentic_predictions.csv", index=False)
print("\\nPer-sprint agentic predictions:")
print(agentic[["repo", "sprint_id", "y_true", "y_pred", "health_status",
               "analysis_source", "latency_s"]].to_string(index=False))

ag_valid = agentic[agentic["y_pred"] >= 0]
if len(ag_valid):
    print(f"\\nAgentic metrics on {len(ag_valid)} successful runs:")
    print(classification_report(ag_valid["y_true"], ag_valid["y_pred"],
                                target_names=["healthy", "at-risk"], zero_division=0))
    print(f"Median latency: {agentic['latency_s'].median():.1f}s   "
          f"LLM-source analysis: {(agentic['analysis_source'] == 'llm').sum()}/{len(agentic)}")
'''

CELL24 = '''benchmark = {
    "experiment":   "Final ML project results (ChromaDB + LangGraph)",
    "n_real_train": int(len(train_bl)),
    "n_real_val":   int(len(real_val)),
    "n_real_test":  int(len(real_test)),
    "n_h3_train":   int(len(train_h3)),
    "cv_bl":        {"mean": cv_bl_mean, "std": cv_bl_std},
    "cv_h3":        {"mean": cv_h3_mean, "std": cv_h3_std},
    "results":      {k: {kk: (None if pd.isna(vv) else float(vv)) for kk, vv in v.items()}
                     for k, v in results.items()},
    "agentic_n_runs":           int(len(agentic)),
    "agentic_n_valid":          int(len(ag_valid)),
    "agentic_latency_median_s": float(agentic["latency_s"].median()) if len(agentic) else None,
    "agentic_llm_source_rate":  float((agentic["analysis_source"] == "llm").mean()) if len(agentic) else None,
}
(OUT / "final_benchmark.json").write_text(json.dumps(benchmark, indent=2))
print(f"Wrote {OUT / 'final_benchmark.json'}")
print("\\nHeadline numbers for the slide deck:")
print(f"  XGB-Baseline  CV F1 = {cv_bl_mean:.3f} \\u00b1 {cv_bl_std:.3f}")
print(f"  XGB-H3        CV F1 = {cv_h3_mean:.3f} \\u00b1 {cv_h3_std:.3f}")
print(f"  XGB-Baseline  test F1 = {results['B2 \\u00b7 XGB-Baseline']['F1 (at-risk)']:.3f}")
print(f"  XGB-H3        test F1 = {results['H3 \\u00b7 XGB + synthetic']['F1 (at-risk)']:.3f}")
if y_pred_llm is not None and (y_pred_llm >= 0).sum() > 0:
    llm_m = results["B4 \\u00b7 Single-LLM zero-shot"]
    print(f"  LLM zero-shot test F1={llm_m['F1 (at-risk)']:.3f}  "
          f"macroF1={llm_m['F1 (macro)']:.3f}  acc={llm_m['Accuracy']:.3f}")
if len(ag_valid):
    ag_m = results["AG \\u00b7 LangGraph orchestrator"]
    print(f"  Agentic (LangGraph) F1 (at-risk) = {ag_m['F1 (at-risk)']:.3f}  "
          f"macroF1={ag_m['F1 (macro)']:.3f}  acc={ag_m['Accuracy']:.3f}  (n={len(ag_valid)})")
    print(f"  Agentic median latency           = {agentic['latency_s'].median():.1f} s")
else:
    print("  Agentic: no valid runs.")
'''

# Also fix cell 17 (has trailing typo: "{len(y_test)}" missing closing paren )
CELL17 = '''def metrics(y_true, y_pred, y_proba=None) -> dict:
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
    "B1 \u00b7 Rule-based (oracle)":  metrics(y_test, y_pred_rule),
    "B2 \u00b7 XGB-Baseline":         metrics(y_test, y_pred_bl, y_proba_bl),
    "H3 \u00b7 XGB + synthetic":      metrics(y_test, y_pred_h3, y_proba_h3),
}
if y_pred_llm is not None and (y_pred_llm >= 0).sum() > 0:
    v = y_pred_llm >= 0
    results["B4 \u00b7 Single-LLM zero-shot"] = metrics(y_test[v], y_pred_llm[v])

if len(ag_valid):
    results["AG \u00b7 LangGraph orchestrator"] = metrics(
        ag_valid["y_true"].values, ag_valid["y_pred"].values
    )
else:
    results["AG \u00b7 LangGraph orchestrator"] = {
        "F1 (at-risk)": np.nan, "F1 (macro)": np.nan,
        "Accuracy": np.nan, "Precision": np.nan, "Recall": np.nan,
    }

results_df = pd.DataFrame(results).T.round(3)
print("=" * 78)
print(f"  FINAL RESULTS  (test set = real-only, n={len(y_test)})")
print("=" * 78)
print(results_df.to_string())
print("=" * 78)
results_df.to_csv(OUT / "results_table.csv")
'''

CELL18 = '''fig, ax = plt.subplots(figsize=(11, 5))
plot_df = results_df[["F1 (at-risk)", "F1 (macro)", "Accuracy"]].dropna(how="all")
plot_df.plot.bar(ax=ax, width=0.75)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score")
ax.set_title(f"Model comparison on real-only test set (n={len(y_test)})")
for c in ax.containers:
    ax.bar_label(c, fmt="%.2f", fontsize=8, padding=2)
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(OUT / "results_barplot.png", dpi=120)
plt.show()
'''

overrides = {15: CELL15, 17: CELL17, 18: CELL18, 24: CELL24}
for idx, src in overrides.items():
    nb["cells"][idx]["source"] = src.splitlines(keepends=True)
    nb["cells"][idx]["outputs"] = []
    nb["cells"][idx]["execution_count"] = None

NB.write_text(json.dumps(nb, indent=1))
print("Rewrote cells:", list(overrides))
