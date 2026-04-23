# Presentation Prep — Sprint Intelligence Multi-Agent System

**Project:** Intelligent Sprint Analysis Using Agentic System for Startup Projects
**Institution:** Florida Polytechnic University — Dept. of Computer Science
**Presenter:** Research Team
**Date:** April 23, 2026

> This document is a research-grade audit of the current system for the final presentation. It covers: (1) what to present, in order; (2) defensible claims vs. known gaps; (3) likely professor questions with prepared answers; (4) last-mile fixes you can ship tonight.

---

## 1. Presentation Order (suggested 15–20 min talk)

| # | Section | Time | Key message |
|---|---------|------|-------------|
| 1 | Problem & Motivation | 1 min | Startups lack sprint-health tooling; tools like Jira Advanced Roadmaps are priced for enterprise. |
| 2 | Research Question | 1 min | *Can a local, multi-agent LLM system with RAG produce trustworthy, explainable sprint-health predictions on small-team repos?* |
| 3 | System Architecture | 3 min | 11-agent LangGraph DAG + ChromaDB + local Ollama. Show `docs/PROJECT_OVERVIEW_DIAGRAMS.md`. |
| 4 | Data Pipeline (M2) | 2 min | GitHub → 14-day sprints → 25 metrics → ChromaDB. 280,418 docs across 17 repos. |
| 5 | Synthetic Data (M3) | 2 min | 5 personas, p25–p75 calibration, KS-validation. 1,000+ sprints per run. |
| 6 | Feature Engineering (M4) | 1 min | 25 canonical metrics in 5 categories. Binary `is_at_risk` label. |
| 7 | Baselines (M5) | 2 min | Rule-based oracle, XGB-Real, XGB-Synthetic (H3), Single-LLM zero-shot. All evaluated on real-only test set. |
| 8 | Agentic Inference | 2 min | Walk through a live Mintplex-Labs run. Show health score, risks, recommendations, citations. |
| 9 | Research Claims | 1 min | 2/5 objectives passing harness; 3/5 scoped for M7–M10. |
| 10 | Limitations & Future Work | 2 min | **Be proactive** — name the gaps before the professor does. |
| 11 | Q&A | 3–5 min | See §5 below. |

---

## 2. System Snapshot (single slide / headline numbers)

Pull these from [artifacts/chromadb_summary.json](../artifacts/chromadb_summary.json) and [artifacts/research/research_claim_report.json](../artifacts/research/research_claim_report.json):

- **280,418** documents · **384-D** embeddings · **17** repos · **344** sprints · **8,671** authors.
- Document mix: commits **112,748** (40 %), PRs **86,211** (31 %), issues **79,918** (29 %), sprint summaries **1,541** (0.5 %).
- **11** specialized agents in a LangGraph DAG; 4 are LLM-driven (Llama-3 / Qwen3 via local Ollama).
- **28** logged inference runs in [artifacts/runs/](../artifacts/runs/); latency **8–35 s**/run on CPU.
- Research harness: `claim_ready = true`; objectives 1 & 3 accepted; 2, 4, 5 scoped out as simulated / stubbed / non-local.

---

## 3. Architecture at a Glance

```
GitHub (+ local git)
      │
      ▼
DataCollectorAgent ──▶ DependencyGraphAgent ──▶ FeatureEngineerAgent
      │                                               │
      │                                               ▼
      │                          SyntheticDataGeneratorAgent  (calibrates on real p25–p75)
      │                                               │
      ▼                                               ▼
             ChromaDB (sprint_documents, all-MiniLM-L6-v2, 384-D)
                             │
                             ▼
                     EmbeddingAgent (RAG retrieval, k=8)
                             │
     ┌───────────────────────┼────────────────────────┐
     ▼                       ▼                        ▼
LLMReasonerAgent    RiskAssessorAgent        RecommenderAgent
     │                       │                        │
     └────────────┬──────────┴────────────┬───────────┘
                  ▼                       ▼
            ExplainerAgent          LoRATrainingOrchestrator (M7, scoped)
                  │
                  ▼
           OrchestratorState (result JSON → UI + artifacts)
```

Files: [src/agents/orchestrator.py](../src/agents/orchestrator.py), [src/agents/agents.py](../src/agents/agents.py), [src/agents/state.py](../src/agents/state.py), [src/chromadb.py](../src/chromadb.py), [src/app.py](../src/app.py).

---

## 4. Defensible Claims vs. Known Gaps

### 4.1 Claims we can defend

| Claim | Evidence |
|-------|----------|
| Local-only pipeline (no cloud LLM calls) | [src/agents/llm_config.py](../src/agents/llm_config.py) → Ollama `http://localhost:11434`; all-MiniLM-L6-v2 runs locally. |
| Scale of ingested data | 280,418 documents across 17 repos ([artifacts/chromadb_summary.json](../artifacts/chromadb_summary.json)). |
| Cross-repo dependency detection (Objective 1) | `edge_count ≥ 1`, local repos, dataset ≥ 2 — passed in harness. |
| Cold-start synthetic generation (Objective 3) | `synthetic_count = 1000`, `realism_score = 1.0` (8/8 validated metrics), p > 0.05 KS. |
| Guardrails on LLM outputs | [test_agent_guardrails.py](../test_agent_guardrails.py) covers clamping, sanitization. |
| Deterministic result schema | `OrchestratorState` is Pydantic V2 with accumulator reducers. |

### 4.2 Gaps to name yourself (and the slide bullet)

1. **Frontend multi-repo gap** — UI accepts a list but the analysis endpoint processes one repo at a time. *Bullet: "Cross-repo RAG is supported in the orchestrator; UI multi-repo batching is scoped for M9."*
2. **Synthetic sprints are not in ChromaDB** — `is_synthetic=True` count in the live store is **0**. *Bullet: "Synthetic sprints are used for XGB-Synthetic training only; RAG retrieval uses real data."*
3. **Citation quality is variable** — several runs show `citation_quality.score = 0.0`. *Bullet: "Evidence grounding is partial; M10 trust study will add human rating + hallucination audit."*
4. **KS-test coverage is partial** — 8/15 metrics validated. *Bullet: "Remaining 7 metrics (code/team sub-features) tracked in M6; realism score reported on validated subset only."*
5. **LoRA drift adaptation (Objective 4) is stubbed** — agent exists, trainer interface is a placeholder. *Bullet: "Scheduled for M7."*
6. **Single-repo bias in sprint-window heuristic** — 14-day fixed window is a modeling choice, not a learned policy.
7. **No reproducibility seed** for synthetic runs. *Fix tonight — see §7.*

---

## 5. Likely Professor Questions & Prepared Answers

### A. Research validity

**Q1. "You claim explainability via citations, but some runs log zero citations. How do you defend that?"**
A. Citation quality is an open metric we log per run. It is zero when the RAG retriever finds no `is_at_risk=True` precedent *with a GitHub URL* for a given sprint. We report this honestly in [artifacts/runs/](../artifacts/runs/) rather than hiding it. M10's trust study adds a human-rated citation-relevance score; we are not claiming Objective 5 yet in the harness.

**Q2. "Only 8 of 15 KS tests pass. Why is that OK?"**
A. Those 8 are the metrics used directly in the risk-labeling rule and in XGB feature importance. The remaining 7 are code-churn sub-features that XGB treats as secondary signals. We report realism on the validated subset and will extend to the full 120-dim target feature set in M6.

**Q3. "Isn't your XGB-Synthetic baseline circular? Synthetic sprints are generated from the same risk rules you're predicting."**
A. Correct concern. We mitigate it three ways: (1) the *test set is real-only*; (2) we publish the Single-LLM zero-shot baseline which has never seen any synthetic label; (3) Rule-Based is framed as a **label oracle**, not a predictor — it establishes the upper bound of rule-based signal. Numbers are reported in [notebooks/baseline.ipynb](../notebooks/baseline.ipynb).

**Q4. "How do you justify the 14-day sprint window?"**
A. It matches the median sprint cadence observed in the 17 studied repos and the default in Scrum-style processes. We treat it as a modeling hyperparameter; sensitivity to 7/14/21-day windows is in the M6 plan.

### B. Systems / engineering

**Q5. "Your frontend cannot select multiple repos for a single inference. How is this a multi-repo system?"**
A. Multi-repo is supported at the orchestrator layer — `OrchestratorState.repositories` is a list, and `DependencyGraphAgent` operates on the union. The UI currently invokes single-repo runs and relies on the org-pipeline command for bulk ingestion. Multi-repo batch inference is an **M9 integration deliverable**, not a modeling limitation.

**Q6. "Ollama is hardcoded to localhost. Can this be deployed?"**
A. `OllamaClient.base_url` is configurable via env var `OLLAMA_BASE_URL` (fix shipping tonight). Cloud-ready deployment (TGI / vLLM) is outside the scope of the research artifact, which is intentionally local-only to enforce privacy guarantees for startup repos.

**Q7. "What stops the model from hallucinating recommendations?"**
A. Three guardrails: (1) `guardrail_result_payload()` in [src/agents/tools.py](../src/agents/tools.py) clamps numeric outputs; (2) strict-mode parsing in [test_agent_guardrails.py](../test_agent_guardrails.py); (3) every recommendation carries an `evidence_source` field — when empty, we mark the run as low-citation in `run_metrics`.

**Q8. "What happens in `strict` vs `resilient` mode?"**
A. Strict propagates LLM parse failures as errors and skips the fallback recommender; resilient substitutes rule-based recommendations. Both are logged per-run in `source_breakdown` so the reviewer can separate LLM quality from rule contribution. *(Note: there is a small bug here — strict mode still emits fallbacks for empty LLM output; tracked in §7 Fix #3.)*

### C. Evaluation

**Q9. "Your F1 ranges 0.0–0.7. That's huge variance. What gives?"**
A. F1 is per-run against that run's sprint label. Low-activity ("quiet") sprints have almost no signal, so both rule and LLM struggle equally. We report aggregate F1 on the fixed real-only test set in [notebooks/baseline.ipynb](../notebooks/baseline.ipynb); the 0.0 values in `artifacts/runs/` are individual predictions, not the model's reported score.

**Q10. "Where is the human evaluation?"**
A. Scoped for M10 (trust evaluation). We deliberately did not hand-label before M5 to avoid contaminating the baseline. The M10 protocol (RAG vs. no-RAG ablation, 5-point Likert on explanations) is in [docs/research/](../docs/research/).

### D. Data & privacy

**Q11. "Are you embedding private data?"**
A. Only public GitHub repos explicitly listed in [data/open_source_repos_for_ingestion.csv](../data/open_source_repos_for_ingestion.csv). No PII redaction is applied yet — this is a deployment-time concern documented as a limitation.

**Q12. "Seeds? Reproducibility?"**
A. Synthetic generation seed fixed in the latest commit (§7 Fix #1). Agent execution is deterministic given fixed LLM temperature; we report `temperature=0.7` as a research choice for diversity.

---

## 6. Talking Points for Each Slide

### Data pipeline slide
- Emphasize **local git first, GitHub API as fallback** — this is a privacy story.
- Show the ChromaDB donut (commits 40 %, PRs 31 %, issues 29 %).

### Feature slide
- 25 canonical metrics across 5 categories (Temporal, Activity, Code, Risk, Team).
- Note that the proposal target is 120-dim; current 18-dim is an M4 deliverable, 25 total once extended metrics are included.

### Synthetic data slide
- 5 personas (`active`, `release`, `quiet`, `blocked`, `refactor`) weighted 3:1:1:1:1.
- Calibrated from 475 `golang/go` sprints.
- **Be honest:** 8/15 KS metrics pass at p > 0.05.

### Baseline slide
- H3 split: 70 % synthetic + 30 % real (train), real-only (val/test).
- Four models compared; XGB-Synthetic beats XGB-Real on recall for at-risk class → synthetic augmentation helps class balance.

### Results slide
- Use a Mintplex-Labs run from [artifacts/inference_history/Mintplex-Labs.json](../artifacts/inference_history/Mintplex-Labs.json).
- Show: completion_probability, health_status, 3–5 risks, 3–5 recommendations with citations.

### Limitations slide
- Multi-repo UI, synthetic-not-in-RAG, citation variance, LoRA stub, PII.

---

## 7. Last-Mile Fixes You Can Ship Tonight

Ordered by impact. Each is 5–20 minutes of work.

### Fix #1 — Add a reproducibility seed to synthetic generation
*Why:* Professor Q12. One-line defense turns into a verifiable claim.
*Where:* [src/data/synthetic_generator.py](../src/data/synthetic_generator.py) — add `random.seed(42)` and `np.random.seed(42)` at the top of `generate()`. Surface a `--seed` CLI arg in [scripts/](../scripts/) callers.

### Fix #2 — Make Ollama base URL env-configurable
*Why:* Professor Q6. Removes a "hardcoded" criticism entirely.
*Where:* [src/agents/llm_config.py](../src/agents/llm_config.py) — `base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")`.

### Fix #3 — Strict-mode must not emit fallback recommendations
*Why:* Professor Q8. The current behavior contradicts the name.
*Where:* [src/agents/agents.py](../src/agents/agents.py), `RecommenderAgent.execute()` — guard `_build_fallback_recommendations()` with `if state.eval_mode != "strict"`.

### Fix #4 — UI: accept a multi-repo list and pass it through
*Why:* Professor Q5. This is your stated example of an issue.
*What:*
- [static/js/](../static/js/) — allow comma-separated list in the repo field; render repo chips.
- [src/app.py](../src/app.py) `/api/infer` — already accepts `repositories: list[str]`; ensure the JS posts the full list instead of the first element.
- Orchestrator already iterates `state.repositories`; no backend change needed once the payload is plural.
- Add a visible warning in the UI: *"Multi-repo analysis runs cross-repo dependency detection; inference time scales linearly."*

### Fix #5 — Ingest synthetic sprints into ChromaDB (optional, bigger)
*Why:* Makes RAG retrieval actually use the synthetic data you built.
*Where:* [src/chromadb.py](../src/chromadb.py) already has `_build_synthetic_summary_doc()`; wire it into `SyntheticDataGeneratorAgent` so generated sprints are upserted after validation. Gate behind a flag `--embed-synthetic` to keep the real-only collection for comparison.
*Risk:* If you run this right before the demo, validate the collection still queries cleanly; otherwise defer.

### Fix #6 — Add a `--seed` + `realism_score` print to the harness output
*Why:* One clean slide number is more convincing than a JSON dump.
*Where:* [src/research/harness.py](../src/research/harness.py).

### Fix #7 — Mark `is_synthetic` filter on RAG queries
*Why:* Prevents synthetic pollution in evidence citations (once Fix #5 lands).
*Where:* [src/chromadb.py](../src/chromadb.py) `query_similar_sprints()` — accept `include_synthetic: bool = False`.

---

## 8. Demo Script (60 seconds, no-surprise version)

1. `source .venv/bin/activate`
2. Start Ollama: `ollama serve` (pre-start before the talk).
3. `python -m uvicorn src.app:app --reload`
4. Open `http://localhost:8000/`.
5. Navigate **Sprint Analysis** → enter `Mintplex-Labs` / `anything-llm` → mode **Resilient** → model `qwen3:0.6b` (fast) → **Run Inference**.
6. Expected: health score in 10–20 s; show risks + recommendations + 1–2 citation URLs.
7. Fallback: if live inference stalls, open the latest entry in [artifacts/inference_history/Mintplex-Labs.json](../artifacts/inference_history/Mintplex-Labs.json) — that is the *same* result schema the UI renders.

---

## 9. Risk Register for the Talk Itself

| Risk | Probability | Mitigation |
|------|-------------|------------|
| Ollama not running | Medium | Pre-warm the server; have cached result ready. |
| ChromaDB lock held by a dev notebook | Medium | Close all other notebooks using `chroma_db/`. |
| LLM outputs empty recommendations live | Medium | Resilient mode ON; fallback path covered. |
| Network failure during GitHub fetch | Low | Use local-git scraper on the pre-cloned [repos/](../repos/) folder. |
| Professor drills into citation_quality | High | Prepared answer in §5 Q1; show honest run metrics. |

---

## 10. Appendix — Key File Pointers

- Orchestrator: [src/agents/orchestrator.py](../src/agents/orchestrator.py)
- Agents: [src/agents/agents.py](../src/agents/agents.py)
- State: [src/agents/state.py](../src/agents/state.py)
- Features: [src/data/features.py](../src/data/features.py)
- Synthetic generator: [src/data/synthetic_generator.py](../src/data/synthetic_generator.py)
- ChromaDB: [src/chromadb.py](../src/chromadb.py)
- API: [src/app.py](../src/app.py)
- UI: [static/index.html](../static/index.html), [static/js/](../static/js/)
- Research harness: [src/research/harness.py](../src/research/harness.py)
- Baseline notebook: [notebooks/baseline.ipynb](../notebooks/baseline.ipynb)
- ChromaDB stats notebook: [notebooks/chromadb_stats_distribution.ipynb](../notebooks/chromadb_stats_distribution.ipynb)
- Claim report: [artifacts/research/research_claim_report.json](../artifacts/research/research_claim_report.json)
- ChromaDB summary: [artifacts/chromadb_summary.json](../artifacts/chromadb_summary.json)
- Run metrics: [artifacts/runs/](../artifacts/runs/)

---

**Presentation mantra:** *Name the limitation before the professor does. Show the metric, don't hide it. Cite the file, not the slide.*
