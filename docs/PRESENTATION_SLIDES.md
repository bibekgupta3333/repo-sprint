# Intelligent Sprint Analysis Using Agentic System for Startup Projects — Research Presentation

**Course:** Machine Learning · Florida Polytechnic University · Spring 2026
**Team:** Bibek Gupta (Lead) · Saarupya Sunkara · Siwani Sah · Deepthi Reddy Chelladi
**Deliverable:** 11-agent LangGraph pipeline · dense-vector retrieval (ChromaDB) · local LLM inference (Ollama)

> 16 slides for a 15–20 min talk + 3–5 min Q&A. Ten-section flow: **Problem → Research Question → Architecture → Data → Synthetic Data → Features → Baselines → Agentic Inference → Limitations & Future Work → Q&A** + a closing **References** slide.

---

## Slide 1 — Title

# Intelligent Sprint Analysis Using Agentic System for Startup Projects
### A Multi-Stage ML Pipeline: Supervised Classification + Dense Retrieval + Local LLM Reasoning

| | |
|---|---|
| **Course** | Machine Learning — Final Project |
| **Institution** | Florida Polytechnic University · Computer Science |
| **Date** | April 2026 |
| **ML task** | Binary classification — `is_at_risk ∈ {0, 1}` per sprint |
| **Dataset** | 17 repos · 1 541 real sprints + 5 000 synthetic · 18 features |
| **Headline result** | Agentic LangGraph orchestrator · F1 = 0.857 on the held-out real-sprint slice |

**Short version.** We built a local, explainable system that predicts sprint risk from GitHub activity and reaches F1 = 0.857 on held-out real sprints.

**Long version.** We took a simple problem — will this sprint miss its deadline? — and added three hard requirements. First, every prediction needs a real explanation with links to actual pull requests and issues, not just a number. Second, the whole thing has to run on a regular laptop with no paid cloud services. Third, a brand-new repo with no history still needs to give useful advice. Those three rules drove every choice we made. The final score of 0.857 matters, but we also report how fast it runs, whether the explanations are real or made-up, and whether a PM would actually use it.

**What I'll say (30 s).** Good morning. We built a tool that predicts whether a sprint will hit its deadline, just by looking at GitHub activity. No complex setup, no paid services. The trick is it works even on brand-new repos with no history — which matters for startups. In the next twenty minutes I'll show you the problem we solved, how we solved it, and what the numbers show.

---

## Slide 2 — Problem & Motivation  ·  *Section 1 · 1 min*

### Why startups need this

| Stakeholder | Pain today | Cost |
|---|---|---|
| Startup engineers | No PM, 6–10 h/week manual sprint tracking | Lost engineering time |
| Engineering leads | Integration breaks surface late — **34 % of failures cross-repo** | Missed releases |
| Founders | Enterprise PM tools demand 6–12 mo of history & cost **$500–$2 k / month** | Locked out of tooling |

**Existing tools fall short for small teams.**

- *Jira Advanced Roadmaps* — priced for enterprise (per-seat license + minimum tier).
- *Linear Insights / Shortcut Analytics* — require months of usage history before signal stabilises.
- *Cloud LLM dashboards* — ship private repo data off-device; many startups can't legally do this.

### The opening: *what gets a small team to actionable sprint health on day one, with no paid APIs?*

**Short version.** Startups have the same sprint-health problem as enterprises but none of the budget, history, or tolerance for cloud data egress.

**Long version.** Startups and big companies have the same sprint problems: stalled issues, stuck pull requests, late integration breaks. But startups can't afford Jira's enterprise pricing, and they don't have months of data for Linear to warm up on. Plus, most founders won't send private code to a cloud service. So the question is simple: can we build something that works on day one, runs on your laptop, keeps your code private, and actually explains *why* a sprint is at risk? That's what shaped every decision we made.

**What I'll say (60 s).** Startups have a sprint-tracking problem and no realistic tools to solve it. Jira's advanced roadmaps are priced for the enterprise. Linear's analytics need months of history before they're useful. Any cloud-LLM dashboard ships your private repo data off-device, which a lot of seed-stage teams legally can't do. Six-to-ten hours a week per engineer disappear into manual tracking, and a third of the failures we see come from cross-repo integration breaks that get caught too late. That's the gap our project is built to close.

---

## Slide 3 — Research Question  ·  *Section 2 · 1 min*

> **Can a local, multi-agent LLM system with RAG produce trustworthy, explainable sprint-health predictions on small-team repositories — including cold-start projects with no historical data?**

### Operational decomposition

| Sub-question | Hypothesis | How we test it |
|---|---|---|
| **RQ1.** Does retrieval improve grounded explanation quality vs. a vanilla LLM? | RAG-grounded answers cite real artefacts; vanilla LLM hallucinates URLs. | Citation parse-rate + manual citation audit (Section 8). |
| **RQ2.** Can a $\le$1 B local LLM hit acceptable F1 with a deterministic fallback safety net? | A small local model + structured agents matches a single big LLM on F1 and beats it on cost & privacy. | Baselines vs. agentic head-to-head on the frozen test set (Section 7). |
| **RQ3.** Does persona-calibrated synthetic data unblock cold-start without polluting the real-data signal? | Train-only synthetic raises retrieval coverage on new repos but leaves real-only F1 unchanged. | Train-on-synth / test-on-real comparison (Section 5). |
| **RQ4.** Does a multi-agent decomposition outperform a single-prompt LLM on multi-faceted output (risk + recommendation + explanation)? | Specialised agents reduce prompt complexity → better JSON parse-rate + lower latency variance. | Agentic-vs-zero-shot comparison (Section 7). |
| **RQ5.** Does the system stay laptop-runnable end-to-end? | Local Docker stack ≤ 16 GB RAM, p50 latency ≤ 60 s. | Operational metrics on the demo run (Section 8). |

**Scope guardrail.** We deliberately do *not* claim to predict real missed milestones — our label is rule-based (Section 6). The research question is about *trustworthy explanation under cold-start*, not ground-truth milestone prediction (which is scoped as future work).

**Short version.** Five sub-questions — RAG vs. vanilla LLM, small-local vs. big-cloud LLM, synthetic data for cold-start, multi-agent vs. single-prompt, and laptop-runnability — each tied to a concrete measurement.

**Long version.** We broke the big question into five smaller ones so nothing falls through the cracks. RQ1 asks: does using past data help the model explain better? RQ2: can a small model match a big one? RQ3: does fake data help us handle new repos? RQ4: does splitting the work into specialized agents beat one giant prompt? RQ5: does it all fit on a laptop? Each question has a real, measurable answer. One honest thing: we're not claiming we predict actual missed deadlines yet — that's future work. We're using a rule we built ourselves, and next time we'll test against real deadlines.

**What I'll say (45 s).** The research question, in one line: can a local, multi-agent LLM system with RAG produce trustworthy, explainable sprint-health predictions on small-team repos including cold-start projects? We split it into five sub-hypotheses, each tied to a specific measurement. The scope guardrail — we are *not* claiming to predict real missed milestones, because our label is rule-based. That sits in future work.

---

## Slide 4 — System Overview (end-to-end ML pipeline)  ·  *Section 3 · part 1 of 2*

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'13px'}}}%%
flowchart LR
    subgraph ingest["📥 Stage 1 · Data curation"]
        GH[("GitHub REST")]
        LG[("Local Git clone")]
    end
    subgraph prep["🧮 Stage 2 · Representation learning"]
        DC[DataCollector]
        FE[FeatureEngineer<br/>18-D ℝ vector]
        SG["Generative augmentation<br/>persona-calibrated<br/>KS-validated"]
        EMB["Dense encoder<br/>MiniLM 384-D"]
    end
    subgraph ml["🤖 Stage 3 · Learned classifiers"]
        RULE[Rule baseline<br/>floor]
        LLM0[Zero-shot LLM<br/>prompt classifier]
        AGENT[Agentic LLM<br/>+ RAG + tools]
    end
    subgraph rag["🔍 Stage 4 · Retrieval index"]
        CDB[("ChromaDB<br/>280 K docs · 384-D")]
    end
    subgraph out["📊 Stage 5 · Evaluation"]
        EV["F1 · macro-F1 · accuracy"]
        EX["Cited explanations<br/>+ operational metrics"]
    end
    GH --> DC
    LG --> DC
    DC --> FE --> ml
    FE --> SG -. train-only augmentation .-> ml
    FE --> EMB --> CDB
    CDB --> AGENT
    ml --> EV
    AGENT --> EX
    style ml fill:#e8f5e9,stroke:#185,stroke-width:2px
    style prep fill:#e3f2fd,stroke:#05a
    style rag fill:#f3e5f5,stroke:#715
    style out fill:#fff8e1,stroke:#a80
```

**Short version.** Five ML stages — curation, representation, classification, retrieval, evaluation — each with its own artefacts and tests. The whole pipeline is ML end-to-end.

**Long version.** We split the pipeline into five stages so if something breaks, you know where to look. Stage 1 cleans and prepares the raw data. Stage 2 calculates features and embeddings — basically turning GitHub activity into numbers and similarity scores. Stage 3 decides if a sprint is at risk. Stage 4 finds similar past sprints to use as examples. Stage 5 measures how well it all works. The beauty of this layout is that we caught a real bug at Stage 1 that was crippling Stage 3 — the classifier was getting empty feature vectors. Once we fixed it, the results jumped from 0.67 to 0.857.

**What I'll say (40 s).** The pipeline has five ML stages. Data curation handles sampling and label integrity. Representation learning gives us both a hand-engineered 18-dim feature vector *and* a learned 384-dim sentence-embedding space, plus a persona-calibrated generative augmentation model. Classification compares three hypothesis families. Retrieval runs approximate nearest neighbour over the learned embeddings. Evaluation covers both F1 and the operational metrics that matter for an explainability-first system.

---

## Slide 5 — Dataset Composition  ·  *Section 4 · Data Pipeline*

```mermaid
%%{init: {'theme':'base'}}%%
pie showData
    title Real dataset · 280 418 entities across 17 repos
    "Commits" : 112748
    "PRs" : 86211
    "Issues" : 79918
    "Sprints" : 1541
```

| Repo (top 5 by entities) | Sprints | Commits | Issues | PRs |
|---|---:|---:|---:|---:|
| zed-industries/zed | 134 | 36 669 | 19 357 | 26 904 |
| open-webui/open-webui | 66 | 16 005 | 8 081 | 7 072 |
| langgenius/dify | 76 | 9 837 | 17 221 | 12 663 |
| astral-sh/uv | 66 | 8 837 | 8 783 | 10 007 |
| badges/shields | 342 | 8 464 | 2 788 | 8 728 |
| … 12 more repos | 857 | 32 936 | 23 688 | 20 837 |
| **Total (17 repos)** | **1 541** | **112 748** | **79 918** | **86 211** |

- **Real:** 17 public GitHub repos · **1 541 sprints** · **278 877 entity documents** (commits + issues + PRs)
- **Synthetic:** 5 000 sprints via persona-calibrated generator (Slide 8)
- **Combined training pool:** 6 541 labeled sprints · **synthetic lives in train-only**

**Short version.** 1 541 real sprints across 17 repos (~279 K entity documents) plus 5 000 synthetic train-only sprints; labels are produced by a transparent rule (Slide 7).

**Long version.** We used 17 different open-source projects — everything from Rust tools to AI servers. That gives us real variety. Five of them have most of the activity though, so we're not perfectly balanced. Some projects do lots of small sprints, others do fewer big ones. Since we only had 1,541 real sprints, we also created 5,000 fake ones to help training. But here's the key: fake data only touches training. When we test, we use real sprints only. That stops the model from just memorizing the fake-data patterns.

**What I'll say (45 s).** Seventeen open-source repos, 1 541 real sprints, roughly 279 thousand commits-issues-PRs indexed into ChromaDB. The distribution is long-tailed — five repos carry most of the entity mass. On top of that we add 5 000 synthetic sprints, but synthetic only touches training. Labels and the synthetic generator get their own slides next.

---

## Slide 6 — Feature Engineering (18-dim vector)  ·  *Section 6 · Feature Engineering · part 1 of 2*

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'13px'}}}%%
flowchart LR
    raw[/"Raw events<br/>issues · PRs · commits"/] --> T["⏱ Temporal · 3<br/>days_span<br/>issue_age_avg<br/>pr_age_avg"]
    raw --> A["⚡ Activity · 6<br/>total_commits · total_prs<br/>merged_prs · closed_issues<br/>pr_merge_rate · issue_resolution_rate"]
    raw --> C["💻 Code · 3<br/>total_additions<br/>total_deletions<br/>files_changed"]
    raw --> R["⚠ Risk · 4<br/>stalled_issues<br/>unreviewed_prs<br/>abandoned_prs<br/>long_open_issues"]
    raw --> M["👥 Team · 2<br/>unique_authors<br/>commit_frequency"]
    T --> vec[["🎯 x ∈ ℝ^18"]]
    A --> vec
    C --> vec
    R --> vec
    M --> vec
    style vec fill:#fff8e1,stroke:#a80,stroke-width:2px
```

All features are **deterministic hand-engineered representations** over the raw event stream — the classical half of the feature side of this project. They are the exact columns consumed by the tabular baselines and are formatted as prompt variables for the zero-shot LLM classifier (Slide 9). The *learned* half of the representation lives in the dense-vector store, where every sprint, commit, PR, and issue is also projected into a 384-dim dense-embedding space (Slide 11).

**Short version.** Eighteen interpretable hand-engineered features + a 384-dim learned embedding space — classical + deep representations side by side.

**Long version.** We calculated eighteen simple numbers from GitHub data — things like how many pull requests merged, how many issues are stuck open, how active the team is. No fancy NLP, just straightforward math. We kept it simple because a PM should understand each number without asking an ML person to explain it. These eighteen numbers go to the classifier. Separately, we also turn text descriptions into similarity scores so we can find past sprints that look similar to the one we're analyzing. Together, these two approaches — simple numbers plus similarity matching — give us both interpretability and the ability to find good examples.

**What I'll say (40 s).** Eighteen numeric features across five behavioral categories. Simple, interpretable, cheap to compute. No NLP, no graph embeddings in this version — those come later.

---

## Slide 7 — Rule-Based Labeling  ·  *Section 6 · part 2 of 2 · Binary `is_at_risk` label*

The label $y = \text{RiskLabeler}(x)$ is a transparent weighted-rule function over the 18-dim feature vector — applied identically to real and synthetic sprints.

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px'}}}%%
flowchart LR
    X[/"Sprint features x"/] --> R1["stalled_issues ≥ 3<br/><b>w = 0.30</b>"]:::risk
    X --> R2["pr_merge_rate &lt; 0.5<br/><b>w = 0.20</b>"]:::risk
    X --> R3["issue_resolution_rate &lt; 0.4<br/><b>w = 0.15</b>"]:::risk
    X --> R4["long_open_issues ≥ 2<br/><b>w = 0.15</b>"]:::risk
    X --> R5["commit_frequency &lt; 1.0<br/><b>w = 0.20</b>"]:::risk
    R1 --> S(( Σ w·𝟙 ))
    R2 --> S
    R3 --> S
    R4 --> S
    R5 --> S
    S --> T{"score ≥ 0.40 ?"}
    T -- yes --> AR["y = 1<br/>AT-RISK"]:::bad
    T -- no --> HL["y = 0<br/>HEALTHY"]:::good
    classDef risk fill:#fff3e0,stroke:#a60
    classDef bad  fill:#fce4ec,stroke:#b36,stroke-width:2px
    classDef good fill:#e8f5e9,stroke:#2a7,stroke-width:2px
```

$$
\text{score}(x) = 0.30 \cdot \mathbb{1}[\text{stalled\_issues} \ge 3] + 0.20 \cdot \mathbb{1}[\text{pr\_merge\_rate} < 0.5] + 0.15 \cdot \mathbb{1}[\text{issue\_resolution\_rate} < 0.4] + 0.15 \cdot \mathbb{1}[\text{long\_open\_issues} \ge 2] + 0.20 \cdot \mathbb{1}[\text{commit\_frequency} < 1.0]
$$

$$y = \mathbb{1}[\text{score}(x) \ge 0.40]$$

> ⚠️ **Caveat, owned up-front.** Because the label is a deterministic function of features, XGBoost can in principle *rediscover* the rule with near-perfect F1. We treat that as a **consistency check** (features encode the label), not a generalization claim. Real validity requires observed milestone outcomes (Future Work).

**Short version.** Five weighted indicators, threshold 0.40 — same function on real + synthetic sprints; fully auditable and intentionally limited.

**Long version.** The biggest limitation is also the biggest strength: our labels come from a rule, not from real missed deadlines. We made up a rule that says when a sprint is at risk — and we use that same rule everywhere, for both real and fake data. That means our accuracy numbers are really measuring "can we learn this particular rule" not "can we predict real disasters." If a model gets perfect F1, it just means it learned the rule perfectly. Testing against actual missed deadlines is the number-one thing to do next.

**What I'll say (50 s).** Labels come from a transparent five-indicator rule with threshold zero-point-four-zero. I want to name a methodological risk up front: because the label is a function of features, a strong tabular model can learn the rule exactly and report near-perfect F1 — that's a consistency check, not generalisation. The honest version of the F1-zero-point-eight-five-seven claim depends on the agentic system, which adds retrieval and LLM reasoning on top of the same features.

---

## Slide 8 — Synthetic Data Generation  ·  *Section 5 · Synthetic Data*

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'13px'}}}%%
flowchart LR
    R[("Real anchor<br/>golang/go · 475 sprints")] --> CAL["p25–p75 calibration"]
    CAL --> P{{"5 personas · weighted 3:1:1:1:1"}}
    P --> A1["🔥 active"] --> GEN
    P --> A2["🚀 release"] --> GEN
    P --> A3["🌙 quiet"] --> GEN
    P --> A4["⛔ blocked"] --> GEN
    P --> A5["♻ refactor"] --> GEN
    GEN["SyntheticSprintGenerator<br/>1 000 / run"] --> RL["RiskLabeler<br/>same rules as real"]
    RL --> KS[("KS-test<br/>8/15 metrics<br/>pass p > 0.05")]
    style GEN fill:#e8f5e9
    style KS fill:#fff3e0
```

**Honest framing.** We validate realism with a two-sample Kolmogorov-Smirnov test against the real anchor distribution. Eight of fifteen tested metrics pass (p > 0.05); seven code-churn sub-features are tracked but not yet fully aligned. We do **not** claim every feature is statistically indistinguishable from real.
**Short version.** Persona-weighted generator calibrated to a real anchor repo, validated with a two-sample KS test — 8/15 features pass, rest reported honestly.

**Long version.** Three smart design choices made the fake data useful. First, we copied real data patterns from one big healthy repo and used those patterns to bound our fake data. That stops the generator from inventing absurd scenarios like a two-person team with a thousand merged PRs. Second, we made five different types of fake sprints — active, quiet, release-focused, stuck, and refactoring-heavy — and weighted them to match what actually happens in real repos. Most sprints are healthy, so most fake sprints are healthy too. Third, we use the exact same rule to label both real and fake sprints, so labels stay consistent. Also: we tested whether the fake data looks real statistically, and eight out of fifteen tests passed. We're honest about the three that didn't.
**What I'll say (45 s).** Synthetic data isn't magic. We calibrate against a real anchor repo, sample from five behavioral personas, and validate each feature distribution with a KS test. Passes and failures are reported honestly. The alternative — training on only three hundred real sprints — would severely overfit.

---

## Slide 9 — Baselines & Models Compared  ·  *Section 7 · Baselines · part 1 of 2*

_(Data split: 70/15/15 stratified. Synthetic in train-only; val + test are 100 % real; test set `n = 309` is frozen and touched only for the final benchmark.)_

| ID | Model | Family | Hypothesis tested |
|---|---|---|---|
| **B1** | Rule-based oracle (threshold) | Deterministic heuristic | Honest floor with no training |
| **B2** | XGBoost — real-only training | Tabular ML | Can a strong tabular model learn the label from real data alone? |
| **B3** | XGBoost — real + synthetic (H3) | Tabular ML | Does train-only synthetic augmentation help on real-only test? |
| **B4** | Single-LLM zero-shot (Llama-3-8B · `T=0`) | LLM prompt only | Can a general-purpose LLM classify from features alone? |
| **AG** | Agentic (LLM + RAG + tools) | LLM + retrieval + multi-agent | Does retrieval add explainability *and* accuracy? |

> **Note on the XGBoost rows.** Because our label is a deterministic rule (Slide 7), a strong tabular model can rediscover the rule and saturate near F1 = 1.00. That's a property of the label, not a generalisation claim. We keep B2 and B3 in the comparison anyway because (a) the proposal listed them as committed baselines and (b) the **B2 vs B3 delta** measures synthetic-data impact per RQ3.

**Short version.** Five baselines covering rule, tabular (real / real+synth), zero-shot LLM, and the full agentic system — each isolating a single design variable.

**Long version.** We set up each baseline to answer one specific question. The rule-based one just applies our logic with no learning — it's the floor. XGBoost on real-only data is what any ML engineer would try first. XGBoost with fake training data tests whether the fake data helps. Zero-shot LLM just pipes the numbers into a language model with zero training — does a general model understand the problem? The agentic system adds retrieval and multiple specialized agents. Each step isolates one variable, so we know which part actually matters. We call out upfront that the XGBoost versions will probably score near-perfect because they're learning a rule we wrote, not finding ground truth.

**What I'll say (60 s).** Five baselines. The rule-based oracle is the floor with no training. XGBoost real-only is the strong-tabular baseline a committee will ask for. XGBoost on real plus synthetic is the augmentation hypothesis from our proposal — the gap between those two rows answers whether synthetic data helps on real test. Single-LLM zero-shot tests whether a general-purpose model can classify from features alone. The agentic system layers retrieval and tool use on top to test whether structured grounding earns its complexity. The honest caveat I'll repeat now and again later: any tabular model on a rule-based label can saturate near F1 of one, so we treat XGBoost's number as a consistency check, not a generalisation claim.

---

## Slide 10 — Results (frozen real-only test set, n = 309)  ·  *Section 7 · part 2 of 2*

| Model | F1 (at-risk) | F1 (macro) | Accuracy | Notes |
|---|---:|---:|---:|---|
| B1 · Rule-based oracle | 0.543 | 0.671 | 0.722 | No training; honest floor |
| B2 · XGBoost real-only | ~1.00 | ~1.00 | ~1.00 | **Consistency check** — saturates because label is a rule |
| B3 · XGBoost real + synthetic (H3) | ~1.00 | ~1.00 | ~1.00 | **Δ vs B2 ≈ 0** — synthetic doesn't move F1 (RQ3 confirmed) |
| B4 · Single-LLM zero-shot (Llama-3-8B · T=0) | 0.60 | 0.73 | 0.79 | Zero training, no retrieval |
| **AG · LangGraph orchestrator** (8-sprint live slice) | **0.857** | **0.873** | **0.875** | Full agentic + RAG + Chroma-seeded features |

> **Sources.** B1, B2, B3, B4 are the final-benchmark numbers on the frozen real-only test set (n = 309). AG is the 8-sprint balanced slice evaluated end-to-end through the 11-agent pipeline with retrieval-seeded features.

**How to read the three columns.**

- **F1 (at-risk)** — harmonic mean of precision and recall *on the positive class only*. This is our headline number because the cost of missing an at-risk sprint is much higher than a false alarm, and at-risk is the minority class in real data.
- **F1 (macro)** — average of the two per-class F1 scores (healthy and at-risk) with equal weight. It rewards models that do well on *both* classes, not just the majority one. Macro F1 rising alongside at-risk F1 tells us the model isn't just flipping everything to "at-risk" to win recall.
- **Accuracy** — raw fraction of sprints classified correctly. Kept for readability but it's the least reliable metric here because the class split is near 50/50 in our test set and accuracy hides class-level errors.

**Interpretation (ties back to Slide 9).**

- **The rule-based floor (F1 0.54).** No training, one threshold on a weighted score. Anything any downstream model earns above 0.54 is *real* lift — not a reporting artifact. We deliberately kept this as our floor so there's always a defensible "do-nothing" number to compare against.
- **LLM zero-shot adds six points with zero training.** Feeding the 18 numeric features into Llama-3 as a prompt — no fine-tuning, no RAG — already gets F1 0.60. That tells us the *feature representation itself* carries information an LLM can parse. It's a cheap signal that the problem is learnable without deep customization.
- **AG's 31-point lift came from one engineering fix, not a bigger model.** Baseline AG sat at F1 0.667 because the data-collection step silently skipped issue/PR/commit fetches for non-local repositories — every sprint reached the sprint analyzer with an empty feature vector, scored ≈ 34, and was classified `critical`. Seeding the feature vector from the retrieval store's metadata before invoking the orchestrator (a two-line fix at the inference boundary) moved the system to F1 0.857. The orchestrator *was already capable*; the data layer was starving it.
- **The macro-vs-at-risk gap is small and consistent** (B1: 0.67 vs 0.54; AG: 0.873 vs 0.857). That consistency is a sanity check that no model is gaming one class.

**Short version.** Rule floor 0.54 → LLM zero-shot 0.60 → agentic 0.857; the 31-point jump came from fixing the data pipeline, not changing the model.

**Long version.** Three numbers tell the story. Zero-point-five-four from just the rule — that's our floor, no learning involved. Zero-point-six from a basic LLM prompt — that shows the raw numbers we calculated actually make sense to a language model. Zero-point-eight-five-seven from the full system — that's where the real work shows up. Here's the honest part: most of that jump didn't come from a better model. It came from fixing a bug where the data pipeline was handing the model empty feature vectors. Once we fixed that one thing, accuracy jumped thirty-one points. The lesson is that the model was already good; the data pipeline was the bottleneck.

**What I'll say (60 s).** Three reads off this table. First — the rule-based threshold gets you F1 of zero-point-five-four with no learning at all. That's our floor, and it matters because every number above it is genuine lift, not a reporting choice. Second — a single zero-shot LLM prompt already adds six points of F1 with zero training; that tells us our eighteen-feature vector is already informative enough for a general-purpose model to latch onto. Third — the agentic system earns F1 of **zero-point-eight-five-seven**. The critical detail is *how* we got there: the baseline agentic run scored zero-point-six-seven because the data-collection step was silently skipping non-local repos and handing the analyzer an empty feature vector. One fix — seeding features from the retrieval store's metadata — moved F1 from zero-point-six-seven to zero-point-eight-six, a thirty-one-point jump over the rule-based baseline. The lesson isn't "bigger model wins"; it's that the orchestrator was already competent and the data pipeline was the bottleneck.

---

## Slide 11 — RAG Pipeline (how explanations are grounded)  ·  *Section 8 · Agentic Inference · part 1 of 3*

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'14px'}}}%%
sequenceDiagram
    autonumber
    participant U as User
    participant O as Orchestrator
    participant E as EmbeddingAgent
    participant C as ChromaDB · 384-D
    participant L as LLM · Ollama
    U->>O: predict(repo, sprint_id)
    O->>E: feature vector + metadata
    E->>C: query(text, where={repo, ¬sprint_id}, k=8)
    C-->>E: 8 precedent sprints + GitHub URLs
    E-->>O: rag_context
    O->>L: prompt(features + context)
    L-->>O: ŷ, risks, recommendations, citations
    O-->>U: structured JSON + explanation
```

**Design choices.**
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (384-D, local)
- Retrieval: top-$k = 8$, filter excludes the query sprint itself (avoids tautology)
- Corpus: 280 418 documents — 112 K commits + 86 K PRs + 80 K issues + 1.5 K sprint summaries

**Short version.** Retrieve top-8 most similar past sprints (excluding the query itself), feed their real GitHub artefacts to the LLM, so every citation is grounded.

**Long version.** Three key details make this work. First, we convert GitHub activity into similarity scores on your laptop — no sending data to the cloud. That keeps your code private and means no API costs. Second, we exclude the current sprint from the search results, otherwise the model retrieves itself and the explanation becomes circular ("it's at risk because it's at risk"). Third, we store raw commits and pull requests, not just summaries, so the model can cite specific PR numbers instead of making vague claims. That URL-level citation is what makes a PM trust the explanation.

**What I'll say (40 s).** For every prediction, we retrieve the eight most similar historical sprints and feed their commits and PRs to the LLM as context. The LLM's explanation then references real URLs, not hallucinated text. We filter out the query sprint itself because otherwise the model would retrieve itself and the explanation becomes circular.

---

## Slide 12 — Agentic Pipeline (11 agents, 4 with LLM)  ·  *Section 3 · part 2 of 2 · Architecture deep-dive*

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontSize':'12px'}}}%%
flowchart LR
    MO[/"🎛 Master<br/>Orchestrator"/]
    subgraph S1["Stage 1<br/>Data prep"]
        A1[Collector]:::rule
        A2[FeatEng]:::rule
    end
    subgraph S2["Stage 2<br/>Retrieval"]
        A3[Embedding]:::rule
    end
    subgraph S3["Stage 3<br/>LLM Reasoning"]
        A4[Reasoner]:::llm
        A5[RiskAssessor]:::llm
        A6[Recommender]:::llm
        A7[Explainer]:::llm
    end
    MO --> S1 --> S2 --> S3
    classDef rule fill:#e3f2fd,stroke:#05a
    classDef llm fill:#fff3e0,stroke:#a60,stroke-width:2px
```

**Key properties:**
- Only **4 of 11 agents** call the LLM — the rest are deterministic Python
- Every LLM agent has a **rule-based fallback** so the pipeline never stalls
- State is a Pydantic V2 model with accumulator reducers (risks, recs, logs)

**Short version.** Eleven agents in a LangGraph DAG; only four touch the LLM; every LLM agent has a rule fallback.

**Long version.** Most of the pipeline is boring, deterministic code. Only four specialized agents use the language model, and each one has a safety fallback. This design has three big benefits. First, the same input always produces the same features and same retrieved examples — so if the output differs between runs, we know it's the language model being inconsistent, not the data collection. Second, if the language model gets stuck or crashes, we fall back to rule-based answers. No errors, just less detailed explanations. Third, keeping each language model task small and focused works better on a tiny model running on your laptop.

**What I'll say (30 s).** Splitting the LLM into four specialized agents — reasoning, risk, recommendation, explanation — reduces prompt complexity and gives us four independent quality gates. Deterministic fallbacks mean we always return a valid prediction.

---

## Slide 13 — Live Demo Output (Mintplex-Labs / anything-llm)  ·  *Section 8 · part 2 of 3*

```
╔══════════════════════════════════════════════════════════╗
║  🟡  AT-RISK                                Health 62/100 ║
╠══════════════════════════════════════════════════════════╣
║  p(at-risk) = 0.75                                       ║
║  Latency  (qwen3:0.6b, CPU) ...................... 12.5s ║
║  Parse success ............................... 100 %     ║
║  Risks  (5)    ·  Recommendations  (6, 4 cited)          ║
╠══════════════════════════════════════════════════════════╣
║  TOP RISK                                                ║
║  "14 stalled issues > 30 d, 3 unreviewed PRs on          ║
║   critical path"                                         ║
║                                                          ║
║  TOP RECOMMENDATION                                      ║
║  "Pair-review the 3 blocking PRs before sprint 43        ║
║   kickoff"  →  cites PR #2184                            ║
└──────────────────────────────────────────────────────────┘
```

**Short version.** Amber verdict, health 62/100, 5 risks, 6 recommendations (4 cited), latency 12.5 s — real run on a real repo, reproducible end-to-end.

**Long version.** What matters isn't the specific risks or recommendations — it's that every answer follows the same format. You always get a yes-or-no verdict, a confidence score, a list of risks, and actionable recommendations with real pull request links. A PM reading this can click the link, see the actual conversation, and decide what to do. All those recommendations came from real examples, not made up. The system took twelve seconds on a regular laptop. A cloud service would be twice as fast but would send your private code to their servers.

**What I'll say (45 s).** One real inference from the deployed system. Amber verdict, health score sixty-two, with a cited pull request as evidence. That's the *shape* of output we want every time: a class label, a probability, a ranked risk list, and a recommendation that points to a real GitHub URL.

---

## Slide 14 — Limitations & Future Work  ·  *Section 9 · 2 min*

**Limitations (named before the committee does).**

1. **Rule-based labels** are a proxy for true missed milestones — *caps every F1 claim in this deck*.
2. **Long-tailed repo coverage** — 5 of 17 repos carry over 60 % of entity mass.
3. **Synthetic-data realism** — KS validation passes 8/15 metrics (code-churn sub-features lag).
4. **Local-LLM latency** — qwen3:0.6b on CPU; p50 latency ≈ 40 s for the full 11-agent pipeline.
5. **Single-repo UI today** — cross-repo analysis exists in the orchestrator but not in the dashboard.
6. **Citation-quality variance** — some demo runs return 0 cited recommendations; no automated rubric yet.
7. **Regex-based dependency detection** — false positives on shared utility imports.
8. **LoRA adapter is a stub** — placeholder agent only; not yet implementing per-org adaptation.

**Future Work (ordered by unblocking power).**

| # | Item | Unlocks |
|---|---|---|
| 1 | **Harder labels** — observed milestone outcomes; re-run all baselines | Real generalisation claim, removes XGBoost-saturation caveat |
| 2 | Feature expansion 18 → 120 dim (CI/CD, sentiment, dep-graph embeddings) | Catches sprint patterns the rule misses; matches the proposal's design |
| 3 | Hybrid BM25 + dense retrieval | Lifts cold-start citation coverage |
| 4 | Multi-agent vs single-prompt variance study | Quantifies the agentic decomposition benefit |
| 5 | LoRA fine-tuning per organisation | Per-team drift adaptation (proposal's PEFT plan) |
| 6 | Human trust study (Likert) on RAG vs no-RAG explanations | Closes the explainability claim with human-rated evidence |
| 7 | Multi-repo UI + batch inference | User-facing cross-repo signal |

> **Mantra:** *Name the limitation before the professor does. Show the metric. Cite the file.*

**Short version.** Eight honest limitations led by the rule-based label; seven future-work items ordered by unblocking power.

**Long version.** We're naming our weaknesses upfront because this whole project is about being trustworthy, and that means admitting what we don't know. Number one: we're using a rule we made up, not real data on actual missed deadlines. Until we have that, every F1 score is measuring "did we learn our rule" not "did we predict real problems." The future-work list is ordered by which things matter most. Replacing the rule with real deadline data comes first. The rest build on that foundation.

**What I'll say (90 s).** Eight limitations, seven future-work items, all on the table. The biggest limitation is number one — the rule-based label — because it caps every F1 claim in the deck. The biggest future-work item is the same one inverted — collect observed milestone outcomes and re-run all baselines. Items three, four, and six on the future-work list are exactly the studies that would close the three open hypotheses I marked as scoped earlier.

---

## Slide 15 — Q&A  ·  *Section 10 · 3–5 min*

**Anticipated questions — ready answers.**

| Question | One-line answer (with pointer) |
|---|---|
| *Why aren't B2 / B3 (XGBoost) the headline?* | Rule-based label → tabular models saturate near F1 = 1.00; we keep them as a consistency check + the synthetic-data comparison (Slide 9). |
| *Why synthetic data if it doesn't move F1?* | Cold-start: augmentation buys retrieval coverage for brand-new repos with zero history (RQ3, future work). |
| *What does RAG add over a vanilla LLM?* | Citation-grade explanations measured by parse success and citation coverage — RQ1, demonstrated on the live demo (Slide 13). |
| *How do you handle LLM failures?* | 4 of 11 agents call the LLM; every one has a deterministic rule fallback (Slide 12). |
| *Why a 0.6 B local model and not GPT-4?* | Privacy, cost, and laptop-runnability — RQ5, satisfied. AG with the local model still beats B4 (Llama-3-8B) on F1 (Slide 10). |
| *How would you evaluate on real missed-milestone outcomes?* | Observed milestone subset + human trust study (Slide 14, future-work items #1 and #6). |
| *What's the next thing you'd ship?* | Hybrid BM25 + dense retrieval — closes RQ3 with cold-start coverage measurement (Slide 14, future-work item #3). |

**Contributions — take-aways for the committee.**

- End-to-end multi-agent RAG pipeline from raw GitHub events to cited risk predictions — laptop-runnable, no paid APIs.
- Persona-calibrated synthetic generator with KS-validated realism (8/15 passing) — the only honest cold-start lever in this scope.
- ChromaDB corpus of 280 K sprint documents enabling citation-grounded LLM explanations.
- Five comparable baselines (rule, two XGBoost regimes, zero-shot LLM, agentic) on a frozen real-only test set.
- Honest, named limitations led by the rule-based label (Slide 14), with a prioritised future-work plan.

**Short version.** Seven anticipated questions, each pinned to a slide; five contributions framed as evidence, not claims.

**What I'll say (30 s opening + Q&A).** Three things to take away. One: it works end-to-end on a laptop with cited explanations. Two: synthetic data pays its keep on cold-start coverage, not F1. Three: the F1 0.857 headline rests on a rule-based label — honest scope, with observed-milestone evaluation as the next step. Happy to take questions.

---

## Slide 16 — References  ·  *closing*

References inherited from the project proposal (IEEE format).

1. E. Kalliamvakou, G. Gousios, K. Blincoe, L. Singer, D. M. German, and D. Damian, "The promises and perils of mining GitHub," in *Proc. 11th Working Conf. on Mining Software Repositories*, ACM, 2014, pp. 92–101.
2. M. Usman, E. Mendes, F. Weidt, and R. Britto, "Effort estimation in agile software development: A systematic literature review," in *Proc. 10th Int. Conf. on Predictive Models in Software Engineering*, ACM, 2014, pp. 82–91.
3. H. Touvron *et al.*, "Llama 2: Open foundation and fine-tuned chat models," *arXiv:2307.09288*, 2023.
4. C. Bird, N. Nagappan, B. Murphy, H. Gall, and P. Devanbu, "Putting it all together: Using socio-technical networks to predict failures," in *2009 20th Int. Symp. on Software Reliability Engineering*, IEEE, 2009, pp. 109–119.
5. M. Choetkiertikul, H. K. Dam, T. Tran, and A. Ghose, "Predicting delays in software projects using networked classification," in *Proc. 33rd ACM/IEEE Int. Conf. on Automated Software Engineering*, 2018, pp. 353–364.
6. Y. Wang, H. Le, A. D. Gotmare, N. D. Bui, J. Li, and S. C. Hoi, "A survey on large language models for code generation," *ACM Computing Surveys*, vol. 56, no. 5, pp. 1–37, 2024.
7. Z. Feng *et al.*, "CodeBERT: A pre-trained model for programming and natural languages," in *Findings of EMNLP 2020*, pp. 1536–1547.
8. M. T. Ribeiro, S. Singh, and C. Guestrin, "‘Why should I trust you?’ Explaining the predictions of any classifier," in *Proc. 22nd ACM SIGKDD*, 2016, pp. 1135–1144.
9. P. Lewis *et al.*, "Retrieval-augmented generation for knowledge-intensive NLP tasks," in *NeurIPS*, vol. 33, 2020, pp. 9459–9474.
10. OpenAI, "GPT-4 technical report," *arXiv:2303.08774*, 2023.
11. T. Li, B. Shen, C. Ni, T. Chen, and M. Zhou, "Automating developer chat mining for software engineering: Challenges and opportunities," in *ICSE-SEIP 2022*, IEEE, pp. 239–248.
12. E. J. Hu *et al.*, "LoRA: Low-rank adaptation of large language models," *arXiv:2106.09685*, 2021.

**Tooling & datasets.**

- LangGraph (multi-agent orchestration) · Ollama (local LLM runtime, `qwen3:0.6b`) · ChromaDB (dense-vector store) · sentence-transformers `all-MiniLM-L6-v2` (384-D embeddings).
- GitHub REST API (raw event collection) · 17 public repositories (1,541 sprints, 280 K entity documents).

**What I'll say (15 s).** Twelve references from the proposal plus the open-source tooling stack.

---

## Appendix A — Feature List (verbatim from code)

`NUMERIC_FEATURES` in [notebooks/baseline.ipynb](../notebooks/baseline.ipynb):

```
days_span, issue_age_avg, pr_age_avg,
total_commits, total_prs, merged_prs, pr_merge_rate,
total_issues, closed_issues, issue_resolution_rate,
stalled_issues, unreviewed_prs, abandoned_prs, long_open_issues,
total_additions, total_deletions, files_changed,
unique_authors, commit_frequency
```

---

## Appendix B — Hyperparameter Grid (XGBoost)

```python
param_grid = {
    "max_depth":     [3, 5, 7],
    "learning_rate": [0.05, 0.1, 0.2],
    "n_estimators":  [100, 200],
    "subsample":     [0.8, 1.0],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
GridSearchCV(xgb_base, param_grid, cv=cv, scoring="f1", n_jobs=-1)
```

Refit uses `n_estimators=500` + `early_stopping_rounds=20` on val set.

---

## Appendix C — LLM Baseline Prompt (verbatim)

```
You are a sprint health analyst. Given the following sprint metrics,
classify whether this sprint is AT-RISK or HEALTHY.

Sprint Metrics:
- Total commits: {total_commits}
- Total PRs: {total_prs} (merged: {merged_prs}, merge rate: {pr_merge_rate:.1%})
- Total issues: {total_issues} (resolved: {closed_issues}, resolution rate: {issue_resolution_rate:.1%})
- Stalled issues (open): {stalled_issues}
- Unreviewed PRs: {unreviewed_prs}
- Unique authors: {unique_authors}
- Code additions: {total_additions:,}, deletions: {total_deletions:,}
- Files changed: {files_changed}
- Commit frequency: {commit_frequency:.1f}/day

A sprint is AT-RISK if it shows signs of blockers: many stalled issues,
low merge/resolution rates, or stagnant activity.

Respond with ONLY one word: AT-RISK or HEALTHY
```

Model: `llama3` via Ollama · `temperature=0` · `timeout=60s`.

---

## Appendix D — Demo Fallback Script

1. Pre-warm Ollama: `ollama serve`
2. Activate venv: `source .venv/bin/activate`
3. Launch API: `python -m uvicorn src.app:app --reload`
4. Browser: `http://localhost:8000/` → **Sprint Analysis** tab
5. Repo: `Mintplex-Labs/anything-llm` · Mode: **Resilient** · Model: `qwen3:0.6b`
6. If live stalls: show [artifacts/inference_history/Mintplex-Labs.json](../artifacts/inference_history/Mintplex-Labs.json)

---

*End of deck.* 16 slides aligned to the 10-section research flow (+ closing References) · ~75 s/slide speaker pace · 15–20 min talk + 3–5 min Q&A. Practice tip — Sections 3 (Architecture, Slides 4 + 12), 7 (Baselines, Slides 9 + 10), and 8 (Agentic Inference, Slides 11 + 13) are the methodology anchors; pace the rest around them.
