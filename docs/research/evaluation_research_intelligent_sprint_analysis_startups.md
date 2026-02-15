# Comprehensive Evaluation Research
# Intelligent Sprint Analysis Using Agentic System for Startup Projects

**Version**: 2.0.0  
**Last Updated**: February 15, 2026  
**Scope**: Small startup teams managing 2-3 GitHub repositories  
**Research Context**: Machine Learning Course Project / Thesis Research  
**Publication Target**: Top-tier ML/SE conference (NeurIPS, ICML, ICSE, FSE)

---

## Executive Summary

This document presents a comprehensive evaluation framework for an LLM-based multi-agent system designed to provide intelligent sprint analysis for small startup teams. The evaluation addresses the unique challenges of startup environments: minimal historical data, resource constraints, and the need for instant deployment without dedicated project managers.

**Key Innovation**: First rigorous evaluation of a locally-deployable, multi-modal LLM agentic system for cross-repository sprint intelligence with synthetic data bootstrapping.

**Evaluation Scope**:
- 500 startup organizations (1,500 repositories total)
- 9,000 sprint instances across 3 years
- 5,000 synthetic scenarios for edge cases
- Human evaluation with 10 startup practitioners
- Local deployment on 16GB RAM laptops

---

## 1. Research Goal & Context

### 1.1 Primary Research Goal

Design and execute a rigorous, multi-faceted evaluation of an agentic sprint-analysis system that provides accurate, explainable, and actionable recommendations for startup teams operating under three critical constraints:
1. **Limited data**: New startups with <3 months history
2. **Limited compute**: Local deployment on developer laptops (16GB RAM)
3. **No dedicated PM**: Tech leads managing 2-3 repos simultaneously

### 1.2 Research Questions

**RQ1: Predictive Performance**  
Can an LLM-based multi-agent system improve sprint outcome prediction and blocker detection for startup projects compared to rule-based and single-model baselines?

**RQ2: Explainability & Trust**  
Do RAG-backed explanations with evidence attribution increase stakeholder trust and recommendation acceptance compared to black-box predictions?

**RQ3: Cold-Start Performance**  
Can synthetic data augmentation enable effective deployment in new organizations without requiring 6-12 months of historical data?

**RQ4: Operational Feasibility**  
Is local deployment on developer-grade hardware (16GB RAM laptops) practically viable for real-time sprint analysis?

**RQ5: Cross-Repository Intelligence**  
Does multi-repository analysis detect dependency-related blockers more effectively than single-repo tools?

**RQ5: Cross-Repository Intelligence**  
Does multi-repository analysis detect dependency-related blockers more effectively than single-repo tools?

### 1.3 Contribution to Field

This evaluation advances the state of the art in:
1. **Multi-modal LLM evaluation**: First comprehensive assessment of code + text + temporal + graph + sentiment fusion for sprint analysis
2. **Cold-start learning**: Novel synthetic data generation and evaluation protocols for organizations without historical data
3. **Explainable AI metrics**: New frameworks for measuring trust and actionability in technical project management
4. **Resource-constrained deployment**: Benchmarking protocols for local ML systems on consumer hardware

---

## 2. Hypotheses & Expected Outcomes

---

## 2. Hypotheses & Expected Outcomes

### H1: Agentic Multi-Modal Superiority
**Hypothesis**: The full agentic system integrating code, text, temporal, graph, sentiment, and CI/CD signals will outperform single-modality and non-agentic baselines on sprint success prediction.

**Prediction**: 
- Full system F1 ≥ 0.85 on sprint outcome classification
- Improvement over rule-based baseline: +0.15 F1
- Improvement over single-LLM baseline: +0.08 F1
- Improvement over no-graph variant: +0.06 F1

**Rationale**: Multi-modal fusion captures complementary signals (code velocity, communication sentiment, dependency risks) that single sources miss.

### H2: RAG-Enhanced Explainability Increases Trust
**Hypothesis**: RAG-backed explanations with evidence attribution will yield significantly higher user trust and recommendation acceptance than black-box predictions.

**Prediction**:
- Trust score (1-5 Likert): 4.2 with RAG vs. 2.8 without RAG
- Recommendation acceptance rate: 78% with RAG vs. 52% without RAG
- Evidence correctness score: ≥ 0.85 (fraction of citations supporting claims)

**Rationale**: Startup practitioners require transparency to trust AI recommendations; cited evidence enables verification and builds confidence.

### H3: Synthetic Data Reduces Cold-Start Penalty
**Hypothesis**: Training with synthetic-augmented data (70% synthetic, 30% real) will maintain performance within 5% relative F1 of real-only models in low-data scenarios.

**Prediction**:
- With 3 months history: Synthetic-augmented F1 ≥ 0.80, Real-only F1 ≈ 0.70
- With 1 week history: Synthetic-augmented F1 ≥ 0.68, Real-only F1 ≈ 0.52
- Zero-shot (synthetic only): F1 ≥ 0.62

**Rationale**: LLM-generated scenarios provide diverse training signal for rare but critical failure modes.

### H4: Local Deployment Meets Startup Constraints
**Hypothesis**: The system will meet operational requirements for startup deployment on standard developer laptops.

**Prediction**:
- p95 analysis latency: ≤ 60 seconds
- Peak RAM usage: ≤ 14GB (fits 16GB laptops)  
- Throughput: ≥ 10 sprint analyses/hour
- Cloud cost: $0 (fully local)

**Rationale**: Quantized models (Llama-3-8B Q4) and efficient vector stores (ChromaDB) enable local execution.

### H5: Cross-Repo Analysis Detects Dependency Blockers
**Hypothesis**: Multi-repository dependency tracking will detect dependency-induced delays missed by single-repo analysis.

**Prediction**:
- Cross-repo blocker detection F1: ≥ 0.82
- Improvement over single-repo analysis: +0.18 F1
- Dependency graph coverage: ≥ 90% of actual dependencies

**Rationale**: 34% of startup delays stem from cross-repo dependencies; explicit dependency modeling addresses this gap.

---

## 3. Evaluation Objectives

### 3.1 Predictive Quality Assessment
**Goal**: Quantify accuracy and calibration of sprint outcome predictions and blocker detection.

**Objectives**:
1. Measure predictive quality for sprint outcomes and blocker detection
2. Compare against multiple baselines (rule-based, classical ML, single LLM)
3. Evaluate early warning capability (lead time before sprint failure)
4. Assess calibration quality (prediction confidence vs. actual outcomes)

### 3.2 Recommendation Usefulness Evaluation
**Goal**: Quantify the practical value of generated recommendations for startup practitioners.

**Objectives**:
1. Measure recommendation relevance and ranking quality
2. Quantify recommendation acceptance rates in human study
3. Assess explanation quality through trust scoring and evidence validation
4. Compare recommendation diversity and coverage across sprint contexts

### 3.3 Operational Feasibility Validation
**Goal**: Demonstrate practical deployability on startup-grade hardware without cloud costs.

**Objectives**:
1. Measure latency distribution under realistic workloads
2. Profile memory and compute resource usage
3. Assess system stability and failure recovery
4. Validate zero-cost local deployment model

### 3.4 Cold-Start Performance Analysis
**Goal**: Validate effectiveness in organizations with minimal historical data.

**Objectives**:
1. Evaluate robustness under limited history scenarios (1 week, 1 month, 3 months)
2. Test cross-organization transfer learning capability
3. Assess synthetic data quality and augmentation benefit
4. Measure time-to-value for new deployments

### 3.5 Practical Impact Measurement
**Goal**: Quantify business-relevant impact on sprint execution decisions.

**Objectives**:
1. Estimate reduction in manual sprint tracking effort
2. Measure improvement in blocker response time
3. Assess decision confidence improvement for tech leads
4. Quantify perceived productivity gains

---

## 4. Comprehensive Evaluation Design

### 4.1 Evaluation Tracks

The evaluation follows a four-track approach combining automated and human assessment:

#### Track 1: Offline Predictive Evaluation
**Purpose**: Measure core predictive accuracy on historical data  
**Data**: 9,000 labeled sprint instances (500 startups, 3 years)  
**Methods**: Classification metrics, calibration analysis, stratified performance  
**Timeline**: Week 2-3 of evaluation period

**Key Analyses**:
- Sprint outcome classification (success/delayed/failed)
- Blocker presence detection (binary + multi-class type)
- Milestone completion prediction
- Early warning analysis (prediction at 25%, 50%, 75% sprint progress)
- Performance stratification by team size, sprint length, repository count

#### Track 2: Online Simulation Evaluation
**Purpose**: Test real-time performance on event streams  
**Data**: GitHub event stream replay from 50 startups  
**Methods**: Sequential evaluation, latency measurement, incremental learning  
**Timeline**: Week 4 of evaluation period

**Key Analyses**:
- Real-time blocker detection accuracy
- End-to-end latency distribution
- Recommendation freshness and relevance
- System stability under continuous operation
- Incremental learning effectiveness

#### Track 3: Human-Centered Evaluation
**Purpose**: Assess practical value and trust from practitioner perspective  
**Participants**: 10 startup practitioners (tech leads, engineering managers, founders)  
**Methods**: Comparative rating study, think-aloud protocol, semi-structured interviews  
**Timeline**: Week 5 of evaluation period

**Key Analyses**:
- Recommendation relevance ratings (Likert 1-5)
- Explanation quality and trust scores
- Recommendation acceptance decisions
- Qualitative feedback on failure modes and adoption barriers
- Decision confidence before/after AI support

#### Track 4: Systems Evaluation
**Purpose**: Validate operational feasibility on target hardware  
**Setup**: MacBook M4 Pro (24GB RAM), Ubuntu laptop (16GB RAM)  
**Methods**: Performance profiling, stress testing, resource monitoring  
**Timeline**: Week 4 of evaluation period

**Key Analyses**:
- Latency percentiles (p50, p95, p99)
- Memory usage (peak and sustained)
- CPU utilization and thermal behavior
- Throughput under concurrent requests
- Failure modes and recovery time

### 4.2 Dataset Construction

#### 4.2.1 Real Startup Data Collection

**Source 1: GitHub Archive**
- Period: March 2023 - February 2026 (3 years)
- Selection criteria: 
  - Organizations with 2-3 core repositories
  - Active development (≥20 events/week)
  - Visible milestone/project management
  - Team size indicators: 3-10 unique contributors
- Total: 500 qualifying startup organizations

**Source 2: Direct GitHub API**
- Target: 20 consenting open-source startups
- Access: Full repository history, API rate limits (5000 req/hr)
- Benefits: Complete data, ground truth labels from actual outcomes

**Data Collection Scope per Organization**:
```yaml
Repository Data:
  - Commits: code diffs, messages, authors, timestamps
  - Issues: titles, descriptions, labels, comments, state transitions
  - Pull Requests: reviews, approvals, merge times, CI results
  - Discussions: team communication, decision rationale
  - Actions: CI/CD pipeline runs, test results, deployment events
  - Dependencies: package.json, requirements.txt, dependencies between repos
  - Documentation: README, wikis, inline comments

Temporal Scope:
  - Full history for analysis
  - Recent 6 months for evaluation (higher quality labels)
```

#### 4.2.2 Sprint Instance Definition & Labeling

**Sprint Definition**:
A sprint is defined as one of:
1. **Explicit milestone** with due date (preferred)
2. **2-week rolling window** from milestone start
3. **Project board phase** with defined goals

**Ground Truth Labels**:

```python
Sprint Outcome (3-class):
  - SUCCESS: ≥90% planned issues closed on time
  - DELAYED: ≥70% issues closed, but >3 days past deadline
  - FAILED: <70% issues closed, or abandoned

Blocker Presence (binary):
  - BLOCKED: ≥1 issue labeled "blocked" or >5 days stuck
  - CLEAR: No significant blockers

Blocker Type (multi-label):
  - TECHNICAL: code complexity, bugs, test failures
  - DEPENDENCY: waiting on external library, cross-repo delay
  - RESOURCE: team capacity, skill gaps
  - REQUIREMENT: unclear spec, scope creep
  - EXTERNAL: third-party delays, infrastructure issues

Recommendation Relevance (human-labeled subset):
  - 1 (Not relevant): Off-topic or inapplicable
  - 2 (Slightly relevant): Correct domain, wrong context
  - 3 (Moderately relevant): Useful, but not actionable
  - 4 (Very relevant): Actionable and timely
  - 5 (Extremely relevant): Critical and immediately actionable
```

**Labeling Process**:
1. **Automated labeling**: GitHub milestone/issue status API
2. **Manual verification**: 500-sprint subset (cross-validated by 2 annotators)
3. **Adjudication**: Third annotator resolves disagreements (target: κ ≥ 0.75)
4. **Documentation**: Annotation guidelines with 30 examples

#### 4.2.3 Synthetic Data Generation

**Purpose**: Enable cold-start deployment and rare event coverage

**Generation Method**:
```python
LLM-Based Scenario Synthesis:
  Model: GPT-4 / Claude-3
  Prompt Template:
    """
    Generate a realistic startup sprint scenario with:
    - Team size: {3-10 developers}
    - Sprint length: {1-4 weeks}
    - Repository count: {2-3}
    - Outcome: {success/delayed/failed}
    - Blocker type: {technical/dependency/resource/requirement/none}
    
    Include:
    1. Initial sprint plan (5-10 issues)
    2. Daily commit activity patterns
    3. Issue creation/closure timeline
    4. PR review dynamics
    5. Blocker emergence and resolution (if applicable)
    6. Final outcome explanation
    """
```

**Synthetic Data Quality Control**:
- Statistical validation: KL divergence vs. real data <0.15
- Feature distribution matching: Jensen-Shannon divergence <0.20
- Expert review: 100-scenario human validation
- Discriminator test: Real vs. synthetic classifier ≤60% accuracy (indistinguishable)

**Synthetic Data Allocation**:
- **Total**: 5,000 synthetic scenarios
- **Edge cases** (30%): Rare blocker types, extreme delays, cascading failures
- **Typical cases** (50%): Common sprint patterns matching real distribution
- **Cross-repo scenarios** (20%): Complex dependency interactions

### 4.3 Data Splits & Cross-Validation

#### 4.3.1 Temporal Split (Primary)

### 5.1 Baselines

1. **Rule-based heuristic baseline** (velocity, burndown, issue closure thresholds).
2. **Classical ML baseline** (e.g., random forest / gradient boosting on handcrafted features).
3. **Single LLM baseline** (no agent orchestration).
4. **LLM without RAG** (same prompts, no retrieval evidence).

### 5.2 Proposed Variants

1. Full agentic workflow (collector + analyzer + risk assessor + recommender + explainer).
2. Full workflow without RAG.
3. Full workflow without cross-repo dependency reasoning.
4. Full workflow without synthetic-data augmentation.

### 5.3 Ablation Studies

- Remove one modality at a time: code, text, temporal, graph, sentiment, CI/CD.
- Remove one agent at a time to estimate contribution.
- Vary retrieval depth (`top-k` = 3, 5, 10).

---

## 6. Metrics

### 6.1 Predictive Performance

- Classification: Accuracy, Precision, Recall, F1, AUROC.
- Calibration: Brier score, Expected Calibration Error.
- Early warning quality: lead time (days before sprint end when risk is first correctly flagged).

### 6.2 Recommendation Quality

- Ranking: NDCG@k, MRR.
- Human relevance score (Likert 1-5).
- Recommendation acceptance rate.

### 6.3 Explanation Quality

- Human trust score.
- Evidence correctness (citation actually supports claim).
- Evidence coverage (fraction of major claims with supporting evidence).

### 6.4 Systems Metrics

- p50/p95 latency from event to recommendation.
- Peak and average RAM/CPU.
- Throughput (analyses/minute).
- Failure rate and retry recovery rate.

### 6.5 Business-Proxy Impact

- Estimated reduction in manual reporting time.
- Estimated improvement in early blocker resolution.
- Decision confidence of team leads before/after AI support.

---

## 7. Statistical Analysis Plan

1. Report mean, standard deviation, and 95% confidence intervals.
2. Use paired significance testing between model variants on matched sprint instances.
3. Report effect sizes (Cohen's d / Cliff's delta where appropriate).
4. Apply multiple-comparison correction for ablations.
5. Include bootstrap analysis for robustness on small startup datasets.

---

## 8. Human Evaluation Protocol

### 8.1 Participants

- 5-10 startup practitioners (tech leads, engineering managers, founders with PM responsibilities).

### 8.2 Procedure

1. Participants review sprint scenarios in two conditions: baseline vs agentic system.
2. They rate recommendations on relevance, actionability, and trust.
3. They indicate whether they would act on each recommendation.
4. Follow-up interview captures failure modes and adoption concerns.

### 8.3 Human Evaluation Outputs

- Quantitative Likert summaries.
- Inter-rater agreement (e.g., weighted kappa).
- Thematic analysis of qualitative feedback.

---

## 9. Experimental Timeline (Startup-Focused)

1. Week 1: Finalize labels, metrics, and baseline implementations.
2. Week 2: Offline baseline benchmarking and sanity checks.
3. Week 3: Agentic variants + ablation runs.
4. Week 4: Online replay and systems stress testing.
5. Week 5: Human evaluation sessions and analysis.
6. Week 6: Final statistical analysis, error analysis, and report.

---

## 10. Validity and Risk Controls

### Internal Validity

- Use strict temporal split to avoid leakage.
- Freeze prompt templates and evaluation scripts before final runs.

### External Validity

- Include multiple startup domains (e.g., SaaS, infra tooling, AI app).
- Report performance stratified by team size and sprint length.

### Construct Validity

- Align proxy metrics (acceptance rate, trust) with interview feedback.
- Validate label quality with dual-annotator checks on a subset.

### Key Risks

- Sparse startup history: mitigated with synthetic augmentation and uncertainty reporting.
- Label ambiguity: mitigated with adjudication rules and annotation guide.
- Overfitting to one organization: mitigated with leave-one-org-out tests.

---

## 11. Reproducibility Requirements

- Versioned dataset snapshots and split manifests.
- Fixed random seeds and environment lockfile.
- Logged experiment metadata (model version, prompts, retrieval config).
- Artifact bundle: metrics tables, plots, ablation matrix, error cases.

---

## 12. Target Success Criteria

1. Sprint risk/blocker F1 >= 0.85 on held-out startup data.
2. Recommendation NDCG@5 >= 0.80.
3. Human trust score >= 4.0/5.
4. p95 latency <60s on local hardware.
5. At least 20% reduction in perceived manual sprint-tracking effort (human study).

---

## 13. Deliverables

1. Evaluation dataset card and split documentation.
2. Baseline and agentic benchmark report.
3. Ablation and error analysis report.
4. Human study results and adoption insights.
5. Final evaluation chapter for thesis/paper submission.

---

## 14. Alignment with Existing Project Docs

- Research goals and hypotheses align with `docs/research/research_objectives.md`.
- Metrics and testing stack align with `docs/architecture/ml_validation_architecture.md`.
- Gap-driven novelty claims align with `docs/research/gap_similar_research.md`.

