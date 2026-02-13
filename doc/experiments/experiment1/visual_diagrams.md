# Visual Architecture Overview

This document contains high-level visual diagrams for quick understanding of the system architecture.

## System Overview: Input → Process → Output

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'fontSize':'16px'}}}%%
graph TB
    subgraph INPUT["📥 INPUT LAYER"]
        GH[GitHub API<br/>GraphQL + REST<br/>5K req/hr]
        
        subgraph INPUT_DATA["Input Datasets"]
            I1[(Milestones<br/>title, dates, status)]
            I2[(Issues<br/>title, body, labels, comments)]
            I3[(Pull Requests<br/>diffs, reviews, CI status)]
            I4[(Commits<br/>messages, changes, authors)]
            I5[(CI/CD Runs<br/>builds, tests, deployments)]
        end
        
        GH --> I1
        GH --> I2
        GH --> I3
        GH --> I4
        GH --> I5
    end
    
    subgraph PROCESS["⚙️ PROCESSING LAYER"]
        subgraph FEATURE_ENG["Feature Engineering (524-dim vector)"]
            F1[Code Features<br/>32-dim<br/>churn, complexity]
            F2[Text Features<br/>384-dim<br/>Sentence-BERT]
            F3[Temporal Features<br/>16-dim<br/>velocity, burndown]
            F4[Graph Features<br/>64-dim<br/>dependencies, networks]
            F5[Sentiment Features<br/>8-dim<br/>tone, morale]
            F6[CI/CD Features<br/>12-dim<br/>quality, speed]
        end
        
        subgraph AGENTS["LLM Agent System"]
            A1[Data Collector<br/>GitHub API fetching]
            A2[Feature Engineer<br/>Multi-modal extraction]
            A3[Embedding Agent<br/>Text → Vectors]
            A4[LLM Reasoner<br/>Llama-3-8B-Q4<br/>~5GB RAM]
            A5[Sprint Analyzer<br/>Pattern recognition]
            A6[Risk Assessor<br/>Blocker detection]
            A7[Recommender<br/>Intervention strategies]
            A8[Explainer<br/>Evidence attribution]
        end
        
        subgraph STORAGE["Storage & Memory"]
            S1[(SQLite<br/>Raw Events<br/>~2GB)]
            S2[(Parquet<br/>Features<br/>~3GB)]
            S3[(ChromaDB<br/>Vectors<br/>~2GB)]
            S4[(PostgreSQL<br/>Results<br/>~1GB)]
            S5[(Redis<br/>Cache<br/>~500MB)]
        end
    end
    
    subgraph OUTPUT["📤 OUTPUT LAYER"]
        subgraph OUTPUT_DATA["Output Datasets"]
            O1[Sprint Analysis<br/>completion probability<br/>health scores<br/>velocity metrics]
            O2[Risk Assessment<br/>blocker types<br/>severity levels<br/>impact estimates]
            O3[Recommendations<br/>actionable steps<br/>success rates<br/>effort estimates]
            O4[Evidence<br/>cited commits/issues/PRs<br/>reasoning chain<br/>similar cases]
            O5[Performance Metrics<br/>50+ KPIs<br/>trends<br/>comparisons]
        end
        
        subgraph INTERFACES["User Interfaces"]
            UI1[REST API<br/>FastAPI]
            UI2[Web Dashboard<br/>React]
            UI3[Reports<br/>Markdown/PDF]
        end
    end
    
    INPUT_DATA --> A1
    A1 --> S1
    S1 --> A2
    A2 --> F1
    A2 --> F2
    A2 --> F3
    A2 --> F4
    A2 --> F5
    A2 --> F6
    F1 --> S2
    F2 --> S2
    F3 --> S2
    F4 --> S2
    F5 --> S2
    F6 --> S2
    S2 --> A3
    A3 --> S3
    S3 --> A4
    A4 --> A5
    A5 --> A6
    A6 --> A7
    A7 --> A8
    A8 --> S4
    S4 --> S5
    S5 --> O1
    S5 --> O2
    S5 --> O3
    S5 --> O4
    S5 --> O5
    O1 --> UI1
    O2 --> UI1
    O3 --> UI1
    O4 --> UI1
    O5 --> UI1
    UI1 --> UI2
    UI1 --> UI3
    
    style INPUT fill:#e1f5ff
    style PROCESS fill:#fff4e1
    style OUTPUT fill:#e8f5e9
    style INPUT_DATA fill:#b3e5fc
    style FEATURE_ENG fill:#ffe0b2
    style AGENTS fill:#ffcc80
    style STORAGE fill:#fff9c4
    style OUTPUT_DATA fill:#c8e6c9
    style INTERFACES fill:#a5d6a7
```

## Data Flow: Example Sprint Analysis

```mermaid
%%{init: {'theme':'base'}}%%
sequenceDiagram
    autonumber
    participant PM as Project Manager
    participant API as REST API
    participant DC as Data Collector
    participant FE as Feature Engineer
    participant LLM as LLM Agent<br/>(Llama-3-8B)
    participant SA as Sprint Analyzer
    participant RA as Risk Assessor
    participant REC as Recommender
    participant EXP as Explainer
    participant DB as Database
    
    PM->>API: Request analysis for<br/>Sprint 24
    API->>DC: Fetch latest data
    DC->>DB: Query milestone data
    DB-->>DC: Return 45 issues,<br/>32 PRs, 187 commits
    DC->>DB: Store raw events
    
    DC->>FE: Process raw data
    FE->>FE: Extract code features (32-dim)
    FE->>FE: Extract text features (384-dim)
    FE->>FE: Extract temporal features (16-dim)
    FE->>FE: Extract graph features (64-dim)
    FE->>FE: Extract sentiment features (8-dim)
    FE->>FE: Extract CI/CD features (12-dim)
    FE->>DB: Store 524-dim feature vector
    
    FE->>LLM: Analyze sprint<br/>with context
    LLM->>DB: Retrieve similar<br/>sprints (RAG)
    DB-->>LLM: 10 historical<br/>examples
    LLM->>SA: Identify patterns
    SA-->>LLM: Velocity declining,<br/>PR #890 blocked
    
    LLM->>RA: Assess risks
    RA-->>LLM: 2 high risks:<br/>- Dependency blocker<br/>- Velocity decline
    
    LLM->>REC: Generate<br/>recommendations
    REC-->>LLM: 3 actions:<br/>- Add reviewers<br/>- Reprioritize<br/>- Team sync
    
    LLM->>EXP: Explain reasoning
    EXP->>DB: Cite evidence
    DB-->>EXP: PR #890, issues #1542-1544
    EXP-->>LLM: Reasoning chain<br/>+ evidence
    
    LLM->>DB: Store results
    DB-->>API: Analysis complete
    API-->>PM: Display:<br/>72.5% completion prob<br/>5 risks<br/>3 recommendations<br/>Full explanation
```

## Memory Budget Visualization (M4 Pro - 24GB RAM)

```mermaid
%%{init: {'theme':'base'}}%%
pie title Memory Allocation on MacBook M4 Pro (24GB)
    "macOS System" : 4
    "LLM (Llama-3-8B-Q4)" : 5
    "ChromaDB Vectors" : 2
    "Feature Processing" : 3
    "PostgreSQL" : 1
    "Redis Cache" : 0.5
    "Python Runtime" : 2
    "Browser Dashboard" : 1.5
    "Safety Buffer" : 5
```

## Feature Engineering Pipeline

```mermaid
%%{init: {'theme':'base'}}%%
graph LR
    subgraph RAW["Raw GitHub Data"]
        R1[Commits<br/>~200/sprint]
        R2[Issues<br/>~45/sprint]
        R3[PRs<br/>~32/sprint]
        R4[Comments<br/>~300/sprint]
        R5[CI Runs<br/>~50/sprint]
    end
    
    subgraph EXTRACTORS["Feature Extractors"]
        E1[Code Analyzer<br/>Diff parsing<br/>AST analysis]
        E2[Text Encoder<br/>Sentence-BERT<br/>all-MiniLM-L6-v2]
        E3[Time Series<br/>Velocity calc<br/>Burndown]
        E4[Graph Builder<br/>NetworkX<br/>Node2Vec]
        E5[Sentiment<br/>VADER<br/>Polarity]
        E6[Quality<br/>Test rates<br/>Build times]
    end
    
    subgraph FEATURES["Feature Vectors"]
        F1[32-dim<br/>Code]
        F2[384-dim<br/>Text]
        F3[16-dim<br/>Temporal]
        F4[64-dim<br/>Graph]
        F5[8-dim<br/>Sentiment]
        F6[12-dim<br/>CI/CD]
    end
    
    COMBINED[524-dim<br/>Combined Vector]
    
    R1 --> E1
    R2 --> E2
    R3 --> E2
    R2 --> E3
    R3 --> E3
    R2 --> E4
    R3 --> E4
    R4 --> E5
    R5 --> E6
    
    E1 --> F1
    E2 --> F2
    E3 --> F3
    E4 --> F4
    E5 --> F5
    E6 --> F6
    
    F1 --> COMBINED
    F2 --> COMBINED
    F3 --> COMBINED
    F4 --> COMBINED
    F5 --> COMBINED
    F6 --> COMBINED
    
    style RAW fill:#ffebee
    style EXTRACTORS fill:#fff3e0
    style FEATURES fill:#e8f5e9
    style COMBINED fill:#c8e6c9
```

## Multi-Agent System Architecture

```mermaid
%%{init: {'theme':'base'}}%%
graph TB
    ORCH[Orchestrator Agent<br/>LangChain]
    
    subgraph AGENTS["Specialized Agents"]
        A1[Data Collector<br/>Tools: GitHub API, SQL]
        A2[Feature Engineer<br/>Tools: Pandas, PyTorch]
        A3[Embedder<br/>Tools: SentenceTransformers]
        A4[LLM Reasoner<br/>Tools: Llama-3-8B, RAG]
        A5[Sprint Analyzer<br/>Tools: Pattern Matching]
        A6[Risk Assessor<br/>Tools: Rule Engine]
        A7[Recommender<br/>Tools: Historical DB]
        A8[Explainer<br/>Tools: Evidence Retrieval]
    end
    
    subgraph MEMORY["Agent Memory"]
        M1[(Short-term<br/>Redis)]
        M2[(Long-term<br/>PostgreSQL)]
        M3[(Vector<br/>ChromaDB)]
    end
    
    ORCH --> A1
    ORCH --> A2
    ORCH --> A3
    ORCH --> A4
    ORCH --> A5
    ORCH --> A6
    ORCH --> A7
    ORCH --> A8
    
    A1 --> M1
    A2 --> M1
    A3 --> M3
    A4 --> M3
    A5 --> M2
    A6 --> M2
    A7 --> M2
    A8 --> M2
    
    M1 -.-> M2
    M2 -.-> M3
    
    style ORCH fill:#e1bee7
    style AGENTS fill:#fff9c4
    style MEMORY fill:#b2dfdb
```

## GitHub Metrics Collection Strategy

```mermaid
%%{init: {'theme':'base'}}%%
graph TB
    START[Start Analysis]
    
    subgraph ORG_LEVEL["Organization Level (1 API call)"]
        O1[GET /orgs/{org}<br/>Basic info]
        O2[GET /orgs/{org}/repos<br/>Repository list]
    end
    
    subgraph REPO_LEVEL["Repository Level (1 GraphQL call per repo)"]
        R1[GraphQL Query:<br/>Milestones + Issues + PRs +<br/>Commits + Reviews + CI]
    end
    
    subgraph DERIVED["Derived Metrics (Computed)"]
        D1[Velocity Metrics<br/>- Current velocity<br/>- Required velocity<br/>- Velocity gap]
        D2[Burndown Analysis<br/>- Ideal vs actual<br/>- Slope<br/>- Deviation]
        D3[Team Metrics<br/>- Active contributors<br/>- Collaboration score<br/>- Response times]
        D4[Quality Metrics<br/>- Code churn<br/>- Review coverage<br/>- CI success rate]
        D5[Risk Indicators<br/>- Blocked issues<br/>- Open PRs<br/>- Failed builds]
    end
    
    CACHE[(Redis Cache<br/>15-min TTL)]
    DB[(PostgreSQL<br/>Persistent Storage)]
    
    START --> O1
    O1 --> O2
    O2 --> R1
    R1 --> CACHE
    CACHE --> D1
    CACHE --> D2
    CACHE --> D3
    CACHE --> D4
    CACHE --> D5
    D1 --> DB
    D2 --> DB
    D3 --> DB
    D4 --> DB
    D5 --> DB
    
    style START fill:#e3f2fd
    style ORG_LEVEL fill:#fff3e0
    style REPO_LEVEL fill:#f3e5f5
    style DERIVED fill:#e8f5e9
    style CACHE fill:#ffebee
    style DB fill:#e0f2f1
```

## Risk Types & Detection

```mermaid
%%{init: {'theme':'base'}}%%
mindmap
  root((Risk Types))
    Dependency Blocker
      PR blocking issues
      Cross-repo dependencies
      External API dependencies
    Velocity Decline
      Issues/day decreasing
      PR merge rate down
      Commit frequency low
    Scope Creep
      New issues added
      Milestone expanded
      Requirements changed
    CI/CD Failures
      Build failures
      Test failures
      Deployment issues
    Review Bottleneck
      PRs awaiting review
      Long review times
      Reviewer capacity
    Team Capacity
      Contributors unavailable
      High workload
      Context switching
    Technical Debt
      Code quality declining
      Complexity increasing
      Refactoring needed
```

## Recommendation Categories

```mermaid
%%{init: {'theme':'base'}}%%
graph TD
    RISKS[Identified Risks]
    
    subgraph REC["Recommendation Engine"]
        R1[Prioritization<br/>- Reorder backlog<br/>- Focus on critical path<br/>- Remove blockers]
        R2[Resource Allocation<br/>- Add reviewers<br/>- Assign more devs<br/>- Balance workload]
        R3[Scope Reduction<br/>- Move non-critical items<br/>- Split milestone<br/>- Defer features]
        R4[Dependency Resolution<br/>- Merge blocking PRs<br/>- Coordinate teams<br/>- Resolve conflicts]
        R5[Process Improvement<br/>- Change workflow<br/>- Automate tasks<br/>- Reduce meetings]
        R6[Communication<br/>- Team sync<br/>- Stakeholder update<br/>- Retrospective]
    end
    
    subgraph HIST["Historical Data (RAG)"]
        H1[(Similar Sprints<br/>Success Cases)]
        H2[(Intervention Outcomes<br/>Success Rates)]
    end
    
    OUTPUT[Ranked Recommendations<br/>with:<br/>- Action steps<br/>- Historical success rate<br/>- Effort estimate<br/>- Expected impact]
    
    RISKS --> R1
    RISKS --> R2
    RISKS --> R3
    RISKS --> R4
    RISKS --> R5
    RISKS --> R6
    
    H1 --> R1
    H1 --> R2
    H1 --> R3
    H1 --> R4
    H1 --> R5
    H1 --> R6
    
    H2 --> R1
    H2 --> R2
    H2 --> R3
    H2 --> R4
    H2 --> R5
    H2 --> R6
    
    R1 --> OUTPUT
    R2 --> OUTPUT
    R3 --> OUTPUT
    R4 --> OUTPUT
    R5 --> OUTPUT
    R6 --> OUTPUT
    
    style RISKS fill:#ffcdd2
    style REC fill:#fff9c4
    style HIST fill:#c5e1a5
    style OUTPUT fill:#b2dfdb
```

## Performance Benchmarks

```mermaid
%%{init: {'theme':'base'}}%%
gantt
    title Processing Time per Organization Size
    dateFormat X
    axisFormat %s seconds
    
    section Small Org (5 repos, 25 milestones)
    Data Collection :0, 120
    Feature Engineering :120, 180
    LLM Analysis :300, 60
    
    section Medium Org (20 repos, 100 milestones)
    Data Collection :0, 480
    Feature Engineering :480, 720
    LLM Analysis :1200, 240
    
    section Large Org (50 repos, 250 milestones)
    Data Collection :0, 1200
    Feature Engineering :1200, 1800
    LLM Analysis :3000, 600
```

## 16-Week Implementation Timeline

```mermaid
%%{init: {'theme':'base'}}%%
gantt
    title Implementation Roadmap
    dateFormat YYYY-MM-DD
    
    section Phase 1: Data & Infrastructure
    Data Collection Infrastructure :done, 2026-02-14, 14d
    Feature Engineering :done, 2026-02-28, 14d
    
    section Phase 2: Model Development
    LLM Integration :active, 2026-03-14, 14d
    Agent System :2026-03-28, 14d
    
    section Phase 3: Training
    LoRA Fine-tuning :2026-04-11, 14d
    RLHF & Optimization :2026-04-25, 14d
    
    section Phase 4: Deployment
    Dashboard & API :2026-05-09, 14d
    Evaluation & Documentation :2026-05-23, 14d
```

---

## Key Takeaways

### ✅ Achievable on M4 Pro (24GB RAM)
- Peak RAM usage: ~22GB
- Storage: ~30GB
- Processing time: 15 sec/milestone

### 🎯 Complete System
- 7 input datasets
- 6 feature modalities (524-dim)
- 8 specialized agents
- 5 output datasets
- 50+ derived metrics

### 🚀 Novel Contributions
- Organization-level intelligence
- Multi-modal LLM fusion
- RAG-based explainability
- LoRA fast adaptation
- Real-time capable architecture

### 📊 Realistic Performance
- Small org: 6 minutes
- Medium org: 25 minutes
- Large org: 65 minutes
- 3 orgs total: 80 minutes

**Status**: Ready for Implementation ✅
