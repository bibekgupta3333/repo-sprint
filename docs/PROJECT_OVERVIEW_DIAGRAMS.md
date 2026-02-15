# Project Overview & Architecture Diagrams
# Lightweight Sprint Intelligence for Small Startups

**Visual Guide to System Components**  
**Target**: 2-3 Repository Startups | 16GB RAM Laptops | <10 Min Setup

---

## 📊 High-Level System Overview

```mermaid
graph TB
    subgraph "External Data Sources"
        GH[GitHub API<br/>2-3 Repos<br/>50-200 events/day]
        SYNTH[Synthetic Data<br/>Pre-generated scenarios<br/>Bootstrap training]
    end
    
    subgraph "Data Collection Layer"
        COLL[Lightweight Collector<br/>Real-time + Batch<br/>Minimal overhead]
    end
    
    subgraph "Multi-Modal Feature Engineering"
        CODE[Code Features<br/>Churn, Complexity<br/>Cross-repo imports (2-3 repos)]
        TEXT[Text Features<br/>Issue/PR semantics<br/>NLP embeddings]
        TEMP[Temporal Features<br/>Velocity, Trends<br/>Sprint patterns]
        GRAPH[Dependency Graph<br/>2-3 repo DAG<br/>Blocker chains]
        SENT[Sentiment<br/>Comment tone<br/>Small team morale]
        CICD[CI/CD Metrics<br/>Test pass rate<br/>Build times]
    end
    
    subgraph "AI/ML Processing (16GB RAM)"
        EMBED[Embedding Agent<br/>Sentence-BERT<br/>~500MB]
        FUSE[Feature Engineer<br/>Multi-Modal Fusion<br/>~1GB]
        LLM[LLM Reasoning<br/>Llama-3-8B Q4<br/>~5GB RAM]
        RAG[RAG Module<br/>ChromaDB Local<br/>~1GB]
    end
    
    subgraph "Analysis & Decision"
        SPRINT[Sprint Analyzer<br/>Health Score]
        RISK[Risk Assessor<br/>Blocker Detection]
        REC[Recommender<br/>Actionable Suggestions]
        EXPL[Explainer<br/>Evidence-Based Reasoning]
    end
    
    subgraph "Storage (Lightweight)"
        SQLITE[(SQLite<br/>Local Events<br/>~500MB)]
        PG[(PostgreSQL<br/>Analysis Results<br/>~200MB)]
        CHROMA[(ChromaDB<br/>Embeddings<br/>~1GB)]
        REDIS[(Redis Cache<br/>Sessions<br/>~100MB)]
    end
    
    subgraph "User Interface (Streamlit)"
        DASH[Dashboard<br/>3 Core Screens<br/>Sprint Health + Blockers + Recs]
    end
    
    subgraph "Feedback Loop (Optional)"
        FEEDBACK[Team Feedback<br/>Accept/Reject<br/>Improve over time]
    end
    
    GH --> COLL
    SYNTH --> COLL
    
    COLL --> CODE
    COLL --> TEXT
    COLL --> TEMP
    COLL --> GRAPH
    COLL --> SENT
    COLL --> CICD
    
    CODE --> FUSE
    TEXT --> FUSE
    TEMP --> FUSE
    GRAPH --> FUSE
    SENT --> FUSE
    CICD --> FUSE
    
    FUSE --> EMBED
    EMBED --> CHROMA
    
    FUSE --> LLM
    CHROMA --> RAG
    RAG --> LLM
    
    LLM --> SPRINT
    LLM --> RISK
    LLM --> REC
    LLM --> EXPL
    
    SPRINT --> PG
    RISK --> PG
    REC --> PG
    EXPL --> PG
    
    PG --> DASH
    CHROMA --> DASH
    REDIS --> DASH
    
    DASH --> FEEDBACK
    FEEDBACK -.->|Improves| LLM
    
    classDef external fill:#e2e8f0,stroke:#94a3b8,color:#000
    classDef data fill:#dbeafe,stroke:#3b82f6,color:#000
    classDef lightweight fill:#dcfce7,stroke:#16a34a,color:#000
    
    class GH,SYNTH external
    class COLL,CODE,TEXT,TEMP,GRAPH,SENT,CICD data
    class SQLITE,CHROMA,REDIS lightweight
    classDef feature fill:#fef3c7,stroke:#f59e0b,color:#000
    classDef ai fill:#d1fae5,stroke:#10b981,color:#000
    classDef analysis fill:#e9d5ff,stroke:#8b5cf6,color:#000
    classDef storage fill:#fecdd3,stroke:#f43f5e,color:#000
    classDef ui fill:#bfdbfe,stroke:#3b82f6,color:#000
    classDef feedback fill:#fed7aa,stroke:#ea580c,color:#000
    
    class GH,GA external
    class COLL data
    class CODE,TEXT,TEMP,GRAPH,SENT,CICD feature
    class EMBED,FUSE,LLM,RAG ai
    class SPRINT,RISK,REC,EXPL analysis
    class PG,CHROMA,REDIS storage
    class DASH ui
    class RLHF feedback
```

---

## 🏗️ Deployment Architecture

```mermaid
graph TB
    subgraph "Developer Machine / Small Startup Server (16GB RAM)"
        subgraph "Docker Compose Environment"
            
            subgraph "Frontend Container"
                ST[Streamlit App<br/>:8501]
            end
            
            subgraph "Backend Container"
                FA[FastAPI Server<br/>:8000<br/>LangGraph Orchestrator]
            end
            
            subgraph "Data Storage"
                PG[(PostgreSQL<br/>:5432<br/>Structured Data)]
                CH[(ChromaDB<br/>:8001<br/>Vector Store)]
                RD[(Redis<br/>:6379<br/>Cache)]
            end
            
            subgraph "LLM Container"
                OL[Ollama Server<br/>:11434<br/>Llama-3-8B-Q4]
            end
            
        end
        
        subgraph "Persistent Volumes"
            V1[postgres_data]
            V2[chromadb_data]
            V3[ollama_models]
            V4[redis_data]
        end
    end
    
    subgraph "External Services"
        GH_API[GitHub API]
        GH_WEBHOOK[GitHub Webhooks]
    end
    
    USER[Startup Tech Lead<br/>Browser] --> ST
    ST --> FA
    FA --> PG
    FA --> CH
    FA --> RD
    FA --> OL
    
    PG --> V1
    CH --> V2
    OL --> V3
    RD --> V4
    
    FA --> GH_API
    GH_WEBHOOK --> FA
    
    classDef frontend fill:#dbeafe,stroke:#3b82f6,color:#000
    classDef backend fill:#d1fae5,stroke:#10b981,color:#000
    classDef storage fill:#fecdd3,stroke:#f43f5e,color:#000
    classDef llm fill:#e9d5ff,stroke:#8b5cf6,color:#000
    classDef volume fill:#fef3c7,stroke:#f59e0b,color:#000
    classDef external fill:#e2e8f0,stroke:#94a3b8,color:#000
    classDef user fill:#fed7aa,stroke:#ea580c,color:#000
    
    class ST frontend
    class FA backend
    class PG,CH,RD storage
    class OL llm
    class V1,V2,V3,V4 volume
    class GH_API,GH_WEBHOOK external
    class USER user
```

---

## 🔄 Data Flow - Real-Time Analysis

```mermaid
sequenceDiagram
    participant PM as Project Manager
    participant ST as Streamlit UI
    participant FA as FastAPI Backend
    participant ORC as Orchestrator Agent
    participant DC as Data Collector
    participant FE as Feature Engineer
    participant EMB as Embedding Agent
    participant LLM as LLM Reasoning
    participant SA as Sprint Analyzer
    participant RA as Risk Assessor
    participant REC as Recommender
    participant EX as Explainer
    participant PG as PostgreSQL
    participant CH as ChromaDB
    participant OL as Ollama
    
    PM->>ST: Request sprint analysis
    ST->>FA: POST /api/analyze/milestone/{id}
    FA->>ORC: Initialize workflow
    
    ORC->>DC: Fetch milestone data
    DC->>PG: Query milestone, issues, PRs, commits
    PG-->>DC: Return data
    DC-->>ORC: Raw data
    
    ORC->>FE: Engineer features
    FE->>FE: Compute 6 modalities
    FE-->>ORC: 524-dim feature vector
    
    ORC->>EMB: Generate embeddings
    EMB->>EMB: Sentence-BERT encode
    EMB->>CH: Store embeddings
    EMB-->>ORC: Embedding vectors
    
    ORC->>LLM: Reason about sprint
    LLM->>CH: Retrieve similar cases (RAG)
    CH-->>LLM: Top-5 similar sprints
    LLM->>OL: LLM inference
    OL-->>LLM: Reasoning output
    LLM-->>ORC: Insights
    
    ORC->>SA: Analyze sprint health
    SA->>SA: Compute health score
    SA->>PG: Save analysis
    SA-->>ORC: Health: 78%, Completion: 65%
    
    ORC->>RA: Assess risks
    RA->>RA: Detect blockers
    RA->>PG: Save risks
    RA-->>ORC: 3 risks identified
    
    ORC->>REC: Generate recommendations
    REC->>REC: Prioritize actions
    REC->>PG: Save recommendations
    REC-->>ORC: 5 recommendations
    
    ORC->>EX: Explain results
    EX->>CH: Fetch evidence
    EX->>EX: Build explanation
    EX-->>ORC: Natural language explanation
    
    ORC-->>FA: Complete analysis
    FA-->>ST: JSON response
    ST-->>PM: Display dashboard
    
    Note over PM,ST: Latency target: <60 seconds
```

---

## 📋 Documentation Structure

```
doc/
├── README.md ────────────────────────┐
│   (Documentation Index)             │
│                                      │
├── planning/                          │
│   └── WBS.md ─────────┐              │
│       (14-week plan)   │              │
│                        │              │
├── architecture/        │              │
│   ├── system_architecture.md ────┐   │
│   ├── database_design.md ────────┤   │
│   └── ml_validation_architecture.md  │
│                                   │   │
├── deployment/                     │   │
│   └── deployment_guide.md ───────┤   │
│                                   │   │
├── design/                         │   │
│   └── figma_design_prompts.md ───┤   │
│                                   │   │
├── research/                       │   │
│   ├── gap_similar_research.md ───┤   │
│   ├── research_objectives.md ────┤   │
│   └── similar_papers_bibliography.md
│                                   │   │
├── experiments/                    │   │
│   └── [Experimental architectures]   │
│                                   │   │
├── thesis_proposal.md ─────────────┤   │
├── quick_reference.md ─────────────┤   │
└── DOCUMENTATION_COMPLETE.md ──────────┘
    (This summary)

Root Files:
├── docker-compose.yml ──────────────────┐
├── .env.example ────────────────────────┤
├── .editorconfig ───────────────────────┤
└── .cursorrules ────────────────────────┘
```

---

## 📊 Research Gaps & Our Solutions

```mermaid
graph LR
    subgraph "Critical Gaps from 50 Papers"
        G1[Project-Level<br/>Intelligence<br/>0/50 papers]
        G2[Real-Time<br/>LLM<br/>1/50 partial]
        G3[Multi-Modal<br/>Fusion<br/>2/50 partial]
        G4[Explainable<br/>AI<br/>0/50 papers]
        G5[Adaptive<br/>Learning<br/>2/50 theory]
    end
    
    subgraph "Our Novel Solutions"
        S1[Multi-Repo<br/>Architecture]
        S2[Streaming<br/>+ LLM]
        S3[6-Modality<br/>Transformer]
        S4[RAG with<br/>Evidence]
        S5[RLHF<br/>Implementation]
    end
    
    G1 -.->|Addresses| S1
    G2 -.->|Addresses| S2
    G3 -.->|Addresses| S3
    G4 -.->|Addresses| S4
    G5 -.->|Addresses| S5
    
    S1 --> O1[F1 > 0.85<br/>Cross-Repo]
    S2 --> O2[<60s<br/>Latency]
    S3 --> O3[+35%<br/>Accuracy]
    S4 --> O4[80%+<br/>Trust Score]
    S5 --> O5[Break 87%<br/>Ceiling]
    
    classDef gap fill:#fecdd3,stroke:#f43f5e,color:#000
    classDef solution fill:#d1fae5,stroke:#10b981,color:#000
    classDef outcome fill:#dbeafe,stroke:#3b82f6,color:#000
    
    class G1,G2,G3,G4,G5 gap
    class S1,S2,S3,S4,S5 solution
    class O1,O2,O3,O4,O5 outcome
```

---

## 🎯 14-Week Project Timeline

```mermaid
gantt
    title 14-Week Research Project Timeline
    dateFormat YYYY-MM-DD
    section Phase 1: Research & Planning
    Literature Review           :done, p1_1, 2026-02-01, 7d
    Gap Analysis               :done, p1_2, after p1_1, 3d
    Research Objectives        :done, p1_3, after p1_2, 2d
    WBS & Architecture         :done, p1_4, after p1_3, 2d
    
    section Phase 2: Data Collection
    Infrastructure Setup       :active, p2_1, 2026-02-15, 7d
    GitHub Data Ingestion      :p2_2, after p2_1, 7d
    Feature Engineering        :p2_3, after p2_2, 7d
    
    section Phase 3: Backend Development
    FastAPI Skeleton           :p3_1, 2026-03-08, 7d
    Database Models            :p3_2, after p3_1, 7d
    API Endpoints              :p3_3, after p3_2, 7d
    
    section Phase 4: Agent Development
    LangGraph Orchestrator     :p4_1, 2026-03-29, 7d
    9 Specialized Agents       :p4_2, after p4_1, 7d
    
    section Phase 5: Frontend Development
    Streamlit Dashboard        :p5_1, 2026-04-12, 7d
    Charts & Visualization     :p5_2, after p5_1, 7d
    
    section Phase 6: ML Validation
    Model Evaluation           :p6_1, 2026-04-26, 7d
    Human Evaluation           :p6_2, after p6_1, 7d
    
    section Milestones
    Documentation Complete     :milestone, m1, 2026-02-14, 0d
    Infrastructure Ready       :milestone, m2, 2026-02-28, 0d
    Backend Complete           :milestone, m3, 2026-03-28, 0d
    Agents Complete            :milestone, m4, 2026-04-11, 0d
    Frontend Complete          :milestone, m5, 2026-04-25, 0d
    Thesis Submission          :crit, milestone, m6, 2026-05-31, 0d
```

---

## 🔍 Technology Stack

```mermaid
graph TB
    subgraph "Frontend Layer"
        ST[Streamlit 1.30+]
        PL[Plotly 5.18+]
        MD[Material Design 3]
    end
    
    subgraph "Backend Layer"
        FA[FastAPI 0.109+]
        PY[Python 3.11+]
        LG[LangGraph 0.0.20+]
        PD[Pydantic 2.5+]
    end
    
    subgraph "AI/ML Layer"
        OL[Ollama + Llama-3-8B-Q4]
        SB[Sentence-BERT<br/>all-MiniLM-L6-v2]
        SK[scikit-learn 1.4+]
        PT[PyTorch 2.1+]
    end
    
    subgraph "Database Layer"
        PG[PostgreSQL 15+]
        CH[ChromaDB 0.4+]
        RD[Redis 7+]
        SA[SQLAlchemy 2.0+]
    end
    
    subgraph "Infrastructure Layer"
        DC[Docker Compose]
        AL[Alembic Migrations]
        NX[Nginx (Optional)]
    end
    
    subgraph "Development Tools"
        PT2[pytest]
        BK[Black Formatter]
        MY[Mypy Type Checker]
        PR[Pre-commit Hooks]
    end
    
    ST --> FA
    PL --> ST
    
    FA --> LG
    FA --> PD
    
    LG --> OL
    LG --> SB
    
    FA --> SA
    SA --> PG
    SA --> CH
    SA --> RD
    
    DC --> ST
    DC --> FA
    DC --> PG
    DC --> CH
    DC --> RD
    DC --> OL
    
    classDef frontend fill:#dbeafe,stroke:#3b82f6,color:#000
    classDef backend fill:#d1fae5,stroke:#10b981,color:#000
    classDef ai fill:#e9d5ff,stroke:#8b5cf6,color:#000
    classDef database fill:#fecdd3,stroke:#f43f5e,color:#000
    classDef infra fill:#fef3c7,stroke:#f59e0b,color:#000
    classDef dev fill:#e2e8f0,stroke:#94a3b8,color:#000
    
    class ST,PL,MD frontend
    class FA,PY,LG,PD backend
    class OL,SB,SK,PT ai
    class PG,CH,RD,SA database
    class DC,AL,NX infra
    class PT2,BK,MY,PR dev
```

---

## 📈 Performance Targets

```mermaid
graph LR
    subgraph "Metrics & Targets"
        M1[Sprint Success<br/>Prediction F1<br/>Target: >0.90<br/>Baseline: 0.74-0.87]
        M2[Blocker Detection<br/>F1<br/>Target: >0.88<br/>Baseline: N/A]
        M3[Analysis Latency<br/>Target: <60s<br/>Baseline: 15-30min]
        M4[Stakeholder Trust<br/>Target: >80%<br/>Baseline: 23%]
        M5[Cold-Start Time<br/>Target: <7 days<br/>Baseline: 6-12 months]
        M6[RAM Footprint<br/>Target: <16GB<br/>Baseline: Cloud-only]
    end
    
    M1 --> R1[+3-16%<br/>Improvement]
    M2 --> R2[Novel<br/>Capability]
    M3 --> R3[15-30×<br/>Faster]
    M4 --> R4[3.5×<br/>Increase]
    M5 --> R5[26-52×<br/>Faster]
    M6 --> R6[Startup<br/>Accessible]
    
    classDef metric fill:#dbeafe,stroke:#3b82f6,color:#000
    classDef result fill:#d1fae5,stroke:#10b981,color:#000
    
    class M1,M2,M3,M4,M5,M6 metric
    class R1,R2,R3,R4,R5,R6 result
```

---

## 🎨 UI/UX Design System

```mermaid
graph TB
    subgraph "Color Palette"
        P1[Primary: #3b82f6]
        P2[Success: #10b981]
        P3[Warning: #f59e0b]
        P4[Danger: #ef4444]
        P5[Neutral: Grays]
    end
    
    subgraph "5 Key Screens"
        S1[Dashboard<br/>Sprint Overview]
        S2[Milestone Analysis<br/>Detailed Metrics]
        S3[Cross-Repo Dependencies<br/>Graph Visualization]
        S4[Recommendations<br/>Actionable Insights]
        S5[Settings<br/>Configuration]
    end
    
    subgraph "Component Library"
        C1[Buttons<br/>3 Variants]
        C2[Cards<br/>Metric Display]
        C3[Badges<br/>Status Indicators]
        C4[Typography<br/>Inter Font]
        C5[Icons<br/>Lucide Icons]
    end
    
    subgraph "Responsive Breakpoints"
        B1[Mobile: 375px]
        B2[Tablet: 768px]
        B3[Desktop: 1440px]
        B4[Ultrawide: 1920px]
    end
    
    P1 --> S1
    P2 --> S2
    P3 --> S3
    P4 --> S4
    P5 --> S5
    
    C1 --> S1
    C2 --> S2
    C3 --> S3
    C4 --> S4
    C5 --> S5
    
    classDef color fill:#dbeafe,stroke:#3b82f6,color:#000
    classDef screen fill:#d1fae5,stroke:#10b981,color:#000
    classDef component fill:#fef3c7,stroke:#f59e0b,color:#000
    classDef breakpoint fill:#e9d5ff,stroke:#8b5cf6,color:#000
    
    class P1,P2,P3,P4,P5 color
    class S1,S2,S3,S4,S5 screen
    class C1,C2,C3,C4,C5 component
    class B1,B2,B3,B4 breakpoint
```

---

## 🧪 Testing Pyramid

```mermaid
graph TB
    L6[Human Evaluation<br/>5-10 PMs<br/>Qualitative Feedback]
    L5[A/B Testing<br/>Variant Comparison<br/>Statistical Significance]
    L4[ML Validation<br/>F1, AUC, RMSE, NDCG<br/>Temporal Split, 5-Fold CV]
    L3[Agent Tests<br/>LangGraph Workflows<br/>Integration Testing]
    L2[Integration Tests<br/>API Endpoints<br/>Database Operations]
    L1[Unit Tests<br/>Individual Functions<br/>>80% Coverage]
    
    L1 --> L2
    L2 --> L3
    L3 --> L4
    L4 --> L5
    L5 --> L6
    
    classDef level1 fill:#10b981,stroke:#059669,color:#fff
    classDef level2 fill:#3b82f6,stroke:#2563eb,color:#fff
    classDef level3 fill:#8b5cf6,stroke:#7c3aed,color:#fff
    classDef level4 fill:#f59e0b,stroke:#d97706,color:#fff
    classDef level5 fill:#ef4444,stroke:#dc2626,color:#fff
    classDef level6 fill:#ec4899,stroke:#db2777,color:#fff
    
    class L1 level1
    class L2 level2
    class L3 level3
    class L4 level4
    class L5 level5
    class L6 level6
```

---

**Document Version**: 1.0.0  
**Last Updated**: February 14, 2026  
**Purpose**: Visual reference for project architecture and workflows  

**Related Documents**:
- [System Architecture](architecture/system_architecture.md) - Detailed technical specs
- [WBS](planning/WBS.md) - Project timeline and tasks
- [Documentation Index](README.md) - Full navigation
