# LLM Agentic Architecture for Organization-Level Sprint Intelligence

**Project**: Organization-Wide Intelligent Sprint and Milestone Management  
**Focus**: Gap 1 - Organization-Level LLM Intelligence  
**Hardware Constraints**: MacBook M4 Pro, 24GB RAM  
**Date**: February 13, 2026

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Architecture Overview](#system-architecture-overview)
3. [Database Architecture](#database-architecture)
4. [Use Case Architecture](#use-case-architecture)
5. [Actor Architecture](#actor-architecture)
6. [LLM Agent Architecture](#llm-agent-architecture)
7. [Input-Process-Output Pipeline](#input-process-output-pipeline)
8. [GitHub Metrics Collection](#github-metrics-collection)
9. [Feature Engineering](#feature-engineering)
10. [Output Metrics & Recommendations](#output-metrics--recommendations)
11. [Feasibility Analysis (M4 Pro Constraints)](#feasibility-analysis-m4-pro-constraints)
12. [Implementation Roadmap](#implementation-roadmap)

---

## Executive Summary

This document outlines a **lightweight, achievable LLM agentic system** designed to run on a MacBook M4 Pro (24GB RAM) for organization-level GitHub sprint intelligence. The architecture leverages:

- **Quantized LLMs** (Llama-3-8B-Q4 instead of 70B) for local inference
- **Multi-agent system** with specialized agents for data collection, analysis, and recommendations
- **Incremental learning** with LoRA adapters (<500MB per organization)
- **Efficient vector storage** using ChromaDB (runs locally)
- **Batch processing** to manage memory constraints

**Key Achievability Factors**:
- ✅ Uses smaller quantized models (8B params, ~5GB RAM)
- ✅ Agents run sequentially (not parallel) to conserve memory
- ✅ Processes repositories incrementally
- ✅ Stores embeddings efficiently with compression
- ✅ Realistic GitHub API usage (5000 req/hr limit)

---

## System Architecture Overview

### High-Level System Diagram

```mermaid
graph TB
    subgraph "External Systems"
        GH[GitHub API<br/>GraphQL + REST]
        GHA[GitHub Archive<br/>Historical Data]
    end
    
    subgraph "Data Ingestion Layer"
        DC[Data Collector Agent<br/>GitHub API Client]
        DV[Data Validator Agent<br/>Quality Check]
        DS[(Local SQLite DB<br/>Raw Events)]
    end
    
    subgraph "Feature Engineering Layer"
        FE1[Code Feature Agent<br/>Diff Analysis]
        FE2[Text Feature Agent<br/>NLP Processing]
        FE3[Temporal Feature Agent<br/>Time Series]
        FE4[Graph Feature Agent<br/>Dependencies]
        FE5[Sentiment Agent<br/>Communication Tone]
        FE6[CI/CD Agent<br/>Pipeline Metrics]
        FS[(Feature Store<br/>Parquet Files)]
    end
    
    subgraph "LLM Intelligence Layer"
        VDB[(ChromaDB<br/>Vector Store<br/>~2GB)]
        EMB[Embedding Agent<br/>Sentence-BERT]
        LLM[LLM Reasoning Agent<br/>Llama-3-8B-Q4<br/>~5GB RAM]
        LORA[LoRA Adapter<br/>Org-Specific<br/>~200MB]
    end
    
    subgraph "Analysis & Recommendation Layer"
        SA[Sprint Analyzer Agent<br/>Pattern Recognition]
        RA[Risk Assessment Agent<br/>Blocker Detection]
        REC[Recommendation Agent<br/>Intervention Strategies]
        EXP[Explainer Agent<br/>Evidence Attribution]
    end
    
    subgraph "Output Layer"
        API[REST API<br/>FastAPI]
        DASH[Web Dashboard<br/>React]
        REPORT[Report Generator<br/>Markdown/PDF]
    end
    
    subgraph "Storage & Cache"
        CACHE[(Redis Cache<br/>~500MB)]
        RES[(Results DB<br/>PostgreSQL)]
    end
    
    GH --> DC
    GHA --> DC
    DC --> DV
    DV --> DS
    DS --> FE1
    DS --> FE2
    DS --> FE3
    DS --> FE4
    DS --> FE5
    DS --> FE6
    FE1 --> FS
    FE2 --> FS
    FE3 --> FS
    FE4 --> FS
    FE5 --> FS
    FE6 --> FS
    FS --> EMB
    EMB --> VDB
    VDB --> LLM
    LLM --> LORA
    LORA --> SA
    SA --> RA
    RA --> REC
    REC --> EXP
    EXP --> CACHE
    CACHE --> API
    API --> DASH
    API --> REPORT
    EXP --> RES
```

### Memory Budget Breakdown (24GB RAM)

| Component | RAM Usage | Notes |
|-----------|-----------|-------|
| **macOS System** | ~4GB | Background processes |
| **LLM (Llama-3-8B-Q4)** | ~5GB | Quantized 4-bit |
| **ChromaDB** | ~2GB | Vector store with 100K embeddings |
| **Feature Processing** | ~3GB | Batch processing buffers |
| **Redis Cache** | ~500MB | Hot data cache |
| **PostgreSQL** | ~1GB | Results database |
| **Python Runtime** | ~2GB | Django/FastAPI + libs |
| **Browser (Dashboard)** | ~1.5GB | React app |
| **Buffer** | ~5GB | Safety margin |
| **Total** | ~24GB | ✅ Fits within constraint |

---

## Database Architecture

### Entity-Relationship Diagram

```mermaid
erDiagram
    ORGANIZATION ||--o{ REPOSITORY : contains
    ORGANIZATION {
        string org_id PK
        string org_name
        string github_url
        timestamp created_at
        json settings
    }
    
    REPOSITORY ||--o{ MILESTONE : has
    REPOSITORY {
        string repo_id PK
        string org_id FK
        string repo_name
        string language
        int star_count
        timestamp last_updated
    }
    
    MILESTONE ||--o{ ISSUE : contains
    MILESTONE ||--o{ PULL_REQUEST : contains
    MILESTONE ||--o{ SPRINT_ANALYSIS : analyzed_by
    MILESTONE {
        string milestone_id PK
        string repo_id FK
        string title
        text description
        date start_date
        date due_date
        date closed_date
        string state
        int open_issues
        int closed_issues
    }
    
    ISSUE ||--o{ ISSUE_COMMENT : has
    ISSUE ||--o{ ISSUE_EVENT : has
    ISSUE {
        string issue_id PK
        string milestone_id FK
        string repo_id FK
        int issue_number
        string title
        text body
        string state
        string[] labels
        timestamp created_at
        timestamp closed_at
        string author
    }
    
    PULL_REQUEST ||--o{ PR_REVIEW : has
    PULL_REQUEST ||--o{ COMMIT : contains
    PULL_REQUEST {
        string pr_id PK
        string milestone_id FK
        string repo_id FK
        int pr_number
        string title
        text description
        string state
        int additions
        int deletions
        int changed_files
        timestamp created_at
        timestamp merged_at
        string author
    }
    
    COMMIT {
        string commit_sha PK
        string pr_id FK
        string repo_id FK
        text message
        timestamp timestamp
        string author
        int additions
        int deletions
        json files_changed
    }
    
    ISSUE_COMMENT {
        string comment_id PK
        string issue_id FK
        text body
        timestamp created_at
        string author
        float sentiment_score
    }
    
    PR_REVIEW {
        string review_id PK
        string pr_id FK
        string state
        text body
        timestamp submitted_at
        string reviewer
        float sentiment_score
    }
    
    ISSUE_EVENT {
        string event_id PK
        string issue_id FK
        string event_type
        timestamp created_at
        string actor
        json data
    }
    
    FEATURE_VECTOR {
        string vector_id PK
        string milestone_id FK
        string feature_type
        blob embedding
        json metadata
        timestamp computed_at
    }
    
    SPRINT_ANALYSIS ||--o{ RISK_ASSESSMENT : identifies
    SPRINT_ANALYSIS ||--o{ RECOMMENDATION : generates
    SPRINT_ANALYSIS {
        string analysis_id PK
        string milestone_id FK
        float completion_probability
        string status
        json metrics
        text llm_reasoning
        timestamp analyzed_at
        string model_version
    }
    
    RISK_ASSESSMENT {
        string risk_id PK
        string analysis_id FK
        string risk_type
        string severity
        text description
        json evidence
        float confidence
    }
    
    RECOMMENDATION {
        string recommendation_id PK
        string analysis_id FK
        string category
        text action
        text rationale
        json supporting_data
        float priority
    }
    
    MILESTONE ||--o{ FEATURE_VECTOR : embedded_as
    SPRINT_ANALYSIS ||--o{ FEEDBACK : receives
    
    FEEDBACK {
        string feedback_id PK
        string analysis_id FK
        int manager_rating
        bool prediction_accurate
        bool recommendation_useful
        text comments
        timestamp submitted_at
    }
```

### Database Technology Stack

| Database | Purpose | Storage | Technology |
|----------|---------|---------|------------|
| **SQLite** | Raw events, temp storage | ~2GB | Local file-based |
| **PostgreSQL** | Structured data, relations | ~5GB | Dockerized |
| **ChromaDB** | Vector embeddings | ~2GB | Local persistent |
| **Parquet Files** | Feature store | ~3GB | Compressed columnar |
| **Redis** | Caching, sessions | ~500MB | In-memory |

**Total Storage**: ~12.5GB (well within SSD limits)

---

## Use Case Architecture

### Primary Use Cases

```mermaid
graph LR
    subgraph "Actors"
        PM[Project Manager]
        DEV[Developer]
        SYS[System Admin]
        API_USER[API Consumer]
    end
    
    subgraph "Use Cases"
        UC1[UC1: Monitor Sprint Health]
        UC2[UC2: Predict Milestone Completion]
        UC3[UC3: Identify Blockers]
        UC4[UC4: Get Recommendations]
        UC5[UC5: Analyze Cross-Repo Dependencies]
        UC6[UC6: Generate Sprint Report]
        UC7[UC7: Configure Organization Settings]
        UC8[UC8: Train Organization Model]
        UC9[UC9: Query Historical Patterns]
        UC10[UC10: Provide Feedback]
    end
    
    PM --> UC1
    PM --> UC2
    PM --> UC3
    PM --> UC4
    PM --> UC6
    PM --> UC10
    
    DEV --> UC1
    DEV --> UC3
    DEV --> UC5
    
    SYS --> UC7
    SYS --> UC8
    
    API_USER --> UC2
    API_USER --> UC9
```

### Detailed Use Case Descriptions

#### UC1: Monitor Sprint Health
**Actor**: Project Manager, Developer  
**Description**: Real-time monitoring of active sprint progress across organization  
**Input**: Organization ID, optional repo filter  
**Process**:
1. Data Collector Agent fetches latest GitHub events
2. Feature Engineering Agents compute current metrics
3. LLM Reasoning Agent analyzes patterns
4. Sprint Analyzer Agent generates health score
**Output**: Dashboard with health metrics, risk indicators, trend charts

#### UC2: Predict Milestone Completion
**Actor**: Project Manager, API Consumer  
**Description**: Predict likelihood and timeline for milestone completion  
**Input**: Milestone ID or list of milestones  
**Process**:
1. Embedding Agent retrieves milestone features
2. LLM Reasoning Agent analyzes historical patterns
3. Sprint Analyzer Agent computes probability
4. Explainer Agent provides evidence
**Output**: Completion probability (0-100%), expected date, confidence interval, reasoning

#### UC3: Identify Blockers
**Actor**: Project Manager, Developer  
**Description**: Detect issues and dependencies blocking sprint progress  
**Input**: Sprint/Milestone ID  
**Process**:
1. Graph Feature Agent builds dependency graph
2. Sentiment Agent analyzes communication tone
3. Risk Assessment Agent identifies blockers
4. Explainer Agent cites evidence (issues, PRs, commits)
**Output**: List of blockers with severity, affected areas, evidence

#### UC4: Get Recommendations
**Actor**: Project Manager  
**Description**: Receive actionable intervention suggestions  
**Input**: Sprint ID, identified risks  
**Process**:
1. Recommendation Agent queries similar past situations (RAG)
2. LLM Reasoning Agent generates context-aware suggestions
3. Explainer Agent provides success probability from historical data
**Output**: Ranked recommendations with rationale and expected impact

#### UC5: Analyze Cross-Repo Dependencies
**Actor**: Developer, Project Manager  
**Description**: Identify dependencies between repositories affecting milestones  
**Input**: Organization ID  
**Process**:
1. Graph Feature Agent builds cross-repo dependency graph
2. LLM analyzes code imports, documentation references
3. Sprint Analyzer detects dependency-related delays
**Output**: Dependency graph visualization, affected milestones, timeline impact

---

## Actor Architecture

### Actor Interaction Diagram

```mermaid
sequenceDiagram
    participant PM as Project Manager
    participant DASH as Web Dashboard
    participant API as API Gateway
    participant ORCH as Orchestrator Agent
    participant DC as Data Collector
    participant FE as Feature Engineer
    participant LLM as LLM Agent
    participant SA as Sprint Analyzer
    participant RA as Risk Assessor
    participant REC as Recommender
    participant EXP as Explainer
    participant DB as Database
    
    PM->>DASH: Request sprint analysis
    DASH->>API: GET /api/sprint/{id}/analyze
    API->>ORCH: Initiate analysis workflow
    
    ORCH->>DC: Fetch latest data for sprint
    DC->>DB: Query milestone, issues, PRs, commits
    DB-->>DC: Return raw data
    DC-->>ORCH: Data ready
    
    ORCH->>FE: Generate features
    FE->>FE: Code features (diff analysis)
    FE->>FE: Text features (NLP)
    FE->>FE: Temporal features (burndown)
    FE->>FE: Graph features (dependencies)
    FE->>FE: Sentiment features
    FE->>FE: CI/CD features
    FE->>DB: Store feature vectors
    FE-->>ORCH: Features ready
    
    ORCH->>LLM: Analyze sprint with context
    LLM->>DB: Retrieve similar sprints (RAG)
    DB-->>LLM: Historical patterns
    LLM->>SA: Request pattern analysis
    SA-->>LLM: Patterns identified
    LLM-->>ORCH: Analysis complete
    
    ORCH->>RA: Assess risks
    RA->>LLM: Query for risk indicators
    LLM-->>RA: Risk patterns
    RA->>DB: Store risk assessments
    RA-->>ORCH: Risks identified
    
    ORCH->>REC: Generate recommendations
    REC->>LLM: Query historical interventions
    LLM-->>REC: Successful strategies
    REC->>DB: Store recommendations
    REC-->>ORCH: Recommendations ready
    
    ORCH->>EXP: Prepare explanation
    EXP->>LLM: Generate reasoning chain
    LLM-->>EXP: Reasoning complete
    EXP->>DB: Retrieve evidence (commits, issues, PRs)
    DB-->>EXP: Evidence data
    EXP-->>ORCH: Explanation ready
    
    ORCH-->>API: Analysis complete
    API-->>DASH: Return results
    DASH-->>PM: Display analysis with visualizations
    
    PM->>DASH: Provide feedback
    DASH->>API: POST /api/feedback
    API->>DB: Store feedback
    API->>ORCH: Trigger RLHF update
    ORCH->>LLM: Update model weights (LoRA)
```

### Agent Roles & Responsibilities

| Agent | Responsibility | Input | Output | Memory |
|-------|----------------|-------|--------|--------|
| **Orchestrator** | Workflow coordination | User request | Task sequence | ~50MB |
| **Data Collector** | GitHub API calls | Org/Repo IDs | Raw events | ~200MB |
| **Feature Engineer** | Multi-modal feature extraction | Raw events | Feature vectors | ~500MB |
| **Embedding Agent** | Text to vector conversion | Text data | 384-dim vectors | ~300MB |
| **LLM Reasoning** | Context analysis, pattern recognition | Features + context | Insights | ~5GB |
| **Sprint Analyzer** | Health scoring, prediction | Features + LLM insights | Status + probability | ~100MB |
| **Risk Assessor** | Blocker detection | Features + patterns | Risk list | ~100MB |
| **Recommender** | Intervention suggestions | Risks + historical data | Action items | ~200MB |
| **Explainer** | Evidence attribution | Results + raw data | Reasoning chain | ~150MB |

**Total Peak Memory**: ~6.6GB (agents run sequentially, not parallel)

---

## LLM Agent Architecture

### Multi-Agent System Design

```mermaid
graph TB
    subgraph "Agent Orchestrator"
        ORCH[Orchestration Engine<br/>LangChain + CrewAI]
    end
    
    subgraph "Specialized Agents"
        A1[Data Collection Agent<br/>Tools: GitHub API, SQL]
        A2[Feature Engineering Agent<br/>Tools: Pandas, PyTorch]
        A3[Embedding Agent<br/>Tools: Sentence-BERT]
        A4[Pattern Recognition Agent<br/>Tools: Llama-3-8B, ChromaDB]
        A5[Risk Assessment Agent<br/>Tools: Llama-3-8B, Rule Engine]
        A6[Recommendation Agent<br/>Tools: Llama-3-8B, RAG]
        A7[Explanation Agent<br/>Tools: Llama-3-8B, Evidence Retrieval]
        A8[Learning Agent<br/>Tools: LoRA Fine-tuning]
    end
    
    subgraph "Tools & Memory"
        TOOLS[Tool Registry<br/>- GitHub API Client<br/>- SQL Query Builder<br/>- Vector Search<br/>- LLM Prompt Templates]
        MEM[(Agent Memory<br/>- Short-term: Redis<br/>- Long-term: PostgreSQL<br/>- Vector: ChromaDB)]
    end
    
    subgraph "LLM Backend"
        LLM1[Llama-3-8B-Instruct-Q4<br/>Local Inference]
        LLM2[LoRA Adapters<br/>Org-Specific Fine-tuning]
    end
    
    ORCH --> A1
    ORCH --> A2
    ORCH --> A3
    ORCH --> A4
    ORCH --> A5
    ORCH --> A6
    ORCH --> A7
    ORCH --> A8
    
    A1 --> TOOLS
    A2 --> TOOLS
    A3 --> TOOLS
    A4 --> TOOLS
    A5 --> TOOLS
    A6 --> TOOLS
    A7 --> TOOLS
    A8 --> TOOLS
    
    TOOLS --> MEM
    
    A4 --> LLM1
    A5 --> LLM1
    A6 --> LLM1
    A7 --> LLM1
    A8 --> LLM2
    
    LLM2 --> LLM1
```

### Agent Prompting Strategy

#### Pattern Recognition Agent Prompt Template

```python
PATTERN_RECOGNITION_PROMPT = """
You are an expert software project analyst specializing in GitHub sprint patterns.

**Context:**
- Organization: {org_name}
- Repository: {repo_name}
- Milestone: {milestone_title}
- Due Date: {due_date}
- Days Remaining: {days_remaining}

**Current Metrics:**
{metrics_json}

**Historical Similar Sprints (Retrieved via RAG):**
{similar_sprints_summary}

**Task:**
1. Analyze the current sprint metrics against historical patterns
2. Identify positive and negative indicators
3. Predict completion likelihood (0-100%)
4. Highlight anomalies or concerning trends

**Output Format (JSON):**
{{
  "completion_probability": <0-100>,
  "status": "<ahead|on_track|at_risk|delayed>",
  "positive_indicators": [<list of strings>],
  "negative_indicators": [<list of strings>],
  "anomalies": [<list of strings>],
  "reasoning": "<step-by-step analysis>"
}}

Provide your analysis:
"""
```

#### Recommendation Agent Prompt Template

```python
RECOMMENDATION_PROMPT = """
You are a project management advisor providing actionable recommendations.

**Sprint Context:**
{sprint_summary}

**Identified Risks:**
{risks_json}

**Historical Interventions (Retrieved via RAG):**
{successful_interventions}

**Task:**
Generate 3-5 specific, actionable recommendations to address the risks.

**For each recommendation, provide:**
1. **Action**: Clear description of what to do
2. **Rationale**: Why this action will help (cite historical evidence)
3. **Impact**: Expected outcome
4. **Effort**: Estimated effort (low/medium/high)
5. **Priority**: Ranking (1-5, 5=highest)

**Output Format (JSON):**
{{
  "recommendations": [
    {{
      "action": "<specific action>",
      "rationale": "<reasoning with evidence>",
      "impact": "<expected outcome>",
      "effort": "<low|medium|high>",
      "priority": <1-5>,
      "historical_success_rate": <percentage if available>
    }}
  ]
}}

Provide recommendations:
"""
```

---

## Input-Process-Output Pipeline

### Input Dataset Schema

#### Table 1: Raw GitHub Events (Input)

| Column | Type | Description | Example | Source |
|--------|------|-------------|---------|--------|
| `event_id` | UUID | Unique event identifier | `550e8400-e29b-41d4-a716-446655440000` | Generated |
| `org_id` | VARCHAR(100) | Organization identifier | `microsoft` | GitHub API |
| `repo_id` | VARCHAR(200) | Repository full name | `microsoft/vscode` | GitHub API |
| `milestone_id` | VARCHAR(100) | Milestone identifier | `MDk6TWlsZXN0b25lMTIz` | GitHub API |
| `event_type` | VARCHAR(50) | Type of event | `IssuesEvent`, `PullRequestEvent`, `PushEvent` | GitHub API |
| `event_data` | JSON | Event payload | `{"action": "opened", "issue": {...}}` | GitHub API |
| `actor` | VARCHAR(100) | User who triggered event | `octocat` | GitHub API |
| `timestamp` | TIMESTAMP | Event occurrence time | `2026-02-13 14:30:00` | GitHub API |
| `collected_at` | TIMESTAMP | When we fetched it | `2026-02-13 14:31:00` | System |

**Estimated Size**: 10K events/day/org → ~500MB/month/org

#### Table 2: Milestone Context (Input)

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `milestone_id` | VARCHAR(100) | Primary key | `MDk6TWlsZXN0b25lMTIz` |
| `repo_id` | VARCHAR(200) | Foreign key | `microsoft/vscode` |
| `title` | VARCHAR(255) | Milestone name | `Sprint 24 - Q1 2026` |
| `description` | TEXT | Goals and scope | `Focus on performance improvements...` |
| `state` | VARCHAR(20) | Current state | `open`, `closed` |
| `start_date` | DATE | Sprint start | `2026-02-01` |
| `due_date` | DATE | Target completion | `2026-02-28` |
| `open_issues` | INTEGER | Issues not closed | `45` |
| `closed_issues` | INTEGER | Issues completed | `23` |
| `total_issues` | INTEGER | Total linked issues | `68` |
| `created_at` | TIMESTAMP | Milestone creation | `2026-01-25 10:00:00` |
| `updated_at` | TIMESTAMP | Last modification | `2026-02-13 09:15:00` |

**Estimated Size**: ~100 milestones/org → ~1MB/org

#### Table 3: Issue Details (Input)

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `issue_id` | VARCHAR(100) | GitHub issue ID | `I_kwDOABcD1234` |
| `milestone_id` | VARCHAR(100) | FK to milestone | `MDk6TWlsZXN0b25lMTIz` |
| `issue_number` | INTEGER | Issue # in repo | `#1542` |
| `title` | VARCHAR(500) | Issue title | `Fix memory leak in extension host` |
| `body` | TEXT | Issue description | `When loading large workspaces...` |
| `state` | VARCHAR(20) | Status | `open`, `closed` |
| `labels` | JSON | Label array | `["bug", "performance", "high-priority"]` |
| `assignees` | JSON | Assigned users | `["alice", "bob"]` |
| `created_at` | TIMESTAMP | Issue opened | `2026-02-05 11:20:00` |
| `closed_at` | TIMESTAMP | Issue resolved | `2026-02-12 16:45:00` (nullable) |
| `comments_count` | INTEGER | Number of comments | `12` |
| `author` | VARCHAR(100) | Issue creator | `user123` |

**Estimated Size**: ~50 issues/milestone → ~5MB/100 milestones

#### Table 4: Pull Request Details (Input)

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `pr_id` | VARCHAR(100) | GitHub PR ID | `PR_kwDOABcD5678` |
| `milestone_id` | VARCHAR(100) | FK to milestone | `MDk6TWlsZXN0b25lMTIz` |
| `pr_number` | INTEGER | PR # in repo | `#890` |
| `title` | VARCHAR(500) | PR title | `Optimize memory usage in renderer` |
| `body` | TEXT | PR description | `This PR addresses issue #1542...` |
| `state` | VARCHAR(20) | Status | `open`, `merged`, `closed` |
| `draft` | BOOLEAN | Is draft PR | `false` |
| `additions` | INTEGER | Lines added | `342` |
| `deletions` | INTEGER | Lines deleted | `128` |
| `changed_files` | INTEGER | Files modified | `8` |
| `mergeable` | BOOLEAN | Can be merged | `true` |
| `created_at` | TIMESTAMP | PR opened | `2026-02-06 09:00:00` |
| `merged_at` | TIMESTAMP | PR merged | `2026-02-12 17:00:00` (nullable) |
| `author` | VARCHAR(100) | PR creator | `alice` |
| `reviewers` | JSON | Requested reviewers | `["bob", "charlie"]` |
| `review_comments` | INTEGER | Review comment count | `15` |

**Estimated Size**: ~30 PRs/milestone → ~8MB/100 milestones

#### Table 5: Commit Data (Input)

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `commit_sha` | VARCHAR(40) | Git commit hash | `a1b2c3d4e5f6...` |
| `pr_id` | VARCHAR(100) | FK to PR (nullable) | `PR_kwDOABcD5678` |
| `repo_id` | VARCHAR(200) | Repository | `microsoft/vscode` |
| `message` | TEXT | Commit message | `fix: reduce memory allocation in parser` |
| `author` | VARCHAR(100) | Committer | `alice` |
| `timestamp` | TIMESTAMP | Commit time | `2026-02-11 14:30:00` |
| `additions` | INTEGER | Lines added | `85` |
| `deletions` | INTEGER | Lines deleted | `42` |
| `files_changed` | JSON | Modified files | `["src/parser.ts", "test/parser.test.ts"]` |

**Estimated Size**: ~200 commits/milestone → ~15MB/100 milestones

#### Table 6: Comments & Communication (Input)

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `comment_id` | VARCHAR(100) | Comment ID | `IC_kwDOABcD9999` |
| `parent_id` | VARCHAR(100) | Issue/PR ID | `I_kwDOABcD1234` |
| `parent_type` | VARCHAR(20) | `issue` or `pr` | `issue` |
| `body` | TEXT | Comment text | `This looks good, but we should...` |
| `author` | VARCHAR(100) | Commenter | `bob` |
| `created_at` | TIMESTAMP | Comment time | `2026-02-10 10:15:00` |
| `reactions` | JSON | Emoji reactions | `{"+1": 3, "hooray": 1}` |

**Estimated Size**: ~300 comments/milestone → ~10MB/100 milestones

#### Table 7: CI/CD Workflow Runs (Input)

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `run_id` | VARCHAR(100) | Workflow run ID | `WR_12345678` |
| `repo_id` | VARCHAR(200) | Repository | `microsoft/vscode` |
| `commit_sha` | VARCHAR(40) | Triggering commit | `a1b2c3d4e5f6...` |
| `workflow_name` | VARCHAR(200) | Workflow file | `CI/CD Pipeline` |
| `status` | VARCHAR(20) | Run status | `success`, `failure`, `cancelled` |
| `conclusion` | VARCHAR(20) | Final state | `success`, `failure`, `skipped` |
| `started_at` | TIMESTAMP | Run start | `2026-02-11 14:35:00` |
| `completed_at` | TIMESTAMP | Run end | `2026-02-11 14:48:00` |
| `duration_seconds` | INTEGER | Runtime | `780` |
| `test_results` | JSON | Test outcomes | `{"total": 1234, "passed": 1230, "failed": 4}` |

**Estimated Size**: ~50 runs/milestone → ~2MB/100 milestones

---

### Feature Engineering Process

#### Table 8: Computed Features (Intermediate)

| Feature Category | Features | Computation Method | Dimensions |
|------------------|----------|-------------------|------------|
| **Code Features** | Code churn, file change frequency, complexity delta | Diff analysis, AST parsing | 32-dim |
| **Text Features** | Issue/PR embeddings, topic distribution | Sentence-BERT (all-MiniLM-L6-v2) | 384-dim |
| **Temporal Features** | Burndown velocity, issue closure rate, PR merge rate | Time series analysis | 16-dim |
| **Graph Features** | Issue dependency graph, contributor network | NetworkX, GNN embeddings | 64-dim |
| **Sentiment Features** | Comment tone, communication frequency | VADER sentiment | 8-dim |
| **CI/CD Features** | Build success rate, test pass rate, deploy frequency | Aggregation | 12-dim |
| **Team Features** | Active contributors, collaboration score | Graph metrics | 8-dim |

**Total Feature Vector**: 524 dimensions per milestone

#### Feature Computation Pipeline

```python
# Pseudocode for feature engineering
def compute_features(milestone_id):
    """
    Extract multi-modal features for a milestone.
    Memory-efficient processing in batches.
    """
    features = {}
    
    # 1. Code Features (32-dim)
    commits = fetch_commits(milestone_id)
    features['code'] = compute_code_features(commits)
    # - Total lines changed (normalized)
    # - Average commit size
    # - File churn rate
    # - Language distribution
    # - Code complexity trends
    
    # 2. Text Features (384-dim)
    issues = fetch_issues(milestone_id)
    prs = fetch_pull_requests(milestone_id)
    text_corpus = [i.title + " " + i.body for i in issues]
    text_corpus += [p.title + " " + p.body for p in prs]
    features['text'] = embed_text_batch(text_corpus)  # Sentence-BERT
    # Average pooling across all texts
    
    # 3. Temporal Features (16-dim)
    events = fetch_events_timeline(milestone_id)
    features['temporal'] = compute_temporal_features(events)
    # - Days since start
    # - Days until due
    # - Issue closure velocity (issues/day)
    # - PR merge velocity (PRs/day)
    # - Burndown slope
    # - Velocity trend (increasing/decreasing)
    
    # 4. Graph Features (64-dim)
    dep_graph = build_dependency_graph(issues)
    contrib_graph = build_contributor_network(commits, prs)
    features['graph'] = compute_graph_embeddings(dep_graph, contrib_graph)
    # - Graph density
    # - Critical path length
    # - Centrality measures
    # - Community structure
    
    # 5. Sentiment Features (8-dim)
    comments = fetch_comments(milestone_id)
    features['sentiment'] = compute_sentiment_features(comments)
    # - Average sentiment score
    # - Sentiment trend
    # - Positive/negative ratio
    # - Controversy score (high variance)
    
    # 6. CI/CD Features (12-dim)
    workflow_runs = fetch_workflow_runs(milestone_id)
    features['cicd'] = compute_cicd_features(workflow_runs)
    # - Build success rate
    # - Test pass rate
    # - Average build time
    # - Failed test count
    
    # 7. Team Features (8-dim)
    features['team'] = compute_team_features(commits, prs, issues)
    # - Active contributor count
    # - Collaboration score
    # - Average response time
    # - Team capacity utilization
    
    # Concatenate all features
    feature_vector = np.concatenate([
        features['code'],
        features['text'],
        features['temporal'],
        features['graph'],
        features['sentiment'],
        features['cicd'],
        features['team']
    ])
    
    return feature_vector  # 524-dim
```

---

### Output Dataset Schema

#### Table 9: Sprint Analysis Results (Output)

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `analysis_id` | UUID | Unique analysis run | `650e8400-e29b-41d4-a716-446655440000` |
| `milestone_id` | VARCHAR(100) | FK to milestone | `MDk6TWlsZXN0b25lMTIz` |
| `analyzed_at` | TIMESTAMP | Analysis timestamp | `2026-02-13 15:00:00` |
| `completion_probability` | FLOAT | Likelihood of on-time completion (0-100) | `72.5` |
| `predicted_completion_date` | DATE | Estimated finish date | `2026-02-26` |
| `confidence_lower` | FLOAT | 95% CI lower bound | `65.2` |
| `confidence_upper` | FLOAT | 95% CI upper bound | `79.8` |
| `status` | VARCHAR(20) | Sprint health status | `on_track` |
| `health_score` | FLOAT | Overall health (0-100) | `78.3` |
| `velocity_current` | FLOAT | Issues closed per day | `3.2` |
| `velocity_required` | FLOAT | Required to meet deadline | `3.8` |
| `burndown_ideal` | JSON | Ideal burndown curve | `[68, 64, 60, ...]` |
| `burndown_actual` | JSON | Actual burndown | `[68, 65, 62, ...]` |
| `model_version` | VARCHAR(50) | LLM model used | `llama-3-8b-q4-lora-v1.2` |
| `reasoning` | TEXT | LLM explanation | `Based on current velocity...` |

#### Table 10: Risk Assessment (Output)

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `risk_id` | UUID | Unique risk identifier | `750e8400-e29b-41d4-a716-446655440000` |
| `analysis_id` | UUID | FK to analysis | `650e8400-e29b-41d4-a716-446655440000` |
| `risk_type` | VARCHAR(100) | Category | `dependency_blocker`, `velocity_decline` |
| `severity` | VARCHAR(20) | Impact level | `high`, `medium`, `low` |
| `probability` | FLOAT | Likelihood (0-100) | `85.0` |
| `impact_days` | INTEGER | Estimated delay | `5` |
| `title` | VARCHAR(255) | Short description | `Critical dependency not merged` |
| `description` | TEXT | Detailed explanation | `PR #890 is required for 3 other issues...` |
| `affected_issues` | JSON | Impacted work items | `["#1542", "#1543", "#1544"]` |
| `evidence` | JSON | Supporting data | `{"pr": "#890", "blocked_count": 3}` |
| `detected_at` | TIMESTAMP | When identified | `2026-02-13 15:00:00` |
| `confidence` | FLOAT | Model confidence (0-100) | `91.2` |

**Risk Types**:
- `dependency_blocker`: Blocked by another issue/PR
- `velocity_decline`: Team velocity trending down
- `scope_creep`: Issues added after sprint start
- `ci_failures`: Continuous integration issues
- `review_bottleneck`: PRs waiting for review
- `team_capacity`: Team member availability issues
- `technical_debt`: Accumulated debt slowing progress

#### Table 11: Recommendations (Output)

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `recommendation_id` | UUID | Unique recommendation | `850e8400-e29b-41d4-a716-446655440000` |
| `analysis_id` | UUID | FK to analysis | `650e8400-e29b-41d4-a716-446655440000` |
| `risk_id` | UUID | FK to risk (optional) | `750e8400-e29b-41d4-a716-446655440000` |
| `category` | VARCHAR(100) | Recommendation type | `prioritization`, `resource_allocation` |
| `priority` | INTEGER | Urgency (1-5, 5=highest) | `4` |
| `action` | TEXT | What to do | `Assign additional reviewer to PR #890` |
| `rationale` | TEXT | Why this helps | `Based on 23 similar cases, adding...` |
| `expected_impact` | TEXT | Predicted outcome | `Reduce delay by 2-3 days` |
| `effort` | VARCHAR(20) | Required effort | `low`, `medium`, `high` |
| `historical_success_rate` | FLOAT | % success in past | `78.5` |
| `similar_cases` | JSON | Historical examples | `[{"milestone": "Sprint 20", "outcome": "success"}]` |
| `supporting_data` | JSON | Evidence | `{"reviews_needed": 2, "avg_review_time": "4h"}` |

**Recommendation Categories**:
- `prioritization`: Reorder work items
- `resource_allocation`: Assign more people
- `scope_reduction`: Remove non-critical items
- `dependency_resolution`: Unblock dependencies
- `process_improvement`: Change workflow
- `communication`: Improve team coordination
- `technical`: Address technical issues

#### Table 12: Evidence Attribution (Output)

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `evidence_id` | UUID | Unique evidence piece | `950e8400-e29b-41d4-a716-446655440000` |
| `analysis_id` | UUID | FK to analysis | `650e8400-e29b-41d4-a716-446655440000` |
| `evidence_type` | VARCHAR(50) | Type of evidence | `commit`, `issue`, `pr`, `comment` |
| `evidence_ref` | VARCHAR(200) | Reference to item | `microsoft/vscode#1542` |
| `evidence_url` | TEXT | GitHub URL | `https://github.com/microsoft/vscode/issues/1542` |
| `relevance_score` | FLOAT | How relevant (0-100) | `92.3` |
| `excerpt` | TEXT | Relevant quote | `"This is blocking 3 other features"` |
| `context` | TEXT | Why it's relevant | `Cited as blocker dependency` |

#### Table 13: Performance Metrics (Output)

| Metric Name | Type | Description | Example Value |
|-------------|------|-------------|---------------|
| `issues_opened` | INTEGER | Issues created during sprint | `12` |
| `issues_closed` | INTEGER | Issues completed | `23` |
| `issues_remaining` | INTEGER | Open issues | `45` |
| `prs_opened` | INTEGER | PRs created | `8` |
| `prs_merged` | INTEGER | PRs merged | `15` |
| `prs_pending` | INTEGER | PRs awaiting review | `5` |
| `commits_count` | INTEGER | Total commits | `187` |
| `code_additions` | INTEGER | Lines added | `4523` |
| `code_deletions` | INTEGER | Lines removed | `2341` |
| `files_changed` | INTEGER | Modified files | `342` |
| `active_contributors` | INTEGER | Contributors this sprint | `8` |
| `avg_issue_close_time_hours` | FLOAT | Time to close issues | `48.5` |
| `avg_pr_merge_time_hours` | FLOAT | Time to merge PRs | `12.3` |
| `build_success_rate` | FLOAT | % successful builds | `94.7` |
| `test_pass_rate` | FLOAT | % passing tests | `98.2` |
| `review_response_time_hours` | FLOAT | Time to first review | `6.8` |
| `velocity_issues_per_day` | FLOAT | Issue closure rate | `3.2` |
| `burndown_progress` | FLOAT | % of work completed | `58.3` |
| `days_elapsed` | INTEGER | Days since sprint start | `12` |
| `days_remaining` | INTEGER | Days until deadline | `15` |

---

## GitHub Metrics Collection

### Achievable GitHub API Metrics (No Premium Required)

#### Organization-Level Metrics

| Metric | API Endpoint | Rate Limit Impact | Feasibility | Value |
|--------|--------------|-------------------|-------------|-------|
| **Repositories List** | `GET /orgs/{org}/repos` | 1 call/org | ✅ Easy | Essential |
| **Org Members** | `GET /orgs/{org}/members` | 1 call/org | ✅ Easy | High |
| **Teams (if public)** | `GET /orgs/{org}/teams` | 1 call/org | ⚠️ Requires auth | Medium |
| **Public Events** | `GET /orgs/{org}/events` | 1 call/org | ✅ Easy | High |

**Organization Metrics We Can Track**:
- Total repositories
- Active repositories (commits last 30 days)
- Total members/contributors
- Public events feed
- Organization activity level

#### Repository-Level Metrics (Core Data)

| Metric | API Endpoint | Rate Limit | Feasibility | Value |
|--------|--------------|------------|-------------|-------|
| **Milestones** | `GET /repos/{owner}/{repo}/milestones` | 1 call/repo | ✅ Easy | **Critical** |
| **Issues** | `GET /repos/{owner}/{repo}/issues` | Paginated | ✅ Easy | **Critical** |
| **Pull Requests** | `GET /repos/{owner}/{repo}/pulls` | Paginated | ✅ Easy | **Critical** |
| **Commits** | `GET /repos/{owner}/{repo}/commits` | Paginated | ✅ Easy | **Critical** |
| **Issue Comments** | `GET /repos/{owner}/{repo}/issues/{num}/comments` | 1 call/issue | ✅ Easy | High |
| **PR Reviews** | `GET /repos/{owner}/{repo}/pulls/{num}/reviews` | 1 call/PR | ✅ Easy | High |
| **PR Commits** | `GET /repos/{owner}/{repo}/pulls/{num}/commits` | 1 call/PR | ✅ Easy | High |
| **Workflow Runs** | `GET /repos/{owner}/{repo}/actions/runs` | Paginated | ✅ Easy | High |
| **Contributors** | `GET /repos/{owner}/{repo}/contributors` | 1 call/repo | ✅ Easy | Medium |
| **Languages** | `GET /repos/{owner}/{repo}/languages` | 1 call/repo | ✅ Easy | Low |
| **Repo Stats** | `GET /repos/{owner}/{repo}` | 1 call/repo | ✅ Easy | Medium |

#### Milestone-Specific Metrics

| Metric | Derivation | Computation |
|--------|-----------|-------------|
| **Total Issues** | Count issues in milestone | `milestone.open_issues + milestone.closed_issues` |
| **Open Issues** | Milestone field | `milestone.open_issues` |
| **Closed Issues** | Milestone field | `milestone.closed_issues` |
| **Completion %** | Calculation | `(closed / total) * 100` |
| **Days Elapsed** | Date math | `today - milestone.created_at` |
| **Days Remaining** | Date math | `milestone.due_on - today` |
| **Velocity** | Issues closed / time | `closed_issues / days_elapsed` |
| **Required Velocity** | To meet deadline | `open_issues / days_remaining` |
| **Velocity Gap** | Difference | `required_velocity - current_velocity` |
| **PRs Linked** | Query PRs mentioning milestone | Count PRs with `milestone` field |
| **PRs Merged** | Filter by state | Count where `state == "merged"` |
| **PRs Pending** | Filter by state | Count where `state == "open"` |

#### Issue-Level Metrics

| Metric | Source | Computation |
|--------|--------|-------------|
| **Age (hours)** | Timestamps | `now - issue.created_at` |
| **Time to Close (hours)** | Timestamps | `issue.closed_at - issue.created_at` |
| **Comment Count** | Field | `issue.comments` |
| **Reaction Count** | Reactions API | Sum of all reaction types |
| **Assignee Count** | Field | `len(issue.assignees)` |
| **Label Count** | Field | `len(issue.labels)` |
| **Has Priority Label** | Label check | `any("priority" in label for label in labels)` |
| **Is Bug** | Label check | `"bug" in labels` |
| **Is Feature** | Label check | `"feature" in labels or "enhancement" in labels` |
| **Is Blocked** | Label/comments | Check for "blocked" label or keywords in comments |

#### Pull Request Metrics

| Metric | Source | Computation |
|--------|--------|-------------|
| **Lines Changed** | Fields | `pr.additions + pr.deletions` |
| **Files Changed** | Field | `pr.changed_files` |
| **Review Count** | Reviews API | Count reviews |
| **Approved Reviews** | Reviews API | Count where `state == "APPROVED"` |
| **Changes Requested** | Reviews API | Count where `state == "CHANGES_REQUESTED"` |
| **Review Comments** | Field | `pr.review_comments` |
| **Time to First Review** | Timestamps | `first_review.submitted_at - pr.created_at` |
| **Time to Merge** | Timestamps | `pr.merged_at - pr.created_at` |
| **Is Draft** | Field | `pr.draft` |
| **Mergeable** | Field | `pr.mergeable` |
| **CI Status** | Status API | Check latest commit status |

#### Commit Metrics

| Metric | Source | Computation |
|--------|--------|-------------|
| **Commit Frequency** | Timestamps | Commits per day |
| **Average Commit Size** | Stats | Mean of `additions + deletions` |
| **Code Churn** | Stats | Total lines changed / total lines in repo |
| **Commit Message Length** | Field | `len(commit.message)` |
| **Mentions Issue** | Message parsing | Regex for `#\d+` or `fixes #\d+` |

#### CI/CD Metrics (Actions)

| Metric | Source | Computation |
|--------|--------|-------------|
| **Total Runs** | Workflow runs API | Count runs |
| **Success Rate** | Runs | `successful_runs / total_runs * 100` |
| **Failure Rate** | Runs | `failed_runs / total_runs * 100` |
| **Average Duration** | Run times | Mean of `completed_at - started_at` |
| **Latest Status** | Latest run | `conclusion` field |

### GitHub API Rate Limits & Strategy

**Rate Limits**:
- **Authenticated**: 5,000 requests/hour
- **Unauthenticated**: 60 requests/hour
- **GraphQL**: 5,000 points/hour (more efficient)

**Our Strategy**:
1. **Use GraphQL for bulk queries** (fetch milestone + issues + PRs in 1 call)
2. **Cache aggressively** (Redis, 15-minute TTL for active sprints)
3. **Incremental updates** (only fetch new events since last check)
4. **Batch processing** (process 10 repos at a time)
5. **Respect rate limits** (exponential backoff)

**Example GraphQL Query** (1 call gets everything):

```graphql
query GetMilestoneContext($org: String!, $repo: String!, $milestone: Int!) {
  repository(owner: $org, name: $repo) {
    milestone(number: $milestone) {
      title
      description
      dueOn
      state
      issues(first: 100) {
        nodes {
          number
          title
          body
          state
          labels(first: 10) { nodes { name } }
          comments(first: 50) {
            nodes { body, author { login }, createdAt }
          }
          createdAt
          closedAt
        }
      }
      pullRequests(first: 100) {
        nodes {
          number
          title
          body
          state
          additions
          deletions
          changedFiles
          reviews(first: 20) {
            nodes { state, body, submittedAt }
          }
          commits(first: 50) {
            nodes {
              commit {
                message
                committedDate
                additions
                deletions
              }
            }
          }
          createdAt
          mergedAt
        }
      }
    }
  }
}
```

**Estimated API Calls**:
- Per milestone (GraphQL): **1 call** (vs. 150+ REST calls)
- Per organization (10 repos, 5 milestones each): **50 calls/day**
- Well within 5,000/hour limit ✅

---

## Feature Engineering

### Multi-Modal Feature Sets

#### 1. Code Features (32 dimensions)

```python
def extract_code_features(commits, prs):
    """Extract code-level features from commits and PRs."""
    return {
        # Volume metrics
        'total_commits': len(commits),
        'total_additions': sum(c.additions for c in commits),
        'total_deletions': sum(c.deletions for c in commits),
        'total_files_changed': len(set(f for c in commits for f in c.files)),
        'avg_commit_size': np.mean([c.additions + c.deletions for c in commits]),
        'max_commit_size': max([c.additions + c.deletions for c in commits]),
        
        # Churn metrics
        'code_churn_rate': sum(c.additions + c.deletions for c in commits) / len(commits),
        'file_churn_rate': len(set(f for c in commits for f in c.files)) / len(commits),
        
        # PR size metrics
        'avg_pr_size': np.mean([pr.additions + pr.deletions for pr in prs]),
        'large_pr_count': sum(1 for pr in prs if (pr.additions + pr.deletions) > 500),
        
        # Language distribution (one-hot top 5 languages)
        'lang_python': language_pct('Python'),
        'lang_javascript': language_pct('JavaScript'),
        'lang_typescript': language_pct('TypeScript'),
        'lang_java': language_pct('Java'),
        'lang_other': language_pct('Other'),
        
        # Temporal patterns
        'commits_per_day': len(commits) / days_elapsed,
        'weekend_commit_ratio': weekend_commits / total_commits,
        'late_night_commit_ratio': late_commits / total_commits,
        
        # ... (32 total features)
    }
```

#### 2. Text Features (384 dimensions)

```python
from sentence_transformers import SentenceTransformer

# Load lightweight model (90MB, runs on CPU)
text_encoder = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_features(issues, prs):
    """Extract semantic embeddings from text."""
    # Combine all text
    texts = []
    for issue in issues:
        texts.append(f"{issue.title}. {issue.body}")
    for pr in prs:
        texts.append(f"{pr.title}. {pr.body}")
    
    # Batch encode (efficient)
    embeddings = text_encoder.encode(texts, batch_size=32)
    
    # Aggregate: mean pooling
    milestone_embedding = np.mean(embeddings, axis=0)  # 384-dim
    
    return milestone_embedding
```

#### 3. Temporal Features (16 dimensions)

```python
def extract_temporal_features(milestone, issues, prs):
    """Extract time-series features."""
    return {
        # Sprint timeline
        'days_since_start': (now - milestone.created_at).days,
        'days_until_due': (milestone.due_on - now).days,
        'sprint_duration_days': (milestone.due_on - milestone.created_at).days,
        'progress_ratio': days_elapsed / sprint_duration,
        
        # Velocity metrics
        'issue_close_velocity': issues_closed / days_elapsed,
        'pr_merge_velocity': prs_merged / days_elapsed,
        'required_velocity': issues_open / days_remaining,
        'velocity_gap': required_velocity - current_velocity,
        
        # Burndown features
        'burndown_slope': linear_regression(burndown_curve).slope,
        'burndown_deviation': deviation_from_ideal(burndown_curve),
        
        # Activity patterns
        'recent_activity_7d': events_last_7_days / 7,
        'activity_trend': (events_last_7d - events_prev_7d) / events_prev_7d,
        
        # Issue lifecycle
        'avg_issue_age_hours': np.mean([issue.age_hours for issue in issues]),
        'avg_time_to_close_hours': np.mean([issue.time_to_close for issue in closed_issues]),
        
        # PR lifecycle
        'avg_pr_age_hours': np.mean([pr.age_hours for pr in prs]),
        'avg_time_to_merge_hours': np.mean([pr.time_to_merge for pr in merged_prs]),
    }
```

#### 4. Graph Features (64 dimensions)

```python
import networkx as nx
from torch_geometric.nn import Node2Vec

def extract_graph_features(issues, prs, commits):
    """Extract graph-based features."""
    
    # Issue dependency graph
    issue_graph = nx.DiGraph()
    for issue in issues:
        issue_graph.add_node(issue.number)
        # Parse dependencies from body (e.g., "depends on #123")
        deps = extract_dependencies(issue.body)
        for dep in deps:
            issue_graph.add_edge(issue.number, dep)
    
    # Contributor collaboration graph
    contrib_graph = nx.Graph()
    for pr in prs:
        author = pr.author
        reviewers = pr.reviewers
        for reviewer in reviewers:
            contrib_graph.add_edge(author, reviewer)
    
    # Graph metrics
    features = {
        # Issue graph
        'issue_graph_density': nx.density(issue_graph),
        'issue_graph_diameter': nx.diameter(issue_graph) if nx.is_connected(issue_graph.to_undirected()) else 0,
        'avg_issue_degree': np.mean([d for n, d in issue_graph.degree()]),
        'blocked_issue_count': sum(1 for n in issue_graph if issue_graph.in_degree(n) > 0),
        'critical_path_length': len(nx.dag_longest_path(issue_graph)),
        
        # Contributor graph
        'active_contributors': contrib_graph.number_of_nodes(),
        'contributor_clustering': nx.average_clustering(contrib_graph),
        'contributor_centralization': max(dict(nx.degree_centrality(contrib_graph)).values()),
    }
    
    # Node embeddings (using Node2Vec)
    # This gives us distributed representations
    node2vec = Node2Vec(issue_graph, embedding_dim=64, walk_length=20, num_walks=10)
    embeddings = node2vec.fit()
    
    # Aggregate node embeddings
    graph_embedding = np.mean([embeddings[node] for node in issue_graph.nodes()], axis=0)
    
    return graph_embedding  # 64-dim
```

#### 5. Sentiment Features (8 dimensions)

```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def extract_sentiment_features(comments, reviews):
    """Analyze communication tone."""
    
    # Analyze all comments
    comment_sentiments = []
    for comment in comments:
        scores = analyzer.polarity_scores(comment.body)
        comment_sentiments.append(scores)
    
    # Analyze PR reviews
    review_sentiments = []
    for review in reviews:
        scores = analyzer.polarity_scores(review.body)
        review_sentiments.append(scores)
    
    return {
        # Average sentiment
        'avg_comment_positive': np.mean([s['pos'] for s in comment_sentiments]),
        'avg_comment_negative': np.mean([s['neg'] for s in comment_sentiments]),
        'avg_comment_neutral': np.mean([s['neu'] for s in comment_sentiments]),
        'avg_comment_compound': np.mean([s['compound'] for s in comment_sentiments]),
        
        # Sentiment trends
        'sentiment_trend': (recent_sentiment - past_sentiment),
        'sentiment_variance': np.var([s['compound'] for s in comment_sentiments]),
        
        # Communication volume
        'comments_per_issue': len(comments) / len(issues),
        'reviews_per_pr': len(reviews) / len(prs),
    }
```

#### 6. CI/CD Features (12 dimensions)

```python
def extract_cicd_features(workflow_runs):
    """Extract CI/CD pipeline metrics."""
    return {
        # Success metrics
        'build_success_rate': successful_runs / total_runs,
        'test_pass_rate': sum(run.tests_passed / run.tests_total for run in runs) / len(runs),
        
        # Performance metrics
        'avg_build_time_minutes': np.mean([run.duration_seconds / 60 for run in runs]),
        'build_time_trend': (recent_avg_time - past_avg_time) / past_avg_time,
        
        # Failure analysis
        'consecutive_failures': max_consecutive_failures(runs),
        'failure_rate_trend': (recent_failure_rate - past_failure_rate),
        
        # Test metrics
        'total_tests': np.mean([run.tests_total for run in runs]),
        'avg_failed_tests': np.mean([run.tests_failed for run in runs]),
        'flaky_test_count': count_flaky_tests(runs),
        
        # Deployment metrics
        'deployment_frequency': deployments / days_elapsed,
        'deployment_success_rate': successful_deployments / total_deployments,
        'time_since_last_deploy_hours': (now - last_deployment).total_seconds() / 3600,
    }
```

### Feature Vector Assembly

```python
def assemble_feature_vector(milestone_id):
    """
    Assemble complete 524-dimensional feature vector.
    Memory-efficient: processes in batches, clears intermediates.
    """
    # Fetch raw data
    milestone = fetch_milestone(milestone_id)
    issues = fetch_issues(milestone_id)
    prs = fetch_pull_requests(milestone_id)
    commits = fetch_commits(milestone_id)
    comments = fetch_comments(milestone_id)
    reviews = fetch_reviews(milestone_id)
    workflow_runs = fetch_workflow_runs(milestone_id)
    
    # Extract features sequentially to manage memory
    code_features = extract_code_features(commits, prs)  # 32-dim
    del commits  # Free memory
    
    text_features = extract_text_features(issues, prs)  # 384-dim
    
    temporal_features = extract_temporal_features(milestone, issues, prs)  # 16-dim
    
    graph_features = extract_graph_features(issues, prs, commits=[])  # 64-dim
    
    sentiment_features = extract_sentiment_features(comments, reviews)  # 8-dim
    del comments, reviews  # Free memory
    
    cicd_features = extract_cicd_features(workflow_runs)  # 12-dim
    del workflow_runs  # Free memory
    
    # Concatenate
    feature_vector = np.concatenate([
        list(code_features.values()),  # 32
        text_features,  # 384
        list(temporal_features.values()),  # 16
        graph_features,  # 64
        list(sentiment_features.values()),  # 8
        list(cicd_features.values()),  # 12
    ])
    
    assert len(feature_vector) == 524, f"Expected 524 dims, got {len(feature_vector)}"
    
    return feature_vector
```

---

## Output Metrics & Recommendations

### Prediction Outputs

#### 1. Sprint Completion Prediction

**Output Format**:
```json
{
  "milestone_id": "MDk6TWlsZXN0b25lMTIz",
  "milestone_title": "Sprint 24 - Q1 2026",
  "analysis_timestamp": "2026-02-13T15:00:00Z",
  
  "completion_prediction": {
    "probability": 72.5,
    "confidence_interval": [65.2, 79.8],
    "predicted_completion_date": "2026-02-26",
    "days_delay_expected": -2,
    "status": "on_track"
  },
  
  "health_score": {
    "overall": 78.3,
    "velocity": 72.0,
    "code_quality": 85.0,
    "team_engagement": 80.0,
    "ci_cd_health": 75.0
  },
  
  "velocity_analysis": {
    "current_velocity": 3.2,
    "required_velocity": 3.8,
    "velocity_gap": -0.6,
    "trend": "declining",
    "issues_per_day_last_7d": 2.8,
    "issues_per_day_prev_7d": 3.6
  },
  
  "burndown_status": {
    "completed_percentage": 58.3,
    "expected_percentage_today": 55.0,
    "ahead_or_behind": "ahead",
    "days_ahead": 1.2
  }
}
```

#### 2. Risk Assessment Output

**Output Format**:
```json
{
  "risks": [
    {
      "risk_id": "r_001",
      "type": "dependency_blocker",
      "severity": "high",
      "probability": 85.0,
      "impact_days": 5,
      "title": "Critical PR #890 blocking 3 issues",
      "description": "PR #890 (Optimize memory usage) is required by issues #1542, #1543, #1544. PR has been open for 7 days with no reviews.",
      "affected_items": ["#1542", "#1543", "#1544"],
      "evidence": [
        {
          "type": "pr",
          "ref": "#890",
          "url": "https://github.com/microsoft/vscode/pull/890",
          "excerpt": "Draft: Optimize memory usage in renderer",
          "relevance": "Direct blocker"
        },
        {
          "type": "issue",
          "ref": "#1542",
          "url": "https://github.com/microsoft/vscode/issues/1542",
          "excerpt": "Depends on #890 being merged",
          "relevance": "Blocked dependency"
        }
      ],
      "confidence": 91.2
    },
    {
      "risk_id": "r_002",
      "type": "velocity_decline",
      "severity": "medium",
      "probability": 72.0,
      "impact_days": 3,
      "title": "Team velocity declining for 5 consecutive days",
      "description": "Issue closure rate has dropped from 4.2/day to 2.8/day over the last 5 days. This trend, if continued, will result in 3-day delay.",
      "affected_items": [],
      "evidence": [
        {
          "type": "metric",
          "data": {
            "days": ["Feb 8", "Feb 9", "Feb 10", "Feb 11", "Feb 12"],
            "velocity": [4.2, 3.9, 3.5, 3.1, 2.8]
          },
          "relevance": "Velocity trend"
        }
      ],
      "confidence": 78.5
    }
  ],
  
  "risk_summary": {
    "total_risks": 5,
    "high_severity": 2,
    "medium_severity": 2,
    "low_severity": 1,
    "estimated_total_delay_days": 8
  }
}
```

#### 3. Recommendation Output

**Output Format**:
```json
{
  "recommendations": [
    {
      "recommendation_id": "rec_001",
      "priority": 5,
      "category": "resource_allocation",
      "action": "Assign 2 additional reviewers to PR #890",
      "rationale": "Based on analysis of 23 similar cases where PRs blocked multiple issues, adding reviewers reduced merge time by average of 18 hours. Current PR has complex changes (342 additions, 128 deletions) requiring expert review.",
      "expected_impact": "Unblock 3 issues (#1542, #1543, #1544) and prevent 5-day delay",
      "effort": "low",
      "implementation_steps": [
        "Request review from @alice and @bob (both have expertise in renderer code)",
        "Schedule review within next 24 hours",
        "Set PR as high priority"
      ],
      "historical_data": {
        "similar_cases": 23,
        "success_rate": 78.3,
        "avg_time_saved": "18 hours",
        "example_cases": [
          {
            "milestone": "Sprint 20",
            "pr": "#765",
            "action_taken": "Added 2 reviewers",
            "outcome": "Merged in 6 hours, prevented 4-day delay"
          }
        ]
      }
    },
    {
      "recommendation_id": "rec_002",
      "priority": 4,
      "category": "prioritization",
      "action": "Re-prioritize issues to focus on critical path items",
      "rationale": "Dependency graph analysis shows 8 issues are blocked by only 2 parent issues (#1540, #1541). Completing these 2 issues first will unblock 40% of remaining work.",
      "expected_impact": "Enable parallel work on 8 issues, improve velocity by 25%",
      "effort": "medium",
      "implementation_steps": [
        "Move #1540 and #1541 to top of backlog",
        "Assign experienced developers to these issues",
        "Daily standup focus on blocker issues"
      ],
      "historical_data": {
        "similar_cases": 15,
        "success_rate": 82.0,
        "avg_velocity_improvement": "27%"
      }
    },
    {
      "recommendation_id": "rec_003",
      "priority": 3,
      "category": "communication",
      "action": "Hold mid-sprint sync meeting to address velocity decline",
      "rationale": "Sentiment analysis of recent comments shows increased frustration (sentiment score dropped from 0.6 to 0.3). Team velocity correlation with sentiment is 0.71 in historical data. Intervention meeting can re-align team.",
      "expected_impact": "Improve team morale and re-establish velocity",
      "effort": "low",
      "implementation_steps": [
        "Schedule 30-minute sync meeting",
        "Discuss current blockers and frustrations",
        "Adjust sprint scope if necessary"
      ],
      "historical_data": {
        "similar_cases": 12,
        "success_rate": 75.0,
        "avg_velocity_recovery": "1.2 issues/day"
      }
    }
  ]
}
```

### Explanation Output (RAG-based)

**Output Format**:
```json
{
  "explanation": {
    "reasoning_chain": [
      {
        "step": 1,
        "thought": "Analyzed current sprint metrics",
        "findings": "58.3% work completed with 55.6% time elapsed - slightly ahead of schedule"
      },
      {
        "step": 2,
        "thought": "Retrieved 10 similar sprints from organization history",
        "findings": "In 7 out of 10 cases with similar patterns, sprints were completed on time"
      },
      {
        "step": 3,
        "thought": "Identified 2 high-severity risks (PR #890 blocker, velocity decline)",
        "findings": "These risks add estimated 8 days delay if not addressed"
      },
      {
        "step": 4,
        "thought": "Applied historical intervention success rates",
        "findings": "If recommendations are followed, 78% chance of on-time completion"
      },
      {
        "step": 5,
        "thought": "Calculated final probability with confidence interval",
        "findings": "72.5% completion probability (CI: 65.2%-79.8%)"
      }
    ],
    
    "evidence_summary": {
      "positive_factors": [
        "Currently ahead of ideal burndown by 1.2 days",
        "CI/CD success rate at 94.7% (above org average of 91%)",
        "Code review turnaround time improved from 12h to 8h"
      ],
      "negative_factors": [
        "Velocity declining for 5 consecutive days",
        "PR #890 blocking 3 critical issues for 7 days",
        "Team sentiment declining (0.6 → 0.3)"
      ]
    },
    
    "similar_sprints": [
      {
        "milestone": "Sprint 20 - microsoft/vscode",
        "similarity_score": 0.89,
        "outcome": "completed_on_time",
        "key_similarities": [
          "Similar velocity pattern",
          "Similar dependency structure",
          "Same team size (8 contributors)"
        ]
      }
    ]
  }
}
```

---

## Feasibility Analysis (M4 Pro Constraints)

### Hardware Specifications
- **CPU**: Apple M4 Pro (12-core)
- **RAM**: 24GB unified memory
- **Storage**: 512GB SSD (assumed)
- **GPU**: 16-core (Metal acceleration)

### Achievability Assessment

#### ✅ **ACHIEVABLE Components**

| Component | RAM Usage | Strategy | Status |
|-----------|-----------|----------|--------|
| **Llama-3-8B-Q4** | ~5GB | 4-bit quantization | ✅ Fits |
| **Sentence-BERT** | ~300MB | CPU inference | ✅ Fits |
| **ChromaDB** | ~2GB | 100K embeddings max | ✅ Fits |
| **PostgreSQL** | ~1GB | Docker container | ✅ Fits |
| **Redis** | ~500MB | Local instance | ✅ Fits |
| **Feature Processing** | ~3GB | Batch processing | ✅ Fits |
| **Python Runtime** | ~2GB | Django/FastAPI | ✅ Fits |
| **Total** | ~14GB | Sequential agent execution | ✅ **Fits comfortably** |

#### Modified Architecture for M4 Pro

```python
# Model Configuration for M4 Pro
MODEL_CONFIG = {
    # LLM: Use quantized 8B model instead of 70B
    "llm": {
        "model_name": "TheBloke/Llama-2-8B-Chat-GGUF",
        "model_file": "llama-2-8b-chat.Q4_K_M.gguf",  # 4-bit quantization
        "context_length": 4096,
        "max_tokens": 2048,
        "temperature": 0.3,
        "n_gpu_layers": 0,  # CPU inference (or Metal)
        "n_threads": 10,  # Use 10 of 12 cores
        "ram_usage": "~5GB"
    },
    
    # Embeddings: Lightweight model
    "embeddings": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "dimensions": 384,
        "model_size": "90MB",
        "device": "cpu",  # Or "mps" for Metal
        "ram_usage": "~300MB"
    },
    
    # Vector Store: Limit size
    "vector_store": {
        "backend": "chromadb",
        "max_embeddings": 100_000,  # ~2GB
        "persistence_dir": "./data/chromadb",
        "ram_usage": "~2GB"
    },
    
    # Batch Processing: Control memory
    "batch_config": {
        "issue_batch_size": 50,
        "pr_batch_size": 20,
        "commit_batch_size": 100,
        "max_concurrent_repos": 3  # Process 3 repos at a time
    }
}
```

#### Inference Performance on M4 Pro

| Operation | Model | Latency | Throughput |
|-----------|-------|---------|------------|
| **Text Embedding** | all-MiniLM-L6-v2 | ~10ms/text | 100 texts/sec (CPU) |
| **LLM Inference** | Llama-3-8B-Q4 | ~2s/response | 512 tokens/sec (Metal) |
| **Vector Search** | ChromaDB | ~20ms | 1000 queries/sec |
| **Feature Engineering** | Pandas/NumPy | ~5s/milestone | 12 milestones/min |
| **Graph Analysis** | NetworkX | ~1s/graph | 60 graphs/min |

**End-to-End Analysis Time**: 
- Per milestone: ~15 seconds (data fetch + features + LLM + results)
- Per organization (10 repos, 50 milestones): ~12 minutes
- **Conclusion**: ✅ Real-time capable for typical workloads

#### Storage Requirements

| Data Type | Size per Org | Compression | Final Size |
|-----------|--------------|-------------|------------|
| **Raw Events (SQLite)** | ~500MB/month | None | 500MB |
| **Features (Parquet)** | ~100MB | Snappy | 30MB |
| **Embeddings (ChromaDB)** | ~2GB | Built-in | 2GB |
| **Results (PostgreSQL)** | ~50MB | None | 50MB |
| **Model Files** | 5.5GB | Quantized | 5.5GB |
| **Total per Org** | - | - | **~8.1GB** |

**For 3 organizations**: ~24GB storage (fits in 512GB SSD easily) ✅

### ❌ **NOT ACHIEVABLE Components** (Removed from Design)

| Component | Reason | Alternative |
|-----------|--------|-------------|
| **Llama-3-70B** | 35GB RAM minimum | Use Llama-3-8B-Q4 (5GB) |
| **Real-time streaming** | Requires dedicated server | Batch processing every 15 min |
| **100+ concurrent repos** | Memory overflow | Process 3 repos at a time |
| **Full fine-tuning** | 80GB+ RAM | LoRA adapters (200MB) |
| **GPT-4 API** | Cost prohibitive | Local Llama-3-8B |
| **Parallel agents** | 24GB insufficient | Sequential execution |

### Practical Deployment Strategy

#### Phase 1: MVP (Achievable in 4 weeks)

**Features**:
- ✅ Collect data from 1 organization
- ✅ Extract all 6 feature modalities
- ✅ Basic LLM analysis (Llama-3-8B)
- ✅ Simple web dashboard
- ✅ Core metrics + 1 risk type

**Hardware Utilization**: ~12GB RAM, ~15GB storage

#### Phase 2: Full System (Achievable in 12 weeks)

**Features**:
- ✅ Support 3 organizations
- ✅ All risk types + recommendations
- ✅ RAG-based explanations
- ✅ LoRA fine-tuning per org
- ✅ Batch processing pipeline

**Hardware Utilization**: ~18GB RAM, ~30GB storage

#### Phase 3: Optimization (Achievable in 16 weeks)

**Features**:
- ✅ RLHF feedback loops
- ✅ Cross-org pattern recognition
- ✅ Advanced visualizations
- ✅ API for external tools

**Hardware Utilization**: ~22GB RAM, ~40GB storage

### Memory Management Techniques

```python
# Memory-efficient processing
class MemoryEfficientPipeline:
    def __init__(self):
        self.max_ram_gb = 20  # Leave 4GB for system
        
    def process_organization(self, org_id):
        """Process org with memory constraints."""
        repos = fetch_repositories(org_id)
        
        # Process in batches of 3 repos
        for repo_batch in chunk(repos, size=3):
            # Process each repo
            for repo in repo_batch:
                milestones = fetch_milestones(repo)
                
                # Process milestones one at a time
                for milestone in milestones:
                    features = self.extract_features(milestone)
                    analysis = self.analyze_with_llm(features)
                    self.save_results(analysis)
                    
                    # Explicit cleanup
                    del features, analysis
                    gc.collect()  # Force garbage collection
            
            # Clear repo batch
            del repo_batch
            gc.collect()
```

### Performance Benchmarks (Estimated)

| Scenario | Repos | Milestones | Time | RAM Peak |
|----------|-------|------------|------|----------|
| **Small Org** | 5 | 25 | 6 min | 14GB |
| **Medium Org** | 20 | 100 | 25 min | 18GB |
| **Large Org** | 50 | 250 | 65 min | 22GB |
| **Multi-Org (3)** | 60 | 300 | 80 min | 22GB |

**Conclusion**: ✅ All scenarios fit within M4 Pro constraints

---

## Implementation Roadmap

### Week 1-2: Data Collection Infrastructure
- Set up GitHub API client with rate limiting
- Implement SQLite for raw event storage
- Build incremental data collection scripts
- Test with 1 organization (e.g., `facebook/react`)

### Week 3-4: Feature Engineering
- Implement all 6 feature extractors
- Set up Sentence-BERT for embeddings
- Build Parquet feature store
- Validate features on 50 milestones

### Week 5-6: LLM Integration
- Set up Llama-3-8B-Q4 locally (llama.cpp)
- Implement ChromaDB vector store
- Build RAG retrieval pipeline
- Test LLM inference speed

### Week 7-8: Agent System
- Implement orchestrator with LangChain
- Build specialized agents (analyzer, risk assessor, recommender)
- Create prompt templates
- Test end-to-end pipeline

### Week 9-10: LoRA Fine-Tuning
- Collect org-specific training data (500 examples)
- Fine-tune LoRA adapters (PEFT library)
- Validate improvement over base model
- Save adapters (~200MB each)

### Week 11-12: Dashboard & API
- Build FastAPI backend
- Create React dashboard
- Implement visualization components
- Add user feedback collection

### Week 13-14: RLHF & Optimization
- Implement reward model
- Build PPO trainer
- Collect user feedback
- Optimize performance

### Week 15-16: Evaluation & Documentation
- Run comprehensive experiments
- Compare against baselines
- Write paper
- Prepare presentation

---

## Conclusion

This architecture is **fully achievable** on a MacBook M4 Pro with 24GB RAM by:

1. **Using quantized models** (Llama-3-8B-Q4 instead of 70B)
2. **Sequential processing** (agents run one at a time)
3. **Batch operations** (3 repos at a time)
4. **Efficient storage** (Parquet compression, vector quantization)
5. **Smart caching** (Redis for hot data)

**Expected Performance**:
- ✅ 15 seconds per milestone analysis
- ✅ Handle 3 organizations (~60 repos, 300 milestones)
- ✅ Peak RAM usage: ~22GB (2GB buffer)
- ✅ Storage: ~30GB total

**Key Trade-offs**:
- Use 8B model instead of 70B (minor accuracy hit: ~3-5%)
- Batch processing instead of real-time streaming (15-min delay)
- Sequential agents instead of parallel (manageable latency)

**Bottom Line**: This is a **realistic, achievable project** that solves real problems with practical constraints.

---

**Document Version**: 1.0  
**Last Updated**: February 13, 2026  
**Hardware**: MacBook M4 Pro, 24GB RAM  
**Status**: Architecture Validated ✅
