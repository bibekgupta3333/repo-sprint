# Work Breakdown Structure (WBS)
# Lightweight Sprint Intelligence for Small Startups

**Project Duration**: 3 Months (12 Weeks)  
**Target**: Small Startups (2-3 GitHub Repositories, 3-10 Developer Teams)  
**Last Updated**: March 24, 2026

---

## 📊 Project Status Overview

| Phase | Status | Start Date | Target End | Actual End | Completion % |
|-------|--------|------------|------------|------------|--------------|
| **Phase 1: Research & Planning** | ✅ Completed | Feb 14, 2026 | Feb 28, 2026 | Mar 1, 2026 | 100% |
| **Phase 2: Data Collection & Infrastructure** | 🟢 In Progress | Mar 1, 2026 | Mar 21, 2026 | - | 75% |
| **Phase 3: Backend Development** | ⚪ Not Started | Mar 22, 2026 | Apr 11, 2026 | - | 0% |
| **Phase 4: Agent Development** | ⚪ Not Started | Apr 12, 2026 | May 2, 2026 | - | 0% |
| **Phase 5: Frontend Development** | ⚪ Not Started | May 3, 2026 | May 16, 2026 | - | 0% |
| **Phase 6: ML Validation & Testing** | ⚪ Not Started | May 17, 2026 | May 30, 2026 | - | 0% |

**Legend**: 🟢 In Progress | ✅ Completed | ⚪ Not Started | 🟡 Blocked | 🔴 At Risk

---

## 🎯 Top 5 Research Objectives

### Objective 1: Instant Setup Multi-Repository Sprint Intelligence
**Gap Addressed**: Existing tools require weeks of setup and historical data  
**Target**: <10 minute setup, analyze 2-3 repositories with zero historical data  
**Success Metric**: >85% accuracy using only synthetic + 1 week of real data

### Objective 2: Ultra-Fast Blocker Detection with RAG
**Gap Addressed**: Current tools have 15-30 min delays, no explanations  
**Target**: <30 second latency from GitHub event to explained recommendation  
**Success Metric**: >80% small team acceptance, <30s p95 latency

### Objective 3: Synthetic Data for Zero-History Startups
**Gap Addressed**: New startups have no historical data to train on  
**Target**: Generate 5K+ realistic sprint scenarios matching small team patterns  
**Success Metric**: Cold-start accuracy within 5% of 6-month historical training

### Objective 4: Laptop-Scale LLM Architecture
**Gap Addressed**: Existing solutions need 64GB+ RAM, cloud GPUs  
**Target**: Run full system on 16GB laptop with local Ollama LLM  
**Success Metric**: <16GB peak RAM, <30GB storage, <30s inference

### Objective 5: Small Team Explainability
**Gap Addressed**: Black-box predictions don't work for 3-person teams  
**Target**: Every recommendation cites specific commits/issues/PRs  
**Success Metric**: >90% recommendations have GitHub evidence, >80% trust score

---

## Phase 1: Research & Planning (Weeks 1-2) - 🟢 35% Complete

### 1.1 Literature Review & Gap Analysis
**Duration**: Week 1 (Feb 14-20, 2026)  
**Owner**: Research Lead  
**Status**: ✅ **COMPLETED**

#### Tasks:
- [x] **1.1.1** Review 50+ research papers on GitHub analytics
  - **Status**: ✅ Completed (Feb 13, 2026)
  - **Output**: `doc/research/gap_similar_research.md`
  - **Notes**: Analyzed papers across 5 domains

- [x] **1.1.2** Identify 10 critical research gaps
  - **Status**: ✅ Completed (Feb 13, 2026)
  - **Output**: Gap analysis in `gap_similar_research.md`
  - **Key Gaps**: Project-level LLM, real-time analysis, explainability

- [x] **1.1.3** Define research objectives and hypotheses
  - **Status**: ✅ Completed (Feb 13, 2026)
  - **Output**: `doc/thesis_proposal.md`
  - **Hypotheses**: 5 testable hypotheses defined

- [ ] **1.1.4** Collect similar research papers (PDF/links)
  - **Status**: 🟡 Blocked - Need access to ACM/IEEE digital library
  - **Target**: Store in `doc/research/papers/`
  - **Action Required**: Set up institutional access or use arXiv alternatives

### 1.2 Architecture Design
**Duration**: Week 1-2 (Feb 14-27, 2026)  
**Owner**: System Architect  
**Status**: 🟢 **IN PROGRESS** (25% complete)

#### Tasks:
- [x] **1.2.1** Design overall LLM agent architecture
  - **Status**: ✅ Completed (Feb 13, 2026)
  - **Output**: `doc/experiments/LLM Agentic Architecture for Organization-Level Sprint Intelligence/llm_agentic_architecture.md`
  - **Agents**: 9 specialized agents defined

- [x] **1.2.2** Design database schema
  - **Status**: ✅ Completed (Feb 13, 2026)
  - **Output**: Embedded in `llm_agentic_architecture.md`
  - **Tables**: 11 tables (7 input, 4 output)

- [ ] **1.2.3** Create system architecture diagrams
  - **Status**: 🟢 In Progress (This task)
  - **Target**: `doc/architecture/system_architecture.md`
  - **Deliverables**: Mermaid diagrams for system, backend, frontend, deployment

- [ ] **1.2.4** Design backend API architecture (FastAPI)
  - **Status**: ⚪ Not Started
  - **Target**: `doc/architecture/backend_architecture.md`
  - **Components**: FastAPI routes, LangGraph orchestrator, RAG pipeline

- [ ] **1.2.5** Design frontend architecture (Streamlit)
  - **Status**: ⚪ Not Started
  - **Target**: `doc/architecture/frontend_architecture.md`
  - **Components**: Dashboard components, real-time updates

### 1.3 Data Strategy
**Duration**: Week 2 (Feb 21-27, 2026)  
**Owner**: Data Engineer  
**Status**: 🟢 **IN PROGRESS** (40% complete)

#### Tasks:
- [x] **1.3.1** Download GitHub Archive sample data
  - **Status**: ✅ Completed (Feb 14, 2026)
  - **Output**: `data/raw/2026-02-07-*.jsonl` (24 files)
  - **Size**: ~500MB raw events

- [x] **1.3.2** Create data processing pipeline
  - **Status**: ✅ Completed (Feb 14, 2026)
  - **Output**: `scripts/prepare_embeddings.py`
  - **Pipeline**: Extract → Embed → Store in ChromaDB

- [x] **1.3.3** Set up local ChromaDB vector store
  - **Status**: ✅ Completed (Feb 14, 2026)
  - **Output**: `data/processed/chromadb/`
  - **Size**: ~100 documents embedded

- [ ] **1.3.4** Design synthetic data generation strategy
  - **Status**: ⚪ Not Started
  - **Target**: `doc/research/synthetic_data_strategy.md`
  - **Goal**: Generate 5K sprint scenarios using LLM

- [ ] **1.3.5** Identify target startup repos for case studies
  - **Status**: ⚪ Not Started
  - **Target**: 2-3 real startup repositories (10-50 contributors)
  - **Criteria**: Active sprints, public repos, diverse tech stacks

### 1.4 Technical Setup & Planning
**Duration**: Week 2 (Feb 21-27, 2026)  
**Owner**: DevOps/Full Stack  
**Status**: 🟢 **IN PROGRESS** (15% complete)

#### Tasks:
- [ ] **1.4.1** Create Work Breakdown Structure (WBS)
  - **Status**: 🟢 In Progress (This document)
  - **Output**: `doc/planning/WBS.md`
  - **Details**: Includes status tracking and timeline

- [ ] **1.4.2** Set up monorepo structure (Turborepo-like)
  - **Status**: ⚪ Not Started
  - **Target**: Root-level `apps/` and `packages/` folders
  - **Structure**: `apps/backend`, `apps/frontend`, `packages/shared`

- [ ] **1.4.3** Create editor configuration (.editorconfig)
  - **Status**: ⚪ Not Started
  - **Output**: `.editorconfig`, `.prettierrc`, `.eslintrc`
  - **Goal**: Consistent code formatting across editors

- [ ] **1.4.4** Create Cursor AI rules
  - **Status**: ⚪ Not Started
  - **Output**: `.cursorrules`
  - **Goal**: Consistent AI-assisted code generation

- [ ] **1.4.5** Set up Docker compose for local development
  - **Status**: ⚪ Not Started
  - **Output**: `docker-compose.yml`
  - **Services**: PostgreSQL, ChromaDB, Ollama, FastAPI, Streamlit

### 1.5 Research Methodology
**Duration**: Week 2 (Feb 21-27, 2026)  
**Owner**: Research Lead  
**Status**: ⚪ **NOT STARTED**

#### Tasks:
- [ ] **1.5.1** Define experimental design
  - **Status**: ⚪ Not Started
  - **Output**: `doc/research/experimental_design.md`
  - **Components**: Control groups, validation splits, metrics

- [ ] **1.5.2** Create evaluation metrics framework
  - **Status**: ⚪ Not Started
  - **Output**: `doc/research/evaluation_metrics.md`
  - **Metrics**: Accuracy, F1, latency, trust score, BLEU (for explanations)

- [ ] **1.5.3** Design ML validation pipeline
  - **Status**: ⚪ Not Started
  - **Output**: `doc/architecture/ml_validation_architecture.md`
  - **Methods**: Cross-validation, A/B testing, human evaluation

- [ ] **1.5.4** Plan ablation studies
  - **Status**: ⚪ Not Started
  - **Output**: `doc/research/ablation_study_plan.md`
  - **Questions**: Impact of each agent, RAG vs. non-RAG, embedding models

---

## Phase 2: Data Collection & Infrastructure (Weeks 3-5) - 🟢 75% Complete

### 2.1 Infrastructure Setup
**Duration**: Week 3 (Mar 1-7, 2026)  
**Owner**: DevOps  
**Status**: 🟢 **IN PROGRESS** (60% complete)

#### Tasks:
- [ ] **2.1.1** Set up PostgreSQL database with schema
  - **Status**: ⚪ Not Started
  - **Output**: Docker container, initialized schema
  - **Tables**: Organizations, Repositories, Milestones, Issues, PRs, Commits, etc.

- [ ] **2.1.2** Set up ChromaDB for vector storage
  - **Status**: ⚪ Not Started
  - **Output**: Persistent ChromaDB instance
  - **Collections**: Issues, PRs, Commits, Documentation

- [ ] **2.1.3** Set up local Ollama with Llama-3-8B
  - **Status**: ⚪ Not Started
  - **Output**: Running Ollama service
  - **Model**: Llama-3-8B-Instruct (quantized)

- [ ] **2.1.4** Set up embedding model (Sentence-BERT)
  - **Status**: ⚪ Not Started
  - **Output**: Local embedding service
  - **Model**: all-MiniLM-L6-v2 (384-dim, free)

- [ ] **2.1.5** Create Docker Compose orchestration
  - **Status**: ⚪ Not Started
  - **Output**: `docker-compose.yml` with all services
  - **Services**: PostgreSQL, ChromaDB, Ollama, Redis, FastAPI

### 2.2 GitHub Data Collection
**Duration**: Week 3-4 (Mar 1-14, 2026)  
**Owner**: Data Engineer  
**Status**: ✅ **COMPLETED** (Mar 24, 2026)

#### Tasks:
- [x] **2.2.1** Implement GitHub API data collector
  - **Status**: ✅ Completed (Mar 2026)
  - **Output**: `src/scrapper/github.py` — full API scraper (issues, PRs, commits, diffs)
  - **Data**: Issues, PRs, commits, diffs, contributors

- [x] **2.2.2** Implement local git data collector (replaces/extends API)
  - **Status**: ✅ Completed (Mar 24, 2026)
  - **Output**: `src/scrapper/local_git.py` — `LocalGitScraper`
  - **Strategy**: Hybrid — local git for commits/diffs (no rate limits), API for issues/PRs
  - **Data**: 65,730 commits, 65,715 diffs, 44,411 PRs/CLs, 17,936 issues from golang/go
  - **Notes**: `--offline` mode extracts PRs/issues from commit trailers (Change-Id, Reviewed-on, Fixes #N)

- [x] **2.2.3** Implement batch diff extraction (git log --numstat -p)
  - **Status**: ✅ Completed (Mar 24, 2026)
  - **Output**: `LocalGitScraper.get_commit_diffs_batch()` — single git process for all diffs
  - **Performance**: 65K commits extracted in ~40 seconds (vs. 30+ min with per-commit API)

- [ ] **2.2.4** Implement real-time GitHub webhook handler
  - **Status**: ⚪ Not Started
  - **Output**: FastAPI webhook endpoint
  - **Events**: Issues, PRs, commits, comments

- [ ] **2.2.5** Set up CI/CD metrics collection (GitHub Actions)
  - **Status**: ⚪ Not Started
  - **Output**: Workflow run data
  - **Metrics**: Build status, test results, duration

### 2.3 Synthetic Data Generation
**Duration**: Week 4-5 (Mar 8-21, 2026)  
**Owner**: ML Engineer  
**Status**: 🟢 **IN PROGRESS** (25% complete)

#### Tasks:
- [x] **2.3.1** Implement sprint preprocessor (real data → sprints)
  - **Status**: ✅ Completed
  - **Output**: `src/data/preprocessor.py` — `SprintPreprocessor`
  - **Result**: 475 sprints from 65K golang/go commits (2-week sprint windows)

- [x] **2.3.2** Implement Chroma document formatter
  - **Status**: ✅ Completed
  - **Output**: `src/data/formatter.py` — `ChromaFormatter`
  - **Result**: 128,552 flat Chroma docs + 475 sprint-aligned docs from golang/go

- [ ] **2.3.3** Develop LLM-based synthetic sprint scenario generator
  - **Status**: ⚪ Not Started
  - **Output**: `src/data/synthetic_generator.py`
  - **Target**: 5K realistic sprint scenarios

- [ ] **2.3.4** Generate synthetic GitHub events
  - **Status**: ⚪ Not Started
  - **Output**: `data/synthetic/sprints/` (JSONL format)
  - **Diversity**: Success, failure, delayed, blocked scenarios

- [ ] **2.3.5** Validate synthetic data realism
  - **Status**: ⚪ Not Started
  - **Method**: Statistical comparison with real data
  - **Metrics**: Event distribution, temporal patterns, vocabulary overlap

- [ ] **2.3.6** Create augmented training dataset
  - **Status**: ⚪ Not Started
  - **Output**: Combined real + synthetic data
  - **Split**: 70% train, 15% val, 15% test

### 2.4 Feature Engineering Pipeline
**Duration**: Week 5 (Mar 15-21, 2026)  
**Owner**: ML Engineer  
**Status**: 🟢 **IN PROGRESS** (50% complete)

#### Tasks:
- [x] **2.4.1** Implement code feature extractor
  - **Status**: ✅ Completed
  - **Output**: `scripts/_core/processor.py` — `Processor`
  - **Features**: Code churn (+9.3M additions, -5.7M deletions), language breakdown (Go, C, Assembly…), files changed per sprint, commit velocity

- [x] **2.4.2** Implement repository analyzer
  - **Status**: ✅ Completed
  - **Output**: `scripts/_core/analyzer.py` — `Analyzer`
  - **Features**: Unique authors (2,830), PR merge rate (100%), issue open/close ratio, additions/deletions per PR

- [x] **2.4.3** Implement sprint-level feature aggregation
  - **Status**: ✅ Completed
  - **Output**: `src/data/preprocessor.py` — groups commits into 2-week sprints with velocity, risk labels, contributor counts
  - **Features**: Sprint velocity, commit frequency, PR throughput, issue resolution rate

- [ ] **2.4.4** Implement text embedding pipeline
  - **Status**: ⚪ Not Started
  - **Output**: `apps/backend/services/feature_engineering/text_features.py`
  - **Model**: Sentence-BERT embeddings (384-dim)

- [ ] **2.4.5** Implement temporal feature extractor
  - **Status**: ⚪ Not Started
  - **Output**: `apps/backend/services/feature_engineering/temporal_features.py`
  - **Features**: Burndown velocity, cycle time, lead time

- [ ] **2.4.6** Implement graph feature extractor
  - **Status**: ⚪ Not Started
  - **Output**: `apps/backend/services/feature_engineering/graph_features.py`
  - **Features**: Dependency graph, contributor network, PageRank

- [ ] **2.4.7** Implement sentiment analysis
  - **Status**: ⚪ Not Started
  - **Output**: `apps/backend/services/feature_engineering/sentiment_features.py`
  - **Model**: VADER or lightweight transformer

- [ ] **2.4.8** Implement CI/CD metrics processor
  - **Status**: ⚪ Not Started
  - **Output**: `apps/backend/services/feature_engineering/cicd_features.py`
  - **Features**: Build success rate, test coverage, deploy frequency

---

## Phase 3: Backend Development (Weeks 6-8) - ⚪ 0% Complete

### 3.1 FastAPI Core Setup
**Duration**: Week 6 (Mar 22-28, 2026)  
**Owner**: Backend Developer  
**Status**: ⚪ **NOT STARTED**

#### Tasks:
- [ ] **3.1.1** Initialize FastAPI project structure
  - **Status**: ⚪ Not Started
  - **Output**: `apps/backend/` with proper structure
  - **Structure**: `api/`, `services/`, `core/`, `utils/`, `models/`

- [ ] **3.1.2** Implement database models (SQLAlchemy)
  - **Status**: ⚪ Not Started
  - **Output**: `apps/backend/models/` (ORM models)
  - **Models**: Organization, Repository, Milestone, Issue, PR, etc.

- [ ] **3.1.3** Create API routes (REST endpoints)
  - **Status**: ⚪ Not Started
  - **Output**: `apps/backend/api/routes/`
  - **Endpoints**: `/organizations`, `/milestones`, `/analysis`, `/recommendations`

- [ ] **3.1.4** Implement authentication & authorization
  - **Status**: ⚪ Not Started
  - **Output**: JWT-based auth
  - **Security**: GitHub OAuth integration

- [ ] **3.1.5** Set up API documentation (OpenAPI/Swagger)
  - **Status**: ⚪ Not Started
  - **Output**: Auto-generated docs at `/docs`

### 3.2 RAG Pipeline Implementation
**Duration**: Week 6-7 (Mar 22 - Apr 4, 2026)  
**Owner**: ML Engineer  
**Status**: ⚪ **NOT STARTED**

#### Tasks:
- [ ] **3.2.1** Implement document indexing service
  - **Status**: ⚪ Not Started
  - **Output**: `apps/backend/services/rag/indexer.py`
  - **Function**: Index GitHub events into ChromaDB

- [ ] **3.2.2** Implement semantic search/retrieval
  - **Status**: ⚪ Not Started
  - **Output**: `apps/backend/services/rag/retriever.py`
  - **Method**: Similarity search with re-ranking

- [ ] **3.2.3** Implement context builder for LLM
  - **Status**: ⚪ Not Started
  - **Output**: `apps/backend/services/rag/context_builder.py`
  - **Function**: Construct prompts with retrieved evidence

- [ ] **3.2.4** Implement citation/evidence tracker
  - **Status**: ⚪ Not Started
  - **Output**: `apps/backend/services/rag/citation_tracker.py`
  - **Function**: Link recommendations to source GitHub items

### 3.3 LLM Integration
**Duration**: Week 7 (Mar 29 - Apr 4, 2026)  
**Owner**: ML Engineer  
**Status**: ⚪ **NOT STARTED**

#### Tasks:
- [ ] **3.3.1** Implement Ollama client wrapper
  - **Status**: ⚪ Not Started
  - **Output**: `apps/backend/services/llm/ollama_client.py`
  - **Features**: Streaming, retry logic, error handling

- [ ] **3.3.2** Create prompt templates library
  - **Status**: ⚪ Not Started
  - **Output**: `apps/backend/services/llm/prompts/`
  - **Templates**: Pattern recognition, recommendation, risk assessment

- [ ] **3.3.3** Implement LLM response parser
  - **Status**: ⚪ Not Started
  - **Output**: `apps/backend/services/llm/parser.py`
  - **Function**: Extract structured data from LLM outputs

- [ ] **3.3.4** Implement caching layer (Redis)
  - **Status**: ⚪ Not Started
  - **Output**: `apps/backend/services/cache.py`
  - **Purpose**: Cache frequent LLM queries

### 3.4 Database Services
**Duration**: Week 8 (Apr 5-11, 2026)  
**Owner**: Backend Developer  
**Status**: ⚪ **NOT STARTED**

#### Tasks:
- [ ] **3.4.1** Implement CRUD services for all entities
  - **Status**: ⚪ Not Started
  - **Output**: `apps/backend/services/crud/`
  - **Entities**: Organization, Milestone, Issue, PR, Analysis, Risk

- [ ] **3.4.2** Implement query optimization
  - **Status**: ⚪ Not Started
  - **Output**: Database indexes, query tuning
  - **Goal**: <100ms query latency

- [ ] **3.4.3** Implement data migration scripts
  - **Status**: ⚪ Not Started
  - **Output**: `apps/backend/migrations/`
  - **Tool**: Alembic for schema versioning

---

## Phase 4: Agent Development (Weeks 9-10) - ⚪ 0% Complete

### 4.1 LangGraph Orchestrator
**Duration**: Week 9 (Apr 12-18, 2026)  
**Owner**: ML Engineer  
**Status**: ⚪ **NOT STARTED**

#### Tasks:
- [ ] **4.1.1** Design LangGraph workflow
  - **Status**: ⚪ Not Started
  - **Output**: `apps/backend/services/agents/graph.py`
  - **Flow**: Data Collector → Feature Engineer → LLM Reasoning → Analyzer → Recommender

- [ ] **4.1.2** Implement agent state management
  - **Status**: ⚪ Not Started
  - **Output**: State persistence between agent calls
  - **Storage**: Redis for session state

- [ ] **4.1.3** Implement agent orchestration logic
  - **Status**: ⚪ Not Started
  - **Output**: Sequential and parallel agent execution
  - **Goal**: <60 second end-to-end latency

### 4.2 Specialized Agents
**Duration**: Week 9-10 (Apr 12-25, 2026)  
**Owner**: ML Engineer  
**Status**: ⚪ **NOT STARTED**

#### Tasks:
- [ ] **4.2.1** Implement Data Collector Agent
  - **Status**: ⚪ Not Started
  - **Output**: `apps/backend/services/agents/data_collector.py`
  - **Function**: Fetch latest GitHub events

- [ ] **4.2.2** Implement Feature Engineering Agent
  - **Status**: ⚪ Not Started
  - **Output**: `apps/backend/services/agents/feature_engineer.py`
  - **Function**: Compute multi-modal features

- [ ] **4.2.3** Implement LLM Reasoning Agent
  - **Status**: ⚪ Not Started
  - **Output**: `apps/backend/services/agents/llm_reasoner.py`
  - **Function**: Analyze patterns with context

- [ ] **4.2.4** Implement Sprint Analyzer Agent
  - **Status**: ⚪ Not Started
  - **Output**: `apps/backend/services/agents/sprint_analyzer.py`
  - **Function**: Compute health score and predictions

- [ ] **4.2.5** Implement Risk Assessor Agent
  - **Status**: ⚪ Not Started
  - **Output**: `apps/backend/services/agents/risk_assessor.py`
  - **Function**: Detect blockers and risks

- [ ] **4.2.6** Implement Recommender Agent
  - **Status**: ⚪ Not Started
  - **Output**: `apps/backend/services/agents/recommender.py`
  - **Function**: Generate actionable interventions

- [ ] **4.2.7** Implement Explainer Agent
  - **Status**: ⚪ Not Started
  - **Output**: `apps/backend/services/agents/explainer.py`
  - **Function**: Provide evidence and reasoning chains

### 4.3 Real-Time Processing
**Duration**: Week 10 (Apr 19-25, 2026)  
**Owner**: Backend Developer  
**Status**: ⚪ **NOT STARTED**

#### Tasks:
- [ ] **4.3.1** Implement event queue (Redis Streams)
  - **Status**: ⚪ Not Started
  - **Output**: `apps/backend/services/event_queue.py`
  - **Purpose**: Buffer incoming GitHub events

- [ ] **4.3.2** Implement background workers
  - **Status**: ⚪ Not Started
  - **Output**: Celery or FastAPI BackgroundTasks
  - **Function**: Process events asynchronously

- [ ] **4.3.3** Implement WebSocket for real-time updates
  - **Status**: ⚪ Not Started
  - **Output**: WebSocket endpoint  
  - **Use Case**: Push analysis results to frontend instantly

---

## Phase 5: Frontend Development (Weeks 11-12) - ⚪ 0% Complete

### 5.1 Streamlit App Setup
**Duration**: Week 11 (May 3-9, 2026)  
**Owner**: Frontend Developer  
**Status**: ⚪ **NOT STARTED**

#### Tasks:
- [ ] **5.1.1** Initialize Streamlit project
  - **Status**: ⚪ Not Started
  - **Output**: `apps/frontend/` structure
  - **Structure**: `pages/`, `components/`, `utils/`, `config/`

- [ ] **5.1.2** Create navigation and routing
  - **Status**: ⚪ Not Started
  - **Output**: Multi-page Streamlit app
  - **Pages**: Dashboard, Repository View, Milestone Analysis, Settings

- [ ] **5.1.3** Implement authentication UI
  - **Status**: ⚪ Not Started
  - **Output**: Login page with GitHub OAuth
  - **Integration**: Connect to FastAPI auth endpoints

### 5.2 Dashboard Components
**Duration**: Week 11-12 (May 3-16, 2026)  
**Owner**: Frontend Developer  
**Status**: ⚪ **NOT STARTED**

#### Tasks:
- [ ] **5.2.1** Create Organization Overview Dashboard
  - **Status**: ⚪ Not Started
  - **Output**: `pages/1_Dashboard.py`
  - **Components**: Active sprints, health scores, trend charts

- [ ] **5.2.2** Create Milestone Analysis View
  - **Status**: ⚪ Not Started
  - **Output**: `pages/2_Milestone_Analysis.py`
  - **Components**: Burndown chart, predictions, risk list

- [ ] **5.2.3** Create Cross-Repo Dependency View
  - **Status**: ⚪ Not Started
  - **Output**: `pages/3_Dependencies.py`
  - **Components**: Dependency graph visualization (Plotly/NetworkX)

- [ ] **5.2.4** Create Recommendations View
  - **Status**: ⚪ Not Started
  - **Output**: `pages/4_Recommendations.py`
  - **Components**: Actionable recommendations with evidence

- [ ] **5.2.5** Create Settings & Configuration
  - **Status**: ⚪ Not Started
  - **Output**: `pages/5_Settings.py`
  - **Features**: Repository selection, notification preferences

### 5.3 Visualization & UX
**Duration**: Week 12 (May 10-16, 2026)  
**Owner**: Frontend Developer  
**Status**: ⚪ **NOT STARTED**

#### Tasks:
- [ ] **5.3.1** Implement real-time data updates
  - **Status**: ⚪ Not Started
  - **Output**: Auto-refresh or WebSocket integration
  - **Frequency**: Every 30 seconds

- [ ] **5.3.2** Create interactive charts (Plotly)
  - **Status**: ⚪ Not Started
  - **Charts**: Burndown, velocity, risk timeline, dependency graph

- [ ] **5.3.3** Implement responsive design
  - **Status**: ⚪ Not Started
  - **Target**: Mobile-friendly layout
  - **Tool**: Streamlit responsive components

- [ ] **5.3.4** Add loading states and error handling
  - **Status**: ⚪ Not Started
  - **Output**: Spinners, error messages, retry buttons

---

## Phase 6: ML Validation & Testing (Weeks 13-14) - ⚪ 0% Complete

### 6.1 ML Model Validation
**Duration**: Week 13 (May 17-23, 2026)  
**Owner**: ML Engineer  
**Status**: ⚪ **NOT STARTED**

#### Tasks:
- [ ] **6.1.1** Implement evaluation pipeline
  - **Status**: ⚪ Not Started
  - **Output**: `apps/backend/evaluation/eval_pipeline.py`
  - **Metrics**: Accuracy, Precision, Recall, F1, MAE, RMSE

- [ ] **6.1.2** Perform k-fold cross-validation
  - **Status**: ⚪ Not Started
  - **Output**: Cross-validation results
  - **Method**: 5-fold temporal split

- [ ] **6.1.3** Conduct ablation studies
  - **Status**: ⚪ Not Started
  - **Output**: `doc/research/ablation_results.md`
  - **Tests**: Remove each agent, RAG vs. no-RAG, embedding models

- [ ] **6.1.4** Perform error analysis
  - **Status**: ⚪ Not Started
  - **Output**: `doc/research/error_analysis.md`
  - **Focus**: False positives, false negatives, edge cases

- [ ] **6.1.5** Benchmark latency and resource usage
  - **Status**: ⚪ Not Started
  - **Metrics**: End-to-end latency, RAM usage, CPU usage
  - **Goal**: <60s latency, <16GB RAM

### 6.2 Unit & Integration Testing
**Duration**: Week 13-14 (May 17-30, 2026)  
**Owner**: Full Stack Developer  
**Status**: ⚪ **NOT STARTED**

#### Tasks:
- [ ] **6.2.1** Write backend unit tests (pytest)
  - **Status**: ⚪ Not Started
  - **Output**: `apps/backend/tests/`
  - **Coverage Target**: >80%

- [ ] **6.2.2** Write integration tests for API endpoints
  - **Status**: ⚪ Not Started
  - **Output**: `apps/backend/tests/integration/`
  - **Tool**: pytest + httpx

- [ ] **6.2.3** Write agent workflow tests
  - **Status**: ⚪ Not Started
  - **Output**: `apps/backend/tests/agents/`
  - **Focus**: LangGraph flows, agent outputs

- [ ] **6.2.4** Write frontend tests (Streamlit test framework)
  - **Status**: ⚪ Not Started
  - **Output**: `apps/frontend/tests/`
  - **Focus**: Component rendering, API calls

- [ ] **6.2.5** Perform end-to-end testing
  - **Status**: ⚪ Not Started
  - **Tool**: Playwright or Selenium
  - **Scenarios**: User workflows from login to analysis

### 6.3 Human Evaluation
**Duration**: Week 14 (May 24-30, 2026)  
**Owner**: Research Lead  
**Status**: ⚪ **NOT STARTED**

#### Tasks:
- [ ] **6.3.1** Recruit 5-10 project managers for evaluation
  - **Status**: ⚪ Not Started
  - **Target**: Managers from startup community
  - **Compensation**: Gift cards or co-authorship

- [ ] **6.3.2** Design evaluation survey
  - **Status**: ⚪ Not Started
  - **Output**: Google Form or Qualtrics survey
  - **Metrics**: Trust score, usefulness, accuracy perception

- [ ] **6.3.3** Conduct user testing sessions
  - **Status**: ⚪ Not Started
  - **Format**: 1-hour sessions with think-aloud protocol
  - **Record**: Screen recordings, notes

- [ ] **6.3.4** Analyze qualitative feedback
  - **Status**: ⚪ Not Started
  - **Output**: `doc/research/user_study_results.md`
  - **Method**: Thematic analysis

### 6.4 Documentation & Deployment
**Duration**: Week 14 (May 24-30, 2026)  
**Owner**: Full Team  
**Status**: ⚪ **NOT STARTED**

#### Tasks:
- [ ] **6.4.1** Write deployment guide
  - **Status**: ⚪ Not Started
  - **Output**: `doc/deployment/deployment_guide.md`
  - **Steps**: Docker setup, configuration, first run

- [ ] **6.4.2** Create user manual
  - **Status**: ⚪ Not Started
  - **Output**: `doc/user_guide.md`
  - **Content**: How to use each feature, interpret results

- [ ] **6.4.3** Write API documentation
  - **Status**: ⚪ Not Started
  - **Output**: OpenAPI spec + usage examples
  - **Tool**: FastAPI auto-docs + custom guides

- [ ] **6.4.4** Create demo video
  - **Status**: ⚪ Not Started
  - **Format**: 5-10 minute walkthrough
  - **Tool**: Loom or OBS

- [ ] **6.4.5** Prepare research paper draft
  - **Status**: ⚪ Not Started
  - **Output**: `doc/research/paper_draft.md`
  - **Format**: Conference submission template (e.g., NeurIPS, ICML)

---

## Ongoing Tasks (Throughout Project)

### Project Management
- [ ] **Weekly team meetings** (Every Monday, 10:00 AM)
  - **Status**: ⚪ Not Started
  - **Agenda**: Progress review, blockers, next week planning

- [ ] **Bi-weekly stakeholder updates** (Every other Friday)
  - **Status**: ⚪ Not Started
  - **Format**: Email summary or presentation

- [ ] **Daily stand-ups (async)** (Slack or Discord)
  - **Status**: ⚪ Not Started
  - **Format**: 3 questions - done yesterday, doing today, blockers

### Documentation
- [ ] **Update WBS weekly**
  - **Status**: 🟢 In Progress
  - **Cadence**: Every Friday EOD

- [ ] **Maintain research log**
  - **Status**: ⚪ Not Started
  - **Output**: `doc/research/research_log.md`
  - **Content**: Experiments, findings, decisions

- [ ] **Code documentation (docstrings)**
  - **Status**: ⚪ Not Started
  - **Standard**: Google-style docstrings
  - **Tool**: Auto-generate with Sphinx

### Version Control
- [ ] **Git branching strategy**
  - **Status**: ⚪ Not Started
  - **Strategy**: GitFlow (main, develop, feature/*, hotfix/*)
  - **PR reviews**: Mandatory for all merges to develop/main

- [ ] **Semantic versioning**
  - **Status**: ⚪ Not Started
  - **Format**: MAJOR.MINOR.PATCH (v0.1.0 → v1.0.0)

---

## Risks & Mitigation

| Risk | Probability | Impact | Mitigation Strategy | Owner |
|------|-------------|--------|---------------------|-------|
| **GitHub API rate limits** | ~~High~~ **Mitigated** | High | ✅ **Resolved** — Local git ingestion (`LocalGitScraper`) replaces API for all commits/diffs. 65K commits extracted in 40s with zero API calls. PRs/issues extracted from commit trailers offline. | Data Engineer |
| **LLM hallucinations** | Medium | High | RAG for grounding, human-in-the-loop validation | ML Engineer |
| **Insufficient training data** | Medium | Medium | Synthetic data generation, transfer learning | ML Engineer |
| **Resource constraints (RAM/CPU)** | Medium | Medium | Quantized models, batch processing, optimization | DevOps |
| **Scope creep** | Medium | High | Strict WBS adherence, weekly scope reviews | Project Lead |
| **Stakeholder unavailability for testing** | Medium | Medium | Early recruitment, incentivize participation | Research Lead |
| **Docker deployment issues** | Low | Medium | Test on multiple platforms, Docker docs | DevOps |

---

## Success Criteria

### Research Success
- [ ] **Novel contribution**: Publish in top-tier ML/SE conference (NeurIPS, ICML, ICSE, FSE)
- [ ] **Real-world impact**: 3+ startups pilot the system
- [ ] **Open source**: 100+ GitHub stars, 10+ contributors

### Technical Success
- [ ] **Performance**: >85% F1 for sprint success prediction
- [ ] **Latency**: <60 seconds end-to-end analysis
- [ ] **Explainability**: >90% recommendations with evidence
- [ ] **Deployment**: Runs on consumer hardware (<16GB RAM)

### User Success
- [ ] **Trust score**: >80% of users trust AI recommendations
- [ ] **Time savings**: Reduce manager overhead by >50%
- [ ] **Adoption**: >70% of tested users would use in production

---

## Resources & Dependencies

### Human Resources
- **Project Lead / Research Lead** (1 person, 20 hrs/week)
- **ML Engineer** (1 person, 20 hrs/week)
- **Backend Developer** (1 person, 15 hrs/week)
- **Frontend Developer** (1 person, 10 hrs/week)
- **DevOps/Full Stack** (1 person, 5 hrs/week)

### Computational Resources
- **Local Development**: MacBook M4 Pro (24GB RAM)
- **Data Storage**: 50GB SSD for data/embeddings
- **GitHub API**: Free tier (5000 requests/hour)

### Software & Tools
- **LLM**: Ollama + Llama-3-8B (free, local)
- **Embedding**: Sentence-BERT (free, local)
- **Vector DB**: ChromaDB (free, local)
- **Database**: PostgreSQL (free, Docker)
- **Backend**: FastAPI, LangGraph (free, Python)
- **Frontend**: Streamlit (free, Python)
- **Deployment**: Docker, Docker Compose (free)

### External Dependencies
- GitHub Archive (free, public dataset)
- 2-3 startup repos for case studies (need permission)
- 5-10 project managers for user testing (need recruitment)

---

## Appendix: Metrics Definitions

### Sprint Success Prediction Metrics
- **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1 Score**: 2 × (Precision × Recall) / (Precision + Recall)
- **AUC-ROC**: Area under ROC curve

### Recommendation Quality Metrics
- **Trust Score**: User survey (1-5 scale)
- **Evidence Coverage**: % recommendations with ≥3 citations
- **Actionability**: User survey (1-5 scale)
- **Acceptance Rate**: % recommendations accepted by users

### System Performance Metrics
- **End-to-End Latency**: GitHub event → recommendation (seconds)
- **Throughput**: Analyses per minute
- **RAM Usage**: Peak memory consumption (GB)
- **Storage**: Database + embeddings size (GB)

### Research Impact Metrics
- **Paper Citations**: Google Scholar citations
- **GitHub Stars**: Repository stars
- **User Adoption**: Active installations
- **Community Engagement**: Issues, PRs, discussions

---

**Document Status**: 🟢 Living Document - Updated Weekly  
**Next Review**: February 21, 2026  
**Owner**: Project Lead

