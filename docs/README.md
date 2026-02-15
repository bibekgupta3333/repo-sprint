# Documentation Index
# Lightweight Sprint Intelligence for Small Startups

**Last Updated**: February 15, 2026  
**Status**: 🟢 Documentation Complete  
**Focus**: 2-3 Repository Small Startup Teams

---

## Quick Navigation

### 🎯 Start Here
- [Main README](../README.md) - Project overview and quick start
- [Thesis Proposal](thesis_proposal.md) - Research foundation (50 papers analyzed)
- [Quick Reference](quick_reference.md) - Command cheat sheet

### 📋 Planning & Management
- [Work Breakdown Structure (WBS)](planning/WBS.md) - 14-week project timeline with status tracking
  - 6 phases, 100+ tasks
  - Risk matrix and mitigation strategies
  - Success criteria and deliverables

### 🏗️ Architecture & Design
- [System Architecture](architecture/system_architecture.md) - Complete system design
  - C4 Context/Container diagrams
  - Component architecture (9 agents)
  - Deployment architecture (Docker)
  - Data flow and security
  
- [Database Design](architecture/database_design.md) - Schema and data models
  - 15 PostgreSQL tables with ERD
  - 4 ChromaDB vector collections
  - SQLAlchemy models
  - Indexing and backup strategies
  
- [ML Validation Architecture](architecture/ml_validation_architecture.md) - Testing framework
  - Multi-level testing pyramid
  - ML model validation (classification, regression, ranking metrics)
  - Agent testing (unit, integration, performance)
  - A/B testing framework
  - Human evaluation protocol
  - Reproducibility guidelines

### 🎨 Design System
- [Figma Design Prompts](design/figma_design_prompts.md) - UI/UX specifications
  - Industry-standard color palette
  - 5 detailed screen designs
  - Component library (buttons, cards, badges)
  - Responsive breakpoints
  - Accessibility guidelines (WCAG AA)

### 🚀 Deployment & Operations
- [Deployment Guide](deployment/deployment_guide.md) - Setup instructions
  - Prerequisites and quick start (5 minutes)
  - Detailed configuration
  - Health checks and troubleshooting
  - Production hardening
  
- [Docker Compose Configuration](../docker-compose.yml) - Container orchestration
  - 6 services (Streamlit, FastAPI, PostgreSQL, ChromaDB, Redis, Ollama)
  - Resource limits (11.5GB total RAM)
  - Volume persistence
  - Health checks

### 🔬 Research & Experiments
- [Research Objectives](research/research_objectives.md) - Top 5 research goals
  - Multi-repo cross-dependency intelligence for 2-3 repos (F1 > 0.85)
  - Real-time RAG blocker detection (<60s latency)
  - Synthetic data generation for new startups (<5% performance gap)
  - LoRA startup adaptation (<500MB footprint)
  - Lightweight local deployment (<16GB RAM)
  - 15 research questions, 5 hypotheses
  - Comprehensive evaluation plan

- [Evaluation Research Plan](research/evaluation_research_intelligent_sprint_analysis_startups.md) - Startup-focused evaluation methodology
  - Offline, online replay, human, and systems evaluation tracks
  - Baselines, ablations, and statistical analysis protocol
  - Success criteria for accuracy, trust, latency, and adoption
  
- [Gap Analysis](research/gap_similar_research.md) - Literature review findings
  - 50 research papers analyzed
  - 10 critical gaps identified
  - Innovation opportunities
  
- **Experiments**:
  - [LLM Agentic Architecture for GitHub Sprint](experiments/LLM%20Agentic%20Architecture%20for%20GitHub%20Sprint/) - Initial sprint-level agent design
  - [LLM Agentic Architecture for Small Startup Sprint Intelligence](experiments/LLM%20Agentic%20Architecture%20for%20Organization-Level%20Sprint%20Intelligence/) - Multi-repo architecture for 2-3 repositories

### ⚙️ Development Configuration
- [Editor Config](../.editorconfig) - Unified code style
  - Python (4 spaces), YAML/JSON (2 spaces)
  - UTF-8 encoding, LF line endings
  
- [Cursor Rules](../.cursorrules) - AI-assisted coding guidelines
  - Python type hints and docstrings
  - FastAPI and LangGraph patterns
  - SQLAlchemy ORM best practices
  - Streamlit caching strategies
  - Security and performance rules
  
- [Environment Template](../.env.example) - Configuration template
  - GitHub API token
  - Database credentials
  - LLM model selection

### 📊 Data
- [Data README](../data/README.md) - Dataset documentation
- [Download Summary](../data/DOWNLOAD_SUMMARY.md) - GitHub Archive download logs
- [Purification Guide](../data/PURIFICATION_GUIDE.md) - Data cleaning instructions

### 📜 Scripts
- [Scripts README](../scripts/README.md) - Automation scripts
  - `download_github_archive.py` - Fetch GitHub Archive data
  - `collect_github_data.py` - Extract sprint metadata
  - `prepare_embeddings.py` - Generate vector embeddings

---

## Documentation Status

| Document | Status | Last Updated | Next Review |
|----------|--------|--------------|-------------|
| WBS | ✅ Complete | Feb 14, 2026 | Weekly |
| System Architecture | ✅ Complete | Feb 14, 2026 | Monthly |
| Database Design | ✅ Complete | Feb 14, 2026 | Monthly |
| ML Validation | ✅ Complete | Feb 14, 2026 | Monthly |
| Deployment Guide | ✅ Complete | Feb 14, 2026 | As needed |
| Research Objectives | ✅ Complete | Feb 14, 2026 | Monthly |
| Figma Design Prompts | ✅ Complete | Feb 14, 2026 | As needed |
| Editor Config | ✅ Complete | Feb 14, 2026 | Stable |
| Cursor Rules | ✅ Complete | Feb 14, 2026 | As needed |

---

## Documentation Conventions

### Markdown Standards
- **Headers**: ATX-style (`#`, `##`, `###`)
- **Code Blocks**: Always specify language (```python, ```yaml, ```bash)
- **Links**: Use relative paths for internal docs
- **Diagrams**: Mermaid syntax for system diagrams

### Versioning
- All major documents include version number (e.g., `v1.0.0`)
- Follow semantic versioning: `MAJOR.MINOR.PATCH`
- Update "Last Updated" date on each change

### File Naming
- Use snake_case for filenames: `system_architecture.md`
- Use descriptive names: `ml_validation_architecture.md` (not `testing.md`)
- Group related docs in subdirectories

### Diagram Guidelines
- Use Mermaid for architecture diagrams (C4, sequences, flowcharts)
- Use ASCII art for simple tables or trees
- Keep diagrams under 20 nodes for readability

---

## How to Use This Documentation

### For New Team Members
1. Read [Main README](../README.md) for project overview
2. Review [Thesis Proposal](thesis_proposal.md) to understand research foundation
3. Study [System Architecture](architecture/system_architecture.md) for technical design
4. Set up local environment using [Deployment Guide](deployment/deployment_guide.md)
5. Check [WBS](planning/WBS.md) to see current project status

### For Development
1. Follow [Cursor Rules](../.cursorrules) for code consistency
2. Refer to [Database Design](architecture/database_design.md) when creating models
3. Use [ML Validation Architecture](architecture/ml_validation_architecture.md) for testing
4. Check [System Architecture](architecture/system_architecture.md) for component interactions

### For Research & Evaluation
1. Review [Research Objectives](research/research_objectives.md) for evaluation criteria
2. Follow [ML Validation Architecture](architecture/ml_validation_architecture.md) for experiment design
3. Consult [Gap Analysis](research/gap_similar_research.md) for positioning

### For Design & Frontend
1. Use [Figma Design Prompts](design/figma_design_prompts.md) for UI specifications
2. Follow color palette and component library from design doc
3. Ensure WCAG AA accessibility compliance

---

## Documentation Maintenance

### Review Schedule
- **Weekly**: WBS status updates
- **Monthly**: Architecture and research objective reviews
- **Quarterly**: Comprehensive documentation audit
- **As Needed**: Deployment guide, design system

### Update Process
1. Make changes to relevant markdown file
2. Update version number and "Last Updated" date
3. If major change, update this index
4. Commit with descriptive message: `docs: update WBS with Phase 2 progress`

### Contributors
- **Research Lead**: Responsible for research objectives, gap analysis
- **Technical Lead**: Responsible for architecture, database design, deployment
- **ML Engineer**: Responsible for ML validation architecture
- **Frontend Developer**: Responsible for design system, Figma prompts

---

## External Resources

### GitHub Repositories
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Ollama Documentation](https://ollama.ai/docs)

### Research Papers
- See [Thesis Proposal](thesis_proposal.md) for full bibliography (50 papers)
- Key papers on multi-agent systems, RAG, graph neural networks

### Design Resources
- [Material Design 3](https://m3.material.io/)
- [WCAG 2.1 Guidelines](https://www.w3.org/WAI/WCAG21/quickref/)

---

**Index Version**: 1.0.0  
**Maintained By**: Documentation Team  
**Contact**: GitHub Issues or project Slack
