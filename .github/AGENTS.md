# Sprint Intelligence Project - Agent Guide

## Project Overview
14-week sprint intelligence research project combining data collection, ML baselines, multi-agent orchestration, and deployment.

## Team Agents

### 🔬 Researcher Agent
**Focus**: Analysis, design, validation, evaluation

**Key Responsibilities**:
- Requirements analysis and specification
- Architecture design and validation
- Literature review and research
- Evaluation framework development
- Cross-functional coordination

**WBS Milestones**:
- M1: Problem Definition
- M3: Synthetic Data Validation
- M6: System Architecture Design
- M10: Evaluation Framework

**Invoke when**:
```
/researcher analyze requirements
/researcher design [component]
/researcher evaluate [approach]
/researcher research [topic]
```

---

### 💻 Developer Agent
**Focus**: Code implementation following KISS principle, processor pattern, modular _core/ structure

**Key Responsibilities**:
- Data pipeline implementation
- Feature extraction modules
- Baseline model implementations
- Model training and fine-tuning
- Integration and orchestration

**Code Principles**:
- **KISS**: Keep code simple, readable, focused
- **Processor Pattern**: Single-responsibility classes
- **_core/ Structure**: Organized, reusable modules
- **Type Hints**: Full type annotations
- **Short & Simple**: Minimal, clear code

**WBS Milestones**:
- M2: Data Pipeline Implementation
- M4: Feature Engineering Module
- M5: Baseline Implementation
- M7: LoRA Fine-Tuning
- M8: RAG Implementation
- M9: Multi-Agent Integration

**Invoke when**:
```
/developer implement [feature]
/developer refactor [module]
/developer fix [bug]
/developer integrate [component]
```

---

## WBS Timeline

| Week | Milestone | Assigned | Focus | Output |
|------|-----------|----------|-------|--------|
| 1 | M1: Problem Definition | Research | Requirements, objectives | Problem specification |
| 2 | M2: Dataset Collection | Developer | Data pipeline | Labeled dataset (9K sprints) |
| 3 | M3: Synthetic Data | Research + Developer | Quality validation | Synthetic dataset (5K sprints) |
| 4 | M4: Feature Engineering | Developer | 120-feature pipeline | Feature module documentation |
| 5 | M5: Baseline Implementation | Developer | Training & evaluation | Baseline benchmarks |
| 6 | M6: System Architecture | Research + Developer | Design & documentation | Architecture diagrams |
| 7 | M7: LoRA Fine-Tuning | Developer | Hyperparameter tuning | Model checkpoints |
| 8 | M8: RAG Implementation | Developer | Embedding & retrieval | RAG pipeline |
| 9 | M9: Multi-Agent Integration | Developer | End-to-end testing | Integrated 6-agent system |
| 10 | M10: Evaluation Framework | Research | Metrics & testing | Evaluation results |
| 11 | M11: Dashboard | Developer | Web interface | Streamlit prototype |
| 12 | M12: Systems Optimization | Developer | Performance tuning | Optimization report |
| 13 | M13: Documentation | All | Writing & editing | Academic paper |
| 14 | M14: Presentation | All | Final presentation | Demo & slides |

---

## Workflow Pattern

### Before Implementation
→ **Researcher** analyzes requirements and designs approach
→ Document specifications and design decisions

### During Implementation
→ **Developer** implements following KISS + processor patterns
→ Stages code in git at each logical checkpoint
→ User reviews and approves commits

### After Implementation
→ **Researcher** validates against requirements
→ Design evaluation/testing strategy
→ Prepare documentation

---

## Current Project Structure

```
scripts/_core/          # Core service modules (KISS principle)
  ├── scraper.py       # GitHub data collection
  ├── processor.py      # Feature extraction (processor pattern)
  ├── analyzer.py       # Data analysis
  └── __init__.py       # Clean exports

scripts/
  ├── ingest.py         # Unified command (download → analyze → process)
  ├── analyze_data.py   # Step-by-step analysis
  └── process_data.py   # Step-by-step processing

data/
  ├── raw/              # Downloaded datasets
  └── processed/        # Extracted features

package.json            # npm scripts for convenience
requirements.txt        # Python dependencies
```

---

## Development Standards

### Code Quality
- ✅ Type hints on all functions
- ✅ Functions < 20 lines
- ✅ Classes do one thing
- ✅ Clear variable names
- ✅ Docstrings for public APIs

### Git Workflow
1. Developer stages code (`git add`)
2. Prepares commit message
3. User reviews and approves: `git commit`
4. Never force-push to main

### Testing
- Test with real data before staging
- Manual testing sufficient for research code
- Use `npm run` commands to validate

---

## Communication Pattern

**Researcher → Developer**: Design specs, requirements, evaluation criteria
**Developer → Researcher**: Implementation status, technical constraints, validation needs
**Both → User**: Progress updates, staged code ready for review, decisions needed
