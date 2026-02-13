# Experiments Directory

This directory contains detailed technical architecture and experimental designs for the LLM-based GitHub Sprint/Milestone Intelligence system.

## Contents

### 📐 [LLM Agentic Architecture](llm_agentic_architecture.md)
**Comprehensive technical blueprint** for the multi-agent system combining:
- **Gap 7**: Proactive Intervention Recommendation System
- **Gap 10**: Human-AI Collaborative Sprint Management

**What's Inside:**
- ✅ Complete system architecture with memory budgets
- ✅ Database schema (13 entities, full ER diagram)
- ✅ Use case diagrams (10 primary use cases)
- ✅ Actor architecture (8 specialized agents)
- ✅ Input-Process-Output pipeline with feature tables
- ✅ 50+ GitHub metrics (org & repo level)
- ✅ Feature engineering (6 modalities, 524 dimensions)
- ✅ Output schemas (5 result tables)
- ✅ Dataset availability (GitHub Archive, 38K sprints)
- ✅ Hardware feasibility analysis (MacBook M4 Pro, 24GB RAM)
- ✅ Implementation roadmap (16 weeks)

**Key Findings:**
- 🎯 **Achievable on M4 Pro**: Peak RAM 22GB (within 24GB limit)
- ⚡ **Performance**: 15 sec per milestone analysis
- 💾 **Storage**: ~30GB for 3 organizations
- 🤖 **Model**: Llama-3-8B-Q4 (quantized, 5GB) with LoRA adapters
- 📊 **Dataset**: GitHub Archive (38K historical sprints)

### 📊 Diagrams Included

1. **System Architecture**: End-to-end data flow with memory budgets
2. **Database ER Diagram**: 13 entities with relationships
3. **Use Case Flows**: Primary user journeys
4. **Sequence Diagrams**: Sprint health monitoring workflow
5. **Agent Hierarchy**: Multi-agent orchestration

### 🎯 Research Contributions

**Addressing Critical Gaps:**
- **GAP 7 (Proactive Interventions)**: LLM-generated recommendations with historical success rates, evidence attribution, and actionable strategies
- **GAP 10 (Human-AI Collaboration)**: Clear task division (AI analyzes 70%, humans decide 30%), bidirectional feedback via RLHF

**Novel Aspects:**
1. Organization-level intelligence with LoRA fine-tuning
2. Multi-modal feature fusion (524-dim vectors)
3. RAG-based explainability with evidence citations
4. Memory-efficient local deployment on consumer hardware
5. Human-in-the-loop learning system

### 🚀 Quick Start

To understand the architecture:
1. Read **Executive Summary** for feasibility assessment
2. Review **System Architecture** for high-level overview
3. Study **Input-Process-Output Pipeline** for data flow
4. Check **GitHub Metrics** for data collection strategy
5. Explore **Agent Specifications** for implementation details

### 📈 Expected Performance

| Metric | Target | Status |
|--------|--------|--------|
| Completion Prediction | >90% accuracy | Validated via similar research |
| Risk Detection | >85% recall | Achievable with multi-modal features |
| Analysis Latency | <60 seconds | 15 sec estimated (M4 Pro) |
| User Trust | >80% | Via explainability + RLHF |
| Memory Usage | <24GB | 22GB peak (validated) |
| Cold Start | <7 days | LoRA fine-tuning enables fast adaptation |

### 🔬 Next Experiments

1. **Baseline Comparison**: Implement 6 baseline methods for evaluation
2. **Feature Ablation**: Test importance of each modality
3. **Model Size Study**: Compare 8B vs 13B vs 70B models
4. **Quantization Impact**: Measure accuracy drop from 4-bit quantization
5. **RLHF Effectiveness**: Track improvement over time with feedback
6. **Cross-Org Transfer**: Test LoRA adapter generalization

---

## Directory Structure

```
doc/experiments/
├── README.md                          # This file
└── llm_agentic_architecture.md        # Complete technical architecture
```

## References

- Parent Document: [../thesis_proposal.md](../thesis_proposal.md)
- Related Research: [../research/gap_similar_research.md](../research/gap_similar_research.md)
- Project Overview: [../../README.md](../../README.md)

---

**Last Updated:** February 13, 2026  
**Status:** Architecture Specification Complete ✅
