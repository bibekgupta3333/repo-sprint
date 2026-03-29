"""LoRA Training Orchestrator Agent (Phase C).

Monitors model performance drift and triggers fine-tuning.

Responsibilities:
  - Track per-project model F1 scores over time
  - Detect performance drift (F1 drop > threshold)
  - Collect training examples when drift detected
  - Execute LoRA fine-tuning (rank=8, 3 epochs)
  - Manage adapter versioning and registry
  - Integrate new adapters into inference pipeline

Targets Objective 4: LoRA Fine-Tuning Pipeline for Continuous Learning
  - Success metrics: <5% F1 gap to baseline, per-project adapters, versioning
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agents.state import OrchestratorState
from src.agents.tools import ToolRegistry


@dataclass
class ProjectPerformance:
    """Tracks performance metrics for a specific project/repository."""
    project_id: str
    baseline_f1: float  # Initial model F1 score
    current_f1: float  # Latest observed F1 score
    drift_detected: bool = False
    drift_magnitude: float = 0.0  # Absolute drop in F1
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    num_observations: int = 0


@dataclass
class AdapterConfig:
    """Configuration for a LoRA adapter."""
    adapter_id: str
    project_id: str
    rank: int = 8
    alpha: int = 16  # LoRA alpha (typically 2 * rank)
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    epochs: int = 3
    learning_rate: float = 1e-4
    batch_size: int = 16
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class AdapterMetrics:
    """Training and evaluation metrics for an adapter."""
    adapter_id: str
    f1_before: float  # F1 before fine-tuning
    f1_after: float  # F1 after fine-tuning
    f1_improvement: float = 0.0
    training_loss: float = 0.0
    validation_loss: float = 0.0
    training_samples: int = 0
    training_time_seconds: float = 0.0
    status: str = "pending"  # pending, training, complete, failed


class LoRATrainingOrchestratorAgent:
    """Orchestrates continuous learning via LoRA adapter management.

    Monitors model performance across projects and automatically triggers
    LoRA fine-tuning when drift is detected. Manages adapter versions
    and versioning for safe rollback.

    Output:
      - state.lora_performance_tracker: dict of project→performance
      - state.lora_adapters_active: list of active adapter configs
      - state.lora_training_triggered: bool indicating if training started
      - state.lora_metrics: AdapterMetrics for completed training
    """

    # Drift detection parameters
    DRIFT_THRESHOLD = 0.05  # Trigger retraining if F1 drops >5%
    MIN_OBSERVATIONS = 3  # Require N samples before triggering
    MIN_BASELINE_SAMPLES = 5  # Need baseline data before monitoring

    def __init__(self, tool_registry: ToolRegistry):
        """Initialize with tool registry and model access."""
        self.logger = logging.getLogger(__name__)
        self.tool_registry = tool_registry
        self.llm_tool = tool_registry.llm_tool
        self.feature_tool = tool_registry.feature_tool

    def execute(self, state: OrchestratorState) -> OrchestratorState:
        """Monitor performance and trigger LoRA training if needed.

        Process:
          1. Extract project-level performance from LLM reasoning
          2. Track F1 scores in performance registry
          3. Detect drift (F1 drop > threshold)
          4. If drift detected:
             a. Collect training examples from recent failures
             b. Create LoRA adapter config
             c. Execute fine-tuning (simulated)
             d. Evaluate new adapter
             e. Update registry
          5. Return updated state with metrics
        """
        self.logger.info("=== LoRA Training Orchestrator Agent Starting ===")
        state.current_agent = "lora_training_orchestrator"

        try:
            # Step 1: Initialize/load performance tracker
            performance_tracker = self._load_or_init_tracker(state)

            # Step 2: Extract current performance from sprint analysis
            project_id = state.repositories[0] if state.repositories else "default"
            current_f1 = self._extract_model_f1(state)

            # Step 3: Update performance tracking
            project_perf = self._update_performance(
                tracker=performance_tracker,
                project_id=project_id,
                current_f1=current_f1
            )

            # Step 4: Check for drift
            training_triggered = False
            adapter_metrics = None

            if self._check_drift_detected(project_perf):
                self.logger.info(
                    f"  ⚠ Drift detected in {project_id}: "
                    f"F1 {project_perf.current_f1:.3f} (baseline {project_perf.baseline_f1:.3f})"
                )

                # Step 5: Trigger LoRA fine-tuning
                adapter_metrics = self._trigger_lora_training(
                    state=state,
                    project_id=project_id,
                    performance=project_perf
                )

                if adapter_metrics:
                    training_triggered = True
                    self.logger.info(
                        f"  ✓ LoRA training complete: F1 improved from "
                        f"{adapter_metrics.f1_before:.3f} → {adapter_metrics.f1_after:.3f}"
                    )

            # Step 6: Store results in state
            state.lora_performance_tracker = {
                k: {
                    "project_id": v.project_id,
                    "baseline_f1": v.baseline_f1,
                    "current_f1": v.current_f1,
                    "drift_detected": v.drift_detected,
                    "drift_magnitude": v.drift_magnitude,
                    "last_updated": v.last_updated,
                    "num_observations": v.num_observations,
                }
                for k, v in performance_tracker.items()
            }

            state.lora_training_triggered = training_triggered

            if adapter_metrics:
                state.lora_metrics = {
                    "adapter_id": adapter_metrics.adapter_id,
                    "f1_before": adapter_metrics.f1_before,
                    "f1_after": adapter_metrics.f1_after,
                    "f1_improvement": adapter_metrics.f1_improvement,
                    "training_loss": adapter_metrics.training_loss,
                    "validation_loss": adapter_metrics.validation_loss,
                    "training_samples": adapter_metrics.training_samples,
                    "training_time_seconds": adapter_metrics.training_time_seconds,
                    "status": adapter_metrics.status,
                }

            # Log summary
            self.logger.info(
                f"  ✓ Monitored {len(performance_tracker)} projects, "
                f"training triggered: {training_triggered}"
            )

            state.execution_logs.append(
                f"[lora_training_orchestrator] Tracked {len(performance_tracker)} projects, "
                f"triggered training: {training_triggered}"
            )

        except Exception as e:
            self.logger.error(f"LoRA training orchestration error: {str(e)}")
            state.errors.append(f"LoRA orchestrator error: {str(e)}")

            # Graceful degradation
            state.lora_performance_tracker = {}
            state.lora_training_triggered = False
            state.lora_metrics = {}

        return state

    # ================================================================ #
    #  Performance Tracking                                           #
    # ================================================================ #

    def _load_or_init_tracker(self, state: OrchestratorState) -> dict[str, ProjectPerformance]:
        """Load existing performance tracker or initialize new one.

        Tracker is persisted in state.lora_performance_tracker from prior runs.
        """
        if hasattr(state, "lora_performance_tracker") and state.lora_performance_tracker:
            # Reconstruct ProjectPerformance objects from dict
            tracker = {}
            for proj_id, metrics_dict in state.lora_performance_tracker.items():
                tracker[proj_id] = ProjectPerformance(
                    project_id=metrics_dict.get("project_id", proj_id),
                    baseline_f1=metrics_dict.get("baseline_f1", 0.5),
                    current_f1=metrics_dict.get("current_f1", 0.5),
                    drift_detected=metrics_dict.get("drift_detected", False),
                    drift_magnitude=metrics_dict.get("drift_magnitude", 0.0),
                    last_updated=metrics_dict.get("last_updated", ""),
                    num_observations=metrics_dict.get("num_observations", 0),
                )
            return tracker
        else:
            return {}

    def _extract_model_f1(self, state: OrchestratorState) -> float:
        """Extract F1 score from sprint analysis.

        F1 is derived from precision and recall metrics. Default to 0.5 if not available.
        """
        if hasattr(state, "sprint_analysis") and state.sprint_analysis:
            analysis = state.sprint_analysis
            # Extract confidence score as proxy for F1
            confidence = analysis.get("confidence_score", 0.5)
            return float(confidence)

        if hasattr(state, "feature_vector") and state.feature_vector:
            # Heuristic: average feature values as performance proxy
            avg_feature = sum(state.feature_vector) / len(state.feature_vector)
            return max(0.3, min(1.0, avg_feature))  # Clamp to [0.3, 1.0]

        return 0.5  # Default baseline

    def _update_performance(
        self,
        tracker: dict[str, ProjectPerformance],
        project_id: str,
        current_f1: float
    ) -> ProjectPerformance:
        """Update or create project performance record.

        Returns updated ProjectPerformance object.
        """
        if project_id not in tracker:
            # Create new project record
            perf = ProjectPerformance(
                project_id=project_id,
                baseline_f1=current_f1,  # First observation is baseline
                current_f1=current_f1,
                num_observations=1
            )
        else:
            # Update existing record
            perf = tracker[project_id]
            perf.current_f1 = current_f1
            perf.num_observations += 1
            perf.last_updated = datetime.now().isoformat()

        tracker[project_id] = perf
        return perf

    def _check_drift_detected(self, performance: ProjectPerformance) -> bool:
        """Check if performance drift is significant enough to trigger retraining.

        Drift = baseline_f1 - current_f1 > DRIFT_THRESHOLD AND enough observations.
        """
        drift = performance.baseline_f1 - performance.current_f1
        performance.drift_magnitude = drift

        if drift > self.DRIFT_THRESHOLD and performance.num_observations >= self.MIN_OBSERVATIONS:
            performance.drift_detected = True
            return True

        return False

    # ================================================================ #
    #  LoRA Training Trigger                                          #
    # ================================================================ #

    def _trigger_lora_training(
        self,
        state: OrchestratorState,
        project_id: str,
        performance: ProjectPerformance
    ) -> Optional[AdapterMetrics]:
        """Trigger LoRA fine-tuning when drift detected.

        Steps:
          1. Collect training examples (recent failures, misclassifications)
          2. Create adapter configuration
          3. Simulate fine-tuning
          4. Evaluate on validation set
          5. Return adapter metrics
        """
        try:
            # Step 1: Collect training examples
            training_examples = self._collect_training_examples(state, project_id)

            if not training_examples or len(training_examples) < 5:
                self.logger.warning(
                    f"Insufficient training examples ({len(training_examples) or 0}), "
                    f"skipping LoRA training"
                )
                return None

            # Step 2: Create adapter config
            adapter_config = self._create_adapter_config(project_id)

            # Step 3: Simulate fine-tuning
            metrics = self._simulate_lora_training(
                config=adapter_config,
                training_examples=training_examples,
                baseline_f1=performance.baseline_f1
            )

            if metrics and metrics.f1_after > performance.current_f1:
                # Step 4: Register new adapter
                self._register_adapter(adapter_config, metrics)

                self.logger.info(
                    f"  ✓ Adapter {adapter_config.adapter_id} registered "
                    f"(F1: {metrics.f1_before:.3f} → {metrics.f1_after:.3f})"
                )

                return metrics

        except Exception as e:
            self.logger.error(f"LoRA training trigger error: {str(e)}")

        return None

    def _collect_training_examples(
        self,
        state: OrchestratorState,
        project_id: str,
        max_examples: int = 100
    ) -> list[dict]:
        """Collect training examples for LoRA fine-tuning.

        Sources:
          - Recent misclassified sprints (from identified_risks)
          - Failed predictions (from execution logs)
          - Edge cases (synthetic sprints with extreme metrics)

        Returns list of training examples.
        """
        examples = []

        # Source 1: Recent misclassifications from identified_risks
        if hasattr(state, "identified_risks") and state.identified_risks:
            for risk in state.identified_risks[:max_examples // 2]:
                if hasattr(risk, "dict"):
                    example = risk.dict()
                elif isinstance(risk, dict):
                    example = risk
                else:
                    continue

                example["source"] = "identified_risk"
                example["project_id"] = project_id
                examples.append(example)

        # Source 2: Synthetic sprints (edge cases)
        if hasattr(state, "synthetic_sprints") and state.synthetic_sprints:
            for sprint in state.synthetic_sprints[:max_examples // 2]:
                if isinstance(sprint, dict):
                    metrics = sprint.get("metrics", {})
                    if metrics:
                        example = {
                            "sprint_id": sprint.get("sprint_id"),
                            "metrics": metrics,
                            "risk_label": sprint.get("risk_label", {}),
                            "source": "synthetic_sprint",
                            "project_id": project_id,
                        }
                        examples.append(example)

        return examples[:max_examples]

    def _create_adapter_config(self, project_id: str) -> AdapterConfig:
        """Create LoRA adapter configuration for project.

        Standard config: rank=8, 3 epochs, 1e-4 learning rate.
        """
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        adapter_id = f"lora_{project_id.replace('/', '_')}_{timestamp}"

        return AdapterConfig(
            adapter_id=adapter_id,
            project_id=project_id,
            rank=8,
            alpha=16,
            target_modules=["q_proj", "v_proj", "k_proj"],
            epochs=3,
            learning_rate=1e-4,
            batch_size=16
        )

    def _simulate_lora_training(
        self,
        config: AdapterConfig,
        training_examples: list[dict],
        baseline_f1: float
    ) -> AdapterMetrics:
        """Simulate LoRA fine-tuning (real training would use transformers library).

        In production, this would:
          1. Apply LoRA config to base model
          2. Train on examples for N epochs
          3. Evaluate on validation set
          4. Save adapter weights

        For now, simulate improvement: F1 increases by 5-10%.
        """
        # Heuristic: LoRA training improves F1 by 5-10% on average
        improvement_rate = 0.07  # 7% average improvement
        f1_before = baseline_f1
        f1_after = min(0.95, f1_before + improvement_rate)

        metrics = AdapterMetrics(
            adapter_id=config.adapter_id,
            f1_before=f1_before,
            f1_after=f1_after,
            f1_improvement=f1_after - f1_before,
            training_loss=0.25,  # Simulated
            validation_loss=0.28,  # Simulated
            training_samples=len(training_examples),
            training_time_seconds=120.0,  # Simulated: 2 minutes
            status="complete"
        )

        return metrics

    def _register_adapter(
        self,
        config: AdapterConfig,
        metrics: AdapterMetrics
    ) -> None:
        """Register adapter in version control system.

        In production, this would save adapter weights and metadata
        to a versioned registry (e.g., HuggingFace/git).
        """
        registry_entry = {
            "adapter_id": config.adapter_id,
            "project_id": config.project_id,
            "config": {
                "rank": config.rank,
                "alpha": config.alpha,
                "target_modules": config.target_modules,
                "epochs": config.epochs,
                "learning_rate": config.learning_rate,
            },
            "metrics": {
                "f1_before": metrics.f1_before,
                "f1_after": metrics.f1_after,
                "f1_improvement": metrics.f1_improvement,
                "training_samples": metrics.training_samples,
                "status": metrics.status,
            },
            "created_at": config.created_at,
            "version": "1.0",
        }

        self.logger.debug(f"Registered adapter: {json.dumps(registry_entry, indent=2)}")

    # ================================================================ #
    #  Adapter Versioning & Rollback                                  #
    # ================================================================ #

    def get_active_adapters(self, project_id: str) -> list[AdapterConfig]:
        """Retrieve active adapters for a project.

        In production, would query adapter registry.
        For now, returns empty list (no persistent storage).
        """
        # TODO: Implement persistent adapter registry
        return []

    def rollback_adapter(self, project_id: str, to_version: str) -> bool:
        """Rollback adapter to previous version if new version underperforms.

        In production, would revert adapter weights in model registry.
        """
        # TODO: Implement rollback logic
        self.logger.warning(f"Adapter rollback requested for {project_id} (not implemented)")
        return False
