#!/usr/bin/env python3
"""
End-to-end workflow test for deep agent framework.
Tests full execution pipeline with mock GitHub data.
"""

import logging
import json
from datetime import datetime, timedelta
from src.agents.state import OrchestratorState, GitHubIssue
from src.agents.orchestrator import MasterOrchestrator
from src.agents.llm_config import OllamaClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)-8s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)


def create_mock_state() -> OrchestratorState:
    """Create a mock OrchestratorState with sample data for testing."""

    now = datetime.now()

    # Mock GitHub issues
    issues = [
        GitHubIssue(
            number=1,
            title="Fix authentication bug",
            body="Users unable to login with OAuth",
            state="closed",
            labels=["bug", "priority-high"],
            created_at=(now - timedelta(days=5)).isoformat(),
        ),
        GitHubIssue(
            number=2,
            title="Add dark mode support",
            body="UI redesign for dark theme",
            state="open",
            labels=["feature", "ui"],
            created_at=(now - timedelta(days=10)).isoformat(),
        ),
        GitHubIssue(
            number=3,
            title="Optimize database queries",
            body="Performance improvement for slow reports",
            state="open",
            labels=["performance", "backend"],
            created_at=(now - timedelta(days=3)).isoformat(),
        ),
    ]

    # Create base state
    state = OrchestratorState(
        repositories=["Mintplex-Labs/anything-llm"],
        eval_mode="resilient",
        github_issues=issues,
        github_prs=[],
        commits=[],
        milestone_data={
            "created_at": (now - timedelta(days=7)).isoformat(),
            "due_on": now.isoformat(),
        },
        created_at=now.isoformat(),
    )

    logger.info(f"Created mock state with {len(issues)} issues")
    return state


def test_orchestrator_invocation():
    """Test full orchestrator invocation with mock data."""

    logger.info("=" * 70)
    logger.info("END-TO-END WORKFLOW TEST")
    logger.info("=" * 70)

    try:
        # Create mock state
        logger.info("\n[Step 1] Creating mock input state...")
        input_state = create_mock_state()

        # Create orchestrator
        logger.info("[Step 2] Initializing MasterOrchestrator...")
        orchestrator = MasterOrchestrator()

        # Execute workflow
        logger.info("[Step 3] Invoking agent workflow...")
        logger.info(f"  • Using {len(input_state.repositories)} repository")
        logger.info(f"  • Processing {len(input_state.github_issues)} mock issues")

        result = orchestrator.invoke(input_state)

        # Validate output
        logger.info("\n[Step 4] Validating workflow output...")

        # Smoke checks: verify orchestration health regardless of strict model quality.
        checks = {
            "Workflow complete": result.workflow_complete,
            "Resilient mode enabled": result.eval_mode == "resilient",
            "Narrative produced": result.narrative_explanation is not None and len(result.narrative_explanation) > 0,
            "Workflow ran without crashes": True,  # If we got here, no crashes
            "Features extracted": result.features is not None and len(result.features) > 0,
            "Run metrics captured": bool(result.run_metrics),
            "Run metrics artifact path": bool(result.run_metrics_artifact),
        }

        all_passed = all(checks.values())

        logger.info("\n[Validation Results]")
        for check, passed in checks.items():
            status = "✓" if passed else "✗"
            logger.info(f"  {status} {check}")

        if result.errors:
            logger.warning("\n[Errors]")
            for error in result.errors:
                logger.warning(f"  - {error}")

        # Print summary
        logger.info("\n[Workflow Summary]")
        logger.info(f"  • Total execution logs: {len(result.execution_logs)}")
        logger.info(f"  • Evaluation mode: {result.eval_mode}")
        logger.info(f"  • Analysis source: {result.analysis_source}")
        logger.info(f"  • Risk source: {result.risk_source}")
        logger.info(f"  • Recommendation source: {result.recommendation_source}")
        logger.info(f"  • Identified risks: {len(result.identified_risks)}")
        logger.info(f"  • Recommendations: {len(result.recommendations)}")

        if result.sprint_analysis:
            analysis = result.sprint_analysis
            logger.info(f"  • Completion probability: {analysis.get('completion_probability', 'N/A')}%")
            logger.info(f"  • Health status: {analysis.get('health_status', 'N/A')}")

        # Print narrative excerpt
        if result.narrative_explanation:
            lines = result.narrative_explanation.split('\n')[:5]
            logger.info("\n[Narrative Excerpt]")
            for line in lines:
                logger.info(f"  {line}")

        logger.info("\n" + "=" * 70)
        if all_passed:
            logger.info("✓ END-TO-END TEST PASSED")
        else:
            logger.info("✗ END-TO-END TEST FAILED")
        logger.info("=" * 70)

        return all_passed

    except Exception as e:
        logger.error(f"\n✗ TEST FAILED: {e}", exc_info=True)
        logger.info("=" * 70)
        return False


if __name__ == "__main__":
    success = test_orchestrator_invocation()
    exit(0 if success else 1)
