"""
Main entry point for sprint intelligence deep agent system.
Supports CLI and API modes with LangChain DeepAgents orchestration.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

from src.agents.orchestrator import create_orchestrator
from src.agents.state import OrchestratorState
from src.agents.llm_config import get_ollama_client, get_ollama_config
from src.agents.tools import sanitize_result_payload
from src.research.harness import run_research_harness


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# CLI Interface
# ============================================================================

def run_sprint_analysis(
    repository_urls: list[str],
    sprint_id: Optional[str] = None,
    milestone_data: Optional[dict] = None,
    output_file: Optional[str] = None,
) -> dict:
    """
    Run complete sprint intelligence analysis on specified repositories.

    Args:
        repository_urls: List of GitHub repository URLs (e.g., ["owner/repo"])
        sprint_id: Optional sprint/milestone identifier
        milestone_data: Optional milestone metadata from GitHub API
        output_file: Optional path to save results JSON

    Returns:
        Dictionary with analysis results
    """
    logger.info("=" * 80)
    logger.info("SPRINT INTELLIGENCE ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"Repositories: {repository_urls}")
    logger.info(f"Sprint ID: {sprint_id or 'Not specified'}")

    # Initialize state
    state = OrchestratorState(
        repositories=repository_urls,
        sprint_id=sprint_id,
        milestone_data=milestone_data or {},
    )

    try:
        # Create and run orchestrator
        logger.info("\nInitializing orchestrator...")
        orchestrator = create_orchestrator()

        logger.info("Executing agent workflow...")
        final_state = orchestrator.invoke(state)

        # Prepare results
        analysis_dict = final_state.sprint_analysis or {}
        results = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "sprint_id": final_state.sprint_id,
            "repositories": final_state.repositories,
            "analysis": {
                "completion_probability": analysis_dict.get("completion_probability"),
                "health_status": analysis_dict.get("health_status"),
                "risks_count": len(final_state.identified_risks),
                "recommendations_count": len(final_state.recommendations),
            },
            "narrative": final_state.narrative_explanation,
            "run_metrics": final_state.run_metrics,
            "run_metrics_artifact": final_state.run_metrics_artifact,
            "risks": [
                r.dict() if hasattr(r, "dict") else r
                for r in final_state.identified_risks
            ],
            "recommendations": [
                r.dict() if hasattr(r, "dict") else r
                for r in final_state.recommendations
            ],
            "execution_logs": final_state.execution_logs,
            "errors": final_state.errors,
        }
        results = sanitize_result_payload(results)

        # Save output if requested
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"\nResults saved to {output_file}")

        logger.info("\n" + "=" * 80)
        logger.info("ANALYSIS COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Completion Probability: {results['analysis']['completion_probability']:.1f}%")
        logger.info(f"Health Status: {results['analysis']['health_status']}")
        logger.info(f"Risks Identified: {results['analysis']['risks_count']}")
        logger.info(f"Recommendations: {results['analysis']['recommendations_count']}")
        logger.info(f"Run metrics artifact: {results.get('run_metrics_artifact')}")

        return results

    except Exception as e:
        logger.error(f"\nAnalysis failed: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
        }


# ============================================================================
# CLI Commands
# ============================================================================

def cmd_analyze():
    """CLI command: analyze a repository."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Sprint Intelligence Analysis"
    )
    parser.add_argument(
        "repositories",
        nargs="+",
        help="GitHub repository URLs (e.g., owner/repo)",
    )
    parser.add_argument(
        "-s", "--sprint-id",
        help="Sprint or milestone identifier",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file for results (JSON)",
    )
    parser.add_argument(
        "--config-only",
        action="store_true",
        help="Show configuration and exit",
    )

    args = parser.parse_args()

    # Show configuration if requested
    if args.config_only:
        print("\n" + "=" * 80)
        print("CONFIGURATION")
        print("=" * 80)
        config = get_ollama_config()
        print(f"Ollama Base URL: {config.base_url}")
        print(f"Model: {config.model_name}")
        print(f"Temperature: {config.temperature}")
        print(f"Max Context: {config.num_ctx}")
        print(f"GPU Layers: {config.num_gpu}")
        return

    # Run analysis
    results = run_sprint_analysis(
        repository_urls=args.repositories,
        sprint_id=args.sprint_id,
        output_file=args.output,
    )

    # Print summary
    if results["status"] == "success":
        analysis = results.get("analysis", {})
        print(f"\nSummary:")
        print(f"  Completion: {analysis.get('completion_probability', 'N/A')}%")
        print(f"  Status: {analysis.get('health_status', 'N/A')}")
        print(f"  Risks: {analysis.get('risks_count', 0)}")
        print(f"  Recommendations: {analysis.get('recommendations_count', 0)}")

        narrative = results.get("narrative", "")
        if narrative:
            print(f"\nNarrative:\n{narrative}")
    else:
        print(f"Error: {results.get('error', 'Unknown error')}")


def cmd_health_check():
    """CLI command: check system health."""
    print("\n" + "=" * 80)
    print("SPRINT INTELLIGENCE SYSTEM HEALTH CHECK")
    print("=" * 80)

    # Check Ollama connection
    try:
        config = get_ollama_config()
        client = get_ollama_client()
        client.close()
        print("✓ Ollama connection: OK")
        print(f"  Model: {config.model_name}")
        print(f"  Endpoint: {config.base_url}")
    except Exception as e:
        print("✗ Ollama connection: FAILED")
        print(f"  Error: {e}")

    # Check dependencies
    try:
        import langgraph
        print("✓ LangGraph: OK")
    except ImportError:
        print("✗ LangGraph: NOT INSTALLED")

    try:
        import chromadb
        print("✓ ChromaDB: OK")
    except ImportError:
        print("✗ ChromaDB: NOT INSTALLED")

    try:
        import pydantic
        print("✓ Pydantic: OK")
    except ImportError:
        print("✗ Pydantic: NOT INSTALLED")

    print("\n" + "=" * 80)


def cmd_research_harness():
    """CLI command: run objective-level research harness."""
    import argparse

    parser = argparse.ArgumentParser(description="Run objective-level research harness")
    parser.add_argument(
        "-o",
        "--output",
        default="artifacts/research/research_harness.json",
        help="Output path for harness artifact JSON",
    )
    parser.add_argument(
        "--claim-output",
        default="artifacts/research/research_claim_report.json",
        help="Output path for strict-only claim report JSON",
    )
    args = parser.parse_args()

    result = run_research_harness(args.output, args.claim_output)
    print("\nResearch Harness Summary")
    print(f"  Mode: {result.get('mode')}")
    print(f"  Status: {result.get('status')}")
    print(f"  Passed: {result.get('passed_objectives')}/{result.get('total_objectives')}")
    print(f"  Claim Ready: {result.get('claim_ready')}")
    print(f"  Artifact: {result.get('artifact_path')}")
    print(f"  Claim Artifact: {result.get('claim_artifact_path')}")

    failed_gates = [g for g in result.get("acceptance_gates", []) if not g.get("passed")]
    if failed_gates:
        print("  Failed Gates:")
        for gate in failed_gates:
            print(f"    - {gate.get('name')}")


# ============================================================================
# Main & Entry Point
# ============================================================================

def main():
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: python -m src.main [command] [args...]")
        print("\nCommands:")
        print("  analyze [repos...]    Analyze repository sprint")
        print("  health-check          Check system health")
        print("  research-harness      Run objective-level research validation")
        print("\nExample:")
        print("  python -m src.main analyze owner/repo")
        sys.exit(1)

    command = sys.argv[1]

    if command == "analyze":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        cmd_analyze()
    elif command == "health-check":
        cmd_health_check()
    elif command == "research-harness":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        cmd_research_harness()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
