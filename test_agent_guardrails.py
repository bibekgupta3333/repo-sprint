import unittest

from src.agents.orchestrator import MasterOrchestrator
from src.agents.state import OrchestratorState
from src.agents.tools import LLMInferenceTool, sanitize_result_payload


class SprintAnalysisGuardrailTests(unittest.TestCase):
    def test_parse_completion_response_normalizes_confidence_and_clamps_health(self) -> None:
        tool = LLMInferenceTool(None)

        parsed = tool._parse_completion_response(
            '{"completion_probability": 90, "health_status": "On Track", "confidence_score": 85, "health_score": 922.01, "reasoning": "stable delivery"}'
        )

        self.assertEqual(parsed["health_status"], "on_track")
        self.assertAlmostEqual(parsed["completion_probability"], 90.0)
        self.assertAlmostEqual(parsed["confidence_score"], 0.85)
        self.assertAlmostEqual(parsed["health_score"], 100.0)

    def test_sprint_analyzer_clamps_out_of_range_llm_inputs(self) -> None:
        state = OrchestratorState(
            repositories=["owner/repo"],
            sprint_analysis={
                "completion_probability": 90,
                "health_status": "On Track",
                "confidence_score": 85,
            },
            features={},
            dependency_graph={},
        )

        result = MasterOrchestrator.sprint_analyzer_node(None, state)

        self.assertAlmostEqual(result.sprint_analysis["confidence_score"], 0.85)
        self.assertGreaterEqual(result.sprint_analysis["health_score"], 0.0)
        self.assertLessEqual(result.sprint_analysis["health_score"], 100.0)

    def test_sanitize_result_payload_guardrails_all_user_facing_values(self) -> None:
        sanitized = sanitize_result_payload({
            "sprint_analysis": {
                "completion_probability": 0.92,
                "health_status": "unknown",
                "confidence_score": 85,
                "health_score": 922.01,
                "delivery_score": 140,
                "momentum_score": -3,
                "quality_score": 101,
                "collaboration_score": 0.35,
                "dependency_risk_score": 160,
                "key_signals": [" stalled reviews ", "", "stalled reviews"],
            },
            "identified_risks": [{
                "risk_type": "Review Bottleneck",
                "severity": 80,
                "description": "slow review queue",
                "affected_issues": ["#12", -3, 12, "foo"],
            }],
            "recommendations": [{
                "title": "Speed reviews",
                "description": "Shorten review latency",
                "priority": "urgent",
                "expected_impact": "Faster merges",
                "action": "Rotate reviewers",
                "evidence_source": 42,
            }],
            "run_metrics": {
                "f1_score": 87,
                "parse_success_rate": 140,
                "fallback_rate": -5,
                "citation_quality": {
                    "total_citations": 2,
                    "non_empty_citations": 5,
                    "score": 175,
                },
                "counts": {
                    "risks": -4,
                    "recommendations": 3,
                    "errors": -1,
                    "execution_logs": 8,
                },
            },
            "evidence_citations": [" https://example.com/1 ", "", "https://example.com/1"],
            "errors": [" bad output ", "", "bad output"],
            "execution_logs": [" step 1 ", "step 1", "step 2"],
        })

        analysis = sanitized["sprint_analysis"]
        self.assertAlmostEqual(analysis["completion_probability"], 92.0)
        self.assertEqual(analysis["health_status"], "on_track")
        self.assertAlmostEqual(analysis["confidence_score"], 0.85)
        self.assertAlmostEqual(analysis["health_score"], 100.0)
        self.assertAlmostEqual(analysis["delivery_score"], 100.0)
        self.assertAlmostEqual(analysis["momentum_score"], 0.0)
        self.assertAlmostEqual(analysis["quality_score"], 100.0)
        self.assertAlmostEqual(analysis["collaboration_score"], 35.0)
        self.assertAlmostEqual(analysis["dependency_risk_score"], 100.0)
        self.assertEqual(analysis["key_signals"], ["stalled reviews"])

        risk = sanitized["identified_risks"][0]
        self.assertEqual(risk["risk_type"], "review_bottleneck")
        self.assertAlmostEqual(risk["severity"], 0.8)
        self.assertEqual(risk["affected_issues"], [12])

        recommendation = sanitized["recommendations"][0]
        self.assertEqual(recommendation["priority"], "high")
        self.assertEqual(recommendation["evidence_source"], "42")

        run_metrics = sanitized["run_metrics"]
        self.assertAlmostEqual(run_metrics["f1_score"], 0.87)
        self.assertAlmostEqual(run_metrics["parse_success_rate"], 1.0)
        self.assertAlmostEqual(run_metrics["fallback_rate"], 0.0)
        self.assertEqual(run_metrics["citation_quality"], {
            "total_citations": 2,
            "non_empty_citations": 2,
            "score": 1.0,
        })
        self.assertEqual(run_metrics["counts"], {
            "risks": 0,
            "recommendations": 3,
            "errors": 0,
            "execution_logs": 8,
        })

        self.assertEqual(sanitized["evidence_citations"], ["https://example.com/1"])
        self.assertEqual(sanitized["errors"], ["bad output"])
        self.assertEqual(sanitized["execution_logs"], ["step 1", "step 2"])


if __name__ == "__main__":
    unittest.main()
