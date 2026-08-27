"""Pytest automated test suite for Aster & Row Support Agent.

Uses clean parameterized test discovery and deterministic assertion checks across all categories.
"""

import pytest
from src.agent import AgentRunner
from evaluation.run_eval import load_all_cases, evaluate_single_case, run_evaluation_suite


@pytest.fixture(scope="session")
def agent():
    """Shared AgentRunner instance for pytest suite."""
    return AgentRunner(tenant_id="aster-and-row")


@pytest.mark.parametrize("case", load_all_cases(), ids=lambda c: f"{c.get('category')}:{c.get('id')}")
def test_evaluation_case(agent, case):
    """Parameterized test verifying all individual evaluation cases pass with zero failures."""
    result = evaluate_single_case(agent, case)
    assert result["passed"] is True, f"Case {result['id']} failed with errors: {result['failures']}"


def test_full_suite_metrics_summary():
    """Verify that the full evaluation suite achieves a 100% pass rate."""
    summary = run_evaluation_suite(tenant_id="aster-and-row")
    assert summary["passed"] == summary["total"]
    assert summary["overall_percentage"] == 100.0
