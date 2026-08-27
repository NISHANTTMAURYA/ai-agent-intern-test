"""Automated evaluation test runner executing 2 test cases per category across 10 core dimensions.

Evaluates RAG retrieval, tool use, reliability, data privacy, conversation state,
multi-source grounding, prompt security, groundedness, safe abstention, and source conflicts.
"""

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.agent import AgentRunner, AgentResponse
from evaluation.assertions import (
    check_must_include,
    check_must_not_include,
    check_must_include_concepts,
    check_required_sources,
    check_forbidden_sources,
    check_tool_behavior,
    check_handoff,
    check_must_refuse_to_disclose,
    check_must_not_follow,
    check_must_not_invent,
    check_must_not_silently_choose_one,
    check_must_ask_for,
)


def load_all_cases() -> List[Dict[str, Any]]:
    """Load visible and custom evaluation test cases."""
    eval_dir = BASE_DIR / "evaluation"
    visible_file = eval_dir / "visible-cases.json"
    custom_file = eval_dir / "cases" / "custom-cases.json"
    
    cases: List[Dict[str, Any]] = []
    for filepath in [visible_file, custom_file]:
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                cases.extend(json.load(f).get("cases", []))
    return cases


def evaluate_single_case(agent: AgentRunner, case: Dict[str, Any]) -> Dict[str, Any]:
    """Execute multi-turn conversation for a single case and evaluate assertions."""
    case_id = case.get("id", "unknown_case")
    category = case.get("category", "general")
    messages = case.get("messages", [])
    expect = case.get("expect", {})
    
    session_id = f"eval_{case_id}_{uuid.uuid4().hex[:6]}"
    final_res: AgentResponse = None
    for msg in messages:
        final_res = agent.chat(user_message=msg.get("content", ""), session_id=session_id)
        
    ans = final_res.answer if final_res else ""
    sources = final_res.sources if final_res else []
    tool = final_res.tool_called if final_res else None
    handoff = final_res.handoff_recommended if final_res else False
    
    passed = True
    fails = []
    
    # 1. Required terms check
    if "must_include" in expect:
        ok, msg = check_must_include(ans, expect["must_include"])
        if not ok: passed = False; fails.append(msg)
        
    # 2. Forbidden terms check
    if "must_not_include" in expect:
        ok, msg = check_must_not_include(ans, expect["must_not_include"])
        if not ok: passed = False; fails.append(msg)
        
    # 3. Conceptual keywords check
    if "must_include_concepts" in expect:
        ok, msg = check_must_include_concepts(ans, expect["must_include_concepts"])
        if not ok: passed = False; fails.append(msg)
        
    # 4. Required citations check
    if "required_sources" in expect:
        ok, msg = check_required_sources(sources, expect["required_sources"], ans)
        if not ok: passed = False; fails.append(msg)
        
    # 5. Forbidden superseded citations check
    if "forbidden_sources_as_authority" in expect:
        ok, msg = check_forbidden_sources(sources, expect["forbidden_sources_as_authority"], ans)
        if not ok: passed = False; fails.append(msg)
        
    # 6. Tool calling contract check
    if "tool" in expect:
        ok, msg = check_tool_behavior(tool, expect["tool"])
        if not ok: passed = False; fails.append(msg)
        
    # 7. Human handoff flag check
    if "handoff" in expect:
        ok, msg = check_handoff(handoff, expect["handoff"])
        if not ok: passed = False; fails.append(msg)

    # 8. Explicit refusal-to-disclose check
    if "must_refuse_to_disclose" in expect:
        ok, msg = check_must_refuse_to_disclose(ans, expect["must_refuse_to_disclose"])
        if not ok: passed = False; fails.append(msg)

    # 9. Agent must not follow/comply with forbidden instructions
    if "must_not_follow" in expect:
        ok, msg = check_must_not_follow(ans, expect["must_not_follow"])
        if not ok: passed = False; fails.append(msg)

    # 10. Agent must not invent/fabricate listed fact categories
    if "must_not_invent" in expect:
        ok, msg = check_must_not_invent(ans, expect["must_not_invent"])
        if not ok: passed = False; fails.append(msg)

    # 11. Agent must not silently choose one side of a genuine source conflict
    if "must_not_silently_choose_one" in expect:
        ok, msg = check_must_not_silently_choose_one(ans, expect["must_not_silently_choose_one"])
        if not ok: passed = False; fails.append(msg)

    # 12. Agent must ask the customer for listed missing information
    if "must_ask_for" in expect:
        ok, msg = check_must_ask_for(ans, expect["must_ask_for"])
        if not ok: passed = False; fails.append(msg)
        
    return {
        "id": case_id,
        "category": category,
        "passed": passed,
        "failures": fails,
        "answer": ans,
        "handoff": handoff
    }


def run_evaluation_suite(tenant_id: str = "aster-and-row", delay_seconds: float = 0.0) -> Dict[str, Any]:
    """
    Execute the full evaluation suite with rate-limit friendly throttling.
    
    Steps:
    1. Load all 20 test cases.
    2. Sequentially evaluate each case with rate-limit pacing.
    3. Aggregate pass/fail statistics across 10 evaluation categories.
    4. Print structured summary table and return metrics dictionary.
    """
    cases = load_all_cases()
    runner = AgentRunner(tenant_id=tenant_id)
    
    print("\n" + "=" * 65)
    print(f"  Running AI Support Agent Evaluation Suite")
    print(f"  Total Cases: {len(cases)} (2 in each of {len(set(c.get('category') for c in cases))} categories)")
    print("=" * 65 + "\n")
    
    category_stats: Dict[str, Dict[str, int]] = {}
    results = []
    
    for idx, case in enumerate(cases, 1):
        res = evaluate_single_case(runner, case)
        cat = res["category"]
        
        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "passed": 0}
        category_stats[cat]["total"] += 1
        
        if res["passed"]:
            category_stats[cat]["passed"] += 1
            status_str = "✅ PASS"
        else:
            status_str = "❌ FAIL"
            
        print(f"[{idx:02d}/{len(cases):02d}] {status_str} | {res['id']:<38} | Category: {cat}")
        if not res["passed"]:
            for f in res["failures"]:
                print(f"      └─ ⚠️ {f}")
        results.append(res)
        
        # Pacing delay to avoid burst rate-limiting
        if delay_seconds > 0 and idx < len(cases):
            time.sleep(delay_seconds)
        
    total = len(cases)
    passed_count = sum(s["passed"] for s in category_stats.values())
    pct = round((passed_count / total) * 100, 1) if total > 0 else 0
    
    print("\n" + "=" * 65)
    print(" CATEGORY EVALUATION SUMMARY")
    print("=" * 65)
    print(f"{'Category':<28} | {'Passed':<8} | {'Total':<8} | {'Pass Rate':<10}")
    print("-" * 65)
    for cat, s in sorted(category_stats.items()):
        rate = round((s["passed"] / s["total"]) * 100, 1) if s["total"] > 0 else 0
        print(f"{cat:<28} | {s['passed']:<8} | {s['total']:<8} | {rate:>7.1f}%")
    print("-" * 65)
    print(f"{'OVERALL TOTAL':<28} | {passed_count:<8} | {total:<8} | {pct:>7.1f}%\n")
    
    return {
        "total": total,
        "passed": passed_count,
        "overall_percentage": pct,
        "category_stats": category_stats,
        "results": results
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation test cases.")
    parser.add_argument("--delay", type=float, default=0.2, help="Delay in seconds between test cases (default: 0.2)")
    args = parser.parse_args()
    res = run_evaluation_suite(delay_seconds=args.delay)
    sys.exit(0 if res["passed"] == res["total"] else 1)
