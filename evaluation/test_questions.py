"""
A small set of test questions for manually or automatically evaluating
the research agent's behavior.

Run with: python -m evaluation.test_questions
(from the research-agent/ root, with dependencies installed and
ANTHROPIC_API_KEY set)
"""

from dotenv import load_dotenv

from agent.graph import run_agent

load_dotenv()

TEST_QUESTIONS = [
    {
        "id": "q1_factual_current",
        "question": "Who is the current CEO of Anthropic?",
        "expects_tool_use": True,
        "notes": "Current-role fact — should trigger at least one web_search.",
    },
    {
        "id": "q2_conceptual",
        "question": "What is LangGraph and how does it differ from LangChain's AgentExecutor?",
        "expects_tool_use": True,
        "notes": "Should search, likely read at least one doc page for accuracy.",
    },
    {
        "id": "q3_multi_hop",
        "question": "What year was the founder of the company that makes Zapier born?",
        "expects_tool_use": True,
        "notes": "Multi-hop — should need 2+ distinct searches/reads.",
    },
    {
        "id": "q4_no_search_needed",
        "question": "What is 15 * 23?",
        "expects_tool_use": False,
        "notes": "Pure reasoning — planner should NOT call tools, should go straight to READY.",
    },
    {
        "id": "q5_ambiguous_source",
        "question": "What are the main differences between Python's asyncio and threading modules?",
        "expects_tool_use": True,
        "notes": "Checks whether planner reads more than just snippets for a technical comparison.",
    },
]


def run_all(verbose: bool = True) -> list[dict]:
    results = []
    for case in TEST_QUESTIONS:
        final_state = run_agent(case["question"])

        tool_calls_made = any(
            getattr(m, "tool_calls", None) for m in final_state["messages"]
        )

        result = {
            "id": case["id"],
            "question": case["question"],
            "expected_tool_use": case["expects_tool_use"],
            "actual_tool_use": tool_calls_made,
            "matched_expectation": tool_calls_made == case["expects_tool_use"],
            "iterations": final_state.get("iterations", 0),
            "final_answer": final_state.get("final_answer"),
        }
        results.append(result)

        if verbose:
            status = "✅" if result["matched_expectation"] else "❌"
            print(f"{status} [{case['id']}] iterations={result['iterations']}")
            print(f"    Q: {case['question']}")
            print(f"    Expected tool use: {case['expects_tool_use']}, got: {tool_calls_made}")
            print(f"    Notes: {case['notes']}")
            print()

    return results


if __name__ == "__main__":
    run_all()