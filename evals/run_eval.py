"""Entry point: run the full evaluation suite against the live app.

    python -m evals.run_eval

Creates/reuses the LangSmith dataset (evals/dataset.py), runs every example
through the real run_pipeline (guardrails -> agent -> guardrails, exactly
like a live request), scores each with the code + LLM-as-judge evaluators
in evaluators.py, and uploads results as a new LangSmith experiment you can
inspect under Datasets & Experiments.
"""
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langsmith.evaluation import evaluate

from app.pipeline import run_pipeline
from evals.dataset import build_or_get_dataset
from evals.evaluators import ALL_EVALUATORS


def target(inputs: dict) -> dict:
    """Wraps the real pipeline — same guardrails -> agent -> guardrails path
    a live request takes — and shapes the result for the evaluators above."""
    query = inputs["query"]
    trace_id = str(uuid.uuid4())
    thread_id = str(uuid.uuid4())
    try:
        response = run_pipeline(query, thread_id, trace_id)
    except Exception as exc:
        return {"answer": str(exc), "status": "error", "tools_used": [], "guardrail_stage": None}

    guardrail_stage = None
    if response.status == "blocked" and response.guardrail_events:
        guardrail_stage = response.guardrail_events[0].stage

    return {
        "answer": response.answer,
        "status": response.status,
        "tools_used": response.tools_used,
        "guardrail_stage": guardrail_stage,
    }


def main():
    dataset = build_or_get_dataset()
    print(f"Running evaluation against dataset '{dataset.name}' ({dataset.id})...")
    results = evaluate(
        target,
        data=dataset.name,
        evaluators=ALL_EVALUATORS,
        experiment_prefix="stock-analyzer-eval",
        max_concurrency=2,
    )
    print("\nDone. View results in LangSmith under Datasets & Experiments.")
    return results


if __name__ == "__main__":
    main()
