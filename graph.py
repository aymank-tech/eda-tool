"""EDA Analysis Graph — LangGraph stateful graph implementing the reflection/self-correction loop."""

from typing import TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, END

SCORE_MAP = {"Excellent": 3, "Solid": 2, "Needs Work": 1}
MODEL_CHOICES = ["openai", "claude"]
MODEL_LABELS = {"openai": "GPT-5.4", "claude": "Claude Sonnet 4"}
MIN_ITERATIONS = 3


class EDAState(TypedDict):
    df: object                                           # pandas DataFrame
    summary: dict                                        # statistical summary from Visualizer
    visualizations: list                                 # [(title, img_b64, description), ...]
    scores: dict                                         # {dimension: (score, [reasons])}
    per_viz_evaluations: list                            # [(title, {dim: score}, [issues]), ...]
    feedback_history: Annotated[list, operator.add]      # accumulated across iterations
    try_count: int                                       # current iteration (0 = not started)
    max_tries: int                                       # max iterations allowed
    openai_api_key: str                                  # OpenAI API key
    anthropic_api_key: str                               # Anthropic API key
    overall_rating: str                                  # "Excellent" / "Solid" / "Needs Work"
    stagnated: bool                                      # True if score stopped improving
    iteration_results: Annotated[list, operator.add]     # full results per iteration for display
    last_viz_model: str                                  # model used for current iteration's visualizer
    visualizer_model: str                                # user-selected: "openai" or "claude"
    scorer_model: str                                    # user-selected: "openai" or "claude"


def _compute_rating(scores):
    """Compute overall rating from dimension scores."""
    values = [SCORE_MAP.get(s, 2) for s, _ in scores.values()]
    avg = sum(values) / len(values)
    if avg >= 2.75:
        return "Excellent"
    elif avg >= 1.75:
        return "Solid"
    return "Needs Work"


def _get_api_key(state, model):
    """Return the API key for the given model choice."""
    if model == "claude":
        return state["anthropic_api_key"]
    return state["openai_api_key"]


# ── Nodes ───────────────────────────────────────────────────────────────────


def visualizer_node(state: EDAState) -> dict:
    """Generate or refine visualizations."""
    import visualizer_agent

    df = state["df"]

    if state["try_count"] == 0:
        # First run — generate from scratch (programmatic)
        summary, visualizations = visualizer_agent.run(df)
        return {
            "summary": summary,
            "visualizations": visualizations,
            "try_count": 1,
            "last_viz_model": "Programmatic",
        }
    else:
        # Subsequent runs — use user-selected model
        model = state["visualizer_model"]
        api_key = _get_api_key(state, model)
        summary, visualizations, _changes = visualizer_agent.refine_with_llm(
            df,
            state["summary"],
            state["visualizations"],
            state["scores"],
            state["per_viz_evaluations"],
            api_key,
            model=model,
        )
        return {
            "summary": summary,
            "visualizations": visualizations,
            "try_count": state["try_count"] + 1,
            "last_viz_model": MODEL_LABELS.get(model, model),
        }


def scorer_node(state: EDAState) -> dict:
    """Score visualizations using the user-selected model."""
    import scorer_agent

    model = state["scorer_model"]
    api_key = _get_api_key(state, model)

    overall, per_viz = scorer_agent.run(
        state["df"],
        state["summary"],
        state["visualizations"],
        api_key,
        model=model,
    )

    rating = _compute_rating(overall)

    iteration_result = {
        "try": state["try_count"],
        "visualizations": list(state["visualizations"]),
        "summary": state["summary"],
        "scores": overall,
        "per_viz_evaluations": per_viz,
        "overall_rating": rating,
        "visualizer_model": state.get("last_viz_model", "Programmatic"),
        "scorer_model": MODEL_LABELS.get(model, model),
    }

    return {
        "scores": overall,
        "per_viz_evaluations": per_viz,
        "overall_rating": rating,
        "feedback_history": [{
            "try": state["try_count"],
            "rating": rating,
            "scores": overall,
            "per_viz": per_viz,
        }],
        "iteration_results": [iteration_result],
    }


def orchestrator_node(state: EDAState) -> dict:
    """Check for stagnation by comparing consecutive iteration scores."""
    history = state.get("feedback_history", [])

    if len(history) >= 2:
        prev_avg = sum(SCORE_MAP.get(s, 2) for s, _ in history[-2]["scores"].values()) / len(history[-2]["scores"])
        curr_avg = sum(SCORE_MAP.get(s, 2) for s, _ in history[-1]["scores"].values()) / len(history[-1]["scores"])
        if curr_avg <= prev_avg:
            return {"stagnated": True}

    return {"stagnated": False}


# ── Routing ─────────────────────────────────────────────────────────────────


def _should_continue(state: EDAState) -> str:
    # Exit immediately on Excellent
    if state["overall_rating"] == "Excellent":
        return "done"

    # Exit on plateau after at least 3 iterations
    if state.get("stagnated", False) and state["try_count"] >= MIN_ITERATIONS:
        return "done"

    # Exit on max tries
    if state["try_count"] >= state["max_tries"]:
        return "done"

    return "retry"


# ── Graph construction ──────────────────────────────────────────────────────


def build_graph():
    graph = StateGraph(EDAState)

    graph.add_node("visualizer", visualizer_node)
    graph.add_node("scorer", scorer_node)
    graph.add_node("orchestrator", orchestrator_node)

    graph.set_entry_point("visualizer")
    graph.add_edge("visualizer", "scorer")
    graph.add_edge("scorer", "orchestrator")
    graph.add_conditional_edges("orchestrator", _should_continue, {
        "retry": "visualizer",
        "done": END,
    })

    return graph.compile()


def run(df, openai_api_key, anthropic_api_key, max_tries=5,
        visualizer_model="openai", scorer_model="claude", on_progress=None):
    """Run the full EDA analysis loop.

    Args:
        df: pandas DataFrame to analyze
        openai_api_key: OpenAI API key
        anthropic_api_key: Anthropic API key
        max_tries: maximum number of Visualize→Score iterations
        visualizer_model: "openai" or "claude"
        scorer_model: "openai" or "claude"
        on_progress: optional callback(state_dict) called after each node

    Returns:
        Final state dict with iteration_results, overall_rating, etc.
    """
    compiled = build_graph()

    initial_state = {
        "df": df,
        "summary": {},
        "visualizations": [],
        "scores": {},
        "per_viz_evaluations": [],
        "feedback_history": [],
        "try_count": 0,
        "max_tries": max_tries,
        "openai_api_key": openai_api_key,
        "anthropic_api_key": anthropic_api_key,
        "overall_rating": "",
        "stagnated": False,
        "iteration_results": [],
        "last_viz_model": "",
        "visualizer_model": visualizer_model,
        "scorer_model": scorer_model,
    }

    final_state = None
    for state in compiled.stream(initial_state, stream_mode="values"):
        final_state = state
        if on_progress:
            on_progress(state)

    return final_state
