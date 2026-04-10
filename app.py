"""EDA Tool — Reflection-based EDA with Visualizer, Scorer, and Orchestrator agents."""

import streamlit as st
import pandas as pd
import numpy as np
import base64
import os
from palmerpenguins import load_penguins

DATASETS_DIR = os.path.join(os.path.dirname(__file__), "datasets")

SCORE_COLORS = {"Excellent": "🟢", "Solid": "🟡", "Needs Work": "🔴"}


def get_available_datasets():
    """Return a dict of {display_name: loader_function}."""
    datasets = {"Palmer Penguins": _load_penguins}

    titanic_csv = os.path.join(DATASETS_DIR, "titanic", "titanic.csv")
    if os.path.exists(titanic_csv):
        datasets["Titanic"] = lambda path=titanic_csv: pd.read_csv(path)

    return datasets


def _load_penguins():
    return load_penguins()


def _load_env_key(name):
    """Load a key from Streamlit secrets or .env file."""
    try:
        return st.secrets[name]
    except (KeyError, FileNotFoundError):
        env_path = os.path.join(os.path.dirname(__file__), ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    if line.strip().startswith(f"{name}="):
                        return line.strip().split("=", 1)[1].strip()
    return ""


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="EDA Tool", layout="wide")
st.title("Exploratory Data Analysis Tool")
st.markdown(
    "Choose a dataset and run the analysis. The tool uses a **reflection loop**: "
    "the Visualizer generates charts, the Scorer evaluates them against a rubric, "
    "and the Orchestrator decides whether to refine and retry."
)

# ── Load API keys ───────────────────────────────────────────────────────────
openai_api_key = _load_env_key("OPENAI_API_KEY")
anthropic_api_key = _load_env_key("ANTHROPIC_API_KEY")

# ── Sidebar ─────────────────────────────────────────────────────────────────
MODEL_OPTIONS = {"GPT-5.4": "openai", "Claude Sonnet 4": "claude"}

with st.sidebar:
    st.header("Settings")
    max_iterations = st.slider("Max iterations", min_value=3, max_value=10, value=5)
    visualizer_model_label = st.selectbox("Visualizer model", list(MODEL_OPTIONS.keys()), index=0)
    scorer_model_label = st.selectbox("Scorer model", list(MODEL_OPTIONS.keys()), index=1)
    visualizer_model = MODEL_OPTIONS[visualizer_model_label]
    scorer_model = MODEL_OPTIONS[scorer_model_label]

# ── Dataset selector ─────────────────────────────────────────────────────────
datasets = get_available_datasets()
dataset_name = st.selectbox("Select a dataset", list(datasets.keys()))

# Reset when dataset changes
if st.session_state.get("current_dataset") != dataset_name:
    st.session_state["current_dataset"] = dataset_name
    st.session_state.pop("result", None)

# ── Run button ──────────────────────────────────────────────────────────────
run_analysis = st.button("Run EDA Analysis", type="primary", use_container_width=True)

if run_analysis:
    if not openai_api_key or not anthropic_api_key:
        missing = []
        if not openai_api_key:
            missing.append("OPENAI_API_KEY")
        if not anthropic_api_key:
            missing.append("ANTHROPIC_API_KEY")
        st.warning(f"Missing API key(s): {', '.join(missing)}. Please add them to your .env file.")
    else:
        from graph import run as run_graph

        df = datasets[dataset_name]()

        # Track progress
        last_try = [0]
        last_scored = [0]

        with st.status("Running EDA analysis loop...", expanded=True) as status:

            def on_progress(state):
                current_try = state.get("try_count", 0)
                iterations = state.get("iteration_results", [])
                viz_model = state.get("last_viz_model", "")

                if current_try > last_try[0]:
                    if current_try == 1:
                        status.write(f"🎨 **Iteration {current_try}:** Generating initial visualizations (Programmatic)...")
                    else:
                        status.write(f"🔄 **Iteration {current_try}:** Generating visualizations with **{visualizer_model_label}**...")
                    last_try[0] = current_try

                if len(iterations) > last_scored[0]:
                    latest = iterations[-1]
                    rating = latest["overall_rating"]
                    icon = SCORE_COLORS.get(rating, "⚪")
                    status.write(f"📊 **Iteration {latest['try']}:** Scored by **{scorer_model_label}** — {icon} **{rating}**")

                    if rating == "Excellent" and current_try >= 3:
                        status.write("✅ Excellent score achieved!")
                    last_scored[0] = len(iterations)

            result = run_graph(
                df, openai_api_key, anthropic_api_key,
                max_tries=max_iterations,
                visualizer_model=visualizer_model,
                scorer_model=scorer_model,
                on_progress=on_progress,
            )

            # Final status
            final_rating = result["overall_rating"]
            n_iterations = len(result["iteration_results"])
            stagnated = result.get("stagnated", False)

            if final_rating == "Excellent":
                status.update(label=f"✅ Analysis complete — Excellent in {n_iterations} iteration(s)", state="complete")
            elif stagnated:
                status.update(label=f"⚠️ Analysis stopped — score plateaued at {final_rating} after {n_iterations} iterations", state="complete")
            else:
                status.update(label=f"⏹️ Analysis complete — {final_rating} after {n_iterations} iteration(s) (max reached)", state="complete")

        st.session_state["result"] = result

# ── Display results ─────────────────────────────────────────────────────────
if "result" in st.session_state:
    result = st.session_state["result"]
    iterations = result["iteration_results"]

    from scorer_agent import RUBRIC

    for i, it in enumerate(iterations):
        is_last = (i == len(iterations) - 1)
        rating = it["overall_rating"]
        icon = SCORE_COLORS.get(rating, "⚪")
        label = f"Iteration {it['try']} — {icon} {rating}"
        if is_last:
            label += " (final)"

        with st.expander(label, expanded=is_last):
            # ── Model info ─────────────────────────────────────────────
            viz_model = it.get("visualizer_model", "Programmatic")
            scorer_model = it.get("scorer_model", "?")
            mc1, mc2 = st.columns(2)
            mc1.info(f"**Visualizer:** {viz_model}")
            mc2.info(f"**Scorer:** {scorer_model}")

            # ── Visualizations ──────────────────────────────────────────
            st.subheader("Visualizations")
            for title, img_b64, description in it["visualizations"]:
                st.markdown(f"#### {title}")
                st.image(base64.b64decode(img_b64), use_container_width=True)
                st.info(description)

            # ── Per-visualization scores ────────────────────────────────
            st.subheader("Per-Visualization Scores & Top Issues")
            for title, scores, top_issues in it["per_viz_evaluations"]:
                st.markdown(f"#### {title}")
                score_cols = st.columns(len(RUBRIC))
                for col, dimension in zip(score_cols, RUBRIC):
                    dim_score = scores[dimension]
                    dim_icon = SCORE_COLORS.get(dim_score, "⚪")
                    col.metric(label=dimension, value=f"{dim_icon} {dim_score}")
                if top_issues:
                    st.markdown("**Top issues:**")
                    for issue in top_issues:
                        st.markdown(f"- {issue}")
                else:
                    st.success("No issues found.")

            # ── Aggregate scores ────────────────────────────────────────
            st.subheader("Aggregate Rubric Scores")
            for dimension, (score, reasons) in it["scores"].items():
                dim_icon = SCORE_COLORS.get(score, "⚪")
                st.markdown(f"### {dim_icon} {dimension} — **{score}**")
                for reason in reasons:
                    st.markdown(f"- {reason}")

            st.divider()
            st.markdown(f"## {icon} Overall Rating: **{rating}**")

    # ── Final summary ───────────────────────────────────────────────────
    st.divider()
    final = iterations[-1]
    final_icon = SCORE_COLORS.get(final["overall_rating"], "⚪")
    n = len(iterations)

    col1, col2, col3 = st.columns(3)
    col1.metric("Final Rating", f"{final_icon} {final['overall_rating']}")
    col2.metric("Iterations", n)
    if result.get("stagnated"):
        col3.metric("Status", "Plateaued")
    elif final["overall_rating"] == "Excellent":
        col3.metric("Status", "Target reached")
    else:
        col3.metric("Status", "Max iterations")
