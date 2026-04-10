"""EDA Tool — Visualizer & Critic Agents for any tabular dataset."""

import streamlit as st
import pandas as pd
import numpy as np
import base64
import os
from palmerpenguins import load_penguins

DATASETS_DIR = os.path.join(os.path.dirname(__file__), "datasets")


def get_available_datasets():
    """Return a dict of {display_name: loader_function}."""
    datasets = {"Palmer Penguins": _load_penguins}

    # Titanic
    titanic_csv = os.path.join(DATASETS_DIR, "titanic", "titanic.csv")
    if os.path.exists(titanic_csv):
        datasets["Titanic"] = lambda path=titanic_csv: pd.read_csv(path)

    return datasets


def _load_penguins():
    return load_penguins()


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="EDA Tool", layout="wide")
st.title("Exploratory Data Analysis Tool")
st.markdown(
    "Choose a dataset, run the **Visualizer Agent** to perform EDA, "
    "the **Critic Agent** to evaluate the results, "
    "then the **Refiner Agent** to improve the visualizations based on feedback."
)

# ── Load API key from Streamlit secrets or .env ─────────────────────────────
openai_api_key = ""
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
except (KeyError, FileNotFoundError):
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.strip().startswith("OPENAI_API_KEY="):
                    openai_api_key = line.strip().split("=", 1)[1].strip()
                    break

# ── Dataset selector ─────────────────────────────────────────────────────────
datasets = get_available_datasets()
dataset_name = st.selectbox("Select a dataset", list(datasets.keys()))

# Reset results when dataset changes
if st.session_state.get("current_dataset") != dataset_name:
    st.session_state["current_dataset"] = dataset_name
    st.session_state.pop("summary", None)
    st.session_state.pop("visualizations", None)
    st.session_state.pop("evaluations", None)
    st.session_state.pop("per_viz_evaluations", None)
    st.session_state.pop("refined_summary", None)
    st.session_state.pop("refined_visualizations", None)
    st.session_state.pop("refiner_changes", None)
    st.session_state.pop("refined_evaluations", None)
    st.session_state.pop("refined_per_viz_evaluations", None)

# ── Buttons ──────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
run_visualizer = col1.button("Run Visualizer Agent", type="primary", use_container_width=True)
run_critic = col2.button("Run Critic Agent", use_container_width=True)
run_refiner = col3.button("Run Refiner Agent", use_container_width=True)

# ── Visualizer ───────────────────────────────────────────────────────────────
if run_visualizer:
    with st.spinner("Visualizer Agent is analyzing the dataset..."):
        import visualizer_agent

        df = datasets[dataset_name]()
        summary, visualizations = visualizer_agent.run(df)
        st.session_state["summary"] = summary
        st.session_state["visualizations"] = visualizations
        st.session_state.pop("evaluations", None)
        st.session_state.pop("per_viz_evaluations", None)
        st.session_state.pop("refined_summary", None)
        st.session_state.pop("refined_visualizations", None)
        st.session_state.pop("refiner_changes", None)
    st.session_state.pop("refined_evaluations", None)
    st.session_state.pop("refined_per_viz_evaluations", None)

if "summary" in st.session_state:
    summary = st.session_state["summary"]
    visualizations = st.session_state["visualizations"]

    st.header("Visualizer Agent Results")

    # ── Statistical Summary ──────────────────────────────────────────────────
    st.subheader("Statistical Summary")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**Dataset shape:** {summary['shape']}")
        st.markdown("**Missing values:**")
        if isinstance(summary["missing_values"], dict):
            st.dataframe(
                pd.DataFrame.from_dict(summary["missing_values"], orient="index", columns=["Count"]),
                use_container_width=True,
            )
        else:
            st.write(summary["missing_values"])

    with col_b:
        # Show categorical value counts
        if summary.get("categorical_counts"):
            for col_name, counts in list(summary["categorical_counts"].items())[:2]:
                st.markdown(f"**{col_name} counts:**")
                st.dataframe(
                    pd.DataFrame.from_dict(counts, orient="index", columns=["Count"]),
                    use_container_width=True,
                )

    if "group_means" in summary:
        st.markdown(f"**Mean values by {summary['group_col']}:**")
        st.dataframe(summary["group_means"], use_container_width=True)

    if "numeric_summary" in summary:
        st.markdown("**Descriptive statistics:**")
        st.dataframe(summary["numeric_summary"], use_container_width=True)

    if "correlations" in summary:
        st.markdown("**Correlation matrix:**")
        st.dataframe(summary["correlations"], use_container_width=True)

    # ── Visualizations ───────────────────────────────────────────────────────
    st.subheader("Visualizations")
    for title, img_b64, description in visualizations:
        st.markdown(f"#### {title}")
        st.image(base64.b64decode(img_b64), use_container_width=True)
        st.info(description)

# ── Critic ───────────────────────────────────────────────────────────────────
if run_critic:
    if "summary" not in st.session_state:
        st.warning("Run the Visualizer Agent first so the Critic has something to evaluate.")
    elif not openai_api_key:
        st.warning("OpenAI API key not found. Please add OPENAI_API_KEY=... to your .env file.")
    else:
        with st.spinner("Critic Agent is consulting ChatGPT (GPT-5.4)..."):
            import critic_agent

            df = datasets[dataset_name]()
            overall_evaluations, per_viz_evaluations = critic_agent.run(
                df,
                st.session_state["summary"],
                st.session_state["visualizations"],
                openai_api_key,
            )
            st.session_state["evaluations"] = overall_evaluations
            st.session_state["per_viz_evaluations"] = per_viz_evaluations
            st.session_state.pop("refined_summary", None)
            st.session_state.pop("refined_visualizations", None)
            st.session_state.pop("refiner_changes", None)
    st.session_state.pop("refined_evaluations", None)
    st.session_state.pop("refined_per_viz_evaluations", None)

if "evaluations" in st.session_state:
    st.header("Critic Agent Evaluation")

    SCORE_COLORS = {
        "Excellent": "🟢",
        "Solid": "🟡",
        "Needs Work": "🔴",
    }

    # ── Per-Visualization Feedback ──────────────────────────────────────────
    st.subheader("Per-Visualization Scores & Top Issues")

    from critic_agent import RUBRIC

    for title, scores, top_issues in st.session_state["per_viz_evaluations"]:
        st.markdown(f"#### {title}")

        # Show rubric scores for this visualization
        score_cols = st.columns(len(RUBRIC))
        for col, dimension in zip(score_cols, RUBRIC):
            dim_score = scores[dimension]
            icon = SCORE_COLORS.get(dim_score, "⚪")
            col.metric(label=dimension, value=f"{icon} {dim_score}")

        # Show top 3 issues
        if top_issues:
            st.markdown("**Top issues:**")
            for issue in top_issues:
                st.markdown(f"- {issue}")
        else:
            st.success("No issues found — this visualization scores well across all dimensions.")

    st.divider()

    # ── Aggregate Rubric Scores ─────────────────────────────────────────────
    st.subheader("Aggregate Rubric Scores")

    for dimension, (score, reasons) in st.session_state["evaluations"].items():
        icon = SCORE_COLORS.get(score, "⚪")
        st.markdown(f"### {icon} {dimension} — **{score}**")
        for reason in reasons:
            st.markdown(f"- {reason}")

    scores = [s for s, _ in st.session_state["evaluations"].values()]
    score_map = {"Excellent": 3, "Solid": 2, "Needs Work": 1}
    avg = sum(score_map[s] for s in scores) / len(scores)
    if avg >= 2.75:
        overall = "Excellent"
    elif avg >= 1.75:
        overall = "Solid"
    else:
        overall = "Needs Work"

    st.divider()
    icon = SCORE_COLORS[overall]
    st.markdown(f"## {icon} Overall Rating: **{overall}**")

# ── Refiner ─────────────────────────────────────────────────────────────────
if run_refiner:
    if "evaluations" not in st.session_state:
        st.warning("Run the Critic Agent first so the Refiner has feedback to act on.")
    elif not openai_api_key:
        st.warning("OpenAI API key not found. Please add OPENAI_API_KEY=... to your .env file.")
    else:
        import visualizer_agent
        import critic_agent

        df = datasets[dataset_name]()

        with st.spinner("Visualizer Agent is refining the visualizations..."):
            refined_summary, refined_visualizations, changes_made = visualizer_agent.refine(
                df,
                st.session_state["summary"],
                st.session_state["visualizations"],
                st.session_state["evaluations"],
                st.session_state["per_viz_evaluations"],
            )
            st.session_state["refined_summary"] = refined_summary
            st.session_state["refined_visualizations"] = refined_visualizations
            st.session_state["refiner_changes"] = changes_made

        with st.spinner("Critic Agent is re-evaluating the refined visualizations (GPT-5.4)..."):
            refined_overall, refined_per_viz = critic_agent.run(
                df,
                refined_summary,
                refined_visualizations,
                openai_api_key,
            )
            st.session_state["refined_evaluations"] = refined_overall
            st.session_state["refined_per_viz_evaluations"] = refined_per_viz

if "refined_visualizations" in st.session_state:
    st.header("Refiner Agent Results")

    # Show what was improved
    st.subheader("Improvements Made")
    for change in st.session_state["refiner_changes"]:
        st.markdown(f"- {change}")

    # Show refined visualizations
    st.subheader("Refined Visualizations")
    for title, img_b64, description in st.session_state["refined_visualizations"]:
        st.markdown(f"#### {title}")
        st.image(base64.b64decode(img_b64), use_container_width=True)
        st.info(description)

if "refined_evaluations" in st.session_state:
    st.header("Critic Re-Evaluation (After Refinement)")

    SCORE_COLORS_R = {"Excellent": "🟢", "Solid": "🟡", "Needs Work": "🔴"}

    from critic_agent import RUBRIC as RUBRIC_R

    # Per-visualization scores
    st.subheader("Per-Visualization Scores & Top Issues")
    for title, scores, top_issues in st.session_state["refined_per_viz_evaluations"]:
        st.markdown(f"#### {title}")
        score_cols = st.columns(len(RUBRIC_R))
        for col, dimension in zip(score_cols, RUBRIC_R):
            dim_score = scores[dimension]
            icon = SCORE_COLORS_R.get(dim_score, "⚪")
            col.metric(label=dimension, value=f"{icon} {dim_score}")
        if top_issues:
            st.markdown("**Top issues:**")
            for issue in top_issues:
                st.markdown(f"- {issue}")
        else:
            st.success("No issues found — this visualization scores well across all dimensions.")

    st.divider()

    # Aggregate scores
    st.subheader("Aggregate Rubric Scores")
    for dimension, (score, reasons) in st.session_state["refined_evaluations"].items():
        icon = SCORE_COLORS_R.get(score, "⚪")
        st.markdown(f"### {icon} {dimension} — **{score}**")
        for reason in reasons:
            st.markdown(f"- {reason}")

    r_scores = [s for s, _ in st.session_state["refined_evaluations"].values()]
    r_score_map = {"Excellent": 3, "Solid": 2, "Needs Work": 1}
    r_avg = sum(r_score_map[s] for s in r_scores) / len(r_scores)
    if r_avg >= 2.75:
        r_overall = "Excellent"
    elif r_avg >= 1.75:
        r_overall = "Solid"
    else:
        r_overall = "Needs Work"

    st.divider()
    icon = SCORE_COLORS_R[r_overall]
    st.markdown(f"## {icon} Overall Rating (Refined): **{r_overall}**")
