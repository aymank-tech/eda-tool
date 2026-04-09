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
    "then run the **Critic Agent** to evaluate the results."
)

# ── Dataset selector ─────────────────────────────────────────────────────────
datasets = get_available_datasets()
dataset_name = st.selectbox("Select a dataset", list(datasets.keys()))

# Reset results when dataset changes
if st.session_state.get("current_dataset") != dataset_name:
    st.session_state["current_dataset"] = dataset_name
    st.session_state.pop("summary", None)
    st.session_state.pop("visualizations", None)
    st.session_state.pop("evaluations", None)

# ── Buttons ──────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
run_visualizer = col1.button("Run Visualizer Agent", type="primary", use_container_width=True)
run_critic = col2.button("Run Critic Agent", use_container_width=True)

# ── Visualizer ───────────────────────────────────────────────────────────────
if run_visualizer:
    with st.spinner("Visualizer Agent is analyzing the dataset..."):
        import visualizer_agent

        df = datasets[dataset_name]()
        summary, visualizations = visualizer_agent.run(df)
        st.session_state["summary"] = summary
        st.session_state["visualizations"] = visualizations
        st.session_state.pop("evaluations", None)

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
    else:
        with st.spinner("Critic Agent is evaluating the results..."):
            import critic_agent

            evaluations = critic_agent.run(
                st.session_state["summary"],
                st.session_state["visualizations"],
            )
            st.session_state["evaluations"] = evaluations

if "evaluations" in st.session_state:
    st.header("Critic Agent Evaluation")

    SCORE_COLORS = {
        "Excellent": "🟢",
        "Solid": "🟡",
        "Needs Work": "🔴",
    }

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
