"""EDA Tool — Visualizer & Critic Agents for the Palmer Penguins dataset."""

import streamlit as st
import pandas as pd
import base64

st.set_page_config(page_title="Penguin EDA Tool", layout="wide")

st.title("🐧 Palmer Penguins — Exploratory Data Analysis")
st.markdown(
    "Use the **Visualizer Agent** to perform EDA, then the **Critic Agent** to evaluate the results."
)

# ── Buttons ──────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
run_visualizer = col1.button("▶ Run Visualizer Agent", type="primary", use_container_width=True)
run_critic = col2.button("▶ Run Critic Agent", use_container_width=True)

# ── Visualizer ───────────────────────────────────────────────────────────────
if run_visualizer:
    with st.spinner("Visualizer Agent is analyzing the dataset…"):
        import visualizer_agent

        summary, visualizations = visualizer_agent.run()
        st.session_state["summary"] = summary
        st.session_state["visualizations"] = visualizations

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

        st.markdown("**Species counts:**")
        st.dataframe(
            pd.DataFrame.from_dict(summary["species_counts"], orient="index", columns=["Count"]),
            use_container_width=True,
        )

    with col_b:
        st.markdown("**Island counts:**")
        st.dataframe(
            pd.DataFrame.from_dict(summary["island_counts"], orient="index", columns=["Count"]),
            use_container_width=True,
        )
        st.markdown("**Per-species means:**")
        st.dataframe(summary["species_means"], use_container_width=True)

    st.markdown("**Descriptive statistics:**")
    st.dataframe(summary["numeric_summary"], use_container_width=True)

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
        with st.spinner("Critic Agent is evaluating the results…"):
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

    # Overall
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
