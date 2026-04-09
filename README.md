# EDA Tool — Palmer Penguins

A browser-based exploratory data analysis tool for the [Palmer Penguins](https://allisonhorst.github.io/palmerpenguins/) dataset, built with Streamlit.

Two agents work together:

- **Visualizer Agent** — generates statistical summaries and four visualizations (feature distributions, bill length vs depth scatter plot, correlation heatmap, box plots by species & sex).
- **Critic Agent** — evaluates the Visualizer's output on Clarity, Insight Depth, Completeness, and Aesthetics.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

The app opens at [http://localhost:8501](http://localhost:8501).

## Usage

1. Click **Run Visualizer Agent** to perform the EDA.
2. Review the statistical summaries and visualizations.
3. Click **Run Critic Agent** to see how the analysis scores against the rubric.
