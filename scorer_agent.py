"""Scorer Agent — uses GPT-5.4 or Claude Opus to evaluate visualizations against a rubric."""

import json
from openai import OpenAI
import anthropic

RUBRIC = {
    "Clarity": {
        "Needs Work": "Hard to read/interpret",
        "Solid": "Readable with minor effort",
        "Excellent": "Immediately clear",
    },
    "Insight Depth": {
        "Needs Work": "Surface stats only",
        "Solid": "At least one non-obvious finding",
        "Excellent": "Multiple meaningful insights",
    },
    "Completeness": {
        "Needs Work": "Major gaps in coverage",
        "Solid": "Distributions and relationships covered",
        "Excellent": "Shape, outliers, and missingness are all addressed",
    },
    "Aesthetics": {
        "Needs Work": "Cluttered or default styling",
        "Solid": "Clean with proper labels",
        "Excellent": "Polished and consistent",
    },
}

SYSTEM_PROMPT = """\
You are an expert data visualization critic. You evaluate exploratory data analysis (EDA) \
visualizations against a structured rubric.

## Rubric

Each visualization must be scored on these four dimensions. The only valid scores are \
"Excellent", "Solid", or "Needs Work".

| Dimension | Needs Work | Solid | Excellent |
|---|---|---|---|
| Clarity | Hard to read/interpret | Readable with minor effort | Immediately clear |
| Insight Depth | Surface stats only | At least one non-obvious finding | Multiple meaningful insights |
| Completeness | Major gaps in coverage | Distributions and relationships covered | Shape, outliers, and missingness are all addressed |
| Aesthetics | Cluttered or default styling | Clean with proper labels | Polished and consistent |

## Your task

You will receive:
1. A dataset sample (first rows and column info)
2. A statistical summary produced by the Visualizer Agent
3. A list of visualizations, each with a title, description, and the actual chart image

For EACH visualization, you must:
- Score it on all 4 rubric dimensions (Excellent / Solid / Needs Work)
- Identify the top 3 issues (or fewer if the visualization is strong)

Then provide aggregate scores across all visualizations for each rubric dimension, with brief reasoning.

## Response format

You MUST respond with valid JSON matching this exact structure (no markdown fencing):
{
  "per_visualization": [
    {
      "title": "...",
      "scores": {"Clarity": "...", "Insight Depth": "...", "Completeness": "...", "Aesthetics": "..."},
      "top_issues": ["issue 1", "issue 2", "issue 3"]
    }
  ],
  "aggregate": {
    "Clarity": {"score": "...", "reasons": ["reason 1", "..."]},
    "Insight Depth": {"score": "...", "reasons": ["..."]},
    "Completeness": {"score": "...", "reasons": ["..."]},
    "Aesthetics": {"score": "...", "reasons": ["..."]}
  }
}
"""


def _build_dataset_context(df):
    """Build a text description of the dataset for the prompt."""
    parts = []
    parts.append(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
    parts.append(f"Columns: {', '.join(df.columns.tolist())}")
    parts.append(f"Dtypes:\n{df.dtypes.to_string()}")
    parts.append(f"\nFirst 5 rows:\n{df.head().to_string()}")

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        parts.append(f"\nMissing values:\n{missing.to_string()}")
    else:
        parts.append("\nNo missing values.")

    return "\n".join(parts)


def _build_summary_context(summary):
    """Convert the Visualizer summary dict to a readable string."""
    parts = []
    parts.append(f"Shape: {summary.get('shape', 'N/A')}")

    mv = summary.get("missing_values", {})
    parts.append(f"Missing values: {mv}")

    if "numeric_summary" in summary:
        parts.append(f"Descriptive statistics:\n{summary['numeric_summary'].to_string()}")

    if "correlations" in summary:
        parts.append(f"Correlation matrix:\n{summary['correlations'].to_string()}")

    if "group_means" in summary:
        parts.append(f"Group means by {summary.get('group_col', 'group')}:\n{summary['group_means'].to_string()}")

    if "categorical_counts" in summary:
        for col, counts in summary["categorical_counts"].items():
            parts.append(f"{col} value counts: {counts}")

    return "\n".join(parts)


def _build_messages(df, summary, visualizations):
    """Build the ChatGPT messages array with text and images."""
    dataset_context = _build_dataset_context(df)
    summary_context = _build_summary_context(summary)

    user_content = []

    user_content.append({
        "type": "text",
        "text": (
            f"## Dataset\n{dataset_context}\n\n"
            f"## Statistical Summary from Visualizer Agent\n{summary_context}\n\n"
            f"## Visualizations\n"
            f"There are {len(visualizations)} visualizations to evaluate. "
            f"Each is shown below with its title and the Visualizer Agent's description.\n"
        ),
    })

    for i, (title, img_b64, description) in enumerate(visualizations, 1):
        user_content.append({
            "type": "text",
            "text": f"\n### Visualization {i}: {title}\nDescription: {description}\n",
        })
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_b64}", "detail": "high"},
        })

    user_content.append({
        "type": "text",
        "text": (
            "\nNow evaluate each visualization against the rubric. "
            "Score each on all 4 dimensions, list the top 3 issues for each, "
            "and provide aggregate scores. Respond with JSON only."
        ),
    })

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _parse_response(response_text):
    """Parse the ChatGPT JSON response into the expected return format."""
    text = response_text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    data = json.loads(text)

    per_viz_evaluations = []
    for item in data["per_visualization"]:
        scores = {}
        for dim in RUBRIC:
            raw = item["scores"].get(dim, "Solid")
            scores[dim] = raw if raw in ("Excellent", "Solid", "Needs Work") else "Solid"
        issues = item.get("top_issues", [])[:3]
        per_viz_evaluations.append((item["title"], scores, issues))

    overall_evaluations = {}
    for dim in RUBRIC:
        agg = data["aggregate"].get(dim, {})
        raw_score = agg.get("score", "Solid")
        score = raw_score if raw_score in ("Excellent", "Solid", "Needs Work") else "Solid"
        reasons = agg.get("reasons", [])
        overall_evaluations[dim] = (score, reasons)

    return overall_evaluations, per_viz_evaluations


def _run_openai(df, summary, visualizations, api_key):
    """Score using OpenAI GPT-5.4."""
    client = OpenAI(api_key=api_key)
    messages = _build_messages(df, summary, visualizations)

    response = client.chat.completions.create(
        model="gpt-5.4",
        messages=messages,
        max_completion_tokens=4096,
        temperature=0.3,
    )

    return _parse_response(response.choices[0].message.content)


def _build_claude_messages(df, summary, visualizations):
    """Build Claude-compatible messages with text and images."""
    dataset_context = _build_dataset_context(df)
    summary_context = _build_summary_context(summary)

    content = []

    content.append({
        "type": "text",
        "text": (
            f"## Dataset\n{dataset_context}\n\n"
            f"## Statistical Summary from Visualizer Agent\n{summary_context}\n\n"
            f"## Visualizations\n"
            f"There are {len(visualizations)} visualizations to evaluate. "
            f"Each is shown below with its title and the Visualizer Agent's description.\n"
        ),
    })

    for i, (title, img_b64, description) in enumerate(visualizations, 1):
        content.append({
            "type": "text",
            "text": f"\n### Visualization {i}: {title}\nDescription: {description}\n",
        })
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": img_b64},
        })

    content.append({
        "type": "text",
        "text": (
            "\nNow evaluate each visualization against the rubric. "
            "Score each on all 4 dimensions, list the top 3 issues for each, "
            "and provide aggregate scores. Respond with JSON only."
        ),
    })

    return content


def _run_claude(df, summary, visualizations, api_key):
    """Score using Claude Sonnet 4."""
    import time
    client = anthropic.Anthropic(api_key=api_key)
    content = _build_claude_messages(df, summary, visualizations)

    for attempt in range(3):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                temperature=0.3,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": content}],
            )
            return _parse_response(response.content[0].text)
        except anthropic.InternalServerError:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                raise


def run(df, summary, visualizations, api_key, model="openai"):
    """Score the Visualizer output.

    Args:
        model: "openai" for GPT-5.4, "claude" for Claude Opus

    Returns:
        overall_evaluations: dict of {dimension: (score, reasons)}
        per_viz_evaluations: list of (title, scores_dict, top_3_issues)
    """
    if model == "claude":
        return _run_claude(df, summary, visualizations, api_key)
    else:
        return _run_openai(df, summary, visualizations, api_key)
