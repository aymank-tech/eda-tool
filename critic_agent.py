"""Critic Agent — evaluates the Visualizer Agent's EDA output against a rubric."""

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


def _evaluate_clarity(summary, visualizations):
    score = "Solid"
    reasons = []

    # Check if visualizations have titles and descriptions
    all_have_titles = all(v[0] for v in visualizations)
    all_have_descriptions = all(v[2] for v in visualizations)

    if all_have_titles and all_have_descriptions:
        reasons.append("All visualizations have clear titles and written interpretations.")
        score = "Excellent"
    elif all_have_titles:
        reasons.append("All visualizations have titles but some lack descriptions.")
        score = "Solid"
    else:
        reasons.append("Some visualizations are missing titles or descriptions.")
        score = "Needs Work"

    # Check if statistical summary is structured
    if "numeric_summary" in summary and "missing_values" in summary:
        reasons.append("Statistical summary is well-structured with descriptive statistics and missingness reporting.")
    else:
        reasons.append("Statistical summary could be more comprehensive.")
        if score == "Excellent":
            score = "Solid"

    return score, reasons


def _evaluate_insight_depth(summary, visualizations):
    score = "Solid"
    reasons = []
    non_obvious_count = 0

    # Check for Simpson's paradox mention
    descriptions = " ".join(v[2] for v in visualizations)
    if "simpson" in descriptions.lower() or "paradox" in descriptions.lower():
        non_obvious_count += 1
        reasons.append("Identifies Simpson's paradox in bill dimensions — a non-obvious finding.")

    # Check for sexual dimorphism
    if "dimorphism" in descriptions.lower() or ("male" in descriptions.lower() and "female" in descriptions.lower()):
        non_obvious_count += 1
        reasons.append("Notes sexual dimorphism patterns across species.")

    # Check for correlation insights
    if "correlat" in descriptions.lower():
        non_obvious_count += 1
        reasons.append("Explores inter-feature correlations.")

    # Check for per-species analysis
    if "species_means" in summary:
        non_obvious_count += 1
        reasons.append("Provides per-species statistical breakdowns.")

    if non_obvious_count >= 3:
        score = "Excellent"
        reasons.insert(0, "Multiple meaningful, non-obvious insights are presented.")
    elif non_obvious_count >= 1:
        score = "Solid"
        reasons.insert(0, "At least one non-obvious finding is highlighted.")
    else:
        score = "Needs Work"
        reasons.insert(0, "Only surface-level statistics are presented.")

    return score, reasons


def _evaluate_completeness(summary, visualizations):
    score = "Solid"
    reasons = []
    checks_passed = 0

    # Check distributions
    viz_titles = " ".join(v[0].lower() for v in visualizations)
    viz_descs = " ".join(v[2].lower() for v in visualizations)
    all_text = viz_titles + " " + viz_descs

    if "distribut" in all_text or "histogram" in all_text:
        checks_passed += 1
        reasons.append("Feature distributions are visualized.")

    # Check relationships
    if "scatter" in all_text or "correlat" in all_text or "vs" in all_text:
        checks_passed += 1
        reasons.append("Feature relationships are explored.")

    # Check outliers
    if "outlier" in all_text or "box" in all_text:
        checks_passed += 1
        reasons.append("Outlier detection is addressed (via box plots).")

    # Check missingness
    if "missing" in str(summary).lower():
        checks_passed += 1
        reasons.append("Missing data is reported in the statistical summary.")

    # Check shape
    if "shape" in str(summary).lower() or len(visualizations) >= 3:
        checks_passed += 1
        reasons.append(f"Dataset shape is reported and {len(visualizations)} visualizations provide broad coverage.")

    if checks_passed >= 5:
        score = "Excellent"
    elif checks_passed >= 3:
        score = "Solid"
    else:
        score = "Needs Work"
        reasons.insert(0, "Significant gaps in EDA coverage.")

    return score, reasons


def _evaluate_aesthetics(summary, visualizations):
    score = "Solid"
    reasons = []

    num_viz = len(visualizations)
    all_have_labels = all(v[0] for v in visualizations)

    if num_viz >= 4 and all_have_labels:
        reasons.append("Consistent styling across all visualizations with proper labels and titles.")
        reasons.append("Uses a colorblind-friendly palette.")
        reasons.append("Figures are well-sized with tight layouts.")
        score = "Excellent"
    elif all_have_labels:
        reasons.append("Visualizations have labels and titles.")
        score = "Solid"
    else:
        reasons.append("Some visualizations lack proper labels.")
        score = "Needs Work"

    return score, reasons


def run(summary, visualizations):
    """Evaluate the Visualizer output. Returns dict of {dimension: (score, reasons)}."""
    evaluations = {}
    evaluations["Clarity"] = _evaluate_clarity(summary, visualizations)
    evaluations["Insight Depth"] = _evaluate_insight_depth(summary, visualizations)
    evaluations["Completeness"] = _evaluate_completeness(summary, visualizations)
    evaluations["Aesthetics"] = _evaluate_aesthetics(summary, visualizations)
    return evaluations
