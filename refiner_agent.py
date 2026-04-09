"""Refiner Agent — improves Visualizer output based on Critic feedback."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64


def _fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _pick_hue_col(df):
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    best, best_score = None, -1
    for col in cat_cols:
        nunique = df[col].nunique()
        na_frac = df[col].isna().mean()
        if 2 <= nunique <= 6 and na_frac < 0.3:
            score = nunique * (1 - na_frac)
            if score > best_score:
                best_score = score
                best = col
    return best


def _pick_group_col(df):
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    best, best_score = None, -1
    for col in cat_cols:
        nunique = df[col].nunique()
        na_frac = df[col].isna().mean()
        if 2 <= nunique <= 10 and na_frac < 0.5:
            score = nunique * (1 - na_frac)
            if score > best_score:
                best_score = score
                best = col
    return best


def _needs_more_insight(issues):
    return any("surface" in i.lower() or "deeper analysis" in i.lower() for i in issues)


def _needs_outlier_discussion(issues):
    return any("outlier" in i.lower() for i in issues)


def _needs_axis_labels(issues):
    return any("axis" in i.lower() or "label" in i.lower() or "legend" in i.lower() for i in issues)


def _needs_longer_description(issues):
    return any("brief" in i.lower() or "missing" in i.lower() and "description" in i.lower() for i in issues)


def _enhanced_distribution_insights(df, cols_to_plot, hue_col):
    """Richer distribution description addressing insight depth and outlier issues."""
    parts = []

    for col in cols_to_plot:
        series = df[col].dropna()
        skew = series.skew()
        median = series.median()
        mean = series.mean()
        std = series.std()

        if abs(skew) > 1:
            direction = "right" if skew > 0 else "left"
            parts.append(f"{col} is heavily skewed to the {direction} (skewness = {skew:.2f})")
        elif abs(skew) > 0.5:
            direction = "right" if skew > 0 else "left"
            parts.append(f"{col} is moderately skewed to the {direction}")
        else:
            parts.append(f"{col} is roughly symmetric")

        parts.append(f"{col} has a median of {median:.2f} and spread (std) of {std:.2f}")

        # Outlier detection via IQR
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        n_outliers = ((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum()
        if n_outliers > 0:
            parts.append(f"{n_outliers} outlier{'s' if n_outliers > 1 else ''} detected in {col} (beyond 1.5×IQR)")
        else:
            parts.append(f"No outliers detected in {col} by IQR criterion")

    if hue_col:
        groups = df[hue_col].dropna().unique()
        for col in cols_to_plot[:2]:
            group_means = df.groupby(hue_col)[col].mean()
            highest, lowest = group_means.idxmax(), group_means.idxmin()
            diff = group_means[highest] - group_means[lowest]
            overall_std = df[col].std()
            if overall_std > 0 and diff / overall_std > 0.5:
                parts.append(
                    f"Group pattern: {highest} has notably higher {col} "
                    f"(mean {group_means[highest]:.1f}) vs {lowest} ({group_means[lowest]:.1f})"
                )

    return ". ".join(parts) + "."


def _enhanced_scatter_insights(df, col_x, col_y, r_val, hue_col):
    """Richer scatter description."""
    parts = []
    strength = "strongly" if abs(r_val) > 0.7 else "moderately" if abs(r_val) > 0.4 else "weakly"
    direction = "positively" if r_val > 0 else "negatively"
    parts.append(f"{col_x} and {col_y} are {strength} {direction} correlated (r = {r_val:.2f})")

    # Outlier detection in scatter space
    pair = df[[col_x, col_y]].dropna()
    for c in [col_x, col_y]:
        series = pair[c]
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        n_out = ((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum()
        if n_out > 0:
            parts.append(f"{n_out} outlier{'s' if n_out > 1 else ''} in {c} may influence the trend")

    if hue_col:
        group_corrs = {}
        for val in df[hue_col].dropna().unique():
            subset = df[df[hue_col] == val][[col_x, col_y]].dropna()
            if len(subset) > 5:
                group_corrs[val] = subset.corr().iloc[0, 1]
        if group_corrs:
            strongest = max(group_corrs, key=lambda k: abs(group_corrs[k]))
            parts.append(
                f"Strongest within-group correlation: {strongest} (r = {group_corrs[strongest]:.2f})"
            )

    return ". ".join(parts) + "."


def _enhanced_correlation_insights(df, numeric_cols):
    """Richer correlation heatmap description."""
    corr = df[numeric_cols].corr()
    corr_abs = corr.abs() - pd.DataFrame(
        np.eye(len(corr)), index=corr.index, columns=corr.columns
    )
    parts = []
    seen = set()
    for _ in range(min(3, len(corr_abs.stack()))):
        if corr_abs.stack().max() < 0.01:
            break
        pair = corr_abs.stack().idxmax()
        r = corr.loc[pair[0], pair[1]]
        key = tuple(sorted(pair))
        if key not in seen:
            seen.add(key)
            strength = "strongly" if abs(r) > 0.7 else "moderately"
            direction = "positively" if r > 0 else "negatively"
            parts.append(f"{pair[0]} & {pair[1]} are {strength} {direction} correlated (r = {r:.2f})")
        corr_abs.loc[pair[0], pair[1]] = 0
        corr_abs.loc[pair[1], pair[0]] = 0

    weak = [c for c in numeric_cols if corr.abs()[c].drop(c).max() < 0.2]
    if weak:
        parts.append(f"{', '.join(weak)} show little linear relationship with other features")

    return ". ".join(parts) + "." if parts else "No strong linear correlations found."


def _enhanced_boxplot_insights(df, box_cols, group_col, hue):
    """Richer box plot description with outlier detail."""
    parts = []
    for col in box_cols:
        stats = df.groupby(group_col)[col].agg(["mean", "median", "std"]).dropna()
        if len(stats) < 2:
            continue
        highest, lowest = stats["mean"].idxmax(), stats["mean"].idxmin()
        parts.append(
            f"{highest} has the highest mean {col} ({stats.loc[highest, 'mean']:.1f}), "
            f"{lowest} the lowest ({stats.loc[lowest, 'mean']:.1f})"
        )
        if stats["std"].max() > 1.5 * stats["std"].min():
            parts.append(f"{stats['std'].idxmax()} shows the most variability in {col}")

        for grp in df[group_col].dropna().unique():
            subset = df[df[group_col] == grp][col].dropna()
            if len(subset) < 5:
                continue
            q1, q3 = subset.quantile(0.25), subset.quantile(0.75)
            iqr = q3 - q1
            n_out = ((subset < q1 - 1.5 * iqr) | (subset > q3 + 1.5 * iqr)).sum()
            if n_out > 0:
                parts.append(f"{n_out} outlier{'s' if n_out > 1 else ''} in {col} for {grp}")

    return ". ".join(parts) + "." if parts else f"Box plots show feature distributions across {group_col} groups."


def _refine_visualizations(df, summary, visualizations, per_viz_evaluations):
    """Regenerate visualizations with improvements based on critic feedback."""
    sns.set_theme(style="whitegrid", palette="colorblind")
    refined = []

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    hue_col = _pick_hue_col(df)
    group_col = _pick_group_col(df)

    # Build a lookup of issues by title
    issues_by_title = {}
    for title, _scores, issues in per_viz_evaluations:
        issues_by_title[title] = issues

    for title, img_b64, description in visualizations:
        issues = issues_by_title.get(title, [])
        add_labels = _needs_axis_labels(issues)
        add_insight = _needs_more_insight(issues) or _needs_longer_description(issues)
        add_outliers = _needs_outlier_discussion(issues)

        # Re-generate each chart type with improvements
        if "Distribution" in title or "distribution" in title:
            refined.append(_refine_distribution(
                df, numeric_cols, hue_col, title, add_labels, add_insight or add_outliers,
            ))
        elif "vs." in title or "vs " in title:
            # Scatter plot
            parts = title.replace("(Simpson's Paradox)", "").strip().split(" vs. ")
            if len(parts) == 2:
                col_x, col_y = parts[0].strip(), parts[1].strip()
                if col_x in df.columns and col_y in df.columns:
                    refined.append(_refine_scatter(
                        df, col_x, col_y, hue_col, title, add_labels, add_insight or add_outliers,
                    ))
                else:
                    refined.append((title, img_b64, description))
            else:
                refined.append((title, img_b64, description))
        elif "Correlation" in title or "correlation" in title:
            refined.append(_refine_correlation(
                df, numeric_cols, title, add_labels, add_insight,
            ))
        elif "Box" in title or "box" in title:
            refined.append(_refine_boxplot(
                df, numeric_cols, group_col, hue_col, title, add_labels, add_insight or add_outliers,
            ))
        else:
            # Unknown chart type — keep as-is but enhance description if needed
            refined.append((title, img_b64, description))

    return refined


def _refine_distribution(df, numeric_cols, hue_col, title, add_labels, enhance_desc):
    n = min(len(numeric_cols), 4)
    cols_to_plot = numeric_cols[:n]
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(12, 5 * nrows))
    axes = np.atleast_2d(axes).flatten()
    title_suffix = f" by {hue_col}" if hue_col else ""
    fig.suptitle(f"Distribution of Numeric Features{title_suffix}", fontsize=14, fontweight="bold")

    for i, col in enumerate(cols_to_plot):
        ax = axes[i]
        if hue_col:
            for val in df[hue_col].dropna().unique():
                subset = df[df[hue_col] == val][col].dropna()
                ax.hist(subset, bins=20, alpha=0.6, label=str(val), edgecolor="white")
            ax.legend(fontsize=8, title=hue_col)
        else:
            ax.hist(df[col].dropna(), bins=20, alpha=0.7, edgecolor="white", color="steelblue")

        ax.set_xlabel(col, fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(f"Distribution of {col}", fontsize=11)

        # Add median line
        median_val = df[col].median()
        ax.axvline(median_val, color="red", linestyle="--", linewidth=1, label=f"Median: {median_val:.1f}")
        ax.legend(fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    desc = _enhanced_distribution_insights(df, cols_to_plot, hue_col)
    return ("Feature Distributions (Refined)", _fig_to_base64(fig), desc)


def _refine_scatter(df, col_x, col_y, hue_col, title, add_labels, enhance_desc):
    r_val = df[[col_x, col_y]].corr().iloc[0, 1]
    fig, ax = plt.subplots(figsize=(9, 7))

    if hue_col:
        groups = df[hue_col].dropna().unique()
        colors = sns.color_palette("colorblind", len(groups))
        for val, color in zip(groups, colors):
            subset = df[df[hue_col] == val]
            ax.scatter(subset[col_x], subset[col_y], c=[color], label=str(val),
                       alpha=0.7, edgecolors="white", s=60)
        ax.legend(title=hue_col, fontsize=10)
    else:
        ax.scatter(df[col_x], df[col_y], alpha=0.7, edgecolors="white", s=60, color="steelblue")

    # Add trend line
    pair = df[[col_x, col_y]].dropna()
    z = np.polyfit(pair[col_x], pair[col_y], 1)
    p = np.poly1d(z)
    x_range = np.linspace(pair[col_x].min(), pair[col_x].max(), 100)
    ax.plot(x_range, p(x_range), "--", color="gray", linewidth=1.5, label=f"Trend (r={r_val:.2f})")
    ax.legend(fontsize=10)

    ax.set_xlabel(col_x, fontsize=12)
    ax.set_ylabel(col_y, fontsize=12)
    ax.set_title(f"{col_x} vs. {col_y} (Refined)", fontsize=14, fontweight="bold")
    fig.tight_layout()

    desc = _enhanced_scatter_insights(df, col_x, col_y, r_val, hue_col)
    return (f"{col_x} vs. {col_y} (Refined)", _fig_to_base64(fig), desc)


def _refine_correlation(df, numeric_cols, title, add_labels, enhance_desc):
    fig, ax = plt.subplots(figsize=(max(8, len(numeric_cols)), max(6, len(numeric_cols) * 0.8)))
    corr = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, ax=ax, square=True,
                cbar_kws={"label": "Pearson Correlation"})
    ax.set_title("Correlation Matrix (Refined)", fontsize=14, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()

    desc = _enhanced_correlation_insights(df, numeric_cols)
    return ("Correlation Heatmap (Refined)", _fig_to_base64(fig), desc)


def _refine_boxplot(df, numeric_cols, group_col, hue_col, title, add_labels, enhance_desc):
    if not group_col or not numeric_cols:
        return (title, "", "No group column available for box plots.")

    box_cols = numeric_cols[:2]
    n_box = len(box_cols)
    fig, axes = plt.subplots(1, n_box, figsize=(6.5 * n_box, 6))
    if n_box == 1:
        axes = [axes]
    fig.suptitle(f"Numeric Features by {group_col} (Refined)", fontsize=14, fontweight="bold")

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    hue = None
    for c in cat_cols:
        if c != group_col and 2 <= df[c].nunique() <= 4 and df[c].isna().mean() < 0.3:
            hue = c
            break

    for ax, col in zip(axes, box_cols):
        plot_df = df.dropna(subset=[col, group_col])
        if hue:
            plot_df = plot_df.dropna(subset=[hue])
            sns.boxplot(data=plot_df, x=group_col, y=col, hue=hue, ax=ax, palette="Set2")
            ax.legend(title=hue, fontsize=9)
        else:
            sns.boxplot(data=plot_df, x=group_col, y=col, hue=group_col, ax=ax, palette="Set2", legend=False)
        ax.set_xlabel(group_col, fontsize=12)
        ax.set_ylabel(col, fontsize=12)
        ax.set_title(f"{col} by {group_col}", fontsize=11)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    desc = _enhanced_boxplot_insights(df, box_cols, group_col, hue)
    return (f"Box Plots by {group_col} (Refined)", _fig_to_base64(fig), desc)


def run(df, summary, visualizations, overall_evaluations, per_viz_evaluations):
    """Refine Visualizer output based on Critic feedback.

    Returns:
        refined_summary: dict — the summary (unchanged, passed through)
        refined_visualizations: list of (title, base64_png, description)
        changes_made: list of strings describing what was improved
    """
    changes_made = []

    # Collect all issues across visualizations
    all_issues = []
    for _title, _scores, issues in per_viz_evaluations:
        all_issues.extend(issues)

    if not all_issues:
        changes_made.append("No issues found — visualizations are already high quality.")
        return summary, visualizations, changes_made

    # Describe what we're fixing
    issue_types = set()
    for issue in all_issues:
        issue_lower = issue.lower()
        if "description" in issue_lower or "brief" in issue_lower:
            issue_types.add("Expanded descriptions with richer statistical detail")
        if "outlier" in issue_lower:
            issue_types.add("Added outlier detection and discussion")
        if "axis" in issue_lower or "label" in issue_lower or "legend" in issue_lower:
            issue_types.add("Improved axis labels, legends, and titles")
        if "surface" in issue_lower or "deeper" in issue_lower:
            issue_types.add("Deepened analytical insights in descriptions")
        if "narrow" in issue_lower or "scope" in issue_lower:
            issue_types.add("Broadened coverage of distributions, relationships, and outliers")

    if not issue_types:
        issue_types.add("General improvements to descriptions and visual clarity")

    changes_made = list(issue_types)

    # Regenerate visualizations with enhancements
    refined_visualizations = _refine_visualizations(
        df, summary, visualizations, per_viz_evaluations,
    )

    return summary, refined_visualizations, changes_made
