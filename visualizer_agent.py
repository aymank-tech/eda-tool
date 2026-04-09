"""Visualizer Agent — performs generic EDA on any tabular dataset."""

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


def generate_statistical_summary(df):
    """Generate a comprehensive statistical summary for any dataframe."""
    summary = {}

    summary["shape"] = f"{df.shape[0]} rows × {df.shape[1]} columns"

    # Missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        summary["missing_values"] = missing.to_dict()
    else:
        summary["missing_values"] = "No missing values"

    # Categorical column value counts
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_summaries = {}
    for col in cat_cols:
        nunique = df[col].nunique()
        if 2 <= nunique <= 20:
            cat_summaries[col] = df[col].value_counts().to_dict()
    summary["categorical_counts"] = cat_summaries

    # Numeric summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        desc = df[numeric_cols].describe().round(2)
        summary["numeric_summary"] = desc

        corr = df[numeric_cols].corr().round(3)
        summary["correlations"] = corr

    # Group-by means for the best categorical grouping column
    group_col = _pick_group_col(df)
    if group_col and numeric_cols:
        group_means = df.groupby(group_col)[numeric_cols].mean().round(2)
        summary["group_col"] = group_col
        summary["group_means"] = group_means

    return summary


def _pick_group_col(df):
    """Pick the best categorical column for grouping (low cardinality, few NAs)."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    best = None
    best_score = -1
    for col in cat_cols:
        nunique = df[col].nunique()
        na_frac = df[col].isna().mean()
        if 2 <= nunique <= 10 and na_frac < 0.5:
            score = nunique * (1 - na_frac)
            if score > best_score:
                best_score = score
                best = col
    return best


def _pick_hue_col(df):
    """Pick the best categorical column for coloring plots (2-6 unique values)."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    best = None
    best_score = -1
    for col in cat_cols:
        nunique = df[col].nunique()
        na_frac = df[col].isna().mean()
        if 2 <= nunique <= 6 and na_frac < 0.3:
            score = nunique * (1 - na_frac)
            if score > best_score:
                best_score = score
                best = col
    return best


# ── Insight generators ───────────────────────────────────────────────────────

def _distribution_insights(df, cols_to_plot, hue_col):
    """Analyze distributions and return data-driven insight text."""
    parts = []

    # Skewness observations
    for col in cols_to_plot:
        series = df[col].dropna()
        skew = series.skew()
        if abs(skew) > 1:
            direction = "right" if skew > 0 else "left"
            parts.append(f"{col} is heavily skewed to the {direction} (skewness = {skew:.2f})")
        elif abs(skew) > 0.5:
            direction = "right" if skew > 0 else "left"
            parts.append(f"{col} is moderately skewed to the {direction}")

    if hue_col:
        groups = df[hue_col].dropna().unique()
        for col in cols_to_plot:
            group_means = df.groupby(hue_col)[col].mean()
            highest = group_means.idxmax()
            lowest = group_means.idxmin()
            if group_means[lowest] != 0:
                ratio = group_means[highest] / group_means[lowest]
            else:
                ratio = float("inf")

            diff = group_means[highest] - group_means[lowest]
            overall_std = df[col].std()
            # Report if difference is meaningful relative to overall spread
            if overall_std > 0 and diff / overall_std > 0.5:
                parts.append(
                    f"{highest} has notably higher {col} (mean {group_means[highest]:.1f}) "
                    f"compared to {lowest} ({group_means[lowest]:.1f})"
                )

        # Check for overlapping vs separated distributions
        separated = []
        overlapping = []
        for col in cols_to_plot[:2]:
            group_ranges = {}
            for val in groups:
                subset = df[df[hue_col] == val][col].dropna()
                if len(subset) > 0:
                    group_ranges[val] = (subset.quantile(0.25), subset.quantile(0.75))
            # Check if any pair of groups has non-overlapping IQRs
            group_list = list(group_ranges.keys())
            any_separated = False
            for i in range(len(group_list)):
                for j in range(i + 1, len(group_list)):
                    r1 = group_ranges[group_list[i]]
                    r2 = group_ranges[group_list[j]]
                    if r1[1] < r2[0] or r2[1] < r1[0]:
                        any_separated = True
            if any_separated:
                separated.append(col)
            else:
                overlapping.append(col)

        if separated:
            parts.append(f"Groups are clearly separated in {', '.join(separated)}")
        if overlapping and separated:
            parts.append(f"but overlap considerably in {', '.join(overlapping)}")

    if not parts:
        parts.append("Distributions appear roughly symmetric with no extreme skewness detected")

    return ". ".join(parts) + "."


def _scatter_insights(df, col_x, col_y, r_val, hue_col):
    """Analyze scatter plot relationship and return data-driven insight text."""
    parts = []

    strength = "strongly" if abs(r_val) > 0.7 else "moderately" if abs(r_val) > 0.4 else "weakly"
    direction = "positively" if r_val > 0 else "negatively"
    parts.append(
        f"{col_x} and {col_y} are {strength} {direction} correlated overall (r = {r_val:.2f})"
    )

    # Check for Simpson's paradox: overall correlation sign differs from within-group
    if hue_col:
        group_corrs = {}
        for val in df[hue_col].dropna().unique():
            subset = df[df[hue_col] == val][[col_x, col_y]].dropna()
            if len(subset) > 5:
                group_corrs[val] = subset.corr().iloc[0, 1]

        if group_corrs:
            signs_within = [np.sign(r) for r in group_corrs.values() if not np.isnan(r)]
            if signs_within and all(s == signs_within[0] for s in signs_within):
                if signs_within[0] != np.sign(r_val) and abs(r_val) > 0.1:
                    parts.append(
                        f"Simpson's paradox detected: the overall correlation is {direction}, "
                        f"but within each {hue_col} group the correlation reverses sign"
                    )
                else:
                    # Report per-group patterns
                    strongest_group = max(group_corrs, key=lambda k: abs(group_corrs[k]))
                    parts.append(
                        f"The relationship is most pronounced in the {strongest_group} group "
                        f"(r = {group_corrs[strongest_group]:.2f})"
                    )

            # Report spread differences
            group_means_x = df.groupby(hue_col)[col_x].mean()
            group_means_y = df.groupby(hue_col)[col_y].mean()
            if group_means_x.std() > 0.1 * group_means_x.mean():
                high_x = group_means_x.idxmax()
                low_x = group_means_x.idxmin()
                parts.append(
                    f"{high_x} tends toward higher {col_x} values while {low_x} tends lower"
                )

    return ". ".join(parts) + "."


def _correlation_insights(df, numeric_cols):
    """Analyze the correlation matrix and return data-driven insight text."""
    corr = df[numeric_cols].corr()
    corr_abs = corr.abs()
    np.fill_diagonal(corr_abs.values, 0)

    parts = []

    # Top 3 strongest correlations
    pairs = []
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
            pairs.append(f"{pair[0]} & {pair[1]} are {strength} {direction} correlated (r = {r:.2f})")
        corr_abs.loc[pair[0], pair[1]] = 0
        corr_abs.loc[pair[1], pair[0]] = 0

    if pairs:
        parts.extend(pairs)

    # Check for features with weak correlations to everything
    max_corr_per_col = corr.abs()
    np.fill_diagonal(max_corr_per_col.values, 0)
    weak_cols = [col for col in numeric_cols if max_corr_per_col[col].max() < 0.2]
    if weak_cols:
        parts.append(f"{', '.join(weak_cols)} show{'s' if len(weak_cols) == 1 else ''} little linear relationship with other features")

    return ". ".join(parts) + "." if parts else "No strong linear correlations found among numeric features."


def _boxplot_insights(df, box_cols, group_col, hue):
    """Analyze box plot data and return data-driven insight text."""
    parts = []

    for col in box_cols:
        group_stats = df.groupby(group_col)[col].agg(["mean", "median", "std"]).dropna()
        if len(group_stats) < 2:
            continue

        # Which group has highest/lowest
        highest = group_stats["mean"].idxmax()
        lowest = group_stats["mean"].idxmin()
        parts.append(
            f"{highest} has the highest mean {col} ({group_stats.loc[highest, 'mean']:.1f}) "
            f"and {lowest} has the lowest ({group_stats.loc[lowest, 'mean']:.1f})"
        )

        # Check for large spread differences
        if group_stats["std"].max() > 1.5 * group_stats["std"].min():
            most_spread = group_stats["std"].idxmax()
            parts.append(f"{most_spread} shows the most variability in {col}")

        # Detect outliers (values beyond 1.5*IQR)
        for grp in df[group_col].dropna().unique():
            subset = df[df[group_col] == grp][col].dropna()
            if len(subset) < 5:
                continue
            q1, q3 = subset.quantile(0.25), subset.quantile(0.75)
            iqr = q3 - q1
            n_outliers = ((subset < q1 - 1.5 * iqr) | (subset > q3 + 1.5 * iqr)).sum()
            if n_outliers > 0:
                parts.append(f"{n_outliers} outlier{'s' if n_outliers > 1 else ''} detected in {col} for {grp}")

    if hue:
        # Check for subgroup pattern (e.g., dimorphism)
        for col in box_cols[:1]:
            sub_means = df.groupby([group_col, hue])[col].mean().unstack()
            if sub_means.shape[1] == 2:
                col_a, col_b = sub_means.columns
                if (sub_means[col_a] > sub_means[col_b]).all() or (sub_means[col_b] > sub_means[col_a]).all():
                    higher = col_a if (sub_means[col_a] > sub_means[col_b]).all() else col_b
                    parts.append(
                        f"{higher} consistently has higher {col} across all {group_col} groups, "
                        f"indicating a clear {hue}-based pattern"
                    )

    return ". ".join(parts) + "." if parts else f"Box plots show the distribution of features across {group_col} groups."


def _find_simpsons_paradox(df, numeric_cols, hue_col):
    """Find a pair of numeric columns exhibiting Simpson's paradox.

    Returns (col_x, col_y, r_overall, {group: r}) or None.
    """
    groups = df[hue_col].dropna().unique()
    if len(groups) < 2:
        return None

    best = None
    best_strength = 0  # how strongly the paradox manifests

    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            cx, cy = numeric_cols[i], numeric_cols[j]
            pair_df = df[[cx, cy, hue_col]].dropna()
            if len(pair_df) < 20:
                continue

            r_overall = pair_df[[cx, cy]].corr().iloc[0, 1]
            if abs(r_overall) < 0.15:
                continue

            group_corrs = {}
            for val in groups:
                subset = pair_df[pair_df[hue_col] == val][[cx, cy]]
                if len(subset) > 10:
                    group_corrs[val] = subset.corr().iloc[0, 1]

            if len(group_corrs) < 2:
                continue

            # Paradox: overall sign differs from majority of within-group signs
            signs_within = [np.sign(r) for r in group_corrs.values() if not np.isnan(r)]
            if not signs_within:
                continue

            majority_sign = np.sign(sum(signs_within))
            if majority_sign != 0 and majority_sign != np.sign(r_overall):
                strength = abs(r_overall) + abs(np.mean(list(group_corrs.values())))
                if strength > best_strength:
                    best_strength = strength
                    best = (cx, cy, r_overall, group_corrs)

    return best


# ── Visualization generators ────────────────────────────────────────────────

def generate_visualizations(df):
    """Generate EDA visualizations for any dataframe. Returns list of (title, base64_png, description)."""
    sns.set_theme(style="whitegrid", palette="colorblind")
    visualizations = []

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    hue_col = _pick_hue_col(df)

    # 1. Distributions of numeric features (optionally colored by hue)
    if numeric_cols:
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
                ax.legend(fontsize=8)
            else:
                ax.hist(df[col].dropna(), bins=20, alpha=0.7, edgecolor="white", color="steelblue")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        desc = _distribution_insights(df, cols_to_plot, hue_col)
        visualizations.append(("Feature Distributions", _fig_to_base64(fig), desc))

    # 2. Scatter plot of the two most correlated numeric features
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr().abs()
        np.fill_diagonal(corr.values, 0)
        max_pair = corr.stack().idxmax()
        col_x, col_y = max_pair
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
        ax.set_xlabel(col_x, fontsize=12)
        ax.set_ylabel(col_y, fontsize=12)
        ax.set_title(f"{col_x} vs. {col_y}", fontsize=14, fontweight="bold")
        fig.tight_layout()

        desc = _scatter_insights(df, col_x, col_y, r_val, hue_col)
        visualizations.append((f"{col_x} vs. {col_y}", _fig_to_base64(fig), desc))

    # 2b. Simpson's paradox scatter plot (if detected and not already shown)
    if hue_col and len(numeric_cols) >= 2:
        paradox_pair = _find_simpsons_paradox(df, numeric_cols, hue_col)
        # Only add if it's a different pair than the one already plotted
        already_plotted = (col_x, col_y) if len(numeric_cols) >= 2 else (None, None)
        if paradox_pair and set(paradox_pair[:2]) != set(already_plotted):
            px, py, r_overall, group_corrs = paradox_pair
            fig, ax = plt.subplots(figsize=(9, 7))
            groups = df[hue_col].dropna().unique()
            colors = sns.color_palette("colorblind", len(groups))
            for val, color in zip(groups, colors):
                subset = df[df[hue_col] == val]
                ax.scatter(subset[px], subset[py], c=[color], label=str(val),
                           alpha=0.7, edgecolors="white", s=60)
            ax.legend(title=hue_col, fontsize=10)
            ax.set_xlabel(px, fontsize=12)
            ax.set_ylabel(py, fontsize=12)
            ax.set_title(f"{px} vs. {py} (Simpson's Paradox)", fontsize=14, fontweight="bold")
            fig.tight_layout()

            direction_overall = "negative" if r_overall < 0 else "positive"
            direction_within = "positive" if r_overall < 0 else "negative"
            group_details = ", ".join(
                f"{g}: r = {r:.2f}" for g, r in group_corrs.items()
            )
            desc = (
                f"Simpson's paradox: the overall correlation between {px} and {py} "
                f"is {direction_overall} (r = {r_overall:.2f}), but within each {hue_col} "
                f"group the correlation is {direction_within} ({group_details}). "
                f"This reversal occurs because the {hue_col} groups occupy different "
                f"regions of the feature space, creating a misleading aggregate trend."
            )
            visualizations.append((
                f"{px} vs. {py} (Simpson's Paradox)",
                _fig_to_base64(fig),
                desc,
            ))

    # 3. Correlation heatmap
    if len(numeric_cols) >= 2:
        fig, ax = plt.subplots(figsize=(max(8, len(numeric_cols)), max(6, len(numeric_cols) * 0.8)))
        corr = df[numeric_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                    center=0, vmin=-1, vmax=1, ax=ax, square=True)
        ax.set_title("Correlation Matrix", fontsize=14, fontweight="bold")
        fig.tight_layout()

        desc = _correlation_insights(df, numeric_cols)
        visualizations.append(("Correlation Heatmap", _fig_to_base64(fig), desc))

    # 4. Box plots of numeric features by group
    group_col = _pick_group_col(df)
    if group_col and numeric_cols:
        box_cols = numeric_cols[:2]
        n_box = len(box_cols)
        fig, axes = plt.subplots(1, n_box, figsize=(6.5 * n_box, 6))
        if n_box == 1:
            axes = [axes]
        fig.suptitle(f"Numeric Features by {group_col}", fontsize=14, fontweight="bold")

        hue = None
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        for c in cat_cols:
            if c != group_col and 2 <= df[c].nunique() <= 4 and df[c].isna().mean() < 0.3:
                hue = c
                break

        for ax, col in zip(axes, box_cols):
            plot_df = df.dropna(subset=[col, group_col])
            if hue:
                plot_df = plot_df.dropna(subset=[hue])
                sns.boxplot(data=plot_df, x=group_col, y=col, hue=hue, ax=ax, palette="Set2")
                ax.legend(title=hue)
            else:
                sns.boxplot(data=plot_df, x=group_col, y=col, hue=group_col, ax=ax, palette="Set2", legend=False)
            ax.set_xlabel(group_col, fontsize=12)
            ax.set_ylabel(col, fontsize=12)
            ax.set_title(col)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        desc = _boxplot_insights(df, box_cols, group_col, hue)
        visualizations.append((f"Box Plots by {group_col}", _fig_to_base64(fig), desc))

    return visualizations


def run(df):
    """Run the full Visualizer Agent on the given dataframe. Returns (summary_dict, visualizations_list)."""
    summary = generate_statistical_summary(df)
    visualizations = generate_visualizations(df)
    return summary, visualizations
