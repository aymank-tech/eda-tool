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
    # Zero out diagonal (self-correlations) using pandas to avoid read-only array issues
    corr_abs = corr_abs - pd.DataFrame(np.eye(len(corr_abs)), index=corr_abs.index, columns=corr_abs.columns)

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
    max_corr_per_col = max_corr_per_col - pd.DataFrame(np.eye(len(max_corr_per_col)), index=max_corr_per_col.index, columns=max_corr_per_col.columns)
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
        corr = corr - pd.DataFrame(np.eye(len(corr)), index=corr.index, columns=corr.columns)
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


# ── Refinement ──────────────────────────────────────────────────────────────

def _enhanced_distribution_insights(df, cols_to_plot, hue_col):
    """Richer distribution description with outlier detection and distribution shape."""
    parts = []
    for col in cols_to_plot:
        series = df[col].dropna()
        skew = series.skew()
        median = series.median()
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

        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        n_outliers = ((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum()
        if n_outliers > 0:
            parts.append(f"{n_outliers} outlier{'s' if n_outliers > 1 else ''} detected in {col} (beyond 1.5×IQR)")
        else:
            parts.append(f"No outliers detected in {col} by IQR criterion")

    if hue_col:
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
    """Richer scatter description with outlier and per-group detail."""
    parts = []
    strength = "strongly" if abs(r_val) > 0.7 else "moderately" if abs(r_val) > 0.4 else "weakly"
    direction = "positively" if r_val > 0 else "negatively"
    parts.append(f"{col_x} and {col_y} are {strength} {direction} correlated (r = {r_val:.2f})")

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
            parts.append(f"Strongest within-group correlation: {strongest} (r = {group_corrs[strongest]:.2f})")

    return ". ".join(parts) + "."


def _enhanced_correlation_insights(df, numeric_cols):
    """Richer correlation heatmap description."""
    corr = df[numeric_cols].corr()
    corr_abs = corr.abs() - pd.DataFrame(np.eye(len(corr)), index=corr.index, columns=corr.columns)
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


def refine(df, summary, visualizations, overall_evaluations, per_viz_evaluations):
    """Refine visualizations based on Critic feedback.

    Takes the original Visualizer output and the Critic's evaluation, then
    regenerates improved visualizations addressing the identified issues.

    Returns:
        refined_summary: dict (passed through unchanged)
        refined_visualizations: list of (title, base64_png, description)
        changes_made: list of strings describing what was improved
    """
    sns.set_theme(style="whitegrid", palette="colorblind")

    # Collect all issues to determine what needs fixing
    all_issues = []
    issues_by_title = {}
    for title, _scores, issues in per_viz_evaluations:
        issues_by_title[title] = issues
        all_issues.extend(issues)

    if not all_issues:
        return summary, visualizations, ["No issues found — visualizations are already high quality."]

    # Categorize improvements needed
    issue_types = set()
    for issue in all_issues:
        il = issue.lower()
        if "description" in il or "brief" in il or "interpretation" in il:
            issue_types.add("Expanded descriptions with richer statistical detail")
        if "outlier" in il:
            issue_types.add("Added outlier detection and discussion")
        if "axis" in il or "label" in il or "legend" in il:
            issue_types.add("Improved axis labels, legends, and titles")
        if "surface" in il or "deeper" in il or "insight" in il:
            issue_types.add("Deepened analytical insights in descriptions")
        if "narrow" in il or "scope" in il or "coverage" in il:
            issue_types.add("Broadened coverage of distributions, relationships, and outliers")
    if not issue_types:
        issue_types.add("General improvements to descriptions and visual clarity")
    changes_made = list(issue_types)

    # Regenerate each visualization with enhancements
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    hue_col = _pick_hue_col(df)
    group_col = _pick_group_col(df)
    refined = []

    for title, img_b64, description in visualizations:
        if "Distribution" in title or "distribution" in title:
            refined.append(_refine_distribution(df, numeric_cols, hue_col))
        elif "vs." in title and "Simpson" not in title:
            parts = title.split(" vs. ")
            if len(parts) == 2 and parts[0].strip() in df.columns and parts[1].strip() in df.columns:
                refined.append(_refine_scatter(df, parts[0].strip(), parts[1].strip(), hue_col))
            else:
                refined.append((title, img_b64, description))
        elif "Correlation" in title or "correlation" in title:
            refined.append(_refine_correlation(df, numeric_cols))
        elif "Box" in title or "box" in title:
            refined.append(_refine_boxplot(df, numeric_cols, group_col))
        else:
            refined.append((title, img_b64, description))

    return summary, refined, changes_made


def _refine_distribution(df, numeric_cols, hue_col):
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

        median_val = df[col].median()
        ax.axvline(median_val, color="red", linestyle="--", linewidth=1, label=f"Median: {median_val:.1f}")
        ax.legend(fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    desc = _enhanced_distribution_insights(df, cols_to_plot, hue_col)
    return ("Feature Distributions (Refined)", _fig_to_base64(fig), desc)


def _refine_scatter(df, col_x, col_y, hue_col):
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


def _refine_correlation(df, numeric_cols):
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


def _refine_boxplot(df, numeric_cols, group_col):
    if not group_col or not numeric_cols:
        return ("Box Plots", "", "No group column available for box plots.")

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


# ── LLM-based refinement ────────────────────────────────────────────────────

_LLM_SYSTEM_PROMPT = """\
You are an expert data visualization coder. You write Python code that creates \
publication-quality EDA visualizations using matplotlib and seaborn.

Your code will be executed with exec(). The DataFrame is available as `df`. \
The modules `matplotlib.pyplot as plt`, `seaborn as sns`, `numpy as np`, \
and `pandas as pd` are pre-imported.

CRITICAL RULES:
1. Create a list called `results` containing tuples of (title_str, fig_object, description_str)
2. Each fig must be a matplotlib Figure object created with plt.subplots() or plt.figure()
3. DO NOT call plt.show()
4. DO NOT call plt.close()
5. Write detailed descriptions (2-3 sentences) with specific statistical observations \
   (mention actual numbers: means, medians, correlations, outlier counts, etc.)
6. Call sns.set_theme(style="whitegrid", palette="colorblind") at the top
7. Every chart MUST have: a descriptive title, labeled axes, and a legend where applicable
8. Create 4-5 diverse visualizations covering: distributions, scatter/relationships, \
   correlation heatmap, group comparisons (box/violin plots), and missing data or outlier analysis
9. Use different chart types from the previous iteration (e.g., violin instead of box, \
   KDE instead of histogram, pair plots, etc.)
10. Directly address every issue raised by the Scorer

Output ONLY valid Python code. No markdown fencing, no explanations, no comments outside the code."""


def _build_llm_user_prompt(df, scores, per_viz_evaluations):
    """Build the user prompt for LLM-based visualization refinement."""
    import json

    col_info_lines = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        n_unique = df[col].nunique()
        n_missing = df[col].isnull().sum()
        sample_vals = df[col].dropna().head(3).tolist()
        col_info_lines.append(
            f"  - {col} (dtype={dtype}, {n_unique} unique, {n_missing} missing, "
            f"sample={sample_vals})"
        )
    col_info_str = "\n".join(col_info_lines)

    feedback_lines = []
    for title, viz_scores, issues in per_viz_evaluations:
        score_str = ", ".join(f"{d}: {s}" for d, s in viz_scores.items())
        issues_str = "; ".join(issues) if issues else "No issues"
        feedback_lines.append(f"  - \"{title}\" [{score_str}]\n    Issues: {issues_str}")
    feedback_str = "\n".join(feedback_lines)

    aggregate_str = json.dumps({d: s for d, (s, _) in scores.items()}, indent=2)
    sample_csv = df.head(10).to_csv(index=False)

    return f"""Generate improved EDA visualizations for this dataset. The previous \
visualizations scored poorly — create SIGNIFICANTLY DIFFERENT charts that address every issue.

## Dataset
Shape: {df.shape[0]} rows × {df.shape[1]} columns
Columns:
{col_info_str}

Sample data (first 10 rows as CSV):
{sample_csv}

## Previous Scorer Feedback
Per-visualization:
{feedback_str}

Aggregate scores:
{aggregate_str}

## Instructions
- Create 4-5 visualizations that are substantially different from the previous ones
- Use chart types like: violin plots, KDE plots, pair plots, swarm plots, heatmaps, \
  bar charts with error bars, etc.
- Every description must include actual numbers from the data
- Directly address each issue listed above
- Make sure axis labels and legends are clear and complete"""


def _call_openai_for_code(user_prompt, api_key):
    """Call OpenAI GPT-5.4 to generate visualization code."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-5.4",
        messages=[
            {"role": "system", "content": _LLM_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_completion_tokens=8192,
        temperature=0.7,
    )
    return response.choices[0].message.content


def _call_claude_for_code(user_prompt, api_key):
    """Call Claude Sonnet 4 to generate visualization code."""
    import anthropic
    import time
    client = anthropic.Anthropic(api_key=api_key)
    for attempt in range(3):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=8192,
                temperature=0.7,
                system=_LLM_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return response.content[0].text
        except anthropic.InternalServerError:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                raise


def _strip_code_fencing(code):
    """Remove markdown code fencing if present."""
    code = code.strip()
    if code.startswith("```"):
        first_newline = code.find("\n")
        code = code[first_newline + 1:] if first_newline != -1 else code[3:]
        if code.endswith("```"):
            code = code[:-3]
        code = code.strip()
    return code


def _execute_viz_code(code, df):
    """Execute generated code and return list of (title, base64_png, description)."""
    namespace = {
        "df": df.copy(),
        "pd": pd,
        "np": np,
        "plt": plt,
        "sns": sns,
    }

    exec(code, namespace)
    raw_results = namespace.get("results", [])

    refined_viz = []
    for item in raw_results:
        if len(item) >= 3:
            title, fig, desc = item[0], item[1], item[2]
            refined_viz.append((str(title), _fig_to_base64(fig), str(desc)))

    return refined_viz


def refine_with_llm(df, summary, visualizations, scores, per_viz_evaluations, api_key, model="openai"):
    """Use an LLM to generate entirely new visualization code based on Scorer feedback.

    Args:
        model: "openai" for GPT-5.4, "claude" for Claude Opus

    Falls back to programmatic refine() if code generation or execution fails.

    Returns:
        refined_summary: dict (passed through unchanged)
        refined_visualizations: list of (title, base64_png, description)
        changes_made: list of strings describing what was improved
    """
    user_prompt = _build_llm_user_prompt(df, scores, per_viz_evaluations)

    # ── Call LLM ─────────────────────────────────────────────────────────────
    try:
        if model == "claude":
            code = _call_claude_for_code(user_prompt, api_key)
            model_name = "Claude Opus"
        else:
            code = _call_openai_for_code(user_prompt, api_key)
            model_name = "GPT-5.4"
    except Exception:
        return refine(df, summary, visualizations, scores, per_viz_evaluations)

    code = _strip_code_fencing(code)

    # ── Execute the generated code ───────────────────────────────────────────
    try:
        refined_viz = _execute_viz_code(code, df)
    except Exception:
        return refine(df, summary, visualizations, scores, per_viz_evaluations)

    if not refined_viz:
        return refine(df, summary, visualizations, scores, per_viz_evaluations)

    changes = [f"Generated entirely new visualizations using {model_name} based on Scorer feedback"]
    return summary, refined_viz, changes
