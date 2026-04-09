"""Visualizer Agent — performs EDA on the Palmer Penguins dataset."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from palmerpenguins import load_penguins
import io
import base64


def _fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def load_data():
    return load_penguins()


def generate_statistical_summary(df):
    """Generate a comprehensive statistical summary."""
    summary = {}

    summary["shape"] = f"{df.shape[0]} rows × {df.shape[1]} columns"

    # Missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        summary["missing_values"] = missing.to_dict()
    else:
        summary["missing_values"] = "No missing values"

    # Species distribution
    summary["species_counts"] = df["species"].value_counts().to_dict()

    # Island distribution
    summary["island_counts"] = df["island"].value_counts().to_dict()

    # Numeric summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    desc = df[numeric_cols].describe().round(2)
    summary["numeric_summary"] = desc

    # Correlations
    corr = df[numeric_cols].corr().round(3)
    summary["correlations"] = corr

    # Per-species stats
    species_stats = df.groupby("species")[numeric_cols].mean().round(2)
    summary["species_means"] = species_stats

    return summary


def generate_visualizations(df):
    """Generate 4 EDA visualizations. Returns list of (title, base64_png, description)."""
    sns.set_theme(style="whitegrid", palette="colorblind")
    visualizations = []

    # 1. Distributions of numeric features by species
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Distribution of Numeric Features by Species", fontsize=14, fontweight="bold")
    numeric_cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    labels = ["Bill Length (mm)", "Bill Depth (mm)", "Flipper Length (mm)", "Body Mass (g)"]
    for ax, col, label in zip(axes.flat, numeric_cols, labels):
        for species in df["species"].dropna().unique():
            subset = df[df["species"] == species][col].dropna()
            ax.hist(subset, bins=20, alpha=0.6, label=species, edgecolor="white")
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    visualizations.append((
        "Feature Distributions by Species",
        _fig_to_base64(fig),
        "Histograms showing the distribution of bill length, bill depth, flipper length, "
        "and body mass for each penguin species. Gentoo penguins are notably larger with "
        "longer flippers and greater body mass. Chinstrap and Adelie overlap in size but "
        "differ in bill dimensions."
    ))

    # 2. Scatter plot: bill length vs bill depth colored by species
    fig, ax = plt.subplots(figsize=(9, 7))
    species_list = df["species"].dropna().unique()
    colors = sns.color_palette("colorblind", len(species_list))
    for species, color in zip(species_list, colors):
        subset = df[df["species"] == species]
        ax.scatter(subset["bill_length_mm"], subset["bill_depth_mm"],
                   c=[color], label=species, alpha=0.7, edgecolors="white", s=60)
    ax.set_xlabel("Bill Length (mm)", fontsize=12)
    ax.set_ylabel("Bill Depth (mm)", fontsize=12)
    ax.set_title("Bill Length vs. Bill Depth by Species", fontsize=14, fontweight="bold")
    ax.legend(title="Species", fontsize=10)
    fig.tight_layout()
    visualizations.append((
        "Bill Length vs. Bill Depth (Simpson's Paradox)",
        _fig_to_base64(fig),
        "This scatter plot reveals Simpson's paradox: overall there appears to be a negative "
        "correlation between bill length and depth, but within each species the correlation is "
        "positive. Adelie penguins have short, deep bills; Chinstraps have long, deep bills; "
        "and Gentoos have long, shallow bills."
    ))

    # 3. Correlation heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    numeric_df = df[numeric_cols].dropna()
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, ax=ax, square=True,
                xticklabels=labels, yticklabels=labels)
    ax.set_title("Correlation Matrix of Numeric Features", fontsize=14, fontweight="bold")
    fig.tight_layout()
    visualizations.append((
        "Correlation Heatmap",
        _fig_to_base64(fig),
        "Flipper length and body mass are strongly positively correlated (r ≈ 0.87). "
        "Bill length correlates moderately with flipper length and body mass. "
        "Bill depth shows weak or negative correlations with other features when species "
        "are pooled — another manifestation of Simpson's paradox."
    ))

    # 4. Box plots by species and sex
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("Body Mass and Flipper Length by Species & Sex", fontsize=14, fontweight="bold")
    df_clean = df.dropna(subset=["sex"])
    sns.boxplot(data=df_clean, x="species", y="body_mass_g", hue="sex", ax=axes[0], palette="Set2")
    axes[0].set_xlabel("Species", fontsize=12)
    axes[0].set_ylabel("Body Mass (g)", fontsize=12)
    axes[0].set_title("Body Mass")
    axes[0].legend(title="Sex")

    sns.boxplot(data=df_clean, x="species", y="flipper_length_mm", hue="sex", ax=axes[1], palette="Set2")
    axes[1].set_xlabel("Species", fontsize=12)
    axes[1].set_ylabel("Flipper Length (mm)", fontsize=12)
    axes[1].set_title("Flipper Length")
    axes[1].legend(title="Sex")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    visualizations.append((
        "Body Mass & Flipper Length by Species and Sex",
        _fig_to_base64(fig),
        "Box plots reveal clear sexual dimorphism across all species — males are consistently "
        "heavier and have longer flippers. Gentoo penguins show the greatest size overall. "
        "Outliers are visible in Chinstrap body mass and Adelie flipper length."
    ))

    return visualizations


def run():
    """Run the full Visualizer Agent. Returns (summary_dict, visualizations_list)."""
    df = load_data()
    summary = generate_statistical_summary(df)
    visualizations = generate_visualizations(df)
    return summary, visualizations
