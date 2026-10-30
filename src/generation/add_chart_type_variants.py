
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.generation.benchmark_generator import (
    shuffle_values,
    get_title,
    choose_subset
)


def line_semantic_status(categories, category_label):
    label = str(category_label).lower()

    ordered_words = [
        "year", "date", "month", "time",
        "age", "quarter", "week"
    ]

    if any(word in label for word in ordered_words):
        return "ordered"

    cleaned = [
        re.sub(r"[^0-9.\-]", "", str(value))
        for value in categories
    ]

    try:
        [float(value) for value in cleaned]
        return "ordered"
    except ValueError:
        return "categorical_unverified"


def pie_is_valid(values):
    return (
        2 <= len(values) <= 8
        and all(value > 0 for value in values)
        and sum(values) > 0
    )


def render_line(
    categories,
    values,
    title,
    category_label,
    value_label,
    output_path
):
    width = max(8, len(categories) * 0.6)

    fig, ax = plt.subplots(
        figsize=(width, 5),
        dpi=120
    )

    positions = np.arange(len(categories))

    ax.plot(
        positions,
        values,
        color="#2CA02C",
        marker="o",
        markersize=7,
        linewidth=2.5
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        categories,
        rotation=40,
        ha="right"
    )

    ax.set_title(title)
    ax.set_xlabel(category_label)
    ax.set_ylabel(value_label)

    ax.grid(
        linestyle="--",
        alpha=0.35
    )

    ax.axhline(0, color="black", linewidth=0.8)

    fig.patch.set_facecolor("white")
    ax.set_facecolor("#F8FFF8")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def render_pie(
    categories,
    values,
    title,
    output_path
):
    total = sum(values)

    def show_original_value(percent):
        value = percent * total / 100

        if abs(value) >= 100:
            return f"{value:.0f}"

        return f"{value:.2f}".rstrip("0").rstrip(".")

    colors = plt.cm.Set3(
        np.linspace(0, 1, len(categories))
    )

    fig, ax = plt.subplots(
        figsize=(8, 7),
        dpi=120
    )

    ax.pie(
        values,
        labels=categories,
        colors=colors,
        autopct=show_original_value,
        startangle=90,
        wedgeprops={
            "edgecolor": "white",
            "linewidth": 1
        }
    )

    ax.set_title(title)
    ax.axis("equal")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def add_chart_type_variants(
    pilot_path,
    benchmark_directory,
    image_directory,
    limit_per_split=None,
    global_seed=42
):
    pilot_df = pd.read_parquet(pilot_path)

    pilot_df = choose_subset(
        pilot_df,
        limit_per_split
    )

    benchmark_directory = Path(benchmark_directory)
    image_directory = Path(image_directory)

    charts_path = (
        benchmark_directory
        / "generated_charts.parquet"
    )

    pairs_path = (
        benchmark_directory
        / "retrieval_pairs.parquet"
    )

    charts_df = pd.read_parquet(charts_path)
    pairs_df = pd.read_parquet(pairs_path)

    new_charts = []
    new_pairs = []

    existing_chart_ids = set(
        charts_df["chart_id"]
    )

    for _, row in tqdm(
        pilot_df.iterrows(),
        total=len(pilot_df),
        desc="Adding line and pie variants"
    ):
        data_id = row["data_id"]
        split = row["split"]

        anchor_id = f"{data_id}__vbar_blue"
        negative_data_id = f"{data_id}__shuffled"
        bar_negative_id = (
            f"{negative_data_id}__vbar_blue"
        )

        # Only extend tables already generated
        if anchor_id not in existing_chart_ids:
            continue

        categories = json.loads(
            row["categories_json"]
        )

        values = [
            float(value)
            for value in json.loads(
                row["values_json"]
            )
        ]

        shuffled_values = shuffle_values(
            values,
            data_id,
            global_seed
        )

        title = get_title(row)

        chart_folder = (
            image_directory
            / split
            / row["source_id"]
        )

        chart_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        # -------------------------
        # Line positive and negative
        # -------------------------

        line_positive_id = (
            f"{data_id}__line_green"
        )

        line_negative_id = (
            f"{negative_data_id}__line_green"
        )

        line_positive_path = (
            chart_folder
            / "positive_line_green.png"
        )

        line_negative_path = (
            chart_folder
            / "negative_line_green.png"
        )

        render_line(
            categories,
            values,
            title,
            row["category_column"],
            row["value_column"],
            line_positive_path
        )

        render_line(
            categories,
            shuffled_values,
            title,
            row["category_column"],
            row["value_column"],
            line_negative_path
        )

        line_status = line_semantic_status(
            categories,
            row["category_column"]
        )

        new_charts.extend([
            {
                "chart_id": line_positive_id,
                "data_id": data_id,
                "source_data_id": data_id,
                "split": split,
                "image_path": str(line_positive_path),
                "chart_type": "line",
                "style_id": "line_green",
                "variant_type": "hard_positive",
                "is_hard_negative": False,
                "semantic_status": line_status
            },
            {
                "chart_id": line_negative_id,
                "data_id": negative_data_id,
                "source_data_id": data_id,
                "split": split,
                "image_path": str(line_negative_path),
                "chart_type": "line",
                "style_id": "line_green",
                "variant_type": "hard_negative",
                "is_hard_negative": True,
                "semantic_status": line_status
            }
        ])

        # Bar anchor → line positive → bar negative
        new_pairs.append({
            "pair_id": f"{data_id}__bar_to_line",
            "task": "chart_chart",
            "split": split,
            "anchor_id": anchor_id,
            "positive_id": line_positive_id,
            "negative_id": bar_negative_id,
            "anchor_modality": "chart",
            "positive_modality": "chart",
            "negative_modality": "chart",
            "source_data_id": data_id,
            "negative_type": "same_style_different_data"
        })

        # Line anchor → bar positive → line negative
        new_pairs.append({
            "pair_id": f"{data_id}__line_to_bar",
            "task": "chart_chart",
            "split": split,
            "anchor_id": line_positive_id,
            "positive_id": anchor_id,
            "negative_id": line_negative_id,
            "anchor_modality": "chart",
            "positive_modality": "chart",
            "negative_modality": "chart",
            "source_data_id": data_id,
            "negative_type": "same_style_different_data"
        })

        # Line chart → matching table
        new_pairs.append({
            "pair_id": f"{data_id}__line_chart_table",
            "task": "chart_table",
            "split": split,
            "anchor_id": line_positive_id,
            "positive_id": data_id,
            "negative_id": negative_data_id,
            "anchor_modality": "chart",
            "positive_modality": "table",
            "negative_modality": "table",
            "source_data_id": data_id,
            "negative_type": "value_shuffled"
        })

        # -------------------------
        # Pie positive and negative
        # -------------------------

        if pie_is_valid(values):
            pie_positive_id = (
                f"{data_id}__pie_palette"
            )

            pie_negative_id = (
                f"{negative_data_id}__pie_palette"
            )

            pie_positive_path = (
                chart_folder
                / "positive_pie.png"
            )

            pie_negative_path = (
                chart_folder
                / "negative_pie.png"
            )

            render_pie(
                categories,
                values,
                title,
                pie_positive_path
            )

            render_pie(
                categories,
                shuffled_values,
                title,
                pie_negative_path
            )

            new_charts.extend([
                {
                    "chart_id": pie_positive_id,
                    "data_id": data_id,
                    "source_data_id": data_id,
                    "split": split,
                    "image_path": str(pie_positive_path),
                    "chart_type": "pie",
                    "style_id": "pie_palette",
                    "variant_type": "hard_positive",
                    "is_hard_negative": False,
                    "semantic_status": "composition_unverified"
                },
                {
                    "chart_id": pie_negative_id,
                    "data_id": negative_data_id,
                    "source_data_id": data_id,
                    "split": split,
                    "image_path": str(pie_negative_path),
                    "chart_type": "pie",
                    "style_id": "pie_palette",
                    "variant_type": "hard_negative",
                    "is_hard_negative": True,
                    "semantic_status": "composition_unverified"
                }
            ])

            new_pairs.extend([
                {
                    "pair_id": f"{data_id}__bar_to_pie",
                    "task": "chart_chart",
                    "split": split,
                    "anchor_id": anchor_id,
                    "positive_id": pie_positive_id,
                    "negative_id": bar_negative_id,
                    "anchor_modality": "chart",
                    "positive_modality": "chart",
                    "negative_modality": "chart",
                    "source_data_id": data_id,
                    "negative_type": "same_style_different_data"
                },
                {
                    "pair_id": f"{data_id}__pie_to_bar",
                    "task": "chart_chart",
                    "split": split,
                    "anchor_id": pie_positive_id,
                    "positive_id": anchor_id,
                    "negative_id": pie_negative_id,
                    "anchor_modality": "chart",
                    "positive_modality": "chart",
                    "negative_modality": "chart",
                    "source_data_id": data_id,
                    "negative_type": "same_style_different_data"
                },
                {
                    "pair_id": f"{data_id}__pie_chart_table",
                    "task": "chart_table",
                    "split": split,
                    "anchor_id": pie_positive_id,
                    "positive_id": data_id,
                    "negative_id": negative_data_id,
                    "anchor_modality": "chart",
                    "positive_modality": "table",
                    "negative_modality": "table",
                    "source_data_id": data_id,
                    "negative_type": "value_shuffled"
                }
            ])

    updated_charts = pd.concat(
        [
            charts_df,
            pd.DataFrame(new_charts)
        ],
        ignore_index=True
    ).drop_duplicates(
        subset=["chart_id"],
        keep="last"
    )

    updated_pairs = pd.concat(
        [
            pairs_df,
            pd.DataFrame(new_pairs)
        ],
        ignore_index=True
    ).drop_duplicates(
        subset=["pair_id"],
        keep="last"
    )

    updated_charts.to_parquet(
        charts_path,
        index=False
    )

    updated_pairs.to_parquet(
        pairs_path,
        index=False
    )

    return updated_charts, updated_pairs
