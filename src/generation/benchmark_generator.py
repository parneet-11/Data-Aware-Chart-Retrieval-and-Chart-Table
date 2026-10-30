
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def stable_seed(text, global_seed=42):
    value = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

    return int(value[:8], 16) + global_seed


def shuffle_values(values, data_id, global_seed=42):
    values = np.array(values, dtype=float)

    rng = np.random.default_rng(
        stable_seed(data_id, global_seed)
    )

    for _ in range(20):
        shuffled = rng.permutation(values)

        if not np.array_equal(shuffled, values):
            return shuffled.tolist()

    # Safe fallback
    shuffled = np.roll(values, 1)

    if np.array_equal(shuffled, values):
        shuffled = values[::-1]

    return shuffled.tolist()


def table_text(columns, rows):
    header = " | ".join(
        str(column) for column in columns
    )

    row_text = [
        " | ".join(str(value) for value in row)
        for row in rows
    ]

    return " ; ".join([header] + row_text)


def table_hash(columns, rows):
    content = json.dumps(
        {
            "columns": columns,
            "rows": rows
        },
        ensure_ascii=False,
        sort_keys=True
    )

    return hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()


def get_title(row):
    title = row.get("title", "")

    if pd.isna(title) or str(title).strip() == "":
        return (
            f"{row['value_column']} by "
            f"{row['category_column']}"
        )

    return str(title)


def render_vertical_blue(
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

    ax.bar(
        positions,
        values,
        color="#4C78A8",
        edgecolor="#2F4B6C",
        linewidth=0.8
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        categories,
        rotation=40,
        ha="right"
    )

    ax.set_title(title, fontsize=12)
    ax.set_xlabel(category_label)
    ax.set_ylabel(value_label)

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.35
    )

    ax.axhline(0, color="black", linewidth=0.8)

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    fig.tight_layout()
    fig.savefig(
        output_path,
        bbox_inches="tight"
    )
    plt.close(fig)


def render_horizontal_orange(
    categories,
    values,
    title,
    category_label,
    value_label,
    output_path
):
    height = max(5, len(categories) * 0.45)

    fig, ax = plt.subplots(
        figsize=(9, height),
        dpi=120
    )

    positions = np.arange(len(categories))

    ax.barh(
        positions,
        values,
        color="#F58518",
        edgecolor="#A65300",
        linewidth=0.8
    )

    ax.set_yticks(positions)
    ax.set_yticklabels(categories)
    ax.invert_yaxis()

    ax.set_title(title, fontsize=13)
    ax.set_xlabel(value_label)
    ax.set_ylabel(category_label)

    ax.grid(
        axis="x",
        linestyle=":",
        alpha=0.4
    )

    ax.axvline(0, color="black", linewidth=0.8)

    fig.patch.set_facecolor("#F4F4F4")
    ax.set_facecolor("#F4F4F4")

    fig.tight_layout()
    fig.savefig(
        output_path,
        bbox_inches="tight"
    )
    plt.close(fig)


def choose_subset(pilot_df, limit_per_split):
    if limit_per_split is None:
        return pilot_df.copy()

    parts = []

    for split in ["train", "val", "test"]:
        split_df = pilot_df[
            pilot_df["split"] == split
        ]

        parts.append(
            split_df.head(limit_per_split)
        )

    return pd.concat(
        parts,
        ignore_index=True
    )


def generate_benchmark(
    pilot_path,
    output_directory,
    image_directory,
    limit_per_split=None,
    global_seed=42
):
    pilot_df = pd.read_parquet(pilot_path)

    pilot_df = choose_subset(
        pilot_df,
        limit_per_split
    )

    output_directory = Path(output_directory)
    image_directory = Path(image_directory)

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    image_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    chart_records = []
    table_records = []
    pair_records = []

    for _, row in tqdm(
        pilot_df.iterrows(),
        total=len(pilot_df),
        desc="Generating chart triplets"
    ):
        data_id = row["data_id"]
        split = row["split"]

        categories = json.loads(
            row["categories_json"]
        )

        values = [
            float(value)
            for value in json.loads(
                row["values_json"]
            )
        ]

        columns = [
            row["category_column"],
            row["value_column"]
        ]

        original_rows = [
            [category, value]
            for category, value in zip(
                categories,
                values
            )
        ]

        shuffled_values = shuffle_values(
            values,
            data_id,
            global_seed
        )

        shuffled_rows = [
            [category, value]
            for category, value in zip(
                categories,
                shuffled_values
            )
        ]

        negative_data_id = (
            f"{data_id}__shuffled"
        )

        anchor_chart_id = (
            f"{data_id}__vbar_blue"
        )

        positive_chart_id = (
            f"{data_id}__hbar_orange"
        )

        negative_chart_id = (
            f"{negative_data_id}__vbar_blue"
        )

        chart_folder = (
            image_directory
            / split
            / row["source_id"]
        )

        chart_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        anchor_path = (
            chart_folder
            / "anchor_vbar_blue.png"
        )

        positive_path = (
            chart_folder
            / "positive_hbar_orange.png"
        )

        negative_path = (
            chart_folder
            / "negative_vbar_blue.png"
        )

        title = get_title(row)

        render_vertical_blue(
            categories=categories,
            values=values,
            title=title,
            category_label=row["category_column"],
            value_label=row["value_column"],
            output_path=anchor_path
        )

        render_horizontal_orange(
            categories=categories,
            values=values,
            title=title,
            category_label=row["category_column"],
            value_label=row["value_column"],
            output_path=positive_path
        )

        # Exactly the same rendering style as the anchor
        render_vertical_blue(
            categories=categories,
            values=shuffled_values,
            title=title,
            category_label=row["category_column"],
            value_label=row["value_column"],
            output_path=negative_path
        )

        chart_records.extend([
            {
                "chart_id": anchor_chart_id,
                "data_id": data_id,
                "source_data_id": data_id,
                "split": split,
                "image_path": str(anchor_path),
                "chart_type": "v_bar",
                "style_id": "vbar_blue",
                "variant_type": "anchor",
                "is_hard_negative": False
            },
            {
                "chart_id": positive_chart_id,
                "data_id": data_id,
                "source_data_id": data_id,
                "split": split,
                "image_path": str(positive_path),
                "chart_type": "h_bar",
                "style_id": "hbar_orange",
                "variant_type": "positive",
                "is_hard_negative": False
            },
            {
                "chart_id": negative_chart_id,
                "data_id": negative_data_id,
                "source_data_id": data_id,
                "split": split,
                "image_path": str(negative_path),
                "chart_type": "v_bar",
                "style_id": "vbar_blue",
                "variant_type": "hard_negative",
                "is_hard_negative": True
            }
        ])

        table_records.extend([
            {
                "data_id": data_id,
                "source_data_id": data_id,
                "split": split,
                "columns_json": json.dumps(columns),
                "rows_json": json.dumps(original_rows),
                "table_text": table_text(
                    columns,
                    original_rows
                ),
                "table_hash": table_hash(
                    columns,
                    original_rows
                ),
                "variant_type": "original",
                "is_generated": False
            },
            {
                "data_id": negative_data_id,
                "source_data_id": data_id,
                "split": split,
                "columns_json": json.dumps(columns),
                "rows_json": json.dumps(shuffled_rows),
                "table_text": table_text(
                    columns,
                    shuffled_rows
                ),
                "table_hash": table_hash(
                    columns,
                    shuffled_rows
                ),
                "variant_type": "hard_negative",
                "is_generated": True
            }
        ])

        # Chart-to-chart triplet
        pair_records.append({
            "pair_id": f"{data_id}__chart_chart",
            "task": "chart_chart",
            "split": split,
            "anchor_id": anchor_chart_id,
            "positive_id": positive_chart_id,
            "negative_id": negative_chart_id,
            "anchor_modality": "chart",
            "positive_modality": "chart",
            "negative_modality": "chart",
            "source_data_id": data_id,
            "negative_type": "same_style_different_data"
        })

        # Anchor chart-to-table triplet
        pair_records.append({
            "pair_id": f"{data_id}__vbar_chart_table",
            "task": "chart_table",
            "split": split,
            "anchor_id": anchor_chart_id,
            "positive_id": data_id,
            "negative_id": negative_data_id,
            "anchor_modality": "chart",
            "positive_modality": "table",
            "negative_modality": "table",
            "source_data_id": data_id,
            "negative_type": "value_shuffled"
        })

        # Positive chart-to-table triplet
        pair_records.append({
            "pair_id": f"{data_id}__hbar_chart_table",
            "task": "chart_table",
            "split": split,
            "anchor_id": positive_chart_id,
            "positive_id": data_id,
            "negative_id": negative_data_id,
            "anchor_modality": "chart",
            "positive_modality": "table",
            "negative_modality": "table",
            "source_data_id": data_id,
            "negative_type": "value_shuffled"
        })

    charts_df = pd.DataFrame(chart_records)
    tables_df = pd.DataFrame(table_records)
    pairs_df = pd.DataFrame(pair_records)

    charts_df.to_parquet(
        output_directory / "generated_charts.parquet",
        index=False
    )

    tables_df.to_parquet(
        output_directory / "benchmark_tables.parquet",
        index=False
    )

    pairs_df.to_parquet(
        output_directory / "retrieval_pairs.parquet",
        index=False
    )

    return charts_df, tables_df, pairs_df
