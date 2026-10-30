
import hashlib
import json
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm


def read_table(table_path):
    df = pd.read_csv(table_path, keep_default_na=False)

    # Convert Pandas/NumPy values into JSON-safe Python values
    table_object = json.loads(
        df.to_json(
            orient="split",
            force_ascii=False
        )
    )

    columns = [str(column) for column in table_object["columns"]]
    rows = table_object["data"]

    return columns, rows


def linearize_table(columns, rows):
    header = " | ".join(columns)

    row_texts = [
        " | ".join(
            "" if value is None else str(value)
            for value in row
        )
        for row in rows
    ]

    return " ; ".join([header] + row_texts)


def create_table_hash(columns, rows):
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


def extract_title(annotation):
    title = (
        annotation
        .get("general_figure_info", {})
        .get("title", "")
    )

    if isinstance(title, str):
        return title

    if isinstance(title, dict):
        return str(title.get("text", ""))

    return ""


def build_chartqa(
    dataset_root,
    output_directory,
    limit_per_split=None
):
    dataset_root = Path(dataset_root)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    table_records = []
    chart_records = []
    query_records = []
    statistics = {}

    for split in ["train", "val", "test"]:
        split_directory = dataset_root / split

        images = {
            path.stem: path
            for path in (split_directory / "png").glob("*.png")
        }

        tables = {
            path.stem: path
            for path in (split_directory / "tables").glob("*.csv")
        }

        annotations = {
            path.stem: path
            for path in (
                split_directory / "annotations"
            ).glob("*.json")
        }

        # Only keep IDs that have image, table and annotation
        sample_ids = sorted(
            set(images)
            & set(tables)
            & set(annotations)
        )

        if limit_per_split is not None:
            sample_ids = sample_ids[:limit_per_split]

        selected_ids = set(sample_ids)

        print(
            f"\nProcessing {split}: "
            f"{len(sample_ids)} charts"
        )

        for sample_id in tqdm(sample_ids):
            with open(
                annotations[sample_id],
                "r",
                encoding="utf-8"
            ) as file:
                annotation = json.load(file)

            columns, rows = read_table(
                tables[sample_id]
            )

            data_id = f"chartqa_{split}_{sample_id}"
            chart_id = f"{data_id}_original"

            table_records.append({
                "data_id": data_id,
                "source_id": sample_id,
                "dataset": "chartqa",
                "split": split,
                "title": extract_title(annotation),
                "columns_json": json.dumps(
                    columns,
                    ensure_ascii=False
                ),
                "rows_json": json.dumps(
                    rows,
                    ensure_ascii=False
                ),
                "table_text": linearize_table(
                    columns,
                    rows
                ),
                "table_hash": create_table_hash(
                    columns,
                    rows
                ),
                "num_rows": len(rows),
                "num_columns": len(columns),
                "table_path": str(tables[sample_id])
            })

            chart_records.append({
                "chart_id": chart_id,
                "data_id": data_id,
                "source_id": sample_id,
                "dataset": "chartqa",
                "split": split,
                "image_path": str(images[sample_id]),
                "annotation_path": str(
                    annotations[sample_id]
                ),
                "chart_type": annotation.get(
                    "type",
                    "unknown"
                ),
                "num_series": len(
                    annotation.get("models", [])
                ),
                "style_id": "original",
                "variant_type": "original",
                "is_original": True
            })

        split_query_count = 0

        for question_source in ["human", "augmented"]:
            question_path = (
                split_directory
                / f"{split}_{question_source}.json"
            )

            with open(
                question_path,
                "r",
                encoding="utf-8"
            ) as file:
                questions = json.load(file)

            for index, question in enumerate(questions):
                sample_id = Path(
                    question["imgname"]
                ).stem

                # During smoke test, only retain selected charts
                if sample_id not in selected_ids:
                    continue

                data_id = f"chartqa_{split}_{sample_id}"
                chart_id = f"{data_id}_original"

                query_records.append({
                    "query_id": (
                        f"chartqa_{split}_"
                        f"{question_source}_{index}"
                    ),
                    "data_id": data_id,
                    "chart_id": chart_id,
                    "source_id": sample_id,
                    "dataset": "chartqa",
                    "split": split,
                    "question_source": question_source,
                    "question": str(question["query"]),
                    "answer": str(question["label"]),
                    "query_type": "unclassified"
                })

                split_query_count += 1

        statistics[split] = {
            "charts": len(sample_ids),
            "tables": len(sample_ids),
            "queries": split_query_count
        }

    tables_df = pd.DataFrame(table_records)
    charts_df = pd.DataFrame(chart_records)
    queries_df = pd.DataFrame(query_records)

    tables_df.to_parquet(
        output_directory / "tables.parquet",
        index=False
    )

    charts_df.to_parquet(
        output_directory / "charts.parquet",
        index=False
    )

    queries_df.to_parquet(
        output_directory / "queries.parquet",
        index=False
    )

    with open(
        output_directory / "dataset_statistics.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(statistics, file, indent=2)

    return tables_df, charts_df, queries_df, statistics
