# Data-Aware Chart Retrieval and Chart-Table Matching

This repository contains the implementation for a group project on
data-aware chart retrieval and chart-table matching.

## Project Goals

1. Natural-language query to chart retrieval
2. Chart-to-table matching
3. Same-data/different-style positive generation
4. Same-style/different-data negative generation
5. Data-aware multimodal retrieval evaluation

## Repository Structure

- `src/` – Source code
- `configs/` – Project configuration files
- `data/benchmark/` – Controlled benchmark metadata and small benchmark files
- `generated_charts/` – Generated chart samples
- `artifacts/` – Local experiment outputs, not tracked by Git
- `reports/` – Local reports and evaluation outputs

## Dataset

The complete raw datasets are not stored in this repository because of
their size. Each team member must download or copy the datasets into:

```text
data/raw/
