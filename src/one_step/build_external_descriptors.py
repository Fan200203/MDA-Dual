from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.one_step.feature_common import zscore


EMPTY = {"", "nan", "none", "null"}


def present(series: pd.Series) -> np.ndarray:
    return (~series.fillna("").astype(str).str.strip().str.lower().isin(EMPTY)).to_numpy(dtype=np.float32)


def numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(frame.get(column, pd.Series(np.zeros(len(frame)))), errors="coerce").fillna(0.0)
    return np.log1p(np.maximum(values.to_numpy(dtype=np.float32), 0.0))


def count_values(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = frame.get(column, pd.Series([""] * len(frame))).fillna("").astype(str)
    return values.map(lambda value: len([item for item in value.split(";") if item.strip()])).to_numpy(dtype=np.float32)


def text_contains(frame: pd.DataFrame, column: str, needle: str) -> np.ndarray:
    values = frame.get(column, pd.Series([""] * len(frame))).fillna("").astype(str).str.lower()
    return values.str.contains(needle, regex=False).to_numpy(dtype=np.float32)


def microbe_descriptors(frame: pd.DataFrame) -> np.ndarray:
    columns = [
        present(frame.get("ncbi_tax_id", pd.Series([""] * len(frame)))),
        present(frame.get("scientific_name", pd.Series([""] * len(frame)))),
        present(frame.get("silva_accession", pd.Series([""] * len(frame)))),
        present(frame.get("tax_rank", pd.Series([""] * len(frame)))),
        present(frame.get("experimental_method", pd.Series([""] * len(frame)))),
        present(frame.get("sample_origin", pd.Series([""] * len(frame)))),
        np.log1p(count_values(frame, "source_hits")),
        np.log1p(count_values(frame, "source_name_hits")),
        text_contains(frame, "match_types", "exact"),
        text_contains(frame, "match_types", "alias"),
        numeric(frame, "peryton_row_count"),
        numeric(frame, "peryton_disease_count"),
        numeric(frame, "peryton_pmid_count"),
        numeric(frame, "peryton_method_count"),
        numeric(frame, "peryton_origin_count"),
        numeric(frame, "peryton_16s_rows"),
        present(frame.get("blast", pd.Series([""] * len(frame)))),
        np.log1p(
            present(frame.get("ncbi_tax_id", pd.Series([""] * len(frame))))
            + present(frame.get("silva_accession", pd.Series([""] * len(frame))))
            + present(frame.get("experimental_method", pd.Series([""] * len(frame))))
            + present(frame.get("sample_origin", pd.Series([""] * len(frame))))
        ),
    ]
    result = np.column_stack(columns).astype(np.float32)
    if result.shape[1] != 18:
        raise AssertionError(result.shape)
    return zscore(result)


def disease_descriptors(frame: pd.DataFrame) -> np.ndarray:
    result = np.column_stack(
        [
            present(frame.get("mesh_id", pd.Series([""] * len(frame)))),
            present(frame.get("mesh_heading", pd.Series([""] * len(frame)))),
            present(frame.get("scientific_name", pd.Series([""] * len(frame)))),
            numeric(frame, "hsdn_symptom_count"),
            numeric(frame, "pubmed_occurrence_sum"),
            numeric(frame, "tfidf_sum"),
            numeric(frame, "dfs1"),
        ]
    ).astype(np.float32)
    if result.shape[1] != 7:
        raise AssertionError(result.shape)
    return zscore(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build transparent 18-D microbe and 7-D disease descriptors.")
    parser.add_argument("--microbes", type=Path, required=True, help="Canonical microbe metadata table used in the experiment (not redistributed).")
    parser.add_argument("--diseases", type=Path, required=True, help="Canonical disease metadata table used in the experiment (not redistributed).")
    parser.add_argument("--output", type=Path, default=Path("artifacts/one_step/base_features"))
    args = parser.parse_args()
    microbe = microbe_descriptors(pd.read_csv(args.microbes))
    disease = disease_descriptors(pd.read_csv(args.diseases))
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "microbe_external_18.npy", microbe)
    np.save(args.output / "disease_external_7.npy", disease)
    print({"microbe_shape": microbe.shape, "disease_shape": disease.shape})


if __name__ == "__main__":
    main()
