from __future__ import annotations

import random
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


LABELS = (-1, 0, 1)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def normalize_name(value: object) -> str:
    """Return a conservative key used only for cross-source entity alignment."""
    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_signed_matrix(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    if frame.index.has_duplicates:
        raise ValueError("Microbe names must be unique in the signed matrix.")
    if frame.columns.has_duplicates:
        raise ValueError("Disease names must be unique in the signed matrix.")
    numeric = frame.apply(pd.to_numeric, errors="raise")
    observed = set(np.unique(numeric.to_numpy(dtype=np.int8)).tolist())
    if not observed.issubset(set(LABELS)):
        raise ValueError(f"Signed matrix contains invalid values: {sorted(observed)}")
    return numeric.astype(np.int8)


def signed_edge_table(matrix: pd.DataFrame) -> pd.DataFrame:
    values = matrix.to_numpy()
    rows, cols = np.nonzero(values)
    return pd.DataFrame(
        {
            "microbe": matrix.index.to_numpy()[rows],
            "disease": matrix.columns.to_numpy()[cols],
            "effect": values[rows, cols].astype(np.int8),
        }
    )


def require_columns(frame: pd.DataFrame, columns: list[str], source: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")

