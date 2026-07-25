from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import load_signed_matrix


def gaussian_interaction_profile(association: np.ndarray) -> np.ndarray:
    squared_norms = np.sum(association * association, axis=1)
    distances = squared_norms[:, None] + squared_norms[None, :] - 2.0 * (association @ association.T)
    scale = float(np.mean(squared_norms))
    if scale <= 0 or not np.isfinite(scale):
        return np.zeros((association.shape[0], association.shape[0]), dtype=np.float32)
    result = np.exp(-np.maximum(distances, 0.0) / scale)
    # The stored GIP matrices used to generate the final 26.5.29 latent files
    # retain self-similarity at one.
    np.fill_diagonal(result, 1.0)
    return result.astype(np.float32)


def svd_latent(features: np.ndarray, dimension: int) -> tuple[np.ndarray, float]:
    centered = features - np.mean(features, axis=0)
    u, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    k = min(dimension, len(singular_values))
    denominator = float(np.sum(singular_values**2))
    explained = float(np.sum(singular_values[:k] ** 2) / denominator) if denominator else 0.0
    return u[:, :k] * singular_values[:k], explained


def zscore(values: np.ndarray) -> np.ndarray:
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    std[std == 0] = 1.0
    return ((values - mean) / std).astype(np.float32)


def generate(matrix: np.ndarray, dimension: int = 128) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    association = (matrix != 0).astype(np.float32)
    gip_m = gaussian_interaction_profile(association)
    gip_d = gaussian_interaction_profile(association.T)
    latent_m, variance_m = svd_latent(np.hstack([gip_m, association]), dimension)
    latent_d, variance_d = svd_latent(np.hstack([gip_d, association.T]), dimension)
    return zscore(latent_m), zscore(latent_d), {
        "microbe_explained_variance": variance_m,
        "disease_explained_variance": variance_d,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the actual 26.5.29 128-D latent block: |A_sign| -> GIP -> [GIP, association profile] -> full SVD -> z-score.")
    parser.add_argument("--matrix", type=Path, default=Path("data/model/A_sign_matrix.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/one_step/latent_128"))
    parser.add_argument("--dimension", type=int, default=128)
    args = parser.parse_args()

    frame = load_signed_matrix(args.matrix)
    latent_m, latent_d, summary = generate(frame.to_numpy(dtype=np.float32), args.dimension)
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "latent_microbe_128.npy", latent_m)
    np.save(args.output / "latent_disease_128.npy", latent_d)
    pd.DataFrame(latent_m, index=frame.index).rename_axis("microbe").to_csv(args.output / "latent_microbe_128.csv")
    pd.DataFrame(latent_d, index=frame.columns).rename_axis("disease").to_csv(args.output / "latent_disease_128.csv")
    payload = {
        "source": "adapted from reproduce_26_5_29/expandtoneo4j/run_pipeline.py",
        "matrix_shape": list(frame.shape),
        "microbe_latent_shape": list(latent_m.shape),
        "disease_latent_shape": list(latent_d.shape),
        **summary,
    }
    (args.output / "latent_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
