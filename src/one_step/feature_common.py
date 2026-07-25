from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.utils.extmath import randomized_svd

from src.common import normalize_name, require_columns


def canonical_names(microbe_path: Path, disease_path: Path) -> tuple[list[str], list[str]]:
    microbes = pd.read_csv(microbe_path)
    diseases = pd.read_csv(disease_path)
    require_columns(microbes, ["microbe"], str(microbe_path))
    require_columns(diseases, ["disease"], str(disease_path))
    return microbes["microbe"].astype(str).tolist(), diseases["disease"].astype(str).tolist()


def _first_index(names: list[str]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for index, name in enumerate(names):
        key = normalize_name(name)
        if key and key not in mapping:
            mapping[key] = index
    return mapping


def source_incidence(edges: pd.DataFrame, microbes: list[str], diseases: list[str]) -> tuple[np.ndarray, dict[str, int]]:
    require_columns(edges, ["microbe", "disease", "effect"], "source association table")
    microbe_index = _first_index(microbes)
    disease_index = _first_index(diseases)
    matrix = np.zeros((len(microbes), len(diseases)), dtype=np.float32)
    matched = 0
    for row in edges.itertuples(index=False):
        m = microbe_index.get(normalize_name(row.microbe))
        d = disease_index.get(normalize_name(row.disease))
        if m is None or d is None:
            continue
        effect = int(row.effect)
        if effect not in (-1, 1):
            continue
        matrix[m, d] += effect
        matched += 1
    matrix = np.sign(matrix).astype(np.float32)
    return matrix, {
        "input_edges": int(len(edges)),
        "matched_edges_before_conflict_collapse": matched,
        "mapped_nonzero_edges": int(np.count_nonzero(matrix)),
        "covered_microbes": int(np.count_nonzero(np.any(matrix != 0, axis=1))),
        "covered_diseases": int(np.count_nonzero(np.any(matrix != 0, axis=0))),
    }


def signed_profile_cosine(profile: np.ndarray) -> np.ndarray:
    normalized = normalize(profile, norm="l2", axis=1, copy=True)
    similarity = normalized @ normalized.T
    covered = np.any(profile != 0, axis=1)
    similarity[~covered, :] = 0.0
    similarity[:, ~covered] = 0.0
    similarity[np.diag_indices_from(similarity)] = covered.astype(np.float32)
    return similarity.astype(np.float32)


def svd64(matrix: np.ndarray, seed: int = 42, dimension: int = 64) -> np.ndarray:
    components = min(dimension, max(1, min(matrix.shape) - 1))
    u, singular_values, _ = randomized_svd(matrix, n_components=components, random_state=seed)
    embedding = (u * singular_values).astype(np.float32)
    embedding = StandardScaler().fit_transform(embedding).astype(np.float32)
    if components < dimension:
        embedding = np.pad(embedding, ((0, 0), (0, dimension - components)))
    return embedding


def build_source_block(
    source: str,
    edge_path: Path,
    microbe_path: Path,
    disease_path: Path,
    output_dir: Path,
    seed: int = 42,
) -> dict[str, object]:
    microbes, diseases = canonical_names(microbe_path, disease_path)
    edges = pd.read_csv(edge_path)
    incidence, coverage = source_incidence(edges, microbes, diseases)
    microbe_similarity = signed_profile_cosine(incidence)
    disease_similarity = signed_profile_cosine(incidence.T)
    microbe_features = svd64(microbe_similarity, seed=seed)
    disease_features = svd64(disease_similarity, seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"{source}_microbe_svd64.npy", microbe_features)
    np.save(output_dir / f"{source}_disease_svd64.npy", disease_features)
    payload = {
        "source": source,
        "construction": "signed source incidence -> within-entity cosine co-profile -> randomized SVD64 -> z-score",
        "copied_reference_similarity_matrix": False,
        "microbe_shape": list(microbe_features.shape),
        "disease_shape": list(disease_features.shape),
        **coverage,
    }
    (output_dir / f"{source}_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return payload


def gaussian_interaction_profile(profile: np.ndarray) -> np.ndarray:
    profile = profile.astype(np.float32)
    squared_norm = np.sum(profile * profile, axis=1)
    scale = float(np.mean(squared_norm))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    distances = squared_norm[:, None] + squared_norm[None, :] - 2.0 * (profile @ profile.T)
    similarity = np.exp(-np.maximum(distances, 0.0) / scale)
    np.fill_diagonal(similarity, 1.0)
    return similarity.astype(np.float32)


def zscore(array: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(array.astype(np.float32)).astype(np.float32)

