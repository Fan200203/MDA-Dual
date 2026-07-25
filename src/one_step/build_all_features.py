from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import load_signed_matrix
from src.one_step.generate_latent_128 import generate


def load_block(path: Path, expected_names: list[str], entity_column: str) -> np.ndarray:
    frame = pd.read_csv(path)
    if entity_column not in frame.columns:
        raise ValueError(f"{path} must contain {entity_column!r}")
    frame[entity_column] = frame[entity_column].astype(str)
    frame = frame.set_index(entity_column)
    missing = sorted(set(expected_names) - set(frame.index))
    if missing:
        raise ValueError(f"{path} is missing {len(missing)} required entities; first: {missing[:5]}")
    return frame.loc[expected_names].to_numpy(dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reassemble the exact 402-D microbe and 391-D disease one-step feature tables.")
    parser.add_argument("--matrix", type=Path, default=Path("data/model/A_sign_matrix.csv"))
    parser.add_argument("--feature-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/one_step/features"))
    args = parser.parse_args()

    matrix_frame = load_signed_matrix(args.matrix)
    microbes = matrix_frame.index.astype(str).tolist()
    diseases = matrix_frame.columns.astype(str).tolist()
    latent_m, latent_d, _ = generate(matrix_frame.to_numpy(dtype=np.float32), 128)

    microbe_blocks = [
        latent_m,
        load_block(args.feature_dir / "microbe_external_18.csv", microbes, "microbe"),
        load_block(args.feature_dir / "microbe_global_gip_64.csv", microbes, "microbe"),
        load_block(args.feature_dir / "peryton_microbe_svd64.csv", microbes, "microbe"),
        load_block(args.feature_dir / "disbiome_microbe_svd64.csv", microbes, "microbe"),
        load_block(args.feature_dir / "hmdad_microbe_svd64.csv", microbes, "microbe"),
    ]
    disease_blocks = [
        latent_d,
        load_block(args.feature_dir / "disease_external_7.csv", diseases, "disease"),
        load_block(args.feature_dir / "disease_global_gip_64.csv", diseases, "disease"),
        load_block(args.feature_dir / "peryton_disease_svd64.csv", diseases, "disease"),
        load_block(args.feature_dir / "disbiome_disease_svd64.csv", diseases, "disease"),
        load_block(args.feature_dir / "hmdad_disease_svd64.csv", diseases, "disease"),
    ]
    microbe = np.hstack(microbe_blocks).astype(np.float32)
    disease = np.hstack(disease_blocks).astype(np.float32)
    if microbe.shape != (2175, 402) or disease.shape != (810, 391):
        raise AssertionError(f"Unexpected feature shapes: {microbe.shape}, {disease.shape}")
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "microbe_features_402.npy", microbe)
    np.save(args.output / "disease_features_391.npy", disease)
    pd.DataFrame(microbe, index=microbes).rename_axis("microbe").to_csv(args.output / "microbe_features_402.csv")
    pd.DataFrame(disease, index=diseases).rename_axis("disease").to_csv(args.output / "disease_features_391.csv")
    manifest = {
        "microbe_shape": list(microbe.shape),
        "disease_shape": list(disease.shape),
        "pair_dimension": 793,
        "block_order": {
            "microbe": ["latent_128", "external_18", "global_gip_64", "peryton_svd64", "disbiome_svd64", "hmdad_svd64"],
            "disease": ["latent_128", "external_7", "global_gip_64", "peryton_svd64", "disbiome_svd64", "hmdad_svd64"],
        },
    }
    (args.output / "feature_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
