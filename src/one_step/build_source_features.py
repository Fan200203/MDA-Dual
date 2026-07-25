from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import load_signed_matrix
from src.one_step.feature_common import signed_profile_cosine, source_incidence, svd64


def build(source: str, edges_path: Path, matrix_path: Path, output_dir: Path, seed: int) -> dict[str, object]:
    matrix = load_signed_matrix(matrix_path)
    microbes = matrix.index.astype(str).tolist()
    diseases = matrix.columns.astype(str).tolist()
    edges = pd.read_csv(edges_path)
    incidence, coverage = source_incidence(edges, microbes, diseases)
    microbe = svd64(signed_profile_cosine(incidence), seed=seed)
    disease = svd64(signed_profile_cosine(incidence.T), seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(microbe, index=microbes).rename_axis("microbe").to_csv(output_dir / f"{source}_microbe_svd64.csv")
    pd.DataFrame(disease, index=diseases).rename_axis("disease").to_csv(output_dir / f"{source}_disease_svd64.csv")
    payload = {
        "source": source,
        "construction": "source signed incidence -> within-entity cosine co-profile -> randomized SVD64 -> z-score",
        "raw_reference_similarity_matrix_included": False,
        "microbe_shape": list(microbe.shape),
        "disease_shape": list(disease.shape),
        **coverage,
    }
    (output_dir / f"{source}_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean-room source-specific feature construction; raw reference similarity matrices are not used.")
    parser.add_argument("--source", choices=("peryton", "disbiome", "hmdad"), required=True)
    parser.add_argument("--edges", type=Path, required=True, help="Officially obtained source association CSV with microbe,disease,effect (-1/+1).")
    parser.add_argument("--matrix", type=Path, default=Path("data/model/A_sign_matrix.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/one_step/source_features"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.edges, args.matrix, args.output, args.seed), indent=2))


if __name__ == "__main__":
    main()
