from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.common import load_signed_matrix
from src.one_step.feature_common import gaussian_interaction_profile, svd64


def main() -> None:
    parser = argparse.ArgumentParser(description="Build single-pass global GIP-M and GIP-D SVD64 blocks.")
    parser.add_argument("--matrix", type=Path, default=Path("data/model/A_sign_matrix.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/one_step/base_features"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    matrix = load_signed_matrix(args.matrix).to_numpy(dtype=np.float32)
    microbe = svd64(gaussian_interaction_profile(matrix), seed=args.seed)
    disease = svd64(gaussian_interaction_profile(matrix.T), seed=args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "microbe_global_gip_svd64.npy", microbe)
    np.save(args.output / "disease_global_gip_svd64.npy", disease)
    print({"microbe_shape": microbe.shape, "disease_shape": disease.shape})


if __name__ == "__main__":
    main()
