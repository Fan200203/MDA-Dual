from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch import nn

from src.common import load_signed_matrix, seed_everything
from src.two_step.representations import (
    TrainedTransE,
    binary_metrics,
    concatenated_node_pair_features,
    degree_features,
    fit_existence_gcn,
    fit_transe_bundle,
    train_existence_mlp,
)


class ScoreMLP(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(1)


def reliable_negative_ranking(
    positives: np.ndarray,
    unlabeled: np.ndarray,
    transe: TrainedTransE,
    iterations: int,
    seed: int,
    inner_epochs: int = 20,
    prediction_batch_size: int = 50000,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the 0323 program's scalar TransE-score bootstrap ranking of U."""
    positive_scores = transe.score_pairs(positives, relation="INCREASE")
    unlabeled_scores = transe.score_pairs(unlabeled, relation="INCREASE")
    valid_positive = np.isfinite(positive_scores)
    valid_unlabeled = np.isfinite(unlabeled_scores)
    positives_scored = positive_scores[valid_positive]
    candidate_indices = np.where(valid_unlabeled)[0]
    candidates_scored = unlabeled_scores[valid_unlabeled]
    if not len(positives_scored) or not len(candidates_scored):
        raise RuntimeError("TransE could not score positive or unlabeled pairs in this fold")

    rng = np.random.default_rng(seed)
    probability_sum = np.zeros(len(candidates_scored), dtype=np.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for iteration in range(iterations):
        torch.manual_seed(seed + iteration)
        sample_size = min(len(positives_scored), len(candidates_scored))
        sampled = rng.choice(len(candidates_scored), size=sample_size, replace=False)
        raw_x = np.concatenate([positives_scored, candidates_scored[sampled]])[:, None]
        raw_y = np.concatenate([np.ones(len(positives_scored)), np.zeros(sample_size)]).astype(np.float32)
        scaler = StandardScaler().fit(raw_x)
        x = torch.as_tensor(scaler.transform(raw_x), dtype=torch.float32, device=device)
        y = torch.as_tensor(raw_y, dtype=torch.float32, device=device)
        model = ScoreMLP(64).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        for _ in range(inner_epochs):
            optimizer.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(model(x), y)
            loss.backward()
            optimizer.step()
        model.eval()
        for start in range(0, len(candidates_scored), prediction_batch_size):
            stop = min(start + prediction_batch_size, len(candidates_scored))
            batch = scaler.transform(candidates_scored[start:stop, None])
            with torch.no_grad():
                probability_sum[start:stop] += torch.sigmoid(
                    model(torch.as_tensor(batch, dtype=torch.float32, device=device))
                ).cpu().numpy()
    ranked_local = np.argsort(probability_sum / iterations)
    ranked_original = candidate_indices[ranked_local]
    return unlabeled[ranked_original], probability_sum[ranked_local] / iterations


def main() -> None:
    parser = argparse.ArgumentParser(description="Five-fold PU association-existence model adapted from 0323_existence_2.py.")
    parser.add_argument("--matrix", type=Path, default=Path("data/model/A_sign_matrix.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/two_step/existence_fold_metrics.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--transe-epochs", type=int, default=100)
    parser.add_argument("--bootstrap-iterations", type=int, default=150)
    parser.add_argument("--bootstrap-epochs", type=int, default=20)
    parser.add_argument("--gcn-epochs", type=int, default=200)
    parser.add_argument("--mlp-epochs", type=int, default=100)
    parser.add_argument("--candidate-cap", type=int, default=0, help="Development only; 0 uses all A_sign=0 pairs.")
    args = parser.parse_args()
    seed_everything(args.seed)

    matrix = load_signed_matrix(args.matrix).to_numpy(dtype=np.int8)
    n_microbes, n_diseases = matrix.shape
    positives = np.argwhere(matrix != 0).astype(np.int32)
    positive_sign = matrix[positives[:, 0], positives[:, 1]].astype(np.int8)
    unlabeled = np.argwhere(matrix == 0).astype(np.int32)
    if args.candidate_cap:
        rng = np.random.default_rng(args.seed)
        unlabeled = unlabeled[rng.choice(len(unlabeled), size=min(args.candidate_cap, len(unlabeled)), replace=False)]

    splitter = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    rows: list[dict[str, float | int]] = []
    for fold, (train_indices, validation_indices) in enumerate(splitter.split(positives), start=1):
        train_positive = positives[train_indices]
        validation_positive = positives[validation_indices]
        transe = fit_transe_bundle(
            train_positive,
            n_microbes,
            n_diseases,
            dimension=50,
            epochs=args.transe_epochs,
            batch_size=256,
            seed=args.seed + fold,
            labels=positive_sign[train_indices],
        )
        ranked_unlabeled, _ = reliable_negative_ranking(
            train_positive,
            unlabeled,
            transe,
            args.bootstrap_iterations,
            args.seed + fold,
            args.bootstrap_epochs,
        )
        needed = len(train_positive) + len(validation_positive)
        if len(ranked_unlabeled) < needed:
            raise RuntimeError(f"Fold {fold}: only {len(ranked_unlabeled)} scoreable U pairs; need {needed}")
        reliable_train = ranked_unlabeled[: len(train_positive)]
        reliable_validation = ranked_unlabeled[len(train_positive) : needed]

        gcn = fit_existence_gcn(
            transe.embeddings,
            train_positive,
            reliable_train,
            n_microbes,
            output_dim=64,
            epochs=args.gcn_epochs,
            seed=args.seed + fold,
        )
        topology = degree_features(np.vstack([train_positive, reliable_train]), n_microbes, n_diseases)
        node_features = np.hstack([transe.embeddings, gcn, topology]).astype(np.float32)
        train_pairs = np.vstack([train_positive, reliable_train])
        train_labels = np.concatenate([np.ones(len(train_positive)), np.zeros(len(reliable_train))])
        classifier = train_existence_mlp(
            concatenated_node_pair_features(train_pairs, node_features, n_microbes),
            train_labels,
            epochs=args.mlp_epochs,
            seed=args.seed + fold,
            learning_rate=0.001,
        )
        validation_pairs = np.vstack([validation_positive, reliable_validation])
        validation_labels = np.concatenate([np.ones(len(validation_positive)), np.zeros(len(reliable_validation))])
        probability = classifier.predict_proba(
            concatenated_node_pair_features(validation_pairs, node_features, n_microbes)
        )
        row = {"fold": fold, **binary_metrics(validation_labels, probability)}
        rows.append(row)
        print(row, flush=True)

    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.mean(numeric_only=True).to_dict())


if __name__ == "__main__":
    main()
