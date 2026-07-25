from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.common import load_signed_matrix, seed_everything
from src.two_step.representations import (
    binary_metrics,
    concatenated_node_pair_features,
    difference_pair_features,
    fit_direction_gcn,
    fit_transe,
)


class DirectionMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(1)


class TorchMLPClassifier:
    def __init__(self, seed: int, epochs: int = 50, batch_size: int = 64, learning_rate: float = 0.001):
        self.seed = seed
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model: DirectionMLP | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TorchMLPClassifier":
        torch.manual_seed(self.seed)
        self.model = DirectionMLP(x.shape[1]).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        dataset = torch.utils.data.TensorDataset(
            torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.model.train()
        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                logits = self.model(batch_x.to(self.device))
                loss = nn.functional.binary_cross_entropy_with_logits(logits, batch_y.to(self.device))
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Classifier is not fitted")
        self.model.eval()
        with torch.no_grad():
            positive = torch.sigmoid(self.model(torch.as_tensor(x, dtype=torch.float32, device=self.device))).cpu().numpy()
        return np.column_stack([1.0 - positive, positive])


def classifiers(seed: int, y_train: np.ndarray):
    positives = max(int(np.sum(y_train == 1)), 1)
    negatives = max(int(np.sum(y_train == 0)), 1)
    return {
        "MLP": TorchMLPClassifier(seed=seed),
        "RandomForest": RandomForestClassifier(
            n_estimators=500, class_weight="balanced", n_jobs=-1, random_state=seed
        ),
        "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed)),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
            scale_pos_weight=negatives / positives,
            verbosity=0,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fold-aware increase/decrease comparison adapted from type4/1.1.")
    parser.add_argument("--matrix", type=Path, default=Path("data/model/A_sign_matrix.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/two_step/direction_fold_metrics.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--transe-epochs", type=int, default=100)
    parser.add_argument("--gcn-epochs", type=int, default=200)
    parser.add_argument(
        "--feature-mode",
        choices=("actual-kge", "manuscript-kge-gcn"),
        default="actual-kge",
        help="actual-kge reproduces the executed type4/1.1 path; manuscript-kge-gcn adds the GCN block described in G8.",
    )
    args = parser.parse_args()
    seed_everything(args.seed)

    matrix = load_signed_matrix(args.matrix).to_numpy(dtype=np.int8)
    n_microbes, n_diseases = matrix.shape
    pairs = np.argwhere(matrix != 0).astype(np.int32)
    labels_sign = matrix[pairs[:, 0], pairs[:, 1]].astype(np.int8)
    labels_binary = (labels_sign > 0).astype(np.int8)
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    rows = []

    for fold, (train_indices, test_indices) in enumerate(splitter.split(pairs, labels_binary), start=1):
        train_pairs, test_pairs = pairs[train_indices], pairs[test_indices]
        train_sign = labels_sign[train_indices]
        transe = fit_transe(
            train_pairs,
            n_microbes,
            n_diseases,
            dimension=128,
            epochs=args.transe_epochs,
            batch_size=256,
            seed=args.seed + fold,
            labels=train_sign,
        )
        if args.feature_mode == "actual-kge":
            x_train = concatenated_node_pair_features(train_pairs, transe, n_microbes)
            x_test = concatenated_node_pair_features(test_pairs, transe, n_microbes)
        else:
            gcn = fit_direction_gcn(
                transe,
                train_pairs,
                train_sign,
                n_microbes,
                output_dim=128,
                epochs=args.gcn_epochs,
                seed=args.seed + fold,
            )
            x_train = difference_pair_features(train_pairs, transe, gcn, n_microbes)
            x_test = difference_pair_features(test_pairs, transe, gcn, n_microbes)
        for name, classifier in classifiers(args.seed + fold, labels_binary[train_indices]).items():
            classifier.fit(x_train, labels_binary[train_indices])
            probability = classifier.predict_proba(x_test)[:, 1]
            rows.append(
                {
                    "fold": fold,
                    "classifier": name,
                    "feature_mode": args.feature_mode,
                    **binary_metrics(labels_binary[test_indices], probability),
                }
            )
            print(rows[-1], flush=True)

    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(output.groupby("classifier").mean(numeric_only=True).to_string())


if __name__ == "__main__":
    main()
