from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier

from src.common import load_signed_matrix


LABEL_ORDER = np.array([-1, 0, 1], dtype=np.int8)
TO_MODEL = {-1: 0, 0: 1, 1: 2}


def sample_pairs(matrix: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    known = np.argwhere(matrix != 0)
    unknown = np.argwhere(matrix == 0)
    zero = unknown[rng.choice(len(unknown), size=len(known), replace=False)]
    samples = np.vstack(
        [
            np.column_stack([known, matrix[known[:, 0], known[:, 1]]]),
            np.column_stack([zero, np.zeros(len(zero), dtype=np.int8)]),
        ]
    ).astype(np.int32)
    rng.shuffle(samples)
    return samples


def pair_features(samples: np.ndarray, microbe: np.ndarray, disease: np.ndarray) -> np.ndarray:
    return np.hstack([microbe[samples[:, 0]], disease[samples[:, 1]]]).astype(np.float32)


def model(seed: int) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=120,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )


def metrics(y_sign: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    y_bin = label_binarize(y_sign, classes=LABEL_ORDER)
    pred_sign = LABEL_ORDER[np.argmax(probabilities, axis=1)]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_sign, pred_sign, labels=LABEL_ORDER, average="macro", zero_division=0
    )
    result = {
        "micro_auc": float(roc_auc_score(y_bin, probabilities, average="micro")),
        "macro_auc": float(roc_auc_score(y_bin, probabilities, average="macro")),
        "micro_aupr": float(average_precision_score(y_bin, probabilities, average="micro")),
        "macro_aupr": float(average_precision_score(y_bin, probabilities, average="macro")),
        "accuracy": float(accuracy_score(y_sign, pred_sign)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }
    for index, label in enumerate(LABEL_ORDER):
        result[f"auc_{int(label):+d}"] = float(roc_auc_score(y_bin[:, index], probabilities[:, index]))
        result[f"aupr_{int(label):+d}"] = float(average_precision_score(y_bin[:, index], probabilities[:, index]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the one-step three-class XGBoost model.")
    parser.add_argument("--matrix", type=Path, default=Path("data/model/A_sign_matrix.csv"))
    parser.add_argument("--microbe-features", type=Path, default=Path("data/features/one_step_microbe_features_402.csv"))
    parser.add_argument("--disease-features", type=Path, default=Path("data/features/one_step_disease_features_391.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/one_step/evaluation"))
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--cv-folds", type=int, default=10)
    args = parser.parse_args()

    matrix_frame = load_signed_matrix(args.matrix)
    matrix = matrix_frame.to_numpy(dtype=np.int8)
    microbe_frame = pd.read_csv(args.microbe_features).set_index("microbe")
    disease_frame = pd.read_csv(args.disease_features).set_index("disease")
    microbe = microbe_frame.loc[matrix_frame.index.astype(str)].to_numpy(dtype=np.float32)
    disease = disease_frame.loc[matrix_frame.columns.astype(str)].to_numpy(dtype=np.float32)
    if microbe.shape != (matrix.shape[0], 402) or disease.shape != (matrix.shape[1], 391):
        raise AssertionError(f"Unexpected feature shapes: {microbe.shape}, {disease.shape}")
    samples = sample_pairs(matrix, args.seed)
    labels = samples[:, 2].astype(np.int8)
    train_idx, test_idx = train_test_split(
        np.arange(len(samples)), test_size=0.2, random_state=args.seed, stratify=labels
    )
    classifier = model(args.seed)
    classifier.fit(
        pair_features(samples[train_idx], microbe, disease),
        np.array([TO_MODEL[int(value)] for value in labels[train_idx]], dtype=np.int8),
    )
    holdout = metrics(labels[test_idx], classifier.predict_proba(pair_features(samples[test_idx], microbe, disease)))
    holdout.update(
        {
            "total_samples": int(len(samples)),
            "training_samples": int(len(train_idx)),
            "test_samples": int(len(test_idx)),
            "training_pseudo_zero_samples": int(np.count_nonzero(labels[train_idx] == 0)),
        }
    )

    folds = []
    splitter = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    for fold, (fit_idx, validation_idx) in enumerate(splitter.split(samples, labels), start=1):
        classifier = model(args.seed + fold)
        classifier.fit(
            pair_features(samples[fit_idx], microbe, disease),
            np.array([TO_MODEL[int(value)] for value in labels[fit_idx]], dtype=np.int8),
        )
        row = {"fold": fold, **metrics(labels[validation_idx], classifier.predict_proba(pair_features(samples[validation_idx], microbe, disease)))}
        folds.append(row)
    fold_frame = pd.DataFrame(folds)
    summary = {
        column: {"mean": float(fold_frame[column].mean()), "std": float(fold_frame[column].std(ddof=0))}
        for column in fold_frame.columns
        if column != "fold"
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "holdout_metrics.json").write_text(json.dumps(holdout, indent=2), encoding="utf-8")
    fold_frame.to_csv(args.output / "cross_validation_fold_metrics.csv", index=False)
    (args.output / "cross_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"holdout": holdout, "cross_validation": summary}, indent=2))


if __name__ == "__main__":
    main()
