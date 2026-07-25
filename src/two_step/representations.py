from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch import nn
from torch_geometric.nn import GCNConv


def entity_name(index: int, node_type: str) -> str:
    return f"{node_type}::{index}"


def labeled_triples(pairs: np.ndarray, labels: np.ndarray | None = None) -> np.ndarray:
    if labels is None:
        relations = np.repeat("ASSOCIATED", len(pairs))
    else:
        relations = np.where(labels > 0, "INCREASE", "DECREASE")
    return np.column_stack(
        [
            [entity_name(int(index), "microbe") for index in pairs[:, 0]],
            relations,
            [entity_name(int(index), "disease") for index in pairs[:, 1]],
        ]
    ).astype(str)


def fit_transe(
    pairs: np.ndarray,
    n_microbes: int,
    n_diseases: int,
    dimension: int,
    epochs: int,
    batch_size: int,
    seed: int,
    labels: np.ndarray | None = None,
) -> np.ndarray:
    return fit_transe_bundle(
        pairs,
        n_microbes,
        n_diseases,
        dimension,
        epochs,
        batch_size,
        seed,
        labels,
    ).embeddings


def graph_edges(pairs: np.ndarray, n_microbes: int) -> torch.Tensor:
    left = torch.as_tensor(pairs[:, 0], dtype=torch.long)
    right = torch.as_tensor(n_microbes + pairs[:, 1], dtype=torch.long)
    return torch.stack([torch.cat([left, right]), torch.cat([right, left])])


class GCNEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.conv1 = GCNConv(input_dim, output_dim * 2)
        self.conv2 = GCNConv(output_dim * 2, output_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.conv2(torch.relu(self.conv1(x, edge_index)), edge_index)


def _pair_dot(z: torch.Tensor, pairs: torch.Tensor, n_microbes: int) -> torch.Tensor:
    left = z[pairs[:, 0]]
    right = z[n_microbes + pairs[:, 1]]
    return torch.sum(left * right, dim=1)


def fit_existence_gcn(
    initial: np.ndarray,
    positive_pairs: np.ndarray,
    negative_pairs: np.ndarray,
    n_microbes: int,
    output_dim: int,
    epochs: int,
    seed: int,
) -> np.ndarray:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.as_tensor(initial, dtype=torch.float32, device=device)
    edge_index = graph_edges(positive_pairs, n_microbes).to(device)
    pairs = torch.as_tensor(np.vstack([positive_pairs, negative_pairs]), dtype=torch.long, device=device)
    labels = torch.cat(
        [torch.ones(len(positive_pairs), device=device), torch.zeros(len(negative_pairs), device=device)]
    )
    encoder = GCNEncoder(initial.shape[1], output_dim).to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)
    for _ in range(epochs):
        optimizer.zero_grad()
        embedding = encoder(x, edge_index)
        loss = nn.functional.binary_cross_entropy_with_logits(_pair_dot(embedding, pairs, n_microbes), labels)
        loss.backward()
        optimizer.step()
    encoder.eval()
    with torch.no_grad():
        return encoder(x, edge_index).cpu().numpy().astype(np.float32)


def fit_direction_gcn(
    initial: np.ndarray,
    pairs: np.ndarray,
    labels_sign: np.ndarray,
    n_microbes: int,
    output_dim: int,
    epochs: int,
    seed: int,
) -> np.ndarray:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.as_tensor(initial, dtype=torch.float32, device=device)
    edge_index = graph_edges(pairs, n_microbes).to(device)
    pair_tensor = torch.as_tensor(pairs, dtype=torch.long, device=device)
    labels = torch.as_tensor(labels_sign > 0, dtype=torch.float32, device=device)
    encoder = GCNEncoder(initial.shape[1], output_dim).to(device)
    head = nn.Linear(output_dim, 1).to(device)
    optimizer = torch.optim.Adam([*encoder.parameters(), *head.parameters()], lr=0.001)
    for _ in range(epochs):
        optimizer.zero_grad()
        embedding = encoder(x, edge_index)
        difference = torch.abs(
            embedding[pair_tensor[:, 0]] - embedding[n_microbes + pair_tensor[:, 1]]
        )
        logits = head(difference).squeeze(1)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()
    encoder.eval()
    with torch.no_grad():
        return encoder(x, edge_index).cpu().numpy().astype(np.float32)


def degree_features(pairs: np.ndarray, n_microbes: int, n_diseases: int) -> np.ndarray:
    out_degree = np.bincount(pairs[:, 0], minlength=n_microbes).astype(np.float32)
    in_degree = np.bincount(pairs[:, 1], minlength=n_diseases).astype(np.float32)
    features = np.zeros((n_microbes + n_diseases, 2), dtype=np.float32)
    features[:n_microbes, 1] = np.log1p(out_degree)
    features[n_microbes:, 0] = np.log1p(in_degree)
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    return (features - mean) / np.where(std > 0, std, 1.0)


def concatenated_node_pair_features(
    pairs: np.ndarray, node_features: np.ndarray, n_microbes: int
) -> np.ndarray:
    return np.hstack(
        [node_features[pairs[:, 0]], node_features[n_microbes + pairs[:, 1]]]
    ).astype(np.float32)


def difference_pair_features(
    pairs: np.ndarray,
    transe: np.ndarray,
    gcn: np.ndarray,
    n_microbes: int,
) -> np.ndarray:
    return np.hstack(
        [
            np.abs(transe[pairs[:, 0]] - transe[n_microbes + pairs[:, 1]]),
            np.abs(gcn[pairs[:, 0]] - gcn[n_microbes + pairs[:, 1]]),
        ]
    ).astype(np.float32)


def binary_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = (probability >= 0.5).astype(np.int8)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, prediction, average="binary", zero_division=0
    )
    return {
        "auroc": float(roc_auc_score(y_true, probability)),
        "aupr": float(average_precision_score(y_true, probability)),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


class ExistenceMLP(nn.Module):
    def __init__(self, input_dim: int = 232):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)


@dataclass
class TrainedExistenceMLP:
    model: ExistenceMLP
    device: torch.device

    def predict_proba(self, features: np.ndarray, batch_size: int = 8192) -> np.ndarray:
        values = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(features), batch_size):
                batch = torch.as_tensor(features[start : start + batch_size], dtype=torch.float32, device=self.device)
                values.append(torch.sigmoid(self.model(batch)).cpu().numpy())
        return np.concatenate(values)


def train_existence_mlp(
    features: np.ndarray,
    labels: np.ndarray,
    epochs: int,
    seed: int,
    learning_rate: float = 0.001,
    batch_size: int = 256,
) -> TrainedExistenceMLP:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ExistenceMLP(features.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    dataset = torch.utils.data.TensorDataset(
        torch.as_tensor(features, dtype=torch.float32), torch.as_tensor(labels, dtype=torch.float32)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            optimizer.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(model(x.to(device)), y.to(device))
            loss.backward()
            optimizer.step()
    return TrainedExistenceMLP(model=model, device=device)


@dataclass
class TrainedTransE:
    """Fold-local PyKEEN model plus embeddings aligned to the repository node order."""

    embeddings: np.ndarray
    model: object
    factory: TriplesFactory
    n_microbes: int

    def score_pairs(
        self,
        pairs: np.ndarray,
        relation: str = "INCREASE",
        batch_size: int = 8192,
    ) -> np.ndarray:
        relation_name = relation if relation in self.factory.relation_to_id else next(iter(self.factory.relation_to_id))
        relation_id = self.factory.relation_to_id[relation_name]
        scores = np.full(len(pairs), np.nan, dtype=np.float32)
        valid_rows: list[int] = []
        encoded: list[tuple[int, int, int]] = []
        for row_index, (microbe_index, disease_index) in enumerate(pairs):
            head = entity_name(int(microbe_index), "microbe")
            tail = entity_name(int(disease_index), "disease")
            if head not in self.factory.entity_to_id or tail not in self.factory.entity_to_id:
                continue
            valid_rows.append(row_index)
            encoded.append((self.factory.entity_to_id[head], relation_id, self.factory.entity_to_id[tail]))
        for start in range(0, len(encoded), batch_size):
            stop = min(start + batch_size, len(encoded))
            batch = torch.as_tensor(encoded[start:stop], dtype=torch.long, device=self.model.device)
            with torch.no_grad():
                values = self.model.score_hrt(batch).detach().cpu().numpy().reshape(-1)
            scores[np.asarray(valid_rows[start:stop], dtype=np.int64)] = values.astype(np.float32)
        return scores


def fit_transe_bundle(
    pairs: np.ndarray,
    n_microbes: int,
    n_diseases: int,
    dimension: int,
    epochs: int,
    batch_size: int,
    seed: int,
    labels: np.ndarray | None = None,
) -> TrainedTransE:
    triples = labeled_triples(pairs, labels)
    factory = TriplesFactory.from_labeled_triples(triples)
    result = pipeline(
        training=factory,
        model="TransE",
        model_kwargs={"embedding_dim": dimension},
        training_loop="sLCWA",
        training_kwargs={"num_epochs": epochs, "batch_size": batch_size, "use_tqdm_batch": False},
        random_seed=seed,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    representation = result.model.entity_representations[0]
    ids = torch.arange(factory.num_entities, device=result.model.device)
    learned = representation(indices=ids).detach().cpu().numpy().astype(np.float32)
    full = np.zeros((n_microbes + n_diseases, dimension), dtype=np.float32)
    for index in range(n_microbes):
        key = entity_name(index, "microbe")
        if key in factory.entity_to_id:
            full[index] = learned[factory.entity_to_id[key]]
    for index in range(n_diseases):
        key = entity_name(index, "disease")
        if key in factory.entity_to_id:
            full[n_microbes + index] = learned[factory.entity_to_id[key]]
    return TrainedTransE(full, result.model, factory, n_microbes)
