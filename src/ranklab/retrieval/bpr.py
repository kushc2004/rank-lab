from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class BPRModel:
    user_index: dict[int, int]
    item_index: dict[int, int]
    user_factors: np.ndarray
    item_factors: np.ndarray
    fallback: float

    def predict(self, users: pd.Series, items: pd.Series) -> np.ndarray:
        values = np.full(len(users), self.fallback, dtype=float)
        for position, (user, item) in enumerate(zip(users, items)):
            u, i = self.user_index.get(int(user)), self.item_index.get(int(item))
            if u is not None and i is not None:
                values[position] = float(self.user_factors[u] @ self.item_factors[i])
        return values

def _torch_device(requested: str):
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "BPR-MF requires PyTorch. Reinstall the project with `python -m pip install -e '.[dev]'`."
        ) from error
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS was requested but is unavailable. Use a PyTorch macOS build with MPS support, "
                "or explicitly set device: cpu."
            )
        return torch, torch.device("mps")
    if requested == "cpu":
        return torch, torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is unavailable. Enable a Kaggle GPU accelerator, "
                "or explicitly set device: cpu."
            )
        return torch, torch.device("cuda")
    if requested == "auto":
        return torch, torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    raise ValueError("device must be one of: mps, cuda, cpu, auto")


def fit_bpr(
    train: pd.DataFrame,
    embedding_dim: int = 32,
    epochs: int = 12,
    learning_rate: float = .05,
    regularization: float = .002,
    seed: int = 42,
    device: str = "mps",
    batch_size: int = 8192,
) -> BPRModel:
    if train.is_random.any():
        raise ValueError("BPR may train only on standard logs")
    train = train.sort_values("timestamp_ms", kind="stable")
    positives = train.loc[train.long_view.eq(1), ["user_id", "item_id", "timestamp_ms"]]
    if positives.empty:
        raise ValueError("BPR requires at least one standard-training long_view")
    users = sorted(train.user_id.unique())
    items = sorted(train.item_id.unique())
    user_index, item_index = {v: n for n, v in enumerate(users)}, {v: n for n, v in enumerate(items)}
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    rng = np.random.default_rng(seed)
    # Build examples in event time. A negative must come from prior exposure;
    # never use an item that is a known positive for the user, including a
    # positive recorded later in the training window.
    known_positive = train.loc[train.long_view.eq(1)].groupby("user_id").item_id.apply(set).to_dict()
    prior_negative: dict[int, list[int]] = {}
    triples = []

    def choose_negative(user_id: int, positive_item: int, forbidden: set[int]) -> int | None:
        """Sample a valid temporal negative without materializing a full pool per event."""
        prior = prior_negative.get(user_id, [])
        if prior:
            # Duplicates preserve the observed exposure distribution. Bounded
            # rejection keeps large histories from making preprocessing quadratic.
            for _ in range(min(64, len(prior))):
                candidate = int(prior[int(rng.integers(len(prior)))])
                if candidate not in forbidden:
                    return candidate
            for candidate in dict.fromkeys(prior):
                if candidate not in forbidden:
                    return int(candidate)
        for _ in range(64):
            candidate = int(items[int(rng.integers(len(items)))])
            if candidate not in forbidden and candidate != positive_item:
                return candidate
        # This is only reached for an unusually dense positive history. It
        # preserves the exact exclusion rule rather than accepting a bad negative.
        for candidate in items:
            if candidate not in forbidden and candidate != positive_item:
                return int(candidate)
        return None

    for row in train.sort_values(["timestamp_ms", "user_id", "item_id"], kind="stable").itertuples(index=False):
        user_id, item_id = int(row.user_id), int(row.item_id)
        if int(row.long_view) == 0:
            prior_negative.setdefault(user_id, []).append(item_id)
            continue
        forbidden = known_positive.get(user_id, set())
        negative_item = choose_negative(user_id, item_id, forbidden)
        if negative_item is not None:
            triples.append((user_index[user_id], item_index[item_id], item_index[negative_item]))
    if not triples:
        raise ValueError("No valid BPR triples after temporal negative filtering")
    torch, torch_device = _torch_device(device)
    torch.manual_seed(seed)
    triples_array = np.asarray(triples, dtype=np.int64)
    triples_tensor = torch.as_tensor(triples_array, device=torch_device)
    user_factors = torch.nn.Parameter(
        torch.randn(len(users), embedding_dim, device=torch_device) * .05
    )
    item_factors = torch.nn.Parameter(
        torch.randn(len(items), embedding_dim, device=torch_device) * .05
    )
    optimizer = torch.optim.Adam((user_factors, item_factors), lr=learning_rate)
    for _ in range(epochs):
        epoch_indices = rng.permutation(len(triples_array))
        for start in range(0, len(triples_array), batch_size):
            batch_indices = torch.as_tensor(
                epoch_indices[start : start + batch_size], device=torch_device
            )
            batch = triples_tensor[batch_indices]
            user_vectors = user_factors[batch[:, 0]]
            positive_vectors = item_factors[batch[:, 1]]
            negative_vectors = item_factors[batch[:, 2]]
            score_difference = (user_vectors * (positive_vectors - negative_vectors)).sum(dim=1)
            ranking_loss = -torch.nn.functional.logsigmoid(score_difference).mean()
            penalty = regularization * (
                user_vectors.square().mean()
                + positive_vectors.square().mean()
                + negative_vectors.square().mean()
            )
            optimizer.zero_grad()
            (ranking_loss + penalty).backward()
            optimizer.step()
    return BPRModel(
        user_index,
        item_index,
        user_factors.detach().cpu().numpy(),
        item_factors.detach().cpu().numpy(),
        fallback=float(train.long_view.mean()),
    )
