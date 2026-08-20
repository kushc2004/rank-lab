from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ranklab.retrieval.bpr import _torch_device


USER_NUMERIC = (
    "is_lowactive_period", "is_live_streamer", "is_video_author",
    "follow_user_num", "fans_user_num", "friend_user_num", "register_days",
)
ITEM_NUMERIC = ("video_duration", "server_width", "server_height")


def _numeric_matrix(frame: pd.DataFrame, columns: Iterable[str]) -> np.ndarray:
    present = [column for column in columns if column in frame]
    if not present:
        return np.zeros((len(frame), 1), dtype=np.float32)
    values = frame[present].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(np.float32)
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    return (values - mean) / np.where(std > 1e-6, std, 1.0)


@dataclass
class TwoTowerArtifacts:
    user_ids: np.ndarray
    item_ids: np.ndarray
    user_embeddings: np.ndarray
    item_embeddings: np.ndarray
    device: str
    metadata: dict | None = None

    def score(self, users: pd.Series, items: pd.Series) -> np.ndarray:
        user_map = {int(value): index for index, value in enumerate(self.user_ids)}
        item_map = {int(value): index for index, value in enumerate(self.item_ids)}
        scores = np.zeros(len(users), dtype=np.float32)
        for position, (user, item) in enumerate(zip(users, items)):
            u, i = user_map.get(int(user)), item_map.get(int(item))
            if u is not None and i is not None:
                scores[position] = self.user_embeddings[u] @ self.item_embeddings[i]
        return scores

    predict = score

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "user_ids.npy", self.user_ids)
        np.save(directory / "item_ids.npy", self.item_ids)
        np.save(directory / "user_eval_embeddings.npy", self.user_embeddings)
        np.save(directory / "item_embeddings.npy", self.item_embeddings)
        (directory / "training_metadata.json").write_text(
            json.dumps(self.metadata or {}, indent=2, sort_keys=True) + "\n"
        )

    @classmethod
    def load(cls, directory: str | Path) -> "TwoTowerArtifacts":
        directory = Path(directory)
        metadata_path = directory / "training_metadata.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
        return cls(
            np.load(directory / "user_ids.npy"),
            np.load(directory / "item_ids.npy"),
            np.load(directory / "user_eval_embeddings.npy"),
            np.load(directory / "item_embeddings.npy"),
            str(metadata.get("device", "loaded")),
            metadata,
        )


def _sample_explicit_negatives(
    train: pd.DataFrame,
    positive_pairs: pd.DataFrame,
    item_ids: np.ndarray,
    item_map: dict[int, int],
    strategy: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, int]]:
    aliases = {"mixed": "exposed_mixed"}
    strategy = aliases.get(strategy, strategy)
    allowed = {"random", "popularity", "exposed", "exposed_mixed"}
    if strategy not in allowed:
        raise ValueError(f"negative_strategy must be one of {sorted(allowed)}")
    required = {"user_id", "item_id", "timestamp_ms"}
    missing = required.difference(positive_pairs.columns)
    if missing:
        raise ValueError(f"positive_pairs missing timestamp contract columns: {sorted(missing)}")

    # User-local candidates and catalog popularity are both evaluated strictly
    # before each target timestamp.  Sampling from the prior-event list with
    # probability n/(n+|I|), otherwise uniformly from the catalog, is exactly
    # equivalent to add-one-smoothed historical item popularity without
    # materializing an O(|I|) probability vector for every target.
    user_events: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    ordered_events = train.sort_values(
        ["timestamp_ms", "user_id", "item_id"], kind="stable"
    )
    for user, frame in ordered_events.groupby("user_id", sort=False):
        user_events[int(user)] = (
            frame["timestamp_ms"].to_numpy(dtype=np.int64),
            frame["item_id"].to_numpy(dtype=np.int64),
            frame["long_view"].to_numpy(dtype=np.int8),
        )
    global_timestamps = ordered_events["timestamp_ms"].to_numpy(dtype=np.int64)
    global_item_indices = ordered_events["item_id"].map(item_map).to_numpy(dtype=np.int64)
    pair_order = np.argsort(
        positive_pairs["timestamp_ms"].to_numpy(dtype=np.int64), kind="stable"
    )
    negatives = np.empty(len(positive_pairs), dtype=np.int64)
    sources = {"exposed": 0, "popularity": 0, "random": 0}
    prior_event_count = 0
    for pair_index in pair_order:
        row = positive_pairs.iloc[int(pair_index)]
        user = int(row["user_id"])
        target_item = int(row["item_id"])
        target_timestamp = int(row["timestamp_ms"])
        prior_event_count = int(
            np.searchsorted(global_timestamps, target_timestamp, side="left")
        )
        timestamps, user_items, rewards = user_events[user]
        stop = int(np.searchsorted(timestamps, target_timestamp, side="left"))
        prior_items = user_items[:stop]
        prior_rewards = rewards[:stop]
        known_positive = set(prior_items[prior_rewards == 1].tolist())
        known_positive.add(target_item)
        exposed = np.unique(prior_items[prior_rewards == 0])
        exposed = np.asarray(
            [item for item in exposed if int(item) not in known_positive], dtype=np.int64
        )
        if strategy in {"exposed", "exposed_mixed"} and len(exposed):
            negative = int(rng.choice(exposed))
            source = "exposed"
        else:
            # Early positive events can legitimately have no strictly-prior
            # non-positive exposure.  Keep the exposed-negative ablation
            # runnable without looking forward: fall back to a uniform catalog
            # draw and expose the fallback count in training metadata.
            use_popularity = strategy in {"popularity", "exposed_mixed"}
            for _ in range(100):
                if use_popularity and prior_event_count and rng.random() < (
                    prior_event_count / (prior_event_count + len(item_ids))
                ):
                    sampled_index = int(
                        global_item_indices[rng.integers(prior_event_count)]
                    )
                    negative = int(item_ids[sampled_index])
                else:
                    negative = int(rng.choice(item_ids))
                if negative not in known_positive:
                    break
            else:
                available = np.asarray([item for item in item_ids if item not in known_positive])
                if not len(available):
                    raise ValueError(f"user {user} has no valid negative catalog item")
                negative = int(rng.choice(available))
            source = "popularity" if use_popularity else "random"
        negatives[int(pair_index)] = item_map[negative]
        sources[source] += 1
    return negatives, sources


def fit_two_tower(
    train: pd.DataFrame,
    users: pd.DataFrame,
    items: pd.DataFrame,
    embedding_dim: int = 64,
    epochs: int = 5,
    batch_size: int = 2048,
    learning_rate: float = 1e-3,
    temperature: float = 0.07,
    seed: int = 42,
    device: str = "auto",
    use_side_features: bool = False,
    history_length: int = 50,
    negative_strategy: str = "exposed_mixed",
    hard_negative_refresh: bool = True,
    hard_negative_epochs: int = 1,
) -> TwoTowerArtifacts:
    """Train a two-tower with in-batch softmax negatives.

    The official Pure side tables do not provide historical availability
    timestamps.  They are therefore disabled in the primary experiment and
    can only be enabled as an explicitly relaxed leakage-sensitivity run.
    """
    if train["is_random"].any():
        raise ValueError("Two-Tower training cannot contain randomized logs")
    if history_length < 1:
        raise ValueError("history_length must be positive")
    positive_events = (
        train.loc[
            train["long_view"].eq(1), ["user_id", "item_id", "timestamp_ms"]
        ]
        .sort_values(["user_id", "timestamp_ms", "item_id"], kind="stable")
        .reset_index(drop=True)
    )
    if positive_events.empty:
        raise ValueError("Two-Tower requires positive standard-log interactions")
    # One target per implicit user-item edge keeps the retrieval objective from
    # over-counting repeated consumption. The target time is the first positive
    # event; its user-history representation is built only from events strictly
    # before that timestamp.
    positives = (
        positive_events.drop_duplicates(["user_id", "item_id"], keep="first")
        .reset_index(drop=True)
    )
    torch, torch_device = _torch_device(device)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    user_ids = np.sort(train["user_id"].unique().astype(np.int64))
    item_ids = np.sort(train["item_id"].unique().astype(np.int64))
    user_map = {int(value): index for index, value in enumerate(user_ids)}
    item_map = {int(value): index for index, value in enumerate(item_ids)}
    pairs = np.asarray(
        [
            (user_map[int(row.user_id)], item_map[int(row.item_id)])
            for row in positives.itertuples(index=False)
        ],
        dtype=np.int64,
    )
    pair_timestamps = positives["timestamp_ms"].to_numpy(dtype=np.int64)
    history_events: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for user, frame in positive_events.groupby("user_id", sort=False):
        user_index = user_map[int(user)]
        history_events[user_index] = (
            frame["timestamp_ms"].to_numpy(dtype=np.int64),
            frame["item_id"].map(item_map).to_numpy(dtype=np.int64),
        )

    def histories_for_pairs(pair_indices: np.ndarray) -> np.ndarray:
        """Materialize bounded histories for one minibatch, not the full corpus."""
        histories = np.full((len(pair_indices), history_length), -1, dtype=np.int64)
        for output_index, pair_index in enumerate(pair_indices):
            user_index = int(pairs[pair_index, 0])
            timestamps, item_indices = history_events[user_index]
            stop = int(np.searchsorted(timestamps, pair_timestamps[pair_index], side="left"))
            start = max(0, stop - history_length)
            sequence = item_indices[start:stop]
            if len(sequence):
                histories[output_index, -len(sequence):] = sequence
        return histories

    # At validation/test/serving time the query timestamp is after the training
    # boundary, so the full strictly-training history is available.
    evaluation_histories = np.full(
        (len(user_ids), history_length), -1, dtype=np.int64
    )
    for user_index, (_, item_indices) in history_events.items():
        sequence = item_indices[-history_length:]
        evaluation_histories[user_index, -len(sequence):] = sequence
    explicit_negatives, negative_sources = _sample_explicit_negatives(
        train, positives[["user_id", "item_id", "timestamp_ms"]], item_ids, item_map,
        negative_strategy, rng
    )

    if use_side_features:
        user_side = users.set_index("user_id").reindex(user_ids)
        item_side = items.rename(columns={"video_id": "item_id"}).set_index("item_id").reindex(item_ids)
        user_values = _numeric_matrix(user_side, USER_NUMERIC)
        item_values = _numeric_matrix(item_side, ITEM_NUMERIC)
    else:
        user_values = np.zeros((len(user_ids), 1), dtype=np.float32)
        item_values = np.zeros((len(item_ids), 1), dtype=np.float32)
    user_meta = torch.as_tensor(user_values, device=torch_device)
    item_meta = torch.as_tensor(item_values, device=torch_device)

    class ItemTower(torch.nn.Module):
        def __init__(self, count: int, metadata_dim: int):
            super().__init__()
            self.identifier = torch.nn.Embedding(count, embedding_dim)
            self.metadata = torch.nn.Linear(metadata_dim, embedding_dim, bias=False)
            self.projection = torch.nn.Sequential(
                torch.nn.ReLU(), torch.nn.Linear(embedding_dim, embedding_dim)
            )

        def forward(self, indices, metadata):
            return torch.nn.functional.normalize(
                self.projection(self.identifier(indices) + self.metadata(metadata)), dim=1
            )

    class UserTower(torch.nn.Module):
        def __init__(self, count: int, metadata_dim: int, item_count: int):
            super().__init__()
            self.identifier = torch.nn.Embedding(count, embedding_dim)
            self.metadata = torch.nn.Linear(metadata_dim, embedding_dim, bias=False)
            self.history_items = torch.nn.Embedding(item_count, embedding_dim)
            self.projection = torch.nn.Sequential(
                torch.nn.ReLU(), torch.nn.Linear(embedding_dim, embedding_dim)
            )

        def forward(self, indices, metadata, histories):
            valid = histories.ge(0)
            safe_histories = histories.clamp_min(0)
            history_vectors = self.history_items(safe_histories)
            mask = valid.unsqueeze(-1).to(history_vectors.dtype)
            pooled_history = (history_vectors * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            combined = self.identifier(indices) + self.metadata(metadata) + pooled_history
            return torch.nn.functional.normalize(self.projection(combined), dim=1)

    user_tower = UserTower(
        len(user_ids), user_meta.shape[1], len(item_ids)
    ).to(torch_device)
    item_tower = ItemTower(len(item_ids), item_meta.shape[1]).to(torch_device)
    optimizer = torch.optim.AdamW(
        list(user_tower.parameters()) + list(item_tower.parameters()), lr=learning_rate
    )

    def train_epochs(epoch_count: int, negatives: np.ndarray) -> None:
        for _ in range(epoch_count):
            epoch_order = rng.permutation(len(pairs))
            for start in range(0, len(pairs), batch_size):
                indices = epoch_order[start : start + batch_size]
                batch = torch.as_tensor(pairs[indices], device=torch_device)
                negative = torch.as_tensor(negatives[indices], device=torch_device)
                histories = torch.as_tensor(
                    histories_for_pairs(indices), device=torch_device
                )
                u = user_tower(
                    batch[:, 0], user_meta[batch[:, 0]], histories
                )
                i = item_tower(batch[:, 1], item_meta[batch[:, 1]])
                n = item_tower(negative, item_meta[negative])
                logits = u @ i.T / temperature
                # Repeated positive items/users are multi-positive collisions,
                # not valid in-batch negatives. Mask them off the diagonal.
                diagonal = torch.eye(len(batch), dtype=torch.bool, device=torch_device)
                same_item = batch[:, 1, None].eq(batch[:, 1][None, :]) & ~diagonal
                same_user = batch[:, 0, None].eq(batch[:, 0][None, :]) & ~diagonal
                user_logits = logits.masked_fill(same_item, -1e9)
                item_logits = logits.T.masked_fill(same_user, -1e9)
                labels = torch.arange(len(batch), device=torch_device)
                contrastive = (
                    torch.nn.functional.cross_entropy(user_logits, labels)
                    + torch.nn.functional.cross_entropy(item_logits, labels)
                ) / 2
                pairwise = torch.nn.functional.softplus(
                    -((u * i).sum(dim=1) - (u * n).sum(dim=1)) / temperature
                ).mean()
                loss = contrastive + pairwise
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    train_epochs(epochs, explicit_negatives)

    hard_negative_count = 0
    if hard_negative_refresh and hard_negative_epochs > 0:
        with torch.no_grad():
            interim_items = item_tower(
                torch.arange(len(item_ids), device=torch_device), item_meta
            ).cpu().numpy()
        hard_negatives = explicit_negatives.copy()
        hard_candidate_k = min(200, len(item_ids))
        with torch.no_grad():
            for start in range(0, len(pairs), batch_size):
                indices = np.arange(start, min(start + batch_size, len(pairs)))
                batch = torch.as_tensor(pairs[indices], device=torch_device)
                histories = torch.as_tensor(
                    histories_for_pairs(indices), device=torch_device
                )
                pair_users = user_tower(
                    batch[:, 0], user_meta[batch[:, 0]], histories
                ).cpu().numpy()
                scores = pair_users @ interim_items.T
                if hard_candidate_k < len(item_ids):
                    top = np.argpartition(
                        -scores, hard_candidate_k - 1, axis=1
                    )[:, :hard_candidate_k]
                    top_scores = np.take_along_axis(scores, top, axis=1)
                    order = np.take_along_axis(
                        top,
                        np.argsort(-top_scores, axis=1, kind="stable"),
                        axis=1,
                    )
                else:
                    order = np.argsort(-scores, axis=1, kind="stable")
                for local_index, pair_index in enumerate(indices):
                    user_index = int(pairs[pair_index, 0])
                    target_index = int(pairs[pair_index, 1])
                    timestamps, positive_item_indices = history_events[user_index]
                    stop = int(
                        np.searchsorted(
                            timestamps, pair_timestamps[pair_index], side="left"
                        )
                    )
                    known = set(positive_item_indices[:stop].tolist())
                    known.add(target_index)
                    selected = next(
                        (
                            int(candidate)
                            for candidate in order[local_index]
                            if int(candidate) not in known
                        ),
                        None,
                    )
                    if selected is not None:
                        hard_negatives[pair_index] = selected
                        hard_negative_count += 1
        train_epochs(hard_negative_epochs, hard_negatives)
    with torch.no_grad():
        all_history_tensor = torch.as_tensor(
            evaluation_histories, device=torch_device
        )
        all_users = user_tower(
            torch.arange(len(user_ids), device=torch_device), user_meta,
            all_history_tensor,
        ).cpu().numpy()
        all_items = item_tower(
            torch.arange(len(item_ids), device=torch_device), item_meta
        ).cpu().numpy()
    metadata = {
        "device": str(torch_device),
        "embedding_dim": int(embedding_dim),
        "epochs": int(epochs),
        "temperature": float(temperature),
        "history_length": int(history_length),
        "history_representation": "mean_pool_last_n_positive_item_embeddings",
        "training_history_contract": "strictly_earlier_than_target_timestamp",
        "evaluation_history_contract": "positive_events_from_standard_training_only",
        "history_positive_events": int(len(positive_events)),
        "negative_strategy": negative_strategy,
        "negative_sources": negative_sources,
        "negative_sampling_contract": "catalog popularity and exposed negatives are strictly before each target timestamp",
        "hard_negative_refresh": bool(hard_negative_refresh),
        "hard_negative_epochs": int(hard_negative_epochs),
        "hard_negative_pairs": hard_negative_count,
        "hard_negative_contract": "one offline top-200 refresh with strict pre-target histories and positives",
        "positive_pairs": int(len(pairs)),
        "side_features": bool(use_side_features),
    }
    return TwoTowerArtifacts(user_ids, item_ids, all_users, all_items, str(torch_device), metadata)
