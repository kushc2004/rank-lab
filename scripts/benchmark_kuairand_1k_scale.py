#!/usr/bin/env python3
"""Benchmark exact and ANN retrieval indexes on the real KuaiRand-1K catalog.

This is an index-scale experiment, separate from the KuaiRand-Pure exposure
experiment.  Item vectors are deterministic feature-hashes of the official
``video_features_basic_1k.csv`` fields; query vectors are mean profiles of a
user's earlier *standard-log* long-viewed items.  Consequently the report
measures exact-versus-HNSW index recall and latency on the real 1K item corpus;
it must not be read as a trained recommender-quality result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ranklab.data.kuairand_1k import OFFICIAL_1K_FILES, raw_paths


def _bucket_and_sign(column: str, value: object, dimension: int) -> tuple[int, float]:
    encoded = f"{column}={value}".encode("utf-8", "surrogatepass")
    digest = hashlib.blake2b(encoded, digest_size=8).digest()
    number = int.from_bytes(digest, "little", signed=False)
    return number % dimension, 1.0 if number >> 63 == 0 else -1.0


def _embed_chunk(frame: pd.DataFrame, dimension: int) -> np.ndarray:
    """Produce stable, normalised feature-hash vectors without learned weights."""
    vectors = np.zeros((len(frame), dimension), dtype=np.float32)
    for column in frame.columns:
        if column == "video_id":
            continue
        codes, values = pd.factorize(frame[column].fillna("__NA__").astype(str), sort=False)
        buckets = np.empty(len(values), dtype=np.intp)
        signs = np.empty(len(values), dtype=np.float32)
        for index, value in enumerate(values):
            buckets[index], signs[index] = _bucket_and_sign(column, value, dimension)
        np.add.at(vectors, (np.arange(len(frame)), buckets[codes]), signs[codes])
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    return vectors


def _catalog(paths: dict[str, Path], dimension: int, chunksize: int) -> tuple[np.ndarray, np.ndarray]:
    item_path = paths["items_basic"]
    header = pd.read_csv(item_path, nrows=0)
    if "video_id" not in header.columns:
        raise ValueError(f"{item_path} has no official video_id column")
    ids: list[np.ndarray] = []
    vectors: list[np.ndarray] = []
    for number, chunk in enumerate(pd.read_csv(item_path, chunksize=chunksize), start=1):
        item_ids = pd.to_numeric(chunk.pop("video_id"), errors="raise").to_numpy(dtype=np.int64)
        if len(np.unique(item_ids)) != len(item_ids):
            raise ValueError(f"duplicate video_id values within basic-feature chunk {number}")
        ids.append(item_ids)
        vectors.append(_embed_chunk(chunk, dimension))
        print(f"[catalog] chunks={number} items={sum(len(part) for part in ids):,}", flush=True)
    catalog_ids = np.concatenate(ids)
    catalog_vectors = np.ascontiguousarray(np.concatenate(vectors), dtype=np.float32)
    if len(np.unique(catalog_ids)) != len(catalog_ids):
        raise ValueError("video_features_basic_1k.csv has duplicate video_id values")
    return catalog_ids, catalog_vectors


def _queries(
    paths: dict[str, Path], ids: np.ndarray, vectors: np.ndarray, chunksize: int, maximum: int
) -> tuple[np.ndarray, dict[str, int]]:
    """Build profiles from earlier standard logs only, never randomized feedback."""
    order = np.argsort(ids)
    sorted_ids = ids[order]
    profiles: dict[int, np.ndarray] = {}
    positives: dict[int, int] = {}
    interaction_path = paths["standard_early"]
    usecols = ["user_id", "video_id", "long_view", "is_rand"]
    for chunk in pd.read_csv(interaction_path, usecols=usecols, chunksize=chunksize):
        if not pd.to_numeric(chunk["is_rand"], errors="raise").eq(0).all():
            raise ValueError("standard_early 1K log contains non-standard interactions")
        positive = chunk.loc[pd.to_numeric(chunk["long_view"], errors="raise").eq(1)]
        if positive.empty:
            continue
        video_ids = pd.to_numeric(positive["video_id"], errors="raise").to_numpy(dtype=np.int64)
        positions = np.searchsorted(sorted_ids, video_ids)
        valid = positions < len(sorted_ids)
        candidate_rows = np.flatnonzero(valid)
        valid[candidate_rows] = (
            sorted_ids[positions[candidate_rows]] == video_ids[candidate_rows]
        )
        for user_id, position in zip(
            pd.to_numeric(positive.loc[valid, "user_id"], errors="raise").to_numpy(dtype=np.int64),
            order[positions[valid]],
        ):
            user = int(user_id)
            if user not in profiles:
                profiles[user] = vectors[position].copy()
                positives[user] = 1
            else:
                profiles[user] += vectors[position]
                positives[user] += 1
    ranked = sorted(profiles, key=lambda user: (-positives[user], user))[:maximum]
    query_vectors = np.vstack([profiles[user] / max(positives[user], 1) for user in ranked]).astype(np.float32)
    query_vectors /= np.maximum(np.linalg.norm(query_vectors, axis=1, keepdims=True), 1e-12)
    return np.ascontiguousarray(query_vectors), {
        "users_with_catalog_matched_standard_positive": len(profiles),
        "selected_queries": len(ranked),
        "standard_positive_events_for_selected_users": int(sum(positives[user] for user in ranked)),
    }


def _latency(index: object, queries: np.ndarray, k: int) -> dict[str, float]:
    samples = []
    for query in queries:
        started = time.perf_counter()
        index.search(query[None, :], k)
        samples.append((time.perf_counter() - started) * 1000)
    return {
        "p50_ms": float(np.percentile(samples, 50)),
        "p95_ms": float(np.percentile(samples, 95)),
        "mean_ms": float(np.mean(samples)),
    }


def _index_bytes(faiss: object, index: object) -> int:
    with tempfile.TemporaryDirectory(prefix="ranklab-faiss-") as directory:
        path = Path(directory) / "index.faiss"
        faiss.write_index(index, str(path))
        return path.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/metrics/kuairand_1k_scale.json"))
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--query-count", type=int, default=100)
    parser.add_argument("--candidate-k", type=int, default=200)
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=80)
    parser.add_argument("--ef-search", type=int, nargs="+", default=[16, 32, 64, 128])
    args = parser.parse_args()
    if min(args.dimension, args.query_count, args.candidate_k, args.chunksize, args.hnsw_m, args.ef_construction) < 1:
        raise SystemExit("all numeric arguments must be positive")
    paths = raw_paths(args.raw_dir)
    try:
        import faiss
    except ModuleNotFoundError as error:
        raise SystemExit("faiss-cpu is required; install the project with `.[full]`") from error

    started = time.perf_counter()
    ids, vectors = _catalog(paths, args.dimension, args.chunksize)
    queries, query_summary = _queries(paths, ids, vectors, args.chunksize, args.query_count)
    if not len(queries):
        raise SystemExit("no standard-log long-view user profiles matched the item catalog")
    print(f"[queries] {len(queries):,} real standard-log user profiles", flush=True)

    exact_started = time.perf_counter()
    exact = faiss.IndexFlatIP(args.dimension)
    exact.add(vectors)
    exact_build_ms = (time.perf_counter() - exact_started) * 1000
    exact_ids = exact.search(queries, args.candidate_k)[1]
    result: dict[str, object] = {
        "experiment": "kuairand_1k_real_catalog_index_scale",
        "claim_boundary": "Index recall/latency only. Item vectors are deterministic official-metadata feature hashes and queries are standard-log historical profiles; this is not trained recommender-quality evaluation.",
        "raw_files": {key: path.name for key, path in paths.items()},
        "catalog": {"items": int(len(ids)), "embedding_dimension": args.dimension, "vector_bytes": int(vectors.nbytes)},
        "queries": query_summary,
        "candidate_k": args.candidate_k,
        "exact": {"backend": "faiss.IndexFlatIP", "build_ms": float(exact_build_ms), "index_bytes": _index_bytes(faiss, exact), **_latency(exact, queries, args.candidate_k)},
        "runtime": {"python": platform.python_version(), "faiss_version": getattr(faiss, "__version__", "unknown"), "faiss_gpu_count": int(getattr(faiss, "get_num_gpus", lambda: 0)())},
    }
    ann = faiss.IndexHNSWFlat(args.dimension, args.hnsw_m, faiss.METRIC_INNER_PRODUCT)
    ann.hnsw.efConstruction = args.ef_construction
    ann_started = time.perf_counter()
    ann.add(vectors)
    tradeoff = []
    for ef_search in args.ef_search:
        ann.hnsw.efSearch = int(ef_search)
        observed = ann.search(queries, args.candidate_k)[1]
        recall = np.mean([len(set(expected) & set(actual)) / args.candidate_k for expected, actual in zip(exact_ids, observed)])
        tradeoff.append({"ef_search": int(ef_search), "recall_at_k_vs_exact": float(recall), **_latency(ann, queries, args.candidate_k)})
        print(f"[hnsw] efSearch={ef_search} recall@{args.candidate_k}={recall:.4f}", flush=True)
    result["hnsw"] = {"backend": "faiss.IndexHNSWFlat", "M": args.hnsw_m, "ef_construction": args.ef_construction, "build_ms": float((time.perf_counter() - ann_started) * 1000), "index_bytes": _index_bytes(faiss, ann), "tradeoff": tradeoff}
    result["elapsed_seconds"] = float(time.perf_counter() - started)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"[complete] {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
