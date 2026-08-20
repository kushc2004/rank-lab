from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np


@dataclass
class ExactInnerProductIndex:
    item_ids: np.ndarray
    embeddings: np.ndarray
    backend: str = "numpy"
    _index: object | None = None

    @classmethod
    def build(cls, item_ids: np.ndarray, embeddings: np.ndarray):
        ids = np.asarray(item_ids, dtype=np.int64)
        vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2 or not len(vectors) or len(ids) != len(vectors):
            raise ValueError("item IDs and embeddings must describe a non-empty 2D index")
        if len(np.unique(ids)) != len(ids):
            raise ValueError("item IDs must be unique")
        if not np.isfinite(vectors).all():
            raise ValueError("item embeddings must be finite")
        try:
            import faiss
            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)
            return cls(ids, vectors, "faiss.IndexFlatIP", index)
        except ModuleNotFoundError:
            return cls(ids, vectors)

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        if queries.ndim != 2 or queries.shape[1] != self.embeddings.shape[1]:
            raise ValueError("queries must be a 2D matrix matching the index dimension")
        if int(k) < 1:
            raise ValueError("k must be positive")
        k = min(int(k), len(self.item_ids))
        if self._index is not None:
            scores, positions = self._index.search(queries, k)
        else:
            matrix = queries @ self.embeddings.T
            positions = np.argpartition(-matrix, k - 1, axis=1)[:, :k]
            scores = np.take_along_axis(matrix, positions, axis=1)
            order = np.argsort(-scores, axis=1, kind="stable")
            positions = np.take_along_axis(positions, order, axis=1)
            scores = np.take_along_axis(scores, order, axis=1)
        return self.item_ids[positions], scores

    def benchmark(self, queries: np.ndarray, k: int = 200) -> dict[str, float | str]:
        samples = []
        for query in queries:
            started = time.perf_counter()
            self.search(query[None, :], k)
            samples.append((time.perf_counter() - started) * 1000)
        return {
            "backend": self.backend,
            "queries": len(samples),
            "p50_ms": float(np.percentile(samples, 50)) if samples else 0.0,
            "p95_ms": float(np.percentile(samples, 95)) if samples else 0.0,
            "index_bytes": int(self.embeddings.nbytes + self.item_ids.nbytes),
        }

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        fallback = directory / "exact_ip_index.npz"
        np.savez_compressed(fallback, item_ids=self.item_ids, embeddings=self.embeddings)
        files = [fallback]
        if self._index is not None:
            import faiss
            native = directory / "faiss.index"
            faiss.write_index(self._index, str(native))
            files.append(native)
        manifest = {
            "format_version": 1,
            "index_type": "IndexFlatIP",
            "metric": "inner_product",
            "backend_at_build": self.backend,
            "items": int(len(self.item_ids)),
            "dimension": int(self.embeddings.shape[1]),
            "files": [
                {
                    "name": path.name,
                    "bytes": int(path.stat().st_size),
                    "sha256": _sha256(path),
                }
                for path in files
            ],
        }
        (directory / "index_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: str | Path) -> "ExactInnerProductIndex":
        """Restore the native FAISS index when available, with NumPy fallback."""
        directory = Path(directory)
        fallback = directory / "exact_ip_index.npz"
        with np.load(fallback, allow_pickle=False) as payload:
            item_ids = np.asarray(payload["item_ids"], dtype=np.int64)
            embeddings = np.ascontiguousarray(payload["embeddings"], dtype=np.float32)
        if embeddings.ndim != 2 or len(item_ids) != len(embeddings):
            raise ValueError("saved exact index has inconsistent IDs and embeddings")
        native = directory / "faiss.index"
        if native.is_file():
            try:
                import faiss
            except ModuleNotFoundError:
                pass
            else:
                index = faiss.read_index(str(native))
                if index.ntotal != len(item_ids) or index.d != embeddings.shape[1]:
                    raise ValueError("FAISS index disagrees with its portable index payload")
                return cls(item_ids, embeddings, "faiss.IndexFlatIP", index)
        return cls(item_ids, embeddings)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
