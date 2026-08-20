from __future__ import annotations

import json
import hashlib
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yaml

from ranklab.data.kuairand import load_interactions, load_side_tables, raw_paths
from ranklab.data.sessionize import add_ranking_contexts
from ranklab.data.splitting import assign_splits, validate_splits
from ranklab.debias.density_ratio import DensityRatioModel, fit_density_ratio, weight_diagnostics
from ranklab.evaluation.analysis import (
    bootstrap_user_metric,
    cohort_report,
    diversity_report,
    item_cohort_report,
    model_ranking_stability,
    paired_user_bootstrap,
    recommendation_behavior,
)
from ranklab.evaluation.calibration import calibration_report, primary_category, user_preference_profiles
from ranklab.evaluation.prediction_calibration import (
    PointwiseProbabilityModel,
    fit_pointwise_probability,
    probability_metrics,
)
from ranklab.evaluation.ranking_metrics import evaluate_ranked_impressions
from ranklab.ranking.candidate_builder import (
    attach_labels,
    observed_ranking_rows,
    retrieval_metrics,
    retrieve_candidates,
)
from ranklab.ranking.lambdarank import LambdaRankModel, fit_lambdarank
from ranklab.ranking.ranker_features import FEATURE_COLUMNS, build_ranker_features
from ranklab.reporting.plots import generate_final_figures
from ranklab.reranking.greedy import rerank_frontier
from ranklab.retrieval.faiss_index import ExactInnerProductIndex
from ranklab.retrieval.two_tower import TwoTowerArtifacts, fit_two_tower
from ranklab.utils.artifacts import atomic_json, environment_snapshot, git_commit, sha256_files


PIPELINE_VERSION = 6


@dataclass
class FullPipeline:
    root: Path
    config: dict
    force: bool = False
    state_path: Path = field(init=False)
    state: dict = field(init=False)
    _interactions: pd.DataFrame | None = field(default=None, init=False)
    _sides: dict[str, pd.DataFrame] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.state_path = self.root / "outputs/full_pipeline_state.json"
        fingerprint = self._fingerprint()
        try:
            existing = json.loads(self.state_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            existing = {}
        if existing.get("fingerprint") != fingerprint or self.force:
            existing = {"pipeline_version": PIPELINE_VERSION, "fingerprint": fingerprint, "stages": {}}
        else:
            existing.setdefault("stages", {})
        self.state = existing

    def _fingerprint(self) -> dict:
        paths = raw_paths(self.config["raw_dir"])
        implementation = [
            path for directory in (self.root / "src/ranklab", self.root / "configs", self.root / "scripts")
            for path in directory.rglob("*") if path.is_file() and path.suffix in {".py", ".yaml", ".sh"}
        ]
        return {
            "pipeline_version": PIPELINE_VERSION,
            "raw_files": {path.name: {"bytes": path.stat().st_size} for path in paths.values()},
            "implementation_sha256": sha256_files(implementation, self.root),
            "config": {key: value for key, value in self.config.items() if not key.endswith("_dir")},
        }

    def _data(self) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        if self._interactions is None:
            self._interactions = assign_splits(load_interactions(self.config["raw_dir"]), self.config)
            validate_splits(self._interactions)
            self._sides = load_side_tables(self.config["raw_dir"])
        assert self._sides is not None
        return self._interactions, self._sides

    def _items(self, sides: dict[str, pd.DataFrame]) -> pd.DataFrame:
        return sides["items_basic"].rename(columns={"video_id": "item_id"}).copy()

    def _write_frame(self, frame: pd.DataFrame, relative: str) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
        return path

    def _write_json(self, payload: object, relative: str) -> Path:
        return atomic_json(self.root / relative, payload)

    def _artifact_record(self, path: Path) -> dict[str, object]:
        """Return a portable, content-addressed record for a derived artifact."""
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
        return {
            "path": str(path.relative_to(self.root)),
            "bytes": int(path.stat().st_size),
            "sha256": digest.hexdigest(),
        }

    def _sample_context_rows(self, frame: pd.DataFrame, maximum: int, seed_offset: int = 0) -> pd.DataFrame:
        rows = add_ranking_contexts(frame, int(self.config["session_gap_minutes"]), int(self.config["context_window_size"]))
        ids = rows["context_id"].drop_duplicates()
        if len(ids) > maximum:
            ids = ids.sample(maximum, random_state=int(self.config["seed"]) + seed_offset)
        return rows.loc[rows["context_id"].isin(set(ids))].copy()

    def _ranker_temporal_slices(self, train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
        timestamps = np.sort(train["timestamp_ms"].unique())
        if len(timestamps) < 2:
            raise ValueError("ranker source split requires at least two training timestamps")
        fraction = float(self.config["ranker_holdout_fraction"])
        if not 0 < fraction < 1:
            raise ValueError("ranker_holdout_fraction must be strictly between zero and one")
        source_count = min(max(int(len(timestamps) * (1 - fraction)), 1), len(timestamps) - 1)
        cutoff = int(timestamps[source_count - 1])
        source = train.loc[train["timestamp_ms"].le(cutoff)].copy()
        ranker = train.loc[train["timestamp_ms"].gt(cutoff)].copy()
        return source, ranker, cutoff

    def _randomized_domain_slices(self, interactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        random_rows = interactions.loc[interactions["split"].eq("randomized_test")].copy()
        adaptation = random_rows.loc[
            random_rows["date"].le(int(self.config["randomized_validation_end"]))
        ].copy()
        evaluation = random_rows.loc[
            random_rows["date"].ge(int(self.config["randomized_evaluation_start"]))
        ].copy()
        if adaptation.empty or evaluation.empty:
            raise ValueError("randomized adaptation/evaluation period is empty")
        # KuaiRand's official calendar date, not raw epoch milliseconds, is
        # authoritative across adjacent daily partitions.  The source clocks
        # have a small timezone-boundary overlap at some day transitions.
        if adaptation["date"].max() >= evaluation["date"].min():
            raise AssertionError("randomized adaptation dates overlap randomized evaluation")
        return adaptation, evaluation

    def _fit_retriever(
        self,
        train: pd.DataFrame,
        sides: dict[str, pd.DataFrame],
        *,
        negative_strategy: str | None = None,
        hard_negative_refresh: bool | None = None,
    ) -> TwoTowerArtifacts:
        return fit_two_tower(
            train, sides["users"], self._items(sides),
            embedding_dim=int(self.config["embedding_dim"]),
            epochs=int(self.config["epochs"]),
            batch_size=int(self.config["batch_size"]),
            learning_rate=float(self.config["learning_rate"]),
            temperature=float(self.config["temperature"]),
            seed=int(self.config["seed"]),
            device=str(self.config["device"]),
            use_side_features=bool(self.config.get("use_side_features", False)),
            history_length=int(self.config["history_length"]),
            negative_strategy=str(negative_strategy or self.config["negative_strategy"]),
            hard_negative_refresh=(
                bool(self.config["hard_negative_refresh"])
                if hard_negative_refresh is None else hard_negative_refresh
            ),
            hard_negative_epochs=int(self.config["hard_negative_epochs"]),
        )

    def _stage_cache_valid(self, name: str) -> bool:
        """Return true only when a recorded stage still has all its artifacts.

        The state fingerprint prevents reuse across incompatible code, config,
        or raw-file layouts. This second check prevents a partial Kaggle output
        restore (or a manually deleted model) from turning a stale state entry
        into a false cache hit.
        """
        record = self.state.get("stages", {}).get(name, {})
        if record.get("status") not in {"complete", "skipped"}:
            return False
        artifacts = record.get("artifacts", [])
        if not isinstance(artifacts, list):
            return False
        if record.get("status") == "skipped" and not artifacts:
            return True
        if not artifacts:
            return False
        for relative in artifacts:
            if not isinstance(relative, str):
                return False
            path = (self.root / relative).resolve()
            try:
                path.relative_to(self.root.resolve())
            except ValueError:
                return False
            if not path.is_file():
                return False
        return True

    def run(self, first: str | None = None, last: str | None = None) -> None:
        stages: list[tuple[str, Callable[[], dict]]] = [
            ("baselines", self.stage_baselines),
            ("two_tower", self.stage_two_tower),
            ("candidates", self.stage_candidates),
            ("rankers", self.stage_rankers),
            ("evaluation", self.stage_evaluation),
            ("analysis", self.stage_analysis),
            ("ope", self.stage_ope),
            ("scale", self.stage_scale),
            ("serving", self.stage_serving),
            ("report", self.stage_report),
        ]
        names = [name for name, _ in stages]
        start = names.index(first) if first else 0
        stop = names.index(last) + 1 if last else len(stages)
        for name, action in stages[start:stop]:
            if self._stage_cache_valid(name) and not self.force:
                print(f"[cache] {name}", flush=True)
                continue
            if self.state.get("stages", {}).get(name, {}).get("status") in {"complete", "skipped"}:
                print(f"[cache-miss] {name}: recorded artifacts are incomplete", flush=True)
            print(f"[start] {name}", flush=True)
            try:
                details = action()
            except Exception as error:
                self.state["stages"][name] = {"status": "failed", "error": repr(error), "updated_at": _now()}
                atomic_json(self.state_path, self.state)
                raise
            status = str(details.pop("status", "complete"))
            self.state["stages"][name] = {"status": status, "updated_at": _now(), **details}
            atomic_json(self.state_path, self.state)
            print(f"[{status}] {name}", flush=True)

    def stage_baselines(self) -> dict:
        expected = [
            "data/manifests/train.parquet", "data/manifests/validation.parquet",
            "data/manifests/standard_test.parquet", "data/manifests/randomized_test.parquet",
            "data/manifests/split_manifest.parquet",
            "outputs/reports/kuairand_data_audit.json",
            "outputs/reports/kuairand_data_audit.md",
            "outputs/reports/feature_leakage_audit.csv",
            "data/features/train_historical.parquet", "data/features/feature_manifest.json",
            "outputs/models/popularity.pkl", "outputs/metrics/popularity.json",
            "outputs/predictions/popularity_standard.parquet",
            "outputs/predictions/popularity_standard_per_group.parquet",
            "outputs/predictions/popularity_randomized.parquet",
            "outputs/predictions/popularity_randomized_per_group.parquet",
            "outputs/models/bpr.pkl", "outputs/metrics/bpr.json",
            "outputs/predictions/bpr_standard.parquet",
            "outputs/predictions/bpr_standard_per_group.parquet",
            "outputs/predictions/bpr_randomized.parquet",
            "outputs/predictions/bpr_randomized_per_group.parquet",
            "outputs/reports/initial_exposure_gap.md", "outputs/run_cache_manifest.json",
        ]
        existing_before = all((self.root / value).is_file() for value in expected)
        subprocess.run(
            [sys.executable, "scripts/run_cached_baselines.py", *self._config_overrides()],
            cwd=self.root,
            check=True,
        )
        return {"artifacts": expected, "artifacts_existed_before_cache_check": existing_before}

    def stage_two_tower(self) -> dict:
        interactions, sides = self._data()
        train = interactions.loc[interactions["split"].eq("train")].copy()
        source_train, ranker_train, cutoff = self._ranker_temporal_slices(train)

        # Ranker-training candidates must come from a retriever that has not
        # learned the held-out ranker labels. The final retriever is trained on
        # all standard training rows and is used only after that boundary.
        source_model = self._fit_retriever(source_train, sides)
        source_dir = self.root / "outputs/models/two_tower_ranker_source"
        source_model.save(source_dir)
        source_index = ExactInnerProductIndex.build(
            source_model.item_ids, source_model.item_embeddings
        )
        source_index_dir = self.root / "data/indices/two_tower_ranker_source_exact"
        source_index.save(source_index_dir)
        model = self._fit_retriever(train, sides)
        model_dir = self.root / "outputs/models/two_tower"
        model.save(model_dir)
        index = ExactInnerProductIndex.build(model.item_ids, model.item_embeddings)
        index_dir = self.root / "data/indices/two_tower_exact"
        index.save(index_dir)
        metrics: dict[str, dict] = {}
        artifacts = [
            str(path.relative_to(self.root))
            for directory in (source_dir, model_dir)
            for path in directory.glob("*") if path.is_file()
        ]
        _, randomized_evaluation = self._randomized_domain_slices(interactions)
        evaluation_splits = {
            "standard": interactions.loc[interactions["split"].eq("standard_test")],
            "randomized": randomized_evaluation,
        }
        for label, evaluation_rows in evaluation_splits.items():
            rows = observed_ranking_rows(
                evaluation_rows, model, int(self.config["session_gap_minutes"]),
                int(self.config["context_window_size"]),
            )
            rows["score"] = rows["retrieval_score"]
            values, per_group = evaluate_ranked_impressions(rows, tuple(self.config["k_values"]), int(self.config["min_group_size"]))
            metrics[label] = values
            prediction_path = self._write_frame(rows, f"outputs/predictions/two_tower_{label}.parquet")
            per_group_path = self._write_frame(
                per_group, f"outputs/predictions/two_tower_{label}_per_group.parquet"
            )
            artifacts.extend([
                str(prediction_path.relative_to(self.root)),
                str(per_group_path.relative_to(self.root)),
            ])
        self._write_json(metrics, "outputs/metrics/two_tower.json")
        if bool(self.config.get("run_retrieval_ablations", False)):
            ablations: dict[str, dict] = {}
            for strategy in self.config["retrieval_ablation_strategies"]:
                for hard_refresh in self.config["retrieval_ablation_hard_refresh"]:
                    name = f"{strategy}__hard_{str(bool(hard_refresh)).lower()}"
                    if (
                        strategy == self.config["negative_strategy"]
                        and bool(hard_refresh) == bool(self.config["hard_negative_refresh"])
                    ):
                        ablation_model = model
                    else:
                        ablation_model = self._fit_retriever(
                            train, sides, negative_strategy=str(strategy),
                            hard_negative_refresh=bool(hard_refresh),
                        )
                        ablation_dir = self.root / f"outputs/models/two_tower_ablation_{name}"
                        ablation_model.save(ablation_dir)
                        artifacts.extend(
                            str(path.relative_to(self.root))
                            for path in sorted(ablation_dir.glob("*")) if path.is_file()
                        )
                    ablations[name] = {}
                    for label, evaluation_rows in evaluation_splits.items():
                        sampled = self._sample_context_rows(
                            evaluation_rows, int(self.config["max_test_contexts"]), 31
                        )
                        rows = observed_ranking_rows(
                            sampled, ablation_model, int(self.config["session_gap_minutes"]),
                            int(self.config["context_window_size"]),
                        )
                        rows["score"] = rows["retrieval_score"]
                        values, _ = evaluate_ranked_impressions(
                            rows, tuple(self.config["k_values"]), int(self.config["min_group_size"])
                        )
                        catalog_candidates = retrieve_candidates(
                            sampled, ablation_model, int(self.config["candidate_k"])
                        )
                        ablations[name][label] = {
                            "exposed_ranking": values,
                            "catalog_retrieval": retrieval_metrics(
                                catalog_candidates, sampled,
                                tuple(int(value) for value in self.config["retrieval_k_values"]),
                            ),
                        }
            self._write_json(ablations, "outputs/metrics/retrieval_negative_ablations.json")
            artifacts.append("outputs/metrics/retrieval_negative_ablations.json")
        index_artifacts = [
            str(path.relative_to(self.root))
            for directory in (source_index_dir, index_dir)
            for path in sorted(directory.glob("*")) if path.is_file()
        ]
        return {
            "artifacts": artifacts + ["outputs/metrics/two_tower.json", *index_artifacts],
            "device": model.device,
            "ranker_source_end_timestamp_ms": cutoff,
            "ranker_source_rows": int(len(source_train)),
            "ranker_training_rows": int(len(ranker_train)),
        }

    def stage_candidates(self) -> dict:
        interactions, sides = self._data()
        train = interactions.loc[interactions["split"].eq("train")].copy()
        _, ranker_train, cutoff = self._ranker_temporal_slices(train)
        source_model = TwoTowerArtifacts.load(self.root / "outputs/models/two_tower_ranker_source")
        final_model = TwoTowerArtifacts.load(self.root / "outputs/models/two_tower")
        source_index = ExactInnerProductIndex.load(
            self.root / "data/indices/two_tower_ranker_source_exact"
        )
        final_index = ExactInnerProductIndex.load(self.root / "data/indices/two_tower_exact")
        outputs = []
        retrieval_report: dict[str, dict] = {}
        training_contracts = (
            (
                "train", ranker_train, source_model, source_index,
                int(self.config["max_train_contexts"]),
            ),
            (
                "validation", interactions.loc[interactions["split"].eq("validation")],
                final_model, final_index, int(self.config["max_validation_contexts"]),
            ),
        )
        for offset, (split, rows, model, index, maximum) in enumerate(training_contracts):
            exposed = self._sample_context_rows(rows, maximum, offset)
            retrieved = retrieve_candidates(
                exposed, model, int(self.config["candidate_k"]), index=index
            )
            retrieval_report[split] = retrieval_metrics(
                retrieved, exposed, tuple(int(value) for value in self.config["retrieval_k_values"])
            )
            candidates = attach_labels(retrieved, exposed, str(self.config["relevance_mode"]))
            featured = build_ranker_features(
                candidates, train, sides["users"], self._items(sides),
                point_in_time=split == "train",
                use_side_features=bool(self.config.get("use_side_features", False)),
            )
            path = self._write_frame(featured, f"data/processed/ranker/{split}.parquet")
            outputs.append(str(path.relative_to(self.root)))
            for date, partition in featured.groupby("date", sort=True):
                feature_path = self._write_frame(
                    partition,
                    f"data/features/ranker/{split}/date={int(date)}/part.parquet",
                )
                outputs.append(str(feature_path.relative_to(self.root)))
        _, randomized_evaluation = self._randomized_domain_slices(interactions)
        evaluation_contracts = (
            ("standard_test", interactions.loc[interactions["split"].eq("standard_test")]),
            ("randomized_test_holdout", randomized_evaluation),
        )
        for offset, (label, rows) in enumerate(evaluation_contracts, start=10):
            exposed = self._sample_context_rows(rows, int(self.config["max_test_contexts"]), offset)
            retrieved = retrieve_candidates(
                exposed, final_model, int(self.config["candidate_k"]), index=final_index
            )
            retrieval_report[label] = retrieval_metrics(
                retrieved, exposed, tuple(int(value) for value in self.config["retrieval_k_values"])
            )
        retrieval_report["contract"] = {
            "ranker_source_end_timestamp_ms": cutoff,
            "train_candidate_retriever": "two_tower_ranker_source",
            "validation_and_test_retriever": "two_tower",
            "candidate_index": "persisted_exact_inner_product_with_faiss_when_available",
            "forced_positives_excluded_from_retrieval_metrics": True,
        }
        self._write_json(
            {
                "primary_features": list(FEATURE_COLUMNS),
                "label_columns": ["label_binary", "label_graded"],
                "primary_label": f"label_{self.config['relevance_mode']}",
                "point_in_time_history": True,
                "history_includes_strictly_earlier_events_only": True,
                "undated_snapshot_side_features_enabled": bool(self.config.get("use_side_features", False)),
                "partitioning": ["split", "date"],
            },
            "data/features/feature_manifest.json",
        )
        outputs.append("data/features/feature_manifest.json")
        self._write_json(retrieval_report, "outputs/metrics/retrieval.json")
        return {"artifacts": outputs + ["outputs/metrics/retrieval.json"]}

    def stage_rankers(self) -> dict:
        train = pd.read_parquet(self.root / "data/processed/ranker/train.parquet")
        validation = pd.read_parquet(self.root / "data/processed/ranker/validation.parquet")
        natural = fit_lambdarank(
            train,
            validation,
            seed=int(self.config["seed"]),
            n_estimators=int(self.config["n_estimators"]),
            early_stopping_rounds=int(self.config["early_stopping_rounds"]),
        )
        natural_path = self.root / "outputs/models/lambdarank_natural.txt"
        natural.save(natural_path)
        natural_importance = self._write_frame(
            natural.feature_importance(), "outputs/metrics/lambdarank_natural_feature_importance.parquet"
        )
        natural.feature_importance().to_csv(
            self.root / "outputs/metrics/lambdarank_natural_feature_importance.csv", index=False
        )
        ablation_artifacts: list[str] = []
        if bool(self.config.get("run_relevance_ablation", False)):
            alternate_mode = "graded" if self.config["relevance_mode"] == "binary" else "binary"
            alternate_train = train.copy()
            alternate_validation = validation.copy()
            alternate_train["label"] = alternate_train[f"label_{alternate_mode}"]
            alternate_validation["label"] = alternate_validation[f"label_{alternate_mode}"]
            alternate = fit_lambdarank(
                alternate_train, alternate_validation, seed=int(self.config["seed"]),
                n_estimators=int(self.config["n_estimators"]),
                early_stopping_rounds=int(self.config["early_stopping_rounds"]),
            )
            alternate_path = self.root / f"outputs/models/lambdarank_relevance_{alternate_mode}.txt"
            alternate.save(alternate_path)
            alternate_importance = self._write_frame(
                alternate.feature_importance(),
                f"outputs/metrics/lambdarank_relevance_{alternate_mode}_feature_importance.parquet",
            )
            alternate.feature_importance().to_csv(
                self.root / f"outputs/metrics/lambdarank_relevance_{alternate_mode}_feature_importance.csv",
                index=False,
            )
            ablation_artifacts.extend(
                [
                    str(alternate_path.relative_to(self.root)),
                    str(alternate_importance.relative_to(self.root)),
                    f"outputs/metrics/lambdarank_relevance_{alternate_mode}_feature_importance.csv",
                ]
            )

        interactions, sides = self._data()
        history = interactions.loc[interactions["split"].eq("train")]
        # The ranker-training rows carry point-in-time aggregates computed with
        # allow_exact_matches=False.  Use those historical counts directly;
        # a full-training item count would expose later training events to an
        # earlier candidate context.
        popularity_weights = 1.0 / np.sqrt(
            train["hist_item_exposure"].fillna(0).to_numpy(dtype=float)
            + float(self.config["popularity_weight_offset"])
        )
        popularity_weights /= max(float(popularity_weights.mean()), 1e-12)
        popularity_weighted = fit_lambdarank(
            train, validation, popularity_weights, seed=int(self.config["seed"]),
            n_estimators=int(self.config["n_estimators"]),
            early_stopping_rounds=int(self.config["early_stopping_rounds"]),
        )
        popularity_weighted_path = self.root / "outputs/models/lambdarank_popularity_weighted.txt"
        popularity_weighted.save(popularity_weighted_path)
        popularity_importance = self._write_frame(
            popularity_weighted.feature_importance(),
            "outputs/metrics/lambdarank_popularity_weighted_feature_importance.parquet",
        )
        popularity_weighted.feature_importance().to_csv(
            self.root / "outputs/metrics/lambdarank_popularity_weighted_feature_importance.csv",
            index=False,
        )
        randomized_adaptation, _ = self._randomized_domain_slices(interactions)
        use_side = bool(self.config.get("use_side_features", False))
        retrieval = TwoTowerArtifacts.load(self.root / "outputs/models/two_tower")
        standard = build_ranker_features(
            observed_ranking_rows(
                interactions.loc[interactions["split"].eq("validation")], retrieval,
                int(self.config["session_gap_minutes"]), int(self.config["context_window_size"]),
            ),
            history, sides["users"], self._items(sides), use_side_features=use_side,
        )
        randomized = build_ranker_features(
            observed_ranking_rows(
                randomized_adaptation, retrieval,
                int(self.config["session_gap_minutes"]), int(self.config["context_window_size"]),
            ),
            history, sides["users"], self._items(sides), use_side_features=use_side,
        )
        density, diagnostics = fit_density_ratio(standard, randomized, seed=int(self.config["seed"]), max_rows_per_domain=int(self.config["max_rows_per_domain"]))
        density_path = self.root / "outputs/models/density_ratio.pkl"
        density.save(density_path)

        raw_weights = density.weights(train, None, False)
        diagnostics["raw"] = weight_diagnostics(raw_weights)
        density_raw = fit_lambdarank(
            train, validation, raw_weights, seed=int(self.config["seed"]),
            n_estimators=int(self.config["n_estimators"]),
            early_stopping_rounds=int(self.config["early_stopping_rounds"]),
        )
        density_raw_path = self.root / "outputs/models/lambdarank_density_raw.txt"
        density_raw.save(density_raw_path)
        density_raw_importance = self._write_frame(
            density_raw.feature_importance(),
            "outputs/metrics/lambdarank_density_raw_feature_importance.parquet",
        )
        density_raw.feature_importance().to_csv(
            self.root / "outputs/metrics/lambdarank_density_raw_feature_importance.csv",
            index=False,
        )
        clip_selection: list[dict[str, object]] = []
        selected_model: LambdaRankModel | None = None
        selected_clip: float | None = None
        selected_value = -np.inf
        for clip in (float(value) for value in self.config["clip_values"]):
            # Select the clipping threshold on the declared randomized
            # adaptation slice.  Global self-normalization is evaluated as a
            # separate ablation after the threshold is frozen.
            weights = density.weights(train, clip, False)
            candidate_model = fit_lambdarank(
                train, validation, weights, seed=int(self.config["seed"]),
                n_estimators=int(self.config["n_estimators"]),
                early_stopping_rounds=int(self.config["early_stopping_rounds"]),
            )
            predictions = randomized.copy()
            predictions["score"] = candidate_model.predict(predictions)
            values, _ = evaluate_ranked_impressions(
                predictions, tuple(self.config["k_values"]), int(self.config["min_group_size"])
            )
            objective_name = f"ndcg@{int(self.config['final_k'])}"
            objective = float(values.get(objective_name, float("nan")))
            selection_score = objective if np.isfinite(objective) else float("-inf")
            clip_selection.append(
                {
                    "clip": clip,
                    "selection_metric": objective_name,
                    "selection_value": objective if np.isfinite(objective) else None,
                    "weight_diagnostics": weight_diagnostics(weights),
                }
            )
            if selected_model is None or selection_score > selected_value:
                selected_value = selection_score
                selected_clip = clip
                selected_model = candidate_model
        if selected_model is None or selected_clip is None:
            raise RuntimeError("density-ratio clipping selection produced no model")
        clipped_weights = density.weights(train, selected_clip, False)
        self_normalized_weights = density.weights(train, selected_clip, True)
        density_clipped = selected_model
        density_clipped_path = self.root / "outputs/models/lambdarank_density_clipped.txt"
        density_clipped.save(density_clipped_path)
        density_clipped_importance = self._write_frame(
            density_clipped.feature_importance(),
            "outputs/metrics/lambdarank_density_clipped_feature_importance.parquet",
        )
        density_clipped.feature_importance().to_csv(
            self.root / "outputs/metrics/lambdarank_density_clipped_feature_importance.csv",
            index=False,
        )
        density_self_normalized = fit_lambdarank(
            train, validation, self_normalized_weights, seed=int(self.config["seed"]),
            n_estimators=int(self.config["n_estimators"]),
            early_stopping_rounds=int(self.config["early_stopping_rounds"]),
        )
        density_self_normalized_path = (
            self.root / "outputs/models/lambdarank_density_self_normalized.txt"
        )
        density_self_normalized.save(density_self_normalized_path)
        density_self_normalized_importance = self._write_frame(
            density_self_normalized.feature_importance(),
            "outputs/metrics/lambdarank_density_self_normalized_feature_importance.parquet",
        )
        density_self_normalized.feature_importance().to_csv(
            self.root
            / "outputs/metrics/lambdarank_density_self_normalized_feature_importance.csv",
            index=False,
        )
        weighted = (
            density_self_normalized
            if bool(self.config["self_normalize"])
            else density_clipped
        )
        weighted_path = self.root / "outputs/models/lambdarank_weighted.txt"
        weighted.save(weighted_path)
        weighted_importance = self._write_frame(
            weighted.feature_importance(), "outputs/metrics/lambdarank_weighted_feature_importance.parquet"
        )
        weighted.feature_importance().to_csv(
            self.root / "outputs/metrics/lambdarank_weighted_feature_importance.csv", index=False
        )
        selected_weights = (
            self_normalized_weights
            if bool(self.config["self_normalize"])
            else clipped_weights
        )
        diagnostics["clipped"] = weight_diagnostics(clipped_weights)
        diagnostics["self_normalized"] = weight_diagnostics(self_normalized_weights)
        diagnostics["selected"] = weight_diagnostics(selected_weights)
        diagnostics["inverse_sqrt_popularity_control"] = {
            **weight_diagnostics(popularity_weights),
            "formula": "1 / sqrt(strict_pre_context_item_exposure + offset), then mean-normalized",
            "offset": float(self.config["popularity_weight_offset"]),
            "temporal_contract": "uses hist_item_exposure computed strictly before each candidate context",
        }
        weight_rows = pd.DataFrame(
            {
                "context_id": train["context_id"].to_numpy(),
                "user_id": train["user_id"].to_numpy(),
                "item_id": train["item_id"].to_numpy(),
                "density_raw": raw_weights,
                "density_clipped": clipped_weights,
                "density_self_normalized": self_normalized_weights,
                "density_selected": selected_weights,
                "inverse_sqrt_popularity": popularity_weights,
            }
        )
        if len(weight_rows) > 200_000:
            weight_rows = weight_rows.sample(200_000, random_state=int(self.config["seed"]))
        weight_sample_path = self._write_frame(
            weight_rows, "outputs/metrics/training_weight_sample.parquet"
        )
        diagnostics.update(
            {
                "clip_selection": clip_selection,
                "selected_clip": selected_clip,
                "selected_primary_weighting": (
                    "clipped_self_normalized"
                    if bool(self.config["self_normalize"])
                    else "clipped"
                ),
                "clip_selection_weighting": "clipped_not_self_normalized",
                "selection_period_end": int(self.config["randomized_validation_end"]),
                "final_randomized_test_was_used_for_selection": False,
            }
        )
        self._write_json(diagnostics, "outputs/metrics/density_ratio.json")

        pointwise = fit_pointwise_probability(train, validation, FEATURE_COLUMNS, int(self.config["seed"]))
        pointwise_path = self.root / "outputs/models/pointwise_probability.pkl"
        pointwise.save(pointwise_path)
        return {
            "artifacts": [
                str(natural_path.relative_to(self.root)), str(weighted_path.relative_to(self.root)),
                str(popularity_weighted_path.relative_to(self.root)),
                str(density_raw_path.relative_to(self.root)),
                str(density_clipped_path.relative_to(self.root)),
                str(density_self_normalized_path.relative_to(self.root)),
                str(density_path.relative_to(self.root)), str(pointwise_path.relative_to(self.root)),
                str(natural_importance.relative_to(self.root)), str(weighted_importance.relative_to(self.root)),
                str(popularity_importance.relative_to(self.root)), str(weight_sample_path.relative_to(self.root)),
                str(density_raw_importance.relative_to(self.root)),
                str(density_clipped_importance.relative_to(self.root)),
                str(density_self_normalized_importance.relative_to(self.root)),
                "outputs/metrics/lambdarank_natural_feature_importance.csv",
                "outputs/metrics/lambdarank_weighted_feature_importance.csv",
                "outputs/metrics/lambdarank_popularity_weighted_feature_importance.csv",
                "outputs/metrics/lambdarank_density_raw_feature_importance.csv",
                "outputs/metrics/lambdarank_density_clipped_feature_importance.csv",
                "outputs/metrics/lambdarank_density_self_normalized_feature_importance.csv",
                "outputs/metrics/density_ratio.json",
            ] + ablation_artifacts,
            "selected_clip": selected_clip,
        }

    def stage_evaluation(self) -> dict:
        interactions, sides = self._data()
        history = interactions.loc[interactions["split"].eq("train")]
        retrieval = TwoTowerArtifacts.load(self.root / "outputs/models/two_tower")
        rankers = {
            "lambdarank_natural": LambdaRankModel.load(
                self.root / "outputs/models/lambdarank_natural.txt"
            ),
            "lambdarank_weighted": LambdaRankModel.load(
                self.root / "outputs/models/lambdarank_weighted.txt"
            ),
            "lambdarank_popularity_weighted": LambdaRankModel.load(
                self.root / "outputs/models/lambdarank_popularity_weighted.txt"
            ),
            "lambdarank_density_raw": LambdaRankModel.load(
                self.root / "outputs/models/lambdarank_density_raw.txt"
            ),
            "lambdarank_density_clipped": LambdaRankModel.load(
                self.root / "outputs/models/lambdarank_density_clipped.txt"
            ),
            "lambdarank_density_self_normalized": LambdaRankModel.load(
                self.root / "outputs/models/lambdarank_density_self_normalized.txt"
            ),
        }
        for mode in ("binary", "graded"):
            path = self.root / f"outputs/models/lambdarank_relevance_{mode}.txt"
            if path.is_file():
                rankers[f"lambdarank_relevance_{mode}"] = LambdaRankModel.load(path)
        pointwise = PointwiseProbabilityModel.load(
            self.root / "outputs/models/pointwise_probability.pkl"
        )
        _, randomized_evaluation = self._randomized_domain_slices(interactions)
        evaluation_splits = {
            "standard": interactions.loc[interactions["split"].eq("standard_test")],
            "randomized": randomized_evaluation,
        }
        all_metrics: dict[str, dict] = {}
        probability_report: dict[str, dict] = {}
        artifacts = []
        for split_label, split_rows in evaluation_splits.items():
            observed = observed_ranking_rows(
                split_rows, retrieval, int(self.config["session_gap_minutes"]),
                int(self.config["context_window_size"]),
            )
            features = build_ranker_features(
                observed, history, sides["users"], self._items(sides),
                use_side_features=bool(self.config.get("use_side_features", False)),
            )
            retrieval_predictions = features.copy()
            retrieval_predictions["score"] = retrieval_predictions["retrieval_score"]
            retrieval_values, retrieval_per_group = evaluate_ranked_impressions(
                retrieval_predictions, tuple(self.config["k_values"]),
                int(self.config["min_group_size"]),
            )
            all_metrics.setdefault("two_tower", {})[split_label] = retrieval_values
            artifacts.extend(
                [
                    str(self._write_frame(
                        retrieval_predictions,
                        f"outputs/predictions/two_tower_{split_label}.parquet",
                    ).relative_to(self.root)),
                    str(self._write_frame(
                        retrieval_per_group,
                        f"outputs/predictions/two_tower_{split_label}_per_group.parquet",
                    ).relative_to(self.root)),
                ]
            )
            for name, model in rankers.items():
                predictions = features.copy()
                predictions["score"] = model.predict(predictions)
                metrics, per_group = evaluate_ranked_impressions(
                    predictions, tuple(self.config["k_values"]), int(self.config["min_group_size"])
                )
                all_metrics.setdefault(name, {})[split_label] = metrics
                artifacts.extend(
                    [
                        str(self._write_frame(
                            predictions, f"outputs/predictions/{name}_{split_label}.parquet"
                        ).relative_to(self.root)),
                        str(self._write_frame(
                            per_group, f"outputs/predictions/{name}_{split_label}_per_group.parquet"
                        ).relative_to(self.root)),
                    ]
                )
            probability_predictions = features.copy()
            probability_predictions["probability"] = pointwise.predict(probability_predictions)
            probability_predictions["score"] = probability_predictions["probability"]
            probability_values, probability_per_group = evaluate_ranked_impressions(
                probability_predictions, tuple(self.config["k_values"]),
                int(self.config["min_group_size"]),
            )
            all_metrics.setdefault("pointwise_probability", {})[split_label] = probability_values
            probability_report[split_label] = probability_metrics(
                probability_predictions["long_view"].astype(int),
                probability_predictions["probability"],
            )
            artifacts.extend(
                [
                    str(self._write_frame(
                        probability_predictions,
                        f"outputs/predictions/pointwise_probability_{split_label}.parquet",
                    ).relative_to(self.root)),
                    str(self._write_frame(
                        probability_per_group,
                        f"outputs/predictions/pointwise_probability_{split_label}_per_group.parquet",
                    ).relative_to(self.root)),
                ]
            )
        all_metrics["evaluation_contract"] = {
            "standard_dates": [int(self.config["standard_test_start"]), int(self.config["standard_test_end"])],
            "randomized_dates": [
                int(self.config["randomized_evaluation_start"]),
                int(self.config["randomized_test_end"]),
            ],
            "randomized_adaptation_rows_excluded": True,
            "ranked_universe": "actually exposed rows within each synthetic context",
        }
        self._write_json(all_metrics, "outputs/metrics/rankers.json")
        self._write_json(probability_report, "outputs/metrics/probability_calibration.json")
        return {
            "artifacts": artifacts + [
                "outputs/metrics/rankers.json", "outputs/metrics/probability_calibration.json"
            ]
        }

    def stage_analysis(self) -> dict:
        interactions, sides = self._data()
        train = interactions.loc[interactions["split"].eq("train")]
        items = self._items(sides)
        item_categories = items[["item_id", "tag"]].copy()
        item_categories["category"] = item_categories["tag"].map(primary_category)
        profiles = user_preference_profiles(train, item_categories)
        ranked_metrics = json.loads((self.root / "outputs/metrics/rankers.json").read_text())
        model_names = [
            name for name, values in ranked_metrics.items()
            if isinstance(values, dict) and "standard" in values and "randomized" in values
        ]
        report: dict[str, dict] = {}
        artifacts: list[str] = []
        per_group_cache: dict[tuple[str, str], pd.DataFrame] = {}
        for name in model_names:
            report[name] = {}
            for split_label in ("standard", "randomized"):
                predictions = pd.read_parquet(
                    self.root / f"outputs/predictions/{name}_{split_label}.parquet"
                ).merge(
                    item_categories[["item_id", "category"]], on="item_id", how="left",
                    validate="many_to_one",
                )
                per_group = pd.read_parquet(
                    self.root / f"outputs/predictions/{name}_{split_label}_per_group.parquet"
                )
                per_group_cache[(name, split_label)] = per_group
                report[name][split_label] = {
                    "behavior": recommendation_behavior(
                        predictions, len(items), int(self.config["final_k"])
                    ),
                    "bootstrap_ndcg": bootstrap_user_metric(
                        per_group, "ndcg", int(self.config["final_k"]),
                        int(self.config["bootstrap_samples"]), int(self.config["seed"]),
                    ),
                    "calibration": calibration_report(
                        predictions, profiles, int(self.config["final_k"])
                    ),
                    "diversity": diversity_report(predictions, int(self.config["final_k"])),
                }
                user_cohort_path = self._write_frame(
                    cohort_report(predictions, train, tuple(self.config["k_values"])),
                    f"outputs/metrics/{name}_{split_label}_user_cohorts.parquet",
                )
                item_cohort_path = self._write_frame(
                    item_cohort_report(predictions, train, tuple(self.config["k_values"])),
                    f"outputs/metrics/{name}_{split_label}_item_cohorts.parquet",
                )
                artifacts.extend(
                    [str(user_cohort_path.relative_to(self.root)), str(item_cohort_path.relative_to(self.root))]
                )

        paired: dict[str, dict] = {}
        comparisons = {
            "density_weighted_minus_natural": "lambdarank_weighted",
            "density_raw_minus_natural": "lambdarank_density_raw",
            "density_clipped_minus_natural": "lambdarank_density_clipped",
            "density_self_normalized_minus_natural": "lambdarank_density_self_normalized",
            "popularity_weighted_minus_natural": "lambdarank_popularity_weighted",
            "ranker_minus_retrieval": "lambdarank_natural",
        }
        for comparison, left_model in comparisons.items():
            right_model = "two_tower" if comparison == "ranker_minus_retrieval" else "lambdarank_natural"
            if left_model not in model_names or right_model not in model_names:
                continue
            paired[comparison] = {}
            for split_label in ("standard", "randomized"):
                paired[comparison][split_label] = paired_user_bootstrap(
                    per_group_cache[(left_model, split_label)],
                    per_group_cache[(right_model, split_label)],
                    metric="ndcg", k=int(self.config["final_k"]),
                    samples=int(self.config["bootstrap_samples"]), seed=int(self.config["seed"]),
                )
        stability = model_ranking_stability(
            {name: ranked_metrics[name] for name in model_names},
            f"ndcg@{int(self.config['final_k'])}",
        )

        frontier_report: dict[str, dict] = {}
        for split_label in ("standard", "randomized"):
            natural = pd.read_parquet(
                self.root / f"outputs/predictions/lambdarank_natural_{split_label}.parquet"
            ).merge(
                item_categories[["item_id", "category"]], on="item_id", how="left",
                validate="many_to_one",
            )
            frontier = rerank_frontier(
                natural, profiles, tuple(float(value) for value in self.config["relevance_weights"]),
                int(self.config["final_k"]),
            )
            frontier_path = self._write_frame(
                frontier, f"outputs/predictions/calibration_frontier_{split_label}.parquet"
            )
            artifacts.append(str(frontier_path.relative_to(self.root)))
            frontier_report[split_label] = {}
            for weight, frame in frontier.groupby("relevance_weight", sort=True):
                reranked = frame.copy()
                reranked["score"] = -reranked["rerank_position"].astype(float)
                metrics, _ = evaluate_ranked_impressions(
                    reranked, tuple(self.config["k_values"]), int(self.config["min_group_size"])
                )
                frontier_report[split_label][str(float(weight))] = {
                    "ranking": metrics,
                    "calibration": calibration_report(
                        reranked, profiles, int(self.config["final_k"])
                    ),
                    "diversity": diversity_report(reranked, int(self.config["final_k"])),
                    "behavior": recommendation_behavior(
                        reranked, len(items), int(self.config["final_k"])
                    ),
                }

        analysis = {
            "models": report,
            "paired_model_comparisons": paired,
            "model_ranking_stability": stability,
            "calibration_diversity_frontier": frontier_report,
        }
        self._write_json(analysis, "outputs/metrics/robustness_and_calibration.json")
        lines = [
            "# Standard vs randomized exposure gap", "",
            "The randomized column uses only the held-out randomized period and was not used to select density-ratio clipping.", "",
            "| model | metric | standard | randomized | standard - randomized |", "|---|---:|---:|---:|---:|",
        ]
        for name in model_names:
            for metric, standard_value in ranked_metrics[name]["standard"].items():
                randomized_value = ranked_metrics[name]["randomized"].get(metric)
                if isinstance(standard_value, (int, float)) and isinstance(randomized_value, (int, float)):
                    lines.append(
                        f"| {name} | {metric} | {standard_value:.6g} | "
                        f"{randomized_value:.6g} | {standard_value - randomized_value:.6g} |"
                    )
        gap_path = self.root / "outputs/reports/exposure_gap.md"
        gap_path.parent.mkdir(parents=True, exist_ok=True)
        gap_path.write_text("\n".join(lines) + "\n")
        artifacts.extend(
            ["outputs/metrics/robustness_and_calibration.json", str(gap_path.relative_to(self.root))]
        )
        return {"artifacts": artifacts}

    def stage_ope(self) -> dict:
        if not bool(self.config.get("optional_ope")):
            path = self._write_json({"status": "skipped", "reason": "Open Bandit Dataset not configured; no propensity values are invented from KuaiRand."}, "outputs/metrics/ope_status.json")
            return {"status": "skipped", "artifacts": [str(path.relative_to(self.root))]}
        command = [
            sys.executable, "scripts/run_ope_benchmark.py",
            "--output", "outputs/metrics/ope.json",
            "--campaign", str(self.config["obd_campaign"]),
            "--behavior-policy", str(self.config["obd_behavior_policy"]),
            "--bootstrap-samples", str(int(self.config["ope_bootstrap_samples"])),
            "--sample-fractions", *(str(float(value)) for value in self.config["ope_sample_fractions"]),
            "--seeds", *(str(int(value)) for value in self.config["ope_seeds"]),
            "--clip-values", *(str(float(value)) for value in self.config["ope_clip_values"]),
        ]
        if self.config.get("obd_data_path"):
            command.extend(["--data-path", str(self.config["obd_data_path"])])
        subprocess.run(command, cwd=self.root, check=True)
        return {"artifacts": ["outputs/metrics/ope.json"]}

    def stage_scale(self) -> dict:
        if not bool(self.config.get("optional_scale")):
            return {"status": "skipped", "reason": "scale experiment disabled"}
        model = TwoTowerArtifacts.load(self.root / "outputs/models/two_tower")
        multiplier = int(self.config["scale_multiplier"])
        # This is explicitly a controlled replicated corpus, not KuaiRand-1K.
        ids = np.arange(len(model.item_ids) * multiplier, dtype=np.int64)
        embeddings = np.tile(model.item_embeddings, (multiplier, 1)).astype(np.float32)
        rng = np.random.default_rng(int(self.config["seed"]))
        embeddings += rng.normal(0, 1e-5, embeddings.shape).astype(np.float32)
        embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
        queries = model.user_embeddings[: min(int(self.config["scale_query_count"]), len(model.user_embeddings))]
        exact_started = time.perf_counter()
        exact = ExactInnerProductIndex.build(ids, embeddings)
        exact_build_ms = (time.perf_counter() - exact_started) * 1000
        exact_ids, _ = exact.search(queries, int(self.config["candidate_k"]))
        exact_metrics = exact.benchmark(queries, int(self.config["candidate_k"]))
        exact_metrics["build_ms"] = float(exact_build_ms)
        result = {
            "corpus": "controlled_replication_with_seeded_jitter",
            "corpus_claim": "controlled scalability study; not a KuaiRand-1K measurement",
            "seed": int(self.config["seed"]),
            "source_items": len(model.item_ids),
            "replication_multiplier": multiplier,
            "replicated_items": len(ids),
            "embedding_dimension": int(embeddings.shape[1]),
            "queries": len(queries),
            "candidate_k": int(self.config["candidate_k"]),
            "exact": exact_metrics,
        }
        try:
            import faiss
            hnsw_m = int(self.config["scale_hnsw_m"])
            ann = faiss.IndexHNSWFlat(
                embeddings.shape[1], hnsw_m, faiss.METRIC_INNER_PRODUCT
            )
            ann.hnsw.efConstruction = int(self.config["scale_hnsw_ef_construction"])
            build_started = time.perf_counter()
            ann.add(np.ascontiguousarray(embeddings))
            build_ms = (time.perf_counter() - build_started) * 1000
            serialized_bytes = int(faiss.serialize_index(ann).nbytes)
            tradeoff = []
            for ef_search in self.config["scale_hnsw_ef_search_values"]:
                ann.hnsw.efSearch = int(ef_search)
                samples = []
                for query in queries:
                    started = time.perf_counter()
                    ann.search(np.ascontiguousarray(query[None, :]), int(self.config["candidate_k"]))
                    samples.append((time.perf_counter() - started) * 1000)
                _, positions = ann.search(
                    np.ascontiguousarray(queries), int(self.config["candidate_k"])
                )
                ann_ids = np.full(positions.shape, -1, dtype=np.int64)
                valid = positions >= 0
                ann_ids[valid] = ids[positions[valid]]
                recall = np.mean([
                    len(set(expected) & set(actual[actual >= 0])) / max(len(expected), 1)
                    for expected, actual in zip(exact_ids, ann_ids)
                ])
                tradeoff.append({
                    "ef_search": int(ef_search),
                    "p50_ms": float(np.percentile(samples, 50)) if samples else 0.0,
                    "p95_ms": float(np.percentile(samples, 95)) if samples else 0.0,
                    "mean_ms": float(np.mean(samples)) if samples else 0.0,
                    "recall_at_k": float(recall),
                })
            result["hnsw"] = {
                "M": hnsw_m,
                "ef_construction": int(self.config["scale_hnsw_ef_construction"]),
                "build_ms": float(build_ms),
                "index_bytes": serialized_bytes,
                "tradeoff": tradeoff,
            }
        except ModuleNotFoundError:
            result["hnsw"] = {"status": "unavailable", "reason": "faiss-cpu is not installed"}
        self._write_json(result, "outputs/metrics/scaling.json")
        return {"artifacts": ["outputs/metrics/scaling.json"], "corpus": "controlled_replication_with_seeded_jitter"}

    def stage_serving(self) -> dict:
        from ranklab.serving.pipeline import RetrievalRankingPipeline
        model = TwoTowerArtifacts.load(self.root / "outputs/models/two_tower")
        ranker = LambdaRankModel.load(self.root / "outputs/models/lambdarank_weighted.txt")
        index = ExactInnerProductIndex.load(self.root / "data/indices/two_tower_exact")
        interactions, _ = self._data()
        history = interactions.loc[interactions["split"].eq("train")]
        serving = RetrievalRankingPipeline(model, ranker, history, index=index)
        sample = model.user_ids[: min(100, len(model.user_ids))].astype(int).tolist()
        result = serving.benchmark(sample, int(datetime.now(timezone.utc).timestamp() * 1000), int(self.config["candidate_k"]))
        result.update({"python": platform.python_version(), "device": model.device})
        self._write_json(result, "outputs/metrics/serving_latency.json")
        (self.root / "outputs/environment.txt").write_text(environment_snapshot())
        return {"artifacts": ["outputs/metrics/serving_latency.json", "outputs/environment.txt"]}

    def stage_report(self) -> dict:
        metric_files = sorted((self.root / "outputs/metrics").glob("*.json"))
        payload = {path.stem: json.loads(path.read_text()) for path in metric_files}
        run_dir = self.root / "outputs/run"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.yaml").write_text(
            yaml.safe_dump(self.config, sort_keys=True), encoding="utf-8"
        )
        commit = git_commit(self.root)
        (run_dir / "git_commit.txt").write_text(commit + "\n", encoding="utf-8")
        environment = environment_snapshot()
        (run_dir / "environment.txt").write_text(environment, encoding="utf-8")
        (run_dir / "seed.txt").write_text(str(int(self.config["seed"])) + "\n", encoding="utf-8")

        data_files = []
        for role, raw_path in raw_paths(self.config["raw_dir"]).items():
            digest = hashlib.sha256()
            with raw_path.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
            data_files.append({
                "role": role,
                "official_file": raw_path.name,
                "bytes": raw_path.stat().st_size,
                "sha256": digest.hexdigest(),
            })
        archive = self.root / "data/raw/KuaiRand-Pure.tar.gz"
        archive_metadata: dict[str, object] = {
            "path": "data/raw/KuaiRand-Pure.tar.gz", "present": archive.is_file(),
            "official_md5": "0820331067a3784d9691136f772b35a7",
        }
        if archive.is_file():
            archive_digest = hashlib.md5(usedforsecurity=False)
            with archive.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    archive_digest.update(chunk)
            archive_metadata.update({
                "bytes": archive.stat().st_size,
                "md5": archive_digest.hexdigest(),
                "matches_official_md5": archive_digest.hexdigest()
                == archive_metadata["official_md5"],
            })
        data_manifest = {
            "dataset": "KuaiRand-Pure",
            "raw_directory": str(self.config["raw_dir"]),
            "official_files": data_files,
            "official_archive": archive_metadata,
        }
        atomic_json(run_dir / "data_manifest.json", data_manifest)

        interactions, _ = self._data()
        splits = []
        for split, frame in interactions.groupby("split", sort=True, observed=True):
            splits.append({
                "split": str(split),
                "rows": int(len(frame)),
                "users": int(frame["user_id"].nunique()),
                "items": int(frame["item_id"].nunique()),
                "minimum_date": int(frame["date"].min()),
                "maximum_date": int(frame["date"].max()),
                "minimum_timestamp_ms": int(frame["timestamp_ms"].min()),
                "maximum_timestamp_ms": int(frame["timestamp_ms"].max()),
            })
        split_manifest = {
            "splits": splits,
            "randomized_adaptation_end": int(self.config["randomized_validation_end"]),
            "randomized_evaluation_start": int(self.config["randomized_evaluation_start"]),
            "randomized_training_rows": 0,
        }
        atomic_json(run_dir / "split_manifest.json", split_manifest)
        split_manifest_path = self._write_frame(
            pd.DataFrame(splits), "outputs/run/split_manifest.parquet"
        )
        feature_manifest_source = self.root / "data/features/feature_manifest.json"
        feature_manifest = json.loads(feature_manifest_source.read_text())
        feature_audit_source = self.root / "outputs/reports/feature_leakage_audit.csv"
        if not feature_audit_source.is_file():
            raise FileNotFoundError(f"Missing required candidate feature audit: {feature_audit_source}")
        feature_audit_path = run_dir / "feature_leakage_audit.csv"
        feature_audit_path.write_bytes(feature_audit_source.read_bytes())
        feature_audit = pd.read_csv(feature_audit_path)
        excluded = feature_audit.loc[
            ~feature_audit["status"].eq("allowed exposure-time context"),
            ["table", "column", "status", "reason"],
        ].to_dict("records")
        feature_manifest["candidate_feature_audit"] = "outputs/run/feature_leakage_audit.csv"
        feature_manifest["non_primary_candidate_columns"] = excluded
        atomic_json(run_dir / "feature_manifest.json", feature_manifest)
        atomic_json(run_dir / "metrics.json", payload)

        per_user_frames: list[pd.DataFrame] = []
        for source in sorted((self.root / "outputs/predictions").glob("*_per_group.parquet")):
            stem = source.stem.removesuffix("_per_group")
            domain = next(
                (value for value in ("randomized", "standard") if stem.endswith(f"_{value}")),
                "unknown",
            )
            model = stem.removesuffix(f"_{domain}") if domain != "unknown" else stem
            frame = pd.read_parquet(source)
            if frame.empty or "user_id" not in frame:
                continue
            numeric = [value for value in ("ndcg", "recall", "mrr") if value in frame]
            group_columns = ["user_id"] + (["k"] if "k" in frame else [])
            per_user = frame.groupby(group_columns, as_index=False)[numeric].mean()
            per_user["contexts"] = frame.groupby(group_columns).size().to_numpy()
            per_user.insert(0, "domain", domain)
            per_user.insert(0, "model", model)
            per_user_frames.append(per_user)
        run_artifacts_extra: list[str] = []
        if per_user_frames:
            per_user_path = self._write_frame(
                pd.concat(per_user_frames, ignore_index=True),
                "outputs/run/metrics_by_user.parquet",
            )
            run_artifacts_extra.append(str(per_user_path.relative_to(self.root)))

        cohort_frames: list[pd.DataFrame] = []
        for source in sorted((self.root / "outputs/metrics").glob("*_cohorts.parquet")):
            frame = pd.read_parquet(source)
            frame.insert(0, "artifact", source.stem)
            frame.insert(
                1, "cohort_type",
                "user_history" if source.stem.endswith("_user_cohorts") else "item_popularity",
            )
            cohort_frames.append(frame)
        if cohort_frames:
            cohort_path = self._write_frame(
                pd.concat(cohort_frames, ignore_index=True, sort=False),
                "outputs/run/metrics_by_cohort.parquet",
            )
            run_artifacts_extra.append(str(cohort_path.relative_to(self.root)))

        prediction_files = sorted((self.root / "outputs/predictions").glob("*.parquet"))
        prediction_manifest = {
            "files": [self._artifact_record(path) for path in prediction_files]
        }
        atomic_json(run_dir / "predictions_manifest.json", prediction_manifest)
        model_and_index_files = sorted(
            path
            for base in (self.root / "outputs/models", self.root / "data/indices")
            if base.is_dir()
            for path in base.rglob("*")
            if path.is_file()
        )
        atomic_json(
            run_dir / "model_index_manifest.json",
            {
                "files": [self._artifact_record(path) for path in model_and_index_files]
            },
        )
        figure_artifacts = generate_final_figures(self.root, interactions, self.config)
        reproducibility = {
            "generated_at": _now(),
            "git_commit": commit,
            "git_dirty": subprocess.run(
                ["git", "status", "--porcelain"], cwd=self.root, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
            ).stdout.strip() != "",
            "pipeline_version": PIPELINE_VERSION,
            "state_file": "outputs/full_pipeline_state.json",
            "randomized_training_rows": 0,
            "ope_uses_kuairand_propensities": False,
            "split_manifest_parquet": str(split_manifest_path.relative_to(self.root)),
            "prediction_artifact_count": len(prediction_files),
            "model_and_index_artifact_count": len(model_and_index_files),
            "final_figures": figure_artifacts,
        }
        atomic_json(run_dir / "reproducibility.json", reproducibility)
        lines = ["# RankLab full experiment report", "", f"Generated: {_now()}", f"Commit: `{commit}`", "", "## Reproducibility", "", "Randomized KuaiRand rows are excluded from recommender and ranker label training. They are used only for randomized evaluation and the declared domain classifier. Historical training features use events with timestamps strictly below each candidate context. OPE is reported only when a propensity-bearing Open Bandit Dataset is explicitly configured. The complete candidate-column availability and leakage audit is preserved in `outputs/run/feature_leakage_audit.csv`.", ""]
        for name, values in payload.items():
            lines.extend([f"## {name}", "", "```json", json.dumps(values, indent=2, sort_keys=True), "```", ""])
        lines.extend(["## Final figures", ""])
        for figure in figure_artifacts:
            relative = Path(os.path.relpath(self.root / figure, self.root / "outputs/reports"))
            lines.append(f"![{Path(figure).stem}]({relative.as_posix()})")
            lines.append("")
        path = self.root / "outputs/reports/full_experiment_report.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines))
        run_artifacts = [
            "outputs/run/config.yaml", "outputs/run/git_commit.txt",
            "outputs/run/environment.txt", "outputs/run/seed.txt",
            "outputs/run/data_manifest.json", "outputs/run/split_manifest.json",
            "outputs/run/split_manifest.parquet",
            "outputs/run/feature_manifest.json", "outputs/run/metrics.json",
            "outputs/run/feature_leakage_audit.csv",
            "outputs/run/predictions_manifest.json", "outputs/run/model_index_manifest.json",
            "outputs/run/reproducibility.json",
        ]
        return {
            "artifacts": [
                str(path.relative_to(self.root)), *run_artifacts,
                *run_artifacts_extra, *figure_artifacts,
            ]
        }

    def _config_overrides(self) -> list[str]:
        # Baseline subprocesses must receive the exact same paths, temporal
        # boundaries, target, seed, and device as the parent pipeline. JSON is
        # also valid YAML, so the existing typed override parser can consume
        # scalars and lists without losing their types.
        overrides: list[str] = []
        for key, value in sorted(self.config.items()):
            if isinstance(value, Path):
                value = str(value)
            try:
                encoded = json.dumps(value, separators=(",", ":"))
            except TypeError:
                continue
            overrides.append(f"{key}={encoded}")
        return overrides


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
