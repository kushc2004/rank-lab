# RankLab

RankLab is an end-to-end, exposure-aware recommender-system experiment on
KuaiRand-Pure. It implements the full project specification: reproducible data
auditing and temporal splits, leakage-safe historical features, popularity and
BPR-MF baselines, Two-Tower retrieval, exact FAISS candidate generation,
LightGBM LambdaRank, randomized-domain density-ratio weighting, randomized-log
evaluation, cohort/bootstrap analysis, calibration reranking, optional
propensity-bearing OPE, controlled scaling, and a read-only Streamlit report.

Randomized KuaiRand rows never enter recommender or ranker label training. The
primary configuration also excludes every undated user/item snapshot feature;
KuaiRand-Pure does not publish a historical availability timestamp for those
fields. `use_side_features=true` exists only as an explicitly relaxed leakage
sensitivity run. KuaiRand propensities are never fabricated.

## Official KuaiRand-Pure data

Download the official Zenodo archive, verify it, and preserve its internal
names:

```bash
bash scripts/download_kuairand_pure.sh
```

The workflow verifies MD5 `0820331067a3784d9691136f772b35a7` and expects the
official `KuaiRand-Pure/data/` hierarchy containing exactly these source files:

```text
log_random_4_22_to_5_08_pure.csv
log_standard_4_08_to_4_21_pure.csv
log_standard_4_22_to_5_08_pure.csv
user_features_pure.csv
video_features_basic_pure.csv
video_features_statistic_pure.csv
```

The source is the official [KuaiRand repository](https://github.com/chongminggao/KuaiRand)
and [Zenodo record](https://zenodo.org/records/10439422). To publish the already
verified archive as a private Kaggle input, run:

```bash
python scripts/publish_kuairand_source.py --create
```

## Full local or Kaggle-GPU run

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[full]'
python scripts/run_full_pipeline.py device=auto
```

`device=auto` chooses CUDA, then Apple MPS, then CPU. Neural retrieval and BPR
benefit from a GPU; LightGBM, auditing, feature construction, metrics, and
reports remain CPU workloads. Configuration overrides use `key=value`, for
example:

```bash
python scripts/run_full_pipeline.py raw_dir=/path/to/KuaiRand-Pure/data epochs=12 candidate_k=300
```

The deterministic stage order is:

```text
baselines -> two_tower -> candidates -> rankers -> evaluation -> analysis
          -> ope -> scale -> serving -> report
```

Every completed stage is atomically recorded in
`outputs/full_pipeline_state.json`. Re-running the same command skips compatible
completed stages. Restored artifacts are accepted only when raw-file sizes,
configuration, and implementation fingerprint match. To restart a portion:

```bash
python scripts/run_full_pipeline.py --from-stage rankers
python scripts/run_full_pipeline.py --from-stage evaluation --to-stage report
python scripts/run_full_pipeline.py --force
```

The final report is `outputs/reports/full_experiment_report.md`; metrics and
per-context predictions remain in `outputs/metrics/` and
`outputs/predictions/`.

## Kaggle notebook and reusable outputs

`notebooks/kaggle_full_pipeline.ipynb` clones this public repository, validates
the attached official dataset, or downloads and checksum-verifies it from the
official source if that input is absent. It restores any attached derived-artifact
cache, runs only missing stages, and writes a fresh cache archive to notebook outputs.
The launch files are `notebooks/kernel-metadata.json` and
`notebooks/kaggle.yml`.

```bash
kaggle kernels push -p .
python3 scripts/watch_kaggle_kernel.py --slug kushchaudhari/ranklab-kuairand-pure-full-pipeline
```

The watcher appends live output to ignored `.kaggle-run.log` without reading or
printing credentials. Use `--once` for a single snapshot.

To package or restore derived artifacts yourself:

```bash
python scripts/publish_kaggle_artifacts.py --no-upload
python scripts/restore_kaggle_artifacts.py artifacts/kaggle/ranklab_artifacts.tar.gz
```

Raw data is never included in that archive. It contains manifests, historical
features, ranker data, indices, models, predictions, metrics, reports, and the
stage-state fingerprint.

## Optional Open Bandit Dataset OPE

KuaiRand-Pure has no documented row-level logging propensities, so the default
OPE stage writes a truthful skipped-status artifact. Install the separate OPE
extra and configure a real Open Bandit Dataset path to enable DM, IPS, SNIPS,
and doubly robust estimates with clipping and effective-sample-size diagnostics:

```bash
python -m pip install -e '.[full,ope]'
python scripts/run_full_pipeline.py optional_ope=true obd_data_path=/path/to/open_bandit_dataset
```

`kaggle_ope/` is an isolated, propensity-valid Kaggle launcher which uses the
OBP adapter's documented source when a local OBD path is not provided. It keeps
OBD estimates out of the KuaiRand-Pure report.

## Real KuaiRand-1K catalog-scale validation

KuaiRand-1K is deliberately a separate scale experiment. The project-side
adapter audits the official extracted hierarchy and the isolated Kaggle kernel
at `kaggle_1k/` attaches the owner-provided `annanet/kuairand-1000` source.
It performs exact-versus-HNSW index benchmarking on real 1K item metadata and
standard-log user histories. It reports index recall and latency, not trained
recommender quality.

```bash
kaggle kernels push -p kaggle_1k
python3 scripts/watch_kaggle_kernel.py --slug kushchaudhari/ranklab-kuairand-1k-scale \
  --output outputs/logs/kaggle_1k_scale.log
```

## Fresh-environment reproduction

After a completed cache archive is available, create an isolated virtual
environment and rerun the documented pipeline without redoing compatible
stages:

```bash
bash scripts/reproduce_fresh_environment.sh \
  --raw-dir /path/to/KuaiRand-Pure/data \
  --cache artifacts/kaggle/ranklab_artifacts.tar.gz
```

## Dashboard

After a completed experiment:

```bash
streamlit run app/streamlit_app.py
```

The dashboard is read-only and displays generated artifacts; it never launches
training from the browser.
