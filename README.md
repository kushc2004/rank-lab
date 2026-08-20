# RankLab (Milestones 0–3)

This implementation is limited to KuaiRand-Pure ingestion, temporal safety,
historical features, popularity/BPR baselines, and standard-versus-randomized
evaluation. It deliberately contains no neural retrieval, FAISS, LambdaRank,
density-ratio weighting, OPE, or UI code.

## Official data acquisition

The official KuaiRand repository documents the archive and checksum. Download
and verify it without changing its internal names:

```bash
bash scripts/download_kuairand_pure.sh
```

This obtains `KuaiRand-Pure.tar.gz` from the official Zenodo record, verifies
MD5 `0820331067a3784d9691136f772b35a7`, and extracts its official
`KuaiRand-Pure/data/` hierarchy under `data/raw/`. The script will not replace
an existing archive or extracted directory. See the official
[KuaiRand repository](https://github.com/chongminggao/KuaiRand) and
[Zenodo record](https://zenodo.org/records/10439422).

To create the private Kaggle input from that exact verified archive:

```bash
python scripts/publish_kuairand_source.py --create
```

## Reproduce the baseline experiment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
./scripts/download_kuairand_pure.sh
python scripts/audit_kuairand.py
python scripts/build_features.py data=kuairand_pure
python scripts/train_popularity.py data=kuairand_pure
python scripts/train_bpr.py data=kuairand_pure
python scripts/evaluate_exposure_gap.py data=kuairand_pure
pytest
```

## Reuse completed artifacts

The cache contains only derived artifacts (split manifests, leakage-safe
features, fitted models, predictions, metrics, and reports), never raw data.
Run the cached sequential pipeline after installing the package:

```bash
python scripts/run_cached_baselines.py
```

It verifies the official CSV file names/sizes and hashes the relevant
implementation/configuration before accepting a cache hit. To create a compact
archive suitable for a Kaggle Dataset or notebook output:

```bash
python scripts/publish_kaggle_artifacts.py --no-upload
```

The archive is written to `artifacts/kaggle/ranklab_artifacts.tar.gz` (ignored
by Git). It can be restored in another checkout with:

```bash
python scripts/restore_kaggle_artifacts.py artifacts/kaggle/ranklab_artifacts.tar.gz
```

The Kaggle notebook automatically restores an attached archive and then calls
the cached runner. It always packages fresh derived outputs at the end. Set
`PUBLISH_CACHE=True` in its last cell only when Kaggle API credentials are
available; the default is private when initially creating the dataset locally.

`train_bpr.py` defaults to `device: mps` in
`configs/retrieval/bpr_mf.yaml`, so on an Apple-silicon Mac its factor
training runs on the Metal GPU. It fails clearly if MPS is unavailable; set
`device: cpu` only when an explicit CPU run is desired.

For Kaggle, enable a GPU accelerator and run
`notebooks/kaggle_baselines.ipynb`; it overrides BPR to `device=cuda`.
`notebooks/kernel-metadata.json` is ready for `kaggle kernels push -p notebooks`.
It declares two private inputs: the checksum-verified official archive
`kushchaudhari/kuairand-pure-official` and the derived cache
`kushchaudhari/ranklab-baseline-artifacts`. Their exact contract is documented
in `notebooks/kaggle.yml`.

The split is deterministic: standard logs through 2022-04-21 train; standard
2022-04-22 through 2022-04-30 validation; standard 2022-05-01 through
2022-05-08 test; all random-intervention rows are randomized test only.

Metrics are only written after commands run on verified local raw data. They
are not inferred from the dataset's published aggregate statistics.
