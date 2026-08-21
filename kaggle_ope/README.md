# Open Bandit Dataset OPE notebook

This is a standalone OPE run. It keeps the OBD propensity-bearing behavior
logs and the separate random-policy on-policy feedback outside of the
KuaiRand-Pure experiment. Internet must remain enabled because the notebook
clones the repository and the OBP adapter may retrieve the official OBD files
when no local `data_path` is configured.

Run it only after the active P100 KuaiRand-Pure kernel is finished:

```sh
kaggle kernels push -p kaggle_ope
python3 scripts/watch_kaggle_kernel.py \
  --slug kushchaudhari/ranklab-open-bandit-ope \
  --output outputs/logs/kaggle_open_bandit_ope.log
```

The P100 request matches the project accelerator policy. The estimators are
CPU-oriented; the notebook does not claim GPU acceleration for OPE.
