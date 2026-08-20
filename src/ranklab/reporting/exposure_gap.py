from __future__ import annotations

import json
from pathlib import Path

def write_exposure_gap(metrics_dir: Path, report_path: Path) -> None:
    results = {}
    for model in ("popularity", "bpr"):
        path = metrics_dir / f"{model}.json"
        if path.exists(): results[model] = json.loads(path.read_text())
    if not results: raise FileNotFoundError("No baseline metric artifacts found; train and evaluate baselines first.")
    lines = ["# Initial exposure-gap report", "", "Randomized-exposure metrics reduce dependence on the historical exposure policy; they are not perfect preference ground truth.", "", "| Model | Metric | Standard test | Randomized test | Gap (standard - randomized) |", "|---|---:|---:|---:|---:|"]
    for model, payload in results.items():
        common = sorted(set(payload["standard"]).intersection(payload["randomized"]) - {"eligible_groups", "eligible_rows"})
        for metric in common:
            standard, random = payload["standard"][metric], payload["randomized"][metric]
            lines.append(f"| {model} | {metric} | {standard:.6f} | {random:.6f} | {standard-random:.6f} |")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")
