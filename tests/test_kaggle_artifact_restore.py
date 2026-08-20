import importlib.util
import sys
from pathlib import Path


def _restore_module():
    script = Path(__file__).parents[1] / "scripts/restore_kaggle_artifacts.py"
    spec = importlib.util.spec_from_file_location("restore_kaggle_artifacts", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_restore_accepts_kaggle_expanded_artifact_directory(tmp_path, monkeypatch):
    source = tmp_path / "input"
    expected = source / "data/manifests/train.parquet"
    expected.parent.mkdir(parents=True)
    expected.write_bytes(b"manifest")
    output = source / "outputs/metrics/popularity.json"
    output.parent.mkdir(parents=True)
    output.write_text("{}")

    module = _restore_module()
    destination = tmp_path / "checkout"
    monkeypatch.setattr(module, "ROOT", destination)
    monkeypatch.setattr(sys, "argv", ["restore_kaggle_artifacts.py", str(source)])
    module.main()

    assert (destination / "data/manifests/train.parquet").read_bytes() == b"manifest"
    assert (destination / "outputs/metrics/popularity.json").read_text() == "{}"
