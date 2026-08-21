#!/usr/bin/env bash
# Reproduce the cached KuaiRand-Pure pipeline from a clean virtual environment.
# This intentionally does not download raw data or invent a source path.
set -euo pipefail

usage() {
  echo "Usage: $0 --raw-dir /path/to/KuaiRand-Pure/data [--cache /path/to/ranklab_artifacts.tar.gz] [--venv /path/to/venv]" >&2
  exit 2
}

raw_dir=""
cache=""
venv_dir=".venv-reproduce"
while (($#)); do
  case "$1" in
    --raw-dir) raw_dir=${2:-}; shift 2 ;;
    --cache) cache=${2:-}; shift 2 ;;
    --venv) venv_dir=${2:-}; shift 2 ;;
    *) usage ;;
  esac
done
[[ -n "$raw_dir" && -d "$raw_dir" ]] || usage
[[ ! -e "$venv_dir" ]] || { echo "Refusing to overwrite existing venv: $venv_dir" >&2; exit 2; }

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python3 -m venv "$venv_dir"
"$venv_dir/bin/python" -m pip install --upgrade pip
"$venv_dir/bin/python" -m pip install -e "${project_root}[full,ope]"
if [[ -n "$cache" ]]; then
  [[ -f "$cache" || -d "$cache" ]] || { echo "Cache does not exist: $cache" >&2; exit 2; }
  "$venv_dir/bin/python" "$project_root/scripts/restore_kaggle_artifacts.py" "$cache"
fi
"$venv_dir/bin/python" "$project_root/scripts/run_full_pipeline.py" "raw_dir=$raw_dir" device=auto
