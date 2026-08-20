#!/usr/bin/env bash
set -euo pipefail

archive_name='KuaiRand-Pure.tar.gz'
expected_md5='0820331067a3784d9691136f772b35a7'
url="https://zenodo.org/records/10439422/files/${archive_name}"
raw_root='data/raw'
archive_path="${raw_root}/${archive_name}"
extract_dir="${raw_root}/KuaiRand-Pure"

mkdir -p "$raw_root"
if [[ -e "$extract_dir" ]]; then
  echo "Refusing to overwrite existing official extraction: $extract_dir" >&2
  exit 1
fi
if [[ ! -e "$archive_path" ]]; then
  curl --fail --location --retry 3 --output "$archive_path" "$url"
fi
if command -v md5sum >/dev/null 2>&1; then
  actual_md5="$(md5sum "$archive_path" | awk '{print $1}')"
elif command -v md5 >/dev/null 2>&1; then
  actual_md5="$(md5 -q "$archive_path")"
else
  echo "Neither md5sum nor md5 is available to verify $archive_path" >&2
  exit 1
fi
if [[ "$actual_md5" != "$expected_md5" ]]; then
  echo "MD5 mismatch for $archive_path: expected $expected_md5, got $actual_md5" >&2
  exit 1
fi
tar -xzf "$archive_path" -C "$raw_root"
test -d "$extract_dir/data"
echo "Verified and extracted official KuaiRand-Pure data at $extract_dir/data"
