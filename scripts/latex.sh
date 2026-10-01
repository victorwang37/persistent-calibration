#!/usr/bin/env bash
# Generate every reproducible table and figure from saved prediction artifacts.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python -m conset.latex
