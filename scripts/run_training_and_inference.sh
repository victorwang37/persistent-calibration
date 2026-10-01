#!/usr/bin/env bash
# Train adapters and regenerate every prediction artifact used by latex.py.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
adapter_root=''
while (($#)); do
    case "$1" in
        --adapter_root)
            adapter_root="${2:?--adapter_root requires a path}"
            shift 2
            ;;
        *)
            echo "Usage: $0 [--adapter_root PATH]" >&2
            exit 2
            ;;
    esac
done

echo 'Assuming the release-provided generated candidates and processed judgments are present under results/.' >&2
adapter_root="${adapter_root:-${SCRATCH:?SCRATCH not set.}}"
bash "$repo_root/scripts/train.sh" --adapter_root "$adapter_root"
bash "$repo_root/scripts/train_dist.sh" --adapter_root "$adapter_root"
bash "$repo_root/scripts/predict.sh" --adapter_root "$adapter_root"
