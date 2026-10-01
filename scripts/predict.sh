#!/usr/bin/env bash
# Run the curated release inference plans after scripts/train.sh completes.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Adapter and hidden-state storage defaults to $SCRATCH. Pass
# --adapter_root PATH to use another location.
adapter_root=''
if [[ "${1:-}" == '--adapter_root' ]]; then
    adapter_root="${2:?--adapter_root requires a path}"
    shift 2
fi
if (($#)); then
    echo "Usage: $0 [--adapter_root PATH]" >&2
    exit 2
fi
adapter_root="${adapter_root:-${SCRATCH:?SCRATCH not set.}}"

learned_configs=(
    conset/inference_configs/olmo7b.yaml
    conset/inference_configs/olmo32b.yaml
    conset/inference_configs/marin8b.yaml
    conset/inference_configs/olmo7b_ablations.yaml
    conset/inference_configs/olmo7b_transfer.yaml
)
for config in "${learned_configs[@]}"; do
    python -m conset.pred_confidence --inference_config "$config" \
        --adapter_root "$adapter_root"
done

surrogate_configs=(
    conset/inference_configs/olmo7b_surrogate.yaml
    conset/inference_configs/olmo32b_surrogate.yaml
    conset/inference_configs/marin8b_surrogate.yaml
)
for config in "${surrogate_configs[@]}"; do
    python -m conset.pred_confidence_surrogate --inference_config "$config" \
        --adapter_root "$adapter_root"
done

python -m conset.pred_confidence_gcm \
    --inference_config conset/inference_configs/gcm.yaml

# These self-consistency artifacts back the SC-surrogate rows in the ablation
# tables.  They use the final non-oracle checkpoint as the surrogate source.
run_sc_surrogate() {
    local dataset="$1" model="$2" surrogate_revision="$3" eval_range="$4"
    shift 4
    python -m conset.sc_surrogate \
        --dataset "$dataset" --model_name "$model" \
        --surrogate_revision "$surrogate_revision" \
        --eval_revisions "$@" --eval_ranges "$eval_range"
}

run_sc_surrogate triviaqa allenai/Olmo-3-1025-7B stage1-step424000 5000-9961 \
    stage1-step566000 stage1-step707000 stage1-step1272000 stage1-step1413814
run_sc_surrogate jeopardy allenai/Olmo-3-1025-7B stage1-step424000 15000-20000 \
    stage1-step566000 stage1-step707000 stage1-step1272000 stage1-step1413814
run_sc_surrogate triviaqa allenai/Olmo-3-1125-32B stage1-step197000 5000-9961 \
    stage1-step262000 stage1-step328000 stage1-step590120 stage1-step656000
run_sc_surrogate jeopardy allenai/Olmo-3-1125-32B stage1-step197000 15000-20000 \
    stage1-step262000 stage1-step328000 stage1-step590120 stage1-step656000
run_sc_surrogate triviaqa marin-community/marin-8b-base jellyfish 5000-9961 \
    phoenix starling deeper-starling
run_sc_surrogate jeopardy marin-community/marin-8b-base jellyfish 15000-20000 \
    phoenix starling deeper-starling

# The variance table uses final-token hidden states from the OLMo 7B
# respective-checkpoint, n_cand=1 adapters.
python -m conset.analyze_variance \
    --inference_config conset/inference_configs/olmo7b_variance.yaml \
    --adapter_root "$adapter_root"
