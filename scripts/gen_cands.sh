#!/usr/bin/env bash
# Generate beam candidates for every dataset/model/checkpoint used downstream.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

models=(
 'allenai/Olmo-3-1025-7B|stage1-step141000 stage1-step283000 stage1-step424000 stage1-step566000 stage1-step707000 stage1-step1272000 stage1-step1413814'
 'allenai/Olmo-3-1125-32B|stage1-step66000 stage1-step131000 stage1-step197000 stage1-step262000 stage1-step328000 stage1-step590120 stage1-step656000'
 'marin-community/marin-8b-base|kestrel ocelot jellyfish phoenix starling deeper-starling'
)
for spec in "${models[@]}"; do
    IFS='|' read -r model revisions <<< "$spec"
    read -r -a revisions <<< "$revisions"
    for dataset_range in 'triviaqa 0 9961' 'jeopardy 0 20000' 'bioasq 0 2719'; do
        read -r dataset beg end <<< "$dataset_range"
        for revision in "${revisions[@]}"; do
            python -m conset.gen_cands --config conset/gen_configs/beam.yaml \
                --dataset "$dataset" --model_name "$model" --revision "$revision" \
                --beg_i "$beg" --end_i "$end"
        done
    done
done
