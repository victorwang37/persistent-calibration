#!/usr/bin/env bash
# Train every LoRA adapter required by conset/latex.py.
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

seeds=(17 18 19)
datasets=(triviaqa jeopardy)
models=(
    'allenai/Olmo-3-1025-7B|stage1-step141000 stage1-step283000 stage1-step424000|stage1-step566000 stage1-step707000 stage1-step1272000 stage1-step1413814'
    'allenai/Olmo-3-1125-32B|stage1-step66000 stage1-step131000 stage1-step197000|stage1-step262000 stage1-step328000 stage1-step590120 stage1-step656000'
    'marin-community/marin-8b-base|kestrel ocelot jellyfish|phoenix starling deeper-starling'
)

train() {
    local config="$1" model="$2" seed="$3"
    shift 3
    python -m conset.train_confidence --config "conset/train_configs/${config}.yaml" \
        --model_name "$model" --revisions "$@" --gen_config_name beam --seed "$seed" \
        --adapter_root "$adapter_root"
}

train_respective() {
    local config="$1" model="$2" seed="$3"
    shift 3
    python -m conset.train_confidence --config "conset/train_configs/${config}.yaml" \
        --model_name "$model" --revisions "$@" --gen_config_name beam --seed "$seed" \
        --use_ckpt_respective_predictor --adapter_root "$adapter_root"
}

for model_spec in "${models[@]}"; do
    IFS='|' read -r model non_oracle oracle <<< "$model_spec"
    read -r -a non_oracle_revisions <<< "$non_oracle"
    read -r -a oracle_revisions <<< "$oracle"
    for dataset in "${datasets[@]}"; do
        for suffix in '' '_ncand2'; do
            config="lora_${dataset}_acc_lr2e-4_bs16_5k${suffix}"
            for seed in "${seeds[@]}"; do
                # OLMo 32B multi-checkpoint conditions run in train_dist.sh.
                if [[ "$model" != 'allenai/Olmo-3-1125-32B' ]]; then
                    train "$config" "$model" "$seed" "${non_oracle_revisions[@]}"
                    train "$config" "$model" "$seed" "${oracle_revisions[@]}"
                fi
                train "$config" "$model" "$seed" "${non_oracle_revisions[-1]}"
                train_respective "$config" "$model" "$seed" "${oracle_revisions[@]}"
            done
        done
    done
done

# Olmo 7B: All-pairs transfer. Ablations of multi-ckpt training. Variance analysis.

olmo_model='allenai/Olmo-3-1025-7B'
olmo_all=(stage1-step141000 stage1-step283000 stage1-step424000
          stage1-step566000 stage1-step707000 stage1-step1272000 stage1-step1413814)
olmo_oracle=(stage1-step566000 stage1-step707000 stage1-step1272000 stage1-step1413814)
for dataset in "${datasets[@]}"; do
    for revision in "${olmo_all[@]}"; do
        train "lora_${dataset}_acc_lr2e-4_bs16_5k" "$olmo_model" 17 "$revision"
    done
    for ((i = 0; i < ${#olmo_all[@]} - 1; ++i)); do
        train "lora_${dataset}_acc_lr2e-4_bs16_5k_ncand2_random" "$olmo_model" 17 \
            "${olmo_all[$i]}" "${olmo_all[$((i + 1))]}"
    done
    for seed in "${seeds[@]}"; do
        train "lora_${dataset}_acc_lr2e-4_bs16_5k_ncand2_onedata" "$olmo_model" "$seed" "${olmo_oracle[@]}"
        train_respective "lora_${dataset}_acc_lr2e-4_bs16_5k_ncand2_pooldata" "$olmo_model" "$seed" "${olmo_oracle[@]}"
    done
done

# The hidden-state variance table uses ten independent respective-checkpoint
# TriviaQA adapters.  Seeds 17--19 were trained in the main loop above.
for seed in 20 21 22 23 24 25 26; do
    train_respective lora_triviaqa_acc_lr2e-4_bs16_5k "$olmo_model" "$seed" \
        "${olmo_oracle[@]}"
done
