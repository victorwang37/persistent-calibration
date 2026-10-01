#!/usr/bin/env bash
# Train OLMo 3 32B multi-checkpoint adapters with Slurm checkpoint parallelism.
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
: "${SLURM_JOB_NODELIST:?Run this script inside a Slurm allocation.}"

model='allenai/Olmo-3-1125-32B'
seeds=(17 18 19)
datasets=(triviaqa jeopardy)
non_oracle=(stage1-step66000 stage1-step131000 stage1-step197000)
oracle=(stage1-step262000 stage1-step328000 stage1-step590120 stage1-step656000)

export MASTER_ADDR
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
port=29500

train_multi() {
    local nodes="$1" config="$2" seed="$3"
    shift 3
    export MASTER_PORT="$port"
    ((port += 1))
    srun -N "$nodes" -n "$nodes" \
        python -m conset.train_confidence_dist \
        --config "conset/train_configs/${config}.yaml" \
        --model_name "$model" --revisions "$@" --gen_config_name beam \
        --seed "$seed" --adapter_root "$adapter_root"
}

for dataset in "${datasets[@]}"; do
    for suffix in '' '_ncand2'; do
        config="lora_${dataset}_acc_lr2e-4_bs16_5k${suffix}"
        for seed in "${seeds[@]}"; do
            train_multi 3 "$config" "$seed" "${non_oracle[@]}"
            train_multi 4 "$config" "$seed" "${oracle[@]}"
        done
    done
done
