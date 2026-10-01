# Reproduction scripts

These Bash scripts reproduce the released tables and figures from the supplied
candidate answer and judgment artifacts.

Install the pinned Python dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

Download `persistent-calibration-results-v1.tar.zst` and its accompanying
`.sha256` file from the matching GitHub Release, then extract the archive from
the repository root. The archive creates the required `results/` directory.

```bash
sha256sum -c persistent-calibration-results-v1.tar.zst.sha256
tar --zstd -xf persistent-calibration-results-v1.tar.zst
```

## Reproduce from released predictions

Render tables and figures from the supplied prediction artifacts:

```bash
bash scripts/latex.sh
```

## Reproduce by rerunning training and inference

Run `train.sh`, `train_dist.sh`, and `predict.sh` with:

```bash
export SCRATCH=/path/to/scratch
bash scripts/run_training_and_inference.sh
```

This complete workflow must run inside a Slurm allocation with at least four
nodes: `train_dist.sh` uses `srun` to train the Olmo 3 32B multi-checkpoint
adapters over three or four checkpoint-parallel ranks.

`$SCRATCH` is the default adapter and hidden-state storage root. Pass
`--adapter_root /path/to/storage` to `train.sh`, `train_dist.sh`, `predict.sh`,
or `run_training_and_inference.sh` to use another location.

`train.sh` writes LoRA adapters to
`$SCRATCH/persistent-calibration/results/conset/`. `predict.sh` reads those
adapters and writes prediction tensors into the checkout's `results/conset/`.
It also runs the public GCM and surrogate-confidence inference plans.
The active plans live in `conset/inference_configs/`; training hyperparameters live
in `conset/train_configs/`.

## Released artifacts

The release includes generated candidate answers and correctness judgments,
which are required inputs to `run_training_and_inference.sh`. `gen_cands.sh` is provided for
reference if candidate generation is needed, but recreating final judgments
also requires the LLM Batch API workflow and supplemental human corrections.
`judge.py` records that judgment format and processing.

BioASQ data is included in this release. TriviaQA and Jeopardy are loaded from
their pinned Hugging Face revisions. Reproduction via
`run_training_and_inference.sh` still requires downloading the public
base-model checkpoints and the public GCM checkpoint, sufficient GPU hardware
(the 32B model may require distributed training), and enough scratch storage
for LoRA adapters.

The hidden-state variance table requires a large cached variance artifact built
from retained LoRA checkpoints and hidden-state streams, neither of which is
released. Therefore `latex.sh` emits a warning and skips that table when the
artifact is unavailable. The
`run_training_and_inference.sh` workflow runs `analyze_variance.py` and
makes it available.
