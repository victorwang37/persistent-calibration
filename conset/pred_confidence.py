"""
Predict confidence on eval checkpoints' predictions using a trained LoRA
adapter. The adapter is applied to each eval checkpoint's base model.

Results are saved to the directory containing the trained model.

Usage:
    python -m conset.pred_confidence \
        --inference_config conset/pred_configs/olmo7b_ncand2_onedata.yaml
"""

import argparse
import os
from collections import defaultdict

import torch
import yaml

from .dataset_configs import get_dataset_config
from .train_confidence import (
    ConfidenceDataset,
    LeftPadCollator,
    forward_confidence,
    get_adapter_dir,
    get_single_digit_token_ids,
    load_group_data,
    load_model_and_tokenizer,
    load_train_config,
    set_adapter_root,
)


def build_eval_examples(qa_examples, group_strs, group_probs, judge_labels,
                        blacklisted_qis=None):
    """Build one eval example per question: the highest-genprob group's answer.

    Returns:
        examples: list of dicts with keys
            'question', 'answer', 'target', 'weight', 'q_i', 'judge_label'
        ('target' and 'weight' are dummies required by ConfidenceDataset.)
    """
    examples = []
    blacklisted_qis = set() if blacklisted_qis is None else blacklisted_qis
    for qi, (qa, groups) in enumerate(zip(qa_examples, group_strs)):
        # Leave this question at NaN / -1 in the saved full-range tensors.
        if qi in blacklisted_qis:
            continue
        if groups is None or len(groups) == 0:
            continue

        # Match accuracy/calibration evaluation: select the highest-probability
        # judged group, never an unjudged/padding slot.
        valid = judge_labels[qi, :len(groups)] != -1
        if not valid.any():
            continue
        probs = group_probs[qi, :len(groups)].clone()
        probs[~valid] = -float('inf')
        gi = probs.argmax().item()
        answer = groups[gi][0]  # representative answer from the group
        examples.append({
            'question': qa.question,
            'answer': answer,
            'target': 0.0,
            'weight': 1.0,
            'q_i': qi,
            'judge_label': judge_labels[qi, gi].item(),
        })
    return examples


def load_eval_examples(dataset_name, model_name, eval_revision,
                       gen_config_name, beg, end, *, task_cache=None):
    """Load one checkpoint's top-answer evaluation examples.

    ``task_cache`` is owned by the caller and is keyed by dataset/range.  It
    avoids rereading the QA dataset and recomputing blacklist indices when
    several checkpoints or adapters use the same evaluation range.

    Returns:
        ``(task, examples, n_blacklisted)``.
    """
    if beg >= end:
        raise ValueError(f'Invalid range for {dataset_name}: {beg}-{end}')
    if task_cache is None:
        task_cache = {}
    task_key = (dataset_name, beg, end)
    if task_key not in task_cache:
        task = get_dataset_config(dataset_name, ranges=[(beg, end)])
        qa_examples = task.load_examples()
        blacklist = task.get_blacklist()
        blacklisted_qis = {
            qi for qi in range(end - beg) if beg + qi in blacklist
        }
        task_cache[task_key] = (task, qa_examples, blacklisted_qis)
    task, qa_examples, blacklisted_qis = task_cache[task_key]
    _, group_strs, group_probs, judge_labels = load_group_data(
        dataset_name, model_name, eval_revision, gen_config_name,
        [(beg, end)], qa_examples=qa_examples)
    examples = build_eval_examples(
        qa_examples, group_strs, group_probs, judge_labels,
        blacklisted_qis=blacklisted_qis)
    return task, examples, len(blacklisted_qis)


@torch.no_grad()
def predict_confidences(model, dataset, collator, digit_token_ids, device,
                        batch_size):
    """Predict confidence for every example in dataset.

    On OOM, halves batch_size and retries.

    Returns a tensor of confidences aligned with dataset order.
    """
    model.eval()
    confs = []
    i = 0
    while i < len(dataset):
        end = min(i + batch_size, len(dataset))
        features = [dataset[j] for j in range(i, end)]
        batch = collator(features)
        try:
            _, prob = forward_confidence(
                model,
                batch['input_ids'].to(device),
                batch['attention_mask'].to(device),
                digit_token_ids)
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            if batch_size == 1:
                raise
            torch.cuda.empty_cache()
            batch_size = max(batch_size // 2, 1)
            print(f"  OOM: reducing eval batch_size to {batch_size}", flush=True)
            continue
        confs.append(prob.float().cpu())
        i = end
        if (i // batch_size) % 10 == 0:
            print(f"  [{i}/{len(dataset)}]", flush=True)

    confs = torch.cat(confs) if confs else torch.empty(0)
    return confs


def _train_config_path(train_config):
    """Return the on-disk YAML path for a config name or YAML filename."""
    if train_config.endswith('.yaml'):
        return train_config if os.path.exists(train_config) else os.path.join(
            'conset/train_configs', train_config)
    return f'conset/train_configs/{train_config}.yaml'


def _save_predictions(examples, confs, beg, end, out_path):
    """Scatter confidence predictions into the standard full-range file."""
    n = end - beg
    conf_full = torch.full((n,), float('nan'))
    label_full = torch.full((n,), -1, dtype=torch.long)
    for ex, conf in zip(examples, confs):
        conf_full[ex['q_i']] = conf
        label_full[ex['q_i']] = ex['judge_label']
    torch.save({
        'confs': conf_full,
        'judge_labels': label_full,
        'eval_range': (beg, end),
    }, out_path)


def run_inference_config(config_path, cli_force=False):
    """Run a YAML inference plan, loading each eval checkpoint only once.

    The plan is intentionally organized around adapters, while execution is
    organized around evaluation checkpoints.  Thus all pending adapters and
    datasets for one eval checkpoint share one loaded base model.  This mode
    is for LoRA adapters.
    """
    with open(config_path) as f:
        plan = yaml.safe_load(f)
    if not isinstance(plan, dict):
        raise ValueError(f'Inference config {config_path} must contain a mapping')

    model_name = plan.get('model_name')
    if not model_name:
        raise ValueError("Inference config is missing required key 'model_name'")
    if 'seed' in plan:
        raise ValueError("Use 'seeds', not deprecated 'seed'")
    default_revisions = plan.get('eval_revisions')
    if default_revisions is not None and not isinstance(default_revisions, list):
        raise ValueError("'eval_revisions' must be a list")
    default_seeds = plan.get('seeds', [17])
    if (not isinstance(default_seeds, list) or not default_seeds
            or not all(isinstance(seed, int) for seed in default_seeds)):
        raise ValueError("'seeds' must be a nonempty list of integers")
    default_epoch = int(plan.get('epoch', 1))
    default_batch_size = int(plan.get('batch_size', 32))
    default_gen_config = plan.get('gen_config_name', 'beam')
    default_is_instruct = bool(plan.get('is_instruct', False))
    default_respective_predictor = plan.get(
        'use_ckpt_respective_predictor', False)
    if not isinstance(default_respective_predictor, bool):
        raise ValueError("'use_ckpt_respective_predictor' must be a boolean")
    force = cli_force or bool(plan.get('force', False))

    dataset_specs = plan.get('datasets')
    adapters = plan.get('adapters')
    if not isinstance(dataset_specs, list) or not dataset_specs:
        raise ValueError("Inference config requires a nonempty 'datasets' list")
    if not isinstance(adapters, list) or not adapters:
        raise ValueError("Inference config requires a nonempty 'adapters' list")

    normalized_datasets = []
    for spec in dataset_specs:
        if not isinstance(spec, dict):
            raise ValueError("Each dataset entry must be a mapping")
        try:
            dataset, beg, end = spec['dataset'], int(spec['beg']), int(spec['end'])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Each dataset entry needs dataset, beg, and end") from exc
        if beg >= end:
            raise ValueError(f"Invalid range for {dataset}: {beg}-{end}")
        normalized_datasets.append((dataset, beg, end))
    dataset_names = {dataset for dataset, _, _ in normalized_datasets}

    # group -> pending adapter work.  Dict insertion order makes the plan's
    # order deterministic while still grouping every adapter behind one base
    # checkpoint load.
    grouped_work = defaultdict(list)
    config_cache = {}
    for adapter_spec in adapters:
        if not isinstance(adapter_spec, dict):
            raise ValueError("Each adapter entry must be a mapping")
        train_config = adapter_spec.get('train_config')
        revisions = adapter_spec.get('revisions')
        if not train_config or not isinstance(revisions, list) or not revisions:
            raise ValueError("Each adapter needs train_config and a nonempty revisions list")
        if 'seed' in adapter_spec:
            raise ValueError(f"Use 'seeds', not deprecated 'seed' for {train_config}")
        train_config = train_config.removesuffix('.yaml')
        if train_config not in config_cache:
            config_cache[train_config] = load_train_config(
                _train_config_path(train_config))
        cfg = config_cache[train_config]

        eval_revisions = adapter_spec.get('eval_revisions', default_revisions)
        if not isinstance(eval_revisions, list) or not eval_revisions:
            raise ValueError(
                f"Adapter {train_config} needs eval_revisions (or plan-level eval_revisions)")
        adapter_seeds = adapter_spec.get('seeds', default_seeds)
        if (not isinstance(adapter_seeds, list) or not adapter_seeds
              or not all(isinstance(seed, int) for seed in adapter_seeds)):
            raise ValueError(f'seeds for {train_config} must be a nonempty list of integers')
        epoch = int(adapter_spec.get('epoch', default_epoch))
        batch_size = int(adapter_spec.get('batch_size', default_batch_size))
        is_instruct = bool(adapter_spec.get('is_instruct', default_is_instruct))
        respective_predictor = adapter_spec.get(
            'use_ckpt_respective_predictor', default_respective_predictor)
        if not isinstance(respective_predictor, bool):
            raise ValueError(
                f'use_ckpt_respective_predictor for {train_config} '
                'must be a boolean')
        gen_config_name = adapter_spec.get('gen_config_name', default_gen_config)
        adapter_dataset_names = adapter_spec.get('datasets')
        if adapter_dataset_names is None:
            adapter_datasets = normalized_datasets
        else:
            if (not isinstance(adapter_dataset_names, list)
                    or not adapter_dataset_names
                    or not all(isinstance(dataset, str) and dataset
                               for dataset in adapter_dataset_names)):
                raise ValueError(
                    f'datasets for {train_config} must be a nonempty list '
                    'of dataset names')
            unknown_datasets = set(adapter_dataset_names) - dataset_names
            if unknown_datasets:
                raise ValueError(
                    f'datasets for {train_config} are not declared at the '
                    f'plan level: {sorted(unknown_datasets)}')
            adapter_datasets = [
                dataset_spec for dataset_spec in normalized_datasets
                if dataset_spec[0] in set(adapter_dataset_names)]
        train_ds_label = '+'.join(cfg['dataset'])
        revision_key = '_'.join(revisions)
        for seed in adapter_seeds:
            adapter_parts = (f'confidence_{train_config}', f'seed{seed}',
                             f'epoch{epoch}')
            adapter_dirs = {}
            for dataset, beg, end in adapter_datasets:
                for eval_rev in eval_revisions:
                    # Respective adapters are stored under the checkpoint on
                    # which each one was trained, just like the legacy CLI mode.
                    adapter_revision_key = (
                        eval_rev if respective_predictor else revision_key)
                    prediction_dir = get_adapter_dir(
                        train_ds_label, model_name, adapter_revision_key,
                        *adapter_parts, storage='home')
                    out_path = os.path.join(
                        prediction_dir,
                        f'pred_confs_{dataset}_{eval_rev}_{beg}-{end}.pt')
                    if os.path.exists(out_path) and not force:
                        print(f"Skipping {train_config} / {eval_rev}: {out_path} exists",
                              flush=True)
                        continue
                    # Do not require an adapter to still be on scratch when every
                    # one of its requested prediction files already exists.
                    if adapter_revision_key not in adapter_dirs:
                        adapter_dirs[adapter_revision_key] = get_adapter_dir(
                            train_ds_label, model_name, adapter_revision_key,
                            *adapter_parts, storage='read')
                    grouped_work[(dataset, beg, end, eval_rev)].append({
                        'train_config': train_config,
                        'adapter_dir': adapter_dirs[adapter_revision_key],
                        'out_path': out_path,
                        'batch_size': batch_size,
                        'is_instruct': is_instruct,
                        'gen_config_name': gen_config_name,
                    })

    if not grouped_work:
        print('All requested prediction files already exist.', flush=True)
        return

    # A checkpoint's tokenizer and base weights are independent of the eval
    # dataset.  Keep the base model resident while processing every dataset
    # and range that needs this checkpoint.
    work_by_eval_revision = defaultdict(list)
    for (dataset_name, beg, end, eval_rev), work_items in grouped_work.items():
        work_by_eval_revision[eval_rev].append(
            (dataset_name, beg, end, work_items))

    task_cache = {}
    from peft import PeftModel
    for eval_rev, dataset_work in work_by_eval_revision.items():
        print(f"=== Loading base model: {model_name} / {eval_rev} "
              f"for {len(dataset_work)} dataset/range group(s) ===", flush=True)
        model, tokenizer = load_model_and_tokenizer(model_name, eval_rev)
        digit_token_ids = get_single_digit_token_ids(tokenizer)
        collator = LeftPadCollator(tokenizer.pad_token_id)

        for dataset_name, beg, end, work_items in dataset_work:
            print(f"=== {dataset_name} {beg}-{end}; eval ckpt: {eval_rev}; "
                  f"{len(work_items)} adapter(s) ===", flush=True)
            gen_config_names = {item['gen_config_name'] for item in work_items}
            if len(gen_config_names) != 1:
                raise ValueError(
                    f"Adapters for {dataset_name} / {eval_rev} specify different "
                    "gen_config_name values; split them into separate inference plans")
            # All adapters in this group consume exactly the same generated answers.
            task, examples, n_blacklisted = load_eval_examples(
                dataset_name, model_name, eval_rev,
                work_items[0]['gen_config_name'], beg, end,
                task_cache=task_cache)
            print(f"  {len(examples)} eval instances "
                  f"({n_blacklisted} blacklisted)", flush=True)

            for item in work_items:
                os.makedirs(os.path.dirname(item['out_path']), exist_ok=True)
                if os.path.exists(item['out_path']) and force:
                    print(f"  Overwriting {item['out_path']}", flush=True)
                print(f"  Adapter: {item['train_config']}", flush=True)
                adapter_model = PeftModel.from_pretrained(model, item['adapter_dir'])
                dataset = ConfidenceDataset(
                    examples, tokenizer, task=task,
                    is_instruct=item['is_instruct'])
                device = next(adapter_model.parameters()).device
                confs = predict_confidences(
                    adapter_model, dataset, collator, digit_token_ids, device,
                    item['batch_size'])
                _save_predictions(examples, confs, beg, end, item['out_path'])
                print(f"  Saved to {item['out_path']}", flush=True)

                # unload() removes LoRA modules without merging their weights,
                # leaving this eval checkpoint ready for the next adapter.
                model = adapter_model.unload()
                del adapter_model
                torch.cuda.empty_cache()

        del model
        torch.cuda.empty_cache()

    print('\nDone.', flush=True)


def main():
    parser = argparse.ArgumentParser(
        description='Predict confidence from a YAML inference plan')
    parser.add_argument('--inference_config', type=str, required=True,
                        help='YAML plan for inference from multiple LoRA adapters')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing prediction files instead of '
                            'skipping them')
    parser.add_argument('--adapter_root', default=os.environ.get('SCRATCH'),
                        help='Root for LoRA adapters; defaults to $SCRATCH')
    args = parser.parse_args()
    set_adapter_root(args.adapter_root)
    run_inference_config(args.inference_config, cli_force=args.force)


if __name__ == "__main__":
    main()
