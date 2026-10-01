"""Build cached per-question hidden-state seed variances for LaTex tables.

Example:
    python -m conset.analyze_variance \
        --inference_config conset/inference_configs/olmo7b_variance.yaml
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import torch
import yaml

from .pred_confidence import load_eval_examples
from .train_confidence import (
    ConfidenceDataset,
    LeftPadCollator,
    adapter_results_root,
    get_adapter_dir,
    load_train_config,
    load_model_and_tokenizer,
    set_adapter_root,
)


def hidden_state_results_dir(dataset, model_name):
    """Return the directory for hidden-state artifacts of one model/dataset."""
    model_part = model_name.rsplit('/', 1)[-1]
    return Path('results') / 'conset' / dataset / model_part / 'hidden_states'


def load_within_question_seed_variances(variance_path):
    """Load cached per-question, across-seed variances from an analysis run.

    The returned payload's ``question_variances[revision]['variances']`` has
    shape ``(questions, layers)``.  Its entries are population variances over
    adapter seeds, summed over hidden dimensions.
    """
    variance_path = Path(variance_path)
    payload = torch.load(variance_path, weights_only=True)
    if not payload.get('revisions') or 'question_variances' not in payload:
        raise ValueError(f'{variance_path} has no per-question seed variances')
    for revision in payload['revisions']:
        try:
            variances = payload['question_variances'][revision]['variances']
        except KeyError as exc:
            raise ValueError(
                f'{variance_path} has no per-question variances for {revision}') from exc
        if variances.ndim != 2:
            raise ValueError(
                f'Expected (questions, layers) variance tensor for {revision}, '
                f'got shape {tuple(variances.shape)}')
    return payload


def question_variance_spearman_summary(question_variances, revisions):
    """Summarize question-wise variance-vs-checkpoint-order correlations.

    Returns ``(means, medians, finite_counts, n_shared_questions)``.  Each
    layer's mean and median are over questions whose Spearman correlation is
    defined (questions with constant variance across checkpoints are omitted).
    """
    from scipy.stats import rankdata
    import numpy as np

    if len(revisions) < 2:
        raise ValueError('Need at least two checkpoints for Spearman correlations')
    by_revision = []
    for q_idxs, variances in question_variances:
        by_revision.append({int(q_i): variances[row_i]
                            for row_i, q_i in enumerate(q_idxs.tolist())})
    common_q_idxs = set.intersection(*(set(values) for values in by_revision))
    if not common_q_idxs:
        raise ValueError('No questions are shared by every requested checkpoint')

    # Rank along the checkpoint axis for every question/layer simultaneously.
    # ``rankdata`` uses average ranks for ties, matching ``spearmanr``.
    ordered_q_idxs = sorted(common_q_idxs)
    values = torch.stack([
        torch.stack([per_revision[q_i] for q_i in ordered_q_idxs])
        for per_revision in by_revision
    ]).detach().cpu().numpy()  # checkpoints x questions x layers
    ranks = rankdata(values, axis=0)
    checkpoint_ranks = np.arange(len(revisions), dtype=np.float64)
    checkpoint_centered = checkpoint_ranks - checkpoint_ranks.mean()
    ranks_centered = ranks - ranks.mean(axis=0, keepdims=True)
    denominators = np.sqrt(
        np.square(checkpoint_centered).sum()
        * np.square(ranks_centered).sum(axis=0))
    correlations = np.divide(
        (checkpoint_centered[:, None, None] * ranks_centered).sum(axis=0),
        denominators, out=np.full_like(denominators, np.nan),
        where=denominators != 0)
    finite = np.isfinite(correlations)
    finite_counts = torch.from_numpy(finite.sum(axis=0))
    means = torch.from_numpy(np.nanmean(correlations, axis=0))
    medians = torch.from_numpy(np.nanmedian(correlations, axis=0))
    return means, medians, finite_counts, len(common_q_idxs)


class _HiddenStateReader:
    """Sequential reader for one streamed ``hidden_states.pt`` output."""

    def __init__(self, output_dir):
        output_dir = Path(output_dir)
        with open(output_dir / 'manifest.json') as f:
            self.manifest = json.load(f)
        self.n_records = self.manifest['n_records']
        self.file = open(output_dir / self.manifest['data_file'], 'rb')
        self.record_i = 0
        self.record = None
        self.offset = 0

    @property
    def n_hidden_states(self):
        return self.manifest['n_hidden_states']

    def ensure_record(self):
        if self.record is not None and self.offset < len(self.record['q_idxs']):
            return True
        if self.record_i == self.n_records:
            self.record = None
            return False
        self.record = torch.load(self.file, weights_only=True)
        self.record_i += 1
        self.offset = 0
        return True

    def close(self):
        self.file.close()


def _seed_variance_by_question(output_dirs, device):
    """Return per-question, per-layer variance across adapter seeds.

    Each value is the population variance across seeds, summed over hidden
    dimensions. The reduction is run on *device* and the retained results are
    kept on CPU for the table's checkpoint-order summary.
    """
    readers = [_HiddenStateReader(path) for path in output_dirs]
    try:
        n_layers = readers[0].n_hidden_states
        hidden_size = readers[0].manifest['hidden_size']
        if n_layers is None or hidden_size is None:
            raise ValueError(f'No hidden states in {output_dirs[0]}')
        for reader in readers[1:]:
            if (reader.n_hidden_states != n_layers
                    or reader.manifest['hidden_size'] != hidden_size):
                raise ValueError('Seed outputs disagree on hidden-state shape')

        q_idx_parts, variance_parts = [], []
        while True:
            available = [reader.ensure_record() for reader in readers]
            if not any(available):
                break
            if not all(available):
                raise ValueError('Seed outputs have different numbers of examples')
            n = min(
                len(reader.record['q_idxs']) - reader.offset
                for reader in readers)
            reference_q_idxs = readers[0].record['q_idxs'][
                readers[0].offset:readers[0].offset + n]
            for reader in readers[1:]:
                q_idxs = reader.record['q_idxs'][
                    reader.offset:reader.offset + n]
                if not torch.equal(q_idxs, reference_q_idxs):
                    raise ValueError(
                        'Seed outputs do not have the same question ordering')
            layer_variances = []
            for layer_i in range(n_layers):
                values = torch.stack([
                    reader.record['hidden_states'][layer_i][
                        reader.offset:reader.offset + n]
                    for reader in readers
                ]).to(device=device, dtype=torch.float32)
                # Sum each question's population seed variance across hidden
                # dimensions, retaining rather than immediately averaging the
                # question axis.
                layer_variances.append(
                    values.var(dim=0, correction=0).sum(dim=-1).double().cpu())
            q_idx_parts.append(reference_q_idxs.cpu())
            variance_parts.append(torch.stack(layer_variances, dim=1))
            for reader in readers:
                reader.offset += n
        if not q_idx_parts:
            raise ValueError('No examples found in hidden-state outputs')
        return torch.cat(q_idx_parts), torch.cat(variance_parts)
    finally:
        for reader in readers:
            reader.close()




def compute_hidden_state_variances(inference_config, *, dataset, train_config):
    """Compute the cached per-question seed variances consumed by LaTex."""
    plan = _parse_plan(inference_config, cli_force=False)
    train_config = train_config.removesuffix('.yaml')
    matching_adapters = [
        adapter for adapter in plan['adapters']
        if adapter['train_config'] == train_config
    ]
    if not matching_adapters:
        raise ValueError(f'{train_config!r} is not an adapter in {inference_config}')
    if dataset not in matching_adapters[0]['datasets']:
        raise ValueError(f'{train_config!r} is not configured for {dataset!r}')
    ranges = [(b, e) for name, b, e in plan['datasets'] if name == dataset]
    if len(ranges) != 1:
        raise ValueError(f'{dataset!r} must have exactly one range in the inference config')
    beg, end = ranges[0]

    seeds = plan['seeds']
    analysis_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Variance computation device: {analysis_device}', flush=True)
    variance_path = hidden_state_results_dir(
        dataset, plan['model_name']) / (
            f'{dataset}_{train_config}_hidden_state_seed_variance.pt')
    revisions = plan['revisions']
    if not revisions:
        raise ValueError('No checkpoints requested')
    variances, question_variances = [], []
    for revision in revisions:
            output_dirs = [
                _output_dir(dataset, plan['model_name'], revision, train_config,
                            seed, plan['epoch'], beg, end)
                for seed in seeds
            ]
            missing = [path / 'manifest.json' for path in output_dirs
                       if not (path / 'manifest.json').exists()]
            if missing:
                raise FileNotFoundError(
                    'Missing hidden-state output(s): '
                    + ', '.join(map(str, missing)))
            print(f'Computing variance for {revision}...', flush=True)
            q_idxs, per_question_variance = \
                _seed_variance_by_question(output_dirs, analysis_device)
            variances.append(per_question_variance.mean(dim=0))
            question_variances.append((q_idxs, per_question_variance))
    matrix = torch.stack(variances, dim=1)
    variance_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
            'variances': matrix,
            'question_variances': {
                revision: {'q_idxs': q_idxs, 'variances': per_question_variance}
                for revision, (q_idxs, per_question_variance)
                in zip(revisions, question_variances)
            },
            'revisions': list(revisions),
            'model_name': plan['model_name'],
            'dataset': dataset,
            'train_config': train_config,
            'eval_range': (beg, end),
            'seeds': seeds,
            'variance_definition': (
                'total variance across seeds, summed over hidden '
                'dimensions and then averaged over questions'),
        }, variance_path)
    print(f'Wrote {variance_path}', flush=True)
    return matrix


def _train_config_path(train_config):
    if train_config.endswith('.yaml'):
        return train_config if os.path.exists(train_config) else os.path.join(
            'conset/train_configs', train_config)
    return f'conset/train_configs/{train_config}.yaml'


def _output_dir(dataset_name, model_name, revision, train_config, seed, epoch,
                beg, end):
    model_part = model_name.rsplit('/', 1)[-1]
    return (adapter_results_root() / 'hidden_states' / dataset_name /
            model_part / revision / f'confidence_{train_config}' /
            f'seed{seed}' / f'epoch{epoch}' / f'{beg}-{end}')


def _parse_plan(config_path, cli_force):
    with open(config_path) as f:
        plan = yaml.safe_load(f)
    if not isinstance(plan, dict):
        raise ValueError(f'Inference config {config_path} must contain a mapping')
    model_name = plan.get('model_name')
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("Inference config requires a nonempty 'model_name'")
    revisions = plan.get('revisions', plan.get('eval_revisions'))
    if (not isinstance(revisions, list) or not revisions
            or not all(isinstance(revision, str) and revision
                       for revision in revisions)):
        raise ValueError(
            "Inference config requires a nonempty 'revisions' list "
            "(or the alias 'eval_revisions')")
    if len(set(revisions)) != len(revisions):
        raise ValueError("'revisions' must not contain duplicates")
    dataset_specs = plan.get('datasets')
    if not isinstance(dataset_specs, list) or not dataset_specs:
        raise ValueError("Inference config requires a nonempty 'datasets' list")
    datasets = []
    for spec in dataset_specs:
        if not isinstance(spec, dict):
            raise ValueError('Each dataset entry must be a mapping')
        try:
            dataset_name = spec['dataset']
            beg, end = int(spec['beg']), int(spec['end'])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                'Each dataset entry needs dataset, beg, and end') from exc
        if not isinstance(dataset_name, str) or not dataset_name or beg >= end:
            raise ValueError(f'Invalid dataset entry: {spec!r}')
        datasets.append((dataset_name, beg, end))
    adapters = plan.get('adapters')
    if not isinstance(adapters, list) or not adapters:
        raise ValueError("Inference config requires a nonempty 'adapters' list")
    dataset_names = {dataset_name for dataset_name, _, _ in datasets}
    adapter_specs = []
    config_cache = {}
    for spec in adapters:
        if not isinstance(spec, dict):
            raise ValueError('Each adapter entry must be a mapping')
        train_config = spec.get('train_config')
        if not isinstance(train_config, str) or not train_config:
            raise ValueError("Each adapter entry needs a nonempty 'train_config'")
        train_config = train_config.removesuffix('.yaml')
        if train_config not in config_cache:
            config_cache[train_config] = load_train_config(
                _train_config_path(train_config))
        cfg = config_cache[train_config]
        if cfg['weight_type'] != 'lora':
            raise ValueError(
                'Hidden-state inference currently supports LoRA adapters only; '
                f'{train_config} has weight_type={cfg["weight_type"]!r}')
        adapter_datasets = spec.get('datasets', cfg['dataset'])
        if (not isinstance(adapter_datasets, list) or not adapter_datasets
                or not all(isinstance(name, str) and name
                           for name in adapter_datasets)):
            raise ValueError(
                f'datasets for {train_config} must be a nonempty list')
        unknown_datasets = set(adapter_datasets) - dataset_names
        if unknown_datasets:
            raise ValueError(
                f'datasets for {train_config} are not declared in the plan: '
                f'{sorted(unknown_datasets)}')
        adapter_specs.append({
            'train_config': train_config,
            'cfg': cfg,
            'datasets': set(adapter_datasets),
        })

    batch_size = int(plan.get('batch_size', 1))
    if batch_size < 1:
        raise ValueError("'batch_size' must be positive")
    seeds = plan.get('seeds', [17])
    if (not isinstance(seeds, list) or not seeds
            or not all(isinstance(seed, int) for seed in seeds)):
        raise ValueError("'seeds' must be a nonempty list of integers")
    epoch = int(plan.get('epoch', 1))
    if epoch < 1:
        raise ValueError("'epoch' must be positive")
    gen_config_name = plan.get('gen_config_name', 'beam')
    is_instruct = plan.get('is_instruct', False)
    if not isinstance(is_instruct, bool):
        raise ValueError("'is_instruct' must be a boolean")
    return {
        'model_name': model_name,
        'revisions': revisions,
        'datasets': datasets,
        'adapters': adapter_specs,
        'seeds': seeds,
        'epoch': epoch,
        'batch_size': batch_size,
        'gen_config_name': gen_config_name,
        'is_instruct': is_instruct,
        'force': cli_force or bool(plan.get('force', False)),
    }


@torch.inference_mode()
def _write_hidden_states(model, dataset, collator, output_dir, *, batch_size,
                         manifest):
    """Run *dataset* and stream final-token post-block states into one file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    data_path = output_dir / 'hidden_states.pt'
    temp_data_path = output_dir / 'hidden_states.pt.tmp'
    n_layers = None
    hidden_size = None
    n_records = 0
    current_batch_size = batch_size
    start = 0
    with open(temp_data_path, 'wb') as data_file:
        while start < len(dataset):
            end = min(start + current_batch_size, len(dataset))
            features = [dataset[index] for index in range(start, end)]
            batch = collator(features)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            try:
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True)
            except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
                is_oom = (isinstance(exc, torch.cuda.OutOfMemoryError)
                          or 'out of memory' in str(exc).lower())
                if not is_oom or current_batch_size == 1:
                    raise
                torch.cuda.empty_cache()
                current_batch_size = max(current_batch_size // 2, 1)
                print(f'  OOM: reducing hidden-state batch_size to '
                      f'{current_batch_size}', flush=True)
                continue

            all_states = out.hidden_states
            if all_states is None:
                raise RuntimeError('Model did not return hidden_states')
            # Exclude hidden_states[0], the embedding/input state.
            states = all_states[1:]
            n_layers = len(states) if n_layers is None else n_layers
            if not states:
                raise RuntimeError('Model returned no post-block hidden states')
            if len(states) != n_layers:
                raise RuntimeError('Inconsistent number of hidden-state layers')

            valid = attention_mask.bool()
            positions = torch.arange(
                attention_mask.shape[1], device=attention_mask.device)
            last_positions = (valid * positions).argmax(dim=1)
            last_states = [
                state[
                    torch.arange(state.shape[0], device=state.device),
                    last_positions.to(state.device),
                ].detach().to(
                    device='cpu', dtype=torch.bfloat16)
                for state in states
            ]
            if hidden_size is None:
                hidden_size = last_states[0].shape[-1]
            elif any(state.shape[-1] != hidden_size for state in last_states):
                raise RuntimeError('Inconsistent hidden-state width')
            # The legacy serialization format supports concatenated torch.save
            # records, so this is one append-only file without accumulating all
            # hidden states in host RAM. Read ``n_records`` records in order.
            torch.save({
                'hidden_states': last_states,
                'q_idxs': batch['q_i'].cpu(),
            }, data_file, _use_new_zipfile_serialization=False)
            n_records += 1
            start = end
            print(f'  [{end}/{len(dataset)}] appended to hidden_states.pt',
                  flush=True)
            del (out, all_states, states, last_states,
                 input_ids, attention_mask)
    os.replace(temp_data_path, data_path)

    manifest.update({
        'n_examples': len(dataset),
        'n_saved_tokens': len(dataset),
        'n_hidden_states': n_layers,
        'hidden_size': hidden_size,
        'data_file': data_path.name,
        'n_records': n_records,
        'serialization': (
            'concatenated torch.save records using legacy serialization; '
            'load n_records records sequentially from data_file'),
    })
    manifest_path = output_dir / 'manifest.json'
    temp_path = output_dir / 'manifest.json.tmp'
    with open(temp_path, 'w') as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write('\n')
    os.replace(temp_path, manifest_path)


def run_inference_config(config_path, cli_force=False):
    """Run respective-checkpoint hidden-state inference from a YAML plan."""
    plan = _parse_plan(config_path, cli_force)
    pending_by_revision = defaultdict(list)
    for dataset_name, beg, end in plan['datasets']:
        for revision in plan['revisions']:
            for adapter in plan['adapters']:
                if dataset_name not in adapter['datasets']:
                    continue
                for seed in plan['seeds']:
                    output_dir = _output_dir(
                        dataset_name, plan['model_name'], revision,
                        adapter['train_config'], seed, plan['epoch'], beg, end)
                    manifest_path = output_dir / 'manifest.json'
                    if manifest_path.exists() and not plan['force']:
                        print(f'Skipping {dataset_name}/{revision}/'
                              f'{adapter["train_config"]}/seed{seed}: '
                              f'{manifest_path} exists', flush=True)
                        continue
                    adapter_dir = get_adapter_dir(
                        '+'.join(adapter['cfg']['dataset']), plan['model_name'],
                        revision, f'confidence_{adapter["train_config"]}',
                        f'seed{seed}', f'epoch{plan["epoch"]}', storage='read')
                    pending_by_revision[revision].append({
                        'dataset_name': dataset_name,
                        'beg': beg,
                        'end': end,
                        'output_dir': output_dir,
                        'adapter_dir': adapter_dir,
                        'train_config': adapter['train_config'],
                        'seed': seed,
                    })
    if not pending_by_revision:
        print('All requested hidden-state outputs already exist.', flush=True)
        return

    task_cache = {}
    for revision, work in pending_by_revision.items():
        print(f'=== Loading model once: {plan["model_name"]} / {revision} '
              f'for {len(work)} dataset/range group(s) ===', flush=True)
        model, tokenizer = load_model_and_tokenizer(plan['model_name'], revision)
        collator = LeftPadCollator(tokenizer.pad_token_id)
        try:
            examples_cache = {}
            from peft import PeftModel
            for item in work:
                dataset_name = item['dataset_name']
                beg, end = item['beg'], item['end']
                output_dir = item['output_dir']
                if (output_dir / 'manifest.json').exists() and plan['force']:
                    print(f'Overwriting {output_dir}', flush=True)
                cache_key = (dataset_name, beg, end, revision,
                             plan['gen_config_name'])
                if cache_key not in examples_cache:
                    examples_cache[cache_key] = load_eval_examples(
                        dataset_name, plan['model_name'], revision,
                        plan['gen_config_name'], beg, end,
                        task_cache=task_cache)
                task, examples, n_blacklisted = examples_cache[cache_key]
                print(f'=== {dataset_name} {beg}-{end}; checkpoint: {revision} ===',
                      flush=True)
                print(f'  {len(examples)} eval instances '
                      f'({n_blacklisted} blacklisted)', flush=True)
                print(f'  Adapter: {item["train_config"]}, seed={item["seed"]}',
                      flush=True)
                adapter_model = None
                try:
                    adapter_model = PeftModel.from_pretrained(
                        model, item['adapter_dir'])
                    dataset = ConfidenceDataset(
                        examples, tokenizer, task=task,
                        is_instruct=plan['is_instruct'])
                    manifest = {
                        'model_name': plan['model_name'],
                        'revision': revision,
                        'dataset': dataset_name,
                        'train_config': item['train_config'],
                        'seed': item['seed'],
                        'epoch': plan['epoch'],
                        'eval_range': [beg, end],
                        'gen_config_name': plan['gen_config_name'],
                        'is_instruct': plan['is_instruct'],
                        'dtype': 'bfloat16',
                        'hidden_state_convention': (
                            'hidden_states[0] is the output after transformer '
                            'block 1; the embedding/input state is excluded; '
                            'each row is the final prompt token that predicts '
                            'the confidence digit'),
                    }
                    _write_hidden_states(
                        adapter_model, dataset, collator, output_dir,
                        batch_size=plan['batch_size'], manifest=manifest)
                    print(f'  Wrote {output_dir / "manifest.json"}', flush=True)
                finally:
                    if adapter_model is not None:
                        model = adapter_model.unload()
                        del adapter_model
                        torch.cuda.empty_cache()
        finally:
            del model
            torch.cuda.empty_cache()
    print('\nDone.', flush=True)


def run_plan(config_path, *, force=False):
    """Generate hidden states and variance caches for every plan adapter/dataset."""
    run_inference_config(config_path, cli_force=force)
    plan = _parse_plan(config_path, cli_force=force)
    for adapter in plan['adapters']:
        for dataset, _beg, _end in plan['datasets']:
            if dataset not in adapter['datasets']:
                continue
            cache_path = hidden_state_results_dir(
                dataset, plan['model_name']) / (
                    f'{dataset}_{adapter["train_config"]}_'
                    'hidden_state_seed_variance.pt')
            if cache_path.exists() and not force:
                print(f'Skipping variance cache: {cache_path}', flush=True)
                continue
            compute_hidden_state_variances(
                config_path, dataset=dataset,
                train_config=adapter['train_config'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inference_config', required=True)
    parser.add_argument('--force', action='store_true',
                        help='Regenerate hidden states and variance caches')
    parser.add_argument('--adapter_root', default=os.environ.get('SCRATCH'),
                        help='Root for LoRA adapters and hidden states; defaults to $SCRATCH')
    args = parser.parse_args()
    set_adapter_root(args.adapter_root)
    run_plan(args.inference_config, force=args.force)


if __name__ == '__main__':
    main()
