"""Generate LaTeX tables from saved confidence-prediction artifacts.

Usage:
    python -m conset.latex

Each public function generates one complete table type and writes it under
``conset/latex/``.
"""

import hashlib
import math
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from conset.evaluation import (
    average_pair_metrics,
    conset_pair_results,
    load_gcm_preds,
    load_genprob_preds,
    load_ensembled_preds,
    load_sc_surrogate_preds,
    parse_adapter_spec,
    pooled_full_metrics,
    subset_metrics,
)
from conset.model_revisions import (
    OLMO_MODEL,
    OLMO_32B_MODEL,
    MARIN_MODEL,
    evaluation_revisions_for_model,
    non_oracle_revisions_for_model,
)
from conset.analyze_variance import (
    hidden_state_results_dir,
    load_within_question_seed_variances,
    question_variance_spearman_summary,
)
from conset.dataset_configs.jeopardy import JeopardyConfig
from conset.dataset_configs.bioasq import BioASQConfig
from conset.utils import parse_ranges
from conset.utils import (
    apply_pava,
    compute_smooth_ece,
    fit_pava,
    paired_question_stratified_bootstrap_across_seeds_differences,
)


BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
BOOTSTRAP_RESAMPLES = 10000
# Each comparison uses this many worker processes; reduce it on a shared node.
BOOTSTRAP_N_JOBS = 64
BOOTSTRAP_SEED = 17
SEEDS = (17, 18, 19)
# When reporting the usual per-checkpoint-pair average, ignore contrast pairs
# too small to give a stable metric.  Pooled evaluation deliberately retains
# every available contrast instance.
MIN_CONSET_SIZE = 500
QWEN_MODEL = 'Qwen/Qwen3-8B'
QWEN_REVISION = 'main'
THREE_MODEL_SPECS = (
    (OLMO_MODEL, 'Olmo 3 7B'),
    (MARIN_MODEL, 'Marin 8B'),
    (OLMO_32B_MODEL, 'Olmo 3 32B'),
)
DATASET_DISPLAY_NAMES = {
    'triviaqa': 'TriviaQA',
    'jeopardy': 'Jeopardy',
    'bioasq': 'BioASQ',
}


def _min_conset_size(dataset):
    """Return the dataset-specific minimum contrast-pair size."""
    return 300 if dataset == 'bioasq' else MIN_CONSET_SIZE

EVAL_RANGES = {
    'triviaqa': ['5000-9961'],
    'jeopardy': ['15000-20000'],
    'bioasq': ['0-2719'],
}
# Historical non-oracle data used solely to fit the SC post-hoc PAVA.  These
# are training-set prefixes, deliberately separate from EVAL_RANGES.
POSTHOC_TRAIN_RANGES = {
    'triviaqa': ['0-5000'],
    'jeopardy': ['0-5000'],
}

PLOTS_DIR = Path('conset/plots')
LATEX_DIR = Path('conset/latex')
BOOTSTRAP_CACHE_DIR = Path('conset/bootstrap_differences')

# (name, source, LaTex label, higher-is-better)
TABLE_METRIC_SPECS = (
    ('delta0_bal', 'conset', r'$\Delta_0^b$ $\scriptstyle\uparrow$', True),
    ('delta0', 'conset', r'$\Delta_0$ $\scriptstyle\uparrow$', True),
    ('delta_bal', 'conset', r'$\Delta^b$ $\scriptstyle\uparrow$', True),
    ('delta', 'conset', r'$\Delta$ $\scriptstyle\uparrow$', True),
    ('conset_auc', 'conset', r'AUC $\scriptstyle\uparrow$', True),
    ('conset_bs', 'conset', r'BS $\scriptstyle\downarrow$', False),
    ('conset_ece', 'conset', r'ECE $\scriptstyle\downarrow$', False),
    ('full_auc', 'full', r'AUC $\scriptstyle\uparrow$', True),
    ('full_bs', 'full', r'BS $\scriptstyle\downarrow$', False),
    ('full_ece', 'full', r'ECE $\scriptstyle\downarrow$', False),
)
TABLE_METRIC_SPEC_BY_NAME = {spec[0]: spec for spec in TABLE_METRIC_SPECS}
TABLE_METRICS = tuple(spec[0] for spec in TABLE_METRIC_SPECS)
SIGTEST_METRICS = TABLE_METRICS
# SIGTEST_METRICS = ()  # disable sigtesting for all metrics


def _bootstrap_difference_cache_path(reference_by_seed, candidate_by_seed, *,
                                     metric, aggregation, higher_is_better):
    """Return a content-addressed path for one bootstrap difference vector."""
    digest = hashlib.sha256(b'question-bootstrap-differences-v1')
    digest.update(repr((metric, aggregation, higher_is_better,
                        BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)).encode())
    for side_name, components_by_seed in (
            ('reference', reference_by_seed), ('candidate', candidate_by_seed)):
        digest.update(side_name.encode())
        digest.update(str(len(components_by_seed)).encode())
        for components in components_by_seed:
            digest.update(str(len(components)).encode())
            for component in components:
                for key in ('class_by_question', 'question_to_row', 'labels',
                            'confs', 'pair_scores', 'class_weights'):
                    digest.update(key.encode())
                    if key not in component:
                        digest.update(b'<missing>')
                        continue
                    value = component[key]
                    if torch.is_tensor(value):
                        value = value.detach().cpu().numpy()
                    value = np.ascontiguousarray(value)
                    digest.update(str((value.dtype.str, value.shape)).encode())
                    digest.update(value.tobytes())
    return BOOTSTRAP_CACHE_DIR / f'{digest.hexdigest()}.npy'

def _method_spec(model_name, train_dataset, oracle, multi_ckpt, n_cand, seed,
                 config_suffix=''):
    """Return the adapter-loading arguments for one Slurm-array method."""
    config = f'lora_{train_dataset}_acc_lr2e-4_bs16_5k'
    if n_cand == 2:
        config += '_ncand2'
    config += config_suffix

    if oracle:
        revisions = evaluation_revisions_for_model(model_name)
    elif multi_ckpt:
        revisions = non_oracle_revisions_for_model(model_name)
    else:
        revisions = (non_oracle_revisions_for_model(model_name)[-1],)

    is_respective = oracle and not multi_ckpt
    adapter = f'{config}:1:{seed}'
    if not is_respective:
        adapter += ':' + ','.join(revisions)
    return adapter, is_respective


def _conset_ece_from_pairs(pair_results):
    """Compute table ECE from the first checkpoint in each pair."""
    confs, labels = [], []
    for pair in pair_results:
        n = pair['n']
        confs.append(pair['confs'][:n])
        labels.append(pair['labels'][:n])
    return compute_smooth_ece(torch.cat(confs), torch.cat(labels))


def _full_set_ece(preds, revisions):
    """Compute full-set ECE as the mean of per-checkpoint ECEs."""
    per_checkpoint = [
        subset_metrics(*preds[revision], proportion=1.0)
        for revision in revisions
    ]
    if any(metrics is None for metrics in per_checkpoint):
        raise ValueError('No valid full-set predictions for a checkpoint')
    return sum(compute_smooth_ece(metrics['confs'], metrics['labels'])
               for metrics in per_checkpoint) / len(per_checkpoint)


def _pair_confidence_gaps(pair_results):
    """Average within-question confidence gaps over contrast checkpoint pairs.

    Each pair contributes equally. Gap magnitudes use the conditioned
    confidence on the correct answer minus that on the incorrect answer;
    importantly, those two answers are from the same question. The signed
    ``Delta_0`` variants instead use raw pair comparisons, which are
    definitionally invariant to an order-preserving BTL transform.
    """
    valid_pairs = [pair for pair in pair_results.values() if pair is not None]
    if not valid_pairs:
        raise ValueError('No checkpoint pair has a nonempty contrast set')
    signed_means, signed_balanced_means = [], []
    means, balanced_means = [], []
    for pair in valid_pairs:
        n = pair['n']
        beg_confs, end_confs = pair['confs'][:n], pair['confs'][n:]
        beg_correct = pair['labels'][:n] == 1
        gaps = torch.where(beg_correct, beg_confs - end_confs,
                           end_confs - beg_confs)
        signed_gaps = 2 * pair['pair_correct'].to(gaps.dtype) - 1
        signed_means.append(signed_gaps.mean().item())
        means.append(gaps.mean().item())
        weights = pair['class_weights'].to(dtype=gaps.dtype, device=gaps.device)
        signed_balanced_means.append(
            (weights * signed_gaps).sum().item() / weights.sum().item())
        balanced_means.append((weights * gaps).sum().item() / weights.sum().item())
    return (
        sum(signed_means) / len(signed_means),
        sum(signed_balanced_means) / len(signed_balanced_means),
        sum(means) / len(means),
        sum(balanced_means) / len(balanced_means),
    )


def _full_question_bootstrap_components(preds, revisions):
    """Build per-checkpoint full-set components on one compact universe.

    The universe is the shared set of valid top-answer positions.  It should
    be precisely the post-blacklist evaluation questions; assert that no
    checkpoint has an additional invalid position instead of silently
    weakening the intended question-level coupling.
    """
    valid_by_revision = []
    for revision in revisions:
        _confs, labels = preds[revision]
        valid_by_revision.append(
            (labels == 1) | (labels == 2) | (labels == 3))
    universe = valid_by_revision[0]
    for revision, valid in zip(revisions[1:], valid_by_revision[1:]):
        if not torch.equal(valid, universe):
            raise ValueError(
                'Question bootstrap requires identical valid top-answer '
                f'positions across checkpoints; {revision} differs')
    if not universe.any():
        raise ValueError('Question bootstrap has no non-blacklisted questions')

    components = []
    for revision in revisions:
        confs, labels = preds[revision]
        binary_labels = (labels[universe] == 1).float()
        n_questions = len(binary_labels)
        components.append({
            'class_by_question': binary_labels.to(torch.int8),
            'question_to_row': torch.arange(n_questions),
            'confs': confs[universe].float(),
            'labels': binary_labels,
        })
    return components, universe


def _conset_question_bootstrap_components(pair_results, universe):
    """Build one improvement/regression component for each retained pair."""
    raw_to_universe = torch.full((len(universe),), -1, dtype=torch.long)
    raw_to_universe[universe] = torch.arange(int(universe.sum()))
    components = []
    for pair_name, pair in pair_results.items():
        if pair is None:
            continue
        raw_questions = pair['question_indices']
        questions = raw_to_universe[raw_questions]
        if (questions < 0).any():
            raise ValueError(
                f'Contrast pair {pair_name} includes a question outside the '
                'post-blacklist bootstrap universe')
        n = pair['n']
        if len(questions) != n:
            raise ValueError(f'Contrast pair {pair_name} has inconsistent size')
        classes = torch.full((int(universe.sum()),), -1, dtype=torch.int8)
        # Class 0 = regression; class 1 = improvement.
        classes[questions[pair['regressed']]] = 0
        classes[questions[pair['improved']]] = 1
        rows = torch.full((int(universe.sum()),), -1, dtype=torch.long)
        rows[questions] = torch.arange(n)
        pair_confs = torch.stack([pair['confs'][:n], pair['confs'][n:]], dim=1)
        pair_labels = torch.stack([pair['labels'][:n], pair['labels'][n:]], dim=1)
        components.append({
            'class_by_question': classes,
            'question_to_row': rows,
            # Conditioning was already applied by conset_pair_results. These
            # are deliberately frozen rather than re-fit per bootstrap draw.
            'confs': pair_confs.float(),
            'labels': pair_labels.float(),
            # Delta_0 uses the raw, signed within-question comparison, while
            # Delta uses the conditioned confidence gap.  The fixed class
            # weights make each direction contribute equally for Delta^b.
            'pair_scores': (2 * pair['pair_correct'] - 1).float(),
            'class_weights': pair['class_weights'].float(),
        })
    if not components:
        raise ValueError('No retained contrast pairs for question bootstrap')
    return components


def _conset_ece_question_bootstrap_components(pair_results, universe):
    """Build contrast ECE components from each pair's first checkpoint."""
    components = _conset_question_bootstrap_components(pair_results, universe)
    for component in components:
        component['confs'] = component['confs'][:, 0]
        component['labels'] = component['labels'][:, 0]
    return components


def _compute_metrics_from_preds(preds, eval_revisions, btl_term='odds',
                                min_conset_size=MIN_CONSET_SIZE):
    """Compute all displayed metrics from one revision -> predictions mapping."""
    full = pooled_full_metrics(preds, revisions=eval_revisions)
    full['full_ece'] = _full_set_ece(preds, eval_revisions)
    full['full_auc'] = full.pop('auc')
    full['full_bs'] = full.pop('bs')
    full.pop('ece')
    pairs = list(combinations(eval_revisions, 2))
    pair_results = conset_pair_results(
        preds, balance_classes=False, condition_on_conset=True,
        btl_term=btl_term, pairs=pairs)
    ignored_pairs = [
        pair_name for pair_name, pair in pair_results.items()
        if pair is not None and pair['n'] < min_conset_size
    ]
    if ignored_pairs:
        print(f'Ignoring {len(ignored_pairs)} contrast pair(s) smaller '
              f'than MIN_CONSET_SIZE={min_conset_size}: '
              + ', '.join(f'{beg} × {end}' for beg, end in ignored_pairs))
        pair_results = {
            pair_name: (None if pair_name in ignored_pairs else pair)
            for pair_name, pair in pair_results.items()
        }
    if not any(pair is not None for pair in pair_results.values()):
        raise ValueError(
            f'No contrast checkpoint pairs remain after applying '
            f'MIN_CONSET_SIZE={min_conset_size}')
    conset = average_pair_metrics(pair_results, ['auc', 'bs'])
    valid_pairs = [pair for pair in pair_results.values() if pair is not None]
    conset['conset_ece'] = sum(
        _conset_ece_from_pairs([pair]) for pair in valid_pairs) / len(valid_pairs)
    pair_metrics = _pair_confidence_gaps(pair_results)
    (conset['delta0'], conset['delta0_bal'],
     conset['delta'], conset['delta_bal']) = pair_metrics
    conset['conset_auc'] = conset.pop('auc')
    conset['conset_bs'] = conset.pop('bs')

    # Retain compact, question-indexed data for paired significance tests.
    # The displayed confidences in pair_results are already conditioned on
    # contrast membership; bootstrap resamples gather these fixed values and
    # never re-fit a temperature or other conditioning transform.
    full_components, universe = _full_question_bootstrap_components(
        preds, eval_revisions)
    full['_question_bootstrap_components'] = full_components
    conset['_question_bootstrap_components'] = \
        _conset_question_bootstrap_components(pair_results, universe)
    conset['_question_bootstrap_ece_components'] = \
        _conset_ece_question_bootstrap_components(pair_results, universe)
    return full, conset


def _load_method_preds(model_name, train_dataset, eval_dataset,
                       oracle, multi_ckpt, n_cand, seed,
                       use_surrogate=False, config_suffix=''):
    """Load predictions for one table method and its evaluation revisions."""
    adapter, is_respective = _method_spec(
        model_name, train_dataset, oracle, multi_ckpt, n_cand, seed,
        config_suffix=config_suffix)
    eval_revisions = list(evaluation_revisions_for_model(model_name))
    eval_ranges = parse_ranges(EVAL_RANGES[eval_dataset])
    preds = load_ensembled_preds(
        [adapter], eval_dataset, model_name, 'beam', eval_revisions,
        eval_ranges, train_dataset=train_dataset,
        use_ckpt_respective_predictor=is_respective,
        use_surrogate=use_surrogate)
    return preds, eval_revisions


def _compute_method_metrics(model_name, train_dataset, eval_dataset,
                            oracle, multi_ckpt, n_cand, seed, use_surrogate=False,
                            config_suffix='', btl_term='odds'):
    preds, eval_revisions = _load_method_preds(
        model_name, train_dataset, eval_dataset, oracle, multi_ckpt, n_cand,
        seed, use_surrogate=use_surrogate, config_suffix=config_suffix,
    )
    return _compute_metrics_from_preds(
        preds, eval_revisions,
        btl_term=btl_term, min_conset_size=_min_conset_size(eval_dataset))


def _eligible_conset_question_mask(preds, eval_revisions,
                                   min_conset_size=MIN_CONSET_SIZE):
    """Return questions appearing in any pair retained by ``MIN_CONSET_SIZE``."""
    first_labels = preds[eval_revisions[0]][1]
    question_mask = torch.zeros_like(first_labels, dtype=torch.bool)
    retained_pairs = 0
    for left_revision, right_revision in combinations(eval_revisions, 2):
        left_labels = preds[left_revision][1]
        right_labels = preds[right_revision][1]
        left_correct = left_labels == 1
        right_correct = right_labels == 1
        left_wrong = (left_labels == 2) | (left_labels == 3)
        right_wrong = (right_labels == 2) | (right_labels == 3)
        contrast_mask = ((left_correct & right_wrong)
                         | (left_wrong & right_correct))
        if int(contrast_mask.sum()) >= min_conset_size:
            question_mask |= contrast_mask
            retained_pairs += 1
    if not retained_pairs:
        raise ValueError(
            f'No contrast pairs remain after applying MIN_CONSET_SIZE='
            f'{min_conset_size}')
    return question_mask, retained_pairs


def make_gcm_plot(*, seeds=SEEDS):
    """Plot GCM's contrast-AUC deficit against contrast-question error rate.

    Each dataset is a line containing one point per base model.  The
    x-coordinate is vanilla Qwen3-8B's top-answer error rate on questions
    occurring in at least one retained contrast pair for that target model.
    The vertical coordinate is oracle multi-checkpoint, two-candidate
    contrast AUC minus the GCM contrast AUC; both use the table's usual
    per-pair averaging.
    """
    path = Path('gcm_plot.png')
    seeds = list(seeds)
    if not seeds:
        raise ValueError('seeds must be nonempty')

    model_specs = THREE_MODEL_SPECS
    datasets = [(dataset, DATASET_DISPLAY_NAMES[dataset])
                for dataset in ('triviaqa', 'jeopardy')]
    points_by_dataset = {}
    for dataset, _dataset_label in datasets:
        ranges = parse_ranges(EVAL_RANGES[dataset])
        _, qwen_labels = load_genprob_preds(
            dataset, QWEN_MODEL, QWEN_REVISION, 'beam', ranges)
        points = []
        for model_name, model_label in model_specs:
            eval_revisions = list(evaluation_revisions_for_model(model_name))
            genprob_preds = load_ensembled_preds(
                None, dataset, model_name, 'beam', eval_revisions, ranges)
            question_mask, retained_pairs = _eligible_conset_question_mask(
                genprob_preds, eval_revisions, _min_conset_size(dataset))
            qwen_valid = question_mask & (
                (qwen_labels == 1) | (qwen_labels == 2) | (qwen_labels == 3))
            if not qwen_valid.any():
                raise ValueError(
                    f'Qwen has no valid answers on {dataset} contrast questions')
            error_rate = 1 - (qwen_labels[qwen_valid] == 1).float().mean().item()

            oracle_per_seed = [
                _compute_method_metrics(
                    model_name, dataset, dataset, oracle=True, multi_ckpt=True,
                    n_cand=2, seed=seed)
                for seed in seeds
            ]
            _, oracle_conset = (
                _average_metric_dicts([full for full, _conset in oracle_per_seed]),
                _average_metric_dicts([conset for _full, conset in oracle_per_seed]),
            )
            _, gcm_conset = _compute_gcm_metrics(
                model_name, dataset, eval_revisions)
            auc_gap = oracle_conset['conset_auc'] - gcm_conset['conset_auc']
            oracle_components_by_seed = [
                conset['_question_bootstrap_components']
                for _full, conset in oracle_per_seed]
            gcm_components_by_seed = [
                gcm_conset['_question_bootstrap_components']]
            bootstrap_differences = \
                paired_question_stratified_bootstrap_across_seeds_differences(
                    oracle_components_by_seed, gcm_components_by_seed,
                    metric='auc', aggregation='mean',
                    higher_is_better=True,
                    confidence_level=BOOTSTRAP_CONFIDENCE_LEVEL,
                    n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED,
                    n_jobs=BOOTSTRAP_N_JOBS,
                    cache_path=_bootstrap_difference_cache_path(
                        oracle_components_by_seed, gcm_components_by_seed,
                        metric='auc', aggregation='mean',
                        higher_is_better=True))
            ci_low, ci_high = np.quantile(
                bootstrap_differences,
                [(1 - BOOTSTRAP_CONFIDENCE_LEVEL) / 2,
                 1 - (1 - BOOTSTRAP_CONFIDENCE_LEVEL) / 2])
            significant = not (ci_low <= 0 <= ci_high)
            print(f'{dataset} / {model_label}: error_rate={error_rate:.4f}, '
                  f'oracle_minus_gcm_auc={auc_gap:.4f}, '
                  f'95%_CI=[{ci_low:.4f}, {ci_high:.4f}], '
                  f'two_sided_significant={significant}, '
                  f'retained_pairs={retained_pairs}', flush=True)
            points.append((model_label, error_rate, auc_gap, ci_low, ci_high))
        points_by_dataset[dataset] = points

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axis = plt.subplots(figsize=(6.5, 4.5))
    model_markers = {'Olmo 3 7B': 'o', 'Marin 8B': 's', 'Olmo 3 32B': '^'}
    dataset_handles = []
    for dataset, dataset_label in datasets:
        points = points_by_dataset[dataset]
        x_values = [point[1] for point in points]
        y_values = [point[2] for point in points]
        line, = axis.plot(x_values, y_values, label=dataset_label)
        dataset_handles.append(Line2D(
            [], [], color=line.get_color(), linewidth=line.get_linewidth(),
            label=dataset_label))
        for model_label, x_value, y_value, ci_low, ci_high in points:
            axis.errorbar(
                x_value, y_value,
                yerr=[[y_value - ci_low], [ci_high - y_value]],
                fmt='none', color=line.get_color(), capsize=3,
                linewidth=1.25, zorder=2)
            axis.scatter(x_value, y_value, color=line.get_color(), s=100,
                         marker=model_markers[model_label], zorder=3)
    axis.axhline(0, color='black', linewidth=1.5, alpha=0.5)
    axis.set_xlabel('Error rate of GCM backbone on contrast questions',fontsize=14)
    axis.set_ylabel('Contrast-set AUC gap:\nOracle minus GCM', fontsize=15)
    axis.tick_params(axis='both', labelsize=12)
    dataset_legend = axis.legend(handles=dataset_handles, title='Dataset',
                                 loc='upper left', fontsize=15, title_fontsize=15)
    axis.add_artist(dataset_legend)
    axis.legend(
        handles=[Line2D([], [], color='black', linestyle='None',
                        marker=marker, markersize=14, label=model_label)
                 for model_label, marker in model_markers.items()],
        title='Model', loc='lower right', fontsize=15, title_fontsize=15)
    axis.grid(alpha=0.25)
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.text(0.62, 0.95, 'Paired bootstrap 95% CIs shown',
              transform=axis.transAxes, ha='center', va='top',
              fontsize=8, color='0.45')
    fig.tight_layout()
    output_path = PLOTS_DIR / path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight', pad_inches=0.02)
    pdf_path = output_path.with_suffix('.pdf')
    fig.savefig(pdf_path, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f'Wrote {output_path} and {pdf_path}')
    return output_path



def plot_full_auc_confusion_matrices(dataset, *, model_name=OLMO_MODEL,
                                     seed=17, revisions=None, fname=None):
    """Write ordinary and random-label transfer matrices side by side.

    Both matrices have seven evaluation-checkpoint columns.  Their axes are
    given the same width and equal-aspect cells; the six-row random matrix is
    north-anchored so its rows align with the first six rows of the ordinary
    seven-row matrix.
    """
    # Okabe-Ito orange identifies checkpoints with randomized labels.
    random_label_color = '#D55E00'
    if dataset not in EVAL_RANGES:
        raise ValueError(f'Unknown dataset {dataset!r}; expected one of '
                         f'{sorted(EVAL_RANGES)}')
    if revisions is None:
        if model_name != OLMO_MODEL:
            raise ValueError(
                'revisions must be provided for models other than OLMo-3-7B')
        non_oracle_revisions = non_oracle_revisions_for_model(OLMO_MODEL)
        revisions = non_oracle_revisions + evaluation_revisions_for_model(OLMO_MODEL)
    revisions = tuple(revisions)
    if len(revisions) < 2 or len(set(revisions)) != len(revisions):
        raise ValueError('revisions must contain at least two unique checkpoints')
    final_step_text = revisions[-1].removeprefix('stage1-step')
    if not final_step_text.isdigit():
        raise ValueError('Percentage checkpoint labels require numeric step revisions')
    final_step = int(final_step_text)

    if fname is None:
        fname = f'{dataset}_full_auc_transfer_combined.png'
    path = Path(fname)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('fname must be relative to conset/plots/')
    output_path = PLOTS_DIR / path
    eval_ranges = parse_ranges(EVAL_RANGES[dataset])

    def load_auc_matrix(adapter_specs):
        aucs = torch.empty((len(adapter_specs), len(revisions)), dtype=torch.float64)
        for train_i, (adapter, description) in enumerate(adapter_specs):
            for eval_i, eval_revision in enumerate(revisions):
                preds = load_ensembled_preds(
                    [adapter], dataset, model_name, 'beam', [eval_revision],
                    eval_ranges, train_dataset=dataset)
                metrics = subset_metrics(*preds[eval_revision], proportion=1.0)
                if metrics is None:
                    raise ValueError(
                        f'No valid full-set predictions for {description} at '
                        f'eval checkpoint {eval_revision}')
                aucs[train_i, eval_i] = metrics['auc']
        return aucs

    ordinary_config = f'lora_{dataset}_acc_lr2e-4_bs16_5k'
    ordinary_specs = [
        (f'{ordinary_config}:1:{seed}:{revision}',
         f'adapter trained on {revision}')
        for revision in revisions
    ]
    train_pairs = list(zip(revisions[:-1], revisions[1:]))
    random_config = f'{ordinary_config}_ncand2_random'
    random_specs = [
        (f'{random_config}:1:{seed}:{first}_{second}',
         f'adapter trained on {first}, {second}')
        for first, second in train_pairs
    ]
    ordinary_aucs = load_auc_matrix(ordinary_specs)
    random_aucs = load_auc_matrix(random_specs)

    def checkpoint_label(revision):
        step = revision.removeprefix('stage1-step')
        return (f'{round(100 * int(step) / final_step):.0f}%'
                if step.isdigit() else revision)

    eval_labels = [checkpoint_label(revision) for revision in revisions]
    random_train_labels = [
        f'{checkpoint_label(first)}, {checkpoint_label(second)}'
        for first, second in train_pairs
    ]

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Explicit axes rectangles keep the physical cell size identical.  The
    # right matrix has six rows, so its top is shared with the left matrix and
    # its bottom lands exactly one left-matrix cell higher.
    figure_width, figure_height = 16, 9
    matrix_width = 0.34
    matrix_height = matrix_width * figure_width / figure_height
    matrix_top = 0.80
    ordinary_rect = [0.10, matrix_top - matrix_height,
                     matrix_width, matrix_height]
    # Reserve the center gutter for the shared color scale.  The randomized
    # matrix is shifted right enough that its long row labels do not collide
    # with that scale.
    random_rect = [0.64,
                   matrix_top - matrix_height * len(train_pairs) / len(revisions),
                   matrix_width,
                   matrix_height * len(train_pairs) / len(revisions)]
    colorbar_rect = [0.458, ordinary_rect[1], 0.02, matrix_height]
    fig = plt.figure(figsize=(figure_width, figure_height))
    ordinary_ax = fig.add_axes(ordinary_rect)
    random_ax = fig.add_axes(random_rect)

    def draw_matrix(ax, aucs, x_labels, y_labels, y_label,
                    y_label_fontsize=16, highlight_superdiagonal=False,
                    color_second_y_label_part=False):
        from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
        from matplotlib.patches import Rectangle

        image = ax.imshow(aucs.numpy(), cmap='viridis', vmin=0.5, vmax=1.0)
        for row in range(aucs.shape[0]):
            for col in range(aucs.shape[1]):
                value = aucs[row, col].item()
                ax.text(col, row, f'{value:.3f}', ha='center', va='center',
                        color='white' if value < 0.72 else 'black',
                        fontsize=16)
        # Draw each border as a filled rim inside its cell, rather than a
        # centered stroke.  This avoids clipping at matrix boundaries and
        # lets adjacent black and red rims share an edge without overlap.
        border_width = 0.05

        def add_cell_rim(row, col, color):
            left, top = col - 0.5, row - 0.5
            for x, y, width, height in (
                    (left, top, 1, border_width),
                    (left, top + 1 - border_width, 1, border_width),
                    (left, top, border_width, 1),
                    (left + 1 - border_width, top, border_width, 1)):
                ax.add_patch(Rectangle(
                    (x, y), width, height, facecolor=color,
                    edgecolor='none', clip_on=True))

        diagonal_length = min(aucs.shape)
        for index in range(diagonal_length):
            add_cell_rim(index, index, 'black')
        if highlight_superdiagonal:
            for row in range(min(aucs.shape[0], aucs.shape[1] - 1)):
                add_cell_rim(row, row + 1, random_label_color)
        ax.set_xticks(range(len(x_labels)), x_labels, fontsize=13)
        ax.set_yticks(range(len(y_labels)))
        if color_second_y_label_part:
            ax.set_yticklabels([])
            for row, label in enumerate(y_labels):
                prefix, second_checkpoint = label.rsplit(', ', maxsplit=1)
                label_box = HPacker(
                    children=[
                        TextArea(f'{prefix}, ', textprops={'fontsize': 13}),
                        TextArea(second_checkpoint, textprops={
                            'fontsize': 13, 'color': random_label_color}),
                    ],
                    align='center', pad=0, sep=0)
                ax.add_artist(AnnotationBbox(
                    label_box, (-0.5, row), xycoords='data',
                    xybox=(-8, 0), boxcoords='offset points',
                    box_alignment=(1, 0.5), frameon=False,
                    annotation_clip=False))
        else:
            ax.set_yticklabels(y_labels, fontsize=13)
        ax.tick_params(axis='both', which='both', length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        ax.set_xlabel('Evaluation checkpoint', fontsize=16, labelpad=12)
        ax.set_ylabel(y_label, fontsize=y_label_fontsize)
        return image

    image = draw_matrix(
        ordinary_ax, ordinary_aucs, eval_labels, eval_labels,
        'Training checkpoint')
    draw_matrix(
        random_ax, random_aucs, eval_labels, random_train_labels, '',
        highlight_superdiagonal=True, color_second_y_label_part=True)
    # Keep this as a compact horizontal footnote rather than a second vertical
    # axis label: the colored row-label components identify the random-label
    # adapters, and a vertical note made the gap between the two matrices look
    # substantially heavier than the ordinary matrix's y-axis label.
    fig.text(random_rect[0] + random_rect[2] / 2,
             random_rect[1] - 0.045,
             'orange = random training labels', ha='center', va='top', fontsize=15,
             color=random_label_color)
    colorbar = fig.colorbar(image, cax=fig.add_axes(colorbar_rect))
    colorbar.set_label('Full-set AUC', rotation=90, labelpad=12, fontsize=15)
    colorbar.ax.tick_params(labelsize=15, length=0)
    colorbar.outline.set_visible(False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight', pad_inches=0.02)
    pdf_path = output_path.with_suffix('.pdf')
    fig.savefig(pdf_path, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f'Wrote {output_path} and {pdf_path}')
    return output_path


def _copy_confidences_to_target_labels(source_confs, target_preds,
                                       eval_revisions, source_name):
    """Pair one fixed confidence vector with every target checkpoint's labels."""
    preds = {}
    for revision in eval_revisions:
        _, target_labels = target_preds[revision]
        if source_confs.shape != target_labels.shape:
            raise ValueError(
                f'{source_name} source and target shapes differ at {revision}: '
                f'{tuple(source_confs.shape)} vs {tuple(target_labels.shape)}')
        preds[revision] = (source_confs, target_labels)
    return preds


def _compute_copy_metrics(model_name, train_dataset, eval_dataset,
                          oracle, multi_ckpt, n_cand, seed, btl_term='odds'):
    """Copy source-checkpoint confidences to all evaluation checkpoints.

    The source is the last checkpoint on which this method was trained. Copy
    prefers an explicit surrogate prediction artifact, but an ordinary source
    checkpoint prediction is equivalent for this baseline and is accepted as
    a fallback. Evaluation labels come directly from each checkpoint's
    generated-answer artifacts, so Copy never requires adapter predictions on
    later target checkpoints merely to recover their labels.
    """
    adapter, is_respective = _method_spec(
        model_name, train_dataset, oracle, multi_ckpt, n_cand, seed)
    if is_respective:
        # A respective predictor has no shared training-revision list.  Use
        # its last evaluation checkpoint as the source deterministically.
        source_revision = evaluation_revisions_for_model(model_name)[-1]
    else:
        _, _, _, train_revisions = parse_adapter_spec(adapter)
        assert train_revisions
        source_revision = train_revisions[-1]
    eval_ranges = parse_ranges(EVAL_RANGES[eval_dataset])
    try:
        source_preds = load_ensembled_preds(
            [adapter], eval_dataset, model_name, 'beam', [source_revision],
            eval_ranges, train_dataset=train_dataset,
            use_ckpt_respective_predictor=is_respective, use_surrogate=True)
    except FileNotFoundError as surrogate_error:
        print('Copy: no surrogate source prediction; falling back to ordinary '
              f'source prediction for {source_revision}', flush=True)
        try:
            source_preds = load_ensembled_preds(
                [adapter], eval_dataset, model_name, 'beam', [source_revision],
                eval_ranges, train_dataset=train_dataset,
                use_ckpt_respective_predictor=is_respective)
        except FileNotFoundError:
            raise surrogate_error
    source_confs, _ = source_preds[source_revision]

    eval_revisions = list(evaluation_revisions_for_model(model_name))
    target_preds = load_ensembled_preds(
        None, eval_dataset, model_name, 'beam', eval_revisions, eval_ranges)
    copy_preds = _copy_confidences_to_target_labels(
        source_confs, target_preds, eval_revisions, 'Copy')
    return _compute_metrics_from_preds(
        copy_preds, eval_revisions,
        btl_term=btl_term, min_conset_size=_min_conset_size(eval_dataset))


def _normalize_branch_methods(methods, argument_name):
    """Validate dict method specifications and make them membership-friendly.

    Dicts cannot themselves be members of a Python ``set``. The public API
    therefore accepts any iterable of dicts and normalizes it to a set of
    immutable ``(oracle, multi_ckpt, n_cand)`` tuples.
    """
    if methods is None:
        return set()
    normalized = set()
    required_keys = {'oracle', 'multi_ckpt', 'n_cand'}
    allowed_keys = required_keys | {'label'}
    for method in methods:
        if not isinstance(method, dict):
            raise TypeError(
                f'{argument_name} entries must be dicts, got {method!r}')
        if not required_keys.issubset(method) or set(method) - allowed_keys:
            raise ValueError(
                f'{argument_name} entries must contain {sorted(required_keys)}, '
                f'with optional label; got {sorted(method)}')
        if 'label' in method and not isinstance(method['label'], str):
            raise TypeError(f'{argument_name} label must be a string')
        oracle = method['oracle']
        multi_ckpt = method['multi_ckpt']
        n_cand = method['n_cand']
        if not isinstance(oracle, bool) or not isinstance(multi_ckpt, bool):
            raise TypeError(
                f'{argument_name} oracle and multi_ckpt values must be bools')
        if not isinstance(n_cand, int) or isinstance(n_cand, bool) or n_cand < 1:
            raise ValueError(
                f'{argument_name} n_cand must be a positive integer')
        normalized.add((oracle, multi_ckpt, n_cand))
    return normalized



def _average_metric_dicts(metric_dicts):
    """Average scalar metrics while retaining per-seed bootstrap components."""
    if len(metric_dicts) == 1:
        return metric_dicts[0]
    averaged = {
        key: sum(metrics[key] for metrics in metric_dicts) / len(metric_dicts)
        for key in metric_dicts[0]
        if not key.startswith('_')
    }
    for key in metric_dicts[0]:
        if key.startswith('_question_bootstrap'):
            averaged[f'{key}_by_seed'] = [metrics[key] for metrics in metric_dicts]
    return averaged


def _average_seed_metrics(seeds, compute_metrics):
    """Run one learned method per seed and average its displayed metrics."""
    per_seed = [compute_metrics(seed) for seed in seeds]
    full_metrics, conset_metrics = zip(*per_seed)
    return (_average_metric_dicts(full_metrics),
            _average_metric_dicts(conset_metrics))


def _compute_self_consistency_metrics(model_name, dataset, eval_revisions, btl_term='odds'):
    """Evaluate highest-generation-probability (self-consistency) confidence."""
    preds = load_ensembled_preds(
        None, dataset, model_name, 'beam', eval_revisions,
        parse_ranges(EVAL_RANGES[dataset]))
    return _compute_metrics_from_preds(
        preds, eval_revisions,
        btl_term=btl_term, min_conset_size=_min_conset_size(dataset))


def _compute_sc_copy_metrics(model_name, dataset, eval_revisions,
                             source_revision=None, btl_term='odds'):
    """Copy one earlier checkpoint's SC confidences to every eval checkpoint.

    Unlike SC surrogate, this does not score a target checkpoint's generated
    answer under the earlier checkpoint.  It simply assigns the earlier
    checkpoint's top-group probability to the corresponding question at every
    target checkpoint, while retaining each target checkpoint's own label.
    """
    if source_revision is None:
        source_revision = non_oracle_revisions_for_model(model_name)[-1]
    ranges = parse_ranges(EVAL_RANGES[dataset])
    print(f'Baseline: SC copy checkpoint {source_revision}')
    source_preds = load_ensembled_preds(
        None, dataset, model_name, 'beam', [source_revision], ranges)
    source_confs, _ = source_preds[source_revision]
    target_preds = load_ensembled_preds(
        None, dataset, model_name, 'beam', eval_revisions, ranges)
    preds = _copy_confidences_to_target_labels(
        source_confs, target_preds, eval_revisions, 'SC-copy')
    return _compute_metrics_from_preds(
        preds, eval_revisions,
        btl_term=btl_term, min_conset_size=_min_conset_size(dataset))


def _compute_sc_surrogate_metrics(model_name, dataset, eval_revisions,
                                  surrogate_revision=None, btl_term='odds'):
    """Evaluate target answers using one fixed checkpoint's SC mass."""
    if surrogate_revision is None:
        surrogate_revision = non_oracle_revisions_for_model(model_name)[-1]
    ranges = parse_ranges(EVAL_RANGES[dataset])
    print(f'Baseline: SC surrogate checkpoint {surrogate_revision}')
    preds = {
        revision: load_sc_surrogate_preds(
            dataset, model_name, surrogate_revision, revision, ranges)
        for revision in eval_revisions
    }
    return _compute_metrics_from_preds(
        preds, eval_revisions,
        btl_term=btl_term, min_conset_size=_min_conset_size(dataset))


def _compute_gcm_metrics(model_name, dataset, eval_revisions, btl_term='odds'):
    """Evaluate the external GCM on every target checkpoint's top answer."""
    ranges = parse_ranges(EVAL_RANGES[dataset])
    preds = {
        revision: load_gcm_preds(dataset, model_name, revision, ranges)
        for revision in eval_revisions
    }
    return _compute_metrics_from_preds(
        preds, eval_revisions,
        btl_term=btl_term, min_conset_size=_min_conset_size(dataset))


def _fit_pava_on_valid_predictions(preds, revisions, error_message):
    """Fit PAVA after pooling valid top-answer predictions across revisions."""
    fit_confs, fit_labels = [], []
    for revision in revisions:
        confs, labels = preds[revision]
        valid = (labels == 1) | (labels == 2) | (labels == 3)
        fit_confs.append(confs[valid].float())
        fit_labels.append((labels[valid] == 1).float())
    if not any(confs.numel() for confs in fit_confs):
        raise ValueError(error_message)
    _, pava_fn = fit_pava(torch.cat(fit_confs), torch.cat(fit_labels))
    return pava_fn


def _compute_posthoc_gcm_metrics(model_name, train_dataset, eval_dataset,
                                 eval_revisions, btl_term='odds'):
    """Fit PAVA on historical GCM outputs and apply it to evaluation outputs."""
    fit_ranges = parse_ranges(POSTHOC_TRAIN_RANGES[train_dataset])
    fit_revisions = non_oracle_revisions_for_model(model_name)
    fit_preds = {
        revision: load_gcm_preds(train_dataset, model_name, revision, fit_ranges)
        for revision in fit_revisions
    }
    pava_fn = _fit_pava_on_valid_predictions(
        fit_preds, fit_revisions,
        'No valid non-oracle GCM predictions available to fit PAVA')

    eval_ranges = parse_ranges(EVAL_RANGES[eval_dataset])
    calibrated_preds = {}
    for revision in eval_revisions:
        confs, labels = load_gcm_preds(
            eval_dataset, model_name, revision, eval_ranges)
        calibrated_preds[revision] = (apply_pava(confs.float(), pava_fn), labels)
    return _compute_metrics_from_preds(
        calibrated_preds, eval_revisions,
        btl_term=btl_term, min_conset_size=_min_conset_size(eval_dataset))


def _compute_posthoc_self_consistency_metrics(model_name, train_dataset,
                                              eval_dataset, eval_revisions,
                                              btl_term='odds'):
    """PAVA-calibrate SC from historical training-set checkpoints.

    The PAVA fit pools the top-answer generation probabilities and judge
    labels from every non-oracle checkpoint over the ``train_dataset`` prefix
    configured in ``POSTHOC_TRAIN_RANGES``. It is then fixed and applied to
    ``eval_dataset`` self-consistency confidences at each evaluation checkpoint;
    evaluation labels therefore do not enter the calibration fit.
    """
    try:
        fit_ranges = parse_ranges(POSTHOC_TRAIN_RANGES[train_dataset])
    except KeyError as exc:
        supported = ', '.join(sorted(POSTHOC_TRAIN_RANGES))
        raise ValueError(
            f'SC post-hoc PAVA is not supported for train_dataset '
            f'{train_dataset!r}; supported: {supported}') from exc
    multi_revisions = non_oracle_revisions_for_model(model_name)
    fit_revisions = list(multi_revisions)
    fit_preds = load_ensembled_preds(
        None, train_dataset, model_name, 'beam', fit_revisions, fit_ranges)

    pava_fn = _fit_pava_on_valid_predictions(
        fit_preds, fit_revisions,
        'No valid non-oracle predictions available to fit SC PAVA')

    eval_preds = load_ensembled_preds(
        None, eval_dataset, model_name, 'beam', eval_revisions,
        parse_ranges(EVAL_RANGES[eval_dataset]))
    calibrated_preds = {
        revision: (apply_pava(confs.float(), pava_fn), labels)
        for revision, (confs, labels) in eval_preds.items()
    }
    return _compute_metrics_from_preds(
        calibrated_preds, eval_revisions,
        btl_term=btl_term, min_conset_size=_min_conset_size(eval_dataset))


def _compute_end_correct_baseline_metrics(model_name, dataset, eval_revisions, btl_term='odds'):
    """Evaluate the baseline that linearly increases with checkpoint order."""
    source_preds = load_ensembled_preds(
        None, dataset, model_name, 'beam', eval_revisions,
        parse_ranges(EVAL_RANGES[dataset]))
    assert len(eval_revisions) > 1

    # The checkpoint order is the model-specific evaluation order: the first
    # checkpoint receives 0, the last receives 1, and intermediate checkpoints
    # are equally spaced between them.
    preds = {}
    for revision_i, revision in enumerate(eval_revisions):
        confs, labels = source_preds[revision]
        confidence = revision_i / (len(eval_revisions) - 1)
        preds[revision] = (confs * 0 + confidence, labels)
    return _compute_metrics_from_preds(
        preds, eval_revisions,
        btl_term=btl_term, min_conset_size=_min_conset_size(dataset))


def _fmt(value, bold=False, underline=False, include_leading_zeros=True):
    if value is None or math.isnan(value):
        return '--'
    text = f'{value:.3f}'
    if text == '-0.000':
        text = '0.000'
    if not include_leading_zeros:
        text = text.removeprefix('0')
    if bold:
        text = rf'\textbf{{{text}}}'
    if underline:
        text = rf'\underline{{{text}}}'
    return text


def _table_metric_values(full, conset, metric_specs):
    """Return the requested displayed values in table-column order."""
    return [
        (conset if source == 'conset' else full)[name]
        if (conset if source == 'conset' else full) is not None else None
        for name, source, _, _ in metric_specs
    ]


def _is_best(value, values, maximize):
    """Whether *value* is tied for the best finite value in its block."""
    finite = [candidate for candidate in values
              if candidate is not None and not math.isnan(candidate)]
    if value is None or math.isnan(value) or not finite:
        return False
    best = max(finite) if maximize else min(finite)
    return math.isclose(value, best, abs_tol=1e-5)


def _question_metric_underlines(group_rows, group_values, metric_col, *,
                                source, maximize, metric, aggregation,
                                components_key, progress_label):
    """Return rows not significantly worse under the question bootstrap."""
    candidates = [
        (row_i, values[metric_col])
        for row_i, values in enumerate(group_values)
        if values[metric_col] is not None
        and not math.isnan(values[metric_col])
    ]
    if not candidates:
        return set(), {}
    best_i = (max if maximize else min)(candidates, key=lambda pair: pair[1])[0]
    data_i = 3 if source == 'full' else 4
    best_data = group_rows[best_i][data_i]
    def per_seed(data, key):
        by_seed_key = f'{key}_by_seed'
        return data[by_seed_key] if by_seed_key in data else [data[key]]

    best_components_by_seed = per_seed(best_data, components_key)
    underlines = set()
    total_tests = len(candidates) - 1
    completed_tests = 0
    print(f'  {progress_label} sigtests: 0/{total_tests}', end='', flush=True)
    for row_i, row in enumerate(group_rows):
        if row_i == best_i or row[data_i] is None:
            continue
        candidate_data = row[data_i]
        candidate_components_by_seed = per_seed(candidate_data, components_key)
        cache_path = _bootstrap_difference_cache_path(
            best_components_by_seed, candidate_components_by_seed,
            metric=metric, aggregation=aggregation,
            higher_is_better=maximize)
        bootstrap_differences = \
            paired_question_stratified_bootstrap_across_seeds_differences(
                best_components_by_seed, candidate_components_by_seed,
                metric=metric, aggregation=aggregation,
                higher_is_better=maximize,
                confidence_level=BOOTSTRAP_CONFIDENCE_LEVEL,
                n_resamples=BOOTSTRAP_RESAMPLES,
                seed=BOOTSTRAP_SEED,
                n_jobs=BOOTSTRAP_N_JOBS, cache_path=cache_path)
        is_significantly_better = bool(np.quantile(
            bootstrap_differences, 1 - BOOTSTRAP_CONFIDENCE_LEVEL) > 0)
        if not is_significantly_better:
            underlines.add(row_i)
        completed_tests += 1
        print(f' {completed_tests}/{total_tests}', end='', flush=True)
    print(flush=True)
    return underlines


def _make_table(model_specs, methods, fname, *, metrics, seeds, btl_term='odds',
                include_leading_zeros=True,
                include_pooldata=(), include_onedata=(),
                include_surrogate=(),
                include_copy=(),
                show_section_headers=True,
                dataset_specs=None, sections=None, model_layout='columns',
                block_label_with_first_regime=False, block_label_fn=None):
    """Shared renderer for one- and multi-model calibration tables.

    ``model_specs`` contains ``(model_name, display_name)`` tuples. With one
    tuple the model-header hierarchy is
    omitted; otherwise columns are model -> Contrast/Full -> metric.
    ``dataset_specs``, when supplied, contains ``(train_dataset,
    eval_dataset, display_name)`` tuples whose result blocks are stacked.
    ``model_layout='columns'`` gives each model its own column block. With
    ``model_layout='blocks'``, there is one metric column block and the
    model/dataset combinations are instead stacked as labeled row blocks.
    ``block_label_with_first_regime`` puts a block label in the metric cells
    of the first Non-oracle/Oracle regime header instead of on its own row.
    ``block_label_fn``, when supplied, maps ``(model_label, eval_dataset,
    dataset_label)`` to a custom row-block label.
    ``sections`` optionally supplies explicit ``{'label', 'methods'}`` groups
    for rendering and within-section comparisons.
    """
    unknown_sigtest_metrics = set(SIGTEST_METRICS) - set(TABLE_METRICS)
    if unknown_sigtest_metrics:
        raise ValueError(
            f'Unknown SIGTEST_METRICS: {sorted(unknown_sigtest_metrics)}')
    if model_layout not in {'columns', 'blocks'}:
        raise ValueError("model_layout must be 'columns' or 'blocks'")
    if block_label_with_first_regime and (sections is not None
                                          or not show_section_headers):
        raise ValueError(
            'block_label_with_first_regime requires regime section headers')
    if block_label_fn is not None and not callable(block_label_fn):
        raise TypeError('block_label_fn must be callable or None')
    path = Path(fname)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('fname must be a filename relative to conset/latex/')
    if not model_specs:
        raise ValueError('model_specs must be nonempty')
    if sections is None:
        section_labels = None
        method_entries = [(method, None) for method in methods]
    else:
        sections = list(sections)
        if not sections:
            raise ValueError('sections must be nonempty')
        section_labels, method_entries = [], []
        for section_i, section in enumerate(sections):
            if set(section) != {'label', 'methods'}:
                raise ValueError(
                    "sections entries must have exactly 'label' and 'methods'")
            if not isinstance(section['label'], str):
                raise TypeError('section label must be a string')
            section_methods = list(section['methods'])
            if not section_methods:
                raise ValueError('section methods must be nonempty')
            section_labels.append(section['label'])
            method_entries.extend((method, section_i) for method in section_methods)
    if dataset_specs is None:
        raise ValueError('dataset_specs must be provided')
    dataset_specs = list(dataset_specs)
    if not dataset_specs:
        raise ValueError('dataset_specs must be nonempty')
    n_display_models = len(model_specs) if model_layout == 'columns' else 1
    seeds = list(seeds)
    if not seeds:
        raise ValueError('seeds must contain at least one seed')
    metric_names = list(metrics)
    if not metric_names or len(set(metric_names)) != len(metric_names):
        raise ValueError('metrics must be a nonempty list of distinct names')
    unknown_metrics = set(metric_names) - set(TABLE_METRIC_SPEC_BY_NAME)
    if unknown_metrics:
        raise ValueError(f'Unknown table metrics: {sorted(unknown_metrics)}')
    metric_specs = [TABLE_METRIC_SPEC_BY_NAME[name] for name in metric_names]
    conset_specs = [spec for spec in metric_specs if spec[1] == 'conset']
    full_specs = [spec for spec in metric_specs if spec[1] == 'full']
    block_specs = conset_specs + full_specs

    baseline_labels = {
        'end_correct': 'End-correct baseline',
        'self_consistency': 'Self-consistency',
        'sc_copy': r'\,\raisebox{0.5ex}{\(\llcorner\)} Copy',
        'sc_surrogate': r'\,\raisebox{0.5ex}{\(\llcorner\)} Surrogate',
        'sc_posthoc': r'\,\raisebox{0.5ex}{\(\llcorner\)} Post-hoc',
        'gcm': 'GCM',
        'gcm_posthoc': r'\,\raisebox{0.5ex}{\(\llcorner\)} Post-hoc',
    }
    baseline_names = set(baseline_labels)
    learned_methods = []
    for method, _section_i in method_entries:
        if set(method) == {'baseline'}:
            if method['baseline'] not in baseline_names:
                raise ValueError(f"Unknown baseline {method['baseline']!r}")
        else:
            allowed_keys = {'oracle', 'multi_ckpt', 'n_cand', 'label'}
            extra_keys = set(method) - allowed_keys
            missing_keys = {'oracle', 'multi_ckpt', 'n_cand'} - set(method)
            if extra_keys or missing_keys:
                raise ValueError(
                    'learned methods must contain oracle, multi_ckpt, and n_cand, '
                    f'with optional label; got {sorted(method)}')
            core_method = {key: method[key]
                           for key in ('oracle', 'multi_ckpt', 'n_cand')}
            _normalize_branch_methods([core_method], 'methods')
            if 'label' in method and not isinstance(method['label'], str):
                raise TypeError('learned method label must be a string')
            learned_methods.append(core_method)
    if len(_normalize_branch_methods(learned_methods, 'methods')) != len(learned_methods):
        raise ValueError('methods contains duplicate learned method specifications')

    pooldata_methods = _normalize_branch_methods(include_pooldata, 'include_pooldata')
    onedata_methods = _normalize_branch_methods(include_onedata, 'include_onedata')
    surrogate_methods = _normalize_branch_methods(include_surrogate, 'include_surrogate')
    copy_methods = _normalize_branch_methods(include_copy, 'include_copy')

    # Each row describes how to compute its metrics and which learned section
    # it belongs to. Branch labels are intentionally independent of methods.
    rows = []
    for method, section_i in method_entries:
        if 'baseline' in method:
            rows.append({'kind': 'baseline', 'method': method,
                         'label': baseline_labels[method['baseline']],
                         'oracle': None, 'section': section_i})
            continue
        oracle, multi_ckpt, n_cand = (
            method['oracle'], method['multi_ckpt'], method['n_cand'])
        key = oracle, multi_ckpt, n_cand
        rows.append({'kind': 'method', 'method': method,
                     'label': method.get('label'),
                     'oracle': oracle, 'section': section_i})
        for kind, selected, label in (
                ('pooldata', pooldata_methods, r'\,\raisebox{0.5ex}{\(\llcorner\)} Pooled data'),
                ('onedata', onedata_methods, r'\,\raisebox{0.5ex}{\(\llcorner\)} Single-ckpt data'),
                ('surrogate', surrogate_methods, r'\,\raisebox{0.5ex}{\(\llcorner\)} Surrogate'),
                ('copy', copy_methods, r'\,\raisebox{0.5ex}{\(\llcorner\)} Copy')):
            if key in selected:
                rows.append({'kind': kind, 'method': method, 'label': label,
                             'oracle': oracle, 'section': section_i})
    def method_label(method):
        if 'label' in method:
            return method['label']
        oracle, multi_ckpt, n_cand = (
            method['oracle'], method['multi_ckpt'], method['n_cand'])
        checkpoints = ('Resp-ckpt' if oracle and not multi_ckpt else
                       'Multi-ckpt' if multi_ckpt else 'Single-ckpt')
        return f'{checkpoints}, ' + rf'$\text{{n}}_{{\text{{cand}}}}$={n_cand}'

    def compute(model_name, train_dataset, eval_dataset, row):
        kind, method = row['kind'], row['method']
        eval_revisions = list(evaluation_revisions_for_model(model_name))
        if kind == 'baseline':
            baseline = method['baseline']
            if baseline == 'end_correct':
                return _compute_end_correct_baseline_metrics(
                    model_name, eval_dataset, eval_revisions, btl_term)
            if baseline == 'self_consistency':
                return _compute_self_consistency_metrics(
                    model_name, eval_dataset, eval_revisions, btl_term)
            if baseline == 'sc_copy':
                return _compute_sc_copy_metrics(
                    model_name, eval_dataset, eval_revisions, btl_term=btl_term)
            if baseline == 'sc_surrogate':
                return _compute_sc_surrogate_metrics(
                    model_name, eval_dataset, eval_revisions, btl_term=btl_term)
            if baseline == 'sc_posthoc':
                return _compute_posthoc_self_consistency_metrics(
                    model_name, train_dataset, eval_dataset, eval_revisions,
                    btl_term=btl_term)
            if baseline == 'gcm':
                return _compute_gcm_metrics(
                    model_name, eval_dataset, eval_revisions, btl_term)
            return _compute_posthoc_gcm_metrics(
                model_name, train_dataset, eval_dataset, eval_revisions, btl_term)
        oracle, multi_ckpt, n_cand = (
            method['oracle'], method['multi_ckpt'], method['n_cand'])
        if kind == 'copy':
            return _average_seed_metrics(
                seeds, lambda seed: _compute_copy_metrics(
                    model_name, train_dataset, eval_dataset, oracle, multi_ckpt,
                    n_cand, seed, btl_term=btl_term))
        suffix = {'pooldata': '_pooldata',
                  'onedata': '_onedata'}.get(kind, '')
        return _average_seed_metrics(
            seeds, lambda seed: _compute_method_metrics(
                model_name, train_dataset, eval_dataset, oracle, multi_ckpt,
                n_cand, seed, use_surrogate=(kind == 'surrogate'),
                config_suffix=suffix, btl_term=btl_term))

    results = []
    for group_train_dataset, group_eval_dataset, group_label in dataset_specs:
        group_results = []
        for row in rows:
            per_model = []
            for model_name, model_label in model_specs:
                try:
                    per_model.append(compute(
                        model_name, group_train_dataset, group_eval_dataset, row))
                except FileNotFoundError as exc:
                    label = row['label'] or method_label(row['method'])
                    print(f'Warning: missing {model_label} predictions for '
                          f'{group_label or group_eval_dataset} {label}: {exc}',
                          file=sys.stderr)
                    per_model.append(None)
            group_results.append(per_model)
        results.append(group_results)

    rendered = [[[
        None if result is None else _table_metric_values(
            result[0], result[1], block_specs)
        for result in per_model
    ] for per_model in group_results] for group_results in results]
    metrics_per_model = len(block_specs)

    def comparison_indices(row_i):
        if section_labels is not None:
            return [i for i, row in enumerate(rows)
                    if row['section'] == rows[row_i]['section']]
        if not show_section_headers:
            return range(len(rows))
        if rows[row_i]['oracle'] is None:
            return ()
        return [i for i, row in enumerate(rows)
                if row['oracle'] == rows[row_i]['oracle']]

    underlines = {}
    if results:
        for group_i, group_results in enumerate(results):
            _group_train, group_eval_dataset, group_label = dataset_specs[group_i]
            progress_dataset = group_label or group_eval_dataset
            for model_i, (_model, model_label) in enumerate(model_specs):
                if section_labels is not None:
                    scopes = [[i for i, row in enumerate(rows)
                               if row['section'] == section_i]
                              for section_i in range(len(section_labels))]
                elif show_section_headers:
                    scopes = [
                        [i for i, row in enumerate(rows) if row['oracle'] is False],
                        [i for i, row in enumerate(rows) if row['oracle'] is True],
                    ]
                else:
                    scopes = [list(range(len(rows)))]
                for scope in scopes:
                    if not scope:
                        continue
                    fake_rows, values = [], []
                    for row_i in scope:
                        result = group_results[row_i][model_i]
                        fake_rows.append((None, None, None,
                                          None if result is None else result[0],
                                          None if result is None else result[1], None))
                        values.append(rendered[group_i][row_i][model_i] or
                                      [None] * metrics_per_model)
                    for metric_i, spec in enumerate(block_specs):
                        name, source, _label, maximize = spec
                        if name in SIGTEST_METRICS:
                            metric = {
                                'delta0': 'delta_0',
                                'delta0_bal': 'delta_0_bal',
                                'conset_ece': 'smoothece',
                                'full_ece': 'smoothece',
                            }.get(name, name.removeprefix(f'{source}_'))
                            aggregation = (
                                'pooled' if source == 'full' and metric != 'smoothece'
                                else 'mean')
                            components_key = (
                                '_question_bootstrap_ece_components'
                                if name == 'conset_ece'
                                else '_question_bootstrap_components')
                            tested_underlines = _question_metric_underlines(
                                fake_rows, values, metric_i,
                                source=source, maximize=maximize, metric=metric,
                                aggregation=aggregation, components_key=components_key,
                                progress_label=f'{progress_dataset} {model_label} {name}')
                            underlines.setdefault((group_i, model_i, metric_i), set()).update(
                                scope[local_i] for local_i in tested_underlines)

    column_spec = 'l' + ''.join(
        'c'
        for _display_model in range(n_display_models) for spec in block_specs)
    table = [
        r'\begin{table}[t]', rf'\caption{{{path.stem.replace("_", " ")}}}',
        rf'\label{{tab:{path.stem}}}', r'\begin{center}',
        r'\setlength{\tabcolsep}{8pt}', r'\small', r'\resizebox{\linewidth}{!}{%',
        r'\renewcommand{\arraystretch}{1.3}',
        rf'\begin{{tabular}}{{{column_spec}}}', r'\toprule',
    ]
    starts = [2 + model_i * metrics_per_model
              for model_i in range(n_display_models)]
    if model_layout == 'columns' and len(model_specs) > 1:
        table.append('& ' + ' & '.join(
            rf'\multicolumn{{{metrics_per_model}}}{{c}}{{{label}}}'
            for _model, label in model_specs) + r' \\')
        table.append(''.join(
            rf'\cmidrule(lr){{{start}-{start + metrics_per_model - 1}}}'
            for start in starts))
    level_headers, level_rules, metric_headers = [], [], []
    for start in starts:
        if conset_specs:
            level_headers.append(rf'\multicolumn{{{len(conset_specs)}}}{{c}}{{Contrast}}')
            level_rules.append(rf'\cmidrule(lr){{{start}-{start + len(conset_specs) - 1}}}')
        if full_specs:
            full_start = start + len(conset_specs)
            level_headers.append(rf'\multicolumn{{{len(full_specs)}}}{{c}}{{Full}}')
            level_rules.append(rf'\cmidrule(lr){{{full_start}-{full_start + len(full_specs) - 1}}}')
        metric_headers.extend(spec[2] for spec in block_specs)
    table.extend([
        '& ' + ' & '.join(level_headers) + r' \\', ''.join(level_rules),
        r'Method & ' + ' & '.join(metric_headers) + r' \\', r'\midrule',
    ])

    def append_row(group_i, row_i, display_model_indices):
        row, cells = rows[row_i], []
        for model_i in display_model_indices:
            result = results[group_i][row_i][model_i]
            if result is None:
                cells.extend(['--'] * metrics_per_model)
                continue
            for metric_i, (value, spec) in enumerate(
                    zip(rendered[group_i][row_i][model_i], block_specs)):
                candidates = [rendered[group_i][i][model_i][metric_i]
                              for i in comparison_indices(row_i)
                              if rendered[group_i][i][model_i] is not None]
                cells.append(_fmt(
                    value,
                    bold=(spec[3] is not None and _is_best(value, candidates, spec[3])),
                    underline=row_i in underlines.get(
                        (group_i, model_i, metric_i), set()),
                    include_leading_zeros=include_leading_zeros))
        label = row['label'] or method_label(row['method'])
        table.append(label + r' & ' + ' & '.join(cells) + r' \\')

    baseline_rows = [i for i, row in enumerate(rows) if row['oracle'] is None]
    learned_by_oracle = {
        oracle: [i for i, row in enumerate(rows) if row['oracle'] is oracle]
        for oracle in (False, True)
    }
    if model_layout == 'columns':
        display_blocks = [
            (group_i, list(range(len(model_specs))), dataset_label)
            for group_i, (_train_dataset, _eval_dataset, dataset_label)
            in enumerate(dataset_specs)
        ]
    else:
        display_blocks = [
            (group_i, [model_i],
             (block_label_fn(model_label, eval_dataset, dataset_label)
              if block_label_fn is not None else
              f'{model_label}, {dataset_label or eval_dataset}'))
            for model_i, (_model_name, model_label)
            in enumerate(model_specs)
            for group_i, (_train_dataset, eval_dataset, dataset_label)
            in enumerate(dataset_specs)
        ]
    for block_i, (group_i, display_model_indices, block_label) in enumerate(display_blocks):
        if block_i:
            table.append(r'\midrule')
        if block_label is not None and not block_label_with_first_regime:
            table.append(
                rf'& \multicolumn{{{n_display_models * metrics_per_model}}}{{c}}'
                rf'{{{block_label}}} \\')
            table.append(
                rf'\cmidrule(lr){{2-{1 + n_display_models * metrics_per_model}}}')
        if section_labels is not None:
            for section_i, section_label in enumerate(section_labels):
                row_indices = [i for i, row in enumerate(rows)
                               if row['section'] == section_i]
                if section_i:
                    table.extend([r'\midrule', r'\addlinespace[3pt]'])
                if show_section_headers:
                    table.append(
                        rf'\multicolumn{{1}}{{l}}{{\:\textit{{{section_label}}}}} & '
                        rf'\multicolumn{{{n_display_models * metrics_per_model}}}{{c}}{{}} \\')
                    table.append(r'\cmidrule(lr){1-1}')
                for row_i in row_indices:
                    append_row(group_i, row_i, display_model_indices)
            continue
        for row_i in baseline_rows:
            append_row(group_i, row_i, display_model_indices)
        if (show_section_headers and baseline_rows
                and any(learned_by_oracle.values())):
            table.append(r'\midrule')
        emitted_group = False
        for oracle in (False, True):
            row_indices = learned_by_oracle[oracle]
            if not row_indices:
                continue
            if emitted_group and show_section_headers:
                table.extend([r'\midrule', r'\addlinespace[3pt]'])
            if show_section_headers:
                section = 'Oracle' if oracle else 'Non-oracle'
                header_block_label = (
                    block_label if block_label_with_first_regime
                    and not emitted_group else '')
                table.append(
                    rf'\multicolumn{{1}}{{l}}{{\:\textit{{{section}}}}} & '
                    rf'\multicolumn{{{n_display_models * metrics_per_model}}}{{c}}'
                    rf'{{{header_block_label}}} \\')
                table.append(r'\cmidrule(lr){1-1}')
                if header_block_label:
                    table.append(
                        rf'\cmidrule(lr){{2-{1 + n_display_models * metrics_per_model}}}')
            for row_i in row_indices:
                append_row(group_i, row_i, display_model_indices)
            emitted_group = True
    table.extend([r'\bottomrule', r'\end{tabular}', r'}', r'\end{center}',
                  r'\end{table}'])
    output_path = LATEX_DIR / path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(table) + '\n')
    print(f'Wrote {output_path}')
    return output_path



def _dataset_specs(datasets):
    """Use dataset labels only when a table contains multiple datasets."""
    datasets = list(datasets)
    if not datasets:
        raise ValueError('datasets must be nonempty')
    return [(dataset, dataset,
             DATASET_DISPLAY_NAMES.get(dataset, dataset)
             if len(datasets) > 1 else None)
            for dataset in datasets]


def _oracle_vs_methods_specs(*, include_end_correct=True,
                             include_sc_posthoc=False):
    """Rows shared by abridged and full oracle-vs-methods tables."""
    methods = [
        {'baseline': 'self_consistency'},
        {'oracle': False, 'multi_ckpt': True, 'n_cand': 2,
         'label': 'Non-oracle training'},
        {'oracle': True, 'multi_ckpt': True, 'n_cand': 2,
         'label': 'Oracle training'},
    ]
    if include_end_correct:
        methods.insert(0, {'baseline': 'end_correct'})
    if include_sc_posthoc:
        methods.insert(2 if include_end_correct else 1,
                       {'baseline': 'sc_posthoc'})
    return methods


def make_table_oracle_vs_methods_abridged(datasets, fname, *, seeds=SEEDS,
                metrics=TABLE_METRICS,
                include_leading_zeros=True, include_surrogate=(),
                include_copy=()):
    """Write the compact model-columns version of this comparison."""
    return _make_table(
        THREE_MODEL_SPECS, _oracle_vs_methods_specs(), fname,
        metrics=metrics, seeds=seeds,
        include_leading_zeros=include_leading_zeros,
        include_surrogate=include_surrogate, include_copy=include_copy,
        show_section_headers=False, dataset_specs=_dataset_specs(datasets))


def make_table_oracle_vs_methods(fname, *, seeds=SEEDS,
                                  metrics=TABLE_METRICS,
                                  include_leading_zeros=True,
                                  include_surrogate=(), include_copy=(),
                                  btl_term='odds'):
    """Write the full row-block version for all models and both datasets.

    This intentionally puts model and dataset in the row-block label, leaving
    a single metric-column block wide enough for every table metric.
    """
    return _make_table(
        THREE_MODEL_SPECS, _oracle_vs_methods_specs(
            include_end_correct=True, include_sc_posthoc=True), fname,
        metrics=metrics, seeds=seeds, btl_term=btl_term,
        include_leading_zeros=include_leading_zeros,
        include_surrogate=include_surrogate, include_copy=include_copy,
        show_section_headers=False,
        dataset_specs=_dataset_specs(['triviaqa', 'jeopardy']),
        model_layout='blocks')


def make_table_transfer_to_bioasq(fname, *, seeds=SEEDS,
                                  metrics=TABLE_METRICS,
                                  include_leading_zeros=True,
                                  include_surrogate=(), include_copy=()):
    """Compare both TriviaQA/Jeopardy transfer directions to BioASQ.

    Results are stacked as adjacent TriviaQA-to-BioASQ and Jeopardy-to-BioASQ
    blocks for each base model, with shared metric columns.
    """
    methods = [
        {'baseline': 'self_consistency'},
        {'oracle': False, 'multi_ckpt': True, 'n_cand': 2,
         'label': 'Non-oracle training'},
        {'oracle': True, 'multi_ckpt': True, 'n_cand': 2,
         'label': 'Oracle training'},
    ]
    return _make_table(
        THREE_MODEL_SPECS, methods, fname, metrics=metrics, seeds=seeds,
        include_leading_zeros=include_leading_zeros,
        include_surrogate=include_surrogate, include_copy=include_copy,
        show_section_headers=False,
        dataset_specs=[
            ('triviaqa', 'bioasq',
             rf'{DATASET_DISPLAY_NAMES["triviaqa"]} $\to$ '
             rf'{DATASET_DISPLAY_NAMES["bioasq"]}'),
            ('jeopardy', 'bioasq',
             rf'{DATASET_DISPLAY_NAMES["jeopardy"]} $\to$ '
             rf'{DATASET_DISPLAY_NAMES["bioasq"]}'),
        ],
        model_layout='blocks')



def make_table_oracle_vs_gcm(fname, *, seeds=SEEDS,
                                     metrics=TABLE_METRICS,
                                     include_leading_zeros=True):
    """Compare GCM, post-hoc GCM, and oracle training for all model/datasets."""
    methods = [
        {'baseline': 'gcm'},
        {'baseline': 'gcm_posthoc'},
        {'oracle': True, 'multi_ckpt': True, 'n_cand': 2,
         'label': 'Oracle training'},
    ]
    return _make_table(
        THREE_MODEL_SPECS, methods, fname, metrics=metrics, seeds=seeds,
        include_leading_zeros=include_leading_zeros,
        show_section_headers=False,
        dataset_specs=_dataset_specs(['triviaqa', 'jeopardy']),
        model_layout='blocks')


def _methods_vs_ablations_sections():
    """Section layout shared by abridged and full ablation tables."""
    non_oracle_method = {'oracle': False, 'multi_ckpt': True, 'n_cand': 2,
                         'label': 'Non-oracle training'}
    return [
        {'label': 'SC', 'methods': [
            {'baseline': 'self_consistency'},
            {'baseline': 'sc_surrogate'},
            {'baseline': 'sc_copy'},
        ]},
        {'label': 'Non-oracle training', 'methods': [non_oracle_method]},
    ]


def make_table_methods_vs_ablations_abridged(datasets, fname, *, seeds=SEEDS,
                metrics=TABLE_METRICS,
                include_leading_zeros=True):
    """Write the compact model-columns version of this ablation table."""
    sections = _methods_vs_ablations_sections()
    non_oracle_method = sections[1]['methods'][0]
    return _make_table(
        THREE_MODEL_SPECS, [], fname, metrics=metrics, seeds=seeds,
        include_leading_zeros=include_leading_zeros,
        include_surrogate=[non_oracle_method], include_copy=[non_oracle_method],
        dataset_specs=_dataset_specs(datasets), sections=sections,
        show_section_headers=False)


def make_table_methods_vs_ablations(fname, *, seeds=SEEDS,
                                    metrics=TABLE_METRICS,
                                    include_leading_zeros=True):
    """Write the full model--dataset row-block version of the ablation table."""
    sections = _methods_vs_ablations_sections()
    non_oracle_method = sections[1]['methods'][0]
    return _make_table(
        THREE_MODEL_SPECS, [], fname, metrics=metrics, seeds=seeds,
        include_leading_zeros=include_leading_zeros,
        include_surrogate=[non_oracle_method], include_copy=[non_oracle_method],
        dataset_specs=_dataset_specs(['triviaqa', 'jeopardy']), sections=sections,
        show_section_headers=False, model_layout='blocks')


def _multi_vs_single_ckpt_specs():
    """Rows shared by abridged and full multi-vs-single tables."""
    return [
        {'oracle': False, 'multi_ckpt': True, 'n_cand': 2, 'label': 'Multi-ckpt'},
        {'oracle': False, 'multi_ckpt': False, 'n_cand': 2, 'label': 'Single-ckpt'},
        {'oracle': True, 'multi_ckpt': True, 'n_cand': 2, 'label': 'Multi-ckpt'},
        {'oracle': True, 'multi_ckpt': False, 'n_cand': 2,
         'label': 'Respective-ckpt'},
    ]


def _multi_vs_single_ckpt_all_ncand_specs():
    """Multi/single rows for both candidate counts, grouped by regime."""
    methods = []
    for oracle, alternate_label in ((False, 'Single-ckpt'), (True, 'Resp-ckpt')):
        for multi_ckpt, checkpoint_label in (
                (True, 'Multi-ckpt'), (False, alternate_label)):
            for n_cand in (2, 1):
                methods.append({
                    'oracle': oracle,
                    'multi_ckpt': multi_ckpt,
                    'n_cand': n_cand,
                    'label': (rf'{checkpoint_label}, '
                              rf'$\text{{n}}_{{\text{{cand}}}}$={n_cand}'),
                })
    return methods


def make_table_multi_vs_single_ckpt_abridged(datasets, fname, *, seeds=SEEDS,
                metrics=TABLE_METRICS,
                include_leading_zeros=True, include_surrogate=(),
                include_copy=()):
    """Write the compact model-columns version of this comparison."""
    return _make_table(
        THREE_MODEL_SPECS, _multi_vs_single_ckpt_specs(), fname,
        metrics=metrics, seeds=seeds,
        include_leading_zeros=include_leading_zeros,
        include_surrogate=include_surrogate, include_copy=include_copy,
        dataset_specs=_dataset_specs(datasets))


def make_table_mvs_ckpt_ans(dataset, fname, *, seeds=SEEDS,
                                    metrics=TABLE_METRICS,
                                    include_leading_zeros=True,
                                    include_surrogate=(), include_copy=()):
    """Write one full table for a dataset with both candidate counts."""
    return _make_table(
        THREE_MODEL_SPECS, _multi_vs_single_ckpt_all_ncand_specs(), fname,
        metrics=metrics, seeds=seeds,
        include_leading_zeros=include_leading_zeros,
        include_surrogate=include_surrogate, include_copy=include_copy,
        dataset_specs=_dataset_specs([dataset]), model_layout='blocks',
        block_label_with_first_regime=True,
        block_label_fn=lambda model_label, _eval_dataset, _dataset_label: model_label)


def make_table_multi_ckpt_vs_ablations(model_name, datasets, fname, *, seeds=SEEDS,
                metrics=TABLE_METRICS,
                include_leading_zeros=True, include_surrogate=(),
                include_copy=(), include_onedata=(), include_pooldata=()):
    """Write the data-revision ablation table; edit this wrapper's methods list."""
    methods = [
        {'oracle': True, 'multi_ckpt': True, 'n_cand': 2, 'label': 'Multi-ckpt'},
        {'oracle': True, 'multi_ckpt': False, 'n_cand': 2, 'label': 'Resp-ckpt'},
    ]
    model_specs = [
        (model_name, model_name.rsplit('/', 1)[-1]),
    ]
    return _make_table(
        model_specs, methods, fname, metrics=metrics, seeds=seeds,
        include_leading_zeros=include_leading_zeros,
        include_surrogate=include_surrogate, include_copy=include_copy,
        include_onedata=include_onedata, include_pooldata=include_pooldata,
        dataset_specs=_dataset_specs(datasets), show_section_headers=False
        )


def make_jeopardy_categories_table(fname, *, n_columns=3):
    """Write a compact appendix table listing the included Jeopardy categories."""
    if n_columns <= 0:
        raise ValueError('n_columns must be positive')

    def display_category(category):
        # ``str.title`` provides the requested first-letter capitalization,
        # but spells ordinal suffixes as e.g. ``19Th``.
        return re.sub(r'(?<=\d)(?:St|Nd|Rd|Th)\b',
                      lambda match: match.group().lower(), category.title())

    categories = sorted(display_category(category)
                        for category in JeopardyConfig.INCLUDE_CATEGORIES)
    n_rows = math.ceil(len(categories) / n_columns)
    rows = [categories[row_i * n_columns:(row_i + 1) * n_columns]
            for row_i in range(n_rows)]
    for row in rows:
        row.extend([''] * (n_columns - len(row)))

    path = Path(fname)
    output_path = LATEX_DIR / path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = [
        r'\begin{table}[t]',
        rf'\caption{{{path.stem.replace("_", " ")}}}',
        rf'\label{{tab:{path.stem}}}',
        r'\begin{center}',
        r'\small',
        r'\begin{tabular}{' + 'l' * n_columns + '}',
        r'\toprule',
    ]
    table.extend(
        ' & '.join(category.replace('&', r'\&') for category in row) + r' \\'
        for row in rows)
    table.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{center}',
        r'\end{table}',
    ])
    output_path.write_text('\n'.join(table) + '\n')
    print(f'Wrote {output_path}')
    return output_path


def make_bioasq_examples_table(fname, *, n_per_type=2):
    """Write an appendix table of post-processing BioASQ examples."""
    if n_per_type <= 0:
        raise ValueError('n_per_type must be positive')
    task = BioASQConfig()
    examples = task._load_all_examples()
    selected_by_type = {}
    for question_type in ('factoid', 'list'):
        selected = [example for example, actual_type in zip(
            examples, task._question_types) if actual_type == question_type]
        if len(selected) < n_per_type:
            raise ValueError(
                f'Only {len(selected)} BioASQ {question_type} examples are available')
        selected_by_type[question_type] = selected[:n_per_type]

    def latex_escape(text):
        replacements = {
            '\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
            '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
            '~': r'\textasciitilde{}', '^': r'\textasciicircum{}',
        }
        escaped = ''.join(replacements.get(char, char) for char in text)
        # Python's list representation writes both quotation marks as '.  Use
        # TeX's opening-quote convention without changing apostrophes within
        # ordinary words (e.g. a possessive or contraction).
        return re.sub(r"(?<![A-Za-z0-9])'", '`', escaped)

    path = Path(fname)
    output_path = LATEX_DIR / path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = [
        r'\begin{table*}[t]',
        rf'\caption{{{path.stem.replace("_", " ")}}}',
        rf'\label{{tab:{path.stem}}}',
        r'\begin{center}',
        r'\small',
        r'\begin{tabular}{lp{0.57\textwidth}p{0.25\textwidth}}',
        r'\toprule',
        r'Type & Question & Reference answer(s) \\',
        r'\midrule',
    ]
    for question_type in ('factoid', 'list'):
        for example in selected_by_type[question_type]:
            table.append(
                f'{question_type.title()} & {latex_escape(example.question)} & '
                f'{latex_escape(example.answer)} ' + r'\\')
    table.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{center}',
        r'\end{table*}',
    ])
    output_path.write_text('\n'.join(table) + '\n')
    print(f'Wrote {output_path}')
    return output_path


def make_conset_breakdown_tables(
        datasets=('triviaqa', 'jeopardy', 'bioasq'), *,
        fname_template='conset-breakdown-{dataset}.tex',
        gen_config_name='beam'):
    """Write one all-model contrast-set breakdown matrix table per dataset.

    In a cell at row ``i`` and column ``j``, ``n (d)`` denotes the total
    number ``n`` of contrast questions between checkpoints ``i`` and ``j``;
    ``d`` is the number on which row checkpoint ``i`` is wrong and column
    checkpoint ``j`` is right.
    """
    model_specs = THREE_MODEL_SPECS

    def checkpoint_labels(revisions):
        steps = [revision.removeprefix('stage1-step') for revision in revisions]
        if all(step.isdigit() for step in steps):
            final_step = int(steps[-1])
            return [f'{round(100 * int(step) / final_step):.0f}\\%'
                    for step in steps]
        return [revision.replace('-', r'\hbox{-}')
                for revision in revisions]

    output_paths = []
    for dataset in datasets:
        if dataset not in EVAL_RANGES:
            raise ValueError(f'Unknown dataset {dataset!r}; expected one of '
                             f'{sorted(EVAL_RANGES)}')
        eval_ranges = parse_ranges(EVAL_RANGES[dataset])
        min_conset_size = _min_conset_size(dataset)
        path = Path(fname_template.format(dataset=dataset))
        table = [
            r'\begin{table*}[ht]',
            rf'\caption{{{path.stem.replace("_", " ")}}}',
            rf'\label{{tab:conset-breakdown-{dataset}}}',
            r'\begin{center}',
            r'\small',
            r'\setlength{\tabcolsep}{5pt}',
            r'\renewcommand{\arraystretch}{1.15}',
        ]

        for model_i, (model_name, model_label) in enumerate(model_specs):
            revisions = list(evaluation_revisions_for_model(model_name))
            preds = {
                revision: load_genprob_preds(
                    dataset, model_name, revision, gen_config_name, eval_ranges)
                for revision in revisions
            }
            pair_results = conset_pair_results(
                preds, pairs=list(combinations(revisions, 2)))
            cells = [['--' if row_i == col_i else None
                      for col_i in range(len(revisions))]
                     for row_i in range(len(revisions))]
            for (row_revision, col_revision), result in pair_results.items():
                if result is None:
                    total = 0
                    forward_count = reverse_count = 0
                else:
                    total = result['n']
                    # ``improved`` is exactly row wrong / column right for
                    # the (row_revision, col_revision) ordering here.
                    forward_count = int(result['improved'].sum())
                    reverse_count = int(result['regressed'].sum())

                def format_cell(direction_count):
                    cell = f'{total} ({direction_count})'
                    return (rf'\textcolor{{gray}}{{{cell}}}'
                            if total < min_conset_size else cell)

                forward_cell = format_cell(forward_count)
                reverse_cell = format_cell(reverse_count)
                row_i = revisions.index(row_revision)
                col_i = revisions.index(col_revision)
                cells[row_i][col_i] = forward_cell
                cells[col_i][row_i] = reverse_cell

            labels = checkpoint_labels(revisions)
            if model_i:
                table.append(r'\vspace{5pt}')
            panel_header = [r'\begin{tabular}{l' + 'c' * len(revisions) + '}']
            # The preceding panel's bottom rule already separates panels;
            # another top rule would create a cramped double line.
            if model_i == 0:
                panel_header.append(r'\toprule')
            panel_header.extend([
                rf'\multicolumn{{{1 + len(revisions)}}}{{c}}{{\textit{{{model_label}}}}} \\',
                rf'\cmidrule(lr){{1-{1 + len(revisions)}}}',
                r'& ' + ' & '.join(labels) + r' \\',
                r'\midrule',
            ])
            table.extend(panel_header)
            table.extend(
                label + ' & ' + ' & '.join(cells[row_i]) + r' \\'
                for row_i, label in enumerate(labels))
            table.extend([r'\bottomrule', r'\end{tabular}'])

        table.extend([r'\end{center}', r'\end{table*}'])
        output_path = LATEX_DIR / path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text('\n'.join(table) + '\n')
        print(f'Wrote {output_path}')
        output_paths.append(output_path)
    return output_paths


def make_checkpoint_training_tokens_table(fname):
    """Write checkpoint training-token counts and evaluation accuracies."""
    # The Marin model card reports these cumulative pre-training-token counts:
    # https://huggingface.co/marin-community/marin-8b-base
    marin_tokens = {
        'kestrel': 2.7,
        'ocelot': 3.78,
        'jellyfish': 4.78,
        'phoenix': 11.1,
        'starling': 12.4,
        'deeper-starling': 12.7,
    }
    final_tokens = {
        OLMO_MODEL: 5.93,
        OLMO_32B_MODEL: 5.50,
    }

    def revisions_for_model(model_name):
        non_oracle = non_oracle_revisions_for_model(model_name)
        return list(dict.fromkeys(
            non_oracle + evaluation_revisions_for_model(model_name)))

    def olmo_percentage(revision, revisions):
        step = revision.removeprefix('stage1-step')
        if not step.isdigit():
            raise ValueError(f'Expected numeric OLMo checkpoint, got {revision!r}')
        final_step = revisions[-1].removeprefix('stage1-step')
        if not final_step.isdigit():
            raise ValueError(f'Expected numeric final OLMo checkpoint, got '
                             f'{revisions[-1]!r}')
        return round(100 * int(step) / int(final_step))

    def token_count(model_name, revision, revisions):
        if model_name == MARIN_MODEL:
            try:
                return marin_tokens[revision]
            except KeyError as exc:
                raise ValueError(
                    f'Missing Marin token count for {revision!r}') from exc
        # Follow the paper's checkpoint labels: round to whole percent before
        # multiplying by the stated final-checkpoint training-token budget.
        return final_tokens[model_name] * olmo_percentage(revision, revisions) / 100

    def format_tokens(model_name, value):
        if model_name in final_tokens:
            # Use ordinary half-up decimal rounding for reported token counts,
            # rather than binary floating-point's ties-to-even formatting.
            return str(Decimal(str(value)).quantize(
                Decimal('0.01'), rounding=ROUND_HALF_UP))
        return f'{value:.3f}'.rstrip('0').rstrip('.')

    accuracies = {}

    def accuracy(model_name, revision, dataset):
        """Top-answer accuracy, excluding only masked blacklist positions."""
        key = (model_name, revision, dataset)
        if key not in accuracies:
            _confs, labels = load_genprob_preds(
                dataset, model_name, revision, 'beam',
                parse_ranges(EVAL_RANGES[dataset]))
            valid = (labels == 1) | (labels == 2) | (labels == 3)
            if not valid.any():
                raise ValueError(
                    f'No valid labels for {model_name}, {revision}, {dataset}')
            accuracies[key] = 100 * (labels[valid] == 1).float().mean().item()
        return accuracies[key]

    path = Path(fname)
    output_path = LATEX_DIR / path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = [
        r'\begin{table*}[t]',
        rf'\caption{{{path.stem.replace("_", " ")}}}',
        rf'\label{{tab:{path.stem}}}',
        r'\begin{center}',
        r'\small',
        r'\setlength{\tabcolsep}{5pt}',
        r'\begin{tabular}{clcccc}',
        r'\toprule',
        r'& & & \multicolumn{3}{c}{Accuracy (\%)} \\',
        r'\cmidrule(lr){4-6}',
        r'Short name & Checkpoint ID & '
        r'\shortstack{Training data\\(trillions of tokens)} & '
        + ' & '.join(DATASET_DISPLAY_NAMES[dataset]
                     for dataset in ('triviaqa', 'jeopardy', 'bioasq')) + r' \\',
        r'\midrule',
    ]
    for model_i, (model_name, model_label) in enumerate(
            THREE_MODEL_SPECS):
        if model_i:
            table.append(r'\midrule')
        table.extend([
            rf'\multicolumn{{6}}{{c}}{{\textit{{{model_label}}}}} \\',
            r'\cmidrule(lr){1-6}',
        ])
        revisions = revisions_for_model(model_name)
        for revision in revisions:
            short_name = (f'{olmo_percentage(revision, revisions)}\\%'
                          if model_name in final_tokens else '')
            table.append(
                f'{short_name} & \\texttt{{{revision}}} & '
                f'{format_tokens(model_name, token_count(model_name, revision, revisions))} & '
                + ' & '.join(
                    f'{accuracy(model_name, revision, dataset):.1f}'
                    for dataset in ('triviaqa', 'jeopardy', 'bioasq')) + ' '
                + r'\\')
    table.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{center}',
        r'\end{table*}',
    ])
    output_path.write_text('\n'.join(table) + '\n')
    print(f'Wrote {output_path}')
    return output_path


def make_hidden_state_variance_table(variance_output, fname, *,
                                     dataset='triviaqa', model_name=None):
    """Write per-layer checkpoint-order/variance Spearman summaries.

    ``variance_output`` is the cached ``.pt`` output of
    :func:`conset.analyze_variance.compute_hidden_state_variances`. For each
    question and layer, the table correlates checkpoint index with
    across-adapter-seed variance; it does not use the cross-dataset variance
    analysis.
    """
    variance_path = Path(variance_output)
    if not variance_path.is_absolute():
        if model_name is None:
            raise ValueError('model_name is required for a relative variance_output')
        variance_path = hidden_state_results_dir(dataset, model_name) / variance_path
    payload = load_within_question_seed_variances(variance_path)
    question_variances = [
        (payload['question_variances'][revision]['q_idxs'],
         payload['question_variances'][revision]['variances'])
        for revision in payload['revisions']
    ]
    means, medians, finite_counts, n_shared_questions = \
        question_variance_spearman_summary(question_variances,
                                           payload['revisions'])

    # Old cached artifacts did not record their base-model identifier; callers
    # can provide it explicitly without recomputing the variances.
    model_name = model_name or payload.get('model_name')
    model_labels = dict(THREE_MODEL_SPECS)
    model_label = (model_labels.get(model_name, model_name.rsplit('/', 1)[-1])
                   if model_name is not None else 'the evaluated model')
    dataset = payload.get('dataset', 'dataset')
    dataset_label = DATASET_DISPLAY_NAMES.get(dataset, dataset)
    path = Path(fname)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('fname must be a filename relative to conset/latex/')
    output_path = LATEX_DIR / path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = [
        r'\begin{table}[t]',
        rf'\caption{{{path.stem.replace("_", " ")}}}',
        rf'\label{{tab:{path.stem}}}',
        r'\centering',
        r'\small',
        r'\begin{tabular}{rcc}',
        r'\toprule',
        r'Layer & Mean $\rho$ & Median $\rho$ \\',
        r'\midrule',
    ]
    table.extend(
        (f'{layer_i} & {mean.item():.3f} & {median.item():.3f} '
         + r'\\' if finite_count else f'{layer_i} & -- & -- ' + r'\\')
        for layer_i, (mean, median, finite_count) in enumerate(
            zip(means, medians, finite_counts), start=1))
    table.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ])
    output_path.write_text('\n'.join(table) + '\n')
    print(f'Wrote {output_path}')
    return output_path


def main():
    abridged_metrics = ('delta0_bal', 'conset_auc', 'full_auc')

    # make_table_oracle_vs_methods_abridged(
    #     datasets=['triviaqa', 'jeopardy'],
    #     fname='oracle-vs-methods-abridged.tex',
    #     metrics=abridged_metrics,
    # )
    # make_table_oracle_vs_methods(
    #     fname='oracle-vs-methods-unabridged.tex',
    # )
    # make_table_oracle_vs_methods(
    #     fname='oracle-vs-methods-unabridged-learn1d.tex',
    #     btl_term='learn_1d',
    # )

    # make_table_methods_vs_ablations_abridged(
    #     datasets=['triviaqa'],
    #     fname='methods-vs-ablations-triviaqa.tex',
    #     metrics=abridged_metrics,
    # )
    # make_table_methods_vs_ablations(
    #     fname='methods-vs-ablations-unabridged.tex',
    # )

    # make_table_multi_vs_single_ckpt_abridged(
    #     datasets=['triviaqa'],
    #     fname='multi-vs-single-ckpt-triviaqa.tex',
    #     metrics=abridged_metrics,
    # )
    # make_table_mvs_ckpt_ans(
    #     dataset='triviaqa',
    #     fname='mvs-ckpt-ans-triviaqa.tex',
    # )
    # make_table_mvs_ckpt_ans(
    #     dataset='jeopardy',
    #     fname='mvs-ckpt-ans-jeopardy.tex',
    # )

    # make_table_multi_ckpt_vs_ablations(
    #     model_name=OLMO_MODEL,
    #     datasets=['triviaqa'],
    #     fname='multi-ckpt-vs-ablations-olmo7b-triviaqa.tex',
    #     include_onedata=[
    #         {'oracle': True, 'multi_ckpt': True, 'n_cand': 2},
    #     ],
    #     include_pooldata=[
    #         {'oracle': True, 'multi_ckpt': False, 'n_cand': 2},
    #     ],
    # )
    # make_table_multi_ckpt_vs_ablations(
    #     model_name=OLMO_MODEL,
    #     datasets=['jeopardy'],
    #     fname='multi-ckpt-vs-ablations-olmo7b-jeopardy.tex',
    #     include_onedata=[
    #         {'oracle': True, 'multi_ckpt': True, 'n_cand': 2},
    #     ],
    #     include_pooldata=[
    #         {'oracle': True, 'multi_ckpt': False, 'n_cand': 2},
    #     ],
    # )

    # make_gcm_plot()
    # make_table_oracle_vs_gcm('oracle-vs-gcm.tex')

    # make_table_transfer_to_bioasq('transfer-to-bioasq.tex')

    # plot_full_auc_confusion_matrices('triviaqa')
    # plot_full_auc_confusion_matrices('jeopardy')

    # try:
    #     make_hidden_state_variance_table(
    #         'triviaqa_lora_triviaqa_acc_lr2e-4_bs16_5k_hidden_state_seed_variance.pt',
    #         'hidden-state-variance.tex', dataset='triviaqa', model_name=OLMO_MODEL)
    # except FileNotFoundError as exc:
    #     print(
    #         'Warning: skipping hidden-state variance table because its '
    #         f'cached variance artifact is unavailable: {exc}',
    #         file=sys.stderr)

    # make_jeopardy_categories_table('jeopardy-categories.tex')
    # make_bioasq_examples_table('bioasq-examples.tex')
    make_conset_breakdown_tables()
    # make_checkpoint_training_tokens_table('checkpoint-training-tokens.tex')


if __name__ == '__main__':
    main()
