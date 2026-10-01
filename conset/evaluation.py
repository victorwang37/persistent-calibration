"""Shared prediction loading and calibration metrics for result tables."""

import os
import re

import numpy as np
import torch
from scipy.optimize import minimize

from .dataset_configs import get_dataset_config
from .model_revisions import OLMO_MODEL, evaluation_revisions_for_model
from .train_confidence import (
    _get_base_result_dir,
    _read_cand_type,
)
from .utils import (
    assemble_ranges,
    compute_auroc,
    compute_brier,
    compute_smooth_ece,
    load_sharded,
)

# External generalized-correctness-model (GCM) prediction artifacts are kept
# outside adapter directories because one GCM is shared by all target models.
DEFAULT_GCM_MODEL = 'Hanqix/GCM-Qwen3-8B-TriviaQA'

# Dataset blacklists are independent of model/checkpoint and expensive for
# datasets whose configuration loads the full source split.
_BLACKLIST_CACHE = {}

DEFAULT_REVISIONS = list(evaluation_revisions_for_model(OLMO_MODEL))


def select_subset(confs, labels, proportion, center='median', complement=False):
    """Select valid confidence predictions from a centered confidence region."""
    right, wrong = labels == 1, (labels == 2) | (labels == 3)
    valid = right | wrong
    n_valid = int(valid.sum())
    if n_valid == 0:
        return None, None
    valid_confs, valid_labels = confs[valid], right[valid].float()
    n_slice = max(1, round(proportion * n_valid))
    if center == 'abs':
        order = (valid_confs - 0.5).abs().argsort()
        selected = order[n_slice:] if complement else order[:n_slice]
        return valid_confs[selected], valid_labels[selected]
    order = valid_confs.argsort()
    valid_confs, valid_labels = valid_confs[order], valid_labels[order]
    start = (n_valid - n_slice) // 2
    end = start + n_slice
    if complement:
        return (torch.cat([valid_confs[:start], valid_confs[end:]]),
                torch.cat([valid_labels[:start], valid_labels[end:]]))
    return valid_confs[start:end], valid_labels[start:end]


def subset_metrics(confs, labels, proportion, center='median', complement=False):
    """Compute calibration metrics for one valid-prediction subset."""
    slice_confs, slice_labels = select_subset(
        confs, labels, proportion, center=center, complement=complement)
    if slice_confs is None:
        return None
    right, wrong = labels == 1, (labels == 2) | (labels == 3)
    metrics = {
        'n_valid': int((right | wrong).sum()),
        'n': len(slice_confs),
        'ece': compute_smooth_ece(slice_confs, slice_labels),
        'bs': compute_brier(slice_confs, slice_labels),
        'auc': compute_auroc(slice_confs, slice_labels),
        'confs': slice_confs,
        'labels': slice_labels,
    }
    return metrics


def pooled_full_metrics(preds, revisions=DEFAULT_REVISIONS):
    """Compute full-set calibration metrics pooled across *revisions*."""
    per_revision = []
    for revision in revisions:
        metric = subset_metrics(*preds[revision], proportion=1.0)
        if metric is None:
            raise ValueError(f'No valid full-set predictions for {revision}')
        per_revision.append(metric)
    confs = torch.cat([metric['confs'] for metric in per_revision])
    labels = torch.cat([metric['labels'] for metric in per_revision])
    return {
        'auc': compute_auroc(confs, labels),
        'bs': compute_brier(confs, labels),
        'ece': compute_smooth_ece(confs, labels),
    }


def parse_adapter_spec(spec):
    """Parse 'CONFIG:EPOCH:SEED[:REV1,REV2,...]' into components.

    Accepts 3-field specs (CONFIG:EPOCH:SEED) with an empty revisions list,
    or 4-field specs (CONFIG:EPOCH:SEED:REV1,REV2,...) with explicit revisions.
    """
    parts = spec.split(':')
    assert len(parts) in (3, 4), (
        f"adapter spec must have 3 or 4 colon-separated fields: {spec}")
    config = parts[0]
    epoch = int(parts[1])
    seed = int(parts[2])
    revisions = parts[3].split(',') if len(parts) == 4 else []
    return config, epoch, seed, revisions


def mask_blacklisted_predictions(confs, labels, dataset, ranges):
    """Represent blacklisted questions as skipped for every prediction source.

    This is intentionally applied while loading, rather than relying on a
    particular prediction artifact having been regenerated after a blacklist
    change.  The local tensor order is the concatenation of ``ranges``.
    """
    if dataset not in _BLACKLIST_CACHE:
        _BLACKLIST_CACHE[dataset] = get_dataset_config(dataset).get_blacklist()
    blacklist = _BLACKLIST_CACHE[dataset]
    canonical_indices = [index for begin, end in ranges
                         for index in range(begin, end)]
    if len(canonical_indices) != len(labels):
        raise ValueError(
            f'Prediction length {len(labels)} does not match ranges {ranges}')
    mask = torch.tensor(
        [index in blacklist for index in canonical_indices],
        dtype=torch.bool, device=labels.device)
    if not mask.any():
        return confs, labels
    confs = confs.clone()
    labels = labels.clone()
    confs[mask] = float('nan')
    labels[mask] = -1
    return confs, labels


def load_preds(adapter_dir, dataset, eval_rev, ranges, use_surrogate=False,
               prediction_prefix=None):
    """Load confidence predictions and their judge labels.

    Confidence files are named ``pred_confs_{dataset}_{eval_rev}_{beg}-{end}.pt``;
    surrogate confidence files use ``pred_confs_surrogate_*``.
    ``prediction_prefix`` supports another explicit prefix, such as the
    external GCM's ``pred_confs_gcm_*``. Each covers the absolute index range
    in its name. The files needed to fulfill *ranges* are discovered
    automatically (a la load_sharded).
    """
    if prediction_prefix is not None:
        if use_surrogate:
            raise ValueError(
                'prediction_prefix is exclusive with surrogate mode')
        prefix = prediction_prefix
    elif use_surrogate:
        prefix = f'pred_confs_surrogate_{dataset}'
    else:
        prefix = f'pred_confs_{dataset}'
    pattern = re.compile(
        rf'^{re.escape(prefix)}_{re.escape(eval_rev)}_(\d+)-(\d+)\.pt$')
    segments = []
    for entry in os.listdir(adapter_dir):
        m = pattern.match(entry)
        if m:
            segments.append((int(m.group(1)), int(m.group(2)),
                             os.path.join(adapter_dir, entry)))
    segments.sort()
    cache = {}

    def _load(path):
        if path not in cache:
            cache[path] = torch.load(path, weights_only=True)
        return cache[path]

    desc = f'{adapter_dir}/{prefix}_{eval_rev}'
    confs = assemble_ranges(
        segments, ranges, lambda p: _load(p)['confs'],
        '.pt', desc)
    judge_labels = assemble_ranges(
        segments, ranges, lambda p: _load(p)['judge_labels'], '.pt', desc)
    return mask_blacklisted_predictions(confs, judge_labels, dataset, ranges)


def gcm_prediction_dir(dataset, model_name, gcm_model=DEFAULT_GCM_MODEL):
    """Return the shared external GCM artifact directory for a target model."""
    target_name = model_name.rsplit('/', 1)[-1]
    gcm_name = gcm_model.rsplit('/', 1)[-1]
    return os.path.join('results', 'conset', dataset, target_name,
                        'gcm', gcm_name)


def load_gcm_preds(dataset, model_name, eval_rev, ranges,
                   gcm_model=DEFAULT_GCM_MODEL):
    """Load one target checkpoint's confidence predictions from the GCM."""
    return load_preds(
        gcm_prediction_dir(dataset, model_name, gcm_model), dataset, eval_rev,
        ranges, prediction_prefix=f'pred_confs_gcm_{dataset}')


def load_sc_surrogate_preds(dataset, model_name, surrogate_revision, eval_rev,
                            ranges):
    """Load SC confidence of *eval_rev* answers under a fixed surrogate ckpt."""
    # Importing here avoids making ordinary calibration evaluation depend on
    # the NLI-generation module at import time.
    from .sc_surrogate import sc_surrogate_prediction_dir
    return load_preds(
        sc_surrogate_prediction_dir(dataset, model_name, surrogate_revision),
        dataset, eval_rev, ranges,
        prediction_prefix=f'pred_confs_sc_surrogate_{dataset}')


def load_genprob_preds(dataset, model_name, eval_rev, gen_config_name,
                       ranges):
    """Use each question's highest genprob as its confidence.

    Returns:
        (confs, judge_labels) for the highest-genprob group of each question.
    """
    base_dir = _get_base_result_dir(dataset, model_name, eval_rev)
    cand_type = _read_cand_type(base_dir, gen_config_name)
    group_probs = load_sharded(
        base_dir, f'{gen_config_name}/{cand_type}_group_probs.pt', ranges)
    judge_labels = load_sharded(
        base_dir, f'{gen_config_name}/judge.pt', ranges)

    # Highest-genprob valid group per question (padding slots masked out)
    masked = group_probs.clone().float()
    masked[judge_labels == -1] = -float('inf')
    gi = masked.argmax(dim=1, keepdim=True)
    confs = group_probs.float().gather(1, gi).squeeze(1)
    labels = judge_labels.gather(1, gi).squeeze(1)
    return mask_blacklisted_predictions(confs, labels, dataset, ranges)


def load_ensembled_preds(adapter_specs, dataset, model_name, gen_config_name,
                         revisions, eval_ranges, train_dataset=None,
                         use_ckpt_respective_predictor=False,
                         use_surrogate=False):
    """Load (confs, judge_labels) for each revision in *revisions*.

    Args:
        adapter_specs: list of 'CONFIG:EPOCH:SEED[:REV1,REV2,...]' specs whose
            predictions are ensembled by arithmetic mean of confidences; None
            for the genprob baseline.
        train_dataset: dataset the adapter was trained on (used to locate the
            adapter directory); defaults to *dataset* (same as eval).
        use_ckpt_respective_predictor: when True, for each eval revision, load
            predictions from the adapter trained on that same revision (the
            revisions field of each spec is ignored, and may be omitted).
        use_surrogate: Load ``pred_confs_surrogate_*`` artifacts rather than
            ordinary confidence artifacts.

    Returns:
        dict of revision -> (confs, judge_labels).
    """
    if train_dataset is None:
        train_dataset = dataset
    preds = {}
    if adapter_specs is None:
        for rev in revisions:
            preds[rev] = load_genprob_preds(
                dataset, model_name, rev, gen_config_name, eval_ranges)
        print("Baseline: genprob confidences")
        return preds

    if use_ckpt_respective_predictor:
        # Parse specs to extract config/epoch/seed (revisions ignored)
        parsed = [parse_adapter_spec(spec) for spec in adapter_specs]

        for rev in revisions:
            adapter_dirs = []
            for config, epoch, seed, _train_revs in parsed:
                adapter_dirs.append(os.path.join(
                    _get_base_result_dir(train_dataset, model_name, rev),
                    f'confidence_{config}', f'seed{seed}', f'epoch{epoch}'))

            per_adapter = [
                load_preds(d, dataset, rev, eval_ranges,
                           use_surrogate=use_surrogate)
                for d in adapter_dirs]
            confs = torch.stack([c for c, _ in per_adapter]).mean(dim=0)
            preds[rev] = (confs, per_adapter[0][1])

        print("Using per-checkpoint respective predictors")
        config_descs = [f'{c}:seed{s}:epoch{e}'
                        for c, e, s, _ in parsed]
        print(f"  Adapter configs: {', '.join(config_descs)}")
        return preds

    # Build adapter directories from specs (standard mode: single adapter)
    adapter_dirs = []
    for spec in adapter_specs:
        config, epoch, seed, train_revs = parse_adapter_spec(spec)
        assert train_revs, (
            f"adapter spec must include training revisions (4-field format) "
            f"when --use_ckpt_respective_predictor is not set: {spec}")
        adapter_dirs.append(os.path.join(
            _get_base_result_dir(train_dataset, model_name, '_'.join(train_revs)),
            f'confidence_{config}', f'seed{seed}', f'epoch{epoch}'))

    # Load and ensemble adapter predictions for each eval revision
    for rev in revisions:
        per_adapter = [
            load_preds(d, dataset, rev, eval_ranges,
                       use_surrogate=use_surrogate)
            for d in adapter_dirs]
        confs = torch.stack([c for c, _ in per_adapter]).mean(dim=0)
        preds[rev] = (confs, per_adapter[0][1])

    for d in adapter_dirs:
        print(f"Adapter: {d}")
    if len(adapter_dirs) > 1:
        print(f"Ensembled over {len(adapter_dirs)} adapters "
              f"(arithmetic mean)")
    return preds


def _learn_1d_conditioning(p_1, p_2, labels_1, n_knots=30):
    """Fit the monotone antisymmetric 1-D Bradley--Terry score from log.txt.

    The learned score has h(.5)=0, h(1-p)=-h(p), and positive increments at
    empirical upper-half quantile knots. It is fitted separately for the
    supplied checkpoint pair using its contrast-direction labels.
    """
    device, output_dtype = p_1.device, p_1.dtype
    p_1_cpu, p_2_cpu = p_1.detach().cpu().double(), p_2.detach().cpu().double()
    targets = (labels_1.detach().cpu() == 1).double()
    upper_values = torch.cat([p_1_cpu, p_2_cpu]).maximum(
        1 - torch.cat([p_1_cpu, p_2_cpu]))

    # c_1,...,c_K are empirical upper-half quantiles. Repeated quantiles can
    # occur after PAVA; collapse them so every interpolation interval is real.
    quantile_levels = torch.arange(1, n_knots + 1, dtype=torch.double) / (n_knots + 1)
    empirical_knots = torch.quantile(upper_values, quantile_levels).clamp(
        min=0.5 + 1e-6, max=1 - 1e-6)
    empirical_knots = torch.unique_consecutive(empirical_knots)
    if empirical_knots.numel() == 0:
        # This occurs only when all inputs are exactly .5. Any positive-side
        # interval is equivalent because every observed score remains zero.
        empirical_knots = torch.tensor([0.75], dtype=torch.double)
    knots = torch.cat([torch.tensor([0.5], dtype=torch.double), empirical_knots])

    # Let a_j = h(c_j) - h(c_{j-1}) be the nonnegative increment over knot
    # interval j.  With knots held fixed, every h(p) is affine in a, so the
    # logistic NLL is convex under the box constraints a_j >= 0.  Keep that
    # direct parametrization rather than enforcing positivity through a
    # nonlinear softplus transformation.
    def h_design(values):
        """Return A such that h(values) == A @ increments."""
        upper = values.maximum(1 - values)
        n_steps = len(knots) - 1
        design = torch.zeros((len(values), n_steps), dtype=torch.double)
        interval = (torch.bucketize(upper, knots, right=False) - 1).clamp(
            min=0, max=n_steps - 1)
        widths = knots[1:] - knots[:-1]
        for interval_i in range(n_steps):
            mask = interval == interval_i
            if not mask.any():
                continue
            # Linear interpolation: all earlier increments have coefficient
            # one, while the current increment gets the in-interval fraction.
            design[mask, :interval_i] = 1
            design[mask, interval_i] = (
                (upper[mask] - knots[interval_i]) / widths[interval_i])
        # Above the final knot, linearly extrapolate using the last slope.
        extrapolate = upper > knots[-1]
        if extrapolate.any():
            design[extrapolate] = 1
            design[extrapolate, -1] += (
                (upper[extrapolate] - knots[-1]) / widths[-1])
        design *= torch.where(values >= 0.5, 1.0, -1.0).unsqueeze(1)
        return design.numpy()

    logit_design = h_design(p_1_cpu) - h_design(p_2_cpu)
    target_values = targets.numpy()
    knot_values = knots.numpy()
    initial_h = np.log(knot_values / (1 - knot_values))
    initial_steps = np.maximum(np.diff(initial_h), 0.0)

    def nll_and_grad(increments):
        logits = logit_design @ increments
        # logaddexp is the stable form of binary cross entropy with logits.
        loss = np.mean(np.logaddexp(0, logits) - target_values * logits)
        probabilities = 1 / (1 + np.exp(-np.clip(logits, -700, 700)))
        gradient = logit_design.T @ (probabilities - target_values)
        return loss, gradient / len(target_values)

    result = minimize(
        nll_and_grad, initial_steps, jac=True, method='L-BFGS-B',
        bounds=[(0.0, None)] * len(initial_steps),
        options={'maxiter': 500, 'gtol': 1e-9, 'ftol': 1e-12})
    if not result.success:
        raise RuntimeError(f'learn_1d L-BFGS-B failed: {result.message}')

    probs_1 = torch.from_numpy(
        1 / (1 + np.exp(-np.clip(logit_design @ result.x, -700, 700))))
    return probs_1.to(device=device, dtype=output_dtype), \
        (1 - probs_1).to(device=device, dtype=output_dtype)


def condition_pair_confidences(confs_1, confs_2, mask=None,
                               btl_term='odds',
                               labels_1=None):
    """Condition paired confidences on exactly one checkpoint being correct.

    For selected entries, returns normalized q_1 and q_2. With
    ``btl_term='odds'`` (the default), their numerators are respectively
    p_1(1-p_2) and (1-p_1)p_2. With ``btl_term='prob'``, the numerators are
    p_1 and p_2. With ``btl_term='learn_1d'``, a monotone 1-D score function
    is fitted such that q_1=sigmoid(h(p_1)-h(p_2)). ``labels_1`` supplies
    the binary target indicating that checkpoint 1 is correct for that mode.
    """
    if btl_term not in {'odds', 'prob', 'learn_1d'}:
        raise ValueError(
            "btl_term must be 'odds', 'prob', or 'learn_1d', got "
            f"{btl_term!r}")
    if mask is None:
        mask = torch.ones_like(confs_1, dtype=torch.bool)
    if mask.shape != confs_1.shape or confs_1.shape != confs_2.shape:
        raise ValueError('confs_1, confs_2, and mask must have the same shape')
    if btl_term == 'learn_1d':
        if labels_1 is None:
            raise ValueError(f"btl_term={btl_term!r} requires labels_1")
        if labels_1.shape != confs_1.shape:
            raise ValueError('labels_1 must have the same shape as confidences')
    out_1, out_2 = confs_1.clone(), confs_2.clone()
    p_1 = confs_1[mask].float().clamp(1e-6, 1 - 1e-6)
    p_2 = confs_2[mask].float().clamp(1e-6, 1 - 1e-6)
    if btl_term == 'learn_1d':
        learned_1, learned_2 = _learn_1d_conditioning(p_1, p_2, labels_1[mask])
        out_1[mask], out_2[mask] = learned_1.to(out_1.dtype), learned_2.to(out_2.dtype)
        return out_1, out_2
    if btl_term == 'odds':
        numerator_1 = p_1 * (1 - p_2)
        numerator_2 = (1 - p_1) * p_2
    else:
        numerator_1, numerator_2 = p_1, p_2

    denominator = numerator_1 + numerator_2
    out_1[mask] = (numerator_1 / denominator).to(out_1.dtype)
    out_2[mask] = (numerator_2 / denominator).to(out_2.dtype)
    return out_1, out_2


def pair_accuracy_scores(correct_confs, incorrect_confs):
    """Return 1 for a win, 0.5 for a tie, and 0 for a loss in each pair."""
    return torch.where(
        correct_confs > incorrect_confs,
        torch.ones_like(correct_confs),
        torch.where(
            correct_confs == incorrect_confs,
            torch.full_like(correct_confs, 0.5),
            torch.zeros_like(correct_confs)))




def contrast_metrics(confs_b, labels_b, confs_e, labels_e,
                     balance_classes=False, condition_on_conset=False,
                     btl_term='odds'):
    """Compute calibration metrics on the contrast set of a ckpt pair.

    Contrast set: instances where one ckpt is right (label 1) and the other is
    wrong (label 2 or 3). Padding labels (-1) are excluded implicitly.

    When *balance_classes* is True, instances from the "regressed" direction
    (beg right, end wrong) and the "improved" direction (beg wrong, end right)
    are reweighted so both directions contribute equally to all metrics.

    Returns:
        dict with aggregate metrics plus directional improvement/regression
        AUC and pair-accuracy diagnostics; or None if the contrast set is
        empty.
    """
    right_b = labels_b == 1
    wrong_b = (labels_b == 2) | (labels_b == 3)
    right_e = labels_e == 1
    wrong_e = (labels_e == 2) | (labels_e == 3)

    regressed = right_b & wrong_e   # beg right, end wrong
    improved = wrong_b & right_e    # beg wrong, end right
    mask = regressed | improved
    n = int(mask.sum())
    if n == 0:
        return None

    raw_pair_confs_b, raw_pair_confs_e = confs_b[mask], confs_e[mask]
    pair_confs_b, pair_confs_e = raw_pair_confs_b, raw_pair_confs_e
    pair_labels_b, pair_labels_e = right_b[mask], right_e[mask]
    if condition_on_conset:
        pair_confs_b, pair_confs_e = condition_pair_confidences(
            pair_confs_b, pair_confs_e, btl_term=btl_term,
            labels_1=pair_labels_b)
    confs = torch.cat([pair_confs_b, pair_confs_e])
    labels = torch.cat([pair_labels_b, pair_labels_e]).float()

    # Per-question class weights (1/n_reg for regressed, 1/n_imp for
    # improved).  Always computed so save_conf_dist can reuse them;
    # only applied to metrics when balance_classes is True.
    n_reg = int(regressed[mask].sum())
    n_imp = int(improved[mask].sum())
    if n_reg > 0 and n_imp > 0:
        class_weights = torch.where(
            regressed[mask],
            torch.tensor(1.0 / n_reg),
            torch.tensor(1.0 / n_imp),
        )
    else:
        class_weights = torch.ones(n)

    # Both the beg-ckpt and end-ckpt datapoints of a question share
    # the same weight, so duplicate.
    weights = torch.cat([class_weights, class_weights]) \
        if balance_classes else None

    # Pair accuracy is a signed within-question comparison, so calculate it
    # from the raw confidences. Every BTL conditioning form is theoretically
    # order-preserving, but learned transforms can otherwise introduce a
    # float32 tie for two distinct raw confidences.
    raw_correct_confs = torch.where(
        right_b[mask], raw_pair_confs_b, raw_pair_confs_e)
    raw_incorrect_confs = torch.where(
        right_b[mask], raw_pair_confs_e, raw_pair_confs_b)
    raw_pair_correct = pair_accuracy_scores(raw_correct_confs, raw_incorrect_confs)
    improved_mask, regressed_mask = improved[mask], regressed[mask]
    return {
        'n': n,
        # Local question positions in the input prediction tensors.  Keeping
        # them lets table code perform a shared-question bootstrap across
        # several contrast checkpoint pairs.
        'question_indices': mask.nonzero(as_tuple=False).flatten(),
        'ece': compute_smooth_ece(confs, labels),
        'bs': compute_brier(confs, labels, weights=weights),
        'auc': compute_auroc(confs, labels, weights=weights),
        'confs': confs,
        # Preserve the unconditioned values as well: callers that report
        # contrast-set confidence levels should not accidentally describe the
        # pair-normalized BTL probabilities as model confidences.
        'raw_confs': torch.cat([raw_pair_confs_b, raw_pair_confs_e]),
        'labels': labels,
        'weights': weights,
        'class_weights': class_weights,
        'improved': improved_mask,
        'regressed': regressed_mask,
        'pair_correct': raw_pair_correct,
    }


def conset_pair_results(preds, balance_classes=False,
                        condition_on_conset=False,
                        btl_term='odds',
                        pairs=None):
    """Evaluate all requested checkpoint pairs on their contrast sets.

    This is the programmatic counterpart to the normal ``--subset conset``
    evaluation path.  It deliberately operates on already-loaded predictions
    so reporting code can reuse the exact contrast-set definitions without
    duplicating prediction-loading logic.
    """
    if pairs is None:
        raise ValueError('conset_pair_results requires explicit checkpoint pairs')

    results = {}
    for left_revision, right_revision in pairs:
        results[(left_revision, right_revision)] = contrast_metrics(
            *preds[left_revision], *preds[right_revision],
            balance_classes=balance_classes,
            condition_on_conset=condition_on_conset, btl_term=btl_term)
    return results


def average_pair_metrics(results, metric_names):
    """Return unweighted averages of named metrics over nonempty pairs."""
    valid = [metric for metric in results.values() if metric is not None]
    if not valid:
        raise ValueError('No checkpoint pair has a nonempty contrast set')
    return {
        name: sum(metric[name] for metric in valid) / len(valid)
        for name in metric_names
    }
