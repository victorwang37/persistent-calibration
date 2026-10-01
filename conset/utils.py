import os
import pickle as pkl
import re
import warnings
from concurrent.futures import ProcessPoolExecutor
from collections import namedtuple
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint, Bounds
from scipy.sparse import lil_matrix, csr_matrix


# ---------------------------------------------------------------------------
# Isotonic regression (PAVA)
# ---------------------------------------------------------------------------

Block = namedtuple('Block', ['val', 'weight', 'beg', 'end'])


def _pava_merge(blocks, i):
    """Merge blocks[i] and blocks[i+1] in place (weighted average)."""
    b1, b2 = blocks[i], blocks[i + 1]
    w = b1.weight + b2.weight
    blocks[i] = Block(
        val=(b1.val * b1.weight + b2.val * b2.weight) / w,
        weight=w, beg=b1.beg, end=b2.end,
    )
    del blocks[i + 1]


def _pava(y):
    """Pool Adjacent Violators Algorithm (left-to-right).

    Finds the non-decreasing sequence minimizing sum((y_i - f_i)^2).

    Args:
        y: 1D array-like, values pre-sorted by the covariate x.

    Returns:
        blocks: list of Block(val, weight, beg, end)
    """
    n = len(y)
    blocks = [Block(val=float(y[i]), weight=1.0, beg=i, end=i+1) for i in range(n)]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i].val > blocks[i + 1].val:
            _pava_merge(blocks, i)
            i = max(0, i - 1)
        else:
            i += 1
    return blocks


def _pava_knots(x, y):
    """Run PAVA and return (knot_x, knot_y) for piecewise linear interpolation.

    Knot x-coordinates are the mean x within each block.
    Adds (0, 0) and (1, 1) as boundary knots if no (0, .) and (1, .) yet.
    """
    idx = np.argsort(x)
    xs = x[idx].astype(np.float64)
    ys = y[idx].astype(np.float64)
    blocks = _pava(ys)
    kx = np.array([xs[b.beg:b.end].mean() for b in blocks])
    ky = np.array([b.val for b in blocks])

    # Add (0, 0) and (1, 1) if no (0, .) and (1, .) yet
    if kx[0] > 0.0:
        kx = np.concatenate([[0.0], kx])
        ky = np.concatenate([[0.0], ky])
    if kx[-1] < 1.0:
        kx = np.concatenate([kx, [1.0]])
        ky = np.concatenate([ky, [1.0]])

    return kx, ky


def fit_pava(x, y):
    """Fit PAVA isotonic regression and return calibrated values.

    Args:
        x: torch tensor of raw confidence values (any shape).
        y: torch tensor of binary labels (same shape as x).

    Returns:
        cal_x: torch tensor of calibrated values pava_fn(x).
        pava_fn: (kx, ky) tuple that can be passed to apply_pava.
    """
    x_flat = x.reshape(-1).numpy()
    y_flat = y.reshape(-1).numpy()
    kx, ky = _pava_knots(x_flat, y_flat)
    cal_x = torch.from_numpy(np.interp(x_flat, kx, ky)).float().reshape(x.shape)
    return cal_x, (kx, ky)


def apply_pava(x, pava_fn):
    """Apply a fitted PAVA to new data.

    Args:
        x: torch tensor of raw confidence values (any shape).
        pava_fn: (kx, ky) tuple returned by fit_pava with do_return_fn=True.

    Returns:
        cal_x: torch tensor of calibrated values pava_fn(x).
    """
    kx, ky = pava_fn
    return torch.from_numpy(np.interp(x.reshape(-1).numpy(), kx, ky)).float().reshape(x.shape)



# ---------------------------------------------------------------------------
# SmoothECE-bandwidth KDE calibration
# ---------------------------------------------------------------------------

def _relplot_smooth_ece():
    """Load the reference ``ml-calibration`` SmoothECE implementation.

    ``relplot`` is expected to be installed, for example with
    ``pip install -e ml-calibration``.  The sibling-checkout fallback is
    temporarily disabled while validating that installation path.
    """
    def import_relplot():
        from relplot import config
        from relplot.kernels import ReflectedGaussianKernel
        from relplot.metrics import smECE
        return config, ReflectedGaussianKernel, smECE

    try:
        return import_relplot()
    except ModuleNotFoundError as exc:
        if exc.name != 'relplot':
            raise
        raise ModuleNotFoundError(
            'SmoothECE requires the installed relplot package. '
            'Run `pip install -e ml-calibration`.') from exc


def smooth_ece_bandwidth(confs, labels, eps=1e-3, n_search=12):
    """Choose the reference implementation's automatic SmoothECE bandwidth.

    ``n_search`` is retained for call compatibility but the reference package
    controls its own binary-search refinement.
    """
    del n_search
    confs_np = (confs.detach().cpu().numpy() if torch.is_tensor(confs)
                else np.asarray(confs))
    labels_np = (labels.detach().cpu().numpy() if torch.is_tensor(labels)
                 else np.asarray(labels))
    if len(confs_np) == 0:
        raise ValueError('SmoothECE requires at least one example')
    _, sigma = _relplot_smooth_ece()[2](
        confs_np.reshape(-1), labels_np.reshape(-1), eps=eps,
        return_width=True)
    return sigma


def compute_smooth_ece(confs, labels, eps=1e-3):
    """Compute the reference implementation's automatically tuned SmoothECE."""
    confs_np = confs.detach().cpu().numpy().astype(np.float64).reshape(-1)
    labels_np = labels.detach().cpu().numpy().astype(np.float64).reshape(-1)
    if len(confs_np) == 0:
        return float('nan')
    return _relplot_smooth_ece()[2](confs_np, labels_np, eps=eps)


def fit_smooth_ece_kde(confs, labels):
    """Fit a SmoothECE-bandwidth reflected-Gaussian KDE calibrator.

    Returns ``(calibrated_confs, kde_fn)``. ``kde_fn`` can be passed to
    :func:`apply_smooth_ece_kde`; it contains the grid, Nadaraya--Watson
    calibration curve, density estimate, and automatically selected bandwidth.
    """
    confs_np = confs.detach().cpu().numpy().astype(np.float64).reshape(-1)
    labels_np = labels.detach().cpu().numpy().astype(np.float64).reshape(-1)
    config, ReflectedGaussianKernel, _ = _relplot_smooth_ece()
    sigma = smooth_ece_bandwidth(confs_np, labels_np)
    n_grid = max(config.smECE_mesh_pts, round(10 / sigma))
    grid = np.linspace(0.0, 1.0, n_grid)
    calibration, density = ReflectedGaussianKernel(sigma).smooth(
        confs_np, labels_np, grid)
    # ``relplot`` returns density up to a common discretization constant;
    # normalize it so this public helper continues to expose a density on
    # [0, 1]. The calibration curve itself is unchanged.
    density /= np.trapz(density, grid)
    calibrated = np.interp(confs_np, grid, calibration).astype(np.float32)
    return (torch.from_numpy(calibrated).reshape(confs.shape),
            (grid, calibration, density, sigma))


def apply_smooth_ece_kde(confs, kde_fn):
    """Apply a calibrator returned by :func:`fit_smooth_ece_kde`."""
    grid, calibration, _density, _sigma = kde_fn
    confs_np = confs.detach().cpu().numpy()
    calibrated = np.interp(confs_np.reshape(-1), grid, calibration)
    return torch.from_numpy(calibrated.astype(np.float32)).reshape(confs.shape)


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------

def plot_reweight_reliability_diagrams(
    correctness, weights, bin_ids, path, checkpoint_labels=None,
):
    """Plot stacked, per-checkpoint composition of reweighting bins.

    Bins are normalized to height one.  Their bottom-to-top order is
    non-conset correct, conset correct, conset incorrect, non-conset
    incorrect.  Each nonempty segment is labelled with its within-bin share,
    full-checkpoint share, and common instance weight.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    correctness = np.asarray(correctness, dtype=int)
    weights = np.asarray(weights, dtype=float)
    bin_ids = np.asarray(bin_ids, dtype=int)
    if not (correctness.shape == weights.shape == bin_ids.shape):
        raise ValueError("correctness, weights, and bin_ids must match")

    n, m = correctness.shape
    n_bins = int(bin_ids.max()) + 1
    if checkpoint_labels is None:
        checkpoint_labels = [f"Checkpoint {c}" for c in range(m)]
    if len(checkpoint_labels) != m:
        raise ValueError("checkpoint_labels must have one entry per checkpoint")

    mixed = (correctness.sum(axis=1) > 0) & (correctness.sum(axis=1) < m)
    categories = [
        ("Non-conset correct", ~mixed, 1, "#4c78a8"),
        ("Conset correct", mixed, 1, "#72b7b2"),
        ("Conset incorrect", mixed, 0, "#f58518"),
        ("Non-conset incorrect", ~mixed, 0, "#e45756"),
    ]
    n_cols = min(2, m)
    n_rows = (m + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 8 * n_rows),
                             squeeze=False, sharey=True)
    bin_spacing = 1.35
    x = np.arange(n_bins) * bin_spacing
    inside_label_min_height = 0.08
    # Three-line callouts are placed in an external lane below the axes.  The
    # gap exceeds their rendered height, so boxes cannot overlap vertically.
    callout_min_gap = 0.16

    for c, ax in enumerate(axes.flat[:m]):
        bottom = np.zeros(n_bins)
        small_segments = [[] for _ in range(n_bins)]
        for name, row_mask, label, color in categories:
            heights = np.zeros(n_bins)
            annotations = [None] * n_bins
            for b in range(n_bins):
                in_bin = bin_ids[:, c] == b
                segment = in_bin & row_mask & (correctness[:, c] == label)
                count = int(segment.sum())
                if not count:
                    continue
                heights[b] = count / in_bin.sum()
                segment_weights = weights[segment, c]
                weight_value = segment_weights[0]
                weight_text = f"{weight_value:.2g}"
                if not np.allclose(segment_weights, segment_weights[0]):
                    weight_value = segment_weights.mean()
                    weight_text = f"{weight_value:.2g}*"
                if weight_value > 5:
                    weight_text = rf"$\mathbf{{{weight_text}}}$"
                annotations[b] = (
                    f"{heights[b]:.1%}\n{count / n:.1%}\n{weight_text}"
                )
            bars = ax.bar(x, heights, bottom=bottom, width=0.82, color=color,
                          edgecolor="white", linewidth=0.6, label=name)
            for b, annotation in enumerate(annotations):
                if annotation is not None:
                    segment_middle = bottom[b] + heights[b] / 2
                    if heights[b] >= inside_label_min_height:
                        ax.text(x[b], segment_middle, annotation,
                                ha="center", va="center", fontsize=6)
                    else:
                        small_segments[b].append(
                            (segment_middle, annotation, color)
                        )
            bottom += heights

        # Small segments receive stacked callouts in dedicated lanes above or
        # below the 0–1 bar region, leaving the composition itself unobscured.
        for b, segments in enumerate(small_segments):
            if not segments:
                continue
            lower = sorted((item for item in segments if item[0] < 0.5),
                           key=lambda item: item[0])
            upper = sorted((item for item in segments if item[0] >= 0.5),
                           key=lambda item: item[0], reverse=True)
            # Offset adjacent bins into alternating callout lanes, preventing
            # labels from different bins from sharing the same horizontal row.
            lane_offset = callout_min_gap * (b % 3)
            callouts = [
                (item, -0.22 - lane_offset - callout_min_gap * i)
                for i, item in enumerate(lower)
            ] + [
                (item, 1.14 + lane_offset + callout_min_gap * i)
                for i, item in enumerate(upper)
            ]
            if b <= 1:
                x_offsets = (0.0, 0.28, 0.50, 0.68)
            elif b >= n_bins - 2:
                x_offsets = (0.0, -0.28, -0.50, -0.68)
            else:
                x_offsets = (0.0, 0.28, -0.28, 0.48)
            for callout_i, ((segment_middle, annotation, color), label_y) in enumerate(callouts):
                ax.annotate(
                    annotation, xy=(x[b], segment_middle),
                    xytext=(x[b] + x_offsets[callout_i], label_y),
                    ha="center", va="center", fontsize=5.5, clip_on=False,
                    bbox={"boxstyle": "round,pad=0.12", "fc": "white",
                          "ec": "none", "alpha": 0.86},
                    arrowprops={"arrowstyle": "-", "color": color,
                                "lw": 0.8, "shrinkA": 0, "shrinkB": 0,
                                "connectionstyle": (
                                    f"arc3,rad={0.15 * (callout_i - 1.5):.3f}"
                                )},
                )

        ax.set_title(checkpoint_labels[c])
        ax.set_xlabel("")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{b / n_bins:.1f}–{(b + 1) / n_bins:.1f}"
                            for b in range(n_bins)], rotation=45, ha="right")
        ax.set_xlim(-0.55, x[-1] + 0.55)
        ax.set_ylim(0, 1)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.grid(axis="y", alpha=0.25)
    for ax in axes[:, 0]:
        ax.set_ylabel("Fraction of bin")
    for ax in axes.flat[m:]:
        ax.remove()

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.20, top=0.80,
                        hspace=0.95, wspace=0.06)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def compute_brier(confs, labels, weights=None):
    """Brier score: mean squared error between confidence and binary label."""
    sq = (confs.float() - labels.float()) ** 2
    if weights is not None:
        return (weights.float() * sq).sum().item() / weights.float().sum().item()
    return sq.mean().item()




def compute_auroc(confs, labels, weights=None):
    """AUROC of confidences against binary labels."""
    from sklearn.metrics import roc_auc_score
    sw = weights.numpy() if weights is not None else None
    return roc_auc_score(labels.numpy(), confs.numpy(), sample_weight=sw)


def _paired_bootstrap_metric_chunk(reference_confs, candidate_confs, labels,
                                   positive, negative, metric_fn, direction,
                                   n_resamples, seed):
    """Compute one independent chunk of paired bootstrap differences."""
    # ProcessPoolExecutor transports NumPy arrays by ordinary serialization;
    # sending Torch tensors instead can invoke its shared-memory transport.
    if not torch.is_tensor(reference_confs):
        reference_confs = torch.from_numpy(reference_confs)
        candidate_confs = torch.from_numpy(candidate_confs)
        labels = torch.from_numpy(labels)
        positive = torch.from_numpy(positive)
        negative = torch.from_numpy(negative)
    generator = torch.Generator().manual_seed(seed)
    metric_differences = np.empty(n_resamples, dtype=np.float64)
    for bootstrap_i in range(n_resamples):
        indices = torch.cat([
            positive[torch.randint(len(positive), (len(positive),), generator=generator)],
            negative[torch.randint(len(negative), (len(negative),), generator=generator)],
        ])
        metric_differences[bootstrap_i] = direction * (
            metric_fn(reference_confs[indices], labels[indices])
            - metric_fn(candidate_confs[indices], labels[indices]))
    return metric_differences


def paired_bootstrap_metric_is_significantly_better(
        reference_confs, candidate_confs, labels, metric_fn, *,
        higher_is_better, confidence_level=0.95, n_resamples=10000, seed=17,
        n_jobs=1):
    """One-sided paired, class-stratified bootstrap metric comparison.

    ``reference_confs`` and ``candidate_confs`` must be predictions for the
    same examples, in the same order.  Each bootstrap resample is shared by
    both predictors, preserving their dependence.  Each resample draws, with
    replacement, exactly the original number of positive examples and exactly
    the original number of negative examples.  ``metric_fn`` receives
    ``(confs, labels)`` tensors and returns a scalar; its direction is
    specified by ``higher_is_better``.  The reference is significant at the
    requested one-sided confidence level when the corresponding percentile
    lower bound on its signed metric improvement is positive.

    ``n_jobs`` splits independent resamples into process-level chunks.  A
    metric function must be top-level and picklable when ``n_jobs > 1``.
    """
    if not 0 < confidence_level < 1:
        raise ValueError('confidence_level must lie strictly between 0 and 1')
    if n_resamples < 1:
        raise ValueError('n_resamples must be positive')
    if n_jobs < 1:
        raise ValueError('n_jobs must be positive')

    reference_confs = reference_confs.detach().cpu().reshape(-1)
    candidate_confs = candidate_confs.detach().cpu().reshape(-1)
    labels = labels.detach().cpu().reshape(-1)
    if not (len(reference_confs) == len(candidate_confs) == len(labels)):
        raise ValueError('paired bootstrap inputs must have the same length')
    if not torch.isfinite(reference_confs).all() or not torch.isfinite(candidate_confs).all():
        raise ValueError('paired bootstrap confidences must be finite')
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError('paired bootstrap labels must be binary')

    positive = (labels == 1).nonzero(as_tuple=False).flatten()
    negative = (labels == 0).nonzero(as_tuple=False).flatten()
    if len(positive) == 0 or len(negative) == 0:
        raise ValueError('paired bootstrap requires both classes')

    direction = 1 if higher_is_better else -1
    n_workers = min(n_jobs, n_resamples)
    if n_workers == 1:
        metric_differences = _paired_bootstrap_metric_chunk(
            reference_confs, candidate_confs, labels, positive, negative,
            metric_fn, direction, n_resamples, seed)
    else:
        chunk_sizes = [n_resamples // n_workers] * n_workers
        for worker_i in range(n_resamples % n_workers):
            chunk_sizes[worker_i] += 1
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(
                    _paired_bootstrap_metric_chunk,
                    reference_confs.numpy(), candidate_confs.numpy(), labels.numpy(),
                    positive.numpy(), negative.numpy(), metric_fn, direction,
                    chunk_size, seed + worker_i,
                )
                for worker_i, chunk_size in enumerate(chunk_sizes)
            ]
            metric_differences = np.concatenate(
                [future.result() for future in futures])

    return bool(np.quantile(metric_differences, 1 - confidence_level) > 0)


def paired_bootstrap_auroc_is_significantly_greater(
        reference_confs, candidate_confs, labels, *,
        confidence_level=0.95, n_resamples=10000, seed=17, n_jobs=1):
    """Whether reference AUROC is significantly higher than candidate AUROC."""
    return paired_bootstrap_metric_is_significantly_better(
        reference_confs, candidate_confs, labels, compute_auroc,
        higher_is_better=True, confidence_level=confidence_level,
        n_resamples=n_resamples, seed=seed, n_jobs=n_jobs)


def paired_bootstrap_smooth_ece_is_significantly_lower(
        reference_confs, candidate_confs, labels, *,
        confidence_level=0.95, n_resamples=10000, seed=17, n_jobs=1):
    """Whether reference smECE is significantly lower than candidate smECE.

    Each resample invokes :func:`compute_smooth_ece`, hence also reruns the
    reference SmoothECE implementation's automatic bandwidth selection.
    """
    return paired_bootstrap_metric_is_significantly_better(
        reference_confs, candidate_confs, labels, compute_smooth_ece,
        higher_is_better=False, confidence_level=confidence_level,
        n_resamples=n_resamples, seed=seed, n_jobs=n_jobs)


# ---------------------------------------------------------------------------
# Question-level shared-sequence bootstrap
# ---------------------------------------------------------------------------

def _as_bootstrap_numpy(value, *, name, dtype=None):
    """Detach one bootstrap input to a contiguous CPU NumPy array."""
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    value = np.asarray(value, dtype=dtype)
    if value.ndim != 1:
        raise ValueError(f'{name} must be one-dimensional, got {value.shape}')
    return np.ascontiguousarray(value)


def _as_bootstrap_values(value, *, name):
    """Detach per-question scalar or vector observations to NumPy."""
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    value = np.asarray(value, dtype=np.float64)
    if value.ndim not in (1, 2):
        raise ValueError(f'{name} must have shape (questions,) or '
                         f'(questions, observations), got {value.shape}')
    return np.ascontiguousarray(value)


def _prepare_question_bootstrap_components(components, *, name):
    """Validate and normalize one method's question-bootstrap components.

    Each component represents one full-set checkpoint or one contrast-set
    checkpoint pair. ``class_by_question`` has one entry for every question in
    the common post-blacklist universe: -1 denotes absence from the component,
    while 0 and 1 denote its two stratification classes. ``question_to_row``
    maps a universe question to its row in the component's confidence/label
    vectors, or -1 when absent.
    """
    prepared = []
    n_questions = None
    for component_i, component in enumerate(components):
        required = {'class_by_question', 'question_to_row', 'confs', 'labels'}
        missing = required - set(component)
        if missing:
            raise ValueError(
                f'{name} component {component_i} is missing {sorted(missing)}')
        classes = _as_bootstrap_numpy(
            component['class_by_question'],
            name=f'{name}[{component_i}].class_by_question', dtype=np.int8)
        row_for_question = _as_bootstrap_numpy(
            component['question_to_row'],
            name=f'{name}[{component_i}].question_to_row', dtype=np.int64)
        confs = _as_bootstrap_values(
            component['confs'], name=f'{name}[{component_i}].confs')
        labels = _as_bootstrap_values(
            component['labels'], name=f'{name}[{component_i}].labels')
        if n_questions is None:
            n_questions = len(classes)
        elif len(classes) != n_questions:
            raise ValueError('All components must use the same question universe')
        if len(row_for_question) != n_questions:
            raise ValueError(
                f'{name} component {component_i} has inconsistent question mapping')
        if confs.shape != labels.shape:
            raise ValueError(
                f'{name} component {component_i} has different confidence and label lengths')
        if not np.isin(classes, (-1, 0, 1)).all():
            raise ValueError('class_by_question values must be -1, 0, or 1')
        member = classes >= 0
        if (row_for_question[member] < 0).any() or (
                row_for_question[member] >= len(confs)).any():
            raise ValueError('Member questions must map to valid component rows')
        if (row_for_question[~member] != -1).any():
            raise ValueError('Nonmember questions must map to -1')
        member_rows = row_for_question[member]
        if (len(member_rows) != len(confs)
                or len(np.unique(member_rows)) != len(member_rows)):
            raise ValueError(
                'Each component row must map to exactly one member question')
        if not np.isfinite(confs).all() or not np.isfinite(labels).all():
            raise ValueError('Bootstrap confidences and labels must be finite')
        if not np.isin(labels, (0.0, 1.0)).all():
            raise ValueError('Bootstrap labels must be binary')
        prepared_component = {
            'class_by_question': classes,
            'question_to_row': row_for_question,
            'confs': confs,
            'labels': labels,
        }
        for key in ('pair_scores', 'class_weights'):
            if key not in component:
                continue
            values = _as_bootstrap_numpy(
                component[key], name=f'{name}[{component_i}].{key}',
                dtype=np.float64)
            if len(values) != len(confs) or not np.isfinite(values).all():
                raise ValueError(
                    f'{name} component {component_i} has invalid {key}')
            if key == 'class_weights' and (values <= 0).any():
                raise ValueError('Bootstrap class_weights must be positive')
            prepared_component[key] = values
        prepared.append(prepared_component)
    if not prepared:
        raise ValueError(f'{name} must contain at least one component')
    return prepared, n_questions


def _validate_paired_question_bootstrap_components(reference, candidate):
    """Require a paired comparison to have identical labels and set geometry."""
    if len(reference) != len(candidate):
        raise ValueError('Paired bootstrap methods have different component counts')
    for component_i, (ref, cand) in enumerate(zip(reference, candidate)):
        for key in ('class_by_question', 'question_to_row', 'labels'):
            if not np.array_equal(ref[key], cand[key]):
                raise ValueError(
                    f'Paired bootstrap components differ in {key}: component '
                    f'{component_i}')


def _sample_shared_question_bootstrap_rows(components, n_questions, rng):
    """Sample all components from one shared IID question sequence.

    Draws that cannot be accepted by any unfinished component are compressed
    away. Conditional on being relevant, the next draw is uniform over the
    active union, exactly matching the original infinite-sequence definition.
    Components process the same draw chunks but enforce their quotas
    independently, preserving their intended question-level dependence.
    """
    remaining = np.array([
        [np.count_nonzero(component['class_by_question'] == class_i)
         for class_i in range(2)]
        for component in components
    ], dtype=np.int64)
    selected_rows = [[] for _ in components]

    while remaining.any():
        active_union = np.zeros(n_questions, dtype=bool)
        for component_i, component in enumerate(components):
            classes = component['class_by_question']
            for class_i in range(2):
                if remaining[component_i, class_i]:
                    active_union |= classes == class_i
        active_questions = np.flatnonzero(active_union)
        if not len(active_questions):
            raise RuntimeError('No active questions remain before quotas were filled')

        # Pick a chunk large enough to finish the slowest currently active
        # class in expectation, plus slack. Rejection-compression makes this
        # particularly important once only a sparse contrast set remains.
        expected_draws = 1
        for component_i, component in enumerate(components):
            class_values = component['class_by_question'][active_questions]
            for class_i in range(2):
                n_remaining = remaining[component_i, class_i]
                if n_remaining:
                    n_available = np.count_nonzero(class_values == class_i)
                    if not n_available:
                        raise RuntimeError(
                            'An unfinished class has no active questions')
                    expected_draws = max(
                        expected_draws,
                        n_remaining * len(active_questions) / n_available)
        chunk_size = max(128, int(np.ceil(expected_draws + 8 * expected_draws ** 0.5)))
        draws = rng.choice(active_questions, size=chunk_size, replace=True)

        for component_i, component in enumerate(components):
            draw_classes = component['class_by_question'][draws]
            for class_i in range(2):
                n_remaining = remaining[component_i, class_i]
                if not n_remaining:
                    continue
                positions = np.flatnonzero(draw_classes == class_i)[:n_remaining]
                if not len(positions):
                    continue
                rows = component['question_to_row'][draws[positions]]
                selected_rows[component_i].append(rows)
                remaining[component_i, class_i] -= len(rows)

    return [np.concatenate(rows) for rows in selected_rows]


def _question_bootstrap_score(components, selected_rows, metric, aggregation):
    """Score one resample after gathering its selected component rows."""
    if aggregation not in {'pooled', 'mean'}:
        raise ValueError(f'Unknown bootstrap aggregation: {aggregation!r}')

    def score(confs, labels):
        confs = torch.from_numpy(confs)
        labels = torch.from_numpy(labels)
        if metric == 'auc':
            return compute_auroc(confs, labels)
        if metric == 'bs':
            return compute_brier(confs, labels)
        if metric == 'smoothece':
            return compute_smooth_ece(confs, labels)
        raise ValueError(f'Unknown bootstrap metric: {metric!r}')

    def delta_score(component, rows):
        if metric in {'delta_0', 'delta_0_bal'}:
            if 'pair_scores' not in component:
                raise ValueError(f'{metric} requires pair_scores')
            values = component['pair_scores'][rows]
        else:
            confs, labels = component['confs'][rows], component['labels'][rows]
            if confs.ndim != 2 or confs.shape[1] != 2:
                raise ValueError(f'{metric} requires two endpoint confidences')
            values = np.where(labels[:, 0] == 1, confs[:, 0] - confs[:, 1],
                              confs[:, 1] - confs[:, 0])
        if metric in {'delta_0_bal', 'delta_bal'}:
            if 'class_weights' not in component:
                raise ValueError(f'{metric} requires class_weights')
            weights = component['class_weights'][rows]
            return float(np.average(values, weights=weights))
        return float(np.mean(values))

    if metric in {'delta_0', 'delta_0_bal', 'delta', 'delta_bal'}:
        if aggregation != 'mean':
            raise ValueError(f'{metric} is defined as a mean over checkpoint pairs')
        return float(np.mean([
            delta_score(component, rows)
            for component, rows in zip(components, selected_rows)
        ]))

    if aggregation == 'pooled':
        confs = np.concatenate([
            component['confs'][rows].reshape(-1) for component, rows
            in zip(components, selected_rows)])
        labels = np.concatenate([
            component['labels'][rows].reshape(-1) for component, rows
            in zip(components, selected_rows)])
        return score(confs, labels)
    return float(np.mean([
        score(component['confs'][rows].reshape(-1),
              component['labels'][rows].reshape(-1))
        for component, rows in zip(components, selected_rows)
    ]))


def _paired_question_bootstrap_metric_chunk(reference, candidate, n_questions,
                                            metric, aggregation, direction,
                                            n_resamples, seed):
    """Compute one worker's paired shared-sequence bootstrap differences."""
    rng = np.random.default_rng(seed)
    differences = np.empty(n_resamples, dtype=np.float64)
    for bootstrap_i in range(n_resamples):
        selected_rows = _sample_shared_question_bootstrap_rows(
            reference, n_questions, rng)
        differences[bootstrap_i] = direction * (
            _question_bootstrap_score(reference, selected_rows, metric, aggregation)
            - _question_bootstrap_score(candidate, selected_rows, metric, aggregation))
    return differences


def _paired_question_bootstrap_across_seeds_chunk(
        reference_by_seed, candidate_by_seed, n_questions, metric, aggregation,
        direction, n_resamples, seed):
    """Compute paired differences after averaging the metric across seeds."""
    rng = np.random.default_rng(seed)
    differences = np.empty(n_resamples, dtype=np.float64)
    sequence_components = reference_by_seed[0]
    for bootstrap_i in range(n_resamples):
        selected_rows = _sample_shared_question_bootstrap_rows(
            sequence_components, n_questions, rng)
        reference_score = np.mean([
            _question_bootstrap_score(components, selected_rows, metric, aggregation)
            for components in reference_by_seed
        ])
        candidate_score = np.mean([
            _question_bootstrap_score(components, selected_rows, metric, aggregation)
            for components in candidate_by_seed
        ])
        differences[bootstrap_i] = direction * (reference_score - candidate_score)
    return differences


def paired_question_stratified_bootstrap_is_significantly_better(
        reference_components, candidate_components, *, metric,
        aggregation='pooled', higher_is_better=True,
        confidence_level=0.95, n_resamples=10000, seed=17, n_jobs=1):
    """One-sided paired bootstrap over shared IID question sequences.

    Unlike :func:`paired_bootstrap_metric_is_significantly_better`, this
    resamples questions rather than flattened checkpoint/question rows. Each
    component independently applies its original two-class quotas to one
    common infinite question sequence. Components can represent full-set
    checkpoints (correct/incorrect classes) or contrast pairs
    (improvement/regression classes).

    ``reference_components`` and ``candidate_components`` are parallel lists
    of dicts containing ``class_by_question``, ``question_to_row``, ``confs``,
    and binary ``labels``. The methods must share every field except
    ``confs``. ``aggregation`` is either ``'pooled'`` or ``'mean'`` over
    components. ``metric`` is one of ``'auc'``, ``'bs'``, ``'ece'``,
    ``'smoothece'``, ``'delta_0'``, ``'delta_0_bal'``, ``'delta'``, or
    ``'delta_bal'``. Delta metrics require contrast-pair components with
    ``pair_scores`` and, for balanced variants, ``class_weights``.
    """
    if not 0 < confidence_level < 1:
        raise ValueError('confidence_level must lie in (0, 1)')
    if n_resamples <= 0 or n_jobs <= 0:
        raise ValueError('n_resamples and n_jobs must be positive')
    reference, n_questions = _prepare_question_bootstrap_components(
        reference_components, name='reference_components')
    candidate, candidate_n_questions = _prepare_question_bootstrap_components(
        candidate_components, name='candidate_components')
    if candidate_n_questions != n_questions:
        raise ValueError('Paired bootstrap methods use different question universes')
    _validate_paired_question_bootstrap_components(reference, candidate)

    direction = 1 if higher_is_better else -1
    n_workers = min(n_jobs, n_resamples)
    if n_workers == 1:
        differences = _paired_question_bootstrap_metric_chunk(
            reference, candidate, n_questions, metric, aggregation, direction,
            n_resamples, seed)
    else:
        chunk_sizes = [n_resamples // n_workers] * n_workers
        for worker_i in range(n_resamples % n_workers):
            chunk_sizes[worker_i] += 1
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(
                    _paired_question_bootstrap_metric_chunk,
                    reference, candidate, n_questions, metric, aggregation,
                    direction, chunk_size, seed + worker_i)
                for worker_i, chunk_size in enumerate(chunk_sizes)
            ]
            differences = np.concatenate([future.result() for future in futures])
    return bool(np.quantile(differences, 1 - confidence_level) > 0)


def paired_question_stratified_bootstrap_across_seeds_differences(
        reference_components_by_seed, candidate_components_by_seed, *, metric,
        aggregation='pooled', higher_is_better=True,
        confidence_level=0.95, n_resamples=10000, seed=17, n_jobs=1,
        cache_path=None):
    """Return paired bootstrap differences for a metric averaged across seeds.

    Each replicate uses one shared question sequence and evaluates the metric
    for every seed before taking the seed mean.  A single seed on one side is
    repeated when compared with a multi-seed method, which is appropriate for
    seed-independent baselines such as self-consistency.
    """
    if not 0 < confidence_level < 1:
        raise ValueError('confidence_level must lie in (0, 1)')
    if n_resamples <= 0 or n_jobs <= 0:
        raise ValueError('n_resamples and n_jobs must be positive')
    if not reference_components_by_seed or not candidate_components_by_seed:
        raise ValueError('Each method must provide at least one seed')
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            differences = np.load(cache_path)
            if differences.shape == (n_resamples,):
                return differences
            warnings.warn(
                f'Ignoring bootstrap cache with wrong shape: {cache_path}')

    n_seeds = max(len(reference_components_by_seed), len(candidate_components_by_seed))
    if len(reference_components_by_seed) == 1:
        reference_components_by_seed = reference_components_by_seed * n_seeds
    if len(candidate_components_by_seed) == 1:
        candidate_components_by_seed = candidate_components_by_seed * n_seeds
    if (len(reference_components_by_seed) != n_seeds
            or len(candidate_components_by_seed) != n_seeds):
        raise ValueError(
            'Methods must have the same seed count, unless one is a '
            'seed-independent single-seed baseline')

    reference_by_seed, candidate_by_seed = [], []
    n_questions = None
    for seed_i, (reference, candidate) in enumerate(zip(
            reference_components_by_seed, candidate_components_by_seed)):
        reference, reference_n_questions = _prepare_question_bootstrap_components(
            reference, name=f'reference_components_by_seed[{seed_i}]')
        candidate, candidate_n_questions = _prepare_question_bootstrap_components(
            candidate, name=f'candidate_components_by_seed[{seed_i}]')
        if reference_n_questions != candidate_n_questions:
            raise ValueError('Paired bootstrap methods use different question universes')
        _validate_paired_question_bootstrap_components(reference, candidate)
        if n_questions is None:
            n_questions = reference_n_questions
            sequence_components = reference
        else:
            if reference_n_questions != n_questions:
                raise ValueError('Bootstrap seeds use different question universes')
            _validate_paired_question_bootstrap_components(
                sequence_components, reference)
        reference_by_seed.append(reference)
        candidate_by_seed.append(candidate)

    direction = 1 if higher_is_better else -1
    n_workers = min(n_jobs, n_resamples)
    worker_args = (reference_by_seed, candidate_by_seed, n_questions, metric,
                   aggregation, direction)
    if n_workers == 1:
        differences = _paired_question_bootstrap_across_seeds_chunk(
            *worker_args, n_resamples, seed)
    else:
        chunk_sizes = [n_resamples // n_workers] * n_workers
        for worker_i in range(n_resamples % n_workers):
            chunk_sizes[worker_i] += 1
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(
                    _paired_question_bootstrap_across_seeds_chunk,
                    *worker_args, chunk_size, seed + worker_i)
                for worker_i, chunk_size in enumerate(chunk_sizes)
            ]
            differences = np.concatenate([future.result() for future in futures])
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = cache_path.with_suffix(cache_path.suffix + '.tmp')
        with temporary_path.open('wb') as file:
            np.save(file, differences)
        os.replace(temporary_path, cache_path)
    return differences


def paired_question_stratified_bootstrap_across_seeds_is_significantly_better(
        reference_components_by_seed, candidate_components_by_seed, *, metric,
        aggregation='pooled', higher_is_better=True,
        confidence_level=0.95, n_resamples=10000, seed=17, n_jobs=1):
    """One-sided test based on paired bootstrap differences across seeds."""
    differences = paired_question_stratified_bootstrap_across_seeds_differences(
        reference_components_by_seed, candidate_components_by_seed,
        metric=metric, aggregation=aggregation,
        higher_is_better=higher_is_better,
        confidence_level=confidence_level, n_resamples=n_resamples,
        seed=seed, n_jobs=n_jobs)
    return bool(np.quantile(differences, 1 - confidence_level) > 0)


def _discover_shards(base_dir):
    """Find all ``{beg}-{end}`` shard subdirectories under *base_dir*.

    Returns a sorted list of ``(beg, end, shard_path)`` tuples.
    """
    shard_re = re.compile(r'^(\d+)-(\d+)$')
    shards = []
    for entry in os.listdir(base_dir):
        m = shard_re.match(entry)
        if m and os.path.isdir(os.path.join(base_dir, entry)):
            shards.append((int(m.group(1)), int(m.group(2)),
                           os.path.join(base_dir, entry)))
    shards.sort()
    return shards


def _load_file(path):
    """Load a ``.pt`` or ``.pkl`` file."""
    if path.endswith('.pt'):
        return torch.load(path, weights_only=True, map_location='cpu')
    elif path.endswith('.pkl'):
        with open(path, 'rb') as f:
            return pkl.load(f)
    else:
        raise ValueError(f"Unsupported file extension: {path}")


def _concat(parts, ext):
    """Concatenate loaded file parts (torch tensors or lists)."""
    if ext == '.pt':
        return torch.cat(parts, dim=0)
    else:
        out = []
        for p in parts:
            out.extend(p)
        return out


def parse_ranges(tokens):
    """Parse range tokens into a list of ``(beg, end)`` tuples.

    Each token is either a ``'BEG-END'`` string or a ``(beg, end)`` pair.
    Ranges must be sorted and non-overlapping.
    """
    ranges = []
    for tok in tokens:
        if isinstance(tok, str):
            beg, end = (int(x) for x in tok.split('-'))
        else:
            beg, end = tok
        assert beg < end, f"invalid range {beg}-{end}"
        assert not ranges or beg >= ranges[-1][1], \
            f"ranges must be sorted and non-overlapping: {tokens}"
        ranges.append((beg, end))
    return ranges


def assemble_ranges(segments, ranges, load_fn, ext, desc):
    """Assemble rows covering absolute index *ranges* from shard *segments*.

    Args:
        segments: list of ``(seg_beg, seg_end, key)`` tuples giving the
            absolute index range each segment's rows cover; *key* is passed
            to *load_fn*.
        ranges: requested absolute ``(beg, end)`` ranges (sorted,
            non-overlapping), e.g. from parse_ranges.
        load_fn: ``key -> data`` with rows indexed along the first axis
            (cached internally, so each key is loaded at most once).
        ext: ``'.pt'`` (torch tensors) or ``'.pkl'`` (lists).
        desc: description of the data source for error messages.

    Returns:
        The concatenated data covering exactly *ranges*, in order.
    """
    cache = {}

    def _cached_load(key):
        if key not in cache:
            cache[key] = load_fn(key)
        return cache[key]

    parts = []
    for beg, end in ranges:
        pos = beg
        while pos < end:
            seg = next(((sb, se, key) for sb, se, key in segments
                        if sb <= pos < se), None)
            if seg is None:
                raise FileNotFoundError(
                    f"No shard covers index {pos} for {desc}")
            seg_beg, seg_end, key = seg
            take_end = min(end, seg_end)
            parts.append(_cached_load(key)[pos - seg_beg:take_end - seg_beg])
            pos = take_end

    return _concat(parts, ext)


def load_sharded(base_dir, filename, ranges):
    """Load *filename* across shard directories and return rows for *ranges*.

    Shard directories are ``{base_dir}/{shard_beg}-{shard_end}/`` (or with an
    extra subdirectory — see *filename*), where the directory name gives the
    absolute index range the shard covers.  *filename* may contain path
    separators, e.g. ``beam/judge.pt``.

    For ``.pt`` files the first dimension is the example axis and rows are
    selected via tensor slicing.  For ``.pkl`` files the data is assumed to be
    a list indexed by example.

    Args:
        base_dir: parent directory that contains the shard subdirectories
            (e.g. ``results/conset/triviaqa/Olmo-3-1025-7B/stage1-step141000``).
        filename: path relative to each shard directory
            (e.g. ``beam/beam_group_strs.pkl``).
        ranges: list of absolute ``(beg, end)`` index ranges (sorted,
            non-overlapping), e.g. from parse_ranges.

    Returns:
        The concatenated and sliced data (torch.Tensor or list).
    """
    shards = _discover_shards(base_dir)
    if not shards:
        raise FileNotFoundError(
            f"No shard directories found under {base_dir}")

    ext = os.path.splitext(filename)[1]
    if ext not in ('.pt', '.pkl'):
        raise ValueError(f"Unsupported file extension '{ext}' for {filename}")

    segments = [(beg, end, os.path.join(path, filename))
                for beg, end, path in shards]
    return assemble_ranges(segments, ranges, _load_file, ext,
                           f'{base_dir}/{filename}')
