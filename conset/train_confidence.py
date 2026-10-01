"""
Train a LoRA adapter to predict confidence from a question and candidate answer.

With multiple --revisions, one shared LoRA adapter is trained across all
checkpoints: each checkpoint trains on its own gen_cands outputs, and the
adapter parameters are tied across the per-checkpoint PEFT models so gradients
from every checkpoint accumulate into the same weights. Batches contain
cfg['batch_size'] questions, but each question contributes its answers/labels
from every checkpoint, so the effective batch size is multiplied by the
number of checkpoints.

Usage:
    python -m conset.train_confidence --config conset/train_configs/lora_triviaqa_acc.yaml \
        --model_name allenai/Olmo-3-1025-7B --revisions stage1-step141000

    python -m conset.train_confidence \
        --config conset/train_configs/lora_triviaqa_acc_lr2e-4.yaml \
        --revisions stage1-step141000 stage1-step283000
"""

import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from .dataset_configs import get_dataset_config
from .utils import load_sharded, parse_ranges


# Generation artifacts stay in the repository checkout.  Adapters can be much
# larger (especially for OLMo 3 32B), so LoRA training writes them to scratch.
# Keep the two roots separate: candidate/judge data and confidence-prediction
# artifacts stay under ``results/conset`` in the checkout, while LoRA adapters
# are stored and loaded exclusively from scratch.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_HOME_CONSET_RESULTS_ROOT = _PROJECT_ROOT / 'results' / 'conset'
_ADAPTER_ROOT = None


def set_adapter_root(adapter_root):
    """Set the scratch-root override used for large adapter artifacts."""
    global _ADAPTER_ROOT
    _ADAPTER_ROOT = Path(adapter_root) if adapter_root else None


def adapter_results_root():
    """Return the root containing LoRA adapters and hidden-state streams."""
    adapter_root = _ADAPTER_ROOT or os.environ.get('SCRATCH')
    if not adapter_root:
        raise RuntimeError(
            'An adapter root is required; pass --adapter_root or set SCRATCH. '
            'Expected $SCRATCH/persistent-calibration/results/conset')
    return Path(adapter_root) / 'persistent-calibration' / 'results' / 'conset'


def _adapter_relative_dir(dataset_name, model_name, revision, *parts):
    half_model_name = model_name.split('/')[-1]
    return Path(dataset_name) / half_model_name / revision / Path(*parts)


def get_adapter_dir(dataset_name, model_name, revision, *parts, storage='read'):
    """Return a scratch adapter directory or a home prediction-output path.

    ``storage='read'`` resolves only the requested scratch adapter directory
    and requires it to exist. ``storage='home'`` is retained solely for the
    confidence-prediction artifacts that are intentionally saved under the
    checkout's ``results/conset`` tree.
    """
    relative = _adapter_relative_dir(dataset_name, model_name, revision, *parts)
    scratch_path = adapter_results_root() / relative
    if storage == 'scratch':
        return str(scratch_path)
    if storage == 'home':
        return str(_HOME_CONSET_RESULTS_ROOT / relative)
    if storage != 'read':
        raise ValueError(f'Unknown adapter storage: {storage!r}')
    if not scratch_path.is_dir():
        raise FileNotFoundError(
            f'Adapter directory not found on scratch: {scratch_path}')
    return str(scratch_path)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_train_config(path):
    """Load and validate a YAML training config."""
    with open(path) as f:
        cfg = yaml.safe_load(f)

    required = [
        'weight_type', 'dataset', 'lr',
        'train_ranges', 'batch_size', 'n_epoch',
    ]
    for key in required:
        if key not in cfg:
            raise ValueError(f"train config {path} missing required key '{key}'")

    if cfg['weight_type'] != 'lora':
        raise ValueError(
            f"weight_type must be 'lora', got {cfg['weight_type']!r}")
    cfg['lr'] = float(cfg['lr'])
    cfg['n_max_cand'] = int(cfg.get('n_max_cand', 1))
    cfg['cand_top_p'] = float(cfg.get('cand_top_p', 1.0))
    cfg['answer_weight_uniform'] = cfg.get('answer_weight_uniform', False)
    cfg['data_revisions'] = cfg.get('data_revisions')
    cfg['random_labels_on_last_ckpt'] = cfg.get(
        'random_labels_on_last_ckpt', False)
    if cfg['n_max_cand'] < 1:
        raise ValueError('n_max_cand must be at least 1')
    if not 0 < cfg['cand_top_p'] <= 1:
        raise ValueError('cand_top_p must be in (0, 1]')
    if not isinstance(cfg['answer_weight_uniform'], bool):
        raise ValueError('answer_weight_uniform must be a boolean')
    if not isinstance(cfg['random_labels_on_last_ckpt'], bool):
        raise ValueError('random_labels_on_last_ckpt must be a boolean')
    if cfg['data_revisions'] is not None:
        if (not isinstance(cfg['data_revisions'], list)
                or not cfg['data_revisions']
                or not all(isinstance(revision, str) and revision
                           for revision in cfg['data_revisions'])):
            raise ValueError(
                'data_revisions must be a nonempty list of revision strings')
        if len(set(cfg['data_revisions'])) != len(cfg['data_revisions']):
            raise ValueError('data_revisions must not contain duplicates')
    supported_keys = {
        'weight_type', 'dataset', 'lr', 'train_ranges',
        'n_max_cand', 'cand_top_p', 'batch_size', 'n_epoch',
        'answer_weight_uniform', 'data_revisions',
        'random_labels_on_last_ckpt',
        'lora_r', 'lora_alpha', 'lora_dropout',
    }
    unused_keys = sorted(set(cfg) - supported_keys)
    if unused_keys:
        print('Warning: ignoring unused train config keys: '
              + ', '.join(unused_keys), flush=True)
    # Normalize dataset to a list (backward compat with single-string configs)
    if isinstance(cfg['dataset'], str):
        cfg['dataset'] = [cfg['dataset']]

    # Normalize train_ranges to per-dataset.
    # Single-dataset backward compat: flat list of strings (e.g. ["0-5000"])
    # is wrapped into a one-element nested list.
    # Multi-dataset: must be a nested list matching the dataset list length.
    raw_ranges = cfg['train_ranges']
    if len(cfg['dataset']) == 1 and raw_ranges and isinstance(raw_ranges[0], str):
        cfg['train_ranges'] = [parse_ranges(raw_ranges)]
    else:
        cfg['train_ranges'] = [parse_ranges(r) for r in raw_ranges]

    if len(cfg['train_ranges']) != len(cfg['dataset']):
        raise ValueError(
            f"train_ranges has {len(cfg['train_ranges'])} entries but "
            f"dataset has {len(cfg['dataset'])} entries")

    return cfg


# ---------------------------------------------------------------------------
# Digit-token confidence extraction
# ---------------------------------------------------------------------------

def get_single_digit_token_ids(tokenizer):
    """Get token IDs for '0', '1', ..., '9' (each must be a single token)."""
    ids = []
    for i in range(10):
        t = str(i)
        token_ids = tokenizer.encode(t, add_special_tokens=False)
        assert len(token_ids) == 1, (
            f"'{t}' tokenizes to {len(token_ids)} tokens {token_ids}, expected 1")
        ids.append(token_ids[0])
    return torch.tensor(ids)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def tokenize_confidence(question, answer, tokenizer, task, ex_i=0,
                        is_instruct=False):
    """Tokenize a confidence-prediction prompt.

    The assistant response is prefilled with "Confidence: " so the model's
    next token is a digit 0-9.

    """
    user_content = task.build_confidence_prompt(question, answer, ex_i)
    if is_instruct:
        if tokenizer.chat_template is None:
            raise ValueError('--is_instruct requires a tokenizer with a chat template')
        msg = [
            {'role': 'user', 'content': user_content},
            {'role': 'assistant', 'content': "Confidence: "},
        ]
        encoded = tokenizer.apply_chat_template(
            [msg],
            add_generation_prompt=False,
            continue_final_message=True,
            enable_thinking=False,
            padding=True,
            return_tensors="pt",
        )
    else:
        text = f"{user_content}\nConfidence: "
        encoded = tokenizer(text, return_tensors="pt")

    if isinstance(encoded, torch.Tensor):
        input_ids = encoded
        attention_mask = torch.ones_like(input_ids)
    else:
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']

    return input_ids.squeeze(0), attention_mask.squeeze(0)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ConfidenceDataset(TorchDataset):
    """Dataset of (question, candidate_answer) pairs with confidence targets.

    Each item has:
        - input_ids, attention_mask: tokenized prompt
        - target: cross-entropy target
        - weight: instance weight (proportional to genprob within question)
        - q_i: question index (for grouping in collator)
    """

    def __init__(self, examples, tokenizer, task=None, is_instruct=False,
                 expected_q_idxs=None):
        """
        Args:
            examples: list of dicts with keys:
                'question', 'answer', 'target', 'weight', 'q_i'
            tokenizer: HuggingFace tokenizer.
            task: optional DatasetConfig for dataset-specific prompts.
            is_instruct: format prompts with the tokenizer chat template.
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.task = task
        self.is_instruct = is_instruct
        self.expected_q_idxs = (None if expected_q_idxs is None
                                else list(expected_q_idxs))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        task = ex.get('task', self.task)
        ex_i = ex.get('ex_i', ex['q_i'])
        ids, mask = tokenize_confidence(
            ex['question'], ex['answer'], self.tokenizer,
            task=task, ex_i=ex_i, is_instruct=self.is_instruct)
        return {
            'input_ids': ids,
            'attention_mask': mask,
            'target': torch.tensor(ex['target'], dtype=torch.float32),
            'weight': torch.tensor(ex['weight'], dtype=torch.float32),
            'q_i': ex['q_i'],
            'ckpt_i': ex.get('ckpt_i', 0),
            'correct': bool(ex.get('correct', False)),
            'is_top_answer': bool(ex.get('is_top_answer', False)),
        }


class LeftPadCollator:
    """Collates variable-length sequences with left-padding."""

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        batch_ids = [f['input_ids'] for f in features]
        batch_masks = [f['attention_mask'] for f in features]
        targets = torch.stack([f['target'] for f in features])
        weights = torch.stack([f['weight'] for f in features])
        qst_idxs = torch.tensor([f['q_i'] for f in features], dtype=torch.long)
        ckpt_idxs = torch.tensor([f['ckpt_i'] for f in features], dtype=torch.long)
        correct = torch.tensor([f['correct'] for f in features], dtype=torch.bool)
        is_top_answer = torch.tensor([f['is_top_answer'] for f in features],
                                     dtype=torch.bool)

        max_len = max(ids.shape[0] for ids in batch_ids)
        padded_ids = torch.full(
            (len(batch_ids), max_len), self.pad_token_id, dtype=batch_ids[0].dtype)
        padded_masks = torch.zeros(
            (len(batch_ids), max_len), dtype=batch_masks[0].dtype)
        for i, (ids, mask) in enumerate(zip(batch_ids, batch_masks)):
            padded_ids[i, max_len - ids.shape[0]:] = ids
            padded_masks[i, max_len - mask.shape[0]:] = mask

        return {
            'input_ids': padded_ids,
            'attention_mask': padded_masks,
            'target': targets,
            'weight': weights,
            'q_i': qst_idxs,
            'ckpt_i': ckpt_idxs,
            'correct': correct,
            'is_top_answer': is_top_answer,
        }


# ---------------------------------------------------------------------------
# Data loading: build examples from gen_cands outputs
# ---------------------------------------------------------------------------

def _get_base_result_dir(dataset_name, model_name, revision):
    """Get the base result directory (without shard suffix).

    Returns e.g. ``results/conset/triviaqa/Olmo-3-1025-7B/stage1-step141000``.
    """
    half_model_name = model_name.split('/')[-1]
    return f"results/conset/{dataset_name}/{half_model_name}/{revision}"


def _read_cand_type(base_dir, gen_config_name):
    """Read the cand_type from the gen_config.yaml stored in the first shard."""
    from .utils import _discover_shards
    shards = _discover_shards(base_dir)
    for _, _, shard_path in shards:
        cfg_path = os.path.join(shard_path, gen_config_name, 'gen_config.yaml')
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                return yaml.safe_load(f)['cand_type']
    raise FileNotFoundError(
        f"No gen_config.yaml found under {base_dir}/*/{gen_config_name}/")


def load_group_data(dataset_name, model_name, revision, gen_config_name,
                    ranges, qa_examples=None):
    """Load QA examples and gen_cands group outputs via cross-shard helper.

    Returns:
        (qa_examples, group_strs, group_probs, judge_labels)
    """
    if qa_examples is None:
        task = get_dataset_config(dataset_name, ranges=ranges)
        qa_examples = task.load_examples()

    base_dir = _get_base_result_dir(dataset_name, model_name, revision)
    cand_type = _read_cand_type(base_dir, gen_config_name)

    group_strs = load_sharded(
        base_dir, f'{gen_config_name}/{cand_type}_group_strs.pkl', ranges)
    group_probs = load_sharded(
        base_dir, f'{gen_config_name}/{cand_type}_group_probs.pt', ranges)
    judge_labels = load_sharded(
        base_dir, f'{gen_config_name}/judge.pt', ranges)

    return qa_examples, group_strs, group_probs, judge_labels


def load_training_data(dataset_name, model_name, revision, gen_config_name,
                       ranges, n_max_cand=1, cand_top_p=1.0,
                       answer_weight_uniform=False,
                       blacklisted_qis=None):
    """Load and prepare training examples from gen_cands group outputs.

    For each question, selects highest-generation-probability distinct answer
    groups, capped by ``n_max_cand`` and by the number needed to reach
    ``cand_top_p`` cumulative generation probability. Weights within a
    question are proportional to the selected groups' generation probabilities
    and sum to 1, unless ``answer_weight_uniform`` is set, in which case each
    selected answer receives equal weight.

    Returns:
        examples: list of dicts with keys
            'question', 'answer', 'target', 'weight', 'q_i'
    """
    # Judge labels provide the binary accuracy training targets.
    qa_examples, group_strs, group_probs, judge_labels = load_group_data(
        dataset_name, model_name, revision, gen_config_name, ranges)

    examples = []
    blacklisted_qis = set() if blacklisted_qis is None else blacklisted_qis

    for qi, (qa, groups) in enumerate(zip(qa_examples, group_strs)):
        if qi in blacklisted_qis:
            continue
        if groups is None or len(groups) == 0:
            continue

        # Padding is excluded.  A no-attempt label (3) is an incorrect answer.
        chosen_grps = []
        for gi in range(len(groups)):
            label = judge_labels[qi, gi].item()
            if label != -1:
                chosen_grps.append(gi)
        if not chosen_grps:
            continue

        # Sort explicitly rather than relying on the serialized group order.
        chosen_grps.sort(key=lambda gi: group_probs[qi, gi].item(), reverse=True)

        if cand_top_p < 1.0:
            cumulative_prob = 0.0
            n_top_p = 0
            for gi in chosen_grps:
                cumulative_prob += group_probs[qi, gi].float().item()
                n_top_p += 1
                if cumulative_prob >= cand_top_p:
                    break
            chosen_grps = chosen_grps[:min(n_max_cand, n_top_p)]
        else:
            chosen_grps = chosen_grps[:n_max_cand]

        # Normalize weights to sum to 1, either by generation probability or
        # uniformly over the selected candidate answers.
        probs = group_probs[qi, chosen_grps].float()
        if answer_weight_uniform:
            weights = torch.full_like(probs, 1.0 / len(chosen_grps))
        else:
            prob_sum = probs.sum()
            assert prob_sum > 0
            weights = probs / prob_sum

        for j, gi in enumerate(chosen_grps):
            answer = groups[gi][0]  # representative answer from the group
            correct = judge_labels[qi, gi].item() == 1

            examples.append({
                'question': qa.question,
                'answer': answer,
                'weight': weights[j].item(),
                'q_i': qi,
                'correct': correct,
                'is_top_answer': j == 0,
            })

    set_training_targets(examples)
    return examples


def set_training_targets(examples):
    """Set binary accuracy targets from each example's correctness label."""
    for ex in examples:
        ex['target'] = float(ex['correct'])


# ---------------------------------------------------------------------------
# WSD Learning Rate Schedule
# ---------------------------------------------------------------------------

class WSDScheduler(torch.optim.lr_scheduler.LambdaLR):
    """Per-epoch Warmup-Stable-Decay (WSD) learning rate schedule.

    Every epoch ends with a cosine decay to 0.1*peak_lr so that each epoch's
    checkpoint is cooled down; the LR jumps back to peak_lr at the start of the
    next epoch (no re-warmup).

    - Warmup: linear to peak_lr over the first warmup_frac of epoch 1 only.
    - Stable: constant at peak_lr.
    - Decay: cosine decay from peak_lr to min_lr_ratio*peak_lr over the last
      decay_frac of every epoch.
    """

    def __init__(self, optimizer, steps_per_epoch, warmup_frac=0.0,
                 decay_frac=0.2, min_lr_ratio=0.1):
        self.steps_per_epoch = steps_per_epoch
        self.warmup_steps = int(steps_per_epoch * warmup_frac)
        self.decay_steps = int(steps_per_epoch * decay_frac)
        self.min_lr_ratio = min_lr_ratio

        def lr_lambda(step):
            if step < self.warmup_steps:
                # Linear warmup (first epoch only). Strictly positive.
                return (step + 1) / self.warmup_steps
            epoch_step = step % self.steps_per_epoch
            stable_end = self.steps_per_epoch - self.decay_steps
            if epoch_step < stable_end:
                return 1.0
            # Cosine decay at the end of every epoch
            decay_step = epoch_step - stable_end
            cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_step / self.decay_steps))
            return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine_decay

        super().__init__(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def apply_lora(model, cfg):
    """Wrap a model with a LoRA adapter built from the train config."""
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=cfg.get('lora_r', 8),
        lora_alpha=cfg.get('lora_alpha', 16),
        target_modules="all-linear",
        lora_dropout=cfg.get('lora_dropout', 0.0),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    return model


def seed_lora_initialization(seed):
    """Reset the RNGs used when PEFT creates LoRA parameters."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tie_trainable_params(src_model, dst_model):
    """Replace dst_model's trainable (LoRA) params with src_model's Parameters.

    After tying, both models literally share the same Parameter objects, so
    gradients from both accumulate together and optimizer steps update both.
    """
    src_params = dict(src_model.named_parameters())
    for name, param in list(dst_model.named_parameters()):
        if not param.requires_grad:
            continue
        mod_path, _, attr = name.rpartition('.')
        setattr(dst_model.get_submodule(mod_path), attr, src_params[name])


def load_model_and_tokenizer(model_name, revision,
                             device_map: 'str | dict' = 'auto'):
    """Load a causal LM (bfloat16) and its tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, revision=revision, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name, revision=revision, device_map=device_map,
        dtype=torch.bfloat16)

    return model, tokenizer


def load_model_and_tokenizer_from_dir(model_dir):
    """Load a full fine-tuned model saved with save_pretrained()."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, device_map='auto', dtype=torch.bfloat16)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Forward pass: confidence from LM head digit logits
# ---------------------------------------------------------------------------

def forward_confidence(model, input_ids, attention_mask, digit_token_ids):
    """Forward pass returning confidence as weighted average over digit tokens.

    The LM head (frozen) produces logits. We select logits for tokens '0'-'9',
    softmax, and compute weighted average.

    Returns:
        prob: confidence in [0, 1]
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    last_logits = outputs.logits[:, -1, :].float()          # [batch, vocab]
    digit_logits = last_logits[:, digit_token_ids]          # [batch, 10]
    digit_probs = torch.softmax(digit_logits, dim=-1)
    values = torch.arange(10, device=digit_logits.device, dtype=digit_logits.dtype) / 9.0
    prob = (digit_probs * values).sum(dim=-1)

    clamped = prob.clamp(1e-6, 1 - 1e-6)
    prob = prob + (clamped - prob).detach()  # straight-through clamp
    logit = torch.logit(prob)

    return logit, prob


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _try_batch(models, dataset, collator, digit_token_ids, optimizer, scheduler,
                batch_idxs, micro_bs, verbose=False, grad_sync_fn=None):
    """Try processing one batch with the given micro_bs.

    ``models`` is a list indexed by each example's ``ckpt_i``; each example is
    forwarded through its own checkpoint's model (single-ckpt training passes a
    one-element list). Models may live on different GPUs (device_map='auto'),
    so each micro-batch is moved to its own model's device.

    Distributed (checkpoint-partitioned) training: entries for checkpoints not
    owned by this rank are None and skipped; delta grads and the batch loss
    are summed across ranks via ``grad_sync_fn`` before the clip/step.

    Args:
        grad_sync_fn: optional callable ``(trainable_params, batch_loss) ->
            global_batch_loss``.  When not None it is called after
            forward/backward to synchronize gradients across ranks (e.g.
            ``all_reduce_delta_grads``).  LoRA training leaves this as None.
    Returns ``(batch_loss, True)`` on success, or ``(0.0, False)`` on OOM.
    On OOM the optimizer/model state is cleaned up so the caller can retry.
    """
    import time as _time

    features = [dataset[i] for i in batch_idxs]
    batch = collator(features)

    ids = batch['input_ids']
    mask = batch['attention_mask']
    targets = batch['target']
    weights = batch['weight']
    ckpt_idxs = batch['ckpt_i']

    optimizer.zero_grad()
    batch_loss = 0.0
    try:
        for ck, model in enumerate(models):
            if model is None:
                continue  # checkpoint owned by another rank
            model_device = next(model.parameters()).device
            sel = (ckpt_idxs == ck).nonzero(as_tuple=True)[0]
            for mib_beg in range(0, len(sel), micro_bs):
                mib = sel[mib_beg:mib_beg + micro_bs]
                mib_ids = ids[mib].to(model_device)
                mib_mask = mask[mib].to(model_device)
                mib_targets = targets[mib].to(model_device)
                mib_weights = weights[mib].to(model_device)

                logit, _ = forward_confidence(
                    model, mib_ids, mib_mask, digit_token_ids)
                per_sample_loss = nn.functional.binary_cross_entropy_with_logits(
                    logit, mib_targets, reduction='none')
                loss = (per_sample_loss * mib_weights).sum()
                loss.backward()
                batch_loss += loss.item()

    except (RuntimeError, torch.cuda.OutOfMemoryError):
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        return 0.0, False

    first_model = next(m for m in models if m is not None)
    trainable = [p for p in first_model.parameters() if p.requires_grad]
    if grad_sync_fn is not None:
        if verbose:
            import torch.distributed as _dist
            print(f"[rank {_dist.get_rank()} {_time.strftime('%H:%M:%S')}] "
                  f"forward/backward done (local loss {batch_loss:.4f}); "
                  f"entering grad all-reduce", flush=True)
        batch_loss = grad_sync_fn(trainable, batch_loss)
        if verbose:
            import torch.distributed as _dist
            print(f"[rank {_dist.get_rank()} {_time.strftime('%H:%M:%S')}] "
                  f"grad all-reduce done", flush=True)

    nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
    optimizer.step()
    scheduler.step()
    if verbose and grad_sync_fn is not None:
        import torch.distributed as _dist
        print(f"[rank {_dist.get_rank()} {_time.strftime('%H:%M:%S')}] "
              f"optimizer step done", flush=True)
    return batch_loss, True


def group_examples_by_question(dataset, batch_size, seed):
    """Create batches where all answers to a question stay together.

    Args:
        dataset: ConfidenceDataset instance.
        batch_size: number of questions per batch.
        seed: random seed for shuffling.
    Returns:
        batches: list of lists of dataset indices.
    """
    # Group dataset indices by question
    qi_to_idxs = {}
    for i in range(len(dataset)):
        qi = dataset.examples[i]['q_i']
        qi_to_idxs.setdefault(qi, []).append(i)

    q_idxs = list(qi_to_idxs.keys())
    if dataset.expected_q_idxs is not None:
        assert q_idxs == dataset.expected_q_idxs, (
            "Training question indices differ from train_ranges with the "
            "blacklist removed: "
            f"expected {len(dataset.expected_q_idxs)}, got {len(q_idxs)}; "
            f"missing={sorted(set(dataset.expected_q_idxs) - set(q_idxs))[:10]}, "
            f"unexpected={sorted(set(q_idxs) - set(dataset.expected_q_idxs))[:10]}")
    rng = np.random.RandomState(seed)
    rng.shuffle(q_idxs)

    batches = []
    current_batch = []
    current_qst_count = 0

    for qi in q_idxs:
        current_batch.extend(qi_to_idxs[qi])
        current_qst_count += 1
        if current_qst_count >= batch_size:
            batches.append(current_batch)
            current_batch = []
            current_qst_count = 0

    if current_batch:
        batches.append(current_batch)

    return batches


def get_contrast_questions(examples):
    """Return (contrast_qs, n_total_qs) for multi-ckpt training examples.

    A contrast question is one whose top (highest-genprob) prediction is
    correct at some trained ckpt and wrong at another.  The first example per
    (q_i, ckpt_i) is treated as the ckpt's top-genprob group.
    """
    top_correct = {}
    seen = set()
    for ex in examples:
        key = (ex['q_i'], ex['ckpt_i'])
        if key not in seen:
            seen.add(key)
            top_correct.setdefault(ex['q_i'], set()).add(ex['correct'])
    contrast_qs = {q for q, corrects in top_correct.items()
                   if len(corrects) == 2}
    return contrast_qs, len(top_correct)


def train_one_epoch(models, dataset, collator, digit_token_ids, optimizer, scheduler,
                    batch_size, micro_bs, seed, epoch,
                    batches=None, cfg=None, weight_info=None,
                    grad_sync_fn=None,
                    step_callback=None, max_batches=None):
    """Train for one epoch with gradient accumulation and question-grouped batches.

    On OOM, halves micro_bs and retries the batch. The reduced micro_bs persists
    for subsequent batches.

    Args:
        batches: optional pre-computed list of index lists. When omitted,
            questions are shuffled uniformly and kept intact within batches.
        grad_sync_fn: optional callable passed through to ``_try_batch`` for
            distributed gradient synchronization.
        step_callback: optional callable invoked after each successful
            optimizer step as ``step_callback(n_batches, batch_loss)``.
        max_batches: when set, train only this prefix of the epoch's normal
            batch order. This is useful for reproducible short runs.

    Returns:
        (avg_loss, micro_bs): average loss over the epoch and the (possibly reduced)
        micro batch size.
    """
    import time as _time
    import torch.distributed as _dist

    for model in models:
        if model is not None:
            model.train()
    if batches is None:
        batches = group_examples_by_question(dataset, batch_size, seed + epoch)
    if max_batches is not None:
        batches = batches[:max_batches]
    total_loss = 0.0
    n_batches = 0

    for batch_idxs in batches:
        while True:
            batch_loss, ok = _try_batch(
                models, dataset, collator, digit_token_ids, optimizer, scheduler,
                batch_idxs, micro_bs,
                verbose=(n_batches < 2 and epoch == 0 and grad_sync_fn is not None),
                grad_sync_fn=grad_sync_fn)
            if ok:
                break
            if micro_bs == 1:
                raise RuntimeError("OOM even with micro_bs=1")
            micro_bs = max(micro_bs // 2, 1)
            tag = (f"[rank {_dist.get_rank()}] "
                   if _dist.is_available() and _dist.is_initialized() else "")
            print(f"  {tag}OOM: reducing micro_bs to {micro_bs}", flush=True)

        total_loss += batch_loss
        n_batches += 1
        if step_callback is not None:
            step_callback(n_batches, batch_loss)

        if n_batches % 10 == 0 and (
                not (_dist.is_available() and _dist.is_initialized())
                or _dist.get_rank() == 0):
            lr = optimizer.param_groups[0]['lr']
            print(f"  [batch {n_batches}/{len(batches)} {_time.strftime('%H:%M:%S')}] "
                  f"loss={batch_loss:.4f} lr={lr:.2e}", flush=True)

    return total_loss / max(n_batches, 1), micro_bs


# ---------------------------------------------------------------------------
# Shared helpers extracted from main()
# ---------------------------------------------------------------------------

def _local_blacklist_indices(task, ranges):
    """Map canonical blacklist indices to local positions in ``ranges``."""
    blacklist = task.get_blacklist()
    n_local = sum(end - beg for beg, end in ranges)
    return {local_i for local_i in range(n_local)
            if task.resolve_ex_i(ranges, local_i) in blacklist}


def expected_training_q_idxs(cfg):
    """Canonical training-question order, excluding dataset blacklists."""
    expected = []
    q_i_offset = 0
    for ds_i, ds_name in enumerate(cfg['dataset']):
        ranges = cfg['train_ranges'][ds_i]
        task = get_dataset_config(ds_name, ranges=ranges)
        n_local = sum(end - beg for beg, end in ranges)
        blacklisted_qis = _local_blacklist_indices(task, ranges)
        expected.extend(q_i_offset + qi for qi in range(n_local)
                        if qi not in blacklisted_qis)
        q_i_offset += n_local
    return expected


def load_all_training_data(cfg, revisions, model_name, gen_config_name,
                           n_max_cand=None, cand_top_p=None,
                           data_revisions=None,
                           random_labels_on_last_ckpt=False,
                           random_label_seed=0,
                           log=print):
    """Load training data across all datasets and checkpoints.

    Iterates over datasets x model checkpoints, loading each checkpoint's own
    prediction artifacts. With ``data_revisions``, every model checkpoint
    instead
    receives a complete copy of the data from every listed revision. Each
    example retains the model checkpoint's ``ckpt_i``.

    With ``random_labels_on_last_ckpt``, the examples used to train the final
    model checkpoint receive independent Bernoulli(0.5) correctness labels.
    This uses a private RNG, so it never advances the question-shuffle RNG.

    Returns:
        train_examples: list of example dicts.
    """
    train_examples = []
    n_max_cand = cfg['n_max_cand'] if n_max_cand is None else n_max_cand
    cand_top_p = cfg['cand_top_p'] if cand_top_p is None else cand_top_p
    answer_weight_uniform = cfg.get('answer_weight_uniform', False)
    random_label_rng = np.random.RandomState(random_label_seed)
    q_i_offset = 0
    for ds_i, ds_name in enumerate(cfg['dataset']):
        ds_ranges = cfg['train_ranges'][ds_i]
        task = get_dataset_config(ds_name, ranges=ds_ranges)
        blacklisted_qis = _local_blacklist_indices(task, ds_ranges)
        n_local = sum(end - beg for beg, end in ds_ranges)
        data_by_training_ckpt = []
        for ckpt_i, rev in enumerate(revisions):
            if data_revisions is not None:
                data_by_training_ckpt.append(list(enumerate(data_revisions)))
            else:
                data_by_training_ckpt.append([(0, rev)])

        # Load each dataset/revision artifact once, then copy its examples for
        # every training model checkpoint that consumes it. The latter copies
        # still produce distinct forwards, but avoid repeated shard reads.
        raw_examples_by_revision = {}
        for _, prediction_rev in (
                pair for pairs in data_by_training_ckpt for pair in pairs):
            if prediction_rev in raw_examples_by_revision:
                continue
            log(f"  Loading {ds_name} data revision {prediction_rev}...")
            raw_examples_by_revision[prediction_rev] = load_training_data(
                ds_name, model_name, prediction_rev,
                gen_config_name, ds_ranges,
                n_max_cand=n_max_cand,
                cand_top_p=cand_top_p,
                answer_weight_uniform=answer_weight_uniform,
                blacklisted_qis=blacklisted_qis)

        for ckpt_i, rev in enumerate(revisions):
            for data_i, prediction_rev in data_by_training_ckpt[ckpt_i]:
                # Do not mutate the cached examples: each model/data pair
                # needs independent checkpoint and global-question metadata.
                examples = [dict(ex)
                            for ex in raw_examples_by_revision[prediction_rev]]
                if (random_labels_on_last_ckpt
                        and ckpt_i == len(revisions) - 1):
                    random_correct = random_label_rng.binomial(
                        1, 0.5, size=len(examples)).astype(bool)
                    for ex, correct in zip(examples, random_correct):
                        ex['correct'] = bool(correct)
                    set_training_targets(examples)
                for ex in examples:
                    ex['ckpt_i'] = ckpt_i
                    ex['data_i'] = data_i
                    ex['data_revision'] = prediction_rev
                    ex['ds_i'] = ds_i
                    ex['ex_i'] = ex['q_i']       # within-dataset index (for ICL)
                    ex['q_i'] += q_i_offset       # globally unique
                    ex['task'] = task
                train_examples.extend(examples)
                if data_revisions is not None:
                    log(f"  {ds_name}/model={rev}, data={prediction_rev}: "
                        f"{len(examples)} training instances")
                else:
                    log(f"  {ds_name}/{rev}: {len(examples)} training instances")
        q_i_offset += n_local
    return train_examples


def prepare_training_weights(train_examples, cfg, n_ckpts, n_data_revisions=1,
                             revisions=None,
                             contrast_qs=None, model_name='allenai/Olmo-3-1025-7B',
                             gen_config_name='beam', log=print):
    """Apply the uniform, question-balanced training weights.

    Each checkpoint/question/data-revision has total vanilla-loss weight
    ``1 / (batch_size * n_ckpts * n_data_revisions)``. ``contrast_qs`` is
    retained as a return value for callers that use it for diagnostics.
    """
    if contrast_qs is None:
        contrast_qs, _ = get_contrast_questions(train_examples)
    for ex in train_examples:
        ex['weight'] /= cfg['batch_size'] * n_ckpts * n_data_revisions
    return train_examples, contrast_qs, {'qi_diff': {}}


def make_save_dir(cfg, model_name, revisions, config_stem, seed,
                  create=True, storage='home'):
    """Build and optionally create the save directory.

    Args:
        create: if True (default), create the directory.  Set to False when
            only rank 0 should create it (distributed training).
        storage: ``'home'`` or ``'scratch'`` adapter root.

    Returns:
        save_dir: path string.
    """
    ds_label = '+'.join(cfg['dataset'])
    save_dir = get_adapter_dir(
        ds_label, model_name, '_'.join(revisions),
        f'confidence_{config_stem}', f'seed{seed}', storage=storage)
    if create:
        os.makedirs(save_dir, exist_ok=True)
    return save_dir


def compute_batches_per_epoch(train_examples, cfg, weight_info):
    """Compute the number of ordinary question-grouped batches per epoch."""
    n_questions = len(set(ex['q_i'] for ex in train_examples))
    return math.ceil(n_questions / cfg['batch_size'])


def log_training_config(cfg, config_stem, revisions, train_examples,
                        weight_info, save_dir, seed, log=print,
                        extra_lines=None):
    """Print a summary of the training configuration."""
    n_ckpts = len(revisions)
    n_questions = len(set(ex['q_i'] for ex in train_examples))
    batches_per_epoch = compute_batches_per_epoch(
        train_examples, cfg, weight_info)
    total_steps = batches_per_epoch * cfg['n_epoch']

    log(f"\nTraining config: {config_stem}")
    log(f"  weight_type:  {cfg['weight_type']}")
    log(f"  n_max_cand:   {cfg['n_max_cand']}")
    log(f"  cand_top_p:   {cfg['cand_top_p']}")
    log(f"  answer_weight_uniform: {cfg['answer_weight_uniform']}")
    log(f"  random_labels_on_last_ckpt: "
        f"{cfg['random_labels_on_last_ckpt']}")
    log(f"  dataset:      {', '.join(cfg['dataset'])}")
    log(f"  revisions:    {revisions}")
    if extra_lines:
        for line in extra_lines:
            log(f"  {line}")
    log(f"  n_epoch:      {cfg['n_epoch']}")
    log(f"  batch_size:   {cfg['batch_size']} questions"
        + (f" x {n_ckpts} ckpts" if n_ckpts > 1 else ""))
    log(f"  lr:           {cfg['lr']}")
    log(f"  seed:         {seed}")
    log(f"  total_steps:  {total_steps}")
    log(f"  n_train:      {len(train_examples)} instances ({n_questions} questions)")
    log(f"  save_dir:     {save_dir}")
    log("")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train confidence prediction model on one or more checkpoints")
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML training config')
    parser.add_argument('--model_name', type=str, default='allenai/Olmo-3-1025-7B',
                        help='HuggingFace model name')
    parser.add_argument('--revisions', type=str, nargs='+', required=True,
                        help='Model revisions/checkpoints to train on jointly '
                             '(multiple revisions share one LoRA adapter)')
    parser.add_argument('--gen_config_name', type=str, default='beam',
                        help='Name of gen config used (matches result subdirectory)')
    parser.add_argument('--seed', type=int, nargs='+', default=[17],
                        help='One or more random seeds; multiple seeds share '
                             'loaded immutable data and base models')
    parser.add_argument('--use_ckpt_respective_predictor', action='store_true',
                        help='Train a separate LoRA adapter per checkpoint '
                             '(saved at single-revision paths)')
    parser.add_argument('--is_instruct', action='store_true',
                        help='Format confidence prompts with the tokenizer '
                             'chat template')
    parser.add_argument('--debug', action='store_true',
                        help='Plot weight diagnostics instead of training')
    parser.add_argument('--adapter_root', default=os.environ.get('SCRATCH'),
                        help='Root for LoRA adapters; defaults to $SCRATCH')
    args = parser.parse_args()
    set_adapter_root(args.adapter_root)

    cfg = load_train_config(args.config)
    config_stem = os.path.splitext(os.path.basename(args.config))[0]
    n_ckpts = len(args.revisions)
    data_revisions = cfg['data_revisions']
    n_data_revisions = len(data_revisions) if data_revisions is not None else 1
    # The cuDNN SDPA backend (prioritized on Hopper in newer torch) produces
    # non-finite gradients in the backward pass on GH200, poisoning the weights.
    torch.backends.cuda.enable_cudnn_sdp(False)

    seeds = args.seed
    if len(set(seeds)) != len(seeds):
        raise ValueError('--seed must not contain duplicates')

    def seed_everything(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    # Match the single-seed setup before the initial data preparation.
    seed_everything(seeds[0])

    def log(*print_args):
        print(*print_args, flush=True)

    # Load training data
    log("Loading training data...")
    if data_revisions is not None:
        log(f"  data_revisions: each model ckpt uses all {n_data_revisions} "
            f"prediction-data revisions: {data_revisions}")
    def prepare_training_data(seed):
        """Build the seed-specific data only when randomized labels require it."""
        random_label_seed = (seed + 0x5EED5EED) % (2 ** 32)
        if cfg['random_labels_on_last_ckpt']:
            log(f"  random_labels_on_last_ckpt: replacing labels for final "
                f"training checkpoint {args.revisions[-1]} "
                f"(private RNG seed {random_label_seed})")
        examples = load_all_training_data(
            cfg, args.revisions, args.model_name, args.gen_config_name,
            data_revisions=data_revisions,
            random_labels_on_last_ckpt=cfg['random_labels_on_last_ckpt'],
            random_label_seed=random_label_seed, log=log)
        expected_q_idxs = expected_training_q_idxs(cfg)
        examples, _, info = prepare_training_weights(
            examples, cfg, n_ckpts, n_data_revisions=n_data_revisions,
            revisions=args.revisions,
            model_name=args.model_name, gen_config_name=args.gen_config_name,
            log=log)
        return examples, expected_q_idxs, info

    # Immutable prediction data is shared across seed runs. Randomized labels
    # are intentionally regenerated with each seed, exactly as independent
    # one-seed invocations would do.
    prepared_by_seed = {}
    if cfg['random_labels_on_last_ckpt']:
        for seed in seeds:
            if seed != seeds[0]:
                log(f"Loading seed-specific training data for seed {seed}...")
            seed_everything(seed)
            prepared_by_seed[seed] = prepare_training_data(seed)
    else:
        shared_prepared = prepare_training_data(seeds[0])
        prepared_by_seed = {seed: shared_prepared for seed in seeds}
    if args.use_ckpt_respective_predictor:
        # Train a separate LoRA adapter per checkpoint, each on that
        # checkpoint's slice of the data. Load each immutable base checkpoint
        # once, then attach a freshly initialized LoRA for every seed.
        for ckpt_i, rev in enumerate(args.revisions):
            log(f"\n--- Training adapter for {rev} "
                f"(ckpt {ckpt_i + 1}/{n_ckpts}) ---")
            base_model, tokenizer = load_model_and_tokenizer(args.model_name, rev)
            digit_token_ids = get_single_digit_token_ids(tokenizer)
            collator = LeftPadCollator(tokenizer.pad_token_id)
            for seed in seeds:
                train_examples, expected_q_idxs, weight_info = prepared_by_seed[seed]
                # Filter to this ckpt's examples; rescale weights to compensate
                # for the 1/n_ckpts factor, then route them to model index 0.
                ckpt_examples = [dict(ex) for ex in train_examples
                                 if ex['ckpt_i'] == ckpt_i]
                for ex in ckpt_examples:
                    ex['weight'] *= n_ckpts
                    ex['ckpt_i'] = 0

                ckpt_save_dir = make_save_dir(
                    cfg, args.model_name, [rev], config_stem, seed,
                    storage='scratch')
                seed_everything(seed)
                seed_lora_initialization(seed)
                model = apply_lora(base_model, cfg)
                ckpt_dataset = ConfidenceDataset(
                    ckpt_examples, tokenizer, is_instruct=args.is_instruct,
                    expected_q_idxs=expected_q_idxs)
                batches_per_epoch = compute_batches_per_epoch(
                    ckpt_examples, cfg, weight_info)
                trainable_params = [p for p in model.parameters()
                                    if p.requires_grad]
                optimizer = torch.optim.AdamW(trainable_params, lr=cfg['lr'])
                scheduler = WSDScheduler(optimizer, batches_per_epoch)
                log_training_config(cfg, config_stem, [rev], ckpt_examples,
                                    weight_info, ckpt_save_dir, seed, log=log)

                micro_bs = cfg['batch_size']
                for epoch in range(cfg['n_epoch']):
                    log(f"=== Seed {seed}; epoch {epoch + 1}/{cfg['n_epoch']} ===")
                    train_loss, micro_bs = train_one_epoch(
                        [model], ckpt_dataset, collator, digit_token_ids,
                        optimizer, scheduler, cfg['batch_size'], micro_bs,
                        seed, epoch, cfg=cfg, weight_info=weight_info)
                    log(f"  Train loss: {train_loss:.4f}  micro_bs: {micro_bs}")
                    epoch_dir = os.path.join(ckpt_save_dir, f'epoch{epoch + 1}')
                    os.makedirs(epoch_dir, exist_ok=True)
                    model.save_pretrained(epoch_dir)
                    log(f"  Saved checkpoint to {epoch_dir}")

                # ``unload`` removes the adapter without merging it, restoring
                # the same immutable base model for the next independent seed.
                base_model = model.unload()
                del model, optimizer, scheduler
                torch.cuda.empty_cache()

            del base_model
            torch.cuda.empty_cache()
    else:
        # Shared adapter across all checkpoints. Keep the immutable base
        # models resident and recreate/tie a new adapter for each seed.
        base_models = []
        tokenizer = None
        for rev in args.revisions:
            print(f"Loading model: {args.model_name} / {rev}", flush=True)
            model, tokenizer = load_model_and_tokenizer(
                args.model_name, rev)
            base_models.append(model)
        assert tokenizer is not None
        digit_token_ids = get_single_digit_token_ids(tokenizer)
        collator = LeftPadCollator(tokenizer.pad_token_id)

        for seed in seeds:
            train_examples, expected_q_idxs, weight_info = prepared_by_seed[seed]
            save_dir = make_save_dir(cfg, args.model_name, args.revisions,
                                     config_stem, seed, storage='scratch')
            seed_everything(seed)
            models = []
            for base_model in base_models:
                seed_lora_initialization(seed)
                models.append(apply_lora(base_model, cfg))
            models[0].print_trainable_parameters()
            for model in models[1:]:
                tie_trainable_params(models[0], model)

            train_dataset = ConfidenceDataset(
                train_examples, tokenizer, is_instruct=args.is_instruct,
                expected_q_idxs=expected_q_idxs)
            batches_per_epoch = compute_batches_per_epoch(
                train_examples, cfg, weight_info)
            trainable_params = [p for p in models[0].parameters()
                                if p.requires_grad]
            optimizer = torch.optim.AdamW(trainable_params, lr=cfg['lr'])
            scheduler = WSDScheduler(optimizer, batches_per_epoch)
            log_training_config(cfg, config_stem, args.revisions,
                                train_examples, weight_info, save_dir,
                                seed, log=log)

            micro_bs = cfg['batch_size']
            for epoch in range(cfg['n_epoch']):
                log(f"=== Seed {seed}; epoch {epoch + 1}/{cfg['n_epoch']} ===")
                train_loss, micro_bs = train_one_epoch(
                    models, train_dataset, collator, digit_token_ids,
                    optimizer, scheduler, cfg['batch_size'], micro_bs,
                    seed, epoch, cfg=cfg, weight_info=weight_info)
                log(f"  Train loss: {train_loss:.4f}  micro_bs: {micro_bs}")
                epoch_dir = os.path.join(save_dir, f'epoch{epoch + 1}')
                os.makedirs(epoch_dir, exist_ok=True)
                models[0].save_pretrained(epoch_dir)
                log(f"  Saved checkpoint to {epoch_dir}")

            # Unload every LoRA wrapper without merging, preserving the base
            # checkpoint tensors for the next independent seed.
            base_models = [model.unload() for model in models]
            del models, optimizer, scheduler
            torch.cuda.empty_cache()

        del base_models
        torch.cuda.empty_cache()

    log("\nDone.")


if __name__ == "__main__":
    main()
