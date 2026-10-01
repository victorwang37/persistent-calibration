"""
Distributed version of train_confidence.py: trains a LoRA confidence adapter
across multiple GPUs (nodes) with checkpoint parallelism.

Checkpoints are spread across GPUs as evenly as possible; each checkpoint is
owned by exactly one rank.  The number of ranks may not exceed the number of
checkpoints.  When launched as plain ``python -m`` (single process), behaves
like train_confidence.py, apart from distributed-specific memory handling.

The global batch (cfg['batch_size'] questions x n_ckpts checkpoints) stays
fixed regardless of the number of GPUs.  A single global SUM all-reduce of the
small LoRA gradients correctly reconstructs the single-process gradient --
training math (LR, step count, WSD schedule, weight normalization) is unchanged.

Usage:
    # Single GPU (same as train_confidence.py):
    python -m conset.train_confidence_dist \\
        --config conset/train_configs/lora_triviaqa_acc_lr2e-4.yaml \\
        --revisions stage1-step141000

    # Multi-node via srun (ckpts spread across nodes):
    srun -N 3 -n 3 python -m conset.train_confidence_dist \\
        --config conset/train_configs/lora_triviaqa_acc_lr2e-4.yaml \\
        --revisions stage1-step141000 stage1-step283000 stage1-step424000

"""

import argparse
import math
import os
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

from .train_confidence import (
    ConfidenceDataset,
    LeftPadCollator,
    WSDScheduler,
    get_adapter_dir,
    apply_lora,
    expected_training_q_idxs,
    forward_confidence,
    get_single_digit_token_ids,
    group_examples_by_question,
    load_all_training_data,
    load_train_config,
    prepare_training_weights,
    seed_lora_initialization,
    set_adapter_root,
    tie_trainable_params,
)


# ---------------------------------------------------------------------------
# Distributed setup shared by the LoRA training entry points.
# ---------------------------------------------------------------------------

def setup_distributed():
    """Initialize torch.distributed if launched as a multi-process job.

    Supports torchrun (``RANK`` in env) and srun with MASTER_ADDR/MASTER_PORT
    exported (rank from ``SLURM_PROCID``).  Keying the SLURM path on
    ``MASTER_ADDR`` avoids false positives: plain ``python`` under sbatch
    still sees SLURM_NTASKS > 1.

    Returns:
        (rank, world_size); (0, 1) without initializing when single-process.
    """
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
    elif ('MASTER_ADDR' in os.environ
          and int(os.environ.get(
              'SLURM_STEP_NUM_TASKS', os.environ.get('SLURM_NTASKS', 1))) > 1):
        rank = int(os.environ['SLURM_PROCID'])
        # A Slurm array job can reserve more tasks than an individual srun
        # step uses (e.g. a three-checkpoint run in a four-node allocation).
        # Prefer the step-local count so every launched rank agrees on the
        # actual process-group size.
        world_size = int(os.environ.get(
            'SLURM_STEP_NUM_TASKS', os.environ['SLURM_NTASKS']))
    else:
        return 0, 1

    torch.cuda.set_device(0)
    dist.init_process_group(
        'nccl', rank=rank, world_size=world_size, timeout=timedelta(hours=2),
        device_id=torch.device('cuda', 0))
    return rank, world_size


# ---------------------------------------------------------------------------
# Model loading (local copy with device_map parameter; the original in
# train_confidence hardcodes device_map='auto')
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_name, revision, device_map=None):
    """Load a causal LM (bfloat16) and its tokenizer."""
    if device_map is None:
        device_map = 'auto'
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, revision=revision, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name, revision=revision, device_map=device_map,
        dtype=torch.bfloat16)

    return model, tokenizer


# ---------------------------------------------------------------------------
# Rank assignment
# ---------------------------------------------------------------------------

def compute_rank_assignment(n_ckpts, world_size, rank):
    """Return the global checkpoint indices owned by this rank."""
    if world_size > n_ckpts:
        raise ValueError(
            f'checkpoint parallelism requires at least one checkpoint per '
            f'rank, but world_size={world_size} exceeds n_ckpts={n_ckpts}')
    chunks = np.array_split(range(n_ckpts), world_size)
    return list(chunks[rank])


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _try_batch(local_models, my_ckpts, dataset, collator, digit_token_ids,
               optimizer, scheduler, trainable_params, batch_idxs,
               micro_bs, device, world_size, rank, debug=False):
    """Process one batch with collective OOM safety and gradient all-reduce.

    ``local_models`` are this rank's checkpoint models (indexed in parallel
    with ``my_ckpts``).  ``batch_idxs`` are this rank's checkpoint-owned
    example indices for this batch (and may be empty when a rank owns no
    examples in a batch).

    Returns ``(batch_loss, ok, grad_sync_seconds)``.  The timing covers only
    the coalesced LoRA-gradient all-reduce.
    """
    optimizer.zero_grad()
    batch_loss = 0.0
    local_ok = True

    if debug:
        import time as _time
        print(f"  [rank {rank} {_time.strftime('%H:%M:%S')}] "
              f"batch start ({len(batch_idxs)} examples)", flush=True)

    try:
        if batch_idxs:
            features = [dataset[i] for i in batch_idxs]
            batch = collator(features)

            ids = batch['input_ids']
            mask = batch['attention_mask']
            targets = batch['target'].to(device)
            weights = batch['weight'].to(device)
            ckpt_idxs = batch['ckpt_i']

            for global_ck, model in zip(my_ckpts, local_models):
                sel = (ckpt_idxs == global_ck).nonzero(as_tuple=True)[0]
                for mib_beg in range(0, len(sel), micro_bs):
                    mib = sel[mib_beg:mib_beg + micro_bs]
                    mib_ids = ids[mib].to(device)
                    mib_mask = mask[mib].to(device)
                    mib_targets = targets[mib]
                    mib_weights = weights[mib]

                    logit, _ = forward_confidence(
                        model, mib_ids, mib_mask, digit_token_ids)
                    per_sample_loss = (
                        nn.functional.binary_cross_entropy_with_logits(
                            logit, mib_targets, reduction='none'))
                    loss = (per_sample_loss * mib_weights).sum()

                    loss.backward()
                    batch_loss += loss.item()

        if debug:
            import time as _time
            print(f"  [rank {rank} {_time.strftime('%H:%M:%S')}] "
                  "forward/backward done", flush=True)

    except (RuntimeError, torch.cuda.OutOfMemoryError):
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # settle CUDA before NCCL collective
        local_ok = False

    # --- Collective OOM agreement: if ANY rank failed, all retry. ---
    if world_size > 1:
        ok_t = torch.tensor(1 if local_ok else 0, device=device)
        dist.all_reduce(ok_t, op=dist.ReduceOp.MIN)
        if ok_t.item() == 0:
            if local_ok:
                optimizer.zero_grad()
            return 0.0, False, 0.0
    elif not local_ok:
        return 0.0, False, 0.0

    # --- Gradient all-reduce (SUM reconstructs single-process gradient) ---
    # PEFT's all-linear OLMo 3 32B adapter has 896 A/B tensors. Issuing one
    # collective per tensor makes inter-node launch latency dominate a batch.
    # ``all_reduce_coalesced`` was fast on three nodes but hung on four, so
    # communicate one explicit flat buffer through the standard all-reduce.
    # This transient FP32 buffer is 256 MiB for the 32B r=8 adapter.
    grad_sync_seconds = 0.0
    if world_size > 1:
        grads = []
        for p in trainable_params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            grads.append(p.grad)
        import time as _time
        sync_start = _time.perf_counter()
        flat_grads = torch.cat([grad.reshape(-1) for grad in grads])
        if debug:
            print(f"  [rank {rank} {_time.strftime('%H:%M:%S')}] "
                  "entering gradient sync", flush=True)
        dist.all_reduce(flat_grads)
        offset = 0
        for grad in grads:
            next_offset = offset + grad.numel()
            grad.copy_(flat_grads[offset:next_offset].view_as(grad))
            offset = next_offset
        del flat_grads
        grad_sync_seconds = _time.perf_counter() - sync_start
        if debug:
            print(f"  [rank {rank} {_time.strftime('%H:%M:%S')}] "
                  f"gradient sync done ({grad_sync_seconds:.2f}s)", flush=True)

    nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
    optimizer.step()
    scheduler.step()

    # All-reduce loss for consistent logging across ranks
    if world_size > 1:
        loss_t = torch.tensor(batch_loss, device=device)
        dist.all_reduce(loss_t)
        batch_loss = loss_t.item()

    return batch_loss, True, grad_sync_seconds


def train_one_epoch(local_models, my_ckpts, my_ckpts_set,
                    dataset, collator, digit_token_ids, optimizer, scheduler,
                    trainable_params, batch_size, micro_bs, device, seed, epoch,
                    world_size, rank):
    """Train one epoch with checkpoint parallelism.

    Every rank builds the same global batches (deterministic seed).  Each rank
    filters to its owned checkpoints, then forwards only those examples.
    Gradients are globally SUM-all-reduced so that the update is identical to
    a single-process run.

    Returns:
        (avg_loss, micro_bs)
    """
    import time as _time

    for model in local_models:
        model.train()

    # Identical batches on all ranks (same seed)
    batches = group_examples_by_question(dataset, batch_size, seed + epoch)
    total_loss = 0.0
    n_batches = 0
    window_batch_start = _time.perf_counter()
    window_grad_sync_seconds = 0.0

    for global_batch_idxs in batches:
        # Filter to this rank's checkpoint-owned work.
        my_idxs = [idx for idx in global_batch_idxs
                    if dataset.examples[idx]['ckpt_i'] in my_ckpts_set]

        while True:
            batch_loss, ok, grad_sync_seconds = _try_batch(
                local_models, my_ckpts, dataset, collator, digit_token_ids,
                optimizer, scheduler, trainable_params,
                my_idxs, micro_bs, device, world_size, rank,
                debug=(epoch == 0 and n_batches < 2))
            if ok:
                break
            if micro_bs == 1:
                raise RuntimeError("OOM even with micro_bs=1")
            micro_bs = max(micro_bs // 2, 1)
            if rank == 0:
                print(f"  OOM: reducing micro_bs to {micro_bs}", flush=True)

        total_loss += batch_loss
        n_batches += 1
        window_grad_sync_seconds += grad_sync_seconds

        if n_batches % 10 == 0 and rank == 0:
            # CUDA work is asynchronous; synchronize once per reporting window
            # so ``batch_s`` measures completed work rather than queued kernels.
            torch.cuda.synchronize(device)
            window_batch_seconds = _time.perf_counter() - window_batch_start
            lr = optimizer.param_groups[0]['lr']
            print(f"  [batch {n_batches}/{len(batches)} {_time.strftime('%H:%M:%S')}] "
                  f"loss={batch_loss:.4f} lr={lr:.2e} "
                  f"batch_s={window_batch_seconds / 10:.2f} "
                  f"grad_sync_s={window_grad_sync_seconds / 10:.2f}",
                  flush=True)
            window_batch_start = _time.perf_counter()
            window_grad_sync_seconds = 0.0

    return total_loss / max(n_batches, 1), micro_bs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Distributed training of LoRA confidence adapter "
                    "(checkpoint parallelism)")
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML training config')
    parser.add_argument('--model_name', type=str,
                        default='allenai/Olmo-3-1025-7B',
                        help='HuggingFace model name')
    parser.add_argument('--revisions', type=str, nargs='+', required=True,
                        help='Model revisions/checkpoints to train on jointly')
    parser.add_argument('--gen_config_name', type=str, default='beam',
                        help='Name of gen config used (matches result subdir)')
    parser.add_argument('--seed', type=int, default=17,
                        help='Random seed')
    parser.add_argument('--is_instruct', action='store_true',
                        help='Format confidence prompts with the tokenizer '
                             'chat template')
    parser.add_argument('--adapter_root', default=os.environ.get('SCRATCH'),
                        help='Root for LoRA adapters; defaults to $SCRATCH')
    args = parser.parse_args()
    set_adapter_root(args.adapter_root)

    # --- Distributed setup ---
    rank, world_size = setup_distributed()

    cfg = load_train_config(args.config)
    config_stem = os.path.splitext(os.path.basename(args.config))[0]
    n_ckpts = len(args.revisions)
    if cfg.get('condition_on_conset_loss', False):
        raise ValueError(
            'condition_on_conset_loss is not yet supported by '
            'train_confidence_dist: it needs differentiable cross-rank '
            'checkpoint-pair communication')

    # The cuDNN SDPA backend produces non-finite gradients on GH200.
    torch.backends.cuda.enable_cudnn_sdp(False)

    # Identical seeds on all ranks (deterministic batching)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # --- Rank assignment ---
    my_ckpts = compute_rank_assignment(n_ckpts, world_size, rank)
    my_ckpts_set = set(my_ckpts)

    if rank == 0:
        print(f"world_size={world_size}, n_ckpts={n_ckpts}")
        for r in range(world_size):
            ck = compute_rank_assignment(n_ckpts, world_size, r)
            revs = [args.revisions[c] for c in ck]
            print(f"  rank {r}: ckpts={revs}")

    # --- Load training data (all ranks load everything for identical batching) ---
    # Do this failure check collectively.  In particular, an ENOLCK from the
    # HF cache must not let the other ranks spend minutes loading 32B models
    # before they discover that one rank has already exited.
    data_error = None
    try:
        if rank == 0:
            print("Loading training data...", flush=True)
        train_examples = load_all_training_data(
            cfg, args.revisions, args.model_name, args.gen_config_name,
            log=print if rank == 0 else lambda *_args, **_kwargs: None)
        expected_q_idxs = expected_training_q_idxs(cfg)
        train_examples, _, _ = prepare_training_weights(
            train_examples, cfg, n_ckpts, revisions=args.revisions,
            model_name=args.model_name, gen_config_name=args.gen_config_name,
            log=print if rank == 0 else lambda *_args, **_kwargs: None)
    except Exception as exc:
        data_error = exc
        print(f"[rank {rank}] Training-data preparation failed: {exc!r}",
              flush=True)

    if world_size > 1:
        data_ready = torch.tensor(
            [int(data_error is None)], device=torch.device('cuda', 0))
        # MIN makes every rank observe a failure from any one rank.
        dist.all_reduce(data_ready, op=dist.ReduceOp.MIN)
        if not data_ready.item():
            if rank == 0:
                print("Aborting before model loading because training-data "
                      "preparation failed on at least one rank.", flush=True)
            if data_error is not None:
                raise RuntimeError(
                    f"Training-data preparation failed on rank {rank}") from data_error
            raise RuntimeError(
                "Training-data preparation failed on another distributed rank")
    elif data_error is not None:
        raise data_error

    # --- Load only this rank's checkpoint models ---
    device_map = {'': 0} if world_size > 1 else 'auto'
    local_models = []
    tokenizer = None
    for ckpt_i in my_ckpts:
        rev = args.revisions[ckpt_i]
        if rank == 0:
            print(f"Loading model: {args.model_name} / {rev}", flush=True)
        model, tokenizer = load_model_and_tokenizer(
            args.model_name, rev, device_map=device_map)
        # Match train_confidence.py: LoRA initialization must not depend on
        # preceding model loads or on which checkpoints are in this run.
        seed_lora_initialization(args.seed)
        model = apply_lora(model, cfg)
        model.gradient_checkpointing_enable()
        local_models.append(model)
    assert tokenizer is not None

    if rank == 0:
        local_models[0].print_trainable_parameters()

    # Tie LoRA params across this rank's local models (if >1)
    for m in local_models[1:]:
        tie_trainable_params(local_models[0], m)

    trainable_params = [p for p in local_models[0].parameters()
                        if p.requires_grad]

    # Broadcast rank-0's LoRA weights so all ranks start identical.  A flat
    # buffer turns hundreds of small NCCL broadcasts into one collective.
    # LoRA parameters all have the same dtype/device, so this is also the
    # same transient-size pattern already used for flat gradient all-reduce.
    if world_size > 1:
        first_param = trainable_params[0]
        assert all(p.data.dtype == first_param.data.dtype and
                   p.data.device == first_param.data.device
                   for p in trainable_params)
        flat_params = torch.cat([p.data.reshape(-1) for p in trainable_params])
        if rank == 0:
            print(f"Broadcasting {len(trainable_params)} LoRA tensors as one "
                  f"{flat_params.numel():,}-element buffer...", flush=True)
        dist.broadcast(flat_params, src=0)
        offset = 0
        for p in trainable_params:
            next_offset = offset + p.numel()
            p.data.copy_(flat_params[offset:next_offset].view_as(p.data))
            offset = next_offset
        del flat_params
        if rank == 0:
            print("LoRA initialization broadcast complete.", flush=True)

    digit_token_ids = get_single_digit_token_ids(tokenizer)

    # Save directory
    save_dir = get_adapter_dir(
        '+'.join(cfg['dataset']), args.model_name, '_'.join(args.revisions),
        f'confidence_{config_stem}', f'seed{args.seed}', storage='scratch')
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)

    # Each example carries its dataset-specific task prompt.
    train_dataset = ConfidenceDataset(
        train_examples, tokenizer, is_instruct=args.is_instruct,
        expected_q_idxs=expected_q_idxs)
    collator = LeftPadCollator(tokenizer.pad_token_id)
    device = next(local_models[0].parameters()).device

    # Initial micro batch size — will auto-reduce on OOM during training
    micro_bs = cfg['batch_size']

    # Compute total training steps for scheduler (unchanged from single-process)
    n_questions = len(set(ex['q_i'] for ex in train_examples))
    batches_per_epoch = math.ceil(n_questions / cfg['batch_size'])
    total_steps = batches_per_epoch * cfg['n_epoch']

    # Optimizer and scheduler over the shared trainable params
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg['lr'])
    scheduler = WSDScheduler(optimizer, batches_per_epoch)

    if rank == 0:
        print(f"\nTraining config: {config_stem}")
        print(f"  weight_type:  {cfg['weight_type']}")
        print(f"  dataset:      {cfg['dataset']}")
        print(f"  revisions:    {args.revisions}")
        print(f"  n_epoch:      {cfg['n_epoch']}")
        print(f"  batch_size:   {cfg['batch_size']} questions x {n_ckpts} ckpts")
        print(f"  lr:           {cfg['lr']}")
        print(f"  seed:         {args.seed}")
        print(f"  total_steps:  {total_steps}")
        print(f"  n_train:      {len(train_examples)} instances "
              f"({n_questions} questions)")
        print(f"  world_size:   {world_size}")
        print(f"  save_dir:     {save_dir}")
        print(flush=True)

    # --- Training loop ---
    for epoch in range(cfg['n_epoch']):
        if rank == 0:
            print(f"=== Epoch {epoch + 1}/{cfg['n_epoch']} ===", flush=True)

        train_loss, micro_bs = train_one_epoch(
            local_models, my_ckpts, my_ckpts_set,
            train_dataset, collator, digit_token_ids,
            optimizer, scheduler, trainable_params,
            cfg['batch_size'], micro_bs, device, args.seed, epoch,
            world_size, rank)

        if rank == 0:
            print(f"  Train loss: {train_loss:.4f}  micro_bs: {micro_bs}",
                  flush=True)

        # Save the adapter (rank 0 only)
        if rank == 0:
            epoch_dir = os.path.join(save_dir, f'epoch{epoch + 1}')
            os.makedirs(epoch_dir, exist_ok=True)
            local_models[0].save_pretrained(epoch_dir)
            print(f"  Saved checkpoint to {epoch_dir}", flush=True)

        if world_size > 1:
            dist.barrier()

    if rank == 0:
        print("\nDone.", flush=True)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
