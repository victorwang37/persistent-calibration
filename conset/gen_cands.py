"""
Standalone candidate generation (beam search or sampling) with YAML config.

Usage:
    python -m conset.gen_cands --config conset/gen_configs/beam.yaml --dataset triviaqa --model_name allenai/Olmo-3-1025-7B --revision stage1-step141000
"""

import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import pickle as pkl
import os
import argparse
import shutil
import yaml

from .dataset_configs import get_dataset_config


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_gen_config(path):
    """Load and validate a YAML generation config.

    Required keys: cand_type (beam|sample), n_cand.
    Optional keys depend on cand_type:
      beam:   length_penalty (default 0.0)
      sample: temperature (default 1.0), top_p
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)

    if 'cand_type' not in cfg:
        raise ValueError(f"gen config {path} missing required key 'cand_type'")
    if cfg['cand_type'] not in ('beam', 'sample'):
        raise ValueError(f"cand_type must be 'beam' or 'sample', got {cfg['cand_type']!r}")
    if 'n_cand' not in cfg:
        raise ValueError(f"gen config {path} missing required key 'n_cand'")

    return cfg


# ---------------------------------------------------------------------------
# Prompt encoding
# ---------------------------------------------------------------------------

def encode_prompt(prompt_text, tokenizer, is_instruct, model_device):
    """Encode a prompt, handling chat template for instruct models."""
    if is_instruct:
        msg = [{'role': 'user', 'content': prompt_text}]
        encoded = tokenizer.apply_chat_template(
            [msg],
            enable_thinking=False,
            add_generation_prompt=True,
            padding=True,
            return_tensors="pt",
        )
    else:
        encoded = tokenizer(prompt_text, return_tensors="pt")

    if isinstance(encoded, torch.Tensor):
        input_ids = encoded.to(model_device)
        attention_mask = torch.ones_like(input_ids)
    else:
        input_ids = encoded['input_ids'].to(model_device)
        attention_mask = encoded['attention_mask'].to(model_device)

    return input_ids, attention_mask


# ---------------------------------------------------------------------------
# Generation functions
# ---------------------------------------------------------------------------

def decode_generated_tokens(tokenizer, sequences, decode_offset):
    """Decode generated suffixes, omitting unused trailing beam slots.

    Transformers beam-search finalization can leave unused trailing slots as
    negative values after a stop string ends a beam.  They are not generated
    text and cannot be passed to a fast tokenizer.  Other invalid-ID patterns
    remain errors rather than being silently altered.
    """
    generated_ids = sequences[:, decode_offset:]
    invalid = (generated_ids < 0) | (generated_ids >= len(tokenizer))
    if not invalid.any():
        return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    trimmed_rows = []
    for row in generated_ids:
        invalid_i = ((row < 0) | (row >= len(tokenizer))).nonzero(as_tuple=False)
        if len(invalid_i) == 0:
            trimmed_rows.append(row)
            continue
        first_invalid = invalid_i[0].item()
        # Only an all-negative suffix is the known unused-slot case.  In
        # particular, do not hide a positive out-of-vocabulary ID.
        if not (row[first_invalid:] < 0).all():
            raise ValueError("model.generate returned token IDs outside the tokenizer vocabulary")
        trimmed_rows.append(row[:first_invalid])

    return tokenizer.batch_decode(trimmed_rows, skip_special_tokens=True)


def sample(prompt_text, model, tokenizer, is_instruct, task, gen_config):
    """Generate n_cand sampled answers for one question. Returns None if prompt is None."""
    if prompt_text is None:
        return None
    n_sample = gen_config['n_cand']
    input_ids, attention_mask = encode_prompt(prompt_text, tokenizer, is_instruct, model.device)

    gen_kwargs = dict(
        do_sample=True,
        temperature=gen_config.get('temperature', 1.0),
        max_new_tokens=task.max_new_tokens,
        return_dict_in_generate=True,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
    )

    if 'top_p' in gen_config:
        gen_kwargs['top_p'] = gen_config['top_p']

    prompt_len = input_ids.shape[1]
    gen_input_ids = input_ids.repeat(n_sample, 1)
    gen_attn_mask = attention_mask.repeat(n_sample, 1)

    stopping = task.get_stopping_criteria(tokenizer, prompt_len, n_sample, is_instruct=is_instruct)
    if stopping is not None:
        gen_kwargs['stopping_criteria'] = stopping

    transformers.set_seed(17)
    outputs = model.generate(
        gen_input_ids,
        attention_mask=gen_attn_mask,
        **gen_kwargs,
    )
    texts = decode_generated_tokens(tokenizer, outputs.sequences, prompt_len)

    del outputs
    torch.cuda.empty_cache()

    return texts


def beam_search(prompt_text, model, tokenizer, is_instruct, task, gen_config):
    """Run beam search for one prompt. Returns None strs and None lls if prompt is None.

    Assumes short-form QA (no CoT) — stops at first newline via stop_strings.
    """
    if prompt_text is None:
        return None, None

    num_beams = gen_config['n_cand']
    length_penalty = gen_config.get('length_penalty', 0.0)

    input_ids, attention_mask = encode_prompt(prompt_text, tokenizer, is_instruct, model.device)

    gen_kwargs = dict(
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        num_beams=num_beams,
        num_return_sequences=num_beams,
        max_new_tokens=task.max_new_tokens,
        length_penalty=length_penalty,
        output_scores=True,
        return_dict_in_generate=True,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        stop_strings=["\n"],
        tokenizer=tokenizer,
    )

    decode_offset = input_ids.shape[1]
    gen_input_ids = input_ids
    gen_attn_mask = attention_mask

    outputs = model.generate(
        gen_input_ids,
        attention_mask=gen_attn_mask,
        **gen_kwargs,
    )

    strs = decode_generated_tokens(tokenizer, outputs.sequences, decode_offset)
    lls = outputs.sequences_scores.cpu()

    return strs, lls


# ---------------------------------------------------------------------------
# Answer grouping
# ---------------------------------------------------------------------------

NLI_MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
ENTAIL_I = 0

def run_nli(task, examples, strs):
    """Compute pairwise NLI between candidate answers.

    Pairs that already match via task.normalize_answer() are assigned
    entailment=1 without running the NLI model, so group_answers can
    use the NLI tensor uniformly.

    Args:
        task: DatasetConfig instance.
        examples: list[QA|None] — QA examples (None for exhausted slots).
        strs: list[list[str]|None] — candidate answer strings per question.

    Returns:
        nlis: (n_qst, max_n, max_n, 3) tensor of NLI scores
              (entailment, neutral, contradiction). Unused entries are -1.
    """
    from transformers import AutoModelForSequenceClassification

    nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
    nli_model = AutoModelForSequenceClassification.from_pretrained(
        NLI_MODEL_NAME, device_map='auto')

    max_n = max(len(s) for s in strs if s is not None)
    nlis = -torch.ones(len(examples), max_n, max_n, 3, dtype=nli_model.dtype)

    for ex_i, ex in enumerate(examples):
        if ex is None or strs[ex_i] is None:
            continue
        n = len(strs[ex_i])
        if n <= 1:
            continue

        # Pre-compute normalized forms for lexical matching
        norms = [task.normalize_answer(s) for s in strs[ex_i]]

        premises, hypotheses = [], []
        nlis_mask = torch.zeros(max_n, max_n, dtype=torch.bool)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if norms[i] == norms[j]:
                    # Lexical match — mark as entailment directly [1, 0, 0]
                    nlis[ex_i, i, j] = torch.nn.functional.one_hot(
                        torch.tensor(ENTAIL_I), 3).to(nlis.dtype)
                else:
                    nlis_mask[i, j] = 1
                    premises.append(
                        f"Question: {ex.question}\nAnswer: {strs[ex_i][i]}")
                    hypotheses.append(f"Answer: {strs[ex_i][j]}")

        if premises:
            inputs = nli_tokenizer(
                premises, hypotheses, return_tensors="pt", padding=True
            ).to(nli_model.device)
            with torch.no_grad():
                nlis[ex_i][torch.nonzero(nlis_mask, as_tuple=True)] = (
                    torch.softmax(nli_model(**inputs).logits, dim=-1).cpu()
                )

    return nlis


def group_answers(task, cand_strs, weights, nlis=None):
    """Group equivalent candidate answers.

    Args:
        task: DatasetConfig instance.
        cand_strs: list[list[str]|None] — candidate strings per question.
        weights: (n_qst, max_n) — per-candidate weights.
        nlis: (n_qst, max_n, max_n, 3) | None — precomputed NLI scores.

    Returns:
        group_strs: list[list[list[str]]|None] — per-question groups
            sorted by descending probability. None for skipped entries.
        group_probs: same shape as weights — group probabilities,
            padded with 0.
    """
    all_groups = []   # list of (list[list[str]] | None)
    group_probs = torch.zeros_like(weights)  # copy shape despite different effective shape

    for ex_i in range(len(cand_strs)):
        if cand_strs[ex_i] is None:
            all_groups.append(None)
            continue

        # Do not put None (ill-formed) answers in any groups
        answers = []
        valid_idxs = []
        for ci, raw in enumerate(cand_strs[ex_i]):
            ans = task.extract_answer(raw)
            if ans:
                answers.append(ans)
                valid_idxs.append(ci)
        valid_idxs = torch.tensor(valid_idxs)

        if not answers:
            all_groups.append([])
            continue

        # Build groups via union-find
        n = len(answers)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i in range(n):
            for j in range(i + 1, n):
                if nlis is None:
                    if task.answers_match(answers[i], answers[j]):
                        union(i, j)
                else:
                    ci, cj = valid_idxs[i], valid_idxs[j]
                    if (nlis[ex_i, ci, cj, ENTAIL_I] >= 0
                            and nlis[ex_i, ci, cj].argmax() == ENTAIL_I
                            and nlis[ex_i, cj, ci].argmax() == ENTAIL_I):
                        union(i, j)

        # Collect groups
        groups = {}
        for i in range(n):
            root = find(i)
            groups.setdefault(root, []).append(i)

        # Build group strs and compute group weights
        groups_with_prob = []
        for members in groups.values():
            group_strs_i = [answers[m] for m in members]
            group_weight = weights[ex_i, valid_idxs[members]].sum().item()
            groups_with_prob.append((group_strs_i, group_weight))

        # Sort by descending probability
        groups_with_prob.sort(key=lambda x: x[1], reverse=True)

        all_groups.append([c[0] for c in groups_with_prob])
        probs = [c[1] for c in groups_with_prob]
        group_probs[ex_i, :len(probs)] = torch.tensor(probs)

    return all_groups, group_probs


def _run_grouping(result_dir, task, examples, gen_config, cand_type, force):
    """Run answer grouping after candidate generation.

    Loads candidates, computes weights, optionally runs NLI, groups answers,
    and saves group files.
    """
    group_strs_save = os.path.join(result_dir, f'{cand_type}_group_strs.pkl')
    group_probs_save = os.path.join(result_dir, f'{cand_type}_group_probs.pt')
    nlis_save = os.path.join(result_dir, f'{cand_type}_nlis.pt')

    if os.path.exists(group_probs_save) and not force:
        print("Grouping results already exist, skipping.", flush=True)
        return

    # Load candidates
    with open(os.path.join(result_dir, f'{cand_type}_strs.pkl'), 'rb') as f:
        cand_strs = pkl.load(f)

    # Load weights
    if cand_type == 'beam':
        beam_lls = torch.load(
            os.path.join(result_dir, 'beam_lls.pt'),
            weights_only=True)

        weights = torch.softmax(beam_lls, dim=-1)
        weights = weights.nan_to_num(0.0)
    else:
        # Uniform weights for sampled candidates
        n_qst = len(cand_strs)
        n_cand = gen_config['n_cand']
        weights = torch.zeros(n_qst, n_cand)
        for ex_i in range(n_qst):
            if cand_strs[ex_i] is not None:
                n = len(cand_strs[ex_i])
                weights[ex_i, :n] = 1.0 / n

    # Compute or load NLI for datasets that need it
    nlis = None
    if task.needs_llm_judge:
        if os.path.exists(nlis_save) and not force:
            print("Loading existing NLI scores...", flush=True)
            nlis = torch.load(nlis_save, weights_only=True)
        else:
            print("Computing pairwise NLI scores...", flush=True)
            nlis = run_nli(task, examples, cand_strs)
            torch.save(nlis, nlis_save)
            print("Saved NLI scores", flush=True)

    # Group answers
    print("Grouping answers...", flush=True)
    group_strs, group_probs = group_answers(
        task, cand_strs, weights, nlis=nlis)

    with open(group_strs_save, 'wb') as f:
        pkl.dump(group_strs, f)
    torch.save(group_probs, group_probs_save)
    print("Saved grouping results", flush=True)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run(args, gen_config):
    # Build dataset config
    ds_kwargs = dict(ranges=[(args.beg_i, args.end_i)])
    task = get_dataset_config(args.dataset, **ds_kwargs)

    # Load model
    print(f"Loading model / revision: {args.model_name} / {args.revision}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, revision=args.revision, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, revision=args.revision, device_map='auto', dtype=torch.bfloat16
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # The cuDNN SDPA backend can fail on certain sequence lengths on GH200.
    torch.backends.cuda.enable_cudnn_sdp(False)

    is_instruct = args.is_instruct
    if is_instruct and tokenizer.chat_template is None:
        raise ValueError('--is_instruct requires a tokenizer with a chat template')

    cand_type = gen_config['cand_type']
    config_stem = os.path.splitext(os.path.basename(args.config))[0]

    half_model_name = args.model_name.split('/')[-1]
    base_result_dir = f"results/conset/{args.dataset}/{half_model_name}/{args.revision}/{args.beg_i}-{args.end_i}"
    result_dir = os.path.join(base_result_dir, config_stem)
    os.makedirs(result_dir, exist_ok=True)

    # Copy gen config into result dir for traceability
    shutil.copy2(args.config, os.path.join(result_dir, 'gen_config.yaml'))

    # Load examples
    examples = task.load_examples()
    n_qst = len(examples)
    print(f"Loaded {n_qst} examples for {task.name}", flush=True)

    # Build prompts (None entries from exhausted BBH subtasks stay as None)
    prompts = [
        task.format_generation_prompt(ex.question if ex else None, is_instruct, ex_i=ex_i)
        for ex_i, ex in enumerate(examples)
    ]

    # Dispatch to beam or sample
    debug = getattr(args, 'debug', False)
    if cand_type == 'beam':
        _run_beam(result_dir, prompts, model, tokenizer,
                  is_instruct, task, gen_config, n_qst, args.force,
                  debug=debug)
    else:
        _run_sample(result_dir, prompts, model, tokenizer,
                    is_instruct, task, gen_config, n_qst, args.force)

    # Free the generation model before grouping (NLI may need GPU memory)
    del model
    torch.cuda.empty_cache()

    _run_grouping(result_dir, task, examples, gen_config, cand_type, args.force)

    print("\nDone.")


def _run_beam(result_dir, prompts, model, tokenizer,
              is_instruct, task, gen_config, n_qst, force,
              debug=False):
    """Run beam search with resume support."""
    strs_save = os.path.join(result_dir, 'beam_strs.pkl')
    lls_save = os.path.join(result_dir, 'beam_lls.pt')
    num_beams = gen_config['n_cand']

    # Resume check
    if os.path.exists(strs_save) and not force:
        with open(strs_save, 'rb') as f:
            beam_strs = pkl.load(f)
        resume_i = len(beam_strs)
        if resume_i >= n_qst:
            print(f"Beam results already complete ({resume_i} examples), skipping.", flush=True)
            return
        beam_lls = torch.load(lls_save)
        print(f"Resuming beam search from {resume_i}/{n_qst}...", flush=True)
    else:
        resume_i = 0
        beam_strs = []
        beam_lls = torch.full((n_qst, num_beams), -float('inf'))
        print("Running beam search...", flush=True)

    for ex_i in range(resume_i, n_qst):
        if debug:
            t0 = time.time()
        strs, lls = beam_search(
            prompts[ex_i], model, tokenizer, is_instruct, task, gen_config)
        if debug:
            print(f"  [beam ex_i={ex_i}] {time.time() - t0:.2f}s", flush=True)
        beam_strs.append(strs)
        if lls is not None:
            beam_lls[ex_i] = lls

        if (ex_i + 1) % 100 == 0 or ex_i + 1 == n_qst:
            with open(strs_save, 'wb') as f:
                pkl.dump(beam_strs, f)
            torch.save(beam_lls, lls_save)
            print(f"  [{ex_i + 1}/{n_qst}]", flush=True)

    print('Saved beam results', flush=True)


def _run_sample(result_dir, prompts, model, tokenizer,
                is_instruct, task, gen_config, n_qst, force):
    """Run sampling with resume support."""
    strs_save = os.path.join(result_dir, 'sample_strs.pkl')

    # Resume check
    if os.path.exists(strs_save) and not force:
        with open(strs_save, 'rb') as f:
            sampled_strs = pkl.load(f)
        resume_i = len(sampled_strs)
        if resume_i >= n_qst:
            print(f"Sample results already complete ({resume_i} examples), skipping.", flush=True)
            return
        print(f"Resuming sampling from {resume_i}/{n_qst}...", flush=True)
    else:
        resume_i = 0
        sampled_strs = []
        print("Sampling generations...", flush=True)

    for ex_i in range(resume_i, n_qst):
        sampled_strs.append(
            sample(prompts[ex_i], model, tokenizer, is_instruct, task,
                   gen_config))

        if (ex_i + 1) % 100 == 0 or ex_i + 1 == n_qst:
            with open(strs_save, 'wb') as f:
                pkl.dump(sampled_strs, f)
            print(f"  [{ex_i + 1}/{n_qst}]", flush=True)

    print('Saved sampled generations', flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Candidate generation (beam/sample) with YAML config")
    parser.add_argument('--config', type=str, default='conset/gen_configs/beam.yaml',
                        help='Path to YAML generation config')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['triviaqa', 'jeopardy', 'bioasq'],
                        help='Dataset to use')
    parser.add_argument('--model_name', type=str, default="allenai/Olmo-3-1025-7B",
                        help='HuggingFace model name')
    parser.add_argument('--revision', type=str, default='main',
                        help='Base-model revision/checkpoint')
    parser.add_argument('--beg_i', type=int, required=True,
                        help='Start index for examples')
    parser.add_argument('--end_i', type=int, required=True,
                        help='End index for examples')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing results')
    parser.add_argument('--is_instruct', action='store_true',
                        help='Format prompts with the tokenizer chat template')
    parser.add_argument('--debug', action='store_true',
                        help='Print per-instance timing')
    args = parser.parse_args()

    gen_config = load_gen_config(args.config)
    t0 = time.time()
    run(args, gen_config)
    print(f"[TIME] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
