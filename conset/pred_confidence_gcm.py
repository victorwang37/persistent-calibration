"""Predict answer correctness with the external TriviaQA GCM.

Unlike :mod:`conset.pred_confidence`, this does not load the target model or
one of this repository's confidence adapters.  It takes each target
checkpoint's top generated answer and asks Hanqix's Qwen3-8B generalized
correctness model (GCM) whether that answer is correct.

The prompt builder below is intentionally copied from
``CalibratedModelAgnosticCorrectness/tuning_models/utils/prompt_utils.py`` so
this script has no dependency on that checkout.

Example:
    python -m conset.pred_confidence_gcm \\
        --inference_config conset/pred_configs/gcm.yaml
"""

import argparse
import os

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .dataset_configs import get_dataset_config
from .train_confidence import load_group_data


# The repository named in the original model release was renamed; this is the
# current public PEFT repository.  It is a LoRA adapter over GCM_BASE_MODEL.
DEFAULT_GCM_MODEL = 'Hanqix/GCM-Qwen3-8B-TriviaQA'
DEFAULT_GCM_BASE_MODEL = 'Qwen/Qwen3-8B'

def build_agnostic_zeroshot_with_response_prompt(example):
    """Build the GCM's model-agnostic zero-shot grading prompt.

    This is a local copy of
    ``_build_agnostic_zeroshot_with_response_prompt`` from the GCM authors'
    repository.  Do not change its wording without retraining or validating
    the external correctness model.
    """
    return (
        'You are grading responses to prompts for correctness, responses could '
        'be generated from multiple LLMs.'
        '\n###Prompt\n' + example['input_prompt'] +
        '\n###Response\n' + example['cleaned_model_completion'] +
        '\n###Instruction\n'
        + "Please respond just 'yes' or 'no' in lowercase if the Response "
          'correctly answers the Prompt: '
    )


def _format_gcm_prompt(tokenizer, question, answer):
    """Create the GCM prompt, then its Qwen chat-template wrapper."""
    prompt = build_agnostic_zeroshot_with_response_prompt({
        'input_prompt': f'Question: {question}\nAnswer:',
        'cleaned_model_completion': answer,
    })
    return tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _single_token_id(tokenizer, text):
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(
            f'GCM requires {text!r} to be one token, got {token_ids}')
    return token_ids[0]


def build_eval_examples(qa_examples, group_strs, group_probs, judge_labels,
                        blacklisted_qis):
    """Select the top generated answer and retain its judge label per question."""
    examples = []
    for qi, (qa, groups) in enumerate(zip(qa_examples, group_strs)):
        if qi in blacklisted_qis or groups is None or len(groups) == 0:
            continue
        group_i = group_probs[qi, :len(groups)].argmax().item()
        examples.append({
            'question': qa.question,
            'answer': groups[group_i][0],
            'q_i': qi,
            'judge_label': judge_labels[qi, group_i].item(),
        })
    return examples


@torch.inference_mode()
def predict_confidences(model, tokenizer, examples, batch_size):
    """Return raw next-token ``P(yes)`` for the GCM prompts.

    This follows the released GCM evaluator: confidence is the vocabulary-wide
    softmax probability of ``yes``, rather than P(yes) renormalized over only
    ``yes`` and ``no``.
    """
    model.eval()
    yes_token_id = _single_token_id(tokenizer, 'yes')
    prompts = [_format_gcm_prompt(tokenizer, ex['question'], ex['answer'])
               for ex in examples]
    device = next(model.parameters()).device
    confs = []
    index = 0
    while index < len(prompts):
        end = min(index + batch_size, len(prompts))
        batch_prompts = prompts[index:end]
        batch = tokenizer(batch_prompts, padding=True, return_tensors='pt')
        try:
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                use_cache=False,
            )
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            if batch_size == 1:
                raise
            torch.cuda.empty_cache()
            batch_size = max(1, batch_size // 2)
            print(f'  OOM: reducing GCM batch_size to {batch_size}', flush=True)
            continue
        confs.append(
            torch.softmax(outputs.logits[:, -1, :].float(), dim=-1)[:, yes_token_id]
            .cpu())
        index = end
        if (index // batch_size) % 10 == 0 or index == len(prompts):
            print(f'  [{index}/{len(prompts)}]', flush=True)
    return torch.cat(confs) if confs else torch.empty(0)


def _default_output_dir(dataset, model_name, gcm_model):
    """Keep GCM artifacts separate from target-model generation artifacts."""
    target_name = model_name.rsplit('/', 1)[-1]
    gcm_name = gcm_model.rsplit('/', 1)[-1]
    return os.path.join('results', 'conset', dataset, target_name,
                        'gcm', gcm_name)


def _save_predictions(examples, confs, beg, end, out_path):
    """Save standard confidence/label tensors aligned to the requested range."""
    conf_full = torch.full((end - beg,), float('nan'))
    labels_full = torch.full((end - beg,), -1, dtype=torch.long)
    for example, confidence in zip(examples, confs):
        conf_full[example['q_i']] = confidence
        labels_full[example['q_i']] = example['judge_label']
    torch.save({
        'confs': conf_full,
        'judge_labels': labels_full,
        'eval_range': (beg, end),
    }, out_path)


def _normalize_prediction_specs(specs, defaults):
    """Validate plan entries and expand them to one target checkpoint each."""
    if not isinstance(specs, list) or not specs:
        raise ValueError("Inference config requires a nonempty 'predictions' list")
    work = []
    output_paths = set()
    for spec in specs:
        if not isinstance(spec, dict):
            raise ValueError('Each prediction entry must be a mapping')
        model_name = spec.get('model_name', defaults.get('model_name'))
        dataset = spec.get('dataset', defaults.get('dataset'))
        if not model_name or not dataset:
            raise ValueError('Each prediction entry needs model_name and dataset')
        try:
            beg = int(spec.get('beg', defaults.get('beg')))
            end = int(spec.get('end', defaults.get('end')))
        except (TypeError, ValueError) as exc:
            raise ValueError('Each prediction entry needs integer beg and end') from exc
        if beg >= end:
            raise ValueError(f'Invalid range for {dataset}: {beg}-{end}')
        revisions = spec.get('eval_revisions', defaults.get('eval_revisions'))
        if not isinstance(revisions, list) or not revisions:
            raise ValueError(
                'Each prediction entry needs a nonempty eval_revisions list')
        batch_size = int(spec.get('batch_size', defaults['batch_size']))
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        gen_config_name = spec.get('gen_config_name',
                                   defaults['gen_config_name'])
        output_dir = spec.get('output_dir') or _default_output_dir(
            dataset, model_name, defaults['gcm_model'])
        for revision in revisions:
            if not isinstance(revision, str) or not revision:
                raise ValueError('eval_revisions must contain nonempty strings')
            out_path = os.path.join(
                output_dir,
                f'pred_confs_gcm_{dataset}_{revision}_{beg}-{end}.pt')
            if out_path in output_paths:
                raise ValueError(
                    f'Inference config requests the same output twice: {out_path}')
            output_paths.add(out_path)
            work.append({
                'model_name': model_name,
                'dataset': dataset,
                'beg': beg,
                'end': end,
                'revision': revision,
                'gen_config_name': gen_config_name,
                'batch_size': batch_size,
                'out_path': out_path,
            })
    return work


def run_prediction_work(work, gcm_model, gcm_base_model, force=False):
    """Run all pending target predictions while keeping the GCM resident."""
    pending = []
    for item in work:
        if os.path.exists(item['out_path']) and not force:
            print(f"Skipping {item['model_name']} / {item['revision']}: "
                  f"{item['out_path']} exists", flush=True)
        else:
            pending.append(item)
    if not pending:
        print('All requested GCM prediction files already exist.', flush=True)
        return

    # Dataset examples and blacklist indices depend only on a dataset/range,
    # not on which target model or checkpoint generated the answers.
    task_cache = {}
    print(f'Loading GCM base model: {gcm_base_model}', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(gcm_base_model, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    base_model = AutoModelForCausalLM.from_pretrained(
        gcm_base_model, device_map='auto', dtype=torch.bfloat16)
    print(f'Loading GCM adapter: {gcm_model}', flush=True)
    model = PeftModel.from_pretrained(base_model, gcm_model)

    for item in pending:
        dataset, beg, end = item['dataset'], item['beg'], item['end']
        task_key = (dataset, beg, end)
        if task_key not in task_cache:
            task = get_dataset_config(dataset, ranges=[(beg, end)])
            qa_examples = task.load_examples()
            blacklist = task.get_blacklist()
            blacklisted_qis = {
                qi for qi in range(end - beg) if beg + qi in blacklist
            }
            task_cache[task_key] = (qa_examples, blacklisted_qis)
        qa_examples, blacklisted_qis = task_cache[task_key]

        print(f"=== {dataset} {beg}-{end}; target: {item['model_name']} / "
              f"{item['revision']} ===", flush=True)
        _, group_strs, group_probs, judge_labels = load_group_data(
            dataset, item['model_name'], item['revision'],
            item['gen_config_name'], [(beg, end)], qa_examples=qa_examples)
        examples = build_eval_examples(
            qa_examples, group_strs, group_probs, judge_labels,
            blacklisted_qis)
        print(f'  {len(examples)} eval instances '
              f'({len(blacklisted_qis)} blacklisted)', flush=True)
        confs = predict_confidences(model, tokenizer, examples,
                                    item['batch_size'])
        os.makedirs(os.path.dirname(item['out_path']), exist_ok=True)
        if os.path.exists(item['out_path']):
            print(f"  Overwriting {item['out_path']}", flush=True)
        _save_predictions(examples, confs, beg, end, item['out_path'])
        print(f"  Saved to {item['out_path']}", flush=True)

    del model
    torch.cuda.empty_cache()
    print('\nDone.', flush=True)


def run_inference_config(config_path, cli_force=False):
    """Run a YAML plan spanning arbitrary target models and datasets.

    The GCM is loaded once for the complete plan.  Each ``predictions`` entry
    supplies ``model_name``, ``dataset``, ``beg``, ``end``, and
    ``eval_revisions``; plan-level values serve as defaults.  ``batch_size``,
    ``gen_config_name``, and ``output_dir`` may also be overridden per entry.
    """
    with open(config_path) as f:
        plan = yaml.safe_load(f)
    if not isinstance(plan, dict):
        raise ValueError('Inference config must contain a mapping')
    gcm_model = plan.get('gcm_model', DEFAULT_GCM_MODEL)
    gcm_base_model = plan.get('gcm_base_model', DEFAULT_GCM_BASE_MODEL)
    defaults = {
        'model_name': plan.get('model_name'),
        'dataset': plan.get('dataset'),
        'beg': plan.get('beg'),
        'end': plan.get('end'),
        'eval_revisions': plan.get('eval_revisions'),
        'batch_size': int(plan.get('batch_size', 8)),
        'gen_config_name': plan.get('gen_config_name', 'beam'),
        'gcm_model': gcm_model,
    }
    work = _normalize_prediction_specs(plan.get('predictions'), defaults)
    force = cli_force or bool(plan.get('force', False))
    run_prediction_work(work, gcm_model, gcm_base_model, force=force)



def main():
    parser = argparse.ArgumentParser(
        description='Predict target-answer correctness with the external GCM')
    parser.add_argument('--inference_config', required=True,
                        help='YAML plan spanning target models, checkpoints, '
                             'datasets, and ranges; the GCM is loaded once')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing GCM prediction artifacts')
    args = parser.parse_args()
    run_inference_config(args.inference_config, cli_force=args.force)


if __name__ == '__main__':
    main()
