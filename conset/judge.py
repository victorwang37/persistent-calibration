"""Judge answers using GPT-5.4 via OpenAI Batch API.

Usage:
    # Build and preview the requests without submitting a batch:
    python -m conset.judge submit [--overwrite]

    # Submit the requests to the OpenAI Batch API:
    python -m conset.judge submit --submit [--overwrite]

    python -m conset.judge retrieve [batch_id]

batch_id can be a full ID string or a positive integer N to use the Nth-last
entry from batch_input/log.txt (e.g. 1 = most recent, 2 = second most recent).
Defaults to 1 if omitted.
"""

import argparse
import json
import os
import pickle
from datetime import datetime
from pathlib import Path

import torch
from openai import OpenAI

from conset.dataset_configs import get_dataset_config

def make_standard_judge_prompt(question: str, gold_ans: str, cand_strs: list[str]) -> str:
    n = len(cand_strs)
    s = 's' if n > 1 else ''
    header = f'Below is a question along with the ground-truth answer and {n} candidate answer{s}. Please determine whether each candidate answer attempts the question (rather than abstaining or not including a well-formed answer), and if so, determine whether the candidate answer is semantically consistent with the ground-truth answer in the context of the question. For each candidate answer, output a line with the answer number followed by a period and a space and exactly one more character: "R" for correct, "W" for incorrect, or "N" for no attempt.'

    cand_list_str = '\n'.join(f'{i+1}. {cand_str}' for i, cand_str in enumerate(cand_strs))

    return (
        f'{header}\n\n'
        f'Question: {question}\n'
        f'Ground-truth answer: {gold_ans}\n'
        f'Candidate answers:\n'
        f'{cand_list_str}'
    )

def make_multi_ans_judge_prompt(question: str, gold_ans: str, cand_strs: list[str]) -> str:
    """Judge prompt for questions with multiple acceptable gold answers.

    ``gold_ans`` is a string representation of the list of reference answers.
    A candidate is correct if it matches *any* of them.
    """
    n = len(cand_strs)
    s = 's' if n > 1 else ''
    header = (
        f'Below is a question along with a list of reference ground-truth '
        f'answers and {n} candidate answer{s}. Please determine whether each '
        f'candidate answer attempts the question (rather than abstaining or '
        f'not including a well-formed answer), and if so, determine whether '
        f'the candidate answer is semantically consistent with ANY of the '
        f'ground-truth answers in the context of the question. For each '
        f'candidate answer, output a line with the answer number followed by '
        f'a period and a space and exactly one more character: "R" for '
        f'correct, "W" for incorrect, or "N" for no attempt.'
    )

    cand_list_str = '\n'.join(f'{i+1}. {cand_str}' for i, cand_str in enumerate(cand_strs))

    return (
        f'{header}\n\n'
        f'Question: {question}\n'
        f'Reference ground-truth answers: {gold_ans}\n'
        f'Candidate answers:\n'
        f'{cand_list_str}'
    )


MODEL = "gpt-5.4"
INPUT_COST_PER_TOKEN = 1.25 / 1e6  # $1.25 per 1M input tokens
OUTPUT_COST_PER_TOKEN = 7.50 / 1e6  # $7.50 per 1M output tokens
DATASET_NAME = "jeopardy"
JSONL_PATH = f"conset/batch_input/{DATASET_NAME}.jsonl"
HUMAN_PATH = f"conset/batch_input/{DATASET_NAME}_output_human.jsonl"
QUEUE_PATH = Path("conset/batch_input/queue.jsonl")

# MODEL_HALF_NAME = "Olmo-3-1025-7B"
# MODEL_HALF_NAME = "Olmo-3-1125-32B"
# MODEL_HALF_NAME = "marin-8b-base"
MODEL_HALF_NAME = "Qwen3-8B"
BASE = Path(f"results/conset/{DATASET_NAME}/{MODEL_HALF_NAME}")
# REVISIONS = [
#     "stage1-step141000",
#     "stage1-step283000",
#     "stage1-step424000",
#     "stage1-step566000",
#     "stage1-step707000",
#     "stage1-step1272000",
#     "stage1-step1413814",
# ]
# REVISIONS = [
#     "stage1-step66000",
#     "stage1-step131000",
#     "stage1-step197000",
#     "stage1-step262000",
#     "stage1-step328000",
#     "stage1-step590120",
#     "stage1-step656000",
# ]
# REVISIONS = [
#     "kestrel",
#     "ocelot",
#     "jellyfish",
#     "phoenix",
#     "starling",
#     "deeper-starling",
# ]
REVISIONS = ["main"]
SHARDS = [(15000, 17500), (17500, 20000)]
CAND_TYPE = "beam"
MAX_N_GROUP = 10
MIN_MAX_OUTPUT_TOKENS = 16
EXTRA_TOKENS = 3  # special tokens the model always produces

LOG_PATH = Path("conset/batch_input/log.txt")


def make_custom_id(revision: str, beg_i: int, end_i: int, q_i: int) -> str:
    """Return the model-qualified identifier used for new judge requests."""
    return f"{MODEL_HALF_NAME}|{revision}|{beg_i}-{end_i}|{q_i}"


def parse_custom_id(custom_id: str) -> tuple[str | None, str, str, int]:
    """Parse either a current four-field or legacy three-field custom ID.

    Current IDs are ``model|revision|beg-end|question_index``.  Legacy IDs
    omit the model field; their model is returned as ``None`` so callers can
    preserve their previous model-agnostic behavior.
    """
    fields = custom_id.split("|")
    if len(fields) == 3:
        model_name = None
        revision, shard, q_i_s = fields
    elif len(fields) == 4:
        model_name, revision, shard, q_i_s = fields
    else:
        raise ValueError(
            f"custom_id must have three or four pipe-separated fields, got "
            f"{custom_id!r}")
    try:
        q_i = int(q_i_s)
    except ValueError as exc:
        raise ValueError(
            f"custom_id has non-integer question index: {custom_id!r}") from exc
    return model_name, revision, shard, q_i


def get_client():
    return OpenAI(
        base_url=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
    )


def resolve_log_entry(batch_id_arg: str) -> dict[str, str]:
    """Resolve batch_id and return parsed log fields.

    batch_id_arg can be a positive integer N (Nth-last log entry) or a
    literal batch ID string.  Returns a dict with keys like 'batch_id',
    'jsonl', etc.
    """
    if batch_id_arg.isdigit():
        n = int(batch_id_arg)
        lines = LOG_PATH.read_text().strip().split("\n")
        if n < 1 or n > len(lines):
            raise ValueError(f"Log has {len(lines)} entries, cannot index {n} from end")
        line = lines[-n]
    else:
        # Search for the line containing this batch_id
        line = None
        for l in LOG_PATH.read_text().strip().split("\n"):
            if batch_id_arg in l:
                line = l
                break
        if line is None:
            return {"batch_id": batch_id_arg}

    fields = {}
    for part in line.split("|"):
        part = part.strip()
        if "=" in part:
            key, val = part.split("=", 1)
            fields[key] = val
    return fields


def iter_shards():
    """Yield (revision, beg_i, end_i, shard_dir) for each shard."""
    for revision in REVISIONS:
        for beg_i, end_i in SHARDS:
            shard_dir = BASE / revision / f"{beg_i}-{end_i}" / CAND_TYPE
            yield revision, beg_i, end_i, shard_dir


def load_group_counts() -> dict[str, int]:
    """Return {custom_id: n_groups} for all shards.

    Used by retrieve() to check for truncated responses. Logic follows submit().
    """
    counts: dict[str, int] = {}
    for revision, beg_i, end_i, shard_dir in iter_shards():
        with open(shard_dir / "beam_group_strs.pkl", "rb") as f:
            group_strs = pickle.load(f)
        for q_i, groups in enumerate(group_strs):
            if groups is None:
                continue
            custom_id = make_custom_id(revision, beg_i, end_i, q_i)
            counts[custom_id] = len(groups)
    return counts


def write_human_queue(custom_ids: set[str], group_counts: dict[str, int],
                      path: Path = QUEUE_PATH):
    """Write default-W human-annotation placeholders for unresolved IDs."""
    with open(path, "w") as f:
        for custom_id in sorted(custom_ids):
            n_groups = group_counts[custom_id]
            entry = {
                "custom_id": custom_id,
                "human_judgment": "\n".join(
                    f"{group_i + 1}. W" for group_i in range(n_groups)),
            }
            f.write(json.dumps(entry) + "\n")
    print(f"Wrote {len(custom_ids)} default-W human-judgment placeholders to {path}")


def report_cost(requests):
    """Print and return the estimated cost for a batch request list."""
    input_tokens = sum(len(r["body"]["input"]) for r in requests) / 4
    output_tokens = sum(r["body"]["max_output_tokens"] for r in requests)
    input_cost = input_tokens * INPUT_COST_PER_TOKEN
    output_cost = output_tokens * OUTPUT_COST_PER_TOKEN
    total_cost = input_cost + output_cost
    print(f"Total requests: {len(requests)}")
    print(f"Estimated cost: ${total_cost:.2f} "
          f"(input ${input_cost:.2f} + output ${output_cost:.2f})")
    return total_cost


def upload_and_create_batch(request_count, estimated_cost):
    """Upload ``JSONL_PATH``, create a batch, and record it in the log."""
    client = get_client()
    with open(JSONL_PATH, "rb") as batch_file:
        batch_input_file = client.files.create(file=batch_file, purpose="batch")
    print(f"Uploaded file: {batch_input_file.id}")

    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={"description": f"{DATASET_NAME} judge"},
    )
    print(f"Batch created: {batch.id}")

    with open(LOG_PATH, "a") as f:
        f.write(f"{datetime.now().isoformat()} | "
                f"batch_id={batch.id} | "
                f"file_id={batch_input_file.id} | "
                f"model={MODEL} | "
                f"requests={request_count} | "
                f"est_cost=${estimated_cost:.2f} | "
                f"jsonl={JSONL_PATH}\n")
    print(f"Logged to {LOG_PATH}")


def submit(args):
    # A preview invocation has already materialized the exact request set in
    # JSONL_PATH.  Submit that file unchanged on a later --submit invocation.
    if args.submit and Path(JSONL_PATH).exists() and not args.overwrite:
        with open(JSONL_PATH) as f:
            requests = [json.loads(line) for line in f if line.strip()]
        if not requests:
            print(f"Existing {JSONL_PATH} contains no requests.")
            return
        print(f"Reusing existing {JSONL_PATH} (pass --overwrite to rebuild it)")
        estimated_cost = report_cost(requests)
        upload_and_create_batch(len(requests), estimated_cost)
        return

    requests = []

    for revision, beg_i, end_i, shard_dir in iter_shards():
        judge_path = shard_dir / "judge.pt"
        if judge_path.exists() and not args.overwrite:
            print(f"Skipping {revision}/{beg_i}-{end_i}/{CAND_TYPE} (judge.pt exists)")
            continue

        with open(shard_dir / "beam_group_strs.pkl", "rb") as f:
            group_strs = pickle.load(f)

        task = get_dataset_config(DATASET_NAME, ranges=[(beg_i, end_i)])
        examples = task.load_examples()
        prompt_fns = task.get_judge_prompt_fns()

        for q_i, (ex, prompt_fn, groups) in enumerate(zip(examples, prompt_fns, group_strs)):
            if groups is None:
                continue
            cand_strs = [group[0] for group in groups]
            prompt = prompt_fn(ex.question, ex.answer, cand_strs)
            custom_id = make_custom_id(revision, beg_i, end_i, q_i)
            max_tokens = max(4 * len(groups) - 1 + EXTRA_TOKENS, MIN_MAX_OUTPUT_TOKENS)
            requests.append({
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": MODEL,
                    "input": prompt,
                    "max_output_tokens": max_tokens,
                    "temperature": 0,
                },
            })

    if not requests:
        print("No requests to submit.")
        return

    estimated_cost = report_cost(requests)

    with open(JSONL_PATH, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")
    print(f"Wrote {JSONL_PATH}")

    if args.submit:
        upload_and_create_batch(len(requests), estimated_cost)


def retrieve(args):
    client = get_client()
    log_entry = resolve_log_entry(args.batch_id)
    batch_id = log_entry["batch_id"]
    jsonl_path = log_entry.get("jsonl", JSONL_PATH)
    batch = client.batches.retrieve(batch_id)
    print(f"Batch:  {batch_id}")
    print(f"Status: {batch.status}")
    if batch.request_counts:
        c = batch.request_counts
        print(f"Progress: {c.completed}/{c.total} completed, {c.failed} failed")

    if batch.status != "completed":
        if batch.status == "failed" and batch.errors:
            for error in batch.errors.data:
                print(f"Error: [{error.code}] {error.message}")
        if batch.status == "failed" and batch.error_file_id:
            error_content = client.files.content(batch.error_file_id)
            if error_content.text.strip():
                print(f"Error file contents:")
                for line in error_content.text.strip().split("\n"):
                    entry = json.loads(line)
                    print(f"  {entry['custom_id']}: {entry.get('error', entry.get('response', {}))}")
        return

    assert batch.output_file_id is not None
    content = client.files.content(batch.output_file_id)

    # Save raw API output, sorted by custom_id (model, revision, shard, q_i).
    # ``parse_custom_id`` also permits a response from a legacy batch.
    raw_path = jsonl_path.replace(".jsonl", "_output.jsonl")
    output_lines = content.text.strip().split("\n")
    def sort_key(l):
        model_name, revision, shard, q_i = parse_custom_id(
            json.loads(l)["custom_id"])
        return (model_name or "", revision, shard, q_i)
    output_lines.sort(key=sort_key)
    with open(raw_path, "w") as f:
        for line in output_lines:
            f.write(line + "\n")
    print(f"Saved raw output to {raw_path}")

    # Load expected group counts from data
    group_counts = load_group_counts()

    # Parse: (revision, shard) -> {(q_i, grp_i): int}
    # Values: 1 = correct (R), 2 = incorrect (W), 3 = no attempt (N)
    LABEL_MAP = {"R": 1, "W": 2, "N": 3}
    shard_results: dict[tuple[str, str], dict[tuple[int, int], int]] = {}
    failed_ids: set[str] = set()

    def parse_judgment(custom_id: str, text: str):
        """Parse judgment text and populate shard_results.

        Returns True if parsing succeeded fully, False otherwise.
        """
        model_name, revision, shard, q_i = parse_custom_id(custom_id)
        if model_name is not None and model_name != MODEL_HALF_NAME:
            raise ValueError(
                f"custom_id {custom_id!r} is for model {model_name!r}, "
                f"not current model {MODEL_HALF_NAME!r}")
        canonical_id = make_custom_id(
            revision, *map(int, shard.split("-")), q_i)
        key = (revision, shard)
        if key not in shard_results:
            shard_results[key] = {}
        resp_lines = [l for l in text.split("\n") if l.strip()]

        ok = True
        n_expected = group_counts.get(canonical_id, len(resp_lines))
        if len(resp_lines) < n_expected:
            print(f"Warning: {custom_id} returned {len(resp_lines)} lines, expected {n_expected}")
            ok = False

        for grp_i, resp_line in enumerate(resp_lines):
            # expected format: "1. R" — take the last non-empty token
            tok = resp_line.strip().rsplit(None, 1)[-1].rstrip(".")
            label = LABEL_MAP.get(tok)
            if label is None:
                print(f"Warning: unexpected line '{resp_line.strip()}' for {custom_id} group {grp_i}, defaulting to N")
                label = 3
                ok = False
            shard_results[key][(q_i, grp_i)] = label
        return ok

    returned_ids = set()
    for line in content.text.strip().split("\n"):
        entry = json.loads(line)
        custom_id = entry["custom_id"]
        _, revision, shard, q_i = parse_custom_id(custom_id)
        canonical_id = make_custom_id(
            revision, *map(int, shard.split("-")), q_i)
        returned_ids.add(canonical_id)
        text = entry["response"]["body"]["output"][0]["content"][0]["text"].strip()
        if not parse_judgment(custom_id, text):
            failed_ids.add(canonical_id)

    # Check for missing responses
    missing = set(group_counts) - returned_ids
    if missing:
        print(f"{len(missing)} requests missing from response")
        failed_ids |= missing

    # Fill in with human annotations (overrides GPT for refusals)
    # Only apply to shards already in shard_results — don't create new judge files.
    if os.path.exists(HUMAN_PATH):
        n_human = 0
        n_skipped = 0
        for line in open(HUMAN_PATH):
            entry = json.loads(line)
            cid = entry["custom_id"]
            human_model, revision, shard, q_i = parse_custom_id(cid)
            if human_model is not None and human_model != MODEL_HALF_NAME:
                n_skipped += 1
                continue
            if (revision, shard) not in shard_results:
                n_skipped += 1
                continue
            if parse_judgment(cid, entry["human_judgment"]):
                canonical_id = make_custom_id(
                    revision, *map(int, shard.split("-")), q_i)
                failed_ids.discard(canonical_id)
            n_human += 1
        print(f"Applied {n_human} human annotations from {HUMAN_PATH}"
              + (f" (skipped {n_skipped} for absent shards)" if n_skipped else ""))

    # Materialize the remaining malformed or missing responses in a compact
    # queue for human review.  Existing human annotations above take priority,
    # so this file contains only IDs that still need supplementation.
    write_human_queue(failed_ids, group_counts)

    if failed_ids:
        print(f"\n{len(failed_ids)} IDs still have parse failures after human labels:")
        print("Not saving any judge.pt files until all judgments are supplemented.")
        return

    print("\nAll missing judgments supplemented by human.")

    # Save per-shard judge.pt
    # Values: 1 = correct, 2 = incorrect, 3 = no attempt, -1 = padding
    for (revision, shard), judgments in sorted(shard_results.items()):
        beg_i, end_i = map(int, shard.split("-"))
        n_qst = end_i - beg_i
        judge = -torch.ones(n_qst, MAX_N_GROUP, dtype=torch.long)
        for (qi, gi), label in judgments.items():
            judge[qi, gi] = label

        out_path = BASE / revision / shard / CAND_TYPE / "judge.pt"
        torch.save(judge, out_path)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")

    p_submit = sub.add_parser("submit", help="Build JSONL, upload, and create batch")
    p_submit.add_argument("--submit", action="store_true",
                          help="Submit instead of just previewing; reuse an "
                               "existing preview JSONL unless --overwrite")
    p_submit.add_argument("--overwrite", action="store_true",
                          help="Re-judge shards that already have judge.pt")

    p_retrieve = sub.add_parser("retrieve", help="Download results and save judge.pt per shard")
    p_retrieve.add_argument("batch_id", nargs="?", default="1",
                            help="Batch ID or N for Nth-last log entry (default: 1)")

    args = parser.parse_args()
    if args.command == "submit":
        submit(args)
    elif args.command == "retrieve":
        retrieve(args)
    else:
        parser.print_help()
