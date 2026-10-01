"""Abstract base class for dataset configurations."""

from __future__ import annotations

import errno
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, List, NamedTuple, Optional

from transformers import StoppingCriteria, StoppingCriteriaList


class QA(NamedTuple):
    question: str
    answer: str


def load_dataset_with_estale_retry(load_fn, *args, max_attempts=5, **kwargs):
    """Call a Hugging Face loader, retrying a transient stale NFS handle."""
    for attempt in range(max_attempts):
        try:
            return load_fn(*args, **kwargs)
        except OSError as exc:
            if exc.errno != errno.ESTALE or attempt == max_attempts - 1:
                raise
            delay_s = 2 ** attempt
            print(f"Stale file handle while loading dataset; retrying in "
                  f"{delay_s}s ({attempt + 1}/{max_attempts - 1})",
                  flush=True)
            time.sleep(delay_s)


# Signature: (question, gold_ans, cand_strs) -> prompt string
JudgePromptFn = Callable[[str, str, list[str]], str]


def extract_between(text, marker=None, delimiter='\n', marker_at_line_start=True,
                    require_delimiter=False):
    """Extract content between marker and delimiter in text.

    Args:
        text: The text to search.
        marker: The string that signals the start of the answer region.
            If None, the answer starts from the beginning of text.
        delimiter: The string that terminates the answer.
        marker_at_line_start: If True, marker must appear at the start of a line.
        require_delimiter: If True, return None unless delimiter is found after marker.

    Returns:
        The stripped content between marker and delimiter, or None if not found.
    """
    if marker is None or marker == "":
        text = text.lstrip()
        marker = ""
        marker_at_line_start = False

    prefix = r'(?:^|\n)\s*' if marker_at_line_start else ''
    match = re.search(prefix + re.escape(marker), text)
    if match is None:
        return None

    after = text[match.end():]
    if delimiter in after:
        return after[:after.index(delimiter)].strip()
    else:
        return None if require_delimiter else after.strip()


class MarkerDelimiterStopping(StoppingCriteria):
    """Stop once every sequence has produced `marker` followed by `delimiter`.

    If marker is None, looks for delimiter in the generated text directly.
    marker_at_line_start: Whether to require the marker to be at the start of a line.
    """

    def __init__(self, tokenizer, prompt_length, marker=None, delimiter='\n',
                 marker_at_line_start=True):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.marker = marker
        self.delimiter = delimiter
        self.marker_at_line_start = marker_at_line_start

    def __call__(self, input_ids, scores, **kwargs):
        for i in range(input_ids.shape[0]):
            text = self.tokenizer.decode(
                input_ids[i, self.prompt_length:], skip_special_tokens=True
            )
            if extract_between(text, self.marker, self.delimiter, self.marker_at_line_start,
                              require_delimiter=True) is None:
                return False

        return True


@dataclass
class DatasetConfig(ABC):
    """Base configuration for a QA dataset used in the DINCO pipeline."""

    name: str
    ranges: list = field(default_factory=list)

    # --- Concrete helpers ---

    def _slice_ranges(self, items):
        """Slice *items* by ``self.ranges`` and concatenate."""
        parts = []
        for beg, end in self.ranges:
            parts.extend(items[beg:end])
        return parts

    @staticmethod
    def resolve_ex_i(ranges, local_i: int) -> int:
        """Map a local index (within sliced ranges) to a global index."""
        for beg, end in ranges:
            span = end - beg
            if local_i < span:
                return beg + local_i
            local_i -= span
        raise IndexError(f"local_i out of range")

    # --- Abstract methods (each subclass must implement) ---

    @abstractmethod
    def load_examples(self) -> List[QA]:
        """Load data as QA pairs (range controlled by ``ranges``).

        Returns:
            List of QA namedtuples.
        """
        ...

    @abstractmethod
    def num_instances(self) -> int:
        """Return the total number of instances in the dataset (ignoring ``ranges``)."""
        ...

    @abstractmethod
    def format_generation_prompt(self, question: Optional[str], is_instruct: bool, ex_i: int = 0) -> Optional[str]:
        """Build few-shot prompt for generation.

        Args:
            question: The question string, or None for exhausted BBH slots.
            is_instruct: Whether the model is instruction-tuned.
            ex_i: Index of the example in the dataset (used by BBH to
                  determine which subtask's ICL examples to use).

        Returns:
            Formatted prompt string, or None if question is None.
        """
        ...

    @abstractmethod
    def extract_answer(self, generated_text: str) -> Optional[str]:
        """Extract the answer from model output.

        Args:
            generated_text: Raw decoded text from model.

        Returns:
            Cleaned answer string, or None if extraction fails.
        """
        ...

    @abstractmethod
    def answers_match(self, a: str, b: str) -> bool:
        """Check if two answers are equivalent for self-consistency.

        Args:
            a: First answer string.
            b: Second answer string.

        Returns:
            True if answers are considered equivalent.
        """
        ...

    @abstractmethod
    def judge_correct(self, generated: str, gold: str) -> Optional[bool]:
        """Determine if generated answer is correct.

        Args:
            generated: Extracted generated answer.
            gold: Gold-standard answer.

        Returns:
            True/False for deterministic judging, None if LLM judge is needed.
        """
        ...

    # --- Concrete methods / properties with defaults ---

    def get_blacklist(self) -> set[int]:
        """Return canonical evaluation-split indices excluded from confidence use.

        The indices refer to the full list returned by the dataset's internal
        evaluation split, before ``ranges`` are applied.  Implementations must
        not remove these examples from :meth:`load_examples`, since generation
        artifacts and prompt indices rely on that stable ordering.
        """
        return set()

    @property
    def needs_llm_judge(self) -> bool:
        """Whether this dataset needs an LLM judge for correctness."""
        return False

    @property
    def max_new_tokens(self) -> int:
        """Maximum new tokens to generate."""
        return 100

    def get_stopping_criteria(self, tokenizer, prompt_len: int, batch_size: int, is_instruct: bool = False) -> Optional["StoppingCriteriaList"]:
        """Custom stopping criteria for generation.

        Returns:
            StoppingCriteriaList or None.
        """
        return None

    def normalize_answer(self, s: str) -> str:
        """Normalize an answer string for deduplication (e.g. in beam search).

        Override for datasets that need special normalization (e.g. lowercase + remove periods).
        Default is identity.
        """
        return s

    def get_icl_prefix(self, is_instruct: bool) -> Optional[str]:
        """Return the shared ICL prefix, or None if not applicable.

        Datasets with a fixed ICL block (same across all instances) can
        override this to enable KV-cache prefilling in gen_cands.
        """
        return None

    def build_confidence_prompt(self, question: str, answer: str,
                                ex_i: int = 0) -> str:
        """Build the user-content string for confidence prediction.

        Override for datasets that need extra context (e.g. passages).
        The returned string is wrapped by the caller for instruct / base
        models and appended with ``"Confidence: "``.

        Args:
            question: The question string.
            answer: The candidate answer.
            ex_i: Index of the example in the loaded dataset (used by
                  datasets that need per-instance context).
        """
        return (
            "Rate your confidence on a scale of 0-9, where 0 is completely "
            "uncertain and 9 is completely certain.\n\n"
            f"Question: {question}\n"
            f"Candidate answer: {answer}"
        )

    def extract_gt_answer(self, qa: QA) -> Optional[str]:
        """Extract ground-truth answer from a QA pair.

        Override for datasets where the answer field needs parsing (e.g. GSM8K).
        """
        return qa.answer

    def get_judge_prompt_fns(self) -> list[JudgePromptFn]:
        """Return a judge prompt function per eval instance.

        The returned list is parallel to :meth:`load_examples`.  Each
        function has signature ``(question, gold_ans, cand_strs) -> str``.
        By default every instance uses :func:`conset.judge.make_standard_judge_prompt`.

        Override for datasets where some instances need a different prompt
        (e.g. BioASQ list questions use ``make_multi_ans_judge_prompt``).
        """
        from conset.judge import make_standard_judge_prompt
        return [make_standard_judge_prompt] * len(self.load_examples())
