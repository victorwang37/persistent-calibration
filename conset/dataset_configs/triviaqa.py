"""TriviaQA dataset configuration."""

import re
from dataclasses import dataclass, field
from typing import List, Optional

from datasets import load_dataset as hf_load_dataset
from transformers import StoppingCriteriaList

from .base import (QA, DatasetConfig, MarkerDelimiterStopping, extract_between,
                   load_dataset_with_estale_retry)


BLACKLIST_QUESTIONS = [
    "Which is the longest length below?",
    "The books were Alice and Jerry in the USA – what were they here?",
    ("Three female trainers have won the Grand National, Jenny Pitman is one "
     "name either of the other two,"),
]
DATASET_REVISION = '0f7faf33a3908546c6fd5b73a660e0f8ff173c2f'


@dataclass
class TriviaQAConfig(DatasetConfig):
    """Configuration for the TriviaQA dataset."""

    name: str = "triviaqa"
    n_icl: int = 5
    max_n_icl: int = 10
    _icl_examples: List[QA] = field(default_factory=list, init=False, repr=False)
    _eval_examples: Optional[List[QA]] = field(default=None, init=False,
                                                repr=False)

    @staticmethod
    def _dedup_questions(ds, answer_key="value") -> List[QA]:
        """Deduplicate a TriviaQA split by question, preserving order.

        Args:
            ds: HuggingFace dataset split (already shuffled if desired).
            answer_key: Key within the 'answer' dict to use as the answer.
        """
        seen = set()
        examples: List[QA] = []
        for row in ds:
            q = row["question"].strip()
            if q not in seen:
                seen.add(q)
                examples.append(QA(q, row["answer"][answer_key].strip()))
        return examples

    def _load_icl_examples(self):
        """Load ICL examples from the train split (cached on first call)."""
        if not self._icl_examples:
            assert self.n_icl <= self.max_n_icl
            ds = load_dataset_with_estale_retry(
                hf_load_dataset, "mandarjoshi/trivia_qa", "rc.nocontext",
                revision=DATASET_REVISION)
            ds = ds["train"].shuffle(seed=17)
            self._icl_examples = self._dedup_questions(ds)[:self.n_icl]

    def _load_eval_examples(self) -> List[QA]:
        """Load and deduplicate validation + train splits (validation first).

        The first max_n_icl train instances are skipped, as they are reserved
        as potential ICL examples.
        """
        if self._eval_examples is None:
            ds = load_dataset_with_estale_retry(
                hf_load_dataset, "mandarjoshi/trivia_qa", "rc.nocontext",
                revision=DATASET_REVISION)
            val = self._dedup_questions(ds["validation"].shuffle(seed=17))
            train = self._dedup_questions(
                ds["train"].shuffle(seed=17))[self.max_n_icl:]
            self._eval_examples = val + train
        return self._eval_examples

    def load_examples(self) -> List[QA]:
        return self._slice_ranges(self._load_eval_examples())

    def num_instances(self) -> int:
        return len(self._load_eval_examples())

    def get_blacklist(self) -> set[int]:
        """Return the uniquely matched bad-question indices in the eval split."""
        self._load_icl_examples()
        icl_blacklist = [example.question for example in self._icl_examples
                         if example.question in BLACKLIST_QUESTIONS]
        assert not icl_blacklist, (
            "TriviaQA blacklist question(s) appear in the ICL set: "
            f"{icl_blacklist!r}")

        examples = self._load_eval_examples()
        blacklist = set()
        for question in BLACKLIST_QUESTIONS:
            matches = [i for i, example in enumerate(examples)
                       if example.question == question]
            assert len(matches) == 1, (
                f"Expected exactly one TriviaQA blacklist match for {question!r}; "
                f"found {len(matches)}")
            blacklist.add(matches[0])
        return blacklist

    def format_generation_prompt(self, question: Optional[str], is_instruct: bool, ex_i: int = 0) -> Optional[str]:
        if question is None:
            return None

        if self.n_icl > 0:
            self._load_icl_examples()
            icl_parts = [f"Prompt: {ex.question}\nAnswer: {ex.answer}"
                         for ex in self._icl_examples]
            icl_block = "\n\n".join(icl_parts)

            if is_instruct:
                prompt = (
                    f"Here are {len(self._icl_examples)} sets of example prompt and answer.\n\n"
                    f"{icl_block}\n\n"
                    "---\n\n"
                    "Now, here is a new prompt to answer. Answer with a concise phrase, as in the examples.\n\n"
                    f"Prompt: {question}\nAnswer:"
                )
            else:
                prompt = f"{icl_block}\n\nPrompt: {question}\nAnswer:"
        else:
            if is_instruct:
                prompt = (
                    "Answer the following question with a concise phrase.\n\n"
                    f"Prompt: {question}\nAnswer:"
                )
            else:
                prompt = f"Prompt: {question}\nAnswer:"

        return prompt

    def extract_answer(self, generated_text: str) -> str:
        return extract_between(generated_text, marker=None, delimiter='\n')

    def normalize_answer(self, s: str) -> str:
        s = self.extract_answer(s)
        s = s.lower()
        s = s.replace('.', '')
        return s

    def answers_match(self, a: str, b: str) -> bool:
        return self.normalize_answer(a) == self.normalize_answer(b)

    def judge_correct(self, generated: str, gold: str) -> Optional[bool]:
        return None  # needs LLM judge

    @property
    def needs_llm_judge(self) -> bool:
        return True
