"""Jeopardy dataset configuration (soldni/jeopardy, all_questions subset)."""

from collections import Counter
from dataclasses import dataclass, field
import re
from typing import List, Optional

from datasets import load_dataset as hf_load_dataset

from .base import (QA, DatasetConfig, extract_between,
                   load_dataset_with_estale_retry)


CLUE_THEN_MISSING_RE = re.compile(
    r"\bclue\b.*\bmissing\b", re.IGNORECASE | re.DOTALL)
DATASET_REVISION = 'aebec90f52498628fc26f7c06c00cd618374026f'


def _is_blacklisted_clue(question: str) -> bool:
    """Whether a category-prefixed Jeopardy question is malformed."""
    # ``[CATEGORY]`` is metadata added by this config, not clue text.
    clue = re.sub(r"^\[[^\]]*\]\s*", "", question).strip()
    is_single_quoted = (
        len(clue) >= 2 and clue[0] == "'" and clue[-1] == "'")
    return bool(CLUE_THEN_MISSING_RE.search(clue)) or not is_single_quoted


@dataclass
class JeopardyConfig(DatasetConfig):
    """Configuration for the Jeopardy dataset."""

    name: str = "jeopardy"
    n_icl: int = 5
    max_n_icl: int = 10
    _icl_examples: List[QA] = field(default_factory=list, init=False, repr=False)
    _all_examples: Optional[List[QA]] = field(default=None, init=False, repr=False)

    INCLUDE_CATEGORIES = set(["LITERATURE", "SCIENCE", "AMERICAN HISTORY", "POTPOURRI", "WORLD HISTORY", "WORD ORIGINS", "HISTORY", "COLLEGES & UNIVERSITIES", "SPORTS", "U.S. CITIES", "WORLD GEOGRAPHY", "BODIES OF WATER", "STATE CAPITALS", "BUSINESS & INDUSTRY", "ANIMALS", "WORLD CAPITALS", "U.S. GEOGRAPHY", "RELIGION", "SHAKESPEARE", "OPERA", "ISLANDS", "BALLET", "FICTIONAL CHARACTERS", "TELEVISION", "PEOPLE", "LANGUAGES", "TRANSPORTATION", "THE BIBLE", "ART & ARTISTS", "BOOKS & AUTHORS", "U.S. HISTORY", "FOOD", "GEOGRAPHY", "ART", "HOLIDAYS & OBSERVANCES", "MUSEUMS", "SCIENCE & NATURE", "AMERICAN LITERATURE", "3-LETTER WORDS", "AMERICANA", "POETS & POETRY", "ANNUAL EVENTS", "POP MUSIC", "AUTHORS", "CLASSICAL MUSIC", "QUOTATIONS", "HODGEPODGE", "MYTHOLOGY", "NONFICTION", "WORLD CITIES", "THE MOVIES", "THE CIVIL WAR", "U.S. PRESIDENTS", "MUSICAL INSTRUMENTS", "FOOD & DRINK", "AROUND THE WORLD", "MUSIC", "4-LETTER WORDS", "HISTORIC NAMES", "COMPOSERS", "ASTRONOMY", "BIOLOGY", "POTENT POTABLES", "MOUNTAINS", "EXPLORERS", "EUROPEAN HISTORY", "MEDICINE", "SCIENTISTS", "ORGANIZATIONS", "TRAVEL & TOURISM", "WEIGHTS & MEASURES", "FIRST LADIES", "FRUITS & VEGETABLES", "ARCHITECTURE", "MAGAZINES", "THE BODY HUMAN", "IN THE DICTIONARY", "AWARDS", "ZOOLOGY", "FAMOUS AMERICANS", "VOCABULARY", "FASHION", "THEATRE", "19th CENTURY AMERICA", "NATURE"])
    EXCLUDE_CATEGORIES = set(["BEFORE & AFTER", "RHYME TIME", "STUPID ANSWERS", "COMMON BONDS", "HOMOPHONES"])
    MIN_COUNT = 150

    def _load_all_examples(self) -> List[QA]:
        """Load and cache all examples from the all_questions subset.

        Each question is prepended with its category in brackets, e.g.
        ``[HISTORY] For the last 8 years of his life...``

        Questions containing ``<a`` tags (media links) are filtered out
        since they reference audio/visual content unusable by a text-only model.
        Only categories with at least ``MIN_COUNT`` instances (after dedup and
        href filtering) are kept, and ``EXCLUDE_CATEGORIES`` are dropped.
        Valid categories should match ``INCLUDE_CATEGORIES``.
        """
        if self._all_examples is None:
            ds = load_dataset_with_estale_retry(
                hf_load_dataset, "soldni/jeopardy", "all_questions",
                revision=DATASET_REVISION)
            rows = ds["train"].shuffle(seed=17)
            triples = []  # (category, question, answer)
            seen_qs = set()
            for row in rows:
                q = row["question"].strip()
                if "<a " in q.lower():
                    continue
                if q not in seen_qs:
                    seen_qs.add(q)
                    cat = row["ee-category"].strip()
                    ans = row["continuation"].strip()
                    triples.append((cat, q, ans))

            # Filter to categories with enough instances
            cat_counts = Counter(cat for cat, _, _ in triples)
            valid_cats = {cat for cat, n in cat_counts.items()
                         if n >= self.MIN_COUNT and cat not in self.EXCLUDE_CATEGORIES}
            assert valid_cats == self.INCLUDE_CATEGORIES, (
                f"Category mismatch:\n"
                f"  extra: {valid_cats - self.INCLUDE_CATEGORIES}\n"
                f"  missing: {self.INCLUDE_CATEGORIES - valid_cats}"
            )
            self._all_examples = [
                QA(f"[{cat}] {q}", ans)
                for cat, q, ans in triples if cat in valid_cats
            ]
        return self._all_examples

    def _load_icl_examples(self):
        """Load ICL examples from the beginning of the shuffled data."""
        if not self._icl_examples:
            assert self.n_icl <= self.max_n_icl
            self._icl_examples = self._load_all_examples()[:self.n_icl]

    def _load_eval_examples(self) -> List[QA]:
        """Eval examples start after the reserved ICL pool."""
        return self._load_all_examples()[self.max_n_icl:]

    def load_examples(self) -> List[QA]:
        return self._slice_ranges(self._load_eval_examples())

    def num_instances(self) -> int:
        return len(self._load_eval_examples())

    def get_blacklist(self) -> set[int]:
        """Return malformed clues' canonical indices in the evaluation split."""
        self._load_icl_examples()
        icl_blacklist = [example.question for example in self._icl_examples
                         if _is_blacklisted_clue(example.question)]
        assert not icl_blacklist, (
            "Jeopardy blacklist question(s) appear in the ICL set: "
            f"{icl_blacklist!r}")

        blacklist = set()
        for index, example in enumerate(self._load_eval_examples()):
            if _is_blacklisted_clue(example.question):
                blacklist.add(index)
        return blacklist

    def format_generation_prompt(self, question: Optional[str],
                                 is_instruct: bool,
                                 ex_i: int = 0) -> Optional[str]:
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
