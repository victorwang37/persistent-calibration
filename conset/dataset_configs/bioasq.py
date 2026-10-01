"""BioASQ dataset configuration (factoid and list questions)"""

import json
import random
from dataclasses import dataclass, field
from typing import List, Optional

from .base import JudgePromptFn, QA, DatasetConfig, extract_between

# Downloaded from https://participants-area.bioasq.org/datasets/
DATA_PATH = "datasets/bioasq_training14b.json"


@dataclass
class BioASQConfig(DatasetConfig):
    """Configuration for the BioASQ dataset (factoid + list questions).
    Use list questions with up to max_n_list_ans answers.
    """

    name: str = "bioasq"
    n_icl: int = 5
    max_n_icl: int = 5
    max_n_list_ans: int = 10
    _all_examples: Optional[List[QA]] = field(default=None, init=False, repr=False)
    _icl_factoid: List[QA] = field(default_factory=list, init=False, repr=False)
    _icl_list: List[QA] = field(default_factory=list, init=False, repr=False)
    _question_types: List[str] = field(default_factory=list, init=False, repr=False)
    _judge_prompt_fns: List = field(default_factory=list, init=False, repr=False)

    LIST_SUFFIX = " Just output one of the answers."

    def _load_all_examples(self) -> List[QA]:
        if self._all_examples is None:
            from conset.judge import make_standard_judge_prompt, make_multi_ans_judge_prompt
            with open(DATA_PATH) as f:
                data = json.load(f)
            instances = data["questions"]
            indices = list(range(len(instances)))
            random.Random(17).shuffle(indices)

            seen_qs = set()
            eval_examples: List[QA] = []
            for i in indices:
                instance = instances[i]
                qst = instance["body"].strip()
                if qst in seen_qs:
                    continue
                seen_qs.add(qst)

                qtype = instance["type"]
                if qtype == "factoid":
                    ea = instance["exact_answer"]
                    assert isinstance(ea, list) and isinstance(ea[0], str), (
                        f"Unexpected factoid exact_answer format: {ea!r}")
                    if not qst[-1] in ".?!" and "?" not in qst:
                        qst += "?"
                    if len(self._icl_factoid) < self.n_icl:
                        self._icl_factoid.append(QA(qst, ea[0].strip()))
                    elif len(ea) == 1:
                        eval_examples.append(QA(qst, ea[0].strip()))
                        self._judge_prompt_fns.append(make_standard_judge_prompt)
                        self._question_types.append(qtype)
                    else:
                        eval_examples.append(QA(qst, str(ea)))
                        self._judge_prompt_fns.append(make_multi_ans_judge_prompt)
                        self._question_types.append(qtype)
                elif qtype == "list":
                    ea = instance["exact_answer"]
                    assert (isinstance(ea, list) and isinstance(ea[0], list)
                            and isinstance(ea[0][0], str)), (
                        f"Unexpected list exact_answer format: {ea!r}")
                    if len(ea) > self.max_n_list_ans:
                        continue
                    if not qst[-1] in ".?!":
                        qst += "."
                    full_qst = qst + self.LIST_SUFFIX
                    if len(self._icl_list) < self.n_icl:
                        self._icl_list.append(QA(full_qst, ea[0][0].strip()))
                    else:
                        eval_examples.append(QA(full_qst, str(ea)))
                        self._judge_prompt_fns.append(make_multi_ans_judge_prompt)
                        self._question_types.append(qtype)

            self._all_examples = eval_examples
        return self._all_examples

    def load_examples(self) -> List[QA]:
        return self._slice_ranges(self._load_all_examples())

    def num_instances(self) -> int:
        return len(self._load_all_examples())

    def format_generation_prompt(self, question: Optional[str],
                                 is_instruct: bool,
                                 ex_i: int = 0) -> Optional[str]:
        if question is None:
            return None

        if self.n_icl > 0:
            self._load_all_examples()  # ensure ICL pools are populated
            abs_i = self.resolve_ex_i(self.ranges, ex_i)
            icl_examples = (self._icl_list if self._question_types[abs_i] == "list"
                            else self._icl_factoid)
            icl_parts = [f"Prompt: {ex.question}\nAnswer: {ex.answer}"
                         for ex in icl_examples]
            icl_block = "\n\n".join(icl_parts)

            if is_instruct:
                prompt = (
                    f"Here are {len(icl_examples)} sets of example prompt and answer.\n\n"
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

    def get_judge_prompt_fns(self) -> list[JudgePromptFn]:
        self._load_all_examples()
        return self._slice_ranges(self._judge_prompt_fns)
