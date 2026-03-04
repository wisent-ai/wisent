"""Basic contrastive pair converters for FullBenchmarkDownloader."""

from typing import Any, Dict, List

from wisent.core.utils.config_tools.constants import DISPLAY_TOP_N_TINY


class BasicConvertersMixin:
    """Mixin providing basic format conversion methods."""

    def _convert_mmmlu_format(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert MMMLU format (instruction, option_a/b/c/d, answer)."""
        instruction = sample.get("instruction", "")
        option_a = sample.get("option_a", "")
        option_b = sample.get("option_b", "")
        option_c = sample.get("option_c", "")
        option_d = sample.get("option_d", "")
        answer = sample.get("answer", "")

        # Map answer letter to option
        options = {"A": option_a, "B": option_b, "C": option_c, "D": option_d}

        correct_answer = options.get(answer, option_a)  # Default to A if answer not found

        # Create pairs with each incorrect option
        pairs = []
        for letter, option in options.items():
            if letter != answer and option:
                pairs.append(
                    {
                        "context": instruction,
                        "good_response": correct_answer,
                        "bad_response": option,
                        "metadata": {
                            "answer_key": answer,
                            "sample_id": sample.get("id", ""),
                            "benchmark_type": "mmmlu",
                        },
                    }
                )

        return pairs

    def _convert_multiple_choice_numeric(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert multiple choice with numeric label (HellaSwag, SWAG)."""
        context = sample.get("ctx", sample.get("query", ""))

        # Handle different choice formats
        if "endings" in sample:
            # HellaSwag format: choices in "endings" list
            choices = sample.get("endings", [])
        elif "ending0" in sample:
            # SWAG format: choices in separate ending0, ending1, etc. fields
            choices = []
            for i in range(4):  # SWAG typically has 4 choices
                ending_key = f"ending{i}"
                if ending_key in sample:
                    choices.append(sample[ending_key])
            # Build context from sent1, sent2, etc.
            sent1 = sample.get("sent1", "")
            sent2 = sample.get("sent2", "")
            context = f"{sent1} {sent2}".strip()
        else:
            choices = sample.get("choices", [])

        correct_idx = int(sample["label"])

        if not choices or correct_idx >= len(choices):
            return []

        correct_answer = choices[correct_idx]
        incorrect_answers = [choices[i] for i in range(len(choices)) if i != correct_idx]

        pairs = []
        for incorrect in incorrect_answers:
            pairs.append(
                {
                    "context": context,
                    "good_response": correct_answer,
                    "bad_response": incorrect,
                    "metadata": {
                        "correct_index": correct_idx,
                        "sample_id": sample.get("id", sample.get("ind", "")),
                        "source": sample.get("source", ""),
                    },
                }
            )

        return pairs

    def _convert_multiple_choice_letter(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert multiple choice with letter answerKey (ARC, OpenBookQA)."""
        question = sample.get("question", "")
        choices_text = sample["choices"]["text"]
        choices_labels = sample["choices"]["label"]
        answer_key = sample["answerKey"]

        # Find correct answer
        correct_idx = None
        for i, label in enumerate(choices_labels):
            if label == answer_key:
                correct_idx = i
                break

        if correct_idx is None:
            return []

        correct_answer = choices_text[correct_idx]
        incorrect_answers = [choices_text[i] for i in range(len(choices_text)) if i != correct_idx]

        pairs = []
        for incorrect in incorrect_answers:
            pairs.append(
                {
                    "context": question,
                    "good_response": correct_answer,
                    "bad_response": incorrect,
                    "metadata": {
                        "answer_key": answer_key,
                        "sample_id": sample.get("id", ""),
                        "source": sample.get("source", ""),
                    },
                }
            )

        return pairs

    def _convert_truthfulqa_mc1(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert TruthfulQA MC1 format."""
        question = sample["question"]
        choices = sample["mc1_targets"]["choices"]
        labels = sample["mc1_targets"]["labels"]

        # Find correct and incorrect answers
        correct_answers = [choices[i] for i, label in enumerate(labels) if label == 1]
        incorrect_answers = [choices[i] for i, label in enumerate(labels) if label == 0]

        if not correct_answers or not incorrect_answers:
            return []

        pairs = []
        for correct in correct_answers:
            for incorrect in incorrect_answers[:DISPLAY_TOP_N_TINY]:  # Limit to 3 incorrect per correct
                pairs.append(
                    {
                        "context": question,
                        "good_response": correct,
                        "bad_response": incorrect,
                        "metadata": {"sample_id": sample.get("id", ""), "benchmark_type": "truthfulqa_mc1"},
                    }
                )

        return pairs

    def _convert_truthfulqa_mc2(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert TruthfulQA MC2 format."""
        question = sample["question"]
        choices = sample["mc2_targets"]["choices"]
        labels = sample["mc2_targets"]["labels"]

        correct_answers = [choices[i] for i, label in enumerate(labels) if label == 1]
        incorrect_answers = [choices[i] for i, label in enumerate(labels) if label == 0]

        if not correct_answers or not incorrect_answers:
            return []

        pairs = []
        for correct in correct_answers:
            for incorrect in incorrect_answers[:self.max_incorrect_per_correct]:  # Limit incorrect per correct
                pairs.append(
                    {
                        "context": question,
                        "good_response": correct,
                        "bad_response": incorrect,
                        "metadata": {"sample_id": sample.get("id", ""), "benchmark_type": "truthfulqa_mc2"},
                    }
                )

        return pairs

    def _convert_textual_entailment(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert textual entailment tasks (CB, RTE)."""
        premise = sample["premise"]
        hypothesis = sample["hypothesis"]
        label = sample["label"]

        # Map different label formats
        if isinstance(label, str):
            if label.lower() in ["entailment", "true", "1"]:
                correct_answer = "Yes, this follows logically."
                incorrect_answer = "No, this does not follow logically."
            elif label.lower() in ["contradiction", "false", "0"]:
                correct_answer = "No, this contradicts the premise."
                incorrect_answer = "Yes, this follows logically."
            else:  # neutral
                correct_answer = "This is neither supported nor contradicted."
                incorrect_answer = "Yes, this follows logically."
        else:
            # Numeric labels: typically 0=entailment, 1=neutral, 2=contradiction
            if label == 0:
                correct_answer = "Yes, this follows logically."
                incorrect_answer = "No, this does not follow logically."
            elif label == 2:
                correct_answer = "No, this contradicts the premise."
                incorrect_answer = "Yes, this follows logically."
            else:  # neutral
                correct_answer = "This is neither supported nor contradicted."
                incorrect_answer = "Yes, this follows logically."

        context = f"Premise: {premise}\nHypothesis: {hypothesis}\nDoes the hypothesis follow from the premise?"

        return [
            {
                "context": context,
                "good_response": correct_answer,
                "bad_response": incorrect_answer,
                "metadata": {
                    "sample_id": sample.get("idx", ""),
                    "original_label": label,
                    "benchmark_type": "textual_entailment",
                },
            }
        ]

    def _convert_boolean_question(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert boolean questions (BoolQ)."""
        question = sample.get("question", "")
        passage = sample.get("passage", "")
        label = sample["label"]

        # Determine correct answer
        if str(label).lower() in ["true", "1"]:
            correct_answer = "Yes"
            incorrect_answer = "No"
        else:
            correct_answer = "No"
            incorrect_answer = "Yes"

        context = f"{passage}\n\nQuestion: {question}" if passage else question

        return [
            {
                "context": context,
                "good_response": correct_answer,
                "bad_response": incorrect_answer,
                "metadata": {"sample_id": sample.get("id", ""), "original_label": label, "benchmark_type": "boolean"},
            }
        ]

    def _convert_winogrande_format(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert Winogrande format (sentence, option1, option2, answer)."""
        sentence = sample.get("sentence", "")
        option1 = sample.get("option1", "")
        option2 = sample.get("option2", "")
        answer = sample.get("answer", "")

        if not sentence or not option1 or not option2 or not answer:
            return []

        # Determine correct and incorrect answers
        if answer == "1":
            correct_answer = option1
            incorrect_answer = option2
        elif answer == "2":
            correct_answer = option2
            incorrect_answer = option1
        else:
            # If answer format is unexpected, default to option1 as correct
            correct_answer = option1
            incorrect_answer = option2

        # Create contrastive pair
        return [
            {
                "question": sentence,  # The sentence with blank to fill
                "good_response": correct_answer,
                "bad_response": incorrect_answer,
                "metadata": {
                    "option1": option1,
                    "option2": option2,
                    "answer": answer,
                    "benchmark_type": "winogrande",
                    "task_type": "coreference_resolution",
                },
            }
        ]
