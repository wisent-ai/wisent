"""QA converters and sample dispatcher for FullBenchmarkDownloader."""

from typing import Any, Dict, List

from wisent.core.constants import MIN_PAGE_TEXT_LENGTH, MIN_SENTENCE_LENGTH, DISPLAY_TRUNCATION_COMPACT, MAX_INCORRECT_PER_CORRECT


class QAConvertersMixin:
    """Mixin providing QA conversion methods and the sample-to-pairs dispatcher."""

    def _convert_sample_to_pairs(self, sample: Dict[str, Any], benchmark_name: str) -> List[Dict[str, Any]]:
        """Convert a single sample to contrastive pairs based on benchmark type."""

        # MMMLU format (instruction, option_a, option_b, option_c, option_d, answer)
        if "instruction" in sample and "option_a" in sample and "answer" in sample:
            return self._convert_mmmlu_format(sample)

        # Multiple Choice with explicit choices and numeric label (HellaSwag, SWAG, etc.)
        if ("endings" in sample and "label" in sample) or ("ending0" in sample and "label" in sample):
            return self._convert_multiple_choice_numeric(sample)

        # Multiple Choice with choices dict and answerKey (ARC, OpenBookQA, etc.)
        if "choices" in sample and "answerKey" in sample:
            return self._convert_multiple_choice_letter(sample)

        # TruthfulQA MC1 format
        if "mc1_targets" in sample:
            return self._convert_truthfulqa_mc1(sample)

        # TruthfulQA MC2 format
        if "mc2_targets" in sample:
            return self._convert_truthfulqa_mc2(sample)

        # SQuAD2 format (id, title, context, question, answers)
        if "context" in sample and "question" in sample and "answers" in sample:
            return self._convert_squad2_format(sample)

        # Textual entailment (premise/hypothesis format like CB, RTE)
        if "premise" in sample and "hypothesis" in sample:
            return self._convert_textual_entailment(sample)

        # Boolean questions (BoolQ)
        if "label" in sample and str(sample["label"]).lower() in ["true", "false", "0", "1"]:
            return self._convert_boolean_question(sample)

        # MBPP format (programming problems with code)
        if "task_id" in sample and "text" in sample and "code" in sample:
            return self._convert_mbpp_format(sample)

        # MATH-500 format (problem, solution, answer, subject, level)
        if (
            "problem" in sample
            and "solution" in sample
            and "answer" in sample
            and "subject" in sample
            and "level" in sample
        ):
            return self._convert_math500_format(sample)

        # WebQS format (question, answers list)
        if "question" in sample and "answers" in sample and isinstance(sample.get("answers"), list):
            return self._convert_webqs_format(sample)

        # NaturalQS format (question, answer as list)
        if "question" in sample and "answer" in sample and isinstance(sample.get("answer"), list):
            return self._convert_naturalqs_format(sample)

        # TriviaQA format (question, answer as dict with aliases)
        if "question" in sample and "answer" in sample and isinstance(sample.get("answer"), dict):
            return self._convert_triviaqa_format(sample)

        # Text generation with question/answer (GSM8K, math problems)
        if "question" in sample and "answer" in sample:
            return self._convert_text_generation(sample)

        # Reading comprehension (CoQA, SQuAD)
        if "story" in sample or "passage" in sample:
            return self._convert_reading_comprehension(sample)

        # SQuAD2 format (id, title, context, question, answers)
        if (
            "id" in sample
            and "title" in sample
            and "context" in sample
            and "question" in sample
            and "answers" in sample
        ):
            return self._convert_squad2_format(sample)

        # Winogrande format (sentence, option1, option2, answer)
        if "sentence" in sample and "option1" in sample and "option2" in sample and "answer" in sample:
            return self._convert_winogrande_format(sample)

        # WikiText format (page)
        if "page" in sample:
            return self._convert_wikitext_format(sample)

        # GPQA format (Question, choice1-4, answer, plus rich metadata)
        if (
            "Question" in sample
            and "choice1" in sample
            and "choice2" in sample
            and "choice3" in sample
            and "choice4" in sample
            and "answer" in sample
        ):
            return self._convert_gpqa_format(sample)

        # HLE format (question, answer, answer_type, category)
        if "question" in sample and "answer" in sample and "answer_type" in sample and "category" in sample:
            return self._convert_hle_format(sample)

        # HumanEval code generation format
        if "task_id" in sample and "canonical_solution" in sample and "prompt" in sample and "test" in sample:
            return self._convert_humaneval_format(sample)

        # MBPP code generation format (task_id, code, prompt, test)
        if "task_id" in sample and "code" in sample and "prompt" in sample and "test" in sample:
            return self._convert_mbpp_format(sample)

        # Arithmetic format (context, completion, _split_origin)
        if "context" in sample and "completion" in sample and "_split_origin" in sample:
            return self._convert_arithmetic_format(sample)

        # Generic multiple choice (catch-all)
        if "choices" in sample:
            return self._convert_generic_multiple_choice(sample)

        print(f"         Warning: Unknown sample format: {list(sample.keys())}")
        return []

    def _convert_wikitext_format(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert WikiText format (page)."""
        page = sample.get("page", "")

        if not page or len(page.strip()) < MIN_PAGE_TEXT_LENGTH:  # Skip very short pages
            return []

        # For WikiText, we create language modeling pairs
        # Split the page into sentences and create good/corrupted pairs
        sentences = page.split(". ")
        if len(sentences) < 2:
            return []

        pairs = []
        for i, sentence in enumerate(sentences):
            if len(sentence.strip()) > MIN_SENTENCE_LENGTH:  # Only use substantial sentences
                # Create a corrupted version by replacing some words
                words = sentence.split()
                if len(words) > 3:
                    # Simple corruption: duplicate a word in the middle
                    mid_idx = len(words) // 2
                    corrupted_words = words.copy()
                    corrupted_words.insert(mid_idx, words[mid_idx])
                    corrupted_sentence = " ".join(corrupted_words)

                    pairs.append(
                        {
                            "question": "Complete the text naturally:",
                            "good_response": sentence.strip(),
                            "bad_response": corrupted_sentence,
                            "metadata": {
                                "benchmark_type": "wikitext",
                                "task_type": "language_modeling",
                                "sentence_index": i,
                            },
                        }
                    )

                    # Limit to 3 pairs per page to avoid too many
                    if len(pairs) >= 3:
                        break

        return pairs

    def _convert_naturalqs_format(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert NaturalQS format (question, answer as list)."""
        question = sample.get("question", "")
        answer_list = sample.get("answer", [])

        if not question or not answer_list:
            return []

        # Take the first answer as the correct one (shortest/most direct)
        correct_answer = answer_list[0] if answer_list else ""

        if not correct_answer:
            return []

        # Generate incorrect answers
        incorrect_answers = []

        # Strategy 1: Use other answers from the list as distractors if available
        if len(answer_list) > 1:
            incorrect_answers.extend(answer_list[1:MAX_INCORRECT_PER_CORRECT + 1])

        # Strategy 2: Generate generic incorrect answers
        if len(incorrect_answers) < 2:
            incorrect_answers.append("I don't know the answer to this question.")
            incorrect_answers.append("This information is not available.")

        # Create contrastive pairs
        pairs = []
        for incorrect in incorrect_answers[:MAX_INCORRECT_PER_CORRECT]:  # Limit pairs
            pairs.append(
                {
                    "context": question,
                    "good_response": correct_answer,
                    "bad_response": incorrect,
                    "metadata": {
                        "benchmark_type": "naturalqs",
                        "task_type": "factual_qa",
                        "total_answers": len(answer_list),
                    },
                }
            )

        return pairs

    def _convert_triviaqa_format(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert TriviaQA format (question, answer as dict with aliases)."""
        question = sample.get("question", "")
        answer_dict = sample.get("answer", {})

        if not question or not answer_dict:
            return []

        # Extract the correct answer from aliases
        aliases = answer_dict.get("aliases", [])
        if not aliases:
            # Use other fields
            correct_answer = (
                answer_dict.get("value", "") or answer_dict.get("normalized_value", "") or str(answer_dict)
            )[:DISPLAY_TRUNCATION_COMPACT]  # Truncate if too long
        else:
            correct_answer = aliases[0]  # Use first alias as primary answer

        if not correct_answer:
            return []

        # Generate incorrect answers
        incorrect_answers = []

        # Strategy 1: Use other aliases as distractors if available
        if len(aliases) > 1:
            incorrect_answers.extend(aliases[1:MAX_INCORRECT_PER_CORRECT + 1])

        # Strategy 2: Generate generic incorrect answers for trivia
        if len(incorrect_answers) < 2:
            incorrect_answers.append("Unknown")
            incorrect_answers.append("I don't know")

        # Create contrastive pairs
        pairs = []
        for incorrect in incorrect_answers[:MAX_INCORRECT_PER_CORRECT]:  # Limit pairs
            pairs.append(
                {
                    "context": question,
                    "good_response": correct_answer,
                    "bad_response": incorrect,
                    "metadata": {
                        "benchmark_type": "triviaqa",
                        "task_type": "trivia_qa",
                        "total_aliases": len(aliases),
                        "entity_name": answer_dict.get("matched_wiki_entity_name", ""),
                    },
                }
            )

        return pairs
