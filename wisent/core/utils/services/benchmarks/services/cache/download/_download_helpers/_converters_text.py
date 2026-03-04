"""Text generation and reading comprehension converters for FullBenchmarkDownloader."""

import random
from typing import Any, Dict, List



class TextConvertersMixin:
    """Mixin providing text generation and reading comprehension conversion methods."""

    def _convert_text_generation(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert text generation tasks (GSM8K, math problems)."""
        question = sample["question"]
        correct_answer = sample["answer"]

        # Generate plausible incorrect answers for math problems
        if any(
            math_keyword in question.lower() for math_keyword in ["dollars", "cost", "price", "how much", "how many"]
        ):
            incorrect_answers = self._generate_math_distractors(correct_answer)
        else:
            # For non-math, create generic incorrect responses
            incorrect_answers = [
                "I don't know the answer to this question.",
                "This question cannot be answered with the given information.",
                "The answer is unclear from the problem statement.",
            ]

        pairs = []
        for incorrect in incorrect_answers:
            pairs.append(
                {
                    "context": question,
                    "good_response": correct_answer,
                    "bad_response": incorrect,
                    "metadata": {"sample_id": sample.get("id", ""), "benchmark_type": "text_generation"},
                }
            )

        return pairs

    def _convert_arithmetic_format(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert arithmetic format (context, completion, _split_origin)."""
        context = sample.get("context", "")
        correct_answer = str(sample.get("completion", "")).strip()

        if not context or not correct_answer:
            return []

        # Extract question from context
        # Context format may be: "Question: What is 8 + 3?\nAnswer:" or just the problem itself
        if "Question:" in context:
            question = context.split("Question:")[1].split("\nAnswer:")[0].strip()
        else:
            question = context.replace("\nAnswer:", "").strip()

        # Generate simple +1 incorrect answer for arithmetic problems
        try:
            if correct_answer.isdigit():
                incorrect_answer = str(int(correct_answer) + 1)
            elif '.' in correct_answer and correct_answer.replace('.', '').replace('-', '').isdigit():
                incorrect_answer = str(float(correct_answer) + 1)
            else:
                incorrect_answer = "Wrong answer"
        except Exception:
            incorrect_answer = "Wrong answer"

        incorrect_answers = [incorrect_answer]

        pairs = []
        for incorrect in incorrect_answers:
            pairs.append(
                {
                    "context": question,
                    "good_response": correct_answer,
                    "bad_response": incorrect,
                    "metadata": {
                        "sample_id": sample.get("id", ""),
                        "benchmark_type": "arithmetic",
                        "split_origin": sample.get("_split_origin", "")
                    },
                }
            )

        return pairs

    def _convert_math500_format(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert MATH-500 format (problem, solution, answer, subject, level)."""
        problem = sample.get("problem", "")
        correct_answer = sample.get("answer", "")
        solution = sample.get("solution", "")
        subject = sample.get("subject", "")
        level = sample.get("level", 0)
        unique_id = sample.get("unique_id", "")

        # Generate mathematical incorrect answers based on correct answer
        incorrect_answers = self._generate_math_distractors(correct_answer)

        pairs = []
        for incorrect in incorrect_answers:
            pairs.append(
                {
                    "context": problem,
                    "good_response": correct_answer,
                    "bad_response": incorrect,
                    "metadata": {
                        "benchmark_type": "math500",
                        "subject": subject,
                        "level": level,
                        "sample_id": unique_id,
                        "has_solution": bool(solution.strip()),
                    },
                }
            )

        return pairs

    def _convert_reading_comprehension(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert reading comprehension tasks (CoQA, SQuAD)."""
        story = sample.get("story", sample.get("passage", ""))

        pairs = []

        # Handle CoQA format with multiple questions
        if "questions" in sample and "answers" in sample:
            questions_data = sample["questions"]
            answers_data = sample["answers"]

            # CoQA format has questions and answers as dicts with lists
            if isinstance(questions_data, dict) and isinstance(answers_data, dict):
                question_texts = questions_data.get("input_text", [])
                answer_texts = answers_data.get("input_text", [])

                for i, (q_text, a_text) in enumerate(zip(question_texts, answer_texts)):
                    context = f"{story}\n\nQuestion: {q_text}"

                    # Generate incorrect answer
                    incorrect_answer = "I cannot find this information in the passage."

                    pairs.append(
                        {
                            "context": context,
                            "good_response": a_text,
                            "bad_response": incorrect_answer,
                            "metadata": {
                                "sample_id": sample.get("id", ""),
                                "question_index": i,
                                "benchmark_type": "reading_comprehension",
                            },
                        }
                    )
            # Handle other formats where questions/answers might be lists directly
            elif isinstance(questions_data, list) and isinstance(answers_data, list):
                for i, (q, a) in enumerate(zip(questions_data, answers_data)):
                    question_text = q.get("input_text", q.get("text", "")) if isinstance(q, dict) else str(q)
                    answer_text = a.get("input_text", a.get("text", "")) if isinstance(a, dict) else str(a)

                    context = f"{story}\n\nQuestion: {question_text}"

                    # Generate incorrect answer
                    incorrect_answer = "I cannot find this information in the passage."

                    pairs.append(
                        {
                            "context": context,
                            "good_response": answer_text,
                            "bad_response": incorrect_answer,
                            "metadata": {
                                "sample_id": sample.get("id", ""),
                                "question_index": i,
                                "benchmark_type": "reading_comprehension",
                            },
                        }
                    )

        return pairs

    def _convert_squad2_format(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert SQuAD2 format (id, title, context, question, answers)."""
        context = sample.get("context", "")
        question = sample.get("question", "")
        answers = sample.get("answers", {})

        if not context or not question:
            return []

        # Handle SQuAD2 answer format
        answer_text = ""
        if isinstance(answers, dict):
            answer_texts = answers.get("text", [])
            if answer_texts and len(answer_texts) > 0:
                answer_text = answer_texts[0]
        elif isinstance(answers, list) and len(answers) > 0:
            if isinstance(answers[0], dict):
                answer_text = answers[0].get("text", "")
            else:
                answer_text = str(answers[0])

        if not answer_text:
            # For unanswerable questions in SQuAD2, create a pair with empty answer
            answer_text = "[No answer available]"

        # Create a contrastive pair using question-answering format
        return [
            {
                "question": f"Context: {context}\n\nQuestion: {question}",
                "good_response": answer_text,
                "bad_response": "[Incorrect answer]",
                "metadata": {
                    "id": sample.get("id", ""),
                    "title": sample.get("title", ""),
                    "benchmark_type": "squad2",
                    "task_type": "reading_comprehension",
                },
            }
        ]

    def _convert_generic_multiple_choice(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generic catch-all for multiple choice formats."""
        question = sample.get("question", sample.get("query", ""))
        choices = sample.get("choices", [])

        if len(choices) < 2:
            return []

        # Assume first choice is correct (this is a catch-all)
        correct_answer = choices[0]
        incorrect_answers = choices[1:]

        pairs = []
        for incorrect in incorrect_answers:
            pairs.append(
                {
                    "context": question,
                    "good_response": correct_answer,
                    "bad_response": incorrect,
                    "metadata": {
                        "sample_id": sample.get("id", ""),
                        "benchmark_type": "generic_multiple_choice",
                        "warning": "Assumed first choice is correct",
                    },
                }
            )

        return pairs

    def _convert_webqs_format(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert WebQS format (question, answers list)."""
        question = sample.get("question", "")
        answers = sample.get("answers", [])

        if not question or not answers:
            return []

        # Take the first answer as the correct one
        correct_answer = answers[0] if answers else ""

        if not correct_answer:
            return []

        # Generate incorrect answers (simple approach)
        incorrect_answers = []

        # Strategy 1: Use other answers from the same dataset if available
        if len(answers) > 1:
            incorrect_answers.extend(answers[1:self.max_incorrect_per_correct + 1])

        # Strategy 2: Generate simple incorrect answers
        if len(incorrect_answers) < 2:
            incorrect_answers.append("Unknown")
            incorrect_answers.append("No information available")

        # Create contrastive pairs
        pairs = []
        for incorrect in incorrect_answers[:self.max_incorrect_per_correct]:  # Limit pairs
            pairs.append(
                {
                    "question": question,
                    "good_response": correct_answer,
                    "bad_response": incorrect,
                    "metadata": {"benchmark_type": "webqs", "task_type": "factual_qa", "url": sample.get("url", "")},
                }
            )

        return pairs
