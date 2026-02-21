"""Benchmark relevance scoring and similarity methods for ClassifierCreator."""

from typing import Dict, List


class ScoringMixin:
    """Mixin providing benchmark relevance scoring and similarity calculation methods."""

    def _calculate_benchmark_relevance(self, issue_type: str, benchmark_concepts: Dict[str, List[str]]) -> List[str]:
        """Calculate relevance scores using semantic similarity."""
        # Calculate relevance scores
        benchmark_scores = []

        for benchmark_name, concepts in benchmark_concepts.items():
            score = self._calculate_semantic_similarity(issue_type, benchmark_name, concepts)

            if score > 0:
                benchmark_scores.append((benchmark_name, score))

        # Sort by relevance score
        benchmark_scores.sort(key=lambda x: x[1], reverse=True)

        return [name for name, score in benchmark_scores]

    def _calculate_semantic_similarity(self, issue_type: str, benchmark_name: str, concepts: List[str]) -> float:
        """Calculate semantic similarity between issue type and benchmark."""
        issue_lower = issue_type.lower()
        benchmark_lower = benchmark_name.lower()

        score = 0.0

        # Direct name matching (highest weight)
        if issue_lower in benchmark_lower or benchmark_lower in issue_lower:
            score += 5.0

        # Concept matching
        for concept in concepts:
            concept_lower = concept.lower()

            # Exact concept match
            if issue_lower == concept_lower:
                score += 4.0
            # Partial concept match
            elif issue_lower in concept_lower or concept_lower in issue_lower:
                score += 2.0
            # Semantic similarity check
            elif self._are_semantically_similar(issue_lower, concept_lower):
                score += 1.5

        # Token-level similarity in benchmark name
        benchmark_tokens = benchmark_lower.replace("_", " ").replace("-", " ").split()
        issue_tokens = issue_lower.replace("_", " ").replace("-", " ").split()

        for issue_token in issue_tokens:
            for benchmark_token in benchmark_tokens:
                if len(issue_token) > 2 and len(benchmark_token) > 2:
                    if issue_token == benchmark_token:
                        score += 3.0
                    elif issue_token in benchmark_token or benchmark_token in issue_token:
                        score += 1.0
                    elif self._are_semantically_similar(issue_token, benchmark_token):
                        score += 0.5

        return score

    def _are_semantically_similar(self, term1: str, term2: str) -> bool:
        """Check if two terms are semantically similar using algorithmic methods."""
        if len(term1) < 3 or len(term2) < 3:
            return False

        # Character-level similarity
        overlap = len(set(term1) & set(term2))
        min_len = min(len(term1), len(term2))
        char_similarity = overlap / min_len

        # Substring similarity
        longer, shorter = (term1, term2) if len(term1) > len(term2) else (term2, term1)
        substring_match = shorter in longer

        # Prefix/suffix similarity
        prefix_len = 0
        suffix_len = 0

        for i in range(min(len(term1), len(term2))):
            if term1[i] == term2[i]:
                prefix_len += 1
            else:
                break

        for i in range(1, min(len(term1), len(term2)) + 1):
            if term1[-i] == term2[-i]:
                suffix_len += 1
            else:
                break

        affix_similarity = (prefix_len + suffix_len) / max(len(term1), len(term2))

        # Combined similarity score
        return char_similarity > 0.6 or substring_match or affix_similarity > 0.4 or prefix_len >= 3 or suffix_len >= 3

    def _prioritize_benchmarks(self, relevant_benchmarks: List[str]) -> List[str]:
        """Prioritize benchmarks algorithmically based on naming patterns and characteristics."""
        benchmark_scores = []

        for benchmark in relevant_benchmarks:
            score = self._calculate_benchmark_quality_score(benchmark)
            benchmark_scores.append((benchmark, score))

        # Sort by quality score (higher is better)
        benchmark_scores.sort(key=lambda x: x[1], reverse=True)
        return [benchmark for benchmark, score in benchmark_scores]

    def _calculate_benchmark_quality_score(self, benchmark_name: str) -> float:
        """Calculate quality score for a benchmark based on naming patterns and characteristics."""
        score = 0.0
        benchmark_lower = benchmark_name.lower()

        # Length heuristic - moderate length names tend to be well-established
        name_length = len(benchmark_name)
        if 8 <= name_length <= 25:
            score += 2.0
        elif name_length < 8:
            score += 0.5  # Very short names might be too simple
        else:
            score += 1.0  # Very long names might be overly specific

        # Component analysis
        parts = benchmark_lower.split("_")
        num_parts = len(parts)

        # Well-structured benchmarks often have 2-3 parts
        if 2 <= num_parts <= 3:
            score += 2.0
        elif num_parts == 1:
            score += 1.5  # Simple names can be good too
        else:
            score += 0.5  # Too many parts might indicate over-specification

        # Indicator of established benchmarks (avoid hardcoding specific names)
        quality_indicators = [
            # Multiple choice indicators (often well-validated)
            ("mc1", 1.5),
            ("mc2", 1.5),
            ("multiple_choice", 1.5),
            # Evaluation methodology indicators
            ("eval", 1.0),
            ("test", 1.0),
            ("benchmark", 1.0),
            # Language understanding indicators
            ("language", 1.0),
            ("understanding", 1.0),
            ("comprehension", 1.0),
            # Logic and reasoning indicators
            ("logic", 1.0),
            ("reasoning", 1.0),
            ("deduction", 1.0),
            # Knowledge assessment indicators
            ("knowledge", 1.0),
            ("question", 1.0),
            ("answer", 1.0),
        ]

        for indicator, points in quality_indicators:
            if indicator in benchmark_lower:
                score += points

        # Penalize very specialized or experimental indicators
        experimental_indicators = [
            "experimental",
            "pilot",
            "demo",
            "sample",
            "tiny",
            "mini",
            "subset",
            "light",
            "debug",
            "test_only",
        ]

        for indicator in experimental_indicators:
            if indicator in benchmark_lower:
                score -= 1.0

        # Bonus for domain diversity indicators
        domain_indicators = ["multilingual", "global", "cross", "multi", "diverse"]

        for indicator in domain_indicators:
            if indicator in benchmark_lower:
                score += 0.5

        return max(0.0, score)  # Ensure non-negative score
