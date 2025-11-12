"""Extractor for TAG-Bench (Table-Augmented Generation) benchmark."""
from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

__all__ = ["TagExtractor"]


class TagExtractor(HuggingFaceBenchmarkExtractor):
    """Extractor for TAG-Bench benchmark.

    TAG-Bench evaluates Table-Augmented Generation: answering natural language
    questions over databases. The benchmark contains 80 queries across different
    database domains.
    """

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """Extract contrastive pairs from TAG-Bench CSV file.

        Args:
            limit: Optional maximum number of pairs to extract

        Returns:
            List of ContrastivePair objects
        """
        # Load the CSV file
        csv_path = Path(__file__).parents[5] / "data" / "tag_queries.csv"

        if not csv_path.exists():
            raise FileNotFoundError(
                f"TAG-Bench data not found at {csv_path}. "
                "Please download from https://github.com/TAG-Research/TAG-Bench"
            )

        pairs: list[ContrastivePair] = []
        all_answers: list[str] = []

        # First pass: collect all answers for negative sampling
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                answer = str(row.get('Answer', '')).strip()
                if answer:
                    all_answers.append(answer)

        # Second pass: create contrastive pairs
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if limit is not None and len(pairs) >= limit:
                    break

                query = str(row.get('Query', '')).strip()
                answer = str(row.get('Answer', '')).strip()
                db = str(row.get('DB used', '')).strip()
                query_type = str(row.get('Query type', '')).strip()

                if not query or not answer:
                    continue

                # Create prompt with database context
                prompt = f"Database: {db}\nQuery: {query}\nAnswer:"

                # Generate negative answer by sampling a different answer
                negative_candidates = [a for a in all_answers if a != answer]
                if negative_candidates:
                    negative_answer = random.choice(negative_candidates)
                else:
                    negative_answer = "unknown"

                # Create contrastive pair
                positive_response = PositiveResponse(model_response=answer)
                negative_response = NegativeResponse(model_response=negative_answer)

                pair = ContrastivePair(
                    prompt=prompt,
                    positive_response=positive_response,
                    negative_response=negative_response,
                    label="tag",
                    metadata={
                        "db": db,
                        "query_type": query_type,
                        "query_id": row.get('Query ID', str(i)),
                    }
                )
                pairs.append(pair)

        return pairs
