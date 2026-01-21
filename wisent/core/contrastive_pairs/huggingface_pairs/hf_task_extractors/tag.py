"""Extractor for TAG-Bench (Table-Augmented Generation) benchmark."""
from __future__ import annotations

import csv
import io
import random
import requests
from pathlib import Path
from typing import Any

from wisent.core.cli_logger import setup_logger
from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

__all__ = ["TagExtractor"]

log = setup_logger(__name__)

# GitHub raw URL for TAG-Bench queries
TAG_GITHUB_URL = "https://raw.githubusercontent.com/TAG-Research/TAG-Bench/main/tag_queries.csv"


class TagExtractor(HuggingFaceBenchmarkExtractor):
    """Extractor for TAG-Bench benchmark.

    TAG-Bench evaluates Table-Augmented Generation: answering natural language
    questions over databases. The benchmark contains 80 queries across different
    database domains.
    """

    evaluator_name = "table_qa"

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
        # Try local file first, then download from GitHub
        csv_path = Path(__file__).parents[5] / "data" / "tag_queries.csv"

        if csv_path.exists():
            log.info(f"Loading TAG-Bench from local file: {csv_path}")
            csv_content = csv_path.read_text(encoding='utf-8')
        else:
            log.info("Downloading TAG-Bench from GitHub...")
            csv_content = self._download_from_github()
            if not csv_content:
                return []

        pairs: list[ContrastivePair] = []
        all_answers: list[str] = []

        # Parse CSV content
        reader = csv.DictReader(io.StringIO(csv_content))
        rows = list(reader)

        # First pass: collect all answers for negative sampling
        for row in rows:
            answer = str(row.get('Answer', '')).strip()
            if answer:
                all_answers.append(answer)

        log.info(f"Loaded {len(rows)} TAG-Bench queries")

        # Second pass: create contrastive pairs
        for i, row in enumerate(rows):
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

    def _download_from_github(self) -> str:
        """Download TAG-Bench CSV from GitHub."""
        try:
            response = requests.get(TAG_GITHUB_URL, timeout=30)
            response.raise_for_status()
            return response.text
        except Exception as e:
            log.error(f"Failed to download TAG-Bench from GitHub: {e}")
            return ""
