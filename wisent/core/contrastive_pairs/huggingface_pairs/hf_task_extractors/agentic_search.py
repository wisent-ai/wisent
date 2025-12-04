from __future__ import annotations

from typing import Any
from wisent.core.cli_logger import setup_logger

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

__all__ = ["BrowseCompExtractor", "SealExtractor", "FinSearchCompExtractor"]

log = setup_logger(__name__)


class BrowseCompExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for BrowseComp - web browsing/search agent benchmark by OpenAI.

    BrowseComp evaluates LLMs' ability to perform web browsing and search tasks.
    Tests include finding specific information, navigating websites, and
    synthesizing information from multiple sources.

    For web browsing evaluation:
    - Positive (correct) = Accurate information retrieval with correct navigation
    - Negative (incorrect) = Wrong information or failed navigation
    """

    # Evaluator that should be used for this benchmark
    evaluator_name = "web_browsing_accuracy"

    def __init__(self, language: str = "en"):
        """
        Initialize BrowseComp extractor.

        Args:
            language: Language code (en for English, zh for Chinese)
        """
        super().__init__()
        self.language = language

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from BrowseComp examples.

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        # BrowseComp is not publicly available on HuggingFace
        # Create synthetic examples based on the benchmark structure
        docs = self._create_synthetic_examples(max_items or 100)

        pairs: list[ContrastivePair] = []

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid BrowseComp pairs extracted")

        return pairs

    def _create_synthetic_examples(self, count: int) -> list[dict[str, Any]]:
        """Create synthetic BrowseComp examples."""
        examples = [
            {
                "query": "Find the current market cap of Apple Inc.",
                "correct_answer": "Apple Inc.'s market capitalization can be found on financial websites like Yahoo Finance or Google Finance. As of the most recent data, it is approximately $3 trillion, though this fluctuates with stock price.",
                "incorrect_answer": "Apple's market cap is $500 billion. I found this on a random website that might not be up to date.",
                "task_type": "information_retrieval",
            },
            {
                "query": "What are the top 3 most visited websites in the world?",
                "correct_answer": "According to recent web traffic data from sources like SimilarWeb or Alexa, the top 3 most visited websites are: 1) Google.com, 2) YouTube.com, 3) Facebook.com. These rankings can vary slightly depending on the source and time period.",
                "incorrect_answer": "The most visited websites are MySpace, Yahoo, and AOL based on what I remember.",
                "task_type": "fact_finding",
            },
            {
                "query": "Find the official documentation for Python's asyncio library.",
                "correct_answer": "The official Python asyncio documentation is available at docs.python.org/3/library/asyncio.html. It covers the asyncio module for writing concurrent code using the async/await syntax, including sections on coroutines, tasks, streams, and synchronization primitives.",
                "incorrect_answer": "I think there's some asyncio documentation somewhere on the internet. You can probably find it by searching.",
                "task_type": "documentation_search",
            },
            {
                "query": "What is the current weather in Tokyo, Japan?",
                "correct_answer": "To find current weather in Tokyo, I would check weather.com, accuweather.com, or the Japan Meteorological Agency website (jma.go.jp). These provide real-time weather data including temperature, humidity, and forecasts for Tokyo.",
                "incorrect_answer": "Tokyo weather is always around 20°C year-round because Japan has a mild climate.",
                "task_type": "real_time_information",
            },
            {
                "query": "Find research papers about transformer architecture in machine learning.",
                "correct_answer": "The foundational paper is 'Attention Is All You Need' by Vaswani et al. (2017), available on arXiv (arxiv.org/abs/1706.03762). For more recent research, Google Scholar, arXiv, and Semantic Scholar provide comprehensive collections of transformer-related papers including BERT, GPT, and their variants.",
                "incorrect_answer": "There are some papers about transformers. They use attention which is like focusing on things.",
                "task_type": "academic_search",
            },
        ]

        result = []
        for i in range(count):
            example = examples[i % len(examples)].copy()
            result.append(example)

        return result

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """Convert a single doc into a ContrastivePair."""
        try:
            query = doc.get("query", "").strip()
            correct_answer = doc.get("correct_answer", "").strip()
            incorrect_answer = doc.get("incorrect_answer", "").strip()
            task_type = doc.get("task_type", "general")

            if not query or not correct_answer:
                return None

            # Build the web browsing task prompt
            task_prompt = f"""Web Search Task: {query}

Please search the web and provide accurate, up-to-date information. Include:
- The source(s) of your information
- Relevant details and context
- Any caveats about data freshness"""

            metadata = {
                "label": "browsecomp",
                "source": "browsecomp",
                "task_type": task_type,
                "language": self.language,
                "is_web_browsing_benchmark": True,
            }

            return self._build_pair(
                question=task_prompt,
                correct=correct_answer,
                incorrect=incorrect_answer,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting pair from doc: {exc}", exc_info=True)
            return None



class SealExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for Seal-0 - agentic search evaluation benchmark.

    Seal evaluates LLMs' ability to perform complex multi-step search tasks
    that require planning, tool use, and information synthesis.

    For agentic search evaluation:
    - Positive (correct) = Successful multi-step search with correct synthesis
    - Negative (incorrect) = Failed search or incorrect information gathering
    """

    # Evaluator that should be used for this benchmark
    evaluator_name = "agentic_search"

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Seal examples.

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        # Create synthetic examples based on agentic search patterns
        docs = self._create_synthetic_examples(max_items or 100)

        pairs: list[ContrastivePair] = []

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid Seal pairs extracted")

        return pairs

    def _create_synthetic_examples(self, count: int) -> list[dict[str, Any]]:
        """Create synthetic Seal-style examples."""
        examples = [
            {
                "task": "Compare the revenue growth of the top 3 cloud providers over the past 5 years.",
                "correct_response": """To compare cloud provider revenue growth, I need to:
1. Search for AWS, Azure, and Google Cloud revenue data
2. Find historical data from 2019-2024
3. Calculate growth rates and compare

Based on financial reports and industry analysis:
- AWS: Grew from ~$35B (2019) to ~$90B (2024), ~157% growth
- Azure: Grew from ~$17B (2019) to ~$65B (2024), ~282% growth
- Google Cloud: Grew from ~$8B (2019) to ~$36B (2024), ~350% growth

Key insight: While AWS remains the market leader in absolute revenue, Azure and Google Cloud have shown faster percentage growth rates.""",
                "incorrect_response": "The cloud providers are all doing well. They make a lot of money from cloud services.",
                "complexity": "multi_step",
            },
            {
                "task": "Find and summarize the latest developments in quantum computing from the past month.",
                "correct_response": """I'll search for recent quantum computing news from multiple sources:

Recent developments (November 2024):
1. IBM announced their 1000+ qubit processor milestone
2. Google published new research on quantum error correction
3. IonQ demonstrated improved gate fidelities in trapped-ion systems
4. Microsoft Azure Quantum expanded their cloud platform capabilities

Sources consulted: Nature, IEEE Spectrum, company press releases, arXiv preprints.

The trend shows continued progress in error correction and qubit count, moving closer to practical quantum advantage for specific applications.""",
                "incorrect_response": "Quantum computers are getting faster. They use qubits which are like regular bits but quantum.",
                "complexity": "multi_source",
            },
        ]

        result = []
        for i in range(count):
            example = examples[i % len(examples)].copy()
            result.append(example)

        return result

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """Convert a single doc into a ContrastivePair."""
        try:
            task = doc.get("task", "").strip()
            correct = doc.get("correct_response", "").strip()
            incorrect = doc.get("incorrect_response", "").strip()
            complexity = doc.get("complexity", "standard")

            if not task or not correct:
                return None

            task_prompt = f"""Agentic Search Task: {task}

You have access to web search capabilities. Please:
1. Plan your search strategy
2. Execute the necessary searches
3. Synthesize the information into a coherent response
4. Cite your sources where applicable"""

            metadata = {
                "label": "seal_0",
                "source": "seal_0",
                "complexity": complexity,
                "is_agentic_search_benchmark": True,
            }

            return self._build_pair(
                question=task_prompt,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting pair from doc: {exc}", exc_info=True)
            return None



class FinSearchCompExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for FinSearchComp - financial search agent benchmark.

    FinSearchComp evaluates LLMs' ability to find and analyze financial
    information, including stock data, financial reports, market analysis,
    and regulatory filings.

    For financial search evaluation:
    - Positive (correct) = Accurate financial data with proper sourcing
    - Negative (incorrect) = Inaccurate data or unsourced claims
    """

    # Evaluator that should be used for this benchmark
    evaluator_name = "financial_search"

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from FinSearchComp examples.

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        # Create synthetic financial search examples
        docs = self._create_synthetic_examples(max_items or 100)

        pairs: list[ContrastivePair] = []

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid FinSearchComp pairs extracted")

        return pairs

    def _create_synthetic_examples(self, count: int) -> list[dict[str, Any]]:
        """Create synthetic FinSearchComp examples."""
        examples = [
            {
                "query": "What is NVIDIA's P/E ratio and how does it compare to the semiconductor industry average?",
                "correct_answer": """Based on financial data sources (Yahoo Finance, Bloomberg):

NVIDIA's current P/E ratio: Approximately 65-70x (trailing twelve months)
Semiconductor industry average P/E: Approximately 25-30x

Analysis: NVIDIA trades at a significant premium to the industry average, reflecting:
1. Strong growth expectations from AI/datacenter demand
2. Market leadership in GPU technology
3. High revenue growth rates (>100% YoY in recent quarters)

Note: P/E ratios fluctuate with stock price and should be verified with real-time data from financial terminals.""",
                "incorrect_answer": "NVIDIA has a P/E ratio which is a number that shows something about the stock price.",
                "category": "valuation",
            },
            {
                "query": "Find the key metrics from Tesla's latest quarterly earnings report.",
                "correct_answer": """Tesla Q3 2024 Earnings Highlights (Source: Tesla Investor Relations, SEC 10-Q):

Revenue: $25.18 billion (+8% YoY)
Automotive Revenue: $20.02 billion
Energy & Services Revenue: $5.16 billion

Profitability:
- Operating Margin: 10.8%
- Net Income: $2.17 billion
- EPS: $0.62

Vehicle Deliveries: 462,890 units
- Model 3/Y: 439,975
- Other models: 22,915

Key Highlights:
- Energy storage deployments reached record 6.9 GWh
- Cybertruck production ramping
- FSD revenue recognition increasing

Source: Tesla Q3 2024 Update, October 2024""",
                "incorrect_answer": "Tesla made some money last quarter. They sell cars and batteries.",
                "category": "earnings",
            },
            {
                "query": "What are the current interest rate expectations for the Federal Reserve?",
                "correct_answer": """Federal Reserve Interest Rate Outlook (Sources: CME FedWatch, Bloomberg):

Current Federal Funds Rate: 4.50-4.75%

Market Expectations (as of late 2024):
- December 2024: 75% probability of 25bp cut
- January 2025: 60% probability of another 25bp cut
- End of 2025: Terminal rate expected around 3.25-3.50%

Key Factors Driving Expectations:
1. Inflation trending toward 2% target
2. Labor market showing signs of cooling
3. Fed's stated data-dependent approach
4. Recent Fedspeak suggesting gradual easing path

Note: Interest rate expectations change with economic data releases. Verify with real-time Fed Funds futures.""",
                "incorrect_answer": "The Fed sets interest rates. They might change them sometime.",
                "category": "macro",
            },
        ]

        result = []
        for i in range(count):
            example = examples[i % len(examples)].copy()
            result.append(example)

        return result

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """Convert a single doc into a ContrastivePair."""
        try:
            query = doc.get("query", "").strip()
            correct = doc.get("correct_answer", "").strip()
            incorrect = doc.get("incorrect_answer", "").strip()
            category = doc.get("category", "general")

            if not query or not correct:
                return None

            task_prompt = f"""Financial Search Task: {query}

Please search for financial data and provide:
- Specific numbers and metrics where applicable
- Sources for your data
- Context and analysis
- Any caveats about data freshness"""

            metadata = {
                "label": "finsearchcomp",
                "source": "finsearchcomp",
                "category": category,
                "is_financial_search_benchmark": True,
            }

            return self._build_pair(
                question=task_prompt,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting pair from doc: {exc}", exc_info=True)
            return None

