from __future__ import annotations

from typing import Any
from wisent.core.cli_logger import setup_logger

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

__all__ = ["ToolBenchExtractor"]

log = setup_logger(__name__)

# ToolBench API categories
TOOLBENCH_CATEGORIES = [
    "Social",
    "Finance",
    "Data",
    "Sports",
    "Entertainment",
    "Travel",
    "Weather",
    "Food",
    "News",
    "Music",
]


class ToolBenchExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for ToolBench/ToolLLM - API Tool Learning Benchmark (ICLR 2024 Spotlight).

    ToolBench evaluates LLMs on real-world API tool use with 16,000+ APIs
    from RapidAPI across 49 categories. Tests both single-tool and multi-tool
    scenarios requiring chains of API calls.

    Related benchmarks:
    - NexusRaven: Function calling evaluation
    - Nexus Function Calling: Commercial API evaluation

    Dataset: Maurus/ToolBench (or OpenBMB/ToolBench on GitHub)

    Schema:
        - query: str (user instruction)
        - query_id: int (unique identifier)
        - api_list: list[dict] (available APIs)
        - answer: str (solution path)
        - category: str (API category)

    For function calling evaluation:
    - Positive (correct) = Correct API sequence with proper parameters
    - Negative (incorrect) = Wrong API calls or incorrect parameters
    """

    # Evaluator that should be used for this benchmark
    evaluator_name = "function_calling"

    def __init__(self, category: str | None = None):
        """
        Initialize ToolBench extractor.

        Args:
            category: Optional filter for API category
        """
        super().__init__()
        self.category = category

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from ToolBench examples.

        Creates pairs for function calling evaluation:
        - Positive (correct) = Correct API usage
        - Negative (incorrect) = Incorrect API usage

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        try:
            docs = self.load_dataset(
                dataset_name="Maurus/ToolBench",
                split="train",
                limit=max_items * 2 if max_items else None,
            )
            log.info(f"Loaded {len(docs)} examples from ToolBench")
        except Exception as e:
            log.warning(f"Failed to load ToolBench from HF: {e}")
            # Create synthetic examples
            docs = self._create_synthetic_examples(max_items or 100)

        pairs: list[ContrastivePair] = []

        for doc in docs:
            # Filter by category if specified
            if self.category:
                doc_category = doc.get("category", "")
                if self.category.lower() not in doc_category.lower():
                    continue

            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid ToolBench pairs extracted")

        return pairs

    def _create_synthetic_examples(self, count: int) -> list[dict[str, Any]]:
        """Create synthetic examples based on ToolBench structure."""
        examples = []

        toolbench_cases = [
            {
                "query": "What's the weather like in New York today?",
                "category": "Weather",
                "api_list": [
                    {"name": "get_current_weather", "parameters": {"city": "str", "units": "str"}},
                    {"name": "get_forecast", "parameters": {"city": "str", "days": "int"}},
                ],
                "correct_call": "get_current_weather(city='New York', units='fahrenheit')",
                "incorrect_call": "get_forecast(city='NY', days=7)",
            },
            {
                "query": "Find me the top 10 trending songs on Spotify",
                "category": "Music",
                "api_list": [
                    {"name": "get_trending_tracks", "parameters": {"limit": "int", "market": "str"}},
                    {"name": "search_tracks", "parameters": {"query": "str", "limit": "int"}},
                ],
                "correct_call": "get_trending_tracks(limit=10, market='US')",
                "incorrect_call": "search_tracks(query='trending', limit=10)",
            },
            {
                "query": "Get the latest stock price for Apple",
                "category": "Finance",
                "api_list": [
                    {"name": "get_stock_quote", "parameters": {"symbol": "str"}},
                    {"name": "get_company_info", "parameters": {"symbol": "str"}},
                ],
                "correct_call": "get_stock_quote(symbol='AAPL')",
                "incorrect_call": "get_company_info(symbol='Apple')",
            },
            {
                "query": "Book a flight from LA to Chicago for next Monday",
                "category": "Travel",
                "api_list": [
                    {"name": "search_flights", "parameters": {"origin": "str", "destination": "str", "date": "str"}},
                    {"name": "book_flight", "parameters": {"flight_id": "str", "passengers": "int"}},
                ],
                "correct_call": "search_flights(origin='LAX', destination='ORD', date='2024-01-15')",
                "incorrect_call": "book_flight(flight_id='unknown', passengers=1)",
            },
            {
                "query": "Send a tweet saying 'Hello World'",
                "category": "Social",
                "api_list": [
                    {"name": "post_tweet", "parameters": {"text": "str"}},
                    {"name": "get_timeline", "parameters": {"count": "int"}},
                ],
                "correct_call": "post_tweet(text='Hello World')",
                "incorrect_call": "get_timeline(count=1)",
            },
            {
                "query": "Get today's top news headlines",
                "category": "News",
                "api_list": [
                    {"name": "get_top_headlines", "parameters": {"country": "str", "category": "str"}},
                    {"name": "search_news", "parameters": {"query": "str", "from_date": "str"}},
                ],
                "correct_call": "get_top_headlines(country='us', category='general')",
                "incorrect_call": "search_news(query='news', from_date='yesterday')",
            },
            {
                "query": "Find restaurants near Times Square",
                "category": "Food",
                "api_list": [
                    {"name": "search_restaurants", "parameters": {"location": "str", "radius": "int"}},
                    {"name": "get_restaurant_details", "parameters": {"restaurant_id": "str"}},
                ],
                "correct_call": "search_restaurants(location='Times Square, NYC', radius=1000)",
                "incorrect_call": "get_restaurant_details(restaurant_id='unknown')",
            },
            {
                "query": "Get the score of yesterday's Lakers game",
                "category": "Sports",
                "api_list": [
                    {"name": "get_game_scores", "parameters": {"team": "str", "date": "str"}},
                    {"name": "get_team_schedule", "parameters": {"team": "str"}},
                ],
                "correct_call": "get_game_scores(team='Lakers', date='yesterday')",
                "incorrect_call": "get_team_schedule(team='LA')",
            },
        ]

        for i in range(count):
            case = toolbench_cases[i % len(toolbench_cases)].copy()
            case["query_id"] = i
            examples.append(case)

        return examples

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single doc into a ContrastivePair.
        """
        try:
            import json

            query_id = doc.get("query_id", 0)
            query = doc.get("query", "").strip()
            api_list_raw = doc.get("api_list", [])
            category = doc.get("category", doc.get("domain", ""))
            answer = doc.get("answer", "")
            correct_call = doc.get("correct_call", "")
            incorrect_call = doc.get("incorrect_call", "")

            # Handle api_list being a JSON string (from HuggingFace)
            if isinstance(api_list_raw, str):
                try:
                    api_list = json.loads(api_list_raw)
                except json.JSONDecodeError:
                    api_list = []
            else:
                api_list = api_list_raw

            if not query:
                log.debug("Skipping: missing query")
                return None

            # Build the function calling prompt
            task_prompt = self._build_function_prompt(query, api_list)

            # Positive = correct API call
            if correct_call:
                correct_response = self._create_correct_response(query, correct_call)
            elif answer:
                correct_response = self._create_correct_response(query, answer)
            else:
                correct_response = self._create_generic_correct_response(query, api_list)

            # Negative = incorrect API call
            if incorrect_call:
                incorrect_response = self._create_incorrect_response(query, incorrect_call)
            else:
                incorrect_response = self._create_generic_incorrect_response(query, api_list)

            metadata = {
                "label": "toolbench",
                "source": "OpenBMB/ToolBench",
                "query_id": query_id,
                "category": category,
                "num_apis": len(api_list) if api_list else 0,
                "is_function_calling_benchmark": True,
            }

            return self._build_pair(
                question=task_prompt,
                correct=correct_response,
                incorrect=incorrect_response,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting pair from doc: {exc}", exc_info=True)
            return None

    def _build_function_prompt(
        self,
        query: str,
        api_list: list[dict[str, Any]],
    ) -> str:
        """Build the function calling task prompt."""
        parts = [f"User Request: {query}"]

        if api_list:
            parts.append("\nAvailable APIs:")
            for api in api_list:
                # Handle both synthetic format and HuggingFace ToolBench format
                if isinstance(api, dict):
                    # HuggingFace ToolBench uses: tool_name, api_name, api_description
                    name = api.get("api_name", api.get("name", "unknown"))
                    tool_name = api.get("tool_name", "")
                    params = api.get("parameters", api.get("required_parameters", []))

                    if tool_name and name != tool_name:
                        full_name = f"{tool_name}.{name}"
                    else:
                        full_name = name

                    if isinstance(params, dict):
                        params_str = ", ".join([f"{k}: {v}" for k, v in params.items()])
                    elif isinstance(params, list):
                        params_str = ", ".join([str(p.get("name", p)) if isinstance(p, dict) else str(p) for p in params])
                    else:
                        params_str = str(params) if params else ""

                    parts.append(f"- {full_name}({params_str})")
                else:
                    parts.append(f"- {api}")

        parts.append(
            "\nSelect the appropriate API and provide the correct function call "
            "with proper parameters to fulfill the user's request."
        )

        return "\n".join(parts)

    def _create_correct_response(self, query: str, api_call: str) -> str:
        """Create a correct function calling response."""
        return (
            f"To fulfill your request, I'll use the following API call:\n\n"
            f"```\n{api_call}\n```\n\n"
            "This is the appropriate API with correctly formatted parameters "
            "to get the information you requested."
        )

    def _create_incorrect_response(self, query: str, api_call: str) -> str:
        """Create an incorrect function calling response."""
        return (
            f"I'll call this API for you:\n\n"
            f"```\n{api_call}\n```\n\n"
            "This should get what you need."
        )

    def _create_generic_correct_response(
        self, query: str, api_list: list[dict[str, Any]]
    ) -> str:
        """Create a generic correct response."""
        if api_list:
            api_name = api_list[0].get("name", "relevant_api")
            return (
                f"I'll use the {api_name} API with the appropriate parameters "
                "based on your request. This API is designed specifically for "
                "this type of query and will return the accurate information you need."
            )
        return "I'll call the appropriate API with the correct parameters for your request."

    def _create_generic_incorrect_response(
        self, query: str, api_list: list[dict[str, Any]]
    ) -> str:
        """Create a generic incorrect response."""
        if api_list and len(api_list) > 1:
            # Use a wrong API
            wrong_api = api_list[-1].get("name", "some_api")
            return (
                f"Let me use {wrong_api} for this. "
                "I'll figure out the parameters as I go."
            )
        return "I'll try calling an API to handle this request somehow."

