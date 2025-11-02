"""Parser setup for the 'evaluate-responses' command."""


def setup_evaluate_responses_parser(parser):
    """Set up the evaluate-responses command parser."""
    parser.add_argument("--input", type=str, required=True, help="Input JSON file with generated responses")
    parser.add_argument("--baseline", type=str, help="Baseline responses JSON file (for personalization comparison)")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file for evaluation results")
    parser.add_argument("--task", type=str, help="Task name (optional, overrides task from input JSON)")
    parser.add_argument("--trait", type=str, help="Personality trait to evaluate (optional, for personalization tasks)")
    parser.add_argument("--trait-description", type=str, help="Description of the personality trait")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
