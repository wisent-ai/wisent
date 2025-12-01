"""Parser setup for the 'generate-responses' command."""


def setup_generate_responses_parser(parser):
    """Set up the generate-responses command parser."""
    parser.add_argument("model", type=str, help="Model name or path")
    parser.add_argument("--task", type=str, required=True, help="Task name (e.g., arc_easy, truthfulqa_mc1)")
    parser.add_argument("--num-questions", type=int, default=10, help="Number of questions to generate responses for (default: 10)")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Maximum tokens to generate (default: 128)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation (default: 0.7)")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p for nucleus sampling (default: 0.95)")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cpu, cuda, mps)")
    parser.add_argument("--output", type=str, required=True, help="Output file path for results")
    parser.add_argument("--use-steering", action="store_true", help="Use steering during generation")
    parser.add_argument("--disable-thinking", action="store_true", help="Disable thinking/reasoning mode (prevents <think> tags for Qwen models)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
