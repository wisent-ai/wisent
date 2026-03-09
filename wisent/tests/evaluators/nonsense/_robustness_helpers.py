"""Extracted from test_robustness.py - main function and __main__ block."""

import argparse


def main():
    """Main function for robustness testing."""
    parser = argparse.ArgumentParser(
        description="Test Math500 robustness with nonsense data"
    )
    parser.add_argument(
        "--model", required=True,
        help="Model to test"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show detailed output"
    )

    args = parser.parse_args()

    from wisent.tests.nonsense.test_robustness import test_math500_robustness

    try:
        test_math500_robustness(
            model=args.model,
            limit=None,
            verbose=args.verbose
        )
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        return 1
    except Exception as e:
        print(f"\nTest failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    return 0


def finish_robustness_output():
    """Print the final failure message for when both systems have problems.

    Called from test_math500_robustness when neither classifier nor lm-eval
    pass the robustness check.
    """
    print("\nROBUSTNESS TEST: ISSUES DETECTED")
    print("   Both evaluation systems may have problems")
