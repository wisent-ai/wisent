"""Extracted from test_cross_benchmark_mode.py - run_all_tests tail and __main__."""


def finish_run_all_tests(failed):
    """Complete the run_all_tests function by returning the exit code.

    Args:
        failed: List of (test_name, error_message) tuples for failed tests

    Returns:
        0 if all tests passed, 1 if any tests failed
    """
    if not failed:
        return 0
    else:
        print(f"\nSome tests failed. Review the output above.")
        return 1
