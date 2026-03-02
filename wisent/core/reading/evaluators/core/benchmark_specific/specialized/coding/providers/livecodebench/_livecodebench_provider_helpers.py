"""Extracted from provider.py - _generate_stdin_test method tail."""


def complete_stdin_test_code(test_code: str) -> str:
    """Complete the stdin test code by appending the main block and returning it.

    Finishes the test code string that was built by _generate_stdin_test by
    adding the if __name__ == '__main__' block and the return statement.

    Args:
        test_code: Partially constructed test code string ending after
                   the print statement for passed tests

    Returns:
        Complete test code string with main block
    """
    test_code += """
if __name__ == '__main__':
    test_stdin()
"""

    return test_code
