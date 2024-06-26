# Inspired by https://github.com/prometheus-eval/prometheus-eval/blob/main/eval/parser.py

import re

pattern = re.compile(
    r"""
    (?:                     # Non-capturing group for various result indicators
        \[RESULT\]          # [RESULT]
        |Score              # Score
        |\[SCORE\]          # [SCORE]
        |\[RESULT\]:        # [RESULT]:
        |Score:             # Score:
        |score:             # score:
        |Result:            # Result:
        |\[Result\]         # [Result]
        |score\s+of         # score of
        |Feedback:          # Feedback:
        |feedback:          # feedback:
    )
    \s*                     # Optional whitespace
    (?:                     # Non-capturing group for optional brackets or parentheses
        \(\s*               # Opening parenthesis with optional whitespace
        |\[\s*              # or opening square bracket with optional whitespace
        |                   # or nothing
    )
    \s*                     # Optional whitespace
    (\d+)                   # Capturing group for one or more digits
    """,
    re.IGNORECASE | re.VERBOSE,
)


def parse_judgment(judgment):
    matches = pattern.search(judgment)

    if matches:
        # Extract the first group that matches (ignoring None)
        result = next((int(match) for match in matches.groups() if match), None)
        if result is not None:
            feedback = (
                judgment.split("[RESULT]")[0].strip()
                if "[RESULT]" in judgment
                else judgment
            )
            return feedback, result

    return None, None


if __name__ == "__main__":
    # Test cases
    test_cases = [
        # Absolute mode test cases (a2a, a2r)
        ("Good job. [RESULT] 3", 3),
        ("Needs improvement. [RESULT] Score: 2", 2),
        ("Well done. [RESULT] Result: 4", 4),
        ("Average. [RESULT] 4/5", 4),
        ("Excellent. [RESULT] 5 out of 5", 5),
        ("Poor performance. [RESULT] score of 1", 1),
        ("Good job. [Result] 3", 3),
        ("Needs improvement. [Result] Score: 2", 2),
        ("Well done. [Result] Result: 4", 4),
        ("Average. [Result] 4/5", 4),
        ("Excellent. [Result] 5 out of 5", 5),
        ("Poor performance. [Result] score of 1", 1),
        ("Good job. [3]", 3),
        ("Good job. (Score 5)", 5),
        ("Good job. [Score 4]", 4),
        ("Good job. score: 3", 3),
        ("Good job. Score: 3", 3),
        ("Good job. score of 1", 1),
        ("Good job. [RESULT] (5)", 5),
    ]

    def run_tests():
        failed_tests = []  # To keep track of failed tests

        for output, expected in test_cases:
            _, result = parse_judgment(output)
            if result != expected:
                failed_tests.append((output, expected, result))

        if failed_tests:
            print("Some tests failed:")
            for output, expected, result in failed_tests:
                print(f"  For input: '{output}', expected: {expected}, got: {result}")
        else:
            print("All tests passed!")

    run_tests()
