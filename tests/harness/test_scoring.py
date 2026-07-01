from harness.scoring import classify_probe_outcome


def test_classify_success_with_boxed_answer() -> None:
    problem = {"ground_truth": "42"}
    completion = r"Reasoning... \boxed{42}"
    assert classify_probe_outcome(problem, completion, truncated=False) == "success"
