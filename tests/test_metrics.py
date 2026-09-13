from src.metrics import evaluate_binary


def test_evaluate_binary_returns_expected_values():
    metrics = evaluate_binary([0, 0, 1, 1], [0, 1, 1, 1])
    assert round(metrics.accuracy, 3) == 0.750
    assert round(metrics.balanced_accuracy, 3) == 0.750
    assert round(metrics.recall, 3) == 1.000
    assert round(metrics.f1, 3) == 0.800
