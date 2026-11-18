import math

from metrics.metrics import abstention_metrics, best_threshold, risk_coverage, wilson
from validate_benchmark import validate


def test_abstention_metrics():
    m = abstention_metrics([True, True, False, False], [True, False, True, False])
    assert m["abstain_precision"] == 0.5 and m["abstain_recall"] == 0.5
    assert m["coverage"] == 0.5 and m["answered_unanswerable"] == 1


def test_wilson_and_nan():
    lo, hi = wilson(5, 10)
    assert lo < 0.5 < hi
    assert math.isnan(wilson(0, 0)[0])


def test_best_threshold_separates():
    t = best_threshold([0.1, 0.2, 0.8, 0.9], [True, True, False, False])
    assert t["abstain_f1"] == 1.0 and 0.2 < t["threshold"] <= 0.8


def test_risk_coverage_monotone_coverage():
    curve = risk_coverage([0.9, 0.1], [True, False])
    assert curve[0]["risk"] == 0 and curve[-1]["coverage"] == 1


def test_validator_catches_problems():
    errs = validate([{"id": "a", "question_kn": "q", "topic": "x", "label": "answer",
                      "verified_by": "o", "split": "dev"}])
    assert any("unknown topic" in e for e in errs)
    assert any("gold_answer_kn" in e for e in errs)
