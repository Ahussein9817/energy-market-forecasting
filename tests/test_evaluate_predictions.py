import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import evaluate_predictions as ep  # noqa: E402


def test_max_abs_diff_numeric_nested_and_nan():
    a = {"x": [1.0, {"y": 2.0}], "n": float("nan")}
    assert ep.max_abs_diff(a, {"x": [1.0, {"y": 2.0}], "n": float("nan")}) == 0.0
    assert ep.max_abs_diff(a, {"x": [1.0, {"y": 2.5}], "n": float("nan")}) == 0.5


def test_max_abs_diff_structure_mismatch_is_inf():
    assert ep.max_abs_diff({"a": 1}, {"b": 1}) == float("inf")
    assert ep.max_abs_diff([1, 2], [1]) == float("inf")
    assert ep.max_abs_diff({"a": "x"}, {"a": "y"}) == float("inf")
