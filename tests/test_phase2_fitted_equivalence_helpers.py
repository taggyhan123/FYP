import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "audit_phase2_fitted_equivalence.py"


def load_audit_module():
    spec = importlib.util.spec_from_file_location("fitted_equivalence", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_equivalence_classes_compare_behavior_not_labels() -> None:
    module = load_audit_module()
    sequences = {
        "first": [("a", "b"), ("a", "c")],
        "second": [("a", "b"), ("a", "c")],
        "third": [("b", "a"), ("a", "c")],
    }

    assert module.equivalence_classes(sequences) == [
        ["first", "second"],
        ["third"],
    ]
