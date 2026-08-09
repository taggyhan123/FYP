import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "audit_frequency_online_structure.py"


def load_audit_module():
    spec = importlib.util.spec_from_file_location("frequency_structure_audit", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_tool(tool_id: str) -> dict:
    return {
        "type": "function",
        "function": {"name": tool_id, "description": "", "parameters": {}},
    }


def make_row(tool_ids: list[str]) -> dict:
    return {"tool_ids": tool_ids, "tools": [make_tool(item) for item in tool_ids]}


def test_structure_analysis_observes_only_after_planning() -> None:
    module = load_audit_module()
    rows = [
        make_row(["shared_a", "novel_z", "shared_b"]),
        make_row(["shared_a", "novel_y", "shared_b"]),
        make_row(["shared_a", "novel_x", "shared_b"]),
    ]

    result = module.analyze_rows(rows)

    assert result["universal_tools"] == 2
    assert result["one_non_universal_tool_per_request"] is True
    positions = result["causal_online_frequency_non_universal_position"]
    # Cold start is alphabetical; after the first observation, unseen tools
    # follow both previously seen shared tools.
    assert positions["first_request"] == 1
    assert positions["last_position_requests"] == 2
