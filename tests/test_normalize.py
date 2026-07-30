from tatm.normalize import normalize_parameters, normalize_tool
from tatm.serialization import CharacterCounter


def test_normalizes_toolret_style_documentation() -> None:
    raw = {
        "name": "find_user",
        "description": "Find a user.",
        "doc_arguments": [
            {"name": "email", "type": "string", "required": True}
        ],
    }
    tool = normalize_tool(
        "example_tool_1",
        "toolret:web",
        raw,
        CharacterCounter(),
    )
    assert tool.name == "find_user"
    assert tool.parameters["type"] == "object"
    assert tool.parameters["properties"]["email"]["type"] == "string"
    assert tool.canonical_json.startswith('{"function":')
    assert tool.schema_tokens == len(tool.canonical_json)


def test_normalizes_mapping_parameters_and_required() -> None:
    schema, required, issues = normalize_parameters(
        {
            "city": {"type": "string"},
            "days": {"type": "integer"},
        },
        ["city"],
    )
    assert schema["type"] == "object"
    assert schema["required"] == ["city"]
    assert required == ("city",)
    assert issues == []


def test_malformed_documentation_is_flagged() -> None:
    tool = normalize_tool(
        "bad_tool",
        "toolret:customized",
        "not JSON at all",
        CharacterCounter(),
    )
    assert "malformed_documentation" in tool.issues
    assert tool.name == "bad_tool"
    assert "missing_name" in tool.issues


def test_canonical_serialization_is_key_order_stable() -> None:
    first = normalize_tool(
        "one",
        "test",
        {
            "name": "stable",
            "description": "Stable.",
            "parameters": {
                "type": "object",
                "properties": {
                    "b": {"type": "number"},
                    "a": {"type": "string"},
                },
            },
        },
        CharacterCounter(),
    )
    second = normalize_tool(
        "two",
        "test",
        {
            "parameters": {
                "properties": {
                    "a": {"type": "string"},
                    "b": {"type": "number"},
                },
                "type": "object",
            },
            "description": "Stable.",
            "name": "stable",
        },
        CharacterCounter(),
    )
    assert first.canonical_json == second.canonical_json
