from __future__ import annotations

import ast
import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from typing import Any

from tatm.models import CanonicalTool
from tatm.serialization import Counter, canonical_function_json, compact_json


_NAME_KEYS = ("name", "tool_name", "function_name", "api_name")
_DESCRIPTION_KEYS = (
    "description",
    "functionality",
    "summary",
    "documentation",
)
_PARAMETER_KEYS = (
    "parameters",
    "input_schema",
    "inputSchema",
    "doc_arguments",
    "arguments",
    "api_arguments",
)
_EXAMPLE_KEYS = ("examples", "example", "example_code", "usage")
_INVALID_NAME = re.compile(r"[^A-Za-z0-9_.:-]+")


def parse_maybe_structured(value: Any) -> tuple[Any, bool]:
    if isinstance(value, (Mapping, list, tuple)):
        return value, True
    if value is None:
        return {}, True
    if not isinstance(value, str):
        return value, False
    stripped = value.strip()
    if not stripped:
        return {}, True
    try:
        return json.loads(stripped), True
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(stripped), True
        except (ValueError, SyntaxError):
            return value, False


def _first(mapping: Mapping[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def _string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return compact_json(value)


def _fallback_name(tool_id: str) -> str:
    candidate = tool_id.rsplit(":", 1)[-1].rsplit("/", 1)[-1]
    candidate = _INVALID_NAME.sub("_", candidate).strip("_")
    return candidate[:128] or "unnamed_tool"


def _normalize_property(value: Any) -> dict[str, Any]:
    parsed, valid = parse_maybe_structured(value)
    if valid and isinstance(parsed, Mapping):
        result = dict(parsed)
    elif isinstance(value, str) and value:
        result = {"description": value}
    else:
        result = {}
    if result.get("type") == "dict":
        result["type"] = "object"
    return result


def normalize_parameters(
    raw: Any,
    top_level_required: Any = None,
) -> tuple[dict[str, Any], tuple[str, ...], list[str]]:
    issues: list[str] = []
    parsed, valid = parse_maybe_structured(raw)
    if not valid:
        issues.append("malformed_parameters")

    if isinstance(parsed, list):
        properties: dict[str, Any] = {}
        for item in parsed:
            if isinstance(item, str):
                properties[item] = {}
            elif isinstance(item, Mapping):
                item_name = _string(_first(item, ("name", "key", "parameter")))
                if item_name:
                    item_schema = {
                        key: value
                        for key, value in item.items()
                        if key not in {"name", "key", "parameter", "required"}
                    }
                    properties[item_name] = _normalize_property(item_schema)
        schema: dict[str, Any] = {"type": "object", "properties": properties}
    elif isinstance(parsed, Mapping):
        schema = dict(parsed)
        if schema.get("type") == "dict":
            schema["type"] = "object"
        schema_keys = {
            "$schema",
            "$defs",
            "additionalProperties",
            "allOf",
            "anyOf",
            "definitions",
            "description",
            "oneOf",
            "properties",
            "required",
            "title",
            "type",
        }
        if not any(key in schema for key in schema_keys):
            schema = {
                "type": "object",
                "properties": {
                    str(key): _normalize_property(value)
                    for key, value in schema.items()
                },
            }
        else:
            schema.setdefault("type", "object")
            properties = schema.get("properties", {})
            if not isinstance(properties, Mapping):
                issues.append("non_object_properties")
                properties = {}
            schema["properties"] = {
                str(key): _normalize_property(value)
                for key, value in properties.items()
            }
    else:
        schema = {"type": "object", "properties": {}}
        issues.append("missing_parameters")

    required_raw = schema.get("required", top_level_required)
    if isinstance(required_raw, str):
        required = (required_raw,)
    elif isinstance(required_raw, (list, tuple, set)):
        required = tuple(str(item) for item in required_raw)
    else:
        required = ()

    properties = schema.setdefault("properties", {})
    if not properties:
        issues.append("empty_parameters")
    unknown_required = sorted(set(required) - set(properties))
    if unknown_required:
        issues.append("required_field_not_in_properties")
    schema["required"] = list(required)
    return schema, required, issues


def normalize_tool(
    tool_id: str,
    source: str,
    raw: Any,
    counter: Counter,
    *,
    domain: str = "",
) -> CanonicalTool:
    parsed, valid = parse_maybe_structured(raw)
    issues: list[str] = []
    if not valid:
        issues.append("malformed_documentation")
        parsed = {"documentation": _string(raw)}
    if not isinstance(parsed, Mapping):
        issues.append("non_object_documentation")
        parsed = {"documentation": _string(parsed)}

    function = parsed.get("function")
    if isinstance(function, Mapping):
        merged: dict[str, Any] = {**parsed, **function}
    else:
        merged = dict(parsed)

    name = _string(_first(merged, _NAME_KEYS))
    if not name:
        issues.append("missing_name")
        name = _fallback_name(tool_id)

    description = _string(_first(merged, _DESCRIPTION_KEYS))
    if not description:
        issues.append("missing_description")

    parameter_raw = _first(merged, _PARAMETER_KEYS)
    parameters, required, parameter_issues = normalize_parameters(
        parameter_raw,
        merged.get("required"),
    )
    issues.extend(parameter_issues)

    examples_value = _first(merged, _EXAMPLE_KEYS)
    if examples_value is None:
        examples: tuple[str, ...] = ()
    elif isinstance(examples_value, (list, tuple)):
        examples = tuple(_string(item) for item in examples_value)
    else:
        examples = (_string(examples_value),)

    inferred_domain = domain or _string(
        _first(merged, ("domain", "server", "framework", "category"))
    )
    canonical_json = canonical_function_json(name, description, parameters)
    return CanonicalTool(
        tool_id=tool_id,
        source=source,
        name=name,
        description=description,
        parameters=parameters,
        required=required,
        examples=examples,
        issues=tuple(sorted(set(issues))),
        domain=inferred_domain,
        canonical_json=canonical_json,
        schema_tokens=counter.count(canonical_json),
    )


def bfcl_tool_id(raw_function: Mapping[str, Any], canonical_json: str) -> str:
    name = _string(raw_function.get("name")) or "unnamed"
    digest = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()[:12]
    safe_name = _INVALID_NAME.sub("_", name).strip("_")[:80] or "unnamed"
    return f"bfcl:{safe_name}:{digest}"
