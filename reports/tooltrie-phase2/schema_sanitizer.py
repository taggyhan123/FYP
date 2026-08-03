#!/usr/bin/env python3
"""Normalize invalid JSON Schema `type` values in a workload's tool payloads.

BFCL/ToolRet ship Python type names ("float", "dict", "int, optional") where
JSON Schema requires one of the seven primitive type names. vLLM tolerates this;
SGLang validates against the metaschema and rejects the whole request.

The mapping is applied identically to every condition, and touches ONLY the
`type` string inside `function.parameters`. Tool names, tool order, tool_ids,
request order, and menu membership are all left untouched, so the ordering
comparison this workload exists to support remains valid.

Optionality ("int, optional") is dropped from the type because JSON Schema
expresses it via the `required` array, which is preserved as-is.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

VALID = {"string", "number", "integer", "object", "array", "boolean", "null"}

MAP = {
    "float": "number", "double": "number", "decimal": "number", "num": "number",
    "int": "integer", "integer64": "integer", "long": "integer",
    "str": "string", "string": "string", "text": "string", "char": "string",
    "dict": "object", "object": "object", "mapping": "object", "json": "object",
    "list": "array", "tuple": "array", "array": "array", "sequence": "array", "set": "array",
    "bool": "boolean", "boolean": "boolean",
    "none": "null", "nonetype": "null", "null": "null",
    "any": "string",
}

stats: Counter[str] = Counter()
unmapped: Counter[str] = Counter()


def norm(raw: str) -> str | None:
    t = raw.strip().lower()
    # Generic containers are arrays regardless of element type: List[str], list[int]...
    if t.startswith(("list[", "sequence[", "tuple[", "set[", "array[")):
        return "array"
    if t.startswith(("dict[", "mapping[")):
        return "object"
    if t.startswith(("optional[", "union[")):
        return "string"
    # "int, optional" / "string (required)" -> take the leading token
    for sep in (",", "(", "|", "/"):
        if sep in t:
            t = t.split(sep)[0].strip()
    t = t.replace(" ", "")
    if t in VALID:
        return t
    if t in MAP:
        return MAP[t]
    unmapped[raw] += 1
    return None


def walk(node):
    if isinstance(node, dict):
        t = node.get("type")
        if isinstance(t, str) and t not in VALID:
            fixed = norm(t)
            if fixed is not None:
                stats[f"{t} -> {fixed}"] += 1
                node["type"] = fixed
            else:
                stats[f"UNMAPPED {t} -> string"] += 1
                node["type"] = "string"
        for v in node.values():
            walk(v)
    elif isinstance(node, list):
        for v in node:
            walk(v)


RESERVED = ("title", "description", "default", "examples", "$comment")


def relocate_root_properties(params: dict) -> None:
    """Move property definitions that leaked to the schema root under `properties`.

    Some ToolRet tools declare a parameter literally named `title`, placed at the
    schema root. JSON Schema reserves `title` as a *string* annotation, so a dict
    there fails the metaschema and SGLang rejects the entire request. Only keys
    that are unmistakably schema objects (dicts carrying a `type`) are moved, and
    only when they collide with a reserved annotation keyword.
    """
    if not isinstance(params, dict):
        return
    moved = {}
    for key in RESERVED:
        val = params.get(key)
        if isinstance(val, dict) and "type" in val:
            moved[key] = params.pop(key)
    if moved:
        props = params.setdefault("properties", {})
        for key, val in moved.items():
            props.setdefault(key, val)
        params.setdefault("type", "object")
        stats[f"relocated root property {sorted(moved)}"] += 1


def main() -> None:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    out = []
    for line in src.open(encoding="utf-8"):
        rec = json.loads(line)
        ids_before = list(rec["tool_ids"])
        names_before = [t["function"]["name"] for t in rec["tools"]]
        for tool in rec["tools"]:
            relocate_root_properties(tool.get("function", {}).get("parameters", {}))
            walk(tool.get("function", {}).get("parameters", {}))
        assert rec["tool_ids"] == ids_before, "tool_ids changed"
        assert [t["function"]["name"] for t in rec["tools"]] == names_before, "tool order changed"
        out.append(rec)
    with dst.open("w", encoding="utf-8") as fh:
        for rec in out:
            fh.write(json.dumps(rec, sort_keys=False) + "\n")
    print(f"{src.name}: {len(out)} records | repairs: {sum(stats.values())}")
    if unmapped:
        print(f"  UNMAPPED (coerced to string): {dict(unmapped)}", file=sys.stderr)


if __name__ == "__main__":
    main()
