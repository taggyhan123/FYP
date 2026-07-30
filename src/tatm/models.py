from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class CanonicalTool:
    tool_id: str
    source: str
    name: str
    description: str
    parameters: dict[str, Any]
    required: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()
    issues: tuple[str, ...] = ()
    domain: str = ""
    canonical_json: str = ""
    schema_tokens: int = 0

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["required"] = list(self.required)
        record["examples"] = list(self.examples)
        record["issues"] = list(self.issues)
        return record

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "CanonicalTool":
        return cls(
            **{
                **record,
                "required": tuple(record.get("required", ())),
                "examples": tuple(record.get("examples", ())),
                "issues": tuple(record.get("issues", ())),
            }
        )

@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    source: str
    query: str
    tool_ids: tuple[str, ...]
    evidence_type: str
    domain: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["tool_ids"] = list(self.tool_ids)
        return record

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "TaskRecord":
        return cls(
            **{
                **record,
                "tool_ids": tuple(record.get("tool_ids", ())),
            }
        )
