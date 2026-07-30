from __future__ import annotations

import json
import urllib.parse
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol


def compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_function_json(
    name: str,
    description: str,
    parameters: dict[str, Any],
) -> str:
    """Serialize one tool in the stable OpenAI-compatible function shape."""
    return compact_json(
        {
            "function": {
                "description": description,
                "name": name,
                "parameters": parameters,
            },
            "type": "function",
        }
    )


class Counter(Protocol):
    model_id: str

    def count(self, text: str) -> int: ...


class TokenCounter:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-0.6B",
        cache_dir: Path | None = None,
    ) -> None:
        from tokenizers import Tokenizer

        self.model_id = model_id
        cache_root = cache_dir or Path(".cache") / "tatm" / "tokenizers"
        safe_model_id = model_id.replace("/", "--")
        tokenizer_path = cache_root / safe_model_id / "tokenizer.json"
        if not tokenizer_path.exists():
            tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
            quoted_model_id = urllib.parse.quote(model_id, safe="/")
            url = (
                f"https://huggingface.co/{quoted_model_id}/resolve/main/"
                "tokenizer.json"
            )
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "TATM-FYP/0.1 tokenizer accounting"},
            )
            temporary = tokenizer_path.with_suffix(".json.part")
            with urllib.request.urlopen(request, timeout=120) as response:
                temporary.write_bytes(response.read())
            temporary.replace(tokenizer_path)
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))

    @lru_cache(maxsize=65_536)
    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)


class CharacterCounter:
    """Deterministic lightweight counter used only in unit tests."""

    model_id = "character-test-counter"

    def count(self, text: str) -> int:
        return len(text)
