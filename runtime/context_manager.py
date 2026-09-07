"""Bounded context assembly for small local coding models."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ContextBudget:
    system_chars: int = 6_000
    task_chars: int = 4_000
    facts_chars: int = 6_000
    code_chars: int = 18_000
    observation_chars: int = 8_000
    tools_chars: int = 6_000
    output_chars: int = 2_000
    total_chars: int = 48_000


@dataclass
class ContextBundle:
    text: str
    prompt_chars: int
    truncated: bool
    included_files: list[str] = field(default_factory=list)
    dropped_sections: list[str] = field(default_factory=list)


class ContextBudgetManager:
    """Build a deduplicated, bounded prompt from structured state."""

    def __init__(self, budget: ContextBudget | None = None) -> None:
        self.budget = budget or ContextBudget()

    @staticmethod
    def _clip(value: Any, limit: int) -> str:
        if isinstance(value, str):
            text = value
        else:
            text = json.dumps(value, ensure_ascii=False, default=str)
        return text[:limit]

    def build(self, *, system: str, task: str, facts: list[str] | None = None,
              files: dict[str, str] | None = None, observation: Any = None,
              tools: Any = None, output_schema: str = "") -> ContextBundle:
        sections: list[tuple[str, str]] = [
            ("SYSTEM", self._clip(system, self.budget.system_chars)),
            ("TASK", self._clip(task, self.budget.task_chars)),
            ("FACTS", self._clip(list(dict.fromkeys(facts or [])), self.budget.facts_chars)),
        ]
        included_files: list[str] = []
        file_parts: list[str] = []
        remaining = self.budget.code_chars
        for path, content in list((files or {}).items()):
            if remaining <= 0:
                break
            excerpt = self._clip(content, min(remaining, 6_000))
            file_parts.append(f"FILE {path}:\n{excerpt}")
            included_files.append(path)
            remaining -= len(excerpt)
        sections.append(("RELEVANT_CODE", "\n\n".join(file_parts)))
        sections.append(("OBSERVATION", self._clip(observation or {}, self.budget.observation_chars)))
        sections.append(("TOOLS", self._clip(tools or [], self.budget.tools_chars)))
        sections.append(("OUTPUT", self._clip(output_schema, self.budget.output_chars)))
        text = "\n\n".join(f"[{name}]\n{body}" for name, body in sections if body)
        truncated = len(text) > self.budget.total_chars
        return ContextBundle(text=text[:self.budget.total_chars], prompt_chars=min(len(text), self.budget.total_chars),
                             truncated=truncated, included_files=included_files,
                             dropped_sections=[name for name, body in sections if not body])

