"""Model router — selects the best local model for a classified task.

Pipeline::

    Task
     ↓
    Check execution_mode (if DIRECT_TOOL → skip model selection)
     ↓
    Score every registered model against task & availability
     ↓
    Filter out score < 0 (hard capability mismatch or unavailable)
     ↓
    Pick highest scorer
     ↓
    If none match → fallback model
     ↓
    RoutingResult
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from models.base import ModelProvider
from models.registry import ModelRegistry
from routing.classifier import ExecutionMode, Task
from routing.scoring import score_model


class RoutingResult(BaseModel):
    """Outcome of routing a task to a model or direct tool."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str           # registry key, e.g. "qwen-coder" or "direct_tool"
    model_id: str             # Ollama tag, e.g. "qwen2.5-coder:7b" or "none"
    score: float
    reason: str
    is_direct_tool: bool = False


class ModelRouter:
    """Deterministic model router.

    Scores every registered model against the task and picks the best.
    Falls back to the designated fallback model if preferred model is unavailable.
    """

    def __init__(self, registry: ModelRegistry):
        self.registry = registry

    def route(
        self,
        task: Task,
        availability: dict[str, bool] | None = None,
    ) -> RoutingResult:
        """Select the best model for *task*, respecting availability."""
        # 1. Direct tool execution check: No LLM needed!
        if task.execution_mode == ExecutionMode.DIRECT_TOOL:
            tool_name = task.direct_tool_name or "tool"
            return RoutingResult(
                model_name="direct_tool",
                model_id="none",
                score=100.0,
                reason=f"Direct deterministic tool: {tool_name} (no LLM required)",
                is_direct_tool=True,
            )

        # 2. Score registered models
        candidates: list[tuple[str, float]] = []

        for config in self.registry.list_models():
            # If availability is provided and model is marked False, skip it
            if availability is not None and not availability.get(config.name, True):
                continue

            s = score_model(task, config)
            if s >= 0:
                candidates.append((config.name, s))

        if not candidates:
            # Nothing matched or preferred models unavailable — try fallback
            fb = self.registry.get_fallback()
            # Check if fallback itself is available
            if fb is not None:
                if availability is None or availability.get(fb.name, True):
                    return RoutingResult(
                        model_name=fb.name,
                        model_id=fb.model_id,
                        score=0.0,
                        reason="Fallback model selected (preferred model unavailable or no capability match)",
                        is_direct_tool=False,
                    )

            raise RuntimeError(
                f"No model can handle task_type={task.task_type.value} "
                f"and no available fallback is configured"
            )

        # Sort descending by score
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_name, best_score = candidates[0]
        best_cfg = self.registry.get_provider(best_name).config

        return RoutingResult(
            model_name=best_name,
            model_id=best_cfg.model,
            score=best_score,
            reason=self._explain(task, best_name, best_score),
            is_direct_tool=False,
        )

    def get_provider_for_task(
        self,
        task: Task,
        availability: dict[str, bool] | None = None,
    ) -> ModelProvider:
        """Convenience: route + return the actual provider instance."""
        result = self.route(task, availability=availability)
        if result.is_direct_tool:
            raise ValueError("Task is configured for direct tool execution; no provider needed.")
        return self.registry.get_provider(result.model_name)

    @staticmethod
    def _explain(task: Task, model_name: str, score: float) -> str:
        return (
            f"{model_name} selected for {task.task_type.value} task "
            f"(score={score:.1f})"
        )
